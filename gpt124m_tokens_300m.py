"""
~300M-parameter GPT-2 style model, trained from scratch on FineWeb-Edu shards.
Built for Kaggle 2x T4 (fp16 autocast + GradScaler, DDP, torch.compile).

Launch:
    torchrun --standalone --nproc_per_node=2 train_gpt2_300m.py

Env vars (all optional):
    MODEL_SIZE    "300m" (304M params, default) or "124m" (GPT-2 small)
    DATA_ROOT     folder with edufineweb_*.npy shards   (default edu_fineweb300M)
    TOTAL_TOKENS  token budget for the LR schedule      (default 3e8)
    MICRO_BATCH   per-GPU micro-batch                   (default 4)
    MAX_HOURS     stop gracefully + checkpoint after N hours (default 0 = off)
    CKPT_DIR      where checkpoints/logs go             (default log)
    RESUME_FROM   checkpoint to resume from             (default CKPT_DIR/ckpt_last.pt if it exists)
    CKPT_EVERY_MIN  save (and upload) a checkpoint every N minutes (default 30)
    HF_REPO       e.g. "user/gpt300m-ckpt": private Hugging Face repo used as persistent storage
    HF_TOKEN      Hugging Face token with WRITE access (use Kaggle Secrets)

With HF_REPO set: on start, if no local checkpoint exists, the latest one is downloaded from the
repo and training resumes from it; during training every checkpoint is uploaded in the background.
"""
import os
import math
import time
import inspect
import datetime
import threading
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import tiktoken


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304   # 50257 padded to a multiple of 128
    n_layer: int = 20
    n_head: int = 16
    n_embd: int = 1024        # -> ~304M parameters


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight   # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"sequence length {T} > block size {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type, verbose=True):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if verbose:
            print(f"decayed tensors: {len(decay_params)} ({sum(p.numel() for p in decay_params):,} params), "
                  f"non-decayed: {len(nodecay_params)} ({sum(p.numel() for p in nodecay_params):,} params), "
                  f"fused AdamW: {use_fused}")
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)


# ----------------------------------------------------------------------------
# data: sharded, rank-aware loader
# ----------------------------------------------------------------------------
def load_tokens(filename):
    npt = np.load(filename).astype(np.int32)
    return torch.tensor(npt, dtype=torch.long)


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, data_root, verbose=True):
        self.B, self.T = B, T
        self.process_rank, self.num_processes = process_rank, num_processes
        assert split in {'train', 'val'}
        shards = sorted(s for s in os.listdir(data_root) if split in s and s.endswith('.npy'))
        assert len(shards) > 0, f"no {split} shards found in {data_root}"
        self.shards = [os.path.join(data_root, s) for s in shards]
        if verbose:
            print(f"found {len(self.shards)} shards for split {split}")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

    # for resuming: all ranks advance in lockstep, so (shard, rank-0 position) is enough
    def state(self):
        return {'shard': self.current_shard,
                'base': self.current_position - self.B * self.T * self.process_rank}

    def load_state(self, s):
        self.current_shard = s['shard']
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = s['base'] + self.B * self.T * self.process_rank


# ----------------------------------------------------------------------------
# DDP setup
# ----------------------------------------------------------------------------
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "DDP needs CUDA"
    init_process_group(backend='nccl', timeout=datetime.timedelta(minutes=60))
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
else:
    ddp_rank, ddp_local_rank, ddp_world_size = 0, 0, 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
master_process = ddp_rank == 0
device_type = 'cuda' if device.startswith('cuda') else 'cpu'

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding('gpt2')

# ----------------------------------------------------------------------------
# hyperparameters
# ----------------------------------------------------------------------------
data_root = os.environ.get("DATA_ROOT", "edu_fineweb300M")
total_tokens = int(float(os.environ.get("TOTAL_TOKENS", 3e8)))
total_batch_size = 262144                       
MODEL_PRESETS = {
    "124m": dict(cfg=dict(n_layer=12, n_head=12, n_embd=768),  micro_batch=16, max_lr=6e-4),
    "300m": dict(cfg=dict(n_layer=20, n_head=16, n_embd=1024), micro_batch=4,  max_lr=3e-4),
}
model_size = os.environ.get("MODEL_SIZE", "300m").lower()
assert model_size in MODEL_PRESETS, f"MODEL_SIZE must be one of {list(MODEL_PRESETS)}"
preset = MODEL_PRESETS[model_size]
B = int(os.environ.get("MICRO_BATCH", preset["micro_batch"]))   
T = 1024
max_hours = float(os.environ.get("MAX_HOURS", 0))
use_compile = True

assert total_batch_size % (B * T * ddp_world_size) == 0, "total_batch_size must be divisible by B*T*world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

max_steps = total_tokens // total_batch_size
warmup_steps = max(20, int(0.04 * max_steps))
max_lr = preset["max_lr"]                        
min_lr = max_lr * 0.1
weight_decay = 0.1

val_every, val_steps = 250, 20
sample_every = 500
ckpt_every_min = float(os.environ.get("CKPT_EVERY_MIN", 30))

log_dir = os.environ.get("CKPT_DIR", "log")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")
ckpt_path = os.path.join(log_dir, "ckpt_last.pt")
resume_path = os.environ.get("RESUME_FROM", ckpt_path)

if master_process:
    print(f"model {model_size} | world size {ddp_world_size} | micro-batch {B}x{T} | grad accum {grad_accum_steps} | "
          f"tokens/step {total_batch_size:,} | max_steps {max_steps} | warmup {warmup_steps}")

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# ----------------------------------------------------------------------------
# build everything
# ----------------------------------------------------------------------------
train_loader = DataLoaderLite(B, T, ddp_rank, ddp_world_size, 'train', data_root, verbose=master_process)
val_loader = DataLoaderLite(B, T, ddp_rank, ddp_world_size, 'val', data_root, verbose=master_process)

model = GPT(GPTConfig(**preset["cfg"])).to(device)
raw_model = model       
if master_process:
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

optimizer = raw_model.configure_optimizers(weight_decay, max_lr, device_type, verbose=master_process)
scaler = torch.amp.GradScaler('cuda')

# ----------------------------------------------------------------------------
# optional remote persistence (Hugging Face Hub): survives Kaggle session loss
# ----------------------------------------------------------------------------
hf_repo = os.environ.get("HF_REPO", "")
hf_token = os.environ.get("HF_TOKEN", "")
_upload_thread = None
print(os.environ["HF_REPO"])
_upload_thread = None


def hf_pull():
    """master only: download ckpt_last.pt (+ log.txt) from the repo if it exists."""
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=hf_token)
    if not api.file_exists(hf_repo, "ckpt_last.pt"):
        print(f"no checkpoint in {hf_repo} yet: starting fresh")
        return
    # any error here propagates on purpose: starting fresh by mistake would overwrite the remote checkpoint
    hf_hub_download(hf_repo, "ckpt_last.pt", token=hf_token, local_dir=log_dir)
    print(f"downloaded ckpt_last.pt from {hf_repo}")
    if api.file_exists(hf_repo, "log.txt"):
        hf_hub_download(hf_repo, "log.txt", token=hf_token, local_dir=log_dir)


def hf_push(final=False, include_final_model=False):
    """master only: upload the latest checkpoint in a background thread (each upload is an atomic commit)."""
    global _upload_thread
    if _upload_thread is not None and _upload_thread.is_alive():
        if not final:
            print("[upload] previous upload still running, skipping this one")
            return
        _upload_thread.join()
    snap = ckpt_path + ".upload"
    if os.path.exists(snap):
        os.remove(snap)
    os.link(ckpt_path, snap)      # stable inode: safe even if ckpt_last.pt is replaced mid-upload
    files = [(snap, "ckpt_last.pt"), (log_file, "log.txt")]
    if include_final_model:
        files.append((os.path.join(log_dir, "model_final.pt"), "model_final.pt"))

    def work():
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        try:
            for local, remote in files:
                api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=hf_repo)
            print(f"[upload] pushed {', '.join(r for _, r in files)} to {hf_repo}")
        except Exception as e:
            print(f"[upload failed] {e}")
            return
        finally:
            if os.path.exists(snap):
                os.remove(snap)
        try:
            api.super_squash_history(repo_id=hf_repo)   # drop older checkpoint versions to save storage
        except Exception as e:
            print(f"[squash skipped] {e}")

    _upload_thread = threading.Thread(target=work)
    _upload_thread.start()
    if final:
        _upload_thread.join()


if hf_repo:
    assert hf_token, "HF_REPO is set but HF_TOKEN is empty"
    if master_process:
        from huggingface_hub import HfApi
        HfApi(token=hf_token).create_repo(hf_repo, private=True, exist_ok=True)   # fails fast on a bad/read-only token
        if not os.path.exists(resume_path):
            hf_pull()
    if ddp:
        dist.barrier()

start_step = 0
if os.path.exists(resume_path):
    ckpt = torch.load(resume_path, map_location=device, weights_only=True)
    assert ckpt['config'] == asdict(raw_model.config), (
        f"checkpoint config {ckpt['config']} does not match MODEL_SIZE={model_size}; "
        f"use a different CKPT_DIR / HF_REPO for a different model")
    raw_model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scaler.load_state_dict(ckpt['scaler'])
    train_loader.load_state(ckpt['loader'])
    start_step = ckpt['step']
    del ckpt
    if master_process:
        print(f"resumed from {resume_path} at step {start_step}")
elif master_process:
    with open(log_file, "w"):
        pass

if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


def save_checkpoint(next_step, val_loss=None, final=False):
    ckpt = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'config': asdict(raw_model.config),
        'step': next_step,
        'loader': train_loader.state(),
        'val_loss': val_loss,
    }
    tmp = ckpt_path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, ckpt_path)                      # atomic: never leaves a half-written file
    if final:
        torch.save({'model': raw_model.state_dict(), 'config': asdict(raw_model.config)},
                   os.path.join(log_dir, "model_final.pt"))


sample_rng = torch.Generator(device=device)
sample_rng.manual_seed(42)

@torch.no_grad()
def generate(prompt, max_new_tokens=60, num_seqs=3, top_k=50, temperature=0.8):
    raw_model.eval()
    tokens = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device)
    tokens = tokens.unsqueeze(0).repeat(num_seqs, 1)
    for _ in range(max_new_tokens):
        with torch.autocast(device_type=device_type, dtype=torch.float16):
            logits, _ = raw_model(tokens[:, -raw_model.config.block_size:])
        logits = logits[:, -1, :].float() / temperature
        logits[:, 50257:] = float('-inf')            # padded vocab ids are not real tokens
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_idx = torch.topk(probs, top_k, dim=-1)
        ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
        tokens = torch.cat((tokens, torch.gather(topk_idx, -1, ix)), dim=1)
    raw_model.train()
    return [enc.decode(row.tolist()) for row in tokens]


# ----------------------------------------------------------------------------
# train
# ----------------------------------------------------------------------------
t_start = time.time()
last_ckpt_time = time.time()
val_loss_value = None

for step in range(start_step, max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # --- validation loss ---
    if step % val_every == 0 or last_step:
        raw_model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = torch.zeros(1, device=device)
            for _ in range(val_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.float16):
                    _, loss = raw_model(x, y)
                val_loss_accum += loss.detach() / val_steps
        raw_model.train()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        val_loss_value = val_loss_accum.item()
        if master_process:
            print(f"validation loss: {val_loss_value:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_value:.4f}\n")

    # --- sample text ---
    if master_process and ((step % sample_every == 0 and step > 0) or last_step):
        for i, s in enumerate(generate("The most important thing to understand about")):
            print(f"> sample {i}: {s!r}")

    # --- one optimizer step (gradient accumulation) ---
    optimizer.zero_grad(set_to_none=True)
    loss_accum = torch.zeros(1, device=device)
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if ddp:   # all-reduce gradients only on the last micro-step
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.float16):
            _, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        scaler.scale(loss).backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    scaler.unscale_(optimizer)                       # unscale BEFORE clipping
    norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    scaler.step(optimizer)
    scaler.update()
    if device_type == 'cuda':
        torch.cuda.synchronize()
    dt = time.time() - t0
    tok_per_sec = total_batch_size / dt

    if master_process:
        msg = (f"step {step:5d}/{max_steps} | loss {loss_accum.item():.6f} | lr {lr:.4e} | "
               f"norm {norm:.4f} | scale {scaler.get_scale():.0f} | dt {dt*1000:.0f}ms | tok/s {tok_per_sec:.0f}")
        if step == start_step + 1 and device_type == 'cuda':
            msg += f" | peak mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB"
        print(msg)
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

    # --- time limit (so a Kaggle session ends cleanly with a checkpoint) ---
    stop = torch.zeros(1, device=device)
    if max_hours > 0 and (time.time() - t_start) > max_hours * 3600:
        stop += 1
    if ddp:
        dist.all_reduce(stop, op=dist.ReduceOp.MAX)
    stop = stop.item() > 0

    # --- checkpoint ---
    ckpt_due = (time.time() - last_ckpt_time) > ckpt_every_min * 60
    if master_process and (ckpt_due or last_step or stop):
        save_checkpoint(step + 1, val_loss_value, final=last_step)
        last_ckpt_time = time.time()
        print(f"saved checkpoint at step {step + 1}")
        if hf_repo:
            hf_push(final=(last_step or stop), include_final_model=last_step)
    if stop:
        if master_process:
            print(f"MAX_HOURS reached: stopping at step {step + 1}. Re-run to resume.")
        break

if ddp:
    destroy_process_group()