# GPT-2 From Scratch: High-Throughput Training on T4 GPUs

A from-scratch PyTorch implementation of the GPT-2 decoder-only transformer, engineered for fast and memory-efficient training on commodity hardware (a single Colab T4 and Kaggle's 2× T4). The project covers the full path from a straightforward fp32 baseline to a multi-GPU, resumable training pipeline, and documents the measured effect of each optimization.

## Highlights

- **≈5× single-GPU speedup** (≈3.4–4.2k → ≈18–20k tokens/s on one T4) from fp16 mixed precision with dynamic loss scaling, `torch.compile`, fused AdamW, and a tensor-core-friendly vocabulary size.
- **≈2× further speedup with DDP** on 2× T4 (≈37–41k tokens/s), with near-linear scaling, because gradients are synchronized only once per optimizer step.
- **≈10× end-to-end** versus the fp32 baseline, on hardware with no TF32 or bf16 support.
- **Large effective batches** (262k–524k tokens per step) on 16 GB GPUs through gradient accumulation.
- **Production-style training loop:** sharded token pipeline, validation loss, text sampling, resumable checkpoints, and optional remote persistence on the Hugging Face Hub.
- **Two model presets:** GPT-2 small (124M) and a ~304M-parameter configuration, switchable with one environment variable.

## Model

Decoder-only transformer following the GPT-2 design:

| Component | Implementation |
|---|---|
| Attention | Multi-head causal self-attention using `F.scaled_dot_product_attention(is_causal=True)` (fused, memory-efficient kernel) |
| MLP | 4× expansion with tanh-approximated GELU |
| Normalization | Pre-LayerNorm blocks with a final LayerNorm |
| Embeddings | Learned token and position embeddings; input and output embeddings are weight-tied |
| Initialization | Normal(0, 0.02); residual projections scaled by `1/√(2·n_layer)` |
| Context length | 1024 tokens |
| Tokenizer | GPT-2 BPE (`tiktoken`), vocabulary padded from 50,257 to 50,304 |

Presets, selected with `MODEL_SIZE`:

- **`124m`:** 12 layers, 12 heads, 768 dim, ~124.5M parameters. Peak LR 6e-4, micro-batch 16 per GPU.
- **`300m`:** 20 layers, 16 heads, 1024 dim, ~304.5M parameters. Peak LR 3e-4, micro-batch 4 per GPU.

## Performance Engineering

Each optimization targets a specific bottleneck. The T4 (Turing, 70 W) has fp16 tensor cores but no TF32 or bf16 support, so the usual Ampere-class shortcuts do not apply. Setting `torch.set_float32_matmul_precision('high')` had no effect on this hardware.

### 1. fp16 mixed precision with `GradScaler`
`torch.autocast(dtype=torch.float16)` runs matmuls and attention on the fp16 tensor cores (about 65 TFLOPS peak on a T4, versus about 8 TFLOPS for fp32), while precision-sensitive operations (softmax, LayerNorm, cross-entropy) and the master weights stay in fp32. A `GradScaler` multiplies the loss before the backward pass so small gradients do not underflow in fp16. Gradients are unscaled before clipping, so the gradient norm is computed on true values. Steps with inf/nan gradients are skipped automatically. Over the benchmark runs the scale stayed at its initial value with no skipped steps.

### 2. `torch.compile`
Fuses chains of elementwise operations (GELU, LayerNorm, residual adds) into single kernels, which cuts memory traffic and kernel-launch overhead. This costs a one-time compilation of about 27–37 s on the first step.

### 3. Fused AdamW
A single fused CUDA kernel updates all parameters instead of looping over them. Weight decay applies only to tensors with two or more dimensions (matmul weights and embeddings), not to biases or LayerNorm parameters.

### 4. Vocabulary padding
The vocabulary is padded from 50,257 to 50,304 (a multiple of 128) so the large output projection and the loss use tensor-core-friendly shapes. The padding ids are masked during sampling.

### 5. Gradient accumulation
The effective batch size is decoupled from GPU memory. A target batch of 262,144 or 524,288 tokens is reached by accumulating gradients over several micro-batches, and the loss is scaled by `1/grad_accum_steps` so the accumulated gradient is a proper mean.

### 6. Distributed Data Parallel (DDP)
- One process per GPU, launched with `torchrun`.
- A rank-aware data loader: each rank reads an interleaved, non-overlapping slice of the token stream.
- Gradient all-reduce is performed **only on the last micro-step** (`require_backward_grad_sync`), so the interconnect cost, which goes over PCIe because T4s have no NVLink, is paid once per optimizer step rather than once per micro-batch.
- The logged loss is averaged across ranks, and logging, sampling and checkpointing run on the master process only.

### 7. Optimization schedule
AdamW (β = 0.9 / 0.95, ε = 1e-8, weight decay 0.1), global gradient-norm clipping at 1.0, and a learning-rate schedule of linear warmup followed by cosine decay to 10% of the peak.

## Results

Benchmarks run 50 optimizer steps on a small text corpus (338,025 tokens), timed with `torch.cuda.synchronize()`. Step 0 includes `torch.compile` and is excluded from the ranges below. Because the corpus is tiny, loss values are only a sanity check, not a quality measure.

### Table 1: Single T4, fp32 baseline vs. optimized (batch 12 × 512)

| | fp32 baseline | Optimized (fp16 + compile + fused AdamW + padded vocab) |
|---|---|---|
| Step time | ≈1.47–1.80 s | ≈309–341 ms |
| Throughput | ≈3.4–4.2k tokens/s | ≈18.0–19.9k tokens/s |
| Speedup | 1× | **≈4.7–5.3×** |
| Step 0 | ≈3.0 s | ≈33 s (includes compilation) |
| Loss at step 49 | ≈6.33 | ≈6.25 |

The optimizations were applied together, so the speedup is a combined figure and the contribution of each was not isolated. The loss trajectories have the same shape (≈11 → ≈6.3 in 50 steps), so mixed precision did not visibly affect convergence. The starting loss differs slightly (10.97 vs 10.94) because of the padded vocabulary and fp16.

<details>
<summary>Screenshots: baseline and optimized runs</summary>

**Before (fp32, batch 12 × 512)**

![Before speedup](assets/Before_speedup.PNG)

**After (fp16 + compile + fused AdamW, batch 12 × 512)**

![After speedup](assets/After_speedup.PNG)

</details>

### Table 2: Single T4 after optimization, effect of batch and sequence length

| Micro-batch × sequence | Tokens per step | Step time | Throughput |
|---|---|---|---|
| 12 × 512 | 6,144 | ≈309–341 ms | ≈18.0–19.9k tokens/s |
| 16 × 512 | 8,192 | ≈400–428 ms | ≈19.1–20.5k tokens/s |
| 16 × 1024 | 16,384 | ≈787–884 ms | ≈18.5–20.8k tokens/s |

Throughput stays flat at roughly 19–20k tokens/s as the batch grows, which shows the T4 is compute-bound after the optimizations. Larger batches therefore buy more tokens per step and full-length (1024) training contexts, not extra speed. Step time drifts up by about 10% over a run, consistent with T4 thermal and power throttling.

<details>
<summary>Screenshots: batch and sequence-length runs</summary>

**Batch 16 × 512**

![Batch 16 x 512](assets/After_speedup_B16_T512.PNG)

**Batch 16 × 1024**

![Batch 16 x 1024](assets/After_speedup_B16_T1024.PNG)

</details>

### Table 3: Multi-GPU scaling and model size

| | 1× T4, 124M | 2× T4 (DDP), 124M | 2× T4 (DDP), 304M |
|---|---|---|---|
| Micro-batch per GPU | 16 × 1024 | 16 × 1024 | 4 × 1024 |
| Gradient accumulation | none | 16 steps | 32 steps |
| Tokens per optimizer step | 16,384 | 524,288 | 262,144 |
| Step time | ≈0.79–0.88 s | ≈12.6–14.2 s | ≈15–17 s |
| Total throughput | ≈18.5–20.8k tokens/s | ≈37–41k tokens/s | ≈15.3–17.2k tokens/s |
| Throughput per GPU | ≈18.5–20.8k | ≈19–20k | ≈7.7–8.6k |
| Scaling vs. 1 GPU | 1× | **≈2×** (near-linear) | n/a |
| Peak GPU memory | n/a | n/a | 11.1 GB per GPU |

Per-GPU throughput is unchanged under DDP, so the communication overhead is negligible. The 304M model processes fewer tokens per second because each token costs about 2.4× the compute (≈2.08 vs ≈0.86 GFLOP per token, forward and backward). Estimated sustained compute is about 17 TFLOP/s per GPU in both cases, so the GPUs are equally busy and the lower token rate is expected.

## Training Pipeline

Beyond the benchmark loop, the training script is set up for long runs on time-limited free GPU sessions:

- **Data:** FineWeb-Edu (`sample-10BT`) streamed from the Hugging Face Hub, tokenized with GPT-2 BPE into `uint16` `.npy` shards (100M tokens per training shard, plus a 10M-token validation shard). The default subset is 300M tokens, so the full dataset is never downloaded.
- **Evaluation:** validation loss every 250 steps, and sample text generated every 500 steps from the uncompiled model.
- **Checkpoint and resume:** a checkpoint every 30 minutes holds the weights, optimizer state, loss-scaler state, step, and exact data-loader position, so a resumed run continues from the same point. Writes are atomic.
- **Remote persistence:** with `HF_REPO` set, each checkpoint is uploaded in the background to a private Hugging Face repo, and a fresh session downloads the latest checkpoint automatically if none exists locally.
- **Session safety:** `MAX_HOURS` stops training cleanly with a final checkpoint, and a mismatched checkpoint (wrong model size) is rejected with a clear error.

## Repository Layout

```
.
├── prep_data.py             # streams and tokenizes FineWeb-Edu into .npy shards
├── train_gpt2_300m.py       # model, data loader, DDP training loop (supports both presets)
├── assets/                  # benchmark screenshots
└── README.md
```

`train_gpt2_300m.py` runs either preset despite its name; select one with `MODEL_SIZE`.

## Usage

**Install**
```bash
pip install torch tiktoken numpy datasets tqdm huggingface_hub
```

**Prepare data** (about 300M tokens by default)
```bash
python prep_data.py
```

**Train on 2 GPUs**
```bash
MODEL_SIZE=124m DATA_ROOT=edu_fineweb300M CKPT_DIR=log124 \
torchrun --standalone --nproc_per_node=2 train_gpt2_300m.py
```

**Resumable run with Hugging Face persistence**
```bash
export HF_TOKEN=...                      # token with write access
export HF_REPO=<username>/gpt124m-ckpt   # created as a private repo if missing
MODEL_SIZE=124m MAX_HOURS=10.5 DATA_ROOT=edu_fineweb300M CKPT_DIR=log124 \
torchrun --standalone --nproc_per_node=2 train_gpt2_300m.py
```

Re-running the same command resumes from the latest checkpoint.

**Configuration (environment variables)**
- `MODEL_SIZE`: `124m` or `300m` (default `300m`).
- `DATA_ROOT`: folder with the `.npy` shards.
- `TOTAL_TOKENS`: token budget that sets the length of the learning-rate schedule (default 3e8).
- `MICRO_BATCH`: per-GPU micro-batch. Lower it if you run out of memory; gradient accumulation adjusts automatically.
- `MAX_HOURS`: stop gracefully after this many hours.
- `CKPT_DIR`, `CKPT_EVERY_MIN`, `RESUME_FROM`: checkpoint location, interval, and an explicit resume path.
- `HF_REPO`, `HF_TOKEN`: remote checkpoint storage.

## Notes and Limitations

- **Hardware-specific choices.** fp16 with a loss scaler is used because the T4 lacks native bf16 and TF32. On Ampere or newer GPUs, bf16 would remove the need for loss scaling.
- **Combined speedup.** The single-GPU optimizations were applied together, so Table 1 reports their combined effect rather than per-optimization gains.
- **Benchmark scale.** Throughput numbers come from short runs. Step time drifts by about 10% over a run on these GPUs, so the figures are given as ranges.
- **Multi-GPU results** were measured on Kaggle's 2× T4 with PCIe interconnect. Scaling on other topologies will differ.

## Possible Next Steps

- KV-cache for faster autoregressive inference.
- Lightweight fine-tuning (LoRA) for domain or style adaptation.
- Longer pretraining runs to improve text quality.