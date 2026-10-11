"""
Tokenize a ~300M-token slice of FineWeb-Edu (sample-10BT) into uint16 .npy shards.

Layout produced:
  shard 0        -> 10M tokens  (validation)
  shards 1..3    -> 100M tokens each (training, 300M total)

Uses streaming, so the full ~10B-token dataset is never downloaded.
Run on a CPU-only session to avoid burning GPU quota:
    python prep_data.py
"""
import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset  # pip install datasets
from tqdm import tqdm              # pip install tqdm

local_dir = os.environ.get("OUT_DIR", "edu_fineweb300M")
remote_name = "sample-10BT"
shard_size = int(1e8)                                  # train shard size (tokens)
val_size = int(1e7)                                    # validation shard size (tokens)
n_train_shards = int(os.environ.get("N_TRAIN_SHARDS", 3))
total_shards = 1 + n_train_shards

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]             # delimits documents


def tokenize(doc):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token id too large for uint16"
    return tokens_np.astype(np.uint16)


def shard_cap(i):
    return val_size if i == 0 else shard_size


def write_shard(i, arr):
    split = "val" if i == 0 else "train"
    path = os.path.join(local_dir, f"edufineweb_{split}_{i:06d}.npy")
    np.save(path, arr)


def main():
    os.makedirs(local_dir, exist_ok=True)
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train", streaming=True)

    nprocs = max(1, (os.cpu_count() or 2) - 1)
    buf = np.empty((shard_size,), dtype=np.uint16)
    shard_index, count = 0, 0
    bar = tqdm(total=shard_cap(0), unit="tokens", desc="Shard 0")
    done = False

    with mp.Pool(nprocs) as pool:
        for tokens in pool.imap(tokenize, fw, chunksize=16):
            while len(tokens) > 0:                      # one doc may span shards
                cap = shard_cap(shard_index)
                take = min(cap - count, len(tokens))
                buf[count:count + take] = tokens[:take]
                count += take
                bar.update(take)
                tokens = tokens[take:]
                if count == cap:
                    write_shard(shard_index, buf[:count])
                    bar.close()
                    shard_index += 1
                    count = 0
                    if shard_index == total_shards:
                        done = True
                        break
                    bar = tqdm(total=shard_cap(shard_index), unit="tokens", desc=f"Shard {shard_index}")
            if done:
                break

    if not done and count > 0:                          # dataset ended early
        write_shard(shard_index, buf[:count])
    print(f"done: wrote {shard_index if done else shard_index + 1} shards to {local_dir}/")


if __name__ == "__main__":
    main()