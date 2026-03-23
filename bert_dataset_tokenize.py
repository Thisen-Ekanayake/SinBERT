import os
import torch
import sentencepiece as spm
from tqdm import tqdm
from multiprocessing import Pool

# ── Config ─────────────────────────────────────────────────────────────────────
INPUT_TXT       = "combined.txt"
SHARD_DIR       = "text_shards"
NUM_SHARDS      = 100

TOKENIZER_MODEL = "tokenizer/unigram_32000_0.9995.model"
OUT_DIR         = "tokenized_chunks"

MAX_LEN         = 512   # doubled from 256 for BERT-base
STRIDE          = 256   # 50% overlap
CHUNK_SIZE      = 10_000  # windows per .pt file
PARALLEL_SHARDS = 4


# ── Step 1: Shard the raw text ─────────────────────────────────────────────────
def shard_text():
    os.makedirs(SHARD_DIR, exist_ok=True)

    print("Counting lines...")
    with open(INPUT_TXT, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    lines_per_shard = total_lines // NUM_SHARDS
    print(f"Total lines:      {total_lines:,}")
    print(f"Lines per shard:  {lines_per_shard:,}")

    writers = [
        open(f"{SHARD_DIR}/shard_{i}.txt", "w", encoding="utf-8")
        for i in range(NUM_SHARDS)
    ]

    with open(INPUT_TXT, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, total=total_lines, desc="Sharding")):
            shard_id = min(i // lines_per_shard, NUM_SHARDS - 1)
            writers[shard_id].write(line)

    for w in writers:
        w.close()

    print("✓ Text sharding complete\n")


# ── Step 2: Tokenize a single shard ───────────────────────────────────────────
def tokenize_text_shard(shard_id):
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL)

    PAD = sp.pad_id()
    EOS = sp.eos_id()

    buffer   = []
    chunk    = []
    chunk_id = 0

    in_path = f"{SHARD_DIR}/shard_{shard_id}.txt"

    with open(in_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Shard {shard_id}", position=shard_id % PARALLEL_SHARDS):
            line = line.strip()
            if not line:
                continue
            tokens = sp.encode(line, out_type=int)
            tokens.append(EOS)
            buffer.extend(tokens)

            while len(buffer) >= MAX_LEN:
                window = buffer[:MAX_LEN]
                buffer = buffer[STRIDE:]

                chunk.append({
                    "input_ids":      torch.tensor(window, dtype=torch.long),
                    "attention_mask": torch.ones(MAX_LEN, dtype=torch.long),
                })

                if len(chunk) == CHUNK_SIZE:
                    out = f"{OUT_DIR}/s{shard_id}_c{chunk_id}.pt"
                    torch.save(chunk, out)
                    chunk.clear()
                    chunk_id += 1

    # Flush remainder — pad to MAX_LEN if anything is left
    if buffer:
        pad_len = MAX_LEN - len(buffer)
        chunk.append({
            "input_ids":      torch.tensor(buffer + [PAD] * pad_len, dtype=torch.long),
            "attention_mask": torch.tensor([1] * len(buffer) + [0] * pad_len, dtype=torch.long),
        })

    if chunk:
        out = f"{OUT_DIR}/s{shard_id}_c{chunk_id}.pt"
        torch.save(chunk, out)

    return shard_id


# ── Step 3: Tokenize all shards in parallel ────────────────────────────────────
def tokenize_all_shards():
    os.makedirs(OUT_DIR, exist_ok=True)

    shard_ids = list(range(NUM_SHARDS))
    total_chunks = 0

    for i in range(0, NUM_SHARDS, PARALLEL_SHARDS):
        batch = shard_ids[i : i + PARALLEL_SHARDS]
        with Pool(PARALLEL_SHARDS) as p:
            p.map(tokenize_text_shard, batch)
        total_chunks += len(batch)
        print(f"  Completed shards {i}–{i + len(batch) - 1}  ({total_chunks}/{NUM_SHARDS})")

    # Report output stats
    import glob
    pt_files = glob.glob(f"{OUT_DIR}/*.pt")
    print(f"\n✓ Tokenization complete")
    print(f"  Output files : {len(pt_files):,}")
    print(f"  Sequence len : {MAX_LEN}")
    print(f"  Stride       : {STRIDE}")

    # Estimate total samples
    sample_count = 0
    for f in pt_files[:5]:
        sample_count += len(torch.load(f))
    avg_per_file = sample_count / min(5, len(pt_files))
    estimated_total = int(avg_per_file * len(pt_files))
    print(f"  Est. samples : ~{estimated_total:,}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    shard_text()
    tokenize_all_shards()