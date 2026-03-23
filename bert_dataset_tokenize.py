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

MAX_LEN         = 512   # includes [CLS] at pos 0 and [SEP] at pos 511
STRIDE          = 256   # 50% overlap (applied to the inner content, not the full 512)
CHUNK_SIZE      = 10_000  # windows per .pt file
PARALLEL_SHARDS = 4

# ── Special token IDs (verified from vocab):
#   id 0 → <pad>   (PAD)
#   id 2 → <s>     (BOS / [CLS])
#   id 3 → </s>    (EOS / [SEP])
#   id 4 → [MASK]
#
# BERT convention  →  SentencePiece equivalent used here
#   [CLS]           →  <s>   (bos_id, id=2)
#   [SEP]           →  </s>  (eos_id, id=3)
#
# Every window is constructed as:
#   [CLS] + content_tokens (510 max) + [SEP]
# so MAX_LEN=512 always holds, and position 0 is always [CLS].


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

    PAD_ID = sp.pad_id()   # 0  → <pad>
    BOS_ID = sp.bos_id()   # 2  → <s>   used as [CLS]
    EOS_ID = sp.eos_id()   # 3  → </s>  used as [SEP]

    # The content budget per window: 512 − 1 ([CLS]) − 1 ([SEP]) = 510
    CONTENT_LEN = MAX_LEN - 2

    buffer   = []   # rolling buffer of raw content tokens (no BOS/EOS)
    chunk    = []
    chunk_id = 0

    in_path = f"{SHARD_DIR}/shard_{shard_id}.txt"

    with open(in_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Shard {shard_id}", position=shard_id % PARALLEL_SHARDS):
            line = line.strip()
            if not line:
                continue

            # Encode without BOS/EOS — we add them manually per-window
            tokens = sp.encode(line, out_type=int)
            buffer.extend(tokens)

            while len(buffer) >= CONTENT_LEN:
                content = buffer[:CONTENT_LEN]
                # Stride advances over content tokens only, not the special tokens
                buffer = buffer[STRIDE:]

                # Build the full 512-token window: [CLS] + content + [SEP]
                window = [BOS_ID] + content + [EOS_ID]   # length = 512

                chunk.append({
                    "input_ids":      torch.tensor(window, dtype=torch.long),
                    "attention_mask": torch.ones(MAX_LEN, dtype=torch.long),
                })

                if len(chunk) == CHUNK_SIZE:
                    out = f"{OUT_DIR}/s{shard_id}_c{chunk_id}.pt"
                    torch.save(chunk, out)
                    chunk.clear()
                    chunk_id += 1

    # Flush remainder — pad to MAX_LEN if anything is left in the buffer
    if buffer:
        content  = buffer[:CONTENT_LEN]   # cap at budget
        real_len = len(content)
        pad_len  = CONTENT_LEN - real_len

        # [CLS] + content + [SEP] + padding
        window = [BOS_ID] + content + [EOS_ID] + [PAD_ID] * pad_len
        attn   = [1] * (real_len + 2) + [0] * pad_len   # +2 for CLS and SEP

        chunk.append({
            "input_ids":      torch.tensor(window, dtype=torch.long),
            "attention_mask": torch.tensor(attn,   dtype=torch.long),
        })

    if chunk:
        out = f"{OUT_DIR}/s{shard_id}_c{chunk_id}.pt"
        torch.save(chunk, out)

    return shard_id


# ── Step 3: Tokenize all shards in parallel ────────────────────────────────────
def tokenize_all_shards():
    os.makedirs(OUT_DIR, exist_ok=True)

    shard_ids    = list(range(NUM_SHARDS))
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
    print(f"  Sequence len : {MAX_LEN}  (2 slots reserved for [CLS] / [SEP])")
    print(f"  Content len  : {MAX_LEN - 2}  (usable content tokens per window)")
    print(f"  Stride       : {STRIDE}  (over content tokens)")

    # Verify first sample of first file looks correct
    if pt_files:
        first_file = sorted(pt_files)[0]
        samples    = torch.load(first_file, weights_only=True)
        sample_ids = samples[0]["input_ids"].tolist()

        sp_check = spm.SentencePieceProcessor()
        sp_check.load(TOKENIZER_MODEL)
        BOS_ID = sp_check.bos_id()
        EOS_ID = sp_check.eos_id()

        first_tok = sp_check.id_to_piece(sample_ids[0])
        last_tok  = sp_check.id_to_piece(sample_ids[-1])
        print(f"\n  Spot-check on {os.path.basename(first_file)}, sample 0:")
        print(f"    Position   0 : id={sample_ids[0]}  piece='{first_tok}'  {'✓' if sample_ids[0] == BOS_ID else '✗ expected BOS id=' + str(BOS_ID)}")
        print(f"    Position 511 : id={sample_ids[-1]}  piece='{last_tok}'  {'✓' if sample_ids[-1] == EOS_ID else '✗ expected EOS id=' + str(EOS_ID)}")
        print(f"    First 10 pieces : {[sp_check.id_to_piece(t) for t in sample_ids[:10]]}")

    # Estimate total samples
    sample_count = 0
    for f in pt_files[:5]:
        sample_count += len(torch.load(f, weights_only=True))
    avg_per_file    = sample_count / min(5, len(pt_files))
    estimated_total = int(avg_per_file * len(pt_files))
    print(f"\n  Est. samples : ~{estimated_total:,}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    shard_text()
    tokenize_all_shards()