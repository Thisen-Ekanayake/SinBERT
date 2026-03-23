"""
verify_chunks.py
────────────────
Run this after bert_dataset_tokenize.py completes and before starting
(or resuming) BERT pre-training. It checks every .pt file in
tokenized_chunks/ and reports any structural issues.

Usage:
    python3 verify_chunks.py

All checks are printed with ✓ / ✗ / ⚠ prefixes. A final PASS / FAIL
summary is printed at the end.
"""

import os
import glob
import random
import math
import torch
import sentencepiece as spm
from tqdm import tqdm

# ── Config — must match bert_dataset_tokenize.py ───────────────────────────────
TOKENIZER_MODEL = "tokenizer/unigram_32000_0.9995.model"
CHUNK_DIR       = "tokenized_chunks"
MAX_LEN         = 512
N_FULL_SCAN     = None    # set to int (e.g. 50) to limit files scanned; None = all
N_DEEP_SAMPLES  = 2000    # number of random samples for deep token-level checks

# ── Load tokenizer ─────────────────────────────────────────────────────────────
sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_MODEL)

PAD_ID   = sp.pad_id()    # 0  → <pad>
BOS_ID   = sp.bos_id()    # 2  → <s>   ([CLS])
EOS_ID   = sp.eos_id()    # 3  → </s>  ([SEP])
MASK_ID  = sp.piece_to_id("[MASK]")
VOCAB_SIZE = sp.get_piece_size()

failures = []   # accumulate all ✗ findings

def fail(msg):
    failures.append(msg)
    print(f"  ✗  {msg}")

def ok(msg):
    print(f"  ✓  {msg}")

def warn(msg):
    print(f"  ⚠  {msg}")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 0 — Environment sanity
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 66)
print("CHECK 0 — Environment sanity")
print("═" * 66)

if not os.path.exists(TOKENIZER_MODEL):
    fail(f"Tokenizer not found: {TOKENIZER_MODEL}")
else:
    ok(f"Tokenizer loaded: {TOKENIZER_MODEL}")

pt_files = sorted(glob.glob(f"{CHUNK_DIR}/*.pt"))
if not pt_files:
    fail(f"No .pt files found in {CHUNK_DIR}")
    print("\n✗  FATAL — nothing to check. Aborting.")
    raise SystemExit(1)
else:
    ok(f"Found {len(pt_files):,} .pt files in {CHUNK_DIR}/")

print(f"\n  Special token IDs:")
print(f"    PAD   id={PAD_ID}  → '{sp.id_to_piece(PAD_ID)}'")
print(f"    BOS   id={BOS_ID}  → '{sp.id_to_piece(BOS_ID)}'  (used as [CLS])")
print(f"    EOS   id={EOS_ID}  → '{sp.id_to_piece(EOS_ID)}'  (used as [SEP])")
print(f"    MASK  id={MASK_ID} → '{sp.id_to_piece(MASK_ID)}'")
print(f"    VOCAB_SIZE = {VOCAB_SIZE}")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 1 — File integrity (loadable, non-empty, correct type)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 66)
print("CHECK 1 — File integrity")
print("═" * 66)

scan_files   = pt_files if N_FULL_SCAN is None else pt_files[:N_FULL_SCAN]
total_samples = 0
corrupt_files = []
empty_files   = []

for path in tqdm(scan_files, desc="Scanning files"):
    try:
        data = torch.load(path, weights_only=True)
    except Exception as e:
        corrupt_files.append((path, str(e)))
        continue

    if not isinstance(data, list) or len(data) == 0:
        empty_files.append(path)
        continue

    total_samples += len(data)

if corrupt_files:
    for path, err in corrupt_files:
        fail(f"Corrupt file: {os.path.basename(path)} — {err}")
else:
    ok(f"All {len(scan_files)} scanned files loaded without errors")

if empty_files:
    for path in empty_files:
        fail(f"Empty file: {os.path.basename(path)}")
else:
    ok(f"No empty files found")

print(f"\n  Total samples across scanned files : {total_samples:,}")
print(f"  Avg samples per file               : {total_samples / len(scan_files):,.0f}")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 2 — Sequence length
# Every sample must have input_ids and attention_mask of exactly MAX_LEN.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 66)
print(f"CHECK 2 — Sequence length (expected {MAX_LEN} for all samples)")
print("═" * 66)

wrong_len_ids  = 0
wrong_len_attn = 0
missing_keys   = 0

for path in tqdm(scan_files, desc="Checking lengths"):
    data = torch.load(path, weights_only=True)
    for sample in data:
        if "input_ids" not in sample or "attention_mask" not in sample:
            missing_keys += 1
            continue
        if sample["input_ids"].shape[0] != MAX_LEN:
            wrong_len_ids += 1
        if sample["attention_mask"].shape[0] != MAX_LEN:
            wrong_len_attn += 1

if missing_keys:
    fail(f"{missing_keys:,} samples missing 'input_ids' or 'attention_mask' key")
else:
    ok("All samples have both required keys")

if wrong_len_ids:
    fail(f"{wrong_len_ids:,} samples have input_ids length ≠ {MAX_LEN}")
else:
    ok(f"All input_ids are exactly length {MAX_LEN}")

if wrong_len_attn:
    fail(f"{wrong_len_attn:,} samples have attention_mask length ≠ {MAX_LEN}")
else:
    ok(f"All attention_masks are exactly length {MAX_LEN}")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 3 — [CLS] / [SEP] boundary tokens
# Position 0 must be BOS_ID ([CLS]).
# The last non-padding token must be EOS_ID ([SEP]).
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 66)
print("CHECK 3 — [CLS] / [SEP] boundary tokens")
print("═" * 66)

# Build a flat index of all samples for random sampling
all_sample_refs = []   # list of (file_path, sample_index)
for path in scan_files:
    data = torch.load(path, weights_only=True)
    for i in range(len(data)):
        all_sample_refs.append((path, i))

# Cache loaded files to avoid redundant disk reads
_file_cache = {}
def get_sample(path, idx):
    if path not in _file_cache:
        _file_cache[path] = torch.load(path, weights_only=True)
    return _file_cache[path][idx]

deep_refs = random.sample(all_sample_refs, min(N_DEEP_SAMPLES, len(all_sample_refs)))

missing_cls = 0
missing_sep = 0
sep_in_wrong_place = 0   # EOS appears before the last real token

for path, idx in tqdm(deep_refs, desc="Checking boundaries"):
    sample  = get_sample(path, idx)
    ids     = sample["input_ids"].tolist()
    attn    = sample["attention_mask"].tolist()
    seq_len = sum(attn)   # number of real (non-padding) tokens

    if ids[0] != BOS_ID:
        missing_cls += 1

    if seq_len > 0 and ids[seq_len - 1] != EOS_ID:
        missing_sep += 1

    # EOS must not appear anywhere except the last real position
    for pos in range(seq_len - 1):
        if ids[pos] == EOS_ID:
            sep_in_wrong_place += 1
            break

checked = len(deep_refs)
print(f"  Checked {checked:,} random samples")

if missing_cls:
    fail(f"[CLS] missing at position 0   : {missing_cls:,} / {checked:,} ({100*missing_cls/checked:.1f}%)")
else:
    ok(f"[CLS] present at position 0 in all {checked:,} samples")

if missing_sep:
    fail(f"[SEP] missing at last real pos : {missing_sep:,} / {checked:,} ({100*missing_sep/checked:.1f}%)")
else:
    ok(f"[SEP] present at last real pos in all {checked:,} samples")

if sep_in_wrong_place:
    warn(f"[SEP] also appears mid-sequence in {sep_in_wrong_place:,} / {checked:,} samples — may indicate sentence boundaries crossing windows")
else:
    ok(f"[SEP] does not appear mid-sequence in any checked sample")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 4 — Padding consistency
# Padding tokens must have attention_mask = 0.
# Non-padding tokens must have attention_mask = 1.
# No padding should appear before real tokens (left-padding not expected).
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 66)
print("CHECK 4 — Padding consistency")
print("═" * 66)

pad_mask_mismatch   = 0   # PAD token but attn=1, or non-PAD but attn=0
left_padded         = 0   # padding before real tokens
samples_with_pad    = 0
fully_packed        = 0

for path, idx in tqdm(deep_refs, desc="Checking padding"):
    sample = get_sample(path, idx)
    ids    = sample["input_ids"].tolist()
    attn   = sample["attention_mask"].tolist()

    has_pad = PAD_ID in ids
    if has_pad:
        samples_with_pad += 1
    else:
        fully_packed += 1

    # Check alignment between PAD positions and attn=0 positions
    for tok, mask in zip(ids, attn):
        if tok == PAD_ID and mask != 0:
            pad_mask_mismatch += 1
            break
        if tok != PAD_ID and mask == 0:
            pad_mask_mismatch += 1
            break

    # Check for left-padding (attn=0 before attn=1)
    seen_real = False
    for mask in attn:
        if mask == 1:
            seen_real = True
        if seen_real and mask == 0:
            break   # trailing pad — fine
    else:
        pass
    # Simpler left-pad check: first token should never be PAD
    if ids[0] == PAD_ID:
        left_padded += 1

if pad_mask_mismatch:
    fail(f"attention_mask / PAD_ID mismatch in {pad_mask_mismatch:,} samples")
else:
    ok(f"attention_mask correctly aligned with PAD_ID in all checked samples")

if left_padded:
    fail(f"Left-padding detected in {left_padded:,} samples (position 0 = PAD)")
else:
    ok(f"No left-padding detected")

print(f"\n  Fully packed samples (no padding) : {fully_packed:,} / {checked:,} ({100*fully_packed/checked:.1f}%)")
print(f"  Samples with trailing padding     : {samples_with_pad:,} / {checked:,} ({100*samples_with_pad/checked:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 5 — Vocabulary range
# All token IDs must be in [0, VOCAB_SIZE).
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 66)
print(f"CHECK 5 — Token ID vocabulary range [0, {VOCAB_SIZE})")
print("═" * 66)

out_of_range_samples = 0
unk_heavy_samples    = 0   # >5% UNK tokens — may indicate encoding issues
UNK_ID               = sp.unk_id()
UNK_THRESHOLD        = 0.05

for path, idx in tqdm(deep_refs, desc="Checking vocab range"):
    sample = get_sample(path, idx)
    ids    = sample["input_ids"]
    attn   = sample["attention_mask"].tolist()
    seq_len = sum(attn)

    if ids.min().item() < 0 or ids.max().item() >= VOCAB_SIZE:
        out_of_range_samples += 1

    real_ids  = ids[:seq_len].tolist()
    unk_count = real_ids.count(UNK_ID)
    if seq_len > 0 and unk_count / seq_len > UNK_THRESHOLD:
        unk_heavy_samples += 1

if out_of_range_samples:
    fail(f"Token IDs out of range in {out_of_range_samples:,} samples")
else:
    ok(f"All token IDs are within [0, {VOCAB_SIZE})")

if unk_heavy_samples:
    warn(f"{unk_heavy_samples:,} samples have >{UNK_THRESHOLD*100:.0f}% UNK tokens — check source text encoding")
else:
    ok(f"No samples have excessive UNK tokens (>{UNK_THRESHOLD*100:.0f}% threshold)")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 6 — Visual decode spot-check
# Decode 5 random samples so you can eyeball the output.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 66)
print("CHECK 6 — Visual decode spot-check (5 random samples)")
print("═" * 66)

spot_refs = random.sample(all_sample_refs, min(5, len(all_sample_refs)))
for i, (path, idx) in enumerate(spot_refs):
    sample  = get_sample(path, idx)
    ids     = sample["input_ids"].tolist()
    attn    = sample["attention_mask"].tolist()
    seq_len = sum(attn)

    first_pieces = [sp.id_to_piece(t) for t in ids[:8]]
    last_pieces  = [sp.id_to_piece(t) for t in ids[max(0, seq_len - 4):seq_len]]
    cls_ok  = "✓" if ids[0] == BOS_ID else "✗"
    sep_ok  = "✓" if ids[seq_len - 1] == EOS_ID else "✗"

    print(f"\n  Sample {i+1}  |  file={os.path.basename(path)}  idx={idx}  seq_len={seq_len}")
    print(f"    First 8 tokens : {first_pieces}")
    print(f"    Last  4 tokens : {last_pieces}")
    print(f"    [CLS] at pos 0 : {cls_ok}   [SEP] at last real pos : {sep_ok}")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 66)
print("FINAL SUMMARY")
print("═" * 66)
print(f"  Files scanned   : {len(scan_files):,} / {len(pt_files):,}")
print(f"  Samples checked : {total_samples:,}  (length/key checks)")
print(f"  Deep-checked    : {checked:,}  (boundary/padding/vocab checks)")

expected_initial_loss = math.log(VOCAB_SIZE)
print(f"\n  Expected initial MLM loss after fix : ~{expected_initial_loss:.2f}  (= ln({VOCAB_SIZE}))")

if failures:
    print(f"\n  ✗  FAIL — {len(failures)} issue(s) found:")
    for f in failures:
        print(f"       • {f}")
    print("\n  Fix the issues above, delete tokenized_chunks/, and re-run")
    print("  bert_dataset_tokenize.py before starting training.")
else:
    print(f"\n  ✓  PASS — all checks passed.")
    print("  Tokenized chunks look correct. Safe to start/resume training.")
print("═" * 66 + "\n")