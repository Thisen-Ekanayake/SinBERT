import os
import glob
import random
import torch
from torch.utils.data import Dataset, random_split
from transformers import (
    BertConfig,
    BertForMaskedLM,
    Trainer,
    TrainingArguments,
)
import sentencepiece as spm
from tqdm import tqdm
import wandb
import math


# ── Config ─────────────────────────────────────────────────────────────────────
TOKENIZER_MODEL    = "tokenizer/unigram_32000_0.9995.model"
CHUNK_DIR          = "tokenized_chunks"
CONFIG_FILE        = "bert_config.json"
CHECKPOINT_DIR     = "bert_checkpoints"
FINAL_MODEL_DIR    = "bert_final_model"
WANDB_RUN_ID_FILE  = "wandb_run_id.txt"

MAX_FILES          = None      # set to int to cap dataset size for testing
TRAIN_RATIO        = 0.9


# ── Verify paths ───────────────────────────────────────────────────────────────
assert os.path.exists(TOKENIZER_MODEL), f"Tokenizer not found: {TOKENIZER_MODEL}"
assert os.path.exists(CHUNK_DIR),       f"Chunk dir not found: {CHUNK_DIR}"
assert os.path.exists(CONFIG_FILE),     f"Config not found: {CONFIG_FILE}"
print("✓ All paths verified")


# ── Tokenizer ──────────────────────────────────────────────────────────────────
sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_MODEL)

PAD_ID     = sp.pad_id()
MASK_ID    = sp.piece_to_id("[MASK]")
VOCAB_SIZE = sp.get_piece_size()

print(f"Vocab size : {VOCAB_SIZE}")
print(f"PAD_ID     : {PAD_ID}")
print(f"MASK_ID    : {MASK_ID}")


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 1 — Special token identity check
# Verifies that IDs 0-5 resolve to the expected special tokens, and that
# [MASK] / [PAD] map to sensible IDs. A wrong MASK_ID is the most common
# cause of loss ≈ 2× ln(vocab_size) at initialisation.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("DIAGNOSTIC 1 — Special token identity check")
print("═" * 60)

EXPECTED_SPECIAL = {
    "[PAD]":   0,
    "[UNK]":   1,
    "[CLS]":   2,
    "[SEP]":   3,
    "[MASK]":  4,
}

diag1_passed = True
for piece, expected_id in EXPECTED_SPECIAL.items():
    actual_id   = sp.piece_to_id(piece)
    back_piece  = sp.id_to_piece(actual_id)
    status = "✓" if actual_id == expected_id else "✗ MISMATCH"
    if actual_id != expected_id:
        diag1_passed = False
    print(f"  {status}  piece_to_id('{piece}') = {actual_id}  |  id_to_piece({actual_id}) = '{back_piece}'")

# Also print first 8 IDs for a raw inspection
print("\n  First 8 token IDs in vocabulary:")
for i in range(8):
    print(f"    id {i:>2} → '{sp.id_to_piece(i)}'")

expected_initial_loss = math.log(VOCAB_SIZE)
print(f"\n  Expected initial MLM loss  : {expected_initial_loss:.3f}  (= ln({VOCAB_SIZE}))")
print(f"  Observed initial loss was  : ~20.59  (reported from run)")
print(f"  Ratio observed / expected  : {20.59 / expected_initial_loss:.2f}×")
if not diag1_passed:
    print("\n  ⚠ MASK_ID or another special token is misaligned — this is very")
    print("    likely the root cause of the inflated loss. Fix the tokenizer")
    print("    special-token assignments before continuing.")
else:
    print("\n  ✓ Special token IDs look correct.")
print("═" * 60 + "\n")


# ── Dataset (full in-memory) ───────────────────────────────────────────────────
# Load everything into RAM up front. With 4.3M samples at 512 tokens each
# this is ~35GB — well within the MI300X's 192GB HBM. Workers can then serve
# batches from RAM with zero disk I/O, keeping the GPU fully fed.
class BertChunkDataset(Dataset):
    def __init__(self, chunk_dir, max_files=None):
        files = sorted(glob.glob(f"{chunk_dir}/*.pt"))
        if max_files:
            files = files[:max_files]
        assert len(files) > 0, f"No .pt files found in {chunk_dir}"

        self.data = []
        print(f"Loading {len(files)} files into memory...")
        for path in tqdm(files, desc="Loading"):
            self.data.extend(torch.load(path, weights_only=True))

        random.shuffle(self.data)
        print(f"✓ Loaded {len(self.data):,} samples across {len(files)} files")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids":      item["input_ids"],
            "attention_mask": item["attention_mask"],
        }


dataset = BertChunkDataset(CHUNK_DIR, max_files=MAX_FILES)

sample = dataset[0]["input_ids"]
print(sample[:50])


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 2 — Chunk boundary token check ([CLS] / [SEP])
# BERT expects every sequence to start with [CLS] and end with [SEP].
# If chunks were tokenized without them, positional context is broken and
# the model never learns proper sentence-level representations.
# ══════════════════════════════════════════════════════════════════════════════
print("═" * 60)
print("DIAGNOSTIC 2 — Chunk boundary token check ([CLS] / [SEP])")
print("═" * 60)

CLS_ID = sp.piece_to_id("[CLS]")
SEP_ID = sp.piece_to_id("[SEP]")
N_CHECK = min(200, len(dataset))

missing_cls = 0
missing_sep = 0
for i in range(N_CHECK):
    ids = dataset[i]["input_ids"].tolist()
    # Find the last non-padding token
    attn = dataset[i]["attention_mask"].tolist()
    seq_len = sum(attn)
    if ids[0] != CLS_ID:
        missing_cls += 1
    if ids[seq_len - 1] != SEP_ID:
        missing_sep += 1

print(f"  Checked {N_CHECK} samples")
print(f"  Missing [CLS] at position 0 : {missing_cls} / {N_CHECK}")
print(f"  Missing [SEP] at last token  : {missing_sep} / {N_CHECK}")

# Print the decoded first 10 tokens of 3 samples for a visual sanity check
print("\n  Decoded first 10 tokens of 3 random samples:")
for i in random.sample(range(len(dataset)), 3):
    ids    = dataset[i]["input_ids"].tolist()
    pieces = [sp.id_to_piece(t) for t in ids[:10]]
    print(f"    sample {i:>7}: {pieces}")

if missing_cls > 0 or missing_sep > 0:
    print("\n  ⚠ Some samples are missing [CLS] or [SEP] boundary tokens.")
    print("    Re-tokenize chunks with these tokens inserted at the boundaries.")
else:
    print("\n  ✓ All checked samples have correct [CLS] / [SEP] boundaries.")
print("═" * 60 + "\n")


# ── MLM Collator ───────────────────────────────────────────────────────────────
# (Assumed to be defined later in the full script — the diagnostic below
#  requires a `collator` object. Replace this stub with your actual collator.)
try:
    collator  # noqa: F821 — already defined earlier in the full script
except NameError:
    from transformers import DataCollatorForLanguageModeling
    collator = DataCollatorForLanguageModeling(
        tokenizer=None,   # placeholder — swap for your real collator
        mlm=True,
        mlm_probability=0.15,
    )
    print("  ℹ  Using a placeholder DataCollatorForLanguageModeling for diagnostics.")
    print("     Replace with your actual collator instance if different.\n")


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 3 — MLM collator label sanity check
# Confirms that:
#   (a) padding positions have label = -100  (not PAD_ID)
#   (b) non-masked positions have label = -100
#   (c) masked positions have valid vocabulary IDs as labels
#   (d) the masking rate is close to 15%
# Loss inflated by ~2× often means padding positions are included in the loss.
# ══════════════════════════════════════════════════════════════════════════════
print("═" * 60)
print("DIAGNOSTIC 3 — MLM collator label sanity check")
print("═" * 60)

BATCH_SIZE_DIAG = 4
diag_samples    = [dataset[i] for i in range(BATCH_SIZE_DIAG)]

try:
    batch = collator(diag_samples)

    input_ids = batch["input_ids"]   # (B, seq_len)
    labels    = batch["labels"]      # (B, seq_len)
    seq_len   = input_ids.shape[1]
    total_tok = labels.numel()

    # ── (a) Padding label check ────────────────────────────────────────────
    pad_positions  = (input_ids == PAD_ID)
    pad_not_masked = (labels[pad_positions] != -100).sum().item()
    print(f"  (a) Padding positions with label ≠ -100 : {pad_not_masked}")
    if pad_not_masked > 0:
        print("      ⚠ Collator is including padding in the loss — set these labels to -100!")
    else:
        print("      ✓ Padding positions correctly excluded from loss.")

    # ── (b) / (c) Active label positions ──────────────────────────────────
    active_mask  = (labels != -100)
    n_active     = active_mask.sum().item()
    mask_rate    = 100.0 * n_active / total_tok
    print(f"\n  (b/c) Active (masked) label positions : {n_active} / {total_tok} ({mask_rate:.1f}%)")
    if abs(mask_rate - 15.0) > 3.0:
        print(f"      ⚠ Masking rate {mask_rate:.1f}% is far from expected 15%.")
    else:
        print(f"      ✓ Masking rate is within 3% of target 15%.")

    # ── (d) Label vocabulary range check ──────────────────────────────────
    active_labels = labels[active_mask]
    out_of_range  = ((active_labels < 0) | (active_labels >= VOCAB_SIZE)).sum().item()
    print(f"\n  (d) Label IDs out of vocab range [0, {VOCAB_SIZE}) : {out_of_range}")
    if out_of_range > 0:
        print("      ⚠ Some label IDs are invalid — check collator masking logic.")
    else:
        print("      ✓ All label IDs are within vocabulary range.")

    # ── Sample label decode ────────────────────────────────────────────────
    print("\n  Decoded label tokens from first sample (first 20 positions):")
    for pos, (inp, lbl) in enumerate(zip(input_ids[0][:20].tolist(), labels[0][:20].tolist())):
        if lbl != -100:
            inp_piece = sp.id_to_piece(inp)
            lbl_piece = sp.id_to_piece(lbl)
            print(f"    pos {pos:>3}: input='{inp_piece}' (id={inp})  label='{lbl_piece}' (id={lbl})")

except Exception as e:
    print(f"  ✗ Could not run collator diagnostic: {e}")
    print("    Ensure `collator` is your actual MLM collator and accepts a list of dataset samples.")

print("═" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 4 — Learning rate warmup progression check
# At step 122 the LR was 2.182e-06, suggesting very slow warmup. Prints the
# expected LR at several checkpoints so you can confirm warmup_steps is sane.
# ══════════════════════════════════════════════════════════════════════════════
print("═" * 60)
print("DIAGNOSTIC 4 — Learning rate warmup progression check")
print("═" * 60)

# Fill in your actual TrainingArguments values here
TARGET_LR    = 1e-4       # peak learning_rate
WARMUP_STEPS = 10000      # warmup_steps (or derived from warmup_ratio)
TOTAL_STEPS  = 45372      # approximate from your run (45372 steps shown in progress bar)

print(f"  Target LR       : {TARGET_LR:.2e}")
print(f"  Warmup steps    : {WARMUP_STEPS}")
print(f"  Total steps     : {TOTAL_STEPS}")
print()
print(f"  {'Step':>8}  {'Expected LR':>14}  {'% of warmup':>12}")
print(f"  {'-'*8}  {'-'*14}  {'-'*12}")
for step in [1, 50, 122, 500, 1000, WARMUP_STEPS, TOTAL_STEPS // 2, TOTAL_STEPS]:
    if step <= WARMUP_STEPS:
        lr = TARGET_LR * (step / WARMUP_STEPS)
    else:
        # Linear decay after warmup (HF default)
        lr = TARGET_LR * max(0.0, (TOTAL_STEPS - step) / (TOTAL_STEPS - WARMUP_STEPS))
    pct = 100.0 * min(step, WARMUP_STEPS) / WARMUP_STEPS
    marker = "  ← your step 122" if step == 122 else ""
    print(f"  {step:>8}  {lr:>14.3e}  {pct:>11.1f}%{marker}")

observed_lr_step122 = 2.182e-6
implied_warmup = int(TARGET_LR / observed_lr_step122 * 122)
print(f"\n  Observed LR at step 122 = {observed_lr_step122:.3e}")
print(f"  This implies warmup_steps ≈ {implied_warmup:,}")
if implied_warmup > TOTAL_STEPS * 0.5:
    print("  ⚠ Warmup is longer than 50% of total training — LR may never reach target.")
else:
    print("  ✓ Warmup length looks reasonable.")
print("═" * 60 + "\n")


print("═" * 60)
print("All diagnostics complete. Review any ⚠ warnings above before")
print("resuming / restarting training.")
print("═" * 60)