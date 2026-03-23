import os
import glob
import random
import math
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

PAD_ID     = sp.pad_id()               # 0  → <pad>
BOS_ID     = sp.bos_id()               # 2  → <s>    used as [CLS]
EOS_ID     = sp.eos_id()               # 3  → </s>   used as [SEP]
MASK_ID    = sp.piece_to_id("[MASK]")  # 4  → [MASK]
VOCAB_SIZE = sp.get_piece_size()

print(f"Vocab size : {VOCAB_SIZE}")
print(f"PAD_ID     : {PAD_ID}  → '{sp.id_to_piece(PAD_ID)}'")
print(f"BOS_ID     : {BOS_ID}  → '{sp.id_to_piece(BOS_ID)}'  (used as [CLS])")
print(f"EOS_ID     : {EOS_ID}  → '{sp.id_to_piece(EOS_ID)}'  (used as [SEP])")
print(f"MASK_ID    : {MASK_ID}  → '{sp.id_to_piece(MASK_ID)}'")


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 1 — Special token identity check
# Vocab layout verified from tokenizer:
#   id 0 → <pad>   (PAD)
#   id 1 → <unk>   (UNK)
#   id 2 → <s>     (BOS / [CLS])
#   id 3 → </s>    (EOS / [SEP])
#   id 4 → [MASK]
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("DIAGNOSTIC 1 — Special token identity check")
print("═" * 60)

EXPECTED_SPECIAL = {
    "<pad>":  0,
    "<unk>":  1,
    "<s>":    2,
    "</s>":   3,
    "[MASK]": 4,
}

diag1_passed = True
for piece, expected_id in EXPECTED_SPECIAL.items():
    actual_id  = sp.piece_to_id(piece)
    back_piece = sp.id_to_piece(actual_id)
    ok = actual_id == expected_id
    if not ok:
        diag1_passed = False
    status = "✓" if ok else "✗ MISMATCH"
    print(f"  {status}  piece_to_id('{piece}') = {actual_id}  |  id_to_piece({actual_id}) = '{back_piece}'")

# Guard: MASK must not be aliased to <unk>
if MASK_ID == sp.unk_id():
    diag1_passed = False
    print(f"\n  ✗ MASK_ID={MASK_ID} resolves to <unk> — [MASK] was not added to the vocab!")
else:
    print(f"\n  ✓ [MASK] id={MASK_ID} is distinct from <unk> id={sp.unk_id()}")

print("\n  First 8 token IDs in vocabulary:")
for i in range(8):
    print(f"    id {i:>2} → '{sp.id_to_piece(i)}'")

expected_initial_loss = math.log(VOCAB_SIZE)
print(f"\n  Expected initial MLM loss : {expected_initial_loss:.3f}  (= ln({VOCAB_SIZE}))")

if not diag1_passed:
    print("\n  ⚠ One or more special token IDs are misaligned.")
    print("    Fix the tokenizer vocabulary before continuing.")
    raise SystemExit(1)
else:
    print("\n  ✓ All special token IDs are correct.")
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


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 2 — Chunk boundary token check (<s> / </s>)
# Every sequence must start with BOS_ID (<s>, id=2) at position 0 and
# end with EOS_ID (</s>, id=3) at the last non-padding position.
# These are the [CLS] / [SEP] equivalents produced by bert_dataset_tokenize.py.
# ══════════════════════════════════════════════════════════════════════════════
print("═" * 60)
print("DIAGNOSTIC 2 — Chunk boundary token check (<s> / </s>)")
print("═" * 60)

N_CHECK     = min(500, len(dataset))
missing_bos = 0
missing_eos = 0

for i in range(N_CHECK):
    ids     = dataset[i]["input_ids"].tolist()
    attn    = dataset[i]["attention_mask"].tolist()
    seq_len = sum(attn)
    if ids[0] != BOS_ID:
        missing_bos += 1
    if ids[seq_len - 1] != EOS_ID:
        missing_eos += 1

print(f"  Checked {N_CHECK} samples")
print(f"  Missing <s>   ([CLS]) at position 0     : {missing_bos} / {N_CHECK}")
print(f"  Missing </s>  ([SEP]) at last real token : {missing_eos} / {N_CHECK}")

print("\n  Decoded first 10 tokens of 3 random samples:")
for i in random.sample(range(len(dataset)), 3):
    ids    = dataset[i]["input_ids"].tolist()
    pieces = [sp.id_to_piece(t) for t in ids[:10]]
    print(f"    sample {i:>7}: {pieces}")

if missing_bos > 0 or missing_eos > 0:
    print("\n  ⚠ Boundary tokens missing — delete tokenized_chunks/ and")
    print("    re-run bert_dataset_tokenize.py before training.")
    raise SystemExit(1)
else:
    print("\n  ✓ All checked samples have correct <s> / </s> boundaries.")
print("═" * 60 + "\n")


# ── MLM Data Collator ──────────────────────────────────────────────────────────
class SimpleMLMCollator:
    """
    Standard BERT MLM masking:
      - 15% of eligible tokens are selected for masking
      - Eligible = not PAD, not BOS (<s> / [CLS]), not EOS (</s> / [SEP])
      - Of selected: 80% → [MASK], 10% → random token, 10% → unchanged

    BOS and EOS are explicitly excluded so the model always sees clean
    [CLS] and [SEP] signals, matching standard BERT behaviour.
    """
    def __init__(self, mask_token_id, pad_token_id, bos_token_id, eos_token_id,
                 vocab_size, mlm_probability=0.15):
        self.mask_token_id   = mask_token_id
        self.pad_token_id    = pad_token_id
        self.bos_token_id    = bos_token_id
        self.eos_token_id    = eos_token_id
        self.vocab_size      = vocab_size
        self.mlm_probability = mlm_probability

    def __call__(self, examples):
        input_ids      = torch.stack([e["input_ids"]      for e in examples])
        attention_mask = torch.stack([e["attention_mask"] for e in examples])

        labels = input_ids.clone()

        # Exclude PAD, BOS ([CLS]), and EOS ([SEP]) from masking eligibility
        ineligible = (
            (input_ids == self.pad_token_id) |
            (input_ids == self.bos_token_id) |
            (input_ids == self.eos_token_id)
        )

        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        probability_matrix.masked_fill_(ineligible, 0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Unselected positions → -100 (ignored by cross-entropy loss)
        labels[~masked_indices] = -100

        # 80% of selected → [MASK]
        replace_with_mask = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool()
            & masked_indices
        )
        input_ids[replace_with_mask] = self.mask_token_id

        # 10% of selected → random token
        # bernoulli(0.5) on the remaining 20% gives 10% overall
        replace_with_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~replace_with_mask
        )
        random_tokens = torch.randint(0, self.vocab_size, labels.shape, dtype=torch.long)
        input_ids[replace_with_random] = random_tokens[replace_with_random]

        # Remaining 10% of selected → unchanged (no action needed)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


data_collator = SimpleMLMCollator(
    mask_token_id=MASK_ID,
    pad_token_id=PAD_ID,
    bos_token_id=BOS_ID,
    eos_token_id=EOS_ID,
    vocab_size=VOCAB_SIZE,
    mlm_probability=0.15,
)
print("✓ MLM collator ready")


# ── Sanity-check collator on a tiny batch ─────────────────────────────────────
_batch  = [dataset[i] for i in range(4)]
_out    = data_collator(_batch)
_masked = (_out["labels"] != -100).sum().item()
_total  = _out["labels"].numel()
print(f"Collator check — input shape: {_out['input_ids'].shape}, "
      f"masked: {_masked}/{_total} ({_masked/_total:.1%})")


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 3 — MLM collator label sanity check
# Placed here so data_collator is guaranteed to be in scope.
# Confirms:
#   (a) PAD / BOS / EOS positions all have label = -100
#   (b) masking rate is ~15% of eligible (non-special) tokens
#   (c) all active label IDs are within [0, VOCAB_SIZE)
#   (d) <s> and </s> positions are never selected for masking
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("DIAGNOSTIC 3 — MLM collator label sanity check")
print("═" * 60)

_diag_batch = [dataset[i] for i in range(4)]
_b          = data_collator(_diag_batch)
_ids        = _b["input_ids"]    # (4, 512)
_lbls       = _b["labels"]       # (4, 512)

# (a) Special token positions must be -100
_special_pos         = (_ids == PAD_ID) | (_ids == BOS_ID) | (_ids == EOS_ID)
_special_not_ignored = (_lbls[_special_pos] != -100).sum().item()
if _special_not_ignored > 0:
    print(f"  ✗ (a) {_special_not_ignored} PAD/BOS/EOS positions have label ≠ -100 — leaking into loss!")
else:
    print(f"  ✓ (a) All PAD / <s> / </s> positions correctly have label = -100")

# (b) Masking rate over eligible tokens only
_eligible_count = (~_special_pos).sum().item()
_n_active       = (_lbls != -100).sum().item()
_mask_rate      = 100.0 * _n_active / _eligible_count if _eligible_count > 0 else 0.0
if abs(_mask_rate - 15.0) > 3.0:
    print(f"  ⚠ (b) Masking rate = {_mask_rate:.1f}% over eligible tokens — expected ~15%")
else:
    print(f"  ✓ (b) Masking rate = {_mask_rate:.1f}% over eligible tokens (target 15%)")

# (c) Label IDs in vocab range
_active_lbls  = _lbls[_lbls != -100]
_out_of_range = ((_active_lbls < 0) | (_active_lbls >= VOCAB_SIZE)).sum().item()
if _out_of_range > 0:
    print(f"  ✗ (c) {_out_of_range} label IDs outside vocab range [0, {VOCAB_SIZE})")
else:
    print(f"  ✓ (c) All label IDs within vocab range [0, {VOCAB_SIZE})")

# (d) BOS / EOS never masked
_bos_masked = (_lbls[_ids == BOS_ID] != -100).sum().item()
_eos_masked = (_lbls[_ids == EOS_ID] != -100).sum().item()
if _bos_masked or _eos_masked:
    print(f"  ✗ (d) <s> masked {_bos_masked}×, </s> masked {_eos_masked}× — must never be masked!")
else:
    print(f"  ✓ (d) <s> ([CLS]) and </s> ([SEP]) are never masked")

# Decoded sample
print("\n  Decoded masked positions from sample 0 (first 20 positions):")
_found = 0
for pos, (inp, lbl) in enumerate(zip(_ids[0][:20].tolist(), _lbls[0][:20].tolist())):
    if lbl != -100:
        print(f"    pos {pos:>3}: input='{sp.id_to_piece(inp)}' (id={inp})  "
              f"label='{sp.id_to_piece(lbl)}' (id={lbl})")
        _found += 1
if _found == 0:
    print("    (no masked positions in first 20 tokens — try a larger sample)")

print("═" * 60 + "\n")


# ── Model ──────────────────────────────────────────────────────────────────────
config = BertConfig.from_json_file(CONFIG_FILE)

# Wire SentencePiece native IDs into HuggingFace BertConfig
config.pad_token_id  = PAD_ID   # 0  → <pad>
config.bos_token_id  = BOS_ID   # 2  → <s>    (used by model internals for loss masking)
config.eos_token_id  = EOS_ID   # 3  → </s>   (used by model internals for loss masking)
config.cls_token_id  = BOS_ID   # 2  → <s>
config.sep_token_id  = EOS_ID   # 3  → </s>
config.mask_token_id = MASK_ID  # 4  → [MASK]

latest_checkpoint = None
if os.path.exists(CHECKPOINT_DIR):
    ckpt_dirs = sorted(
        glob.glob(f"{CHECKPOINT_DIR}/checkpoint-*"),
        key=lambda x: int(x.split("-")[-1])
    )
    if ckpt_dirs:
        latest_checkpoint = ckpt_dirs[-1]

if latest_checkpoint:
    print(f"✓ Resuming from checkpoint: {latest_checkpoint}")
    model = BertForMaskedLM.from_pretrained(latest_checkpoint)
else:
    print("✓ Training from scratch")
    model = BertForMaskedLM(config)

print(f"Parameters         : {model.num_parameters():,}")
print(f"Hidden size        : {config.hidden_size}")
print(f"Layers             : {config.num_hidden_layers}")
print(f"Attention heads    : {config.num_attention_heads}")
print(f"Intermediate size  : {config.intermediate_size}")


# ── Train / Validation Split ───────────────────────────────────────────────────
train_size = int(TRAIN_RATIO * len(dataset))
val_size   = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print(f"Train samples : {len(train_dataset):,}")
print(f"Val samples   : {len(val_dataset):,}")


# ── W&B ───────────────────────────────────────────────────────────────────────
wandb_run_id = None
if os.path.exists(WANDB_RUN_ID_FILE):
    with open(WANDB_RUN_ID_FILE) as f:
        wandb_run_id = f.read().strip()
    print(f"✓ Resuming W&B run: {wandb_run_id}")
else:
    print("✓ Starting new W&B run")

if wandb.run is None:
    run = wandb.init(
        project="bert-pretraining",
        name="bert-base-110M-512len",
        id=wandb_run_id,
        resume="allow",
        config={
            "model_params":       model.num_parameters(),
            "hidden_size":        config.hidden_size,
            "num_layers":         config.num_hidden_layers,
            "num_heads":          config.num_attention_heads,
            "intermediate_size":  config.intermediate_size,
            "max_seq_len":        512,
            "mlm_probability":    0.15,
            "learning_rate":      1e-4,
            "effective_batch":    128 * 2,
            "warmup_ratio":       0.10,
            "scheduler":          "cosine",
            "epochs":             3,
            "vocab_size":         VOCAB_SIZE,
            "pad_id":             PAD_ID,
            "bos_id (cls)":       BOS_ID,
            "eos_id (sep)":       EOS_ID,
            "mask_id":            MASK_ID,
        }
    )
    with open(WANDB_RUN_ID_FILE, "w") as f:
        f.write(run.id)
    print(f"✓ W&B run ID saved: {run.id}")


# ── Training Arguments ────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,

    # Batch — MI300X has 192GB HBM3, push it hard
    per_device_train_batch_size=128,
    per_device_eval_batch_size=256,
    gradient_accumulation_steps=2,      # effective batch = 256

    # Learning rate
    learning_rate=1e-4,
    warmup_ratio=0.10,
    weight_decay=0.01,

    # Schedule
    num_train_epochs=3,
    lr_scheduler_type="cosine",

    # Precision — BF16 is more stable than FP16, MI300X has native BF16 support
    bf16=True,
    fp16=False,

    # Checkpointing & logging
    logging_steps=100,
    save_steps=2000,
    save_total_limit=3,
    eval_steps=2000,
    eval_strategy="steps",

    # Disable best-model loading so checkpointing plays nicely with resume
    load_best_model_at_end=False,
    metric_for_best_model="eval_loss",

    # I/O — data is in RAM so workers only add overhead; pin_memory speeds
    # up CPU→GPU transfers via page-locked memory
    dataloader_num_workers=0,
    dataloader_pin_memory=True,

    # Reporting
    report_to="wandb",

    push_to_hub=False,
)


# ── Trainer ────────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)
print("✓ Trainer ready")


# ── Train ──────────────────────────────────────────────────────────────────────
print("\nStarting training...")
trainer.train(resume_from_checkpoint=latest_checkpoint)

trainer.save_model(FINAL_MODEL_DIR)
print(f"✓ Model saved to {FINAL_MODEL_DIR}")


# ── Final Evaluation ───────────────────────────────────────────────────────────
eval_results = trainer.evaluate()
print("\n=== Final Evaluation ===")
for k, v in eval_results.items():
    print(f"  {k}: {v:.4f}")

wandb.finish()