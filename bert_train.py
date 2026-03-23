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


# ── Config ─────────────────────────────────────────────────────────────────────
TOKENIZER_MODEL    = "tokenizer/unigram_32000_0.9995.model"
CHUNK_DIR          = "tokenized_chunks"
CONFIG_FILE        = "bert_config.json"
CHECKPOINT_DIR     = "bert_checkpoints"
FINAL_MODEL_DIR    = "bert_final_model"
WANDB_RUN_ID_FILE  = "wandb_run_id.txt"

CACHE_FILES        = 10        # max .pt files held in memory at once
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

PAD_ID  = sp.pad_id()
MASK_ID = sp.piece_to_id("[MASK]")
VOCAB_SIZE = sp.get_piece_size()

print(f"Vocab size : {VOCAB_SIZE}")
print(f"PAD_ID     : {PAD_ID}")
print(f"MASK_ID    : {MASK_ID}")


# ── Dataset (lazy loading with LRU file cache) ─────────────────────────────────
class BertChunkDataset(Dataset):
    def __init__(self, chunk_dir, max_files=None):
        self.files = sorted(glob.glob(f"{chunk_dir}/*.pt"))
        if max_files:
            self.files = self.files[:max_files]
        assert len(self.files) > 0, f"No .pt files found in {chunk_dir}"

        # Build index without loading all data into RAM
        print(f"Indexing {len(self.files)} files...")
        self.index = []   # list of (file_idx, sample_idx)
        for file_idx, path in enumerate(tqdm(self.files, desc="Indexing")):
            data = torch.load(path, weights_only=True)
            self.index.extend((file_idx, i) for i in range(len(data)))

        random.shuffle(self.index)
        self._cache = {}   # file_idx → list[dict]
        print(f"✓ Dataset indexed: {len(self.index):,} samples across {len(self.files)} files")

    def _load(self, file_idx):
        if file_idx not in self._cache:
            # Evict oldest entry if cache is full
            if len(self._cache) >= CACHE_FILES:
                self._cache.pop(next(iter(self._cache)))
            self._cache[file_idx] = torch.load(self.files[file_idx], weights_only=True)
        return self._cache[file_idx]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, sample_idx = self.index[idx]
        item = self._load(file_idx)[sample_idx]
        return {
            "input_ids":      item["input_ids"],
            "attention_mask": item["attention_mask"],
        }


dataset = BertChunkDataset(CHUNK_DIR, max_files=MAX_FILES)


# ── MLM Data Collator ──────────────────────────────────────────────────────────
class SimpleMLMCollator:
    """
    Standard BERT MLM masking:
      - 15% of non-padding tokens are selected
      - Of those: 80% → [MASK], 10% → random token, 10% → unchanged
    """
    def __init__(self, mask_token_id, pad_token_id, vocab_size, mlm_probability=0.15):
        self.mask_token_id  = mask_token_id
        self.pad_token_id   = pad_token_id
        self.vocab_size     = vocab_size
        self.mlm_probability = mlm_probability

    def __call__(self, examples):
        input_ids      = torch.stack([e["input_ids"]      for e in examples])
        attention_mask = torch.stack([e["attention_mask"] for e in examples])

        labels = input_ids.clone()

        # Select 15% of tokens, excluding padding
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        probability_matrix.masked_fill_(input_ids == self.pad_token_id, 0.0)
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
        # (bernoulli(0.5) on the remaining 20% gives 10% overall)
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
    vocab_size=VOCAB_SIZE,
    mlm_probability=0.15,
)
print("✓ MLM collator ready")


# ── Sanity-check collator on a tiny batch ─────────────────────────────────────
_batch = [dataset[i] for i in range(4)]
_out   = data_collator(_batch)
_masked = (_out["labels"] != -100).sum().item()
_total  = _out["labels"].numel()
print(f"Collator check — input shape: {_out['input_ids'].shape}, "
      f"masked: {_masked}/{_total} ({_masked/_total:.1%})")


# ── Model ──────────────────────────────────────────────────────────────────────
config = BertConfig.from_json_file(CONFIG_FILE)

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
            "effective_batch":    128 * 2,   # per_device * accum
            "warmup_ratio":       0.10,
            "scheduler":          "cosine",
            "epochs":             3,
            "vocab_size":         VOCAB_SIZE,
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

    # I/O
    dataloader_num_workers=8,

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