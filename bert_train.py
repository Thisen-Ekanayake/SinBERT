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
    TrainerCallback,
    DataCollatorForLanguageModeling
)
import sentencepiece as spm
import wandb
import pynvml
import time

# ==================== POWER CONSUMPTION LOGGING ====================
pynvml.nvmlInit()
GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)

class GPUEfficiencyCallback(TrainerCallback):
    def __init__(self, log_every_n_steps=10):
        self.log_every_n_steps = log_every_n_steps
        self.energy_joules = 0.0
        self.total_tokens = 0
        self.last_time = None

    def on_step_begin(self, args, state, control, **kwargs):
        if self.last_time is None:
            self.last_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_every_n_steps != 0:
            return

        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        # GPU telemetry
        power_watts = pynvml.nvmlDeviceGetPowerUsage(GPU_HANDLE) / 1000.0
        util = pynvml.nvmlDeviceGetUtilizationRates(GPU_HANDLE).gpu
        temp = pynvml.nvmlDeviceGetTemperature(
            GPU_HANDLE, pynvml.NVML_TEMPERATURE_GPU
        )

        # Energy integration
        self.energy_joules += power_watts * dt
        energy_wh = self.energy_joules / 3600.0

        # Token counting
        batch = kwargs.get("inputs", None)
        if batch is not None and "input_ids" in batch:
            tokens_this_step = batch["input_ids"].numel()
            self.total_tokens += tokens_this_step
        else:
            tokens_this_step = 0

        # Efficiency metrics
        watts_per_token = (
            power_watts / tokens_this_step if tokens_this_step > 0 else 0.0
        )
        joules_per_token = (
            self.energy_joules / self.total_tokens
            if self.total_tokens > 0
            else 0.0
        )

        # Log to W&B
        wandb.log(
            {
                "gpu/power_watts": power_watts,
                "gpu/energy_wh": energy_wh,
                "gpu/utilization_pct": util,
                "gpu/temperature_c": temp,
                "train/tokens_step": tokens_this_step,
                "train/tokens_total": self.total_tokens,
                "efficiency/watts_per_token": watts_per_token,
                "efficiency/joules_per_token": joules_per_token,
            },
            step=state.global_step,
        )



# ==================== CONFIGURATION ====================
TOKENIZER_MODEL = "tokenizer/unigram_32000_0.9995.model"
CHUNK_DIR = "tokenized_chunks"
CONFIG_FILE = "bert_config.json"
CHECKPOINT_DIR = "bert_checkpoints"
WANDB_RUN_ID_FILE = "wandb_run_id.txt"  # File to store W&B run ID

# Verify paths
assert os.path.exists(TOKENIZER_MODEL)
assert os.path.exists(CHUNK_DIR)
assert os.path.exists(CONFIG_FILE)
print("✓ All paths verified")



# ==================== TOKENIZER ====================
sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_MODEL)

PAD_ID = sp.pad_id()
MASK_ID = sp.piece_to_id("[MASK]")

print("Vocab size:", sp.get_piece_size())
print("PAD_ID:", PAD_ID)
print("MASK_ID:", MASK_ID)



# ==================== DATASET ====================
class BertChunkDataset(Dataset):
    def __init__(self, chunk_dir, max_files=None):
        self.files = sorted(glob.glob(f"{chunk_dir}/*.pt"))
        if max_files:
            self.files = self.files[:max_files]
        
        assert len(self.files) > 0, "No chunk files found"
        
        # Load all data into memory
        self.data = []
        print(f"Loading {len(self.files)} files...")
        for file_idx, file in enumerate(self.files):
            if file_idx % 10 == 0:
                print(f"  Loaded {file_idx}/{len(self.files)} files...")
            file_data = torch.load(file)
            self.data.extend(file_data)
        
        print(f"✓ Dataset loaded with {len(self.data)} samples")
        random.shuffle(self.data)  # Shuffle once

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"]
        }

# Initialize dataset with a limit for testing
dataset = BertChunkDataset(CHUNK_DIR)  # Start with 20 files max_files=10



# ==================== DATA COLLATOR ====================
class SimpleMLMCollator:
    def __init__(self, mask_token_id, pad_token_id, vocab_size, mlm_probability=0.15):
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        self.mlm_probability = mlm_probability

    def __call__(self, examples):
        input_ids = torch.stack([e["input_ids"] for e in examples])
        attention_mask = torch.stack([e["attention_mask"] for e in examples])

        labels = input_ids.clone()

        # Do not mask padding
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        probability_matrix.masked_fill_(input_ids == self.pad_token_id, 0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        # 80% -> [MASK]
        mask_replace_prob = 0.8
        mask_replace = torch.bernoulli(torch.full(labels.shape, mask_replace_prob)).bool() & masked_indices
        input_ids[mask_replace] = self.mask_token_id

        # 10% -> random token (correct probability: 0.1 / 0.15 ≈ 0.6667)
        random_replace_prob = 0.6667
        random_replace = torch.bernoulli(torch.full(labels.shape, random_replace_prob)).bool() & masked_indices & ~mask_replace
        random_tokens = torch.randint(
            low=0,
            high=self.vocab_size,
            size=labels.shape,
            dtype=torch.long
        )
        input_ids[random_replace] = random_tokens[random_replace]

        # 10% -> unchanged (already handled by not being masked or replaced)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

data_collator = SimpleMLMCollator(
    mask_token_id=MASK_ID,
    pad_token_id=PAD_ID,
    vocab_size=sp.get_piece_size(),
    mlm_probability=0.15
)

print("✓ Custom MLM collator ready")



# ==================== TEST BATCH ====================
# Test with a small batch
batch = [dataset[i] for i in range(4)]
out = data_collator(batch)

print("input_ids shape:", out["input_ids"].shape)
print("labels shape:", out["labels"].shape)

# Count masked tokens
masked_tokens = (out["labels"] != -100).sum().item()
total_tokens = out["labels"].numel()
print(f"Masked tokens: {masked_tokens} ({masked_tokens/total_tokens:.1%} of all tokens)")



# ==================== MODEL ====================
config = BertConfig.from_json_file(CONFIG_FILE)

# Check for existing checkpoints
checkpoint_found = False
latest_checkpoint = None

if os.path.exists(CHECKPOINT_DIR):
    # Look for checkpoint folders (checkpoint-*)
    checkpoint_dirs = glob.glob(f"{CHECKPOINT_DIR}/checkpoint-*")
    if checkpoint_dirs:
        # Get the latest checkpoint by step number
        checkpoint_dirs.sort(key=lambda x: int(x.split('-')[-1]))
        latest_checkpoint = checkpoint_dirs[-1]
        checkpoint_found = True
        print(f"✓ Found checkpoint: {latest_checkpoint}")

if checkpoint_found:
    print(f"✓ Resuming from checkpoint: {latest_checkpoint}")
    model = BertForMaskedLM.from_pretrained(latest_checkpoint)
else:
    print("✓ Training from scratch")
    model = BertForMaskedLM(config)

print(f"Model parameters: {model.num_parameters():,}")
print(f"Hidden size: {config.hidden_size}")
print(f"Num layers: {config.num_hidden_layers}")
print(f"Num attention heads: {config.num_attention_heads}")



# ==================== DATA SPLIT ====================
# Split dataset into train and validation
train_ratio = 0.9
train_size = int(train_ratio * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")



# ==================== WANDB ====================
# Get or create W&B run ID
wandb_run_id = None

if os.path.exists(WANDB_RUN_ID_FILE):
    with open(WANDB_RUN_ID_FILE, 'r') as f:
        wandb_run_id = f.read().strip()
    print(f"✓ Found existing W&B run ID: {wandb_run_id}")
else:
    print("✓ Starting new W&B run")

# Initialize W&B with resume capability
if wandb.run is None:
    run = wandb.init(
        project="bert-finetuning",
        name="bert_lr1e-5_bs128_cosine_whole_dataset",
        id=wandb_run_id,           # Use existing run ID if available
        resume="allow",             # Allow resuming
        config={
            "lr": 1e-5,
            "effective_batch_size": 16 * 8,
            "scheduler": "cosine",
            "epochs": 2,
        }
    )
    
    # Save the run ID for future resumptions
    with open(WANDB_RUN_ID_FILE, 'w') as f:
        f.write(run.id)
    print(f"✓ W&B run ID saved: {run.id}")



# ==================== TRAINING ARGS ====================
training_args = TrainingArguments(
    output_dir="bert_checkpoints",
    overwrite_output_dir=False, # set to false to start from existing checkpoints
    
    # Batch sizes
    per_device_train_batch_size=16,  # Reduced for stability
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=8,    # Effective batch = 128
    
    # Learning rate
    learning_rate=1e-5,  # Lower learning rate for BERT
    warmup_ratio=0.05,
    # warmup_steps=5000,
    weight_decay=0.01,
    
    # Training schedule
    num_train_epochs=2,
    lr_scheduler_type="cosine",
    
    # Checkpointing
    logging_steps=100,
    eval_steps=25000,  # Added evaluation steps
    save_steps=2000,
    save_total_limit=3,
    
    # Evaluation
    eval_strategy="steps",
    load_best_model_at_end=False, # disable for resuming
    metric_for_best_model="eval_loss",
    
    # Optimization
    fp16=True,
    dataloader_num_workers=4,
    
    # Reporting
    report_to="wandb", # reporting to wandb
    
    no_cuda=False,
    
    # Push to hub (disabled by default)
    push_to_hub=False
)



# ==================== TRAINER ====================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    # Optional: compute metrics if needed
    # compute_metrics=compute_metrics,
    callbacks=[GPUEfficiencyCallback(log_every_n_steps=10)]
)

print("✓ Trainer ready with train/validation split")



# ==================== TRAINING ====================
print("Starting training...")

# Check if we should resume from checkpoint
resume_from_checkpoint = None
if os.path.exists(CHECKPOINT_DIR):
    checkpoint_dirs = glob.glob(f"{CHECKPOINT_DIR}/checkpoint-*")
    if checkpoint_dirs:
        # get the latest checkpoint by step number
        checkpoint_dirs.sort(key=lambda x: int(x.split('-')[-1]))
        resume_from_checkpoint = checkpoint_dirs[-1]
        print(f"Resuming from: {resume_from_checkpoint}")

trainer.train(resume_from_checkpoint=resume_from_checkpoint)

# Save the final model
trainer.save_model("bert_final_model")
print("✓ Training complete and model saved")



# ==================== EVALUATION ====================
# Evaluate on validation set
eval_results = trainer.evaluate()
print("\n=== Final Evaluation Results ===")
for key, value in eval_results.items():
    print(f"{key}: {value:.4f}")
    
# Finish W&B run
wandb.finish()