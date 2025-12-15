# SinBERT: A Sinhala BERT-Style Masked Language Model

SinBERT is a **Sinhala-focused BERT-style masked language model** trained from scratch on a large, cleaned Sinhala text corpus.  
The repository contains:

- **Preprocessing utilities** tailored for Sinhala text
- Scripts/notebooks to **tokenize data** and **train a compact BERT (6-layer) masked LM**
- Saved **training checkpoints** and a **final exportable model**

---

## Project Structure

- **`data/`**: Raw and intermediate text data.
  - **`*.parquet` / `*.jsonl` / `*.txt`**: Original or partially cleaned Sinhala corpora.
  - **`culturax/`**: A sub-corpus (e.g., CulturaX) and its filtered/normalized variants.
- **`preprocessing_utils/`**: Standalone scripts to clean and normalize Sinhala text.
- **`text_shards/`**: Large corpora split into many shard files for easier processing.
- **`tokenizer/`**:
  - Placeholder directory for a SentencePiece tokenizer.
  > The tokenizer is not publicly available at the moment. However, you can train your own tokenizer using [this repository](https://github.com/Thisen-Ekanayake/Sinhala-Tokenizer-Training.git "Sinhala-Tokenizer-Training"), which contains the exact code used to train the tokenizer for this project.
- **`utilities/`**:
  - Multiple Jupyter notebooks for exploratory analysis and dataset inspection.
- **`bert_config.json`**: Configuration for the BERT model (6 layers, 6 heads, 384 hidden size, vocab size 32000, max position 512, etc.).
- **`bert_checkpoints/`**: Intermediate training checkpoints (configs, optimizer state, scheduler, RNG, `training_args.bin`).
- **`bert_final_model/`**: Final HF-compatible model directory (config + weights + `training_args.bin`).
- **`bert_dataset_tokenize.ipynb`**: Notebook to build HF-style datasets from tokenized data.
- **`bert_train.ipynb`**: Notebook that trains `BertForMaskedLM` using the tokenized dataset.
- **`requirements.txt`**: Python dependencies (PyTorch, Transformers, SentencePiece, Datasets, etc.).

---

## Installation

- **Python version**: >= 3.10 (tested with Python 3.13)

1. **Create and activate an environment** (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

---

## End-to-End Workflow

### 1. Prepare and Clean the Raw Corpus

- Place your raw Sinhala data into `data/` (TXT, JSONL, or Parquet).
- Use the scripts in `preprocessing_utils/` to iteratively clean the text:
  - **`text_normalizer.py`**: Unicode NFC normalization and removal of zero-width/invisible chars (except ZWJ).
  - **`remove_non_sinhala_characters.py`**: Keep only Sinhala script, Sinhala digits, and a safe punctuation set.
  - **`clean_repeated_punctuation.py`**, **`remove_extra_whitespace.py`**, **`remove_empty_punctuations.py`**.
  - **`remove_unmatched_brackets_and_dates.py`**: Remove date-like patterns and stray brackets/quotes.
  - **`split_lines_by_length.py`**: Split into short/main/long files by word count.
  - **`filter_sinhala_lines_and_rename.py`**: Filter by Sinhala content and minimal length and produce descriptive filenames.

Typical usage pattern (each script is standalone):

```bash
python preprocessing_utils/text_normalizer.py
python preprocessing_utils/remove_non_sinhala_characters.py
# ... run other scripts as needed, adjusting input/output paths in each script
```

After cleaning, you typically combine all cleaned text into a single large file such as `combined.txt`.

---

### 2. Train or Reuse the Tokenizer

The tokenizer is **not distributed with this repository**. Instead, you should train your own tokenizer following the steps described above.

Training a **SentencePiece unigram** tokenizer is fast and lightweight, even on modest hardware. Once trained, place the generated `.model` and `.vocab` files inside the `tokenizer/` directory before proceeding.

---

### 3. Create a HuggingFace Dataset (Notebook)

- Use `bert_dataset_tokenize.ipynb` to:
  - Build a `datasets.Dataset` or `DatasetDict` suitable for masked language modeling (MLM).
  - Optionally shuffle/split into train/validation sets.

The notebook is written in an exploratory style; you can adapt the cell parameters (paths, splits) as needed.

---

### 4. Train SinBERT (Notebook)

- **Model configuration** is defined in `bert_config.json`:
  - `model_type`: `bert`
  - `architectures`: `["BertForMaskedLM"]`
  - `vocab_size`: `32000`
  - `hidden_size`: `384`
  - `num_hidden_layers`: `6`
  - `num_attention_heads`: `6`
  - `max_position_embeddings`: `512`
  - `mlm_probability`: `0.15`

- Use `bert_train.ipynb` to:
  - Load the prepared dataset.
  - Instantiate `BertForMaskedLM` with `bert_config.json`.
  - Train using HuggingFace `transformers` + `accelerate` on your hardware.
  - Log metrics (e.g., with `wandb` if enabled).
  - Save checkpoints to `bert_checkpoints/` and export the final model to `bert_final_model/`.

You can resume training from any checkpoint folder under `bert_checkpoints/` by pointing the notebook or training script to the desired `checkpoint-XXXX` directory.

---

## Using the Trained Model

The final model in `bert_final_model/` is compatible with HuggingFace Transformers.  
You can load it as follows, together with the SentencePiece tokenizer:

```python
from transformers import AutoModelForMaskedLM
import sentencepiece as spm

model = AutoModelForMaskedLM.from_pretrained("bert_final_model")

sp = spm.SentencePieceProcessor()
sp.load("tokenizer/unigram_32000_0.9995.model")

text = "ආයුබෝවන්! ඔබට කොහොමද?"
ids = sp.encode(text, out_type=int)
```

From here you can:

- Run **masked language modeling** experiments.
- Fine-tune SinBERT for downstream Sinhala NLP tasks (classification, NER, etc.) using standard HF workflows.

---

## Notes and Tips

- **Performance**:
  - Large corpora are split into many shards under `text_shards/` for easier streaming and parallelism.
  - `text_normalizer.py` uses multithreading (`ThreadPoolExecutor`) and automatically picks up all CPU cores.
- **Reproducibility**:
  - Checkpoints store `rng_state.pth`, `scaler.pt`, `optimizer.pt`, and `scheduler.pt` for reproducible training restarts.
  - `training_args.bin` in both checkpoints and `bert_final_model/` capture the HF `TrainingArguments`.
- **Customization**:
  - You can adjust model depth/width by editing `bert_config.json`.

---

## Training Data

Training data consists of a mixture of publicly available multilingual datasets and a custom Sinhala corpus:

- MADLAD-400
- CulturaX
- Author-curated Sinhala dataset (≈147 million words)

All datasets were preprocessed and normalized prior to training.

---

## Citation / Attribution

If you use or build upon SinBERT in academic or industrial work, please cite this repository or acknowledge it in your paper, report, or documentation.