import sentencepiece as spm
import json
import os

print("Script started")

model_path = "tokenizer/unigram_32000_0.9995.model"
text_path = "combined.txt"
output_path = "tokenized_dataset.jsonl"

assert os.path.exists(model_path), f"Model file not found: {model_path}"
assert os.path.exists(text_path), f"Text file not found: {text_path}"

print("Files confirmed")

sp = spm.SentencePieceProcessor()
sp.load(model_path)
print("Tokenizer loaded")

block_size = 512
input_ids = []

with open(text_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # Tokenize each line
        encoded = sp.encode(line, out_type=int)
        input_ids.extend(encoded)

print(f"Total tokens collected: {len(input_ids)}")
print(f"First 10 tokens: {input_ids[:10]}")

blocks = [input_ids[i:i+block_size] for i in range(0, len(input_ids), block_size)]

print(f"Total blocks (512 tokens each): {len(blocks)}")

if not blocks:
    print("No blocks created. Input might still be too short.")
else:
    with open(output_path, "w", encoding="utf-8") as out_file:
        for block in blocks:
            out_file.write(json.dumps({"input_ids": block}) + "\n")

    print(f"Wrote {len(blocks)} blocks to {output_path}")
