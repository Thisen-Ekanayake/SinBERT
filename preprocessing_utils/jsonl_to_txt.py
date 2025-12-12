import json

def jsonl_to_txt(jsonl_path, txt_path):
    with open(jsonl_path, "r", encoding="utf-8") as jsonl_file, \
         open(txt_path, "w", encoding="utf-8") as txt_file:

        for line in jsonl_file:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                text = obj.get("text", "")

                # Replace literal \n with actual newlines
                text = text.replace("\\n", "\n")

                # Split lines and remove first two
                parts = text.split("\n")
                cleaned = "\n".join(parts[2:])

                txt_file.write(cleaned + "\n\n")

            except Exception as e:
                print("Skipping line due to error:", e)

if __name__ == "__main__":
    jsonl_to_txt("si_clean_0000.jsonl", "output.txt")