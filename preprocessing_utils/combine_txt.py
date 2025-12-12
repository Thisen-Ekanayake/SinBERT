"""
combine 2 text files into one
"""

# ==== CONFIG ====
file1_path = "main_5856427_116349202.txt"
file2_path = "long_filtered.txt"
output_path = "combined.txt"
# ==================

with open(output_path, "w", encoding="utf-8") as outfile:
    # Write content of the first file
    with open(file1_path, "r", encoding="utf-8") as f1:
        for line in f1:
            outfile.write(line)
    
    # Write a newline between files (optional)
    outfile.write("\n")
    
    # Write content of the second file
    with open(file2_path, "r", encoding="utf-8") as f2:
        for line in f2:
            outfile.write(line)

print(f"Files combined successfully into '{output_path}'.")
