import os

def count_words(line: str) -> int:
    """
    Count the number of words in a line.

    Args:
        line (str): A single line of text.

    Returns:
        int: Number of space-separated words.
    """
    return len(line.strip().split())

def rename_file_with_stats(group_name: str, lines: list[str], output_dir: str) -> str:
    """
    Saves a group of lines into a new file named based on group name,
    line count, and word count.

    Format: <group_name>_<line_count>_<word_count>.txt

    Args:
        group_name (str): Prefix for the file name (e.g., 'short', 'main').
        lines (list): List of text lines.
        output_dir (str): Directory to save the output file.

    Returns:
        str: Full path of the saved file.
    """
    line_count = len(lines)
    word_count = sum(count_words(line) for line in lines)
    new_filename = f"{group_name}_{line_count}_{word_count}.txt"
    new_path = os.path.join(output_dir, new_filename)

    with open(new_path, "w", encoding="utf-8") as f:
        # Ensure lines are stripped and end with a newline
        f.writelines([line.strip() + "\n" for line in lines])

    print(f"Saved: {new_filename}")
    return new_path

def split_lines_by_word_count(input_file: str):
    """
    Splits a text file's lines into 3 categories:
    - Short: lines with < 5 words
    - Main: lines with 5–50 words
    - Long: lines with > 50 words

    Each group is saved into a new file with a name containing
    the line and word count.

    Args:
        input_file (str): Path to the input .txt file.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    short_lines = []
    main_lines = []
    long_lines = []

    for line in lines:
        word_count = count_words(line)
        if word_count < 5:
            short_lines.append(line)
        elif word_count > 50:
            long_lines.append(line)
        else:
            main_lines.append(line)

    output_dir = os.path.dirname(input_file)

    rename_file_with_stats("short", short_lines, output_dir)
    rename_file_with_stats("main", main_lines, output_dir)
    rename_file_with_stats("long", long_lines, output_dir)

# === MAIN EXECUTION ===
# Replace with your actual file path
if __name__ == "__main__":
    file_path = "output_path.normalized.txt"
    split_lines_by_word_count(file_path)