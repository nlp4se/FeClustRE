import os
import pandas as pd
from csv import reader
import io
from collections import Counter

def infer_delimiter_and_columns(lines, candidates=[",", ";", "\t"]):
    best_delim = None
    best_count = 0
    best_column_len = None

    for delim in candidates:
        parsed = []
        try:
            parsed = [list(reader([line], delimiter=delim))[0] for line in lines]
        except Exception:
            continue

        col_lengths = [len(row) for row in parsed]
        most_common = Counter(col_lengths).most_common(1)
        if most_common:
            count, col_len = most_common[0][1], most_common[0][0]
            if count > best_count and col_len >= 5:
                best_count = count
                best_column_len = col_len
                best_delim = delim

    return best_delim, best_column_len

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "..", "..", "data", "input", "endpoint_1_process_reviews", "ai_assistants")
    input_path = os.path.join(input_dir, "REFSQ26-AIAssistants.csv")
    output_path = os.path.join(input_dir, "REFSQ26-cleaned_ai_assistants.csv")

    # Read lines safely and remove nulls
    with open(input_path, "r", encoding="latin1", errors="ignore") as f:
        lines = [line.replace('\x00', '').strip() for line in f if line.strip()]

    # Infer delimiter from full dataset if needed
    delim, expected_columns = infer_delimiter_and_columns(lines, [",", ";", "\t"])
    if delim is None:
        raise ValueError("Could not infer a consistent delimiter or column count.")

    # Parse and clean rows
    cleaned_rows = []
    for line in lines:
        try:
            row = list(reader([line], delimiter=delim))[0]
            row = [cell.strip() for cell in row]
            if len(row) == expected_columns:
                cleaned_rows.append(row)
        except Exception:
            continue

    if len(cleaned_rows) < 2:
        raise ValueError("Not enough valid rows found after cleaning.")

    df = pd.DataFrame(cleaned_rows[1:], columns=cleaned_rows[0])
    df.to_csv(output_path, index=False)
    print(f"Cleaned CSV saved to: {output_path}")

if __name__ == "__main__":
    main()
