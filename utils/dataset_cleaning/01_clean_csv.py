import os
import csv
import pandas as pd
import io
import re


def is_valid_app_package(pkg):
    if pd.isna(pkg) or not pkg:
        return False
    pkg_str = str(pkg).strip()
    if len(pkg_str) < 3:
        return False
    if pkg_str.startswith('.') or pkg_str.endswith('.'):
        return False
    if re.match(r'^[\d._-]+$', pkg_str):
        return False
    return bool(re.match(r'^[a-zA-Z][a-zA-Z0-9._-]*$', pkg_str))


def clean_row_data(row, header):
    cleaned = []
    for i, cell in enumerate(row):
        cell = str(cell).strip()

        if i < len(header):
            col_name = header[i].lower()

            if 'score' in col_name:
                try:
                    score = int(float(cell)) if cell and cell != '' else None
                    if score and 1 <= score <= 5:
                        cleaned.append(str(score))
                    else:
                        cleaned.append('')
                except:
                    cleaned.append('')

            elif 'review' in col_name and len(cell) < 5:
                cleaned.append('')

            else:
                cleaned.append(cell)
        else:
            cleaned.append(cell)

    return cleaned


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "..", "..", "data", "input", "endpoint_1_process_reviews", "ai_assistants")
    input_path = os.path.join(input_dir, "REFSQ26-AIAssistants.csv")
    output_path = os.path.join(input_dir, "REFSQ26-cleaned_ai_assistants.csv")

    try:
        with open(input_path, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()
    except UnicodeDecodeError:
        with open(input_path, "r", encoding="latin1", errors="replace") as f:
            raw_text = f.read()

    raw_text = raw_text.replace("\x00", "").replace("\ufeff", "")

    cleaned_rows = []
    reader = csv.reader(io.StringIO(raw_text), delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)

    header_row = None
    app_package_idx = None

    for row_num, row in enumerate(reader):
        if row_num == 0:
            header_row = [cell.strip() for cell in row if cell.strip()]
            if len(header_row) >= 5:
                cleaned_rows.append(header_row)
                try:
                    app_package_idx = header_row.index('app_package')
                except ValueError:
                    try:
                        app_package_idx = [i for i, col in enumerate(header_row) if 'package' in col.lower()][0]
                    except IndexError:
                        app_package_idx = None
            continue

        cleaned_row = clean_row_data(row, header_row)

        if len(cleaned_row) >= len(header_row) and any(cell for cell in cleaned_row):
            if len(cleaned_row) > len(header_row):
                cleaned_row = cleaned_row[:len(header_row)]
            elif len(cleaned_row) < len(header_row):
                cleaned_row.extend([''] * (len(header_row) - len(cleaned_row)))

            if app_package_idx is not None and app_package_idx < len(cleaned_row):
                if not is_valid_app_package(cleaned_row[app_package_idx]):
                    continue

            if any(cell.strip() for cell in cleaned_row):
                cleaned_rows.append(cleaned_row)

    if len(cleaned_rows) < 2:
        raise ValueError("Not enough valid rows found after cleaning.")

    df = pd.DataFrame(cleaned_rows[1:], columns=cleaned_rows[0])

    if 'app_package' in df.columns:
        initial_count = len(df)
        df = df[df['app_package'].apply(is_valid_app_package)]
        filtered_count = len(df)
        print(f"Filtered out {initial_count - filtered_count} rows with invalid app packages")

    df = df.dropna(how='all')
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Cleaned CSV saved to: {output_path}")
    print(f"Final dataset contains {len(df)} rows")


if __name__ == "__main__":
    main()