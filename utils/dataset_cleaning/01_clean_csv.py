import os
import csv
import pandas as pd
import io
import re
from pathlib import Path


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


def clean_text(text):
    if pd.isna(text) or text == '':
        return None

    text = str(text)
    text = re.sub(r'=L|=M|=O|=%|=-|=\n', ' ', text)
    text = re.sub(r'[^\w\s.,!?()-]', ' ', text)
    text = ' '.join(text.split())

    if len(text.strip()) < 5:
        return None

    return text.strip()


def clean_app_name(name):
    """Clean and validate app_name"""
    if pd.isna(name) or not name:
        return ''

    name = str(name).strip()
    name = re.sub(r'[^\x20-\x7E]', '', name)  # Remove non-ASCII
    name = name.strip()

    # Heuristic filter: too short or no alphabetic character
    if len(name) < 3 or not re.search(r'[a-zA-Z]', name):
        return ''

    # Junky symbol patterns
    if re.search(r'[=<>@#\$%^&*~]+', name):
        return ''

    # Remove known junk phrases (case-insensitive)
    if any(bad in name.lower() for bad in ['fix the replies', 'test', 'asdf', 'lorem']):
        return ''

    # Should contain at least one capitalized word (heuristic for real app names)
    if not re.search(r'\b[A-Z][a-z]+', name):
        return ''

    return name


def clean_row_data(row, header):
    cleaned = []
    for i, cell in enumerate(row):
        if cell is None or pd.isna(cell):
            cell = ''
        else:
            cell = str(cell).strip()

        if i < len(header):
            col_name = header[i].lower()

            if 'score' in col_name:
                try:
                    if cell and cell.lower() not in ['nan', 'none', 'null']:
                        score = int(float(cell))
                        cleaned.append(str(score if 1 <= score <= 5 else 1))
                    else:
                        cleaned.append('1')
                except Exception:
                    cleaned.append('1')

            elif 'review' in col_name:
                cleaned_review = clean_text(cell)
                cleaned.append(cleaned_review if cleaned_review else '')

            elif 'app_name' in col_name:
                cell = clean_app_name(cell)
                cleaned.append(cell)

            else:
                cleaned.append(cell if cell.lower() not in ['nan', 'none', 'null'] else '')
        else:
            cleaned.append(cell if cell.lower() not in ['nan', 'none', 'null'] else '')

    return cleaned


def clean_dataset(input_path, output_dir):
    """Clean the CSV dataset"""

    print(f"Loading dataset from {input_path}...")

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
            cleaned_row = cleaned_row[:len(header_row)]
            if app_package_idx is not None and app_package_idx < len(cleaned_row):
                if not is_valid_app_package(cleaned_row[app_package_idx]):
                    continue

            if any(cell.strip() for cell in cleaned_row):
                cleaned_rows.append(cleaned_row)

    if len(cleaned_rows) < 2:
        raise ValueError("Not enough valid rows found after cleaning.")

    df = pd.DataFrame(cleaned_rows[1:], columns=cleaned_rows[0])
    print(f"Original dataset: {len(df)} rows")

    if 'app_package' in df.columns:
        initial = len(df)
        df = df[df['app_package'].apply(is_valid_app_package)]
        print(f"Filtered out {initial - len(df)} rows with invalid app packages")

    if 'review' in df.columns:
        initial = len(df)
        df = df[df['review'].str.len() > 0]
        print(f"Filtered out {initial - len(df)} rows with empty reviews")

    if 'app_name' in df.columns:
        initial = len(df)
        df['app_name'] = df['app_name'].apply(clean_app_name)
        df = df[df['app_name'].str.len() >= 3]
        print(f"Filtered out {initial - len(df)} rows with invalid app names")

    df = df.dropna(how='all')
    for col in df.columns:
        df[col] = df[col].fillna('').replace(['nan', 'NaN', 'None', 'null', 'NULL'], '')

    print(f"After cleaning: {len(df)} rows")

    Path(output_dir).mkdir(exist_ok=True)
    full_output = os.path.join(output_dir, "full_cleaned_dataset.csv")
    df.to_csv(full_output, index=False, encoding='utf-8')
    print(f"Full cleaned dataset saved to: {full_output}")

    return df, full_output


def split_by_apps(df, output_dir):
    """Split dataset by apps"""
    apps = df['app_name'].unique()
    print(f"Found {len(apps)} unique apps:")

    app_stats = []
    for app in apps:
        app_df = df[df['app_name'] == app].copy()
        clean_app_name = re.sub(r'[^\w\s-]', '', app).strip()
        clean_app_name = re.sub(r'\s+', '_', clean_app_name)
        app_file = os.path.join(output_dir, f"{clean_app_name}.csv")
        app_df.to_csv(app_file, index=False, encoding='utf-8')

        stats = {
            'app_name': app,
            'reviews': len(app_df),
            'avg_score': round(app_df['score'].astype(float).mean(), 2),
            'filename': app_file
        }
        app_stats.append(stats)
        print(f"  - {app}: {stats['reviews']} reviews (avg score: {stats['avg_score']}) -> {app_file}")

    return app_stats


def create_mini_subsets(app_stats, output_dir, min_reviews=15, max_reviews=50):
    """Create mini subsets for testing"""
    mini_dir = os.path.join(output_dir, "mini_subsets")
    Path(mini_dir).mkdir(exist_ok=True)
    print(f"\nCreating mini subsets ({min_reviews}-{max_reviews} reviews per app)...")

    selected_apps = []
    for stat in app_stats:
        if stat['reviews'] >= min_reviews:
            app_df = pd.read_csv(stat['filename'])

            if len(app_df) > max_reviews:
                try:
                    sampled_df = app_df.groupby('score', group_keys=False).apply(
                        lambda x: x.sample(min(len(x), max_reviews // 5))
                    ).sample(n=max_reviews).reset_index(drop=True)
                except Exception:
                    sampled_df = app_df.sample(n=max_reviews).reset_index(drop=True)
            else:
                sampled_df = app_df.copy()

            clean_app_name = Path(stat['filename']).stem
            mini_file = os.path.join(mini_dir, f"{clean_app_name}_mini.csv")
            sampled_df.to_csv(mini_file, index=False, encoding='utf-8')

            selected_apps.append({
                'app_name': stat['app_name'],
                'original_reviews': stat['reviews'],
                'mini_reviews': len(sampled_df),
                'avg_score': round(sampled_df['score'].astype(float).mean(), 2),
                'filename': mini_file
            })

            print(f"  - {stat['app_name']}: {len(sampled_df)} reviews -> {mini_file}")

    if selected_apps:
        combined_mini = [pd.read_csv(app['filename']) for app in selected_apps]
        combined_df = pd.concat(combined_mini, ignore_index=True)
        combined_file = os.path.join(mini_dir, "combined_mini_dataset.csv")
        combined_df.to_csv(combined_file, index=False, encoding='utf-8')
        print(f"\nCombined mini dataset: {len(combined_df)} reviews from {len(selected_apps)} apps")
        print(f"Saved to: {combined_file}")

        print("\nMini dataset summary:")
        summary = combined_df.groupby('app_name').agg({
            'review': 'count',
            'score': lambda x: round(pd.to_numeric(x, errors='coerce').mean(), 2)
        }).rename(columns={'review': 'Reviews', 'score': 'Avg Score'})
        print(summary)

    return selected_apps


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "..", "..", "data", "input", "endpoint_1_process_reviews", "ai_assistants")
    input_path = os.path.join(input_dir, "REFSQ26-AIAssistants.csv")
    output_dir = os.path.join(input_dir, "cleaned_data")

    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found!")
        print("Please check the file path and update if necessary.")
        return

    df, full_output = clean_dataset(input_path, output_dir)
    app_stats = split_by_apps(df, output_dir)
    mini_apps = create_mini_subsets(app_stats, output_dir, min_reviews=15, max_reviews=50)

    print("\nProcessing complete!")
    print(f"Check the '{output_dir}' directory for all outputs")
    print(f"Use '{os.path.join(output_dir, 'mini_subsets', 'combined_mini_dataset.csv')}' for testing")


if __name__ == "__main__":
    main()
