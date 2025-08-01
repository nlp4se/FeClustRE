import os
import pandas as pd
import re


def sanitize_filename(name: str, fallback: str = "app") -> str:
    if pd.isna(name) or not str(name).strip():
        return fallback
    name = re.sub(r'[^a-zA-Z0-9_.-]', '_', str(name)).strip('_')
    return name if name else fallback


def is_valid_package(pkg):
    if pd.isna(pkg):
        return False
    pkg_str = str(pkg).strip()
    if not pkg_str or pkg_str.lower() in ['nan', 'null', 'none']:
        return False
    return len(pkg_str) > 0


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "..", "data", "input", "endpoint_1_process_reviews", "ai_assistants")
    input_path = os.path.join(data_dir, "REFSQ26-cleaned_ai_assistants.csv")
    output_dir = os.path.join(data_dir, "split_by_app")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path, low_memory=False)

    if "app_package" not in df.columns:
        print("Available columns:", df.columns.tolist())
        raise KeyError("'app_package' column missing in cleaned data.")

    df = df[df["app_package"].notna()]
    df = df[df["app_package"].apply(lambda x: str(x).strip() != "")]

    print(f"Total rows after filtering: {len(df)}")

    if len(df) == 0:
        print("No valid data found after filtering.")
        return

    valid_packages = df[df["app_package"].apply(is_valid_package)]

    print(f"Rows with valid packages: {len(valid_packages)}")

    if len(valid_packages) == 0:
        print("No valid app packages found. Using all non-empty packages.")
        valid_packages = df

    app_count = 0
    for app_pkg, group in valid_packages.groupby("app_package"):
        if len(group) > 0:
            filename = sanitize_filename(app_pkg) + ".csv"
            group.to_csv(os.path.join(output_dir, filename), index=False)
            app_count += 1
            print(f"Created {filename} with {len(group)} rows")

    print(f"Filtered and split {app_count} CSVs saved to: {output_dir}")


if __name__ == "__main__":
    main()