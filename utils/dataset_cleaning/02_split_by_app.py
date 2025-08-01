import os
import pandas as pd
import re

def sanitize_filename(name: str, fallback: str = "app") -> str:
    name = re.sub(r'[^a-zA-Z0-9_.-]', '_', str(name)).strip('_')
    return name if name else fallback

def is_valid_package(pkg: str) -> bool:
    return bool(re.fullmatch(r'[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)+', str(pkg)))

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "..", "data", "input", "endpoint_1_process_reviews", "ai_assistants")
    input_path = os.path.join(data_dir, "REFSQ26-cleaned_ai_assistants.csv")
    output_dir = os.path.join(data_dir, "split_by_app")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path, low_memory=False)

    if "app_package" not in df.columns:
        raise KeyError("'app_package' column missing in cleaned data.")

    # Filter out broken or missing entries
    df = df[df["app_package"].notna()]
    df = df[df["app_package"].apply(is_valid_package)]

    for idx, (app_pkg, group) in enumerate(df.groupby("app_package")):
        filename = sanitize_filename(app_pkg) + ".csv"
        group.to_csv(os.path.join(output_dir, filename), index=False)

    print(f"Filtered and split CSVs saved to: {output_dir}")

if __name__ == "__main__":
    main()
