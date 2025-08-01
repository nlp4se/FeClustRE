import os
import pandas as pd


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "..", "data", "input", "endpoint_1_process_reviews", "ai_assistants")
    input_path = os.path.join(data_dir, "REFSQ26-cleaned_ai_assistants.csv")
    output_path = os.path.join(data_dir, "REFSQ26-mini_ai_assistants.csv")

    df = pd.read_csv(input_path, low_memory=False)

    if "app_package" not in df.columns:
        print("Available columns:", df.columns.tolist())
        raise KeyError("'app_package' column is missing in the CSV")

    df = df[df["app_package"].notna()]
    df = df[df["app_package"].apply(lambda x: str(x).strip() != "")]

    n = 100

    def sample_group(group):
        if len(group) <= n:
            return group
        return group.sample(n=n, random_state=42)

    mini_df = df.groupby("app_package", group_keys=False).apply(sample_group)
    mini_df = mini_df.reset_index(drop=True)

    mini_df.to_csv(output_path, index=False)

    app_counts = mini_df.groupby("app_package").size()
    total_apps = len(app_counts)
    total_reviews = len(mini_df)

    print(f"Mini CSV with samples from {total_apps} apps ({total_reviews} total reviews) saved to: {output_path}")


if __name__ == "__main__":
    main()