import pandas as pd


def count_apps_and_reviews(file_path):

    df = pd.read_csv(file_path, engine='python', encoding='utf-8', on_bad_lines='skip')

    print("=== APPS AND REVIEW COUNTS ===")
    print(f"Total records: {len(df)}")

    # App counts
    app_counts = df['app_name'].value_counts()
    print(f"\nUnique apps: {len(app_counts)}")

    for app, count in app_counts.items():
         print(f"{app}: {count} reviews")


if __name__ == "__main__":
    count_apps_and_reviews('full_cleaned_dataset.csv')