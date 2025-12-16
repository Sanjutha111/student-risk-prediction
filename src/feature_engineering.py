import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/processed/clean_student_data.csv")
OUTPUT_PATH = Path("data/processed/model_ready_data.csv")

def main():
    df = pd.read_csv(INPUT_PATH)

    # Convert yes/no columns to 1/0 automatically
    yes_no_cols = df.select_dtypes(include="object").columns

    for col in yes_no_cols:
        unique_vals = df[col].unique()
        if set(unique_vals) == {"yes", "no"} or set(unique_vals) == {"no", "yes"}:
            df[col] = df[col].map({"yes": 1, "no": 0})

    # One-hot encode remaining categorical columns
    categorical_cols = df.select_dtypes(include="object").columns

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Final safety check
    if df.select_dtypes(include="object").shape[1] > 0:
        raise ValueError("Categorical columns still present after encoding")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("✅ Model-ready data saved to:", OUTPUT_PATH)

if __name__ == "__main__":
    main()
