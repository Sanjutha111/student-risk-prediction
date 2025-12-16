import pandas as pd
from pathlib import Path


RAW_DATA_PATH = Path("data/raw/student-mat.csv")
PROCESSED_DATA_PATH = Path("data/processed/clean_student_data.csv")

# --------------------
# Load data
# --------------------
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")

# --------------------
# Clean and preprocess
# --------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Create target variable
    df["AtRisk"] = (df["G3"] < 10).astype(int)

    # Drop leakage columns
    df = df.drop(columns=["G1", "G2", "G3"])

    # Sanity check for missing values
    if df.isna().sum().any():
        raise ValueError("Dataset contains missing values")

    return df

# --------------------
# Save cleaned data
# --------------------
def save_data(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def main():
    df = load_data(RAW_DATA_PATH)
    df_clean = clean_data(df)
    save_data(df_clean, PROCESSED_DATA_PATH)
    print("Cleaned dataset saved to:", PROCESSED_DATA_PATH)

if __name__ == "__main__":
    main()
