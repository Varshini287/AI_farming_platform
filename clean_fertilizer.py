# src/preprocessing/clean_fertilizer.py
import pandas as pd

def clean_fertilizer(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)

    # 1. Drop rows with any nulls
    df = df.dropna()

    # 2. Ensure key columns are numeric
    numeric_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=[c for c in numeric_cols if c in df.columns])

    # 3. Save cleaned data
    df.to_csv(output_csv, index=False)
    print(f"[clean_fertilizer] Saved cleaned fertilizer data to {output_csv}")

if __name__ == "__main__":
    clean_fertilizer(
        input_csv="data/data_core.csv",
        output_csv="data/processed/fertilizer_clean.csv"
    )
