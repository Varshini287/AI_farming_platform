# clean_crop.py
import pandas as pd

def clean_crop(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)

    # 1. Drop any rows with missing values
    df = df.dropna()

    # 2. Remove invalid pH entries (e.g. pH <= 0)
    if 'ph' in df.columns:
        df = df[df['ph'] > 0]

    # 3. Clip extreme N, P, K values to a reasonable range [0,100]
    for col in ['N', 'P', 'K']:
        if col in df.columns:
            df[col] = df[col].clip(lower=0, upper=100)

    # 4. Save cleaned data
    df.to_csv(output_csv, index=False)
    print(f"[clean_crop] Saved cleaned crop data to {output_csv}")

if __name__ == "__main__":
    clean_crop(
        input_csv="data/Crop_recommendation.csv",
        output_csv="data/processed/crop_recommendation_clean.csv"
    )
