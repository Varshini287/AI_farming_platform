import pandas as pd
import os
import sys

RAW_CSV = "market_price/raw/raw_area.csv"
OUT_CSV = "market_price/data/area.csv"

def clean_area():
    if not os.path.isfile(RAW_CSV):
        print(f"❌ Raw file not found: {RAW_CSV}")
        sys.exit(1)

    # 1) Load raw
    df = pd.read_csv(RAW_CSV)
    print("Columns in raw_area.csv:", list(df.columns))

    # 2) Rename columns to our schema
    #    Based on your screenshot, the headers are:
    #      State, District, Crop, Year, Season, Area, Area Units, Production, Production Units, Yield
    df = df.rename(columns={
        "State":         "state_name",
        "District":      "district",
        "Crop":          "commodity_name",
        "Year":          "year",
        "Area":          "area"
        # we ignore Area Units, Production, Production Units, Yield
    })

    # 3) Select only the fields we need
    keep = ["state_name", "district", "year", "commodity_name", "area"]
    df = df[keep]

    # 4) Filter to Karnataka
    df = df[df["state_name"].str.upper().str.contains("KARNATAKA")]

    # 5) Write out
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"✅ Wrote {OUT_CSV} ({len(df)} rows)")

if __name__ == "__main__":
    clean_area()
