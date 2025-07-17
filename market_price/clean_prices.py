import pandas as pd

def clean_prices():
    df = pd.read_csv("market_price/raw/raw_prices.csv")
    # Rename & parse
    df = df.rename(columns={
        "market":       "market_name",
        "date":         "arrival_date",
        "modal_price":  "modal_price",        # keep same
        "commodity_name":"commodity_name",    # keep same
        "state":        "state_name"
    })
    df["arrival_date"] = pd.to_datetime(df["arrival_date"], errors="coerce")
    # Filter to Karnataka
    df = df[df.state_name.str.upper() == "KARNATAKA"]
    # Select only needed columns
    out = df[["commodity_name","market_name","arrival_date","modal_price"]]
    out.to_csv("market_price/data/prices.csv", index=False)
    print("→ Cleansed prices.csv:", len(out), "rows")

if __name__ == "__main__":
    clean_prices()
