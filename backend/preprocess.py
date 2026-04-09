import os
import pandas as pd

# Define paths
INPUT_DIR = "/Users/Janith/Desktop/invesment-portfolio-analysis/data"         
OUTPUT_DIR = "/Users/Janith/Desktop/invesment-portfolio-analysis/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "master_top50_companies.csv")


os.makedirs(OUTPUT_DIR, exist_ok=True)

all_dfs = []

#  Load and preprocess each CSV
for file_name in os.listdir(INPUT_DIR):
    if file_name.endswith(".csv"):
        file_path = os.path.join(INPUT_DIR, file_name)
        df = pd.read_csv(file_path)

        if df.empty:
            print(f" Skipping empty CSV: {file_name}")
            continue

        # Convert date column if exists
        if "lastTradedTime" in df.columns:
            df["lastTradedTime"] = pd.to_datetime(df["lastTradedTime"], errors="coerce")
            df = df.dropna(subset=["lastTradedTime"])  # remove invalid dates

        all_dfs.append(df)

#  Combine all data
if not all_dfs:
    print(" No valid data to combine.")
else:
    master_df = pd.concat(all_dfs, ignore_index=True)

    # Filter by date range
    START_DATE = pd.to_datetime("2023-01-01")
    END_DATE = pd.to_datetime("2025-09-23")
    master_df = master_df[(master_df["lastTradedTime"] >= START_DATE) & 
                          (master_df["lastTradedTime"] <= END_DATE)]

    # Sort and reset index
    master_df.sort_values(by="lastTradedTime", inplace=True)
    master_df.reset_index(drop=True, inplace=True)

    #  Drop unnecessary columns if they exist
    cols_to_drop = ["crossingTradeVol", "crossingVolume", "marketCapPercentage", "quantity", "status"]
    master_df = master_df.drop(columns=[col for col in cols_to_drop if col in master_df.columns])

    #  Clean strings (strip spaces, replace empty/NaN with "null")
    master_df = master_df.astype(str)  # treat everything as string
    master_df = master_df.map(lambda x: x.strip() if isinstance(x, str) else x)  #  modern replacement
    master_df = master_df.replace("", "null").fillna("null")

    #  Save final cleaned CSV
    master_df.to_csv(OUTPUT_FILE, index=False)

    print(f" Final cleaned CSV saved at: {OUTPUT_FILE}, rows: {len(master_df)}, columns: {len(master_df.columns)}")
    print(f" Remaining columns: {list(master_df.columns)}")