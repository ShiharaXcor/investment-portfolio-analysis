import os
import yfinance as yf
import pandas as pd


OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


START_DATE = "2023-01-01"
END_DATE = "2025-09-24"




companies = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "NVDA": "NVIDIA",
    "META": "Meta",
    "TSLA": "Tesla",
    "BRK-B": "Berkshire Hathaway",
    "V": "Visa",
    "JPM": "JPMorgan Chase",
    "JNJ": "Johnson & Johnson",
    "WMT": "Walmart",
    "PG": "Procter & Gamble",
    "MA": "Mastercard",
    "UNH": "UnitedHealth",
    "XOM": "Exxon Mobil",
    "HD": "Home Depot",
    "PFE": "Pfizer",
    "BAC": "Bank of America",
    "KO": "Coca-Cola",
    "PEP": "PepsiCo",
    "NKE": "Nike",
    "DIS": "Disney",
    "ORCL": "Oracle",
    "CSCO": "Cisco",
    "INTC": "Intel",
    "ADBE": "Adobe",
    "CRM": "Salesforce",
    "NFLX": "Netflix",
    "AVGO": "Broadcom",
    "COST": "Costco",
    "T": "AT&T",
    "VZ": "Verizon",
    "TM": "Toyota",
    "SAP": "SAP",
    "BABA": "Alibaba",
    "TSM": "TSMC",
    "SONY": "Sony",
    "BP": "BP",
    "SHEL": "Shell",
    "HSBC": "HSBC",
    "RIO": "Rio Tinto",
    "UL": "Unilever",
    "NVS": "Novartis",
    "LLY": "Eli Lilly",
    "ABBV": "AbbVie",
    "MRK": "Merck",
    "ASML": "ASML Holding",
    "RDS-A": "Royal Dutch Shell"
}

def fetch_and_save(symbol, name, start, end):
    print(f" Fetching {name} ({symbol})")
    try:
        # Fetch only trading days
        df = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
        
        if df.empty:
            raise ValueError("No data returned")

        # Flatten MultiIndex if exists
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure all OHLCV columns exist
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                df[col] = pd.NA

        df.reset_index(inplace=True)  # 'Date' column is trading date

        # Remove rows with no volume (no trade)
        df = df[df["Volume"] > 0]

        # Fetch market cap
        ticker_info = yf.Ticker(symbol).info
        market_cap = ticker_info.get("marketCap", None)

        # Prepare final dataframe
        df_final = pd.DataFrame({
            "id": range(1, len(df)+1),
            "name": name,
            "symbol": symbol,
            "quantity": 1,
            "percentageChange": ((df["Close"] - df["Open"]) / df["Open"]) * 100,
            "change": df["Close"] - df["Open"],
            "price": df["Close"],
            "previousClose": df["Close"].shift(1),
            "high": df["High"],
            "low": df["Low"],
            "turnover": df["Close"] * df["Volume"],
            "sharevolume": df["Volume"],
            "tradevolume": df["Volume"],
            "marketCap": market_cap,
            "marketCapPercentage": "",
            "open": df["Open"],
            "closingPrice": df["Close"],
            "crossingVolume": "",
            "crossingTradeVol": "",
            "status": "Active",
            "lastTradedTime": df["Date"]      
        })

        safe_name = name.replace(" ", "_")
        df_final.to_csv(os.path.join(OUTPUT_DIR, f"{safe_name}.csv"), index=False)
        print(f" Saved {safe_name}.csv")

    except Exception as e:
        print(f" Error fetching {symbol}: {e}")
        # Create empty CSV in case of failure
        df_empty = pd.DataFrame(columns=[
            "id","name","symbol","quantity","percentageChange","change","price","previousClose",
            "high","low","lastTradedTime","issueDate","turnover","sharevolume","tradevolume",
            "marketCap","marketCapPercentage","open","closingPrice","crossingVolume","crossingTradeVol","status"
        ])
        safe_name = name.replace(" ", "_")
        df_empty.to_csv(os.path.join(OUTPUT_DIR, f"{safe_name}.csv"), index=False)


for sym, name in companies.items():
    fetch_and_save(sym, name, START_DATE, END_DATE)



    