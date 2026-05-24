# Block 0 — download and prepare price/return data

import yfinance as yf
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

TICKERS = ["SPY", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "VNQ", "GLD", "XLP"]
START = "2007-01-01"
END = "2024-12-31"

raw = yf.download(
    TICKERS,
    start=START,
    end=END,
    auto_adjust=True,
    progress=False,
)["Close"]

# drop market holidays (all-NaN rows), then forward-fill up to 3 days
# to handle calendar mismatches between US and non-US ETF listings
raw = raw.dropna(how="all")
prices_daily = raw.ffill(limit=3).dropna()
prices_daily = prices_daily[TICKERS]

log_returns_daily = np.log(prices_daily / prices_daily.shift(1)).dropna()

# resample to last business day of each month
prices_monthly = prices_daily.resample("BME").last()[TICKERS]
log_returns_monthly = np.log(prices_monthly / prices_monthly.shift(1)).dropna()

prices_daily.to_csv("data/block_0/prices_daily.csv")
prices_monthly.to_csv("data/block_0/prices_monthly.csv")
log_returns_daily.to_csv("data/block_0/log_returns_daily.csv")
log_returns_monthly.to_csv("data/block_0/log_returns_monthly.csv")

print("Block 0 complete: saved prices and returns to data/block_0")
