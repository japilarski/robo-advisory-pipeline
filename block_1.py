# Block 1 — signal model: walk-forward expected returns (3 variants)

import pandas as pd
from sklearn.linear_model import LinearRegression
import pandas_datareader.data as web
import warnings

warnings.filterwarnings("ignore")

TICKERS = ["SPY", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "VNQ", "GLD", "XLP"]
TRAIN_WINDOW = 36  # months
STEP = 1
PERIODS_YEAR = 12

log_returns_monthly = pd.read_csv(
    "data/block_0/log_returns_monthly.csv", index_col=0, parse_dates=True
)

ff_raw = web.DataReader("F-F_Research_Data_Factors", "famafrench", start="2006-01-01")[0]
ff_factors = ff_raw / 100
ff_factors.columns = ["MKT_RF", "SMB", "HML", "RF"]

# use year-month strings for robust merging (PeriodIndex can be finicky)
ff_factors.index = ff_factors.index.strftime("%Y-%m")
ff_factors = ff_factors[~ff_factors.index.duplicated(keep="last")]

ym_index = log_returns_monthly.index.strftime("%Y-%m")
ff_aligned = ff_factors.reindex(ym_index)
ff_aligned.index = log_returns_monthly.index

results = []
n_periods = len(log_returns_monthly)
dates = log_returns_monthly.index

for start_idx in range(0, n_periods - TRAIN_WINDOW, STEP):
    end_idx = start_idx + TRAIN_WINDOW
    forecast_idx = end_idx

    if forecast_idx >= n_periods:
        break

    forecast_date = dates[forecast_idx]
    train_returns = log_returns_monthly.iloc[start_idx:end_idx]
    train_ff = ff_aligned.iloc[start_idx:end_idx]

    if len(train_ff) < TRAIN_WINDOW:
        continue

    row_base = {"date": forecast_date}

    # historical mean
    mu_hist = train_returns.mean() * PERIODS_YEAR
    results.append({**row_base, "variant": "historical_mean", **mu_hist.to_dict()})

    # CAPM
    mkt_excess = train_ff["MKT_RF"].values.reshape(-1, 1)
    rf_mean = train_ff["RF"].mean()
    erp = train_ff["MKT_RF"].mean()

    mu_capm = {}
    for ticker in TICKERS:
        excess_ret = (train_returns[ticker] - train_ff["RF"]).values
        reg = LinearRegression(fit_intercept=True).fit(mkt_excess, excess_ret)
        beta = reg.coef_[0]
        mu_capm[ticker] = (rf_mean + beta * erp) * PERIODS_YEAR

    results.append({**row_base, "variant": "capm", **mu_capm})

    # Fama-French 3-factor
    factors_3f = train_ff[["MKT_RF", "SMB", "HML"]].values
    factor_means = train_ff[["MKT_RF", "SMB", "HML"]].mean()

    mu_ff3f = {}
    for ticker in TICKERS:
        excess_ret = (train_returns[ticker] - train_ff["RF"]).values
        reg = LinearRegression(fit_intercept=True).fit(factors_3f, excess_ret)
        betas = reg.coef_
        mu_ff3f[ticker] = (rf_mean + betas @ factor_means.values) * PERIODS_YEAR

    results.append({**row_base, "variant": "ff3f", **mu_ff3f})

expected_returns = pd.DataFrame(results).set_index(["date", "variant"])
expected_returns.to_csv("data/block_1/expected_returns.csv")

print(f"Saved expected_returns.csv ({expected_returns.shape})")
