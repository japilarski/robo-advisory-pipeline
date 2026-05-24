# Block 2 — covariance matrix estimation (Ledoit-Wolf, walk-forward)

import pandas as pd
import numpy as np
import pickle
from sklearn.covariance import LedoitWolf
import warnings

warnings.filterwarnings("ignore")

TICKERS = ["SPY", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "VNQ", "GLD", "XLP"]
TRAIN_WINDOW = 36
STEP = 1
PERIODS_YEAR = 12

log_returns_monthly = pd.read_csv(
    "data/block_0/log_returns_monthly.csv", index_col=0, parse_dates=True
)
TICKERS_ORDERED = log_returns_monthly.columns.tolist()

covariance_matrices = {}

n_periods = len(log_returns_monthly)
dates = log_returns_monthly.index

for start_idx in range(0, n_periods - TRAIN_WINDOW, STEP):
    end_idx = start_idx + TRAIN_WINDOW
    forecast_idx = end_idx

    if forecast_idx >= n_periods:
        break

    forecast_date = dates[forecast_idx]
    train = log_returns_monthly[TICKERS_ORDERED].iloc[start_idx:end_idx].values

    lw = LedoitWolf().fit(train)
    sigma = lw.covariance_ * PERIODS_YEAR  # annualise

    covariance_matrices[forecast_date] = pd.DataFrame(
        sigma, index=TICKERS, columns=TICKERS
    )

with open("data/block_2/covariance_matrices.pkl", "wb") as f:
    pickle.dump(covariance_matrices, f)

print(f"Saved covariance_matrices.pkl ({len(covariance_matrices)} matrices)")

# sanity checks
first_date = dates[TRAIN_WINDOW]
sigma_lw = covariance_matrices[first_date]

assert np.allclose(sigma_lw, sigma_lw.T), "Matrix is not symmetric!"
assert (np.diag(sigma_lw.values) > 0).all(), "Negative variance on diagonal!"

eigenvalues = np.linalg.eigvalsh(sigma_lw.values)
assert (eigenvalues >= -1e-10).all(), f"Matrix not PSD! Min eigenvalue: {eigenvalues.min()}"
