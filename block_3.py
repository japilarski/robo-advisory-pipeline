# Block 3 — portfolio optimization (Markowitz, Mean-CVaR, Risk Parity)

import numpy as np
import pandas as pd
import pickle
import warnings
from scipy.optimize import minimize
import pandas_datareader.data as web

warnings.filterwarnings("ignore")

TICKERS = ["SPY", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "VNQ", "GLD", "XLP"]
N = len(TICKERS)
MAX_WEIGHT = 0.40
MIN_WEIGHT = 0.00
CVAR_ALPHA = 0.05
CVAR_SIMULATIONS = 5_000

expected_returns = pd.read_csv(
    "data/block_1/expected_returns.csv", index_col=[0, 1], parse_dates=True
)

with open("data/block_2/covariance_matrices.pkl", "rb") as f:
    covariance_matrices = pickle.load(f)

ff_raw = web.DataReader("F-F_Research_Data_Factors", "famafrench", start="2006-01-01")[0]
ff_raw = ff_raw / 100
ff_raw.index = ff_raw.index.strftime("%Y-%m")
rf_series = ff_raw["RF"]

forecast_dates = expected_returns.index.get_level_values("date").unique().sort_values()


def max_sharpe(mu, sigma, rf, max_w=MAX_WEIGHT):
    n = len(mu)
    bounds = [(MIN_WEIGHT, max_w)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    w0 = np.ones(n) / n

    def neg_sharpe(w):
        port_ret = w @ mu
        port_vol = np.sqrt(w @ sigma @ w)
        if port_vol < 1e-10:
            return 0.0
        return -(port_ret - rf) / port_vol

    result = minimize(
        neg_sharpe,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )
    if result.success:
        w = np.clip(result.x, 0, 1)
        return w / w.sum()
    return np.ones(n) / n


def mean_cvar(
        mu,
        sigma,
        alpha=CVAR_ALPHA,
        n_sim=CVAR_SIMULATIONS,
        max_w=MAX_WEIGHT,
        random_state=42,
):
    rng = np.random.default_rng(random_state)
    n = sigma.shape[0]

    try:
        scenarios = rng.multivariate_normal(mu.values, sigma, size=n_sim)
    except np.linalg.LinAlgError:
        sigma_reg = sigma + np.eye(n) * 1e-6
        scenarios = rng.multivariate_normal(mu.values, sigma_reg, size=n_sim)

    bounds = [(MIN_WEIGHT, max_w)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    w0 = np.ones(n) / n

    def portfolio_cvar(w):
        port_returns = scenarios @ w
        losses = -port_returns
        var_threshold = np.quantile(losses, 1 - alpha)
        tail_losses = losses[losses >= var_threshold]
        return tail_losses.mean()

    result = minimize(
        portfolio_cvar,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 2000},
    )
    if result.success:
        w = np.clip(result.x, 0, 1)
        return w / w.sum()
    return np.ones(n) / n


def risk_parity(sigma, max_w=MAX_WEIGHT):
    n = sigma.shape[0]
    bounds = [(1e-6, max_w)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    w0 = np.ones(n) / n

    def risk_contribution_objective(w):
        port_var = w @ sigma @ w
        rc = w * (sigma @ w)
        rc_target = port_var / n
        return np.sum((rc - rc_target) ** 2)

    result = minimize(
        risk_contribution_objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if result.success:
        w = np.clip(result.x, 0, 1)
        return w / w.sum()
    return np.ones(n) / n


results = []
signal_variants = ["historical_mean", "capm", "ff3f"]

for date in forecast_dates:
    if date not in covariance_matrices:
        continue
    sigma = covariance_matrices[date].values

    ym = date.strftime("%Y-%m")
    rf_monthly = rf_series.get(ym, 0.0)
    rf_annual = rf_monthly * 12

    w_rp = risk_parity(sigma)
    results.append(
        {"date": date, "model": "risk_parity", "signal": "none", **dict(zip(TICKERS, w_rp))}
    )

    for variant in signal_variants:
        try:
            mu = expected_returns.loc[(date, variant), TICKERS]
        except KeyError:
            continue

        w_mvo = max_sharpe(mu, sigma, rf_annual)
        results.append(
            {"date": date, "model": "markowitz", "signal": variant, **dict(zip(TICKERS, w_mvo))}
        )

        w_cvar = mean_cvar(mu, sigma)
        results.append(
            {"date": date, "model": "mean_cvar", "signal": variant, **dict(zip(TICKERS, w_cvar))}
        )

weights_df = pd.DataFrame(results).set_index(["date", "model", "signal"])
weights_df.to_csv("data/block_3/portfolio_weights.csv")
print(f"Saved portfolio_weights.csv ({weights_df.shape})")

# sanity checks
sums = weights_df.sum(axis=1)
assert np.allclose(sums, 1.0, atol=1e-4), f"Weights don't sum to 1: {sums.describe()}"
assert (weights_df.values >= -1e-6).all(), "Negative weights found!"
max_w_actual = weights_df.max(axis=1).max()
assert max_w_actual <= MAX_WEIGHT + 1e-4, f"Max weight exceeded: {max_w_actual:.4f}"
