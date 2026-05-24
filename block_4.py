# Block 4 — backtest (gross, no transaction costs)

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

TICKERS = ["SPY", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "VNQ", "GLD", "XLP"]
THRESHOLD = 0.05
INITIAL_VALUE = 1.0

weights_df = pd.read_csv(
    "data/block_3/portfolio_weights.csv", index_col=[0, 1, 2], parse_dates=True
)
weights_df.index.names = ["date", "model", "signal"]

log_returns_monthly = pd.read_csv(
    "data/block_0/log_returns_monthly.csv", index_col=0, parse_dates=True
)[TICKERS]

simple_returns = np.exp(log_returns_monthly) - 1

combinations = weights_df.index.droplevel("date").unique()


def get_benchmark_weights(name):
    w = pd.Series(0.0, index=TICKERS)
    if name == "60_40":
        w["SPY"] = 0.60
        w["TLT"] = 0.40
    elif name == "equal_weight":
        w[:] = 1.0 / len(TICKERS)
    return w


def simulate_portfolio(target_weights_series, returns, strategy="calendar", threshold=THRESHOLD):
    monthly_returns = {}
    portfolio_values = {}
    rebalance_dates = []

    sim_dates = sorted(set(target_weights_series.keys()) & set(returns.index))
    if not sim_dates:
        return pd.Series(dtype=float), pd.Series(dtype=float), []

    current_weights = target_weights_series[sim_dates[0]].reindex(TICKERS).fillna(0.0)
    current_weights = current_weights / current_weights.sum()
    port_value = INITIAL_VALUE
    rebalance_dates.append(sim_dates[0])

    for date in sim_dates:
        target_w = target_weights_series[date].reindex(TICKERS).fillna(0.0)
        target_w = target_w / target_w.sum()

        if strategy == "calendar":
            do_rebalance = True
        elif strategy == "threshold":
            drift = (current_weights - target_w).abs().max()
            do_rebalance = drift > threshold
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if do_rebalance:
            current_weights = target_w.copy()
            rebalance_dates.append(date)

        month_returns = returns.loc[date].reindex(TICKERS).fillna(0.0)
        port_return = current_weights @ month_returns

        # update weights to reflect price drift
        new_asset_values = current_weights * (1 + month_returns)
        current_weights = new_asset_values / new_asset_values.sum()

        port_value *= 1 + port_return
        monthly_returns[date] = port_return
        portfolio_values[date] = port_value

    return pd.Series(monthly_returns), pd.Series(portfolio_values), rebalance_dates


all_returns = {}
all_values = {}

for model, signal in combinations:
    subset = weights_df.xs((model, signal), level=("model", "signal"))
    target_weights_series = {date: subset.loc[date] for date in subset.index}

    for strategy in ["calendar", "threshold"]:
        key = (model, signal, strategy)
        rets, vals, _ = simulate_portfolio(target_weights_series, simple_returns, strategy=strategy)
        all_returns[key] = rets
        all_values[key] = vals

for bench_name in ["60_40", "equal_weight"]:
    w = get_benchmark_weights(bench_name)
    first_date = simple_returns.index[
        simple_returns.index >= weights_df.index.get_level_values("date").min()
        ][0]
    bench_dates = simple_returns.index[simple_returns.index >= first_date]
    target_weights_series = {date: w for date in bench_dates}

    rets, vals, _ = simulate_portfolio(target_weights_series, simple_returns, strategy="calendar")
    all_returns[(bench_name, "none", "calendar")] = rets
    all_values[(bench_name, "none", "calendar")] = vals

returns_df = pd.DataFrame(all_returns)
returns_df.index.name = "date"
returns_df.columns = pd.MultiIndex.from_tuples(
    returns_df.columns, names=["model", "signal", "rebalancing"]
)

values_df = pd.DataFrame(all_values)
values_df.index.name = "date"
values_df.columns = pd.MultiIndex.from_tuples(
    values_df.columns, names=["model", "signal", "rebalancing"]
)

returns_flat = returns_df.copy()
returns_flat.columns = [f"{m}|{s}|{r}" for m, s, r in returns_df.columns]
returns_flat.to_csv("data/block_4/portfolio_returns.csv")

values_flat = values_df.copy()
values_flat.columns = [f"{m}|{s}|{r}" for m, s, r in values_df.columns]
values_flat.to_csv("data/block_4/portfolio_values.csv")

print(
    f"Saved portfolio_returns.csv {returns_flat.shape} and portfolio_values.csv {values_flat.shape}"
)
