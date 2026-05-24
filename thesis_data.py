import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("data/thesis/csv")
OUT.mkdir(parents=True, exist_ok=True)

# load inputs
log_returns = pd.read_csv(
    "data/block_0/log_returns_monthly.csv",
    index_col=0,
    parse_dates=True,
    date_format="ISO8601",
)
net_values = pd.read_csv(
    "data/block_5/portfolio_values_net.csv", index_col=0, parse_dates=True
)
gross_values = pd.read_csv(
    "data/block_4/portfolio_values.csv", index_col=0, parse_dates=True
)
net_returns = pd.read_csv(
    "data/block_5/portfolio_returns_net.csv", index_col=0, parse_dates=True
)
weights_df = pd.read_csv(
    "data/block_3/portfolio_weights.csv",
    index_col=[0, 1, 2],
    parse_dates=True,
    date_format="ISO8601",
)
weights_df.index.names = ["date", "model", "signal"]

csv_dir = Path("data/block_6/csv")
metrics_gross = pd.read_csv(csv_dir / "metrics_gross.csv", index_col=0)
metrics_net = pd.read_csv(csv_dir / "metrics_net.csv", index_col=0)
drag_df = pd.read_csv(csv_dir / "drag_decomposition.csv", index_col=0)
bair_df = pd.read_csv(csv_dir / "beta_alpha_ir.csv", index_col=0)

TICKERS = ["SPY", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "VNQ", "GLD", "XLP"]
ASSET_CLASSES = {
    "SPY": "US equities",
    "IWM": "US equities",
    "XLP": "US equities",
    "EFA": "Global equities",
    "EEM": "Global equities",
    "TLT": "Bonds",
    "IEF": "Bonds",
    "LQD": "Bonds",
    "VNQ": "Alternatives",
    "GLD": "Alternatives",
}
LABEL_MAP = {
    "markowitz|historical_mean|calendar": "MKT HM CAL",
    "markowitz|historical_mean|threshold": "MKT HM THR",
    "markowitz|capm|calendar": "MKT CAPM CAL",
    "markowitz|capm|threshold": "MKT CAPM THR",
    "markowitz|ff3f|calendar": "MKT FF3F CAL",
    "markowitz|ff3f|threshold": "MKT FF3F THR",
    "mean_cvar|historical_mean|calendar": "MCVAR HM CAL",
    "mean_cvar|historical_mean|threshold": "MCVAR HM THR",
    "mean_cvar|capm|calendar": "MCVAR CAPM CAL",
    "mean_cvar|capm|threshold": "MCVAR CAPM THR",
    "mean_cvar|ff3f|calendar": "MCVAR FF3F CAL",
    "mean_cvar|ff3f|threshold": "MCVAR FF3F THR",
    "risk_parity|none|calendar": "RP CAL",
    "risk_parity|none|threshold": "RP THR",
    "60_40|none|calendar": "60/40 CAL",
    "equal_weight|none|calendar": "EW CAL",
}

PANEL_CONFIGS = [
    ("markowitz", "historical_mean"),
    ("markowitz", "capm"),
    ("markowitz", "ff3f"),
    ("mean_cvar", "historical_mean"),
    ("mean_cvar", "capm"),
    ("mean_cvar", "ff3f"),
    ("risk_parity", "none"),
]

simple_returns = np.exp(log_returns[TICKERS]) - 1
MONTHS = len(simple_returns)


def save(df, name):
    df.to_csv(OUT / f"{name}.csv")


# 1. asset summary (risk/return and price charts)
rolling_max = (1 + simple_returns).cumprod().cummax()
max_dd = ((1 + simple_returns).cumprod() / rolling_max - 1).min()

asset_summary = pd.DataFrame(
    {
        "asset_class": pd.Series(ASSET_CLASSES),
        "cagr_pct": ((1 + simple_returns).prod() ** (12 / MONTHS) - 1) * 100,
        "annual_volatility_pct": simple_returns.std(ddof=1) * np.sqrt(12) * 100,
        "sharpe_ratio": (
            ((1 + simple_returns).prod() ** (12 / MONTHS) - 1)
            / (simple_returns.std(ddof=1) * np.sqrt(12))
        ),
        "max_drawdown_pct": max_dd * 100,
        "skewness": simple_returns.skew(),
        "kurtosis": simple_returns.kurt(),
    }
).round(4)
save(asset_summary, "assets_summary")

# 2. asset correlation matrix
save(simple_returns.corr().round(4), "assets_correlation")

# 3. normalized asset prices
prices_norm = (1 + simple_returns).cumprod()
prices_norm = (prices_norm / prices_norm.iloc[0]).round(4)
save(prices_norm, "assets_prices_normalized")

# 4. gross and net metrics
mg = metrics_gross.copy()
mn = metrics_net.copy()
mg.index = [LABEL_MAP.get(i, i) for i in mg.index]
mn.index = [LABEL_MAP.get(i, i) for i in mn.index]
mg.columns = [f"{c}_gross" for c in mg.columns]
mn.columns = [f"{c}_net" for c in mn.columns]
save(pd.concat([mg, mn], axis=1), "metrics_all")

# 5. drag decomposition (TC and tax)
drag_out = drag_df.copy()
drag_out.index = [LABEL_MAP.get(i, i) for i in drag_out.index]
save(drag_out, "drag_decomposition")

# 6. beta, alpha, and information ratio
bair_out = bair_df.copy()
bair_out.index = [LABEL_MAP.get(i, i) for i in bair_out.index]
save(bair_out, "beta_alpha_ir")

# 7. drawdown time series (net)
dd_data = {}
for col in net_values.columns:
    v = net_values[col].dropna()
    dd_data[LABEL_MAP.get(col, col)] = (v / v.cummax() - 1).reindex(net_values.index)
save(pd.DataFrame(dd_data).round(6), "drawdowns_net")

# 8. HHI time series per model/signal
N = len(TICKERS)
hhi_data = {}
for model, signal in PANEL_CONFIGS:
    try:
        w = weights_df.xs((model, signal), level=("model", "signal"))[TICKERS]
        hhi_data[f"{model}|{signal}"] = (w**2).sum(axis=1)
    except KeyError:
        pass
save(pd.DataFrame(hhi_data).round(6), "hhi_time_series")

# 9. average portfolio weights
weight_summaries = []
for model, signal in PANEL_CONFIGS:
    try:
        w = weights_df.xs((model, signal), level=("model", "signal"))[TICKERS]
        row = w.mean()
        row.name = f"{model}|{signal}"
        weight_summaries.append(row)
    except KeyError:
        pass
save(pd.DataFrame(weight_summaries).round(4), "average_weights")

# 10. net portfolio values (time series)
pv_out = net_values.copy()
pv_out.columns = [LABEL_MAP.get(c, c) for c in pv_out.columns]
save(pv_out.round(6), "portfolio_values_net")

# 11. monthly net returns
ret_out = net_returns.copy()
ret_out.columns = [LABEL_MAP.get(c, c) for c in ret_out.columns]
save(ret_out.round(6), "monthly_returns_net")

print(f"All data saved to: {OUT.resolve()}")
