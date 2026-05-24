# Block 6 — metrics, charts, and exports

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# config

TICKERS = ["SPY", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "VNQ", "GLD", "XLP"]
N = len(TICKERS)
MONTHS_PER_YEAR = 12
RF = 0.0
VAR_LEVEL = 0.05
BENCHMARK_MARKET = "SPY"
BENCHMARK_IR = "60_40|none|calendar"
EPS = 1e-10

BENCHMARKS = ["60_40|none|calendar", "equal_weight|none|calendar"]

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

MODEL_COLORS = {
    "risk_parity": "#e6550d",
    "markowitz": "#1f77b4",
    "mean_cvar": "#2ca02c",
}

BENCH_STYLES = {
    "60_40|none|calendar": ("#222222", "--", 1.4),
    "equal_weight|none|calendar": ("#888888", ":", 1.4),
}

ASSET_CLASSES = {
    "SPY": "Akcje USA",
    "IWM": "Akcje USA",
    "XLP": "Akcje USA",
    "EFA": "Akcje globalne",
    "EEM": "Akcje globalne",
    "TLT": "Obligacje",
    "IEF": "Obligacje",
    "LQD": "Obligacje",
    "VNQ": "Alternatywne",
    "GLD": "Alternatywne",
}
CLASS_COLORS = {
    "Akcje USA": "#1f77b4",
    "Akcje globalne": "#2ca02c",
    "Obligacje": "#d62728",
    "Alternatywne": "#9467bd",
}

# Unique color per ticker (consistent across charts)
TICKER_COLOR_MAP = dict(zip(TICKERS, sns.color_palette("tab10", N)))

HHI_EQUAL_WEIGHT = 1.0 / N
HHI_60_40 = 0.6**2 + 0.4**2

# shared legend handles (reused across charts)
MODEL_LEGEND_HANDLES = [
    Line2D([0], [0], color=MODEL_COLORS["markowitz"], linewidth=2.0, label="Markowitz"),
    Line2D([0], [0], color=MODEL_COLORS["mean_cvar"], linewidth=2.0, label="Mean-CVaR"),
    Line2D(
        [0], [0], color=MODEL_COLORS["risk_parity"], linewidth=2.0, label="Risk Parity"
    ),
    Line2D(
        [0],
        [0],
        color="#222222",
        linewidth=1.4,
        linestyle="--",
        label="60/40 CAL (benchmark)",
    ),
    Line2D(
        [0],
        [0],
        color="#888888",
        linewidth=1.4,
        linestyle=":",
        label="EW CAL (benchmark)",
    ),
]

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)

# output folders
base_out = Path("data/block_6")
csv_dir = base_out / "csv"
pdf_dir = base_out / "pdf"
csv_dir.mkdir(parents=True, exist_ok=True)
pdf_dir.mkdir(parents=True, exist_ok=True)


def short_label(variant: str) -> str:
    return LABEL_MAP.get(variant, variant)


def save_pdf(fig, path: Path) -> None:
    with PdfPages(path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# load inputs

gross_values = pd.read_csv(
    "data/block_4/portfolio_values.csv", index_col=0, parse_dates=True
)
gross_returns = pd.read_csv(
    "data/block_4/portfolio_returns.csv", index_col=0, parse_dates=True
)
net_values = pd.read_csv(
    "data/block_5/portfolio_values_net.csv", index_col=0, parse_dates=True
)
net_returns = pd.read_csv(
    "data/block_5/portfolio_returns_net.csv", index_col=0, parse_dates=True
)
costs_log = pd.read_csv(
    "data/block_5/transaction_costs_log.csv", index_col=0, parse_dates=True
)
tax_log = pd.read_csv("data/block_5/tax_log.csv")
log_returns = pd.read_csv(
    "data/block_0/log_returns_monthly.csv", index_col=0, parse_dates=True
)[TICKERS]
weights_df = pd.read_csv(
    "data/block_3/portfolio_weights.csv", index_col=[0, 1, 2], parse_dates=True
)
weights_df.index.names = ["date", "model", "signal"]

simple_returns_assets = np.exp(log_returns) - 1
MODELS = [c for c in net_values.columns if c not in BENCHMARKS]


# --- metric functions ---


def total_return(values: pd.Series) -> float:
    v = values.dropna()
    return float(v.iloc[-1] / v.iloc[0] - 1) if len(v) >= 2 else np.nan


def cagr(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return np.nan
    return float(np.prod(1.0 + r.values) ** (1.0 / (len(r) / MONTHS_PER_YEAR)) - 1)


def annualised_vol(returns: pd.Series) -> float:
    r = returns.dropna()
    return float(r.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR)) if len(r) >= 2 else np.nan


def sharpe(returns: pd.Series) -> float:
    c, v = cagr(returns), annualised_vol(returns)
    return float((c - RF) / v) if v and v >= EPS else np.nan


def downside_deviation(returns: pd.Series, mar: float = 0.0) -> float:
    r = returns.dropna()
    neg = r[r < mar / MONTHS_PER_YEAR] - mar / MONTHS_PER_YEAR
    return (
        float(np.sqrt((neg**2).mean()) * np.sqrt(MONTHS_PER_YEAR)) if len(neg) else 0.0
    )


def sortino(returns: pd.Series) -> float:
    c, dd = cagr(returns), downside_deviation(returns, mar=RF)
    return float((c - RF) / dd) if dd >= EPS else np.nan


def max_drawdown(values: pd.Series) -> float:
    v = values.dropna()
    return float((v / v.cummax() - 1.0).min()) if not v.empty else np.nan


def calmar(returns: pd.Series, values: pd.Series) -> float:
    c, mdd = cagr(returns), max_drawdown(values)
    return float(c / abs(mdd)) if mdd and abs(mdd) >= EPS else np.nan


def var_historic(returns: pd.Series, level: float = VAR_LEVEL) -> float:
    r = returns.dropna()
    return float(np.percentile(r, level * 100)) if len(r) >= 2 else np.nan


def cvar_historic(returns: pd.Series, level: float = VAR_LEVEL) -> float:
    r = returns.dropna()
    var = var_historic(r, level)
    return float(r[r <= var].mean()) if not np.isnan(var) else np.nan


def skewness(returns: pd.Series) -> float:
    r = returns.dropna()
    return float(r.skew()) if len(r) >= 3 else np.nan


def excess_kurtosis(returns: pd.Series) -> float:
    r = returns.dropna()
    return float(r.kurt()) if len(r) >= 4 else np.nan


def hhi_concentration(weights_series: pd.DataFrame) -> float:
    hhi_vals = (weights_series**2).sum(axis=1)
    return float(hhi_vals.mean()) if not hhi_vals.empty else np.nan


def compute_all_metrics(returns: pd.Series, values: pd.Series) -> dict:
    return {
        "total_return": total_return(values),
        "cagr": cagr(returns),
        "volatility": annualised_vol(returns),
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "max_drawdown": max_drawdown(values),
        "calmar": calmar(returns, values),
        "var_5pct": var_historic(returns),
        "cvar_5pct": cvar_historic(returns),
        "skewness": skewness(returns),
        "excess_kurtosis": excess_kurtosis(returns),
    }


def get_hhi(model: str, signal: str) -> float:
    try:
        return hhi_concentration(
            weights_df.xs((model, signal), level=("model", "signal"))
        )
    except KeyError:
        return np.nan


def hhi_for_benchmark(name: str) -> float:
    if name == "60_40":
        return 0.6**2 + 0.4**2
    if name == "equal_weight":
        return N * (1.0 / N) ** 2
    return np.nan


spy_returns = simple_returns_assets[BENCHMARK_MARKET]


def beta_alpha_jensen(port_returns, market_returns=None):
    if market_returns is None:
        market_returns = spy_returns
    common = port_returns.dropna().index.intersection(market_returns.dropna().index)
    if len(common) < 12:
        return np.nan, np.nan, np.nan
    rp = port_returns.loc[common] - RF / MONTHS_PER_YEAR
    rm = market_returns.loc[common] - RF / MONTHS_PER_YEAR
    slope, intercept, r_value, _, _ = stats.linregress(rm, rp)
    return float(slope), float(intercept * MONTHS_PER_YEAR), float(r_value**2)


def information_ratio(port_returns, bench_returns):
    common = port_returns.dropna().index.intersection(bench_returns.dropna().index)
    if len(common) < 12:
        return np.nan
    active = port_returns.loc[common] - bench_returns.loc[common]
    te = active.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR)
    return float(active.mean() * MONTHS_PER_YEAR / te) if te >= EPS else np.nan


# main metric loop
records_gross, records_net, records_bair = [], [], []
bench_ir_returns = (
    net_returns[BENCHMARK_IR] if BENCHMARK_IR in net_returns.columns else None
)

for col in gross_values.columns:
    parts = col.split("|")
    model, signal, strategy = parts[0], parts[1], parts[2]

    g = compute_all_metrics(gross_returns[col], gross_values[col])
    g["hhi"] = (
        get_hhi(model, signal)
        if model not in ["60_40", "equal_weight"]
        else hhi_for_benchmark(model)
    )
    g.update({"variant": col, "model": model, "signal": signal, "strategy": strategy})
    records_gross.append(g)

    if col in net_returns.columns and col in net_values.columns:
        n = compute_all_metrics(net_returns[col], net_values[col])
        n["hhi"] = g["hhi"]
        n.update(
            {"variant": col, "model": model, "signal": signal, "strategy": strategy}
        )
        records_net.append(n)

    if col in net_returns.columns:
        beta, alpha_j, r2 = beta_alpha_jensen(net_returns[col])
        ir = (
            information_ratio(net_returns[col], bench_ir_returns)
            if bench_ir_returns is not None
            else np.nan
        )
        records_bair.append(
            {
                "variant": col,
                "model": model,
                "signal": signal,
                "strategy": strategy,
                "beta": beta,
                "alpha_jensen_ann": alpha_j,
                "r_squared": r2,
                "information_ratio": ir,
            }
        )

col_order = [
    "model",
    "signal",
    "strategy",
    "total_return",
    "cagr",
    "volatility",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "var_5pct",
    "cvar_5pct",
    "skewness",
    "excess_kurtosis",
    "hhi",
]

metrics_gross = pd.DataFrame(records_gross).set_index("variant")
metrics_net = pd.DataFrame(records_net).set_index("variant")
metrics_bair = pd.DataFrame(records_bair).set_index("variant")
metrics_gross = metrics_gross[[c for c in col_order if c in metrics_gross.columns]]
metrics_net = metrics_net[[c for c in col_order if c in metrics_net.columns]]

numeric_cols = [c for c in col_order if c not in ["model", "signal", "strategy"]]
metrics_combined = pd.concat(
    [
        metrics_gross[["model", "signal", "strategy"]],
        metrics_gross[numeric_cols].add_suffix("_gross"),
        metrics_net[numeric_cols].add_suffix("_net"),
    ],
    axis=1,
)

drag_records = []
for col in gross_values.columns:
    if col not in net_values.columns:
        continue
    parts = col.split("|")
    model = parts[0]
    signal = parts[1] if len(parts) > 1 else "none"
    strategy = parts[2] if len(parts) > 2 else "calendar"
    g_final = gross_values[col].dropna().iloc[-1]
    n_final = net_values[col].dropna().iloc[-1]
    tc_total = costs_log[col].sum() if col in costs_log.columns else np.nan
    tax_total = tax_log[
        (tax_log["model"] == model)
        & (tax_log["signal"] == signal)
        & (tax_log["strategy"] == strategy)
    ]["tax"].sum()
    drag_records.append(
        {
            "variant": col,
            "model": model,
            "signal": signal,
            "strategy": strategy,
            "gross_final_value": g_final,
            "net_final_value": n_final,
            "total_drag": (g_final - n_final) / g_final,
            "tc_total": tc_total,
            "tax_total": tax_total,
            "tc_drag_pct": tc_total / 1.0,
            "tax_drag_pct": tax_total / 1.0,
        }
    )
drag_df = pd.DataFrame(drag_records).set_index("variant")

metrics_gross.to_csv(csv_dir / "metrics_gross.csv")
metrics_net.to_csv(csv_dir / "metrics_net.csv")
metrics_combined.to_csv(csv_dir / "metrics_combined.csv")
drag_df.to_csv(csv_dir / "drag_decomposition.csv")
metrics_bair.to_csv(csv_dir / "beta_alpha_ir.csv")

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 160)

# --- performance & cost charts ---

PANEL_ORDER = [k for k in LABEL_MAP if k not in BENCHMARKS]

# 1. portfolio value curves
fig, axes = plt.subplots(5, 3, figsize=(14, 16), sharex=True, sharey=True)
axes_flat = axes.flatten()

for idx, col in enumerate(PANEL_ORDER):
    ax = axes_flat[idx]
    for bench in BENCHMARKS:
        clr, ls, lw = BENCH_STYLES[bench]
        ax.plot(
            net_values.index,
            net_values[bench],
            color=clr,
            linestyle=ls,
            linewidth=lw,
            alpha=0.6,
            zorder=1,
        )
    color = MODEL_COLORS.get(col.split("|")[0], "#333333")
    ax.plot(net_values.index, net_values[col], color=color, linewidth=2.0, zorder=2)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}×"))
    ax.set_title(LABEL_MAP[col], fontsize=9, fontweight="bold", pad=4)
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(labelsize=8)
    if idx % 3 == 0:
        ax.set_ylabel("Wartość (start=1.0)", fontsize=9)

axes_flat[14].set_visible(False)
fig.autofmt_xdate(rotation=30)
fig.legend(
    handles=MODEL_LEGEND_HANDLES,
    loc="lower center",
    ncol=5,
    fontsize=9,
    framealpha=0.85,
    bbox_to_anchor=(0.5, -0.01),
)
fig.suptitle(
    "Ścieżki wartości portfela — netto (skala logarytmiczna)",
    fontsize=12,
    fontweight="bold",
)
fig.tight_layout(rect=[0, 0.04, 1, 1])
save_pdf(fig, pdf_dir / "portfolio_curves.pdf")


# 2. drawdown chart
def drawdown_series(values: pd.Series) -> pd.Series:
    v = values.dropna()
    return v / v.cummax() - 1.0


# Compute global min drawdown to set dynamic scale
all_dd_min = min(
    drawdown_series(net_values[col]).min()
    for col in list(PANEL_ORDER) + BENCHMARKS
    if col in net_values.columns
)
y_bottom = max(all_dd_min * 1.15, -1.0)  # 15% padding, not less than -100%

fig, axes = plt.subplots(5, 3, figsize=(14, 17), sharex=True, sharey=True)
axes_flat = axes.flatten()

for idx, col in enumerate(PANEL_ORDER):
    ax = axes_flat[idx]
    for bench in BENCHMARKS:
        clr, ls, lw = BENCH_STYLES[bench]
        dd_b = drawdown_series(net_values[bench])
        ax.fill_between(dd_b.index, dd_b.values, 0, color=clr, alpha=0.08)
        ax.plot(
            dd_b.index,
            dd_b.values,
            color=clr,
            linestyle=ls,
            linewidth=lw,
            alpha=0.6,
            zorder=1,
        )
    color = MODEL_COLORS.get(col.split("|")[0], "#333333")
    dd = drawdown_series(net_values[col])
    ax.fill_between(dd.index, dd.values, 0, color=color, alpha=0.25, zorder=2)
    ax.plot(dd.index, dd.values, color=color, linewidth=1.8, zorder=3)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_title(LABEL_MAP[col], fontsize=9, fontweight="bold", pad=4)
    ax.set_ylim(y_bottom, 0.02)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)
    if idx % 3 == 0:
        ax.set_ylabel("Drawdown", fontsize=9)

# hide empty panel but keep x-axis labels for column 3
axes_flat[14].set_visible(False)

# force date labels on the last visible panel in column 3 (idx=11)
axes_flat[11].tick_params(axis="x", labelbottom=True, labelsize=8)
axes_flat[11].xaxis.set_tick_params(which="both", labelbottom=True)
for label in axes_flat[11].get_xticklabels():
    label.set_visible(True)
    label.set_rotation(30)
    label.set_ha("right")

axes_flat[14].set_visible(False)
fig.autofmt_xdate(rotation=30)
fig.legend(
    handles=MODEL_LEGEND_HANDLES,
    loc="lower center",
    ncol=5,
    fontsize=9,
    framealpha=0.85,
    bbox_to_anchor=(0.5, -0.01),
)
fig.suptitle("Przebieg drawdown w czasie — netto", fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0.04, 1, 0.97])
save_pdf(fig, pdf_dir / "drawdown_chart.pdf")

# 3. metrics heatmap
heat_cols = ["sharpe", "sortino", "calmar"]
heat_titles = ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"]
HEAT_CMAPS = ["Blues", "Greens", "Purples"]
VARIANTS_ORDER = list(LABEL_MAP.values())

mg = metrics_gross[heat_cols].copy()
mn = metrics_net[heat_cols].copy()
mg.index = [LABEL_MAP.get(i, i) for i in mg.index]
mn.index = [LABEL_MAP.get(i, i) for i in mn.index]
mg = mg.reindex(VARIANTS_ORDER)
mn = mn.reindex(VARIANTS_ORDER)

fig, axes = plt.subplots(1, 3, figsize=(15, 7), gridspec_kw={"wspace": 0.06})
for ax, col, title, cmap in zip(axes, heat_cols, heat_titles, HEAT_CMAPS):
    panel = pd.DataFrame(
        {"Brutto": mg[col], "Netto": mn[col]}, index=VARIANTS_ORDER
    ).astype(float)
    vmin, vmax = panel.min().min(), panel.max().max()
    norm = Normalize(vmin=vmin, vmax=vmax)
    ax.imshow(panel.values, cmap=cmap, norm=norm, aspect="auto")
    for i in range(panel.shape[0]):
        for j in range(panel.shape[1]):
            val = panel.iloc[i, j]
            txt = "—" if np.isnan(val) else f"{val:.2f}"
            bg = plt.get_cmap(cmap)(norm(val) if not np.isnan(val) else 0.5)
            lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color="white" if lum < 0.55 else "black",
            )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Brutto", "Netto"], fontsize=9, fontweight="bold")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
    if ax is axes[0]:
        ax.set_yticks(range(len(VARIANTS_ORDER)))
        ax.set_yticklabels(VARIANTS_ORDER, fontsize=8.5)
    else:
        ax.set_yticks(range(len(VARIANTS_ORDER)))
        ax.set_yticklabels([""] * len(VARIANTS_ORDER))
    for y in np.arange(-0.5, len(VARIANTS_ORDER), 1):
        ax.axhline(y, color="white", linewidth=0.8)
    ax.axvline(0.5, color="white", linewidth=0.8)
    ax.axhline(13.5, color="black", linewidth=1.5, linestyle="--", alpha=0.6)
    fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.35, pad=0.02, aspect=18
    ).ax.tick_params(labelsize=8)

fig.suptitle(
    "Heatmapa wskaźników jakości portfela — brutto vs netto",
    fontsize=12,
    fontweight="bold",
    y=1.02,
)
fig.tight_layout()
save_pdf(fig, pdf_dir / "metrics_heatmap.pdf")

# 4. CAGR gross vs net
variants = metrics_gross.index.tolist()
xlabels = [LABEL_MAP.get(v, v) for v in variants]
x = np.arange(len(variants))
width = 0.38

fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(
    x - width / 2,
    metrics_gross["cagr"] * 100,
    width,
    label="Brutto",
    color="#1f77b4",
    alpha=0.85,
)
ax.bar(
    x + width / 2,
    metrics_net["cagr"] * 100,
    width,
    label="Netto",
    color="#1f77b4",
    alpha=0.40,
    hatch="///",
    edgecolor="#1f77b4",
)
ax.set_xticks(x)
ax.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=8)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=1))
ax.set_title(
    "CAGR brutto vs netto dla wszystkich wariantów", fontsize=10, fontweight="bold"
)
ax.set_xlabel("Wariant", fontsize=9)
ax.set_ylabel("CAGR (%)", fontsize=9)
ax.legend(fontsize=9, framealpha=0.85)
ax.grid(True, axis="y", alpha=0.3)
ax.tick_params(labelsize=8)
fig.tight_layout()
save_pdf(fig, pdf_dir / "cagr_gross_vs_net.pdf")

# 5. monthly returns boxplot
BOX_COLORS = [
    MODEL_COLORS.get(col.split("|")[0], "#888888") for col in net_returns.columns
]
ret_data = [net_returns[col].dropna().values * 100 for col in net_returns.columns]
xlabels_r = [LABEL_MAP.get(col, col) for col in net_returns.columns]

fig, ax = plt.subplots(figsize=(13, 5))
bp = ax.boxplot(
    ret_data,
    patch_artist=True,
    medianprops={"color": "black", "linewidth": 1.5},
    flierprops={"marker": "o", "markersize": 2.5, "alpha": 0.4},
    whiskerprops={"linewidth": 0.8},
    capprops={"linewidth": 0.8},
)
for patch, color in zip(bp["boxes"], BOX_COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
ax.set_xticks(range(1, len(xlabels_r) + 1))
ax.set_xticklabels(xlabels_r, rotation=35, ha="right", fontsize=8)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=1))
ax.set_title(
    "Rozkład miesięcznych zwrotów netto — wszystkie warianty",
    fontsize=10,
    fontweight="bold",
)
ax.set_ylabel("Miesięczny zwrot (%)", fontsize=9)
ax.grid(True, axis="y", alpha=0.3)
ax.tick_params(labelsize=8)
ax.legend(
    handles=[
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS["markowitz"],
            linewidth=6,
            alpha=0.75,
            label="Markowitz",
        ),
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS["mean_cvar"],
            linewidth=6,
            alpha=0.75,
            label="Mean-CVaR",
        ),
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS["risk_parity"],
            linewidth=6,
            alpha=0.75,
            label="Risk Parity",
        ),
        Line2D([0], [0], color="#888888", linewidth=6, alpha=0.75, label="Benchmarki"),
    ],
    fontsize=9,
    framealpha=0.85,
    loc="upper right",
)
fig.tight_layout()
save_pdf(fig, pdf_dir / "returns_boxplot.pdf")

# 6. transaction costs vs total drag
SCATTER_COLORS = [MODEL_COLORS.get(v.split("|")[0], "#888888") for v in drag_df.index]

fig, ax = plt.subplots(figsize=(8, 5.5))
for i, (variant, row) in enumerate(drag_df.iterrows()):
    x_val = row["tc_total"] * 100
    y_val = row["total_drag"] * 100
    ax.scatter(
        x_val,
        y_val,
        color=SCATTER_COLORS[i],
        s=80,
        zorder=3,
        edgecolors="white",
        linewidth=0.6,
    )
    ax.annotate(
        LABEL_MAP.get(variant, variant),
        (x_val, y_val),
        textcoords="offset points",
        xytext=(5, 3),
        fontsize=7.5,
        color=SCATTER_COLORS[i],
    )
max_x = drag_df["tc_total"].max() * 100 * 1.1
ax.plot(
    [0, max_x],
    [0, max_x],
    color="gray",
    linestyle="--",
    linewidth=0.9,
    label="Drag = TC (bez podatku)",
    zorder=1,
)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(decimals=2))
ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=1))
ax.set_title(
    "Koszty transakcyjne a łączny uszczerbek na wartości (brutto − netto)",
    fontsize=10,
    fontweight="bold",
)
ax.set_xlabel("Łączne koszty transakcyjne (% wartości initial)", fontsize=9)
ax.set_ylabel("Całkowity drag (%)", fontsize=9)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=8)
ax.legend(
    handles=[
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS["markowitz"],
            linewidth=6,
            alpha=0.85,
            label="Markowitz",
        ),
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS["mean_cvar"],
            linewidth=6,
            alpha=0.85,
            label="Mean-CVaR",
        ),
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS["risk_parity"],
            linewidth=6,
            alpha=0.85,
            label="Risk Parity",
        ),
        Line2D([0], [0], color="#888888", linewidth=6, alpha=0.85, label="Benchmarki"),
        Line2D(
            [0], [0], color="gray", linestyle="--", linewidth=1.2, label="Drag = TC"
        ),
    ],
    fontsize=9,
    framealpha=0.85,
)
fig.tight_layout()
save_pdf(fig, pdf_dir / "cost_vs_drag.pdf")

# 7. normalized ETF prices
ASSET_GROUPS = {
    "Akcje USA": ["SPY", "IWM", "XLP"],
    "Akcje globalne": ["EFA", "EEM"],
    "Obligacje": ["TLT", "IEF", "LQD"],
    "Alternatywne": ["VNQ", "GLD"],
}
simple_returns_plot = np.exp(log_returns) - 1
prices_norm = (1 + simple_returns_plot).cumprod()
prices_norm = prices_norm / prices_norm.iloc[0]

fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
axes = axes.flatten()
for ax, (group_name, tickers_in_group) in zip(axes, ASSET_GROUPS.items()):
    for ticker in tickers_in_group:
        if ticker not in prices_norm.columns:
            continue
        ax.plot(
            prices_norm.index,
            prices_norm[ticker],
            label=ticker,
            color=TICKER_COLOR_MAP[ticker],
            linewidth=1.8,
        )
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_title(group_name, fontsize=10, fontweight="bold")
    ax.set_ylabel("Znormalizowana cena (start = 1.0)", fontsize=9)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}×"))
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

fig.suptitle("Znormalizowane ceny ETF w okresie próby", fontsize=12, fontweight="bold")
fig.autofmt_xdate(rotation=30)
fig.tight_layout()
save_pdf(fig, pdf_dir / "asset_prices_normalized.pdf")


# --- HHI concentration analysis ---

PANEL_CONFIGS = [
    ("markowitz", "historical_mean", "MKT HM"),
    ("markowitz", "capm", "MKT CAPM"),
    ("markowitz", "ff3f", "MKT FF3F"),
    ("mean_cvar", "historical_mean", "MCVAR HM"),
    ("mean_cvar", "capm", "MCVAR CAPM"),
    ("mean_cvar", "ff3f", "MCVAR FF3F"),
    ("risk_parity", "none", "RP"),
]


def compute_hhi_series(model: str, signal: str) -> pd.Series:
    try:
        w = weights_df.xs((model, signal), level=("model", "signal"))[TICKERS]
        return (w**2).sum(axis=1)
    except KeyError:
        return pd.Series(dtype=float)


# HHI bar chart
ORDERED_LABELS = [
    "MKT HM CAL",
    "MKT HM THR",
    "MKT CAPM CAL",
    "MKT CAPM THR",
    "MKT FF3F CAL",
    "MKT FF3F THR",
    "MCVAR HM CAL",
    "MCVAR HM THR",
    "MCVAR CAPM CAL",
    "MCVAR CAPM THR",
    "MCVAR FF3F CAL",
    "MCVAR FF3F THR",
    "RP CAL",
    "RP THR",
]
hhi_series = metrics_gross["hhi"].copy()
hhi_series.index = [LABEL_MAP.get(i, i) for i in hhi_series.index]
hhi_ordered = hhi_series.reindex(ORDERED_LABELS)

bar_colors = []
for label in ORDERED_LABELS:
    if "MKT" in label:
        bar_colors.append(MODEL_COLORS["markowitz"])
    elif "MCVAR" in label:
        bar_colors.append(MODEL_COLORS["mean_cvar"])
    elif "RP" in label:
        bar_colors.append(MODEL_COLORS["risk_parity"])

fig, ax = plt.subplots(figsize=(12, 5.5))
bars = ax.bar(
    range(len(ORDERED_LABELS)),
    hhi_ordered.values,
    color=bar_colors,
    alpha=0.85,
    edgecolor="white",
    linewidth=0.6,
)
ax.axhline(HHI_EQUAL_WEIGHT, color="#2ca02c", linestyle="--", linewidth=1.2, alpha=0.8)
for bar, val in zip(bars, hhi_ordered.values):
    if not np.isnan(val):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.003,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
        )
ax.set_xticks(range(len(ORDERED_LABELS)))
ax.set_xticklabels(ORDERED_LABELS, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("HHI (średnie w czasie)", fontsize=9)
ax.set_title(
    "Koncentracja portfela — wskaźnik Herfindahla-Hirschmana (HHI)",
    fontsize=10,
    fontweight="bold",
)
ax.set_ylim(0, hhi_ordered.dropna().max() * 1.15)
ax.grid(True, axis="y", alpha=0.3)
ax.tick_params(labelsize=8)
ax.legend(
    handles=[
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS["markowitz"],
            linewidth=6,
            alpha=0.85,
            label="Markowitz",
        ),
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS["mean_cvar"],
            linewidth=6,
            alpha=0.85,
            label="Mean-CVaR",
        ),
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS["risk_parity"],
            linewidth=6,
            alpha=0.85,
            label="Risk Parity",
        ),
        Line2D(
            [0],
            [0],
            color="#2ca02c",
            linestyle="--",
            linewidth=1.5,
            label=f"EW ref. ({HHI_EQUAL_WEIGHT:.3f})",
        ),
    ],
    fontsize=9,
    framealpha=0.85,
    loc="upper right",
    ncol=2,
)
fig.tight_layout()
save_pdf(fig, pdf_dir / "hhi_barplot.pdf")

# HHI evolution
fig, axes = plt.subplots(4, 2, figsize=(13, 14), sharex=True, sharey=True)
axes_flat = axes.flatten()

for idx, (model, signal, label) in enumerate(PANEL_CONFIGS):
    ax = axes_flat[idx]
    hhi_ts = compute_hhi_series(model, signal)
    if hhi_ts.empty:
        ax.set_visible(False)
        continue
    color = MODEL_COLORS.get(model, "#333333")
    hhi_smooth = hhi_ts.rolling(3, min_periods=1).mean()
    ax.fill_between(hhi_ts.index, hhi_ts.values, alpha=0.2, color=color)
    ax.plot(hhi_ts.index, hhi_ts.values, color=color, linewidth=0.8, alpha=0.5)
    ax.plot(hhi_smooth.index, hhi_smooth.values, color=color, linewidth=1.8)
    ax.axhline(
        HHI_EQUAL_WEIGHT, color="#2ca02c", linestyle="--", linewidth=1.0, alpha=0.7
    )
    ax.axhline(HHI_60_40, color="#222222", linestyle=":", linewidth=1.0, alpha=0.6)
    ax.set_title(label, fontsize=9, fontweight="bold", pad=4)
    ax.set_ylim(0, 0.8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)
    if idx % 2 == 0:
        ax.set_ylabel("HHI", fontsize=9)

axes_flat[7].set_visible(False)
fig.autofmt_xdate(rotation=30)
fig.legend(
    handles=[
        Line2D(
            [0], [0], color=MODEL_COLORS["markowitz"], linewidth=2, label="Markowitz"
        ),
        Line2D(
            [0], [0], color=MODEL_COLORS["mean_cvar"], linewidth=2, label="Mean-CVaR"
        ),
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS["risk_parity"],
            linewidth=2,
            label="Risk Parity",
        ),
        Line2D(
            [0],
            [0],
            color="#2ca02c",
            linestyle="--",
            linewidth=1.5,
            label=f"EW ref. ({HHI_EQUAL_WEIGHT:.3f})",
        ),
        Line2D(
            [0],
            [0],
            color="#222222",
            linestyle=":",
            linewidth=1.5,
            label=f"60/40 ref. ({HHI_60_40:.3f})",
        ),
    ],
    loc="lower center",
    ncol=5,
    fontsize=9,
    framealpha=0.85,
    bbox_to_anchor=(0.5, -0.01),
)
fig.suptitle(
    "Ewolucja koncentracji portfela w czasie — wskaźnik HHI",
    fontsize=12,
    fontweight="bold",
)
fig.tight_layout(rect=[0, 0.04, 1, 1])
save_pdf(fig, pdf_dir / "hhi_evolution.pdf")


# --- risk context charts ---

simple_returns = np.exp(log_returns) - 1

# A. asset correlation matrix
corr = simple_returns.corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

fig, ax = plt.subplots(figsize=(9, 7.5))
sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    ax=ax,
    cbar_kws={"shrink": 0.7, "label": "Współczynnik korelacji Pearsona"},
    annot_kws={"size": 9},
)
ax.set_title(
    "Macierz korelacji miesięcznych zwrotów ETF\n(pełny okres próby)",
    fontsize=12,
    fontweight="bold",
    pad=12,
)
ax.tick_params(labelsize=8)
for lbl in ax.get_xticklabels():
    ticker = lbl.get_text()
    lbl.set_color(CLASS_COLORS.get(ASSET_CLASSES.get(ticker, ""), "black"))
    lbl.set_fontweight("bold")
for lbl in ax.get_yticklabels():
    ticker = lbl.get_text()
    lbl.set_color(CLASS_COLORS.get(ASSET_CLASSES.get(ticker, ""), "black"))
    lbl.set_fontweight("bold")
ax.legend(
    handles=[
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=color,
            markersize=10,
            label=cls,
        )
        for cls, color in CLASS_COLORS.items()
    ],
    loc="upper right",
    fontsize=9,
    framealpha=0.85,
    title="Klasa aktywów",
)
fig.tight_layout()
save_pdf(fig, pdf_dir / "asset_correlation.pdf")

# B. risk-return scatter
MONTHS = len(simple_returns)
cagr_assets = ((1 + simple_returns).prod() ** (12 / MONTHS) - 1) * 100
vol_assets = simple_returns.std(ddof=1) * np.sqrt(12) * 100

fig, ax = plt.subplots(figsize=(8, 6))
for ticker in TICKERS:
    cls = ASSET_CLASSES.get(ticker, "")
    color = CLASS_COLORS.get(cls, "gray")
    ax.scatter(
        vol_assets[ticker],
        cagr_assets[ticker],
        color=color,
        s=120,
        zorder=3,
        edgecolors="white",
        linewidth=0.8,
    )
    ax.annotate(
        ticker,
        (vol_assets[ticker], cagr_assets[ticker]),
        textcoords="offset points",
        xytext=(7, 3),
        fontsize=9,
        fontweight="bold",
        color=color,
    )
x_line = np.linspace(0, vol_assets.max() * 1.15, 100)
ax.plot(
    x_line,
    0.5 * x_line,
    color="gray",
    linestyle="--",
    linewidth=0.9,
    alpha=0.6,
    label="Sharpe = 0.5",
)
ax.plot(
    x_line,
    1.0 * x_line,
    color="gray",
    linestyle=":",
    linewidth=0.9,
    alpha=0.6,
    label="Sharpe = 1.0",
)
ax.axhline(0, color="black", linewidth=0.7, alpha=0.4)
ax.set_xlabel("Roczna zmienność (%)", fontsize=9)
ax.set_ylabel("CAGR (%)", fontsize=9)
ax.set_title(
    "Profil ryzyko–zwrot poszczególnych ETF\n(pełny okres próby, dane miesięczne)",
    fontsize=12,
    fontweight="bold",
)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=1))
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=8)
ax.legend(
    handles=[
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markersize=9,
            label=cls,
        )
        for cls, color in CLASS_COLORS.items()
    ]
    + [
        Line2D([0], [0], color="gray", linestyle="--", label="Sharpe = 0.5"),
        Line2D([0], [0], color="gray", linestyle=":", label="Sharpe = 1.0"),
    ],
    fontsize=9,
    framealpha=0.85,
)
fig.tight_layout()
save_pdf(fig, pdf_dir / "asset_risk_return.pdf")

# C. portfolio weights evolution
WEIGHT_PANEL_ORDER = [
    ("markowitz", "historical_mean"),
    ("markowitz", "capm"),
    ("markowitz", "ff3f"),
    ("mean_cvar", "historical_mean"),
    ("mean_cvar", "capm"),
    ("mean_cvar", "ff3f"),
    ("risk_parity", "none"),
]
WEIGHT_LABEL_MAP = {
    ("markowitz", "historical_mean"): "MKT HM",
    ("markowitz", "capm"): "MKT CAPM",
    ("markowitz", "ff3f"): "MKT FF3F",
    ("mean_cvar", "historical_mean"): "MCVAR HM",
    ("mean_cvar", "capm"): "MCVAR CAPM",
    ("mean_cvar", "ff3f"): "MCVAR FF3F",
    ("risk_parity", "none"): "RP",
}

fig, axes = plt.subplots(
    7, 2, figsize=(14, 22), sharex=True, gridspec_kw={"hspace": 0.35, "wspace": 0.08}
)

for row_idx, (model, signal) in enumerate(WEIGHT_PANEL_ORDER):
    for col_idx, strategy in enumerate(["calendar", "threshold"]):
        ax = axes[row_idx, col_idx]
        try:
            w = weights_df.xs((model, signal), level=("model", "signal"))
        except KeyError:
            ax.set_visible(False)
            continue
        w = w[TICKERS].sort_index()
        ax.stackplot(
            w.index,
            [w[t].values for t in TICKERS],
            labels=TICKERS,
            colors=[TICKER_COLOR_MAP[t] for t in TICKERS],
            alpha=0.85,
        )
        suffix = "CAL" if strategy == "calendar" else "THR"
        ax.set_title(
            f"{WEIGHT_LABEL_MAP[(model, signal)]} — {suffix}",
            fontsize=9,
            fontweight="bold",
            pad=3,
        )
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        if col_idx == 0:
            ax.set_ylabel("Udział w portfelu", fontsize=9)

fig.autofmt_xdate(rotation=30)
fig.legend(
    handles=[
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=TICKER_COLOR_MAP[t],
            markersize=10,
            label=t,
        )
        for t in TICKERS
    ],
    loc="lower center",
    ncol=5,
    fontsize=9,
    framealpha=0.85,
    bbox_to_anchor=(0.5, -0.01),
    title="ETF",
)
fig.suptitle(
    "Ewolucja wag portfela w czasie — wszystkie warianty\n"
    "Kolumna lewa: Calendar Rebalancing | Kolumna prawa: Threshold Rebalancing",
    fontsize=12,
    fontweight="bold",
)
fig.subplots_adjust(
    left=0.07, right=0.97, top=0.95, bottom=0.06, hspace=0.35, wspace=0.08
)
save_pdf(fig, pdf_dir / "weights_evolution.pdf")

print(f"All outputs saved to: {base_out.resolve()}")
