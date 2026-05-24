import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# figure sizes
SIZE_1P = (8.0, 4.8)  # 1 panel (scatter, barplot, boxplot)
SIZE_2P = (8.0, 8.5)  # 2 panels, vertical
SIZE_3P = (8.0, 10.0)  # 3 panels, vertical
SIZE_SQ = (7.2, 6.2)  # square (correlation matrix)
SIZE_HM = (9.5, 5.5)  # 2-column heatmap

# style
plt.rcParams.update(
    {
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "figure.dpi": 300,
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
    }
)

# palette and fixed colors
MODEL_COLORS = {
    "risk_parity": "#e6550d",
    "markowitz": "#1f77b4",
    "mean_cvar": "#2ca02c",
    "benchmark": "#888888",
}
BENCH_STYLES = {
    "60_40|none|calendar": ("#222222", "--", 1.5),
    "equal_weight|none|calendar": ("#888888", ":", 1.5),
}
SIGNAL_LS = {
    "historical_mean": ("solid", 1.5),
    "capm": ("dashed", 1.5),
    "ff3f": ("dotted", 1.5),
    "none": ("solid", 1.5),
}
REBAL_ALPHA = {"calendar": 1.0, "threshold": 0.55}

CLASS_COLORS = {
    "Akcje USA": "#1f77b4",
    "Akcje globalne": "#2ca02c",
    "Obligacje": "#d62728",
    "Alternatywne": "#9467bd",
}
TICKER_COLORS = [
    "#1f77b4",
    "#aec7e8",
    "#6baed6",
    "#2ca02c",
    "#98df8a",
    "#d62728",
    "#ff9896",
    "#e6550d",
    "#9467bd",
    "#c5b0d5",
]

LABEL_MAP = {
    "markowitz|historical_mean|calendar": "MKT HM KAL",
    "markowitz|historical_mean|threshold": "MKT HM PROG",
    "markowitz|capm|calendar": "MKT CAPM KAL",
    "markowitz|capm|threshold": "MKT CAPM PROG",
    "markowitz|ff3f|calendar": "MKT FF3F KAL",
    "markowitz|ff3f|threshold": "MKT FF3F PROG",
    "mean_cvar|historical_mean|calendar": "MCVAR HM KAL",
    "mean_cvar|historical_mean|threshold": "MCVAR HM PROG",
    "mean_cvar|capm|calendar": "MCVAR CAPM KAL",
    "mean_cvar|capm|threshold": "MCVAR CAPM PROG",
    "mean_cvar|ff3f|calendar": "MCVAR FF3F KAL",
    "mean_cvar|ff3f|threshold": "MCVAR FF3F PROG",
    "risk_parity|none|calendar": "RP KAL",
    "risk_parity|none|threshold": "RP PROG",
    "60_40|none|calendar": "60/40 KAL",
    "equal_weight|none|calendar": "EW KAL",
}
VARIANTS_ORDER = list(LABEL_MAP.values())

TICKERS = ["SPY", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "VNQ", "GLD", "XLP"]
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

BENCHMARKS = ["60_40|none|calendar", "equal_weight|none|calendar"]

# paths and data
OUT = Path("data/thesis/pdf")
OUT.mkdir(parents=True, exist_ok=True)

net_values = pd.read_csv(
    "data/block_5/portfolio_values_net.csv", index_col=0, parse_dates=True
)
net_returns = pd.read_csv(
    "data/block_5/portfolio_returns_net.csv", index_col=0, parse_dates=True
)
log_returns = pd.read_csv(
    "data/block_0/log_returns_monthly.csv",
    index_col=0,
    parse_dates=True,
    date_format="ISO8601",
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

simple_returns = np.exp(log_returns[TICKERS]) - 1
MONTHS = len(simple_returns)


# helpers
def save(fig, name: str):
    with PdfPages(OUT / f"{name}.pdf") as pdf:
        pdf.savefig(fig)
    plt.close(fig)


def panel_label(ax, text: str):
    ax.text(
        0.02,
        0.97,
        text,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2),
    )


def drawdown_series(s: pd.Series) -> pd.Series:
    v = s.dropna()
    return v / v.cummax() - 1.0


def model_of(key: str) -> str:
    return key.split("|")[0]


def signal_of(key: str) -> str:
    return key.split("|")[1]


def rebal_of(key: str) -> str:
    return key.split("|")[2]


# 1. normalized etf prices  (2 files: equities | bonds+alts)

prices_norm = (1 + simple_returns).cumprod()
prices_norm = prices_norm / prices_norm.iloc[0]

ASSET_GROUPS = {
    "Akcje USA": {
        "tickers": ["SPY", "IWM", "XLP"],
        "colors": ["#1f77b4", "#aec7e8", "#6baed6"],
    },
    "Akcje globalne": {"tickers": ["EFA", "EEM"], "colors": ["#2ca02c", "#98df8a"]},
    "Obligacje": {
        "tickers": ["TLT", "IEF", "LQD"],
        "colors": ["#d62728", "#ff9896", "#e6550d"],
    },
    "Alternatywne": {"tickers": ["VNQ", "GLD"], "colors": ["#9467bd", "#c5b0d5"]},
}

for fname, groups in [
    ("asset_prices_equities", ["Akcje USA", "Akcje globalne"]),
    ("asset_prices_bonds_alt", ["Obligacje", "Alternatywne"]),
]:
    fig, axes = plt.subplots(2, 1, figsize=SIZE_2P, sharex=True)
    for ax, gname in zip(axes, groups):
        g = ASSET_GROUPS[gname]
        for ticker, color in zip(g["tickers"], g["colors"]):
            ax.plot(
                prices_norm.index,
                prices_norm[ticker],
                label=ticker,
                color=color,
                lw=1.3,
            )
        ax.axhline(1.0, color="gray", ls="--", lw=0.7, alpha=0.5)
        ax.set_ylabel("Wartość znormalizowana (start = 1,0)", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}×"))
        ax.legend(
            loc="upper left", fontsize=8, framealpha=0.65, title=gname, title_fontsize=8
        )
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Data")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout(pad=0.8)
    save(fig, fname)

# 2. correlation matrix  (square size, colorbar inside)

corr = simple_returns.corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

fig, ax = plt.subplots(figsize=SIZE_SQ)
sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    ax=ax,
    cbar_kws={"shrink": 0.6, "aspect": 20},
    annot_kws={"size": 8.5},
)

for label in ax.get_xticklabels():
    t = label.get_text()
    label.set_color(CLASS_COLORS.get(ASSET_CLASSES.get(t, ""), "black"))
    label.set_fontweight("bold")
for label in ax.get_yticklabels():
    t = label.get_text()
    label.set_color(CLASS_COLORS.get(ASSET_CLASSES.get(t, ""), "black"))
    label.set_fontweight("bold")

leg_els = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor=c, markersize=10, label=cls)
    for cls, c in CLASS_COLORS.items()
]
ax.legend(handles=leg_els, loc="upper right", fontsize=8, framealpha=0.8)
fig.tight_layout(pad=0.8)
save(fig, "asset_correlation")

# 3. asset risk-return profile

cagr_assets = ((1 + simple_returns).prod() ** (12 / MONTHS) - 1) * 100
vol_assets = simple_returns.std(ddof=1) * np.sqrt(12) * 100

fig, ax = plt.subplots(figsize=SIZE_1P)
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
        lw=0.8,
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
ax.plot(x_line, 0.5 * x_line, color="gray", ls="--", lw=0.9, alpha=0.6)
ax.plot(x_line, 1.0 * x_line, color="gray", ls=":", lw=0.9, alpha=0.6)
ax.axhline(0, color="black", lw=0.7, alpha=0.4)
ax.set_xlabel("Roczna zmienność (%)")
ax.set_ylabel("CAGR (%)")
ax.xaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=1))
ax.grid(True)

leg_class = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=cls)
    for cls, c in CLASS_COLORS.items()
]
leg_lines = [
    Line2D([0], [0], color="gray", ls="--", label="Sharpe = 0,5"),
    Line2D([0], [0], color="gray", ls=":", label="Sharpe = 1,0"),
]
ax.legend(handles=leg_class + leg_lines, fontsize=8, framealpha=0.8)
fig.tight_layout(pad=0.8)
save(fig, "asset_risk_return")

# 4. portfolio value paths  (7 files, 2 panels: cal + thr)

CURVE_GROUPS = [
    ("markowitz", "historical_mean", "curves_mkw_hm"),
    ("markowitz", "capm", "curves_mkw_capm"),
    ("markowitz", "ff3f", "curves_mkw_ff3f"),
    ("mean_cvar", "historical_mean", "curves_mcvar_hm"),
    ("mean_cvar", "capm", "curves_mcvar_capm"),
    ("mean_cvar", "ff3f", "curves_mcvar_ff3f"),
    ("risk_parity", "none", "curves_rp"),
]
MODEL_NAMES = {
    "markowitz": "Markowitz",
    "mean_cvar": "Mean-CVaR",
    "risk_parity": "Parytet ryzyka",
}

BENCH_LEGEND = [
    Line2D([0], [0], color="#222222", lw=1.5, ls="--", label="60/40 (benchmark)"),
    Line2D([0], [0], color="#888888", lw=1.5, ls=":", label="EW (benchmark)"),
]

for model, signal, fname in CURVE_GROUPS:
    cal_key = f"{model}|{signal}|calendar"
    thr_key = f"{model}|{signal}|threshold"
    color = MODEL_COLORS[model]

    fig, axes = plt.subplots(2, 1, figsize=SIZE_2P, sharex=True)
    for ax, col in zip(axes, [cal_key, thr_key]):
        if col not in net_values.columns:
            ax.set_visible(False)
            continue
        for bench in BENCHMARKS:
            clr, ls, lw = BENCH_STYLES[bench]
            ax.plot(
                net_values.index,
                net_values[bench],
                color=clr,
                ls=ls,
                lw=lw,
                alpha=0.7,
                zorder=1,
            )
        ax.plot(net_values.index, net_values[col], color=color, lw=2.0, zorder=2)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}×"))
        ax.set_ylabel("Wartość portfela (skala log)")
        ax.grid(True, which="both", alpha=0.25)
        panel_label(ax, LABEL_MAP.get(col, col))

    axes[-1].set_xlabel("Data")
    fig.autofmt_xdate(rotation=30)

    leg_handles = [
        Line2D([0], [0], color=color, lw=2.0, label=MODEL_NAMES[model])
    ] + BENCH_LEGEND
    axes[0].legend(handles=leg_handles, fontsize=8, framealpha=0.8, loc="lower right")
    fig.tight_layout(pad=0.8)
    save(fig, fname)

# 5. drawdown  — 3 panels (by model group), no fill

DD_GROUPS = [
    ("markowitz", ["historical_mean", "capm", "ff3f"]),
    ("mean_cvar", ["historical_mean", "capm", "ff3f"]),
    ("risk_parity", ["none"]),
]
SIGNAL_LABELS = {
    "historical_mean": "Śr. hist.",
    "capm": "CAPM",
    "ff3f": "FF3F",
    "none": "",
}
MODEL_LABELS = {
    "markowitz": "Markowitz",
    "mean_cvar": "Mean-CVaR",
    "risk_parity": "Parytet ryzyka",
}

fig, axes = plt.subplots(3, 1, figsize=SIZE_3P, sharex=True, sharey=True)

for ax, (model, signals) in zip(axes, DD_GROUPS):
    color = MODEL_COLORS[model]

    for signal in signals:
        ls_style, lw = SIGNAL_LS[signal]
        for rebal in ["calendar", "threshold"]:
            col = f"{model}|{signal}|{rebal}"
            if col not in net_values.columns:
                continue
            dd = drawdown_series(net_values[col])
            sig_lbl = SIGNAL_LABELS[signal]
            reb_lbl = "KAL" if rebal == "calendar" else "PROG"
            lbl = f"{sig_lbl} {reb_lbl}".strip() if sig_lbl else reb_lbl
            ax.plot(
                dd.index,
                dd.values,
                color=color,
                ls=ls_style,
                lw=lw,
                alpha=REBAL_ALPHA[rebal],
                label=lbl,
            )

    for bench in BENCHMARKS:
        clr, ls, lw = BENCH_STYLES[bench]
        dd = drawdown_series(net_values[bench])
        ax.plot(
            dd.index,
            dd.values,
            color=clr,
            ls=ls,
            lw=lw,
            zorder=5,
            label=LABEL_MAP.get(bench, bench),
        )

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("Obsunięcie (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower left", framealpha=0.8, ncol=2)
    panel_label(ax, MODEL_LABELS[model])

axes[-1].set_xlabel("Data")
fig.autofmt_xdate(rotation=30)
fig.tight_layout(pad=0.8)
save(fig, "drawdown")

# 6. cagr gross vs net  — colors by model, hatching for net

CAGR_ORDER = [
    "risk_parity|none|calendar",
    "risk_parity|none|threshold",
    "markowitz|historical_mean|calendar",
    "markowitz|historical_mean|threshold",
    "markowitz|capm|calendar",
    "markowitz|capm|threshold",
    "markowitz|ff3f|calendar",
    "markowitz|ff3f|threshold",
    "mean_cvar|historical_mean|calendar",
    "mean_cvar|historical_mean|threshold",
    "mean_cvar|capm|calendar",
    "mean_cvar|capm|threshold",
    "mean_cvar|ff3f|calendar",
    "mean_cvar|ff3f|threshold",
    "60_40|none|calendar",
    "equal_weight|none|calendar",
]

x_pos = np.arange(len(CAGR_ORDER))
w = 0.38
xlabels = [LABEL_MAP.get(v, v) for v in CAGR_ORDER]


def bar_color(key):
    m = model_of(key)
    return MODEL_COLORS.get(m, MODEL_COLORS["benchmark"])


fig, ax = plt.subplots(figsize=SIZE_1P)

for i, key in enumerate(CAGR_ORDER):
    if key not in metrics_gross.index:
        continue
    c = bar_color(key)
    brutto = metrics_gross.loc[key, "cagr"] * 100
    netto = metrics_net.loc[key, "cagr"] * 100
    ax.bar(i - w / 2, brutto, w, color=c, alpha=0.85, edgecolor="white", lw=0.5)
    ax.bar(
        i + w / 2,
        netto,
        w,
        color=c,
        alpha=0.85,
        edgecolor="white",
        hatch="///",
        linewidth=0,
    )

ax.set_xticks(x_pos)
ax.set_xticklabels(xlabels, rotation=40, ha="right", fontsize=7.5)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=1))
ax.set_ylabel("CAGR (%)")
ax.grid(True, axis="y")

# Separators between model groups
for sep in [1.5, 7.5, 13.5]:
    ax.axvline(sep, color="gray", lw=0.8, ls="--", alpha=0.4)

leg_model = [
    mpatches.Patch(facecolor=MODEL_COLORS["risk_parity"], label="Parytet ryzyka"),
    mpatches.Patch(facecolor=MODEL_COLORS["markowitz"], label="Markowitz"),
    mpatches.Patch(facecolor=MODEL_COLORS["mean_cvar"], label="Mean-CVaR"),
    mpatches.Patch(facecolor=MODEL_COLORS["benchmark"], label="Benchmarki"),
]
leg_style = [
    mpatches.Patch(facecolor="gray", alpha=0.85, label="Brutto (pełny)"),
    mpatches.Patch(
        facecolor="gray", alpha=0.85, hatch="///", label="Netto (kreskowanie)"
    ),
]
ax.legend(
    handles=leg_model + leg_style,
    fontsize=7.5,
    framealpha=0.85,
    loc="upper left",
    ncol=2,
)
fig.tight_layout(pad=0.8)
save(fig, "cagr_gross_vs_net")

# 7. monthly returns boxplot  — colors by model

BOX_ORDER = CAGR_ORDER
ret_data = []
xlabels_b = []
box_colors = []
for key in BOX_ORDER:
    if key not in net_returns.columns:
        continue
    ret_data.append(net_returns[key].dropna().values * 100)
    xlabels_b.append(LABEL_MAP.get(key, key))
    box_colors.append(bar_color(key))

fig, ax = plt.subplots(figsize=SIZE_1P)
bp = ax.boxplot(
    ret_data,
    patch_artist=True,
    medianprops={"color": "black", "linewidth": 1.5},
    flierprops={"marker": "o", "markersize": 2.5, "alpha": 0.4},
    whiskerprops={"linewidth": 0.9},
    capprops={"linewidth": 0.9},
)
for patch, color in zip(bp["boxes"], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
for whisker, color in zip(bp["whiskers"], [c for c in box_colors for _ in range(2)]):
    whisker.set_color(color)
for cap, color in zip(bp["caps"], [c for c in box_colors for _ in range(2)]):
    cap.set_color(color)
for flier, color in zip(bp["fliers"], box_colors):
    flier.set_markerfacecolor(color)
    flier.set_markeredgecolor(color)

ax.set_xticks(range(1, len(xlabels_b) + 1))
ax.set_xticklabels(xlabels_b, rotation=40, ha="right", fontsize=7.5)
ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=1))
ax.set_ylabel("Miesięczny zwrot (%)")
ax.grid(True, axis="y")

for sep in [2.5, 8.5, 14.5]:
    ax.axvline(sep, color="gray", lw=0.8, ls="--", alpha=0.4)

leg_model_b = [
    mpatches.Patch(
        facecolor=MODEL_COLORS["risk_parity"], alpha=0.75, label="Parytet ryzyka"
    ),
    mpatches.Patch(facecolor=MODEL_COLORS["markowitz"], alpha=0.75, label="Markowitz"),
    mpatches.Patch(facecolor=MODEL_COLORS["mean_cvar"], alpha=0.75, label="Mean-CVaR"),
    mpatches.Patch(facecolor=MODEL_COLORS["benchmark"], alpha=0.75, label="Benchmarki"),
]
ax.legend(handles=leg_model_b, fontsize=8, framealpha=0.85, loc="lower right")
fig.tight_layout(pad=0.8)
save(fig, "returns_boxplot")

# 8. metrics heatmaps

SPLIT_L = VARIANTS_ORDER[:8]  # MKT (6) + MCVAR HM (2)
SPLIT_R = VARIANTS_ORDER[8:]  # MCVAR CAPM–FF3F (4) + RP (2) + benchmarks (2)

HEAT_METRICS = [
    ("sharpe", "heatmap_sharpe"),
    ("sortino", "heatmap_sortino"),
    ("calmar", "heatmap_calmar"),
]

mg_h = metrics_gross[["sharpe", "sortino", "calmar"]].copy()
mn_h = metrics_net[["sharpe", "sortino", "calmar"]].copy()
mg_h.index = [LABEL_MAP.get(i, i) for i in mg_h.index]
mn_h.index = [LABEL_MAP.get(i, i) for i in mn_h.index]
mg_h = mg_h.reindex(VARIANTS_ORDER)
mn_h = mn_h.reindex(VARIANTS_ORDER)


def draw_half_heatmap(ax, rows, col, norm, show_cbar=False, fig=None):
    panel = pd.DataFrame(
        {"Brutto": mg_h.loc[rows, col], "Netto": mn_h.loc[rows, col]}
    ).astype(float)
    ax.imshow(panel.values, cmap="RdYlGn", norm=norm, aspect="auto")

    for i in range(panel.shape[0]):
        for j in range(panel.shape[1]):
            val = panel.iloc[i, j]
            txt = "—" if np.isnan(val) else f"{val:.2f}"
            bg = plt.cm.RdYlGn(norm(val) if not np.isnan(val) else 0.5)
            lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="black" if lum > 0.45 else "white",
            )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Brutto", "Netto"], fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8.5)

    for y in np.arange(-0.5, len(rows), 1):
        ax.axhline(y, color="white", lw=0.8)
    ax.axvline(0.5, color="white", lw=0.8)


for col, fname in HEAT_METRICS:
    all_vals = pd.concat([mg_h[col], mn_h[col]]).dropna()
    norm = Normalize(vmin=all_vals.min(), vmax=all_vals.max())

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=SIZE_HM, gridspec_kw={"wspace": 0.65}
    )

    draw_half_heatmap(ax_l, SPLIT_L, col, norm)
    draw_half_heatmap(ax_r, SPLIT_R, col, norm)

    # Separator: active strategies | benchmarks (last 2 rows of right column)
    ax_r.axhline(5.5, color="black", lw=1.2, ls="--", alpha=0.5)

    # Shared colorbar
    sm = ScalarMappable(norm=norm, cmap="RdYlGn")
    cb = fig.colorbar(sm, ax=[ax_l, ax_r], shrink=0.8, aspect=25, pad=0.02)
    cb.ax.tick_params(labelsize=8)

    fig.tight_layout(pad=0.8)
    save(fig, fname)

# 9. transaction costs vs drag

fig, ax = plt.subplots(figsize=SIZE_1P)
for i, (variant, row) in enumerate(drag_df.iterrows()):
    x_val = row["tc_total"] * 100
    y_val = row["total_drag"] * 100
    lbl = LABEL_MAP.get(variant, variant)
    color = MODEL_COLORS.get(model_of(variant), MODEL_COLORS["benchmark"])
    ax.scatter(x_val, y_val, color=color, s=70, zorder=3)
    ax.annotate(
        lbl,
        (x_val, y_val),
        textcoords="offset points",
        xytext=(5, 3),
        fontsize=7,
        color=color,
    )
max_x = drag_df["tc_total"].max() * 100 * 1.1
ax.plot(
    [0, max_x],
    [0, max_x],
    color="gray",
    ls="--",
    lw=0.9,
    label="Drag = TC (bez podatku Belki)",
)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(decimals=2))
ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=1))
ax.set_xlabel("Łączne koszty transakcyjne (%)")
ax.set_ylabel("Całkowity drag (%)")
ax.legend(fontsize=8)
ax.grid(True)

leg_c = [
    mpatches.Patch(facecolor=MODEL_COLORS[m], label=MODEL_NAMES[m])
    for m in ["risk_parity", "markowitz", "mean_cvar"]
]
leg_c.append(mpatches.Patch(facecolor=MODEL_COLORS["benchmark"], label="Benchmarki"))
ax.legend(
    handles=[Line2D([0], [0], color="gray", ls="--", label="Drag = TC")] + leg_c,
    fontsize=8,
    framealpha=0.85,
)
fig.tight_layout(pad=0.8)
save(fig, "cost_vs_drag")

# 10. portfolio weights evolution  (7 files, 2 panels: cal + thr)

WEIGHTS_GROUPS = [
    ("markowitz", "historical_mean", "weights_mkw_hm"),
    ("markowitz", "capm", "weights_mkw_capm"),
    ("markowitz", "ff3f", "weights_mkw_ff3f"),
    ("mean_cvar", "historical_mean", "weights_mcvar_hm"),
    ("mean_cvar", "capm", "weights_mcvar_capm"),
    ("mean_cvar", "ff3f", "weights_mcvar_ff3f"),
    ("risk_parity", "none", "weights_rp"),
]

TICKER_LEGEND = [
    Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        markerfacecolor=TICKER_COLORS[i],
        markersize=10,
        label=TICKERS[i],
    )
    for i in range(len(TICKERS))
]

for model, signal, fname in WEIGHTS_GROUPS:
    try:
        w = weights_df.xs((model, signal), level=("model", "signal"))[
            TICKERS
        ].sort_index()
    except KeyError:
        continue

    fig, axes = plt.subplots(2, 1, figsize=SIZE_2P, sharex=True)
    for ax, strategy in zip(axes, ["calendar", "threshold"]):
        col_key = f"{model}|{signal}|{strategy}"
        ax.stackplot(
            w.index,
            [w[t].values for t in TICKERS],
            labels=TICKERS,
            colors=TICKER_COLORS,
            alpha=0.85,
        )
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_ylabel("Udział w portfelu (%)")
        panel_label(ax, LABEL_MAP.get(col_key, col_key))

    axes[-1].set_xlabel("Data")
    fig.autofmt_xdate(rotation=30)
    axes[0].legend(
        handles=TICKER_LEGEND, fontsize=7.5, loc="upper right", ncol=5, framealpha=0.8
    )
    fig.tight_layout(pad=0.8)
    save(fig, fname)

# 11. hhi — averages (barplot)

N = len(TICKERS)
HHI_EW = 1.0 / N
HHI_60_40 = 0.6**2 + 0.4**2

hhi_series = metrics_gross["hhi"].copy()
hhi_series.index = [LABEL_MAP.get(i, i) for i in hhi_series.index]
hhi_ordered = hhi_series.reindex(VARIANTS_ORDER)

bar_colors_hhi = []
for lbl in VARIANTS_ORDER:
    if "MKT" in lbl:
        bar_colors_hhi.append(MODEL_COLORS["markowitz"])
    elif "MCVAR" in lbl:
        bar_colors_hhi.append(MODEL_COLORS["mean_cvar"])
    elif "RP" in lbl:
        bar_colors_hhi.append(MODEL_COLORS["risk_parity"])
    else:
        bar_colors_hhi.append(MODEL_COLORS["benchmark"])

fig, ax = plt.subplots(figsize=SIZE_1P)
bars = ax.bar(
    range(len(VARIANTS_ORDER)),
    hhi_ordered.values,
    color=bar_colors_hhi,
    alpha=0.85,
    edgecolor="white",
    lw=0.6,
)
ax.axhline(
    HHI_EW,
    color="#2ca02c",
    ls="--",
    lw=1.2,
    alpha=0.8,
    label=f"EW (HHI = {HHI_EW:.3f})",
)
ax.axhline(
    HHI_60_40,
    color="#222222",
    ls=":",
    lw=1.2,
    alpha=0.8,
    label=f"60/40 (HHI = {HHI_60_40:.3f})",
)
ax.axvline(13.5, color="gray", lw=0.8, alpha=0.5)

for bar, val in zip(bars, hhi_ordered.values):
    if not np.isnan(val):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.003,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
        )

ax.set_xticks(range(len(VARIANTS_ORDER)))
ax.set_xticklabels(VARIANTS_ORDER, rotation=40, ha="right", fontsize=7.5)
ax.set_ylabel("HHI (średnia w czasie)")
ax.set_ylim(0, max(hhi_ordered.dropna()) * 1.18)
ax.grid(True, axis="y")

leg_hhi = [
    mpatches.Patch(facecolor=MODEL_COLORS["markowitz"], alpha=0.85, label="Markowitz"),
    mpatches.Patch(facecolor=MODEL_COLORS["mean_cvar"], alpha=0.85, label="Mean-CVaR"),
    mpatches.Patch(
        facecolor=MODEL_COLORS["risk_parity"], alpha=0.85, label="Parytet ryzyka"
    ),
    mpatches.Patch(facecolor=MODEL_COLORS["benchmark"], alpha=0.85, label="Benchmarki"),
    Line2D([0], [0], color="#2ca02c", ls="--", lw=1.5, label=f"Ref. EW ({HHI_EW:.3f})"),
    Line2D(
        [0], [0], color="#222222", ls=":", lw=1.5, label=f"Ref. 60/40 ({HHI_60_40:.3f})"
    ),
]
ax.legend(handles=leg_hhi, fontsize=8, framealpha=0.8, loc="upper left", ncol=2)
fig.tight_layout(pad=0.8)
save(fig, "hhi_barplot")

# 12. hhi — time evolution


def compute_hhi(model, signal):
    try:
        w = weights_df.xs((model, signal), level=("model", "signal"))[TICKERS]
        return (w**2).sum(axis=1)
    except KeyError:
        return pd.Series(dtype=float)


HHI_GROUPS = [
    (
        "markowitz",
        [("historical_mean", "MKT HM"), ("capm", "MKT CAPM"), ("ff3f", "MKT FF3F")],
        "hhi_evol_mkw",
    ),
    (
        "mean_cvar",
        [
            ("historical_mean", "MCVAR HM"),
            ("capm", "MCVAR CAPM"),
            ("ff3f", "MCVAR FF3F"),
        ],
        "hhi_evol_mcvar",
    ),
    ("risk_parity", [("none", "Parytet ryzyka")], "hhi_evol_rp"),
]

for model, signals, fname in HHI_GROUPS:
    nrows = len(signals)
    size = SIZE_3P if nrows == 3 else SIZE_1P
    fig, axes = plt.subplots(nrows, 1, figsize=size, sharex=True, sharey=True)
    axes_list = np.atleast_1d(axes)
    color = MODEL_COLORS[model]

    for ax, (signal, label) in zip(axes_list, signals):
        hhi_ts = compute_hhi(model, signal)
        if hhi_ts.empty:
            ax.set_visible(False)
            continue
        hhi_smooth = hhi_ts.rolling(3, min_periods=1).mean()
        ax.plot(hhi_ts.index, hhi_ts.values, color=color, lw=0.8, alpha=0.35)
        ax.fill_between(hhi_smooth.index, hhi_smooth.values, alpha=0.15, color=color)
        ax.plot(hhi_smooth.index, hhi_smooth.values, color=color, lw=1.8)
        ax.axhline(HHI_EW, color="#2ca02c", ls="--", lw=1.0, alpha=0.7)
        ax.axhline(HHI_60_40, color="#222222", ls=":", lw=1.0, alpha=0.6)
        ax.set_ylim(0, 0.85)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
        ax.set_ylabel("HHI")
        ax.grid(True, alpha=0.25)
        panel_label(ax, label)

    axes_list[-1].set_xlabel("Data")
    fig.autofmt_xdate(rotation=30)

    leg_hhi_ev = [
        Line2D([0], [0], color=color, lw=0.8, alpha=0.35, label="HHI (miesięcznie)"),
        Line2D([0], [0], color=color, lw=2.0, label="HHI (krocząca 3M)"),
        Line2D(
            [0], [0], color="#2ca02c", ls="--", lw=1.5, label=f"Ref. EW ({HHI_EW:.3f})"
        ),
        Line2D(
            [0],
            [0],
            color="#222222",
            ls=":",
            lw=1.5,
            label=f"Ref. 60/40 ({HHI_60_40:.3f})",
        ),
    ]
    axes_list[0].legend(
        handles=leg_hhi_ev, fontsize=8, framealpha=0.8, loc="upper right"
    )
    fig.tight_layout(pad=0.8)
    save(fig, fname)

print(f"All charts saved to: {OUT.resolve()}")
