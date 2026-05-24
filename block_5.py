# Block 5 — backtest with transaction costs and annual Belka tax

import numpy as np
import pandas as pd
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

TICKERS = ["SPY", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "VNQ", "GLD", "XLP"]
N = len(TICKERS)
THRESHOLD = 0.05
INITIAL_VALUE = 1.0
TRANSACTION_COST = 0.001  # 0.10% per side
TAX_RATE = 0.19  # Belka
OUTPUT_DIR = Path("data/block_5")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
        w[:] = 1.0 / N
    else:
        raise ValueError(f"Unknown benchmark: {name}")
    return w


class TaxLedger:
    def __init__(self, tickers, loss_carryforward_years=5):
        self.tickers = tickers
        self.loss_carryforward_years = loss_carryforward_years
        self.cost_basis = pd.Series(0.0, index=tickers, dtype=float)
        self.annual_realised_net = 0.0
        self.tax_log = {}
        self._loss_pool: list[tuple[int, float]] = []

    def initialise(self, weights, portfolio_value):
        self.cost_basis = weights * portfolio_value

    def rebalance_to_target(self, asset_values_before, target_weights, portfolio_value_before_cost):
        new_target_values = target_weights * portfolio_value_before_cost

        for ticker in self.tickers:
            old_val = float(asset_values_before[ticker])
            new_val = float(new_target_values[ticker])
            basis = float(self.cost_basis[ticker])

            if old_val > 1e-12 and new_val < old_val:
                sold_value = old_val - new_val
                sell_fraction = sold_value / old_val
                realised_pl = sell_fraction * (old_val - basis)
                self.annual_realised_net += realised_pl
                self.cost_basis[ticker] = basis * (1 - sell_fraction)
            elif new_val > old_val:
                buy_value = new_val - old_val
                self.cost_basis[ticker] = basis + buy_value

    def settle_year_end_tax(self, year: int) -> float:
        gross_income = self.annual_realised_net
        self.annual_realised_net = 0.0

        if gross_income >= 0:
            # consume only tranches due this year; keep the rest for later
            deductible = 0.0
            surviving = []
            for target_year, tranche in self._loss_pool:
                if target_year == year:
                    deductible += tranche
                elif target_year > year:
                    surviving.append((target_year, tranche))
            self._loss_pool = surviving

            taxable = max(0.0, gross_income - deductible)
            tax = taxable * TAX_RATE
        else:
            loss = abs(gross_income)
            tranche = loss / self.loss_carryforward_years
            for k in range(1, self.loss_carryforward_years + 1):
                self._loss_pool.append((year + k, tranche))
            tax = 0.0

        self.tax_log[year] = tax
        return tax


def simulate_portfolio_net(target_weights_series, returns, strategy="calendar", threshold=THRESHOLD):
    monthly_returns = {}
    portfolio_values = {}
    cost_log = {}

    sim_dates = sorted(set(target_weights_series.keys()) & set(returns.index))
    if not sim_dates:
        return (
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            TaxLedger(TICKERS),
        )

    first_date = sim_dates[0]
    current_weights = target_weights_series[first_date].reindex(TICKERS).fillna(0.0)
    current_weights = current_weights / current_weights.sum()
    portfolio_value = INITIAL_VALUE

    ledger = TaxLedger(TICKERS)
    ledger.initialise(current_weights, portfolio_value)

    for i, date in enumerate(sim_dates):
        start_value = portfolio_value

        target_w = target_weights_series[date].reindex(TICKERS).fillna(0.0)
        target_w = target_w / target_w.sum()

        asset_values_before = current_weights * portfolio_value

        if strategy == "calendar":
            do_rebalance = True
        elif strategy == "threshold":
            drift = (current_weights - target_w).abs().max()
            do_rebalance = drift > threshold
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        tc = 0.0
        if do_rebalance:
            turnover = (target_w - current_weights).abs().sum() / 2.0
            tc = 2.0 * TRANSACTION_COST * turnover * portfolio_value

            ledger.rebalance_to_target(
                asset_values_before=asset_values_before,
                target_weights=target_w,
                portfolio_value_before_cost=portfolio_value,
            )

            portfolio_value -= tc
            current_weights = target_w.copy()

        month_returns = returns.loc[date].reindex(TICKERS).fillna(0.0)
        asset_values_after_return = current_weights * portfolio_value * (1.0 + month_returns)
        portfolio_value = asset_values_after_return.sum()
        current_weights = asset_values_after_return / portfolio_value

        is_last_observation = i == len(sim_dates) - 1
        next_year_differs = not is_last_observation and sim_dates[i + 1].year != date.year

        if next_year_differs or is_last_observation:
            tax_paid = ledger.settle_year_end_tax(date.year)
            portfolio_value -= tax_paid

            if portfolio_value > 0:
                asset_values_after_tax = current_weights * portfolio_value
                current_weights = asset_values_after_tax / asset_values_after_tax.sum()

        monthly_returns[date] = (portfolio_value / start_value) - 1.0
        portfolio_values[date] = portfolio_value
        cost_log[date] = tc

    return (
        pd.Series(monthly_returns),
        pd.Series(portfolio_values),
        pd.Series(cost_log),
        ledger,
    )


all_returns = {}
all_values = {}
all_costs = {}
all_tax_logs = {}

for model, signal in combinations:
    subset = weights_df.xs((model, signal), level=("model", "signal"))
    target_weights_series = {date: subset.loc[date] for date in subset.index}

    for strategy in ["calendar", "threshold"]:
        key = (model, signal, strategy)
        rets, vals, costs, ledger = simulate_portfolio_net(
            target_weights_series=target_weights_series,
            returns=simple_returns,
            strategy=strategy,
        )
        all_returns[key] = rets
        all_values[key] = vals
        all_costs[key] = costs
        all_tax_logs[key] = ledger.tax_log

for bench_name in ["60_40", "equal_weight"]:
    w = get_benchmark_weights(bench_name)
    first_date = simple_returns.index[
        simple_returns.index >= weights_df.index.get_level_values("date").min()
        ][0]
    bench_dates = simple_returns.index[simple_returns.index >= first_date]
    target_weights_series = {date: w for date in bench_dates}

    key = (bench_name, "none", "calendar")
    rets, vals, costs, ledger = simulate_portfolio_net(
        target_weights_series=target_weights_series,
        returns=simple_returns,
        strategy="calendar",
    )
    all_returns[key] = rets
    all_values[key] = vals
    all_costs[key] = costs
    all_tax_logs[key] = ledger.tax_log


def flatten_df(d):
    df = pd.DataFrame(d)
    df.index.name = "date"
    df.columns = [f"{m}|{s}|{r}" for m, s, r in df.columns]
    return df


returns_flat = flatten_df(all_returns)
values_flat = flatten_df(all_values)
costs_flat = flatten_df(all_costs)

returns_flat.to_csv(OUTPUT_DIR / "portfolio_returns_net.csv")
values_flat.to_csv(OUTPUT_DIR / "portfolio_values_net.csv")
costs_flat.to_csv(OUTPUT_DIR / "transaction_costs_log.csv")

tax_rows = []
for (model, signal, strategy), tax_dict in all_tax_logs.items():
    for year, tax in tax_dict.items():
        tax_rows.append(
            {"model": model, "signal": signal, "strategy": strategy, "year": year, "tax": tax}
        )

tax_log_df = pd.DataFrame(tax_rows)
tax_log_df.to_csv(OUTPUT_DIR / "tax_log.csv", index=False)

print(
    "Saved net backtest outputs: "
    f"returns {returns_flat.shape}, values {values_flat.shape}, "
    f"costs {costs_flat.shape}, tax {tax_log_df.shape}"
)
