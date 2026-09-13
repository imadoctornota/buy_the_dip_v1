"""Does Buying the Dip Work for Individual Stocks?

Code accompanying the YouTube analysis of drawdown threshold crossings and
simple portfolio rules for AAPL, MSFT, TSLA, NVDA, AMZN, META, and NFLX.

This is an exploratory historical analysis, not investment advice.
"""



# ==============================================================================
# 2. Imports and settings


# ==============================================================================
# What this step is doing — and why


from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 140)

DATA_DIR = Path("data")
FIGURE_DIR = Path("figures")

DATA_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

# Core experiment settings
FOCUS_TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "NFLX"]

START_DATE = "2000-01-01"
END_DATE = "2026-09-12"  # yfinance end is exclusive; includes data through 2026-09-11

# ~1 trading year
LOOKBACK = 252

# Allow a drawdown estimate once ~6 months of history exists.
MIN_HISTORY = 126

THRESHOLDS = [-0.10, -0.20, -0.30, -0.50]

HORIZONS = {
    "1m": 21,
    "3m": 63,
    "1y": 252,
}

print("Focus tickers:", FOCUS_TICKERS)


# ==============================================================================
# 3. Download adjusted price history


# ==============================================================================
# What this step is doing — and why


def download_prices(tickers, start="2000-01-01", end=None):
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        actions=False,
        threads=True,
        progress=False,
        group_by="column",
    )

    # yfinance returns different shapes for one ticker vs multiple tickers.
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy()
        close.columns = [tickers[0]]

    close = close.sort_index().dropna(how="all")
    return close


focus_path = DATA_DIR / "focus_stock_prices.parquet"

# Set to True to redownload the same frozen date range.
REFRESH_FOCUS_DATA = False

if focus_path.exists() and not REFRESH_FOCUS_DATA:
    prices = pd.read_parquet(focus_path)
    print(f"Loaded cached data from {focus_path}")
else:
    prices = download_prices(FOCUS_TICKERS, start=START_DATE, end=END_DATE)
    prices.to_parquet(focus_path)
    print(f"Downloaded and saved data to {focus_path}")

print(prices.tail())
print()
print("Non-missing daily observations:")
print(prices.count().sort_values(ascending=False).to_frame("n_days"))


# ==============================================================================
# 4. Quick price-history sanity check


# ==============================================================================
# What this step is doing — and why


fig, ax = plt.subplots(figsize=(12, 6))

normalized = prices / prices.apply(lambda s: s.dropna().iloc[0])

for ticker in normalized.columns:
    ax.plot(normalized.index, normalized[ticker], label=ticker)

ax.set_title("Growth of $1 Since Each Stock's First Available Observation")
ax.set_ylabel("Growth of $1")
ax.set_xlabel("Date")
ax.legend()
ax.set_yscale("log")

plt.tight_layout()
plt.show()


# ==============================================================================
# 5. Calculate drawdown from the trailing 52-week high


# ==============================================================================
# What this step is doing — and why


rolling_high = prices.rolling(
    LOOKBACK,
    min_periods=MIN_HISTORY,
).max()

drawdown = prices / rolling_high - 1

print(drawdown.tail())


# ==============================================================================
# 6. Visualize drawdowns


ticker = "AAPL"

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(drawdown.index, drawdown[ticker] * 100)

for threshold in THRESHOLDS:
    ax.axhline(threshold * 100, linestyle="--", linewidth=1)

ax.set_title(f"{ticker}: Drawdown from Trailing 52-Week High")
ax.set_ylabel("Drawdown (%)")
ax.set_xlabel("Date")

plt.tight_layout()
plt.show()


# ==============================================================================
# 7. Detect dip events


# ==============================================================================
# What this step is doing — and why


def detect_dip_events(drawdown_df, thresholds=THRESHOLDS):
    events = []

    for ticker in drawdown_df.columns:
        dd = drawdown_df[ticker].dropna()

        for threshold in thresholds:
            crossed = (dd <= threshold) & (dd.shift(1) > threshold)
            event_dates = dd.index[crossed.fillna(False)]

            for date in event_dates:
                events.append({
                    "ticker": ticker,
                    "date": date,
                    "threshold": threshold,
                    "actual_drawdown": dd.loc[date],
                })

    events = pd.DataFrame(events)

    if not events.empty:
        events = events.sort_values(
            ["ticker", "date", "threshold"]
        ).reset_index(drop=True)

    return events


events = detect_dip_events(drawdown)

print(events.head(20))

print("Event counts:")
print(
    events.groupby(["ticker", "threshold"])
    .size()
    .unstack(fill_value=0)
)


# ==============================================================================
# 8. Add forward returns after each dip


# ==============================================================================
# What this step is doing — and why


def add_forward_returns(events_df, prices_df, horizons=HORIZONS):
    output = []

    for _, event in events_df.iterrows():
        ticker = event["ticker"]
        date = event["date"]

        s = prices_df[ticker].dropna()

        if date not in s.index:
            continue

        i = s.index.get_loc(date)
        entry_i = i + 1

        if entry_i >= len(s):
            continue

        entry_price = s.iloc[entry_i]

        row = event.to_dict()
        row["entry_date"] = s.index[entry_i]
        row["entry_price"] = entry_price

        for name, horizon in horizons.items():
            exit_i = entry_i + horizon

            if exit_i < len(s):
                exit_price = s.iloc[exit_i]
                row[f"return_{name}"] = exit_price / entry_price - 1
            else:
                row[f"return_{name}"] = np.nan

        output.append(row)

    return pd.DataFrame(output)


results = add_forward_returns(events, prices)

print(results.head(20))


# ==============================================================================
# 9. Summarize post-dip returns


# ==============================================================================
# What this step is doing — and why


def summarize_dip_results(results_df, horizon="1y"):
    col = f"return_{horizon}"

    summary = (
        results_df
        .groupby(["ticker", "threshold"])
        .agg(
            n=(col, "count"),
            mean_return=(col, "mean"),
            median_return=(col, "median"),
            win_rate=(col, lambda x: (x > 0).mean()),
        )
        .reset_index()
    )

    return summary


summary_1y = summarize_dip_results(results, horizon="1y")

summary_1y["threshold_pct"] = summary_1y["threshold"] * 100
summary_1y["mean_return_pct"] = summary_1y["mean_return"] * 100
summary_1y["median_return_pct"] = summary_1y["median_return"] * 100
summary_1y["win_rate_pct"] = summary_1y["win_rate"] * 100

print(
    summary_1y[
        [
            "ticker",
            "threshold_pct",
            "n",
            "mean_return_pct",
            "median_return_pct",
            "win_rate_pct",
        ]
    ].sort_values(["ticker", "threshold_pct"], ascending=[True, False])
)


# ==============================================================================
# 10. Calculate each stock's unconditional forward returns


# ==============================================================================
# What this step is doing — and why


def baseline_returns(prices_df, horizons=HORIZONS):
    rows = []

    for ticker in prices_df.columns:
        s = prices_df[ticker].dropna()

        for name, horizon in horizons.items():
            # Signal today, enter next close, exit after horizon trading days.
            entry = s.shift(-1)
            exit_price = s.shift(-(horizon + 1))

            returns = (exit_price / entry - 1).dropna()

            rows.append({
                "ticker": ticker,
                "horizon": name,
                "n": len(returns),
                "mean_return": returns.mean(),
                "median_return": returns.median(),
                "win_rate": (returns > 0).mean(),
            })

    return pd.DataFrame(rows)


baseline = baseline_returns(prices)

print(baseline)


# ==============================================================================
# 11. Compute the dip edge


# ==============================================================================
# What this step is doing — and why
# ==============================================================================
# What the current output says


baseline_1y = (
    baseline[baseline["horizon"] == "1y"]
    [["ticker", "mean_return", "median_return", "win_rate"]]
    .rename(columns={
        "mean_return": "baseline_mean_return",
        "median_return": "baseline_median_return",
        "win_rate": "baseline_win_rate",
    })
)

comparison_1y = summary_1y.merge(
    baseline_1y,
    on="ticker",
    how="left",
)

comparison_1y["dip_edge"] = (
    comparison_1y["mean_return"]
    - comparison_1y["baseline_mean_return"]
)

comparison_1y["median_dip_edge"] = (
    comparison_1y["median_return"]
    - comparison_1y["baseline_median_return"]
)

comparison_1y["win_rate_edge"] = (
    comparison_1y["win_rate"]
    - comparison_1y["baseline_win_rate"]
)

comparison_1y["dip_edge_pct"] = comparison_1y["dip_edge"] * 100
comparison_1y["median_dip_edge_pct"] = comparison_1y["median_dip_edge"] * 100
comparison_1y["win_rate_edge_pct"] = comparison_1y["win_rate_edge"] * 100

print(
    comparison_1y[
        [
            "ticker",
            "threshold_pct",
            "n",
            "mean_return_pct",
            "dip_edge_pct",
            "median_dip_edge_pct",
            "win_rate_edge_pct",
        ]
    ].sort_values(["ticker", "threshold_pct"], ascending=[True, False])
)


# ==============================================================================
# 12. Plot: 1-year dip edge by stock and threshold


plot_df = comparison_1y.pivot(
    index="ticker",
    columns="threshold_pct",
    values="dip_edge_pct",
).sort_index()

ax = plot_df.plot(
    kind="bar",
    figsize=(12, 6),
)

ax.axhline(0, linewidth=1)
ax.set_title("Extra 1-Year Return After a Dip vs. an Ordinary Day")
ax.set_ylabel("Dip Edge (percentage points)")
ax.set_xlabel("Stock")
ax.legend(title="Drawdown threshold (%)")

plt.tight_layout()

output_path = FIGURE_DIR / "dip_edge_by_stock.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"Saved: {output_path}")


# ==============================================================================
# 13. Plot: pooled results across focus stocks


# ==============================================================================
# What this step is doing — and why


pooled_1y = (
    results
    .groupby("threshold")
    .agg(
        n=("return_1y", "count"),
        mean_return=("return_1y", "mean"),
        median_return=("return_1y", "median"),
        win_rate=("return_1y", lambda x: (x > 0).mean()),
    )
    .reset_index()
)

pooled_1y["threshold_pct"] = pooled_1y["threshold"] * 100
pooled_1y["mean_return_pct"] = pooled_1y["mean_return"] * 100
pooled_1y["median_return_pct"] = pooled_1y["median_return"] * 100
pooled_1y["win_rate_pct"] = pooled_1y["win_rate"] * 100

print(pooled_1y)


fig, ax = plt.subplots(figsize=(9, 5))

x = np.arange(len(pooled_1y))

ax.bar(
    x - 0.18,
    pooled_1y["mean_return_pct"],
    width=0.36,
    label="Mean 1Y return",
)

ax.bar(
    x + 0.18,
    pooled_1y["median_return_pct"],
    width=0.36,
    label="Median 1Y return",
)

ax.set_xticks(x)
ax.set_xticklabels(
    [f"{int(abs(v))}% dip" for v in pooled_1y["threshold_pct"]]
)

ax.set_title("Forward 1-Year Return After Drawdown Threshold Crossings")
ax.set_ylabel("Return (%)")
ax.legend()

plt.tight_layout()

output_path = FIGURE_DIR / "pooled_forward_returns.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"Saved: {output_path}")


# ==============================================================================
# 14. Inspect the actual historical events


inspect_ticker = "TSLA"
inspect_threshold = -0.30

cols = [
    "ticker",
    "date",
    "threshold",
    "actual_drawdown",
    "entry_date",
    "return_1m",
    "return_3m",
    "return_1y",
]

print(
    results.loc[
        (results["ticker"] == inspect_ticker)
        & (results["threshold"] == inspect_threshold),
        cols,
    ]
    .sort_values("date")
    .reset_index(drop=True)
)


# ==============================================================================
# 15. Optional: impose a cooldown between dip events


# ==============================================================================
# What this step is doing — and why


def apply_event_cooldown(events_df, prices_df, cooldown_days=63):
    kept = []

    for (ticker, threshold), group in events_df.groupby(["ticker", "threshold"]):
        s = prices_df[ticker].dropna()
        group = group.sort_values("date")

        last_position = None

        for _, row in group.iterrows():
            if row["date"] not in s.index:
                continue

            pos = s.index.get_loc(row["date"])

            if last_position is None or (pos - last_position) >= cooldown_days:
                kept.append(row.to_dict())
                last_position = pos

    return pd.DataFrame(kept)


events_cooldown = apply_event_cooldown(
    events,
    prices,
    cooldown_days=63,   # ~3 months
)

results_cooldown = add_forward_returns(
    events_cooldown,
    prices,
)

print("Original events:", len(events))
print("Cooldown-filtered events:", len(events_cooldown))

print(
    events_cooldown.groupby(["ticker", "threshold"])
    .size()
    .unstack(fill_value=0)
)


# ==============================================================================
# A sensitivity check you should actually look at


# Recompute the 1-year dip edge after imposing the cooldown.
results_cooldown = add_forward_returns(events_cooldown, prices)
summary_cooldown_1y = summarize_dip_results(results_cooldown, horizon="1y")

cooldown_comparison = summary_cooldown_1y.merge(
    baseline_1y,
    on="ticker",
    how="left",
)

cooldown_comparison["dip_edge_pct"] = (
    cooldown_comparison["mean_return"]
    - cooldown_comparison["baseline_mean_return"]
) * 100

cooldown_comparison["threshold_pct"] = cooldown_comparison["threshold"] * 100

print(
    cooldown_comparison[
        ["ticker", "threshold_pct", "n", "mean_return", "dip_edge_pct"]
    ]
    .assign(mean_return_pct=lambda x: x["mean_return"] * 100)
    .drop(columns="mean_return")
    .sort_values(["ticker", "threshold_pct"], ascending=[True, False])
)


# ==============================================================================
# Portfolio-level test: does "buy the dip" work as an actual allocation rule?
# ==============================================================================
# Default portfolio rules


# ==============================================================================
# Why the portfolio test is a separate question


PORTFOLIO_THRESHOLDS = [-0.10, -0.20, -0.30, -0.50]
POSITION_WEIGHT = 0.20
MAX_POSITIONS = 5
MIN_DIVERSIFIED_POSITIONS = 2

# Keep cash at 0% for now. This is conservative for dip strategies that
# spend substantial time out of the market.
CASH_DAILY_RETURN = 0.0


# ==============================================================================
# Build monthly portfolio weights


# ==============================================================================
# What this step is doing — and why


def month_end_trading_dates(index):
    index = pd.DatetimeIndex(index)
    s = pd.Series(index, index=index)
    dates = s.groupby(index.to_period("M")).last().tolist()
    return pd.DatetimeIndex(dates)


def build_monthly_dip_weights(
    prices_df,
    drawdown_df,
    threshold=-0.20,
    max_positions=5,
    position_weight=0.20,
    mode="fixed_slot",
    min_positions=1,
):
    cols = list(prices_df.columns)
    rebalance_dates = month_end_trading_dates(prices_df.index)

    weights = pd.DataFrame(0.0, index=rebalance_dates, columns=cols)
    signal_rows = []

    for dt in rebalance_dates:
        px = prices_df.loc[dt]
        dd = drawdown_df.loc[dt]

        available = px.notna() & dd.notna()
        eligible = dd[available & (dd <= threshold)].sort_values()

        # More-negative drawdown = deeper dip.
        selected = list(eligible.head(max_positions).index)

        if len(selected) < min_positions:
            selected = []

        if selected:
            if mode == "fixed_slot":
                # Avoid leverage if settings are later changed.
                w = min(position_weight, 1.0 / len(selected))
                weights.loc[dt, selected] = w
            elif mode == "fully_invested":
                weights.loc[dt, selected] = 1.0 / len(selected)
            else:
                raise ValueError("mode must be 'fixed_slot' or 'fully_invested'")

        signal_rows.append({
            "date": dt,
            "threshold": threshold,
            "n_eligible": len(eligible),
            "n_selected": len(selected),
            "invested_fraction": weights.loc[dt].sum(),
            "cash_fraction": 1.0 - weights.loc[dt].sum(),
            "selected": ", ".join(selected),
        })

    signals = pd.DataFrame(signal_rows).set_index("date")
    return weights, signals


def build_equal_weight_benchmark(prices_df, drawdown_df):
    cols = list(prices_df.columns)
    rebalance_dates = month_end_trading_dates(prices_df.index)
    weights = pd.DataFrame(0.0, index=rebalance_dates, columns=cols)

    for dt in rebalance_dates:
        available = prices_df.loc[dt].notna() & drawdown_df.loc[dt].notna()
        names = list(available[available].index)
        if names:
            weights.loc[dt, names] = 1.0 / len(names)

    return weights


# ==============================================================================
# Convert target weights into realized returns


# ==============================================================================
# Corrected execution logic


def run_weighted_portfolio(
    prices_df,
    rebalance_weights,
    cash_daily_return=0.0,
    execution_lag_days=1,
):
    """Simulate a portfolio that truly rebalances only on scheduled dates.

    Timing convention
    -----------------
    A target weight observed at the close on signal day t is executed at the
    close after `execution_lag_days` trading days. With the default lag of 1,
    a month-end signal is traded at the next trading day's close, and the new
    holdings first earn the following close-to-close return.

    Between rebalances, weights are allowed to drift with asset returns.
    """
    prices_df = prices_df.sort_index()
    stock_returns = prices_df.pct_change(fill_method=None)
    index = prices_df.index
    cols = list(prices_df.columns)

    # Map each signal date to the first return date on which the new portfolio
    # should be active. If we execute at t+1 close, the first earned return is
    # t+1 close -> t+2 close, i.e. return date t+2.
    effective_targets = {}
    for signal_date, target in rebalance_weights.iterrows():
        if signal_date not in index:
            continue
        signal_pos = index.get_loc(signal_date)
        effective_pos = signal_pos + execution_lag_days + 1
        if effective_pos < len(index):
            effective_targets[index[effective_pos]] = target.fillna(0.0).copy()

    current_weights = pd.Series(0.0, index=cols)
    current_cash = 1.0

    portfolio_returns = []
    wealth_values = []
    weight_rows = []
    exposure_values = []
    cash_values = []
    position_counts = []

    wealth = 1.0

    for dt in index:
        # Rebalance at the prior close before earning today's close-to-close return.
        if dt in effective_targets:
            target = effective_targets[dt].reindex(cols).fillna(0.0).clip(lower=0.0)
            total_target = target.sum()
            if total_target > 1.0 + 1e-12:
                raise ValueError(f"Target weights exceed 100% on {dt}: {total_target:.4f}")
            current_weights = target
            current_cash = 1.0 - total_target

        weight_rows.append(current_weights.copy())
        exposure_values.append(current_weights.sum())
        cash_values.append(current_cash)
        position_counts.append((current_weights > 0).sum())

        r = stock_returns.loc[dt].reindex(cols).fillna(0.0)
        day_return = float((current_weights * r).sum() + current_cash * cash_daily_return)
        portfolio_returns.append(day_return)

        wealth *= (1.0 + day_return)
        wealth_values.append(wealth)

        # Let weights drift after today's returns instead of resetting them daily.
        asset_values = current_weights * (1.0 + r)
        cash_value = current_cash * (1.0 + cash_daily_return)
        gross = asset_values.sum() + cash_value

        if gross > 0:
            current_weights = asset_values / gross
            current_cash = cash_value / gross

    daily_weights = pd.DataFrame(weight_rows, index=index, columns=cols)

    return {
        "returns": pd.Series(portfolio_returns, index=index, name="portfolio_return"),
        "wealth": pd.Series(wealth_values, index=index, name="wealth"),
        "weights": daily_weights,
        "exposure": pd.Series(exposure_values, index=index, name="exposure"),
        "cash_weight": pd.Series(cash_values, index=index, name="cash_weight"),
        "position_count": pd.Series(position_counts, index=index, name="position_count"),
    }


benchmark_rebalance_weights = build_equal_weight_benchmark(prices, drawdown)

benchmark_portfolio = run_weighted_portfolio(
    prices,
    benchmark_rebalance_weights,
    cash_daily_return=CASH_DAILY_RETURN,
    execution_lag_days=1,
)


# ==============================================================================
# Portfolio return, risk, and upside/downside capture


# ==============================================================================
# What these metrics are telling you


def monthly_compound_returns(daily_returns):
    grouped = daily_returns.groupby(daily_returns.index.to_period("M"))
    monthly = grouped.apply(lambda r: (1.0 + r).prod() - 1.0)
    monthly.index = monthly.index.to_timestamp("M")
    return monthly


def max_drawdown_from_wealth(wealth):
    return (wealth / wealth.cummax() - 1.0).min()


def portfolio_metrics(portfolio, benchmark_returns=None, label="Strategy"):
    r = portfolio["returns"].dropna()

    active = portfolio["exposure"].reindex(r.index).fillna(0) > 0
    start = active[active].index[0] if active.any() else r.index[0]

    r_eval = r.loc[start:]
    wealth_eval = (1.0 + r_eval).cumprod()

    years = max(
        (r_eval.index[-1] - r_eval.index[0]).days / 365.25,
        1 / 365.25,
    )

    total_return = wealth_eval.iloc[-1] - 1.0
    cagr = wealth_eval.iloc[-1] ** (1.0 / years) - 1.0
    ann_vol = r_eval.std() * np.sqrt(252)
    sharpe_0rf = (
        r_eval.mean() / r_eval.std() * np.sqrt(252)
        if r_eval.std() > 0 else np.nan
    )

    monthly = monthly_compound_returns(r_eval)

    out = {
        "strategy": label,
        "start": r_eval.index[0],
        "end": r_eval.index[-1],
        "total_return": total_return,
        "cagr": cagr,
        "annualized_vol": ann_vol,
        "sharpe_0rf": sharpe_0rf,
        "max_drawdown": max_drawdown_from_wealth(wealth_eval),
        "best_month": monthly.max(),
        "worst_month": monthly.min(),
        "positive_months": (monthly > 0).mean(),
        "avg_invested_fraction": portfolio["exposure"].loc[start:].mean(),
        "avg_positions": portfolio["position_count"].loc[start:].mean(),
        "pct_days_2plus_positions": (
            portfolio["position_count"].loc[start:] >= 2
        ).mean(),
    }

    if benchmark_returns is not None:
        aligned = pd.concat(
            [
                r_eval.rename("strategy"),
                benchmark_returns.rename("benchmark"),
            ],
            axis=1,
            join="inner",
        ).dropna()

        sm = monthly_compound_returns(aligned["strategy"])
        bm = monthly_compound_returns(aligned["benchmark"])

        m = pd.concat(
            [sm.rename("strategy"), bm.rename("benchmark")],
            axis=1,
        ).dropna()

        up = m["benchmark"] > 0
        down = m["benchmark"] < 0

        avg_up_strategy = m.loc[up, "strategy"].mean() if up.any() else np.nan
        avg_up_benchmark = m.loc[up, "benchmark"].mean() if up.any() else np.nan
        avg_down_strategy = m.loc[down, "strategy"].mean() if down.any() else np.nan
        avg_down_benchmark = m.loc[down, "benchmark"].mean() if down.any() else np.nan

        out["avg_return_up_months"] = avg_up_strategy
        out["avg_return_down_months"] = avg_down_strategy

        out["upside_capture"] = (
            avg_up_strategy / avg_up_benchmark
            if pd.notna(avg_up_benchmark) and avg_up_benchmark != 0
            else np.nan
        )
        out["downside_capture"] = (
            avg_down_strategy / avg_down_benchmark
            if pd.notna(avg_down_benchmark) and avg_down_benchmark != 0
            else np.nan
        )

    return out


# ==============================================================================
# Run all dip thresholds


portfolio_runs = {}
metric_rows = []

metric_rows.append(
    portfolio_metrics(
        benchmark_portfolio,
        benchmark_returns=benchmark_portfolio["returns"],
        label="Equal-weight benchmark",
    )
)

for threshold in PORTFOLIO_THRESHOLDS:
    threshold_label = int(abs(threshold) * 100)

    fixed_weights, fixed_signals = build_monthly_dip_weights(
        prices,
        drawdown,
        threshold=threshold,
        max_positions=MAX_POSITIONS,
        position_weight=POSITION_WEIGHT,
        mode="fixed_slot",
        min_positions=1,
    )

    fixed = run_weighted_portfolio(
        prices,
        fixed_weights,
        cash_daily_return=CASH_DAILY_RETURN,
    )

    portfolio_runs[f"fixed_{threshold_label}"] = {
        "portfolio": fixed,
        "signals": fixed_signals,
    }

    metric_rows.append(
        portfolio_metrics(
            fixed,
            benchmark_returns=benchmark_portfolio["returns"],
            label=f"Fixed-slot {threshold_label}% dip",
        )
    )

    full_weights, full_signals = build_monthly_dip_weights(
        prices,
        drawdown,
        threshold=threshold,
        max_positions=MAX_POSITIONS,
        mode="fully_invested",
        min_positions=MIN_DIVERSIFIED_POSITIONS,
    )

    full = run_weighted_portfolio(
        prices,
        full_weights,
        cash_daily_return=CASH_DAILY_RETURN,
    )

    portfolio_runs[f"full_{threshold_label}"] = {
        "portfolio": full,
        "signals": full_signals,
    }

    metric_rows.append(
        portfolio_metrics(
            full,
            benchmark_returns=benchmark_portfolio["returns"],
            label=f"Fully invested {threshold_label}% dip basket",
        )
    )


portfolio_summary = pd.DataFrame(metric_rows)

display_summary = portfolio_summary[
    [
        "strategy",
        "start",
        "end",
        "cagr",
        "annualized_vol",
        "sharpe_0rf",
        "max_drawdown",
        "avg_invested_fraction",
        "avg_positions",
        "pct_days_2plus_positions",
        "upside_capture",
        "downside_capture",
    ]
].copy()

for col in [
    "cagr",
    "annualized_vol",
    "max_drawdown",
    "avg_invested_fraction",
    "pct_days_2plus_positions",
    "upside_capture",
    "downside_capture",
]:
    display_summary[col] = display_summary[col] * 100

print(
    display_summary.rename(columns={
        "cagr": "CAGR (%)",
        "annualized_vol": "Ann. vol (%)",
        "max_drawdown": "Max drawdown (%)",
        "avg_invested_fraction": "Avg invested (%)",
        "avg_positions": "Avg positions",
        "pct_days_2plus_positions": "Days with 2+ positions (%)",
        "upside_capture": "Upside capture (%)",
        "downside_capture": "Downside capture (%)",
    }).round(2)
)


# ==============================================================================
# Portfolio wealth curves


# ==============================================================================
# How to interpret this figure


wealth_to_plot = pd.DataFrame({
    "Equal-weight benchmark": benchmark_portfolio["wealth"],
    "20% dip: fixed slots": portfolio_runs["fixed_20"]["portfolio"]["wealth"],
    "30% dip: fixed slots": portfolio_runs["fixed_30"]["portfolio"]["wealth"],
    "20% dip: fully invested basket": portfolio_runs["full_20"]["portfolio"]["wealth"],
})

fig, ax = plt.subplots(figsize=(12, 6))

for col in wealth_to_plot.columns:
    ax.plot(wealth_to_plot.index, wealth_to_plot[col], label=col)

ax.set_title("Portfolio Growth: Dip Strategies vs Equal-Weight Benchmark")
ax.set_ylabel("Growth of $1")
ax.set_xlabel("Date")
ax.set_yscale("log")
ax.legend()

plt.tight_layout()

output_path = FIGURE_DIR / "portfolio_wealth_comparison.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"Saved: {output_path}")


# ==============================================================================
# Capital deployment and number of simultaneous positions


p20 = portfolio_runs["fixed_20"]["portfolio"]
signals20 = portfolio_runs["fixed_20"]["signals"]

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(p20["exposure"].index, p20["exposure"] * 100)
ax.set_title("20% Dip Strategy: Portfolio Capital Deployed")
ax.set_ylabel("Invested (%)")
ax.set_xlabel("Date")
ax.set_ylim(0, 105)
plt.tight_layout()

output_path = FIGURE_DIR / "dip_portfolio_exposure.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"Average invested: {p20['exposure'].mean()*100:.1f}%")
print(f"Average positions: {p20['position_count'].mean():.2f}")
print(f"Saved: {output_path}")


position_distribution = (
    signals20["n_selected"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(position_distribution.index.astype(str), position_distribution.values)
ax.set_title("20% Dip Strategy: Number of Positions at Monthly Rebalance")
ax.set_xlabel("Number of selected securities")
ax.set_ylabel("Share of rebalances (%)")
plt.tight_layout()

output_path = FIGURE_DIR / "dip_portfolio_position_count.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.show()

print(
    position_distribution.rename("share_of_rebalances_pct").to_frame().round(2)
)

print(f"Saved: {output_path}")


# ==============================================================================
# Which stocks dominate the portfolio?


weights_20 = portfolio_runs["fixed_20"]["portfolio"]["weights"]

security_exposure = pd.DataFrame({
    "average_weight_pct": weights_20.mean() * 100,
    "pct_days_held_pct": (weights_20 > 0).mean() * 100,
})

print(
    security_exposure
    .sort_values("average_weight_pct", ascending=False)
    .round(2)
)


# ==============================================================================
# Threshold sensitivity


fixed_only = portfolio_summary[
    portfolio_summary["strategy"].str.startswith("Fixed-slot")
].copy()

threshold_labels = (
    fixed_only["strategy"]
    .str.extract(r"(\d+)%")[0]
    .astype(int)
)

threshold_table = pd.DataFrame({
    "dip_threshold_pct": threshold_labels.values,
    "cagr_pct": fixed_only["cagr"].values * 100,
    "ann_vol_pct": fixed_only["annualized_vol"].values * 100,
    "max_drawdown_pct": fixed_only["max_drawdown"].values * 100,
    "avg_invested_pct": fixed_only["avg_invested_fraction"].values * 100,
    "avg_positions": fixed_only["avg_positions"].values,
    "upside_capture_pct": fixed_only["upside_capture"].values * 100,
    "downside_capture_pct": fixed_only["downside_capture"].values * 100,
}).sort_values("dip_threshold_pct")

print(threshold_table.round(2))


# ==============================================================================
# Optional position-size sensitivity


POSITION_SIZE_GRID = [0.10, 0.15, 0.20, 0.25]
position_size_rows = []

for slot_weight in POSITION_SIZE_GRID:
    max_pos = int(np.floor(1.0 / slot_weight))

    w, sig = build_monthly_dip_weights(
        prices,
        drawdown,
        threshold=-0.20,
        max_positions=max_pos,
        position_weight=slot_weight,
        mode="fixed_slot",
        min_positions=1,
    )

    p = run_weighted_portfolio(
        prices,
        w,
        cash_daily_return=CASH_DAILY_RETURN,
    )

    metrics = portfolio_metrics(
        p,
        benchmark_returns=benchmark_portfolio["returns"],
        label=f"{int(slot_weight*100)}% per position",
    )

    position_size_rows.append(metrics)

position_size_summary = pd.DataFrame(position_size_rows)

print(
    pd.DataFrame({
        "position_size_pct": [x * 100 for x in POSITION_SIZE_GRID],
        "cagr_pct": position_size_summary["cagr"] * 100,
        "max_drawdown_pct": position_size_summary["max_drawdown"] * 100,
        "avg_invested_pct": position_size_summary["avg_invested_fraction"] * 100,
        "avg_positions": position_size_summary["avg_positions"],
        "upside_capture_pct": position_size_summary["upside_capture"] * 100,
        "downside_capture_pct": position_size_summary["downside_capture"] * 100,
    }).round(2)
)


# ==============================================================================
# How to interpret the portfolio section
