"""Event study and simple portfolio test for the 'buy the dip' question.

The script downloads price history at runtime and keeps it in memory. It does
not save or redistribute vendor-sourced price histories.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "NFLX"]
START_DATE = "2000-01-01"
# yfinance treats end as exclusive. This includes September 11, 2026.
END_DATE = "2026-09-12"

LOOKBACK = 252
THRESHOLDS = [-0.10, -0.20, -0.30, -0.50]
HORIZONS = {"1m": 21, "3m": 63, "1y": 252}
ENTRY_LAG = 1
COOLDOWN_DAYS = 63

FIGURE_DIR = Path("figures")
OUTPUT_DIR = Path("outputs")


def download_adjusted_close(tickers):
    raw = yf.download(
        tickers=tickers,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=True,
        group_by="column",
    )

    if raw.empty:
        raise RuntimeError("No price data were returned.")

    if isinstance(raw.columns, pd.MultiIndex):
        field = "Adj Close" if "Adj Close" in raw.columns.get_level_values(0) else "Close"
        prices = raw[field].copy()
    else:
        field = "Adj Close" if "Adj Close" in raw.columns else "Close"
        prices = raw[[field]].copy()
        prices.columns = [tickers[0]]

    prices = prices.reindex(columns=tickers).sort_index().dropna(how="all")
    missing = prices.columns[prices.notna().sum() == 0].tolist()
    if missing:
        raise RuntimeError(f"No usable history returned for: {missing}")
    return prices


def trailing_drawdown(prices, lookback=LOOKBACK):
    high = prices.rolling(lookback, min_periods=1).max()
    return prices / high - 1.0


def detect_crossings(drawdown, thresholds=THRESHOLDS):
    rows = []
    for ticker in drawdown.columns:
        series = drawdown[ticker].dropna()
        for threshold in thresholds:
            crossed = (series <= threshold) & (series.shift(1) > threshold)
            for date in series.index[crossed.fillna(False)]:
                rows.append(
                    {
                        "ticker": ticker,
                        "signal_date": date,
                        "threshold": threshold,
                        "drawdown": float(series.loc[date]),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["ticker", "signal_date", "threshold", "drawdown"])
    return pd.DataFrame(rows).sort_values(["ticker", "threshold", "signal_date"]).reset_index(drop=True)


def apply_cooldown(events, prices, cooldown_days=COOLDOWN_DAYS):
    kept = []
    for (ticker, threshold), group in events.groupby(["ticker", "threshold"], sort=False):
        trading_index = prices[ticker].dropna().index
        positions = pd.Series(np.arange(len(trading_index)), index=trading_index)
        last_position = None
        for row in group.sort_values("signal_date").itertuples(index=False):
            if row.signal_date not in positions.index:
                continue
            position = int(positions.loc[row.signal_date])
            if last_position is None or position - last_position >= cooldown_days:
                kept.append(row._asdict())
                last_position = position
    return pd.DataFrame(kept)


def add_forward_returns(events, prices, horizons=HORIZONS, entry_lag=ENTRY_LAG):
    rows = []
    for event in events.itertuples(index=False):
        series = prices[event.ticker].dropna()
        if event.signal_date not in series.index:
            continue
        signal_i = series.index.get_loc(event.signal_date)
        entry_i = signal_i + entry_lag
        if entry_i >= len(series):
            continue

        row = event._asdict()
        row["entry_date"] = series.index[entry_i]
        for label, horizon in horizons.items():
            exit_i = entry_i + horizon
            row[f"return_{label}"] = (
                float(series.iloc[exit_i] / series.iloc[entry_i] - 1.0)
                if exit_i < len(series)
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def ordinary_forward_means(prices, horizons=HORIZONS, entry_lag=ENTRY_LAG):
    rows = []
    for ticker in prices.columns:
        series = prices[ticker].dropna()
        for label, horizon in horizons.items():
            forward = series.shift(-(entry_lag + horizon)) / series.shift(-entry_lag) - 1.0
            rows.append(
                {
                    "ticker": ticker,
                    "horizon": label,
                    "ordinary_mean": float(forward.mean()),
                    "ordinary_median": float(forward.median()),
                }
            )
    return pd.DataFrame(rows)


def summarize_events(results, baseline, horizons=HORIZONS):
    rows = []
    for threshold in THRESHOLDS:
        for label in horizons:
            col = f"return_{label}"
            subset = results.loc[results["threshold"].eq(threshold), ["ticker", col]].dropna()
            per_stock = subset.groupby("ticker")[col].agg(["mean", "median", "count"]).reset_index()
            base = baseline.query("horizon == @label")
            per_stock = per_stock.merge(base, on="ticker", how="left")
            eligible = per_stock.loc[per_stock["count"] > 0].copy()
            if eligible.empty:
                continue
            rows.append(
                {
                    "threshold": threshold,
                    "horizon": label,
                    "events": int(eligible["count"].sum()),
                    "stocks": int(len(eligible)),
                    "mean_post_dip": float(eligible["mean"].mean()),
                    "ordinary_mean": float(eligible["ordinary_mean"].mean()),
                    "equal_stock_edge": float((eligible["mean"] - eligible["ordinary_mean"]).mean()),
                    "median_post_dip": float(eligible["median"].mean()),
                }
            )
    return pd.DataFrame(rows)


def month_end_signal_dates(index):
    s = pd.Series(index=index, data=index)
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).last().values)


def simulate_monthly_strategy(prices, drawdown, threshold):
    common = prices.dropna(how="any").copy()
    dd = drawdown.reindex(common.index)[common.columns]
    signal_dates = month_end_signal_dates(common.index)

    shares = pd.Series(0.0, index=common.columns)
    cash = 1.0
    records = []
    pending_target = None

    for i, date in enumerate(common.index):
        px = common.loc[date]
        wealth = cash + float((shares * px).sum())

        if pending_target is not None:
            target_weights = pending_target
            shares = wealth * target_weights / px
            cash = wealth * (1.0 - float(target_weights.sum()))
            pending_target = None
            wealth = cash + float((shares * px).sum())

        exposure = float((shares * px).sum() / wealth) if wealth > 0 else 0.0
        records.append(
            {
                "date": date,
                "wealth": wealth,
                "exposure": exposure,
                "positions": int((shares.abs() > 0).sum()),
            }
        )

        if date in signal_dates and i + 1 < len(common.index):
            active = dd.loc[date].le(threshold)
            # Fixed 1/N slots: unused slots remain cash.
            target = pd.Series(0.0, index=common.columns)
            target.loc[active] = 1.0 / len(common.columns)
            pending_target = target

    return pd.DataFrame(records).set_index("date")


def equal_weight_buy_and_hold(prices):
    common = prices.dropna(how="any")
    weights = pd.Series(1.0 / common.shape[1], index=common.columns)
    shares = weights / common.iloc[0]
    wealth = common.mul(shares, axis=1).sum(axis=1)
    return wealth / wealth.iloc[0]


def performance_metrics(wealth, label, exposure=None, positions=None):
    wealth = wealth.dropna()
    returns = wealth.pct_change().dropna()
    years = (wealth.index[-1] - wealth.index[0]).days / 365.25
    drawdown = wealth / wealth.cummax() - 1.0
    row = {
        "strategy": label,
        "start": wealth.index[0],
        "end": wealth.index[-1],
        "total_return": wealth.iloc[-1] / wealth.iloc[0] - 1.0,
        "cagr": (wealth.iloc[-1] / wealth.iloc[0]) ** (1.0 / years) - 1.0,
        "annualized_vol": returns.std() * np.sqrt(252),
        "sharpe_0rf": returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else np.nan,
        "max_drawdown": drawdown.min(),
    }
    if exposure is not None:
        row["avg_exposure"] = exposure.reindex(wealth.index).mean()
    if positions is not None:
        row["avg_positions"] = positions.reindex(wealth.index).mean()
    return row


def save_figures(summary, portfolio_runs, benchmark):
    FIGURE_DIR.mkdir(exist_ok=True)

    three_month = summary.query("horizon == '3m'").copy()
    if not three_month.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar((-100 * three_month["threshold"]).astype(int).astype(str) + "%", 100 * three_month["equal_stock_edge"])
        ax.axhline(0, linewidth=1)
        ax.set(xlabel="Drawdown threshold", ylabel="3-month edge vs ordinary days (percentage points)", title="Forward returns after dip signals")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "dip_event_study.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(benchmark.index, benchmark, label="Equal-weight buy and hold")
    for threshold, run in portfolio_runs.items():
        ax.plot(run.index, run["wealth"], label=f"Buy at {abs(threshold):.0%} drawdown")
    ax.set_yscale("log")
    ax.set(xlabel="Date", ylabel="Growth of $1", title="Monthly dip strategies")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "dip_portfolios.png", dpi=180)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    prices = download_adjusted_close(TICKERS)
    drawdown = trailing_drawdown(prices)
    events = detect_crossings(drawdown)
    results = add_forward_returns(events, prices)
    baseline = ordinary_forward_means(prices)
    summary = summarize_events(results, baseline)

    cooldown_events = apply_cooldown(events, prices)
    cooldown_results = add_forward_returns(cooldown_events, prices)
    cooldown_summary = summarize_events(cooldown_results, baseline)

    common = prices.dropna(how="any")
    benchmark = equal_weight_buy_and_hold(common)
    portfolio_runs = {}
    metrics = [performance_metrics(benchmark, "Equal-weight buy and hold")]
    for threshold in [-0.10, -0.20, -0.30, -0.50]:
        run = simulate_monthly_strategy(prices, drawdown, threshold)
        portfolio_runs[threshold] = run
        metrics.append(
            performance_metrics(
                run["wealth"],
                f"Buy at {abs(threshold):.0%} drawdown",
                exposure=run["exposure"],
                positions=run["positions"],
            )
        )

    portfolio_summary = pd.DataFrame(metrics)

    # Only aggregate tables are written. Event-level and price-level data stay in memory.
    summary.to_csv(OUTPUT_DIR / "dip_event_summary.csv", index=False)
    cooldown_summary.to_csv(OUTPUT_DIR / "dip_cooldown_summary.csv", index=False)
    portfolio_summary.to_csv(OUTPUT_DIR / "dip_portfolio_summary.csv", index=False)
    save_figures(summary, portfolio_runs, benchmark)

    print("\nEvent-study summary")
    print(summary.to_string(index=False))
    print("\n63-trading-day cooldown sensitivity")
    print(cooldown_summary.to_string(index=False))
    print("\nPortfolio summary")
    print(portfolio_summary.to_string(index=False))
    print("\nRaw price histories and event-level rows were not saved.")


if __name__ == "__main__":
    main()
