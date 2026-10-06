# Does Buying the Dip Actually Work?

Code for the analysis behind my **Buy the Dip** video.

The question is simple: after an individual stock falls substantially from its recent high, does that fact by itself predict better future returns?

The study uses seven widely followed stocks: **AAPL, MSFT, TSLA, NVDA, AMZN, META, and NFLX**.

## What is tested

A dip is defined relative to the highest adjusted close over the trailing 252 trading days. The script tests first crossings below four drawdown levels:

- 10%
- 20%
- 30%
- 50%

The signal is observed at the close. Entry is at the **next trading day's close**, so the triggering close is never treated as an executable entry price. Forward returns are measured over approximately 1 month, 3 months, and 1 year.

The event study compares post-dip returns with each stock's own ordinary forward returns. A 63-trading-day cooldown is also used as a sensitivity check so that repeated crossings during the same decline do not dominate the results.

The script then asks a separate portfolio question: what happens if a portfolio waits for dip signals and rebalances monthly? Signals are observed at month-end and trades occur at the next close. Each stock has a fixed 1/7 portfolio slot, so unused slots remain in cash and portfolio weights drift between rebalances.

## Running the analysis

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python buying_the_dip.py
```

The analysis is frozen through **September 11, 2026**. `yfinance` downloads the required history when the script runs.

## Data handling

This repository intentionally contains **no cached Yahoo Finance price history and no event-level export**. Price data are downloaded at runtime, used in memory, and discarded when the process ends. The script writes only aggregate summary tables and figures locally, and those generated directories are ignored by Git.

## Files

- `buying_the_dip.py` - event study, cooldown sensitivity, and monthly portfolio simulation
- `requirements.txt` - Python dependencies
- `.gitignore` - keeps local price data, caches, and generated artifacts out of Git history

## Limitations

This is an exploratory demonstration, not a production trading strategy. The stock universe was chosen with hindsight from companies that are prominent today, creating substantial survivorship and selection bias. The analysis also ignores taxes and most real-world implementation costs, uses simple thresholds that were not selected through a formal train/test process, and contains observations that are not fully independent across stocks and market regimes.

Results from this small survivor-selected sample should not be generalized to all stocks or interpreted as expected future performance.

## Disclaimer

For educational and informational purposes only. Nothing here is financial or investment advice.
