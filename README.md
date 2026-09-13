Does Buying the Dip Actually Work?

Code accompanying my video testing a simple question: after an individual stock falls substantially from its recent high, are its subsequent returns actually better than usual?

The analysis focuses on seven widely followed stocks:

AAPL, MSFT, TSLA, NVDA, AMZN, META, and NFLX.

Rather than treating every day below a drawdown threshold as a new observation, the event study looks at threshold crossings below 10%, 20%, 30%, and 50% drawdowns from the trailing 252-trading-day high. Forward returns are then measured from the next trading day's close, which avoids entering at a price that would not have been known when the signal was observed.

The code also compares post-dip returns with each stock's own unconditional forward-return distribution, applies a 63-trading-day cooldown as a sensitivity analysis, and tests whether the idea translates into a simple monthly-rebalanced portfolio strategy.

What the script does

The script:

downloads adjusted daily price history with yfinance;

calculates drawdowns from each stock's trailing 52-week high;

detects first crossings of the 10%, 20%, 30%, and 50% drawdown thresholds;

calculates 1-month, 3-month, and 1-year forward returns;

compares post-dip returns with each stock's ordinary forward returns;

repeats the event analysis with a 63-trading-day cooldown;

runs monthly portfolio backtests using several dip thresholds;

compares the dip strategies with an equal-weight benchmark;

reports return, volatility, drawdown, exposure, diversification, and upside/downside-capture statistics;

saves the main figures to figures/.

The portfolio simulation observes the signal at month-end and executes the rebalance at the next trading day's close. Portfolio weights are then allowed to drift naturally until the next rebalance rather than being reset every day.

Running it

Python 3.10+ is recommended.

pip install yfinance pandas numpy matplotlib pyarrow
python buying_the_dip.py

The script caches downloaded prices in data/focus_stock_prices.parquet. Set REFRESH_FOCUS_DATA = True in the script if you want to redownload the frozen sample.

Reproducibility

The public version freezes the analysis at September 11, 2026 so that rerunning the code later does not silently extend the sample and produce a different experiment.

Yahoo Finance can occasionally revise historical adjusted-price data, so very small differences are still possible across download dates or library versions.

Important limitations

This is an exploratory demonstration, not evidence of a production-ready trading strategy.

Most importantly, the stock universe was chosen from companies that are prominent today. That creates substantial survivorship and selection bias: the analysis does not include the many companies that experienced large drawdowns and later disappeared, were acquired, or never became major winners.

Other limitations include:

only seven stocks are studied;

taxes, transaction costs, spreads, and slippage are ignored;

cash is modeled with a 0% return;

the thresholds and portfolio rules are simple and were not optimized using a formal train/test framework;

observations from the same stock and market regime are not fully independent;

strong historical performance of this particular stock set should not be interpreted as an expected future return.

The purpose of the project is to test a common investing claim carefully and show how the answer changes when the question is defined more precisely.

Files

buying_the_dip.py — complete analysis and portfolio backtest

data/ — cached price data created when the script runs

figures/ — figures created by the analysis

Disclaimer

This project is for educational and informational purposes only and is not financial or investment advice.
