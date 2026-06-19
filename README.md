# Volatility-Managed Factor Investing: Forecasting vs Portfolio Performance
By Stein de Bruin, Erasmus School of Economics


Do more sophisticated volatility forecasts produce better volatility-managed factor portfolios after transaction costs?

Volatility-managed portfolios scale exposure to a factor by the inverse of a variance forecast, taking less risk when volatility is expected to be high. This project asks whether a *better* variance forecast produces a *better* portfolio, once realistic transaction costs and leverage limits are imposed.

## Method

Three variance forecasts of increasing sophistication are compared:

- **RV** — a random-walk benchmark (realised variance of the past 22 trading days)
- **HAR** — the heterogeneous autoregressive model of Corsi (2009)
- **RF** — a random forest on the same set of features

Each forecast feeds a Moreira-Muir (2017) volatility-managed portfolio for the six Fama-French factors (Mkt, SMB, HML, RMW, CMA, Mom) over a common Jan 1963 - Apr 2026 sample. Forecasts use an expanding window with monthly refitting; the Random Forest is retuned annually. Portfolios are capped at 3x leverage and charged 10 bps on drift-adjusted turnover.

Both halves of the question are tested formally: forecast accuracy with out-of-sample R², QLIKE loss, and Diebold-Mariano tests. Portfolio performance with net Sharpe ratios and a stationary block bootstrap for Sharpe differences.

## Main result

Better volatility forecasts do not produce better volatility-managed portfolios after costs. More sophisticated forecasts are sometimes significantly more accurate, but this accuracy does not translate into net Sharpe ratios. In one case the more accurate forecast produces a significantly worse portfolio. Performance is driven far more by which factor is managed than by which forecast is used. The mechanism is turnover: the smoother HAR forecast trades less and pays lower costs.

## Structure

```
volatility-managed-factor-backtest/
├── main.py              # runs the whole pipeline end to end
├── analysis.ipynb       # the analysis narrative with results and figures
├── src/                 # importable library code
│   ├── data.py          # download and align Fama-French factor returns
│   ├── features.py      # realised variance and HAR features
│   ├── forecasting.py   # RV, HAR and random forest forecasts (cached)
│   ├── portfolio.py     # Moreira-Muir (2017) volatility-managed weights
│   ├── backtest.py      # drift-adjusted turnover and transaction costs
│   ├── evaluation.py    # forecast accuracy and portfolio metrics
│   └── plots.py         # figures
├── results/             # cached forecasts
└── figures/             # output figures
```

## Usage

```
pip install -r requirements.txt
python main.py
```

Factor data is downloaded from Kenneth French's data library on first run and cached locally. Forecasts are cached in `results/forecasts/`; to regenerate them from scratch, delete that folder and rerun (around 15 minutes for the Random Forest across all factors).