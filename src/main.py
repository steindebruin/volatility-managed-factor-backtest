import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd

from data import load_factor_data
from features import create_volatility_features, factors
from forecasting import get_forecasts
from portfolio import construct_vol_managed_returns
from backtest import run_backtest
from evaluation import (
    models,
    sharpe_ratio,
    performance_metrics,
    forecast_accuracy_table,
    sharpe_difference_table,
)
from plots import (
    plot_cumulative_returns,
    plot_drawdowns,
    plot_turnover,
    plot_forecast_accuracy,
)


# run the full pipeline for every factor and model
def run_pipeline(daily_returns, monthly_returns):
    results = {}
    sharpe_rows = {}
    performance_rows = {}
    r2_rows = {}
    dm_rows = {}
    sharpe_tests = {}

    for factor in factors:
        # variance forecasts (RV, HAR, RF) and the realised forward variance target
        forecasts = get_forecasts(daily_returns, factor)
        realised = create_volatility_features(daily_returns, factor)["rv_fwd"].reindex(forecasts.index)

        results[factor] = {}
        net_returns = {}
        for model in models:
            # build the vol-managed portfolio and net out transaction costs
            managed = construct_vol_managed_returns(monthly_returns[factor], forecasts[model])
            result = run_backtest(managed)
            results[factor][model] = result
            net_returns[model] = result["net_return"]

            # portfolio performance: Sharpe and the full metric set
            sharpe_rows.setdefault(factor, {})[model] = sharpe_ratio(result["net_return"])
            performance_rows[(factor, model)] = performance_metrics(result, result["factor_return"])

        # forecast accuracy (R^2_oos, QLIKE, DM tests) and Sharpe-difference tests
        accuracy, dm = forecast_accuracy_table(forecasts, realised)
        r2_rows[factor] = accuracy["r2_oos"]
        dm_rows[factor] = dm
        sharpe_tests[factor] = sharpe_difference_table(net_returns)

    tables = {
        "sharpe": pd.DataFrame(sharpe_rows).T[models],
        "performance": pd.DataFrame(performance_rows).T,
        "r2": pd.DataFrame(r2_rows).T[models],
        "dm": pd.DataFrame(dm_rows).T,
        "sharpe_tests": sharpe_tests,
    }
    return results, tables


# print every result table to the console
def print_tables(tables):
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")

    print("Net Sharpe ratio")
    print(tables["sharpe"])
    print()
    print("Performance summary")
    print(tables["performance"])
    print()
    print("Out-of-sample R^2")
    print(tables["r2"])
    print()
    print("Diebold-Mariano p-values on QLIKE loss")
    print(tables["dm"])
    print()
    print("Sharpe-difference p-values")
    for factor in factors:
        print(f"\n{factor}")
        print(tables["sharpe_tests"][factor])


if __name__ == "__main__":
    # load aligned daily and monthly factor returns
    daily_returns, monthly_returns = load_factor_data()

    # forecasting, portfolio construction, backtesting and evaluation for all factors
    results, tables = run_pipeline(daily_returns, monthly_returns)

    print_tables(tables)

    # save the four figures to figures/
    plot_cumulative_returns(results)
    plot_drawdowns(results)
    plot_turnover(results)
    plot_forecast_accuracy(tables["r2"])

    print("\nFinished.")