import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from features import factors

models = ["RV", "HAR", "RF"]
model_colors = {"RV": "#999999", "HAR": "#1f77b4", "RF": "#d62728"}
factor_color = "#cccccc"

figure_dir = Path(__file__).resolve().parent.parent / "figures"


def _save(fig, name):
    figure_dir.mkdir(exist_ok=True)
    fig.savefig(figure_dir / name, bbox_inches="tight")


# 2x3 small multiples of cumulative net return per factor, with the unmanaged factor as a grey reference line
def plot_cumulative_returns(results):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for ax, factor in zip(axes.flat, factors):
        frames = results[factor]
        factor_cum = (1 + frames["RV"]["factor_return"]).cumprod()
        ax.plot(factor_cum.index, factor_cum, color=factor_color, linewidth=1, label="Factor")
        for model in models:
            cum = (1 + frames[model]["net_return"]).cumprod()
            ax.plot(cum.index, cum, color=model_colors[model], linewidth=1, label=model)
        ax.set_title(factor)
        ax.set_yscale("log")
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.supylabel("Cumulative return (log scale)")
    fig.tight_layout()
    _save(fig, "cumulative_returns.pdf")
    plt.show()


# 2x3 small multiples of drawdown over time per factor
def plot_drawdowns(results):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for ax, factor in zip(axes.flat, factors):
        frames = results[factor]
        for model in models:
            cum = (1 + frames[model]["net_return"]).cumprod()
            drawdown = cum / cum.cummax() - 1
            ax.plot(drawdown.index, drawdown, color=model_colors[model], linewidth=1, label=model)
        ax.set_title(factor)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.supylabel("Drawdown")
    fig.tight_layout()
    _save(fig, "drawdowns.pdf")
    plt.show()


# grouped bar chart of average monthly turnover by factor and model
def plot_turnover(results):
    data = pd.DataFrame({
        model: [results[f][model]["turnover"].mean() for f in factors]
        for model in models
    }, index=factors)

    fig, ax = plt.subplots(figsize=(9, 5))
    data.plot.bar(ax=ax, color=[model_colors[m] for m in models], width=0.75)
    ax.set_ylabel("Average monthly turnover")
    ax.set_xlabel("")
    ax.legend(frameon=False)
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    _save(fig, "turnover.pdf")
    plt.show()


# grouped bar chart of out-of-sample R^2 by factor and model
def plot_forecast_accuracy(r2_table):
    fig, ax = plt.subplots(figsize=(9, 5))
    r2_table[models].plot.bar(ax=ax, color=[model_colors[m] for m in models], width=0.75)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Out-of-sample $R^2$")
    ax.set_xlabel("")
    ax.legend(frameon=False)
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    _save(fig, "forecast_accuracy.pdf")
    plt.show()


if __name__ == "__main__":
    from data import load_factor_data
    from features import create_volatility_features
    from forecasting import get_forecasts
    from portfolio import construct_vol_managed_returns
    from backtest import run_backtest
    from evaluation import r2_oos

    daily_returns, monthly_returns = load_factor_data()

    # run the full pipeline for every factor and model
    results = {}
    r2_rows = {}
    for factor in factors:
        forecasts = get_forecasts(daily_returns, factor)
        realised = create_volatility_features(daily_returns, factor)["rv_fwd"].reindex(forecasts.index)

        results[factor] = {}
        r2_rows[factor] = {}
        for model in models:
            managed = construct_vol_managed_returns(monthly_returns[factor], forecasts[model])
            results[factor][model] = run_backtest(managed)
            r2_rows[factor][model] = r2_oos(realised, forecasts[model])

    r2_table = pd.DataFrame(r2_rows).T[models]

    plot_cumulative_returns(results)
    plot_drawdowns(results)
    plot_turnover(results)
    plot_forecast_accuracy(r2_table)