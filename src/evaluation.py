import numpy as np
import pandas as pd
from scipy import stats

models = ["RV", "HAR", "RF"]


# Campbell-Thompson (2008) out-of-sample R^2 vs the expanding historical mean of RV
def r2_oos(realized, forecast):
    realized, forecast = realized.align(forecast, join="inner")
    benchmark = realized.expanding().mean().shift(1)  # mean through t-1, no lookahead

    valid = benchmark.notna()
    realized, forecast, benchmark = realized[valid], forecast[valid], benchmark[valid]

    sse_model = ((realized - forecast) ** 2).sum()
    sse_bench = ((realized - benchmark) ** 2).sum()
    return 1 - sse_model / sse_bench


# QLIKE loss for variance forecasts; lower is better
def qlike_loss(realized, forecast):
    realized, forecast = realized.align(forecast, join="inner")
    ratio = realized / forecast
    return ratio - np.log(ratio) - 1


def qlike(realized, forecast):
    return qlike_loss(realized, forecast).mean()


# Diebold-Mariano test on a loss differential for 1-step-ahead forecasts,
# with the Harvey-Leybourne-Newbold small-sample correction: returns (statistic, two-sided p-value).
# lower loss for model a gives a negative stat.
def diebold_mariano(loss_a, loss_b):
    d = (loss_a - loss_b).dropna().to_numpy()
    n = len(d)
    d_mean = d.mean()

    # 1-step horizon: long-run variance is just the sample variance of d
    var_d = np.var(d, ddof=0) / n
    dm = d_mean / np.sqrt(var_d)

    # HLN correction for horizon h = 1
    hln = np.sqrt((n + 1 - 2 + 1 / n) / n) * dm

    p = 2 * (1 - stats.t.cdf(abs(hln), df=n - 1))
    return hln, p


# accuracy table: R^2_oos and QLIKE per model, plus DM p-values on the ladder
def forecast_accuracy_table(forecasts, realized):
    rows = {}
    for model in models:
        rows[model] = {
            "r2_oos": r2_oos(realized, forecasts[model]),
            "qlike": qlike(realized, forecasts[model]),
        }
    table = pd.DataFrame(rows).T

    losses = {m: qlike_loss(realized, forecasts[m]) for m in models}
    ladder = [("HAR", "RV"), ("RF", "RV"), ("RF", "HAR")]
    dm = {}
    for a, b in ladder:
        _, p = diebold_mariano(losses[a], losses[b])
        dm[f"{a} vs {b}"] = p

    return table, pd.Series(dm, name="DM p-value (QLIKE)")


# annualised Sharpe ratio of (already excess) net returns
def sharpe_ratio(net_return):
    r = net_return.dropna()
    return np.sqrt(12) * r.mean() / r.std()


# geometric annualised return
def annual_return(net_return):
    r = net_return.dropna()
    return (1 + r).prod() ** (12 / len(r)) - 1


# maximum drawdown of the compounded net return
def max_drawdown(net_return):
    cumulative = (1 + net_return.dropna()).cumprod()
    running_max = cumulative.cummax()
    return (cumulative / running_max - 1).min()


# average monthly turnover
def average_turnover(turnover):
    return turnover.dropna().mean()


# Moreira-Muir alpha: managed net return regressed on the unmanaged factor;
# returns annualised alpha and its t-statistic
def alpha(net_return, factor_return):
    r, f = net_return.align(factor_return, join="inner")
    r, f = r.dropna(), f.reindex(r.index)
    x = np.column_stack([np.ones(len(f)), f.to_numpy()])
    beta, _, _, _ = np.linalg.lstsq(x, r.to_numpy(), rcond=None)
    resid = r.to_numpy() - x @ beta
    sigma2 = resid @ resid / (len(r) - 2)
    se = np.sqrt(sigma2 * np.linalg.inv(x.T @ x)[0, 0])
    return beta[0] * 12, beta[0] / se


# per-portfolio performance metrics from a backtest result frame
def performance_metrics(result, factor_return):
    return {
        "sharpe": sharpe_ratio(result["net_return"]),
        "annual_return": annual_return(result["net_return"]),
        "max_drawdown": max_drawdown(result["net_return"]),
        "turnover": average_turnover(result["turnover"]),
        "alpha": alpha(result["net_return"], factor_return)[0],
    }


# stationary block bootstrap (Politis-Romano): geometric block lengths, same
# block indices drawn for both series so their dependence is preserved
def _stationary_bootstrap_indices(n, block_length, rng):
    idx = np.empty(n, dtype=int)
    p = 1.0 / block_length
    i = rng.integers(0, n)
    for t in range(n):
        if t == 0 or rng.random() < p:
            i = rng.integers(0, n)
        else:
            i = (i + 1) % n
        idx[t] = i
    return idx


# paired test of the Sharpe difference between two net-return series
def sharpe_difference_test(net_a, net_b, block_length=6, reps=10000, seed=0):
    a, b = net_a.align(net_b, join="inner")
    a, b = a.dropna(), b.reindex(a.index).dropna()
    a, b = a.align(b, join="inner")
    a, b = a.to_numpy(), b.to_numpy()
    n = len(a)

    def sharpe_diff(x, y):
        return np.sqrt(12) * (x.mean() / x.std() - y.mean() / y.std())

    observed = sharpe_diff(a, b)

    rng = np.random.default_rng(seed)
    diffs = np.empty(reps)
    for k in range(reps):
        idx = _stationary_bootstrap_indices(n, block_length, rng)
        diffs[k] = sharpe_diff(a[idx], b[idx])

    # two-sided p-value: how often the bootstrapped difference falls on the
    # opposite side of zero from the observed difference
    if observed >= 0:
        p = 2 * np.mean(diffs <= 0)
    else:
        p = 2 * np.mean(diffs >= 0)
    return observed, min(p, 1.0)


# upper-triangular table of Sharpe-difference p-values across the three models
def sharpe_difference_table(net_returns, block_length=6, reps=10000):
    table = pd.DataFrame(index=models, columns=models, dtype=float)
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            _, p = sharpe_difference_test(net_returns[a], net_returns[b], block_length, reps)
            table.loc[a, b] = p
    return table


if __name__ == "__main__":
    from data import load_factor_data
    from features import create_volatility_features, factors
    from forecasting import get_forecasts
    from portfolio import construct_vol_managed_returns
    from backtest import run_backtest

    daily_returns, monthly_returns = load_factor_data()

    sharpe_rows = {}
    performance_rows = {}
    dm_rows = {}
    sharpe_tests = {}

    for factor in factors:
        forecasts = get_forecasts(daily_returns, factor)
        realised = create_volatility_features(daily_returns, factor)["rv_fwd"].reindex(forecasts.index)

        net_returns = {}
        for model in models:
            managed = construct_vol_managed_returns(monthly_returns[factor], forecasts[model])
            result = run_backtest(managed)
            net_returns[model] = result["net_return"]

            sharpe_rows.setdefault(factor, {})[model] = sharpe_ratio(result["net_return"])
            performance_rows[(factor, model)] = performance_metrics(result, result["factor_return"])

        _, dm = forecast_accuracy_table(forecasts, realised)
        dm_rows[factor] = dm
        sharpe_tests[factor] = sharpe_difference_table(net_returns)

    sharpe_table = pd.DataFrame(sharpe_rows).T[models]
    performance_table = pd.DataFrame(performance_rows).T
    dm_table = pd.DataFrame(dm_rows).T

    pd.set_option("display.float_format", lambda v: f"{v:.3f}")

    print("Net Sharpe ratio")
    print(sharpe_table)
    print()
    print("Performance summary")
    print(performance_table)
    print()
    print("Diebold-Mariano p-values on QLIKE loss")
    print(dm_table)
    print()
    print("Sharpe-difference p-values")
    for factor in factors:
        print(f"\n{factor}")
        print(sharpe_tests[factor])