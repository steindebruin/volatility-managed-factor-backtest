import pandas as pd

transaction_cost = 0.001  # 10 bps per unit traded


# vectorised backtest: drift-adjusted turnover and net-of-cost returns
def run_backtest(managed, tcost=transaction_cost):
    weight = managed["weight"]
    factor_return = managed["factor_return"]
    gross_return = managed["managed_return"]

    # weight after drifting with the factor over the month, before rebalancing
    drifted_weight = (weight * (1 + factor_return)).shift(1)

    # turnover traded at each rebalance; first month trades in from zero
    turnover = (weight - drifted_weight).abs()
    turnover.iloc[0] = abs(weight.iloc[0])

    cost = tcost * turnover
    net_return = gross_return - cost

    return pd.DataFrame({
        "weight": weight,
        "turnover": turnover,
        "gross_return": gross_return,
        "cost": cost,
        "net_return": net_return,
        "factor_return": factor_return,
    })


if __name__ == "__main__":
    from data import load_factor_data
    from forecasting import get_forecasts
    from portfolio import construct_vol_managed_returns

    daily_returns, monthly_returns = load_factor_data()
    forecasts = get_forecasts(daily_returns, "Mkt")
    managed = construct_vol_managed_returns(monthly_returns["Mkt"], forecasts["RF"])

    result = run_backtest(managed)
    print(result.shape)
    print(result.describe().round(5))