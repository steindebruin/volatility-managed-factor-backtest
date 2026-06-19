import pandas as pd

max_leverage = 3.0

# scaling constant c set on the training window so the managed factor matches
# the factor's unconditional variance there; frozen for the whole OOS path
def _scaling_constant(factor_returns, variance_forecast, train_end):
    train = variance_forecast.index <= train_end
    weights_unscaled = 1.0 / variance_forecast[train]
    managed = weights_unscaled * factor_returns.reindex(variance_forecast.index)[train]
    return factor_returns.reindex(variance_forecast.index)[train].std() / managed.std()


# Moreira-Muir (2017) vol-managed returns: weight c / forecast (made at month t),
# applied to the factor return realised in month t+1
def construct_vol_managed_returns(factor_returns, variance_forecast, train_years=20, cap=max_leverage):
    forecast = variance_forecast.dropna()

    train_end = forecast.index[train_years * 12 - 1] if len(forecast) > train_years * 12 else forecast.index[len(forecast) // 2]
    c = _scaling_constant(factor_returns, forecast, train_end)

    # weight set at month t from the forecast of t+1 variance
    weight_t = (c / forecast).clip(lower=0.0, upper=cap)

    # apply it to next month's return: shift the weight onto the t+1 index
    weight = weight_t.shift(1).dropna()
    aligned_factor = factor_returns.reindex(weight.index)
    managed = weight * aligned_factor

    return pd.DataFrame({"weight": weight, "managed_return": managed, "factor_return": aligned_factor})


if __name__ == "__main__":
    from data import load_factor_data
    from forecasting import get_forecasts

    daily_returns, monthly_returns = load_factor_data()
    forecasts = get_forecasts(daily_returns, "Mkt")

    result = construct_vol_managed_returns(monthly_returns["Mkt"], forecasts["RF"])
    print(result.shape)
    print(result.describe())