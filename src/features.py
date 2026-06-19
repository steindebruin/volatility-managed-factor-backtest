import pandas as pd

factors = ["Mkt", "SMB", "HML", "RMW", "CMA", "Mom"]

# Corsi (2009) horizons in trading days: day, week, month
horizons = {"rv_d": 1, "rv_w": 5, "rv_m": 22}
forward = 22  # trading days in the forecast target


# daily realised variance: squared daily factor return as proxy
def daily_rv(daily_returns, factor):
    return daily_returns[factor] ** 2


# classical HAR features on the last trading day of each month, plus the forward 22-day target
def create_volatility_features(daily_returns, factor):
    rv = daily_rv(daily_returns, factor)

    # backward-looking averages over each Corsi (2009) horizon
    table = pd.DataFrame(index=rv.index)
    for name, h in horizons.items():
        table[name] = rv.rolling(h).mean()

    # random-walk benchmark: realised variance of the past 22 trading days
    table["rv_benchmark"] = rv.rolling(forward).sum()

    # forward 22-day realised variance: sum of RV over the next 22 trading days
    forward_sum = rv[::-1].rolling(forward).sum()[::-1].shift(-1)
    table["rv_fwd"] = forward_sum

    # keep the last trading day of each month
    month_end = table.groupby(rv.index.to_period("M")).tail(1)
    month_end.index = month_end.index.to_period("M").to_timestamp("M")

    return month_end.dropna()


if __name__ == "__main__":
    from data import load_factor_data

    daily_returns, _ = load_factor_data()
    features = create_volatility_features(daily_returns, "Mkt")
    print(features.shape)
    print(features.head())