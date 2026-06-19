import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

predictors = ["rv_d", "rv_w", "rv_m"]
target = "rv_fwd"

forecast_dir = Path(__file__).resolve().parent.parent / "results" / "forecasts"

# random forest hyperparameter grid
rf_grid = {
    "n_estimators": [200, 500],
    "max_depth": [3, 5, None],
    "min_samples_leaf": [1, 5, 10], # guards against overfitting the short sample
}


# RV benchmark: random walk, realised variance of the past 22 trading days
def forecast_rv(features):
    return features["rv_benchmark"].rename("RV")


# expanding-window HAR regression, refit each month
def forecast_har(features, initial_train_years=20, factor=""):
    x = features[predictors].to_numpy()
    y = features[target].to_numpy()
    start = initial_train_years * 12

    preds = {}
    for t in tqdm(range(start, len(features)), desc=f"HAR {factor}".strip()):
        model = LinearRegression().fit(x[:t], y[:t])
        preds[features.index[t]] = model.predict(x[t:t + 1])[0]

    return pd.Series(preds, name="HAR")


# pick RF hyperparameters on the most recent 20% of the training block (= validation sample)
def _tune_rf(x_train, y_train):
    split = int(len(x_train) * 0.8)
    x_fit, x_val = x_train[:split], x_train[split:]
    y_fit, y_val = y_train[:split], y_train[split:]

    best_params, best_mse = None, np.inf
    for n, depth, leaf in product(rf_grid["n_estimators"], rf_grid["max_depth"], rf_grid["min_samples_leaf"]):
        model = RandomForestRegressor(
            n_estimators=n, max_depth=depth, min_samples_leaf=leaf,
            random_state=0, n_jobs=-1,
        ).fit(x_fit, y_fit)
        mse = np.mean((model.predict(x_val) - y_val) ** 2)
        if mse < best_mse:
            best_params, best_mse = {"n_estimators": n, "max_depth": depth, "min_samples_leaf": leaf}, mse

    return best_params


# expanding-window RF, refit monthly, hyperparameters retuned once a year
def forecast_rf(features, initial_train_years=20, factor=""):
    x = features[predictors].to_numpy()
    y = features[target].to_numpy()
    start = initial_train_years * 12

    preds = {}
    params = None
    for t in tqdm(range(start, len(features)), desc=f"RF {factor}".strip()):
        # retune on the first forecast and every 12 months after
        if (t - start) % 12 == 0:
            params = _tune_rf(x[:t], y[:t])

        # refit on the full window, including the validation sample
        model = RandomForestRegressor(random_state=0, n_jobs=-1, **params).fit(x[:t], y[:t])
        preds[features.index[t]] = model.predict(x[t:t + 1])[0]

    return pd.Series(preds, name="RF")


# the three forecasts for one factor on the common OOS index, cached to results
def get_forecasts(daily_returns, factor, initial_train_years=20):
    path = forecast_dir / f"{factor}.csv"

    # read the cache if it exists, otherwise compute and save
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)

    from features import create_volatility_features

    features = create_volatility_features(daily_returns, factor)
    har = forecast_har(features, initial_train_years, factor)
    rf = forecast_rf(features, initial_train_years, factor)
    rv = forecast_rv(features).reindex(har.index)  # align RV to the OOS months

    forecasts = pd.concat([rv, har, rf], axis=1)

    forecast_dir.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(path)
    return forecasts


if __name__ == "__main__":
    from data import load_factor_data
    from features import factors

    daily_returns, _ = load_factor_data()
    for factor in factors:
        forecasts = get_forecasts(daily_returns, factor)
        print(factor, forecasts.shape)