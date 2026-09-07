"""7-day OHLC forecasting.

Generalizes the RandomForestRegressor approach in finalmodel.ipynb (which
predicted HDB's Close from other tickers' Close prices) into a self-contained
pipeline that works for any ticker:

1. Fit a regressor to predict "tomorrow's Close" from lag/technical features
   built purely from the ticker's own OHLCV history (src/features.py).
2. Forecast the Close price recursively, one trading day at a time, for the
   requested horizon (default 7 trading days) — each step's prediction feeds
   back in as a lag feature for the next step.
3. Reconstruct a full OHLC candle around each forecast Close using the
   ticker's own historical average Open/High/Low-to-Close relationships
   (src/features.intraday_range_stats), since the regressor only predicts
   the closing price directly.

This is a classical-ML approach (matching the repo's existing
scikit-learn-based methodology) rather than a deep-learning model, so it
trains in seconds per ticker and has no GPU/heavy-framework requirement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data import next_trading_days
from src.features import build_feature_frame, intraday_range_stats, latest_feature_row

MODEL_FACTORIES = {
    "Random Forest": lambda: RandomForestRegressor(
        n_estimators=300, max_depth=8, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": lambda: GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
    ),
}


@dataclass
class BacktestMetrics:
    mae: float
    rmse: float
    mape: float
    directional_accuracy: float
    n_test: int


@dataclass
class ForecastResult:
    model_name: str
    backtest: BacktestMetrics
    history: pd.DataFrame  # original OHLCV
    forecast: pd.DataFrame  # Open/High/Low/Close indexed by forecast date
    feature_importances: pd.Series


def _chronological_split(feats: pd.DataFrame, test_frac: float = 0.15):
    n_test = max(20, int(len(feats) * test_frac))
    train, test = feats.iloc[:-n_test], feats.iloc[-n_test:]
    return train, test


def _fit_and_backtest(feats: pd.DataFrame, model_name: str):
    train, test = _chronological_split(feats)
    feature_cols = [c for c in feats.columns if c != "target"]

    model = MODEL_FACTORIES[model_name]()
    model.fit(train[feature_cols], train["target"])

    y_pred = model.predict(test[feature_cols])
    y_true = test["target"].to_numpy()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

    actual_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(y_pred[1:] - y_true[:-1])
    directional_accuracy = float(np.mean(actual_dir == pred_dir) * 100) if len(actual_dir) else float("nan")

    metrics = BacktestMetrics(
        mae=mae, rmse=rmse, mape=mape,
        directional_accuracy=directional_accuracy, n_test=len(test),
    )

    # Refit on the full dataset for the actual forward forecast.
    full_model = MODEL_FACTORIES[model_name]()
    full_model.fit(feats[feature_cols], feats["target"])
    importances = getattr(full_model, "feature_importances_", None)
    imp_series = (
        pd.Series(importances, index=feature_cols).sort_values(ascending=False)
        if importances is not None else pd.Series(dtype=float)
    )
    return full_model, feature_cols, metrics, imp_series


def _recursive_close_forecast(df: pd.DataFrame, model, feature_cols: list[str], horizon: int) -> list[float]:
    working = df.copy()
    preds: list[float] = []
    for _ in range(horizon):
        row = latest_feature_row(working)[feature_cols]
        next_close = float(model.predict(row)[0])
        preds.append(next_close)

        # Append a synthetic row so the next iteration's lag/MA features
        # account for this step's prediction. Volume is held at its recent
        # average since future volume is unobservable.
        next_date = working.index[-1] + pd.Timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += pd.Timedelta(days=1)
        new_row = pd.DataFrame(
            {
                "Open": [next_close],
                "High": [next_close],
                "Low": [next_close],
                "Close": [next_close],
                "Volume": [working["Volume"].tail(10).mean()],
            },
            index=[next_date],
        )
        working = pd.concat([working, new_row])
    return preds


def _reconstruct_ohlc(df: pd.DataFrame, closes: list[float], dates: list) -> pd.DataFrame:
    stats = intraday_range_stats(df)
    prev_close = float(df["Close"].iloc[-1])

    rows = []
    for close, date in zip(closes, dates):
        open_px = prev_close * (1 + stats["open_offset_mean"])
        high_px = max(open_px, close) * (1 + stats["high_offset_mean"])
        low_px = min(open_px, close) * (1 - stats["low_offset_mean"])
        rows.append({"Date": date, "Open": open_px, "High": high_px, "Low": low_px, "Close": close})
        prev_close = close

    return pd.DataFrame(rows).set_index("Date")


def run_forecast(df: pd.DataFrame, model_name: str = "Random Forest", horizon: int = 7) -> ForecastResult:
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(MODEL_FACTORIES)}.")

    feats = build_feature_frame(df)
    model, feature_cols, metrics, importances = _fit_and_backtest(feats, model_name)

    closes = _recursive_close_forecast(df, model, feature_cols, horizon)
    dates = next_trading_days(df.index[-1].date(), horizon)
    forecast_df = _reconstruct_ohlc(df, closes, dates)

    return ForecastResult(
        model_name=model_name,
        backtest=metrics,
        history=df,
        forecast=forecast_df,
        feature_importances=importances,
    )
