"""Module 3 — Anticipation de la qualité de l'eau à H+1 (STL + LSTM), section 5.3.

La composante saisonnière S_t issue de la décomposition STL est calculée UNE
SEULE FOIS puis stockée telle quelle dans seasonality.db (table immuable,
lecture seule ensuite — voir db.py). Le LSTM est entraîné sur la composante de
tendance T_t, avec une fenêtre glissante de 3 pas de temps (T_t-3, T_t-2, T_t-1)
-> T_t, comme décrit en section 4.4.3.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

from data_gen import generate_hourly_series
import db

VARS = ["PH", "TEMP", "DO", "TURBIDITY"]
COL_MAP = {"PH": "s_ph", "TEMP": "s_temp", "DO": "s_do", "TURBIDITY": "s_turbidity"}
IDX_MAP = {"TEMP": 0, "DO": 1, "PH": 2, "TURBIDITY": 3}  # ordre renvoyé par db.lookup


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def decompose_and_store():
    """Décompose chaque variable (STL, période journalière = 24h), stocke la
    saisonnalité en base (une seule fois, table immuable) et renvoie les
    tendances T_t pour l'entraînement du LSTM."""
    df = generate_hourly_series()
    trends = {"date": df["date"].values, "heure": df["heure"].values}
    seasonal_records = df[["date", "heure"]].copy()

    for var in VARS:
        stl = STL(df[var], period=24, robust=True)
        res = stl.fit()
        trends[var] = res.trend.values
        seasonal_records[COL_MAP[var]] = res.seasonal.values

    db.init_db()
    if not db.is_populated():
        seasonal_records = seasonal_records.drop_duplicates(subset=["date", "heure"])
        db.populate_from_stl(seasonal_records)

    return pd.DataFrame(trends)


def make_windows(series, lag=3):
    X, y = [], []
    for i in range(lag, len(series)):
        X.append(series[i - lag:i])
        y.append(series[i])
    return np.array(X), np.array(y)


def train_lstm_models(trend_df):
    from tensorflow import keras

    results, models, scalers = {}, {}, {}

    for var in VARS:
        series = trend_df[var].values.reshape(-1, 1)
        scaler = StandardScaler()
        series_s = scaler.fit_transform(series).flatten()

        X, y = make_windows(series_s, lag=3)
        split = int(len(X) * 0.85)  # découpage chronologique, pas aléatoire
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        X_train = X_train.reshape((-1, 3, 1))
        X_test = X_test.reshape((-1, 3, 1))

        model = keras.Sequential(
            [
                keras.layers.Input(shape=(3, 1)),
                keras.layers.LSTM(16),
                keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        model.fit(X_train, y_train, epochs=8, batch_size=32, verbose=0)

        y_pred = model.predict(X_test, verbose=0).flatten()
        results[var] = {
            "RMSE": round(rmse(y_test, y_pred), 4),
            "MAE": round(mean_absolute_error(y_test, y_pred), 4),
            "MAPE": round(mean_absolute_percentage_error(y_test + 1e-6, y_pred + 1e-6), 4),
            "R2": round(r2_score(y_test, y_pred), 4),
        }
        models[var] = model
        scalers[var] = scaler

    return results, models, scalers


def predict_h_plus_1(models, scalers, trend_df, var, date, heure):
    """Prédit T_t à H+1 pour `var` à partir des 3 derniers points connus, puis
    lit S_t (lecture seule, immuable) dans seasonality.db pour reconstituer :
    valeur_finale = tendance_LSTM + S_t."""
    row_idx = trend_df.index[(trend_df["date"] == date) & (trend_df["heure"] == heure)]
    if len(row_idx) == 0 or row_idx[0] < 3:
        return None
    idx = row_idx[0]

    scaler = scalers[var]
    window = trend_df[var].values[idx - 3: idx]
    window_s = scaler.transform(window.reshape(-1, 1)).flatten().reshape(1, 3, 1)

    pred_s = models[var].predict(window_s, verbose=0).flatten()[0]
    pred_trend = float(scaler.inverse_transform([[pred_s]])[0][0])

    next_hour = (heure + 1) % 24
    next_date = date + 1 if next_hour == 0 else date

    seasonal_row = db.lookup(next_date, next_hour)
    s_val = float(seasonal_row[IDX_MAP[var]]) if seasonal_row is not None else 0.0

    final_value = pred_trend + s_val
    return final_value, pred_trend, s_val, next_date, next_hour
