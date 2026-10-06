"""Thar Weather AI — feature engineering and model training.

Predicts, for Mithi (Tharparkar, Sindh, Pakistan):
  1. Tomorrow's maximum temperature  (regression)
  2. Whether it will rain tomorrow   (classification, rain = >= 1.0 mm)

Data: Open-Meteo historical archive (ERA5 reanalysis), daily values
2015-2025 for the grid point nearest Mithi (24.78 N, 69.82 E), plus
daily mean relative humidity and surface pressure aggregated from the
hourly archive (24 hourly values per day).

Features use only information available *before* the target day:
yesterday's values, last-week values, rolling 7-day means, the 3-day
pressure tendency, and the season (day-of-year encoded as sin/cos).
No leakage from the future.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

FEATURES = [
    "doy_sin", "doy_cos",
    "tmax_lag1", "tmin_lag1", "tmean_lag1", "precip_lag1", "wind_lag1",
    "rh_lag1", "press_lag1",
    "tmax_lag7", "precip_lag7",
    "tmax_roll7", "tmin_roll7", "precip_roll7", "rh_roll7", "press_roll7",
    "press_trend",
]
RAIN_MM = 1.0  # a day counts as "rain" at/above this many mm


def load_csv(path):
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_history(data_dir="data"):
    """Load the bundled dataset (stored as two CSV era files)."""
    import glob
    frames = [pd.read_csv(p, parse_dates=["date"])
              for p in sorted(glob.glob(f"{data_dir}/mithi_weather_*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values("date").reset_index(drop=True)


def build_features(df):
    """Turn a daily weather table into a supervised learning table."""
    df = df.copy()
    doy = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    for col, new in [("temperature_2m_max", "tmax_lag1"),
                     ("temperature_2m_min", "tmin_lag1"),
                     ("temperature_2m_mean", "tmean_lag1"),
                     ("precipitation_sum", "precip_lag1"),
                     ("wind_speed_10m_max", "wind_lag1")]:
        df[new] = df[col].shift(1)
    df["tmax_lag7"] = df["temperature_2m_max"].shift(7)
    df["precip_lag7"] = df["precipitation_sum"].shift(7)
    df["tmax_roll7"] = df["temperature_2m_max"].shift(1).rolling(7).mean()
    df["tmin_roll7"] = df["temperature_2m_min"].shift(1).rolling(7).mean()
    df["precip_roll7"] = df["precipitation_sum"].shift(1).rolling(7).sum()
    # Humidity + pressure (daily means from the hourly archive), same
    # lag/rolling pattern as the other variables — known before the
    # target day, so still no future leakage.
    df["rh_lag1"] = df["relative_humidity_2m_mean"].shift(1)
    df["press_lag1"] = df["surface_pressure_mean"].shift(1)
    df["rh_roll7"] = df["relative_humidity_2m_mean"].shift(1).rolling(7).mean()
    df["press_roll7"] = df["surface_pressure_mean"].shift(1).rolling(7).mean()
    # 3-day surface-pressure tendency ending yesterday: falling pressure
    # is a classic early sign of unsettled weather.
    df["press_trend"] = (df["surface_pressure_mean"].shift(1)
                         - df["surface_pressure_mean"].shift(4))
    df["target_tmax"] = df["temperature_2m_max"]
    df["target_rain"] = (df["precipitation_sum"] >= RAIN_MM).astype(int)
    return df.dropna().reset_index(drop=True)


def train(df_feat, split_date="2024-01-01"):
    """Time-based split: train on the past, test on the future."""
    train = df_feat[df_feat["date"] < split_date]
    test = df_feat[df_feat["date"] >= split_date]
    reg = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    reg.fit(train[FEATURES], train["target_tmax"])
    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf.fit(train[FEATURES], train["target_rain"])
    pred_tmax = reg.predict(test[FEATURES])
    pred_rain = clf.predict(test[FEATURES])
    metrics = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "tmax_mae_c": round(float(mean_absolute_error(test["target_tmax"], pred_tmax)), 2),
        "rain_accuracy": round(float(accuracy_score(test["target_rain"], pred_rain)), 3),
        "rain_base_rate_test": round(float(test["target_rain"].mean()), 3),
        "split_date": split_date,
    }
    return reg, clf, metrics


def train_final(df_feat):
    """Train on ALL data (used by the app for live predictions)."""
    reg = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    reg.fit(df_feat[FEATURES], df_feat["target_tmax"])
    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf.fit(df_feat[FEATURES], df_feat["target_rain"])
    return reg, clf
