"""Download Mithi's weather history, train the models, print honest metrics.

Data source: Open-Meteo archive API (free, no API key) — ERA5 reanalysis
for the grid point nearest Mithi, Tharparkar (24.74 N, 69.81 E).

Usage:  python train_model.py
Writes: data/mithi_weather_2015_2020.csv, data/mithi_weather_2021_2025.csv,
        metrics.json
"""
import json

import pandas as pd
import requests

import model

LAT, LON = 24.7433, 69.8066  # Mithi, Tharparkar, Sindh, Pakistan
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PARAMS = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": "2015-01-01",
    "end_date": "2025-12-31",
    "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
             "precipitation_sum,wind_speed_10m_max",
    "timezone": "Asia/Karachi",
}


def download():
    r = requests.get(ARCHIVE_URL, params=PARAMS, timeout=120)
    r.raise_for_status()
    daily = r.json()["daily"]
    df = pd.DataFrame(daily).rename(columns={"time": "date"})
    return df


def main():
    df = download()
    df[df["date"] < "2021-01-01"].to_csv("data/mithi_weather_2015_2020.csv", index=False)
    df[df["date"] >= "2021-01-01"].to_csv("data/mithi_weather_2021_2025.csv", index=False)
    print(f"Downloaded {len(df)} days: {df['date'].min()} to {df['date'].max()}")

    feat = model.build_features(model.load_history())
    _, _, metrics = model.train(feat)
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Test results (trained on 2015-2023, tested on 2024-2025):")
    print(f"  Tomorrow's max temperature: mean error {metrics['tmax_mae_c']} °C")
    print(f"  Rain prediction accuracy:  {metrics['rain_accuracy']:.1%} "
          f"(rain falls on {metrics['rain_base_rate_test']:.1%} of test days)")


if __name__ == "__main__":
    main()
