"""Thar Weather AI — a Streamlit app by Ronak Kumar.

Machine-learning weather predictions for Mithi, Tharparkar — one of the
hottest, driest places in Pakistan, where a good forecast matters for
farmers, schools, and daily life.

Run:  streamlit run app.py
Data: Open-Meteo (ERA5 archive 2015-2025 + live forecast API).
"""
import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

import model

LAT, LON = 24.7433, 69.8066
DATA_DIR = Path(__file__).parent / "data"
METRICS = Path(__file__).parent / "metrics.json"

st.set_page_config(page_title="Thar Weather AI — Mithi", page_icon="🌦️")
st.title("🌦️ Thar Weather AI")
st.caption("Machine-learning weather forecasts for **Mithi, Tharparkar, Sindh** — "
           "trained on 11 years of real local weather data (Open-Meteo / ERA5).")


@st.cache_data
def load_history():
    return model.load_history(str(DATA_DIR))


@st.cache_resource
def load_models():
    feat = model.build_features(load_history())
    reg, clf = model.train_final(feat)
    return reg, clf


def feature_row_for_tomorrow(recent: pd.DataFrame) -> pd.DataFrame:
    """Build one feature row predicting the day after `recent` ends."""
    recent = recent.sort_values("date").reset_index(drop=True)
    target_date = recent["date"].iloc[-1] + pd.Timedelta(days=1)
    ext = pd.concat([recent, pd.DataFrame([{
        "date": target_date,
        "temperature_2m_max": float("nan"), "temperature_2m_min": float("nan"),
        "temperature_2m_mean": float("nan"), "precipitation_sum": float("nan"),
        "wind_speed_10m_max": float("nan"),
    }])], ignore_index=True)
    feat = model.build_features(ext)
    return feat.iloc[[-1]]


def fetch_recent():
    """Last 14 days of observed weather from the Open-Meteo forecast API."""
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": LAT, "longitude": LON, "past_days": 14, "forecast_days": 1,
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
                 "precipitation_sum,wind_speed_10m_max",
        "timezone": "Asia/Karachi",
    }, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["daily"]).rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df.iloc[:14]  # observed days only, not the forecast day


hist = load_history()

tab1, tab2, tab3 = st.tabs(["🔮 Tomorrow's prediction", "📊 Mithi's climate", "🤖 How it works"])

with tab1:
    reg, clf = load_models()
    try:
        recent = fetch_recent()
        row = feature_row_for_tomorrow(recent)
        tmax = float(reg.predict(row[model.FEATURES])[0])
        rain_p = float(clf.predict_proba(row[model.FEATURES])[0][1])
        target = (recent["date"].iloc[-1] + pd.Timedelta(days=1)).strftime("%A, %d %B %Y")
        c1, c2 = st.columns(2)
        c1.metric(f"Predicted max temperature — {target}", f"{tmax:.1f} °C")
        c2.metric("Chance of rain (≥ 1 mm)", f"{rain_p:.0%}")
        st.write("Based on the last 14 days of observed weather in Mithi.")
    except Exception as e:
        st.warning(f"Live data is unreachable right now ({e}). "
                   "The climate tab below works fully offline.")

with tab2:
    st.subheader("11 years of Mithi weather (2015–2025)")
    monthly = hist.copy()
    monthly["month"] = monthly["date"].dt.month
    m = monthly.groupby("month").agg(
        avg_max=("temperature_2m_max", "mean"),
        avg_min=("temperature_2m_min", "mean"),
        rain_mm=("precipitation_sum", "sum"),
    )
    m["rain_mm"] = m["rain_mm"] / 11  # average per year
    st.write("Average temperatures by month — May and June cross 40 °C:")
    st.bar_chart(m[["avg_max", "avg_min"]])
    st.write("Average rainfall by month — almost all of it arrives with the monsoon:")
    st.bar_chart(m[["rain_mm"]])
    hottest = hist.loc[hist["temperature_2m_max"].idxmax()]
    st.info(f"Hottest day in the record: **{hottest['temperature_2m_max']:.1f} °C** "
            f"on {hottest['date'].strftime('%d %B %Y')}. "
            f"Average year: only ~{(hist['precipitation_sum'].sum() / 11):.0f} mm of rain.")

with tab3:
    st.subheader("The model")
    st.write(
        "Two Random Forest models (scikit-learn). Features: yesterday's and last "
        "week's temperature, rain and wind, rolling 7-day averages, and the season "
        "(day-of-year as sin/cos) — only information known *before* the target day, "
        "so there is no cheating from the future."
    )
    if METRICS.exists():
        met = json.loads(METRICS.read_text())
        st.write(
            f"**Honest test results** — trained on 2015–2023, tested on 2024–2025 "
            f"({met['test_rows']} days the model had never seen):\n\n"
            f"- Max-temperature mean error: **{met['tmax_mae_c']} °C**\n"
            f"- Rain-day accuracy: **{met['rain_accuracy']:.1%}** "
            f"(rain falls on only {met['rain_base_rate_test']:.1%} of days in Mithi, "
            f"so this number must be read with care)"
        )
    st.caption("Built by Ronak Kumar · Data: Open-Meteo (ERA5) · "
               "Code: github.com/Ronakkumar343/weather-ai-predictor")
