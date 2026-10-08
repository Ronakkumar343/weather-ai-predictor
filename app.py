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
    """Build one feature row predicting the day after `recent` ends.

    The appended target-day row carries 0.0 placeholder raw values. Every
    feature is a lag/rolling value taken from *earlier* days, so the
    placeholders never enter the features — they only fill the target
    columns, which prediction ignores. (Appending NaN instead would make
    build_features' dropna() delete the target row entirely, and the
    caller would silently get the last observed day's row.)
    """
    recent = recent.sort_values("date").reset_index(drop=True)
    target_date = recent["date"].iloc[-1] + pd.Timedelta(days=1)
    ext = pd.concat([recent, pd.DataFrame([{
        "date": target_date,
        "temperature_2m_max": 0.0, "temperature_2m_min": 0.0,
        "temperature_2m_mean": 0.0, "precipitation_sum": 0.0,
        "wind_speed_10m_max": 0.0,
        "relative_humidity_2m_mean": 0.0,
        "surface_pressure_mean": 0.0,
    }])], ignore_index=True)
    feat = model.build_features(ext)
    return feat.iloc[[-1]]


def weekly_forecast(reg, clf, recent: pd.DataFrame, days: int = 7,
                    rainy_day_mm: float = None) -> pd.DataFrame:
    """Recursive 7-day outlook, built with the project's real pipeline.

    Day +1 is predicted from the observed days, exactly like the
    tomorrow forecast. For days +2..+7 the model's own predictions are
    rolled forward as inputs: the predicted max temperature becomes
    that day's temperature_2m_max, and every step re-runs
    model.build_features, so the lag/rolling features are the project's
    real ones — nothing is mocked. Columns the model does not predict
    are filled with documented approximations:
      * min temp  = predicted max − the recent average day/night range,
      * mean temp = average of the predicted max and min,
      * precipitation = rain probability × the average amount that
        falls on a rainy day in the reference history (expected value),
      * wind, humidity, pressure = last observed value (persistence).

    Because later days are built on earlier predictions, errors
    compound — only day +1 carries the verified accuracy. The result
    columns are: date, tmax_pred, rain_prob.
    """
    recent = recent.sort_values("date").reset_index(drop=True)
    if rainy_day_mm is None:
        rainy = recent.loc[recent["precipitation_sum"] >= model.RAIN_MM,
                           "precipitation_sum"]
        rainy_day_mm = float(rainy.mean()) if len(rainy) else 5.0
    diurnal = float((recent["temperature_2m_max"]
                     - recent["temperature_2m_min"]).mean())
    last = recent.iloc[-1]
    work = recent.copy()
    rows = []
    for _ in range(days):
        row = feature_row_for_tomorrow(work)
        tmax = float(reg.predict(row[model.FEATURES])[0])
        rain_p = float(clf.predict_proba(row[model.FEATURES])[0][1])
        date = work["date"].iloc[-1] + pd.Timedelta(days=1)
        rows.append({"date": date, "tmax_pred": tmax, "rain_prob": rain_p})
        tmin = tmax - diurnal
        work = pd.concat([work, pd.DataFrame([{
            "date": date,
            "temperature_2m_max": tmax,
            "temperature_2m_min": tmin,
            "temperature_2m_mean": (tmax + tmin) / 2,
            "precipitation_sum": rain_p * rainy_day_mm,
            "wind_speed_10m_max": float(last["wind_speed_10m_max"]),
            "relative_humidity_2m_mean": float(last["relative_humidity_2m_mean"]),
            "surface_pressure_mean": float(last["surface_pressure_mean"]),
        }])], ignore_index=True)
    return pd.DataFrame(rows)


def fetch_recent():
    """Last 14 days of observed weather from the Open-Meteo forecast API.

    The daily values come from the forecast API's daily endpoint; the
    humidity and pressure columns are daily means aggregated from its
    hourly endpoint, exactly as in the training data.
    """
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": LAT, "longitude": LON, "past_days": 14, "forecast_days": 1,
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
                 "precipitation_sum,wind_speed_10m_max",
        "timezone": "Asia/Karachi",
    }, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["daily"]).rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.iloc[:14]  # observed days only, not the forecast day

    rh = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": LAT, "longitude": LON, "past_days": 14, "forecast_days": 1,
        "hourly": "relative_humidity_2m,surface_pressure",
        "timezone": "Asia/Karachi",
    }, timeout=30)
    rh.raise_for_status()
    hourly = pd.DataFrame(rh.json()["hourly"])
    hourly["date"] = pd.to_datetime(hourly["time"]).dt.normalize()
    agg = hourly.groupby("date").agg(
        relative_humidity_2m_mean=("relative_humidity_2m", "mean"),
        surface_pressure_mean=("surface_pressure", "mean"),
    ).reset_index()
    return df.merge(agg, on="date", how="left")


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

        rainy_hist = hist.loc[hist["precipitation_sum"] >= model.RAIN_MM,
                              "precipitation_sum"]
        rainy_day_mm = float(rainy_hist.mean()) if len(rainy_hist) else None
        week = weekly_forecast(reg, clf, recent, rainy_day_mm=rainy_day_mm)
        st.divider()
        st.subheader("🗓️ Next 7 days — model outlook")
        cols = st.columns(7)
        for col, (_, r) in zip(cols, week.iterrows()):
            with col:
                st.markdown(f"**{r['date'].strftime('%a')}**  \n"
                            f"{r['date'].strftime('%d %b')}")
                st.markdown(f"### {r['tmax_pred']:.0f} °C")
                st.caption(f"🌧️ rain {r['rain_prob']:.0%}")
        st.line_chart(week.set_index("date")["tmax_pred"], height=180)
        st.caption(
            "Honest note: these 7 days are **model estimates**, made by feeding "
            "each day's prediction back in as the next day's input (recursive "
            "forecasting). Only tomorrow's prediction is verified — mean error "
            "1.01 °C on 731 unseen days of 2024–2025. Days further out are built "
            "on earlier predictions, so their accuracy is worse, and no accuracy "
            "figure is claimed for them."
        )
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
        "week's temperature, rain, wind, humidity and pressure, rolling 7-day "
        "averages, the 3-day pressure tendency, and the season (day-of-year as "
        "sin/cos) — only information known *before* the target day, so there is "
        "no cheating from the future."
    )
    st.write(
        "The **7-day outlook** reuses these same two models recursively: each "
        "day's prediction is fed back as the next day's input. That is why only "
        "tomorrow's number carries the verified accuracy below — the further "
        "out a day is, the more it rests on earlier predictions instead of "
        "real observations."
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
