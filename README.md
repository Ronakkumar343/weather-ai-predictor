# 🌦️ Thar Weather AI — weather prediction for Mithi, Tharparkar

**Machine-learning weather forecasts for one of the hottest, driest places in Pakistan** — built by **Ronak Kumar**, a Class 9 student from Mithi who wanted a forecast that actually knows his hometown.

## Why this exists

Global weather apps barely know Tharparkar exists. Mithi's climate is extreme and specific: summers cross 45 °C, rain arrives in a few monsoon bursts, and a good local forecast matters — for farmers deciding when to sow, schools planning around heat, and families planning travel. So this project trains a model on **Mithi's own weather history** instead of borrowing a generic one.

## What it does

- **Tomorrow's maximum temperature** — regression model
- **Tomorrow's rain chance** (≥ 1 mm) — classification model
- **Mithi climate explorer** — 11 years of local weather: monthly temperature and rainfall patterns, record days

## The data (real, local, 2015–2025)

- Source: **Open-Meteo archive (ERA5 reanalysis)**, grid point nearest Mithi (24.78° N, 69.82° E)
- **4,018 days** of daily max/min/mean temperature, rainfall, and wind — bundled in `data/` (two CSVs: 2015–2020 and 2021–2025), refreshable anytime with `train_model.py`
- Hottest day in the record: **48.1 °C on 27 May 2024**. Average year: only ~339 mm of rain.

## The model

Two Random Forest models (scikit-learn, 300 trees each). Features use **only what is known before the target day** — yesterday's and last week's weather, rolling 7-day averages, and the season (day-of-year as sin/cos). No future leakage.

### Honest test results

Trained on 2015–2023, tested on **731 days of 2024–2025 the model had never seen**:

| Prediction | Result |
|---|---|
| Tomorrow's max temperature | **Mean error 1.02 °C** |
| Rain tomorrow (≥ 1 mm) | **93.0% accuracy** |

Read the rain number honestly: rain falls on only 8.9% of days in Mithi, so even "always say no rain" scores 91.1%. The model beats that baseline — and in the app it reports a rain *probability*, which is more useful than a yes/no in a desert.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

- `app.py` — the Streamlit app (live prediction uses the Open-Meteo forecast API for the last 14 observed days; the climate tab works fully offline)
- `model.py` — feature engineering + training (shared by the app and the trainer)
- `train_model.py` — re-downloads the data, retrains, and rewrites `metrics.json`

## Roadmap

- [ ] Add humidity and pressure features from the hourly archive
- [ ] Compare Random Forest against gradient boosting and a simple LSTM
- [ ] Weekly forecast view, not just tomorrow
- [ ] Sindhi/Urdu language toggle for local users

---

*Part of my journey from Mithi to MIT — learning AI by building things my own hometown needs.*
