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
- Since October 2026 the dataset also carries **daily mean relative humidity and surface pressure**, aggregated from the Open-Meteo **hourly** archive (96,432 hourly values, averaged 24-per-day) and stored as two extra columns in the same CSVs
- Hottest day in the record: **48.1 °C on 27 May 2024**. Average year: only ~339 mm of rain.

## The model

Two Random Forest models (scikit-learn, 300 trees each). Features use **only what is known before the target day** — yesterday's and last week's weather, rolling 7-day averages, the season (day-of-year as sin/cos), and since October 2026 also humidity and pressure (yesterday's daily means, 7-day rolling means, and the 3-day pressure tendency). No future leakage.

### Honest test results

Trained on 2015–2023, tested on **731 days of 2024–2025 the model had never seen** — the same split before and after the humidity/pressure features were added, so the numbers are directly comparable:

| Prediction | Before (12 features) | Now (17 features) |
|---|---|---|
| Tomorrow's max temperature | Mean error 1.02 °C | **Mean error 1.01 °C** |
| Rain tomorrow (≥ 1 mm) | 93.0% accuracy | **93.3% accuracy** |

The new features helped, but honestly only a little — about a hundredth of a degree on temperature, and on rain two extra correctly-called days out of 731 (682 vs 680). That small gain is reported as measured, not dressed up. Read the rain number honestly too: rain falls on only 8.9% of days in Mithi, so even "always say no rain" scores 91.1%. The model beats that baseline — and in the app it reports a rain *probability*, which is more useful than a yes/no in a desert.

### Model comparison (Oct 2026)

Is Random Forest actually the right choice? `compare_models.py` races it against gradient boosting and a **persistence baseline** (predict tomorrow = yesterday, no learning at all) — same 17 features for both learned models, same split (train 2015–2023, test 731 days of 2024–2025), gradient boosting with scikit-learn defaults, nothing tuned on the test set. Full numbers in `model_comparison.json`:

| Model | Tomorrow's max temp — mean error | Rain tomorrow — accuracy |
|---|---|---|
| Persistence (tomorrow = yesterday) | 1.024 °C | 91.9% |
| Gradient Boosting | 1.014 °C | 92.3% |
| **Random Forest (used in the app)** | **1.009 °C** | **93.3%** |

Random Forest wins on both tasks, so it stays — but honestly by very little on temperature: Mithi's max temperature is so stable day-to-day that just repeating yesterday scores 1.024 °C, only 0.015 °C worse than the forest. The clearer win is on rain (93.3% vs 91.9% for persistence and 91.1% for always saying "no rain"). Temperature errors are shown to 3 decimals here because at 2 decimals all three models round to 1.01–1.02 °C and the real ordering disappears.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

- `app.py` — the Streamlit app (live prediction uses the Open-Meteo forecast API for the last 14 observed days; the climate tab works fully offline)
- `model.py` — feature engineering + training (shared by the app and the trainer)
- `train_model.py` — re-downloads the data, retrains, and rewrites `metrics.json`
- `compare_models.py` — races Random Forest vs gradient boosting vs a persistence baseline on the same test split, and rewrites `model_comparison.json`

## Roadmap

- [x] Add humidity and pressure features from the hourly archive (Oct 2026 — small real gains, see table above)
- [x] Compare Random Forest against gradient boosting and a persistence baseline (Oct 2026 — Random Forest stays, by a small margin; see comparison table above)
- [ ] Try a simple LSTM for comparison
- [ ] Weekly forecast view, not just tomorrow
- [ ] Sindhi/Urdu language toggle for local users

---

*Part of my journey from Mithi to MIT — learning AI by building things my own hometown needs.*
