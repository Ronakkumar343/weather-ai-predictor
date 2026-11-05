# 🌦️ Thar Weather AI — weather prediction for Mithi, Tharparkar

**Machine-learning weather forecasts for one of the hottest, driest places in Pakistan** — built by **Ronak Kumar**, a Class 9 student from Mithi who wanted a forecast that actually knows his hometown.

## Why this exists

Global weather apps barely know Tharparkar exists. Mithi's climate is extreme and specific: summers cross 45 °C, rain arrives in a few monsoon bursts, and a good local forecast matters — for farmers deciding when to sow, schools planning around heat, and families planning travel. So this project trains a model on **Mithi's own weather history** instead of borrowing a generic one.

## What it does

- **Tomorrow's maximum temperature** — regression model
- **Tomorrow's rain chance** (≥ 1 mm) — classification model
- **7-day outlook** — the same models run recursively (each day's prediction feeds the next day); useful shape of the week ahead, but only tomorrow's number is verified — see the honesty note below
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

### How sure is it? Prediction intervals (Nov 2026)

A single number hides its own uncertainty, so the max-temperature model
now also reports a range. Method: **split-conformal prediction**
(`probabilistic.py`, full numbers in `probabilistic_metrics.json`). The
same Random Forest and the same 2024-01-01 split as above; the 731-day
test period is split chronologically in half — the first 365 days
(2024-01-01 to 2024-12-30) calibrate the interval width from the
model's absolute errors, and the second 366 days (2024-12-31 to
2025-12-31) are the held-out validation slice. Calibration days are
never scored as validation days.

| Band | Half-width | Validation coverage (366 days) | Mean width |
|---|---|---|---|
| 80% band | ±1.69 °C | 85.8% | 3.38 °C |
| 90% band | ±2.42 °C | 93.4% | 4.83 °C |

Read honestly: both bands over-covered slightly on validation (85.8%
vs the 80% target, 93.4% vs 90%) — the intervals are a touch wider
than they need to be, which is the safe direction for a forecast.
The point-forecast accuracy is unchanged by this step (still 1.01 °C
mean error); intervals only describe its uncertainty. Limits: the
validation slice is a single year (366 days), the intervals cover
tomorrow only — no interval is claimed for days 2–7 of the outlook.

Rain probabilities were put through the same split (Nov 2026,
`probabilistic.py`, full numbers in `probabilistic_metrics.json`):
the Random Forest rain probability was recalibrated on the 365-day
calibration slice only — isotonic mapping first, Platt logistic
scaling second — and scored on the held-out 366 days (31 rainy days).
The honest result is **no gain**: neither method beat the raw
probability on validation, so the app keeps the raw probability.

| Rain probability | Validation Brier score (lower is better) |
|---|---|
| Raw Random Forest | **0.0481** |
| Isotonic recalibration | 0.0504 |
| Platt recalibration | 0.0517 |

Why it failed is visible in the calibration curves (also in
`probabilistic_metrics.json`): with only 34 rainy days in the
calibration slice, the isotonic map collapses to coarse steps
(including a 0.99 step for the highest forecasts), so it chases
calibration noise instead of a real miscalibration — and the raw
curve was already close to the diagonal where most forecasts live
(269 of 366 validation days fall in the 0.0–0.1 bin, mean forecast
0.018 vs 0.011 observed). Recalibration is therefore **not**
adopted; claiming a "calibrated" rain chance on these numbers
would be dressing up a worse forecast.

### The 7-day outlook (Oct 2026)

The app also shows a week ahead. It is **not** a second model and it makes no new accuracy claim: the same two Random Forests are run **recursively** — predict tomorrow, write that prediction into the weather table as if observed, predict the day after from it, and so on for 7 days, re-running the exact same feature pipeline (`model.build_features`) at every step. Columns the model does not predict are filled with simple, stated approximations: min temperature from the recent average day/night range, precipitation as the rain probability times the average rainy-day amount in the record, and wind/humidity/pressure held at their last observed values.

Read the week view honestly: **the verified numbers above (1.01 °C, 93.3%) apply to tomorrow only.** Each later day is built on earlier predictions, so errors compound and accuracy degrades day by day; no accuracy figure is claimed for days 2–7, in the app or here.

One correctness fix shipped with this feature: the one-day helper previously appended an all-NaN row for the target day, which `build_features`' `dropna()` then deleted — so the "tomorrow" prediction was actually computed from the feature row of the last *observed* day. It now builds the target day's own row (placeholder raw values that never enter the lag features), and the headline prediction and day 1 of the outlook are the same number by construction.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

- `app.py` — the Streamlit app (live prediction + recursive 7-day outlook use the Open-Meteo forecast API for the last 14 observed days; the climate tab works fully offline). Tomorrow's card shows the measured 80% temperature range and the raw rain chance, with the plain-language explanation of what the range means; no interval is shown for days 2–7
- `model.py` — feature engineering + training (shared by the app and the trainer)
- `train_model.py` — re-downloads the data, retrains, and rewrites `metrics.json`
- `compare_models.py` — races Random Forest vs gradient boosting vs a persistence baseline on the same test split, and rewrites `model_comparison.json`
- `probabilistic.py` — split-conformal max-temperature intervals and rain-probability calibration (`python probabilistic.py` rewrites `probabilistic_metrics.json`); `tests/test_probabilistic.py` — 11 tests for the interval math and the no-leakage split; `tests/test_rain_calibration.py` — 10 tests for Brier scoring, monotonic calibration maps, unfitted-calibrator errors, and the real-pipeline no-gain result (21 tests total)

## Roadmap

- [x] Add humidity and pressure features from the hourly archive (Oct 2026 — small real gains, see table above)
- [x] Compare Random Forest against gradient boosting and a persistence baseline (Oct 2026 — Random Forest stays, by a small margin; see comparison table above)
- [ ] Try a simple LSTM for comparison
- [x] Weekly forecast view, not just tomorrow (Oct 2026 — recursive multi-step outlook; the verified accuracy figures apply to tomorrow only, see "The 7-day outlook" above)
- [x] Measured max-temperature prediction intervals (Nov 2026 — split-conformal 80%/90% bands, calibrated on 2024, validated on 2025; see "How sure is it?" above)
- [x] Calibrated rain probabilities (Nov 2026 — tried isotonic + Platt on the calibration slice; neither improved the Brier score on the held-out slice, so the raw probability is kept and the no-gain result is reported above, not hidden)
- [x] Streamlit display of the range + raw rain chance (Nov 2026 — tomorrow's card shows the measured 80% temperature range with its plain-language explanation and the raw rain chance with the no-gain recalibration note; no interval for days 2–7)
- [ ] Sindhi/Urdu language toggle for local users

---

*Part of my journey from Mithi to MIT — learning AI by building things my own hometown needs.*
