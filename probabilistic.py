"""Thar Weather AI — probabilistic outlook: split-conformal intervals.

A point forecast ("tomorrow's max will be 34 °C") hides how sure the
model is. This module turns the existing Random Forest max-temperature
model into honest prediction *intervals*: "tomorrow's max will be
between 32 °C and 36 °C".

Method (split conformal prediction, no new model, no new data):
  1. Train the same Random Forest as model.train on data before
     2024-01-01 (3,280 rows) and predict the held-out test period
     (2024-01-01 onward, 731 days).
  2. Split that test period chronologically in half. The FIRST half
     is the calibration slice: the interval half-width is the
     finite-sample-corrected quantile of the absolute residuals
     |actual - predicted| on calibration days only.
  3. Measure on the SECOND half (the validation slice) only: the
     fraction of validation days whose real max fell inside the
     interval (empirical coverage) and the mean interval width.
     Calibration days are never scored as validation days.

So a reported "80% band" means: on the validation slice, about 80%
of real maxima landed inside the band — measured, not assumed.
Point-forecast accuracy (MAE 1.01 °C) is unchanged by this step;
intervals only describe its uncertainty.

Rain probabilities get the same treatment, with a different honest
outcome possible: the Random Forest rain probability is recalibrated
on the calibration slice only (isotonic mapping, and Platt scaling
as a second method), then scored on the held-out validation slice.
If neither method lowers the Brier score there, the result is
reported as no-gain and the raw probability is kept — calibration
is not adopted on the strength of the calibration slice alone.
"""
import json

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

import model

CALIBRATION_FRACTION = 0.5  # first half of the test period calibrates
LEVELS = (0.8, 0.9)  # the bands this project reports


def conformal_quantile(abs_residuals, level):
    """Finite-sample split-conformal half-width for a target coverage.

    For n calibration residuals and target coverage `level`, the
    half-width is the ceil((n + 1) * level) / n quantile of the
    absolute residuals, taken with method="higher" so the result is
    always one of the observed residuals. The (n + 1) correction is
    what gives the finite-sample coverage guarantee; without it a
    plain quantile systematically under-covers on small samples.
    """
    residuals = np.asarray(abs_residuals, dtype=float)
    if residuals.size == 0:
        raise ValueError("conformal_quantile needs at least one residual")
    if np.any(residuals < 0):
        raise ValueError("residuals must be absolute values (>= 0)")
    if not 0 < level < 1:
        raise ValueError("level must be strictly between 0 and 1")
    n = residuals.size
    rank = int(np.ceil((n + 1) * level))
    q_level = min(1.0, rank / n)
    return float(np.quantile(residuals, q_level, method="higher"))


def make_intervals(predictions, half_width):
    """Symmetric intervals: prediction +/- half_width."""
    preds = np.asarray(predictions, dtype=float)
    return preds - half_width, preds + half_width


def empirical_coverage(actual, lower, upper):
    """Fraction of actual values inside [lower, upper] (inclusive)."""
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if actual.size == 0:
        raise ValueError("empirical_coverage needs at least one value")
    return float(np.mean((actual >= lower) & (actual <= upper)))


def mean_width(lower, upper):
    """Mean interval width in the same units as the predictions."""
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))


def split_calibration_validation(n, calibration_fraction=CALIBRATION_FRACTION):
    """Chronological split of n test rows: (calibration, validation).

    Returns index arrays. The split is chronological (earliest rows
    calibrate) so a later validation day can never leak into the
    calibration set, and the two sets are disjoint by construction.
    """
    if n < 2:
        raise ValueError("need at least 2 rows to split calibration/validation")
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be strictly between 0 and 1")
    n_cal = int(n * calibration_fraction)
    return np.arange(n_cal), np.arange(n_cal, n)


def evaluate_intervals(df_feat, split_date="2024-01-01", levels=LEVELS):
    """Train once, calibrate on the first test half, score the second.

    Returns a JSON-serialisable dict with, per level: the calibration
    half-width, and the validation coverage + mean width + sample size.
    The calibration date range is reported so a reader can check the
    two slices do not overlap.
    """
    train = df_feat[df_feat["date"] < split_date]
    test = df_feat[df_feat["date"] >= split_date].reset_index(drop=True)
    reg, _, _ = model.train(df_feat, split_date=split_date)
    # reg is fitted on the pre-split training rows only; predict the
    # held-out test period, then split it into calibration/validation.
    pred = reg.predict(test[model.FEATURES])
    actual = test["target_tmax"].to_numpy(dtype=float)
    cal_idx, val_idx = split_calibration_validation(len(test))
    cal_residuals = np.abs(actual[cal_idx] - pred[cal_idx])
    bands = {}
    for level in levels:
        half_width = conformal_quantile(cal_residuals, level)
        lower, upper = make_intervals(pred[val_idx], half_width)
        bands[f"{int(round(level * 100))}pct"] = {
            "target_coverage": level,
            "half_width_c": round(half_width, 2),
            "validation_coverage": round(
                empirical_coverage(actual[val_idx], lower, upper), 3),
            "validation_mean_width_c": round(mean_width(lower, upper), 2),
        }
    return {
        "method": "split-conformal, symmetric intervals from absolute residuals",
        "model": "RandomForestRegressor (n_estimators=300, random_state=42), "
                 "same 17 features and 2024-01-01 split as metrics.json",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "calibration_rows": int(len(cal_idx)),
        "validation_rows": int(len(val_idx)),
        "calibration_dates": [str(test["date"].iloc[cal_idx[0]].date()),
                              str(test["date"].iloc[cal_idx[-1]].date())],
        "validation_dates": [str(test["date"].iloc[val_idx[0]].date()),
                             str(test["date"].iloc[val_idx[-1]].date())],
        "bands": bands,
    }


def _validate_probs_actual(probs, actual):
    """Shared checks for calibration helpers: matching, non-empty,
    probabilities in [0, 1], labels 0/1."""
    probs = np.asarray(probs, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if probs.size == 0 or actual.size == 0:
        raise ValueError("calibration needs at least one sample")
    if probs.shape != actual.shape:
        raise ValueError("probabilities and labels must have the same length")
    if np.any(probs < 0) or np.any(probs > 1):
        raise ValueError("probabilities must lie in [0, 1]")
    if not np.all((actual == 0) | (actual == 1)):
        raise ValueError("labels must be 0 or 1")
    return probs, actual


def brier_score(actual, probs):
    """Mean squared error between forecast probabilities and 0/1
    outcomes. 0 is perfect; a constant 0.5 forecast scores 0.25
    regardless of the outcomes."""
    probs, actual = _validate_probs_actual(probs, actual)
    return float(np.mean((probs - actual) ** 2))


def fit_isotonic_calibrator(cal_probs, cal_actual):
    """Isotonic (monotone, piecewise-constant) recalibration map.

    Fitted on the calibration slice only. The fitted map is
    non-decreasing by construction: a higher raw probability can
    never map to a lower calibrated probability. With only ~34 rainy
    calibration days this map is coarse — that is exactly what the
    validation Brier score then exposes.
    """
    probs, actual = _validate_probs_actual(cal_probs, cal_actual)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(probs, actual)
    return calibrator


def fit_platt_calibrator(cal_probs, cal_actual):
    """Platt scaling: a logistic regression on the raw probability.

    The canonical alternative to isotonic calibration, also fitted
    on the calibration slice only. One smooth sigmoid instead of
    isotonic's steps, so it cannot chase single calibration days.
    """
    probs, actual = _validate_probs_actual(cal_probs, cal_actual)
    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(probs.reshape(-1, 1), actual)
    return calibrator


def apply_calibrator(calibrator, probs):
    """Apply a fitted isotonic or Platt calibrator, clipped to [0, 1].

    An unfitted calibrator raises sklearn's NotFittedError here —
    checked explicitly, because IsotonicRegression.predict on its
    own raises a bare AttributeError when unfitted. Callers must
    fit on calibration data first, never on the slice being scored.
    """
    from sklearn.utils.validation import check_is_fitted
    check_is_fitted(calibrator)
    probs = np.asarray(probs, dtype=float)
    if probs.size == 0:
        raise ValueError("apply_calibrator needs at least one probability")
    if np.any(probs < 0) or np.any(probs > 1):
        raise ValueError("probabilities must lie in [0, 1]")
    if isinstance(calibrator, IsotonicRegression):
        calibrated = calibrator.predict(probs)
    elif isinstance(calibrator, LogisticRegression):
        calibrated = calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]
    else:
        raise TypeError("calibrator must be isotonic or Platt (logistic)")
    return np.clip(calibrated, 0.0, 1.0)


def calibration_curve(actual, probs, n_bins=10):
    """Binned calibration curve: per 0.1-wide probability bin, the
    count, the mean forecast probability and the observed rain rate.

    Empty bins are kept (count 0, rates 0.0) so the curve always has
    n_bins entries and a reader can see where the model never
    forecasts. Bins are [lo, hi) except the last, which is [0.9, 1.0].
    """
    probs, actual = _validate_probs_actual(probs, actual)
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    curve = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        if b == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        count = int(mask.sum())
        curve.append({
            "bin": [round(lo, 2), round(hi, 2)],
            "count": count,
            "mean_predicted": round(float(probs[mask].mean()), 3) if count else 0.0,
            "observed_rate": round(float(actual[mask].mean()), 3) if count else 0.0,
        })
    return curve


def evaluate_rain_calibration(df_feat, split_date="2024-01-01"):
    """Recalibrate rain probabilities on calibration, score on validation.

    Same Random Forest, same 2024-01-01 split and same chronological
    calibration/validation slices as evaluate_intervals: the
    calibrators see calibration days only, and every Brier score and
    curve below is measured on the held-out validation slice. The
    `improved_*` flags are computed, not assumed: a method counts as
    an improvement only if its validation Brier score is strictly
    below the raw model's.
    """
    train = df_feat[df_feat["date"] < split_date]
    test = df_feat[df_feat["date"] >= split_date].reset_index(drop=True)
    _, clf, _ = model.train(df_feat, split_date=split_date)
    rain_col = list(clf.classes_).index(1)
    probs = clf.predict_proba(test[model.FEATURES])[:, rain_col]
    actual = test["target_rain"].to_numpy(dtype=float)
    cal_idx, val_idx = split_calibration_validation(len(test))
    isotonic = fit_isotonic_calibrator(probs[cal_idx], actual[cal_idx])
    platt = fit_platt_calibrator(probs[cal_idx], actual[cal_idx])
    iso_val = apply_calibrator(isotonic, probs[val_idx])
    platt_val = apply_calibrator(platt, probs[val_idx])
    raw_brier = brier_score(actual[val_idx], probs[val_idx])
    iso_brier = brier_score(actual[val_idx], iso_val)
    platt_brier = brier_score(actual[val_idx], platt_val)
    return {
        "method": "isotonic (primary) and Platt logistic scaling, "
                  "both fitted on the calibration slice only",
        "model": "RandomForestClassifier (n_estimators=300, random_state=42), "
                 "same 17 features and 2024-01-01 split as metrics.json",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "calibration_rows": int(len(cal_idx)),
        "validation_rows": int(len(val_idx)),
        "calibration_dates": [str(test["date"].iloc[cal_idx[0]].date()),
                              str(test["date"].iloc[cal_idx[-1]].date())],
        "validation_dates": [str(test["date"].iloc[val_idx[0]].date()),
                             str(test["date"].iloc[val_idx[-1]].date())],
        "calibration_rain_days": int(actual[cal_idx].sum()),
        "validation_rain_days": int(actual[val_idx].sum()),
        "raw_brier": round(raw_brier, 4),
        "isotonic_brier": round(iso_brier, 4),
        "platt_brier": round(platt_brier, 4),
        "improved_isotonic": bool(iso_brier < raw_brier),
        "improved_platt": bool(platt_brier < raw_brier),
        "calibration_curve_raw": calibration_curve(actual[val_idx], probs[val_idx]),
        "calibration_curve_isotonic": calibration_curve(actual[val_idx], iso_val),
        "calibration_curve_platt": calibration_curve(actual[val_idx], platt_val),
    }


def main(data_dir="data"):
    df_feat = model.build_features(model.load_history(data_dir))
    results = evaluate_intervals(df_feat)
    results["rain_calibration"] = evaluate_rain_calibration(df_feat)
    with open("probabilistic_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Split-conformal max-temperature intervals "
          f"(calibration {results['calibration_rows']} days "
          f"{results['calibration_dates'][0]}..{results['calibration_dates'][1]}, "
          f"validation {results['validation_rows']} days "
          f"{results['validation_dates'][0]}..{results['validation_dates'][1]}):")
    for name, band in results["bands"].items():
        print(f"  {name} band: +/-{band['half_width_c']} °C, "
              f"validation coverage {band['validation_coverage']:.1%}, "
              f"mean width {band['validation_mean_width_c']} °C")
    rain = results["rain_calibration"]
    print("Rain-probability calibration "
          f"(calibration {rain['calibration_rows']} days, "
          f"validation {rain['validation_rows']} days, "
          f"{rain['validation_rain_days']} rainy validation days):")
    print(f"  raw Brier {rain['raw_brier']}, "
          f"isotonic Brier {rain['isotonic_brier']} "
          f"(improved: {rain['improved_isotonic']}), "
          f"Platt Brier {rain['platt_brier']} "
          f"(improved: {rain['improved_platt']})")


if __name__ == "__main__":
    main()
