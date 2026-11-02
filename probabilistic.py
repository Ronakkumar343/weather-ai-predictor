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
"""
import json

import numpy as np

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


def main(data_dir="data"):
    df_feat = model.build_features(model.load_history(data_dir))
    results = evaluate_intervals(df_feat)
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


if __name__ == "__main__":
    main()
