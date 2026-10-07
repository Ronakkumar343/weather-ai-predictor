"""Compare models for Thar Weather AI on the SAME honest test split.

Queue item 3b: Random Forest vs Gradient Boosting vs a persistence
baseline, for both predictions the project makes:

  1. Tomorrow's maximum temperature (regression, scored by MAE in deg C)
  2. Whether it will rain tomorrow, >= 1.0 mm (classification, accuracy)

Rules, so the comparison is fair and means what it says:
  - Same features for both learned models: model.FEATURES (17 features,
    only information known before the target day).
  - Same time-based split as model.train(): train 2015-2023,
    test 2024-2025 (731 days the learned models never trained on).
  - Persistence needs no training: tomorrow's max temperature is
    predicted as yesterday's max temperature (tmax_lag1), and rain
    tomorrow is predicted as rain yesterday (precip_lag1 >= 1.0 mm).
  - Gradient Boosting is scikit-learn's GradientBoostingRegressor /
    GradientBoostingClassifier with default settings except
    random_state=42. Random Forest matches model.py (300 trees,
    random_state=42). Nothing is tuned on the test set.

Usage:  python compare_models.py
Writes: model_comparison.json (and prints the table the README uses)
"""
import json

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import accuracy_score, mean_absolute_error

import model

SPLIT_DATE = "2024-01-01"


def main():
    feat = model.build_features(model.load_history())
    train = feat[feat["date"] < SPLIT_DATE]
    test = feat[feat["date"] >= SPLIT_DATE]

    # --- Tomorrow's maximum temperature (regression) ---
    tmax_pred = {
        "persistence": test["tmax_lag1"],
        "random_forest": RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1
        ).fit(train[model.FEATURES], train["target_tmax"]).predict(test[model.FEATURES]),
        "gradient_boosting": GradientBoostingRegressor(
            random_state=42
        ).fit(train[model.FEATURES], train["target_tmax"]).predict(test[model.FEATURES]),
    }
    # 3 decimals for MAE here (metrics.json uses 2): at 2 decimals all
    # three models round to 1.01-1.02 C and the real ordering is hidden.
    tmax_mae = {
        name: round(float(mean_absolute_error(test["target_tmax"], pred)), 3)
        for name, pred in tmax_pred.items()
    }

    # --- Rain tomorrow (classification) ---
    rain_pred = {
        "persistence": (test["precip_lag1"] >= model.RAIN_MM).astype(int),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1
        ).fit(train[model.FEATURES], train["target_rain"]).predict(test[model.FEATURES]),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=42
        ).fit(train[model.FEATURES], train["target_rain"]).predict(test[model.FEATURES]),
    }
    rain_acc = {
        name: round(float(accuracy_score(test["target_rain"], pred)), 3)
        for name, pred in rain_pred.items()
    }

    results = {
        "split_date": SPLIT_DATE,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "features": len(model.FEATURES),
        "tmax_mae_c": tmax_mae,
        "rain_accuracy": rain_acc,
        "rain_base_rate_test": round(float(test["target_rain"].mean()), 3),
    }
    with open("model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Train {len(train)} rows (< {SPLIT_DATE}), test {len(test)} rows, "
          f"{len(model.FEATURES)} features")
    print(f"{'Model':<20} {'Tmax MAE (C)':>12} {'Rain accuracy':>13}")
    for name in ["persistence", "random_forest", "gradient_boosting"]:
        print(f"{name:<20} {tmax_mae[name]:>12.3f} {rain_acc[name]:>13.1%}")
    print(f"Rain falls on {results['rain_base_rate_test']:.1%} of test days, "
          f"so always predicting 'no rain' scores "
          f"{1 - results['rain_base_rate_test']:.1%}.")


if __name__ == "__main__":
    main()
