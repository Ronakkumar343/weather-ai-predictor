"""Tests for the rain-probability calibration in probabilistic.py.

Pure-function tests run on synthetic probabilities only — no model,
no data files — so they stay fast. The one integration test runs the
real pipeline end to end on the bundled dataset and checks the
no-leakage split plus the honest improved/no-gain bookkeeping (it
asserts the flag logic, not that calibration must improve — on this
dataset it does not).
"""
import os
import sys
import unittest

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import probabilistic


class BrierScoreTests(unittest.TestCase):
    def test_perfect_and_constant(self):
        self.assertEqual(probabilistic.brier_score([0, 1], [0.0, 1.0]), 0.0)
        self.assertEqual(probabilistic.brier_score([0, 1], [0.5, 0.5]), 0.25)

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            probabilistic.brier_score([], [])
        with self.assertRaises(ValueError):
            probabilistic.brier_score([0, 1], [0.5])
        with self.assertRaises(ValueError):
            probabilistic.brier_score([0, 1], [0.5, 1.2])
        with self.assertRaises(ValueError):
            probabilistic.brier_score([0, 2], [0.5, 0.5])


class IsotonicTests(unittest.TestCase):
    def test_mapping_is_monotonic(self):
        probs = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9])
        actual = np.array([0, 0, 1, 0, 1, 1, 1], dtype=float)
        calibrator = probabilistic.fit_isotonic_calibrator(probs, actual)
        mapped = probabilistic.apply_calibrator(calibrator, np.linspace(0, 1, 21))
        self.assertTrue(np.all(np.diff(mapped) >= 0))
        self.assertTrue(np.all((mapped >= 0) & (mapped <= 1)))

    def test_unfitted_raises(self):
        with self.assertRaises(NotFittedError):
            probabilistic.apply_calibrator(IsotonicRegression(), [0.3, 0.7])

    def test_fit_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            probabilistic.fit_isotonic_calibrator([], [])
        with self.assertRaises(ValueError):
            probabilistic.fit_isotonic_calibrator([0.2, 1.5], [0, 1])


class PlattTests(unittest.TestCase):
    def test_mapping_is_monotonic_on_increasing_data(self):
        probs = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9])
        actual = np.array([0, 0, 0, 1, 1, 1, 1], dtype=float)
        calibrator = probabilistic.fit_platt_calibrator(probs, actual)
        mapped = probabilistic.apply_calibrator(calibrator, np.linspace(0, 1, 21))
        self.assertTrue(np.all(np.diff(mapped) >= -1e-9))
        self.assertTrue(np.all((mapped >= 0) & (mapped <= 1)))

    def test_unfitted_raises(self):
        with self.assertRaises(NotFittedError):
            probabilistic.apply_calibrator(LogisticRegression(), [0.3, 0.7])

    def test_unknown_calibrator_type_raises(self):
        with self.assertRaises(TypeError):
            probabilistic.apply_calibrator(object(), [0.3])


class CalibrationCurveTests(unittest.TestCase):
    def test_counts_sum_and_rates_in_range(self):
        actual = np.array([0, 1, 0, 1, 1], dtype=float)
        probs = np.array([0.05, 0.15, 0.85, 0.95, 0.45])
        curve = probabilistic.calibration_curve(actual, probs)
        self.assertEqual(len(curve), 10)
        self.assertEqual(sum(bin_["count"] for bin_ in curve), 5)
        for bin_ in curve:
            self.assertGreaterEqual(bin_["observed_rate"], 0.0)
            self.assertLessEqual(bin_["observed_rate"], 1.0)
        # The 0.0-0.1 bin holds the 0.05 forecast (no rain) and the
        # 0.9-1.0 bin holds the 0.95 forecast (rain).
        self.assertEqual(curve[0]["count"], 1)
        self.assertEqual(curve[0]["observed_rate"], 0.0)
        self.assertEqual(curve[9]["count"], 1)
        self.assertEqual(curve[9]["observed_rate"], 1.0)


class IntegrationTests(unittest.TestCase):
    def test_real_pipeline_no_leakage_and_honest_flags(self):
        import model
        df_feat = model.build_features(
            model.load_history(os.path.join(os.path.dirname(__file__),
                                            "..", "data")))
        rain = probabilistic.evaluate_rain_calibration(df_feat)
        self.assertEqual(rain["calibration_rows"], 365)
        self.assertEqual(rain["validation_rows"], 366)
        self.assertEqual(rain["calibration_rows"] + rain["validation_rows"], 731)
        self.assertLess(rain["calibration_dates"][1],
                        rain["validation_dates"][0])
        for key in ("raw_brier", "isotonic_brier", "platt_brier"):
            self.assertGreaterEqual(rain[key], 0.0)
            self.assertLessEqual(rain[key], 1.0)
        # Flag logic, not a hoped-for outcome: the flag must match
        # the actual Brier comparison, whichever way it goes.
        self.assertEqual(rain["improved_isotonic"],
                         rain["isotonic_brier"] < rain["raw_brier"])
        self.assertEqual(rain["improved_platt"],
                         rain["platt_brier"] < rain["raw_brier"])
        for curve_key in ("calibration_curve_raw",
                          "calibration_curve_isotonic",
                          "calibration_curve_platt"):
            curve = rain[curve_key]
            self.assertEqual(len(curve), 10)
            self.assertEqual(sum(bin_["count"] for bin_ in curve), 366)


if __name__ == "__main__":
    unittest.main()
