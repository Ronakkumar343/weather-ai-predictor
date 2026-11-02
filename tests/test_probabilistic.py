"""Tests for probabilistic.py (split-conformal max-temp intervals).

Pure-function tests run on synthetic residuals only — no model, no
data files — so they stay fast. The one integration test runs the
real pipeline end to end on the bundled dataset.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import probabilistic


class ConformalQuantileTests(unittest.TestCase):
    def test_widening_with_level(self):
        residuals = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
        q80 = probabilistic.conformal_quantile(residuals, 0.8)
        q90 = probabilistic.conformal_quantile(residuals, 0.9)
        self.assertLessEqual(q80, q90)
        self.assertIn(q80, residuals)
        self.assertIn(q90, residuals)

    def test_higher_level_never_narrower(self):
        rng = np.random.default_rng(42)
        residuals = np.abs(rng.normal(0, 1.5, 200))
        prev = 0.0
        for level in (0.5, 0.7, 0.8, 0.9, 0.95):
            q = probabilistic.conformal_quantile(residuals, level)
            self.assertGreaterEqual(q, prev)
            prev = q

    def test_finite_sample_correction(self):
        # n = 4, level 0.8: ceil(5 * 0.8) / 4 = 1.0 -> max residual,
        # where a plain 80% quantile would return a smaller value.
        residuals = np.array([0.5, 1.0, 1.5, 2.0])
        self.assertEqual(probabilistic.conformal_quantile(residuals, 0.8), 2.0)

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            probabilistic.conformal_quantile([], 0.8)
        with self.assertRaises(ValueError):
            probabilistic.conformal_quantile([0.5, -1.0], 0.8)
        with self.assertRaises(ValueError):
            probabilistic.conformal_quantile([1.0], 1.0)
        with self.assertRaises(ValueError):
            probabilistic.conformal_quantile([1.0], 0.0)


class CoverageAndWidthTests(unittest.TestCase):
    def test_coverage_counts_boundaries(self):
        actual = np.array([10.0, 12.0, 14.0])
        lower, upper = probabilistic.make_intervals([12.0, 12.0, 12.0], 2.0)
        self.assertEqual(probabilistic.empirical_coverage(actual, lower, upper), 1.0)

    def test_coverage_partial(self):
        actual = np.array([10.0, 20.0])
        lower, upper = probabilistic.make_intervals([10.0, 10.0], 1.0)
        self.assertEqual(probabilistic.empirical_coverage(actual, lower, upper), 0.5)

    def test_mean_width_is_twice_half_width(self):
        lower, upper = probabilistic.make_intervals([30.0, 31.0], 1.5)
        self.assertAlmostEqual(probabilistic.mean_width(lower, upper), 3.0)

    def test_coverage_rejects_empty(self):
        with self.assertRaises(ValueError):
            probabilistic.empirical_coverage([], [], [])


class SplitTests(unittest.TestCase):
    def test_split_is_disjoint_and_chronological(self):
        cal, val = probabilistic.split_calibration_validation(731)
        self.assertEqual(len(cal), 365)
        self.assertEqual(len(val), 366)
        self.assertTrue(np.all(cal < val.min()))  # calibration strictly first
        self.assertEqual(len(set(cal.tolist()) & set(val.tolist())), 0)

    def test_split_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            probabilistic.split_calibration_validation(1)
        with self.assertRaises(ValueError):
            probabilistic.split_calibration_validation(100, 0.0)


class IntegrationTests(unittest.TestCase):
    def test_real_pipeline_no_leakage(self):
        import model
        df_feat = model.build_features(
            model.load_history(os.path.join(os.path.dirname(__file__),
                                            "..", "data")))
        results = probabilistic.evaluate_intervals(df_feat)
        # Calibration and validation slices are disjoint, in order,
        # and together cover the whole test period.
        self.assertEqual(results["test_rows"], 731)
        self.assertEqual(results["calibration_rows"]
                         + results["validation_rows"], 731)
        self.assertLess(results["calibration_dates"][1],
                        results["validation_dates"][0])
        # Wider target band must be at least as wide on validation.
        self.assertGreaterEqual(
            results["bands"]["90pct"]["validation_mean_width_c"],
            results["bands"]["80pct"]["validation_mean_width_c"])
        # Coverage is a real measured fraction, not an assumed one.
        for band in results["bands"].values():
            self.assertGreaterEqual(band["validation_coverage"], 0.0)
            self.assertLessEqual(band["validation_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
