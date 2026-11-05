"""Tests for the item-8c Streamlit display: tomorrow's temperature
range and the raw rain chance.

The range helpers are pure functions on the measured numbers in
probabilistic_metrics.json — the tests pin the honest wording (the
measured 85.8% coverage, not the 80% target; the raw rain chance,
not a 'calibrated' one) and the AppTest smoke test proves the app
still renders end to end with zero exceptions.
"""
import json
import unittest
from pathlib import Path

import app

ROOT = Path(__file__).parent.parent

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # streamlit not installed (e.g. a minimal CI env)
    AppTest = None


class TomorrowRangeTests(unittest.TestCase):
    def test_range_is_symmetric_and_width_is_twice_half_width(self):
        low, high = app.tomorrow_range(34.0, 1.69)
        self.assertAlmostEqual(low, 32.31)
        self.assertAlmostEqual(high, 35.69)
        self.assertAlmostEqual(high - low, 2 * 1.69)

    def test_range_explanation_uses_measured_coverage_not_target(self):
        text = app.range_explanation({"validation_coverage": 0.858})
        self.assertIn("85.8%", text)
        self.assertIn("8 in 10", text)
        self.assertIn("366", text)

    def test_rain_note_says_raw_and_reports_no_gain(self):
        prob = json.loads((ROOT / "probabilistic_metrics.json").read_text())
        note = app.rain_chance_note(prob)
        self.assertIn("raw", note)
        self.assertIn("did not improve", note)
        self.assertIn("0.0481", note)
        self.assertIn("0.0504", note)
        self.assertIn("0.0517", note)
        self.assertNotIn("calibrated chance", note)

    def test_loaded_metrics_match_the_measured_bands(self):
        prob = app.load_probabilistic_metrics()
        self.assertIsNotNone(prob)
        band = prob["bands"]["80pct"]
        self.assertEqual(band["half_width_c"], 1.69)
        self.assertEqual(band["validation_coverage"], 0.858)
        self.assertEqual(prob["rain_calibration"]["raw_brier"], 0.0481)


@unittest.skipIf(AppTest is None, "streamlit not installed")
class AppSmokeTests(unittest.TestCase):
    def test_app_renders_with_no_exceptions(self):
        at = AppTest.from_file(str(ROOT / "app.py"))
        # The app trains two Random Forests and fetches 14 days of
        # live data before its first render — far past AppTest's
        # 3-second default timeout.
        at.run(timeout=300)
        self.assertFalse(at.exception, f"app failed: {at.exception}")


if __name__ == "__main__":
    unittest.main()
