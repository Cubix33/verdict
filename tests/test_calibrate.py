"""Regression tests for verdict.calibrate - the tier logic and serialisation.

These exist because both behaviours were shipped wrong once:
- thresholds serialised as bare `Infinity`, which Python's json accepts and
  every strict JSON parser rejects (issue #2);
- a single per-document-maximum threshold applied per region, which made
  forged regions structurally undetectable (issue #2, recall section).
"""
import json

import pytest

from verdict.calibrate import Calibration


def _cal(**thresholds) -> Calibration:
    return Calibration(customer_id="test", n_samples=10, thresholds=thresholds)


class TestTiers:
    def test_screen_and_strong_are_distinct_bounds(self):
        cal = _cal(ela_z={"hi": 20.0, "lo": float("-inf"),
                          "screen_hi": 5.0, "screen_lo": float("-inf"),
                          "scale": 2.0})
        # Between the tiers: a lead, not damning.
        assert cal.exceeds("ela_z", 7.0, tier="screen")
        assert not cal.exceeds("ela_z", 7.0, tier="strong")
        # Past both.
        assert cal.exceeds("ela_z", 25.0, tier="screen")
        assert cal.exceeds("ela_z", 25.0, tier="strong")
        # Under both.
        assert not cal.exceeds("ela_z", 2.0, tier="screen")

    def test_both_sided_metric(self):
        cal = _cal(hf_ratio={"hi": 4.0, "lo": 0.05,
                             "screen_hi": 2.5, "screen_lo": 0.2, "scale": 1.0})
        assert cal.exceeds("hf_ratio", 0.1, tier="screen")      # too smooth
        assert not cal.exceeds("hf_ratio", 0.1, tier="strong")
        assert cal.exceeds("hf_ratio", 0.01, tier="strong")

    def test_missing_metric_never_exceeds(self):
        cal = _cal()
        assert not cal.exceeds("ela_z", 999.0)
        assert cal.severity("ela_z", 999.0) == 0.0

    def test_severity_is_tier_relative(self):
        cal = _cal(ela_z={"hi": 20.0, "lo": float("-inf"),
                          "screen_hi": 5.0, "screen_lo": float("-inf"),
                          "scale": 2.0})
        assert cal.severity("ela_z", 9.0, tier="screen") == pytest.approx(2.0)
        assert cal.severity("ela_z", 9.0, tier="strong") == 0.0


class TestSerialisation:
    def test_no_bare_infinity_in_json(self):
        cal = _cal(grid_ratio={"hi": float("inf"), "lo": 0.2,
                               "screen_hi": float("inf"), "screen_lo": 0.4,
                               "scale": 0.1})
        text = json.dumps(cal.to_dict())
        assert "Infinity" not in text          # strict-JSON safety

    def test_round_trip_preserves_one_sided_bounds(self):
        cal = _cal(grid_ratio={"hi": float("inf"), "lo": 0.2,
                               "screen_hi": float("inf"), "screen_lo": 0.4,
                               "scale": 0.1})
        back = Calibration.from_dict(json.loads(json.dumps(cal.to_dict())))
        assert back.thresholds["grid_ratio"]["hi"] == float("inf")
        assert back.thresholds["grid_ratio"]["lo"] == 0.2
        assert back.exceeds("grid_ratio", 0.3, tier="screen")   # side=low
        assert not back.exceeds("grid_ratio", 0.3, tier="strong")

    def test_pre_screen_tier_files_fall_back_to_strong(self):
        """A calibration file written before the screen tier existed must load
        and behave exactly as it did then - conservative, never crashing."""
        old = {"customer_id": "legacy",
               "thresholds": {"ela_z": {"hi": 20.0, "lo": None, "scale": 5.0}}}
        cal = Calibration.from_dict(old)
        assert cal.exceeds("ela_z", 21.0, tier="screen")
        assert not cal.exceeds("ela_z", 10.0, tier="screen")
