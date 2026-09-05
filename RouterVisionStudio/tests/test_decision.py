from __future__ import annotations

import unittest
from types import SimpleNamespace

from production_app import verdict_for
from router_vision.interlock import VisionVerdict


def result(status: str, *, passed: bool = True, note: str = ""):
    return SimpleNamespace(
        status=status,
        note=note,
        checked=12,
        run=SimpleNamespace(passed=passed, message="machine failure" if not passed else ""),
    )


class DecisionTests(unittest.TestCase):
    def test_good_is_pass(self):
        verdict, _reason = verdict_for(result("GOOD"))
        self.assertIs(verdict, VisionVerdict.PASS)

    def test_model_ng_is_ng(self):
        verdict, _reason = verdict_for(result("NG", note="position 8"))
        self.assertIs(verdict, VisionVerdict.NG)

    def test_missing_data_fails_closed(self):
        verdict, _reason = verdict_for(result("FAULT"))
        self.assertIs(verdict, VisionVerdict.FAULT)

    def test_router_failure_is_ng(self):
        verdict, reason = verdict_for(result("FAULT", passed=False))
        self.assertIs(verdict, VisionVerdict.NG)
        self.assertEqual(reason, "machine failure")


if __name__ == "__main__":
    unittest.main()
