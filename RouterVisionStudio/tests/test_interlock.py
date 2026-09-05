from __future__ import annotations

import unittest

from router_vision.interlock import GateState, SimulatedConveyorLink, VisionVerdict


class InterlockTests(unittest.TestCase):
    def test_release_requires_complete_matching_pass(self):
        link = SimulatedConveyorLink()
        link.connect()
        self.assertFalse(link.signals.release)

        link.begin_inspection("PANEL-1")
        self.assertFalse(link.signals.release)
        link.publish("PANEL-1", VisionVerdict.PASS, "all positions passed")

        self.assertEqual(link.state, GateState.PASS)
        self.assertTrue(link.signals.result_valid)
        self.assertTrue(link.signals.release)

    def test_ng_never_releases(self):
        link = SimulatedConveyorLink()
        link.connect()
        link.begin_inspection("PANEL-2")
        link.publish("PANEL-2", VisionVerdict.NG, "position 4 failed")

        self.assertEqual(link.state, GateState.NG)
        self.assertTrue(link.signals.result_valid)
        self.assertTrue(link.signals.ng_signal)
        self.assertFalse(link.signals.release)

    def test_identity_mismatch_fails_closed(self):
        link = SimulatedConveyorLink()
        link.connect()
        link.begin_inspection("PANEL-3")
        link.publish("OTHER", VisionVerdict.PASS, "wrong panel")

        self.assertEqual(link.state, GateState.FAULT)
        self.assertFalse(link.signals.result_valid)
        self.assertFalse(link.signals.release)

    def test_disconnect_clears_release(self):
        link = SimulatedConveyorLink()
        link.connect()
        link.begin_inspection("PANEL-4")
        link.publish("PANEL-4", VisionVerdict.PASS, "pass")
        self.assertTrue(link.signals.release)

        link.disconnect()
        self.assertEqual(link.state, GateState.OFFLINE)
        self.assertFalse(link.signals.release)

    def test_operator_hold_rejects_late_pass(self):
        link = SimulatedConveyorLink()
        link.connect()
        link.begin_inspection("PANEL-5")
        link.fault("operator requested hold")
        link.publish("PANEL-5", VisionVerdict.PASS, "late result")

        self.assertEqual(link.state, GateState.FAULT)
        self.assertFalse(link.signals.result_valid)
        self.assertFalse(link.signals.release)


if __name__ == "__main__":
    unittest.main()
