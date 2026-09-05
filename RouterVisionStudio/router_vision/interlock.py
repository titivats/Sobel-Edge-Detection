"""Fail-safe conveyor interlock state machine.

The production UI uses this module in simulation mode.  A future PLC adapter
must preserve the same rule: absence of a complete, matching PASS result means
HOLD.  This module deliberately contains no hardware access.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GateState(str, Enum):
    OFFLINE = "OFFLINE-HOLD"
    READY = "READY-HOLD"
    INSPECTING = "INSPECTING-HOLD"
    PASS = "PASS-RELEASE"
    NG = "NG-HOLD"
    FAULT = "FAULT-HOLD"


class VisionVerdict(str, Enum):
    PASS = "PASS"
    NG = "NG"
    FAULT = "FAULT"


@dataclass(frozen=True)
class GateSignals:
    connected: bool
    healthy: bool
    heartbeat: bool
    result_valid: bool
    pass_signal: bool
    ng_signal: bool
    release: bool


class SimulatedConveyorLink:
    """In-memory PLC handshake used during development.

    RELEASE is derived rather than commanded directly.  This makes the safe
    state explicit: a crash, disconnect, mismatched panel or missing result can
    never leave RELEASE true.
    """

    def __init__(self) -> None:
        self.state = GateState.OFFLINE
        self.active_panel = ""
        self.reason = "simulation link not started"
        self._connected = False
        self._healthy = False
        self._heartbeat = False
        self._result_valid = False
        self._pass = False
        self._ng = False

    @property
    def signals(self) -> GateSignals:
        release = (
            self._connected
            and self._healthy
            and self._result_valid
            and self._pass
            and not self._ng
            and self.state is GateState.PASS
        )
        return GateSignals(
            connected=self._connected,
            healthy=self._healthy,
            heartbeat=self._heartbeat,
            result_valid=self._result_valid,
            pass_signal=self._pass,
            ng_signal=self._ng,
            release=release,
        )

    def connect(self) -> None:
        self._connected = True
        self._healthy = True
        self._clear_result()
        self.active_panel = ""
        self.state = GateState.READY
        self.reason = "simulation connected; conveyor held until PASS"

    def disconnect(self, reason: str = "link disconnected") -> None:
        self._connected = False
        self._healthy = False
        self._clear_result()
        self.state = GateState.OFFLINE
        self.reason = reason

    def toggle_heartbeat(self) -> None:
        if self._connected and self._healthy:
            self._heartbeat = not self._heartbeat
        else:
            self._heartbeat = False

    def begin_inspection(self, panel_id: str) -> None:
        if not self._connected or not self._healthy:
            self.fault("cannot inspect while conveyor link is unhealthy")
            return
        self._clear_result()
        self.active_panel = panel_id
        self.state = GateState.INSPECTING
        self.reason = f"waiting for complete vision result: {panel_id}"

    def publish(self, panel_id: str, verdict: VisionVerdict, reason: str) -> None:
        if not self._connected or not self._healthy:
            self.fault("result arrived while conveyor link was unhealthy")
            return
        if not panel_id or panel_id != self.active_panel:
            self.fault("panel identity mismatch; result rejected")
            return
        if self.state is not GateState.INSPECTING:
            self.fault("late result rejected because the gate is no longer inspecting")
            return

        if verdict is VisionVerdict.PASS:
            self._result_valid = True
            self._pass = True
            self._ng = False
            self.state = GateState.PASS
        elif verdict is VisionVerdict.NG:
            self._result_valid = True
            self._pass = False
            self._ng = True
            self.state = GateState.NG
        else:
            self.fault(reason or "vision result invalid")
            return
        self.reason = reason

    def acknowledge(self, panel_id: str | None = None) -> None:
        if panel_id and self.active_panel and panel_id != self.active_panel:
            self.fault("PLC acknowledgement did not match the active panel")
            return
        if not self._connected or not self._healthy:
            self.disconnect("cannot acknowledge while link is unhealthy")
            return
        self._clear_result()
        self.active_panel = ""
        self.state = GateState.READY
        self.reason = "result acknowledged; ready for next panel"

    def fault(self, reason: str) -> None:
        self._clear_result()
        self._ng = True
        self.state = GateState.FAULT
        self.reason = reason

    def _clear_result(self) -> None:
        self._result_valid = False
        self._pass = False
        self._ng = False
