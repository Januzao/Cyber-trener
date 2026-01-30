from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Hint:
    ts_ms: int
    issue_code: str
    message: str
    severity: str          # "info" | "warn" | "critical"
    view: str              # "front" | "side"
    confidence: float

