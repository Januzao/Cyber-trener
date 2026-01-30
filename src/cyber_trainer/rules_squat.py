from __future__ import annotations

from typing import Dict, List, Optional

from .features import FrontMetrics, SideMetrics, ema
from .rules_common import Hint


class SquatRuleEngine:
    """Simple squat rules:
    - Back rounding (side torso angle too small)
    - Knee collapse/asymmetry (front view)
    """

    def __init__(self, frames_required: int = 6) -> None:
        self.frames_required = frames_required
        self._torso_angle_s: Optional[float] = None
        self._knee_in_s: Optional[float] = None
        self._asym_s: Optional[float] = None
        self._cnt: Dict[str, int] = {}

    def _count(self, key: str, cond: bool) -> bool:
        if cond:
            self._cnt[key] = self._cnt.get(key, 0) + 1
        else:
            self._cnt[key] = 0
        return self._cnt[key] >= self.frames_required

    def update(self, side: SideMetrics, front: FrontMetrics) -> List[Hint]:
        hints: List[Hint] = []
        if not side.ok:
            return hints

        # smooth
        self._torso_angle_s = ema(self._torso_angle_s, side.torso_hip_angle, alpha=0.35)
        if front.ok:
            self._knee_in_s = ema(self._knee_in_s, front.knee_inward_norm, alpha=0.35)
            self._asym_s = ema(self._asym_s, front.asymmetry_norm, alpha=0.35)

        conf_side = min(1.0, max(0.0, (side.avg_vis - 0.45) / 0.4))
        conf_front = min(1.0, max(0.0, (front.avg_vis - 0.45) / 0.4)) if front.ok else 0.0

        # Back rounding: torso angle too small
        if self._torso_angle_s is not None:
            if self._count("sq_back_round", self._torso_angle_s < 110.0):
                hints.append(
                    Hint(
                        ts_ms=side.ts_ms,
                        issue_code="SQ_BACK_NEUTRAL",
                        message="Keep your back neutral during squat.",
                        severity="warn",
                        view="side",
                        confidence=conf_side,
                    )
                )
                self._cnt["sq_back_round"] = 0

        # Knee collapse / asymmetry (front)
        if front.ok and self._knee_in_s is not None and self._asym_s is not None:
            cond = (self._knee_in_s > 0.20) or (self._asym_s > 0.15)
            if self._count("sq_knees", cond):
                hints.append(
                    Hint(
                        ts_ms=front.ts_ms,
                        issue_code="SQ_KNEES_COLLAPSE",
                        message="Keep knees tracking over toes; avoid collapse.",
                        severity="info",
                        view="front",
                        confidence=conf_front,
                    )
                )
                self._cnt["sq_knees"] = 0

        return hints
