from __future__ import annotations

from typing import Dict, List, Optional

from .features import FrontMetrics, SideMetrics, ema
from .rules_common import Hint


class LungeRuleEngine:
    """Simple lunge rules (front and side cues):
    - Knee tracking (avoid medial collapse)
    - Torso upright (avoid excessive forward lean)
    """

    def __init__(self, frames_required: int = 6) -> None:
        self.frames_required = frames_required
        self._knee_in_s: Optional[float] = None
        self._asym_s: Optional[float] = None
        self._torso_angle_s: Optional[float] = None
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

        self._torso_angle_s = ema(self._torso_angle_s, side.torso_hip_angle, alpha=0.35)
        if front.ok:
            self._knee_in_s = ema(self._knee_in_s, front.knee_inward_norm, alpha=0.35)
            self._asym_s = ema(self._asym_s, front.asymmetry_norm, alpha=0.35)

        conf_side = min(1.0, max(0.0, (side.avg_vis - 0.45) / 0.4))
        conf_front = min(1.0, max(0.0, (front.avg_vis - 0.45) / 0.4)) if front.ok else 0.0

        # Knee collapse
        if front.ok and self._knee_in_s is not None:
            if self._count("lunge_knees", self._knee_in_s > 0.18):
                hints.append(
                    Hint(
                        ts_ms=front.ts_ms,
                        issue_code="LU_KNEES",
                        message="Keep your front knee tracking over the toes.",
                        severity="info",
                        view="front",
                        confidence=conf_front,
                    )
                )
                self._cnt["lunge_knees"] = 0

        # Torso lean too far forward
        if self._torso_angle_s is not None:
            if self._count("lunge_lean", self._torso_angle_s < 95.0):
                hints.append(
                    Hint(
                        ts_ms=side.ts_ms,
                        issue_code="LU_LEAN",
                        message="Keep your torso more upright during the lunge.",
                        severity="info",
                        view="side",
                        confidence=conf_side,
                    )
                )
                self._cnt["lunge_lean"] = 0

        return hints
