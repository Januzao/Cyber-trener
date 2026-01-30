from __future__ import annotations

from typing import Dict, List, Optional

from .features import FrontMetrics, SideMetrics, ema
from .rules_common import Hint


class PlankRuleEngine:
    """Simple plank rules:
    - Body alignment (side): torso should be roughly straight (no sagging or hiking)
    - Head/neck neutral (proxy via torso angle)
    """

    def __init__(self, frames_required: int = 6) -> None:
        self.frames_required = frames_required
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
        conf_side = min(1.0, max(0.0, (side.avg_vis - 0.45) / 0.4))

        # For a good plank the torso-hip-ankle angle should be near 180 degrees (straight line)
        if self._torso_angle_s is not None:
            if self._count("plank_sag", self._torso_angle_s < 160.0):
                hints.append(
                    Hint(
                        ts_ms=side.ts_ms,
                        issue_code="PLANK_SAG",
                        message="Keep your body straight; avoid sagging at the hips.",
                        severity="warn",
                        view="side",
                        confidence=conf_side,
                    )
                )
                self._cnt["plank_sag"] = 0

        return hints
