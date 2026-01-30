from __future__ import annotations

from typing import Dict, List, Optional

from .features import FrontMetrics, SideMetrics, ema
from .rules_common import Hint


class PushupRuleEngine:
    """Simple push-up rules:
    - Hips sagging or hips too high (side view)
    - Elbow flare is not checked (no direct landmark), so we use simple proxies
    """

    def __init__(self, frames_required: int = 6) -> None:
        self.frames_required = frames_required
        self._hip_sh_dy_s: Optional[float] = None
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

        # normalized vertical separation between hip and shoulder
        hip_sh_dy = abs(side.hip_y - side.shoulder_y) / max(1.0, side.torso_len)
        self._hip_sh_dy_s = ema(self._hip_sh_dy_s, hip_sh_dy, alpha=0.35)

        conf_side = min(1.0, max(0.0, (side.avg_vis - 0.45) / 0.4))

        if self._hip_sh_dy_s is not None:
            # hips sagging (hip lower -> larger dy)
            if self._count("push_sag", self._hip_sh_dy_s > 0.38):
                hints.append(
                    Hint(
                        ts_ms=side.ts_ms,
                        issue_code="PU_SAG",
                        message="Hips are sagging — tighten your core.",
                        severity="warn",
                        view="side",
                        confidence=conf_side,
                    )
                )
                self._cnt["push_sag"] = 0

            # hips too high (too straight, not engaging core)
            if self._count("push_high", self._hip_sh_dy_s < 0.18):
                hints.append(
                    Hint(
                        ts_ms=side.ts_ms,
                        issue_code="PU_HIPS_HIGH",
                        message="Lower your hips slightly to keep a straight plank line.",
                        severity="info",
                        view="side",
                        confidence=conf_side,
                    )
                )
                self._cnt["push_high"] = 0

        return hints
