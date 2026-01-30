from __future__ import annotations

from typing import Dict, List, Optional

from .features import FrontMetrics, SideMetrics, ema
from .rules_common import Hint


class BicepsRuleEngine:
    """Biceps curl (standing, side view preferred) rules.

    We do NOT try to perfectly count reps here; we only generate form hints and a
    'correct/incorrect' signal.

    Signals used (side view):
    - elbow_angle (shoulder-elbow-wrist): flexion/extension.
    - wrist_y, elbow_y: to detect elbow swinging forward/back via wrist_x drift proxy is not available,
      so we use elbow_y instability as a weak indicator + angle speed.
    - We also use a simple motion gate outside this engine (in server) to decide when to score.
    """

    def __init__(self, frames_required: int = 6) -> None:
        self.frames_required = frames_required

        self._elbow_angle_s: Optional[float] = None
        self._elbow_y_s: Optional[float] = None

        self._cnt: Dict[str, int] = {}

        # For speed / momentum checks
        self._prev_ts_ms: Optional[int] = None
        self._prev_angle: Optional[float] = None

        # Dynamic ROM tracking (per set)
        self._angle_min: Optional[float] = None
        self._angle_max: Optional[float] = None

    def reset(self) -> None:
        self._elbow_angle_s = None
        self._elbow_y_s = None
        self._cnt = {}
        self._prev_ts_ms = None
        self._prev_angle = None
        self._angle_min = None
        self._angle_max = None

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

        # Smooth the elbow angle and elbow vertical position
        self._elbow_angle_s = ema(self._elbow_angle_s, side.elbow_angle, alpha=0.35)
        self._elbow_y_s = ema(self._elbow_y_s, side.elbow_y, alpha=0.35)

        conf_side = min(1.0, max(0.0, (side.avg_vis - 0.45) / 0.4))

        # Track ROM extremes to give ROM hints (after you start moving)
        if self._elbow_angle_s is not None and self._elbow_angle_s > 1e-3:
            if self._angle_min is None:
                self._angle_min = self._elbow_angle_s
                self._angle_max = self._elbow_angle_s
            else:
                self._angle_min = min(self._angle_min, self._elbow_angle_s)
                self._angle_max = max(self._angle_max, self._elbow_angle_s)

        # --- Rule: too fast / momentum (warn) ---
        # If elbow angle changes extremely fast, it's likely swinging.
        ang_speed = 0.0
        if self._prev_ts_ms is not None and self._prev_angle is not None and self._elbow_angle_s is not None:
            dt = max(1e-3, (side.ts_ms - self._prev_ts_ms) / 1000.0)
            ang_speed = abs(self._elbow_angle_s - self._prev_angle) / dt  # deg/s

        if self._count("bi_fast", ang_speed > 280.0):
            hints.append(
                Hint(
                    ts_ms=side.ts_ms,
                    issue_code="BI_FAST",
                    message="Slow down — avoid using momentum on curls.",
                    severity="warn",
                    view="side",
                    confidence=conf_side,
                )
            )
            self._cnt["bi_fast"] = 0

        # --- Rule: elbow drifting (warn) ---
        # With only 2D, we approximate elbow stability by how much elbow_y jitter is present.
        # If elbow_y changes too much relative to torso_len, likely swinging/cheating.
        # We use a simple instant delta via previous smoothed elbow_y.
        # (This is intentionally conservative to avoid false positives.)
        if self._prev_ts_ms is not None and self._elbow_y_s is not None:
            # Use shoulder_y as reference: if elbow_y departs a lot from its typical relation to shoulder.
            # Acceptable range ~0.12 torso_len.
            rel = abs(self._elbow_y_s - side.shoulder_y) / max(1.0, side.torso_len)
            if self._count("bi_elbow_swing", rel > 0.42):
                hints.append(
                    Hint(
                        ts_ms=side.ts_ms,
                        issue_code="BI_SWING",
                        message="Keep your elbows stable — don’t swing them during curls.",
                        severity="warn",
                        view="side",
                        confidence=conf_side,
                    )
                )
                self._cnt["bi_elbow_swing"] = 0

        # --- Rule: limited ROM top/bottom (info) ---
        # After we observed some movement (range), warn if range is too small.
        if self._angle_min is not None and self._angle_max is not None:
            rom = self._angle_max - self._angle_min
            # Typical curl ROM for elbow flexion is big; in 2D we accept smaller.
            if rom > 25.0:
                # If you never get "top" flexion (small angle), hint
                if self._count("bi_top_rom", self._angle_min > 70.0):
                    hints.append(
                        Hint(
                            ts_ms=side.ts_ms,
                            issue_code="BI_TOP_ROM",
                            message="Curl higher — aim for full elbow flexion at the top.",
                            severity="info",
                            view="side",
                            confidence=conf_side,
                        )
                    )
                    self._cnt["bi_top_rom"] = 0

                # If you never extend down (large angle), hint
                if self._count("bi_bottom_rom", self._angle_max < 145.0):
                    hints.append(
                        Hint(
                            ts_ms=side.ts_ms,
                            issue_code="BI_BOTTOM_ROM",
                            message="Lower the weight more — get closer to full extension at the bottom.",
                            severity="info",
                            view="side",
                            confidence=conf_side,
                        )
                    )
                    self._cnt["bi_bottom_rom"] = 0

        self._prev_ts_ms = side.ts_ms
        self._prev_angle = self._elbow_angle_s

        return hints
