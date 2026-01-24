from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .config import DeadliftThresholds
from .features import FrontMetrics, SideMetrics, ema


@dataclass(frozen=True)
class Hint:
    ts_ms: int
    issue_code: str
    message: str
    severity: str          # "info" | "warn" | "critical"
    view: str              # "front" | "side"
    confidence: float


class DeadliftRuleEngine:
    """
    Minimal rule engine with:
    - smoothing
    - debounce (frames_required)
    - basic "lifting" detection using hip vertical velocity
    """

    def __init__(self, thr: DeadliftThresholds, lift_vel_px_s: float) -> None:
        self.thr = thr
        self.lift_vel_px_s = lift_vel_px_s

        # state for velocities
        self._prev_ts_ms: Optional[int] = None
        self._prev_hip_y: Optional[float] = None
        self._prev_sh_y: Optional[float] = None

        # smoothed metrics
        self._torso_angle_s: Optional[float] = None
        self._bar_dx_s: Optional[float] = None
        self._knee_in_s: Optional[float] = None
        self._asym_s: Optional[float] = None

        # debounce counters
        self._cnt: Dict[str, int] = {}

        # lifting phase detect
        self._is_lifting: bool = False
        self._lift_start_ms: Optional[int] = None

    def _count(self, key: str, cond: bool) -> bool:
        if cond:
            self._cnt[key] = self._cnt.get(key, 0) + 1
        else:
            self._cnt[key] = 0
        return self._cnt[key] >= self.thr.frames_required

    def update(self, side: SideMetrics, front: FrontMetrics) -> List[Hint]:
        hints: List[Hint] = []

        # Gate: require side metrics for lifting detection
        if not side.ok:
            return hints

        # Smooth key metrics (EMA)
        self._torso_angle_s = ema(self._torso_angle_s, side.torso_hip_angle, alpha=0.35)
        self._bar_dx_s = ema(self._bar_dx_s, side.bar_dx_norm, alpha=0.35)

        if front.ok:
            self._knee_in_s = ema(self._knee_in_s, front.knee_inward_norm, alpha=0.35)
            self._asym_s = ema(self._asym_s, front.asymmetry_norm, alpha=0.35)

        # Velocity (y down in image, so moving up => y decreases)
        is_lifting_now = False
        hip_up_speed = 0.0
        sh_up_speed = 0.0

        if self._prev_ts_ms is not None and self._prev_hip_y is not None and self._prev_sh_y is not None:
            dt = max(1e-3, (side.ts_ms - self._prev_ts_ms) / 1000.0)
            hip_up_speed = (self._prev_hip_y - side.hip_y) / dt
            sh_up_speed = (self._prev_sh_y - side.shoulder_y) / dt
            is_lifting_now = hip_up_speed > self.lift_vel_px_s

        # update lift phase
        if is_lifting_now and not self._is_lifting:
            self._lift_start_ms = side.ts_ms
        self._is_lifting = is_lifting_now

        # Basic confidence from pose visibility
        conf_side = min(1.0, max(0.0, (side.avg_vis - 0.45) / 0.4))
        conf_front = min(1.0, max(0.0, (front.avg_vis - 0.45) / 0.4)) if front.ok else 0.0

        # === RULE 1: Back rounding / too much lean proxy (SIDE) ===
        if self._is_lifting and self._torso_angle_s is not None:
            cond = self._torso_angle_s < self.thr.min_torso_hip_angle_deg
            if self._count("back_rounding", cond):
                hints.append(
                    Hint(
                        ts_ms=side.ts_ms,
                        issue_code="DL_BACK_NEUTRAL",
                        message="Keep your back neutral.",
                        severity="warn",
                        view="side",
                        confidence=conf_side,
                    )
                )
                self._cnt["back_rounding"] = 0

        # === RULE 2: Early hip rise (SIDE) ===
        if self._is_lifting and self._lift_start_ms is not None:
            if (side.ts_ms - self._lift_start_ms) <= 900:  # early part of the pull
                if sh_up_speed > 1e-3:
                    ratio = hip_up_speed / sh_up_speed
                    cond = ratio > self.thr.early_hip_ratio
                    if self._count("early_hips", cond):
                        hints.append(
                            Hint(
                                ts_ms=side.ts_ms,
                                issue_code="DL_EARLY_HIPS",
                                message="Don’t shoot your hips up early.",
                                severity="warn",
                                view="side",
                                confidence=conf_side,
                            )
                        )
                        self._cnt["early_hips"] = 0

        # === RULE 3: Bar drifting away (SIDE, approx via wrist vs ankle dx) ===
        if self._is_lifting and self._bar_dx_s is not None:
            cond = self._bar_dx_s > self.thr.bar_far_norm_dx
            if self._count("bar_far", cond):
                hints.append(
                    Hint(
                        ts_ms=side.ts_ms,
                        issue_code="DL_BAR_CLOSE",
                        message="Keep the bar close to your legs.",
                        severity="info",
                        view="side",
                        confidence=conf_side,
                    )
                )
                self._cnt["bar_far"] = 0

        # === RULE 4: Knee collapse / asymmetry (FRONT) ===
        if self._is_lifting and front.ok and self._knee_in_s is not None and self._asym_s is not None:
            cond_in = self._knee_in_s > self.thr.knee_inward_norm
            cond_as = self._asym_s > self.thr.asymmetry_norm
            cond = cond_in or cond_as

            if self._count("knees", cond):
                hints.append(
                    Hint(
                        ts_ms=front.ts_ms,
                        issue_code="DL_KNEES_OVER_TOES",
                        message="Keep knees tracking over toes.",
                        severity="info",
                        view="front",
                        confidence=conf_front,
                    )
                )
                self._cnt["knees"] = 0

        # store previous
        self._prev_ts_ms = side.ts_ms
        self._prev_hip_y = side.hip_y
        self._prev_sh_y = side.shoulder_y

        return hints
