from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .features import FrontMetrics, SideMetrics


@dataclass
class ActivityState:
    phase: str               # "idle" | "ready" | "active"
    active: bool
    motion_px_s: float
    since_ms: int


class MotionGate:
    """Detects when the user has *started* the selected exercise.

    Why this exists:
    - Pose visibility alone should NOT increase the 'form score'.
    - We only want to score once the user begins moving in a way consistent with the exercise.

    Implementation:
    - Track per-frame motion (px/s) of a few robust landmarks from side metrics:
      hip_y, shoulder_y, wrist_y.
    - Consider the set 'active' when motion stays above a threshold for N frames.
    - Drop back to 'ready' after a cool-down duration with low motion.
    """

    def __init__(
        self,
        min_vis: float,
        active_frames: int = 4,
        inactive_ms: int = 900,
    ) -> None:
        self.min_vis = float(min_vis)
        self.active_frames = int(active_frames)
        self.inactive_ms = int(inactive_ms)

        self._prev_ts_ms: Optional[int] = None
        self._prev_hip_y: Optional[float] = None
        self._prev_sh_y: Optional[float] = None
        self._prev_wr_y: Optional[float] = None

        self._above_cnt: int = 0
        self._last_active_ms: int = 0
        self._phase: str = "idle"

    def reset(self) -> None:
        self._prev_ts_ms = None
        self._prev_hip_y = None
        self._prev_sh_y = None
        self._prev_wr_y = None
        self._above_cnt = 0
        self._last_active_ms = 0
        self._phase = "idle"

    def _motion_threshold(self, exercise: str) -> float:
        # Tuned for 12 FPS and typical laptop cams.
        # If your camera is far away (small person in frame), lower thresholds ~20–30%.
        base: Dict[str, float] = {
            "deadlift": 40.0,
            "squat": 35.0,
            "lunge": 32.0,
            "pushups": 28.0,
            "biceps": 18.0,
            "unknown": 45.0,
        }
        return float(base.get(exercise, 35.0))

    def update(self, exercise: str, side: SideMetrics, front: FrontMetrics) -> ActivityState:
        ts = int(side.ts_ms if side.ok else (front.ts_ms if front.ok else 0))

        # If pose is not usable -> idle and no scoring
        base_vis = 0.0
        if side.ok:
            base_vis = max(base_vis, side.avg_vis)
        if front.ok:
            base_vis = max(base_vis, front.avg_vis)

        if base_vis < self.min_vis:
            self._phase = "idle"
            self._above_cnt = 0
            return ActivityState(phase=self._phase, active=False, motion_px_s=0.0, since_ms=ts)

        # We have decent pose => at least READY (if an exercise is selected/known)
        if self._phase == "idle":
            self._phase = "ready"

        # Motion estimate
        motion = 0.0
        if side.ok and self._prev_ts_ms is not None and self._prev_hip_y is not None and self._prev_sh_y is not None:
            dt = max(1e-3, (ts - self._prev_ts_ms) / 1000.0)
            d1 = abs(side.hip_y - self._prev_hip_y) / dt
            d2 = abs(side.shoulder_y - self._prev_sh_y) / dt
            d3 = 0.0
            if self._prev_wr_y is not None:
                d3 = abs(side.wrist_y - self._prev_wr_y) / dt
            motion = max(d1, d2, d3)

        thr = self._motion_threshold(exercise)

        if motion > thr:
            self._above_cnt += 1
            self._last_active_ms = ts
        else:
            self._above_cnt = max(0, self._above_cnt - 1)

        # Become ACTIVE after N consecutive-ish frames above threshold
        if self._phase != "active" and self._above_cnt >= self.active_frames:
            self._phase = "active"
            self._last_active_ms = ts

        # Drop back to READY if we were ACTIVE but motion is low for some time
        if self._phase == "active" and (ts - self._last_active_ms) > self.inactive_ms:
            self._phase = "ready"
            self._above_cnt = 0

        # Store prev for next motion estimate
        if side.ok:
            self._prev_ts_ms = ts
            self._prev_hip_y = side.hip_y
            self._prev_sh_y = side.shoulder_y
            self._prev_wr_y = side.wrist_y

        return ActivityState(phase=self._phase, active=(self._phase == "active"), motion_px_s=float(motion), since_ms=int(self._last_active_ms))
