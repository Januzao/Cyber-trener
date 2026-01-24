from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .features import SideMetrics


@dataclass
class RepState:
    reps: int = 0
    sets: int = 0
    in_rep: bool = False


class DeadliftRepCounter:
    """Very simple deadlift rep counter based on hip vertical movement.

    Notes:
    - In image coordinates, *y grows downward*.
    - When the hip goes up, hip_y decreases.

    This is intentionally minimal for MVP:
    - detects the "up" phase using hip vertical speed
    - counts a rep when the lift ends and the vertical displacement was large enough
    """

    def __init__(
        self,
        lift_vel_px_s: float,
        min_displacement_px: float = 55.0,
        min_rep_ms: int = 450,
        target_reps_per_set: int = 10,
    ) -> None:
        self.lift_vel_px_s = float(lift_vel_px_s)
        self.min_displacement_px = float(min_displacement_px)
        self.min_rep_ms = int(min_rep_ms)
        self.target_reps_per_set = int(target_reps_per_set)

        self.state = RepState()

        self._prev_ts_ms: Optional[int] = None
        self._prev_hip_y: Optional[float] = None

        self._in_lift: bool = False
        self._lift_start_ms: Optional[int] = None
        self._start_y: Optional[float] = None
        self._min_y: Optional[float] = None

    def reset(self) -> None:
        self.state = RepState()
        self._prev_ts_ms = None
        self._prev_hip_y = None
        self._in_lift = False
        self._lift_start_ms = None
        self._start_y = None
        self._min_y = None

    def update(self, side: SideMetrics) -> RepState:
        if not side.ok:
            return self.state

        # velocity (px/s)
        hip_up_speed = 0.0
        if self._prev_ts_ms is not None and self._prev_hip_y is not None:
            dt = max(1e-3, (side.ts_ms - self._prev_ts_ms) / 1000.0)
            hip_up_speed = (self._prev_hip_y - side.hip_y) / dt

        is_lifting_now = hip_up_speed > self.lift_vel_px_s

        # start of a rep
        if is_lifting_now and not self._in_lift:
            self._in_lift = True
            self.state.in_rep = True
            self._lift_start_ms = side.ts_ms
            self._start_y = float(side.hip_y)
            self._min_y = float(side.hip_y)

        # update min hip height
        if self._in_lift and self._min_y is not None:
            self._min_y = min(self._min_y, float(side.hip_y))

        # end of a rep
        if not is_lifting_now and self._in_lift:
            dur = 0
            if self._lift_start_ms is not None:
                dur = int(side.ts_ms - self._lift_start_ms)

            disp = 0.0
            if self._start_y is not None and self._min_y is not None:
                disp = float(self._start_y - self._min_y)

            if dur >= self.min_rep_ms and disp >= self.min_displacement_px:
                self.state.reps += 1

                # naive set counting
                if self.state.reps >= self.target_reps_per_set:
                    self.state.sets += 1
                    self.state.reps = 0

            self._in_lift = False
            self.state.in_rep = False
            self._lift_start_ms = None
            self._start_y = None
            self._min_y = None

        # store prev
        self._prev_ts_ms = int(side.ts_ms)
        self._prev_hip_y = float(side.hip_y)

        return self.state
