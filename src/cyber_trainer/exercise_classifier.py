from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .features import FrontMetrics, SideMetrics


_ALLOWED = {"deadlift", "squat", "lunge", "pushups", "biceps", "plank", "unknown"}


@dataclass(frozen=True)
class ExerciseGuess:
    """Classifier output.

    name: one of:
      deadlift | squat | lunge | pushups | biceps | plank | unknown
    """

    name: str
    confidence: float
    reason: str


class ExerciseClassifier:
    """A lightweight heuristic classifier (NOT a trained ML model).

    In practice you should prefer *forced* mode from UI, because heuristics in 2D
    are sensitive to camera angle and user anthropometry.
    """

    def __init__(self, min_visibility: float = 0.55) -> None:
        self.min_visibility = float(min_visibility)
        self._forced: Optional[str] = None

    @property
    def forced(self) -> Optional[str]:
        return self._forced

    def set_forced(self, name: Optional[str]) -> None:
        """Force a specific exercise or None for auto."""
        if name is None:
            self._forced = None
            return

        if name not in _ALLOWED or name == "unknown":
            raise ValueError("Unsupported exercise. Allowed: deadlift | squat | lunge | pushups | biceps | plank | None")

        self._forced = name

    def classify(self, side: SideMetrics, front: FrontMetrics) -> ExerciseGuess:
        # forced mode
        if self._forced is not None:
            return ExerciseGuess(name=self._forced, confidence=1.0, reason="forced by user")

        # visibility gate
        base_vis = 0.0
        if side.ok:
            base_vis = max(base_vis, side.avg_vis)
        if front.ok:
            base_vis = max(base_vis, front.avg_vis)

        if base_vis < self.min_visibility or not side.ok:
            return ExerciseGuess(name="unknown", confidence=0.0, reason="pose not visible enough")

        bar_dx = float(side.bar_dx_norm)
        torso_len = max(1.0, float(side.torso_len))
        hip_sh_dy = abs(float(side.hip_y) - float(side.shoulder_y)) / torso_len

        # --- pushups / plank family ---
        if hip_sh_dy < 0.35 and bar_dx > 1.15:
            conf = min(0.92, 0.62 + (0.35 - hip_sh_dy) * 0.9)
            return ExerciseGuess(name="pushups", confidence=float(conf), reason="body almost horizontal (pushup/plank)")

        # --- deadlift ---
        if bar_dx < 0.85:
            conf = min(0.95, 0.65 + (0.85 - bar_dx) * 0.7)
            return ExerciseGuess(name="deadlift", confidence=float(conf), reason="hands close to shins/ankles")

        # --- squat ---
        if 0.85 <= bar_dx < 1.35 and hip_sh_dy > 0.6:
            conf = min(0.88, 0.55 + (bar_dx - 0.85) * 0.25)
            return ExerciseGuess(name="squat", confidence=float(conf), reason="upright stance with hands away from ankles")

        # --- biceps curl hint (very weak in auto) ---
        # If elbow angle is changing a lot while torso stays upright, could be curls.
        if side.elbow_angle > 1e-3 and hip_sh_dy > 0.55 and 45.0 <= side.elbow_angle <= 165.0:
            conf = 0.45 + min(0.25, base_vis * 0.25)
            return ExerciseGuess(name="biceps", confidence=float(min(0.7, conf)), reason="arm flexion detected (weak cue)")

        # --- lunge (very weak in auto) ---
        # With current metrics, lunges are hard to detect robustly.
        # We keep it as unknown unless forced.
        conf = 0.35 + (min(1.0, base_vis) * 0.25)
        return ExerciseGuess(name="unknown", confidence=float(min(0.7, conf)), reason="not enough distinctive cues")
