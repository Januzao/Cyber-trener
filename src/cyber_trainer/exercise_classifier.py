from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .features import FrontMetrics, SideMetrics


_ALLOWED = {"deadlift", "squat", "plank", "unknown"}


@dataclass(frozen=True)
class ExerciseGuess:
    """Classifier output.

    name: one of: deadlift | squat | plank | unknown
    """

    name: str
    confidence: float
    reason: str


class ExerciseClassifier:
    """A lightweight heuristic classifier.

    Important: this is NOT a trained ML model. It is meant only as a quick MVP
    that distinguishes a few basic movements based on simple, interpretable cues.
    """

    def __init__(self, min_visibility: float = 0.55) -> None:
        self.min_visibility = min_visibility
        self._forced: Optional[str] = None

    @property
    def forced(self) -> Optional[str]:
        return self._forced

    def set_forced(self, name: Optional[str]) -> None:
        """Force a specific exercise ("deadlift" | "squat" | "plank") or None for auto."""
        if name is None:
            self._forced = None
            return

        if name not in _ALLOWED or name == "unknown":
            raise ValueError("Unsupported exercise. Allowed: deadlift | squat | plank | None")

        self._forced = name

    def classify(self, side: SideMetrics, front: FrontMetrics) -> ExerciseGuess:
        # forced mode
        if self._forced is not None:
            return ExerciseGuess(name=self._forced, confidence=1.0, reason="forced by user")

        # visibility gate
        if not side.ok or side.avg_vis < self.min_visibility:
            return ExerciseGuess(name="unknown", confidence=0.0, reason="pose not visible enough")

        bar_dx = float(side.bar_dx_norm)
        torso_len = max(1.0, float(side.torso_len))
        hip_sh_dy = abs(float(side.hip_y) - float(side.shoulder_y)) / torso_len

        # Plank/push-up style: body nearly horizontal + hands far from ankles
        if hip_sh_dy < 0.35 and bar_dx > 1.15:
            conf = min(0.9, 0.6 + (0.35 - hip_sh_dy) * 0.8)
            return ExerciseGuess(name="plank", confidence=float(conf), reason="body almost horizontal")

        # Deadlift: wrists close to ankles in x (bar is near the shins)
        if bar_dx < 0.85:
            conf = min(0.95, 0.65 + (0.85 - bar_dx) * 0.7)
            return ExerciseGuess(name="deadlift", confidence=float(conf), reason="hands close to shins/ankles")

        # Squat: upright stance, hands not near ankles
        if 0.85 <= bar_dx < 1.35 and hip_sh_dy > 0.6:
            conf = min(0.85, 0.55 + (bar_dx - 0.85) * 0.25)
            return ExerciseGuess(name="squat", confidence=float(conf), reason="upright stance with hands away from ankles")

        # fallback
        conf = 0.35 + (min(1.0, side.avg_vis) * 0.25)
        if front.ok:
            conf += min(0.25, front.avg_vis * 0.25)

        return ExerciseGuess(
            name="unknown",
            confidence=float(min(0.7, conf)),
            reason="not enough distinctive cues",
        )
