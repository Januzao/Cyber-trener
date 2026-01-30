from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ScoreState:
    score: int
    status: str            # "neutral" | "correct" | "incorrect"
    message: str


class FormScoreTracker:
    """Turns (pose visibility + rule hints + activity) into a smooth score line.

    Key behavior (fix for your issue):
    - When you simply stand in frame: score stays near 0 (or neutral).
    - After you select an exercise AND start moving (activity=active): score rises and reacts to form.
    """

    def __init__(self) -> None:
        self._score_f: float = 0.0

    def reset(self) -> None:
        self._score_f = 0.0

    def update(
        self,
        *,
        active: bool,
        base_vis: float,
        hint_severity: Optional[str],
        hint_message: str,
        analyzing_message: str = "Почни виконувати вправу…",
    ) -> ScoreState:
        base_vis = max(0.0, min(1.0, float(base_vis)))

        # If we are not active yet, keep score low (do not reward just being in frame)
        if not active:
            # gentle decay to 0
            self._score_f *= 0.90
            s = int(max(0.0, min(100.0, self._score_f)))
            return ScoreState(score=s, status="neutral", message=analyzing_message)

        # If active but pose quality is poor, don't claim correctness
        if base_vis < 0.45:
            self._score_f *= 0.90
            s = int(max(0.0, min(100.0, self._score_f)))
            return ScoreState(score=s, status="neutral", message="Погана видимість — стань повністю в кадр.")

        # Target score based on hints (higher is better)
        if hint_severity is None:
            target = 92.0
            status = "correct"
            message = "Добре! Техніка виглядає правильно."
        elif hint_severity == "info":
            target = 72.0
            status = "neutral"
            message = hint_message
        elif hint_severity == "warn":
            target = 48.0
            status = "incorrect"
            message = hint_message
        else:  # "critical"
            target = 28.0
            status = "incorrect"
            message = hint_message

        # Scale by visibility: if tracking is weak, compress score
        target *= (0.55 + 0.45 * base_vis)

        # Smooth towards target
        self._score_f = 0.82 * self._score_f + 0.18 * target
        s = int(max(0.0, min(100.0, self._score_f)))

        return ScoreState(score=s, status=status, message=message)
