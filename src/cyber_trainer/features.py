from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .pose import Point, PoseResult


def dist(a: Point, b: Point) -> float:
    return float(math.hypot(a.x - b.x, a.y - b.y))


def angle_deg(a: Point, b: Point, c: Point) -> float:
    """Angle ABC in degrees, with B as vertex."""
    ba = np.array([a.x - b.x, a.y - b.y], dtype=np.float32)
    bc = np.array([c.x - b.x, c.y - b.y], dtype=np.float32)
    nba = float(np.linalg.norm(ba))
    nbc = float(np.linalg.norm(bc))
    if nba < 1e-6 or nbc < 1e-6:
        return 0.0
    cosv = float(np.dot(ba, bc) / (nba * nbc))
    cosv = max(-1.0, min(1.0, cosv))
    return float(math.degrees(math.acos(cosv)))


def ema(prev: Optional[float], cur: float, alpha: float) -> float:
    if prev is None:
        return cur
    return alpha * cur + (1.0 - alpha) * prev


def pick_body_side(lm: dict[str, Point]) -> str:
    """In side-view one side landmarks are often more visible.

    Choose 'left' or 'right' based on visibility of key joints.
    """
    left_keys = ["left_shoulder", "left_hip", "left_knee", "left_ankle", "left_wrist", "left_elbow"]
    right_keys = ["right_shoulder", "right_hip", "right_knee", "right_ankle", "right_wrist", "right_elbow"]

    def _avg(keys: list[str]) -> float:
        vals = [lm[k].vis for k in keys if k in lm]
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))

    left_vis = _avg(left_keys)
    right_vis = _avg(right_keys)
    return "left" if left_vis >= right_vis else "right"


@dataclass
class SideMetrics:
    ok: bool
    ts_ms: int
    avg_vis: float

    # Core (used by deadlift/squat/lunge/pushup)
    torso_hip_angle: float          # angle at hip: shoulder-hip-ankle
    hip_y: float
    shoulder_y: float
    bar_dx_norm: float              # |wrist.x - ankle.x| / torso_len
    torso_len: float

    # Added for curls + richer motion gating
    wrist_x: float
    wrist_y: float
    elbow_y: float
    elbow_angle: float              # angle at elbow: shoulder-elbow-wrist (0..180)


@dataclass
class FrontMetrics:
    ok: bool
    ts_ms: int
    avg_vis: float
    knee_inward_norm: float         # max of (left inward, right inward)
    asymmetry_norm: float           # |left_inward - right_inward|
    hip_width: float


def compute_side_metrics(pose: PoseResult, ts_ms: int) -> SideMetrics:
    if not pose.ok:
        return SideMetrics(False, ts_ms, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)

    lm = pose.landmarks
    side = pick_body_side(lm)

    sh = lm[f"{side}_shoulder"]
    hip = lm[f"{side}_hip"]
    ank = lm[f"{side}_ankle"]
    wr = lm[f"{side}_wrist"]
    el = lm.get(f"{side}_elbow")

    torso_len = max(1.0, dist(sh, hip))
    torso_hip_angle = angle_deg(sh, hip, ank)
    bar_dx_norm = abs(wr.x - ank.x) / torso_len

    elbow_angle = 0.0
    elbow_y = 0.0
    if el is not None:
        elbow_angle = angle_deg(sh, el, wr)
        elbow_y = float(el.y)

    return SideMetrics(
        ok=True,
        ts_ms=ts_ms,
        avg_vis=float(pose.avg_vis),
        torso_hip_angle=float(torso_hip_angle),
        hip_y=float(hip.y),
        shoulder_y=float(sh.y),
        bar_dx_norm=float(bar_dx_norm),
        torso_len=float(torso_len),
        wrist_x=float(wr.x),
        wrist_y=float(wr.y),
        elbow_y=float(elbow_y),
        elbow_angle=float(elbow_angle),
    )


def compute_front_metrics(pose: PoseResult, ts_ms: int) -> FrontMetrics:
    if not pose.ok:
        return FrontMetrics(False, ts_ms, 0.0, 0.0, 0.0, 1.0)

    lm = pose.landmarks
    lh, rh = lm["left_hip"], lm["right_hip"]
    la, ra = lm["left_ankle"], lm["right_ankle"]
    lk, rk = lm["left_knee"], lm["right_knee"]

    hip_width = max(1.0, abs(rh.x - lh.x))

    # inward collapse: knees drifting toward center relative to ankle
    left_inward = max(0.0, (lk.x - la.x) / hip_width)      # left knee moved right from left ankle
    right_inward = max(0.0, (ra.x - rk.x) / hip_width)     # right knee moved left from right ankle

    knee_inward_norm = max(left_inward, right_inward)
    asymmetry_norm = abs(left_inward - right_inward)

    return FrontMetrics(
        ok=True,
        ts_ms=ts_ms,
        avg_vis=float(pose.avg_vis),
        knee_inward_norm=float(knee_inward_norm),
        asymmetry_norm=float(asymmetry_norm),
        hip_width=float(hip_width),
    )
