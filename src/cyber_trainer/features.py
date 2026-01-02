from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .pose import Point, PoseResult


def dist(a: Point, b: Point) -> float:
    return float(math.hypot(a.x - b.x, a.y - b.y))


def angle_deg(a: Point, b: Point, c: Point) -> float:
    """
    Angle ABC in degrees, with B as vertex.
    """
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
    """
    In side-view one side landmarks are often more visible.
    Choose 'left' or 'right' based on visibility of key joints.
    """
    left_keys = ["left_shoulder", "left_hip", "left_knee", "left_ankle", "left_wrist"]
    right_keys = ["right_shoulder", "right_hip", "right_knee", "right_ankle", "right_wrist"]

    left_vis = sum(lm[k].vis for k in left_keys if k in lm) / len(left_keys)
    right_vis = sum(lm[k].vis for k in right_keys if k in lm) / len(right_keys)
    return "left" if left_vis >= right_vis else "right"


@dataclass
class SideMetrics:
    ok: bool
    ts_ms: int
    avg_vis: float
    torso_hip_angle: float          # angle at hip: shoulder-hip-ankle
    hip_y: float
    shoulder_y: float
    bar_dx_norm: float              # |wrist.x - ankle.x| / torso_len
    torso_len: float


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
        return SideMetrics(False, ts_ms, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    lm = pose.landmarks
    side = pick_body_side(lm)

    sh = lm[f"{side}_shoulder"]
    hip = lm[f"{side}_hip"]
    ank = lm[f"{side}_ankle"]
    wr = lm[f"{side}_wrist"]

    torso_len = max(1.0, dist(sh, hip))
    torso_hip_angle = angle_deg(sh, hip, ank)
    bar_dx_norm = abs(wr.x - ank.x) / torso_len

    return SideMetrics(
        ok=True,
        ts_ms=ts_ms,
        avg_vis=pose.avg_vis,
        torso_hip_angle=torso_hip_angle,
        hip_y=hip.y,
        shoulder_y=sh.y,
        bar_dx_norm=bar_dx_norm,
        torso_len=torso_len,
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
        avg_vis=pose.avg_vis,
        knee_inward_norm=float(knee_inward_norm),
        asymmetry_norm=float(asymmetry_norm),
        hip_width=float(hip_width),
    )
