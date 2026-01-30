from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# MediaPipe Pose indices
L_SH, R_SH = 11, 12
L_HIP, R_HIP = 23, 24

@dataclass
class PoseFeatureState:
    prev_xy: Optional[np.ndarray] = None  # (33,2)

def _normalize_landmarks_xy_vis(xy: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """
    xy: (33,2) in [0..1] from MediaPipe
    vis: (33,1)
    returns: (33,3) -> x,y,vis normalized:
      - center at hip center
      - scale by shoulder distance (fallback hip distance)
    """
    hip_center = (xy[L_HIP] + xy[R_HIP]) / 2.0
    xy2 = xy - hip_center

    sh_dist = float(np.linalg.norm(xy2[L_SH] - xy2[R_SH]) + 1e-6)
    hip_dist = float(np.linalg.norm(xy2[L_HIP] - xy2[R_HIP]) + 1e-6)
    scale = sh_dist if sh_dist > 1e-3 else hip_dist

    xy2 = xy2 / scale
    return np.concatenate([xy2.astype(np.float32), vis.astype(np.float32)], axis=1)  # (33,3)

def mp_landmarks_to_feat33x5(
    lm_xy: np.ndarray,
    lm_vis: np.ndarray,
    state: PoseFeatureState,
) -> Tuple[np.ndarray, float]:
    """
    lm_xy: (33,2) raw mediapipe x,y
    lm_vis: (33,1) raw mediapipe visibility
    returns:
      feat: (33,5) -> x,y,vis,vx,vy (vx/vy in normalized space)
      motion: scalar motion proxy (mean abs velocity)
    """
    base33x3 = _normalize_landmarks_xy_vis(lm_xy, lm_vis)  # (33,3)
    xy = base33x3[:, :2]  # normalized xy

    if state.prev_xy is None:
        v = np.zeros_like(xy, dtype=np.float32)
    else:
        v = (xy - state.prev_xy).astype(np.float32)

    state.prev_xy = xy.copy()

    feat = np.concatenate([base33x3, v], axis=1).astype(np.float32)  # (33,5)

    motion = float(np.mean(np.abs(v)))  # simple gate for "is moving"
    return feat, motion
