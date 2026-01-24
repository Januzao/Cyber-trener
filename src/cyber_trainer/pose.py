from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cv2
import mediapipe as mp
import numpy as np


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    vis: float


@dataclass(frozen=True)
class PoseResult:
    ok: bool
    avg_vis: float
    landmarks: Dict[str, Point]  # pixel coords


class PoseEstimator:
    """MediaPipe Pose wrapper producing pixel-space landmarks for downstream logic + UI."""

    def __init__(self) -> None:
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def infer(self, frame_bgr: np.ndarray) -> PoseResult:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._pose.process(rgb)

        if not res.pose_landmarks:
            return PoseResult(ok=False, avg_vis=0.0, landmarks={})

        lm = res.pose_landmarks.landmark

        # Landmarks needed by:
        # - rule engine (shoulder/hip/knee/ankle/wrist)
        # - UI overlay (nose + elbows)
        want = {
            "nose": self._mp_pose.PoseLandmark.NOSE,
            "left_shoulder": self._mp_pose.PoseLandmark.LEFT_SHOULDER,
            "right_shoulder": self._mp_pose.PoseLandmark.RIGHT_SHOULDER,
            "left_elbow": self._mp_pose.PoseLandmark.LEFT_ELBOW,
            "right_elbow": self._mp_pose.PoseLandmark.RIGHT_ELBOW,
            "left_wrist": self._mp_pose.PoseLandmark.LEFT_WRIST,
            "right_wrist": self._mp_pose.PoseLandmark.RIGHT_WRIST,
            "left_hip": self._mp_pose.PoseLandmark.LEFT_HIP,
            "right_hip": self._mp_pose.PoseLandmark.RIGHT_HIP,
            "left_knee": self._mp_pose.PoseLandmark.LEFT_KNEE,
            "right_knee": self._mp_pose.PoseLandmark.RIGHT_KNEE,
            "left_ankle": self._mp_pose.PoseLandmark.LEFT_ANKLE,
            "right_ankle": self._mp_pose.PoseLandmark.RIGHT_ANKLE,
        }

        out: Dict[str, Point] = {}
        vis_sum = 0.0
        for name, idx in want.items():
            p = lm[int(idx)]
            x = float(p.x * w)
            y = float(p.y * h)
            v = float(getattr(p, "visibility", 0.0))
            out[name] = Point(x=x, y=y, vis=v)
            vis_sum += v

        avg_vis = vis_sum / max(1, len(want))
        return PoseResult(ok=True, avg_vis=avg_vis, landmarks=out)
