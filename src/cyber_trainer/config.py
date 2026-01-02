from __future__ import annotations

from dataclasses import dataclass


# === STREAM URLS (EDIT THESE) ===
# Example for Android IP Webcam:
#   http://192.168.1.50:8080/video
FRONT_STREAM_URL = "http://127.0.0.1:8080/video"
SIDE_STREAM_URL = "http://127.0.0.1:8081/video"


@dataclass(frozen=True)
class DeadliftThresholds:
    # Side view
    min_torso_hip_angle_deg: float = 118.0   # lower => too much lean / rounding proxy (during lifting)
    bar_far_norm_dx: float = 0.55            # wrist-to-ankle horizontal distance / torso_len
    early_hip_ratio: float = 1.45            # hip_up_speed / shoulder_up_speed during early lift

    # Front view
    knee_inward_norm: float = 0.22           # normalized inward knee drift
    asymmetry_norm: float = 0.18             # left vs right difference

    # Debounce
    frames_required: int = 6                 # condition must hold for N processed frames to trigger


@dataclass(frozen=True)
class AppConfig:
    # Streams
    front_url: str = FRONT_STREAM_URL
    side_url: str = SIDE_STREAM_URL

    # Processing
    process_fps: int = 12                    # try 8–15 depending on laptop
    display_scale: float = 0.9               # window scale

    # Pose
    min_pose_visibility: float = 0.55        # average visibility gate

    # Hinting
    enable_voice: bool = True
    max_hints_per_sec: float = 1.0
    per_issue_cooldown_sec: float = 5.0

    # Movement detection
    lift_velocity_px_per_s: float = 40.0     # hip moving up faster than this => lifting

    thresholds: DeadliftThresholds = DeadliftThresholds()
