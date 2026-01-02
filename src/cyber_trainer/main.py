from __future__ import annotations

import time
from typing import List, Optional

import cv2

from .config import AppConfig
from .features import compute_front_metrics, compute_side_metrics
from .hinting import HintManager
from .pose import PoseEstimator
from .rules_deadlift import DeadliftRuleEngine, Hint
from .summary import SessionSummary, save_summary
from .video_stream import LatestFrameGrabber


def _put_text(img, text: str, line: int = 0) -> None:
    y = 30 + 28 * line
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)


def main() -> None:
    cfg = AppConfig()

    print("Starting streams...")
    front_grabber = LatestFrameGrabber("front-grabber", cfg.front_url)
    side_grabber = LatestFrameGrabber("side-grabber", cfg.side_url)
    front_grabber.start()
    side_grabber.start()

    pose_front = PoseEstimator()
    pose_side = PoseEstimator()

    hint_mgr = HintManager(
        enable_voice=cfg.enable_voice,
        max_hints_per_sec=cfg.max_hints_per_sec,
        per_issue_cooldown_sec=cfg.per_issue_cooldown_sec,
    )
    engine = DeadliftRuleEngine(cfg.thresholds, lift_vel_px_s=cfg.lift_velocity_px_per_s)

    session_on = False
    session_start_ms = 0
    session_hints: List[Hint] = []
    last_hint_text: str = ""

    tick = 1.0 / max(1, cfg.process_fps)
    next_t = time.time()

    print("Ready.")
    print("Controls: S=start, E=end, Q=quit")

    try:
        while True:
            # timing
            now = time.time()
            if now < next_t:
                time.sleep(max(0.0, next_t - now))
            next_t = max(next_t + tick, time.time())

            fp = front_grabber.get_latest()
            sp = side_grabber.get_latest()

            # if missing frames, just show last or blank
            front_img = fp.frame.copy() if fp is not None else None
            side_img = sp.frame.copy() if sp is not None else None

            # Pose + rules only if we have both frames
            if front_img is not None and side_img is not None:
                front_pose = pose_front.infer(front_img)
                side_pose = pose_side.infer(side_img)

                if front_pose.ok:
                    for p in front_pose.landmarks.values():
                        if p.vis > 0.5:
                            cv2.circle(front_img, (int(p.x), int(p.y)), 4, (0, 255, 255), -1)
                if side_pose.ok:
                    for p in side_pose.landmarks.values():
                        if p.vis > 0.5:
                            cv2.circle(side_img, (int(p.x), int(p.y)), 4, (0, 255, 255), -1)

                if session_on:
                    # gate on pose visibility
                    if side_pose.avg_vis >= cfg.min_pose_visibility:
                        side_m = compute_side_metrics(side_pose, sp.ts_ms)
                        front_m = compute_front_metrics(front_pose, fp.ts_ms) if front_pose.ok else None

                        # even if front missing, we pass a "not ok" metrics container
                        if front_m is None:
                            from .features import FrontMetrics
                            front_m = FrontMetrics(False, fp.ts_ms, 0.0, 0.0, 0.0, 1.0)

                        hints = engine.update(side_m, front_m)
                        for h in hints:
                            if hint_mgr.allow(h):
                                hint_mgr.emit(h)
                                session_hints.append(h)
                                last_hint_text = f"{h.issue_code}: {h.message}"

                # overlays
                _put_text(front_img, f"Front | session={'ON' if session_on else 'OFF'}", 0)
                _put_text(side_img, f"Side  | session={'ON' if session_on else 'OFF'}", 0)
                if last_hint_text:
                    _put_text(front_img, f"Last: {last_hint_text}", 1)
                    _put_text(side_img, f"Last: {last_hint_text}", 1)

            # display
            if front_img is not None:
                if cfg.display_scale != 1.0:
                    front_img = cv2.resize(front_img, None, fx=cfg.display_scale, fy=cfg.display_scale)
                cv2.imshow("Front", front_img)
            if side_img is not None:
                if cfg.display_scale != 1.0:
                    side_img = cv2.resize(side_img, None, fx=cfg.display_scale, fy=cfg.display_scale)
                cv2.imshow("Side", side_img)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("s"), ord("S")) and not session_on:
                session_on = True
                session_hints = []
                session_start_ms = int(time.time() * 1000)
                last_hint_text = ""
                print("Session started.")
            if key in (ord("e"), ord("E")) and session_on:
                session_on = False
                end_ms = int(time.time() * 1000)
                summary = SessionSummary(started_ts_ms=session_start_ms, ended_ts_ms=end_ms, hints=session_hints)
                path = save_summary(summary)
                print("Session ended.")
                print("Counts:", summary.counts())
                print("Saved:", path)

    finally:
        hint_mgr.shutdown()
        front_grabber.stop()
        side_grabber.stop()
        cv2.destroyAllWindows()
