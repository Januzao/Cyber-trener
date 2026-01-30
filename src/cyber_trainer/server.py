from __future__ import annotations

import argparse
import asyncio
import json
import time
import threading
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .activity_gate import MotionGate
from .config import AppConfig
from .exercise_classifier import ExerciseClassifier
from .features import compute_front_metrics, compute_side_metrics
from .form_scoring import FormScoreTracker
from .hinting import HintManager
from .pose import PoseEstimator, PoseResult
from .reps import DeadliftRepCounter
from .rules_biceps import BicepsRuleEngine
from .rules_common import Hint as CommonHint
from .rules_deadlift import DeadliftRuleEngine, Hint as DeadliftHint
from .rules_lunge import LungeRuleEngine
from .rules_pushup import PushupRuleEngine
from .rules_squat import SquatRuleEngine
from .summary import SessionSummary, save_summary
from .video_stream import LatestFrameGrabber


import os

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    # Reuse the exact architecture used during training.
    from .ml.train_squat_pairs import SquatFormTCN
except Exception:  # pragma: no cover
    SquatFormTCN = None  # type: ignore



def _pose_to_keypoints(pose: PoseResult) -> List[Dict[str, Any]]:
    # UI expects: {x, y, confidence, label}
    out: List[Dict[str, Any]] = []
    if not pose.ok:
        return out

    for label, p in pose.landmarks.items():
        out.append(
            {
                "x": float(p.x),
                "y": float(p.y),
                "confidence": float(p.vis),
                "label": str(label),
            }
        )

    return out


# Which keypoints should be highlighted in UI for each issue.
ISSUE_TO_ERRORS: Dict[str, List[str]] = {
    # Deadlift
    "DL_BACK_NEUTRAL": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    "DL_EARLY_HIPS": ["left_hip", "right_hip", "left_knee", "right_knee"],
    "DL_BAR_CLOSE": ["left_wrist", "right_wrist", "left_ankle", "right_ankle"],
    "DL_KNEES_OVER_TOES": ["left_knee", "right_knee", "left_ankle", "right_ankle"],

    # Squat
    "SQ_BACK_NEUTRAL": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    "SQ_KNEES_COLLAPSE": ["left_knee", "right_knee", "left_ankle", "right_ankle"],

    # Lunge
    "LU_KNEES": ["left_knee", "right_knee", "left_ankle", "right_ankle"],
    "LU_LEAN": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],

    # Pushups
    "PU_SAG": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    "PU_HIPS_HIGH": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],

    # Biceps curls
    "BI_FAST": ["left_wrist", "right_wrist", "left_elbow", "right_elbow"],
    "BI_SWING": ["left_wrist", "right_wrist", "left_elbow", "right_elbow"],
    "BI_TOP_ROM": ["left_wrist", "right_wrist", "left_elbow", "right_elbow"],
    "BI_BOTTOM_ROM": ["left_wrist", "right_wrist", "left_elbow", "right_elbow"],
}



# ------------------------------------------------------------
# Squat ML (correct/incorrect) — trained on your own videos
# ------------------------------------------------------------
MP_LABELS_33: List[str] = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# indices used by MediaPipe Pose
_L_SH, _R_SH = 11, 12
_L_HIP, _R_HIP = 23, 24


def _pose_to_np_xy_vis(pose: PoseResult) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (xy, vis) in MediaPipe landmark order.

    xy: (33,2)  vis: (33,1)
    """
    if not pose.ok:
        return None, None

    xy = np.zeros((33, 2), dtype=np.float32)
    vis = np.zeros((33, 1), dtype=np.float32)

    for i, name in enumerate(MP_LABELS_33):
        p = pose.landmarks.get(name)
        if p is None:
            continue
        xy[i, 0] = float(p.x)
        xy[i, 1] = float(p.y)
        vis[i, 0] = float(p.vis)

    return xy, vis


def _normalize_xy_vis(xy: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """Normalize as in dataset builder: center at hip, scale by shoulder width."""
    hip_center = (xy[_L_HIP] + xy[_R_HIP]) / 2.0
    xy2 = xy - hip_center

    sh_dist = float(np.linalg.norm(xy2[_L_SH] - xy2[_R_SH]) + 1e-6)
    hip_dist = float(np.linalg.norm(xy2[_L_HIP] - xy2[_R_HIP]) + 1e-6)
    scale = sh_dist if sh_dist > 1e-3 else hip_dist
    xy2 = xy2 / scale

    return np.concatenate([xy2.astype(np.float32), vis.astype(np.float32)], axis=1)  # (33,3) -> x,y,vis


class _FeatState:
    def __init__(self) -> None:
        self.prev_xy: Optional[np.ndarray] = None  # (33,2) normalized


def _to_feat33x5(xy: np.ndarray, vis: np.ndarray, st: _FeatState) -> tuple[np.ndarray, float]:
    """Build features (33,5) = x,y,vis,vx,vy + motion proxy."""
    base = _normalize_xy_vis(xy, vis)  # (33,3)
    nxy = base[:, :2]

    if st.prev_xy is None:
        v = np.zeros_like(nxy, dtype=np.float32)
    else:
        v = (nxy - st.prev_xy).astype(np.float32)

    st.prev_xy = nxy.copy()

    feat = np.concatenate([base, v], axis=1).astype(np.float32)  # (33,5)
    motion = float(np.mean(np.abs(v)))
    return feat, motion


def _safe_torch_load(path: str):
    if torch is None:
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=True)  # type: ignore[arg-type]
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        return None


class SquatMlRuntime:
    """Realtime inference wrapper for squat_form.pt.

    - keeps a sliding window for front + side
    - outputs p(correct) in [0,1]
    - provides an EMA-smoothed score (0..100)
    """

    def __init__(self, ckpt_path: str, window_default: int = 45) -> None:
        self.enabled = False
        self.window = int(window_default)
        self.in_dim = 0

        self._model = None
        self._buf_f: List[np.ndarray] = []
        self._buf_s: List[np.ndarray] = []

        self._front_st = _FeatState()
        self._side_st = _FeatState()

        self._ema = 0.0

        if torch is None or SquatFormTCN is None:
            return

        ckpt = _safe_torch_load(ckpt_path)
        if not isinstance(ckpt, dict):
            return

        try:
            self.window = int(ckpt.get("window", window_default))
            self.in_dim = int(ckpt.get("in_dim", 0))
            model = SquatFormTCN(in_dim=self.in_dim)
            model.load_state_dict(ckpt["model_state"], strict=True)
            model.eval()
            self._model = model
            self.enabled = True
        except Exception:
            self.enabled = False
            self._model = None

    def reset(self) -> None:
        self._buf_f = []
        self._buf_s = []
        self._front_st = _FeatState()
        self._side_st = _FeatState()
        self._ema = 0.0

    def _push(self, feat_f: np.ndarray, feat_s: np.ndarray) -> None:
        self._buf_f.append(feat_f.reshape(-1).astype(np.float32))
        self._buf_s.append(feat_s.reshape(-1).astype(np.float32))
        if len(self._buf_f) > self.window:
            self._buf_f.pop(0)
        if len(self._buf_s) > self.window:
            self._buf_s.pop(0)

    def ready(self) -> bool:
        return self.enabled and len(self._buf_f) == self.window and len(self._buf_s) == self.window

    def update(
        self,
        front_pose: PoseResult,
        side_pose: PoseResult,
        *,
        is_active: bool,
        match_ok: bool,
        motion_gate_ok: bool,
    ) -> tuple[int, str, Optional[float], bool]:
        """Return (score0_100, status, p_correct, incorrect)."""
        if not self.enabled:
            return 0, "neutral", None, False

        xy_f, vis_f = _pose_to_np_xy_vis(front_pose)
        xy_s, vis_s = _pose_to_np_xy_vis(side_pose)

        if xy_f is None or xy_s is None or vis_f is None or vis_s is None:
            # No pose -> decay to 0
            self._ema = 0.85 * self._ema
            return int(round(100.0 * self._ema)), "neutral", None, False

        feat_f, motion_f = _to_feat33x5(xy_f, vis_f, self._front_st)
        feat_s, motion_s = _to_feat33x5(xy_s, vis_s, self._side_st)

        self._push(feat_f, feat_s)

        # If user selected squat but is doing something else (classifier mismatch), force score down.
        if not match_ok:
            self._ema = 0.70 * self._ema
            return int(round(100.0 * self._ema)), "neutral", None, False

        # Only score when the session is active AND motion gate says "moving".
        if not (is_active and motion_gate_ok and self.ready()):
            self._ema = 0.90 * self._ema
            return int(round(100.0 * self._ema)), "neutral", None, False

        assert self._model is not None

        wf = np.stack(self._buf_f, axis=0)
        ws = np.stack(self._buf_s, axis=0)
        x = np.concatenate([wf, ws], axis=1).astype(np.float32)  # (W,F)

        xt = torch.from_numpy(x).unsqueeze(0)
        with torch.no_grad():
            logit = float(self._model(xt).squeeze(0).item())
        p = float(1.0 / (1.0 + np.exp(-logit)))

        # EMA smoothing for UI scale
        alpha = 0.25
        self._ema = (1.0 - alpha) * self._ema + alpha * p

        status = "good" if p >= 0.5 else "bad"
        incorrect = bool(p < 0.5)
        return int(round(100.0 * self._ema)), status, p, incorrect


def _severity_rank(sev: str) -> int:
    if sev == "critical":
        return 3
    if sev == "warn":
        return 2
    if sev == "info":
        return 1
    return 0


def _hint_severity(h: Any) -> str:
    return str(getattr(h, "severity", ""))


def _hint_issue(h: Any) -> str:
    return str(getattr(h, "issue_code", ""))


def _hint_message(h: Any) -> str:
    return str(getattr(h, "message", ""))


def _hint_view(h: Any) -> str:
    return str(getattr(h, "view", ""))


class ConnectionManager:
    def __init__(self) -> None:
        self._active: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._active.add(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._active.discard(ws)

    async def broadcast_json(self, payload: Dict[str, Any]) -> None:
        async with self._lock:
            clients = list(self._active)

        dead: List[WebSocket] = []
        for ws in clients:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)

        for ws in dead:
            await self.disconnect(ws)


class EngineThread(threading.Thread):
    """Background thread that:

    - reads 2 MJPEG streams (front + side)
    - runs MediaPipe Pose
    - classifies/uses forced exercise selection
    - runs rule engines for 5 exercises
    - publishes state snapshots for the UI
    """

    def __init__(self, cfg: AppConfig, publish_cb) -> None:
        super().__init__(daemon=True, name="engine_thr")
        self.cfg = cfg
        self._publish = publish_cb

        self.front_thr = LatestFrameGrabber(name="front_thr", url=cfg.front_url)
        self.side_thr = LatestFrameGrabber(name="side_thr", url=cfg.side_url)

        self.front_pose = PoseEstimator()
        self.side_pose = PoseEstimator()

        self.classifier = ExerciseClassifier(min_visibility=cfg.min_pose_visibility)

        # rule engines
        self.deadlift_engine = DeadliftRuleEngine(thr=cfg.thresholds, lift_vel_px_s=cfg.lift_velocity_px_per_s)
        self.squat_engine = SquatRuleEngine(frames_required=cfg.thresholds.frames_required)
        self.lunge_engine = LungeRuleEngine(frames_required=cfg.thresholds.frames_required)
        self.pushup_engine = PushupRuleEngine(frames_required=cfg.thresholds.frames_required)
        self.biceps_engine = BicepsRuleEngine(frames_required=cfg.thresholds.frames_required)

        # rep counting (kept for deadlift as in MVP)
        self.rep_counter = DeadliftRepCounter(lift_vel_px_s=cfg.lift_velocity_px_per_s)

        # activity + score behavior (fix: score doesn't rise just from visibility)
        self.motion_gate = MotionGate(min_vis=cfg.min_pose_visibility, active_frames=4, inactive_ms=900)
        self.score_tracker = FormScoreTracker()

        # Squat ML model (optional)
        self.squat_ml = SquatMlRuntime(ckpt_path=os.path.join(os.getcwd(), 'models', 'squat_form.pt'))

        # hinting (global rate-limit + per-issue cooldown)
        self.hint_mgr = HintManager(
            enable_voice=cfg.enable_voice,
            max_hints_per_sec=cfg.max_hints_per_sec,
            per_issue_cooldown_sec=cfg.per_issue_cooldown_sec,
        )

        # session state
        self._ctrl_lock = threading.Lock()
        self._session_on: bool = True  # MVP: run immediately
        self._session_started_ms: Optional[int] = None
        self._session_hints: List[Any] = []

        # UI hint persistence
        self._active_hint: Optional[Any] = None
        self._active_hint_until_ms: int = 0

        # stop flag
        self._stop = threading.Event()

        # latest state exposed for new WS clients
        self.last_state: Optional[Dict[str, Any]] = None

    def stop(self) -> None:
        self._stop.set()
        try:
            self.front_thr.stop()
        except Exception:
            pass
        try:
            self.side_thr.stop()
        except Exception:
            pass
        try:
            self.hint_mgr.shutdown()
        except Exception:
            pass

    def start_session(self) -> None:
        with self._ctrl_lock:
            self._session_on = True
            self._session_started_ms = int(time.time() * 1000)
            self._session_hints = []
            self._active_hint = None
            self._active_hint_until_ms = 0
            self.rep_counter.reset()
            self.motion_gate.reset()
            self.score_tracker.reset()
            try:
                self.squat_ml.reset()
            except Exception:
                pass
            try:
                self.squat_ml.reset()
            except Exception:
                pass
            self.biceps_engine.reset()

    def stop_session(self) -> Optional[str]:
        with self._ctrl_lock:
            if not self._session_on:
                return None
            self._session_on = False

            started = self._session_started_ms or int(time.time() * 1000)
            ended = int(time.time() * 1000)
            summary = SessionSummary(started_ts_ms=started, ended_ts_ms=ended, hints=list(self._session_hints))
            path = save_summary(summary)
            return path

    def set_exercise(self, name: Optional[str]) -> None:
        # name: "deadlift" | "squat" | "lunge" | "pushups" | "biceps" | "plank" | None
        with self._ctrl_lock:
            self.classifier.set_forced(name)
            # Reset gates so score starts only when you begin the new exercise
            self.motion_gate.reset()
            self.score_tracker.reset()
            if name == "biceps":
                self.biceps_engine.reset()

    def _run_engines(self, ex_name: str, side_m, front_m) -> List[Any]:
        if ex_name == "deadlift":
            return self.deadlift_engine.update(side_m, front_m)
        if ex_name == "squat":
            return self.squat_engine.update(side_m, front_m)
        if ex_name == "lunge":
            return self.lunge_engine.update(side_m, front_m)
        if ex_name == "pushups":
            return self.pushup_engine.update(side_m, front_m)
        if ex_name == "biceps":
            return self.biceps_engine.update(side_m, front_m)
        return []

    def run(self) -> None:
        self.front_thr.start()
        self.side_thr.start()

        self._session_started_ms = int(time.time() * 1000)

        tick_s = max(1.0 / max(1, self.cfg.process_fps), 0.02)
        next_t = time.perf_counter()

        while not self._stop.is_set():
            now = time.perf_counter()
            if now < next_t:
                time.sleep(min(0.005, next_t - now))
                continue

            next_t = now + tick_s

            front_pkt = self.front_thr.get_latest()
            side_pkt = self.side_thr.get_latest()

            front_w = 0
            front_h = 0
            side_w = 0
            side_h = 0
            if front_pkt is not None:
                front_h, front_w = front_pkt.frame.shape[:2]
            if side_pkt is not None:
                side_h, side_w = side_pkt.frame.shape[:2]

            # if streams are not available yet, still publish heartbeat
            if front_pkt is None and side_pkt is None:
                payload = {
                    "type": "state",
                    "ts_ms": int(time.time() * 1000),
                    "connected": True,
                    "session": {"on": self._session_on},
                    "exercise": {"name": "unknown", "confidence": 0.0, "mode": "auto"},
                    "views": {
                        "front": {"ok": False, "avg_vis": 0.0, "keypoints": [], "errors": [], "frame_w": 0, "frame_h": 0, "mjpeg": self.cfg.front_url},
                        "side": {"ok": False, "avg_vis": 0.0, "keypoints": [], "errors": [], "frame_w": 0, "frame_h": 0, "mjpeg": self.cfg.side_url},
                    },
                    "form": {"status": "neutral", "message": "Очікування потоків камер…", "score": 0, "pose_color": "green", "ml": {"enabled": False}},
                    "reps": asdict(self.rep_counter.state),
                }
                self.last_state = payload
                self._publish(payload)
                continue

            # run pose inference
            front_pose = self.front_pose.infer(front_pkt.frame) if front_pkt is not None else PoseResult(False, 0.0, {})
            side_pose = self.side_pose.infer(side_pkt.frame) if side_pkt is not None else PoseResult(False, 0.0, {})

            if side_pkt is not None:
                ts_ms = side_pkt.ts_ms
            elif front_pkt is not None:
                ts_ms = front_pkt.ts_ms
            else:
                ts_ms = int(time.time() * 1000)

            # compute metrics
            side_m = compute_side_metrics(side_pose, ts_ms=ts_ms)
            front_m = compute_front_metrics(front_pose, ts_ms=ts_ms)

            # exercise guess
            guess = self.classifier.classify(side_m, front_m)
            mode = "forced" if self.classifier.forced is not None else "auto"

            # Decide which exercise we are actually evaluating
            eval_ex = self.classifier.forced if self.classifier.forced is not None else guess.name

            # Activity gate: score only when you start moving
            act = self.motion_gate.update(eval_ex, side_m, front_m)
            is_active = bool(self._session_on and act.active and eval_ex != "unknown")

            # rules/hints: only when active (fix: no instant scoring on just standing)
            hints: List[Any] = []
            if is_active:
                hints = self._run_engines(eval_ex, side_m, front_m)

            # store to session timeline
            if self._session_on and hints:
                self._session_hints.extend(hints)

            # choose most important hint for UI
            if hints:
                top = max(hints, key=lambda h: _severity_rank(_hint_severity(h)))
                self._active_hint = top
                self._active_hint_until_ms = ts_ms + 2500

                # voice feedback
                try:
                    if self.hint_mgr.allow(top):
                        self.hint_mgr.emit(top)
                except Exception:
                    pass

            # rep counting (only in deadlift, as before)
            if self._session_on and eval_ex == "deadlift":
                self.rep_counter.update(side_m)

            # Determine current active hint (persist a bit for UI)
            active_hint = self._active_hint if (self._active_hint is not None and ts_ms <= self._active_hint_until_ms) else None

            # errors (highlight joints)
            errors: List[str] = []
            if active_hint is not None:
                errors = ISSUE_TO_ERRORS.get(_hint_issue(active_hint), [])

            # base visibility for scoring
            base_vis = 0.0
            if side_pose.ok:
                base_vis = max(base_vis, side_pose.avg_vis)
            if front_pose.ok:
                base_vis = max(base_vis, front_pose.avg_vis)

            # UI messages when not active yet
            if not self._session_on:
                analyzing_msg = "Сесію зупинено."
            elif eval_ex == "unknown":
                analyzing_msg = "Вибери вправу (або увімкни forced режим) і стань повністю в кадр."
            elif act.phase == "ready":
                analyzing_msg = "Почни виконувати вправу — тоді почнеться оцінка техніки."
            else:
                analyzing_msg = "Почни виконувати вправу…"

            hint_sev = _hint_severity(active_hint) if active_hint is not None else None
            hint_msg = _hint_message(active_hint) if active_hint is not None else ""

            score_state = self.score_tracker.update(
                active=is_active,
                base_vis=base_vis,
                hint_severity=hint_sev,
                hint_message=hint_msg,
                analyzing_message=analyzing_msg,
            )


            # ------------------------------------------------------------
            # Optional: ML-based squat technique score (trained on your data)
            # ------------------------------------------------------------
            form_status = score_state.status
            form_message = score_state.message
            form_score = int(score_state.score)

            ml_info: Dict[str, Any] = {"enabled": False}
            ml_incorrect = False

            if eval_ex == "squat" and getattr(self, "squat_ml", None) is not None and self.squat_ml.enabled:
                # If user forced "squat" but classifier thinks it's NOT squat -> decay score.
                match_ok = True
                if self.classifier.forced == "squat":
                    match_ok = (guess.name == "squat" and float(guess.confidence) >= 0.55)

                ml_score, ml_status, ml_p, ml_incorrect = self.squat_ml.update(
                    front_pose,
                    side_pose,
                    is_active=is_active,
                    match_ok=match_ok,
                    motion_gate_ok=bool(act.active),
                )

                ml_info = {
                    "enabled": True,
                    "p_correct": None if ml_p is None else float(ml_p),
                    "status": str(ml_status),
                    "score": int(ml_score),
                    "match_ok": bool(match_ok),
                }

                # Override the UI score with ML output
                form_score = int(ml_score)
                if ml_p is None:
                    form_status = "neutral"
                    form_message = analyzing_msg
                else:
                    form_status = "good" if ml_p >= 0.5 else "bad"
                    form_message = f"ML техніка присідання: {form_score}/100"

            # If ML says incorrect squat, highlight the whole skeleton in red (fallback via 'errors')
            if ml_incorrect:
                errors = list(MP_LABELS_33)

            payload = {
                "type": "state",
                "ts_ms": int(ts_ms),
                "connected": True,
                "session": {"on": bool(self._session_on)},
                "exercise": {
                    "name": str(eval_ex),
                    "confidence": float(guess.confidence),
                    "reason": str(guess.reason),
                    "mode": mode,
                    "phase": act.phase,
                    "motion_px_s": float(act.motion_px_s),
                },
                "views": {
                    "front": {
                        "ok": bool(front_pose.ok),
                        "avg_vis": float(front_pose.avg_vis),
                        "keypoints": _pose_to_keypoints(front_pose),
                        "errors": errors if front_pose.ok else [],
                        "mjpeg": self.cfg.front_url,
                        "frame_w": int(front_w),
                        "frame_h": int(front_h),
                    },
                    "side": {
                        "ok": bool(side_pose.ok),
                        "avg_vis": float(side_pose.avg_vis),
                        "keypoints": _pose_to_keypoints(side_pose),
                        "errors": errors if side_pose.ok else [],
                        "mjpeg": self.cfg.side_url,
                        "frame_w": int(side_w),
                        "frame_h": int(side_h),
                    },
                },
                "form": {
                    "status": form_status,
                    "message": form_message,
                    "score": int(form_score),
                    "pose_color": "red" if ml_incorrect else "green",
                    "ml": ml_info,
                },
                "reps": asdict(self.rep_counter.state),
            }

            self.last_state = payload
            self._publish(payload)


class RealtimeHub:
    """Glue between the engine thread and WebSocket clients."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.manager = ConnectionManager()

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue: Optional[asyncio.Queue] = None

        self.engine = EngineThread(cfg=cfg, publish_cb=self._publish_from_thread)

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=2)
        self.engine.start()
        asyncio.create_task(self._broadcaster())

    def _publish_from_thread(self, payload: Dict[str, Any]) -> None:
        if self._loop is None or self._queue is None:
            return

        def _put() -> None:
            assert self._queue is not None
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except Exception:
                    pass
            try:
                self._queue.put_nowait(payload)
            except Exception:
                pass

        self._loop.call_soon_threadsafe(_put)

    async def _broadcaster(self) -> None:
        assert self._queue is not None
        while True:
            payload = await self._queue.get()
            await self.manager.broadcast_json(payload)


cfg = AppConfig()
app = FastAPI(title="Cyber-Trener Realtime API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

hub = RealtimeHub(cfg)


@app.on_event("startup")
async def _startup() -> None:
    await hub.start()


@app.on_event("shutdown")
async def _shutdown() -> None:
    try:
        hub.engine.stop()
    except Exception:
        pass


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"ok": True, "session_on": bool(hub.engine._session_on), "forced_exercise": hub.engine.classifier.forced}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await hub.manager.connect(ws)

    # send latest snapshot instantly (so UI doesn't wait for next tick)
    if hub.engine.last_state is not None:
        try:
            await ws.send_json(hub.engine.last_state)
        except Exception:
            pass

    try:
        while True:
            raw = await ws.receive_text()
            if not raw:
                continue

            try:
                msg = json.loads(raw)
            except Exception:
                await ws.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            if not isinstance(msg, dict):
                continue

            mtype = str(msg.get("type", ""))

            if mtype == "control":
                action = str(msg.get("action", ""))

                if action == "start":
                    hub.engine.start_session()
                    await ws.send_json({"type": "ack", "action": "start", "ok": True})

                elif action == "stop":
                    path = hub.engine.stop_session()
                    await ws.send_json({"type": "ack", "action": "stop", "ok": True})
                    if path:
                        await ws.send_json({"type": "summary", "path": path})

                elif action == "set_exercise":
                    name = msg.get("name")
                    if name is None:
                        hub.engine.set_exercise(None)
                        await ws.send_json({"type": "ack", "action": "set_exercise", "ok": True, "name": None})
                    else:
                        try:
                            hub.engine.set_exercise(str(name))
                            await ws.send_json({"type": "ack", "action": "set_exercise", "ok": True, "name": str(name)})
                        except Exception as e:
                            await ws.send_json({"type": "error", "message": str(e)})

                else:
                    await ws.send_json({"type": "error", "message": f"Unknown action: {action}"})

            elif mtype == "ping":
                await ws.send_json({"type": "pong", "ts_ms": int(time.time() * 1000)})

    except WebSocketDisconnect:
        await hub.manager.disconnect(ws)
    except Exception:
        await hub.manager.disconnect(ws)
