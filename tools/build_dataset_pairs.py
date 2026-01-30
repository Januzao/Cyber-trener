import os
import glob
import json
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

@dataclass
class PairMeta:
    exercise: str
    form: int
    front_path: str
    side_path: str
    fps: float
    length: int

def _normalize_landmarks(arr33x4: np.ndarray) -> np.ndarray:
    # arr: (33,4) x,y,z,vis
    L_HIP, R_HIP = 23, 24
    L_SH, R_SH = 11, 12

    xy = arr33x4[:, :2].astype(np.float32)
    vis = arr33x4[:, 3:4].astype(np.float32)

    hip_center = (xy[L_HIP] + xy[R_HIP]) / 2.0
    xy = xy - hip_center

    sh_dist = np.linalg.norm(xy[L_SH] - xy[R_SH]) + 1e-6
    hip_dist = np.linalg.norm(xy[L_HIP] - xy[R_HIP]) + 1e-6
    scale = sh_dist if sh_dist > 1e-3 else hip_dist
    xy = xy / scale

    return np.concatenate([xy, vis], axis=1).astype(np.float32)  # (33,3)

def _extract_seq(video_path: str, target_fps: int = 30) -> Tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1:
        fps = float(target_fps)

    step = max(int(round(fps / target_fps)), 1)

    seq: List[np.ndarray] = []
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step != 0:
                i += 1
                continue
            i += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if not res.pose_landmarks:
                seq.append(np.zeros((33, 3), dtype=np.float32))
                continue

            lm = res.pose_landmarks.landmark
            arr = np.zeros((33, 4), dtype=np.float32)
            for k in range(33):
                arr[k, 0] = lm[k].x
                arr[k, 1] = lm[k].y
                arr[k, 2] = lm[k].z
                arr[k, 3] = lm[k].visibility

            seq.append(_normalize_landmarks(arr))

    cap.release()
    if not seq:
        raise RuntimeError(f"Empty sequence: {video_path}")

    return np.stack(seq, axis=0), float(target_fps)  # (T,33,3)

def _add_velocity(seq33x3: np.ndarray) -> np.ndarray:
    # (T,33,3) => (T,33,5) x,y,vis,vx,vy
    xy = seq33x3[..., :2]
    v = np.zeros_like(xy, dtype=np.float32)
    v[1:] = xy[1:] - xy[:-1]
    return np.concatenate([seq33x3, v], axis=-1).astype(np.float32)

def _pair_files(dir_path: str) -> List[Tuple[str, str]]:
    fronts = sorted(glob.glob(os.path.join(dir_path, "*_front.*")))
    pairs = []
    for f in fronts:
        base = os.path.basename(f)
        stem = base.rsplit("_front", 1)[0]
        side = None
        for ext in (".mp4", ".mov", ".mkv", ".avi", ".MOV", ".MP4"):
            cand = os.path.join(dir_path, stem + "_side" + ext)
            if os.path.exists(cand):
                side = cand
                break
        if side is None:
            raise RuntimeError(f"Missing side pair for: {f}")
        pairs.append((f, side))
    if not pairs:
        raise RuntimeError(f"No *_front.* found in {dir_path}")
    return pairs

def build(root: str, out_npz: str) -> None:
    X_front: List[np.ndarray] = []
    X_side: List[np.ndarray] = []
    y_form: List[int] = []
    metas: List[PairMeta] = []

    for form_name, form_val in [("correct", 1), ("incorrect", 0)]:
        d = os.path.join(root, "squat", form_name)
        if not os.path.isdir(d):
            continue
        pairs = _pair_files(d)

        for fp, sp in pairs:
            seq_f, fps = _extract_seq(fp, target_fps=30)
            seq_s, _ = _extract_seq(sp, target_fps=30)

            feat_f = _add_velocity(seq_f)  # (Tf,33,5)
            feat_s = _add_velocity(seq_s)  # (Ts,33,5)

            T = min(feat_f.shape[0], feat_s.shape[0])
            feat_f = feat_f[:T]
            feat_s = feat_s[:T]

            X_front.append(feat_f)
            X_side.append(feat_s)
            y_form.append(int(form_val))
            metas.append(PairMeta("squat", int(form_val), fp, sp, fps, T))

            print(f"[OK] {form_name}: {os.path.basename(fp)} + {os.path.basename(sp)} | T={T}")

    if not X_front:
        raise RuntimeError("No data found. Check dataset paths and filenames.")

    meta_json = json.dumps([m.__dict__ for m in metas], ensure_ascii=False)
    np.savez_compressed(
        out_npz,
        X_front=np.array(X_front, dtype=object),
        X_side=np.array(X_side, dtype=object),
        y_form=np.array(y_form, dtype=np.int64),
        meta=np.array(meta_json),
    )
    print(f"Saved: {out_npz} | samples={len(X_front)}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="path to dataset/ directory")
    ap.add_argument("--out", required=True, help="output .npz path")
    args = ap.parse_args()
    build(args.root, args.out)
