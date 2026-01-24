# Cyber‑Trener (2‑camera realtime MVP)

This project streams **two phone cameras** (front + side), runs **MediaPipe Pose**,
classifies the exercise (MVP heuristic), detects a few **deadlift form issues**, counts reps,
and sends everything to the UI via **WebSocket**.

## What works right now

### Backend (Python)
- 2× MJPEG input streams (`front_url`, `side_url`)
- MediaPipe pose landmarks (pixel coords)
- Exercise classifier (deadlift / squat / plank / unknown)
- Deadlift rule engine (hinge angle, bar drift, knee collapse)
- Rep counting (deadlift only)
- Rate‑limited voice hints (pyttsx3)
- WebSocket API: `/ws` → live state packets for the UI
- Session summary saved to JSON when you stop a session

### Frontend (React + Vite)
- Two video cards (front + side)
- Pose skeleton overlay on each stream
- Live feedback panel (correct / incorrect / analyzing)
- Reps/sets + accuracy graph
- Controls: Start / Stop, Exercise mode (Auto / forced)

---

## Requirements

### Backend
- Python **3.11**
- Working camera MJPEG stream(s). Recommended apps:
  - **Android:** “IP Webcam”
  - **iOS:** any MJPEG / IP camera app exposing a `.../video` endpoint

### Frontend
- Node.js 18+ (or 20+)

---

## 1) Configure phones (MJPEG URLs)

Example URLs from “IP Webcam” (Android):

- Side camera (phone 1):
  - `http://192.168.0.101:8080/video`
- Front camera (phone 2):
  - `http://192.168.0.102:8080/video`

Edit:

`src/cyber_trainer/config.py`

```py
front_url: str = "http://192.168.0.102:8080/video"
side_url: str = "http://192.168.0.101:8080/video"
```

> Tip: open each URL in a browser. If you see a live video stream – it’s correct.

---

## 2) Run backend (Python)

From the project folder:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -U pip
pip install -e .

# run the realtime server
python -m cyber_trainer.server --host 0.0.0.0 --port 8000
```

You should see uvicorn logs and the server ready.

WebSocket endpoint:
- `ws://localhost:8000/ws`

---

## 3) Run frontend (React)

Open a second terminal in the same project root and run:

```bash
npm install
npm run dev
```

Open in browser:
- `http://localhost:5173`

### If backend is on another PC / IP
Run the UI with:

```bash
VITE_WS_URL=ws://<BACKEND_IP>:8000/ws npm run dev
```

---

## 4) How to test quickly

1) Start backend (`python -m cyber_trainer.server`)
2) Start UI (`npm run dev`)
3) Verify in UI:
   - **Engine Connected** status is green
   - two camera cards show your streams
   - skeleton overlay appears (Pose OK)
4) Press **Start** and do a few deadlifts
5) Press **Stop** → backend saves a JSON summary (`sessions/session_*.json`)

---

## Notes

- If pose quality is low:
  - increase lighting
  - step back so full body fits the frame
  - keep the camera stable
- Deadlift logic expects **side view** to be the main signal.
- Rep counter is enabled **only for deadlift** in this MVP.
