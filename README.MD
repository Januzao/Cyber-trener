# Cyber Trainer (MVP)

A minimal laptop app that:
- reads two MJPEG streams (front + side) from phones,
- runs MediaPipe Pose,
- detects a few deadlift form issues,
- speaks real-time voice hints (rate-limited),
- saves a session summary to JSON.

## Requirements
- Python 3.11.x
- Two phones streaming MJPEG over local Wi-Fi (e.g., Android "IP Webcam" app)

## Setup
1) Create venv and install:
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
