import { useEffect, useMemo, useRef, useState } from 'react';
import { AlertTriangle, Video } from 'lucide-react';

interface PoseKeypoint {
  x: number;
  y: number;
  confidence: number;
  label: string;
}

interface ViewState {
  ok: boolean;
  avg_vis: number;
  keypoints: PoseKeypoint[];
  errors: string[];
  frame_w: number;
  frame_h: number;
  mjpeg?: string;
}

interface CameraFeedProps {
  title: string;
  connected: boolean;
  poseData: ViewState | null;
  /** Rotates BOTH the MJPEG image and skeleton overlay (degrees). */
  rotateDeg?: 0 | 90 | 180 | 270;
}

// Minimal skeleton connections for a clear demo
const POSE_CONNECTIONS: Array<[string, string]> = [
  ['nose', 'left_shoulder'],
  ['nose', 'right_shoulder'],
  ['left_shoulder', 'right_shoulder'],

  ['left_shoulder', 'left_elbow'],
  ['left_elbow', 'left_wrist'],

  ['right_shoulder', 'right_elbow'],
  ['right_elbow', 'right_wrist'],

  ['left_shoulder', 'left_hip'],
  ['right_shoulder', 'right_hip'],
  ['left_hip', 'right_hip'],

  ['left_hip', 'left_knee'],
  ['left_knee', 'left_ankle'],

  ['right_hip', 'right_knee'],
  ['right_knee', 'right_ankle'],
];

export function CameraFeed({ title, connected, poseData, rotateDeg = 0 }: CameraFeedProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapperRef = useRef<HTMLDivElement | null>(null);

  const [boxSize, setBoxSize] = useState<{ w: number; h: number }>({ w: 0, h: 0 });

  const frameW = poseData?.frame_w ?? 0;
  const frameH = poseData?.frame_h ?? 0;

  // Rotacja nadal jako CSS (tak jak u Ciebie)
  const rotateStyle = useMemo(() => {
    return {
      transform: `rotate(${rotateDeg}deg)`,
      transformOrigin: 'center center',
    } as React.CSSProperties;
  }, [rotateDeg]);

  // Mierzymy rozmiar kontenera (żeby policzyć "object-cover" dla overlay)
  useEffect(() => {
    const el = wrapperRef.current;
    if (!el) return;

    const update = () => {
      const r = el.getBoundingClientRect();
      setBoxSize({ w: Math.max(0, r.width), h: Math.max(0, r.height) });
    };

    update();

    const ro = new ResizeObserver(() => update());
    ro.observe(el);

    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapperRef.current;
    if (!canvas || !wrap) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const containerW = boxSize.w;
    const containerH = boxSize.h;

    // Jak jeszcze nie ma wymiarów — nie rysuj
    if (containerW <= 1 || containerH <= 1) return;

    const w = frameW > 0 ? frameW : 640;
    const h = frameH > 0 ? frameH : 360;

    // DPI scaling (żeby nie było rozmyte)
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const cw = Math.floor(containerW * dpr);
    const ch = Math.floor(containerH * dpr);

    if (canvas.width !== cw) canvas.width = cw;
    if (canvas.height !== ch) canvas.height = ch;

    // Czyścimy
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!poseData || !poseData.keypoints || poseData.keypoints.length === 0) return;

    const keypoints = poseData.keypoints;
    const errors = poseData.errors || [];

    // UWAGA: symulujemy object-fit: cover dla overlay
    // skala = max(w kontenera / w klatki, h kontenera / h klatki)
    // offset = centrowanie (bo cover docina)
    const scale = Math.max(containerW / w, containerH / h);
    const offsetX = (containerW - w * scale) / 2;
    const offsetY = (containerH - h * scale) / 2;

    // Funkcja mapowania punktu z układu klatki (px) → układ kontenera (px)
    const mapPoint = (x: number, y: number) => {
      const mx = (x * scale + offsetX) * dpr;
      const my = (y * scale + offsetY) * dpr;
      return { mx, my };
    };

    // Linie
    ctx.lineWidth = 3 * dpr;

    for (const [a, b] of POSE_CONNECTIONS) {
      const p1 = keypoints.find((k) => k.label === a);
      const p2 = keypoints.find((k) => k.label === b);
      if (!p1 || !p2) continue;
      if (p1.confidence <= 0.5 || p2.confidence <= 0.5) continue;

      const hasError = errors.includes(a) || errors.includes(b);
      const m1 = mapPoint(p1.x, p1.y);
      const m2 = mapPoint(p2.x, p2.y);

      ctx.beginPath();
      ctx.strokeStyle = hasError ? '#ef4444' : '#10b981';
      ctx.moveTo(m1.mx, m1.my);
      ctx.lineTo(m2.mx, m2.my);
      ctx.stroke();
    }

    // Punkty
    for (const kp of keypoints) {
      if (kp.confidence <= 0.5) continue;

      const hasError = errors.includes(kp.label);
      const { mx, my } = mapPoint(kp.x, kp.y);

      // glow
      ctx.beginPath();
      ctx.arc(mx, my, 8 * dpr, 0, Math.PI * 2);
      ctx.fillStyle = hasError ? 'rgba(239, 68, 68, 0.28)' : 'rgba(16, 185, 129, 0.28)';
      ctx.fill();

      // dot
      ctx.beginPath();
      ctx.arc(mx, my, 4 * dpr, 0, Math.PI * 2);
      ctx.fillStyle = hasError ? '#ef4444' : '#10b981';
      ctx.fill();

      // center
      ctx.beginPath();
      ctx.arc(mx, my, 2 * dpr, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
    }
  }, [poseData, frameW, frameH, boxSize]);

  const showStream = Boolean(poseData?.mjpeg);
  const poseOk = Boolean(poseData?.ok);
  const avgVis = poseData?.avg_vis ?? 0;

  return (
    <div
      ref={wrapperRef}
      className="relative w-full h-full bg-slate-900 rounded-xl overflow-hidden border border-slate-800"
    >
      {/* Video */}
      {showStream ? (
        <img
          src={poseData?.mjpeg}
          alt={title}
          className="absolute inset-0 w-full h-full object-cover" // ✅ cover zamiast contain
          style={rotateStyle}
        />
      ) : (
        <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-slate-800 to-slate-900">
          <div className="text-center">
            <Video className="w-14 h-14 text-slate-600 mx-auto mb-2" />
            <p className="text-slate-300">{title}</p>
            <p className="text-xs text-slate-500 mt-1">No MJPEG stream</p>
          </div>
        </div>
      )}

      {/* Canvas overlay */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full pointer-events-none"
        style={rotateStyle}
      />

      {/* Top status */}
      <div className="absolute top-3 left-3 right-3 flex items-center justify-between">
        <div className="bg-slate-950/60 backdrop-blur-sm px-3 py-2 rounded-lg border border-slate-800 flex items-center gap-2">
          <span className="text-xs text-slate-200">{title}</span>
          <span className="text-[10px] text-slate-500">
            {frameW > 0 && frameH > 0 ? `${frameW}×${frameH}` : '—'}
          </span>
          <span className="text-[10px] text-slate-500">rot: {rotateDeg}°</span>
        </div>

        <div className="bg-slate-950/60 backdrop-blur-sm px-3 py-2 rounded-lg border border-slate-800">
          <div className="flex items-center gap-2 text-xs">
            <span className={connected ? 'text-emerald-300' : 'text-red-300'}>
              {connected ? 'WS OK' : 'WS DOWN'}
            </span>
            <div
              className={`w-1.5 h-1.5 rounded-full ${
                connected ? 'bg-emerald-400' : 'bg-red-400'
              } animate-pulse`}
            />
          </div>
        </div>
      </div>

      {/* Bottom pose indicator */}
      <div className="absolute bottom-3 left-3 bg-slate-950/60 backdrop-blur-sm px-3 py-2 rounded-lg border border-slate-800">
        <div className="flex items-center gap-2 text-xs">
          {poseOk ? (
            <>
              <div className="w-2 h-2 bg-emerald-400 rounded-full" />
              <span className="text-slate-200">Pose OK</span>
              <span className="text-slate-500">vis: {Math.round(avgVis * 100)}%</span>
            </>
          ) : (
            <>
              <AlertTriangle className="w-4 h-4 text-amber-300" />
              <span className="text-slate-300">Analyzing…</span>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
