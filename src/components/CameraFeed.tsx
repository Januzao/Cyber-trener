import { useEffect, useRef } from 'react';
import { Video } from 'lucide-react';

interface PoseKeypoint {
  x: number;
  y: number;
  confidence: number;
  label: string;
}

interface PoseData {
  keypoints: PoseKeypoint[];
  errors: string[];
}

interface CameraFeedProps {
  poseData: PoseData | null;
}

// Skeleton connections for drawing lines between keypoints
const POSE_CONNECTIONS = [
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

export function CameraFeed({ poseData }: CameraFeedProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Initialize webcam
  useEffect(() => {
    const initCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 1280, height: 720 },
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.log('Camera access not available in this environment');
      }
    };

    initCamera();

    return () => {
      if (videoRef.current?.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
        tracks.forEach((track) => track.stop());
      }
    };
  }, []);

  // Draw pose skeleton overlay
  useEffect(() => {
    if (!canvasRef.current || !poseData) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const { keypoints, errors } = poseData;

    // Draw connections (skeleton lines)
    ctx.lineWidth = 3;
    POSE_CONNECTIONS.forEach(([start, end]) => {
      const startPoint = keypoints.find((kp) => kp.label === start);
      const endPoint = keypoints.find((kp) => kp.label === end);

      if (startPoint && endPoint && startPoint.confidence > 0.5 && endPoint.confidence > 0.5) {
        const hasError = errors.includes(start) || errors.includes(end);
        
        ctx.beginPath();
        ctx.strokeStyle = hasError ? '#ef4444' : '#10b981';
        ctx.moveTo(startPoint.x, startPoint.y);
        ctx.lineTo(endPoint.x, endPoint.y);
        ctx.stroke();
      }
    });

    // Draw keypoints
    keypoints.forEach((kp) => {
      if (kp.confidence > 0.5) {
        const hasError = errors.includes(kp.label);
        
        // Outer circle (glow effect)
        ctx.beginPath();
        ctx.arc(kp.x, kp.y, 8, 0, 2 * Math.PI);
        ctx.fillStyle = hasError ? 'rgba(239, 68, 68, 0.3)' : 'rgba(16, 185, 129, 0.3)';
        ctx.fill();

        // Inner circle
        ctx.beginPath();
        ctx.arc(kp.x, kp.y, 4, 0, 2 * Math.PI);
        ctx.fillStyle = hasError ? '#ef4444' : '#10b981';
        ctx.fill();

        // White center
        ctx.beginPath();
        ctx.arc(kp.x, kp.y, 2, 0, 2 * Math.PI);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
      }
    });
  }, [poseData]);

  return (
    <div className="relative w-full h-full bg-slate-900 rounded-lg overflow-hidden border border-slate-700">
      {/* Video element (hidden, used for actual camera feed) */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="absolute inset-0 w-full h-full object-cover"
      />

      {/* Placeholder when camera is not available */}
      <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-slate-800 to-slate-900">
        <div className="text-center">
          <Video className="w-24 h-24 text-slate-600 mx-auto mb-4" />
          <p className="text-slate-400">Camera Feed Placeholder</p>
          <p className="text-sm text-slate-500 mt-2">Live video stream from webcam</p>
        </div>
      </div>

      {/* Canvas overlay for pose skeleton */}
      <canvas
        ref={canvasRef}
        width={640}
        height={540}
        className="absolute inset-0 w-full h-full pointer-events-none"
      />

      {/* Processing indicator */}
      <div className="absolute top-4 left-4 bg-slate-900/80 backdrop-blur-sm px-3 py-2 rounded-lg border border-slate-700">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
          <span className="text-xs text-slate-300">Processing frames in real-time</span>
        </div>
      </div>

      {/* Python engine connection indicator */}
      <div className="absolute top-4 right-4 bg-slate-900/80 backdrop-blur-sm px-3 py-2 rounded-lg border border-slate-700">
        <div className="text-xs text-slate-400">
          <div className="flex items-center gap-2">
            <span className="text-emerald-400">Python ML Engine</span>
            <div className="w-1.5 h-1.5 bg-emerald-400 rounded-full" />
          </div>
        </div>
      </div>
    </div>
  );
}
