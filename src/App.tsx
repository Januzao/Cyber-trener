import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { CameraFeed } from './components/CameraFeed';
import { FeedbackPanel } from './components/FeedbackPanel';
import { MetricsSidebar } from './components/MetricsSidebar';
import { Dumbbell, Play, Square, Wifi, WifiOff } from 'lucide-react';

type FormStatus = 'correct' | 'incorrect' | 'neutral';

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

interface BackendState {
  type: 'state';
  ts_ms: number;
  connected: boolean;
  session: { on: boolean };
  exercise: {
    name: string;
    confidence: number;
    reason?: string;
    mode?: 'auto' | 'forced';
  };
  views: {
    front: ViewState;
    side: ViewState;
  };
  form: {
    status: FormStatus;
    message: string;
    score: number;
  };
  reps: {
    reps: number;
    sets: number;
    in_rep: boolean;
  };
}

type ExerciseMode = 'auto' | 'deadlift' | 'squat' | 'plank';

function getWsUrl(): string {
  // You can override in Vite via: VITE_WS_URL=ws://<ip>:8000/ws
  const envUrl = (import.meta as any).env?.VITE_WS_URL as string | undefined;
  if (envUrl && typeof envUrl === 'string' && envUrl.trim().length > 0) return envUrl.trim();
  return 'ws://localhost:8000/ws';
}

export default function App() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const reconnectAttemptRef = useRef<number>(0);
  const connectWsRef = useRef<() => void>(() => {});

  const [engineConnected, setEngineConnected] = useState(false);
  const [backendState, setBackendState] = useState<BackendState | null>(null);
  const [lastError, setLastError] = useState<string>('');
  const [accuracyHistory, setAccuracyHistory] = useState<Array<{ time: number; accuracy: number }>>([]);
  const [exerciseMode, setExerciseMode] = useState<ExerciseMode>('auto');

  const wsUrl = useMemo(() => getWsUrl(), []);

  const clearReconnectTimer = () => {
    if (reconnectTimerRef.current !== null) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  };

  const scheduleReconnect = useCallback(() => {
    clearReconnectTimer();

    // exponential backoff: 0.5s, 1s, 2s, 4s (max 6s)
    const attempt = reconnectAttemptRef.current;
    const delay = Math.min(6000, 500 * Math.pow(2, attempt));
    reconnectAttemptRef.current = Math.min(5, attempt + 1);

    reconnectTimerRef.current = window.setTimeout(() => {
      connectWsRef.current();
    }, delay);
  }, []);

  const connectWs = useCallback(() => {
    try {
      setLastError('');
      wsRef.current?.close();
    } catch {
      // ignore
    }

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      reconnectAttemptRef.current = 0;
      setEngineConnected(true);
      setLastError('');

      // ask backend to switch to the selected mode right away
      const forced = exerciseMode === 'auto' ? null : exerciseMode;
      ws.send(
        JSON.stringify({
          type: 'control',
          action: 'set_exercise',
          name: forced,
        })
      );
    };

    ws.onclose = () => {
      setEngineConnected(false);
      scheduleReconnect();
    };

    ws.onerror = () => {
      setEngineConnected(false);
    };

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (!msg || typeof msg !== 'object') return;

        if (msg.type === 'state') {
          const st = msg as BackendState;
          setBackendState(st);
          setAccuracyHistory((prev) => {
            const next = [...prev, { time: st.ts_ms, accuracy: Math.round(st.form.score) }];
            return next.slice(-60);
          });
          return;
        }

        if (msg.type === 'error') {
          setLastError(String(msg.message || 'Unknown error'));
          return;
        }

        if (msg.type === 'summary') {
          // optional notification for user
          // eslint-disable-next-line no-console
          console.log('Summary saved at:', msg.path);
        }
      } catch (e) {
        setLastError('WS message parse error');
      }
    };
  }, [exerciseMode, scheduleReconnect, wsUrl]);

  // Keep a stable reference for reconnect timers
  connectWsRef.current = connectWs;

  // always keep a reference to the latest connect function
  connectWsRef.current = connectWs;

  useEffect(() => {
    connectWs();
    return () => {
      clearReconnectTimer();
      try {
        wsRef.current?.close();
      } catch {
        // ignore
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const sendControl = useCallback((payload: any) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify(payload));
  }, []);

  const startSession = useCallback(() => {
    sendControl({ type: 'control', action: 'start' });
  }, [sendControl]);

  const stopSession = useCallback(() => {
    sendControl({ type: 'control', action: 'stop' });
  }, [sendControl]);

  const setExercise = useCallback(
    (mode: ExerciseMode) => {
      setExerciseMode(mode);
      const forced = mode === 'auto' ? null : mode;
      sendControl({ type: 'control', action: 'set_exercise', name: forced });
    },
    [sendControl]
  );

  const sessionOn = backendState?.session?.on ?? false;
  const formStatus: FormStatus = backendState?.form?.status ?? 'neutral';
  const feedbackMessage = backendState?.form?.message ?? '';
  const repCount = backendState?.reps?.reps ?? 0;
  const setCount = backendState?.reps?.sets ?? 0;

  const exerciseName = backendState?.exercise?.name ?? 'unknown';
  const exerciseConfidence = backendState?.exercise?.confidence ?? 0;
  const exerciseModeLabel = backendState?.exercise?.mode ?? 'auto';

  return (
    <div className="min-h-screen bg-slate-950 text-white">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-sm px-6 py-4">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-xl text-white flex items-center gap-2">
              <Dumbbell className="w-5 h-5 text-emerald-400" />
              Cyber‑Trener (Realtime)
            </h1>
            <p className="text-sm text-slate-400 mt-1">
              Live feedback from Python engine (2 cameras → pose → exercise → hints)
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            {/* Connection */}
            <div className="flex items-center gap-2 px-3 py-2 rounded-lg border border-slate-700 bg-slate-900/40">
              {engineConnected ? (
                <>
                  <Wifi className="w-4 h-4 text-emerald-400" />
                  <span className="text-sm text-slate-200">Engine Connected</span>
                </>
              ) : (
                <>
                  <WifiOff className="w-4 h-4 text-red-400" />
                  <span className="text-sm text-slate-300">Disconnected</span>
                </>
              )}
            </div>

            {/* Exercise selector */}
            <div className="flex items-center gap-2 px-3 py-2 rounded-lg border border-slate-700 bg-slate-900/40">
              <span className="text-xs text-slate-400">Exercise</span>
              <select
                value={exerciseMode}
                onChange={(e) => setExercise(e.target.value as ExerciseMode)}
                className="bg-slate-900 text-slate-200 text-sm border border-slate-700 rounded px-2 py-1 outline-none"
              >
                <option value="auto">Auto</option>
                <option value="deadlift">Deadlift</option>
                <option value="squat">Squat</option>
                <option value="plank">Plank</option>
              </select>
            </div>

            {/* Session controls */}
            <div className="flex items-center gap-2">
              <button
                onClick={startSession}
                disabled={!engineConnected}
                className="flex items-center gap-2 px-3 py-2 rounded-lg border border-emerald-500/40 bg-emerald-500/10 text-emerald-300 hover:bg-emerald-500/20 disabled:opacity-50"
              >
                <Play className="w-4 h-4" />
                Start
              </button>
              <button
                onClick={stopSession}
                disabled={!engineConnected}
                className="flex items-center gap-2 px-3 py-2 rounded-lg border border-red-500/40 bg-red-500/10 text-red-300 hover:bg-red-500/20 disabled:opacity-50"
              >
                <Square className="w-4 h-4" />
                Stop
              </button>
            </div>
          </div>
        </div>

        {/* live status line */}
        <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-slate-400">
          <span className="px-2 py-1 rounded border border-slate-800 bg-slate-900/40">
            Session: <span className={sessionOn ? 'text-emerald-300' : 'text-slate-300'}>{sessionOn ? 'ON' : 'OFF'}</span>
          </span>
          <span className="px-2 py-1 rounded border border-slate-800 bg-slate-900/40">
            Exercise: <span className="text-slate-200">{exerciseName}</span>
            <span className="text-slate-500"> ({exerciseModeLabel}, {Math.round(exerciseConfidence * 100)}%)</span>
          </span>
          <span className="px-2 py-1 rounded border border-slate-800 bg-slate-900/40">
            WS: <span className="text-slate-200">{wsUrl}</span>
          </span>
          {lastError && (
            <span className="px-2 py-1 rounded border border-red-500/30 bg-red-500/10 text-red-300">
              {lastError}
            </span>
          )}
        </div>
      </header>

      {/* Main Layout */}
      <div className="flex h-[calc(100vh-140px)]">
        {/* Primary Area */}
        <div className="flex-1 w-[70%] relative p-6">
          <div className="h-full flex flex-col gap-4">
            {/* Camera Feeds */}
            <div className="flex-1 flex items-stretch justify-center gap-4">
              {/* Each feed is a vertical 9:16 panel. Height is constrained by the available space. */}
              <div className="h-full max-h-full" style={{ aspectRatio: '9 / 16', maxWidth: '42vh' }}>
                <CameraFeed
                  title="Side view"
                  connected={engineConnected}
                  poseData={backendState?.views?.side ?? null}
                  rotateDeg={0}
                />
              </div>

              <div className="h-full max-h-full" style={{ aspectRatio: '9 / 16', maxWidth: '42vh' }}>
                <CameraFeed
                  title="Front view"
                  connected={engineConnected}
                  poseData={backendState?.views?.front ?? null}
                  rotateDeg={0}
                />
              </div>
            </div>

            {/* Feedback Panel */}
            <FeedbackPanel status={formStatus} message={feedbackMessage} />
          </div>
        </div>

        {/* Metrics Sidebar */}
        <div className="w-[30%] border-l border-slate-800 bg-slate-900/30">
          <MetricsSidebar repCount={repCount} setCount={setCount} accuracyHistory={accuracyHistory} />
        </div>
      </div>
    </div>
  );
}
