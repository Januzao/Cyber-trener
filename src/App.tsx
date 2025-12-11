import { useState, useEffect } from 'react';
import { CameraFeed } from './components/CameraFeed';
import { FeedbackPanel } from './components/FeedbackPanel';
import { MetricsSidebar } from './components/MetricsSidebar';

export default function App() {
  const [formStatus, setFormStatus] = useState<'correct' | 'incorrect' | 'neutral'>('neutral');
  const [feedbackMessage, setFeedbackMessage] = useState('');
  const [repCount, setRepCount] = useState(0);
  const [setCount, setSetCount] = useState(0);
  const [accuracyHistory, setAccuracyHistory] = useState<Array<{ time: number; accuracy: number }>>([]);
  const [poseData, setPoseData] = useState<any>(null);

  // Simulate real-time data from Python processing engine
  useEffect(() => {
    const simulateRealTimeData = () => {
      // Mock pose skeleton data (keypoints)
      const mockPoseData = {
        keypoints: [
          { x: 320, y: 150, confidence: 0.95, label: 'nose' },
          { x: 310, y: 180, confidence: 0.92, label: 'left_shoulder' },
          { x: 330, y: 180, confidence: 0.93, label: 'right_shoulder' },
          { x: 300, y: 250, confidence: 0.88, label: 'left_elbow' },
          { x: 340, y: 250, confidence: 0.89, label: 'right_elbow' },
          { x: 290, y: 310, confidence: 0.85, label: 'left_wrist' },
          { x: 350, y: 310, confidence: 0.87, label: 'right_wrist' },
          { x: 305, y: 320, confidence: 0.91, label: 'left_hip' },
          { x: 335, y: 320, confidence: 0.90, label: 'right_hip' },
          { x: 310, y: 420, confidence: 0.86, label: 'left_knee' },
          { x: 330, y: 420, confidence: 0.87, label: 'right_knee' },
          { x: 315, y: 520, confidence: 0.84, label: 'left_ankle' },
          { x: 325, y: 520, confidence: 0.85, label: 'right_ankle' },
        ],
        errors: [] as string[],
      };

      // Randomly simulate form correction feedback
      const random = Math.random();
      if (random > 0.7) {
        setFormStatus('incorrect');
        const errors = [
          'Lower your hips - keep back straight',
          'Knees should not extend past toes',
          'Keep your chest up and core engaged',
          'Distribute weight evenly on both feet',
        ];
        const selectedError = errors[Math.floor(Math.random() * errors.length)];
        setFeedbackMessage(selectedError);
        mockPoseData.errors = ['left_knee', 'right_knee'];
      } else if (random > 0.4) {
        setFormStatus('correct');
        setFeedbackMessage('Perfect form! Keep it up!');
        mockPoseData.errors = [];
      } else {
        setFormStatus('neutral');
        setFeedbackMessage('');
        mockPoseData.errors = [];
      }

      setPoseData(mockPoseData);

      // Update accuracy history
      const accuracy = 60 + Math.random() * 35;
      setAccuracyHistory((prev) => {
        const newHistory = [
          ...prev,
          { time: Date.now(), accuracy: Math.round(accuracy) },
        ];
        return newHistory.slice(-20); // Keep last 20 data points
      });
    };

    // Initial data
    simulateRealTimeData();

    // Update every 2 seconds to simulate real-time processing
    const interval = setInterval(simulateRealTimeData, 2000);

    return () => clearInterval(interval);
  }, []);

  // Simulate rep counting
  useEffect(() => {
    const repInterval = setInterval(() => {
      if (Math.random() > 0.7) {
        setRepCount((prev) => {
          const newCount = prev + 1;
          if (newCount >= 10 && newCount % 10 === 0) {
            setSetCount((s) => s + 1);
            return 0;
          }
          return newCount;
        });
      }
    }, 3000);

    return () => clearInterval(repInterval);
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-white">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-sm px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl text-white">Real-Time Pose Trainer</h1>
            <p className="text-sm text-slate-400 mt-1">
              Live feedback powered by Python ML processing engine
            </p>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
              <span className="text-sm text-slate-300">Engine Connected</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Layout */}
      <div className="flex h-[calc(100vh-80px)]">
        {/* Primary Area - Camera Feed (70%) */}
        <div className="flex-1 w-[70%] relative p-6">
          <div className="h-full flex flex-col gap-4">
            {/* Camera Feed with Pose Overlay */}
            <div className="flex-1 relative">
              <CameraFeed poseData={poseData} />
            </div>

            {/* Feedback Panel - Overlay style */}
            <FeedbackPanel status={formStatus} message={feedbackMessage} />
          </div>
        </div>

        {/* Metrics Sidebar (30%) */}
        <div className="w-[30%] border-l border-slate-800 bg-slate-900/30">
          <MetricsSidebar
            repCount={repCount}
            setCount={setCount}
            accuracyHistory={accuracyHistory}
          />
        </div>
      </div>
    </div>
  );
}
