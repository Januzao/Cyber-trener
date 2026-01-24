import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip } from 'recharts';
import { Activity, Target, TrendingUp } from 'lucide-react';

interface MetricsSidebarProps {
  repCount: number;
  setCount: number;
  accuracyHistory: Array<{ time: number; accuracy: number }>;
}

export function MetricsSidebar({ repCount, setCount, accuracyHistory }: MetricsSidebarProps) {
  // Calculate current average accuracy
  const currentAccuracy =
    accuracyHistory.length > 0
      ? Math.round(
          accuracyHistory.reduce((sum, item) => sum + item.accuracy, 0) / accuracyHistory.length
        )
      : 0;

  // Format data for chart
  const chartData = accuracyHistory.map((item, index) => ({
    index,
    accuracy: item.accuracy,
  }));

  return (
    <div className="h-full flex flex-col p-6 overflow-y-auto">
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-lg text-white mb-1">Performance Metrics</h2>
        <p className="text-xs text-slate-400">Live data from pose estimation engine</p>
      </div>

      {/* Counters Section */}
      <div className="space-y-4 mb-8">
        {/* Rep Counter */}
        <div className="bg-gradient-to-br from-blue-500/10 to-blue-600/10 border border-blue-500/30 rounded-lg p-5">
          <div className="flex items-center gap-3 mb-3">
            <div className="p-2 bg-blue-500/20 rounded-lg">
              <Activity className="w-5 h-5 text-blue-400" />
            </div>
            <span className="text-sm text-slate-300">Current Reps</span>
          </div>
          <div className="text-4xl text-white tabular-nums">
            {repCount.toString().padStart(2, '0')}
          </div>
          <div className="mt-2 text-xs text-slate-400">of 10 target</div>
        </div>

        {/* Set Counter */}
        <div className="bg-gradient-to-br from-purple-500/10 to-purple-600/10 border border-purple-500/30 rounded-lg p-5">
          <div className="flex items-center gap-3 mb-3">
            <div className="p-2 bg-purple-500/20 rounded-lg">
              <Target className="w-5 h-5 text-purple-400" />
            </div>
            <span className="text-sm text-slate-300">Sets Completed</span>
          </div>
          <div className="text-4xl text-white tabular-nums">
            {setCount.toString().padStart(2, '0')}
          </div>
          <div className="mt-2 text-xs text-slate-400">total sets</div>
        </div>

        {/* Current Accuracy */}
        <div className="bg-gradient-to-br from-emerald-500/10 to-emerald-600/10 border border-emerald-500/30 rounded-lg p-5">
          <div className="flex items-center gap-3 mb-3">
            <div className="p-2 bg-emerald-500/20 rounded-lg">
              <TrendingUp className="w-5 h-5 text-emerald-400" />
            </div>
            <span className="text-sm text-slate-300">Avg Accuracy</span>
          </div>
          <div className="text-4xl text-white tabular-nums">
            {currentAccuracy}%
          </div>
          <div className="mt-2 text-xs text-slate-400">form accuracy score</div>
        </div>
      </div>

      {/* Real-time Accuracy Graph */}
      <div className="flex-1 bg-slate-800/50 border border-slate-700 rounded-lg p-4">
        <div className="mb-4">
          <h3 className="text-sm text-white mb-1">Real-Time Accuracy</h3>
          <p className="text-xs text-slate-400">Live form quality tracking</p>
        </div>

        <div className="h-64">
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <XAxis
                  dataKey="index"
                  stroke="#64748b"
                  tick={{ fill: '#64748b', fontSize: 10 }}
                  tickLine={false}
                />
                <YAxis
                  domain={[0, 100]}
                  stroke="#64748b"
                  tick={{ fill: '#64748b', fontSize: 10 }}
                  tickLine={false}
                  label={{
                    value: 'Accuracy %',
                    angle: -90,
                    position: 'insideLeft',
                    style: { fill: '#64748b', fontSize: 10 },
                  }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1e293b',
                    border: '1px solid #334155',
                    borderRadius: '8px',
                    fontSize: '12px',
                  }}
                  labelStyle={{ color: '#cbd5e1' }}
                />
                <Line
                  type="monotone"
                  dataKey="accuracy"
                  stroke="#10b981"
                  strokeWidth={2}
                  dot={{ fill: '#10b981', r: 3 }}
                  activeDot={{ r: 5 }}
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center text-slate-500 text-sm">
              Waiting for accuracy data...
            </div>
          )}
        </div>

        {/* Live update indicator */}
        <div className="mt-4 pt-3 border-t border-slate-700/50">
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-400">Data stream</span>
            <div className="flex items-center gap-2">
              <div className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse" />
              <span className="text-emerald-400">Live updates</span>
            </div>
          </div>
        </div>
      </div>

      {/* Processing Info */}
      <div className="mt-6 p-4 bg-slate-800/30 border border-slate-700/50 rounded-lg">
        <p className="text-xs text-slate-400 leading-relaxed">
          All metrics are computed in real-time by the Python processing engine using pose
          estimation ML models. Data is streamed continuously for immediate visual feedback.
        </p>
      </div>
    </div>
  );
}
