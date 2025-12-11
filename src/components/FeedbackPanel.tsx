import { CheckCircle2, AlertTriangle, Info } from 'lucide-react';

interface FeedbackPanelProps {
  status: 'correct' | 'incorrect' | 'neutral';
  message: string;
}

export function FeedbackPanel({ status, message }: FeedbackPanelProps) {
  const getStatusConfig = () => {
    switch (status) {
      case 'correct':
        return {
          bgColor: 'bg-emerald-500/20',
          borderColor: 'border-emerald-500/50',
          textColor: 'text-emerald-400',
          icon: CheckCircle2,
          iconColor: 'text-emerald-400',
          label: 'CORRECT FORM',
        };
      case 'incorrect':
        return {
          bgColor: 'bg-red-500/20',
          borderColor: 'border-red-500/50',
          textColor: 'text-red-400',
          icon: AlertTriangle,
          iconColor: 'text-red-400',
          label: 'FORM ERROR',
        };
      default:
        return {
          bgColor: 'bg-blue-500/10',
          borderColor: 'border-blue-500/30',
          textColor: 'text-blue-400',
          icon: Info,
          iconColor: 'text-blue-400',
          label: 'ANALYZING',
        };
    }
  };

  const config = getStatusConfig();
  const Icon = config.icon;

  return (
    <div
      className={`${config.bgColor} ${config.borderColor} border-2 rounded-lg p-6 backdrop-blur-sm transition-all duration-300`}
    >
      <div className="flex items-start gap-4">
        {/* Icon */}
        <div className={`${status === 'correct' ? 'animate-pulse' : ''}`}>
          <Icon className={`w-8 h-8 ${config.iconColor}`} strokeWidth={2.5} />
        </div>

        {/* Content */}
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-2">
            <span className={`text-sm font-medium ${config.textColor} tracking-wide`}>
              {config.label}
            </span>
            {status !== 'neutral' && (
              <div className="flex-1 h-1 bg-gradient-to-r from-current to-transparent opacity-30 rounded" />
            )}
          </div>

          {message && (
            <p className="text-white text-lg mt-2">
              {message}
            </p>
          )}

          {status === 'neutral' && !message && (
            <p className="text-slate-400 text-sm mt-1">
              Waiting for pose detection...
            </p>
          )}
        </div>
      </div>

      {/* Real-time feedback indicator */}
      {status !== 'neutral' && (
        <div className="mt-4 pt-4 border-t border-slate-700/50">
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-400">Real-time ML analysis</span>
            <div className="flex items-center gap-2">
              <div className={`w-1.5 h-1.5 rounded-full ${status === 'correct' ? 'bg-emerald-400' : 'bg-red-400'} animate-pulse`} />
              <span className={config.textColor}>Live feedback active</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
