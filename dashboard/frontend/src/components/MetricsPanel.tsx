import type { MetricsData } from '../types/ocean';
import { BarChart3, Target, CheckCircle2 } from 'lucide-react';

interface MetricsPanelProps {
  metrics: MetricsData | null;
  selectedDepth: number;
}

export const MetricsPanel: React.FC<MetricsPanelProps> = ({
  metrics,
  selectedDepth,
}) => {
  if (!metrics) return null;

  const overall = metrics.overall;
  const currentDepthMetrics = metrics.by_depth.find(
    (row) => row.Depth_m !== undefined && Math.abs(row.Depth_m - selectedDepth) < 0.1
  );

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-cyan-400" />
          <h3 className="text-sm font-semibold text-slate-200 tracking-wide uppercase">
            Test Set Evaluation Performance (2023–2024 Independent Test)
          </h3>
        </div>
        <div className="flex items-center gap-1.5 text-xs text-emerald-400 font-medium">
          <CheckCircle2 className="w-3.5 h-3.5" />
          <span>Verified Evaluation Pipeline</span>
        </div>
      </div>

      {/* Two Columns: Overall Performance vs Current Depth Performance */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Overall Test Performance */}
        <div className="bg-slate-950/70 border border-slate-800/80 rounded-lg p-3.5 flex flex-col gap-2.5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-300">Overall Test Performance</span>
            <span className="text-[10px] text-slate-500 font-mono">
              {overall?.N ? `${overall.N.toLocaleString()} points` : 'All depths & dates'}
            </span>
          </div>

          <div className="grid grid-cols-3 gap-2">
            <div className="bg-slate-900/90 rounded p-2 border border-slate-800">
              <div className="text-[10px] text-slate-400 font-medium">RMSE</div>
              <div className="text-base font-bold text-cyan-300 font-mono">
                {overall ? `${overall.RMSE_C.toFixed(4)} °C` : '--'}
              </div>
            </div>
            <div className="bg-slate-900/90 rounded p-2 border border-slate-800">
              <div className="text-[10px] text-slate-400 font-medium">MAE</div>
              <div className="text-base font-bold text-cyan-300 font-mono">
                {overall ? `${overall.MAE_C.toFixed(4)} °C` : '--'}
              </div>
            </div>
            <div className="bg-slate-900/90 rounded p-2 border border-slate-800">
              <div className="text-[10px] text-slate-400 font-medium">Bias</div>
              <div className="text-base font-bold text-cyan-300 font-mono">
                {overall ? `${overall.Bias_C.toFixed(4)} °C` : '--'}
              </div>
            </div>
            <div className="bg-slate-900/90 rounded p-2 border border-slate-800">
              <div className="text-[10px] text-slate-400 font-medium">Correlation</div>
              <div className="text-base font-bold text-emerald-400 font-mono">
                {overall ? overall.Correlation.toFixed(4) : '--'}
              </div>
            </div>
            <div className="bg-slate-900/90 rounded p-2 border border-slate-800">
              <div className="text-[10px] text-slate-400 font-medium">R² Score</div>
              <div className="text-base font-bold text-emerald-400 font-mono">
                {overall ? overall.R2.toFixed(4) : '--'}
              </div>
            </div>
            <div className="bg-slate-900/90 rounded p-2 border border-slate-800">
              <div className="text-[10px] text-slate-400 font-medium">Status</div>
              <div className="text-xs font-bold text-cyan-400 font-mono pt-1">
                Converged
              </div>
            </div>
          </div>
        </div>

        {/* Current Depth Performance */}
        <div className="bg-slate-950/70 border border-slate-800/80 rounded-lg p-3.5 flex flex-col gap-2.5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <Target className="w-3.5 h-3.5 text-cyan-400" />
              <span className="text-xs font-semibold text-slate-300">
                Depth Level Performance: <span className="text-cyan-300 font-bold">{selectedDepth} m</span>
              </span>
            </div>
            <span className="text-[10px] text-slate-500 font-mono">
              {currentDepthMetrics?.N ? `${currentDepthMetrics.N.toLocaleString()} valid cells` : ''}
            </span>
          </div>

          <div className="grid grid-cols-3 gap-2">
            <div className="bg-slate-900/90 rounded p-2 border border-slate-800">
              <div className="text-[10px] text-slate-400 font-medium">Depth RMSE</div>
              <div className="text-base font-bold text-cyan-300 font-mono">
                {currentDepthMetrics ? `${currentDepthMetrics.RMSE_C.toFixed(4)} °C` : '--'}
              </div>
            </div>
            <div className="bg-slate-900/90 rounded p-2 border border-slate-800">
              <div className="text-[10px] text-slate-400 font-medium">Depth MAE</div>
              <div className="text-base font-bold text-cyan-300 font-mono">
                {currentDepthMetrics ? `${currentDepthMetrics.MAE_C.toFixed(4)} °C` : '--'}
              </div>
            </div>
            <div className="bg-slate-900/90 rounded p-2 border border-slate-800">
              <div className="text-[10px] text-slate-400 font-medium">Depth Bias</div>
              <div className="text-base font-bold text-cyan-300 font-mono">
                {currentDepthMetrics ? `${currentDepthMetrics.Bias_C.toFixed(4)} °C` : '--'}
              </div>
            </div>
            <div className="bg-slate-900/90 rounded p-2 border border-slate-800">
              <div className="text-[10px] text-slate-400 font-medium">Correlation</div>
              <div className="text-base font-bold text-emerald-400 font-mono">
                {currentDepthMetrics ? currentDepthMetrics.Correlation.toFixed(4) : '--'}
              </div>
            </div>
            <div className="bg-slate-900/90 rounded p-2 border border-slate-800">
              <div className="text-[10px] text-slate-400 font-medium">R² Score</div>
              <div className="text-base font-bold text-emerald-400 font-mono">
                {currentDepthMetrics ? currentDepthMetrics.R2.toFixed(4) : '--'}
              </div>
            </div>
            <div className="bg-slate-900/90 rounded p-2 border border-slate-800">
              <div className="text-[10px] text-slate-400 font-medium">Target Ground</div>
              <div className="text-xs font-bold text-slate-300 font-mono pt-1">
                GLORYS
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
