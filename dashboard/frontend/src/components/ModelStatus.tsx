import React from 'react';
import type { ModelInfo, TestPeriod } from '../types/ocean';
import { Waves, Cpu, Clock, Calendar, CheckCircle2 } from 'lucide-react';

interface ModelStatusProps {
  model?: ModelInfo | null;
  testPeriod?: TestPeriod | null;
  totalDates?: number;
}

export const ModelStatus: React.FC<ModelStatusProps> = ({
  model,
  testPeriod,
  totalDates,
}) => {
  return (
    <header className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl flex flex-col md:flex-row md:items-center justify-between gap-6">
      {/* Title and Subtitle */}
      <div className="flex items-start gap-4">
        <div className="p-3 bg-cyan-950 border border-cyan-500/40 rounded-xl text-cyan-400 shadow-lg shadow-cyan-500/10">
          <Waves className="w-8 h-8" />
        </div>
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-black text-white tracking-tight">
              OceanEmbed
            </h1>
            <span className="flex items-center gap-1.5 px-2.5 py-0.5 bg-emerald-950/80 text-emerald-400 border border-emerald-500/30 rounded-full text-xs font-semibold">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span>Model Ready</span>
            </span>
          </div>
          <p className="text-xs text-slate-300 font-medium tracking-wide mt-1">
            Satellite Embedding-Based Subsurface Ocean Temperature Reconstruction
          </p>
          <p className="text-[11px] text-slate-500 mt-0.5">
            Demonstrating 3D Subsurface Ocean Thermocline Predictions over the Arabian Sea & Indian Ocean Domain
          </p>
        </div>
      </div>

      {/* Confirmed Model Metadata Badges */}
      <div className="flex flex-wrap items-center gap-3">
        {/* Model Spec */}
        <div className="flex items-center gap-2 bg-slate-950/80 px-3 py-2 rounded-lg border border-slate-800">
          <Cpu className="w-4 h-4 text-cyan-400" />
          <div className="text-left">
            <div className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold">Architecture</div>
            <div className="text-xs font-bold text-slate-200 font-mono">
              {model?.architecture || 'Temporal U-Net (3D-2D)'}
            </div>
          </div>
        </div>

        {/* Input Variables */}
        <div className="flex items-center gap-2 bg-slate-950/80 px-3 py-2 rounded-lg border border-slate-800">
          <Clock className="w-4 h-4 text-cyan-400" />
          <div className="text-left">
            <div className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold">Input Window</div>
            <div className="text-xs font-bold text-slate-200 font-mono">
              {model ? `${model.input_variables} surface vars × ${model.time_window_days} days` : '7 surface vars × 7 days'}
            </div>
          </div>
        </div>

        {/* Output Depths */}
        <div className="flex items-center gap-2 bg-slate-950/80 px-3 py-2 rounded-lg border border-slate-800">
          <CheckCircle2 className="w-4 h-4 text-cyan-400" />
          <div className="text-left">
            <div className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold">Target Levels</div>
            <div className="text-xs font-bold text-slate-200 font-mono">
              {model ? `${model.depth_levels} Depths (0–1000m)` : '15 Depths (0–1000m)'}
            </div>
          </div>
        </div>

        {/* Test Period */}
        <div className="flex items-center gap-2 bg-slate-950/80 px-3 py-2 rounded-lg border border-slate-800">
          <Calendar className="w-4 h-4 text-cyan-400" />
          <div className="text-left">
            <div className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold">Test Period</div>
            <div className="text-xs font-bold text-slate-200 font-mono">
              {testPeriod ? `${testPeriod.start.slice(0, 4)}–${testPeriod.end.slice(0, 4)} (${totalDates ?? testPeriod.total_days} days)` : `2023–2024 (${totalDates ?? 731} days)`}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};
