import React, { useState } from 'react';
import { DepthSlice } from './DepthSlice';
import type { CompareData } from '../types/ocean';
import { Activity, Layers } from 'lucide-react';

interface ComparisonViewProps {
  compareData: CompareData;
  latitudes: number[];
  longitudes: number[];
  selectedDate: string;
  selectedDepth: number;
  minTemp: number;
  maxTemp: number;
}

export const ComparisonView: React.FC<ComparisonViewProps> = ({
  compareData,
  latitudes,
  longitudes,
  selectedDate,
  selectedDepth,
  minTemp,
  maxTemp,
}) => {
  const [showErrorPanel, setShowErrorPanel] = useState<boolean>(true);
  const m = compareData.depth_metrics;

  return (
    <div className="flex flex-col gap-4">
      {/* Side-by-Side Dual View Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Left: Model Prediction */}
        <DepthSlice
          values={compareData.prediction}
          minVal={minTemp}
          maxVal={maxTemp}
          latitudes={latitudes}
          longitudes={longitudes}
          selectedDate={selectedDate}
          selectedDepth={selectedDepth}
          mode="prediction"
          title="MODEL PREDICTION (Temporal U-Net)"
        />

        {/* Right: GLORYS Reference Target */}
        <DepthSlice
          values={compareData.target}
          minVal={minTemp}
          maxVal={maxTemp}
          latitudes={latitudes}
          longitudes={longitudes}
          selectedDate={selectedDate}
          selectedDepth={selectedDepth}
          mode="target"
          title="GLORYS REFERENCE TARGET"
        />
      </div>

      {/* Optional Absolute Error Slice */}
      {showErrorPanel && (
        <div className="mt-1">
          <DepthSlice
            values={compareData.error}
            minVal={0.0}
            maxVal={compareData.error_max || 4.0}
            latitudes={latitudes}
            longitudes={longitudes}
            selectedDate={selectedDate}
            selectedDepth={selectedDepth}
            mode="error"
            title="ABSOLUTE ERROR (|PREDICTION - GLORYS|)"
            isError={true}
          />
        </div>
      )}

      {/* Synchronized Metrics Bar for this Depth */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4 shadow-lg">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-cyan-400" />
          <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
            Evaluation Metrics at Depth {selectedDepth} m:
          </span>
        </div>

        {m ? (
          <div className="flex flex-wrap items-center gap-6 text-xs font-mono">
            <div className="flex items-center gap-1.5">
              <span className="text-slate-400">RMSE:</span>
              <span className="font-bold text-cyan-300">{m.RMSE_C.toFixed(4)} °C</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="text-slate-400">MAE:</span>
              <span className="font-bold text-cyan-300">{m.MAE_C.toFixed(4)} °C</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="text-slate-400">Bias:</span>
              <span className="font-bold text-cyan-300">{m.Bias_C.toFixed(4)} °C</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="text-slate-400">Correlation:</span>
              <span className="font-bold text-emerald-400">{m.Correlation.toFixed(4)}</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="text-slate-400">R²:</span>
              <span className="font-bold text-emerald-400">{m.R2.toFixed(4)}</span>
            </div>
            {m.N && (
              <div className="flex items-center gap-1.5">
                <span className="text-slate-500">Cells:</span>
                <span className="text-slate-400">{m.N.toLocaleString()}</span>
              </div>
            )}
          </div>
        ) : (
          <div className="text-xs text-slate-500 italic">
            Metrics for depth {selectedDepth}m unavailable in evaluation CSV.
          </div>
        )}

        <button
          onClick={() => setShowErrorPanel(!showErrorPanel)}
          className="flex items-center gap-1.5 text-xs text-slate-300 hover:text-white px-2.5 py-1 bg-slate-800 hover:bg-slate-700 rounded border border-slate-700 transition"
        >
          <Layers className="w-3.5 h-3.5 text-amber-400" />
          <span>{showErrorPanel ? 'Hide Absolute Error' : 'Show Absolute Error'}</span>
        </button>
      </div>
    </div>
  );
};
