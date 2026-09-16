import React, { useRef, useEffect, useState, useMemo } from 'react';
import { gridToCanvas } from '../utils/colormap';
import type { FieldMode } from '../types/ocean';

interface DepthSliceProps {
  values: (number | null)[][];
  minVal: number;
  maxVal: number;
  latitudes: number[];
  longitudes: number[];
  selectedDate: string;
  selectedDepth: number;
  mode: FieldMode;
  title?: string;
  units?: string;
  isError?: boolean;
}

export const DepthSlice: React.FC<DepthSliceProps> = ({
  values,
  minVal,
  maxVal,
  latitudes,
  longitudes,
  selectedDate,
  selectedDepth,
  mode,
  title,
  units = '°C',
  isError = false,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoverInfo, setHoverInfo] = useState<{
    lat: number;
    lon: number;
    val: number | null;
    x: number;
    y: number;
  } | null>(null);

  const numRows = values.length;
  const numCols = values[0]?.length || 0;

  // Draw colormapped grid onto canvas
  useEffect(() => {
    if (!canvasRef.current || !values || values.length === 0) return;
    const offscreen = gridToCanvas(values, minVal, maxVal, isError);
    if (!offscreen) return;

    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;

    canvasRef.current.width = offscreen.width;
    canvasRef.current.height = offscreen.height;
    ctx.drawImage(offscreen, 0, 0);
  }, [values, minVal, maxVal, isError]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || numRows === 0 || numCols === 0) return;

    const rect = canvas.getBoundingClientRect();
    const clientX = e.clientX - rect.left;
    const clientY = e.clientY - rect.top;

    const normX = Math.max(0, Math.min(1, clientX / rect.width));
    const normY = Math.max(0, Math.min(1, clientY / rect.height));

    const colIdx = Math.min(numCols - 1, Math.floor(normX * numCols));
    // canvas y=0 is North (lat row index = numRows - 1)
    const rowIdx = Math.min(numRows - 1, Math.floor((1 - normY) * numRows));

    const val = values[rowIdx]?.[colIdx] ?? null;
    const lat = latitudes[rowIdx] ?? 0;
    const lon = longitudes[colIdx] ?? 0;

    setHoverInfo({
      lat,
      lon,
      val,
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    });
  };

  const handleMouseLeave = () => {
    setHoverInfo(null);
  };

  const latTicks = useMemo(() => {
    if (!latitudes || latitudes.length === 0) return [];
    return [5, 10, 15, 20, 25, 30];
  }, [latitudes]);

  const lonTicks = useMemo(() => {
    if (!longitudes || longitudes.length === 0) return [];
    return [45, 55, 65, 75, 85, 95, 105];
  }, [longitudes]);

  return (
    <div className="relative flex flex-col bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-xl select-none" ref={containerRef}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div>
          <span className="text-xs font-semibold tracking-wider uppercase text-cyan-400">
            {title || `${mode.toUpperCase()} 2D SLICE`}
          </span>
          <h3 className="text-sm font-medium text-slate-200">
            Depth: <span className="font-bold text-white">{selectedDepth} m</span> &bull; {selectedDate}
          </h3>
        </div>
        <div className="flex items-center gap-2 text-xs text-slate-400">
          <span>Range: {minVal.toFixed(1)} to {maxVal.toFixed(1)} {units}</span>
        </div>
      </div>

      {/* Main Map Box */}
      <div className="relative flex items-center justify-center p-2 bg-slate-950/70 rounded-lg border border-slate-800/80">
        {/* Y Axis Labels (Latitude) */}
        <div className="absolute left-1 top-2 bottom-6 flex flex-col justify-between text-[10px] font-mono text-slate-500 pointer-events-none select-none">
          {latTicks.slice().reverse().map((lat) => (
            <span key={lat}>{lat}°N</span>
          ))}
        </div>

        {/* Canvas */}
        <div className="relative ml-8 mb-5 overflow-hidden rounded border border-slate-700/50">
          <canvas
            ref={canvasRef}
            className="w-full h-auto cursor-crosshair block"
            style={{
              aspectRatio: '241 / 101',
              imageRendering: 'pixelated',
            }}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
          />

          {/* Tooltip Overlay */}
          {hoverInfo && (
            <div
              className="absolute pointer-events-none z-20 px-2.5 py-1.5 bg-slate-950/95 border border-cyan-500/50 rounded shadow-2xl text-[11px] font-mono text-slate-200 transition-transform -translate-x-1/2 -translate-y-full"
              style={{
                left: `${hoverInfo.x}px`,
                top: `${hoverInfo.y - 8}px`,
              }}
            >
              <div className="font-bold text-cyan-300">
                {hoverInfo.val !== null ? `${hoverInfo.val.toFixed(2)} ${units}` : 'Land / Masked'}
              </div>
              <div className="text-slate-400 text-[10px]">
                {hoverInfo.lat.toFixed(2)}°N, {hoverInfo.lon.toFixed(2)}°E
              </div>
            </div>
          )}
        </div>

        {/* X Axis Labels (Longitude) */}
        <div className="absolute bottom-1 left-9 right-3 flex justify-between text-[10px] font-mono text-slate-500 pointer-events-none select-none">
          {lonTicks.map((lon) => (
            <span key={lon}>{lon}°E</span>
          ))}
        </div>
      </div>
    </div>
  );
};
