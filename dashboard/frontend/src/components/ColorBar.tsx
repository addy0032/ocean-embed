import React, { useId } from 'react';

interface ColorBarProps {
  minVal: number;
  maxVal: number;
  onChangeRange?: (min: number, max: number) => void;
  units?: string;
  isError?: boolean;
}

export const ColorBar: React.FC<ColorBarProps> = ({
  minVal,
  maxVal,
  onChangeRange,
  units = '°C',
  isError = false,
}) => {
  const minInputId = useId();
  const maxInputId = useId();
  // Thermal gradient vs Error gradient
  const gradientCss = isError
    ? 'linear-gradient(to right, rgb(255,255,210), rgb(254,217,118), rgb(254,153,41), rgb(217,72,30), rgb(160,20,50), rgb(100,0,60))'
    : 'linear-gradient(to right, rgb(10,20,70), rgb(24,65,140), rgb(30,130,160), rgb(40,175,130), rgb(160,200,60), rgb(240,190,40), rgb(235,90,30), rgb(200,30,30))';

  const stepCount = 5;
  const ticks = [];
  const range = maxVal - minVal;
  for (let i = 0; i <= stepCount; i++) {
    const val = minVal + (range * (i / stepCount));
    ticks.push(val);
  }

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-lg flex flex-col gap-2">
      <div className="flex items-center justify-between text-xs">
        <span className="font-semibold text-slate-300 uppercase tracking-wider">
          {isError ? 'Absolute Error Scale' : 'Temperature Scale'} ({units})
        </span>
        {onChangeRange && (
          <div className="flex items-center gap-2 text-slate-400 font-mono text-[11px]">
            <span>Range:</span>
            <label htmlFor={minInputId} className="sr-only">Min Temperature</label>
            <input
              id={minInputId}
              type="number"
              value={minVal}
              step={0.5}
              onChange={(e) => onChangeRange(parseFloat(e.target.value) || 0, maxVal)}
              className="w-14 bg-slate-950 border border-slate-700 rounded px-1 py-0.5 text-center text-slate-200"
            />
            <span>to</span>
            <label htmlFor={maxInputId} className="sr-only">Max Temperature</label>
            <input
              id={maxInputId}
              type="number"
              value={maxVal}
              step={0.5}
              onChange={(e) => onChangeRange(minVal, parseFloat(e.target.value) || 30)}
              className="w-14 bg-slate-950 border border-slate-700 rounded px-1 py-0.5 text-center text-slate-200"
            />
          </div>
        )}
      </div>

      {/* Gradient bar */}
      <div
        className="w-full h-4 rounded-md border border-slate-700/80 shadow-inner"
        style={{ background: gradientCss }}
      />

      {/* Numeric Ticks */}
      <div className="flex justify-between text-[10px] font-mono text-slate-400 px-0.5">
        {ticks.map((t, idx) => (
          <span key={idx}>{t.toFixed(1)} {units}</span>
        ))}
      </div>
    </div>
  );
};
