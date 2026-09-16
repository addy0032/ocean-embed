import React, { useEffect, useId } from 'react';
import type { ViewMode, DisplayMode } from '../types/ocean';
import { Play, Pause, ChevronLeft, ChevronRight, Sliders, Box, Map, Layers } from 'lucide-react';

interface ControlsProps {
  dates: string[];
  selectedDate: string;
  onSelectDate: (date: string) => void;
  depths: number[];
  selectedDepth: number;
  onSelectDepth: (depth: number) => void;
  viewMode: ViewMode;
  onSelectViewMode: (mode: ViewMode) => void;
  displayMode: DisplayMode;
  onSelectDisplayMode: (mode: DisplayMode) => void;
  isPlaying: boolean;
  onTogglePlay: () => void;
  playbackSpeed: number;
  onSelectSpeed: (speed: number) => void;
  verticalExaggeration: number;
  onChangeVerticalExaggeration: (val: number) => void;
}

export const Controls: React.FC<ControlsProps> = ({
  dates,
  selectedDate,
  onSelectDate,
  depths,
  selectedDepth,
  onSelectDepth,
  viewMode,
  onSelectViewMode,
  displayMode,
  onSelectDisplayMode,
  isPlaying,
  onTogglePlay,
  playbackSpeed,
  onSelectSpeed,
  verticalExaggeration,
  onChangeVerticalExaggeration,
}) => {
  const currentDateIdx = dates.indexOf(selectedDate);
  const dateSliderId = useId();
  const vertExagSliderId = useId();

  // Format date nicely (e.g. "15 August 2023")
  const formattedDate = React.useMemo(() => {
    if (!selectedDate) return '';
    try {
      const [y, m, d] = selectedDate.split('-').map(Number);
      const dateObj = new Date(Date.UTC(y, m - 1, d));
      return dateObj.toLocaleDateString('en-US', {
        day: 'numeric',
        month: 'long',
        year: 'numeric',
        timeZone: 'UTC',
      });
    } catch {
      return selectedDate;
    }
  }, [selectedDate]);

  const handleStep = (direction: 'prev' | 'next') => {
    if (dates.length === 0) return;
    let nextIdx = currentDateIdx + (direction === 'next' ? 1 : -1);
    if (nextIdx < 0) nextIdx = dates.length - 1;
    if (nextIdx >= dates.length) nextIdx = 0;
    onSelectDate(dates[nextIdx]);
  };

  // Playback timer effect
  useEffect(() => {
    if (!isPlaying || dates.length === 0) return;

    // interval in ms based on playbackSpeed (1x = 600ms per day)
    const intervalMs = Math.round(600 / playbackSpeed);

    const timer = setInterval(() => {
      let nextIdx = currentDateIdx + 1;
      if (nextIdx >= dates.length) nextIdx = 0;
      onSelectDate(dates[nextIdx]);
    }, intervalMs);

    return () => clearInterval(timer);
  }, [isPlaying, dates, currentDateIdx, playbackSpeed, onSelectDate]);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl flex flex-col gap-5">
      {/* Top Row: Mode Pickers & View Toggle */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-800">
        {/* Field Mode Toggle */}
        <div className="flex items-center gap-1.5 bg-slate-950 p-1 rounded-lg border border-slate-800">
          <span className="text-[11px] font-semibold text-slate-400 px-2 uppercase tracking-wider">Mode:</span>
          {(['prediction', 'target', 'error', 'compare'] as ViewMode[]).map((mode) => {
            const isSelected = viewMode === mode;
            let label = mode.toUpperCase();
            if (mode === 'prediction') label = 'MODEL';
            if (mode === 'target') label = 'GLORYS';
            if (mode === 'compare') label = 'COMPARE';

            return (
              <button
                key={mode}
                onClick={() => onSelectViewMode(mode)}
                className={`px-3 py-1.5 rounded text-xs font-semibold tracking-wide transition ${
                  isSelected
                    ? 'bg-cyan-500 text-slate-950 shadow-md font-bold'
                    : 'text-slate-300 hover:text-white hover:bg-slate-800'
                }`}
              >
                {label}
              </button>
            );
          })}
        </div>

        {/* View Switcher (3D vs 2D) - available unless compare mode */}
        {viewMode !== 'compare' && (
          <div className="flex items-center gap-1.5 bg-slate-950 p-1 rounded-lg border border-slate-800">
            <button
              onClick={() => onSelectDisplayMode('3d')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-semibold transition ${
                displayMode === '3d'
                  ? 'bg-slate-800 text-cyan-300 border border-cyan-500/40'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Box className="w-3.5 h-3.5" />
              <span>3D Volume</span>
            </button>
            <button
              onClick={() => onSelectDisplayMode('2d')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-semibold transition ${
                displayMode === '2d'
                  ? 'bg-slate-800 text-cyan-300 border border-cyan-500/40'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Map className="w-3.5 h-3.5" />
              <span>2D Slice Map</span>
            </button>
          </div>
        )}

        {/* Vertical Exaggeration Slider (for 3D view) */}
        {viewMode !== 'compare' && displayMode === '3d' && (
          <div className="flex items-center gap-2 text-xs text-slate-300 bg-slate-950/70 px-3 py-1.5 rounded-lg border border-slate-800">
            <Sliders className="w-3.5 h-3.5 text-cyan-400" />
            <label htmlFor={vertExagSliderId} className="cursor-pointer">Vertical Exag: <span className="font-bold text-cyan-300">{verticalExaggeration.toFixed(1)}x</span></label>
            <input
              id={vertExagSliderId}
              type="range"
              min="1.0"
              max="8.0"
              step="0.5"
              value={verticalExaggeration}
              onChange={(e) => onChangeVerticalExaggeration(parseFloat(e.target.value))}
              className="w-20 accent-cyan-400 cursor-pointer"
            />
          </div>
        )}
      </div>

      {/* Middle Row: Date Navigation & Playback Slider */}
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold tracking-wider text-slate-400 uppercase">Target Date:</span>
            <span className="text-base font-bold text-white px-2.5 py-0.5 bg-slate-800 rounded border border-slate-700 font-mono">
              {formattedDate}
            </span>
            <span className="text-xs text-slate-500 font-mono">({selectedDate})</span>
          </div>

          {/* Play / Step / Speed Controls */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => handleStep('prev')}
              className="p-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded border border-slate-700 transition"
              title="Previous Day"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>

            <button
              onClick={onTogglePlay}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-bold transition ${
                isPlaying
                  ? 'bg-amber-500 hover:bg-amber-400 text-slate-950'
                  : 'bg-cyan-500 hover:bg-cyan-400 text-slate-950'
              }`}
            >
              {isPlaying ? <Pause className="w-4 h-4 fill-current" /> : <Play className="w-4 h-4 fill-current" />}
              <span>{isPlaying ? 'PAUSE' : 'PLAY'}</span>
            </button>

            <button
              onClick={() => handleStep('next')}
              className="p-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded border border-slate-700 transition"
              title="Next Day"
            >
              <ChevronRight className="w-4 h-4" />
            </button>

            {/* Speed Selector */}
            <div className="flex items-center gap-1 ml-2 bg-slate-950 p-0.5 rounded border border-slate-800">
              {[0.5, 1, 2, 5].map((speed) => (
                <button
                  key={speed}
                  onClick={() => onSelectSpeed(speed)}
                  className={`px-1.5 py-0.5 text-[11px] font-mono rounded ${
                    playbackSpeed === speed
                      ? 'bg-slate-800 text-cyan-300 font-bold'
                      : 'text-slate-500 hover:text-slate-300'
                  }`}
                >
                  {speed}x
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Date Range Slider */}
        <div className="flex items-center gap-3">
          <span className="text-[10px] font-mono text-slate-500">{dates[0]}</span>
          <label htmlFor={dateSliderId} className="sr-only">Date Slider</label>
          <input
            id={dateSliderId}
            type="range"
            min={0}
            max={Math.max(0, dates.length - 1)}
            value={currentDateIdx >= 0 ? currentDateIdx : 0}
            onChange={(e) => onSelectDate(dates[parseInt(e.target.value, 10)])}
            className="flex-1 accent-cyan-400 cursor-pointer h-2 bg-slate-800 rounded-lg appearance-none"
          />
          <span className="text-[10px] font-mono text-slate-500">{dates[dates.length - 1]}</span>
        </div>
      </div>

      {/* Bottom Row: Depth Level Selector */}
      <div className="flex flex-col gap-2 pt-2 border-t border-slate-800">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-cyan-400" />
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Subsurface Depth:</span>
            <span className="text-base font-extrabold text-cyan-300 bg-cyan-950/60 px-2.5 py-0.5 rounded border border-cyan-500/40 font-mono">
              Depth: {selectedDepth} m
            </span>
          </div>
          <span className="text-xs text-slate-500">15 Standard Levels (0m - 1000m)</span>
        </div>

        {/* 15 Depth Buttons */}
        <div className="flex flex-wrap items-center gap-1.5">
          {depths.map((d) => {
            const isSelected = d === selectedDepth;
            return (
              <button
                key={d}
                onClick={() => onSelectDepth(d)}
                className={`px-2.5 py-1 rounded text-xs font-mono transition ${
                  isSelected
                    ? 'bg-cyan-500 text-slate-950 font-bold shadow-lg shadow-cyan-500/20'
                    : 'bg-slate-800/80 hover:bg-slate-700 text-slate-300 border border-slate-700/60'
                }`}
              >
                {d}m
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};
