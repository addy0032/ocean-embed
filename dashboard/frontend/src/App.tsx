import { useState, useEffect, useCallback, useRef } from 'react';
import { oceanApi } from './api/oceanApi';
import type { Metadata, FieldData, CompareData, MetricsData, ViewMode, DisplayMode } from './types/ocean';
import { ModelStatus } from './components/ModelStatus';
import { OceanScene } from './components/OceanScene';
import { DepthSlice } from './components/DepthSlice';
import { ComparisonView } from './components/ComparisonView';
import { Controls } from './components/Controls';
import { ColorBar } from './components/ColorBar';
import { MetricsPanel } from './components/MetricsPanel';
import { Loader2, AlertCircle, RefreshCw } from 'lucide-react';

export function App() {
  // Metadata & Metrics
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [metaLoading, setMetaLoading] = useState<boolean>(true);
  const [metaError, setMetaError] = useState<string | null>(null);

  // Active User Selections
  const [selectedDate, setSelectedDate] = useState<string>('');
  const [selectedDepth, setSelectedDepth] = useState<number>(100.0);
  const [viewMode, setViewMode] = useState<ViewMode>('prediction');
  const [displayMode, setDisplayMode] = useState<DisplayMode>('3d');

  // Animation & 3D Parameters
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1.0);
  const [verticalExaggeration, setVerticalExaggeration] = useState<number>(3.0);

  // Data Payloads
  const [fieldData, setFieldData] = useState<FieldData | null>(null);
  const [compareData, setCompareData] = useState<CompareData | null>(null);
  const [dataLoading, setDataLoading] = useState<boolean>(false);
  const [dataError, setDataError] = useState<string | null>(null);

  // Custom Colorbar Range Controls
  const [customRange, setCustomRange] = useState<{ min: number; max: number } | null>(null);

  // Request counter to avoid race conditions
  const requestIdRef = useRef<number>(0);

  // 1. Initial Load of Metadata and Evaluation Metrics
  const loadInitialData = useCallback(async () => {
    setMetaLoading(true);
    setMetaError(null);
    try {
      const [meta, met] = await Promise.all([
        oceanApi.getMetadata(),
        oceanApi.getMetrics(),
      ]);
      setMetadata(meta);
      setMetrics(met);

      if (meta.dates && meta.dates.length > 0) {
        setSelectedDate(meta.dates[0]);
      }
      if (meta.depths && meta.depths.length > 0) {
        // Default to 100m if available, otherwise mid depth
        const defaultDepth = meta.depths.includes(100.0) ? 100.0 : meta.depths[0];
        setSelectedDepth(defaultDepth);
      }
    } catch (err: any) {
      console.error('Failed to load initial metadata:', err);
      setMetaError(err.message || 'Failed to connect to OceanEmbed FastAPI backend.');
    } finally {
      setMetaLoading(false);
    }
  }, []);

  useEffect(() => {
    loadInitialData();
  }, [loadInitialData]);

  // 2. Fetch Field or Comparison Data when Date, Depth, or ViewMode changes
  useEffect(() => {
    if (!selectedDate || selectedDepth === undefined) return;

    const currentReqId = ++requestIdRef.current;
    setDataLoading(true);
    setDataError(null);

    const fetchData = async () => {
      try {
        if (viewMode === 'compare') {
          const comp = await oceanApi.getCompare(selectedDate, selectedDepth);
          if (currentReqId === requestIdRef.current) {
            setCompareData(comp);
            setFieldData(null);
            setDataLoading(false);
          }
        } else {
          const field = await oceanApi.getField(selectedDate, selectedDepth, viewMode);
          if (currentReqId === requestIdRef.current) {
            setFieldData(field);
            setCompareData(null);
            setDataLoading(false);
          }
        }
      } catch (err: any) {
        if (currentReqId === requestIdRef.current) {
          console.error('Failed to fetch ocean field data:', err);
          setDataError(err.message || 'Error fetching field data from API.');
          setDataLoading(false);
        }
      }
    };

    fetchData();
  }, [selectedDate, selectedDepth, viewMode]);

  // Derived Min / Max for Colormap
  const activeMin = customRange
    ? customRange.min
    : (viewMode === 'compare'
        ? (compareData?.temp_min ?? 10.0)
        : (viewMode === 'error' ? 0.0 : (fieldData?.min ?? 10.0)));

  const activeMax = customRange
    ? customRange.max
    : (viewMode === 'compare'
        ? (compareData?.temp_max ?? 32.0)
        : (viewMode === 'error' ? (fieldData?.max ?? 4.0) : (fieldData?.max ?? 32.0)));

  // Error boundary / Backend down state
  if (metaLoading) {
    return (
      <div className="min-h-screen bg-slate-950 flex flex-col items-center justify-center p-6 text-slate-200">
        <Loader2 className="w-10 h-10 text-cyan-400 animate-spin mb-4" />
        <h2 className="text-xl font-bold tracking-tight">Connecting to OceanEmbed Engine...</h2>
        <p className="text-sm text-slate-400 mt-1">Opening Zarr stores and initializing subsurface grid.</p>
      </div>
    );
  }

  if (metaError) {
    return (
      <div className="min-h-screen bg-slate-950 flex flex-col items-center justify-center p-6 text-slate-200">
        <div className="max-w-lg bg-slate-900 border border-rose-500/40 rounded-2xl p-6 shadow-2xl text-center flex flex-col items-center">
          <AlertCircle className="w-12 h-12 text-rose-400 mb-3" />
          <h2 className="text-lg font-bold text-white mb-2">Backend Connection Failed</h2>
          <p className="text-xs text-slate-300 font-mono bg-slate-950 p-3 rounded border border-slate-800 mb-4 text-left w-full overflow-x-auto">
            {metaError}
          </p>
          <div className="text-xs text-slate-400 mb-5 text-left">
            <p className="font-semibold text-slate-200 mb-1">Make sure the FastAPI backend is running:</p>
            <code className="bg-slate-950 px-2 py-1 rounded text-cyan-300 block font-mono text-[11px]">
              uv run uvicorn dashboard.backend.main:app --port 8000 --reload
            </code>
          </div>
          <button
            onClick={loadInitialData}
            className="flex items-center gap-2 px-4 py-2 bg-cyan-500 hover:bg-cyan-400 text-slate-950 rounded-lg text-sm font-bold transition shadow-lg shadow-cyan-500/20"
          >
            <RefreshCw className="w-4 h-4" />
            <span>Retry Connection</span>
          </button>
        </div>
      </div>
    );
  }

  const depths = metadata?.depths || [];
  const dates = metadata?.dates || [];
  const latitudes = metadata?.latitudes || [];
  const longitudes = metadata?.longitudes || [];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-4 md:p-6 flex flex-col gap-5 max-w-[1440px] mx-auto">
      {/* 1. Model Header & Spec Badges */}
      <ModelStatus
        model={metadata?.model || null}
        testPeriod={metadata?.test_period || null}
        totalDates={dates.length}
      />

      {/* 2. Interactive Navigation Controls (Dates, Depths, Modes, Speeds) */}
      <Controls
        dates={dates}
        selectedDate={selectedDate}
        onSelectDate={setSelectedDate}
        depths={depths}
        selectedDepth={selectedDepth}
        onSelectDepth={setSelectedDepth}
        viewMode={viewMode}
        onSelectViewMode={(mode) => {
          setViewMode(mode);
          setCustomRange(null); // reset range to auto
        }}
        displayMode={displayMode}
        onSelectDisplayMode={setDisplayMode}
        isPlaying={isPlaying}
        onTogglePlay={() => setIsPlaying(!isPlaying)}
        playbackSpeed={playbackSpeed}
        onSelectSpeed={setPlaybackSpeed}
        verticalExaggeration={verticalExaggeration}
        onChangeVerticalExaggeration={setVerticalExaggeration}
      />

      {/* 3. Main Visualization Stage */}
      <main className="relative flex flex-col gap-4">
        {/* Loading Spinner Overlay */}
        {dataLoading && (
          <div className="absolute inset-0 bg-slate-950/60 backdrop-blur-[2px] z-30 flex items-center justify-center rounded-xl">
            <div className="flex items-center gap-3 bg-slate-900/90 border border-slate-700 px-4 py-2 rounded-lg shadow-2xl text-xs font-mono text-cyan-300">
              <Loader2 className="w-4 h-4 animate-spin text-cyan-400" />
              <span>Loading {viewMode} ({selectedDepth}m)...</span>
            </div>
          </div>
        )}

        {/* Error Alert */}
        {dataError && (
          <div className="p-3 bg-rose-950/60 border border-rose-500/40 rounded-lg text-xs text-rose-200 flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>{dataError}</span>
          </div>
        )}

        {/* View Switcher Output */}
        {viewMode === 'compare' ? (
          compareData ? (
            <ComparisonView
              compareData={compareData}
              latitudes={latitudes}
              longitudes={longitudes}
              selectedDate={selectedDate}
              selectedDepth={selectedDepth}
              minTemp={activeMin}
              maxTemp={activeMax}
            />
          ) : null
        ) : displayMode === '3d' ? (
          <OceanScene
            depths={depths}
            selectedDepth={selectedDepth}
            onSelectDepth={setSelectedDepth}
            activeValues={fieldData?.values || []}
            minVal={activeMin}
            maxVal={activeMax}
            isError={viewMode === 'error'}
            verticalExaggeration={verticalExaggeration}
          />
        ) : (
          <DepthSlice
            values={fieldData?.values || []}
            minVal={activeMin}
            maxVal={activeMax}
            latitudes={latitudes}
            longitudes={longitudes}
            selectedDate={selectedDate}
            selectedDepth={selectedDepth}
            mode={fieldData?.mode || 'prediction'}
            isError={viewMode === 'error'}
          />
        )}
      </main>

      {/* 4. Colormap Legend & Range Adjustment */}
      <ColorBar
        minVal={activeMin}
        maxVal={activeMax}
        onChangeRange={(min, max) => setCustomRange({ min, max })}
        isError={viewMode === 'error'}
      />

      {/* 5. Test Performance Metrics Panel */}
      <MetricsPanel
        metrics={metrics}
        selectedDepth={selectedDepth}
      />

      {/* Footer */}
      <footer className="text-center text-xs text-slate-500 py-3 border-t border-slate-900 flex flex-col md:flex-row items-center justify-between gap-2">
        <div>
          OceanEmbed &copy; 2026 &bull; Smart India Hackathon Demonstration Dashboard
        </div>
        <div className="font-mono text-[11px] text-slate-400">
          Trained Checkpoint: fully_trained_temporal_unet_best_32batch.pt &bull; Target: GLORYS Reanalysis
        </div>
      </footer>
    </div>
  );
}

export default App;
