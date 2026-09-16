export type FieldMode = 'prediction' | 'target' | 'error';
export type ViewMode = 'prediction' | 'target' | 'error' | 'compare';
export type DisplayMode = '3d' | '2d';

export interface GridInfo {
  latitude: number;
  longitude: number;
  depth: number;
  time: number;
}

export interface ModelInfo {
  name: string;
  architecture: string;
  checkpoint: string;
  input_variables: number;
  time_window_days: number;
  depth_levels: number;
  status: string;
}

export interface TestPeriod {
  start: string;
  end: string;
  total_days: number;
}

export interface Metadata {
  title: string;
  dates: string[];
  depths: number[];
  latitude_range: [number, number];
  longitude_range: [number, number];
  latitudes: number[];
  longitudes: number[];
  grid: GridInfo;
  units: string;
  test_period: TestPeriod;
  model: ModelInfo;
}

export interface FieldData {
  date: string;
  depth: number;
  mode: FieldMode;
  min: number | null;
  max: number | null;
  mean: number | null;
  units: string;
  values: (number | null)[][];
}

export interface MetricsRow {
  Depth_m?: number;
  RMSE_C: number;
  MAE_C: number;
  Bias_C: number;
  Correlation: number;
  R2: number;
  N?: number;
}

export interface CompareData {
  date: string;
  depth: number;
  prediction: (number | null)[][];
  target: (number | null)[][];
  error: (number | null)[][];
  temp_min: number;
  temp_max: number;
  error_max: number;
  units: string;
  depth_metrics: MetricsRow | null;
}

export interface MetricsData {
  overall: MetricsRow | null;
  summary: Record<string, number> | null;
  by_depth: MetricsRow[];
}
