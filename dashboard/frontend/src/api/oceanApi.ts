import type { Metadata, FieldData, CompareData, MetricsData } from '../types/ocean';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

class OceanApi {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl.replace(/\/+$/, '');
  }

  async getMetadata(): Promise<Metadata> {
    const res = await fetch(`${this.baseUrl}/api/metadata`);
    if (!res.ok) {
      throw new Error(`Failed to fetch metadata (${res.status}): ${await res.text()}`);
    }
    return res.json();
  }

  async getField(date: string, depth: number, mode: string = 'prediction'): Promise<FieldData> {
    const params = new URLSearchParams({
      date,
      depth: depth.toString(),
      mode,
    });
    const res = await fetch(`${this.baseUrl}/api/field?${params.toString()}`);
    if (!res.ok) {
      throw new Error(`Failed to fetch field data (${res.status}): ${await res.text()}`);
    }
    return res.json();
  }

  async getCompare(date: string, depth: number): Promise<CompareData> {
    const params = new URLSearchParams({
      date,
      depth: depth.toString(),
    });
    const res = await fetch(`${this.baseUrl}/api/compare?${params.toString()}`);
    if (!res.ok) {
      throw new Error(`Failed to fetch comparison data (${res.status}): ${await res.text()}`);
    }
    return res.json();
  }

  async getMetrics(): Promise<MetricsData> {
    const res = await fetch(`${this.baseUrl}/api/metrics`);
    if (!res.ok) {
      throw new Error(`Failed to fetch metrics (${res.status}): ${await res.text()}`);
    }
    return res.json();
  }
}

export const oceanApi = new OceanApi();
