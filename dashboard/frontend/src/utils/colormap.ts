/**
 * Scientific colormaps and pixel buffer generation for ocean temperature and error.
 */

export interface RGB {
  r: number;
  g: number;
  b: number;
}

// cmocean thermal-like scientific colormap stops:
// Deep blue -> Teal -> Green -> Yellow -> Orange -> Red -> White
const THERMAL_STOPS: [number, RGB][] = [
  [0.00, { r: 10,  g: 20,  b: 70  }], // deep ocean navy
  [0.15, { r: 24,  g: 65,  b: 140 }], // royal blue
  [0.30, { r: 30,  g: 130, b: 160 }], // cyan / teal
  [0.45, { r: 40,  g: 175, b: 130 }], // ocean green
  [0.60, { r: 160, g: 200, b: 60  }], // yellow-green
  [0.75, { r: 240, g: 190, b: 40  }], // warm golden yellow
  [0.88, { r: 235, g: 90,  b: 30  }], // vibrant orange-red
  [1.00, { r: 200, g: 30,  b: 30  }], // deep crimson
];

// Error colormap: light yellow-green -> bright orange -> dark red -> deep violet
const ERROR_STOPS: [number, RGB][] = [
  [0.00, { r: 255, g: 255, b: 210 }], // near-zero error (light pale yellow)
  [0.20, { r: 254, g: 217, b: 118 }], // yellow
  [0.40, { r: 254, g: 153, b: 41  }], // bright orange
  [0.65, { r: 217, g: 72,  b: 30  }], // red-orange
  [0.85, { r: 160, g: 20,  b: 50  }], // crimson
  [1.00, { r: 100, g: 0,   b: 60  }], // deep violet-purple
];

// Land / invalid cell color: sleek dark slate charcoal
const LAND_COLOR: RGB = { r: 22, g: 28, b: 40 };

function interpolateStops(stops: [number, RGB][], t: number): RGB {
  const clamped = Math.max(0, Math.min(1, t));
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, c0] = stops[i];
    const [t1, c1] = stops[i + 1];
    if (clamped >= t0 && clamped <= t1) {
      const factor = (clamped - t0) / (t1 - t0);
      return {
        r: Math.round(c0.r + (c1.r - c0.r) * factor),
        g: Math.round(c0.g + (c1.g - c0.g) * factor),
        b: Math.round(c0.b + (c1.b - c0.b) * factor),
      };
    }
  }
  return stops[stops.length - 1][1];
}

export function getColor(val: number | null, min: number, max: number, isError: boolean = false): RGB & { a: number } {
  if (val === null || isNaN(val)) {
    return { ...LAND_COLOR, a: 255 };
  }
  const range = max - min || 1.0;
  const normalized = (val - min) / range;
  const rgb = interpolateStops(isError ? ERROR_STOPS : THERMAL_STOPS, normalized);
  return { ...rgb, a: 255 };
}

/**
 * Converts 2D grid [101, 241] to ImageData [241 x 101].
 * Inverts row order so North (latitude 30°N, index 100) is at the TOP (y = 0).
 */
export function gridToImageData(
  values: (number | null)[][],
  minVal: number,
  maxVal: number,
  isError: boolean = false
): ImageData | null {
  if (!values || values.length === 0) return null;
  const numRows = values.length;     // 101
  const numCols = values[0].length;  // 241

  const imgData = new ImageData(numCols, numRows);
  const data = imgData.data;

  for (let r = 0; r < numRows; r++) {
    // Invert row: r=0 (5.0°N) -> y = numRows - 1 - r (bottom)
    //             r=100 (30.0°N) -> y = 0 (top)
    const y = numRows - 1 - r;
    const row = values[r];
    for (let x = 0; x < numCols; x++) {
      const val = row[x];
      const pixelIdx = (y * numCols + x) * 4;
      const c = getColor(val, minVal, maxVal, isError);
      data[pixelIdx] = c.r;
      data[pixelIdx + 1] = c.g;
      data[pixelIdx + 2] = c.b;
      data[pixelIdx + 3] = c.a;
    }
  }
  return imgData;
}

/**
 * Creates an HTMLCanvasElement from 2D values for use as a Three.js Texture or 2D display.
 */
export function gridToCanvas(
  values: (number | null)[][],
  minVal: number,
  maxVal: number,
  isError: boolean = false
): HTMLCanvasElement | null {
  const imgData = gridToImageData(values, minVal, maxVal, isError);
  if (!imgData) return null;

  const canvas = document.createElement('canvas');
  canvas.width = imgData.width;
  canvas.height = imgData.height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  ctx.putImageData(imgData, 0, 0);
  return canvas;
}
