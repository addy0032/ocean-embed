# OceanEmbed: Satellite Embedding-Based Subsurface Ocean Temperature Reconstruction

Interactive 3D and 2D visualization dashboard for subsurface ocean temperature reconstruction from satellite surface observations using a Temporal U-Net (3D-2D).

Built for Smart India Hackathon (SIH) demonstration.

---

## System Architecture

```text
Trained Model Checkpoint (.pt)
        │
        ▼
scripts/ml/generate_predictions.py
        │
        ▼
evaluation/predictions.zarr  [time: 731, depth: 15, latitude: 101, longitude: 241]
        │
   ┌────┴──────────────────────────┐
   ▼                               ▼
predictions.zarr          y_target_025deg.zarr (GLORYS)
   │                               │
   └───────────────┬───────────────┘
                   ▼
       FastAPI Backend (Port 8000)
       - GET /api/metadata
       - GET /api/field
       - GET /api/compare
       - GET /api/metrics
                   │
                   ▼
     React + Three.js Frontend (Port 5173)
       - 15-Layer 3D Mesh Volume (R3F + OrbitControls)
       - 2D Spatial Depth Slice Map
       - Side-by-Side Model vs GLORYS Comparison
       - Time Playback Animation (0.5x – 5x)
       - Quantitative Evaluation Metrics Panel
```

---

## Dataset & Model Specifications

* **Model Architecture**: Temporal U-Net (3D Conv temporal-spatial encoder + 2D U-Net spatial decoder)
* **Checkpoint**: `checkpoints/fully_trained_temporal_unet_best_32batch.pt`
* **Input**: 7 surface variables × 7 days (`[B, 7, 7, 101, 241]`)
  * SST, SSS, ADT (Sea-level), U Current, V Current, U Wind, V Wind
* **Output**: 15 subsurface depth levels (`[B, 15, 101, 241]`)
  * 0m, 5m, 10m, 20m, 30m, 50m, 75m, 100m, 125m, 150m, 200m, 300m, 500m, 700m, 1000m
* **Spatial Domain**: Arabian Sea & Northern Indian Ocean (5.0°N to 30.0°N, 45.0°E to 105.0°E, 0.25° resolution)
* **Independent Test Period**: `2023-01-01` to `2024-12-31` (731 days, completely independent of training)
* **Target Reference**: GLORYS12V1 ocean reanalysis

---

## How to Run the System

### Step 1 — Generate Offline Predictions (One-time)

Run the GPU-accelerated prediction generation script to compute and inverse-normalize the full 2023–2024 test dataset into `evaluation/predictions.zarr`:

```bash
uv run python scripts/ml/generate_predictions.py
```

This step:
1. Loads the trained checkpoint and normalization statistics.
2. Runs batched inference on all test dates.
3. Inverse-normalizes raw predictions back to °C without masking raw arrays.
4. Writes `evaluation/predictions.zarr` with chunking `{"time": 1, "depth": -1, "latitude": -1, "longitude": -1}`.
5. Runs automated validation checks on dimensions, coordinates, and value ranges.

> **Note**: The backend never executes PyTorch inference. All inference is pre-computed offline.

---

### Step 2 — Start the FastAPI Backend

Start the lightweight FastAPI server to serve metadata, field slices, and metrics:

```bash
uv run uvicorn dashboard.backend.main:app --port 8000 --reload
```

Backend API documentation is available at `http://localhost:8000/docs`.

---

### Step 3 — Start the Frontend Development Server

In a new terminal, install frontend dependencies (if not already done) and start the Vite dev server:

```bash
cd dashboard/frontend
pnpm install
pnpm dev
```

---

### Step 4 — Open Browser

Open your browser to:

```text
http://localhost:5173
```

---

## Dashboard Capabilities

1. **3D Ocean Volume View**:
   - 15 layered horizontal planes along Y axis corresponding to depths from 0m to 1000m.
   - Depth increases downward along negative Y.
   - Visual vertical depth indicator with tick markers and active sphere.
   - Configurable vertical exaggeration (1x to 8x) to clearly reveal thermocline structures.
   - Interactive orbit rotation, zoom, pan, and a "Reset View" button.

2. **2D Depth Slice Map**:
   - High-contrast scientific canvas with latitude on Y axis and longitude on X axis.
   - Interactive mouse hover displaying exact Lat, Lon, and Temperature in °C.
   - Preserves bathymetric/land validity mask (masked cells shown in slate).

3. **Prediction / GLORYS / Error Modes**:
   - **MODEL**: Reconstructed subsurface temperatures predicted by Temporal U-Net.
   - **GLORYS**: GLORYS reanalysis reference target.
   - **ERROR**: Absolute reconstruction error ($|\text{Prediction} - \text{Target}|$) in °C.

4. **Side-by-Side Comparison Mode**:
   - Dual synchronized panels (MODEL vs GLORYS) sharing an identical color scale.
   - Optional absolute error panel.
   - Live metrics summary for the selected depth level.

5. **Time Animation Controller**:
   - Play/Pause animation stepping through all 731 days of the 2023–2024 test period.
   - Speed selection: 0.5x, 1x, 2x, 5x.
   - Slider and step-forward/backward buttons with formatted date badge (`DD MMMM YYYY`).

6. **Quantitative Evaluation Metrics**:
   - Displays confirmed test set performance (overall RMSE = 0.7229 °C, Correlation = 0.9956, R² = 0.9913).
   - Dynamic depth-wise metrics card updating with selected depth level.
