# OceanEmbed Dashboard Guide

This directory contains the FastAPI backend and Vite + React + Three.js frontend for the OceanEmbed interactive demonstration dashboard.

## Directory Layout

```text
dashboard/
├── backend/
│   ├── main.py          # FastAPI application with REST endpoints
│   └── config.py        # Zarr dataset paths and server configuration
│
└── frontend/
    ├── src/
    │   ├── api/
    │   │   └── oceanApi.ts        # REST API client
    │   ├── components/
    │   │   ├── OceanScene.tsx     # 3D 15-layer ocean scene (Three.js / R3F)
    │   │   ├── DepthSlice.tsx     # 2D spatial canvas map
    │   │   ├── ComparisonView.tsx # Side-by-side Model vs GLORYS
    │   │   ├── Controls.tsx       # Date, depth, animation & mode controls
    │   │   ├── ColorBar.tsx       # Scientific colormap & range controls
    │   │   ├── MetricsPanel.tsx   # Depth & overall performance metrics
    │   │   └── ModelStatus.tsx    # Header & verified architecture status
    │   ├── types/
    │   │   └── ocean.ts           # TypeScript interfaces
    │   ├── utils/
    │   │   └── colormap.ts        # cmocean-thermal and error colormaps
    │   ├── App.tsx
    │   └── main.tsx
    ├── package.json
    └── vite.config.ts
```

## Quick Start

1. Ensure predictions have been generated:
   ```bash
   uv run python scripts/ml/generate_predictions.py
   ```

2. Start the FastAPI backend:
   ```bash
   uv run uvicorn dashboard.backend.main:app --port 8000 --reload
   ```

3. Start the React frontend:
   ```bash
   cd dashboard/frontend
   pnpm install
   pnpm dev
   ```

4. Navigate to `http://localhost:5173`.
