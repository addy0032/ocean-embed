# OceanEmbed Data Directory Specification & Volume Reference

This document provides a comprehensive reference for the oceanographic datasets used in **OceanEmbed**, including the raw datasets downloaded from the **Copernicus Marine Environment Monitoring Service (CMEMS)** and the **Copernicus Climate Change Service (C3S)**, the regridded and harmonized intermediate products, and the model-ready input/target Zarr stores.

It details storage volumes (compressed on-disk size vs. fully expanded uncompressed array size), data compression ratios, coordinate grids, variable schemas, and the cross-environment distribution between the Windows workspace and the WSL2 environment.

---

## Table of Contents

1. [Executive Summary & High-Level Metrics](#1-executive-summary--high-level-metrics)
2. [Dual Environment Storage Architecture (Windows vs. WSL)](#2-dual-environment-storage-architecture-windows-vs-wsl)
3. [Raw Data Sources & Copernicus Products](#3-raw-data-sources--copernicus-products)
   - [Sea Surface Height & Geostrophic Currents (DUACS ADT/UGOS/VGOS)](#31-sea-surface-height--geostrophic-currents-duacs)
   - [Sea Surface Salinity (Multi-Observation SSS)](#32-sea-surface-salinity-multi-observation-sss)
   - [Sea Surface Temperature (C3S Ultra-High Resolution SST)](#33-sea-surface-temperature-c3s-l4-sst)
   - [Ocean Surface Wind (Scatterometer Hourly Blended Wind)](#34-ocean-surface-wind-hourly-scatterometer)
   - [Subsurface Ocean Physics (GLORYS12V1 Reanalysis)](#35-subsurface-ocean-physics-glorys12v1-reanalysis)
4. [Raw Data Volume & Compression Analysis](#4-raw-data-volume--compression-analysis)
5. [Processed Data Pipeline & Catalog](#5-processed-data-pipeline--catalog)
   - [Reference Grid & Master Land/Sea Mask](#51-reference-grid--master-landsea-mask)
   - [Regridded Single-Variable Stores](#52-regridded-single-variable-stores)
   - [Harmonized Aligned Datasets (WSL)](#53-harmonized-aligned-datasets-wsl)
   - [Production Machine Learning Zarr Stores (Windows & WSL)](#54-production-machine-learning-zarr-stores)
6. [Processed Data Volume & Compression Analysis](#6-processed-data-volume--compression-analysis)
7. [Comprehensive Parameter & Variable Dictionary](#7-comprehensive-parameter--variable-dictionary)

---

## 1. Executive Summary & High-Level Metrics

The dataset covers a **10-year span (2015–2024)** across the **Northern Indian Ocean, Arabian Sea, and Bay of Bengal** ($5.0^\circ\text{N} - 30.0^\circ\text{N}$, $45.0^\circ\text{E} - 105.0^\circ\text{E}$). The primary objective is to reconstruct 3D subsurface temperature profiles (0 to 1,000 meters across 15 standard oceanographic depth layers) from multi-sensor satellite surface observations using deep learning.

| Metric | Raw Downloads | Processed & ML Datasets (Unified) |
| :--- | :--- | :--- |
| **Total Temporal Range** | 10 Years (2015-01-01 to 2024-12-31) | 10 Years (2015-01-01 to 2024-12-31, 3,653 days) |
| **Time Steps** | 3,653 daily steps (87,673 steps for Wind) | 3,653 uniform daily steps |
| **Spatial Grid** | Multi-resolution ($0.05^\circ, 0.083^\circ, 0.125^\circ$) | Uniform $0.25^\circ \times 0.25^\circ$ grid ($101 \times 241$) |
| **Storage Format** | Zarr Format 3 / Directory Store | Zarr Format 3 (compressed with Zstandard `zstd`) |
| **Total Compressed On-Disk Size** | **~40.91 GB** (41,895 MB) | **~12.28 GB** (Windows + WSL unique stores) |
| **Total Uncompressed Array Size** | **~366.86 GB** (375,667 MB) | **~24.58 GB** (expanded memory footprints) |
| **Overall Compression Savings** | **8.97× compression** (88.85% space saved) | **2.0× to 2.7× compression** |

---

## 2. Dual Environment Storage Architecture (Windows vs. WSL)

Because oceanographic data processing involves high-throughput I/O under Linux (xESMF conservative regridding, Dask multiprocessing) alongside Windows development, datasets are distributed across two paths in the project:

- **Windows Project Path**: `c:\Data\sih-26\ps-66\data\`
- **WSL Project Path**: `\\wsl.localhost\Ubuntu\home\addy\projects\sih-26\ps-66\data\` (native WSL: `/home/addy/projects/sih-26/ps-66/data/`)

### File Location Breakdown

```
ps-66/data/
├── raw/                                           [WINDOWS ONLY: 40.91 GB]
│   ├── glorys/ (10 annual Zarr stores)            ── 18.01 GB on-disk (212.64 GB uncompressed)
│   ├── sealevel/ (DUACS ADT/UGOS/VGOS Zarr)       ──  1.06 GB on-disk (  7.95 GB uncompressed)
│   ├── sss/ (Multi-Obs MY & NRT Zarrs)            ──  0.45 GB on-disk (  2.65 GB uncompressed)
│   ├── sst/ (C3S 0.05° L4 SST Zarr)               ──  1.20 GB on-disk ( 16.42 GB uncompressed)
│   └── wind/ (Hourly 0.125° Scatterometer Zarr)   ── 20.20 GB on-disk (127.20 GB uncompressed)
│
└── processed/                                     [WINDOWS & WSL DISTRIBUTED]
    ├── reference_grid_025deg.zarr/                ── Windows & WSL (76 KB)
    ├── surface_mask_025deg.zarr/                  ── Windows & WSL (76 KB)
    ├── sealevel/ (Regridded 0.25° Zarr)           ── Windows (482 MB)
    ├── sss/ (Regridded 0.25° Zarr)                ── Windows & WSL (145 MB)
    ├── sst/ (Regridded 0.25° Zarr)                ── Windows & WSL (130 MB)
    ├── wind/ (Daily averaged 0.25° Zarr)          ── WSL ONLY (691 MB)
    ├── glorys/ (15-depth regridded 0.25° Zarr)    ── WSL ONLY (1.90 GB)
    ├── aligned/ (8 individual masked stores)      ── WSL ONLY (5.10 GB total)
    ├── X_inputs_025deg.zarr/ (7-channel inputs)   ── Windows & WSL (1.03 GB Windows / 1.30 GB WSL)
    ├── X_inputs_025deg_t7.zarr/ (t=7 chunks)      ── Windows ONLY (1.03 GB)
    └── y_target_025deg.zarr/ (15 depths target)   ── Windows & WSL (1.86 GB Windows / 2.00 GB WSL)
```

---

## 3. Raw Data Sources & Copernicus Products

All raw oceanographic data were acquired from the **Copernicus Marine Environment Monitoring Service (CMEMS)** and the **Copernicus Climate Change Service (C3S)** via the `copernicusmarine` Python client. The context and parameters originate from the scripts in [scripts/download/](file:///c:/Data/sih-26/ps-66/scripts/download).

### 3.1. Sea Surface Height & Geostrophic Currents (DUACS)
* **Script**: [`scripts/download/download_adt_ugos_vgos.py`](file:///c:/Data/sih-26/ps-66/scripts/download/download_adt_ugos_vgos.py)
* **Copernicus Product Title**: Global Ocean Gridded L4 Sea Surface Heights And Derived Variables Reprocessed (1993-Ongoing)
* **Copernicus Dataset ID**: `cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D`
* **Target Output**: `data/raw/sealevel/cmems_sealevel_adt_ugos_vgos_2015_2024_0125deg.zarr`
* **Spatial Bounding Box**: Longitude $45.0^\circ\text{E} - 105.0^\circ\text{E}$, Latitude $5.0^\circ\text{N} - 30.0^\circ\text{N}$
* **Native Grid Resolution**: $0.125^\circ \times 0.125^\circ$ ($202 \times 482$ grid cells)
* **Temporal Range**: `2015-01-01T00:00:00` to `2024-12-31T00:00:00` (3,653 daily steps)
* **Downloaded Variables**:
  - `adt`: Absolute Dynamic Topography ($m$)
  - `ugos`: Surface Geostrophic Eastward Sea Water Velocity ($m/s$)
  - `vgos`: Surface Geostrophic Northward Sea Water Velocity ($m/s$)

### 3.2. Sea Surface Salinity (Multi-Observation SSS)
* **Script**: [`scripts/download/download_sss.py`](file:///c:/Data/sih-26/ps-66/scripts/download/download_sss.py)
* **Copernicus Product Title**: Global Ocean Multi-Observation Reprocessed & NRT Sea Surface Salinity
* **Copernicus Dataset IDs**:
  - Multi-Year (MY): `cmems_obs-mob_glo_phy-sss_my_multi_P1D` (2015-01-01 to 2024-12-15; 3,637 days)
  - Near Real-Time (NRT): `cmems_obs-mob_glo_phy-sss_nrt_multi_P1D` (2024-12-16 to 2024-12-31; 16 days)
* **Target Outputs**:
  - `data/raw/sss/multiobs_sss_my_2015_2024-12-15_0125deg.zarr`
  - `data/raw/sss/multiobs_sss_nrt_2024-12-16_2024-12-31_0125deg.zarr`
* **Spatial Bounding Box**: Longitude $45.0^\circ\text{E} - 105.0^\circ\text{E}$, Latitude $5.0^\circ\text{N} - 30.0^\circ\text{N}$
* **Native Grid Resolution**: $0.125^\circ \times 0.125^\circ$ ($202 \times 482$ grid cells)
* **Downloaded Variable**:
  - `sos`: Sea Surface Salinity ($10^{-3}$ or $\text{PSU}$)

### 3.3. Sea Surface Temperature (C3S L4 SST)
* **Script**: [`scripts/download/download_sst.py`](file:///c:/Data/sih-26/ps-66/scripts/download/download_sst.py)
* **Copernicus Product Title**: Copernicus Climate Change Service (C3S) Global Sea Surface Temperature Reprocessed L4
* **Copernicus Dataset ID**: `C3S-GLO-SST-L4-REP-OBS-SST`
* **Target Output**: `data/raw/sst/c3s_sst_2015_2024_005deg.zarr`
* **Spatial Bounding Box**: Longitude $45.0^\circ\text{E} - 105.0^\circ\text{E}$, Latitude $5.0^\circ\text{N} - 30.0^\circ\text{N}$
* **Native Grid Resolution**: High-resolution $0.05^\circ \times 0.05^\circ$ ($502 \times 1,202$ grid cells)
* **Temporal Range**: `2015-01-01T00:00:00` to `2024-12-31T00:00:00` (3,653 daily steps)
* **Downloaded Variable**:
  - `analysed_sst`: Analysed Sea Surface Temperature ($\text{Kelvin}$)

### 3.4. Ocean Surface Wind (Hourly Scatterometer)
* **Script**: [`scripts/download/download_wind.py`](file:///c:/Data/sih-26/ps-66/scripts/download/download_wind.py)
* **Copernicus Product Title**: Global Ocean Daily & Hourly Gridded Sea Surface Winds from Scatterometer Reprocessed L4
* **Copernicus Dataset ID**: `cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H`
* **Target Output**: `data/raw/wind/cmems_wind_2015_2024_0125deg_hourly.zarr`
* **Spatial Bounding Box**: Longitude $45.0^\circ\text{E} - 105.0^\circ\text{E}$, Latitude $5.0^\circ\text{N} - 30.0^\circ\text{N}$
* **Native Grid Resolution**: $0.125^\circ \times 0.125^\circ$ ($202 \times 482$ grid cells)
* **Temporal Range**: `2015-01-01T00:00:00` to `2024-12-31T23:59:59` (**87,673 hourly timesteps**)
* **Downloaded Variables**:
  - `eastward_wind`: 10m Eastward Wind Component ($m/s$)
  - `northward_wind`: 10m Northward Wind Component ($m/s$)

### 3.5. Subsurface Ocean Physics (GLORYS12V1 Reanalysis)
* **Script**: [`scripts/download/download_glorys.py`](file:///c:/Data/sih-26/ps-66/scripts/download/download_glorys.py)
* **Copernicus Product Title**: Global Ocean Physics Reanalysis (GLORYS12V1)
* **Copernicus Dataset ID**: `cmems_mod_glo_phy_my_0.083deg_P1D-m`
* **Target Directory**: `data/raw/glorys/` (10 annual stores split by year via `copernicusmarine.subset_split_on`)
* **Spatial Bounding Box**: Longitude $45.0^\circ\text{E} - 105.0^\circ\text{E}$, Latitude $5.0^\circ\text{N} - 30.0^\circ\text{N}$
* **Native Grid Resolution**: Eddy-resolving $1/12^\circ \approx 0.0833^\circ$ ($301 \times 721$ grid cells)
* **Native Vertical Levels**: 36 standard depth levels from $0.494\text{ m}$ to $1,062.44\text{ m}$
* **Temporal Range**: 2015 to 2024 (3,653 days total: 365 days in standard years, 366 in leap years 2016, 2020, 2024)
* **Downloaded Variable**:
  - `thetao`: Sea Water Potential Temperature ($^\circ\text{C}$)

---

## 4. Raw Data Volume & Compression Analysis

The raw data downloaded from Copernicus is stored in Zarr format. The table below compares the **compressed on-disk size** against the **uncompressed in-memory size** (the memory that dense NumPy/xarray arrays occupy when expanded as `prod(shape) * itemsize`).

All raw Copernicus float variables are downloaded as 64-bit double-precision floats (`float64`, 8 bytes per number).

| Raw Dataset | Store Name / Identifier | Array Dimensions | Variable(s) & Dtype | On-Disk (Compressed) | In-Memory (Uncompressed) | Compression Ratio | Space Saved |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sea Level** | `cmems_sealevel_adt_ugos_vgos_2015_2024_0125deg.zarr` | `time: 3653`<br>`lat: 202, lon: 482` | `adt`, `ugos`, `vgos`<br>(all `float64`) | **1.064 GB**<br>(1,089 MB) | **7.950 GB**<br>(8,141 MB) | **7.47×** | 86.6% |
| **SSS (MY)** | `multiobs_sss_my_2015_2024-12-15_0125deg.zarr` | `time: 3637`<br>`depth: 1, lat: 202, lon: 482` | `sos`<br>(`float64`) | **0.446 GB**<br>(457 MB) | **2.638 GB**<br>(2,702 MB) | **5.92×** | 83.1% |
| **SSS (NRT)** | `multiobs_sss_nrt_2024-12-16_2024-12-31_0125deg.zarr` | `time: 16`<br>`depth: 1, lat: 202, lon: 482` | `sos`<br>(`float64`) | **0.002 GB**<br>(2 MB) | **0.012 GB**<br>(12 MB) | **6.00×** | 83.3% |
| **SST** | `c3s_sst_2015_2024_005deg.zarr` | `time: 3653`<br>`lat: 502, lon: 1202` | `analysed_sst`<br>(`float64`) | **1.195 GB**<br>(1,224 MB) | **16.423 GB**<br>(16,817 MB) | **13.74×** | 92.7% |
| **Wind** | `cmems_wind_2015_2024_0125deg_hourly.zarr` | `time: 87673`<br>`lat: 202, lon: 482` | `eastward_wind`, `northward_wind`<br>(both `float64`) | **20.201 GB**<br>(20,686 MB) | **127.200 GB**<br>(130,253 MB) | **6.30×** | 84.1% |
| **GLORYS 2015** | `..._2015-01-01T00-00-00-2015-12-31T00-00-00.zarr` | `time: 365, depth: 36`<br>`lat: 301, lon: 721` | `thetao` (`float64`) | 1.791 GB | 21.246 GB | 11.86× | 91.6% |
| **GLORYS 2016** | `..._2016-01-01T00-00-00-2016-12-31T00-00-00.zarr` | `time: 366, depth: 36`<br>`lat: 301, lon: 721` | `thetao` (`float64`) | 1.795 GB | 21.305 GB | 11.87× | 91.6% |
| **GLORYS 2017** | `..._2017-01-01T00-00-00-2017-12-31T00-00-00.zarr` | `time: 365, depth: 36`<br>`lat: 301, lon: 721` | `thetao` (`float64`) | 1.795 GB | 21.246 GB | 11.84× | 91.6% |
| **GLORYS 2018** | `..._2018-01-01T00-00-00-2018-12-31T00-00-00.zarr` | `time: 365, depth: 36`<br>`lat: 301, lon: 721` | `thetao` (`float64`) | 1.802 GB | 21.246 GB | 11.79× | 91.5% |
| **GLORYS 2019** | `..._2019-01-01T00-00-00-2019-12-31T00-00-00.zarr` | `time: 365, depth: 36`<br>`lat: 301, lon: 721` | `thetao` (`float64`) | 1.804 GB | 21.246 GB | 11.78× | 91.5% |
| **GLORYS 2020** | `..._2020-01-01T00-00-00-2020-12-31T00-00-00.zarr` | `time: 366, depth: 36`<br>`lat: 301, lon: 721` | `thetao` (`float64`) | 1.802 GB | 21.305 GB | 11.82× | 91.5% |
| **GLORYS 2021** | `..._2021-01-01T00-00-00-2021-12-31T00-00-00.zarr` | `time: 365, depth: 36`<br>`lat: 301, lon: 721` | `thetao` (`float64`) | 1.802 GB | 21.246 GB | 11.79× | 91.5% |
| **GLORYS 2022** | `..._2022-01-01T00-00-00-2022-12-31T00-00-00.zarr` | `time: 365, depth: 36`<br>`lat: 301, lon: 721` | `thetao` (`float64`) | 1.806 GB | 21.246 GB | 11.76× | 91.5% |
| **GLORYS 2023** | `..._2023-01-01T00-00-00-2023-12-31T00-00-00.zarr` | `time: 365, depth: 36`<br>`lat: 301, lon: 721` | `thetao` (`float64`) | 1.805 GB | 21.246 GB | 11.77× | 91.5% |
| **GLORYS 2024** | `..._2024-01-01T00-00-00-2024-12-31T00-00-00.zarr` | `time: 366, depth: 36`<br>`lat: 301, lon: 721` | `thetao` (`float64`) | 1.804 GB | 21.305 GB | 11.81× | 91.5% |
| **GLORYS Total** | *All 10 annual stores combined (3,653 days)* | `time: 3653, depth: 36`<br>`lat: 301, lon: 721` | `thetao` (`float64`) | **18.006 GB**<br>(18,438 MB) | **212.640 GB**<br>(217,743 MB) | **11.81×** | 91.5% |
| **TOTAL RAW** | **Complete Copernicus Raw Data Store** | — | — | **40.914 GB** | **366.863 GB** | **8.97×** | **88.85%** |

> **Key Takeaway**: Without Zarr compression, storing the raw datasets would require **366.86 GB** of disk space. Compressed Zarr storage with Blosc/Zstd saves **325.95 GB (88.85%)**, making it feasible to store the entire 10-year multi-sensor archive in just **40.91 GB**.

---

## 5. Processed Data Pipeline & Catalog

The raw satellite observations are regridded, harmonized, and masked using the scripts located in [`scripts/preprocessing/`](file:///c:/Data/sih-26/ps-66/scripts/preprocessing).

```
Raw Multi-Resolution Data
  ├── SST (0.05°)
  ├── SSS (0.125°)
  ├── ADT/UGOS/VGOS (0.125°)
  ├── Wind (0.125° Hourly) ──> [Daily Averaging]
  └── GLORYS (0.083°, 36 levels) ──> [1D PCHIP Vertical Interpolation: 15 Depths]
        │
        ▼
  xESMF Conservative-Normed Regridding (scripts/preprocessing/regridding/)
        │
        ▼
  Single-Variable Processed Zarrs (0.25° grid, 101 x 241, float32)
        │
        ▼
  scripts/preprocessing/harmonizing/create_master_mask.py
        │
        ▼
  surface_mask_025deg.zarr & reference_grid_025deg.zarr
        │
        ▼
  scripts/preprocessing/harmonizing/apply_master_mask.py ──> data/processed/aligned/*.zarr
        │
        ▼
  scripts/preprocessing/harmonizing/build_training_dataset.py
        │
        ├──> data/processed/X_inputs_025deg.zarr (7 surface channels: ADT, SST, SSS, U_curr, V_curr, U_wind, V_wind)
        ├──> data/processed/X_inputs_025deg_t7.zarr (Rechunked with 7-day temporal windows)
        └──> data/processed/y_target_025deg.zarr (15 depth levels target + depth_valid_mask)
```

### 5.1. Reference Grid & Master Land/Sea Mask

#### `reference_grid_025deg.zarr`
* **Path**: Windows & WSL (`data/processed/reference_grid_025deg.zarr`)
* **Coordinates**:
  - `latitude`: 101 points from $5.0^\circ\text{N}$ to $30.0^\circ\text{N}$ ($\Delta = 0.25^\circ$)
  - `longitude`: 241 points from $45.0^\circ\text{E}$ to $105.0^\circ\text{E}$ ($\Delta = 0.25^\circ$)
  - `lat_b`: 102 cell boundary latitudes
  - `lon_b`: 242 cell boundary longitudes
* **Size**: 76 KB on-disk | 0.01 MB uncompressed

#### `surface_mask_025deg.zarr`
* **Path**: Windows & WSL (`data/processed/surface_mask_025deg.zarr`)
* **Variables**: `surface_mask` (`bool`, shape `[101, 241]`)
* **Description**: True for ocean cells, False for land/inland water. Built by intersecting valid ocean masks across all satellite observation streams.
* **Size**: 76 KB on-disk | 0.03 MB uncompressed

---

### 5.2. Regridded Single-Variable Stores

All variables are converted to single-precision float (`float32`, 4 bytes) during preprocessing to reduce memory footprint by 50% while preserving physical accuracy.

#### 1. `sealevel_2015_2024_025deg.zarr`
* **Location**: Windows (`data/processed/sealevel/sealevel_2015_2024_025deg.zarr`)
* **Dimensions**: `{'time': 3653, 'latitude': 101, 'longitude': 241}`
* **Variables**:
  - `adt`: Absolute Dynamic Topography ($m$)
  - `ugos`: Zonal Geostrophic Velocity ($m/s$)
  - `vgos`: Meridional Geostrophic Velocity ($m/s$)
* **On-Disk Size**: 482 MB (folder total: 488.61 MB including HTML regridding report)
* **Uncompressed In-Memory Size**: 0.994 GB (1,017.61 MB)
* **Compression Ratio**: 2.11×

#### 2. `sss_2015_2024_025deg.zarr`
* **Location**: Windows & WSL (`data/processed/sss/sss_2015_2024_025deg.zarr`)
* **Dimensions**: `{'time': 3653, 'latitude': 101, 'longitude': 241}`
* **Variables**:
  - `sos`: Sea Surface Salinity ($10^{-3}$, PSU)
* **Notes**: Seamlessly combines the Multi-Year (2015-01-01 to 2024-12-15) and NRT (2024-12-16 to 2024-12-31) products.
* **On-Disk Size**: 145 MB (Windows) / 189 MB (WSL folder total)
* **Uncompressed In-Memory Size**: 0.331 GB (339.22 MB)
* **Compression Ratio**: 2.34×

#### 3. `sst_2015_2024_025deg.zarr`
* **Location**: Windows & WSL (`data/processed/sst/sst_2015_2024_025deg.zarr`)
* **Dimensions**: `{'time': 3653, 'latitude': 101, 'longitude': 241}`
* **Variables**:
  - `analysed_sst`: Sea Surface Temperature ($^\circ\text{C}$ or $\text{K}$)
* **On-Disk Size**: 130 MB (Windows) / 176 MB (WSL folder total)
* **Uncompressed In-Memory Size**: 0.331 GB (339.22 MB)
* **Compression Ratio**: 2.61×

#### 4. `wind_2015_2024_025deg.zarr`
* **Location**: WSL (`/home/addy/projects/sih-26/ps-66/data/processed/wind/wind_2015_2024_025deg.zarr`)
* **Dimensions**: `{'time': 3653, 'latitude': 101, 'longitude': 241}`
* **Variables**:
  - `northward_wind`: 10m Northward Wind ($m/s$)
  - `eastward_wind`: 10m Eastward Wind ($m/s$)
* **Notes**: Daily averaged from 87,673 hourly time slices and regridded using conservative area-weighting.
* **On-Disk Size**: 691 MB (WSL folder total: 705 MB)
* **Uncompressed In-Memory Size**: 0.663 GB (678.42 MB)
* **Compression Ratio**: 0.98×

#### 5. `glorys_temperature_2015_2024_025deg.zarr`
* **Location**: WSL (`/home/addy/projects/sih-26/ps-66/data/processed/glorys/glorys_temperature_2015_2024_025deg.zarr`)
* **Dimensions**: `{'time': 3653, 'depth': 15, 'latitude': 101, 'longitude': 241}`
* **Variables**:
  - `thetao`: Sea Water Potential Temperature ($^\circ\text{C}$)
  - `depth_valid_mask`: Static bathymetric mask per depth (`uint8`, shape `[15, 101, 241]`)
* **Vertical Depth Levels (15 layers)**:
  `[0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0]` meters
* **On-Disk Size**: 1.90 GB (WSL folder total: 2.00 GB)
* **Uncompressed In-Memory Size**: 4.969 GB (5,088.29 MB)
* **Compression Ratio**: 2.62×

---

### 5.3. Harmonized Aligned Datasets (WSL)

Generated by [`apply_master_mask.py`](file:///c:/Data/sih-26/ps-66/scripts/preprocessing/harmonizing/apply_master_mask.py) in `data/processed/aligned/` to ensure all fields share the exact same valid ocean mask prior to training dataset assembly:

| Aligned Store | Disk Size | Uncomp. Size | Shape | Variables |
| :--- | :--- | :--- | :--- | :--- |
| `adt_aligned_025deg.zarr` | 586 MB | 0.994 GB | `[3653, 101, 241]` | `adt`, `ugos`, `vgos` |
| `sss_aligned_025deg.zarr` | 182 MB | 0.331 GB | `[3653, 101, 241]` | `sos` |
| `sst_aligned_025deg.zarr` | 159 MB | 0.331 GB | `[3653, 101, 241]` | `analysed_sst` |
| `u_curr_aligned_025deg.zarr` | 586 MB | 0.994 GB | `[3653, 101, 241]` | `ugos`, `adt`, `vgos` |
| `v_curr_aligned_025deg.zarr` | 586 MB | 0.994 GB | `[3653, 101, 241]` | `vgos`, `adt`, `ugos` |
| `u_wind_aligned_025deg.zarr` | 540 MB | 0.663 GB | `[3653, 101, 241]` | `eastward_wind`, `northward_wind` |
| `v_wind_aligned_025deg.zarr` | 541 MB | 0.663 GB | `[3653, 101, 241]` | `northward_wind`, `eastward_wind` |
| `target_aligned_025deg.zarr` | 2.00 GB | 4.969 GB | `[3653, 15, 101, 241]` | `thetao`, `depth_valid_mask` |
| **Total `aligned/` Directory** | **5.10 GB** | **11.94 GB** | — | — |

---

### 5.4. Production Machine Learning Zarr Stores

These are the primary datasets loaded by PyTorch DataLoaders ([scripts/ml/dataloader.py](file:///c:/Data/sih-26/ps-66/scripts/ml/dataloader.py)) and used by the training/inference pipelines:

#### 1. `X_inputs_025deg.zarr`
* **Locations**: Windows (`data/processed/X_inputs_025deg.zarr`) and WSL
* **Dimensions**: `time: 3653`, `latitude: 101`, `longitude: 241`
* **Chunking**: `(1, 101, 241)`
* **Channels (7 Surface Variables)**:
  1. `adt`: Absolute Dynamic Topography ($m$)
  2. `sst`: Sea Surface Temperature ($^\circ\text{C}$)
  3. `sss`: Sea Surface Salinity ($\text{PSU}$)
  4. `u_curr`: Geostrophic Current Zonal Component ($m/s$)
  5. `v_curr`: Geostrophic Current Meridional Component ($m/s$)
  6. `u_wind`: 10m Eastward Wind ($m/s$)
  7. `v_wind`: 10m Northward Wind ($m/s$)
* **On-Disk Size**: 1.032 GB (1,057.01 MB on Windows) / 1.30 GB (WSL)
* **Uncompressed In-Memory Size**: 2.319 GB (2,374.39 MB)
* **Compression Ratio**: 2.25×

#### 2. `X_inputs_025deg_t7.zarr`
* **Locations**: Windows (`data/processed/X_inputs_025deg_t7.zarr`)
* **Dimensions & Variables**: Identical to `X_inputs_025deg.zarr`
* **Chunking**: Optimized 7-day temporal window chunking for Temporal U-Net sliding window batch extraction
* **On-Disk Size**: 1.032 GB (1,057.01 MB)
* **Uncompressed In-Memory Size**: 2.319 GB (2,374.39 MB)
* **Compression Ratio**: 2.25×

#### 3. `y_target_025deg.zarr`
* **Locations**: Windows (`data/processed/y_target_025deg.zarr`) and WSL
* **Dimensions**: `time: 3653`, `depth: 15`, `latitude: 101`, `longitude: 241`
* **Chunking**: `(1, 15, 101, 241)`
* **Variables**:
  - `target`: Subsurface Temperature ($^\circ\text{C}$), shape `[3653, 15, 101, 241]`, `float32`
  - `depth_valid_mask`: Static bathymetric validity mask, shape `[15, 101, 241]`, `bool`
* **Target Depth Levels**: `[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]` meters
* **On-Disk Size**: 1.856 GB (1,900.28 MB on Windows) / 2.00 GB (WSL)
* **Uncompressed In-Memory Size**: 4.969 GB (5,088.29 MB)
* **Compression Ratio**: 2.68×

---

## 6. Processed Data Volume & Compression Analysis

Summary of compressed vs. uncompressed sizes for the processed and ML-ready stores:

| Dataset / Store Name | Location | Primary Variables | Compressed (Disk) | Uncompressed (Memory) | Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `sealevel_2015_2024_025deg.zarr` | Windows | `adt`, `ugos`, `vgos` | 482 MB | 1,018 MB (0.99 GB) | 2.11× |
| `sss_2015_2024_025deg.zarr` | Windows/WSL | `sos` | 145 MB | 339 MB (0.33 GB) | 2.34× |
| `sst_2015_2024_025deg.zarr` | Windows/WSL | `analysed_sst` | 130 MB | 339 MB (0.33 GB) | 2.61× |
| `wind_2015_2024_025deg.zarr` | WSL | `eastward_wind`, `northward_wind` | 691 MB | 678 MB (0.66 GB) | 0.98× |
| `glorys_temperature_2015_2024_025deg.zarr` | WSL | `thetao`, `depth_valid_mask` | 1.90 GB | 5,088 MB (4.97 GB) | 2.62× |
| `surface_mask_025deg.zarr` | Windows/WSL | `surface_mask` | 0.01 MB | 0.03 MB | 3.00× |
| `reference_grid_025deg.zarr` | Windows/WSL | *(grid coordinates & bounds)* | 0.01 MB | 0.01 MB | 1.00× |
| `aligned/*.zarr` (8 stores) | WSL | 7 surface inputs + target aligned | 5.10 GB | 11,940 MB (11.94 GB) | 2.34× |
| `X_inputs_025deg.zarr` | Windows/WSL | 7 surface channels | 1.03 GB | 2,374 MB (2.32 GB) | 2.25× |
| `X_inputs_025deg_t7.zarr` | Windows | 7 surface channels (t=7 chunked) | 1.03 GB | 2,374 MB (2.32 GB) | 2.25× |
| `y_target_025deg.zarr` | Windows/WSL | `target` (15 depths) + mask | 1.86 GB | 5,088 MB (4.97 GB) | 2.68× |

---

## 7. Comprehensive Parameter & Variable Dictionary

The table below catalogs every variable across both the raw Copernicus datasets and the processed machine learning datasets:

| Variable Name | Role | Raw Copernicus Name | Standard Name (CF Convention) | Units | Valid Physical Range | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `adt` | Input Feature | `adt` | `sea_surface_height_above_geoid` | $m$ | $-1.5 \text{ to } +1.5$ | Absolute Dynamic Topography: Sea surface height above the geoid ($\text{ADT} = \text{SLA} + \text{MDT}$). |
| `sst` / `analysed_sst` | Input Feature | `analysed_sst` | `sea_surface_temperature` | $^\circ\text{C}$ / $\text{K}$ | $15.0 \text{ to } 35.0$ ($^\circ\text{C}$) | Foundation Sea Surface Temperature derived from satellite infrared/microwave radiometers. |
| `sss` / `sos` | Input Feature | `sos` | `sea_surface_salinity` | $\text{PSU}$ / $10^{-3}$ | $20.0 \text{ to } 42.0$ | Practical Salinity at sea surface from multi-satellite microwave radiometers (SMOS, SMAP). |
| `u_curr` / `ugos` | Input Feature | `ugos` | `surface_geostrophic_eastward_sea_water_velocity` | $m/s$ | $-2.5 \text{ to } +2.5$ | Zonal (Eastward) component of surface geostrophic current derived from altimetry gradients. |
| `v_curr` / `vgos` | Input Feature | `vgos` | `surface_geostrophic_northward_sea_water_velocity` | $m/s$ | $-2.5 \text{ to } +2.5$ | Meridional (Northward) component of surface geostrophic current derived from altimetry gradients. |
| `u_wind` | Input Feature | `eastward_wind` | `eastward_wind` | $m/s$ | $-30.0 \text{ to } +30.0$ | 10-meter Eastward neutral wind speed component from spaceborne scatterometers. |
| `v_wind` | Input Feature | `northward_wind` | `northward_wind` | $m/s$ | $-30.0 \text{ to } +30.0$ | 10-meter Northward neutral wind speed component from spaceborne scatterometers. |
| `target` / `thetao` | Target Label | `thetao` | `sea_water_potential_temperature` | $^\circ\text{C}$ | $2.0 \text{ to } 35.0$ | Subsurface potential temperature reconstructed across 15 vertical layers (0m to 1000m). |
| `surface_mask` | Mask | — | `sea_binary_mask` | boolean | $\{0, 1\}$ | Static 2D surface mask ($101 \times 241$); 1 indicates valid ocean, 0 indicates land. |
| `depth_valid_mask`| Mask | — | `bathymetric_valid_mask` | boolean / uint8 | $\{0, 1\}$ | Static 3D bathymetric mask ($15 \times 101 \times 241$); prevents training on subsurface levels deeper than seafloor. |
| `time` | Coordinate | `time` | `time` | ISO 8601 | 2015-01-01 to 2024-12-31 | Daily time coordinate index (3,653 days). |
| `depth` | Coordinate | `depth` | `depth` | $m$ | $0 \text{ to } 1000$ | Depth below ocean surface: $[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]\text{ m}$. |
| `latitude` | Coordinate | `latitude` | `latitude` | degrees_north | $5.0^\circ\text{N} \text{ to } 30.0^\circ\text{N}$ | Regular latitude grid with $0.25^\circ$ spacing (101 points). |
| `longitude` | Coordinate | `longitude` | `longitude` | degrees_east | $45.0^\circ\text{E} \text{ to } 105.0^\circ\text{E}$ | Regular longitude grid with $0.25^\circ$ spacing (241 points). |

---

*Documentation compiled and verified against dataset archives on 2026-09-26.*
