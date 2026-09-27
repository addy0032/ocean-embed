"""
Download Argo float temperature profiles for independent in-situ validation.

Source: Ifremer ERDDAP (Argo GDAC mirror), dataset "ArgoFloats".
Period: 2023-06-01 to 2023-12-31 (SW monsoon -> post-monsoon -> NE monsoon
onset), inside the 2023-2024 test period.

Fetched month by month to keep each request small; the raw per-level rows
(raw + adjusted pressure/temperature with QC flags and data mode) are saved
as-is so QC selection happens in scripts/ml/evaluate_argo.py.
"""

import time
import urllib.parse
from pathlib import Path

import pandas as pd

ERDDAP_URL = "https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.csv"

START = "2023-06-01"
END = "2024-01-01"  # exclusive

MIN_LON, MAX_LON = 45, 105
MIN_LAT, MAX_LAT = 5, 30
MAX_PRES = 1100  # dbar, a little below 1000 m so the deepest level can be interpolated

VARIABLES = [
    "platform_number", "cycle_number", "direction", "data_mode",
    "time", "time_qc", "latitude", "longitude", "position_qc",
    "pres", "pres_qc", "pres_adjusted", "pres_adjusted_qc",
    "temp", "temp_qc", "temp_adjusted", "temp_adjusted_qc",
]

# QC flags / modes are single characters and blank for real-time adjusted fields
STR_COLUMNS = {c: str for c in VARIABLES if c.endswith("_qc") or c in ("data_mode", "direction")}

OUTPUT_PATH = Path("data/raw/argo/argo_temp_2023-06_2023-12.csv")


def fetch(start, end, retries=3):
    constraints = [
        f"time>={start}T00:00:00Z", f"time<{end}T00:00:00Z",
        f"latitude>={MIN_LAT}", f"latitude<={MAX_LAT}",
        f"longitude>={MIN_LON}", f"longitude<={MAX_LON}",
        f"pres<={MAX_PRES}",
    ]
    query = ",".join(VARIABLES) + "&" + "&".join(constraints)
    url = ERDDAP_URL + "?" + urllib.parse.quote(query, safe=",&=")

    for attempt in range(1, retries + 1):
        try:
            # second row of ERDDAP CSV holds units
            return pd.read_csv(url, skiprows=[1], dtype=STR_COLUMNS)
        except Exception as e:
            print(f"  attempt {attempt} failed: {e}")
            time.sleep(10 * attempt)
    raise RuntimeError(f"ERDDAP request failed for {start} -> {end}")


def main():
    months = pd.date_range(START, END, freq="MS")
    frames = []

    for start, end in zip(months[:-1], months[1:]):
        print(f"Fetching {start:%Y-%m} ...")
        df = fetch(f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}")
        n_prof = df.groupby(["platform_number", "cycle_number", "direction"]).ngroups
        print(f"  {len(df):,} levels, {n_prof} profiles")
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(data):,} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
