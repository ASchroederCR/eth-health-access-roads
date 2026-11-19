"""
31_chirps_normal.py -- CHIRPS 1991-2020 daily rainfall for the forecast's calendar days, so forecast and observed rain
can be shown as % of normal.

For each year 1991-2020 it pulls CHIRPS v3 daily (UCSB-CHC/CHIRPS/V3/DAILY_RNL, 0.05 deg, from 1981) for the 10 days
before the run plus its 15 forecast days, block-averaged 2x2 onto PARDIG's 0.1 deg grid (the grid WeatherNext 3 is
cached on). Window totals for any span (days 1-3, 4-7, 8-15, all 15, the 10 observed days) are sums of these days.

DAILY_RNL splits CHIRPS pentads into days with ERA5; DAILY_SAT (used in 08_/09_) uses IMERG and starts in 1998. Multi-day
totals of the two agree closely because both sum back to the same pentads.

Output
  data/raw/chirps/chirps_normal_<MMDD>_<MMDD>.npz   rain [year, day, y, x] mm/day, years, days (month-day), lat, lon

    python scripts\\31_chirps_normal.py --init 2026-11-19 [--project ...]
"""
import argparse, datetime as dt, io, os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import ee

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTD = os.path.join(ROOT, "data", "raw", "chirps"); os.makedirs(OUTD, exist_ok=True)
CHIRPS_ID = "UCSB-CHC/CHIRPS/V3/DAILY_RNL"
X0, Y0, RES, NX, NY = 32.95, 15.05, 0.1, 151, 118          # PARDIG grid, as in 08_
GRID = {"dimensions": {"width": NX, "height": NY}, "crsCode": "EPSG:4326",
        "affineTransform": {"scaleX": RES, "shearX": 0, "translateX": X0, "shearY": 0, "scaleY": -RES, "translateY": Y0}}
YEARS = range(1991, 2021)
NOBS, NFC = 10, 15


def pixels(img):
    raw = ee.data.computePixels({"expression": img.toFloat().unmask(-9999), "fileFormat": "NPY", "grid": GRID})
    a = np.load(io.BytesIO(raw)) if isinstance(raw, (bytes, bytearray)) else raw
    out = np.stack([a[n].astype("float32") for n in a.dtype.names]); out[out <= -9998] = np.nan
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True); ap.add_argument("--project", default=os.environ.get("EE_PROJECT"))
    a = ap.parse_args()
    ee.Initialize(project=a.project)
    init = dt.date.fromisoformat(a.init)
    first = init - dt.timedelta(days=NOBS)
    md = [(first + dt.timedelta(days=i)).strftime("%m-%d") for i in range(NOBS + NFC)]
    out = os.path.join(OUTD, f"chirps_normal_{md[0].replace('-', '')}_{md[-1].replace('-', '')}.npz")
    if os.path.exists(out): raise SystemExit(f"exists: {out}")
    C = ee.ImageCollection(CHIRPS_ID).select("precipitation"); proj = C.first().projection()

    def year(y):
        d0 = dt.date(y, first.month, first.day)
        days = [d0 + dt.timedelta(days=i) for i in range(NOBS + NFC)]
        ims = [C.filterDate(d.isoformat(), (d + dt.timedelta(days=1)).isoformat()).first().setDefaultProjection(proj)
                .reduceResolution(ee.Reducer.mean(), maxPixels=16).rename(f"d{i:02d}") for i, d in enumerate(days)]
        r = pixels(ee.ImageCollection(ims).toBands()); print(f"  {y} ok", flush=True); return r

    with ThreadPoolExecutor(6) as ex:
        R = np.stack(list(ex.map(year, YEARS)))
    lon = X0 + RES * (np.arange(NX) + 0.5); lat = Y0 - RES * (np.arange(NY) + 0.5)
    np.savez_compressed(out, rain=R.astype("float32"), years=np.array(list(YEARS)), days=np.array(md), lat=lat, lon=lon)
    print(f"wrote {out}: {R.shape}")
