"""
27_wn3_hourly.py -- hourly WeatherNext 3 rain (ensemble mean and 90th percentile) for Ethiopia, for intense-rain
hazard. 21_ keeps daily totals only; short downpours need the hourly fields.

Cache: data/raw/weathernext3/hourly/wn3h_eth_<YYYYMMDDHH>.npz with mean, p90 [hour 1..360, row, col] in 0.1 mm
(int16), on the same 0.1 deg grid as 21_ and PARDIG.

    python scripts\\27_wn3_hourly.py --init 2026-11-19 [2026-11-01 ...]     (no -I: earthengine-api in user site)
"""
import argparse, datetime as dt, io, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "weathernext3", "hourly"); os.makedirs(RAW, exist_ok=True)
WN3 = "projects/gcp-public-data-weathernext/assets/weathernext_3_0_0_0p1deg"
X0, Y0, RES, NX, NY = 32.95, 15.05, 0.1, 151, 118
NH = 360


def path_for(init):
    return os.path.join(RAW, f"wn3h_eth_{init:%Y%m%d%H}.npz")


def load_hourly(init):
    z = np.load(path_for(init))
    return {"mean": z["mean"].astype("float32") / 10, "p90": z["p90"].astype("float32") / 10}


def fetch(init):
    import ee
    p = path_for(init)
    if os.path.exists(p): return p
    run = ee.ImageCollection(WN3).filter(ee.Filter.eq("start_time", f"{init:%Y-%m-%dT%H:%M:%S}Z")).sort("forecast_hour")
    n = run.size().getInfo()
    if n < NH: raise SystemExit(f"run {init} has {n} of {NH} hourly images")
    grid = {"dimensions": {"width": NX, "height": NY}, "crsCode": "EPSG:4326",
            "affineTransform": {"scaleX": RES, "shearX": 0, "translateX": X0, "shearY": 0, "scaleY": -RES, "translateY": Y0}}
    out = {}
    for stat in ["mean", "p90"]:
        parts = []
        for h0 in range(0, NH, 120):                                  # 120 bands per request
            sub = run.filter(ee.Filter.rangeContains("forecast_hour", h0 + 1, h0 + 120))
            img = sub.select(f"total_precipitation_1hr_{stat}").map(
                lambda i: i.max(0).multiply(10000).round().toInt16()).toBands()
            raw = ee.data.computePixels({"expression": img, "fileFormat": "NPY", "grid": grid})
            a = np.load(io.BytesIO(raw)); parts.append(np.stack([a[k] for k in a.dtype.names]).astype("int16"))
        out[stat] = np.concatenate(parts)
    np.savez_compressed(p, **out, init=init.isoformat())
    return p


if __name__ == "__main__":
    import ee
    ap = argparse.ArgumentParser(); ap.add_argument("--init", nargs="+", required=True)
    ap.add_argument("--project", default=os.environ.get("EE_PROJECT")); a = ap.parse_args()
    ee.Initialize(project=a.project)
    for d in a.init:
        init = dt.datetime.fromisoformat(d)
        try:
            p = fetch(init); z = np.load(p)
            print(f"{init:%Y-%m-%d}: {z['mean'].shape}, max hourly p90 {z['p90'].max() / 10:.1f} mm", flush=True)
        except Exception as e:
            print(f"{init:%Y-%m-%d}: FAILED {str(e)[:150]}", flush=True)
