"""
21_weathernext3_forecast.py -- WeatherNext 3 15-day rainfall for Ethiopia from Earth Engine, as daily totals.

Collection projects/gcp-public-data-weathernext/assets/weathernext_3_0_0_0p1deg (0.1 deg, global):
  * 15-day runs at 00/06/12/18 UTC (forecast_hour 1..360, hourly); 48 h runs every other hour; archive from Jan 2024.
  * No individual members: each field is the ensemble mean and p10/p25/p50/p75/p90.
  * Rain fields: total_precipitation_1hr (used here), experimental_tp_1hr, imerg_tp_1hr; metres per hour.
This script uses 00 UTC runs so forecast days are UTC days (day d = forecast hours 24(d-1)+1 .. 24d), and sums the
hourly mean, p50 and p90 into daily totals (mm). Summed hourly percentiles are not the percentiles of the daily total:
the summed p90 is an upper wet-end envelope, the summed p50 a low "typical" path; the mean adds up exactly.

Cache: data/raw/weathernext3/wn3_eth_<YYYYMMDDHH>.npz with mean, p50, p90 [day, row, col] mm, mask, init and the grid.
Other scripts load it with load_run().

    python scripts\\21_weathernext3_forecast.py [--init 2026-11-19]      (no -I: earthengine-api is in user site-packages)
"""
import argparse, datetime as dt, glob, io, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "weathernext3"); os.makedirs(RAW, exist_ok=True)
WN3 = "projects/gcp-public-data-weathernext/assets/weathernext_3_0_0_0p1deg"
FIELD = "total_precipitation_1hr"
X0, Y0, RES, NX, NY = 32.95, 15.05, 0.1, 151, 118          # same 0.1 deg grid as the PARDIG / CHIRPS work
NDAY = 15


def load_run(init=None):
    """Latest (or given) cached run -> dict(mean, p50, p90, mask, init, x0, y0, res, nx, ny)."""
    f = (os.path.join(RAW, f"wn3_eth_{np.datetime64(init, 'D').astype(str).replace('-', '')}00.npz") if init else
         sorted(glob.glob(os.path.join(RAW, "wn3_eth_??????????.npz")))[-1])
    z = np.load(f)
    return {k: z[k] for k in z.files} | {"path": f}


def fetch(init):
    import ee
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    from shapely.geometry import shape
    path = os.path.join(RAW, f"wn3_eth_{init:%Y%m%d%H}.npz")
    if os.path.exists(path): return path
    run = ee.ImageCollection(WN3).filter(ee.Filter.eq("start_time", f"{init:%Y-%m-%dT%H:%M:%S}Z"))
    n = run.size().getInfo()
    if n < 24 * NDAY: raise SystemExit(f"run {init} incomplete in Earth Engine ({n} of {24 * NDAY} hourly images)")
    grid = {"dimensions": {"width": NX, "height": NY}, "crsCode": "EPSG:4326",
            "affineTransform": {"scaleX": RES, "shearX": 0, "translateX": X0, "shearY": 0, "scaleY": -RES, "translateY": Y0}}
    out = {}
    for stat in ["mean", "p50", "p90"]:
        b = f"{FIELD}_{stat}"
        days = [run.filter(ee.Filter.rangeContains("forecast_hour", 24 * d + 1, 24 * d + 24)).select(b).sum()
                  .max(0).multiply(1000).rename(f"d{d + 1:02d}") for d in range(NDAY)]
        raw = ee.data.computePixels({"expression": ee.Image.cat(days).unmask(-9999), "fileFormat": "NPY", "grid": grid})
        a = np.load(io.BytesIO(raw)); arr = np.stack([a[k].astype("float32") for k in a.dtype.names]); arr[arr <= -9998] = np.nan
        out[stat] = arr
        print(f"  {stat}: fetched", flush=True)
    gj = json.load(open(adm1_path(), encoding="utf-8"))           # OCHA COD-AB 2025 regions
    mask = rasterize([(shape(f["geometry"]), 1) for f in gj["features"]], out_shape=(NY, NX),
                     transform=from_origin(X0, Y0, RES, RES), all_touched=True) == 1
    np.savez_compressed(path, **out, mask=mask, init=init.isoformat(), x0=X0, y0=Y0, res=RES, nx=NX, ny=NY,
                        field=FIELD, source=WN3)
    return path


def adm1_path():
    """OCHA COD-AB 2025 admin-1 regions (14 + the contested area), cached GeoJSON; built from the COD-AB download by
    40_woreda_population.py when missing."""
    p = os.path.join(ROOT, "data", "raw", "boundaries", "cod_ab_eth", "eth_adm1.geojson")
    if not os.path.exists(p):
        import importlib.util
        sp = importlib.util.spec_from_file_location("wpop", os.path.join(ROOT, "scripts", "40_woreda_population.py"))
        m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); m.adm1_geojson()
    return p


def latest_complete_00z():
    import ee
    d = dt.datetime.now(dt.timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    for _ in range(5):
        n = ee.ImageCollection(WN3).filter(ee.Filter.eq("start_time", f"{d:%Y-%m-%dT00:00:00}Z")).size().getInfo()
        if n >= 24 * NDAY: return d
        d -= dt.timedelta(days=1)
    raise SystemExit("no complete 00Z WeatherNext 3 run in the last 5 days")


if __name__ == "__main__":
    import ee
    ap = argparse.ArgumentParser(); ap.add_argument("--init"); ap.add_argument("--project", default=os.environ.get("EE_PROJECT"))
    a = ap.parse_args()
    ee.Initialize(project=a.project)
    init = dt.datetime.fromisoformat(a.init) if a.init else latest_complete_00z()
    print(f"WeatherNext 3 run {init:%Y-%m-%d %H}Z", flush=True)
    p = fetch(init)
    r = load_run(init.date().isoformat())
    m = r["mask"]
    print(f"wrote {p}: national 15-day mean {r['mean'].sum(0)[m].mean():.1f} mm (p50-sum {r['p50'].sum(0)[m].mean():.1f}, "
          f"p90-sum {r['p90'].sum(0)[m].mean():.1f})")
