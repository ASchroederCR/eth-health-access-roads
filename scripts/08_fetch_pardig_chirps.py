"""
08_fetch_pardig_chirps.py -- pull one year of daily rainfall over Ethiopia from Google Earth Engine:

  * PARDIG_1_NRT (Google WeatherNext precipitation nowcast, 15-min frames, 0.1 deg, m/h)
  * CHIRPS v3 daily, IMERG-disaggregated (UCSB-CHC/CHIRPS/V3/DAILY_SAT, 0.05 deg, mm/day)

Both go onto PARDIG's native 0.1 deg grid (CHIRPS is block-averaged 2x2 on the server) and are
cached as data/raw/pardig/eth_daily_<start>_<end>.npz with arrays [day, row, col] in mm/day,
plus the PARDIG frame count per day and an Ethiopia land mask (LSIB 2017).

The year ends on the last complete UTC day that PARDIG has, so the two series overlap fully.
Rerunning skips chunks already on disk.

Run with the user Python (earthengine-api is installed there, not in ArcGIS Pro's env):
    python -I scripts\\08_fetch_pardig_chirps.py [--days 365]
"""
import argparse, io, os, datetime as dt
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import ee

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "pardig"); os.makedirs(RAW, exist_ok=True)

PARDIG_ID = "projects/gcp-public-data-weathernext/assets/PARDIG_1_NRT"
CHIRPS_ID = "UCSB-CHC/CHIRPS/V3/DAILY_SAT"
# Ethiopia (32.99-47.99 E, 3.40-14.89 N) on PARDIG's grid: cell edges fall on x.x5 degrees
X0, Y0, RES, NX, NY = 32.95, 15.05, 0.1, 151, 118
GRID = {"dimensions": {"width": NX, "height": NY}, "crsCode": "EPSG:4326",
        "affineTransform": {"scaleX": RES, "shearX": 0, "translateX": X0,
                            "shearY": 0, "scaleY": -RES, "translateY": Y0}}
CHUNK = 10   # days per request (PARDIG: 10 x 96 frames)


def pixels(img):
    """Fetch a multiband image on GRID as a float32 array [band, row, col]; masked -> NaN."""
    raw = ee.data.computePixels({"expression": img.toFloat().unmask(-9999), "fileFormat": "NPY", "grid": GRID})
    a = np.load(io.BytesIO(raw)) if isinstance(raw, (bytes, bytearray)) else raw
    out = np.stack([a[n].astype("float32") for n in a.dtype.names])
    out[out <= -9998] = np.nan
    return out


def pardig_day(d0):
    frames = PARDIG.filterDate(d0, d0.advance(1, "day"))
    # mean rate (m/h) x 24 h x 1000 = mm/day; equals sum x 0.25 h x 1000 when all 96 frames exist,
    # and stays unbiased on days with a few frames missing
    return frames.mean().multiply(24 * 1000).rename(d0.format("'p'YYYYMMdd"))


def chirps_day(d0):
    im = CHIRPS.filterDate(d0, d0.advance(1, "day")).first()
    return (im.setDefaultProjection(CHIRPS_PROJ)
              .reduceResolution(ee.Reducer.mean(), maxPixels=16)
              .rename(d0.format("'c'YYYYMMdd")))


def stack(fn, days):
    return ee.ImageCollection([fn(ee.Date(d.isoformat())) for d in days]).toBands()


def fetch_chunk(k, days):
    path = os.path.join(RAW, "chunks", f"{days[0]:%Y%m%d}_{days[-1]:%Y%m%d}.npz")
    if os.path.exists(path): return path
    p = pixels(stack(pardig_day, days)); c = pixels(stack(chirps_day, days))
    np.savez_compressed(path, pardig=p, chirps=c)
    print(f"  chunk {k:02d} {days[0]} .. {days[-1]} ok", flush=True)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=os.environ.get("EE_PROJECT")); ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--start", help="YYYY-MM-DD; with --end, fetch that window instead of the latest year")
    ap.add_argument("--end")
    a = ap.parse_args()
    ee.Initialize(project=a.project)
    PARDIG = ee.ImageCollection(PARDIG_ID).select("precipitation")
    CHIRPS = ee.ImageCollection(CHIRPS_ID).select("precipitation")
    CHIRPS_PROJ = CHIRPS.first().projection()

    last = dt.datetime.fromtimestamp(PARDIG.aggregate_max("system:time_start").getInfo() / 1000, dt.timezone.utc)
    end = (last.date() if last.hour == 23 and last.minute == 45 else last.date() - dt.timedelta(days=1))
    c_last = dt.datetime.fromtimestamp(CHIRPS.aggregate_max("system:time_start").getInfo() / 1000, dt.timezone.utc).date()
    end = min(end, c_last); start = end - dt.timedelta(days=a.days - 1)
    if a.start and a.end:
        start, end = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)
    days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    print(f"PARDIG last frame {last:%Y-%m-%d %H:%M} UTC; CHIRPS last day {c_last}; window {start} .. {end}")

    out = os.path.join(RAW, f"eth_daily_{start:%Y%m%d}_{end:%Y%m%d}.npz")
    os.makedirs(os.path.join(RAW, "chunks"), exist_ok=True)
    chunks = [days[i:i + CHUNK] for i in range(0, len(days), CHUNK)]
    with ThreadPoolExecutor(4) as ex:
        paths = list(ex.map(lambda kc: fetch_chunk(*kc), enumerate(chunks)))

    # frames per day (to flag PARDIG gaps) and the Ethiopia mask
    t = PARDIG.filterDate(start.isoformat(), (end + dt.timedelta(days=1)).isoformat()) \
              .aggregate_array("system:time_start").getInfo()
    tday = np.array([dt.datetime.fromtimestamp(x / 1000, dt.timezone.utc).date() for x in t])
    nframes = np.array([(tday == d).sum() for d in days])
    eth = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(ee.Filter.eq("country_na", "Ethiopia"))
    mask = pixels(ee.Image(0).paint(eth, 1).rename("eth"))[0] == 1

    P = np.concatenate([np.load(p)["pardig"] for p in paths]); C = np.concatenate([np.load(p)["chirps"] for p in paths])
    lon = X0 + RES * (np.arange(NX) + 0.5); lat = Y0 - RES * (np.arange(NY) + 0.5)
    np.savez_compressed(out, pardig=P, chirps=C, nframes=nframes, mask=mask, lon=lon, lat=lat,
                        dates=np.array([d.isoformat() for d in days]))
    print(f"wrote {out}: {P.shape}, {mask.sum()} Ethiopia cells, "
          f"{(nframes < 96).sum()} days with <96 PARDIG frames (min {nframes.min()})")
