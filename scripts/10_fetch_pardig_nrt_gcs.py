"""
10_fetch_pardig_nrt_gcs.py -- recent PARDIG NRT frames straight from the public GCS bucket.

Earth Engine's PARDIG_1_NRT collection lags by a few weeks; gs://pardig_1/NRT/zarr/2026_to_present/
holds the same 15-min, 0.1 deg analysis up to the last hour or so (one global Zarr per frame,
~1.2 MB each). This is near-real-time observation, not a forecast.

Writes data/raw/pardig/pardig_nrt_daily_<start>_<end>.nc: daily mm (UTC days, complete days only)
over the same Ethiopia grid as 08_fetch_pardig_chirps.py, plus frames per day.
Daily total = mean frame rate (m/h) x 24 x 1000, which equals the sum x 0.25 h x 1000 when all
96 frames exist.

    python scripts\\10_fetch_pardig_nrt_gcs.py [--start 2026-10-21] [--end YYYY-MM-DD] [--offset 6]
"""
import argparse, os, datetime as dt
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, xarray as xr, zarr
from obstore.store import GCSStore

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "pardig")
BUCKET, PREFIX = "pardig_1", "NRT/zarr/2026_to_present"
LAT, LON = slice(3.25, 15.05), slice(32.95, 48.05)      # cell centres 3.3-15.0 N, 33.0-48.0 E
STORE = GCSStore(bucket=BUCKET, skip_signature=True)


def load(t):
    p = f"{PREFIX}/{t:%Y%m%dT%H%M%S}/predictions.zarr"
    try:
        ds = xr.open_zarr(zarr.storage.ObjectStore(GCSStore(bucket=BUCKET, prefix=p, skip_signature=True),
                                                   read_only=True), chunks=None)
        return t, ds["precipitation"].sel(latitude=LAT, longitude=LON).values.astype("float32")
    except Exception:
        return t, None                                    # frame missing or not yet published


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--start", default="2026-10-21"); ap.add_argument("--end")
    ap.add_argument("--offset", type=int, default=0, help="window start hour UTC, e.g. 6 for 06-06 UTC days")
    a = ap.parse_args()
    newest = sorted(STORE.list_with_delimiter(f"{PREFIX}/")["common_prefixes"])[-1].rsplit("/", 1)[-1]
    newest = pd.Timestamp(dt.datetime.strptime(newest, "%Y%m%dT%H%M%S"))
    off = pd.Timedelta(hours=a.offset)
    last_full = (newest + pd.Timedelta(minutes=15) - off).normalize() - pd.Timedelta(days=1)
    end = min(pd.Timestamp(a.end), last_full) if a.end else last_full
    start = pd.Timestamp(a.start)
    print(f"newest frame {newest}; fetching {start:%Y-%m-%d} .. {end:%Y-%m-%d}")

    times = pd.date_range(start + off, end + off + pd.Timedelta(hours=23, minutes=45), freq="15min")
    with ThreadPoolExecutor(24) as ex:
        res = dict(ex.map(load, times))
    ok = [t for t in times if res[t] is not None]
    ref = xr.open_zarr(zarr.storage.ObjectStore(GCSStore(bucket=BUCKET, prefix=f"{PREFIX}/{ok[0]:%Y%m%dT%H%M%S}/predictions.zarr",
                                                         skip_signature=True), read_only=True), chunks=None)
    lat = ref.latitude.sel(latitude=LAT).values; lon = ref.longitude.sel(longitude=LON).values
    da = xr.DataArray(np.stack([res[t] for t in ok]), dims=("time", "latitude", "longitude"),
                      coords={"time": ok, "latitude": lat, "longitude": lon})
    g = da.assign_coords(time=da.time - off).resample(time="1D")   # window starting at `off` -> its date
    daily = (g.mean() * 24 * 1000).sortby("latitude", ascending=False)
    daily.name = "precipitation"; daily.attrs = {"units": "mm/day", "source": f"gs://{BUCKET}/{PREFIX}",
                                                 "note": f"24 h windows starting {a.offset:02d} UTC on the labelled date; mean 15-min rate x 24 h"}
    frames = g.count().isel(latitude=0, longitude=0).astype("int16")
    out = xr.Dataset({"precipitation": daily, "frames": frames})
    path = os.path.join(RAW, f"pardig_nrt_daily_{start:%Y%m%d}_{end:%Y%m%d}" + (f"_{a.offset:02d}z" if a.offset else "") + ".nc")
    out.to_netcdf(path)
    print(f"wrote {path}: {daily.shape}, {len(times) - len(ok)} of {len(times)} frames missing")
