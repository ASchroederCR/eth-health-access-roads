"""
28_pardig_hourly.py -- observed hourly rain from PARDIG's 15-min near-real-time frames (GCS), for calibrating the
intense-rain hazard. Hour = mean of its four 15-min rates (m/h) x 1000 -> mm in that hour.

Writes data/raw/pardig/pardig_hourly_max_<start>_<end>.npz with, per UTC day [day, row, col]:
  max1h   wettest single hour, mm
  max3h   wettest 3 consecutive hours, mm
  total   daily total, mm (check against 10_)
on the same 0.1 deg Ethiopia grid as 08_ / 10_.

    python scripts\\28_pardig_hourly.py --start 2026-11-01 --end 2026-11-18
"""
import argparse, importlib.util, os
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("p10", os.path.join(ROOT, "scripts", "10_fetch_pardig_nrt_gcs.py"))
p10 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p10)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--start", required=True); ap.add_argument("--end", required=True); a = ap.parse_args()
    s, e = pd.Timestamp(a.start), pd.Timestamp(a.end)
    times = pd.date_range(s, e + pd.Timedelta(hours=23, minutes=45), freq="15min")
    with ThreadPoolExecutor(24) as ex:
        res = dict(ex.map(p10.load, times))
    ok = [t for t in times if res[t] is not None]
    shp = res[ok[0]].shape
    print(f"{len(ok)} of {len(times)} frames; grid {shp}")
    days = pd.date_range(s, e)
    M1, M3, TT = (np.zeros((len(days),) + shp, "float32") for _ in range(3))
    for di, d in enumerate(days):
        hrs = []
        for h in range(24):
            fr = [res[t] for t in pd.date_range(d + pd.Timedelta(hours=h), periods=4, freq="15min") if res.get(t) is not None]
            hrs.append(np.nan_to_num(np.mean(fr, 0)) * 1000 if fr else np.zeros(shp, "float32"))
        H = np.stack(hrs)
        M1[di] = H.max(0); M3[di] = np.stack([H[i:i + 3].sum(0) for i in range(22)]).max(0); TT[di] = H.sum(0)
    # same orientation as the daily files: latitude north -> south
    lat = p10.xr.open_zarr(p10.zarr.storage.ObjectStore(p10.GCSStore(bucket=p10.BUCKET, prefix=f"{p10.PREFIX}/{ok[0]:%Y%m%dT%H%M%S}/predictions.zarr",
                                                                     skip_signature=True), read_only=True), chunks=None).latitude.sel(latitude=p10.LAT).values
    if lat[0] < lat[-1]:
        M1, M3, TT = M1[:, ::-1], M3[:, ::-1], TT[:, ::-1]
    out = os.path.join(p10.RAW, f"pardig_hourly_max_{s:%Y%m%d}_{e:%Y%m%d}.npz")
    np.savez_compressed(out, max1h=M1, max3h=M3, total=TT, dates=np.array([d.strftime("%Y-%m-%d") for d in days]))
    print(f"wrote {out}; Ethiopia-wide wettest hour {M1.max():.1f} mm; days {len(days)}")
