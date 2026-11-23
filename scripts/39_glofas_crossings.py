"""
39_glofas_crossings.py -- GloFAS river-flow forecasts at the road river crossings (38_), as the chance of a flood.

GloFAS (Copernicus Emergency Management Service, Global Flood Awareness System, LISFLOOD on a 0.05 deg river network)
issues a 51-member ensemble of daily river discharge every day. For each crossing this script:
  1. matches the crossing to a GloFAS river cell: within 2 cells (~11 km), upstream area within a factor of 2 of MERIT
     Hydro's at the crossing, closest match in log(area) then distance. Crossings on rivers draining < 250 km2 are not
     matched (GloFAS does not resolve them reliably), nor are rivers whose GloFAS 2-year flood is under 10 m3/s (dry
     beds in the model, where any flow would count as a "flood"); 14_ treats both as small streams (flash-flood rule).
  2. takes the forecast issued on the run date (or the day before if today's is not out yet), lined up to the same
     calendar days as the WeatherNext 3 run;
  3. for each forecast window (days 1-3, 4-7, 8-15) and each GloFAS flood threshold (2-, 5- and 20-year return level of
     daily discharge, GloFAS v4 thresholds), the highest share of ensemble members above it on any day of the window.
14_ then decides which crossings carry a flood risk (see flood_crossings there).

Needs: access to the CEMS Early Warning Data Store (EWDS) with the key in ~/.cdsapirc (terms of use accepted), and the
GloFAS v4 threshold and upstream-area files (downloaded once from ECMWF's GloFAS auxiliary-data page, clipped here).

Outputs
  data/raw/glofas/glofas_static_eth.npz                       thresholds and upstream area, clipped (once)
  data/raw/glofas/glofas_xings_<YYYYMMDD>.npz                 forecast at the matched cells (cache)
  data/processed/roads/eth_glofas_xings_<init>.parquet        per crossing: cell, areas, p2/p5/p20 per window, peak Q

    python scripts\\39_glofas_crossings.py --init 2026-11-23
"""
import argparse, glob, os, time
import numpy as np, pandas as pd, geopandas as gpd, shapely

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "glofas"); os.makedirs(RAW, exist_ok=True)
STATIC = os.path.join(RAW, "glofas_static_eth.npz")
XING = os.path.join(ROOT, "data", "processed", "roads", "eth_flood_crossings.parquet")
DL = os.environ.get("GLOFAS_DL", os.path.join(os.path.expanduser("~"), "tools", "glofas_dl"))   # global files, outside synced folders
BOX = (32.9, 48.1, 3.3, 15.1)                    # lon0, lon1, lat0, lat1
RP = (2, 5, 20)
MIN_UPA, MATCH_RAD, MATCH_RATIO = 250.0, 2, 2.0
MIN_RL2 = 10.0                                   # m3/s: below this the 2-year "flood" is a trickle in a mostly dry bed
WINDOWS = [(1, 3), (4, 7), (8, 15)]
P_FLAG = 0.30                                    # share of members above a threshold that counts as a flood signal


def clip(path, var):
    import xarray as xr
    ds = xr.open_dataset(path)
    lat = "lat" if "lat" in ds.dims else "latitude"; lon = "lon" if "lon" in ds.dims else "longitude"
    la = ds[lat].values
    sl = slice(BOX[3], BOX[2]) if la[0] > la[-1] else slice(BOX[2], BOX[3])
    d = ds[var].sel({lon: slice(BOX[0], BOX[1]), lat: sl})
    return d.values.astype("float32"), d[lon].values, d[lat].values


def static():
    if os.path.exists(STATIC): return dict(np.load(STATIC))
    out = {}
    up, lon, lat = clip(os.path.join(DL, "uparea_glofas_v4_0.nc"), "uparea")
    out["upa_km2"], out["lon"], out["lat"] = up / 1e6, lon, lat
    for r in RP:
        a, lo, la = clip(os.path.join(DL, f"flood_threshold_glofas_v4_rl_{r}.0.nc"), f"rl_{r}.0")
        assert np.allclose(lo, lon) and np.allclose(la, lat), "threshold grid differs from upstream-area grid"
        out[f"rl{r}"] = a
    np.savez_compressed(STATIC, **out); print(f"wrote {STATIC}: {up.shape}", flush=True)
    return out


def match(X, S):
    """-> (row, col) of the GloFAS cell for each crossing, -1 where none qualifies."""
    lon, lat, A = S["lon"], S["lat"], S["upa_km2"]
    dx, dy = lon[1] - lon[0], lat[1] - lat[0]
    cx = np.rint((shapely.get_x(X.geometry.values) - lon[0]) / dx).astype(int)
    cy = np.rint((shapely.get_y(X.geometry.values) - lat[0]) / dy).astype(int)
    rr, cc = np.full(len(X), -1), np.full(len(X), -1)
    u = X.upa_km2.values.astype(float)
    for i in np.flatnonzero(u >= MIN_UPA):
        best = None
        for j in range(cy[i] - MATCH_RAD, cy[i] + MATCH_RAD + 1):
            for k in range(cx[i] - MATCH_RAD, cx[i] + MATCH_RAD + 1):
                if not (0 <= j < A.shape[0] and 0 <= k < A.shape[1]): continue
                a = A[j, k]
                if not np.isfinite(a) or a <= 0: continue
                if u[i] >= 65534:                       # 38_ caps catchments at 65,534 km2: any GloFAS cell that large matches
                    ratio = 0.0 if a >= 30000 else np.inf
                else:
                    ratio = abs(np.log(a / u[i]))
                if ratio > np.log(MATCH_RATIO): continue
                key = (round(ratio, 2), (j - cy[i]) ** 2 + (k - cx[i]) ** 2)
                if best is None or key < best[0]: best = (key, j, k)
        if best: rr[i], cc[i] = best[1], best[2]
    return rr, cc


def fetch_forecast(day, cells, S):
    """Ensemble daily discharge [member, lead day 1..16, cell] for forecast issued on `day` (00 UTC)."""
    cache = os.path.join(RAW, f"glofas_xings_{day:%Y%m%d}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        if z["cells"].shape == cells.shape and (z["cells"] == cells).all(): return z["q"]
    import cdsapi, xarray as xr
    key = [l.split(":", 1)[1].strip() for l in open(os.path.expanduser("~/.cdsapirc")) if l.lower().startswith("key")][0]
    c = cdsapi.Client(url="https://ewds.climate.copernicus.eu/api", key=key, quiet=True, timeout=600)
    tmp = os.path.join(DL, f"glofas_fc_{day:%Y%m%d}.nc")
    if not os.path.exists(tmp):
        c.retrieve("cems-glofas-forecast", {
            "system_version": ["operational"], "hydrological_model": ["lisflood"],
            "product_type": ["control_forecast", "ensemble_perturbed_forecasts"],
            "variable": "river_discharge_in_the_last_24_hours",
            "year": [f"{day:%Y}"], "month": [f"{day:%m}"], "day": [f"{day:%d}"],
            "leadtime_hour": [str(24 * k) for k in range(1, 17)],
            "data_format": "netcdf", "download_format": "unarchived",
            "area": [BOX[3], BOX[0], BOX[2], BOX[1]]}, tmp + ".part")
        os.replace(tmp + ".part", tmp)
    # control and ensemble members come back as two netCDFs in a zip (their structures differ)
    import zipfile, tempfile, shutil
    parts, tdir = [tmp], None
    if zipfile.is_zipfile(tmp):
        tdir = tempfile.mkdtemp(dir=DL)
        with zipfile.ZipFile(tmp) as z_: z_.extractall(tdir)
        parts = sorted(glob.glob(os.path.join(tdir, "*.nc")))
    blocks, la, lo = [], None, None
    for f_ in parts:
        with xr.open_dataset(f_) as ds:
            da = ds[[k for k in ds.data_vars if k.lower().startswith("dis")][0]]
            lat = "latitude" if "latitude" in da.dims else "lat"; lon = "longitude" if "longitude" in da.dims else "lon"
            step = "forecast_period" if "forecast_period" in da.dims else "step"
            da = da.squeeze([d for d in da.dims if d not in (lat, lon, step, "number") and da.sizes[d] == 1])
            if "number" not in da.dims: da = da.expand_dims("number")
            blocks.append(da.transpose("number", step, lat, lon).values.astype("float32"))
            la, lo = da[lat].values, da[lon].values
    if tdir: shutil.rmtree(tdir)
    arr = np.concatenate(blocks, 0)                             # [member, lead day, lat, lon]
    yy = np.array([np.argmin(np.abs(la - S["lat"][j])) for j in cells[:, 0]])
    xx = np.array([np.argmin(np.abs(lo - S["lon"][k])) for k in cells[:, 1]])
    q = arr[:, :, yy, xx]                                       # [member, lead, cell]; lead 1 = discharge over the run day
    np.savez_compressed(cache, q=q, cells=cells)
    os.remove(tmp)                                              # keep only the crossings, not the whole grid
    print(f"GloFAS {day:%Y-%m-%d}: {q.shape[0]} members x {q.shape[1]} days at {q.shape[2]} cells", flush=True)
    return q


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--init", required=True); a = ap.parse_args()
    init = pd.Timestamp(a.init); tag = f"{init:%Y%m%d}"
    S = static()
    X = gpd.read_parquet(XING).reset_index(drop=True)
    rr, cc = match(X, S)
    dry = (rr >= 0) & (S["rl2"][np.maximum(rr, 0), np.maximum(cc, 0)] < MIN_RL2)
    print(f"  {dry.sum():,} matched cells have a 2-year flood under {MIN_RL2:.0f} m3/s (ephemeral or model-dry rivers): left to the flash-flood rule", flush=True)
    rr[dry], cc[dry] = -1, -1
    ok = rr >= 0
    print(f"crossings {len(X):,}; on rivers >= {MIN_UPA:.0f} km2: {(X.upa_km2 >= MIN_UPA).sum():,}; matched to a GloFAS cell: {ok.sum():,}", flush=True)
    cells, inv = np.unique(np.c_[rr[ok], cc[ok]], axis=0, return_inverse=True)
    q, used, err = None, None, None
    for back in (0, 1):                                         # today's forecast, else yesterday's
        day = init - pd.Timedelta(days=back)
        try:
            q = fetch_forecast(day, cells, S); used = day; break
        except Exception as e:
            err = e; print(f"GloFAS forecast for {day:%Y-%m-%d} not available: {str(e)[:300]}", flush=True)
    if q is None: raise SystemExit(f"no GloFAS forecast: {err}")
    off = (init - used).days                                    # lead day L of the GloFAS run = forecast day L - off
    R = X[["osm_id", "structure", "upa_km2", "exposed"]].copy()
    R["g_row"], R["g_col"] = rr, cc
    R["g_upa_km2"] = np.where(ok, S["upa_km2"][np.maximum(rr, 0), np.maximum(cc, 0)], np.nan).round(0)
    for r in RP:
        R[f"rl{r}"] = np.where(ok, S[f"rl{r}"][np.maximum(rr, 0), np.maximum(cc, 0)], np.nan).round(1)
    ci = np.full(len(X), -1); ci[ok] = inv.ravel()
    for w, (d0, d1) in enumerate(WINDOWS, start=1):
        leads = [d + off - 1 for d in range(d0, d1 + 1) if 0 <= d + off - 1 < q.shape[1]]
        qq = q[:, leads, :]                                     # [member, day, cell]
        for r in RP:
            th = np.where(ok, R[f"rl{r}"].values, np.nan)
            p = np.full(len(X), np.nan)
            pc = (qq[:, :, ci[ok]] > th[ok][None, None, :]).mean(0).max(0)   # worst day in the window
            p[ok] = pc
            R[f"p{r}_w{w}"] = np.round(p, 2)
        med = np.full(len(X), np.nan); med[ok] = np.median(qq[:, :, ci[ok]], 0).max(0); R[f"qmed_w{w}"] = np.round(med, 1)
    R["glofas_date"] = f"{used:%Y-%m-%d}"
    out = os.path.join(ROOT, "data", "processed", "roads", f"eth_glofas_xings_{tag}.parquet")
    pd.DataFrame(R).to_parquet(out)
    for w in (1, 2, 3):
        print(f"  window {w}: P(> 2-yr) >= {P_FLAG:.0%} at {(R[f'p2_w{w}'] >= P_FLAG).sum():,} crossings; 5-yr {(R[f'p5_w{w}'] >= P_FLAG).sum():,}; "
              f"20-yr {(R[f'p20_w{w}'] >= P_FLAG).sum():,}", flush=True)
    print(f"wrote {out} (GloFAS forecast of {used:%Y-%m-%d})", flush=True)
