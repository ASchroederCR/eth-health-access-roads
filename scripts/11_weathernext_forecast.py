"""
11_weathernext_forecast.py -- 15-day rainfall outlook for Ethiopia from Google WeatherNext 2
(64-member ensemble, 0.25 deg, 6-hourly accumulations to 360 h) via Google Earth Engine, plus a check
of an earlier run against PARDIG.

  * Uses 00 UTC runs so forecast days line up with the UTC days of PARDIG and CHIRPS
    (day 1 = init date, forecast hours 6-24).
  * Per member, the 60 x 6 h steps are summed into 15 daily totals on the server and fetched for
    Ethiopia only; cached in data/raw/weathernext/wn2_eth_<YYYYMMDDHH>.npz as [member, day, row, col] mm.

Outputs (<init> = run date):
  charts/ETH_wn2_outlook_maps.png      ensemble-mean totals (days 1-7, 8-15), chance of >= 25 mm, wet-end total
  charts/ETH_wn2_plume.png             national daily rainfall: last 15 days PARDIG + 64-member forecast
  charts/ETH_wn2_adm1.png              15-day total per region: ensemble mean and 10-90 % range
  charts/ETH_wn2_verify_<init>.png     an earlier run against what PARDIG then observed
  data/processed/forecast/eth_wn2_mean15d_<init>.tif, eth_wn2_p25mm_<init>.tif
  data/processed/tables/eth_wn2_adm1_<init>.csv, eth_wn2_daily_<init>.csv, eth_wn2_verify_<init>.csv

    python scripts\\11_weathernext_forecast.py [--init 2026-11-16] [--verify 2026-11-01]
(no -I: earthengine-api lives in user site-packages)
"""
import argparse, datetime as dt, glob, io, json, os
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import BoundaryNorm, ListedColormap
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import shape
import ee

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "weathernext"); os.makedirs(RAW, exist_ok=True)
PRAW = os.path.join(ROOT, "data", "raw", "pardig")
TB = os.path.join(ROOT, "data", "processed", "tables"); FC = os.path.join(ROOT, "data", "processed", "forecast")
OUT = os.path.join(ROOT, "charts"); BND = os.path.join(ROOT, "data", "raw", "boundaries")
WN2 = "projects/gcp-public-data-weathernext/assets/weathernext_2_0_0"
NMEM, NDAY = 64, 15
# Ethiopia on the WN2 0.25 deg grid (cell edges at x.125 / x.375 ...)
X0, Y0, RES, NX, NY = 32.875, 15.125, 0.25, 61, 48
GRID = {"dimensions": {"width": NX, "height": NY}, "crsCode": "EPSG:4326",
        "affineTransform": {"scaleX": RES, "shearX": 0, "translateX": X0, "shearY": 0, "scaleY": -RES, "translateY": Y0}}
TRANSFORM = from_origin(X0, Y0, RES, RES)
lon = X0 + RES * (np.arange(NX) + 0.5); lat = Y0 - RES * (np.arange(NY) + 0.5)
THRESH = 25   # mm in 15 days, for the probability map

# --- style: same tokens as 06_charts.py -------------------------------------------
INK, INK2, MUTED, GRIDC, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
C_FC, C_OBS = "#2a78d6", "#eb6834"
SEQ = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2, "axes.grid": True,
                     "grid.color": GRIDC, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})


def title(ax, t, sub=None):
    ax.set_title(t, loc="left", fontsize=13, color=INK, fontweight="bold", pad=22 if sub else 8)
    if sub: ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9.5, color=INK2, va="bottom")


def foot(fig, txt):
    fig.text(0.01, -0.45 / fig.get_figheight(), txt, fontsize=8, color=MUTED, ha="left", va="top")


def datefmt(ax, step=3):
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=step)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))


def save(fig, path):
    import tempfile, shutil, time
    tmp = os.path.join(tempfile.gettempdir(), "enso_" + os.path.basename(path))
    fig.savefig(tmp); plt.close(fig)
    for i in range(10):
        try: shutil.copyfile(tmp, path); os.remove(tmp); return
        except OSError: time.sleep(2 + i)
    raise OSError(f"could not write {path} (file locked?) - temp copy left at {tmp}")


# --- Earth Engine ---------------------------------------------------------------------
def pixels(img):
    raw = ee.data.computePixels({"expression": img.toFloat().unmask(-9999), "fileFormat": "NPY", "grid": GRID})
    a = np.load(io.BytesIO(raw)) if isinstance(raw, (bytes, bytearray)) else raw
    out = np.stack([a[n].astype("float32") for n in a.dtype.names]); out[out <= -9998] = np.nan
    return out


def member_daily(run, m):
    mem = run.filter(ee.Filter.eq("ensemble_member", str(m))).select("total_precipitation_6hr")
    days = [mem.filter(ee.Filter.rangeContains("forecast_hour", 24 * d + 6, 24 * d + 24)).sum()
              .max(0).multiply(1000).rename(f"d{d + 1:02d}") for d in range(NDAY)]
    return pixels(ee.Image.cat(days))


def fetch_run(init):
    path = os.path.join(RAW, f"wn2_eth_{init:%Y%m%d%H}.npz")
    if os.path.exists(path): return path
    run = ee.ImageCollection(WN2).filter(ee.Filter.eq("start_time", f"{init:%Y-%m-%dT%H:%M:%S}Z"))
    n = run.size().getInfo()
    if n < NMEM * 60: raise SystemExit(f"run {init} incomplete in Earth Engine ({n} of {NMEM * 60} images)")
    print(f"fetching WeatherNext 2 run {init:%Y-%m-%d %H}Z: {NMEM} members x {NDAY} days", flush=True)
    with ThreadPoolExecutor(8) as ex:
        arr = np.stack(list(ex.map(lambda m: member_daily(run, m), range(NMEM))))
    eth = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(ee.Filter.eq("country_na", "Ethiopia"))
    mask = pixels(ee.Image(0).paint(eth, 1))[0] == 1
    np.savez_compressed(path, precip=arr, mask=mask, init=init.isoformat())
    return path


def latest_complete_00z():
    d = dt.datetime.now(dt.timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    for _ in range(5):
        n = ee.ImageCollection(WN2).filter(ee.Filter.eq("start_time", f"{d:%Y-%m-%dT00:00:00}Z")).size().getInfo()
        if n >= NMEM * 60: return d
        d -= dt.timedelta(days=1)
    raise SystemExit("no complete 00Z run in the last 5 days")


# --- shared geometry ------------------------------------------------------------------
gj = json.load(open(os.path.join(BND, "geoboundaries_adm1", "ETH_ADM1.geojson"), encoding="utf-8"))
NAMES = [f["properties"]["shapeName"] for f in gj["features"]]
GEOMS = [shape(f["geometry"]) for f in gj["features"]]


def zones_for(transform, shp):
    return rasterize([(g, i + 1) for i, g in enumerate(GEOMS)], out_shape=shp, transform=transform, fill=0,
                     dtype="int16", all_touched=False)


def area_w(latv, shp): return np.cos(np.deg2rad(latv))[:, None] * np.ones(shp)


def regional_means(a, mask, zones, w):
    """a [..., y, x] -> dict name -> area mean [...] (national under 'Ethiopia')."""
    out = {"Ethiopia": (a[..., mask] * w[mask]).sum(-1) / w[mask].sum()}
    for i, n in enumerate(NAMES):
        k = mask & (zones == i + 1)
        if k.sum(): out[n] = (a[..., k] * w[k]).sum(-1) / w[k].sum()
    return out


def draw_outline(ax):
    for g in GEOMS:
        for poly in (g.geoms if g.geom_type == "MultiPolygon" else [g]):
            x, y = poly.exterior.xy; ax.plot(x, y, color="#ffffff", lw=0.6); ax.plot(x, y, color=INK2, lw=0.35)


EXT = [X0, X0 + NX * RES, Y0 - NY * RES, Y0]


def mapax(ax):
    ax.set_xlim(EXT[0], EXT[1]); ax.set_ylim(EXT[2], EXT[3]); ax.set_aspect("equal"); ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)


def load_pardig():
    """Daily PARDIG on its 0.1 deg grid, Earth Engine year + GCS top-up, as an xarray DataArray."""
    e = np.load(sorted(glob.glob(os.path.join(PRAW, "eth_daily_*.npz")))[-1])
    da = xr.DataArray(e["pardig"], dims=("time", "latitude", "longitude"),
                      coords={"time": pd.to_datetime(e["dates"]), "latitude": e["lat"], "longitude": e["lon"]})
    for f in sorted(glob.glob(os.path.join(PRAW, "pardig_nrt_daily_*.nc"))):
        g = xr.open_dataset(f).precipitation.load()
        g = g.assign_coords(latitude=e["lat"], longitude=e["lon"])   # same grid, checked in 10_
        da = xr.concat([da, g.sel(time=~g.time.isin(da.time))], "time")
    return da.sortby("time"), e["mask"]


# --- main -----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=os.environ.get("EE_PROJECT")); ap.add_argument("--init"); ap.add_argument("--verify")
    a = ap.parse_args()
    ee.Initialize(project=a.project)
    init = dt.datetime.fromisoformat(a.init) if a.init else latest_complete_00z()
    d = np.load(fetch_run(init)); P, mask = d["precip"], d["mask"]          # [mem, day, y, x]
    days = pd.date_range(init, periods=NDAY, freq="D")
    tag = f"{init:%Y%m%d}"
    SRC = (f"Source: Google WeatherNext 2, 64-member ensemble run {init:%d %b %Y} 00 UTC (0.25°), via Google "
           f"Earth Engine. Forecast days are UTC days {days[0]:%d %b} – {days[-1]:%d %b %Y}.")
    zones = zones_for(TRANSFORM, mask.shape); w = area_w(lat, mask.shape)

    # ---- fields
    tot = P.sum(1)                                     # [mem, y, x] 15-day totals
    wk1, wk2 = P[:, :7].sum(1).mean(0), P[:, 7:].sum(1).mean(0)
    mean15 = tot.mean(0); p_thr = (tot >= THRESH).mean(0) * 100; p90 = np.percentile(tot, 90, axis=0)
    for f in (wk1, wk2, mean15, p_thr, p90): f[~mask] = np.nan

    def tif(name, arr):
        with rasterio.open(os.path.join(FC, name), "w", driver="GTiff", height=NY, width=NX, count=1, dtype="float32",
                           crs="EPSG:4326", transform=TRANSFORM, nodata=np.nan, compress="deflate") as r:
            r.write(arr.astype("float32"), 1)
    tif(f"eth_wn2_mean15d_{tag}.tif", mean15); tif(f"eth_wn2_p{THRESH}mm_{tag}.tif", p_thr)

    # ---- tables
    reg = regional_means(tot, mask, zones, w)          # name -> [mem]
    adm = pd.DataFrame([{"region": n, "mean_mm": v.mean(), "p10_mm": np.percentile(v, 10),
                         "median_mm": np.median(v), "p90_mm": np.percentile(v, 90),
                         f"members_ge_{THRESH}mm_pct": (v >= THRESH).mean() * 100} for n, v in reg.items()])
    adm.round(2).to_csv(os.path.join(TB, f"eth_wn2_adm1_{tag}.csv"), index=False)
    nat = regional_means(P, mask, zones, w)["Ethiopia"]   # [mem, day]
    dly = pd.DataFrame({"date": days.date, "mean": nat.mean(0), "p10": np.percentile(nat, 10, 0),
                        "median": np.median(nat, 0), "p90": np.percentile(nat, 90, 0)})
    dly.round(3).to_csv(os.path.join(TB, f"eth_wn2_daily_{tag}.csv"), index=False)

    # ---- chart 1: maps
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 9.4))
    lv_tot = [0, 2, 5, 10, 25, 50, 75, 100, 150]
    panels = [(axs[0, 0], wk1, f"Days 1–7 ({days[0]:%d}–{days[6]:%d %b})", lv_tot, "Ensemble-mean rainfall, mm"),
              (axs[0, 1], wk2, f"Days 8–15 ({days[7]:%d}–{days[-1]:%d %b})", lv_tot, "Ensemble-mean rainfall, mm"),
              (axs[1, 0], p_thr, f"Chance of ≥ {THRESH} mm over 15 days", [0, 10, 20, 35, 50, 65, 80, 90, 100],
               "% of the 64 members"),
              (axs[1, 1], p90, "Wet end: 15-day total, 90th percentile", lv_tot, "mm (1 in 10 members wetter)")]
    for ax, f, t, lv, lab in panels:
        im = ax.imshow(np.ma.masked_invalid(f), extent=EXT, cmap=ListedColormap(SEQ), norm=BoundaryNorm(lv, len(SEQ)),
                       interpolation="nearest")
        draw_outline(ax); mapax(ax); ax.set_title(t, loc="left", fontsize=11, color=INK, fontweight="bold")
        cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, aspect=28, orientation="horizontal", extend="max" if lv[-1] != 100 else "neither")
        cb.set_label(lab, color=INK2, fontsize=8.5); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8, colors=INK2)
    n0 = adm.iloc[0]
    fig.suptitle(f"Ethiopia 15-day rainfall outlook from {days[0]:%d %b %Y}", x=0.01, ha="left", y=1.0, va="bottom",
                 fontsize=14, color=INK, fontweight="bold")
    fig.text(0.01, 0.985, f"National 15-day total: ensemble mean {n0.mean_mm:.0f} mm, 80% of members between "
             f"{n0.p10_mm:.0f} and {n0.p90_mm:.0f} mm.", fontsize=9.5, color=INK2)
    foot(fig, SRC); save(fig, os.path.join(OUT, "ETH_wn2_outlook_maps.png"))

    # ---- chart 2: plume with recent PARDIG
    pdg, pmask = load_pardig()
    plat = pdg.latitude.values; pw = area_w(plat, pmask.shape)
    obs = pdg.sel(time=slice(init - dt.timedelta(days=15), init - dt.timedelta(days=1)))
    obs_nat = (obs.values[:, pmask] * pw[pmask]).sum(-1) / pw[pmask].sum()
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for m in range(NMEM): ax.plot(days, nat[m], color=C_FC, lw=0.5, alpha=0.18)
    ax.fill_between(days, dly.p10, dly.p90, color=C_FC, alpha=0.15, lw=0, label="Forecast, 10–90% of members")
    ax.plot(days, dly["mean"], color=C_FC, lw=2.2, label="Forecast, ensemble mean")
    if len(obs_nat): ax.plot(obs.time.values, obs_nat, color=C_OBS, lw=2.2, label="Observed (PARDIG)")
    ax.axvline(init, color=AXIS, lw=1, ls="--"); ax.text(init, ax.get_ylim()[1], " run start", color=MUTED, fontsize=8.5, va="top")
    ax.set_ylim(bottom=0); ax.set_ylabel("mm/day, national mean"); ax.legend(frameon=False, loc="upper left", fontsize=9)
    datefmt(ax)
    title(ax, "Daily rainfall over Ethiopia: last 15 days and next 15",
          "Thin lines: the 64 ensemble members. Area-weighted national mean.")
    foot(fig, SRC + " Observed: PARDIG_1_NRT."); save(fig, os.path.join(OUT, "ETH_wn2_plume.png"))

    # ---- chart 3: admin-1 range
    r = adm[adm.region != "Ethiopia"].sort_values("mean_mm").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(r) + 1.6)); y = np.arange(len(r))
    ax.hlines(y, r.p10_mm, r.p90_mm, color=C_FC, alpha=0.35, lw=6, label="10–90% of members")
    ax.scatter(r.mean_mm, y, s=60, color=C_FC, edgecolor=SURF, linewidth=2, zorder=3, label="Ensemble mean")
    for yi, row in zip(y, r.itertuples()):
        ax.text(row.p90_mm + 2, yi, f"{getattr(row, f'members_ge_{THRESH}mm_pct'):.0f}% ≥ {THRESH} mm", va="center",
                fontsize=8.5, color=INK2)
    ax.set_yticks(y, r.region); ax.grid(axis="y", visible=False); ax.set_xlim(0, r.p90_mm.max() * 1.3)
    ax.set_xlabel(f"Rainfall {days[0]:%d %b} – {days[-1]:%d %b}, mm"); ax.legend(frameon=False, loc="lower right", fontsize=9)
    title(ax, "15-day rainfall outlook by region", "Area-weighted regional mean; label: share of members with ≥ "
          f"{THRESH} mm")
    foot(fig, SRC); save(fig, os.path.join(OUT, "ETH_wn2_adm1.png"))

    # ---- verification of an earlier run against PARDIG
    vinit = dt.datetime.fromisoformat(a.verify) if a.verify else init - dt.timedelta(days=NDAY)
    vdays = pd.date_range(vinit, periods=NDAY, freq="D")
    if vdays[-1] <= pdg.time.values.max():
        V = np.load(fetch_run(vinit))["precip"]
        vnat = regional_means(V, mask, zones, w)["Ethiopia"]
        o = pdg.sel(time=vdays)
        o_nat = (o.values[:, pmask] * pw[pmask]).sum(-1) / pw[pmask].sum()
        pz = zones_for(from_origin(pdg.longitude.values[0] - 0.05, plat[0] + 0.05, 0.1, 0.1), pmask.shape)
        oreg = regional_means(o.values.sum(0), pmask, pz, pw)       # name -> scalar
        freg = regional_means(V.sum(1), mask, zones, w)              # name -> [mem]
        ver = pd.DataFrame([{"region": n, "observed_pardig_mm": oreg[n], "fc_mean_mm": freg[n].mean(),
                             "fc_p10_mm": np.percentile(freg[n], 10), "fc_p90_mm": np.percentile(freg[n], 90),
                             "obs_rank_pct": (freg[n] < oreg[n]).mean() * 100} for n in freg])
        ver.round(2).to_csv(os.path.join(TB, f"eth_wn2_verify_{vinit:%Y%m%d}.csv"), index=False)
        r_d = np.corrcoef(vnat.mean(0), o_nat)[0, 1]
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.35})
        ax = axs[0]
        for m in range(NMEM): ax.plot(vdays, vnat[m], color=C_FC, lw=0.5, alpha=0.18)
        ax.fill_between(vdays, np.percentile(vnat, 10, 0), np.percentile(vnat, 90, 0), color=C_FC, alpha=0.15, lw=0,
                        label="Forecast, 10–90% of members")
        ax.plot(vdays, vnat.mean(0), color=C_FC, lw=2.2, label="Forecast, ensemble mean")
        ax.plot(vdays, o_nat, color=C_OBS, lw=2.2, label="Observed (PARDIG)")
        ax.set_ylim(bottom=0); ax.set_ylabel("mm/day, national mean"); ax.legend(frameon=False, loc="upper right", fontsize=9)
        datefmt(ax, 2)
        title(ax, "National daily rainfall", f"Correlation of ensemble mean with observed r = {r_d:.2f}")
        ax = axs[1]; vv = ver[ver.region != "Ethiopia"].sort_values("observed_pardig_mm").reset_index(drop=True)
        yy = np.arange(len(vv))
        ax.hlines(yy, vv.fc_p10_mm, vv.fc_p90_mm, color=C_FC, alpha=0.35, lw=6, label="Forecast 10–90%")
        ax.scatter(vv.fc_mean_mm, yy, s=50, color=C_FC, edgecolor=SURF, linewidth=2, zorder=3, label="Forecast mean")
        ax.scatter(vv.observed_pardig_mm, yy, s=50, color=C_OBS, marker="D", edgecolor=SURF, linewidth=1.5, zorder=4,
                   label="Observed (PARDIG)")
        ax.set_yticks(yy, vv.region); ax.grid(axis="y", visible=False); ax.set_xlim(left=0)
        ax.set_xlabel("15-day total, mm"); ax.legend(frameon=False, loc="lower right", fontsize=8.5)
        inside = ((vv.observed_pardig_mm >= vv.fc_p10_mm) & (vv.observed_pardig_mm <= vv.fc_p90_mm)).sum()
        title(ax, "15-day totals by region", f"Observed inside the 10–90% range in {inside} of {len(vv)} regions")
        fig.suptitle(f"How the {vinit:%d %b} run did: forecast vs observed, {vdays[0]:%d} – {vdays[-1]:%d %b %Y}",
                     x=0.01, ha="left", y=1.02, va="bottom", fontsize=14, color=INK, fontweight="bold")
        foot(fig, f"Forecast: WeatherNext 2 run {vinit:%d %b %Y} 00 UTC, 64 members. Observed: PARDIG_1_NRT, which "
                  f"read ~11% wetter than CHIRPS v3 over the past year, so this is a check against PARDIG, not ground truth.")
        save(fig, os.path.join(OUT, f"ETH_wn2_verify_{vinit:%Y%m%d}.png"))
        print(ver.round(1).to_string(index=False)); print(f"daily national r = {r_d:.2f}")
    else:
        print(f"skip verification: PARDIG ends {pd.Timestamp(pdg.time.values.max()):%Y-%m-%d}, run {vinit:%Y-%m-%d} needs to {vdays[-1]:%Y-%m-%d}")

    pd.set_option("display.width", 200)
    print(adm.round(1).to_string(index=False)); print(dly.round(2).to_string(index=False))
