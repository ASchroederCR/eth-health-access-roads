"""
33_percent_of_normal.py -- forecast and recent observed rain as % of the CHIRPS 1991-2020 normal for the same calendar
days, plus where the forecast total falls in those 30 years (below / near / above normal, by terciles).

Steps
  * Forecast: WeatherNext 3 daily rain with the monthly bias correction (24_/26_, the amounts shown on the map), summed
    over days 1-3, 4-7, 8-15 and all 15 days. Observed: PARDIG's last 10 days before the run.
  * Both are in PARDIG terms. PARDIG runs drier than CHIRPS in these weeks (32_), so each is divided by the region's
    PARDIG / CHIRPS ratio before comparing with the CHIRPS normal. A region drier than 0.5 mm/day in CHIRPS over those
    weeks uses the national ratio (a ratio of tiny totals is noise); ratios are clipped to 0.6-1.4.
  * Normal: mean of the 30 yearly totals for the same days (31_). Cells whose 15-day normal is under 5 mm (or a
    window's normal under 1 mm) are left out: a few mm there swings the percentage wildly.
  * Terciles: below if the total is under the 30-year 33rd percentile, above if over the 67th. The forecast is an
    ensemble mean, which smooths away extremes, so "above normal" is if anything understated, mostly in days 8-15.

    import:  pn = load("pn", "33_percent_of_normal.py"); F = pn.fields(init, daily, obs10, shape)
    script:  python scripts\\33_percent_of_normal.py --init 2026-11-19     -> charts/ETH_rain_percent_of_normal.png
                                                                           data/processed/tables/eth_rain_percent_of_normal_<init>.csv
"""
import argparse, glob, importlib.util, json, os
import numpy as np, pandas as pd
from rasterio.transform import from_origin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NORMD = os.path.join(ROOT, "data", "raw", "chirps")
RATIO = os.path.join(ROOT, "data", "processed", "pardig_chirps_ratio.json")
WINDOWS = [(1, 3), (4, 7), (8, 15), (1, 15)]
MIN15, MINW = 5.0, 1.0
NOBS = 10


def load(name, file):
    s = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", file))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def normal_file(init):
    first = pd.Timestamp(init) - pd.Timedelta(days=NOBS)
    f = os.path.join(NORMD, f"chirps_normal_{first:%m%d}_{(pd.Timestamp(init) + pd.Timedelta(days=14)):%m%d}.npz")
    if not os.path.exists(f): raise SystemExit(f"missing {f}: run 31_chirps_normal.py --init {pd.Timestamp(init):%Y-%m-%d}")
    return f


def ratio_raster(shape):
    wn = load("wn", "11_weathernext_forecast.py")
    r = json.load(open(RATIO)); R, mm = r["ratio_pardig_over_chirps"], r["chirps_mm_per_day"]
    nat = float(np.clip(R["Ethiopia"], 0.6, 1.4))
    zones = wn.zones_for(from_origin(32.95, 15.05, 0.1, 0.1), shape)
    out = np.full(shape, nat, "float32"); used = {}
    for i, n in enumerate(wn.NAMES):
        v = float(np.clip(R[n], 0.6, 1.4)) if mm.get(n, 0) >= 0.5 else nat
        out[zones == i + 1] = v; used[n] = round(v, 2)
    return out, used, r["window"]


def fields(init, daily, obs10, shape):
    """daily [15, y, x] corrected forecast mm/day; obs10 [10, y, x] PARDIG mm/day (NaN if missing).
    Returns pct / tercile per window key ('w1','w2','w3','all'), observed pct, and the normals."""
    e = np.load(normal_file(init)); N = e["rain"]                   # [year, 25, y, x]
    rr, used, win = ratio_raster(shape)
    out = {"ratio": rr, "ratio_used": used, "ratio_window": win}
    keys = ["w1", "w2", "w3", "all"]
    for k, (a, b) in zip(keys, WINDOWS):
        ny = N[:, NOBS + a - 1:NOBS + b].sum(1)                      # [year, y, x]
        nm = ny.mean(0); lo, hi = np.percentile(ny, [100 / 3, 200 / 3], axis=0)
        fc = np.nan_to_num(daily[a - 1:b]).sum(0) / rr
        ok = nm >= (MIN15 if k == "all" else MINW)
        pct = np.where(ok, 100 * fc / np.maximum(nm, 1e-6), np.nan)
        ter = np.where(fc < lo, -1, np.where(fc > hi, 1, 0)).astype("float32"); ter[~ok] = np.nan
        out[k] = dict(pct=pct, ter=ter, normal=nm, fc=fc)
    no = N[:, :NOBS].sum(1); nm = no.mean(0)
    ob = np.nansum(obs10, 0) / rr; ok = (nm >= MIN15 * NOBS / 15) & np.isfinite(obs10).all(0)
    out["obs"] = dict(pct=np.where(ok, 100 * ob / np.maximum(nm, 1e-6), np.nan), normal=nm, obs=ob,
                      ter=np.where(~ok, np.nan, np.where(ob < np.percentile(no, 100 / 3, axis=0), -1,
                                                          np.where(ob > np.percentile(no, 200 / 3, axis=0), 1, 0))))
    return out


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    ap = argparse.ArgumentParser(); ap.add_argument("--init", required=True); a = ap.parse_args()
    wn = load("wn", "11_weathernext_forecast.py"); wn3 = load("wn3", "21_weathernext3_forecast.py")
    rr = load("rr", "14_roads_rain_risk.py")
    z = wn3.load_run(a.init); init = pd.Timestamp(str(z["init"])); mask = z["mask"]
    H, fld = rr.hazard(z)
    P, _ = wn.load_pardig(); have = set(pd.to_datetime(P.time.values))
    obs10 = np.stack([P.sel(time=init - pd.Timedelta(days=k)).values if (init - pd.Timedelta(days=k)) in have
                      else np.full(mask.shape, np.nan) for k in range(NOBS, 0, -1)])
    F = fields(init, fld["daily"], obs10, mask.shape)
    print("ratios used:", F["ratio_used"])

    # ---- regional table
    zones = wn.zones_for(from_origin(32.95, 15.05, 0.1, 0.1), mask.shape)
    lat = 15.05 - 0.1 * (np.arange(mask.shape[0]) + 0.5); w = wn.area_w(lat, mask.shape)
    rows = []
    for name, m in [("Ethiopia", mask)] + [(n, mask & (zones == i + 1)) for i, n in enumerate(wn.NAMES)]:
        row = {"region": name, "pardig_over_chirps_used": F["ratio_used"].get(name, round(float(np.clip(json.load(open(RATIO))["ratio_pardig_over_chirps"]["Ethiopia"], .6, 1.4)), 2))}
        for k in ("obs", "w1", "w2", "w3", "all"):
            f = F[k]; num = f["obs"] if k == "obs" else f["fc"]
            row[f"{k}_mm_chirps_terms"] = float((num[m] * w[m]).sum() / w[m].sum())
            row[f"{k}_normal_mm"] = float((f["normal"][m] * w[m]).sum() / w[m].sum())
            row[f"{k}_pct_normal"] = 100 * row[f"{k}_mm_chirps_terms"] / max(row[f"{k}_normal_mm"], 1e-6)
            t = f["ter"][m]; ok = np.isfinite(t); ww = w[m][ok]
            for lab, v in (("below", -1), ("above", 1)):
                row[f"{k}_area_{lab}_pct"] = float(100 * ((t[ok] == v) * ww).sum() / max(ww.sum(), 1e-9))
        rows.append(row)
    T = pd.DataFrame(rows).round(1)
    T.to_csv(os.path.join(wn.TB, f"eth_rain_percent_of_normal_{init:%Y%m%d}.csv"), index=False)
    pd.set_option("display.width", 200)
    print(T[["region", "pardig_over_chirps_used", "obs_pct_normal", "w1_pct_normal", "w2_pct_normal", "w3_pct_normal",
             "all_pct_normal", "all_area_below_pct", "all_area_above_pct"]].to_string(index=False))

    # ---- chart: observed 10 days, forecast 15 days (% of normal), forecast tercile
    BR = ["#8c510a", "#d8b365", "#f0efec", "#9ec5f4", "#256abf"]
    bounds = [0, 50, 80, 120, 200, 1e6]
    cm = ListedColormap(BR); nrm = BoundaryNorm(bounds, cm.N)
    tcm = ListedColormap([BR[1], BR[2], BR[3]]); tn = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], 3)
    ext = [32.95, 32.95 + 0.1 * mask.shape[1], 15.05 - 0.1 * mask.shape[0], 15.05]
    plt.rcParams.update({"hatch.color": wn.MUTED, "hatch.linewidth": 0.5})
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.9))
    end_obs = init - pd.Timedelta(days=1); start_obs = init - pd.Timedelta(days=NOBS)
    d15 = init + pd.Timedelta(days=14)
    panels = [(F["obs"]["pct"], cm, nrm, f"Observed, last 10 days ({start_obs:%d}–{end_obs:%d %b})", "% of normal"),
              (F["all"]["pct"], cm, nrm, f"Forecast, next 15 days ({init:%d %b}–{d15:%d %b})", "% of normal"),
              (F["all"]["ter"], tcm, tn, "Forecast 15-day total vs 1991–2020", "tercile")]
    for ax, (arr, c, n, ttl, _) in zip(axs, panels):
        A = np.where(mask, arr, np.nan)
        ax.imshow(A, extent=ext, cmap=c, norm=n, interpolation="nearest")
        nc = (mask & ~np.isfinite(arr)).astype(float)
        ax.contourf(np.linspace(ext[0], ext[1], mask.shape[1]), np.linspace(ext[3], ext[2], mask.shape[0]), nc,
                    levels=[0.5, 1.5], colors="none", hatches=["//////"])
        wn.draw_outline(ax); wn.mapax(ax); ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_title(ttl, loc="left", fontsize=11, color=wn.INK, fontweight="bold", pad=6)
    import matplotlib.patches as mp
    labs = ["< 50%", "50–80%", "80–120%", "120–200%", "> 200%"]
    axs[1].legend([mp.Patch(color=c) for c in BR] + [mp.Patch(facecolor="none", edgecolor=wn.MUTED, hatch="//////", lw=0)], labs + ["dry season, not compared"],
                  loc="lower left", bbox_to_anchor=(-0.62, -0.2), ncol=6, frameon=False, fontsize=9, handlelength=1.2)
    axs[2].legend([mp.Patch(color=c) for c in (BR[1], BR[2], BR[3])], ["below normal (driest third)", "near normal",
                  "above normal (wettest third)"], loc="lower left", bbox_to_anchor=(0, -0.2), ncol=1, frameon=False,
                  fontsize=9, handlelength=1.2)
    tn_ = T.set_index("region").loc["Ethiopia"]
    fig.text(0.01, 0.985, "Rain compared with normal for these days of the year", fontsize=15, fontweight="bold",
             color=wn.INK, va="top")
    fig.text(0.01, 0.925, f"Ethiopia: the last 10 days had {tn_.obs_pct_normal:.0f}% of normal; the next 15 days are forecast "
             f"at {tn_.all_pct_normal:.0f}% (days 1–3 {tn_.w1_pct_normal:.0f}%, 4–7 {tn_.w2_pct_normal:.0f}%, 8–15 "
             f"{tn_.w3_pct_normal:.0f}%). {tn_.all_area_above_pct:.0f}% of the compared area is forecast above normal, "
             f"{tn_.all_area_below_pct:.0f}% below.", fontsize=10, color=wn.INK2, va="top")
    import textwrap
    wn.foot(fig, textwrap.fill(
        f"Normal: CHIRPS v3 daily (ERA5-split, UCSB-CHC/CHIRPS/V3/DAILY_RNL) 1991–2020, same calendar days, on a 0.1° grid. "
        f"Forecast: WeatherNext 3 run {init:%d %b %Y} 00Z, ensemble mean, bias-corrected to PARDIG by month; observed: PARDIG. "
        f"Both put into CHIRPS terms by the regional PARDIG ÷ CHIRPS ratio for {F['ratio_window']} 2023–25 ({min(F['ratio_used'].values()):.2f}–"
        f"{max(F['ratio_used'].values()):.2f} by region; regions drier than 0.5 mm/day then use the national "
        f"{np.clip(json.load(open(RATIO))['ratio_pardig_over_chirps']['Ethiopia'], .6, 1.4):.2f}). Cells with a 15-day normal under 5 mm are not compared. The ensemble mean smooths extremes, so 'above "
        f"normal' is conservative, mostly late in the 15 days.", 230))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.12, wspace=0.04)
    wn.save(fig, os.path.join(wn.OUT, "ETH_rain_percent_of_normal.png"))
    print("saved")
