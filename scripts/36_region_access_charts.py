"""
36_region_access_charts.py -- one chart + map per region: how much longer health centres take to reach their nearest
hospital over the 15-day forecast, by forecast window and as a 15-day mean.

Trip: health centre -> the hospital it reaches fastest by road at normal speeds (17_). Added minutes = forecast time -
normal time on that same trip, with roads slowed by each window's own tier (days 1-3, 4-7, 8-15).
15-day mean added minutes = the day-weighted mean over the 15 days, taking each window's added minutes for each of its
days: (3 x days 1-3 + 4 x days 4-7 + 8 x days 8-15) / 15. The worst-window figure ("all 15 days" elsewhere in the
project) is shown alongside for reference.

Regions are OCHA COD-AB 2025 admin-1 polygons (14 regions + the contested area); each health centre is assigned by
location (nearest region within ~1 km if it falls just outside a boundary).
Facilities > 10 km from a mapped road are left out.

Outputs (<init> = forecast run date)
  charts/regions/ETH_hc_access_<region>.png                     per region: map + windows + most affected centres
  charts/ETH_hc_access_regions.png                              all regions side by side
  data/processed/tables/eth_hc_access_by_region_<init>.csv      per region summary
  data/processed/tables/eth_hc_access_15day_mean_<init>.csv     every health centre: added minutes per window + 15-day mean

    python scripts\\36_region_access_charts.py [--init 2026-11-19]
"""
import argparse, glob, importlib.util, os, re, textwrap
import numpy as np, pandas as pd, geopandas as gpd, shapely
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("wn", os.path.join(ROOT, "scripts", "11_weathernext_forecast.py"))
wn = importlib.util.module_from_spec(spec); spec.loader.exec_module(wn)
INK, INK2, MUTED, SURF = wn.INK, wn.INK2, wn.MUTED, wn.SURF
PROC = os.path.join(ROOT, "data", "processed", "roads")
RDIR = os.path.join(wn.OUT, "regions"); os.makedirs(RDIR, exist_ok=True)
DAYS = {1: 3, 2: 4, 3: 8}                                   # days in each forecast window
WLAB = {1: "Days 1–3", 2: "Days 4–7", 3: "Days 8–15"}
DLV = [0, 5, 15, 30, 60, 1e9]; DLAB = ["< 5 min", "5–15", "15–30", "30–60", "60+ min"]
DCOL = ["#d9d8d2", "#f6cf7a", "#ec9a3b", "#c95a1c", "#7d2a0c"]
C_WIN = "#e0701f"; C_MEAN = "#7d2a0c"; C_WORST = "#c9c8c2"
COSLAT = np.cos(np.deg2rad(9.0))


def slug(s): return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def hrs(m): return f"{m / 60:.1f} h" if m >= 90 else f"{m:.0f} min"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--init", default=None); a = ap.parse_args()
    if not a.init:                                        # latest run with access results
        a.init = pd.Timestamp(sorted(glob.glob(os.path.join(wn.TB, "eth_health_access_2*.csv")))[-1].split("_")[-1][:8]).strftime("%Y-%m-%d")
    tag = a.init.replace("-", ""); init = pd.Timestamp(a.init)
    d = pd.read_csv(os.path.join(wn.TB, f"eth_health_access_{tag}.csv"))
    need = [f"to_hosp_added_w{w}_min" for w in DAYS]
    if not set(need) <= set(d.columns): raise SystemExit("per-window columns missing: rerun 17_health_access.py")
    hc = d[(d.Type == "Health Center") & ~d.off_network & d.to_hosp_min.notna()].copy()
    # region by location (OCHA COD-AB 2025 polygons)
    REG = gpd.GeoDataFrame({"region": wn.NAMES}, geometry=wn.GEOMS, crs="EPSG:4326")
    pts = gpd.GeoDataFrame(hc[["Id"]], geometry=gpd.points_from_xy(hc.Longitude, hc.Latitude), crs="EPSG:4326")
    j = gpd.sjoin_nearest(pts.to_crs(32637), REG.to_crs(32637), how="left", max_distance=1000)
    hc["region"] = j[~j.index.duplicated()].region.reindex(hc.index).values
    print(f"{hc.region.isna().sum()} health centres outside every region, left out")
    hc = hc[hc.region.notna()]
    hc["mean15_added_min"] = sum(DAYS[w] * hc[f"to_hosp_added_w{w}_min"] for w in DAYS) / 15
    hc["mean15_added_min"] = hc.mean15_added_min.round(1)
    hosp = d[d.Type == "Hospital"]
    keep = ["Id", "Name", "region", "admin1Name", "admin2Name", "admin3Name", "Latitude", "Longitude", "nearest_hospital", "to_hosp_min"] + need + ["mean15_added_min", "to_hosp_added_min"]
    hc[keep].rename(columns={"to_hosp_added_min": "worst_window_added_min"}).to_csv(
        os.path.join(wn.TB, f"eth_hc_access_15day_mean_{tag}.csv"), index=False)

    g = gpd.read_parquet(os.path.join(PROC, f"eth_roads_scored_{tag}.parquet"), columns=["tier", "highway", "geometry"])
    sev = g[g.tier.isin(["Elevated", "Severe"])]
    main = g[g.highway.str.replace("_link", "").isin(["motorway", "trunk", "primary", "secondary"])]
    def segs(df):
        xy, idx = shapely.get_coordinates(df.geometry.values, return_index=True)
        return np.split(xy, np.flatnonzero(np.diff(idx)) + 1)
    SEV, MAIN = segs(sev), segs(main)
    cmap, norm = ListedColormap(DCOL), BoundaryNorm(DLV, len(DCOL))
    span = f"{init:%d %b} – {init + pd.Timedelta(days=14):%d %b %Y}"

    rows = []
    regions = hc.region.value_counts().index.tolist()
    for reg in regions:
        R = hc[hc.region == reg]
        st = {"region": reg, "health_centres": len(R),
              "median_normal_min": round(float(R.to_hosp_min.median()), 1),
              **{f"mean_added_{WLAB[w].lower().replace(' ', '_').replace('–', '_')}": round(float(R[f'to_hosp_added_w{w}_min'].mean()), 1) for w in DAYS},
              "mean_added_15day_mean": round(float(R.mean15_added_min.mean()), 1),
              "mean_added_worst_window": round(float(R.to_hosp_added_min.mean()), 1),
              "hc_15day_mean_30plus": int((R.mean15_added_min >= 30).sum()),
              **{f"hc_30plus_w{w}": int((R[f'to_hosp_added_w{w}_min'] >= 30).sum()) for w in DAYS},
              "hc_worst_window_30plus": int((R.to_hosp_added_min >= 30).sum())}
        rows.append(st)

        # --- figure: map | windows + top centres -------------------------------------------------
        fig = plt.figure(figsize=(15, 8.6))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[0.8, 1.2], wspace=0.5, hspace=0.5)
        ax = fig.add_subplot(gs[:, 0]); b1 = fig.add_subplot(gs[0, 1]); b2 = fig.add_subplot(gs[1, 1])
        x0, x1 = R.Longitude.quantile([0.0, 1.0]); y0, y1 = R.Latitude.quantile([0.0, 1.0])
        pad = max(0.15, 0.08 * max(x1 - x0, y1 - y0)); x0 -= pad; x1 += pad; y0 -= pad; y1 += pad
        # square-ish frame matching the panel
        cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, max((x1 - x0) * COSLAT, (y1 - y0)) / 2
        x0, x1, y0, y1 = cx - half / COSLAT * 1.05, cx + half / COSLAT * 1.05, cy - half, cy + half
        ax.add_collection(LineCollection(MAIN, colors="#bdbcb5", linewidths=0.5, zorder=1))
        ax.add_collection(LineCollection(SEV, colors="#e7c3ad", linewidths=0.45, zorder=1.5))
        wn.draw_outline(ax)
        oth = hc[hc.region != reg]
        ax.scatter(oth.Longitude, oth.Latitude, s=6, color="#c9c8c2", lw=0, zorder=2, alpha=0.6)
        Rs = R.sort_values("mean15_added_min")
        ax.scatter(Rs.Longitude, Rs.Latitude, c=Rs.mean15_added_min, cmap=cmap, norm=norm, s=np.where(Rs.mean15_added_min >= 30, 46, 22),
                   edgecolors=np.where(Rs.mean15_added_min >= 30, SURF, "none"), linewidths=0.5, zorder=3)
        hv = hosp[hosp.Longitude.between(x0, x1) & hosp.Latitude.between(y0, y1)]
        ax.scatter(hv.Longitude, hv.Latitude, marker="s", s=34, facecolors="none", edgecolors="#000000", linewidths=1.0, zorder=4)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect(1 / COSLAT); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s_ in ax.spines.values(): s_.set_color("#d9d8d2")
        ax.set_title("Health centres by 15-day mean added minutes to their hospital", loc="left", fontsize=11, color=INK, pad=6)
        hd = [Line2D([], [], marker="o", ls="", color=c, markersize=7, label=l) for c, l in zip(DCOL, DLAB)]
        hd += [Line2D([], [], marker="s", ls="", markerfacecolor="none", markeredgecolor="#000000", markeredgewidth=1.1, markersize=7, label="hospital"),
               Line2D([], [], color="#e7c3ad", lw=2, label="Elevated / Severe road (worst window)"),
               Line2D([], [], color="#bdbcb5", lw=2, label="main road"),
               Line2D([], [], marker="o", ls="", color="#c9c8c2", markersize=5, label="other regions' centres")]
        ax.legend(handles=hd, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=4, frameon=False, fontsize=8.5,
                  columnspacing=0.9, handletextpad=0.3)

        # windows: mean added minutes and number of centres 30+ min longer
        lab = [WLAB[1], WLAB[2], WLAB[3], "15-day mean", "Worst window"]
        vals = [R[f"to_hosp_added_w{w}_min"].mean() for w in DAYS] + [R.mean15_added_min.mean(), R.to_hosp_added_min.mean()]
        n30 = [int((R[f"to_hosp_added_w{w}_min"] >= 30).sum()) for w in DAYS] + [int((R.mean15_added_min >= 30).sum()), int((R.to_hosp_added_min >= 30).sum())]
        cols = [C_WIN] * 3 + [C_MEAN, C_WORST]
        b1.bar(range(5), vals, color=cols, width=0.62)
        top = max(vals) if max(vals) > 0 else 1
        for i, (v, n) in enumerate(zip(vals, n30)):
            b1.text(i, v + top * 0.03, f"+{v:.0f} min", ha="center", va="bottom", fontsize=9.5, color=INK, fontweight="medium")
            b1.text(i, v + top * 0.13, f"{n} at 30+", ha="center", va="bottom", fontsize=8, color=MUTED)
        b1.set_xticks(range(5), lab, fontsize=9); b1.tick_params(axis="x", length=0)
        b1.set_ylim(0, top * 1.42); b1.set_ylabel("mean added minutes"); b1.grid(axis="x", visible=False)
        b1.axvline(2.5, color=MUTED, lw=0.7, ls=":")
        wn.title(b1, "Added time to the nearest hospital, by forecast window", f"mean over {len(R)} health centres (grey: how many are 30+ min longer) · normal trip median {hrs(R.to_hosp_min.median())}")

        # most affected centres by 15-day mean
        T = R.nlargest(12, "mean15_added_min").iloc[::-1]
        y = np.arange(len(T))
        b2.barh(y, T.mean15_added_min, color=C_MEAN, height=0.6, zorder=3)
        b2.scatter(T.to_hosp_added_min, y, marker="|", s=110, color=INK2, lw=1.6, zorder=4)
        for yi, r in zip(y, T.itertuples()):
            b2.text(r.mean15_added_min, yi, f"  +{r.mean15_added_min:.0f}", va="center", fontsize=8.5, color=INK, zorder=5)
        names = [textwrap.shorten(f"{n.strip()} · {z}", 38, placeholder="…") for n, z in zip(T.Name, T.admin2Name)]
        b2.set_yticks(y, names, fontsize=8.5); b2.grid(axis="y", visible=False)
        b2.set_xlim(0, max(T.to_hosp_added_min.max(), T.mean15_added_min.max()) * 1.18 + 1)
        b2.set_xlabel("added minutes to the nearest hospital  (bar: 15-day mean · tick: worst window)")
        wn.title(b2, "Most affected health centres", "by 15-day mean added minutes")

        fig.suptitle(f"{reg}: how much longer health centres take to reach their hospital under the forecast rain",
                     x=0.01, ha="left", fontsize=13.5, fontweight="bold", color=INK, y=0.995)
        fig.text(0.01, 0.955, f"WeatherNext 3 run {init:%d %b %Y}, 15 days ({span}) · road + slope model · "
                 f"15-day mean = (3 × days 1–3 + 4 × days 4–7 + 8 × days 8–15) ÷ 15", fontsize=9.5, color=INK2)
        wn.foot(fig, "Trip: each health centre to the hospital it reaches fastest by road at normal speeds; added minutes = forecast minus normal on that trip, "
                     "roads slowed by each window's tier (Minor 0.9, Elevated 0.75, Severe 0.5).\nRegions: OCHA COD-AB 2025 admin-1, centres assigned by location. Centres > 10 km from a mapped road left out. Speeds and slowdowns are illustrative, not calibrated. "
                     "Roads © OpenStreetMap contributors.")
        wn.save(fig, os.path.join(RDIR, f"ETH_hc_access_{slug(reg)}.png")); print(f"  {reg}: {len(R)} centres")

    # drop region charts from earlier runs whose region no longer exists (e.g. the old 11-region names)
    made = {f"ETH_hc_access_{slug(r)}.png" for r in regions}
    for f_ in glob.glob(os.path.join(RDIR, "ETH_hc_access_*.png")):
        if os.path.basename(f_) not in made: os.remove(f_); print(f"  removed old chart {os.path.basename(f_)}")
    S = pd.DataFrame(rows).sort_values("mean_added_15day_mean", ascending=False)
    S.to_csv(os.path.join(wn.TB, f"eth_hc_access_by_region_{tag}.csv"), index=False)
    print(S.to_string(index=False))

    # --- all regions side by side ---------------------------------------------------------------
    fig, (c1, c2) = plt.subplots(1, 2, figsize=(15, 6.4), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.35})
    S2 = S.iloc[::-1]; y = np.arange(len(S2)); h = 0.2
    wk = [c for c in S.columns if c.startswith("mean_added_days")]
    for i, (c, w) in enumerate(zip(wk, DAYS)):
        c1.barh(y + (1 - i) * h, S2[c], height=h, color=["#f6cf7a", "#ec9a3b", "#c95a1c"][i], label=WLAB[w], zorder=3)
    c1.scatter(S2.mean_added_15day_mean, y, marker="D", s=34, color=C_MEAN, zorder=4, label="15-day mean")
    for yi, v in zip(y, S2.mean_added_15day_mean): c1.text(v + 0.9, yi + 0.02, f"{v:.0f}", va="center", fontsize=8.5, color=C_MEAN, fontweight="medium")
    c1.set_yticks(y, [f"{r}  ({n})" for r, n in zip(S2.region, S2.health_centres)], fontsize=9.5); c1.grid(axis="y", visible=False)
    c1.set_xlabel("mean added minutes, health centre → nearest hospital"); c1.legend(frameon=False, fontsize=9, loc="lower right")
    wn.title(c1, "Mean added time by region and forecast window", "(number of health centres)")
    pct = 100 * S2.hc_15day_mean_30plus / S2.health_centres
    c2.barh(y, pct, color=C_MEAN, height=0.6, zorder=3)
    for yi, p_, n in zip(y, pct, S2.hc_15day_mean_30plus): c2.text(p_, yi, f"  {p_:.0f}%  ({n})", va="center", fontsize=8.5, color=INK)
    c2.set_yticks(y, S2.region, fontsize=9.5); c2.grid(axis="y", visible=False); c2.set_xlim(0, max(10, pct.max() * 1.35))
    c2.set_xlabel("% of health centres")
    wn.title(c2, "Health centres 30+ min longer, 15-day mean", "share of the region's centres (count)")
    fig.suptitle("Health centre to hospital: added travel time under the forecast rain, by region", x=0.01, ha="left", fontsize=13.5,
                 fontweight="bold", color=INK, y=1.07)
    fig.text(0.01, 1.005, f"WeatherNext 3 run {init:%d %b %Y}, 15 days · road + slope model · 15-day mean = (3 × days 1–3 + 4 × days 4–7 + 8 × days 8–15) ÷ 15",
             fontsize=9.5, color=INK2)
    wn.foot(fig, "Trip: each health centre to the hospital it reaches fastest by road at normal speeds; added minutes = forecast minus normal on that trip. "
                 "Regions: OCHA COD-AB 2025 admin-1, centres assigned by location. Centres > 10 km from a mapped road left out. Speeds and slowdowns are illustrative, not calibrated.")
    wn.save(fig, os.path.join(wn.OUT, "ETH_hc_access_regions.png"))
