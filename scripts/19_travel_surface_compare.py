"""
19_travel_surface_compare.py -- how different would the accessibility results be with the Malaria Atlas Project
motorized travel-time surface instead of (or alongside) our road + slope model?

Inputs
  data/travel-surfaces/202001_Global_Motorized_Travel_Time_to_Healthcare_ETH.tiff
      MAP 2020 (Weiss et al. 2020, Nature Medicine) motorized travel time to the nearest healthcare facility, minutes,
      ~1 km. NOTE: the supplied clip only has values for north-west Ethiopia (~34.3-37.5 E, 10.1-14.5 N).
      It is a finished travel-time result, not a friction (speed) surface, so it can be compared with our model
      cell by cell but cannot be re-routed to our hospitals.
  data/processed/roads/eth_roads_scored_<init>.parquet   our scored roads (14_), same speeds as 17_health_access.py
  data/health_facilities/ethiopian-health-facilities.csv approved Hospitals, Health Centers, Health Posts
  data/raw/exposure/worldpop_2020_1km/eth_ppp_2020_1km_Aggregated.tif   population weights

Method
  For every MAP cell with a value, our travel time to the nearest facility (any of the three types) =
    min( walk straight to the nearest facility at 5 km/h,
         walk to the nearest road vertex at 5 km/h + road-network time to the nearest facility )
  under normal speeds (ours_normal) and forecast speeds (ours_forecast), multi-source Dijkstra from all facilities.
  MAP has no rain. "MAP + forecast" applies our per-cell forecast/normal ratio to MAP's baseline, i.e. what MAP's
  times would become if each place slowed by the same factor our model gives it.
  Differences between ours and MAP mix three things: off-road travel (MAP uses land-cover speeds, we walk at 5 km/h),
  the road network and speeds (MAP: OSM + Google roads 2019-20; ours: current OSM with class speeds), and the
  facility list (MAP's global compilation vs the Ethiopian register).

Outputs
  charts/ETH_travel_surface_compare_map.png     MAP vs our normal times, their difference, forecast added time on each
  charts/ETH_travel_surface_compare_stats.png   cell scatter, population beyond 30/60/120 min under four versions
  data/processed/tables/eth_travel_surface_compare_<init>.csv   summary by zone

    python scripts\\19_travel_surface_compare.py [--init 2026-11-17]
"""
import argparse, glob, importlib.util, os
import numpy as np, pandas as pd, geopandas as gpd, rasterio, shapely
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("ha", os.path.join(ROOT, "scripts", "17_health_access.py"))
ha = importlib.util.module_from_spec(spec); spec.loader.exec_module(ha)
wn = ha.wn
INK, INK2, MUTED, SURF = wn.INK, wn.INK2, wn.MUTED, wn.SURF
MAPF = os.path.join(ROOT, "data", "travel-surfaces", "202001_Global_Motorized_Travel_Time_to_Healthcare_ETH.tiff")
POPF = os.path.join(ROOT, "data", "raw", "exposure", "worldpop_2020_1km", "eth_ppp_2020_1km_Aggregated.tif")
TT_LV = [0, 15, 30, 60, 120, 240, 1e9]; TT_LAB = ["<15", "15–30", "30–60", "1–2 h", "2–4 h", "4 h+"]
TT_COL = ["#f4f1e6", "#d5e4ef", "#9ec5f4", "#5c93d6", "#2c5f9e", "#163a63"]
DIF_LV = [-1e9, -60, -20, -5, 5, 20, 60, 1e9]; DIF_LAB = ["ours 60+ faster", "20–60 faster", "5–20 faster", "±5",
                                                         "5–20 slower", "20–60 slower", "ours 60+ slower"]
DIF_COL = ["#1d5f8a", "#5c93d6", "#b7d3ee", "#eeede8", "#f2c38f", "#e0701f", "#9a3412"]
ADD_LV = [0, 5, 15, 30, 60, 1e9]; ADD_COL = ["#ebeae4", "#f6cf7a", "#ec9a3b", "#c95a1c", "#7d2a0c"]
ADD_LAB = ["<5 min", "5–15", "15–30", "30–60", "60+ min"]


def multi_source(G, n_nodes, src_nodes, src_cost):
    """Minutes from the nearest source via a virtual node linked to each source node at src_cost."""
    N = n_nodes
    G = G.tocoo()
    rows = np.r_[G.row, np.full(len(src_nodes), N)]; cols = np.r_[G.col, src_nodes]
    data = np.r_[G.data, src_cost + 1e-6]
    H = sparse.coo_matrix((data, (rows, cols)), shape=(N + 1, N + 1)).tocsr()
    return dijkstra(H, directed=False, indices=N)[:N]


def classed(ax, a, ext, lv, col):
    cmap = ListedColormap(col); norm = BoundaryNorm(lv, cmap.N)
    return ax.imshow(a, extent=ext, cmap=cmap, norm=norm, interpolation="nearest")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--init"); a_ = ap.parse_args()
    PROC = os.path.join(ROOT, "data", "processed", "roads")
    f = (os.path.join(PROC, f"eth_roads_scored_{pd.Timestamp(a_.init):%Y%m%d}.parquet") if a_.init else
         sorted(glob.glob(os.path.join(PROC, "eth_roads_scored_*.parquet")))[-1])
    tag = os.path.basename(f).split("_")[-1].split(".")[0]

    # --- MAP surface: valid cells ------------------------------------------------------
    with rasterio.open(MAPF) as d:
        M = d.read(1).astype("float32"); tr = d.transform
    valid = M >= 0
    rr, cc = np.nonzero(valid)
    cx = tr.c + (cc + 0.5) * tr.a; cy = tr.f + (rr + 0.5) * tr.e
    x0, x1, y0, y1 = cx.min(), cx.max(), cy.min(), cy.max()
    print(f"MAP cells with values: {valid.sum():,}  ({x0:.2f}-{x1:.2f} E, {y0:.2f}-{y1:.2f} N)")
    with rasterio.open(POPF) as d:
        P = d.read(1).astype("float32"); ptr = d.transform
    pc = np.clip(((cx - ptr.c) / ptr.a).astype(int), 0, P.shape[1] - 1)
    pr = np.clip(((cy - ptr.f) / ptr.e).astype(int), 0, P.shape[0] - 1)
    pop = np.where(P[pr, pc] > 0, P[pr, pc], 0)

    # --- our network, normal + forecast --------------------------------------------------
    g = gpd.read_parquet(f)
    pad = 1.0                                                           # route over a margin around the clip
    g = g.cx[x0 - pad:x1 + pad, y0 - pad:y1 + pad].reset_index(drop=True)
    nodes, a, b, seg, km = ha.build_graph(g)
    v0 = ha.base_speed(g.highway, g.surface_class, g["rough"] if "rough" in g.columns else None); sf = g.speed_factor.values
    vs = np.where(seg >= 0, v0[np.maximum(seg, 0)], ha.WALK); fs = np.where(seg >= 0, sf[np.maximum(seg, 0)], 1.0)
    N = len(nodes)
    G0 = sparse.coo_matrix((km / vs * 60 + 1e-6, (a, b)), shape=(N, N)).tocsr(); G0.sum_duplicates()
    G1 = sparse.coo_matrix((km / (vs * fs) * 60 + 1e-6, (a, b)), shape=(N, N)).tocsr(); G1.sum_duplicates()
    tree = cKDTree(np.c_[nodes[:, 0] * ha.COSLAT, nodes[:, 1]])

    hf = pd.read_csv(os.path.join(ROOT, "data", "health_facilities", "ethiopian-health-facilities.csv"))
    hf = hf[(hf.Status == "Approved") & hf.Type.isin(ha.KEEP)]
    hf = hf[(hf.Longitude > x0 - pad) & (hf.Longitude < x1 + pad) & (hf.Latitude > y0 - pad) & (hf.Latitude < y1 + pad)]
    fn, fgap = ha.snap(tree, hf.Longitude.values, hf.Latitude.values)
    fconn = fgap / ha.WALK * 60
    T0 = multi_source(G0, N, fn, fconn); T1 = multi_source(G1, N, fn, fconn)

    cn, cgap = ha.snap(tree, cx, cy)
    ftree = cKDTree(np.c_[hf.Longitude.values * ha.COSLAT, hf.Latitude.values])
    dd, _ = ftree.query(np.c_[cx * ha.COSLAT, cy])
    walk_direct = dd * 111.32 / ha.WALK * 60
    o0 = np.minimum(walk_direct, cgap / ha.WALK * 60 + T0[cn])
    o1 = np.minimum(walk_direct, cgap / ha.WALK * 60 + T1[cn])
    m0 = M[rr, cc].astype(float)
    ratio = np.where(o0 > 0.5, o1 / np.maximum(o0, 0.5), 1.0)
    m1 = m0 * ratio
    print(f"{len(hf):,} facilities, graph {N:,} nodes")

    # --- summaries ---------------------------------------------------------------------------
    def wq(v, w, q):
        o = np.argsort(v); cw = np.cumsum(w[o]); return float(v[o][np.searchsorted(cw, q * cw[-1])])
    W = pop; tot = W.sum()
    rows = []
    for name, v in [("MAP 2020 (no rain)", m0), ("Ours, normal speeds", o0), ("Ours, forecast speeds", o1),
                    ("MAP + our forecast slowdown", m1)]:
        rows.append({"version": name, "pop_median_min": round(wq(v, W, 0.5), 1), "pop_p90_min": round(wq(v, W, 0.9), 1),
                     **{f"pop_share_over_{t}min": round(float(W[v > t].sum() / tot), 3) for t in (30, 60, 120)},
                     **{f"pop_over_{t}min": round(float(W[v > t].sum())) for t in (30, 60, 120)}})
    S = pd.DataFrame(rows); print(S.to_string(index=False))
    r_ = np.corrcoef(np.log1p(o0), np.log1p(m0))[0, 1]
    diff = o0 - m0
    print(f"cells: log-time correlation ours vs MAP {r_:.2f}; median ours-MAP {np.median(diff):+.1f} min; "
          f"pop-weighted median {wq(diff, W, 0.5):+.1f} min; population in clip {tot / 1e6:.2f} M")
    add_o = o1 - o0; add_m = m1 - m0
    print(f"pop-weighted mean added minutes: ours {np.average(add_o, weights=W):.1f}, MAP-based {np.average(add_m, weights=W):.1f}; "
          f"people with 30+ added: ours {W[add_o >= 30].sum() / 1e3:,.0f}k, MAP-based {W[add_m >= 30].sum() / 1e3:,.0f}k")

    # by zone (ADM2 names from the facility register are not polygons -> use ADM1 geoBoundaries regions)
    reg = gpd.GeoDataFrame({"region": wn.NAMES}, geometry=wn.GEOMS, crs="EPSG:4326")
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(cx, cy), crs="EPSG:4326")
    rg = gpd.sjoin(pts, reg, how="left", predicate="within").groupby(level=0).region.first().fillna("outside").values
    Z = pd.DataFrame({"region": rg, "pop": W, "map": m0, "ours": o0, "ours_fc": o1, "map_fc": m1})
    Z = (Z.groupby("region").apply(lambda d: pd.Series({
        "population": round(d["pop"].sum()), "cells": len(d),
        "median_map_min": round(wq(d["map"].values, d["pop"].values + 1e-9, .5), 1),
        "median_ours_min": round(wq(d["ours"].values, d["pop"].values + 1e-9, .5), 1),
        "pop_over_60_map": round(d["pop"][d["map"] > 60].sum()), "pop_over_60_ours": round(d["pop"][d["ours"] > 60].sum()),
        "pop_over_60_ours_fc": round(d["pop"][d["ours_fc"] > 60].sum()), "pop_over_60_map_fc": round(d["pop"][d["map_fc"] > 60].sum())}),
        include_groups=False).sort_values("population", ascending=False))
    Z.to_csv(os.path.join(wn.TB, f"eth_travel_surface_compare_{tag}.csv"))
    S.to_csv(os.path.join(wn.TB, f"eth_travel_surface_compare_summary_{tag}.csv"), index=False)
    print(Z.to_string())

    # --- map: 2 x 2 ---------------------------------------------------------------------------
    rmin, rmax, cmin, cmax = rr.min(), rr.max(), cc.min(), cc.max()
    ext = [tr.c + cmin * tr.a, tr.c + (cmax + 1) * tr.a, tr.f + (rmax + 1) * tr.e, tr.f + rmin * tr.e]
    def grid(v):
        A = np.full((rmax - rmin + 1, cmax - cmin + 1), np.nan, "float32"); A[rr - rmin, cc - cmin] = v; return A
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 13), gridspec_kw={"wspace": 0.04, "hspace": 0.14})
    panels = [(axs[0, 0], grid(m0), TT_LV, TT_COL, "MAP 2020 motorized travel time to nearest facility"),
              (axs[0, 1], grid(o0), TT_LV, TT_COL, "Our road model, normal speeds"),
              (axs[1, 0], grid(diff), DIF_LV, DIF_COL, "Difference: ours − MAP (normal conditions)"),
              (axs[1, 1], grid(add_m), ADD_LV, ADD_COL, "Forecast added time, applied to the MAP baseline")]
    for ax, A, lv, col, ttl in panels:
        classed(ax, A, ext, lv, col)
        for gm in wn.GEOMS:
            for poly in (gm.geoms if gm.geom_type == "MultiPolygon" else [gm]):
                xs, ys = poly.exterior.xy; ax.plot(xs, ys, color=INK2, lw=0.4)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal"); ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
        ax.set_title(ttl, loc="left", fontsize=10.5, color=INK, pad=4)
    from matplotlib.patches import Patch
    def leg(ax, col, lab, title):
        ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(col, lab)], loc="lower left", fontsize=8, title=title,
                  title_fontsize=8.5, frameon=True, facecolor=SURF, edgecolor="none", ncol=2)
    leg(axs[0, 0], TT_COL, TT_LAB, "minutes"); leg(axs[0, 1], TT_COL, TT_LAB, "minutes")
    leg(axs[1, 0], DIF_COL, DIF_LAB, "ours vs MAP"); leg(axs[1, 1], ADD_COL, ADD_LAB, "added by forecast")
    fig.suptitle("How a published travel-time surface compares with our road + slope model", x=0.125, ha="left",
                 fontsize=13.5, fontweight="bold", color=INK, y=0.965)
    fig.text(0.125, 0.937, f"North-west Ethiopia (extent of the supplied MAP clip) · time to the nearest health post, centre or hospital · "
             f"forecast: WeatherNext 3 run {pd.Timestamp(tag):%d %b %Y}", fontsize=9.5, color=INK2)
    fig.text(0.125, 0.085, "MAP: Malaria Atlas Project 2020 motorized travel time to healthcare (Weiss et al. 2020), ~1 km; it uses land-cover off-road "
             "speeds, OSM + Google roads and its own facility list.\nOurs: OSM roads (HeiGIT surfaces) with class speeds, 5 km/h walking off-road, approved facilities "
             "from the Ethiopian register. Bottom right: MAP times x our forecast/normal ratio for each cell.", fontsize=8, color=MUTED, va="top")
    wn.save(fig, os.path.join(wn.OUT, "ETH_travel_surface_compare_map.png"))

    # --- stats -----------------------------------------------------------------------------------
    fig, (s1, s2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.3})
    lo, hi = 1, 1000
    hb = s1.hexbin(np.clip(m0, lo, hi), np.clip(o0, lo, hi), C=W, reduce_C_function=np.sum, gridsize=45, xscale="log",
                   yscale="log", bins="log", cmap=ListedColormap(["#e6effa", "#b7d3ee", "#7fb0e6", "#3f84d1", "#245f9e", "#123a63"]),
                   mincnt=1, linewidths=0)
    s1.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls="--"); s1.text(700, 350, "1:1", color=INK2, fontsize=9, ha="right")
    s1.set_xlim(lo, hi); s1.set_ylim(lo, hi)
    s1.set_xlabel("MAP 2020 travel time, min (log)"); s1.set_ylabel("our road model, normal speeds, min (log)")
    wn.title(s1, "Cell by cell, normal conditions", f"~1 km cells shaded by population · log-time r = {r_:.2f}")
    vers = S.version.tolist(); xs_ = np.arange(3); wd = 0.2
    cols = ["#8e9590", "#5c93d6", "#c95a1c", "#e7a76a"]
    for i, (vname, c) in enumerate(zip(vers, cols)):
        vals = [S.loc[i, f"pop_share_over_{t}min"] * 100 for t in (30, 60, 120)]
        bars = s2.bar(xs_ + (i - 1.5) * wd, vals, width=wd * 0.92, color=c, label=vname)
        for x_, v_ in zip(xs_ + (i - 1.5) * wd, vals):
            s2.text(x_, v_ + 0.6, f"{v_:.0f}", ha="center", fontsize=8, color=INK2)
    s2.set_xticks(xs_, ["over 30 min", "over 1 hour", "over 2 hours"]); s2.set_ylabel("share of population, %")
    s2.grid(axis="x", visible=False); s2.legend(loc="upper right", frameon=False, fontsize=8.5)
    wn.title(s2, "People beyond a travel-time threshold", f"{tot / 1e6:.1f} M people in the clip (WorldPop 2020)")
    wn.foot(fig, "Ours = OSM road network (HeiGIT surfaces) with class speeds and 5 km/h walking to the road; MAP = Malaria Atlas Project 2020 motorized surface. "
                 "'MAP + our forecast slowdown' multiplies MAP times by our forecast/normal ratio per cell.")
    wn.save(fig, os.path.join(wn.OUT, "ETH_travel_surface_compare_stats.png"))
