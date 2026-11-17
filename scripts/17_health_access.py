"""
17_health_access.py -- which Ethiopian health facilities lose the most referral access when forecast rain slows the
roads around them.

Inputs
  data/health_facilities/ethiopian-health-facilities.csv     facility register (name, type, ownership, admin areas, lat/lon)
  data/processed/roads/eth_roads_scored_<init>.parquet       every vehicle road segment with its slowdown tier (14_)

Method
  * Road graph: every road vertex is a node (OSM ways that meet share a vertex), consecutive vertices are edges.
  * Normal speed by road class (km/h): motorway 80, trunk 70, primary 60, secondary 50, tertiary 40, unclassified /
    road 30, residential 20, service / track 15, living street 10; links 80% of their class; unpaved roads x 0.7.
    Forecast speed = normal x the segment's tier speed factor (Minor 0.9, Elevated 0.75, Severe 0.5).
  * Mapping gaps: dead-end vertices within 30 m of another road vertex are linked (at 5 km/h), so a way that stops
    just short of a junction does not force a long detour.
  * Facilities join the network at the nearest road vertex, covering the gap at 5 km/h (walking / carrying).
  * Each facility is assigned the hospital it reaches fastest by road at normal speeds (health posts also: the
    nearest health centre). Its travel time to that same facility is then measured under normal and forecast speeds,
    so added minutes = forecast - normal for one fixed trip (no switching to another hospital when roads slow).
    Facilities more than 10 km from any mapped road are kept but flagged.
  * Linked facilities: the health centres / posts for which a hospital is the nearest by road at normal speeds. This is
    a road-network grouping, not a population catchment or the official referral pathway. Hospitals are ranked by how
    many linked facilities take >= 30 minutes longer to reach them.
  * Local exposure: share of road length within 5 km that is Elevated or Severe, and the tier of the nearest road.
  Counted: approved Hospitals, Health Centers and Health Posts. Clinics and pharmacies are left out of the ranking.

Outputs (<init> = forecast run date)
  data/processed/tables/eth_health_access_<init>.csv              every counted facility
  data/processed/tables/eth_health_access_top_<init>.csv          most affected hospitals (by linked facilities), health centres, posts
  data/processed/tables/eth_health_access_woreda_<init>.csv       health posts with >= 30 added minutes, by woreda
  charts/ETH_health_access_map.png                                facilities by added minutes to the nearest hospital
  charts/ETH_health_access_top.png                                most affected health centres and hospitals

    python scripts\\17_health_access.py [--init 2026-11-17]
"""
import argparse, glob, importlib.util, os
import numpy as np, pandas as pd, geopandas as gpd, shapely
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("wn", os.path.join(ROOT, "scripts", "11_weathernext_forecast.py"))
wn = importlib.util.module_from_spec(spec); spec.loader.exec_module(wn)
INK, INK2, MUTED, SURF = wn.INK, wn.INK2, wn.MUTED, wn.SURF
TCOL = {"Normal": "#c9c8c2", "Minor": "#f2b33d", "Elevated": "#e0701f", "Severe": "#a8320f"}

SPEED = {"motorway": 80, "trunk": 70, "primary": 60, "secondary": 50, "tertiary": 40, "unclassified": 30, "road": 30,
         "residential": 20, "service": 15, "track": 15, "living_street": 10}
WALK, KEEP = 5.0, ["Hospital", "Health Center", "Health Post"]
COSLAT = np.cos(np.deg2rad(9.0))
# added-minutes classes for maps (sequential, light -> dark)
DLV = [0, 5, 15, 30, 60, 1e9]; DLAB = ["< 5 min", "5–15", "15–30", "30–60", "60+ min"]
DCOL = ["#d9d8d2", "#f6cf7a", "#ec9a3b", "#c95a1c", "#7d2a0c"]


def base_speed(hw, surf):
    h = hw.str.replace("_link", "", regex=False)
    v = h.map(SPEED).fillna(20).values * np.where(hw.str.endswith("_link"), 0.8, 1.0)
    return v * np.where(surf.values == "unpaved", 0.7, 1.0)


def build_graph(g):
    geo = g.geometry.values
    xy, idx = shapely.get_coordinates(geo, return_index=True)
    key = np.rint(xy[:, 0] * 1e6).astype(np.int64) * 20_000_000 + np.rint(xy[:, 1] * 1e6).astype(np.int64)
    uk, first, inv = np.unique(key, return_index=True, return_inverse=True)
    nodes = xy[first]
    same = idx[:-1] == idx[1:]
    a, b, seg = inv[:-1][same], inv[1:][same], idx[:-1][same]
    lon1, lat1, lon2, lat2 = map(np.deg2rad, (xy[:-1][same, 0], xy[:-1][same, 1], xy[1:][same, 0], xy[1:][same, 1]))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    km = 2 * 6371.0 * np.arcsin(np.sqrt(h))
    keep = a != b
    a, b, seg, km = a[keep], b[keep], seg[keep], km[keep]
    # bridge mapping gaps: dead-end vertices within 30 m of another road vertex get a short link (seg = -1)
    deg = np.bincount(np.concatenate([a, b]), minlength=len(nodes))
    ends = np.flatnonzero(deg == 1)
    tree = cKDTree(np.c_[nodes[:, 0] * COSLAT, nodes[:, 1]])
    pairs = tree.query_ball_point(np.c_[nodes[ends, 0] * COSLAT, nodes[ends, 1]], r=0.03 / 111.32)
    ba, bb = [], []
    for e, nb in zip(ends, pairs):
        for n in nb:
            if n != e: ba.append(e); bb.append(n)
    ba, bb = np.array(ba, int), np.array(bb, int)
    bk = np.hypot((nodes[ba, 0] - nodes[bb, 0]) * COSLAT, nodes[ba, 1] - nodes[bb, 1]) * 111.32
    print(f"bridged {len(np.unique(ba)):,} dead ends with {len(ba):,} links <= 30 m")
    return nodes, np.r_[a, ba], np.r_[b, bb], np.r_[seg, np.full(len(ba), -1)], np.r_[km, bk]


def own_target(G0, G1, fac_nodes, fac_conn, tgt_nodes, tgt_conn, tgt_ids):
    """Assign each facility the target (hospital / health centre) it reaches first at normal speeds, then time the
    trip to that same target under normal (G0) and forecast (G1) speeds. Minutes include both walking links.
    Forecast speeds are at least half of normal, so each forecast search can stop at twice the group's normal time."""
    un = np.unique(tgt_nodes)
    tc = pd.Series(tgt_conn).groupby(tgt_nodes).min().reindex(un).values         # shortest link among targets on a node
    tid = pd.Series(tgt_ids).groupby(tgt_nodes).first().reindex(un).values
    d0, _, src = dijkstra(G0, directed=False, indices=un, min_only=True, return_predecessors=True)
    s = src[fac_nodes]
    pos = np.where(s >= 0, np.searchsorted(un, np.maximum(s, 0)), -1)
    t0 = np.where(pos >= 0, d0[fac_nodes] + fac_conn + tc[np.maximum(pos, 0)], np.nan)
    t1 = np.full(len(fac_nodes), np.nan)
    order = np.argsort(pos, kind="stable"); ps = pos[order]
    starts = np.searchsorted(ps, np.arange(len(un))); ends = np.searchsorted(ps, np.arange(len(un)), side="right")
    for k in range(len(un)):
        grp = order[starts[k]:ends[k]]
        if not len(grp): continue
        lim = 2.0 * d0[fac_nodes[grp]].max() + 1.0
        d = dijkstra(G1, directed=False, indices=un[k], limit=lim)
        t1[grp] = d[fac_nodes[grp]] + fac_conn[grp] + tc[k]
    return t0, t1, np.where(pos >= 0, tid[np.maximum(pos, 0)], -1)


def snap(tree, lon, lat):
    d, i = tree.query(np.c_[lon * COSLAT, lat])
    return i, d * 111.32


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--init"); a_ = ap.parse_args()
    PROC = os.path.join(ROOT, "data", "processed", "roads")
    f = (os.path.join(PROC, f"eth_roads_scored_{pd.Timestamp(a_.init):%Y%m%d}.parquet") if a_.init else
         sorted(glob.glob(os.path.join(PROC, "eth_roads_scored_*.parquet")))[-1])
    tag = os.path.basename(f).split("_")[-1].split(".")[0]
    g = gpd.read_parquet(f)
    print(f"roads {len(g):,} segments, run {tag}")

    # --- facilities ----------------------------------------------------------------------
    hf = pd.read_csv(os.path.join(ROOT, "data", "health_facilities", "ethiopian-health-facilities.csv"))
    hf = hf[(hf.Status == "Approved") & hf.Type.isin(KEEP)].reset_index(drop=True)
    print(hf.Type.value_counts().to_dict())

    # --- graph ---------------------------------------------------------------------------
    nodes, a, b, seg, km = build_graph(g)
    v0 = base_speed(g.highway, g.surface_class)
    sf = g.speed_factor.values
    vs = np.where(seg >= 0, v0[np.maximum(seg, 0)], WALK); fs = np.where(seg >= 0, sf[np.maximum(seg, 0)], 1.0)
    w0 = km / vs * 60; w1 = km / (vs * fs) * 60
    print(f"graph {len(nodes):,} nodes, {len(a):,} edges")
    tree = cKDTree(np.c_[nodes[:, 0] * COSLAT, nodes[:, 1]])
    hf["node"], hf["road_gap_km"] = snap(tree, hf.Longitude.values, hf.Latitude.values)
    conn = hf.road_gap_km.values / WALK * 60

    N = len(nodes)
    G0 = sparse.coo_matrix((w0 + 1e-6, (a, b)), shape=(N, N)).tocsr(); G0.sum_duplicates()
    G1 = sparse.coo_matrix((w1 + 1e-6, (a, b)), shape=(N, N)).tocsr(); G1.sum_duplicates()
    for tgt, label in [("Hospital", "hosp"), ("Health Center", "hc")]:
        s = (hf.Type == tgt).values
        t0, t1, tid = own_target(G0, G1, hf.node.values, conn, hf.node.values[s], conn[s], hf.index.values[s])
        hf[f"to_{label}_min"], hf[f"to_{label}_fc_min"] = t0.round(1), t1.round(1)
        hf[f"{label}_idx"] = tid
        print(f"{tgt}: forecast times to each facility's own {tgt.lower()} done")
    hf["nearest_hospital"] = hf.Name.reindex(hf.hosp_idx.values).str.strip().values
    hf["nearest_health_centre"] = hf.Name.reindex(hf.hc_idx.values).str.strip().values
    for c in ["to_hosp", "to_hc"]:
        hf[f"{c}_added_min"] = (hf[f"{c}_fc_min"] - hf[f"{c}_min"]).round(1)
        hf.loc[~np.isfinite(hf[f"{c}_min"]), [f"{c}_min", f"{c}_fc_min", f"{c}_added_min"]] = np.nan
    hf.loc[hf.Type != "Health Post", ["to_hc_min", "to_hc_fc_min", "to_hc_added_min"]] = np.nan
    hf.loc[hf.Type == "Hospital", ["to_hosp_min", "to_hosp_fc_min", "to_hosp_added_min"]] = 0.0
    hf["off_network"] = hf.road_gap_km > 10

    # --- local exposure within 5 km ---------------------------------------------------------
    mid = shapely.line_interpolate_point(g.geometry.values, 0.5, normalized=True)
    mt = cKDTree(np.c_[shapely.get_x(mid) * COSLAT, shapely.get_y(mid)])
    lenkm = (g.osm_length.values / 1000)
    bad = g.tier.isin(["Elevated", "Severe"]).values
    nb = mt.query_ball_point(np.c_[hf.Longitude.values * COSLAT, hf.Latitude.values], r=5 / 111.32)
    tot = np.array([lenkm[i].sum() for i in nb]); badk = np.array([lenkm[i][bad[i]].sum() for i in nb])
    hf["roads_5km_km"] = tot.round(1)
    hf["share_elev_severe_5km"] = np.where(tot > 0, badk / np.maximum(tot, 1e-9), np.nan).round(2)
    _, ns = mt.query(np.c_[hf.Longitude.values * COSLAT, hf.Latitude.values])
    hf["nearest_road_tier"] = g.tier.values[ns]
    hf["slip_within_5km"] = [bool(g.slide_exposure.values[i].any()) for i in nb]

    cols = ["Id", "Name", "Type", "Ownership", "admin1Name", "admin2Name", "admin3Name", "Latitude", "Longitude",
            "road_gap_km", "off_network", "to_hosp_min", "to_hosp_fc_min", "to_hosp_added_min", "to_hc_min", "to_hc_fc_min",
            "to_hc_added_min", "nearest_hospital", "nearest_health_centre", "hosp_idx", "nearest_road_tier", "share_elev_severe_5km", "slip_within_5km", "roads_5km_km"]
    hf["road_gap_km"] = hf.road_gap_km.round(2)
    res = hf[cols].copy()
    res.to_csv(os.path.join(wn.TB, f"eth_health_access_{tag}.csv"), index=False)

    # --- rankings ----------------------------------------------------------------------------
    on = ~res.off_network
    top_hc = res[on & (res.Type == "Health Center")].nlargest(25, "to_hosp_added_min")
    top_hp = res[on & (res.Type == "Health Post")].nlargest(25, "to_hc_added_min")
    sub_ = res[on & res.Type.isin(["Health Center", "Health Post"])]
    cat = (sub_[sub_.hosp_idx >= 0].assign(hit=sub_.to_hosp_added_min >= 30).groupby("hosp_idx")
           .agg(linked_facilities=("Id", "size"), linked_30min_plus=("hit", "sum"),
                linked_median_added_min=("to_hosp_added_min", "median")))
    H_ = res[res.Type == "Hospital"].join(cat, how="left").fillna({"linked_facilities": 0, "linked_30min_plus": 0})
    res = res.join(cat)                                                        # hospital rows carry linked-facility stats
    res.drop(columns=["hosp_idx"]).to_csv(os.path.join(wn.TB, f"eth_health_access_{tag}.csv"), index=False)
    top_h = H_.sort_values(["linked_30min_plus", "linked_median_added_min"], ascending=False).head(25)
    top = pd.concat([top_hc.assign(rank_on="added min to its nearest hospital"),
                     top_hp.assign(rank_on="added min to its nearest health centre"),
                     top_h.assign(rank_on="linked facilities taking >= 30 min longer to reach it")]).drop(columns=["hosp_idx"])
    top.to_csv(os.path.join(wn.TB, f"eth_health_access_top_{tag}.csv"), index=False)
    hp = res[on & (res.Type == "Health Post")]
    wor = (hp.assign(hit=hp.to_hc_added_min >= 30).groupby(["admin1Name", "admin2Name", "admin3Name"])
           .agg(health_posts=("Id", "size"), posts_30min_plus=("hit", "sum"), median_added_min=("to_hc_added_min", "median"))
           .query("posts_30min_plus > 0").sort_values("posts_30min_plus", ascending=False))
    wor.to_csv(os.path.join(wn.TB, f"eth_health_access_woreda_{tag}.csv"))

    def summ(df, c):
        x = df[c].dropna()
        return {k: int((x >= v).sum()) for k, v in [(">=15", 15), (">=30", 30), (">=60", 60)]} | {"n": len(x), "median_added": round(float(x.median()), 1)}
    print("health centres -> hospital", summ(res[on & (res.Type == "Health Center")], "to_hosp_added_min"))
    print("health posts -> hospital", summ(res[on & (res.Type == "Health Post")], "to_hosp_added_min"))
    print("health posts -> health centre", summ(res[on & (res.Type == "Health Post")], "to_hc_added_min"))
    print("hospitals with >=50% Elevated/Severe roads within 5 km:", int((res[res.Type == "Hospital"].share_elev_severe_5km >= 0.5).sum()))
    print("off network (>10 km from a road):", res.off_network.groupby(res.Type).sum().to_dict())
    print(top_hc[["Name", "admin2Name", "to_hosp_min", "to_hosp_fc_min", "to_hosp_added_min"]].head(10).to_string())
    print(wor.head(10).to_string())

    # --- map --------------------------------------------------------------------------------
    cmap = ListedColormap(DCOL); norm = BoundaryNorm(DLV, cmap.N)
    pts = res[on & res.Type.isin(["Health Center", "Health Post"])].copy()
    pts["added"] = np.where(pts.Type == "Health Post", pts.to_hc_added_min, pts.to_hosp_added_min)
    pts = pts.dropna(subset=["added"]).sort_values("added")
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.6), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.06})
    sev = g[g.tier.isin(["Elevated", "Severe"])]
    coords, idx = shapely.get_coordinates(sev.geometry.values, return_index=True)
    segs = np.split(coords, np.flatnonzero(np.diff(idx)) + 1)
    for k, (ax, (sub, ttl)) in enumerate(zip(axs, [(pts[pts.Type == "Health Center"], "Health centres: added minutes to the nearest hospital"),
                                                   (pts[pts.Type == "Health Post"], "Health posts: added minutes to the nearest health centre")])):
        ax.add_collection(LineCollection(segs, colors="#e7c3ad", linewidths=0.25, zorder=1))
        wn.draw_outline(ax); wn.mapax(ax)
        ax.scatter(sub.Longitude, sub.Latitude, c=sub.added, cmap=cmap, norm=norm, s=5 if k else 9, linewidths=0, zorder=3)
        big = sub[sub.added >= 30]
        ax.scatter(big.Longitude, big.Latitude, c=big.added, cmap=cmap, norm=norm, s=16 if k else 26,
                   edgecolors=SURF, linewidths=0.4, zorder=4)
        ax.set_title(ttl, loc="left", fontsize=11, color=INK, pad=4)
        n30 = int((sub.added >= 30).sum())
        ax.text(0.02, 0.02, f"{n30:,} of {len(sub):,} at 30+ added minutes", transform=ax.transAxes, fontsize=9, color=INK2)
    hosp = res[res.Type == "Hospital"]
    axs[0].scatter(hosp.Longitude, hosp.Latitude, marker="P", s=22, color=INK, edgecolors=SURF, linewidths=0.4, zorder=5)
    hd = [Line2D([], [], marker="o", ls="", color=c, markersize=7, label=l) for c, l in zip(DCOL, DLAB)]
    hd += [Line2D([], [], marker="P", ls="", color=INK, markersize=7, label="hospital"),
           Line2D([], [], color="#e7c3ad", lw=2, label="Elevated / Severe road")]
    fig.legend(handles=hd, loc="lower center", ncol=7, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Which health facilities lose the most referral access under the forecast rain", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK, y=1.0)
    fig.text(0.01, 0.955, f"Extra travel time when roads slow to their forecast tier speeds · WeatherNext 2 run {pd.Timestamp(tag):%d %b %Y}, 15 days",
             fontsize=9.5, color=INK2)
    fig.text(0.01, -0.03, "Shortest road travel time, normal vs forecast speeds. Normal speed by road class (trunk 70 … track 15 km/h, unpaved x 0.7); "
                          "forecast = normal x tier factor (Minor 0.9, Elevated 0.75, Severe 0.5).\nFacilities join the road network at the nearest road "
                          "vertex at 5 km/h. Approved facilities from the Ethiopian health facility register; facilities > 10 km from a mapped road left out. "
                          "Speeds are illustrative, not calibrated.", fontsize=8, color=MUTED, va="top")
    wn.save(fig, os.path.join(wn.OUT, "ETH_health_access_map.png"))

    # --- top health centres / hospitals ---------------------------------------------------------
    t2 = top_hc.head(20).iloc[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 7.4), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.72})
    y = np.arange(len(t2))
    a1.hlines(y, t2.to_hosp_min / 60, t2.to_hosp_fc_min / 60, color=AXC if (AXC := "#c3c2b7") else None, lw=2, zorder=1)
    a1.scatter(t2.to_hosp_min / 60, y, color="#8e9590", s=36, zorder=2, label="normal")
    a1.scatter(t2.to_hosp_fc_min / 60, y, color=TCOL["Severe"], s=36, zorder=3, label="forecast")
    for yi, (_, r) in zip(y, t2.iterrows()):
        a1.text(r.to_hosp_fc_min / 60, yi, f"  +{r.to_hosp_added_min:.0f} min", va="center", fontsize=8.5, color=INK2)
    a1.set_yticks(y, [f"{n[:26]} · {w}" for n, w in zip(t2.Name.str.strip(), t2.admin2Name)], fontsize=8.5)
    a1.set_xlabel("hours to the nearest hospital"); a1.grid(axis="y", visible=False)
    a1.set_xlim(0, t2.to_hosp_fc_min.max() / 60 * 1.22)
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, frameon=False, fontsize=9)
    wn.title(a1, "Health centres: biggest added time to their hospital", "Top 20, road travel to the nearest hospital under normal vs forecast speeds")
    t3 = top_h.head(15).iloc[::-1]
    yy = np.arange(len(t3))
    a2.barh(yy, t3.linked_facilities, color="#e1e0d9", height=0.62, label="linked facilities")
    a2.barh(yy, t3.linked_30min_plus, color=TCOL["Elevated"], height=0.62, label="take 30+ min longer to reach it")
    for yi, (_, r) in zip(yy, t3.iterrows()):
        a2.text(r.linked_facilities, yi, f"  {int(r.linked_30min_plus)} of {int(r.linked_facilities)}"
                + ("  ▲" if r.slip_within_5km else ""), va="center", fontsize=8.5, color=INK2)
    a2.set_yticks(yy, [f"{n.strip()[:26]} · {w}" for n, w in zip(t3.Name, t3.admin2Name)], fontsize=8.5)
    a2.set_xlim(0, t3.linked_facilities.max() * 1.3); a2.set_xlabel("health centres + posts for which this is the nearest hospital")
    a2.grid(axis="y", visible=False)
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, frameon=False, fontsize=9)
    wn.title(a2, "Hospitals that become hardest to reach", "Top 15 by linked facilities with 30+ added minutes")
    wn.foot(fig, "Linked facilities = health centres and posts for which this is the nearest hospital by road at normal speeds (a road-network grouping, "
                 "not a population catchment). Times are to that same hospital under both speeds. Ranks only facilities within 10 km of a mapped road. "
                 "▲ = a road within 5 km of the hospital is flagged for slip / rock-fall exposure.\n"
                 f"Full lists: data/processed/tables/eth_health_access_top_{tag}.csv")
    wn.save(fig, os.path.join(wn.OUT, "ETH_health_access_top.png"))
