"""
20_friction_access.py -- rerun the health-facility referral analysis on the Malaria Atlas Project motorized friction
surface and compare it with the road + slope model (17_health_access.py).

Inputs
  data/raw/friction/map_friction_motorized_2019_v5_1_eth.tif   MAP 2019 v5.1 motorized friction, minutes per metre,
      ~1 km (Weiss et al. 2020), from Earth Engine projects/malariaatlasproject/assets/accessibility/friction_surface/2019_v5_1
  data/processed/roads/eth_roads_scored_<init>.parquet          our roads with slowdown tiers (14_)
  data/processed/tables/eth_health_access_<init>.csv            road-model results per facility (17_)
  data/travel-surfaces/202001_Global_Motorized_Travel_Time_to_Healthcare_ETH.tiff   MAP's published travel time
      (north-west clip only), used to check the friction routing

Method
  * Graph: every friction cell inside Ethiopia (+ ~5 km) is a node, linked to its 8 neighbours; a link costs
    distance (m) x mean friction of the two cells (min/m). Off-road travel, road speeds and land cover all come from MAP.
  * Forecast: a cell crossed by one of our scored roads is slowed by that road's tier factor (fastest road in the cell
    decides); friction / factor. Cells without a road keep MAP's friction (rain effects off-road are not modelled).
  * Facilities sit in their cell. As in 17_: each facility is assigned the hospital it reaches first at normal
    friction (health posts also their nearest health centre), and the forecast time is measured to that same facility.
  * Check: time from each cell to the nearest facility in our list, on MAP friction, against MAP's published raster
    in the north-west clip -> isolates the effect of the facility list (same friction, different facilities).

Outputs
  data/processed/tables/eth_health_access_friction_<init>.csv   per facility: road-model and friction-model times
  data/processed/tables/eth_health_access_friction_summary_<init>.csv
  charts/ETH_friction_vs_road_access.png    normal times and forecast added minutes, road vs friction, counts
  charts/ETH_friction_vs_road_map.png       health centres' added minutes under each model, side by side
  charts/ETH_friction_vs_road_hospitals.png hospital rankings under each model

    python scripts\\20_friction_access.py [--init 2026-11-17]
"""
import argparse, glob, importlib.util, os
import numpy as np, pandas as pd, geopandas as gpd, rasterio, shapely
from rasterio.features import rasterize
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("ha", os.path.join(ROOT, "scripts", "17_health_access.py"))
ha = importlib.util.module_from_spec(spec); spec.loader.exec_module(ha)
wn = ha.wn
INK, INK2, MUTED, SURF = wn.INK, wn.INK2, wn.MUTED, wn.SURF
FRIC = os.path.join(ROOT, "data", "raw", "friction", "map_friction_motorized_2019_v5_1_eth.tif")
MAPTT = os.path.join(ROOT, "data", "travel-surfaces", "202001_Global_Motorized_Travel_Time_to_Healthcare_ETH.tiff")
C_ROAD, C_FRIC = "#2a78d6", "#c95a1c"


def cell_graph(F, ok, tr):
    """8-neighbour graph over valid cells; edge minutes = metres x mean friction."""
    H, W = F.shape
    nid = np.full((H, W), -1, np.int64); nid[ok] = np.arange(ok.sum())
    lat = tr.f + (np.arange(H) + 0.5) * tr.e
    dx = tr.a * 111320 * np.cos(np.deg2rad(lat)); dy = abs(tr.e) * 110574
    A, B, Wt = [], [], []
    for di, dj in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        ra, rb = slice(0, H - di), slice(di, H)
        ca, cb = (slice(0, W - dj), slice(dj, W)) if dj >= 0 else (slice(-dj, W), slice(0, W + dj))
        a, b, fa, fb = nid[ra, ca], nid[rb, cb], F[ra, ca], F[rb, cb]
        m = (a >= 0) & (b >= 0)
        rows = np.nonzero(m)[0]                                   # row index of the 'a' cell (ra starts at 0)
        if di and dj: d = np.hypot(dx[rows], dy)
        elif dj: d = dx[rows]
        else: d = np.full(len(rows), dy)
        A.append(a[m]); B.append(b[m]); Wt.append(d * 0.5 * (fa[m] + fb[m]))
    return nid, np.concatenate(A), np.concatenate(B), np.concatenate(Wt)


def to_rc(tr, x, y, shape):
    c = np.clip(((x - tr.c) / tr.a).astype(int), 0, shape[1] - 1)
    r = np.clip(((y - tr.f) / tr.e).astype(int), 0, shape[0] - 1)
    return r, c


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--init"); a_ = ap.parse_args()
    PROC = os.path.join(ROOT, "data", "processed", "roads")
    f = (os.path.join(PROC, f"eth_roads_scored_{pd.Timestamp(a_.init):%Y%m%d}.parquet") if a_.init else
         sorted(glob.glob(os.path.join(PROC, "eth_roads_scored_*.parquet")))[-1])
    tag = os.path.basename(f).split("_")[-1].split(".")[0]

    with rasterio.open(FRIC) as d:
        F = d.read(1).astype("float32"); tr = d.transform
    eth = shapely.union_all([gm.buffer(0) for gm in wn.GEOMS]).buffer(0.05)
    inside = rasterize([(eth, 1)], out_shape=F.shape, transform=tr, fill=0) == 1
    ok = inside & np.isfinite(F) & (F > 0)
    print(f"friction cells used: {ok.sum():,}")

    # --- forecast friction: slow cells crossed by our scored roads -------------------------------
    g = gpd.read_parquet(f)
    v0 = ha.base_speed(g.highway, g.surface_class)
    dens = shapely.segmentize(g.geometry.values, 0.002)
    xy, idx = shapely.get_coordinates(dens, return_index=True)
    r, c = to_rc(tr, xy[:, 0], xy[:, 1], F.shape)
    order = np.argsort(v0[idx], kind="stable")                    # fastest road written last -> it decides
    SF = np.ones(F.shape, "float32")
    SF[r[order], c[order]] = g.speed_factor.values[idx][order]
    F1 = F / SF
    print(f"cells with a scored road: {(np.isfinite(F) & (SF < 1)).sum():,} slowed by the forecast")

    nid, A, B, W0 = cell_graph(F, ok, tr)
    _, _, _, W1 = cell_graph(F1, ok, tr)
    N = int(ok.sum())
    G0 = sparse.coo_matrix((W0 + 1e-6, (A, B)), shape=(N, N)).tocsr()
    G1 = sparse.coo_matrix((W1 + 1e-6, (A, B)), shape=(N, N)).tocsr()
    print(f"graph {N:,} nodes, {len(A):,} edges")

    # --- facilities ------------------------------------------------------------------------------
    hf = pd.read_csv(os.path.join(ROOT, "data", "health_facilities", "ethiopian-health-facilities.csv"))
    hf = hf[(hf.Status == "Approved") & hf.Type.isin(ha.KEEP)].reset_index(drop=True)
    fr, fc = to_rc(tr, hf.Longitude.values, hf.Latitude.values, F.shape)
    node = nid[fr, fc]
    keep = node >= 0
    hf, node = hf[keep].reset_index(drop=True), node[keep]
    zero = np.zeros(len(hf))
    res = hf[["Id", "Name", "Type", "admin1Name", "admin2Name", "admin3Name", "Latitude", "Longitude"]].copy()
    for tgt, label in [("Hospital", "hosp"), ("Health Center", "hc")]:
        s = (hf.Type == tgt).values
        t0, t1, tid = ha.own_target(G0, G1, node, zero, node[s], zero[s], hf.index.values[s])
        res[f"fr_to_{label}_min"], res[f"fr_to_{label}_fc_min"], res[f"fr_{label}_idx"] = t0.round(1), t1.round(1), tid
        print(f"{tgt}: done")
    res["fr_nearest_hospital"] = res.Name.reindex(res.fr_hosp_idx.values).str.strip().values
    res["fr_nearest_hospital_id"] = res.Id.reindex(res.fr_hosp_idx.values).values
    for l in ["hosp", "hc"]:
        res[f"fr_to_{l}_added_min"] = (res[f"fr_to_{l}_fc_min"] - res[f"fr_to_{l}_min"]).round(1)
    res.loc[res.Type != "Health Post", ["fr_to_hc_min", "fr_to_hc_fc_min", "fr_to_hc_added_min"]] = np.nan

    road = pd.read_csv(os.path.join(wn.TB, f"eth_health_access_{tag}.csv"))
    road = road[["Id", "off_network", "to_hosp_min", "to_hosp_fc_min", "to_hosp_added_min", "to_hc_min", "to_hc_fc_min",
                 "to_hc_added_min", "nearest_hospital"]]
    X = res.merge(road, on="Id", how="left")
    X.drop(columns=["fr_hosp_idx", "fr_hc_idx"]).to_csv(os.path.join(wn.TB, f"eth_health_access_friction_{tag}.csv"), index=False)

    # --- comparisons -------------------------------------------------------------------------------
    hc = X[(X.Type == "Health Center") & ~X.off_network.fillna(False).astype(bool)].dropna(subset=["to_hosp_min", "fr_to_hosp_min"])
    hp = X[(X.Type == "Health Post") & ~X.off_network.fillna(False).astype(bool)].dropna(subset=["to_hc_min", "fr_to_hc_min"])
    rl = lambda a, b: np.corrcoef(np.log1p(a), np.log1p(b))[0, 1]
    sp = lambda a, b: pd.Series(a).rank().corr(pd.Series(b).rank())
    top_r = set(hc.nlargest(25, "to_hosp_added_min").Id); top_f = set(hc.nlargest(25, "fr_to_hosp_added_min").Id)
    same_h = (hc.nearest_hospital.str.strip() == hc.fr_nearest_hospital).mean()

    def linked(df, idx_col, add_col):
        d = df.assign(hit=df[add_col] >= 30).groupby(idx_col).agg(n=("Id", "size"), hit=("hit", "sum"))
        return d
    both = X[X.Type.isin(["Health Center", "Health Post"]) & ~X.off_network.fillna(False).astype(bool)]
    road_link = both.groupby("nearest_hospital").agg(n=("Id", "size"), hit=("to_hosp_added_min", lambda s: (s >= 30).sum()))
    fr_link = both.groupby("fr_nearest_hospital").agg(n=("Id", "size"), hit=("fr_to_hosp_added_min", lambda s: (s >= 30).sum()))
    L = road_link.join(fr_link, lsuffix="_road", rsuffix="_fric", how="outer").fillna(0)
    top10_r = set(L.nlargest(10, "hit_road").index); top10_f = set(L.nlargest(10, "hit_fric").index)

    S = pd.DataFrame([
        ("health centres compared", len(hc), len(hc)),
        ("median normal minutes, health centre to its hospital", round(hc.to_hosp_min.median(), 1), round(hc.fr_to_hosp_min.median(), 1)),
        ("health centres over 2 h to their hospital (normal)", int((hc.to_hosp_min > 120).sum()), int((hc.fr_to_hosp_min > 120).sum())),
        ("health centres 30+ min longer under the forecast", int((hc.to_hosp_added_min >= 30).sum()), int((hc.fr_to_hosp_added_min >= 30).sum())),
        ("health centres 60+ min longer under the forecast", int((hc.to_hosp_added_min >= 60).sum()), int((hc.fr_to_hosp_added_min >= 60).sum())),
        ("health posts 30+ min longer to their health centre", int((hp.to_hc_added_min >= 30).sum()), int((hp.fr_to_hc_added_min >= 30).sum())),
        ("median added minutes, health centres with any added time",
         round(hc.to_hosp_added_min[hc.to_hosp_added_min > 0.5].median(), 1), round(hc.fr_to_hosp_added_min[hc.fr_to_hosp_added_min > 0.5].median(), 1)),
    ], columns=["measure", "road_model", "friction_model"])
    extra = {"log-time correlation, health centre normal times": round(rl(hc.to_hosp_min, hc.fr_to_hosp_min), 2),
             "rank correlation, health centre added minutes": round(sp(hc.to_hosp_added_min.values, hc.fr_to_hosp_added_min.values), 2),
             "top-25 health centres in common": len(top_r & top_f),
             "health centres assigned the same nearest hospital": f"{same_h:.0%}",
             "rank correlation, hospitals by linked facilities 30+": round(sp(L.hit_road.values, L.hit_fric.values), 2),
             "top-10 hospitals in common": len(top10_r & top10_f)}
    S = pd.concat([S, pd.DataFrame([(k, v, "") for k, v in extra.items()], columns=S.columns)])
    S.to_csv(os.path.join(wn.TB, f"eth_health_access_friction_summary_{tag}.csv"), index=False)
    print(S.to_string(index=False))

    # facility-list check in the MAP clip: our facilities on MAP friction vs MAP's own published raster
    with rasterio.open(MAPTT) as d:
        M = d.read(1).astype("float32"); mtr = d.transform
    src = np.unique(node)
    G0v = G0.tocoo(); NN = N
    Hh = sparse.coo_matrix((np.r_[G0v.data, np.full(len(src), 1e-6)], (np.r_[G0v.row, np.full(len(src), NN)], np.r_[G0v.col, src])),
                           shape=(NN + 1, NN + 1)).tocsr()
    near = dijkstra(Hh, directed=False, indices=NN)[:NN]
    mr, mc = np.nonzero(M >= 0)
    mx = mtr.c + (mc + 0.5) * mtr.a; my = mtr.f + (mr + 0.5) * mtr.e
    rr_, cc_ = to_rc(tr, mx, my, F.shape); nn = nid[rr_, cc_]; okm = nn >= 0
    ours_on_fric = near[nn[okm]]; mapv = M[mr[okm], mc[okm]]
    print(f"facility-list check (NW clip): median our-list-on-MAP-friction {np.median(ours_on_fric):.1f} min vs MAP published "
          f"{np.median(mapv):.1f} min; share of cells where ours is faster: {(ours_on_fric < mapv - 1).mean():.0%}; "
          f"log r = {rl(ours_on_fric, mapv):.2f}")
    open(os.path.join(wn.TB, f"eth_health_access_friction_summary_{tag}.csv"), "a", encoding="utf-8").write(
        f"facility-list check NW clip: median minutes our facilities on MAP friction vs MAP published,{np.median(ours_on_fric):.1f},{np.median(mapv):.1f}\n")

    # --- chart 1: road vs friction --------------------------------------------------------------------
    fig, (c1, c2, c3) = plt.subplots(1, 3, figsize=(17, 5.6), gridspec_kw={"width_ratios": [1, 1, 1.1], "wspace": 0.32})
    lo, hi = 3, 1500
    c1.scatter(np.clip(hc.to_hosp_min, lo, hi), np.clip(hc.fr_to_hosp_min, lo, hi), s=5, alpha=0.35, color=INK2, linewidths=0)
    c1.plot([lo, hi], [lo, hi], color=INK2, ls="--", lw=1); c1.set_xscale("log"); c1.set_yscale("log")
    c1.set_xlim(lo, hi); c1.set_ylim(lo, hi)
    c1.set_xlabel("road model, minutes (log)"); c1.set_ylabel("MAP friction model, minutes (log)")
    wn.title(c1, "Health centre to its hospital, normal", f"{len(hc):,} health centres · log r = {extra['log-time correlation, health centre normal times']}")
    m_ = 400
    c2.scatter(np.clip(hc.to_hosp_added_min, 0, m_), np.clip(hc.fr_to_hosp_added_min, 0, m_), s=6, alpha=0.4, color=INK2, linewidths=0)
    c2.plot([0, m_], [0, m_], color=INK2, ls="--", lw=1); c2.set_xlim(-5, m_); c2.set_ylim(-5, m_)
    c2.axvline(30, color="#c3c2b7", lw=0.8); c2.axhline(30, color="#c3c2b7", lw=0.8)
    c2.set_xlabel("road model, added minutes"); c2.set_ylabel("MAP friction model, added minutes")
    wn.title(c2, "Added by the forecast", f"rank r = {extra['rank correlation, health centre added minutes']} · "
             f"{extra['top-25 health centres in common']} of the top 25 in common")
    labs = ["Centre → hospital\n30+ min longer", "Centre → hospital\n60+ min longer", "Post → centre\n30+ min longer"]
    rv = [S.iloc[3, 1], S.iloc[4, 1], S.iloc[5, 1]]; fv = [S.iloc[3, 2], S.iloc[4, 2], S.iloc[5, 2]]
    xs = np.arange(3)
    c3.bar(xs - 0.2, rv, 0.38, color=C_ROAD, label="road + slope model"); c3.bar(xs + 0.2, fv, 0.38, color=C_FRIC, label="MAP friction model")
    for x_, v_ in zip(np.r_[xs - 0.2, xs + 0.2], rv + fv):
        c3.text(x_, v_, f"{int(v_):,}", ha="center", va="bottom", fontsize=8.5, color=INK2)
    c3.axvline(1.5, color=MUTED, lw=0.8, ls=":")                    # different trip either side
    c3.set_xticks(xs, labs, fontsize=8.5); c3.grid(axis="x", visible=False); c3.legend(frameon=False, fontsize=9, loc="upper right")
    wn.title(c3, "Facilities losing access under the forecast", "count, on two separate trips")
    wn.foot(fig, "Road model: HeiGIT/OSM roads with class speeds, 5 km/h walking to the road. MAP friction model: Malaria Atlas Project 2019 motorized "
                 "friction (~1 km; roads, land cover, off-road speeds), cells crossed by a scored road slowed by its forecast tier.\n"
                 "Both: approved facilities from the Ethiopian register, each trip to the facility's own nearest hospital (health posts: health centre) "
                 f"at normal speeds; WeatherNext 3 run {pd.Timestamp(tag):%d %b %Y}.")
    wn.save(fig, os.path.join(wn.OUT, "ETH_friction_vs_road_access.png"))

    # --- chart 2: maps -----------------------------------------------------------------------------------
    DLV = [0, 5, 15, 30, 60, 1e9]; DCOL = ["#d9d8d2", "#f6cf7a", "#ec9a3b", "#c95a1c", "#7d2a0c"]
    cmap = ListedColormap(DCOL); norm = BoundaryNorm(DLV, cmap.N)
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.6), gridspec_kw={"wspace": 0.04})
    for ax, col, ttl in [(axs[0], "to_hosp_added_min", "Road + slope model"), (axs[1], "fr_to_hosp_added_min", "MAP friction model")]:
        d_ = hc.sort_values(col)
        wn.draw_outline(ax); wn.mapax(ax)
        ax.scatter(d_.Longitude, d_.Latitude, c=d_[col].clip(lower=0), cmap=cmap, norm=norm, s=10, linewidths=0, zorder=3)
        big = d_[d_[col] >= 30]
        ax.scatter(big.Longitude, big.Latitude, c=big[col], cmap=cmap, norm=norm, s=26, edgecolors=SURF, linewidths=0.4, zorder=4)
        ax.set_title(ttl, loc="left", fontsize=11, color=INK, pad=4)
        ax.text(0.02, 0.02, f"{int((d_[col] >= 30).sum()):,} health centres at 30+ added minutes", transform=ax.transAxes, fontsize=9, color=INK2)
    hd = [Line2D([], [], marker="o", ls="", color=c_, markersize=7, label=l_) for c_, l_ in
          zip(DCOL, ["< 5 min", "5–15", "15–30", "30–60", "60+ min"])]
    fig.legend(handles=hd, loc="lower center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Health centres: added minutes to their hospital under the forecast, two travel models", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK, y=1.0)
    fig.text(0.01, 0.935, "Same facilities, same hospital assignment rule and same road slowdowns; only the base travel surface differs",
             fontsize=9.5, color=INK2)
    wn.save(fig, os.path.join(wn.OUT, "ETH_friction_vs_road_map.png"))

    # --- chart 3: hospitals ------------------------------------------------------------------------------
    T = L.assign(name=L.index).sort_values("hit_road", ascending=False).head(15).iloc[::-1]
    T2 = L.assign(name=L.index).sort_values("hit_fric", ascending=False).head(15).iloc[::-1]
    fig, (h1, h2) = plt.subplots(1, 2, figsize=(15, 6.8), gridspec_kw={"wspace": 0.75})
    for ax, D, main, other, ttl in [(h1, T, "hit_road", "hit_fric", "Top 15 under the road model"),
                                    (h2, T2, "hit_fric", "hit_road", "Top 15 under the MAP friction model")]:
        y = np.arange(len(D))
        ax.barh(y + 0.19, D[main], 0.36, color=C_ROAD if main == "hit_road" else C_FRIC,
                label="road model" if main == "hit_road" else "friction model")
        ax.barh(y - 0.19, D[other], 0.36, color=C_FRIC if main == "hit_road" else C_ROAD,
                label="friction model" if main == "hit_road" else "road model")
        ax.set_yticks(y, [n[:34] for n in D.name], fontsize=8.5); ax.grid(axis="y", visible=False)
        ax.set_xlabel("linked health centres + posts taking 30+ min longer")
        ax.legend(frameon=False, fontsize=9, loc="lower right"); wn.title(ax, ttl, "hospitals that become hardest to reach")
    wn.foot(fig, f"Linked = facilities for which this is the nearest hospital at normal speeds under that model. "
                 f"Top-10 hospitals in common: {extra['top-10 hospitals in common']}; rank correlation {extra['rank correlation, hospitals by linked facilities 30+']}.")
    wn.save(fig, os.path.join(wn.OUT, "ETH_friction_vs_road_hospitals.png"))
