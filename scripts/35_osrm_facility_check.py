"""
35_osrm_facility_check.py -- every facility's normal-speed trips (17_) timed again on OSRM, to flag where the road model
disagrees with a standard router.

Two trips, kept apart as in 17_:
  hospital  every health centre and health post -> the hospital 17_ assigned it
  centre    every health post -> the health centre 17_ assigned it (matched by name, nearest of that name)
The trips are timed with OSRM's table service on the public demo server (router.project-osrm.org, car profile), batched
so one request times all the facilities going to one hospital (or a pack of health centres): about 530 requests at one
per second. Each end snaps to the nearest OSRM road. 17_'s walking legs (5 km/h to the road at both ends) are taken off
its time, so the comparison is road time against road time.

ratio = ours / OSRM. The road model runs about 1.3x OSRM everywhere (unpaved roads at 70% speed; OSRM drives untagged
roads at full speed), so a facility is flagged against that typical ratio, not against 1:
  slow_outlier  ratio > 2.5x the median ratio, and >= 30 min more: likely a mapping gap or detour in the road graph
  fast_outlier  ratio < 0.4x the median ratio, and >= 30 min less: OSRM detours (or our graph uses a link OSRM won't)
  no_route      OSRM found no route (disconnected road piece around one end)

Outputs (<init> = forecast run date)
  data/processed/tables/eth_osrm_facility_check_<init>.csv    one row per facility and trip
  data/processed/tables/eth_osrm_facility_check_<init>.json   summary by trip, type and region
  charts/ETH_osrm_facility_check.png                          map of ratios by trip, and the distribution

    python scripts\\35_osrm_facility_check.py [--init 2026-11-19]
"""
import argparse, glob, json, os, time
import numpy as np, pandas as pd, requests
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib import font_manager as _fm
import importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TB = os.path.join(ROOT, "data", "processed", "tables")
spec = importlib.util.spec_from_file_location("wn", os.path.join(ROOT, "scripts", "11_weathernext_forecast.py"))
wn = importlib.util.module_from_spec(spec); spec.loader.exec_module(wn)
for _f in glob.glob(os.path.join(ROOT, "assets", "fonts", "roboto", "*.ttf")): _fm.fontManager.addfont(_f)
plt.rcParams.update({"font.family": ["Roboto", "DejaVu Sans"]})
INK, INK2, MUTED = wn.INK, wn.INK2, wn.MUTED
URL = "https://router.project-osrm.org/table/v1/driving/"
WALK = 60 / 5
CHUNK = 200
COSLAT = np.cos(np.deg2rad(9.0))


def table(s, pts, src, dst):
    """pts: list of (lon, lat); src/dst: index lists. Returns (minutes, km) arrays [len(src), len(dst)]."""
    q = ";".join(f"{x:.6f},{y:.6f}" for x, y in pts)
    url = f"{URL}{q}?sources={';'.join(map(str, src))}&destinations={';'.join(map(str, dst))}&annotations=duration,distance"
    for i in range(8):
        try:
            r = s.get(url, headers={"User-Agent": "road-access-check/1.0"}, timeout=120)
        except requests.RequestException:
            time.sleep(10 * (i + 1)); continue
        if r.status_code in (429, 502, 503, 504): time.sleep(15 * (i + 1)); continue
        j = r.json()
        if j.get("code") != "Ok": raise RuntimeError(j.get("message", j.get("code")))
        d = np.array([[np.nan if v is None else v for v in row] for row in j["durations"]], float) / 60
        k = np.array([[np.nan if v is None else v for v in row] for row in j["distances"]], float) / 1000
        return d, k
    raise RuntimeError("server kept failing")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--init", default="2026-11-19"); a = ap.parse_args()
    tag = a.init.replace("-", "")
    d = pd.read_csv(os.path.join(TB, f"eth_health_access_{tag}.csv"))
    d = d[~d.off_network].copy()
    H = d[d.Type == "Hospital"].set_index("Id"); C = d[d.Type == "Health Center"]
    out = os.path.join(TB, f"eth_osrm_facility_check_{tag}.csv")
    done = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame(columns=["Id", "trip"])
    have = set(zip(done.Id, done.trip)); rows = done.to_dict("records")

    # --- jobs: (trip, [facility rows], [target rows]) ------------------------------------------
    jobs = []
    f = d[d.Type.isin(["Health Center", "Health Post"]) & d.nearest_hospital_id.notna() & d.to_hosp_min.notna()].copy()
    f["hid"] = f.nearest_hospital_id.astype(int); f = f[f.hid.isin(H.index)]
    for hid, g in f.groupby("hid"):
        g = g[np.array([(i, "hospital") not in have for i in g.Id], bool)]
        for k in range(0, len(g), CHUNK - 1):
            jobs.append(("hospital", g.iloc[k:k + CHUNK - 1], H.loc[[hid]].reset_index()))
    # health post -> its health centre, by name; where a name repeats, the nearest centre of that name
    p = d[(d.Type == "Health Post") & d.nearest_health_centre.notna() & d.to_hc_min.notna()].copy()
    cidx = {n: g for n, g in C.groupby("Name")}
    def pick(r):
        g = cidx.get(r.nearest_health_centre)
        if g is None: return -1
        dd = np.hypot((g.Longitude - r.Longitude) * COSLAT, g.Latitude - r.Latitude)
        return int(g.Id.values[np.argmin(dd.values)])
    p["cid"] = [pick(r) for r in p.itertuples()]
    p = p[(p.cid >= 0).values & np.array([(i, "centre") not in have for i in p.Id], bool)]
    Cx = C.set_index("Id")
    pack, n = [], 0
    for cid, g in p.groupby("cid"):
        if n + len(g) + 1 > CHUNK and pack:
            jobs.append(("centre", pd.concat([x for x, _ in pack]), Cx.loc[[c for _, c in pack]].reset_index())); pack, n = [], 0
        pack.append((g, cid)); n += len(g) + 1
    if pack: jobs.append(("centre", pd.concat([x for x, _ in pack]), Cx.loc[[c for _, c in pack]].reset_index()))
    print(f"{len(jobs)} requests to send ({sum(len(j[1]) for j in jobs):,} trips)", flush=True)

    s = requests.Session(); gap = dict(zip(d.Id, d.road_gap_km))
    for i, (trip, src, dst) in enumerate(jobs):
        pts = list(zip(src.Longitude, src.Latitude)) + list(zip(dst.Longitude, dst.Latitude))
        tmin, tkm = table(s, pts, list(range(len(src))), list(range(len(src), len(pts))))
        tgt_col = "hid" if trip == "hospital" else "cid"
        col = {t: j for j, t in enumerate(dst.Id)}
        for r_, (_, r) in enumerate(src.iterrows()):
            t = int(r[tgt_col]); j = col[t]
            ours_total = r.to_hosp_min if trip == "hospital" else r.to_hc_min
            rows.append({"Id": int(r.Id), "Name": r.Name, "Type": r.Type, "region": r.admin1Name, "zone": r.admin2Name,
                         "woreda": r.admin3Name, "Latitude": r.Latitude, "Longitude": r.Longitude, "trip": trip,
                         "target_id": t, "target": dst.Name.iloc[j],
                         "gap_km": r.road_gap_km, "target_gap_km": gap.get(t, np.nan), "ours_total_min": ours_total,
                         "ours_road_min": round(ours_total - WALK * (r.road_gap_km + gap.get(t, 0)), 1),
                         "osrm_min": None if np.isnan(tmin[r_, j]) else round(tmin[r_, j], 1),
                         "osrm_km": None if np.isnan(tkm[r_, j]) else round(tkm[r_, j], 2)})
        if (i + 1) % 25 == 0 or i == len(jobs) - 1:
            pd.DataFrame(rows).to_csv(out, index=False); print(f"  {i + 1}/{len(jobs)}", flush=True)
        time.sleep(1.0)

    R = pd.DataFrame(rows)
    # both ends join the road at (almost) the same point: no road leg worth comparing
    short = (R.ours_road_min < 1) & (R.osrm_min.fillna(0) < 1)
    R["ratio"] = (R.ours_road_min.clip(lower=0) / R.osrm_min.where(R.osrm_min > 0)).where(~short & (R.ours_road_min >= 1))
    flag = pd.Series("", index=R.index)
    med = R.groupby("trip").ratio.transform("median")
    diff = R.ours_road_min - R.osrm_min
    flag[R.osrm_min.isna()] = "no_route"
    flag[short] = "same_road_point"
    flag[(R.ratio > 2.5 * med) & (diff >= 30)] = "slow_outlier"
    flag[(R.ratio < 0.4 * med) & (diff <= -30)] = "fast_outlier"
    R["flag"] = flag; R["ratio_vs_typical"] = (R.ratio / med).round(2); R["ratio"] = R.ratio.round(3)
    R.to_csv(out, index=False)

    st = {}
    for trip, g in R.groupby("trip"):
        ok = g[g.ratio.notna()]
        st[trip] = {"trips": len(g), "median_ratio": round(float(ok.ratio.median()), 3),
                    "iqr": [round(float(ok.ratio.quantile(q)), 3) for q in (0.25, 0.75)],
                    "within_25pct_of_typical": round(float(ok.ratio_vs_typical.between(0.8, 1.25).mean()), 3),
                    "spearman": round(float(ok[["ours_road_min", "osrm_min"]].corr("spearman").iloc[0, 1]), 3),
                    "flags": g.flag.replace("", "ok").value_counts().to_dict(),
                    "by_type": ok.groupby("Type").ratio.median().round(3).to_dict(),
                    "by_region": ok.groupby("region").ratio.median().round(3).to_dict()}
    json.dump(st, open(out.replace(".csv", ".json"), "w"), indent=1); print(json.dumps(st, indent=1))

    # --- chart: ratio relative to typical, by trip -------------------------------------------------
    LV = [0, 0.4, 0.8, 1.25, 2.5, 1e9]; LAB = ["< 0.4× (OSRM much slower)", "0.4–0.8×", "within 25%", "1.25–2.5×", "> 2.5× (ours much slower)"]
    COL = ["#2f6aa8", "#9ec5f4", "#d4d3cc", "#ec9a3b", "#a8320f"]
    cmap, norm = ListedColormap(COL), BoundaryNorm(LV, 5)
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.8), gridspec_kw={"wspace": 0.05})
    for ax, (trip, ttl) in zip(axs, [("hospital", "Trip to the nearest hospital · health centres and posts"),
                                       ("centre", "Trip from a health post to its health centre")]):
        g = R[(R.trip == trip) & R.ratio.notna()].copy()
        g["dev"] = np.abs(np.log(g.ratio_vs_typical)); g = g.sort_values("dev")
        wn.draw_outline(ax); wn.mapax(ax)
        ax.scatter(g.Longitude, g.Latitude, c=g.ratio_vs_typical, cmap=cmap, norm=norm, s=np.where(g.dev > np.log(2.5), 14, 4), linewidths=0, zorder=3)
        nr = R[(R.trip == trip) & (R.flag == "no_route")]
        ax.scatter(nr.Longitude, nr.Latitude, marker="x", s=14, color=INK, lw=0.8, zorder=4)
        S = st[trip]
        ax.set_title(ttl, loc="left", fontsize=11, color=INK, pad=4)
        ax.text(0.02, -0.03, f"{S['trips']:,} trips · typical ratio {S['median_ratio']:.2f} · {100 * S['within_25pct_of_typical']:.0f}% within 25% of it\n"
                f"flagged: {S['flags'].get('slow_outlier', 0)} much slower, {S['flags'].get('fast_outlier', 0)} much faster, {S['flags'].get('no_route', 0)} no OSRM route\n"
                f"{S['flags'].get('same_road_point', 0):,} not compared (both ends on the same road point)",
                transform=ax.transAxes, fontsize=9, color=INK2, va="top")
    hd = [Line2D([], [], ls="", label="Our road time ÷ OSRM, vs the typical ratio:")]
    hd += [Line2D([], [], marker="o", ls="", color=c, markersize=7, label=l) for c, l in zip(COL, LAB)]
    hd += [Line2D([], [], marker="x", ls="", color=INK, markersize=6, label="no OSRM route")]
    fig.legend(handles=hd, loc="lower center", ncol=7, frameon=False, fontsize=9, handletextpad=0.3, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Where the road model disagrees with OSRM, facility by facility", x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK, y=1.0)
    fig.text(0.01, 0.935, f"Normal-speed road time on each facility's own trip vs OSRM car (public demo server) · run {pd.Timestamp(a.init):%d %b %Y}",
             fontsize=9.5, color=INK2)
    fig.text(0.01, -0.11, "Typical ratio ≈ 1.3: we slow unpaved roads to 70%, OSRM drives untagged roads at full speed. Flags: much slower = ratio > 2.5× typical and 30+ min more "
             "(likely a gap or detour in our road graph); much faster = < 0.4× typical and 30+ min less. Walking legs removed from our time. Data © OpenStreetMap contributors.",
             fontsize=8, color=MUTED, va="top", wrap=True)
    wn.save(fig, os.path.join(wn.OUT, "ETH_osrm_facility_check.png"))
