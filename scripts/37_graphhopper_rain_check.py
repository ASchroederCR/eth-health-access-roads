"""
37_graphhopper_rain_check.py -- an independent router applies the same forecast road tiers, to check the added travel
time that 17_ reports.

17_ times trips on its own road graph (its own speeds by road class, surface and slope). This script gives the same
rain penalties to GraphHopper (open-source router, its own car speeds from OSM road class, surface and track type, no
slope) and asks whether the added minutes come out the same. Agreement means the forecast deltas do not hinge on 17_'s
base speeds or graph; disagreement shows where they do.

How the tiers reach GraphHopper
  One pass over the OSM extract writes four copies (days 1-3, 4-7, 8-15 and the worst tier over the 15 days, which
  gives the headline numbers). In each copy every vehicle way carries its tier in the `hgv` tag (Minor = designated,
  Elevated = destination, Severe = delivery; real hgv tags are removed). GraphHopper's car profile never reads hgv, so
  the tag only carries the tier. Each copy becomes a graph with two profiles, both fastest-route (distance_influence 0):
    normal  GraphHopper's car speeds
    rain    the same, times 0.9 / 0.75 / 0.5 on Minor / Elevated / Severe ways (17_'s SPEED_F)
  Ways 17_ treats as closed are left as GraphHopper finds them (it applies its own access rules).

Trips (as 17_ and 35_): every health centre -> the hospital 17_ assigned it, and every health post -> its health centre
(matched by name, nearest of that name). Each end snaps to GraphHopper's nearest road, the same snap in both profiles,
so walking legs cancel in the added minutes. Added = rain - normal on GraphHopper vs 17_'s to_*_added_* columns.

Needs Java 17+ and the GraphHopper web jar: GH_HOME (default ~/tools) holding graphhopper/graphhopper-web-*.jar and a
jdk-*/ folder (or java on PATH).

Outputs (<init> = forecast run date)
  data/processed/tables/eth_graphhopper_rain_check_<init>.csv   one row per facility, trip and window
  data/processed/tables/eth_graphhopper_rain_check_<init>.json  agreement by trip and window
  charts/ETH_graphhopper_rain_check.png

    python scripts\\37_graphhopper_rain_check.py [--init 2026-11-20] [--threads 12]
"""
import argparse, glob, json, os, shutil, subprocess, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
import importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TB = os.path.join(ROOT, "data", "processed", "tables"); PROC = os.path.join(ROOT, "data", "processed", "roads")
OSMD = os.path.join(ROOT, "data", "raw", "osm"); GHD = os.path.join(OSMD, "graphhopper")
spec = importlib.util.spec_from_file_location("wn", os.path.join(ROOT, "scripts", "11_weathernext_forecast.py"))
wn = importlib.util.module_from_spec(spec); spec.loader.exec_module(wn)
for _f in glob.glob(os.path.join(ROOT, "assets", "fonts", "roboto", "*.ttf")): _fm.fontManager.addfont(_f)
plt.rcParams.update({"font.family": ["Roboto", "DejaVu Sans"]})
INK, INK2, MUTED = wn.INK, wn.INK2, wn.MUTED
C_OURS, C_GH = "#256abf", "#e0701f"

SPEED_F = {"Minor": 0.9, "Elevated": 0.75, "Severe": 0.5}            # 17_
HGV = {"Minor": "designated", "Elevated": "destination", "Severe": "delivery"}
SCEN = {"w1": "tier_w1", "w2": "tier_w2", "w3": "tier_w3", "worst": "tier"}
SCEN_LAB = {"w1": "days 1–3", "w2": "days 4–7", "w3": "days 8–15", "worst": "all 15 days"}
OURS = {("hospital", "w1"): "to_hosp_added_w1_min", ("hospital", "w2"): "to_hosp_added_w2_min",
        ("hospital", "w3"): "to_hosp_added_w3_min", ("hospital", "worst"): "to_hosp_added_min",
        ("centre", "w1"): "to_hc_added_w1_min", ("centre", "w2"): "to_hc_added_w2_min",
        ("centre", "w3"): "to_hc_added_w3_min", ("centre", "worst"): "to_hc_added_min"}
COSLAT = np.cos(np.deg2rad(9.0))
PORT = 8989


def tag_extracts(pbf, tiers, tag):
    """tiers: DataFrame osm_id + one column per scenario. Writes one tagged copy of the extract per scenario."""
    import osmium
    os.makedirs(GHD, exist_ok=True)
    outs = {s: os.path.join(GHD, f"eth_tiers_{tag}_{s}.osm.pbf") for s in SCEN}
    if all(os.path.exists(p) for p in outs.values()): return outs
    look = {s: {int(i): HGV[t] for i, t in zip(tiers.osm_id, tiers[c]) if t in HGV} for s, c in SCEN.items()}
    tmp = {s: p.replace(".osm.pbf", ".part.osm.pbf") for s, p in outs.items()}     # renamed only when complete
    for p in list(outs.values()) + list(tmp.values()):
        if os.path.exists(p): os.remove(p)
    W = {s: osmium.SimpleWriter(p) for s, p in tmp.items()}

    class H(osmium.SimpleHandler):
        def node(self, n):
            for w in W.values(): w.add_node(n)

        def way(self, w):
            base = {t.k: t.v for t in w.tags if t.k != "hgv"}
            for s, wr in W.items():
                v = look[s].get(w.id)
                wr.add_way(w.replace(tags={**base, "hgv": v}) if v else (w.replace(tags=base) if "hgv" in w.tags else w))

        def relation(self, r):
            for w in W.values(): w.add_relation(r)

    t0 = time.time(); H().apply_file(pbf)
    for w in W.values(): w.close()
    for s in outs: os.replace(tmp[s], outs[s])
    print(f"tagged extracts written in {time.time() - t0:.0f} s: " + ", ".join(f"{s} {len(v):,} ways" for s, v in look.items()), flush=True)
    return outs


def java_cmd(home):
    j = sorted(glob.glob(os.path.join(home, "jdk-*", "bin", "java.exe")) + glob.glob(os.path.join(home, "jdk-*", "bin", "java")))
    return j[0] if j else shutil.which("java")


CONFIG = """graphhopper:
  datareader.file: "{pbf}"
  graph.location: "{loc}"
  graph.encoded_values: car_access, car_average_speed, road_access, hgv
  profiles:
    - name: normal
      custom_model: {{"distance_influence": 0, "priority": [{{"if": "!car_access", "multiply_by": "0"}}], "speed": [{{"if": "true", "limit_to": "car_average_speed"}}]}}
    - name: rain
      custom_model: {{"distance_influence": 0, "priority": [{{"if": "!car_access", "multiply_by": "0"}}], "speed": [{{"if": "true", "limit_to": "car_average_speed"}}, {{"if": "hgv == DESIGNATED", "multiply_by": "{mi}"}}, {{"else_if": "hgv == DESTINATION", "multiply_by": "{el}"}}, {{"else_if": "hgv == DELIVERY", "multiply_by": "{se}"}}]}}
  profiles_ch:
    - profile: normal
    - profile: rain
  profiles_lm: []
  prepare.min_network_size: 200
  routing.snap_preventions_default: tunnel, bridge, ferry
  import.osm.ignored_highways: footway,cycleway,path,pedestrian,steps,bridleway
server:
  application_connectors:
  - type: http
    port: {port}
    bind_host: localhost
  admin_connectors:
  - type: http
    port: {aport}
    bind_host: localhost
logging:
  level: WARN
"""


def start_server(java, jar, pbf, scen, log):
    loc = os.path.join(GHD, f"graph_{os.path.basename(pbf).replace('.osm.pbf', '')}")
    cfg = os.path.join(GHD, f"config_{scen}.yml")
    open(cfg, "w").write(CONFIG.format(pbf=pbf.replace("\\", "/"), loc=loc.replace("\\", "/"), port=PORT, aport=PORT + 1,
                                       mi=SPEED_F["Minor"], el=SPEED_F["Elevated"], se=SPEED_F["Severe"]))
    p = subprocess.Popen([java, "-Xmx8g", "-jar", jar, "server", cfg], stdout=log, stderr=subprocess.STDOUT, cwd=GHD)
    t0 = time.time()
    while time.time() - t0 < 3600:
        if p.poll() is not None: raise RuntimeError(f"GraphHopper exited ({p.returncode}); see {log.name}")
        try:
            if requests.get(f"http://localhost:{PORT}/health", timeout=2).ok:
                print(f"  {scen}: graph ready after {time.time() - t0:.0f} s", flush=True); return p
        except requests.RequestException:
            pass
        time.sleep(5)
    p.kill(); raise RuntimeError("GraphHopper did not start within an hour")


def route_all(pairs, profile, threads):
    """pairs: array [n, 4] lat0, lon0, lat1, lon1. Returns minutes (NaN where no route)."""
    s = requests.Session()
    a = requests.adapters.HTTPAdapter(pool_connections=threads, pool_maxsize=threads); s.mount("http://", a)
    url = f"http://localhost:{PORT}/route"

    def one(r):
        q = {"point": [f"{r[0]:.6f},{r[1]:.6f}", f"{r[2]:.6f},{r[3]:.6f}"], "profile": profile,
             "calc_points": "false", "instructions": "false"}
        for k in range(3):
            try:
                j = s.get(url, params=q, timeout=60)
                if j.status_code == 400: return np.nan          # no route / point not snapped
                j.raise_for_status(); return j.json()["paths"][0]["time"] / 60000
            except requests.RequestException:
                time.sleep(2 * (k + 1))
        return np.nan
    with ThreadPoolExecutor(threads) as ex:
        return np.array(list(ex.map(one, pairs, chunksize=64)), float)


def trips(d):
    """Health centre -> its hospital, health post -> its health centre (as 35_)."""
    d = d[~d.off_network]
    H = d[d.Type == "Hospital"].set_index("Id"); C = d[d.Type == "Health Center"]
    f = d[(d.Type == "Health Center") & d.nearest_hospital_id.notna() & d.to_hosp_min.notna()].copy()
    f["tid"] = f.nearest_hospital_id.astype(int); f = f[f.tid.isin(H.index)]
    f["trip"] = "hospital"; f["tlat"] = H.Latitude.reindex(f.tid).values; f["tlon"] = H.Longitude.reindex(f.tid).values
    f["target"] = H.Name.reindex(f.tid).values
    p = d[(d.Type == "Health Post") & d.nearest_health_centre.notna() & d.to_hc_min.notna()].copy()
    cidx = {n: g for n, g in C.groupby("Name")}

    def pick(r):
        g = cidx.get(r.nearest_health_centre)
        if g is None: return -1
        dd = np.hypot((g.Longitude - r.Longitude) * COSLAT, g.Latitude - r.Latitude)
        return int(g.Id.values[np.argmin(dd.values)])
    p["tid"] = [pick(r) for r in p.itertuples()]; p = p[p.tid >= 0]
    Cx = C.set_index("Id")
    p["trip"] = "centre"; p["tlat"] = Cx.Latitude.reindex(p.tid).values; p["tlon"] = Cx.Longitude.reindex(p.tid).values
    p["target"] = Cx.Name.reindex(p.tid).values
    return pd.concat([f, p], ignore_index=True)


def agree(o, g):
    """Agreement of added minutes, ours (o) vs GraphHopper (g)."""
    ok = o.notna() & g.notna()
    o, g = o[ok], g[ok]
    close = (o - g).abs() <= np.maximum(5, 0.25 * np.maximum(o, g))
    o30, g30 = o >= 30, g >= 30
    return {"trips": int(ok.sum()), "spearman": round(float(pd.concat([o, g], axis=1).corr("spearman").iloc[0, 1]), 3),
            "median_added_ours": round(float(o.median()), 1), "median_added_gh": round(float(g.median()), 1),
            "mean_added_ours": round(float(o.mean()), 1), "mean_added_gh": round(float(g.mean()), 1),
            "within_5min_or_25pct": round(float(close.mean()), 3),
            "n30_ours": int(o30.sum()), "n30_gh": int(g30.sum()), "n30_both": int((o30 & g30).sum()),
            "n60_ours": int((o >= 60).sum()), "n60_gh": int((g >= 60).sum())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--init", default=None)
    ap.add_argument("--threads", type=int, default=12); a = ap.parse_args()
    if a.init is None:
        a.init = pd.Timestamp(sorted(glob.glob(os.path.join(TB, "eth_health_access_2*.csv")))[-1][-12:-4]).strftime("%Y-%m-%d")
    tag = a.init.replace("-", "")
    home = os.environ.get("GH_HOME", os.path.join(os.path.expanduser("~"), "tools"))
    jar = sorted(glob.glob(os.path.join(home, "graphhopper", "graphhopper-web-*.jar")))[-1]; java = java_cmd(home)
    print(f"GraphHopper {os.path.basename(jar)} with {java}", flush=True)

    r = pd.read_parquet(os.path.join(PROC, f"eth_roads_scored_{tag}.parquet"), columns=["osm_id"] + list(SCEN.values()))
    pbf = sorted(glob.glob(os.path.join(OSMD, "ethiopia-*.osm.pbf")))[-1]
    outs = tag_extracts(pbf, r, tag)

    d = pd.read_csv(os.path.join(TB, f"eth_health_access_{tag}.csv"))
    T = trips(d).reset_index(drop=True)
    pairs = T[["Latitude", "Longitude", "tlat", "tlon"]].to_numpy()
    print(f"{len(T):,} trips: {(T.trip == 'hospital').sum():,} centre -> hospital, {(T.trip == 'centre').sum():,} post -> centre", flush=True)

    res = {}
    log = open(os.path.join(GHD, f"graphhopper_{tag}.log"), "a")
    for s in SCEN:
        p = start_server(java, jar, outs[s], s, log)
        try:
            t0 = time.time()
            if "normal" not in res: res["normal"] = route_all(pairs, "normal", a.threads)
            res[s] = route_all(pairs, "rain", a.threads)
            print(f"  {s}: routed in {time.time() - t0:.0f} s; no route {int(np.isnan(res[s]).sum())}", flush=True)
        finally:
            p.terminate()
            try: p.wait(60)
            except subprocess.TimeoutExpired: p.kill()
            time.sleep(3)

    rows = []
    for s in SCEN:
        for trip in ("hospital", "centre"):
            m = (T.trip == trip).values
            rows.append(pd.DataFrame({"Id": T.Id[m].values, "Name": T.Name[m].values, "Type": T.Type[m].values,
                                      "region": T.admin1Name[m].values, "Latitude": T.Latitude[m].values, "Longitude": T.Longitude[m].values,
                                      "trip": trip, "window": s, "target_id": T.tid[m].values, "target": T.target[m].values,
                                      "ours_normal_min": (T.to_hosp_min if trip == "hospital" else T.to_hc_min)[m].values,
                                      "ours_added_min": T[OURS[(trip, s)]][m].values,
                                      "gh_normal_min": res["normal"][m].round(1), "gh_rain_min": res[s][m].round(1),
                                      "gh_added_min": (res[s][m] - res["normal"][m]).round(1)}))
    R = pd.concat(rows, ignore_index=True)
    out = os.path.join(TB, f"eth_graphhopper_rain_check_{tag}.csv"); R.to_csv(out, index=False)

    st = {"init": a.init, "graphhopper": os.path.basename(jar), "speed_factors": SPEED_F}
    for trip in ("hospital", "centre"):
        st[trip] = {}
        for s in SCEN:
            g = R[(R.trip == trip) & (R.window == s)]
            st[trip][s] = agree(g.ours_added_min, g.gh_added_min)
        g = R[(R.trip == trip) & (R.window == "worst")]
        ok = g.ours_normal_min.notna() & g.gh_normal_min.notna()
        st[trip]["normal_time_spearman"] = round(float(g[ok][["ours_normal_min", "gh_normal_min"]].corr("spearman").iloc[0, 1]), 3)
        st[trip]["normal_time_median_ratio"] = round(float((g.ours_normal_min / g.gh_normal_min.where(g.gh_normal_min > 1))[ok].median()), 3)
        st[trip]["no_route"] = int(g.gh_normal_min.isna().sum())
        dd = pd.read_csv(os.path.join(TB, f"eth_health_access_{tag}.csv"), usecols=["Id", "road_gap_km"]).set_index("Id").road_gap_km
        st[trip]["no_route_median_road_gap_km"] = round(float(dd.reindex(g.Id[g.gh_normal_min.isna()]).median()), 2)
        # slowdown as a share of the trip's normal time: removes the two routers' different base speeds
        m = ok & (g.ours_normal_min > 5) & (g.gh_normal_min > 5)
        ro, rh = (g.ours_added_min / g.ours_normal_min)[m], (g.gh_added_min / g.gh_normal_min)[m]
        st[trip]["relative_added_worst"] = {"median_ours": round(float(ro.median()), 3), "median_gh": round(float(rh.median()), 3),
                                            "mean_ours": round(float(ro.mean()), 3), "mean_gh": round(float(rh.mean()), 3),
                                            "spearman": round(float(pd.concat([ro, rh], axis=1).corr("spearman").iloc[0, 1]), 3)}
        rg = {}
        for reg, x in g.groupby("region"):
            rg[reg] = {"n30_ours": int((x.ours_added_min >= 30).sum()), "n30_gh": int((x.gh_added_min >= 30).sum()),
                       "mean_ours": round(float(x.ours_added_min.mean()), 1), "mean_gh": round(float(x.gh_added_min.mean()), 1)}
        st[trip]["by_region_worst"] = rg
    json.dump(st, open(out.replace(".csv", ".json"), "w"), indent=1); print(json.dumps(st, indent=1))

    # --- chart -------------------------------------------------------------------------------------------
    fig = plt.figure(figsize=(15, 11.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.8], wspace=0.42, hspace=0.95)
    tr = symlog = lambda v: np.sign(v) * np.log10(1 + np.abs(v))
    TK = [0, 5, 15, 30, 60, 120, 240]
    for k, (trip, ttl) in enumerate([("hospital", "Health centre → hospital"), ("centre", "Health post → health centre")]):
        ax = fig.add_subplot(gs[0, k])
        g = R[(R.trip == trip) & (R.window == "worst")].dropna(subset=["ours_added_min", "gh_added_min"])
        x, y = tr(g.ours_added_min.clip(lower=0)), tr(g.gh_added_min.clip(lower=0))
        ax.scatter(x, y, s=5 if trip == "hospital" else 2.5, color=C_OURS, alpha=0.35, linewidths=0, zorder=3)
        lim = tr(max(240, g[["ours_added_min", "gh_added_min"]].max().max()))
        ax.plot([0, lim], [0, lim], color=INK2, lw=0.8, zorder=2)
        ax.axvline(tr(30), color=MUTED, lw=0.6, ls=":"); ax.axhline(tr(30), color=MUTED, lw=0.6, ls=":")
        ax.set_xticks(tr(np.array(TK)), TK); ax.set_yticks(tr(np.array(TK)), TK); ax.set_xlim(-0.05, lim); ax.set_ylim(-0.05, lim)
        ax.set_xlabel("Added minutes, our road model", fontsize=9.5, color=INK2); ax.set_ylabel("Added minutes, GraphHopper", fontsize=9.5, color=INK2)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=8.5, colors=INK2); ax.grid(color="#ecebe7", lw=0.6, zorder=0)
        S = st[trip]["worst"]; RL = st[trip]["relative_added_worst"]
        ax.set_title(ttl, loc="left", fontsize=11, color=INK, fontweight="bold", pad=6)
        ax.text(0.0, -0.2, f"{S['trips']:,} trips · rank correlation {S['spearman']:.2f}\n{100 * S['within_5min_or_25pct']:.0f}% within 5 min or 25%\n"
                f"30+ min: ours {S['n30_ours']:,}, GraphHopper {S['n30_gh']:,}, both {S['n30_both']:,}\n"
                f"normal trip: ours {st[trip]['normal_time_median_ratio']:.2f}× GraphHopper's\n"
                f"slowdown ÷ normal time, median:\n   ours {100 * RL['median_ours']:.0f}%, GraphHopper {100 * RL['median_gh']:.0f}%",
                transform=ax.transAxes, fontsize=8.8, color=INK2, va="top", ha="left", linespacing=1.35)
    # bars: 30+ counts by window, both trips
    for k, (trip, ttl) in enumerate([("hospital", "Health centres 30+ min longer to their hospital"),
                                     ("centre", "Health posts 30+ min longer to their health centre")]):
        ax = fig.add_subplot(gs[1, k])
        xs = np.arange(4); o = [st[trip][s]["n30_ours"] for s in SCEN]; g_ = [st[trip][s]["n30_gh"] for s in SCEN]
        ax.bar(xs - 0.2, o, 0.38, color=C_OURS, label="our road model", zorder=3)
        ax.bar(xs + 0.2, g_, 0.38, color=C_GH, label="GraphHopper", zorder=3)
        for xi, v in zip(xs - 0.2, o): ax.text(xi, v, f"{v:,}", ha="center", va="bottom", fontsize=8.5, color=INK2)
        for xi, v in zip(xs + 0.2, g_): ax.text(xi, v, f"{v:,}", ha="center", va="bottom", fontsize=8.5, color=INK2)
        ax.set_xticks(xs, [SCEN_LAB[s] for s in SCEN], fontsize=9); ax.set_ylim(0, max(o + g_) * 1.18)
        for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", labelsize=8.5, colors=INK2, length=0); ax.grid(axis="y", color="#ecebe7", lw=0.6, zorder=0)
        ax.set_title(ttl, loc="left", fontsize=10.5, color=INK, pad=6)
    # right column: region comparison, centre -> hospital, worst window
    ax = fig.add_subplot(gs[:, 2])
    rg = pd.DataFrame(st["hospital"]["by_region_worst"]).T.sort_values("n30_ours")
    rg = rg[(rg.n30_ours + rg.n30_gh) > 0]
    y = np.arange(len(rg))
    ax.barh(y + 0.2, rg.n30_ours, 0.38, color=C_OURS, zorder=3, label="our road model")
    ax.barh(y - 0.2, rg.n30_gh, 0.38, color=C_GH, zorder=3, label="GraphHopper")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.99, 0.965))
    for yi, v in zip(y + 0.2, rg.n30_ours): ax.text(v, yi, f" {int(v)}", va="center", fontsize=8, color=INK2)
    for yi, v in zip(y - 0.2, rg.n30_gh): ax.text(v, yi, f" {int(v)}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, rg.index, fontsize=9); ax.set_xlim(0, rg[["n30_ours", "n30_gh"]].max().max() * 1.18)
    for sp in ("top", "right", "bottom"): ax.spines[sp].set_visible(False)
    ax.tick_params(axis="x", labelsize=8.5, colors=INK2, length=0); ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#ecebe7", lw=0.6, zorder=0)
    ax.set_title("Health centres 30+ min longer, by region\n(all 15 days)", loc="left", fontsize=10.5, color=INK, pad=6)
    fig.suptitle("GraphHopper with the same forecast road tiers: do the added minutes agree?", x=0.01, ha="left", fontsize=13.5,
                 fontweight="bold", color=INK, y=0.995)
    fig.text(0.01, 0.955, f"WeatherNext 3 run {pd.Timestamp(a.init):%d %b %Y} · each trip timed normally and under the forecast on both routers · "
             f"top row: worst tier over the 15 days (axes stretched at the low end)", fontsize=9.5, color=INK2)
    wn.foot(fig, f"GraphHopper {st['graphhopper'].replace('graphhopper-web-', '').replace('.jar', '')} car speeds (road class, surface, track type; no slope), fastest route, "
            "with 17_'s tier speeds: Minor ×0.9, Elevated ×0.75, Severe ×0.5.\nOur road model: own speeds by class, surface, roughness and slope. "
            f"Same trips and targets on both. Not compared: {st['hospital']['no_route'] + st['centre']['no_route']:,} trips GraphHopper could not snap "
            f"(facility several km from a road; median {st['centre']['no_route_median_road_gap_km']:.1f} km). Data © OpenStreetMap contributors.")
    wn.save(fig, os.path.join(wn.OUT, "ETH_graphhopper_rain_check.png"))
