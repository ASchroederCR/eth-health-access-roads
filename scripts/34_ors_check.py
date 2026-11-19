"""
34_ors_check.py -- check the access model's normal-speed travel times against openrouteservice (ORS) driving-car.

17_ routes each facility to its nearest hospital on the OSM road graph with fixed speeds by road class (unpaved x 0.7)
and walks the gap to the road at 5 km/h at both ends. ORS routes the same trip with its own OSM speed model (maxspeed,
surface, tracktype, smoothness, access, one-ways, turn costs) and snaps each end onto the road with no walking time.
To compare like with like, the walk legs are taken off 17_'s time and only trips whose ends are within 2 km of a road
are sampled.

Sample: ~200 health centres and posts (not off-network), stratified by region and by 17_'s travel-time quintile, with a
fixed seed. Each trip is one ORS directions request (free key: 40/min, 2,000/day).

Key: environment variable ORS_API_KEY, or the first line of ~/.ors_key (kept out of the repo).

Outputs (<init> = forecast run date whose access table is checked)
  data/processed/tables/eth_ors_check_<init>.csv       every sampled trip: ours vs ORS minutes and ORS km
  charts/ETH_ors_check.png                              ours vs ORS, and the ratio by road type of the nearest road

    python scripts\\34_ors_check.py [--init 2026-11-19] [--n 200]
"""
import argparse, glob, json, os, time
import numpy as np, pandas as pd, requests
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TB = os.path.join(ROOT, "data", "processed", "tables"); CH = os.path.join(ROOT, "charts")
for _f in glob.glob(os.path.join(ROOT, "assets", "fonts", "roboto", "*.ttf")): _fm.fontManager.addfont(_f)
plt.rcParams.update({"font.family": ["Roboto", "DejaVu Sans"], "font.size": 9.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#888", "axes.labelcolor": "#333",
                     "xtick.color": "#555", "ytick.color": "#555", "figure.dpi": 150, "savefig.dpi": 200})
URL = "https://api.openrouteservice.org/v2/directions/driving-car"
OSRM = "https://router.project-osrm.org/route/v1/driving/{a0},{a1};{b0},{b1}?overview=false&radiuses=2500;2500"
WALK = 60 / 5                                         # minutes per km at 5 km/h, as in 17_
INK, MUTED, ACC = "#1f2a2e", "#6b7478", "#2b6f8a"


def key():
    k = os.environ.get("ORS_API_KEY")
    p = os.path.expanduser("~/.ors_key")
    if not k and os.path.exists(p): k = open(p).readline().strip()
    if not k: raise SystemExit("no ORS key: set ORS_API_KEY or put it in ~/.ors_key")
    return k


def route_osrm(s, k, a, b):
    for i in range(6):
        r = s.get(OSRM.format(a0=a[0], a1=a[1], b0=b[0], b1=b[1]), headers={"User-Agent": "road-access-check/1.0"}, timeout=60)
        if r.status_code == 429: time.sleep(20 * (i + 1)); continue
        j = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        if r.status_code != 200 or j.get("code") != "Ok": return np.nan, np.nan, (j.get("message") or r.text)[:120]
        return j["routes"][0]["duration"] / 60, j["routes"][0]["distance"] / 1000, ""
    return np.nan, np.nan, "rate limited"


def route(s, k, a, b):
    body = {"coordinates": [a, b], "radiuses": [2500, 2500], "instructions": False, "preference": "fastest"}
    for i in range(6):
        r = s.post(URL, json=body, headers={"Authorization": k, "Content-Type": "application/json"}, timeout=60)
        if r.status_code == 429: time.sleep(20 * (i + 1)); continue
        if r.status_code != 200: return np.nan, np.nan, r.json().get("error", {}).get("message", r.text)[:120] if r.headers.get("content-type", "").startswith("application/json") else r.text[:120]
        sm = r.json()["routes"][0]["summary"]
        return sm.get("duration", 0) / 60, sm.get("distance", 0) / 1000, ""
    return np.nan, np.nan, "rate limited"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--init", default="2026-11-19"); ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--router", choices=["ors", "osrm"], default="ors")
    a = ap.parse_args(); tag = a.init.replace("-", "")
    d = pd.read_csv(os.path.join(TB, f"eth_health_access_{tag}.csv"))
    hosp = d[d.Type == "Hospital"].set_index("Id")
    f = d[d.Type.isin(["Health Center", "Health Post"]) & ~d.off_network & d.to_hosp_min.notna() & d.nearest_hospital_id.notna()].copy()
    f["hid"] = f.nearest_hospital_id.astype(int)
    f = f[f.hid.isin(hosp.index)]
    f["h_gap"] = hosp.road_gap_km.reindex(f.hid).values
    f = f[(f.road_gap_km <= 2) & (f.h_gap <= 2)]
    f["ours_road_min"] = f.to_hosp_min - WALK * (f.road_gap_km + f.h_gap)
    f = f[f.ours_road_min > 5]
    f["q"] = pd.qcut(f.ours_road_min, 5, labels=False)
    rng = np.random.default_rng(42)
    per = f.groupby(["admin1Name", "q"]).size()
    take = (per / per.sum() * a.n).apply(np.ceil).astype(int)                       # proportional, >=1 per stratum
    smp = pd.concat([g.sample(min(len(g), take[k]), random_state=int(rng.integers(1e9))) for k, g in f.groupby(["admin1Name", "q"])])
    smp = smp.sample(min(a.n, len(smp)), random_state=42) if len(smp) > a.n else smp
    print(f"{len(f):,} eligible trips, sampling {len(smp)}")

    RT = a.router; NAME = {"ors": "ORS driving-car", "osrm": "OSRM car"}[RT]
    out = os.path.join(TB, f"eth_{RT}_check_{tag}.csv")
    done = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame(columns=["Id"])
    k = key() if RT == "ors" else None; s = requests.Session(); fn = route if RT == "ors" else route_osrm; rows = done.to_dict("records"); seen = set(done.Id)
    for i, r in enumerate(smp.itertuples()):
        if r.Id in seen: continue
        h = hosp.loc[r.hid]
        t, km, err = fn(s, k, [r.Longitude, r.Latitude], [h.Longitude, h.Latitude])
        rows.append({"Id": r.Id, "Name": r.Name, "Type": r.Type, "region": r.admin1Name, "hospital": h.Name,
                     "road_gap_km": r.road_gap_km, "hosp_gap_km": r.h_gap, "nearest_road_tier": r.nearest_road_tier,
                     "ours_total_min": r.to_hosp_min, "ours_road_min": round(r.ours_road_min, 1),
                     "ors_min": round(t, 1) if t == t else np.nan, "ors_km": round(km, 2) if km == km else np.nan, "error": err})
        if (i + 1) % 20 == 0:
            pd.DataFrame(rows).to_csv(out, index=False); print(f"  {i + 1}/{len(smp)}", flush=True)
        time.sleep(1.6 if RT == "ors" else 1.1)                                                              # 40 requests / minute
    R = pd.DataFrame(rows); R.to_csv(out, index=False)

    ok = R[R.ors_min.notna() & (R.ors_min > 0)].copy()
    ok["ratio"] = ok.ours_road_min / ok.ors_min
    lr = np.log(ok.ratio)
    st = {"n": len(ok), "failed": int(R.ors_min.isna().sum()), "median_ratio": float(ok.ratio.median()),
          "iqr": [float(ok.ratio.quantile(.25)), float(ok.ratio.quantile(.75))],
          "within_25pct": float((ok.ratio.between(0.8, 1.25)).mean()),
          "spearman": float(ok[["ours_road_min", "ors_min"]].corr("spearman").iloc[0, 1]),
          "ors_kmh_median": float((ok.ors_km / (ok.ors_min / 60)).median())}
    st["by_type"] = ok.groupby("Type").ratio.median().round(3).to_dict()
    st["by_region"] = ok.groupby("region").ratio.agg(["median", "size"]).round(3).to_dict("index")
    st["by_length"] = ok.groupby(pd.cut(ok.ors_min, [0, 30, 60, 120, 240, 1e4])).ratio.median().round(3).astype(float).to_dict()
    st["by_length"] = {str(k): v for k, v in st["by_length"].items()}
    print(json.dumps(st, indent=1))
    json.dump(st, open(os.path.join(TB, f"eth_{RT}_check_{tag}.json"), "w"), indent=1)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    m = max(ok.ours_road_min.max(), ok.ors_min.max()) * 1.05
    a1.plot([1, m], [1, m], color=MUTED, lw=1, ls="--", zorder=1)
    for lo, hi in ((0.8, 1.25),):
        xs = np.array([1, m]); a1.fill_between(xs, xs * lo, xs * hi, color="#d9e4e8", zorder=0, lw=0)
    for t_, mk in (("Health Center", "o"), ("Health Post", "^")):
        z = ok[ok.Type == t_]
        a1.scatter(z.ors_min, z.ours_road_min, s=22, marker=mk, facecolor="none", edgecolor=ACC if mk == "o" else INK, lw=0.9,
                   label=f"{t_.replace('Center', 'centre').replace('Post', 'post')} ({len(z)})", zorder=3)
    a1.set_xscale("log"); a1.set_yscale("log"); a1.set_xlim(3, m); a1.set_ylim(3, m)
    for ax in (a1,):
        ax.set_xticks([5, 10, 30, 60, 120, 300, 600]); ax.set_xticklabels(["5", "10", "30", "60", "120", "300", "600"])
        ax.set_yticks([5, 10, 30, 60, 120, 300, 600]); ax.set_yticklabels(["5", "10", "30", "60", "120", "300", "600"])
        ax.minorticks_off()
    a1.set_xlabel(f"{NAME}, minutes"); a1.set_ylabel("this model (road part, normal speeds), minutes")
    a1.legend(frameon=False, loc="upper left", fontsize=8.5)
    a1.text(0.98, 0.04, "dashed: equal  ·  band: within 25%", transform=a1.transAxes, ha="right", color=MUTED, fontsize=8)
    a1.set_title("Trip to the nearest hospital", loc="left", fontsize=11, fontweight="medium", color=INK)

    bins = [0, 30, 60, 120, 240, 1e4]; lab = ["<30", "30–60", "60–120", "120–240", "240+"]
    ok["b"] = pd.cut(ok.ors_min, bins, labels=lab)
    for j, l in enumerate(lab):
        v = ok.ratio[ok.b == l]
        if not len(v): continue
        a2.scatter(np.full(len(v), j) + rng.uniform(-0.18, 0.18, len(v)), v, s=12, color=ACC, alpha=0.45, lw=0)
        a2.plot([j - 0.3, j + 0.3], [v.median()] * 2, color=INK, lw=2.2)
        a2.text(j + 0.33, v.median(), f"{v.median():.2f}", va="center", fontsize=8.5, color=INK)
        a2.text(j, 3.4, f"n={len(v)}", ha="center", fontsize=7.5, color=MUTED)
    a2.axhline(1, color=MUTED, lw=1, ls="--"); a2.set_yscale("log")
    a2.set_yticks([0.33, 0.5, 0.75, 1, 1.5, 2, 3]); a2.set_yticklabels(["0.33", "0.5", "0.75", "1", "1.5", "2", "3"]); a2.minorticks_off()
    a2.set_ylim(0.25, 4); a2.set_xticks(range(len(lab))); a2.set_xticklabels(lab)
    a2.set_xlabel(f"{NAME.split()[0]} trip time, minutes"); a2.set_ylabel(f"this model ÷ {NAME.split()[0]}  (above 1 = we are slower)")
    a2.set_title("Ratio by trip length (bar = median)", loc="left", fontsize=11, fontweight="medium", color=INK)
    fig.text(0.01, 0.955, f"Normal-speed travel times: this model vs {NAME.split()[0]}", ha="left", fontsize=13.5, fontweight="medium", color=INK)
    fig.text(0.01, 0.9, f"{st['n']} sampled health centre / post → nearest hospital trips · median ratio {st['median_ratio']:.2f} · "
             f"{100 * st['within_25pct']:.0f}% within 25% · rank correlation {st['spearman']:.2f}", fontsize=9.5, color=MUTED)
    fig.text(0.01, 0.015, f"Walking legs (5 km/h to the road) removed from this model's time; {NAME.split()[0]} snaps each end to the nearest road within 2.5 km. "
             "Sample stratified by region and trip-time quintile, ends within 2 km of a road. " + ("ORS © openrouteservice.org / HeiGIT" if RT == "ors" else "OSRM demo server (project-osrm.org)") + ", data © OpenStreetMap contributors.",
             fontsize=7.3, color=MUTED, wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 0.88))
    fig.savefig(os.path.join(CH, f"ETH_{RT}_check.png")); print(f"wrote charts/ETH_{RT}_check.png")
