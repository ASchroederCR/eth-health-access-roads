"""
43_acled_zones.py -- political violence events and fatalities by zone and month since 1 January 2026, for the web map's
time-series layer. Source: ACLED's aggregated monthly counts for Ethiopia on HDX (dataset ethiopia-acled-conflict-data,
"political violence events and fatalities by month-year"), refreshed weekly by ACLED; fetched here when the cached copy
is more than 3 days old. Use is subject to ACLED's Terms of Use and Attribution Policy (https://acleddata.com/terms-of-use/).

The HDX file uses an older zone set. Drawn on the OCHA COD-AB 2025 zones:
  - by pcode where the zone is unchanged;
  - West Gojam = 2025 West Gojam + North Gojam + Bahir Dar town (carved out of it); Sidama (one unit in ACLED) = the five
    2025 Sidama zones; Western Tigray = contested Area 1;
  - not covered, drawn as such: Central and South Ethiopia (the file's former-SNNP zones carry no events in any year),
    contested Areas 2 and 3, and two new zones whose events sit in a former parent zone (East Borena, Afar Mahi/Zone 6).

Outputs
  data/raw/acled/eth_pv_by_month.xlsx          cached HDX file
  webmap/data/conflict.json                    {"src", "asof", "months", "z": [[name, region, [events], [fatalities]]], ...}
  webmap/data/conflict.geojson                 zone outlines (simplified ~200 m), property i = row in z, -1 = not covered

    python scripts\\43_acled_zones.py [--refresh]
"""
import argparse, json, os, re, sys, time
import importlib.util
import numpy as np, pandas as pd, geopandas as gpd, requests, shapely

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "acled"); os.makedirs(RAW, exist_ok=True)
XL = os.path.join(RAW, "eth_pv_by_month.xlsx"); META = os.path.join(RAW, "eth_pv_by_month.json")
WEB = os.path.join(ROOT, "webmap", "data")
API = "https://data.humdata.org/api/3/action/package_show?id=ethiopia-acled-conflict-data"
START = 2026
MERGE = {"ET0307": ["ET0307", "ET0313", "ET0314"], "ET1600": ["ET1601", "ET1602", "ET1603", "ET1604", "ET1605"], "ET0105": ["ET9901"]}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def fetch(force=False):
    if not force and os.path.exists(XL) and time.time() - os.path.getmtime(XL) < 3 * 86400:
        return json.load(open(META)) if os.path.exists(META) else {}
    d = requests.get(API, timeout=60, headers={"User-Agent": "data-client/1.0"}).json()["result"]
    res = next(r for r in d["resources"] if "political_violence" in r["name"].lower())
    r = requests.get(res["url"], timeout=300, headers={"User-Agent": "data-client/1.0"}); r.raise_for_status()
    open(XL, "wb").write(r.content)
    m = re.search(r"TO (\d{4}-\d{2}-\d{2})", d.get("dataset_date", ""))
    meta = {"asof": m.group(1) if m else None, "file": os.path.basename(res["url"]), "modified": res.get("last_modified")}
    json.dump(meta, open(META, "w")); print(f"fetched {meta['file']} (data to {meta['asof']})")
    return meta


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(); ap.add_argument("--refresh", action="store_true"); a = ap.parse_args()
    meta = fetch(a.refresh)
    d = pd.read_excel(XL, sheet_name="Data")
    d = d[(d.Year >= START) & (d.Admin1 != "SNNP")].copy()
    d["m"] = d.Year.astype(str) + "-" + d.Month.map({m: f"{i + 1:02d}" for i, m in enumerate(MONTHS)})
    months = sorted(d.m.unique())
    spec = importlib.util.spec_from_file_location("w", os.path.join(ROOT, "scripts", "40_woreda_population.py"))
    w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
    Z = gpd.read_file(w.gdb_path(), layer="eth_admin2", engine="pyogrio").to_crs(4326)
    Z["geometry"] = shapely.make_valid(Z.geometry.values)
    geo = dict(zip(Z.adm2_pcode, Z.geometry)); used = set()
    rows, feats = [], []
    for (pc, a1, a2), g in d.groupby(["Admin2 Pcode", "Admin1", "Admin2"]):
        parts = [p for p in MERGE.get(pc, [pc]) if p in geo]
        if not parts: print(f"  no 2025 outline for {a2} ({a1}, {pc}): {int(g.Events.sum())} events left off the map"); continue
        s = g.set_index("m").reindex(months).fillna(0)
        feats.append({"i": len(rows), "nm": a2, "geometry": shapely.union_all([geo[p] for p in parts])})
        rows.append([a2, a1, s.Events.astype(int).tolist(), s.Fatalities.astype(int).tolist()]); used |= set(parts)
    nc = Z[~Z.adm2_pcode.isin(used)]
    for r in nc.itertuples(): feats.append({"i": -1, "nm": f"{r.adm2_name} ({r.adm1_name})", "geometry": r.geometry})
    G = gpd.GeoDataFrame(feats, crs=4326)
    G["geometry"] = shapely.set_precision(shapely.simplify(G.geometry.values, 0.002), 1e-4)
    G.to_file(os.path.join(WEB, "conflict.geojson"), driver="GeoJSON", engine="pyogrio")
    out = {"src": "ACLED (Armed Conflict Location & Event Data), political violence events and fatalities by zone and month, via HDX",
           "asof": meta.get("asof"), "months": months, "z": rows, "nc": sorted(f"{r.adm2_name} ({r.adm1_name})" for r in nc.itertuples())}
    json.dump(out, open(os.path.join(WEB, "conflict.json"), "w", encoding="utf-8"), separators=(",", ":"), ensure_ascii=False)
    ev = sum(sum(r[2]) for r in rows); fa = sum(sum(r[3]) for r in rows)
    print(f"ACLED political violence {months[0]}..{months[-1]} (to {meta.get('asof')}): {len(rows)} zones, {ev:,} events, {fa:,} fatalities; "
          f"{len(nc)} 2025 zones not covered: {', '.join(out['nc'])}")
