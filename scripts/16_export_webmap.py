"""
16_export_webmap.py -- data files for the interactive road / rain web map (webmap/index.html).

Reads the outputs of 14_roads_rain_risk.py (scored roads GeoPackage), the WeatherNext 3 run it used (21_),
GEM-15 elevation (15_fetch_gem15_dem.py) and the ADM1 boundaries, and writes compact web files to webmap/data/:
  roads_severe.json, roads_elevated.json, roads_minor.json   flagged segments, simplified ~30 m, compact
  roads_context.json       trunk / primary / secondary roads (all tiers) as a light base network
  rain.geojson             0.25 deg forecast cells: 15-day mean, wettest 3 days, P(day >= 20 mm), hazard
  adm1.geojson             region outlines
  slip.json                slip / rock-fall exposed road stretches (compact, as the road files)
  flood.json               flood-flagged low-lying stretches (14_; as slip.json)
  xings.json               flood-flagged river crossings (points: {"cols", "lut", "f": [[values..., x*1e4, y*1e4]]})
  mask.geojson             everything outside Ethiopia, drawn white on the map
  facilities.geojson       hospitals, health centres, health posts with access results (17_health_access.py) and,
                           where available, the MAP friction-surface results (20_friction_access.py) as fr* fields
  friction.png             MAP 2019 motorized friction as 4 travel-speed classes in the slowdown-tier colours
  hillshade.png            GEM-15 hillshade at 0.01 deg, transparent outside Ethiopia
  summary.json             run, totals by tier, by region, slip exposure

    python scripts\\16_export_webmap.py [--init 2026-11-17]
"""
import argparse, glob, importlib.util, json, os
import numpy as np, pandas as pd, geopandas as gpd, shapely, rasterio
from rasterio.features import rasterize
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "webmap", "data"); os.makedirs(OUT, exist_ok=True)
TIERS = ["Normal", "Minor", "Elevated", "Severe"]
HW = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "track", "service",
      "living_street", "road"]


def r4(geom, tol):
    s = shapely.simplify(geom, tol, preserve_topology=False)
    return shapely.set_precision(s, 1e-4)


def dump(path, gdf, props):
    """Compact layout: {"cols": [...], "lut": {col: [labels]}, "f": [[v1, v2, ..., [x0, y0, dx1, dy1, ...]], ...]}
    with coordinates as integers of 1e-4 deg, delta-encoded after the first point; the page rebuilds GeoJSON."""
    lut, cols = {}, {}
    for k, v in props.items():
        col = gdf[v]
        if not pd.api.types.is_numeric_dtype(col):
            labels = sorted(col.fillna("").unique().tolist()); idx = {l: i for i, l in enumerate(labels)}
            lut[k] = labels; cols[k] = col.fillna("").map(idx).tolist()
        else:
            cols[k] = [int(x) if float(x).is_integer() else float(x) for x in col.tolist()]
    feats = []
    for i, gm in enumerate(gdf.geometry.values):
        if gm is None or gm.is_empty: continue
        c = np.rint(shapely.get_coordinates(gm) * 1e4).astype(np.int64)
        if len(c) < 2: continue
        d = np.vstack([c[:1], np.diff(c, axis=0)]).ravel().tolist()
        feats.append([cols[k][i] for k in props] + [d])
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"cols": list(props), "lut": lut, "f": feats}, f, separators=(",", ":"), ensure_ascii=False)
    return len(feats), os.path.getsize(path) / 1e6


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--init"); a = ap.parse_args()
    spec3 = importlib.util.spec_from_file_location("wn3", os.path.join(ROOT, "scripts", "21_weathernext3_forecast.py"))
    wn3 = importlib.util.module_from_spec(spec3); spec3.loader.exec_module(wn3)
    z = wn3.load_run(a.init); init = pd.Timestamp(str(z["init"])); tag = f"{init:%Y%m%d}"
    g = gpd.read_file(os.path.join(ROOT, "data", "processed", "roads", f"eth_roads_rain_risk_{tag}.gpkg"), engine="pyogrio")
    regions = sorted(g.region.unique())
    days = pd.date_range(init, periods=z["mean"].shape[0])

    # --- flagged roads, compact properties --------------------------------------------
    g["t"] = g.tier.map({t: i for i, t in enumerate(TIERS)})
    g["nm"] = g.name_en.fillna(g.name).fillna("")
    g["sf"] = np.where(g.surface_class == "paved", "paved", "unpaved")
    g["sk"] = g.surface_source.str.startswith("OSM").astype(int)        # 1 = surface from OSM tag / Mapillary model
    g["bad"] = g.smoothness.fillna("")
    g["km"] = (g.osm_length / 1000).round(2)
    g["pk"] = pd.to_datetime(g.peak_day).dt.strftime("%d %b")
    g["sl"] = g.slide_exposure.astype(int)
    g["slide_km"] = g.slide_km.astype(float).round(2)
    g["landslide_susc_max"] = g.landslide_susc_max.astype(int)
    g["tw"] = g.tier_without_terrain.map({t: i for i, t in enumerate(TIERS)})
    for w_ in (1, 2, 3):                                   # per forecast window: tier, accumulation, intensity, slip km
        g[f"t{w_}"] = g[f"tier_w{w_}"].map({t: i for i, t in enumerate(TIERS)})
        g[f"slide_km_w{w_}"] = g[f"slide_km_w{w_}"].astype(float).round(2)
    props = {"t": "t", "tw": "tw", "hw": "highway", "nm": "nm", "sf": "sf", "sk": "sk", "cond": "bad", "km": "km",
             "tc": "terrain_class", "gr": "road_grade_pct", "sm": "terrain_slope_mean_deg", "sx": "terrain_slope_max_deg",
             "el": "elev_mean_m", "r15": "rain_15d_mm", "r3": "wettest_3d_mm", "p20": "p90_wettest_day_mm", "pk": "pk",
             "sl": "sl", "slk": "slide_km", "su": "landslide_susc_max", "rg": "region", "R": "rain_hazard",
             "S": "surface_score", "T": "steep_score",
             "t1": "t1", "t2": "t2", "t3": "t3", "a1": "acc_w1", "a2": "acc_w2", "a3": "acc_w3",
             "i1": "int_w1", "i2": "int_w2", "i3": "int_w3", "s1": "slide_km_w1", "s2": "slide_km_w2", "s3": "slide_km_w3"}
    for c_, nd in [("km", 2), ("road_grade_pct", 1), ("terrain_slope_mean_deg", 1), ("terrain_slope_max_deg", 1),
                   ("rain_15d_mm", 1), ("wettest_3d_mm", 1), ("p90_wettest_day_mm", 1)]:
        g[c_] = g[c_].astype(float).round(nd)
    g["elev_mean_m"] = g.elev_mean_m.round(0).astype(int)
    g["geometry"] = r4(g.geometry.values, 0.0003)
    for t in ["Severe", "Elevated", "Minor"]:
        n, mb = dump(os.path.join(OUT, f"roads_{t.lower()}.json"), g[g.tier == t].reset_index(drop=True), props)
        print(f"{t}: {n} segments, {mb:.1f} MB")

    # --- context: main roads, all tiers -------------------------------------------------
    import importlib.util
    spec = importlib.util.spec_from_file_location("rr", os.path.join(ROOT, "scripts", "14_roads_rain_risk.py"))
    rr = importlib.util.module_from_spec(spec); spec.loader.exec_module(rr)
    allr, _ = rr.drop_seismic(rr.load_roads())
    ctx = allr[allr.highway.isin(["motorway", "trunk", "primary", "secondary", "tertiary"])].copy()
    ctx["geometry"] = r4(ctx.geometry.values, 0.001)
    ctx["c"] = ctx.highway.map({"motorway": 0, "trunk": 0, "primary": 1, "secondary": 2, "tertiary": 3})
    n, mb = dump(os.path.join(OUT, "roads_context.json"), ctx.reset_index(drop=True), {"c": "c"})
    print(f"context: {n} segments, {mb:.1f} MB")

    # --- slip / rock-fall exposed stretches (14_) ---------------------------------------------
    pcs = gpd.read_parquet(os.path.join(ROOT, "data", "processed", "roads", f"eth_slip_pieces_{tag}.parquet"))
    for w_ in ("w1", "w2", "w3"): pcs[w_] = pcs[w_].astype(int)
    pcs = pcs.dissolve(by=["seg", "w1", "w2", "w3"], aggfunc={"km": "sum"}).reset_index()   # join pieces of one segment
    pcs["geometry"] = shapely.line_merge(pcs.geometry.values)
    pcs = pcs.explode(index_parts=False).reset_index(drop=True)
    pcs["km"] = pcs.km.round(2)
    n, mb = dump(os.path.join(OUT, "slip.json"), pcs, {"km": "km", "w1": "w1", "w2": "w2", "w3": "w3"})
    print(f"slip stretches: {n}, {mb:.1f} MB")

    # --- flood-flagged stretches and river crossings (14_) ----------------------------------------
    fpp = os.path.join(ROOT, "data", "processed", "roads", f"eth_flood_pieces_{tag}.parquet")
    fxp = os.path.join(ROOT, "data", "processed", "roads", f"eth_flood_xings_{tag}.parquet")
    if os.path.exists(fpp) and os.path.exists(fxp):
        fp = gpd.read_parquet(fpp)
        for w_ in ("w1", "w2", "w3"): fp[w_] = fp[w_].astype(int)
        if len(fp):
            fp = fp.dissolve(by=["seg", "w1", "w2", "w3"], aggfunc={"km": "sum"}).reset_index()
            fp["geometry"] = shapely.line_merge(fp.geometry.values)
            fp = fp.explode(index_parts=False).reset_index(drop=True); fp["km"] = fp.km.round(2)
        n, mb = dump(os.path.join(OUT, "flood.json"), fp, {"km": "km", "w1": "w1", "w2": "w2", "w3": "w3"})
        X = gpd.read_parquet(fxp); X = X[X[["w1", "w2", "w3"]].any(axis=1)].reset_index(drop=True)
        SRC = ["osm", "merit"]
        clean = lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
        ST = ["ford", "none", "culvert", "bridge", "assumed bridge"]; BY = ["", "river", "flash"]
        pc = lambda v: -1 if v is None or not np.isfinite(v) else int(round(100 * float(v)))
        cols = ["st", "src", "up", "w1", "w2", "w3", "lv1", "lv2", "lv3", "by1", "by2", "by3"] + \
               [f"p{rp}_{w_}" for w_ in (1, 2, 3) for rp in (2, 5, 20)] + ["rd", "wn"]
        rows = []
        for r in X.itertuples():
            d_ = r._asdict()
            rows.append([ST.index(r.structure) if r.structure in ST else 1, SRC.index(r.source), int(round(float(r.upa_km2))),
                         int(r.w1), int(r.w2), int(r.w3), int(r.lv1), int(r.lv2), int(r.lv3),
                         BY.index(r.by1), BY.index(r.by2), BY.index(r.by3)]
                        + [pc(d_.get(f"p{rp}_w{w_}")) for w_ in (1, 2, 3) for rp in (2, 5, 20)]
                        + [clean(r.road), clean(r.waterway_name), int(round(r.geometry.x * 1e4)), int(round(r.geometry.y * 1e4))])
        with open(os.path.join(OUT, "xings.json"), "w", encoding="utf-8") as f_:
            json.dump({"cols": cols, "lut": {"st": ST, "src": SRC, "by1": BY, "by2": BY, "by3": BY}, "f": rows,
                       "glofas": str(X.glofas_date.iloc[0]) if "glofas_date" in X.columns and len(X) else None},
                      f_, separators=(",", ":"), ensure_ascii=False)
        print(f"flood stretches: {n}, {mb:.1f} MB; crossings flagged: {len(rows)}")
    else:
        for f_ in ("flood.json", "xings.json"):
            with open(os.path.join(OUT, f_), "w") as fh: json.dump({"cols": [], "lut": {}, "f": []}, fh)

    # --- outside-Ethiopia mask -------------------------------------------------------------------
    import importlib.util as _iu
    eth = shapely.union_all([gm.buffer(0) for gm in rr.wn.GEOMS]).buffer(0.002).simplify(0.002)
    mask = shapely.box(20, -5, 60, 25).difference(eth)
    gpd.GeoDataFrame(geometry=[mask], crs="EPSG:4326").to_file(os.path.join(OUT, "mask.geojson"), driver="GeoJSON", engine="pyogrio")

    # --- rain cells ----------------------------------------------------------------------
    mask = z["mask"]
    H, fld = rr.hazard(z)
    wn = rr.wn
    WH = rr.window_hazards(z, init, fld["daily_h"])
    import json as _json
    _th = _json.load(open(rr.INTJ))["thresholds"] if os.path.exists(rr.INTJ) else None
    _hp = os.path.join(ROOT, "data", "raw", "weathernext3", "hourly", f"wn3h_eth_{init:%Y%m%d%H}.npz")
    HOURLY = None
    if os.path.exists(_hp):
        _z = np.load(_hp); HOURLY = {"mean": _z["mean"].astype("float32") / 10, "p90": _z["p90"].astype("float32") / 10}
        _band_key = [next(f"days_{a}_{b}" for a, b in rr.WINDOWS if a <= h // 24 + 1 <= b) for h in range(360)]
    _P, _ = wn.load_pardig(); _have = set(pd.to_datetime(_P.time.values))
    OBS10 = np.stack([np.nan_to_num(_P.sel(time=init - pd.Timedelta(days=k)).values) if (init - pd.Timedelta(days=k)) in _have
                      else np.full(mask.shape if hasattr(mask, "shape") else z["mask"].shape, np.nan) for k in range(10, 0, -1)])
    PN = None
    try:
        _s = importlib.util.spec_from_file_location("pn", os.path.join(ROOT, "scripts", "33_percent_of_normal.py"))
        _pn = importlib.util.module_from_spec(_s); _s.loader.exec_module(_pn)
        PN = _pn.fields(init, fld["daily"], OBS10, mask.shape)
    except SystemExit as e_:
        print("no % of normal:", e_)

    def pv(k, i, j, f="pct"):
        """% of normal (int, -1 = not compared) or tercile (-1/0/1, 9 = not compared)."""
        v = PN[k][f][i, j] if PN else np.nan
        if not np.isfinite(v): return -1 if f == "pct" else 9
        return int(round(float(v))) if f == "pct" else int(v)
    gx0, gy0, gres = float(z["x0"]), float(z["y0"]), float(z["res"])
    feats = []
    for i in range(int(z["ny"])):
        for j in range(int(z["nx"])):
            if not mask[i, j]: continue
            x0, y1 = round(gx0 + j * gres, 4), round(gy0 - i * gres, 4)
            x1, y0 = round(x0 + gres, 4), round(y1 - gres, 4)
            feats.append({"type": "Feature", "properties": {
                "m": round(float(fld["mean15"][i, j]), 1), "r3": round(float(fld["r3"][i, j]), 1),
                "p20": round(float(fld["p90max"][i, j]), 1), "h": int(H[i, j]),
                "a1": int(WH["A"][0][i, j]), "a2": int(WH["A"][1][i, j]), "a3": int(WH["A"][2][i, j]),
                "i1": int(WH["I"][0][i, j]), "i2": int(WH["I"][1][i, j]), "i3": int(WH["I"][2][i, j]),
                "pn": pv("all", i, j), "pn1": pv("w1", i, j), "pn2": pv("w2", i, j), "pn3": pv("w3", i, j),
                "po": pv("obs", i, j), "pt": pv("all", i, j, "ter")},
                "geometry": {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}})
    json.dump({"type": "FeatureCollection", "features": feats}, open(os.path.join(OUT, "rain.geojson"), "w"),
              separators=(",", ":"))

    # --- region outlines -------------------------------------------------------------------
    adm = gpd.GeoDataFrame({"name": wn.NAMES}, geometry=wn.GEOMS, crs="EPSG:4326")
    adm["geometry"] = shapely.set_precision(shapely.simplify(adm.geometry.values, 0.003), 1e-4)
    adm.to_file(os.path.join(OUT, "adm1.geojson"), driver="GeoJSON", engine="pyogrio")

    # --- hillshade -------------------------------------------------------------------------
    with rasterio.open(os.path.join(ROOT, "data", "raw", "dem", "gem15_eth_0p001.tif")) as d:
        e = d.read(1, out_shape=(d.height // 10, d.width // 10), resampling=rasterio.enums.Resampling.average).astype("float32")
        tr = d.transform * d.transform.scale(10, 10); bounds = d.bounds
    e[e <= 0] = np.nan
    lat = bounds.top - (np.arange(e.shape[0]) + 0.5) * 0.01
    dy, dx = np.gradient(e, 1110.0, 1110.0 * np.cos(np.deg2rad(lat))[:, None].mean())
    dy = -dy                                                               # rows run north -> south
    slp = np.arctan(2.0 * np.hypot(dx, dy)); asp = np.arctan2(-dx, dy)
    az, alt = np.deg2rad(315), np.deg2rad(45)
    hs = np.sin(alt) * np.cos(slp) + np.cos(alt) * np.sin(slp) * np.cos(az - asp)
    inside = rasterize([(gm, 1) for gm in wn.GEOMS], out_shape=e.shape, transform=tr, fill=0) == 1
    # dark-on-transparent shading: shadows carry alpha, lit slopes fade out (works on light and dark grounds)
    shade = np.clip((0.85 - np.nan_to_num(hs, nan=0.85)) / 0.85, 0, 1)
    alpha = (shade * 200 * inside).astype("uint8")
    rgba = np.zeros(e.shape + (4,), "uint8"); rgba[..., 3] = alpha
    Image.fromarray(rgba, "RGBA").save(os.path.join(OUT, "hillshade.png"), optimize=True)
    print("hillshade", e.shape, os.path.getsize(os.path.join(OUT, "hillshade.png")) / 1e6, "MB",
          "bounds", [bounds.left, bounds.bottom, bounds.right, bounds.top])

    # --- health facilities (from 17_health_access.py) ------------------------------------------
    hfa = pd.read_csv(os.path.join(ROOT, "data", "processed", "tables", f"eth_health_access_{tag}.csv"))
    hfa["Name"] = hfa.Name.str.strip()
    FRF = os.path.join(ROOT, "data", "processed", "tables", f"eth_health_access_friction_{tag}.csv")
    HAS_FR = os.path.exists(FRF)
    WINS = (1, 2, 3)                                     # forecast windows: days 1-3, 4-7, 8-15
    if HAS_FR:
        fr = pd.read_csv(FRF)
        fcols = ["fr_to_hosp_min", "fr_to_hosp_fc_min", "fr_to_hosp_added_min", "fr_to_hc_min", "fr_to_hc_fc_min",
                 "fr_to_hc_added_min", "fr_nearest_hospital"] + [f"fr_to_{t}_added_w{w}_min" for t in ("hosp", "hc") for w in WINS
                 if f"fr_to_{t}_added_w{w}_min" in fr.columns] + (["fr_nearest_hospital_id"] if "fr_nearest_hospital_id" in fr.columns else [])
        hfa = hfa.merge(fr[["Id"] + fcols], on="Id", how="left")
        hfa["fr_nearest_hospital"] = hfa.fr_nearest_hospital.str.strip()
        lk = hfa[hfa.Type.isin(["Health Center", "Health Post"]) & ~hfa.off_network]
        frl = lk.groupby("fr_nearest_hospital").agg(fcf=("Id", "size"), fc30=("fr_to_hosp_added_min", lambda s_: int((s_ >= 30).sum())),
                                                    fcm=("fr_to_hosp_added_min", "median"),
                                                    **{f"fc30_{w}": (f"fr_to_hosp_added_w{w}_min", lambda s_: int((s_ >= 30).sum()))
                                                       for w in WINS if f"fr_to_hosp_added_w{w}_min" in lk.columns})
        hfa = hfa.merge(frl, left_on="Name", right_index=True, how="left")
        hfa.loc[hfa.Type != "Hospital", [c for c in hfa.columns if c in ("fcf", "fc30", "fcm") or c.startswith("fc30_")]] = np.nan
        print(f"friction results merged for {hfa.fr_to_hosp_min.notna().sum():,} facilities")
    else:
        print("no friction results for this run (20_friction_access.py) - web map will offer the road model only")
    TY = {"Hospital": 2, "Health Center": 1, "Health Post": 0}
    hfeats = []
    for r in hfa.itertuples(index=False):
        num = lambda v, nd=0: None if pd.isna(v) else (round(float(v), nd) if nd else int(round(float(v))))
        p = {"n": r.Name, "ty": TY[r.Type], "a1": r.admin1Name, "a2": r.admin2Name, "a3": r.admin3Name,
             "h0": num(r.to_hosp_min), "h1": num(r.to_hosp_fc_min), "c0": num(r.to_hc_min), "c1": num(r.to_hc_fc_min),
             "d": num(r.to_hosp_added_min), "dc": num(r.to_hc_added_min) if r.Type == "Health Post" else None, "ch": r.nearest_hospital if isinstance(r.nearest_hospital, str) else "",
             "hc": r.nearest_health_centre if (r.Type == "Health Post" and isinstance(r.nearest_health_centre, str)) else "",
             "nt": r.nearest_road_tier, "s5": num(r.share_elev_severe_5km, 2), "sl": int(bool(r.slip_within_5km)),
             "gap": num(r.road_gap_km, 1), "off": int(bool(r.off_network))}
        p["id"] = int(r.Id)
        # added minutes per forecast window (days 1-3, 4-7, 8-15); d / dc = all 15 days (worst window)
        rd = r._asdict()
        for w in WINS:
            p[f"d{w}"] = num(rd.get(f"to_hosp_added_w{w}_min"))
            if r.Type == "Health Post": p[f"dc{w}"] = num(rd.get(f"to_hc_added_w{w}_min"))
            if r.Type == "Hospital":
                p[f"c30_{w}"] = num(rd.get(f"linked_30min_plus_w{w}")) or 0; p[f"cm_{w}"] = num(rd.get(f"linked_median_added_w{w}_min"))
            if HAS_FR:
                p[f"fd{w}"] = num(rd.get(f"fr_to_hosp_added_w{w}_min"))
                if r.Type == "Health Post": p[f"fdc{w}"] = num(rd.get(f"fr_to_hc_added_w{w}_min"))
                if r.Type == "Hospital": p[f"fc30_{w}"] = num(rd.get(f"fc30_{w}")) or 0
        if isinstance(getattr(r, "nearest_hospital_id", None), (int, float)) and not pd.isna(r.nearest_hospital_id):
            p["hid"] = int(r.nearest_hospital_id)
        if r.Type == "Hospital":
            p |= {"cf": num(r.linked_facilities), "c30": num(r.linked_30min_plus), "cm": num(r.linked_median_added_min)}
            # local rain series at the hospital's 0.1 deg cell: 10 observed days, 15 forecast days
            ci = int(np.clip((float(z["y0"]) - r.Latitude) / float(z["res"]), 0, int(z["ny"]) - 1))
            cj = int(np.clip((r.Longitude - float(z["x0"])) / float(z["res"]), 0, int(z["nx"]) - 1))
            fxd = WH["fxmax"][:, ci, cj]
            ints = [0] * 15
            if _th:
                for d_ in range(15):
                    key = next(f"days_{a}_{b}" for a, b in rr.WINDOWS if a <= d_ + 1 <= b)
                    ints[d_] = 2 if fxd[d_] >= _th[key]["very_intense"] else 1 if fxd[d_] >= _th[key]["intense"] else 0
            if HOURLY is not None:
                hm, hw = HOURLY["mean"][:360, ci, cj], HOURLY["p90"][:360, ci, cj]
                i1 = [h for h in range(360) if _th and hw[h] >= _th[_band_key[h]]["intense"]]
                i2 = [h for h in range(360) if _th and hw[h] >= _th[_band_key[h]]["very_intense"]]
                pk = int(np.argmax(hw))
                p["hs"] = {"t0": f"{init:%Y-%m-%dT%H}", "m": [round(float(v), 1) for v in hm.reshape(120, 3).sum(1)],
                           "w": [round(float(v), 1) for v in hw.reshape(120, 3).sum(1)], "i1": i1, "i2": i2,
                           "pk": pk, "pkv": round(float(hw[pk]), 1), "obs10": round(float(np.nansum(OBS10[:, ci, cj])), 1),
                           "pn": pv("all", ci, cj), "po": pv("obs", ci, cj)}
            p["rs"] = {"o": [None if np.isnan(v) else round(float(v), 1) for v in OBS10[:, ci, cj]],
                       "f": [round(float(v), 1) for v in fld["daily"][:, ci, cj]],
                       "w": [round(float(v), 1) for v in np.nan_to_num(z["p90"])[:, ci, cj]],
                       "i": ints, "t0": f"{init:%Y-%m-%d}"}
        if HAS_FR:
            if "fr_nearest_hospital_id" in hfa.columns and not pd.isna(r.fr_nearest_hospital_id):
                p["fhid"] = int(r.fr_nearest_hospital_id)
            p |= {"fh0": num(r.fr_to_hosp_min), "fh1": num(r.fr_to_hosp_fc_min), "fc0": num(r.fr_to_hc_min),
                  "fc1": num(r.fr_to_hc_fc_min), "fd": num(r.fr_to_hosp_added_min), "fdc": num(r.fr_to_hc_added_min) if r.Type == "Health Post" else None,
                  "fch": r.fr_nearest_hospital if isinstance(r.fr_nearest_hospital, str) else ""}
            if r.Type == "Hospital":
                p |= {"fcf": num(r.fcf) or 0, "fc30": num(r.fc30) or 0, "fcm": num(r.fcm)}
            if r.Type == "Hospital": p["fd"] = 0
        hfeats.append({"type": "Feature", "properties": p,
                       "geometry": {"type": "Point", "coordinates": [round(r.Longitude, 5), round(r.Latitude, 5)]}})
    with open(os.path.join(OUT, "facilities.geojson"), "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": hfeats}, fh, separators=(",", ":"), ensure_ascii=False)
    print(f"facilities: {len(hfeats)}, {os.path.getsize(os.path.join(OUT, 'facilities.geojson')) / 1e6:.1f} MB")
    on = ~hfa.off_network
    top_hc = hfa[on & (hfa.Type == "Health Center")].nlargest(10, "to_hosp_added_min")
    top_h = hfa[hfa.Type == "Hospital"].sort_values(["linked_30min_plus", "linked_median_added_min"], ascending=False).head(10)
    hc_ = hfa[on & (hfa.Type == "Health Center")]; hp_ = hfa[on & (hfa.Type == "Health Post")]
    health = {"hc_n": len(hc_), "hc_30": int((hc_.to_hosp_added_min >= 30).sum()), "hc_60": int((hc_.to_hosp_added_min >= 60).sum()),
              "hp_n": len(hp_), "hp_30": int((hp_.to_hc_added_min >= 30).sum()), "hp_h30": int((hp_.to_hosp_added_min >= 30).sum()),
              "top_hc": [{"n": r.Name, "a2": r.admin2Name, "h0": round(r.to_hosp_min), "h1": round(r.to_hosp_fc_min),
                          "x": r.Longitude, "y": r.Latitude} for r in top_hc.itertuples()],
              "top_h": [{"n": r.Name, "a2": r.admin2Name, "cf": int(r.linked_facilities),
                         "c30": int(r.linked_30min_plus), "x": r.Longitude, "y": r.Latitude} for r in top_h.itertuples()]}
    for w in WINS:                                       # the same block per forecast window (road model)
        k, kc = f"to_hosp_added_w{w}_min", f"to_hc_added_w{w}_min"
        tw = hfa[on & (hfa.Type == "Health Center")].nlargest(10, k)
        thw = hfa[hfa.Type == "Hospital"].sort_values([f"linked_30min_plus_w{w}", f"linked_median_added_w{w}_min"], ascending=False).head(10)
        health.setdefault("win", []).append({
            "hc_30": int((hc_[k] >= 30).sum()), "hc_60": int((hc_[k] >= 60).sum()),
            "hp_30": int((hp_[kc] >= 30).sum()), "hp_h30": int((hp_[k] >= 30).sum()),
            "top_hc": [{"n": r.Name, "a2": r.admin2Name, "h0": round(r.to_hosp_min), "h1": round(r.to_hosp_min + getattr(r, k)),
                        "x": r.Longitude, "y": r.Latitude} for r in tw.itertuples()],
            "top_h": [{"n": r.Name, "a2": r.admin2Name, "cf": int(r.linked_facilities), "c30": int(getattr(r, f"linked_30min_plus_w{w}")),
                       "x": r.Longitude, "y": r.Latitude} for r in thw.itertuples()]})
    if HAS_FR:
        hcf = hfa[on & (hfa.Type == "Health Center")].dropna(subset=["fr_to_hosp_added_min"])
        hpf = hfa[on & (hfa.Type == "Health Post")].dropna(subset=["fr_to_hc_added_min"])
        tf = hcf.nlargest(10, "fr_to_hosp_added_min")
        th = hfa[hfa.Type == "Hospital"].sort_values(["fc30", "fcm"], ascending=False).head(10)
        health["fr"] = {"hc_n": len(hcf), "hc_30": int((hcf.fr_to_hosp_added_min >= 30).sum()),
                        "hc_60": int((hcf.fr_to_hosp_added_min >= 60).sum()), "hp_n": len(hpf),
                        "hp_30": int((hpf.fr_to_hc_added_min >= 30).sum()), "hp_h30": int((hpf.fr_to_hosp_added_min >= 30).sum()),
                        "top_hc": [{"n": r.Name, "a2": r.admin2Name, "h0": round(r.fr_to_hosp_min), "h1": round(r.fr_to_hosp_fc_min),
                                    "x": r.Longitude, "y": r.Latitude} for r in tf.itertuples()],
                        "top_h": [{"n": r.Name, "a2": r.admin2Name, "cf": int(r.fcf or 0), "c30": int(r.fc30 or 0),
                                   "x": r.Longitude, "y": r.Latitude} for r in th.itertuples()],
                        "hc_median_normal": round(float(hcf.fr_to_hosp_min.median()), 1),
                        "road_hc_median_normal": round(float(hc_.to_hosp_min.median()), 1)}
        for w in WINS:
            k = f"fr_to_hosp_added_w{w}_min"
            if k not in hfa.columns: break
            hcw = hcf.dropna(subset=[k]); kc = f"fr_to_hc_added_w{w}_min"
            tfw = hcw.nlargest(10, k); thw = hfa[hfa.Type == "Hospital"].sort_values([f"fc30_{w}", "fcm"], ascending=False).head(10)
            health["fr"].setdefault("win", []).append({
                "hc_30": int((hcw[k] >= 30).sum()), "hc_60": int((hcw[k] >= 60).sum()),
                "hp_30": int((hpf[kc] >= 30).sum()), "hp_h30": int((hpf[f"fr_to_hosp_added_w{w}_min"] >= 30).sum()),
                "top_hc": [{"n": r.Name, "a2": r.admin2Name, "h0": round(r.fr_to_hosp_min), "h1": round(r.fr_to_hosp_min + getattr(r, k)),
                            "x": r.Longitude, "y": r.Latitude} for r in tfw.itertuples()],
                "top_h": [{"n": r.Name, "a2": r.admin2Name, "cf": int(r.fcf or 0), "c30": int(getattr(r, f"fc30_{w}") or 0),
                           "x": r.Longitude, "y": r.Latitude} for r in thw.itertuples()]})
        # friction speed image (km/h classes), transparent outside Ethiopia
        FRT = os.path.join(ROOT, "data", "raw", "friction", "map_friction_motorized_2019_v5_1_eth.tif")
        with rasterio.open(FRT) as d_:
            Fr = d_.read(1).astype("float32"); ftr = d_.transform; fb = d_.bounds
        spd = 60.0 / (Fr * 1000.0)
        inside_ = rasterize([(gm, 1) for gm in rr.wn.GEOMS], out_shape=Fr.shape, transform=ftr, fill=0) == 1
        # same palette as the road slowdown tiers: slower travel = "worse" tier colour
        SPC = [(0, 3, (168, 50, 15)),       # Severe   #a8320f   < 3 km/h
               (3, 10, (224, 112, 31)),     # Elevated #e0701f   3-10 km/h
               (10, 30, (242, 179, 61)),    # Minor    #f2b33d   10-30 km/h
               (30, 1e9, (201, 200, 194))]  # Normal   #c9c8c2   >= 30 km/h
        img = np.zeros(Fr.shape + (4,), "uint8")
        for lo_, hi_, rgb in SPC:
            k_ = inside_ & np.isfinite(spd) & (spd >= lo_) & (spd < hi_)
            img[k_, :3] = rgb; img[k_, 3] = 235
        Image.fromarray(img, "RGBA").save(os.path.join(OUT, "friction.png"), optimize=True)
        health["fr"]["bounds"] = [fb.left, fb.bottom, fb.right, fb.top]
        print(f"friction.png {os.path.getsize(os.path.join(OUT, 'friction.png')) / 1e6:.1f} MB")

    # --- summary ---------------------------------------------------------------------------
    tab = pd.read_csv(os.path.join(ROOT, "data", "processed", "tables", f"eth_roads_rain_risk_adm1_{tag}.csv"), index_col=0)
    tab = tab[tab.index != "Unassigned"]
    reg_bbox = {n: [round(v, 3) for v in gm.bounds] for n, gm in zip(wn.NAMES, wn.GEOMS)}
    _lat = float(z["y0"]) - float(z["res"]) * (np.arange(int(z["ny"])) + 0.5)
    _w = np.cos(np.deg2rad(_lat))[:, None] * np.ones(z["mask"].shape)
    if PN:
        _ter = PN["all"]["ter"]; _tok = z["mask"] & np.isfinite(_ter)
    win = []
    for w_, (a_, b_) in enumerate(rr.WINDOWS, start=1):
        kmw = g.groupby(f"tier_w{w_}").km.sum()
        win.append({"label": f"Days {a_}–{b_}", "dates": f"{days[a_ - 1]:%d %b} – {days[b_ - 1]:%d %b}",
                    **{t: round(float(kmw.get(t, 0))) for t in TIERS[1:]},
                    "slip": round(float(tab[f"slide_km_w{w_}"].sum())) if f"slide_km_w{w_}" in tab.columns else None,
                    "flood_km": round(float(tab[f"flood_km_w{w_}"].sum())) if f"flood_km_w{w_}" in tab.columns else None,
                    "flood_x": int(tab[f"flood_xings_w{w_}"].sum()) if f"flood_xings_w{w_}" in tab.columns else None,
                    "intense_share": round(float(np.mean(WH["I"][w_ - 1][z["mask"]] >= 1)), 3)})
    summ = {"init": f"{init:%d %b %Y}", "start": f"{days[0]:%d %b}", "end": f"{days[-1]:%d %b %Y}",
            "total_km": round(float(tab[TIERS].sum().sum())),
            "tiers": {t: round(float(tab[t].sum())) for t in TIERS},
            "slide_km": round(float(tab.slide_exposure_km.sum())),
            "flood_km": round(float(tab.flood_km.sum())) if "flood_km" in tab.columns else None,
            "flood_x": int(tab.flood_xings.sum()) if "flood_xings" in tab.columns else None,
            "raised_km": round(float(tab.raised_by_terrain_km.sum())),
            "regions": [{"name": n, **{t: round(float(tab.loc[n, t])) for t in TIERS},
                         "share": float(tab.loc[n, "share_slowed_%"]), "slide": round(float(tab.loc[n, "slide_exposure_km"])),
                         "bbox": reg_bbox.get(n)} for n in tab.sort_values("share_slowed_%", ascending=False).index],
            "hillshade_bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top], "health": health, "windows": win,
            "normal": None if not PN else {
                "ratio_window": PN["ratio_window"],
                **{k: round(float(100 * (np.nan_to_num(PN[k]["obs" if k == "obs" else "fc"])[z["mask"]] * _w[z["mask"]]).sum()
                                  / max((PN[k]["normal"][z["mask"]] * _w[z["mask"]]).sum(), 1e-6)))
                   for k in ("obs", "w1", "w2", "w3", "all")},
                **{lab: round(float(100 * (_w[_tok] * (_ter[_tok] == v)).sum() / _w[_tok].sum()))
                   for lab, v in (("above", 1), ("below", -1))}}}
    if _th: summ["int_th"] = _th                    # forecast wettest-hour thresholds (mm) per window, for the page's labels
    json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(json.dumps({k: summ[k] for k in ["init", "tiers", "slide_km", "raised_km"]}))
