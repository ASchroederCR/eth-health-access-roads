"""
16_export_webmap.py -- data files for the interactive road / rain web map (webmap/index.html).

Reads the outputs of 14_roads_rain_risk.py (scored roads GeoPackage), the WeatherNext 2 run it used,
GEM-15 elevation (15_fetch_gem15_dem.py) and the ADM1 boundaries, and writes compact web files to webmap/data/:
  roads_severe.json, roads_elevated.json, roads_minor.json   flagged segments, simplified ~30 m, compact
  roads_context.json       trunk / primary / secondary roads (all tiers) as a light base network
  rain.geojson             0.25 deg forecast cells: 15-day mean, wettest 3 days, P(day >= 20 mm), hazard
  adm1.geojson             region outlines
  slip.json                slip / rock-fall exposed road stretches (compact, as the road files)
  mask.geojson             everything outside Ethiopia, drawn white on the map
  facilities.geojson       hospitals, health centres, health posts with access results (17_health_access.py)
  hillshade.png            GEM-15 hillshade at 0.01 deg, transparent outside Ethiopia
  summary.json             run, totals by tier, by region, slip exposure

    python scripts\\16_export_webmap.py [--init 2026-11-17]
"""
import argparse, glob, json, os
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
    wraw = os.path.join(ROOT, "data", "raw", "weathernext")
    f = (os.path.join(wraw, f"wn2_eth_{pd.Timestamp(a.init):%Y%m%d}00.npz") if a.init else
         sorted(glob.glob(os.path.join(wraw, "wn2_eth_??????????.npz")))[-1])
    z = np.load(f); init = pd.Timestamp(str(z["init"])); tag = f"{init:%Y%m%d}"
    g = gpd.read_file(os.path.join(ROOT, "data", "processed", "roads", f"eth_roads_rain_risk_{tag}.gpkg"), engine="pyogrio")
    regions = sorted(g.region.unique())
    days = pd.date_range(init, periods=z["precip"].shape[1])

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
    props = {"t": "t", "tw": "tw", "hw": "highway", "nm": "nm", "sf": "sf", "sk": "sk", "cond": "bad", "km": "km",
             "tc": "terrain_class", "gr": "road_grade_pct", "sm": "terrain_slope_mean_deg", "sx": "terrain_slope_max_deg",
             "el": "elev_mean_m", "r15": "rain_15d_mm", "r3": "wettest_3d_mm", "p20": "p_day_ge20mm", "pk": "pk",
             "sl": "sl", "slk": "slide_km", "su": "landslide_susc_max", "rg": "region", "R": "rain_hazard",
             "S": "surface_score", "T": "steep_score"}
    for c_, nd in [("km", 2), ("road_grade_pct", 1), ("terrain_slope_mean_deg", 1), ("terrain_slope_max_deg", 1),
                   ("rain_15d_mm", 1), ("wettest_3d_mm", 1), ("p_day_ge20mm", 2)]:
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
    ctx = allr[allr.highway.isin(["motorway", "trunk", "primary", "secondary"])].copy()
    ctx["geometry"] = r4(ctx.geometry.values, 0.001)
    ctx["c"] = ctx.highway.map({"motorway": 0, "trunk": 0, "primary": 1, "secondary": 2})
    n, mb = dump(os.path.join(OUT, "roads_context.json"), ctx.reset_index(drop=True), {"c": "c"})
    print(f"context: {n} segments, {mb:.1f} MB")

    # --- slip / rock-fall exposed stretches (14_) ---------------------------------------------
    pcs = gpd.read_parquet(os.path.join(ROOT, "data", "processed", "roads", f"eth_slip_pieces_{tag}.parquet"))
    pcs = pcs.dissolve(by="seg", aggfunc={"km": "sum"}).reset_index()           # join pieces of one segment
    pcs["geometry"] = shapely.line_merge(pcs.geometry.values)
    pcs = pcs.explode(index_parts=False).reset_index(drop=True)
    pcs["km"] = pcs.km.round(2)
    n, mb = dump(os.path.join(OUT, "slip.json"), pcs, {"km": "km"})
    print(f"slip stretches: {n}, {mb:.1f} MB")

    # --- outside-Ethiopia mask -------------------------------------------------------------------
    import importlib.util as _iu
    eth = shapely.union_all([gm.buffer(0) for gm in rr.wn.GEOMS]).buffer(0.002).simplify(0.002)
    mask = shapely.box(20, -5, 60, 25).difference(eth)
    gpd.GeoDataFrame(geometry=[mask], crs="EPSG:4326").to_file(os.path.join(OUT, "mask.geojson"), driver="GeoJSON", engine="pyogrio")

    # --- rain cells ----------------------------------------------------------------------
    arr, mask = z["precip"], z["mask"]
    H, fld = rr.hazard(arr)
    wn = rr.wn
    feats = []
    for i in range(wn.NY):
        for j in range(wn.NX):
            if not mask[i, j]: continue
            x0, y1 = wn.X0 + j * wn.RES, wn.Y0 - i * wn.RES
            x1, y0 = x0 + wn.RES, y1 - wn.RES
            feats.append({"type": "Feature", "properties": {
                "m": round(float(fld["mean15"][i, j]), 1), "r3": round(float(fld["r3"][i, j]), 1),
                "p20": round(float(fld["p20"][i, j]), 2), "h": int(H[i, j])},
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
    TY = {"Hospital": 2, "Health Center": 1, "Health Post": 0}
    hfeats = []
    for r in hfa.itertuples(index=False):
        num = lambda v, nd=0: None if pd.isna(v) else (round(float(v), nd) if nd else int(round(float(v))))
        added = r.to_hc_added_min if r.Type == "Health Post" else r.to_hosp_added_min
        p = {"n": r.Name, "ty": TY[r.Type], "a1": r.admin1Name, "a2": r.admin2Name, "a3": r.admin3Name,
             "h0": num(r.to_hosp_min), "h1": num(r.to_hosp_fc_min), "c0": num(r.to_hc_min), "c1": num(r.to_hc_fc_min),
             "d": num(added), "ch": r.nearest_hospital if isinstance(r.nearest_hospital, str) else "",
             "hc": r.nearest_health_centre if (r.Type == "Health Post" and isinstance(r.nearest_health_centre, str)) else "",
             "nt": r.nearest_road_tier, "s5": num(r.share_elev_severe_5km, 2), "sl": int(bool(r.slip_within_5km)),
             "gap": num(r.road_gap_km, 1), "off": int(bool(r.off_network))}
        if r.Type == "Hospital":
            p |= {"cf": num(r.linked_facilities), "c30": num(r.linked_30min_plus), "cm": num(r.linked_median_added_min)}
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
              "hp_n": len(hp_), "hp_30": int((hp_.to_hc_added_min >= 30).sum()),
              "top_hc": [{"n": r.Name, "a2": r.admin2Name, "h0": round(r.to_hosp_min), "h1": round(r.to_hosp_fc_min),
                          "x": r.Longitude, "y": r.Latitude} for r in top_hc.itertuples()],
              "top_h": [{"n": r.Name, "a2": r.admin2Name, "cf": int(r.linked_facilities),
                         "c30": int(r.linked_30min_plus), "x": r.Longitude, "y": r.Latitude} for r in top_h.itertuples()]}

    # --- summary ---------------------------------------------------------------------------
    tab = pd.read_csv(os.path.join(ROOT, "data", "processed", "tables", f"eth_roads_rain_risk_adm1_{tag}.csv"), index_col=0)
    tab = tab[tab.index != "Unassigned"]
    reg_bbox = {n: [round(v, 3) for v in gm.bounds] for n, gm in zip(wn.NAMES, wn.GEOMS)}
    summ = {"init": f"{init:%d %b %Y}", "start": f"{days[0]:%d %b}", "end": f"{days[-1]:%d %b %Y}",
            "total_km": round(float(tab[TIERS].sum().sum())),
            "tiers": {t: round(float(tab[t].sum())) for t in TIERS},
            "slide_km": round(float(tab.slide_exposure_km.sum())),
            "raised_km": round(float(tab.raised_by_terrain_km.sum())),
            "regions": [{"name": n, **{t: round(float(tab.loc[n, t])) for t in TIERS},
                         "share": float(tab.loc[n, "share_slowed_%"]), "slide": round(float(tab.loc[n, "slide_exposure_km"])),
                         "bbox": reg_bbox.get(n)} for n in tab.sort_values("share_slowed_%", ascending=False).index],
            "hillshade_bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top], "health": health}
    json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(json.dumps({k: summ[k] for k in ["init", "tiers", "slide_km", "raised_km"]}))
