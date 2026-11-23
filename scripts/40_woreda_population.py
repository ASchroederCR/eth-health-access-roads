"""
40_woreda_population.py -- woreda (admin 3) boundaries with WorldPop population and children under 5, for the
hospital-network population profiles on the web map (16_).

Boundaries: OCHA common operational dataset (COD-AB) for Ethiopia on HDX (eth_admin_boundaries, 2025; CC BY-IGO).
Population: WorldPop 2020 age/sex structure, constrained to settled areas, 100 m (Earth Engine
WorldPop/GP/100m/pop_age_sex_cons_unadj): total = "population"; under 5 = M_0 + M_1 + F_0 + F_1 (ages 0 and 1-4).
Summed over each woreda on Earth Engine at 100 m.

Outputs (run once; static)
  data/raw/boundaries/eth_woredas.parquet                    woreda outlines (simplified ~50 m) with the numbers below
  data/processed/tables/eth_woreda_population_2020.csv       pcode, woreda, zone, region, area, population, under 5

    python scripts\\40_woreda_population.py
"""
import glob, io, os, zipfile
import numpy as np, pandas as pd, geopandas as gpd, requests, pyogrio, shapely

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BD = os.path.join(ROOT, "data", "raw", "boundaries", "cod_ab_eth"); os.makedirs(BD, exist_ok=True)
URL = ("https://data.humdata.org/dataset/cb58fa1f-687d-4cac-81a7-655ab1efb2d0/resource/22ce2861-a868-4bfd-b13a-d88c3642c38b/"
       "download/eth_admin_boundaries.gdb.zip")
OUT_G = os.path.join(ROOT, "data", "raw", "boundaries", "eth_woredas.parquet")
OUT_T = os.path.join(ROOT, "data", "processed", "tables", "eth_woreda_population_2020.csv")
WP = "WorldPop/GP/100m/pop_age_sex_cons_unadj"


def woredas():
    gdb = glob.glob(os.path.join(BD, "**", "*.gdb"), recursive=True)
    if not gdb:
        r = requests.get(URL, timeout=600, headers={"User-Agent": "boundary-download/1.0"}); r.raise_for_status()
        zipfile.ZipFile(io.BytesIO(r.content)).extractall(BD)
        gdb = glob.glob(os.path.join(BD, "**", "*.gdb"), recursive=True)
    layers = [l[0] for l in pyogrio.list_layers(gdb[0])]
    lay = next(l for l in layers if "adm3" in l.lower() or "admin3" in l.lower())
    g = gpd.read_file(gdb[0], layer=lay, engine="pyogrio").to_crs(4326)
    cols = {c.lower(): c for c in g.columns}
    pick = lambda *names: next(cols[n] for n in names if n in cols)
    out = gpd.GeoDataFrame({"pcode": g[pick("adm3_pcode", "admin3pcode")], "woreda": g[pick("adm3_en", "admin3name_en", "adm3_name")],
                            "zone": g[pick("adm2_en", "admin2name_en", "adm2_name")], "region": g[pick("adm1_en", "admin1name_en", "adm1_name")]},
                           geometry=shapely.make_valid(g.geometry.values), crs=4326)
    print(f"{lay}: {len(out)} woredas", flush=True)
    return out


def worldpop(g):
    import ee
    ee.Initialize(project=os.environ.get("EE_PROJECT"))
    img = ee.ImageCollection(WP).filter(ee.Filter.eq("country", "ETH")).first()
    img = img.select("population").addBands(img.select(["M_0", "M_1", "F_0", "F_1"]).reduce(ee.Reducer.sum()).rename("u5"))
    pop, u5 = np.zeros(len(g)), np.zeros(len(g))
    simp = shapely.simplify(g.geometry.values, 0.001)
    for k in range(0, len(g), 60):
        feats = [ee.Feature(ee.Geometry(shapely.geometry.mapping(simp[i])), {"i": int(i)}) for i in range(k, min(k + 60, len(g)))]
        res = img.reduceRegions(ee.FeatureCollection(feats), ee.Reducer.sum(), scale=100).getInfo()
        for f in res["features"]:
            i = f["properties"]["i"]; pop[i] = f["properties"].get("population") or 0; u5[i] = f["properties"].get("u5") or 0
        print(f"  {min(k + 60, len(g))}/{len(g)}", flush=True)
    return pop, u5


def grid_sums(polys, chunk=80):
    """WorldPop 2020 population and under 5 summed over the 100 m cells inside each polygon (Earth Engine).
    polys: list of shapely polygons (EPSG:4326). -> (pop, u5) float arrays."""
    import ee
    try: ee.Number(1).getInfo()
    except Exception: ee.Initialize(project=os.environ.get("EE_PROJECT"))
    img = ee.ImageCollection(WP).filter(ee.Filter.eq("country", "ETH")).first()
    img = img.select("population").addBands(img.select(["M_0", "M_1", "F_0", "F_1"]).reduce(ee.Reducer.sum()).rename("u5"))
    pop, u5 = np.zeros(len(polys)), np.zeros(len(polys))
    simp = shapely.simplify(np.asarray(polys, dtype=object), 0.0005)
    for k in range(0, len(polys), chunk):
        feats = [ee.Feature(ee.Geometry(shapely.geometry.mapping(simp[i])), {"i": int(i)}) for i in range(k, min(k + chunk, len(polys)))
                 if simp[i] is not None and not simp[i].is_empty]
        if not feats: continue
        res = img.reduceRegions(ee.FeatureCollection(feats), ee.Reducer.sum(), scale=100).getInfo()
        for f in res["features"]:
            i = f["properties"]["i"]; pop[i] = f["properties"].get("population") or 0; u5[i] = f["properties"].get("u5") or 0
        print(f"  grid sums {min(k + chunk, len(polys))}/{len(polys)}", flush=True)
    return pop, u5


if __name__ == "__main__":
    g = woredas()
    g["area_km2"] = g.to_crs(32637).area / 1e6
    g["population"], g["under5"] = worldpop(g)
    g["population"], g["under5"] = g.population.round(0), g.under5.round(0)
    t = g.drop(columns="geometry"); t["area_km2"] = t.area_km2.round(1)
    os.makedirs(os.path.dirname(OUT_T), exist_ok=True); t.to_csv(OUT_T, index=False)
    g["geometry"] = shapely.simplify(g.geometry.values, 0.0005)
    g.to_parquet(OUT_G)
    print(f"wrote {OUT_T} and {OUT_G}: {len(g)} woredas, population {g.population.sum():,.0f}, under 5 {g.under5.sum():,.0f} "
          f"({g.under5.sum() / g.population.sum():.1%})")
