"""
38_flood_exposure.py -- static flood exposure of the road network: low-lying stretches beside rivers, and river
crossings without a bridge. 14_ combines it with each forecast window's rain to flag roads that could flood.

Terrain and water, on the same ~110 m (0.001 deg) grid as the GEM-15 terrain (Earth Engine, 1 deg tiles, cached):
  hnd_m        MERIT Hydro height above the nearest drainage (HAND), lowest in the cell, m
  upa_near     largest MERIT Hydro upstream catchment within 1 km, km2 (how big a river is nearby)
  upa_cell     largest MERIT Hydro upstream catchment in the cell, km2 (finds river channels the road crosses)
  gsw_occ      JRC Global Surface Water occurrence 1984-2021, highest in the cell, % of time under water
  -> data/raw/flood/flood_terrain_eth_0p001.tif

River crossings (-> data/processed/roads/eth_flood_crossings.parquet, one point per crossing):
  osm     a vehicle road meets an OSM river, stream, canal or drain
  merit   a road meets a MERIT channel with >= 50 km2 upstream where OSM maps no waterway within 300 m (and leaves
          the channel again within ~300 m: longer runs are a road beside the river, not across it)
  structure   bridge (the road way is tagged bridge), culvert (the waterway is tagged tunnel=culvert), ford (the road
              way is tagged ford, or a ford node lies within 30 m), assumed bridge (nothing mapped, but the road is trunk,
              primary or secondary), or none mapped
  upa_km2     catchment at the crossing (largest MERIT upstream area within ~1 cell)
  exposed     ford on >= 1 km2, none mapped on >= 10 km2, or culvert on >= 100 km2 (bridges are not flagged)
Thresholds are first guesses, not calibrated against observed road floods.

Low-lying exposure is judged by 14_ on its ~110 m samples along each road (LOWLAND rule below, imported there).

    python scripts\\38_flood_exposure.py [--refresh]
"""
import argparse, glob, io, json, os, re, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, geopandas as gpd, shapely, rasterio, pyogrio
from rasterio.transform import from_origin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "flood"); TILES = os.path.join(RAW, "tiles")
TIF = os.path.join(RAW, "flood_terrain_eth_0p001.tif")
PROC = os.path.join(ROOT, "data", "processed", "roads"); CROSS = os.path.join(PROC, "eth_flood_crossings.parquet")
ROADS = os.path.join(ROOT, "data", "raw", "roads", "eth_vehicle_roads_osm.parquet")
OSMD = os.path.join(ROOT, "data", "raw", "osm")
X0, X1, Y0, Y1, RES, TS = 33, 48, 3, 15, 0.001, 1
N = int(round(TS / RES))
BANDS = ["hnd_m", "upa_near", "upa_cell", "gsw_occ"]

# rules (14_ imports these)
LOW_HAND_M, NEAR_UPA_KM2, GSW_PCT, GSW_HAND_M = 3.0, 100.0, 10.0, 5.0
CH_UPA_KM2 = 50.0                                   # MERIT channel big enough to count as a crossing
EXPOSE = {"ford": 1.0, "none": 10.0, "culvert": 100.0}
WATERWAYS = {"river", "stream", "canal", "drain"}
MAIN = {"trunk", "primary", "secondary"}       # an unmapped crossing on these is assumed bridged (not flagged)


def lowland(hnd, upa_near, gsw):
    """Sample is low-lying flood ground: <= 3 m above a drainage with a >= 100 km2 river within 1 km, or ground
    under water >= 10% of the time (JRC) and <= 5 m above drainage."""
    return ((hnd <= LOW_HAND_M) & (upa_near >= NEAR_UPA_KM2)) | ((gsw >= GSW_PCT) & (hnd <= GSW_HAND_M))


_GRID = {}


def sample(x, y):
    """-> dict of band arrays at points (nearest cell): hnd_m (m; NaN where no data), upa_near, upa_cell (km2), gsw_occ (%).
    Stored as uint16: HAND in decimetres, catchment in km2 capped at 65,534, 65,535 = no data."""
    if not _GRID:
        with rasterio.open(TIF) as d:
            _GRID["a"], _GRID["tr"] = d.read(), d.transform
    a, tr = _GRID["a"], _GRID["tr"]
    c = np.clip(((np.asarray(x) - tr.c) / tr.a).astype(int), 0, a.shape[2] - 1)
    r = np.clip(((np.asarray(y) - tr.f) / tr.e).astype(int), 0, a.shape[1] - 1)
    v = {b: a[i, r, c] for i, b in enumerate(BANDS)}
    nd = v["hnd_m"] == 65535
    out = {"hnd_m": np.where(nd, np.nan, v["hnd_m"] / 10.0)}
    for b in ("upa_near", "upa_cell", "gsw_occ"):
        out[b] = np.where(v[b] == 65535, 0, v[b]).astype("float32")
    return out


def to_u16(a):
    """[4, y, x] float (hnd m, upa km2, upa km2, gsw %) -> uint16; -1 (no data) -> 65535."""
    o = np.empty(a.shape, "uint16")
    o[0] = np.where(a[0] < 0, 65535, np.clip(np.round(a[0] * 10), 0, 65534))
    for k in (1, 2, 3): o[k] = np.where(a[k] < 0, 65535, np.clip(np.ceil(a[k]), 0, 65534))
    return o


# --- 1. rasters from Earth Engine -------------------------------------------------------------------
def fetch_rasters():
    import ee
    from shapely.geometry import shape, box
    from shapely.ops import unary_union
    ee.Initialize(project=os.environ.get("EE_PROJECT"))
    os.makedirs(TILES, exist_ok=True)
    proj = ee.Projection("EPSG:4326", [RES, 0, 0, 0, -RES, 0])
    m = ee.Image("MERIT/Hydro/v1_0_1")
    hnd = m.select("hnd").reduceResolution(ee.Reducer.min(), maxPixels=64).reproject(proj)
    upa_near = m.select("upa").focalMax(1000, "circle", "meters").reduceResolution(ee.Reducer.max(), maxPixels=64).reproject(proj)
    upa_cell = m.select("upa").reduceResolution(ee.Reducer.max(), maxPixels=64).reproject(proj)
    gsw = (ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0)
           .reduceResolution(ee.Reducer.max(), maxPixels=256).reproject(proj))
    img = ee.Image.cat([hnd, upa_near, upa_cell, gsw]).rename(BANDS).toFloat().unmask(-1)
    import importlib.util
    spec = importlib.util.spec_from_file_location("wn", os.path.join(ROOT, "scripts", "11_weathernext_forecast.py"))
    wn = importlib.util.module_from_spec(spec); spec.loader.exec_module(wn)
    eth = unary_union(wn.GEOMS).buffer(0.05)
    jobs = [(i, j) for i in range(int((X1 - X0) / TS)) for j in range(int((Y1 - Y0) / TS))
            if eth.intersects(box(X0 + i * TS, Y1 - (j + 1) * TS, X0 + (i + 1) * TS, Y1 - j * TS))]

    def one(ij):
        i, j = ij; p = os.path.join(TILES, f"t_{i:02d}_{j:02d}.npy")
        if os.path.exists(p): return
        grid = {"dimensions": {"width": N, "height": N}, "crsCode": "EPSG:4326",
                "affineTransform": {"scaleX": RES, "shearX": 0, "translateX": X0 + i * TS,
                                    "shearY": 0, "scaleY": -RES, "translateY": Y1 - j * TS}}
        for k in range(5):
            try:
                raw = ee.data.computePixels({"expression": img, "fileFormat": "NPY", "grid": grid})
                a = np.load(io.BytesIO(raw)); np.save(p, to_u16(np.stack([a[b] for b in BANDS]))); return
            except Exception as e:
                if k == 4: raise
                time.sleep(10 * (k + 1))
    t0 = time.time()
    with ThreadPoolExecutor(6) as ex: list(ex.map(one, jobs))
    print(f"{len(jobs)} tiles in {time.time() - t0:.0f} s", flush=True)
    H, W = int((Y1 - Y0) / RES), int((X1 - X0) / RES)
    out = np.full((4, H, W), 65535, "uint16")
    for i, j in jobs:
        out[:, j * N:(j + 1) * N, i * N:(i + 1) * N] = np.load(os.path.join(TILES, f"t_{i:02d}_{j:02d}.npy"))
    with rasterio.open(TIF, "w", driver="GTiff", width=W, height=H, count=4, dtype="uint16", crs="EPSG:4326", nodata=65535,
                       transform=from_origin(X0, Y1, RES, RES), compress="deflate", tiled=True, predictor=2) as d:
        d.write(out)
        for k, b in enumerate(BANDS): d.set_band_description(k + 1, b)
    for i, j in jobs: os.remove(os.path.join(TILES, f"t_{i:02d}_{j:02d}.npy"))
    print(f"wrote {TIF}; tiles removed", flush=True)


# --- 2. crossings ---------------------------------------------------------------------------------------
TAG = re.compile(r'"([^"]+)"=>"((?:[^"\\]|\\.)*)"')
def tags(s): return dict(TAG.findall(s)) if isinstance(s, str) else {}


def crossings():
    pbf = sorted(glob.glob(os.path.join(OSMD, "ethiopia-*.osm.pbf")))[-1]
    R = gpd.read_parquet(ROADS)[["osm_id", "highway", "geometry"]]
    t = pyogrio.read_dataframe(pbf, layer="lines", columns=["osm_id", "highway", "other_tags"],
                               where="highway IS NOT NULL AND (other_tags LIKE '%\"bridge\"%' OR other_tags LIKE '%\"ford\"%')",
                               read_geometry=False)
    t = t[t.osm_id.notna()]; tg = [tags(s) for s in t.other_tags]
    oid = t.osm_id.astype(float).astype("int64").values
    BR = {i for i, d in zip(oid, tg) if d.get("bridge", "no") not in ("no",)}
    FW = {i for i, d in zip(oid, tg) if d.get("ford", "no") not in ("no",)}
    fn = pyogrio.read_dataframe(pbf, layer="points", columns=["osm_id", "other_tags"], where="other_tags LIKE '%\"ford\"%'")
    fn = fn[[tags(s).get("ford", "no") != "no" for s in fn.other_tags]]
    W = pyogrio.read_dataframe(pbf, layer="lines", columns=["osm_id", "name", "waterway", "other_tags"], where="waterway IS NOT NULL")
    W = W[W.waterway.isin(WATERWAYS)].reset_index(drop=True)
    W["culvert"] = [tags(s).get("tunnel") == "culvert" for s in W.other_tags]
    print(f"OSM: {len(W):,} waterways, {len(BR):,} bridge ways, {len(FW):,} ford ways, {len(fn):,} ford nodes", flush=True)

    # 2a. road x OSM waterway
    tree = shapely.STRtree(W.geometry.values)
    ri, wi = tree.query(R.geometry.values, predicate="intersects")
    P = shapely.intersection(R.geometry.values[ri], W.geometry.values[wi])
    rows = []
    for a, b, p in zip(ri, wi, P):
        for q in shapely.get_parts(p):
            if q.geom_type != "Point":
                q = shapely.centroid(q)
            rows.append((a, b, q))
    C = gpd.GeoDataFrame({"ri": [r[0] for r in rows], "wi": [r[1] for r in rows]}, geometry=[r[2] for r in rows], crs="EPSG:4326")
    C["source"] = "osm"; C["waterway"] = W.waterway.values[C.wi]; C["waterway_name"] = W.name.values[C.wi]
    C["culvert"] = W.culvert.values[C.wi]
    # a waterway crossing the same road way more than once within 50 m counts once
    C["k"] = (shapely.get_x(C.geometry.values) / 0.0005).round().astype(int).astype(str) + "_" + (shapely.get_y(C.geometry.values) / 0.0005).round().astype(int).astype(str)
    C = C.drop_duplicates(["ri", "k"]).drop(columns=["k"])
    print(f"  road x OSM waterway: {len(C):,} crossings", flush=True)

    # 2b. road x MERIT channel (>= 50 km2) away from any OSM crossing: densify roads at ~50 m, runs of channel cells
    g = R.geometry.values; deg = shapely.length(g)
    n = np.maximum(2, np.ceil(deg / 0.00045).astype(int) + 1)
    seg = np.repeat(np.arange(len(g)), n); start = np.repeat(np.cumsum(n) - n, n)
    frac = (np.arange(len(seg)) - start) / np.repeat(n - 1, n)
    pts = shapely.line_interpolate_point(g[seg], frac, normalized=True)
    x, y = shapely.get_x(pts), shapely.get_y(pts)
    up = sample(x, y)["upa_cell"]
    hit = up >= CH_UPA_KM2
    first = hit & np.r_[True, (seg[1:] != seg[:-1]) | ~hit[:-1]]
    last = hit & np.r_[(seg[1:] != seg[:-1]) | ~hit[1:], True]
    s0, s1 = np.flatnonzero(first), np.flatnonzero(last)
    mid = (s0 + s1) // 2
    M = gpd.GeoDataFrame({"ri": seg[mid]}, geometry=shapely.points(x[mid], y[mid]), crs="EPSG:4326")
    M = M[(s1 - s0) < 7]                                    # a run of > ~300 m is a road along the river, not across
    near = shapely.STRtree(C.geometry.values).query(M.geometry.values, predicate="dwithin", distance=0.0027)
    M = M[~np.isin(np.arange(len(M)), near[0])]
    M["source"] = "merit"; M["waterway"] = "MERIT channel"; M["waterway_name"] = None; M["culvert"] = False
    print(f"  road x MERIT channel (>= {CH_UPA_KM2:.0f} km2, no OSM waterway within 300 m): {len(M):,}", flush=True)
    X = pd.concat([C.drop(columns=["wi"]), M], ignore_index=True)
    X = gpd.GeoDataFrame(X, geometry="geometry", crs="EPSG:4326")

    # 2c. structure and catchment
    X["osm_id"] = R.osm_id.values[X.ri]; X["highway"] = R.highway.values[X.ri]
    fd = np.zeros(len(X), bool)
    if len(fn):
        q = shapely.STRtree(fn.geometry.values).query(X.geometry.values, predicate="dwithin", distance=0.00027)
        fd[np.unique(q[0])] = True
    main = X.highway.str.replace("_link", "").isin(MAIN).values
    X["structure"] = np.where(X.osm_id.isin(list(BR)), "bridge", np.where(X.culvert, "culvert",
                     np.where(X.osm_id.isin(list(FW)) | fd, "ford", np.where(main, "assumed bridge", "none"))))
    xx, yy = shapely.get_x(X.geometry.values), shapely.get_y(X.geometry.values)
    best = np.zeros(len(X), "float32")
    for dx in (-RES, 0, RES):
        for dy in (-RES, 0, RES):
            best = np.maximum(best, sample(xx + dx, yy + dy)["upa_cell"])
    X["upa_km2"] = best.round(1)
    X["exposed"] = [(s in EXPOSE) and (u >= EXPOSE[s]) for s, u in zip(X.structure, X.upa_km2)]
    X = X.drop(columns=["ri", "culvert"])
    X.to_parquet(CROSS)
    print(X.groupby(["source", "structure"]).agg(n=("exposed", "size"), exposed=("exposed", "sum")).to_string())
    print(f"wrote {CROSS}: {len(X):,} crossings, {int(X.exposed.sum()):,} exposed", flush=True)
    return X


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--refresh", action="store_true"); a = ap.parse_args()
    if a.refresh or not os.path.exists(TIF): fetch_rasters()
    X = crossings()
    # low-lying share of the network, for the record
    R = gpd.read_parquet(ROADS)[["osm_id", "geometry"]]
    g = R.geometry.values; n = np.maximum(2, np.ceil(shapely.length(g) / 0.001).astype(int) + 1)
    seg = np.repeat(np.arange(len(g)), n); start = np.repeat(np.cumsum(n) - n, n)
    pts = shapely.line_interpolate_point(g[seg], (np.arange(len(seg)) - start) / np.repeat(n - 1, n), normalized=True)
    s = sample(shapely.get_x(pts), shapely.get_y(pts))
    low = lowland(s["hnd_m"], s["upa_near"], s["gsw_occ"])
    print(f"low-lying samples: {low.mean():.1%} of road samples; segments with any: {np.bincount(seg, low, minlength=len(g)).astype(bool).sum():,}")
