"""
15_fetch_gem15_dem.py -- Google GEM-15 bare-earth terrain (dtm, 15 m) for Ethiopia via Earth Engine,
reduced to a 0.001 deg (~110 m) grid with two bands:
  elev_m     dtm elevation, m (Earth Engine pyramid / nearest resampling)
  slope_deg  mean of the 15 m terrain slope within each ~110 m cell, degrees x 4 stored as uint8 (0.25 deg steps)
Slope is computed per source tile in its own UTM projection at 15 m, then averaged to ~105 m
before mosaicking, so it keeps the steepness that a coarse DEM would smooth away.

Fetched in 0.25 x 0.25 deg tiles (larger ones hit the Earth Engine memory limit) over Ethiopia only (cached in data/raw/dem/tiles/) and written to
data/raw/dem/gem15_eth_0p001.tif (2 bands, EPSG:4326).

    python scripts\\15_fetch_gem15_dem.py
(no -I: earthengine-api lives in user site-packages)
"""
import io, os
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np, rasterio
from shapely.geometry import shape, box
from shapely.ops import unary_union
from rasterio.transform import from_origin
import ee

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "dem"); TILES = os.path.join(RAW, "tiles"); os.makedirs(TILES, exist_ok=True)
OUT = os.path.join(RAW, "gem15_eth_0p001.tif")
GEM = "projects/climate-and-sustainability/assets/gem-15_v1"
X0, X1, Y0, Y1, RES = 33, 48, 3, 15, 0.001      # whole degrees: Ethiopia is 33.0-48.0 E, 3.4-14.9 N
TS = 0.25                                        # tile size, deg
N = int(round(TS / RES))                         # pixels per tile

ee.Initialize(project=os.environ.get("EE_PROJECT"))


def per_tile(img):
    dtm = img.select("dtm")
    proj = dtm.projection()
    slope = (ee.Terrain.slope(dtm).reduceResolution(ee.Reducer.mean(), maxPixels=64)
             .reproject(proj.atScale(105)))
    return dtm.rename("elev_m").addBands(slope.rename("slope"))


def image():
    c = ee.ImageCollection(GEM).filterBounds(ee.Geometry.Rectangle([X0, Y0, X1, Y1])).map(per_tile)
    m = c.mosaic()
    return ee.Image.cat(m.select("elev_m").round().toInt16(),
                        m.select("slope").multiply(4).round().clamp(0, 255).toUint8().rename("slope_deg"))


IMG = None


def fetch(i, j):
    x, y = X0 + i * TS, Y0 + j * TS
    path = os.path.join(TILES, f"t_{i:03d}_{j:03d}.npz")
    if os.path.exists(path): return path
    grid = {"dimensions": {"width": N, "height": N}, "crsCode": "EPSG:4326",
            "affineTransform": {"scaleX": RES, "shearX": 0, "translateX": x, "shearY": 0, "scaleY": -RES, "translateY": y + TS}}
    for k in range(4):
        try:
            raw = ee.data.computePixels({"expression": IMG.unmask(0), "fileFormat": "NPY", "grid": grid})
            break
        except Exception as e:
            if k == 3: raise
            print(f"retry {i},{j}: {str(e)[:100]}", flush=True)
    a = np.load(io.BytesIO(raw)) if isinstance(raw, (bytes, bytearray)) else raw
    np.savez_compressed(path, elev=a["elev_m"].astype("int16"), slope=a["slope_deg"].astype("uint8"))
    return path


if __name__ == "__main__":
    IMG = image()
    bnd = json.load(open(os.path.join(ROOT, "data", "raw", "boundaries", "geoboundaries_adm1", "ETH_ADM1.geojson"), encoding="utf-8"))
    eth = unary_union([shape(f["geometry"]) for f in bnd["features"]]).buffer(0.02)
    NI, NJ = int((X1 - X0) / TS), int((Y1 - Y0) / TS)
    jobs = [(i, j) for i in range(NI) for j in range(NJ)
            if eth.intersects(box(X0 + i * TS, Y0 + j * TS, X0 + (i + 1) * TS, Y0 + (j + 1) * TS))]
    print(f"{len(jobs)} tiles", flush=True)
    done = [0]
    def run(j):
        fetch(*j); done[0] += 1
        if done[0] % 100 == 0: print(f"{done[0]} / {len(jobs)}", flush=True)
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(run, jobs))
    W, H = NI * N, NJ * N
    elev = np.zeros((H, W), "int16"); slope = np.zeros((H, W), "uint8")
    for i, j in jobs:
        t = np.load(os.path.join(TILES, f"t_{i:03d}_{j:03d}.npz"))
        r0, c0 = (NJ - 1 - j) * N, i * N
        elev[r0:r0 + N, c0:c0 + N] = t["elev"]; slope[r0:r0 + N, c0:c0 + N] = t["slope"]
    prof = dict(driver="GTiff", height=H, width=W, count=2, dtype="int16", crs="EPSG:4326",
                transform=from_origin(X0, Y1, RES, RES), compress="deflate", tiled=True, predictor=2)
    with rasterio.open(OUT, "w", **prof) as d:
        d.write(elev, 1); d.write(slope.astype("int16"), 2)
        d.set_band_description(1, "elev_m"); d.set_band_description(2, "slope_deg_x4")
    print(f"wrote {OUT}: {H} x {W}; elev {elev[elev > 0].min()}-{elev.max()} m; "
          f"slope mean {slope[elev > 0].mean() / 4:.1f} deg")
