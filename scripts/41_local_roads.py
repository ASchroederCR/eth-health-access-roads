"""
41_local_roads.py -- the local road network (everything below tertiary: unclassified, track, residential, service,
link roads) as 1-degree tiles for the web map's faint base layer, loaded only for the tiles in view once zoomed in.
The flagged roads (16_) are mostly on these classes; without them they read as disconnected pieces.

Same roads as the analysis (14_ load_roads, seismic cut-lines dropped), simplified ~30 m.
Static: rerun when the OSM extract changes.

Outputs
  webmap/data/local/index.json        {"cells": ["x_y", ...]}   x = floor(lon), y = floor(lat)
  webmap/data/local/<x>_<y>.json      compact lines (same layout as 16_ dump), property c: 0 unclassified / link,
                                      1 track, 2 residential / service

    python scripts\\41_local_roads.py
"""
import importlib.util, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "webmap", "data", "local")


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", file))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


if __name__ == "__main__":
    sys.argv = sys.argv[:1]
    ex, rr = load("ex", "16_export_webmap.py"), load("rr", "14_roads_rain_risk.py")
    allr, _ = rr.drop_seismic(rr.load_roads())
    loc = allr[~allr.highway.isin(["motorway", "trunk", "primary", "secondary", "tertiary"])].copy()
    loc["c"] = np.select([loc.highway == "track", loc.highway.isin(["residential", "service", "living_street"])], [1, 2], 0)
    loc["geometry"] = ex.r4(loc.geometry.values, 0.0003)
    loc = loc[~loc.geometry.is_empty & loc.geometry.notna()]
    p = loc.geometry.representative_point()
    loc["cell"] = [f"{int(np.floor(x))}_{int(np.floor(y))}" for x, y in zip(p.x, p.y)]
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        if f.endswith(".json"): os.remove(os.path.join(OUT, f))
    cells, tot = [], 0
    for k, grp in loc.groupby("cell"):
        n, mb = ex.dump(os.path.join(OUT, f"{k}.json"), grp.reset_index(drop=True), {"c": "c"}); cells.append(k); tot += mb
    json.dump({"cells": cells}, open(os.path.join(OUT, "index.json"), "w"))
    print(f"local roads: {len(loc):,} ways, {loc.osm_length.sum() / 1000:,.0f} km in {len(cells)} tiles, {tot:.1f} MB "
          f"(largest {max(os.path.getsize(os.path.join(OUT, c + '.json')) for c in cells) / 1e6:.1f} MB)")
