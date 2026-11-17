"""
14_roads_rain_risk.py -- example: which Ethiopian roads may slow down under the WeatherNext 2 15-day rainfall
forecast, given their surface (paved / unpaved).

Inputs
  data/raw/roads/heigit_eth_roadsurface_lines.geojson   HeiGIT Mapillary road surface lines (HDX, Nov 2024)
      surface = OSM surface tag first, then the Mapillary image model (combined_surface_osm_priority).
      About 3/4 of the road length has neither; for those the surface is inferred from the road class,
      using the share of each class that is paved among the labelled roads.
  data/raw/weathernext/wn2_eth_<init>.npz               64-member daily rainfall from 11_weathernext_forecast.py
  data/raw/dem/gem15_eth_0p001.tif                      Google GEM-15 bare-earth elevation + mean 15 m slope on a
                                                        ~110 m grid, from 15_fetch_gem15_dem.py

Method (illustrative thresholds, not calibrated against observed travel times)
  Three factors per road segment, each sampled every ~110 m along the line:
  R  rain hazard 0-3 at the segment's 0.25 deg cell, from the ensemble:
      3 high      >= 50% of members have a day >= 20 mm, or ensemble-mean wettest 3 days >= 40 mm
      2 moderate  >= 50% of members have a day >= 10 mm, or wettest 3 days >= 20 mm
      1 low       15-day ensemble mean >= 10 mm
  S  surface 0-2: paved 0, unpaved 1; +1 if OSM smoothness is bad/very_bad/horrible/very_horrible/impassable
  T  steepness 0-2, after the Ethiopian Roads Authority terrain classes (flat/rolling < 25 % ground slope,
     mountainous 25-50 %, escarpment > 50 %) plus the road's own gradient:
      0 flat/rolling  mean terrain slope < 14 deg and road grade < 5 %
      1 mountainous   terrain 14-26.6 deg or grade 5-10 %
      2 escarpment    terrain >= 26.6 deg or grade >= 10 %
     road grade = total climb + descent along the line / its length (smoothed by the 110 m DEM).
  Slowdown tier = R + S + T - 1, clipped to 0-3 (Normal, Minor, Elevated, Severe), and at most one tier above
  R, so light rain alone cannot make a road Severe; no forecast rain (R = 0) = Normal.
  Flat roads give the same tiers as before (unpaved: Minor/Elevated/Severe for low/moderate/high rain;
  paved one lower); steep ground adds up to two tiers.
  Separate flag slide_exposure: R >= 2 and the steepest ~110 m cell along the road >= 26.6 deg - places
  where rock fall, slips or wash-outs could block the road, not just slow it.
  Assumed speed factors: Minor 0.9, Elevated 0.75, Severe 0.5 (may be impassable).
  Footpaths, tracks under construction etc. are dropped; only vehicle roads are scored.
  Also dropped: ~5,200 km of seismic survey cut-lines around the Calub / Hilala gas fields (44-45.5 E,
  5.5-6.8 N) - unnamed, dead-straight lines > 2 km that OSM maps as tracks/unclassified roads.

Outputs (<init> = run date)
  charts/ETH_roads_rain_risk_map.png      roads by slowdown tier over the 15-day forecast, national + hotspot zoom
  charts/ETH_roads_rain_risk_adm1.png     km of road per tier by region
  charts/ETH_roads_rain_risk_daily.png    probability-weighted km of road under >= 10 mm rain, per forecast day
  charts/ETH_roads_rain_risk_factors.png  how surface and steepness change the tier, km by factor combination
  data/processed/roads/eth_roads_rain_risk_<init>.gpkg   scored segments (tier Minor+), for GIS
  data/processed/roads/eth_roads_scored_<init>.parquet   every vehicle segment with tier + speed factor (routing)
  data/processed/tables/eth_roads_rain_risk_adm1_<init>.csv

    python scripts\\14_roads_rain_risk.py [--init 2026-11-17]
(no -I: needs the helpers in 11_weathernext_forecast.py, which import earthengine-api from user site-packages)
"""
import argparse, glob, importlib.util, os
import numpy as np, pandas as pd, geopandas as gpd, shapely
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("wn", os.path.join(ROOT, "scripts", "11_weathernext_forecast.py"))
wn = importlib.util.module_from_spec(spec); spec.loader.exec_module(wn)

RRAW = os.path.join(ROOT, "data", "raw", "roads")
SRC = os.path.join(RRAW, "heigit_eth_roadsurface_lines.geojson")
CACHE = os.path.join(RRAW, "eth_vehicle_roads.parquet")
PROC = os.path.join(ROOT, "data", "processed", "roads"); os.makedirs(PROC, exist_ok=True)

VEHICLE = ["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary",
           "secondary_link", "tertiary", "tertiary_link", "unclassified", "residential", "living_street",
           "service", "track", "road"]
BAD = {"bad", "very_bad", "horrible", "very_horrible", "impassable"}
TIERS = ["Normal", "Minor", "Elevated", "Severe"]
SPEED = {"Normal": 1.0, "Minor": 0.9, "Elevated": 0.75, "Severe": 0.5}
# ordinal warm ramp for severity (light -> dark), grey for no change
TCOL = {"Normal": "#c9c8c2", "Minor": "#f2b33d", "Elevated": "#e0701f", "Severe": "#a8320f"}
TLW = {"Normal": 0.15, "Minor": 0.45, "Elevated": 0.7, "Severe": 1.0}
RAIN_SEQ = ["#fcfcfb", "#e6effa", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf"]
RAIN_LV = [0, 5, 10, 25, 50, 100, 150, 400]
INK, INK2, MUTED, SURF = wn.INK, wn.INK2, wn.MUTED, wn.SURF


def load_roads():
    if os.path.exists(CACHE): return gpd.read_parquet(CACHE)
    cols = ["highway", "surface", "smoothness", "combined_surface_osm_priority", "osm_length", "name", "name:en"]
    g = gpd.read_file(SRC, columns=cols, engine="pyogrio", use_arrow=True)
    g = g[g.highway.isin(VEHICLE)].rename(columns={"combined_surface_osm_priority": "surf_known", "name:en": "name_en"})
    g = g.reset_index(drop=True); g.to_parquet(CACHE); return g


SEISMIC_BOX = (44.0, 5.5, 45.5, 6.8)    # Calub / Hilala gas fields, Ogaden


def drop_seismic(g):
    """Drop the grid of seismic survey cut-lines around the Ogaden gas fields that OSM maps as roads:
    unnamed track/unclassified/residential/service lines > 2 km that are dead straight (chord / length > 0.995)."""
    geo = g.geometry.values
    st = shapely.distance(shapely.get_point(geo, 0), shapely.get_point(geo, -1)) / np.maximum(shapely.length(geo), 1e-12)
    mid = shapely.line_interpolate_point(geo, 0.5, normalized=True); x, y = shapely.get_x(mid), shapely.get_y(mid)
    x0, y0, x1, y1 = SEISMIC_BOX
    m = ((st > 0.995) & (g.osm_length > 2000).values & g.name.isna().values
         & g.highway.isin(["track", "unclassified", "residential", "service"]).values
         & (x > x0) & (x < x1) & (y > y0) & (y < y1))
    print(f"dropped {m.sum()} probable seismic cut-lines, {g.osm_length[m].sum() / 1000:,.0f} km")
    return g[~m].reset_index(drop=True), g.osm_length[m].sum() / 1000


def classify_surface(g):
    lab = g[g.surf_known.notna()]
    share = (lab.osm_length * (lab.surf_known == "paved")).groupby(lab.highway).sum() / lab.osm_length.groupby(lab.highway).sum()
    p = g.highway.map(share).fillna(0.0)
    g["surface_class"] = g.surf_known.where(g.surf_known.notna(), np.where(p >= 0.5, "paved", "unpaved"))
    g["surface_source"] = np.where(g.surf_known.notna(), "OSM tag / Mapillary model", "inferred from road class")
    return g, share


def hazard(arr):
    """arr [member, day, y, x] mm -> hazard level 0-3 per cell plus the fields behind it."""
    mean15 = arr.sum(1).mean(0)
    dmax = arr.max(1)
    p20, p10 = (dmax >= 20).mean(0), (dmax >= 10).mean(0)
    r3 = np.stack([arr[:, i:i + 3].sum(1) for i in range(arr.shape[1] - 2)], 1).max(1).mean(0)
    h = np.zeros(mean15.shape, "int8")
    h[mean15 >= 10] = 1
    h[(p10 >= 0.5) | (r3 >= 20)] = 2
    h[(p20 >= 0.5) | (r3 >= 40)] = 3
    return h, dict(mean15=mean15, p20=p20, p10=p10, r3=r3)


DEM = os.path.join(ROOT, "data", "raw", "dem", "gem15_eth_0p001.tif")
HF = os.path.join(ROOT, "data", "health_facilities", "ethiopian-health-facilities.csv")
TERRAIN = ["flat/rolling", "mountainous", "escarpment"]


def terrain(geoms, length_m):
    """Sample GEM-15 every ~110 m along each line -> road grade %, mean and max terrain slope (deg)."""
    import rasterio
    with rasterio.open(DEM) as d:
        elev = d.read(1); slope = d.read(2).astype("float32") / 4; tr = d.transform
    deg = shapely.length(geoms)
    n = np.maximum(2, np.ceil(deg / 0.001).astype(int) + 1)          # samples per segment
    seg = np.repeat(np.arange(len(geoms)), n)
    start = np.repeat(np.cumsum(n) - n, n)
    frac = (np.arange(len(seg)) - start) / np.repeat(n - 1, n)
    pts = shapely.line_interpolate_point(np.asarray(geoms)[seg], frac, normalized=True)
    col = ((shapely.get_x(pts) - tr.c) / tr.a).astype(int); row = ((shapely.get_y(pts) - tr.f) / tr.e).astype(int)
    col = np.clip(col, 0, elev.shape[1] - 1); row = np.clip(row, 0, elev.shape[0] - 1)
    z = elev[row, col].astype("float32"); s = slope[row, col]
    ok = z > 0                                                        # 0 = no data (water, outside)
    same = (seg[1:] == seg[:-1]) & ok[1:] & ok[:-1]
    climb = np.bincount(seg[1:][same], np.abs(np.diff(z))[same], minlength=len(geoms))
    grade = np.where(length_m > 0, 100 * climb / np.maximum(length_m, 1), 0)
    cnt = np.bincount(seg[ok], minlength=len(geoms))
    smean = np.bincount(seg[ok], s[ok], minlength=len(geoms)) / np.maximum(cnt, 1)
    smax = np.zeros(len(geoms), "float32"); np.maximum.at(smax, seg[ok], s[ok])
    elev_mean = np.bincount(seg[ok], z[ok], minlength=len(geoms)) / np.maximum(cnt, 1)
    return grade, smean, smax, elev_mean, (elev, tr)


def to_cell(x, y):
    c = np.clip(((x - wn.X0) / wn.RES).astype(int), 0, wn.NX - 1)
    r = np.clip(((wn.Y0 - y) / wn.RES).astype(int), 0, wn.NY - 1)
    return r, c


def lines(ax, gg, color, lw, z):
    coords, idx = shapely.get_coordinates(gg.geometry.values, return_index=True)
    cuts = np.flatnonzero(np.diff(idx)) + 1
    ax.add_collection(LineCollection(np.split(coords, cuts), colors=color, linewidths=lw, zorder=z,
                                     capstyle="round"))


def rain_bg(ax, mean15):
    cmap = ListedColormap(RAIN_SEQ); norm = BoundaryNorm(RAIN_LV, cmap.N)
    a = np.where(MASK, mean15, np.nan)
    return ax.imshow(a, extent=wn.EXT, cmap=cmap, norm=norm, interpolation="nearest", zorder=0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--init"); a = ap.parse_args()
    f = (os.path.join(wn.RAW, f"wn2_eth_{pd.Timestamp(a.init):%Y%m%d}00.npz") if a.init else
         sorted(glob.glob(os.path.join(wn.RAW, "wn2_eth_??????????.npz")))[-1])
    z = np.load(f); arr, MASK = z["precip"], z["mask"]
    init = pd.Timestamp(str(z["init"])); days = pd.date_range(init, periods=arr.shape[1])
    tag, span = f"{init:%Y%m%d}", f"{days[0]:%d %b} – {days[-1]:%d %b %Y}"
    print(f"forecast {init:%Y-%m-%d} 00Z, {span}")

    H, fld = hazard(arr)
    roads, seis_km = drop_seismic(load_roads())
    g, share = classify_surface(roads)
    miss = g.osm_length.isna()                             # ~1% lack osm_length: measure them in UTM 37N
    g.loc[miss, "osm_length"] = g[miss].geometry.to_crs(32637).length.round(0)
    km = g.osm_length / 1000
    mid = shapely.line_interpolate_point(g.geometry.values, 0.5, normalized=True)
    mx, my = shapely.get_x(mid), shapely.get_y(mid)
    r, c = to_cell(mx, my)
    hz = H[r, c]
    unp = (g.surface_class == "unpaved").values
    S = unp.astype(int) + g.smoothness.isin(BAD).values.astype(int)
    grade, smean, smax, elev_mean, (DEMA, DEMT) = terrain(g.geometry.values, g.osm_length.values)
    T = np.where((smean >= 26.6) | (grade >= 10), 2, np.where((smean >= 14) | (grade >= 5), 1, 0))
    tier = np.clip(hz + S + T - 1, 0, 3)
    tier = np.where(hz == 0, 0, np.minimum(tier, hz + 1))
    tier_flat = np.where(hz == 0, 0, np.minimum(np.clip(hz + S - 1, 0, 3), hz + 1))   # same, ignoring terrain
    g["rain_hazard"], g["surface_score"], g["steep_score"] = hz, S, T
    g["terrain_class"] = np.array(TERRAIN)[T]
    g["road_grade_pct"], g["terrain_slope_mean_deg"] = grade.round(1), smean.round(1)
    g["terrain_slope_max_deg"], g["elev_mean_m"] = smax.round(1), elev_mean.round(0)
    g["slide_exposure"] = (hz >= 2) & (smax >= 26.6)
    g["tier"] = pd.Categorical(np.array(TIERS)[tier], TIERS, ordered=True)
    g["tier_without_terrain"] = np.array(TIERS)[tier_flat]
    g["speed_factor"] = g.tier.map(SPEED).astype(float)
    g["rain_15d_mm"] = fld["mean15"][r, c].round(1)
    g["wettest_3d_mm"] = fld["r3"][r, c].round(1)
    g["p_day_ge20mm"] = fld["p20"][r, c].round(2)
    daymean = arr.mean(0)                                   # [day, y, x]
    g["peak_day"] = days[daymean[:, r, c].argmax(0)].strftime("%Y-%m-%d")
    reg = gpd.GeoDataFrame({"region": wn.NAMES}, geometry=wn.GEOMS, crs="EPSG:4326")
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(mx, my), crs="EPSG:4326")
    g["region"] = gpd.sjoin(pts, reg, how="left", predicate="within").groupby(level=0).region.first().values
    g["region"] = g.region.fillna("Unassigned")

    # --- tables & GIS -----------------------------------------------------------------
    tab = pd.crosstab(g.region, g.tier, values=km, aggfunc="sum").reindex(columns=TIERS).fillna(0).round(0)
    tab["share_slowed_%"] = (100 * tab[TIERS[1:]].sum(1) / tab.sum(1)).round(1)
    tab["slide_exposure_km"] = km[g.slide_exposure].groupby(g.region[g.slide_exposure]).sum().reindex(tab.index).fillna(0).round(0)
    tab["raised_by_terrain_km"] = km[tier > tier_flat].groupby(g.region[tier > tier_flat]).sum().reindex(tab.index).fillna(0).round(0)
    tab = tab.sort_values("share_slowed_%", ascending=False)
    tab.to_csv(os.path.join(wn.TB, f"eth_roads_rain_risk_adm1_{tag}.csv"))
    flag = g[g.tier != "Normal"].copy(); flag["tier"] = flag.tier.astype(str)
    flag.drop(columns=["surf_known"]).to_file(os.path.join(PROC, f"eth_roads_rain_risk_{tag}.gpkg"),
                                              layer="roads_rain_risk", driver="GPKG", engine="pyogrio")
    full = g[["highway", "surface_class", "osm_length", "tier", "tier_without_terrain", "speed_factor", "slide_exposure",
              "region", "geometry"]].copy()
    full["tier"] = full.tier.astype(str)
    full.to_parquet(os.path.join(PROC, f"eth_roads_scored_{tag}.parquet"))       # every segment, for routing (17_)
    tot = km.groupby(g.tier, observed=False).sum()
    known = km[g.surface_source.str.startswith("OSM")].sum() / km.sum()
    print(tot.round(0).to_string()); print(f"surface known for {known:.0%} of length")
    print(tab.to_string())
    print("terrain class km:", km.groupby(g.terrain_class).sum().round(0).to_dict())
    print(f"raised by terrain: {km[tier > tier_flat].sum():,.0f} km; slide exposure {km[g.slide_exposure].sum():,.0f} km")
    print("without terrain:", km.groupby(tier_flat).sum().round(0).to_dict())

    # --- map: national + hotspot zoom ---------------------------------------------------
    sev = g[g.tier.isin(["Severe", "Elevated"])]
    w = (sev.osm_length * np.where(sev.tier == "Severe", 2, 1))
    bx, by = np.floor(mx[sev.index] / 0.5) * 0.5, np.floor(my[sev.index] / 0.5) * 0.5
    hot = pd.Series(w.values, index=pd.MultiIndex.from_arrays([bx, by])).groupby(level=[0, 1]).sum()
    # densest 2.5 x 2 deg window of Severe/Elevated road length
    best, box = -1, None
    for (x0, y0) in hot.index:
        for dx in (0, -1, -2):
            for dy in (0, -1):
                xa, ya = x0 + dx, y0 + dy
                s = hot[(hot.index.get_level_values(0) >= xa) & (hot.index.get_level_values(0) < xa + 2.5) &
                        (hot.index.get_level_values(1) >= ya) & (hot.index.get_level_values(1) < ya + 2)].sum()
                if s > best: best, box = s, (xa, ya, xa + 2.5, ya + 2)

    fig, axs = plt.subplots(1, 2, figsize=(15, 6.4), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.04})
    for k, ax in enumerate(axs):
        if k == 1:                                          # hillshade from GEM-15 under the zoom
            c0, r0 = ~DEMT * (box[0], box[3]); c1, r1 = ~DEMT * (box[2], box[1])
            sub = DEMA[int(r0):int(r1), int(c0):int(c1)].astype("float32"); sub[sub <= 0] = np.nan
            dy, dx = np.gradient(sub, 110.0, 110.0 * np.cos(np.deg2rad((box[1] + box[3]) / 2)))
            slp = np.arctan(np.hypot(dx, dy)); asp = np.arctan2(-dx, dy)
            az, alt = np.deg2rad(315), np.deg2rad(45)
            hs = np.sin(alt) * np.cos(slp) + np.cos(alt) * np.sin(slp) * np.cos(az - asp)
            ax.imshow(hs, extent=[box[0], box[2], box[1], box[3]], cmap="gray", vmin=-0.2, vmax=1.1, zorder=0,
                      interpolation="bilinear")
        im = rain_bg(ax, fld["mean15"])
        if k == 1: im.set_alpha(0.5); im.set_zorder(0.5)
        for t in TIERS:
            sub = g[g.tier == t]
            if len(sub): lines(ax, sub, TCOL[t], TLW[t] * (1 if k == 0 else 1.8), TIERS.index(t) + 1)
        wn.draw_outline(ax); wn.mapax(ax)
    ax = axs[1]; ax.set_xlim(box[0], box[2]); ax.set_ylim(box[1], box[3])
    sl = g.slide_exposure.values & (mx > box[0]) & (mx < box[2]) & (my > box[1]) & (my < box[3])
    ax.scatter(mx[sl], my[sl], marker="^", s=16, color=INK, edgecolor=SURF, linewidth=0.5, zorder=8)
    # health facilities (approved hospitals / health centres) from the national register
    hf = pd.read_csv(HF, usecols=["Type", "Status", "Latitude", "Longitude"])
    hf = hf[(hf.Status == "Approved")]
    hosp, hc = hf[hf.Type == "Hospital"], hf[hf.Type == "Health Center"]
    inb = lambda d: d[(d.Longitude > box[0]) & (d.Longitude < box[2]) & (d.Latitude > box[1]) & (d.Latitude < box[3])]
    ax.scatter(inb(hc).Longitude, inb(hc).Latitude, marker="o", s=15, facecolor=SURF, edgecolor=INK, linewidth=0.8, zorder=9)
    ax.scatter(inb(hosp).Longitude, inb(hosp).Latitude, marker="P", s=70, color=INK, edgecolor=SURF, linewidth=0.7, zorder=10)
    axs[0].scatter(hosp.Longitude, hosp.Latitude, marker="P", s=11, color=INK, edgecolor=SURF, linewidth=0.3, zorder=9)
    ax.legend(handles=[Line2D([], [], marker="^", ls="", color=INK, markeredgecolor=SURF, markersize=6,
                              label=f"slip / rock-fall exposure ({km[g.slide_exposure].sum():,.0f} km nationally)"),
                       Line2D([], [], marker="P", ls="", color=INK, markeredgecolor=SURF, markersize=8, label="hospital (both maps)"),
                       Line2D([], [], marker="o", ls="", markerfacecolor=SURF, markeredgecolor=INK, markersize=5, label="health centre")],
              loc="upper left", bbox_to_anchor=(0, -0.01), fontsize=8.5, frameon=False)
    for s in ax.spines.values(): s.set_visible(True); s.set_color(INK2); s.set_linewidth(0.8)
    axs[0].add_patch(Rectangle(box[:2], box[2] - box[0], box[3] - box[1], fill=False, ec=INK, lw=1.1, zorder=10))
    # label the regions inside the zoom
    for n, geom in zip(wn.NAMES, wn.GEOMS):
        p = geom.representative_point()
        if box[0] < p.x < box[2] and box[1] < p.y < box[3]:
            ax.text(p.x, p.y, n, fontsize=8.5, color=INK2, ha="center", zorder=11,
                    bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec="none", alpha=0.8))
    wn.title(axs[0], "Roads that may slow down under the next 15 days of forecast rain",
             f"WeatherNext 2, run {init:%d %b %Y} 00Z ({span}) · rain × road surface × terrain steepness (GEM-15)")
    axs[1].set_title(f"Hotspot: {box[1]:.1f}–{box[3]:.1f}°N, {box[0]:.1f}–{box[2]:.1f}°E, over GEM-15 hillshade", loc="left",
                     fontsize=10.5, color=INK, pad=6)
    hd = [Line2D([], [], color=TCOL[t], lw=3 if t != "Normal" else 1.5,
                 label=f"{t}{'' if t == 'Normal' else f' (~{int(100 * (1 - SPEED[t]))}% slower)'} · {tot[t]:,.0f} km")
          for t in TIERS[::-1]]
    leg = axs[0].legend(handles=hd, loc="upper left", bbox_to_anchor=(0, -0.01), ncol=2, fontsize=8.5,
                        frameon=False, title="Road slowdown tier", title_fontsize=9, columnspacing=1.2)
    leg._legend_box.align = "left"
    cax = axs[1].inset_axes([0.64, -0.12, 0.36, 0.03])     # right of the slip-marker legend
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=RAIN_LV[1:-1])
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7.5, length=0, colors=INK2)
    cax.set_title("15-day rain, mm (ensemble mean)", fontsize=8, color=INK2, loc="left")
    fig.canvas.draw()                                       # footnote goes under the lowest legend / colour bar
    yb = min([ax_.get_position().y0 for ax_ in axs] + [cax.get_position().y0 - 0.03] +
             [l_.get_window_extent().transformed(fig.transFigure.inverted()).y0 for l_ in (leg, axs[1].get_legend())])
    fig.text(0.01, yb - 0.02, "Tier = rain hazard (0–3) + surface (paved 0, unpaved 1, +1 if tagged bad/horrible) + steepness (0–2) − 1, "
                 "capped at one tier above the rain hazard; no rain = Normal.\nRain: high = half the members give a day ≥ 20 mm or the "
                 "wettest 3 days average ≥ 40 mm; moderate = a day ≥ 10 mm or 3 days ≥ 20 mm; low = 15-day mean ≥ 10 mm. "
                 "Steepness: mountainous = terrain ≥ 14° or road grade ≥ 5 %; escarpment = ≥ 26.6° or ≥ 10 %.\n"
                 "Slip / rock-fall exposure: moderate+ rain and a ≥ 26.6° cell along the road. Speed factors are illustrative, not calibrated. "
                 f"Roads: HeiGIT Mapillary road surface (OSM, Nov 2024); surface known for {known:.0%} of length, the rest inferred from road class.\n"
                 f"Health facilities: approved hospitals and health centres from the Ethiopian health facility register. "
                 f"Terrain: Google GEM-15 bare-earth model, slope from 15 m averaged to ~110 m, sampled every ~110 m along each road. "
                 f"Vehicle roads only, {km.sum():,.0f} km; {seis_km:,.0f} km of seismic survey lines in the Ogaden removed.",
             fontsize=8, color=MUTED, ha="left", va="top")
    wn.save(fig, os.path.join(wn.OUT, "ETH_roads_rain_risk_map.png"))

    # --- km by region -----------------------------------------------------------------
    t2 = tab[tab.index != "Unassigned"].sort_values("share_slowed_%")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.4, 1], "wspace": 0.35})
    left = np.zeros(len(t2))
    for t in TIERS[1:][::-1]:
        a1.barh(t2.index, t2[t], left=left, color=TCOL[t], height=0.62, edgecolor=SURF, linewidth=1, label=t)
        left += t2[t].values
    for i, (v, s) in enumerate(zip(left, t2["share_slowed_%"])):
        if v > 0: a1.text(v, i, f"  {v:,.0f} km", va="center", fontsize=8.5, color=INK2)
    a1.set_xlabel("road length likely slowed, km"); a1.grid(axis="y", visible=False)
    a1.legend(loc="lower right", fontsize=9, frameon=False, title="Tier", title_fontsize=9)
    wn.title(a1, "Road length likely slowed, by region", f"15-day forecast from {init:%d %b %Y}")
    a2.barh(t2.index, t2["share_slowed_%"], color=TCOL["Elevated"], height=0.62)
    for i, s in enumerate(t2["share_slowed_%"]):
        a2.text(s, i, f"  {s:.0f}%", va="center", fontsize=8.5, color=INK2)
    a2.set_yticklabels([]); a2.set_xlabel("share of the region's road length, %"); a2.grid(axis="y", visible=False)
    a2.set_xlim(0, max(10, t2["share_slowed_%"].max() * 1.18))
    wn.title(a2, "Share of the network", "Minor + Elevated + Severe")
    wn.foot(fig, "Vehicle roads from HeiGIT Mapillary road surface lines; segments assigned to the region containing their midpoint. "
                 "Tiers as in ETH_roads_rain_risk_map.png.")
    wn.save(fig, os.path.join(wn.OUT, "ETH_roads_rain_risk_adm1.png"))

    # --- when: km of road under likely >= 10 mm per day ---------------------------------
    # timing differs between members, so use probability-weighted km: sum of length x P(>= 10 mm that day)
    pday = (arr >= 10).mean(0)[:, r, c]                    # [day, segment]
    kmv = km.values
    du = (pday * (kmv * unp)).sum(1); dp = (pday * (kmv * ~unp)).sum(1)
    x = np.arange(len(days))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.bar(x, du, width=0.7, color=TCOL["Elevated"], label="unpaved", edgecolor=SURF, linewidth=1)
    ax.bar(x, dp, width=0.7, bottom=du, color=TCOL["Normal"], label="paved", edgecolor=SURF, linewidth=1)
    for i, v in enumerate(du + dp):
        ax.text(i, v, f"{v / 1000:.1f}k", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, [f"{d:%d %b}" for d in days], fontsize=8.5)
    ax.set_ylabel("expected road km under ≥ 10 mm"); ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    wn.title(ax, "When the wet days fall on the road network",
             "Road length × chance of ≥ 10 mm that UTC day (share of 64 members), summed over all segments")
    wn.foot(fig, f"WeatherNext 2 64-member ensemble, run {init:%d %b %Y} 00Z; each segment takes the 0.25° cell at its midpoint. "
                 "Members disagree on timing, so the risk is spread over several days rather than concentrated on one.")
    wn.save(fig, os.path.join(wn.OUT, "ETH_roads_rain_risk_daily.png"))
    # --- factors: surface x terrain, km by tier, for roads with any forecast rain --------
    wet = hz > 0
    combo = pd.DataFrame({"surface": np.where(unp, "unpaved", "paved"), "terrain": g.terrain_class.values,
                          "tier": g.tier.values, "km": kmv})[wet]
    pv = combo.pivot_table(index=["surface", "terrain"], columns="tier", values="km", aggfunc="sum",
                           observed=False).reindex(columns=TIERS).fillna(0)
    pv = pv.reindex(pd.MultiIndex.from_product([["paved", "unpaved"], TERRAIN])).fillna(0)
    lab = [f"{s_} · {t_}" for s_, t_ in pv.index]
    sh = pv.div(pv.sum(1).replace(0, np.nan), axis=0) * 100
    fig, (b1, b2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.2], "wspace": 0.08})
    yy = np.arange(len(pv))[::-1]
    for ax_, data, xl in ((b1, pv, "road km"), (b2, sh, "share of the group's km, %")):
        left = np.zeros(len(pv))
        for t in TIERS:
            ax_.barh(yy, data[t].values, left=left, color=TCOL[t], height=0.62, edgecolor=SURF, linewidth=1,
                     label=t)
            left += np.nan_to_num(data[t].values)
        ax_.set_xlabel(xl); ax_.grid(axis="y", visible=False)
    b1.set_yticks(yy, lab)
    for i, v in zip(yy, pv.sum(1).values):
        b1.text(v, i, f"  {v:,.0f}", va="center", fontsize=8.5, color=INK2)
    b1.set_xlim(0, pv.sum(1).max() * 1.2)
    b2.set_yticks(yy, []); b2.set_xlim(0, 100)
    b1.legend(loc="upper right", frameon=False, fontsize=9, title="Tier", title_fontsize=9)
    wn.title(b1, "Roads under forecast rain, by surface and terrain", f"Road length with any rain hazard (R ≥ 1), run {init:%d %b %Y}")
    wn.title(b2, "Steeper ground pushes roads up a tier", "Tier mix within each group")
    raised = km[tier > tier_flat].sum()
    wn.foot(fig, f"Terrain raises {raised:,.0f} km of road at least one tier compared with rain and surface alone. "
                 "Terrain classes after the Ethiopian Roads Authority bands, from Google GEM-15 (see ETH_roads_rain_risk_map.png).")
    wn.save(fig, os.path.join(wn.OUT, "ETH_roads_rain_risk_factors.png"))
    print("hotspot box", box)
