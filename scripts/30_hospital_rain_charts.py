"""
30_hospital_rain_charts.py -- static charts of the 15-day rain forecast at hospitals in the wettest forecast zones,
showing when intense rain is possible.

Uses the per-hospital series in webmap/data/facilities.geojson (16_), so the charts match the web map:
  hs  3-hourly WeatherNext 3 forecast at the hospital's 0.1 deg cell (ensemble mean and wet end = 90th percentile,
      raw, mm per 3 h), hours where intense / very intense rain is possible (29_ thresholds), wettest hour
  rs  daily forecast, bias-corrected (monthly mapping), and PARDIG observed for the 10 days before the run
Selection: hospitals whose 15-day bias-corrected forecast is >= 50 mm or that have any intense-rain hours; ranked by
intense-rain hours, then forecast total; the top 12 are charted.

Outputs
  charts/ETH_hospital_rain_forecast.png      12 small multiples + locator map
  data/processed/tables/eth_hospital_intense_rain_<init>.csv   every hospital with forecast intense rain, with periods

    python -I scripts\\30_hospital_rain_charts.py
"""
import json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
C_MEAN, C_INT1, C_INT2 = "#6da7ec", "#cdb7e6", "#8656bd"
RAIN_SEQ = ["#fcfcfb", "#e6effa", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf"]
RAIN_LV = [0, 5, 10, 25, 50, 100, 150, 1e4]
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.dpi": 200,
                     "savefig.bbox": "tight"})


def spans(hrs):
    out = []
    for h in sorted(hrs):
        if out and h <= out[-1][1] + 1: out[-1][1] = h
        else: out.append([h, h])
    return out


if __name__ == "__main__":
    d = json.load(open(os.path.join(ROOT, "webmap", "data", "facilities.geojson"), encoding="utf-8"))
    S = json.load(open(os.path.join(ROOT, "webmap", "data", "summary.json"), encoding="utf-8"))
    rows = []
    for f in d["features"]:
        p = f["properties"]
        if p["ty"] != 2 or "hs" not in p: continue
        hs, rs = p["hs"], p["rs"]
        rows.append({"name": p["n"].strip(), "zone": p["a2"], "woreda": p["a3"], "region": p["a1"], "lon": f["geometry"]["coordinates"][0],
                     "lat": f["geometry"]["coordinates"][1], "total15": float(np.nansum(rs["f"])), "obs10": hs["obs10"],
                     "ih": len(hs["i1"]), "ih2": len(hs["i2"]), "pkv": hs["pkv"], "pk": hs["pk"], "hs": hs,
                     "linked": p.get("cf"), "linked30": p.get("c30")})
    D = pd.DataFrame(rows)
    t0 = pd.Timestamp(D.hs.iloc[0]["t0"] + ":00")
    tag = f"{t0:%Y%m%d}"
    at = lambda h: t0 + pd.Timedelta(hours=int(h))

    # --- table: every hospital with forecast intense rain ----------------------------------------------
    def period_txt(hs):
        return "; ".join(f"{at(a):%d %b %H:%M}–{at(b + 1):%H:%M}" for a, b in spans(hs["i1"]))
    T = D[D.ih > 0].sort_values(["ih", "total15"], ascending=False)
    T.assign(intense_periods_utc=T.hs.map(period_txt),
             very_intense_periods_utc=T.hs.map(lambda h: "; ".join(f"{at(a):%d %b %H:%M}–{at(b + 1):%H:%M}" for a, b in spans(h["i2"]))),
             wettest_hour_utc=T.pk.map(lambda h: f"{at(h):%d %b %H:%M}"))[
        ["name", "zone", "woreda", "region", "intense_periods_utc", "very_intense_periods_utc", "wettest_hour_utc", "pkv",
         "total15", "obs10", "linked", "linked30"]].rename(columns={"pkv": "wettest_hour_wet_end_mm", "total15": "forecast_15d_mm_corrected",
                                                                  "obs10": "observed_last_10d_mm", "linked": "linked_facilities",
                                                                  "linked30": "linked_30min_plus"}).round(1).to_csv(
        os.path.join(ROOT, "data", "processed", "tables", f"eth_hospital_intense_rain_{tag}.csv"), index=False)
    print(f"{len(T)} hospitals with forecast intense rain")

    # --- selection -------------------------------------------------------------------------------------
    C = D[(D.total15 >= 50) | (D.ih > 0)].sort_values(["ih", "total15"], ascending=False).head(12).reset_index(drop=True)
    print(C[["name", "zone", "ih", "total15", "pkv"]].round(1).to_string())

    # --- figure: locator + 12 panels ---------------------------------------------------------------------
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(3, 5, width_ratios=[1.35, 1, 1, 1, 1], hspace=0.75, wspace=0.28)
    # locator map with the 15-day forecast rain cells
    axm = fig.add_subplot(gs[0:2, 0])
    rain = json.load(open(os.path.join(ROOT, "webmap", "data", "rain.geojson")))
    xs = sorted({round(ft["geometry"]["coordinates"][0][0][0], 4) for ft in rain["features"]})
    ys = sorted({round(ft["geometry"]["coordinates"][0][0][1], 4) for ft in rain["features"]})
    res = 0.1; x0, y0 = min(xs), min(ys); nx = int(round((max(xs) - x0) / res)) + 1; ny = int(round((max(ys) - y0) / res)) + 1
    G = np.full((ny, nx), np.nan)
    for ft in rain["features"]:
        cx, cy = ft["geometry"]["coordinates"][0][0]
        G[int(round((cy - y0) / res)), int(round((cx - x0) / res))] = ft["properties"]["m"]
    cmap = ListedColormap(RAIN_SEQ); norm = BoundaryNorm(RAIN_LV, cmap.N)
    axm.imshow(G, origin="lower", extent=[x0, x0 + nx * res, y0, y0 + ny * res], cmap=cmap, norm=norm, interpolation="nearest")
    adm = json.load(open(os.path.join(ROOT, "webmap", "data", "adm1.geojson"), encoding="utf-8"))
    for ft in adm["features"]:
        geo = ft["geometry"]; polys = geo["coordinates"] if geo["type"] == "MultiPolygon" else [geo["coordinates"]]
        for poly in polys:
            ring = np.array(poly[0]); axm.plot(ring[:, 0], ring[:, 1], color=INK2, lw=0.4)
    axm.scatter(D.lon, D.lat, s=5, color="#b8b6ae", zorder=3, linewidths=0)
    for i, r in C.iterrows():
        axm.scatter(r.lon, r.lat, s=70, color=C_INT2 if r.ih else INK, edgecolor=SURF, linewidth=0.8, zorder=4)
        axm.text(r.lon, r.lat, f"{i + 1}", color=SURF, fontsize=6.5, ha="center", va="center", zorder=5, fontweight="bold")
    axm.set_aspect("equal"); axm.set_xticks([]); axm.set_yticks([]); [sp.set_visible(False) for sp in axm.spines.values()]
    axm.set_title("Where they are", loc="left", fontsize=11, color=INK)
    axm.legend(handles=[Line2D([], [], marker="o", ls="", color=C_INT2, markersize=8, label="intense rain forecast"),
                        Line2D([], [], marker="o", ls="", color=INK, markersize=8, label="wet zone, no intense rain"),
                        Line2D([], [], marker="o", ls="", color="#b8b6ae", markersize=4, label="other hospitals")]
               + [Patch(color=c, label=l) for c, l in zip(RAIN_SEQ[2:], ["10–25 mm", "25–50", "50–100", "100–150", "150+"])],
               loc="upper left", bbox_to_anchor=(0, -0.02), fontsize=7.5, frameon=False, ncol=2, title="15-day forecast rain (bias-corrected)",
               title_fontsize=7.5)

    slots = [(r, c) for r in range(3) for c in range(1, 5)][:12]
    ymax = max(max(max(r.hs["w"]), 1) for _, r in C.iterrows())
    for i, r in C.iterrows():
        ax = fig.add_subplot(gs[slots[i][0], slots[i][1]])
        hs = r.hs; n = len(hs["m"]); x = np.arange(n) * 3
        wid = lambda a, b: (min(a, (a + b + 1) / 2 - 3), max(b + 1, (a + b + 1) / 2 + 3))     # at least 6 h wide so it shows
        for a, b in spans(hs["i1"]):
            ax.axvspan(*wid(a, b), color=C_INT1, alpha=0.95, lw=0, zorder=0)
            ax.plot((a + b + 1) / 2, ymax * 1.07, marker="v", color=C_INT2, ms=7, zorder=4, clip_on=False)
        for a, b in spans(hs["i2"]):
            ax.axvspan(*wid(a, b), color=C_INT2, alpha=0.95, lw=0, zorder=0)
        ax.fill_between(x + 1.5, hs["m"], step=None, color=C_MEAN, alpha=0.8, lw=0, zorder=2)
        ax.plot(x + 1.5, hs["w"], color=INK2, lw=0.9, ls=(0, (2, 1.5)), zorder=3)
        for dd in (3, 7):
            ax.axvline(dd * 24, color=INK2, lw=0.8, zorder=1)
        ax.set_xlim(0, 360); ax.set_ylim(0, ymax * 1.12)
        ax.set_xticks([0, 72, 168, 360], [f"{at(h):%d %b}" for h in (0, 72, 168, 359)], fontsize=7.5)
        ax.grid(axis="y", color="#e1e0d9", lw=0.5); ax.tick_params(axis="y", labelsize=7.5)
        if slots[i][1] == 1: ax.set_ylabel("mm per 3 h", fontsize=8)
        nm = r["name"]
        for a_, b_ in [("university Teaching Hospital", "Univ. Teaching H."), ("Comprehensive specialized Hospital", "Comp. Spec. H."),
                       ("Comperhensive Specialized  Hospital", "Comp. Spec. H."), ("Primary Hospital", "PH"), ("Primery Hospital", "PH"),
                       ("Primary Hosiptal", "PH"), ("General Hospital", "GH"), ("General hospital", "GH")]:
            nm = nm.replace(a_, b_)
        ax.set_title(f"{i + 1}. {nm}", loc="left", fontsize=9.5, color=INK, pad=26, fontweight="bold")
        sp_ = spans(hs["i1"])
        line2 = (f"intense rain: {len(sp_)} period{'s' if len(sp_) > 1 else ''}, first {at(sp_[0][0]):%d %b %H}:00 UTC" if r.ih else "no intense rain forecast")
        ax.text(0, 1.03, f"{r.zone} · 15-day forecast {r.total15:.0f} mm\n{line2}", transform=ax.transAxes, fontsize=7.3,
                color=C_INT2 if r.ih else INK2, va="bottom", linespacing=1.3)
    hd = [Patch(color=C_MEAN, label="forecast rain, ensemble mean (mm per 3 h)"), Line2D([], [], color=INK2, ls=(0, (2, 1.5)), label="wet end (90th percentile)"),
          Patch(color=C_INT1, label="intense rain possible (≥ 10 mm in an hour)"), Patch(color=C_INT2, label="very intense (≥ 20 mm in an hour)"),
          Line2D([], [], color=INK2, lw=0.8, label="forecast windows: days 1–3 | 4–7 | 8–15")]
    fig.legend(handles=hd, loc="lower center", ncol=5, frameon=False, fontsize=8.5, bbox_to_anchor=(0.55, 0.045))
    fig.suptitle("Hospitals in the wettest forecast zones: when intense rain is possible", x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK, y=0.995)
    fig.text(0.01, 0.962, f"WeatherNext 3 run {t0:%d %b %Y} 00Z, 15 days · 12 hospitals with forecast intense rain or ≥ 50 mm bias-corrected forecast, "
             f"ranked by intense-rain hours then total · {len(T)} hospitals nationally have intense rain forecast", fontsize=9.5, color=INK2)
    fig.text(0.01, 0.02, "Series: raw WeatherNext 3 at each hospital's 0.1° cell (amounts run low; the corrected 15-day total is in each subtitle). Intense rain: the wettest "
             "hour of the hourly 90th percentile above thresholds calibrated so it flags as often as PARDIG observed ≥ 10 / 20 mm in an hour (1–18 Nov 2026).\n"
             "At the scale used here, flags are rarely false (6–14% with no downpour within ~20 km in the window), but many short-range downpours are missed. "
             "Days 8–15 are lower confidence. Times are UTC (Ethiopia = UTC + 3).", fontsize=7.8, color=MUTED, va="bottom")
    out = os.path.join(ROOT, "charts", "ETH_hospital_rain_forecast.png")
    fig.savefig(out); print("wrote", out)
