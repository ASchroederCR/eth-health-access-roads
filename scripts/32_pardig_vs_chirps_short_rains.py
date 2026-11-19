"""
32_pardig_vs_chirps_short_rains.py -- does it matter that the forecast is corrected to PARDIG rather than CHIRPS?
Compares the two observed datasets over the Oct-Dec short rains of 2023, 2024 and 2025 on the statistics the road
hazards use: rain totals, heavy days (>= 10 / 20 mm), wet-day 95th percentile and 10-day running totals in the
accumulation classes (>= 25 / 50 / 100 mm).

It also saves PARDIG / CHIRPS total-rain ratios for the forecast's calendar weeks (run date +/- 15 days, pooled over the
three years) per region, which 33_ uses to put the PARDIG-scaled forecast into CHIRPS terms before dividing by the
CHIRPS 1991-2020 normal.

Inputs
  data/raw/pardig/eth_daily_*.npz        PARDIG and CHIRPS v3 (DAILY_SAT) daily on PARDIG's 0.1 deg grid (08_)
Outputs
  data/processed/tables/eth_pardig_chirps_short_rains.csv   per region x season (and pooled) statistics
  data/processed/pardig_chirps_ratio.json                   PARDIG / CHIRPS total ratio per region, forecast weeks
  charts/ETH_pardig_chirps_short_rains.png

    python scripts\\32_pardig_vs_chirps_short_rains.py --init 2026-11-19
"""
import argparse, datetime as dt, glob, importlib.util, json, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from rasterio.transform import from_origin


def load(name, file):
    s = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", file))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TB = os.path.join(ROOT, "data", "processed", "tables")
RATIO = os.path.join(ROOT, "data", "processed", "pardig_chirps_ratio.json")
SEASONS = [2023, 2024, 2025]
MIN_CELLS = 30            # regions smaller than this (Addis Ababa, Dire Dawa, Hareri) are in the table, not the chart


def load_obs():
    P, C, D = [], [], []
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "pardig", "eth_daily_*.npz"))):
        e = np.load(f); P.append(e["pardig"]); C.append(e["chirps"]); D.append(pd.to_datetime(e["dates"])); mask = e["mask"]
    d = pd.DatetimeIndex(np.concatenate(D)); keep = ~d.duplicated()
    o = np.argsort(d[keep]); return np.concatenate(P)[keep][o], np.concatenate(C)[keep][o], d[keep][o], mask


def run10(a):
    """10-day running totals along axis 0 (first 9 days dropped)."""
    c = np.cumsum(a, 0); out = c[9:].copy(); out[1:] -= c[:-10]; return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--init", required=True); a = ap.parse_args()
    init = dt.date.fromisoformat(a.init)
    wn = load("wn", "11_weathernext_forecast.py")
    P, C, D, mask = load_obs()
    zones = wn.zones_for(from_origin(32.95, 15.05, 0.1, 0.1), mask.shape)
    lat = 15.05 - 0.1 * (np.arange(mask.shape[0]) + 0.5); w = wn.area_w(lat, mask.shape)
    regions = [("Ethiopia", mask)] + [(n, mask & (zones == i + 1)) for i, n in enumerate(wn.NAMES)]

    rows, pooled = [], {}
    for yr in SEASONS + ["pooled"]:
        sel = D.year.isin(SEASONS) & D.month.isin([10, 11, 12]) if yr == "pooled" else (D.year == yr) & D.month.isin([10, 11, 12])
        if yr != "pooled" and sel.sum() < 92: raise SystemExit(f"season {yr}: only {sel.sum()} days on disk (run 08_)")
        # 10-day running totals within each season (never across the gap between seasons)
        r10p, r10c = [], []
        for y in (SEASONS if yr == "pooled" else [yr]):
            s = (D.year == y) & D.month.isin([10, 11, 12]); r10p.append(run10(P[s])); r10c.append(run10(C[s]))
        r10p, r10c = np.concatenate(r10p), np.concatenate(r10c)
        Ps, Cs = P[sel], C[sel]
        for name, m in regions:
            p, c, ww = Ps[:, m], Cs[:, m], w[m]
            nday = p.shape[0]
            def tot(x): return float((x.sum(0) * ww).sum() / ww.sum()) / (len(SEASONS) if yr == "pooled" else 1)
            def frac(x, t): return float(((x >= t) * ww).sum() / (ww.sum() * x.shape[0]))
            def csi(f, o, t): A, B = f >= t, o >= t; return float((A & B).sum() / max((A | B).sum(), 1))
            row = {"region": name, "season": yr, "cells": int(m.sum()),
                   "chirps_mm": tot(c), "pardig_mm": tot(p)}
            row["ratio_total"] = row["pardig_mm"] / max(row["chirps_mm"], 1e-6)
            for t in (1, 10, 20):
                row[f"chirps_ge{t}"] = frac(c, t); row[f"pardig_ge{t}"] = frac(p, t)
                row[f"ratio_ge{t}"] = row[f"pardig_ge{t}"] / max(row[f"chirps_ge{t}"], 1e-9)
            row["chirps_wet_p95"] = float(np.percentile(c[c >= 1], 95)) if (c >= 1).sum() > 50 else np.nan
            row["pardig_wet_p95"] = float(np.percentile(p[p >= 1], 95)) if (p >= 1).sum() > 50 else np.nan
            rp, rc = r10p[:, m], r10c[:, m]
            for t in (25, 50, 100):
                row[f"chirps_10d_ge{t}"] = frac(rc, t); row[f"pardig_10d_ge{t}"] = frac(rp, t)
                row[f"ratio_10d_ge{t}"] = row[f"pardig_10d_ge{t}"] / max(row[f"chirps_10d_ge{t}"], 1e-9)
                row[f"csi_10d_ge{t}"] = csi(rp, rc, t)
            row["csi_day_ge10"] = csi(p, c, 10)
            rows.append(row)
    T = pd.DataFrame(rows)
    T.round(4).to_csv(os.path.join(TB, "eth_pardig_chirps_short_rains.csv"), index=False)

    # ---- ratio for the forecast's calendar weeks: run date -15 .. +30 days (covers the 15 forecast days), 3 years pooled
    doy0 = pd.Timestamp(init).dayofyear
    sel = D.year.isin(SEASONS) & (D.dayofyear >= doy0 - 15) & (D.dayofyear <= doy0 + 30)
    ratio, cmm = {}, {}
    for name, m in regions:
        p, c = P[sel][:, m].sum(0), C[sel][:, m].sum(0); ww = w[m]
        ratio[name] = round(float((p * ww).sum() / max((c * ww).sum(), 1e-6)), 3)
        cmm[name] = round(float((c * ww).sum() / ww.sum() / sel.sum()), 2)
    json.dump({"window": f"{(pd.Timestamp(init) - pd.Timedelta(days=15)):%d %b} - {(pd.Timestamp(init) + pd.Timedelta(days=30)):%d %b}",
               "years": SEASONS, "ratio_pardig_over_chirps": ratio, "chirps_mm_per_day": cmm,
               "note": "area-weighted total-rain ratio; CHIRPS v3 DAILY_SAT on PARDIG's 0.1 deg grid"},
              open(RATIO, "w"), indent=1)
    print("forecast-weeks ratio PARDIG/CHIRPS:", ratio)

    # ---- console summary
    pd.set_option("display.width", 200)
    print(T[T.season == "pooled"][["region", "cells", "chirps_mm", "pardig_mm", "ratio_total", "ratio_ge10", "ratio_ge20",
                                    "chirps_wet_p95", "pardig_wet_p95", "ratio_10d_ge25", "ratio_10d_ge50", "csi_10d_ge25", "csi_10d_ge50"]].round(2).to_string(index=False))

    # ---- chart: ratio dot plots + national tail curve
    INK, INK2, MUTED, AXIS = wn.INK, wn.INK2, wn.MUTED, wn.AXIS
    C_CH, C_PD = wn.C_FC, wn.C_OBS            # blue = CHIRPS, orange = PARDIG (same slots as 09_)
    Tp = T[T.season == "pooled"].set_index("region")
    big = [n for n in Tp.index if Tp.loc[n, "cells"] >= MIN_CELLS and n != "Ethiopia"]
    order = ["Ethiopia"] + sorted(big, key=lambda n: -Tp.loc[n, "chirps_mm"])
    fig = plt.figure(figsize=(15, 6.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.25], wspace=0.18, left=0.1, right=0.98, top=0.8, bottom=0.17)
    panels = [("ratio_total", "Total rain"), ("ratio_ge20", "Days ≥ 20 mm"), ("ratio_10d_ge50", "10-day totals ≥ 50 mm")]
    yy = np.arange(len(order))[::-1]
    for k, (col, ttl) in enumerate(panels):
        ax = fig.add_subplot(gs[k])
        ax.axvline(1, color=INK2, lw=0.9, zorder=1)
        for j, n in enumerate(order):
            vs = [T[(T.region == n) & (T.season == s)][col].iloc[0] for s in SEASONS]
            ax.scatter(np.clip(vs, 0.2, 5), [yy[j]] * 3, s=22, color="#c3c2b7", zorder=2, linewidths=0)
            v = Tp.loc[n, col]
            ax.scatter([np.clip(v, 0.2, 5)], [yy[j]], s=60, color=INK if n == "Ethiopia" else INK2, zorder=3,
                       edgecolors=wn.SURF, linewidths=1.5)
            ax.text(4.4, yy[j], f"{v:.2f}", va="center", ha="right", fontsize=8.5, color=INK2)
        ax.set_xscale("log"); ax.set_xlim(0.25, 4.5)
        ax.set_xticks([0.25, 0.5, 1, 2, 4]); ax.set_xticklabels(["¼", "½", "1", "2", "4"])
        ax.minorticks_off(); ax.set_xlim(0.25, 4.5)
        ax.set_yticks(yy); ax.set_yticklabels(order if k == 0 else [], fontsize=9)
        ax.grid(axis="y", visible=False); ax.set_ylim(-0.7, len(order) - 0.3)
        ax.set_title(ttl, loc="left", fontsize=11, color=INK, fontweight="semibold", pad=4)
        ax.set_xlabel("PARDIG ÷ CHIRPS", fontsize=9)
        if k == 0:
            ax.get_yticklabels()[0].set_fontweight("semibold")
    # national exceedance curve
    ax = fig.add_subplot(gs[3])
    sel = D.year.isin(SEASONS) & D.month.isin([10, 11, 12])
    xs = np.arange(1, 81)
    for arr, col, lab in ((C[sel][:, mask], C_CH, "CHIRPS"), (P[sel][:, mask], C_PD, "PARDIG")):
        f = np.array([(arr >= x).mean() for x in xs]) * 100
        ax.plot(xs, f, color=col, lw=2)
        i = 55; ax.text(xs[i] + 1, f[i] * 1.25, lab, color=INK2, fontsize=9, fontweight="semibold")
    ax.set_yscale("log"); ax.set_xlim(1, 80)
    ax.set_yticks([0.01, 0.1, 1, 10, 50]); ax.set_yticklabels(["0.01%", "0.1%", "1%", "10%", "50%"])
    ax.set_xlabel("daily rain, mm", fontsize=9)
    ax.set_title("Share of cell-days at least this wet", loc="left", fontsize=11, color=INK, fontweight="semibold", pad=4)
    fig.text(0.1, 0.94, "PARDIG vs CHIRPS over the Oct–Dec short rains, 2023–2025",
             fontsize=15, fontweight="bold", color=INK, ha="left")
    pn = Tp.loc["Ethiopia"]
    fig.text(0.1, 0.885, f"Ethiopia: PARDIG has {pn.ratio_total:.2f}× CHIRPS's rain, {pn.ratio_ge20:.2f}× its days ≥ 20 mm and "
             f"{pn.ratio_10d_ge50:.2f}× its 10-day totals ≥ 50 mm.  Dark dot: three seasons pooled; grey: each season.  "
             f"1 = same.", fontsize=10, color=INK2, ha="left")
    import textwrap
    wn.foot(fig, textwrap.fill("PARDIG: Google's 15-min precipitation analysis summed to UTC days. CHIRPS v3 daily (UCSB-CHC/CHIRPS/V3/DAILY_SAT, "
                 "gauge-blended, days split with IMERG), averaged to PARDIG's 0.1° grid. Cell-days in Ethiopia, Oct–Dec of 2023, "
                 "2024, 2025; area-weighted. Regions: geoBoundaries ADM1 (Addis Ababa, Dire Dawa, Hareri too small to show; in the table).", 210))
    wn.save(fig, os.path.join(wn.OUT, "ETH_pardig_chirps_short_rains.png"))
    print("saved chart and table")
