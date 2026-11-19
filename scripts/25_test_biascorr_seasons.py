"""
25_test_biascorr_seasons.py -- does the WeatherNext 3 bias correction (24_, trained on the Oct-Dec short rains) hold up in
other seasons, in particular the dry Kiremt (Jun-Sep) of 2026? Compared with a season-specific correction.

Test seasons (00Z runs every 6th day; PARDIG from all cached Earth Engine years + GCS top-up):
  Belg 2024 (Mar-May), Belg 2026, Kiremt 2025 (Jun-Sep), Kiremt 2026
Versions of the forecast daily rain:
  raw        WeatherNext 3 ensemble mean as published
  current    quantile mapping in data/processed/wn3_qm.npz (trained on Oct-Dec 2025 + early Nov 2026)
  seasonal   quantile mapping trained on the same season in the other year (Belg 2024 <-> 2026, Kiremt 2025 <-> 2026),
             same method as 24_ (lead band x region, national fallback)
Scores per season and version: national forecast / observed by lead band, CSI for >= 1 mm and >= 10 mm days, and the
rain hazard classes (plain rule) against PARDIG for runs whose 15 days are all observed: share flagged vs observed,
hit rate, false alarm ratio. Context: PARDIG national rain per month for Kiremt 2023, 2025, 2026 and Belg 2024, 2026.

Outputs
  charts/ETH_wn3_biascorr_seasons.png
  data/processed/tables/eth_wn3_biascorr_seasons.csv, eth_wn3_biascorr_seasons_hazard.csv, eth_pardig_season_context.csv

    python scripts\\25_test_biascorr_seasons.py
"""
import glob, importlib.util, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt


def load(name, file):
    s = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", file))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEASONS = [("Belg 2024", "2024-03-01", "2024-05-31"), ("Belg 2026", "2026-03-01", "2026-05-31"),
           ("Kiremt 2025", "2025-06-01", "2025-09-30"), ("Kiremt 2026", "2026-06-01", "2026-09-30")]
PARTNER = {"Belg 2024": "Belg 2026", "Belg 2026": "Belg 2024", "Kiremt 2025": "Kiremt 2026", "Kiremt 2026": "Kiremt 2025"}
METHODS = ["raw", "current", "seasonal"]
MCOL = {"raw": "#c95a1c", "current": "#3f8f5f", "seasonal": "#2a78d6"}
MLAB = {"raw": "raw WeatherNext 3", "current": "current correction (short rains)", "seasonal": "seasonal correction (other year)"}


def pardig_all(wn):
    """date -> daily PARDIG [y, x] mm from every cached Earth Engine npz plus the GCS top-up files; and the mask."""
    import xarray as xr
    D, mask, lat = {}, None, None
    for f in sorted(glob.glob(os.path.join(wn.PRAW, "eth_daily_*.npz"))):
        e = np.load(f); mask = e["mask"] if mask is None else mask; lat = e["lat"]
        for i, d in enumerate(pd.to_datetime(e["dates"])):
            if np.isfinite(e["pardig"][i]).any(): D.setdefault(d, np.nan_to_num(e["pardig"][i]).astype("float32"))
    for f in sorted(glob.glob(os.path.join(wn.PRAW, "pardig_nrt_daily_*.nc"))):
        if f.endswith("_06z.nc"): continue
        g = xr.open_dataset(f)
        for d, a in zip(pd.to_datetime(g.time.values), g.precipitation.values):
            D.setdefault(d, np.nan_to_num(a).astype("float32"))
    return D, mask, lat


if __name__ == "__main__":
    wn = load("wn", "11_weathernext_forecast.py"); wn3 = load("wn3", "21_weathernext3_forecast.py")
    bc = load("bc", "24_bias_correct_wn3.py")
    D, pmask, lat = pardig_all(wn)
    print(f"PARDIG days available: {len(D):,} ({min(D):%Y-%m-%d} .. {max(D):%Y-%m-%d})")
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones(pmask.shape)
    def nat(a, m): return float((a[m] * w[m]).sum() / w[m].sum())
    def csi(f, o, t): a, b = f >= t, o >= t; return (a & b).sum() / max((a | b).sum(), 1)

    runs = {}
    for name, a, b in SEASONS:
        fs = []
        for f in sorted(glob.glob(os.path.join(wn3.RAW, "wn3_eth_??????????.npz"))):
            t = pd.Timestamp(os.path.basename(f)[8:16])
            if pd.Timestamp(a) <= t <= pd.Timestamp(b): fs.append(f)
        runs[name] = fs
        print(f"{name}: {len(fs)} runs")
    R0 = dict(np.load(runs[SEASONS[0][0]][0]))
    REG = bc.region_raster(float(R0["x0"]), float(R0["y0"]), float(R0["res"]), int(R0["nx"]), int(R0["ny"]), wn.GEOMS)
    QM_CUR = dict(np.load(bc.QMF))

    def pairs_for(files):
        out = []
        for f in files:
            R = dict(np.load(f)); init = pd.Timestamp(str(R["init"])); m = R["mask"] & pmask; fm = np.nan_to_num(R["mean"])
            for d in range(fm.shape[0]):
                day = init + pd.Timedelta(days=d)
                if day in D: out.append((bc.band_of(d + 1), fm[d], D[day], m))
        return out
    QM_SEAS = {}
    for name, _, _ in SEASONS:
        fq, oq, used = bc.fit(pairs_for(runs[PARTNER[name]]), REG, len(wn.GEOMS))
        QM_SEAS[name] = {"fq": fq, "oq": oq, "regions": REG}

    rows, hz = [], []
    for name, a, b in SEASONS:
        for f in runs[name]:
            R = dict(np.load(f)); init = pd.Timestamp(str(R["init"])); m = R["mask"] & pmask
            V = {"raw": np.nan_to_num(R["mean"]), "current": bc.correct(R, QM_CUR), "seasonal": bc.correct(R, QM_SEAS[name])}
            for d in range(15):
                day = init + pd.Timedelta(days=d)
                if day not in D: continue
                o = D[day]
                for k, v in V.items():
                    rows.append({"season": name, "run": f"{init:%Y-%m-%d}", "lead_day": d + 1, "band": bc.band_of(d + 1), "method": k,
                                 "obs": nat(o, m), "fc": nat(v[d], m), "csi1": csi(v[d][m], o[m], 1), "csi10": csi(v[d][m], o[m], 10)})
            days = [init + pd.Timedelta(days=d) for d in range(15)]
            if all(d_ in D for d_ in days):
                ho = bc.rule(np.stack([D[d_] for d_ in days]))
                for k, v in V.items():
                    hf = bc.rule(v)
                    for lvl, lab in [(2, "moderate+"), (3, "high")]:
                        x_, y_ = (hf >= lvl) & m, (ho >= lvl) & m
                        hz.append({"season": name, "run": f"{init:%Y-%m-%d}", "method": k, "class": lab,
                                   "fc_share": x_.sum() / m.sum(), "obs_share": y_.sum() / m.sum(),
                                   "hit_rate": (x_ & y_).sum() / max(y_.sum(), 1), "false_alarm_ratio": (x_ & ~y_).sum() / max(x_.sum(), 1)})
    T = pd.DataFrame(rows); H = pd.DataFrame(hz)
    S = (T.groupby(["season", "method", "band"]).agg(obs=("obs", "sum"), fc=("fc", "sum"), csi1=("csi1", "mean"), csi10=("csi10", "mean"))
         .assign(ratio=lambda d: d.fc / d.obs).reset_index())
    S["band"] = S.band.map({0: "days 1-3", 1: "days 4-7", 2: "days 8-15"})
    SA = (T.groupby(["season", "method"]).agg(obs=("obs", "sum"), fc=("fc", "sum"), csi1=("csi1", "mean"), csi10=("csi10", "mean"))
          .assign(ratio=lambda d: d.fc / d.obs).reset_index())
    HA = H.groupby(["season", "method", "class"])[["fc_share", "obs_share", "hit_rate", "false_alarm_ratio"]].mean().reset_index()
    S.round(3).to_csv(os.path.join(wn.TB, "eth_wn3_biascorr_seasons.csv"), index=False)
    HA.round(3).to_csv(os.path.join(wn.TB, "eth_wn3_biascorr_seasons_hazard.csv"), index=False)
    print(SA.round(3).to_string(index=False))
    print(HA[HA["class"] == "moderate+"].round(3).to_string(index=False))
    print(HA[HA["class"] == "high"].round(3).to_string(index=False))

    # --- context: PARDIG national rain per month in each season-year -----------------------------------
    ctx = []
    for name, a, b in [("Kiremt 2023", "2023-06-01", "2023-09-30"), ("Kiremt 2025", "2025-06-01", "2025-09-30"),
                       ("Kiremt 2026", "2026-06-01", "2026-09-30"), ("Belg 2024", "2024-03-01", "2024-05-31"),
                       ("Belg 2026", "2026-03-01", "2026-05-31")]:
        for mth in pd.period_range(a, b, freq="M"):
            ds = [d for d in pd.date_range(mth.start_time, mth.end_time) if d in D]
            if len(ds) < 0.9 * mth.days_in_month: continue
            ctx.append({"season": name, "month": mth.strftime("%b"), "mm": sum(nat(D[d], pmask) for d in ds) * mth.days_in_month / len(ds)})
    C = pd.DataFrame(ctx); C.round(1).to_csv(os.path.join(wn.TB, "eth_pardig_season_context.csv"), index=False)
    print(C.groupby("season", sort=False).mm.sum().round(0).to_string())

    # --- chart -------------------------------------------------------------------------------------------
    INK2 = wn.INK2
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), gridspec_kw={"hspace": 0.45, "wspace": 0.28})
    sn = [s[0] for s in SEASONS]; x = np.arange(len(sn)); wd = 0.26
    ax = axs[0, 0]
    for i, mth in enumerate(METHODS):
        v = [SA[(SA.season == s) & (SA.method == mth)].ratio.iloc[0] for s in sn]
        ax.bar(x + (i - 1) * wd, v, wd * 0.92, color=MCOL[mth], label=MLAB[mth])
        for xx, vv in zip(x + (i - 1) * wd, v): ax.text(xx, vv, f"{vv:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.axhline(1, color=INK2, ls="--", lw=1); ax.set_xticks(x, sn); ax.set_ylabel("forecast ÷ observed, national (all lead days)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left"); ax.grid(axis="x", visible=False)
    wn.title(ax, "Amount", "1 = right on average")
    ax = axs[0, 1]
    for i, mth in enumerate(METHODS):
        v = [SA[(SA.season == s) & (SA.method == mth)].csi10.iloc[0] for s in sn]
        ax.bar(x + (i - 1) * wd, v, wd * 0.92, color=MCOL[mth])
        for xx, vv in zip(x + (i - 1) * wd, v): ax.text(xx, vv, f"{vv:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(x, sn); ax.set_ylabel("critical success index, ≥ 10 mm days"); ax.grid(axis="x", visible=False)
    wn.title(ax, "Catching 10 mm days", "daily, 0.1° cells, all lead days")
    ax = axs[1, 0]
    hm = HA[HA["class"] == "moderate+"]
    labs, hr, far, share = [], [], [], []
    for s in sn:
        for mth in METHODS:
            r = hm[(hm.season == s) & (hm.method == mth)]
            if not len(r): continue
            labs.append(f"{s} · {mth}"); hr.append(r.hit_rate.iloc[0]); far.append(r.false_alarm_ratio.iloc[0])
            share.append((r.fc_share.iloc[0], r.obs_share.iloc[0]))
    y = np.arange(len(labs))[::-1]
    ax.barh(y + 0.19, hr, 0.36, color="#3f8f5f", label="hit rate"); ax.barh(y - 0.19, far, 0.36, color="#b8b6ae", label="false alarm ratio")
    for yi, (f_, o_) in zip(y, share): ax.text(1.01, yi, f"flags {f_:.0%} vs {o_:.0%}", va="center", fontsize=7.5, color=INK2, transform=ax.get_yaxis_transform())
    ax.set_yticks(y, labs, fontsize=8); ax.set_xlim(0, 1); ax.grid(axis="y", visible=False); ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    wn.title(ax, "Moderate+ rain hazard vs PARDIG", "runs with all 15 days observed")
    ax = axs[1, 1]
    order = ["Jun", "Jul", "Aug", "Sep"]
    for s_, c_, ls_ in [("Kiremt 2023", "#9aa19c", "-"), ("Kiremt 2025", "#5c93d6", "-"), ("Kiremt 2026", "#c95a1c", "-")]:
        d_ = C[C.season == s_].set_index("month").reindex(order)
        ax.plot(order, d_.mm, color=c_, lw=2.2, marker="o", ls=ls_, label=f"{s_}: {d_.mm.sum():.0f} mm")
    ax.set_ylabel("PARDIG national rain, mm per month"); ax.set_ylim(bottom=0); ax.legend(frameon=False, fontsize=9)
    wn.title(ax, "How dry was this summer?", "Kiremt national rainfall, PARDIG")
    wn.foot(fig, "Current correction: quantile mapping trained on Oct–Dec 2025 short rains + early Nov 2026. Seasonal: the same method trained on the same season "
                 "in the other year (Belg 2024 ↔ 2026, Kiremt 2025 ↔ 2026). WeatherNext 3 00Z runs every 6th day; PARDIG UTC days on the 0.1° grid.")
    wn.save(fig, os.path.join(wn.OUT, "ETH_wn3_biascorr_seasons.png"))
