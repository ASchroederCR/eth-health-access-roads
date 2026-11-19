"""
23_verify_wn3_pardig.py -- how well did WeatherNext 3 forecast early-November 2026 rainfall in Ethiopia, judged against
Google PARDIG (observed, 0.1 deg, UTC days)? WeatherNext 2 shown alongside where its runs are cached.

Inputs
  data/raw/weathernext3/wn3_eth_202611DD00.npz   WN3 00Z runs (21_), daily mean / p50 / p90 on the PARDIG 0.1 deg grid
  data/raw/weathernext/wn2_eth_2026110100.npz    WN2 1 Nov 00Z run, 64 members at 0.25 deg (11_)
  PARDIG daily via 11_weathernext_forecast.load_pardig() (Earth Engine year + GCS near-real-time top-up, 10_)

Measures
  * 1 Nov run vs observed 1-15 Nov: national daily rain, 15-day maps, pattern correlation, bias.
  * Lead-time skill, all cached 00Z runs, every forecast day whose date PARDIG covers: national bias ratio
    (forecast / observed), cell correlation of daily totals, critical success index for >= 1 mm and >= 10 mm days.
  * Hazard check: the calibrated forecast rain classes (14_ hazard(), wet-end thresholds from 22_) vs the same rule
    applied to what PARDIG observed (a day >= 10 / 20 mm, 3 days >= 20 / 40 mm, 15-day >= 10 mm), for runs whose
    15 days PARDIG fully covers. WN2 member rule on its 1 Nov run for comparison.
  * Wet-end coverage: share of cells where the observed wettest day <= the forecast wettest-day wet end (p90).

Outputs
  charts/ETH_wn3_verify_nov.png      1 Nov run: national daily series, 15-day maps
  charts/ETH_wn3_verify_skill.png    lead-time bias / correlation / CSI, hazard hit rate and false alarms
  data/processed/tables/eth_wn3_verify_leadtime.csv, eth_wn3_verify_hazard.csv

    python scripts\\23_verify_wn3_pardig.py
"""
import glob, importlib.util, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap


def load(name, file):
    s = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", file))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wn = load("wn", "11_weathernext_forecast.py")
wn3 = load("wn3", "21_weathernext3_forecast.py")
rr = load("rr", "14_roads_rain_risk.py")
cal = load("cal", "22_calibrate_wn3_hazard.py")
INK, INK2, MUTED, SURF = wn.INK, wn.INK2, wn.MUTED, wn.SURF
C_OBS, C_WN3, C_WN2 = "#52514e", "#c95a1c", "#2a78d6"
SEQ = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
LV = [0, 2, 5, 10, 25, 50, 100, 200, 1e4]


def obs_hazard(o):
    """Same rule as the forecast, on observed daily totals [day, y, x]."""
    t15, dmax = o.sum(0), o.max(0)
    r3 = np.stack([o[i:i + 3].sum(0) for i in range(o.shape[0] - 2)]).max(0)
    h = np.zeros(t15.shape, int); h[t15 >= 10] = 1
    h[(dmax >= 10) | (r3 >= 20)] = 2; h[(dmax >= 20) | (r3 >= 40)] = 3
    return h


def csi(f, o, t):
    a, b = f >= t, o >= t
    hit = (a & b).sum(); return hit / max((a | b).sum(), 1)


if __name__ == "__main__":
    P, pmask = wn.load_pardig()
    P = P.sel(time=slice("2026-10-25", None)).load()
    pdays = pd.to_datetime(P.time.values)
    obs_end = pdays[-1]
    # PARDIG frames sanity: only keep complete days (the GCS top-up file carries frame counts)
    print(f"PARDIG daily {pdays[0]:%d %b} .. {obs_end:%d %b %Y}; grid {P.shape[1:]}")
    runs = sorted(glob.glob(os.path.join(wn3.RAW, "wn3_eth_202611??00.npz")))
    runs = [r for r in runs if pd.Timestamp(os.path.basename(r)[8:16]) <= obs_end]
    lat = P.latitude.values
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones(P.shape[1:])

    def nat(a, m): return float((a[m] * w[m]).sum() / w[m].sum())

    rows, hz_rows = [], []
    for f in runs:
        R = wn3.load_run(None) if False else dict(np.load(f))
        init = pd.Timestamp(str(R["init"])); m3 = R["mask"] & pmask
        assert abs(float(R["x0"]) + 0.05 - float(P.longitude.values[0])) < 1e-6, "grid mismatch"
        fm = np.nan_to_num(R["mean"]); fp90 = np.nan_to_num(R["p90"])
        for d in range(fm.shape[0]):
            day = init + pd.Timedelta(days=d)
            if day > obs_end: break
            o = np.nan_to_num(P.sel(time=day).values)
            fo, oo = fm[d][m3], o[m3]
            rows.append({"run": f"{init:%Y-%m-%d}", "lead_day": d + 1, "date": f"{day:%Y-%m-%d}",
                         "obs_nat_mm": round(nat(o, m3), 2), "fc_nat_mm": round(nat(fm[d], m3), 2),
                         "corr": round(float(np.corrcoef(fo, oo)[0, 1]), 3) if fo.std() > 0 and oo.std() > 0 else np.nan,
                         "csi_1mm": round(csi(fo, oo, 1), 3), "csi_10mm": round(csi(fo, oo, 10), 3),
                         "obs_share_ge10": round(float((oo >= 10).mean()), 4), "fc_share_ge10": round(float((fo >= 10).mean()), 4),
                         "p90_share_ge10": round(float((fp90[d][m3] >= 10).mean()), 4)})
        if init + pd.Timedelta(days=14) <= obs_end:          # full 15 days observed -> hazard check
            o15 = np.nan_to_num(P.sel(time=slice(init, init + pd.Timedelta(days=14))).values)
            ho = obs_hazard(o15)
            hf, _ = rr.hazard(R)
            for k, lab in [(2, "moderate+"), (3, "high")]:
                a, b = (hf >= k) & m3, (ho >= k) & m3
                hz_rows.append({"run": f"{init:%Y-%m-%d}", "model": "WN3 calibrated", "class": lab,
                                "fc_share": round(float(a.sum() / m3.sum()), 4), "obs_share": round(float(b.sum() / m3.sum()), 4),
                                "hit_rate": round(float((a & b).sum() / max(b.sum(), 1)), 3),
                                "false_alarm_ratio": round(float((a & ~b).sum() / max(a.sum(), 1)), 3)})
            cover = float((o15.max(0)[m3] <= fp90.max(0)[m3]).mean())
            hz_rows.append({"run": f"{init:%Y-%m-%d}", "model": "WN3 wet end", "class": "obs wettest day <= p90 wettest day",
                            "fc_share": np.nan, "obs_share": np.nan, "hit_rate": round(cover, 3), "false_alarm_ratio": np.nan})
    L = pd.DataFrame(rows); L.to_csv(os.path.join(wn.TB, "eth_wn3_verify_leadtime.csv"), index=False)

    # WN2 1 Nov run (0.25 deg): national series + member-rule hazard vs observed (observed aggregated to 0.25)
    z2 = np.load(os.path.join(wn.RAW, "wn2_eth_2026110100.npz")); a2 = z2["precip"]; m2 = z2["mask"]
    lat2 = wn.lat; w2 = np.cos(np.deg2rad(lat2))[:, None] * np.ones(m2.shape)
    wn2_nat = np.array([(a2[:, d][:, m2] * w2[m2]).sum(1) / w2[m2].sum() for d in range(15)]).T     # [member, day]
    o15 = np.nan_to_num(P.sel(time=slice("2026-11-01", "2026-11-15")).values)
    # PARDIG to 0.25: mean of 0.1 cells whose centres fall in each 0.25 cell
    plon, plat = P.longitude.values, P.latitude.values
    ci = ((plon - wn.X0) / wn.RES).astype(int); ri = ((wn.Y0 - plat) / wn.RES).astype(int)
    o25 = np.zeros((15,) + m2.shape); cnt = np.zeros(m2.shape)
    okr, okc = (ri >= 0) & (ri < wn.NY), (ci >= 0) & (ci < wn.NX)
    for i in np.flatnonzero(okr):
        for j in np.flatnonzero(okc):
            o25[:, ri[i], ci[j]] += o15[:, i, j]; cnt[ri[i], ci[j]] += 1
    o25 = o25 / np.maximum(cnt, 1)
    h2, ho2 = cal.wn2_member_classes(a2), obs_hazard(o25)
    for k, lab in [(2, "moderate+"), (3, "high")]:
        a, b = (h2 >= k) & m2, (ho2 >= k) & m2
        hz_rows.append({"run": "2026-11-01", "model": "WN2 members", "class": lab, "fc_share": round(float(a.sum() / m2.sum()), 4),
                        "obs_share": round(float(b.sum() / m2.sum()), 4), "hit_rate": round(float((a & b).sum() / max(b.sum(), 1)), 3),
                        "false_alarm_ratio": round(float((a & ~b).sum() / max(a.sum(), 1)), 3)})
    HZ = pd.DataFrame(hz_rows); HZ.to_csv(os.path.join(wn.TB, "eth_wn3_verify_hazard.csv"), index=False)

    # summaries
    print(L.groupby("lead_day").agg(n=("run", "size"), obs=("obs_nat_mm", "sum"), fc=("fc_nat_mm", "sum"),
                                    corr=("corr", "mean"), csi1=("csi_1mm", "mean"), csi10=("csi_10mm", "mean"))
          .assign(ratio=lambda d: (d.fc / d.obs).round(2)).round(3).to_string())
    print(HZ.to_string(index=False))

    # --- chart 1: 1 Nov run ----------------------------------------------------------------------------
    R1 = dict(np.load(os.path.join(wn3.RAW, "wn3_eth_2026110100.npz"))); m3 = R1["mask"] & pmask
    days = pd.date_range("2026-11-01", periods=15)
    obs_nat = [nat(o15[d], m3) for d in range(15)]
    f_mean = [nat(np.nan_to_num(R1["mean"][d]), m3) for d in range(15)]
    f_p90 = [nat(np.nan_to_num(R1["p90"][d]), m3) for d in range(15)]
    fig = plt.figure(figsize=(15, 9.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.25], hspace=0.28, wspace=0.05)
    ax = fig.add_subplot(gs[0, :])
    ax.fill_between(days, np.percentile(wn2_nat, 10, 0), np.percentile(wn2_nat, 90, 0), color=C_WN2, alpha=0.13, lw=0,
                    label="WeatherNext 2, 10–90% of members")
    ax.plot(days, wn2_nat.mean(0), color=C_WN2, lw=1.8, label="WeatherNext 2 mean (0.25°)")
    ax.plot(days, f_p90, color=C_WN3, lw=1.2, ls="--", label="WeatherNext 3 wet end (p90, summed hourly)")
    ax.plot(days, f_mean, color=C_WN3, lw=2.2, label="WeatherNext 3 mean (0.1°)")
    ax.plot(days, obs_nat, color=C_OBS, lw=0, marker="o", ms=6, label="PARDIG observed")
    wn.datefmt(ax, 2); ax.set_ylabel("national mean rain, mm/day"); ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, ncol=2)
    tot_o, tot_3, tot_2 = sum(obs_nat), sum(f_mean), float(wn2_nat.mean(0).sum())
    wn.title(ax, "Forecasts from 1 Nov 2026 00Z against what fell",
             f"Ethiopia, UTC days 1–15 Nov · 15-day national total: PARDIG {tot_o:.1f} mm, WeatherNext 3 {tot_3:.1f} mm, WeatherNext 2 {tot_2:.1f} mm")
    cmap = ListedColormap(SEQ); norm = BoundaryNorm(LV, cmap.N)
    ext = [float(P.longitude.values[0]) - 0.05, float(P.longitude.values[-1]) + 0.05, float(lat[-1]) - 0.05, float(lat[0]) + 0.05]
    tot_obs = np.where(pmask, o15.sum(0), np.nan)
    f3 = np.where(R1["mask"], np.nan_to_num(R1["mean"]).sum(0), np.nan)
    f2 = np.where(m2, a2.mean(0).sum(0), np.nan)
    r3c = np.corrcoef(np.log1p(f3[m3]), np.log1p(tot_obs[m3]))[0, 1]
    mm = m2 & (cnt > 0); r2c = np.corrcoef(np.log1p(f2[mm]), np.log1p(o25.sum(0)[mm]))[0, 1]
    panels = [(tot_obs, ext, "PARDIG observed"), (f3, ext, f"WeatherNext 3 mean · pattern r = {r3c:.2f}"),
              (f2, wn.EXT, f"WeatherNext 2 mean · pattern r = {r2c:.2f}")]
    for k, (A, e, t) in enumerate(panels):
        a_ = fig.add_subplot(gs[1, k]); im = a_.imshow(A, extent=e, cmap=cmap, norm=norm, interpolation="nearest")
        wn.draw_outline(a_); wn.mapax(a_); a_.set_title(t, loc="left", fontsize=10, color=INK, pad=3)
    a_ = fig.add_subplot(gs[1, 3])
    ratio = np.where(m3, (f3 + 1) / (tot_obs + 1), np.nan)
    dl = [0, 0.25, 0.5, 0.8, 1.25, 2, 4, 100]; dc = ["#9a3412", "#e0701f", "#f2c38f", "#eeede8", "#b7d3ee", "#5c93d6", "#1d5f8a"]
    a_.imshow(ratio, extent=ext, cmap=ListedColormap(dc), norm=BoundaryNorm(dl, 7), interpolation="nearest")
    wn.draw_outline(a_); wn.mapax(a_); a_.set_title("WeatherNext 3 ÷ PARDIG (15-day)", loc="left", fontsize=10, color=INK, pad=3)
    from matplotlib.patches import Patch
    a_.legend(handles=[Patch(color=c, label=l) for c, l in zip(dc, ["< ¼", "¼–½", "½–0.8", "0.8–1.25", "1.25–2", "2–4", "> 4×"])],
              loc="upper left", bbox_to_anchor=(0, -0.02), fontsize=7.5, frameon=False, ncol=4)
    cax = fig.add_axes([0.13, 0.06, 0.4, 0.014]); cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=LV[1:-1])
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8, length=0); cax.set_title("15-day total, mm", fontsize=8.5, color=INK2, loc="left")
    fig.text(0.01, 0.015, "PARDIG: Google's 15-min precipitation analysis, summed to UTC days. WeatherNext 3: projects/gcp-public-data-weathernext/assets/"
             "weathernext_3_0_0_0p1deg, hourly ensemble mean summed to UTC days; wet end = hourly 90th percentiles summed (an envelope).", fontsize=8, color=MUTED)
    wn.save(fig, os.path.join(wn.OUT, "ETH_wn3_verify_nov.png"))

    # --- chart 2: skill by lead + hazard ------------------------------------------------------------
    G = L.groupby("lead_day").agg(n=("run", "size"), obs=("obs_nat_mm", "sum"), fc=("fc_nat_mm", "sum"),
                                  corr=("corr", "mean"), csi1=("csi_1mm", "mean"), csi10=("csi_10mm", "mean"))
    G["ratio"] = G.fc / G.obs
    # WN2 1 Nov run per lead (single run) for reference
    w2_rows = []
    for d in range(15):
        f_ = a2.mean(0)[d][mm]; o_ = o25[d][mm]
        w2_rows.append({"lead_day": d + 1, "csi1": csi(f_, o_, 1), "csi10": csi(f_, o_, 10)})
    W2 = pd.DataFrame(w2_rows).set_index("lead_day")
    fig, (s1, s2, s3) = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"wspace": 0.5, "width_ratios": [1, 1, 1.1]})
    s1.bar(G.index, G.ratio, color=C_WN3, width=0.7)
    s1.axhline(1, color=INK2, lw=1, ls="--"); s1.set_xlabel("lead day"); s1.set_ylabel("forecast ÷ observed, national")
    for x_, v_, n_ in zip(G.index, G.ratio, G.n): s1.text(x_, v_, f"{v_:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    s1.grid(axis="x", visible=False)
    run_lab = ", ".join(sorted({r[8:10].lstrip("0") for r in L.run.unique()}, key=int))
    wn.title(s1, "Amount: WeatherNext 3 vs PARDIG", f"00Z runs of {run_lab} Nov · 1 = right on average")
    one = L[L.run == "2026-11-01"].set_index("lead_day")
    s2.plot(G.index, G.csi1, color=C_WN3, lw=2.4, marker="o", ms=4, label="WN3, all runs, ≥ 1 mm")
    s2.plot(G.index, G.csi10, color=C_WN3, lw=2.4, ls="--", marker="o", ms=4, label="WN3, all runs, ≥ 10 mm")
    s2.plot(one.index, one.csi_1mm, color=C_WN3, lw=1, alpha=0.7, label="WN3, 1 Nov run only, ≥ 1 mm")
    s2.plot(W2.index, W2.csi1, color=C_WN2, lw=1, alpha=0.9, label="WN2, 1 Nov run only, ≥ 1 mm")
    s2.set_xlabel("lead day"); s2.set_ylabel("critical success index"); s2.set_ylim(0, 1)
    s2.legend(frameon=False, fontsize=8, loc="upper right")
    wn.title(s2, "Where it rains: daily rain / no-rain", "share of cells right when either says rain")
    H3 = HZ[HZ.model == "WN3 calibrated"]
    labs, hr, far = [], [], []
    for (run, cl), d_ in H3.groupby(["run", "class"], sort=False):
        labs.append(f"{run[8:]} Nov\n{cl}"); hr.append(d_.hit_rate.iloc[0]); far.append(d_.false_alarm_ratio.iloc[0])
    H2 = HZ[HZ.model == "WN2 members"]
    for cl in ["moderate+", "high"]:
        d_ = H2[H2["class"] == cl]; labs.append(f"WN2 01 Nov\n{cl}"); hr.append(d_.hit_rate.iloc[0]); far.append(d_.false_alarm_ratio.iloc[0])
    y_ = np.arange(len(labs))[::-1]
    s3.barh(y_ + 0.19, hr, 0.36, color="#3f8f5f", label="hit rate (observed area caught)")
    s3.barh(y_ - 0.19, far, 0.36, color="#b8b6ae", label="false alarm ratio (flagged, not observed)")
    s3.set_yticks(y_, [l.replace(chr(10), " · ") for l in labs], fontsize=8); s3.set_xlim(0, 1); s3.grid(axis="y", visible=False)
    s3.legend(frameon=False, fontsize=8, loc="lower right")
    wn.title(s3, "Rain hazard classes vs what PARDIG gives", "same rule on observed rain; runs with all 15 days observed")
    wn.foot(fig, f"Days up to {obs_end:%d %b}; runs per lead day fall from {G.n.iloc[0]} at day 1 to {G.n.iloc[-1]} at day 15, so the last days are noisy. "
                 "Daily totals on the 0.1° PARDIG grid (WN2 at 0.25°, PARDIG averaged to it). Hazard: forecast = calibrated WN3 rule (wet end 13 / 24 mm) "
                 "or WN2 64-member rule; observed = a day ≥ 10 / 20 mm, 3 days ≥ 20 / 40 mm, 15-day ≥ 10 mm from PARDIG.")
    wn.save(fig, os.path.join(wn.OUT, "ETH_wn3_verify_skill.png"))
