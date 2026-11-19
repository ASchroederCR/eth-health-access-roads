"""
26_monthly_biascorr_wn3.py -- year-round bias correction for WeatherNext 3 daily rain: a quantile mapping for each
calendar month, pooled over every year available, tested leave-one-year-out against PARDIG.

Why: the short-rains mapping (24_) gets the hazard classes right in Belg and Kiremt too, but inflates rain amounts there
1.5-2.5x (25_). A mapping trained on one season of one year is unstable because WN3's bias changes from year to year.

Method
  * Pairs: every cached WN3 00Z run (21_) x PARDIG day (all cached Earth Engine years + GCS top-up), 0.1 deg cells.
  * Mapping for month M uses pairs whose forecast day falls in M-1, M or M+1 (all years), per lead band (1-3, 4-7,
    8-15 days) x ADM1 region; regions with < 3,000 observed wet samples use the national mapping for that month and
    band; a month-band with < 20,000 samples uses the all-months mapping.
  * Each forecast day is corrected with the mapping of the month it falls in.
  * Test, leave-one-year-out: runs started in year Y are corrected with a mapping trained on the other years only, and
    scored against PARDIG with raw WN3 and the short-rains mapping (24_, note it was trained on Oct-Dec 2025 and Nov 2026,
    so it is in-sample for those). Scores by season-year: national forecast/observed, CSI >= 1 / >= 10 mm, and the
    rain hazard classes (plain rule) for runs with all 15 days observed.
  * Final mapping trained on all years -> data/processed/wn3_qm_monthly.npz; 24_.correct() uses it when present.

Outputs
  data/processed/wn3_qm_monthly.npz
  data/processed/tables/eth_wn3_biascorr_monthly_test.csv, eth_wn3_biascorr_monthly_test_hazard.csv
  charts/ETH_wn3_biascorr_monthly.png

    python scripts\\26_monthly_biascorr_wn3.py
"""
import glob, importlib.util, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt


def load(name, file):
    s = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", file))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QMM = os.path.join(ROOT, "data", "processed", "wn3_qm_monthly.npz")
MIN_MB = 20000


def season_of(m):
    return {1: "Bega", 2: "Bega", 3: "Belg", 4: "Belg", 5: "Belg", 6: "Kiremt", 7: "Kiremt", 8: "Kiremt", 9: "Kiremt",
            10: "Short rains", 11: "Short rains", 12: "Short rains"}[m]


if __name__ == "__main__":
    wn = load("wn", "11_weathernext_forecast.py"); wn3 = load("wn3", "21_weathernext3_forecast.py")
    bc = load("bc", "24_bias_correct_wn3.py"); s25 = load("s25", "25_test_biascorr_seasons.py")
    D, pmask, lat = s25.pardig_all(wn)
    files = sorted(glob.glob(os.path.join(wn3.RAW, "wn3_eth_??????????.npz")))
    R0 = dict(np.load(files[0])); MASK = R0["mask"] & pmask
    REG = bc.region_raster(float(R0["x0"]), float(R0["y0"]), float(R0["res"]), int(R0["nx"]), int(R0["ny"]), wn.GEOMS)
    regv = REG[MASK]; NREG = len(wn.GEOMS); nb = len(bc.BANDS)
    w = (np.cos(np.deg2rad(lat))[:, None] * np.ones(pmask.shape))[MASK]

    # --- pairs as vectors over the Ethiopia cells -----------------------------------------------------
    P = []          # (run_idx, year, init, lead, band, valid_month, fvec, ovec)
    runs = []
    for f in files:
        R = dict(np.load(f)); init = pd.Timestamp(str(R["init"])); fm = np.nan_to_num(R["mean"])
        k = len(runs); runs.append((f, init))
        for d in range(fm.shape[0]):
            day = init + pd.Timedelta(days=d)
            if day in D:
                P.append((k, init.year, init, d + 1, bc.band_of(d + 1), day.month, fm[d][MASK].astype("float32"), D[day][MASK]))
    print(f"{len(runs)} runs, {len(P):,} forecast days with observations; years {sorted({p[1] for p in P})}")
    by_month = {m: sum(1 for p in P if p[5] == m) for m in range(1, 13)}
    print("forecast days per valid month:", by_month)

    def fit_monthly(pairs):
        """-> fq, oq [12, band, region(0 = national), Q]"""
        fq = np.zeros((12, nb, NREG + 1, len(bc.Q)), "float32"); oq = np.zeros_like(fq)
        allf = {b: np.concatenate([p[6] for p in pairs if p[4] == b]) for b in range(nb)}
        allo = {b: np.concatenate([p[7] for p in pairs if p[4] == b]) for b in range(nb)}
        allr = {b: np.concatenate([regv for p in pairs if p[4] == b]) for b in range(nb)}
        for m in range(1, 13):
            win = {(m - 2) % 12 + 1, m, m % 12 + 1}
            for b in range(nb):
                sel = [p for p in pairs if p[4] == b and p[5] in win]
                if sum(len(p[6]) for p in sel) < MIN_MB:
                    f_, o_, r_ = allf[b], allo[b], allr[b]
                else:
                    f_ = np.concatenate([p[6] for p in sel]); o_ = np.concatenate([p[7] for p in sel])
                    r_ = np.concatenate([regv for _ in sel])
                fq[m - 1, b, 0], oq[m - 1, b, 0] = np.quantile(f_, bc.Q), np.quantile(o_, bc.Q)
                for r in range(1, NREG + 1):
                    k = r_ == r
                    if (o_[k] >= 1).sum() >= bc.MIN_WET:
                        fq[m - 1, b, r], oq[m - 1, b, r] = np.quantile(f_[k], bc.Q), np.quantile(o_[k], bc.Q)
                    else:
                        fq[m - 1, b, r], oq[m - 1, b, r] = fq[m - 1, b, 0], oq[m - 1, b, 0]
        return fq, oq

    def correct_vec(fvec, band, month, fq, oq):
        out = np.empty_like(fvec)
        for r in np.unique(regv):
            k = regv == r
            out[k] = bc.apply_map(fvec[k], fq[month - 1, band, r], oq[month - 1, band, r])
        return out

    QM_CUR = dict(np.load(bc.QMF))
    def correct_current(fvec, band):
        out = np.empty_like(fvec)
        for r in np.unique(regv):
            k = regv == r
            out[k] = bc.apply_map(fvec[k], QM_CUR["fq"][band, r], QM_CUR["oq"][band, r])
        return out

    def nat(v): return float((v * w).sum() / w.sum())
    def csi(f, o, t): a, b = f >= t, o >= t; return (a & b).sum() / max((a | b).sum(), 1)

    rows, hz = [], []
    years = sorted({p[1] for p in P})
    for Y in years:
        fq, oq = fit_monthly([p for p in P if p[1] != Y])
        test = [p for p in P if p[1] == Y]
        cache = {}
        for (k, yr, init, lead, b, mo, fv, ov) in test:
            vers = {"raw": fv, "current (short rains)": correct_current(fv, b), "monthly, other years": correct_vec(fv, b, mo, fq, oq)}
            season = f"{season_of(init.month)} {init.year if init.month > 2 or season_of(init.month) != 'Bega' else init.year}"
            for name, v in vers.items():
                rows.append({"season": season, "run": k, "lead": lead, "method": name, "obs": nat(ov), "fc": nat(v),
                             "csi1": csi(v, ov, 1), "csi10": csi(v, ov, 10)})
                cache.setdefault((k, name), {})[lead] = v
            cache.setdefault((k, "obs"), {})[lead] = ov
        for (k, name), dd in cache.items():
            if name == "obs" or len(dd) < 15 or len(cache[(k, "obs")]) < 15: continue
            fc = np.stack([dd[l] for l in range(1, 16)])[:, None, :]; ob = np.stack([cache[(k, "obs")][l] for l in range(1, 16)])[:, None, :]
            hf, ho = bc.rule(fc)[0], bc.rule(ob)[0]
            init = runs[k][1]
            for lvl, lab in [(2, "moderate+"), (3, "high")]:
                x_, y_ = hf >= lvl, ho >= lvl
                hz.append({"season": f"{season_of(init.month)} {init.year}", "method": name, "class": lab,
                           "fc_share": x_.mean(), "obs_share": y_.mean(),
                           "hit_rate": (x_ & y_).sum() / max(y_.sum(), 1), "false_alarm_ratio": (x_ & ~y_).sum() / max(x_.sum(), 1)})
        print(f"tested {Y}: {len(test):,} forecast days")
    T = pd.DataFrame(rows); H = pd.DataFrame(hz)
    SA = (T.groupby(["season", "method"]).agg(days=("obs", "size"), obs=("obs", "sum"), fc=("fc", "sum"), csi1=("csi1", "mean"), csi10=("csi10", "mean"))
          .assign(ratio=lambda d: d.fc / d.obs).reset_index())
    HA = H.groupby(["season", "method", "class"])[["fc_share", "obs_share", "hit_rate", "false_alarm_ratio"]].mean().reset_index()
    SA.round(3).to_csv(os.path.join(wn.TB, "eth_wn3_biascorr_monthly_test.csv"), index=False)
    HA.round(3).to_csv(os.path.join(wn.TB, "eth_wn3_biascorr_monthly_test_hazard.csv"), index=False)
    print(SA.round(2).to_string(index=False)); print(HA[HA["class"] == "moderate+"].round(2).to_string(index=False))
    print(HA[HA["class"] == "high"].round(2).to_string(index=False))

    fqA, oqA = fit_monthly(P)
    np.savez_compressed(QMM, fq=fqA, oq=oqA, regions=REG, q=bc.Q, bands=np.array(bc.BANDS), monthly=True,
                        trained_on=np.array([os.path.basename(f) for f, _ in runs]))
    print("wrote", QMM)

    # --- chart ----------------------------------------------------------------------------------------------
    INK2 = wn.INK2
    order = [s for s in ["Bega 2024", "Belg 2024", "Short rains 2025", "Kiremt 2025", "Bega 2026", "Belg 2026", "Kiremt 2026",
                         "Short rains 2026"] if s in set(SA.season)]
    METH = ["raw", "current (short rains)", "monthly, other years"]
    COL = {"raw": "#c95a1c", "current (short rains)": "#9aa19c", "monthly, other years": "#3f8f5f"}
    fig, axs = plt.subplots(3, 1, figsize=(15, 13), gridspec_kw={"hspace": 0.55})
    x = np.arange(len(order)); wd = 0.26
    for ax, col, ttl, sub, ref in [(axs[0], "ratio", "Amount", "forecast ÷ observed, national, all lead days · 1 = right", 1),
                                  (axs[1], "csi10", "Catching 10 mm days", "critical success index, daily 0.1° cells", None)]:
        for i, mth in enumerate(METH):
            v = [SA[(SA.season == s) & (SA.method == mth)][col].iloc[0] for s in order]
            ax.bar(x + (i - 1) * wd, v, wd * 0.92, color=COL[mth], label=mth)
            for xx, vv in zip(x + (i - 1) * wd, v): ax.text(xx, vv, f"{vv:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
        if ref: ax.axhline(ref, color=INK2, ls="--", lw=1)
        ax.set_xticks(x, order); ax.grid(axis="x", visible=False); wn.title(ax, ttl, sub)
    axs[0].legend(frameon=False, fontsize=9, ncol=3, loc="upper left")
    ax = axs[2]; hm = HA[HA["class"] == "moderate+"]
    for i, mth in enumerate(METH):
        hr = [hm[(hm.season == s) & (hm.method == mth)].hit_rate.mean() for s in order]
        fa = [hm[(hm.season == s) & (hm.method == mth)].false_alarm_ratio.mean() for s in order]
        ax.bar(x + (i - 1) * wd, hr, wd * 0.92, color=COL[mth], label=f"{mth}: hit rate")
        ax.scatter(x + (i - 1) * wd, fa, marker="_", s=260, color="#0b0b0b", zorder=3, label="false alarm ratio" if i == 2 else None)
    ax.set_xticks(x, order); ax.set_ylim(0, 1); ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper left")
    wn.title(ax, "Moderate+ rain hazard vs PARDIG", "bars = share of observed area caught; black ticks = false alarm ratio")
    wn.foot(fig, "Monthly: quantile mapping per calendar month (± 1 month, all other years), lead band and region, so each season-year is scored by a "
                 "mapping that never saw it.\nCurrent: the short-rains mapping (Oct–Dec 2025 + early Nov 2026), in-sample for those two seasons. "
                 "WeatherNext 3 00Z runs every 3rd–6th day; PARDIG UTC days on the 0.1° grid.")
    wn.save(fig, os.path.join(wn.OUT, "ETH_wn3_biascorr_monthly.png"))
