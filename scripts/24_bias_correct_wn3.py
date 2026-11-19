"""
24_bias_correct_wn3.py -- bias-correct WeatherNext 3 daily rain against PARDIG by quantile mapping, test it on runs it
was not trained on, and save the mapping the road scoring uses (14_).

Why: against PARDIG (23_), WN3's ensemble-mean daily rain in early Nov 2026 was 40-60% too low and almost never reached
10 mm, so the hazard rule under-flagged (it caught 21-53% of the area PARDIG made moderate or higher).

Method
  * Pairs: WN3 00Z run daily ensemble mean (21_) and PARDIG daily total for the same UTC day and 0.1 deg cell.
  * Groups: lead band (days 1-3, 4-7, 8-15) x ADM1 region; a region with < 3,000 wet (>= 1 mm) observed samples in a
    band uses the all-Ethiopia mapping for that band.
  * Mapping: empirical quantiles of forecast and observed values (201 levels, dense in the wet tail); a forecast value
    is replaced by the observed value at the same quantile; above the top quantile it is scaled by obs/fc at the top.
  * Corrected rain feeds the original rain rule directly (a day >= 10 / 20 mm, 3 days >= 20 / 40 mm, 15 days >= 10 mm),
    the rule PARDIG itself is judged by in 23_, so no wet-end thresholds are needed.
  * Test: train on the Oct-Dec 2025 short-rains runs (every 3rd day from 22 Oct 2025; PARDIG starts 22 Oct 2025), apply to
    the Nov 2026 runs (1-10, 16, 17 Nov), compare with raw WN3 and with the wet-end rule (22_).
  * Final mapping: trained on both periods -> data/processed/wn3_qm.npz.

Outputs
  data/processed/wn3_qm.npz                         mapping used by 14_ (bands, region raster, quantile tables)
  data/processed/tables/eth_wn3_biascorr_test.csv   test-period metrics, raw vs corrected
  charts/ETH_wn3_biascorrection.png

    python scripts\\24_bias_correct_wn3.py
"""
import glob, importlib.util, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from rasterio.features import rasterize
from rasterio.transform import from_origin


def load(name, file):
    s = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", file))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QMF = os.path.join(ROOT, "data", "processed", "wn3_qm.npz")
BANDS = [(1, 3), (4, 7), (8, 15)]
Q = np.unique(np.r_[np.linspace(0, 0.9, 91), np.linspace(0.9, 0.99, 46), np.linspace(0.99, 0.9999, 64)])
MIN_WET = 3000


def band_of(lead):
    for i, (a, b) in enumerate(BANDS):
        if a <= lead <= b: return i
    return len(BANDS) - 1


def region_raster(x0, y0, res, nx, ny, geoms):
    return rasterize([(g, i + 1) for i, g in enumerate(geoms)], out_shape=(ny, nx),
                     transform=from_origin(x0, y0, res, res), fill=0, dtype="int16")


def fit(pairs, regions, nreg):
    """pairs: list of (band, fc[y,x], obs[y,x], mask) -> fq, oq arrays [band, region(0 = national), Q]."""
    nb = len(BANDS)
    F = [[[] for _ in range(nreg + 1)] for _ in range(nb)]; O = [[[] for _ in range(nreg + 1)] for _ in range(nb)]
    for b, f, o, m in pairs:
        F[b][0].append(f[m]); O[b][0].append(o[m])
        for r in range(1, nreg + 1):
            k = m & (regions == r)
            if k.any(): F[b][r].append(f[k]); O[b][r].append(o[k])
    fq = np.zeros((nb, nreg + 1, len(Q))); oq = np.zeros_like(fq); used = np.zeros((nb, nreg + 1), bool)
    for b in range(nb):
        for r in range(nreg + 1):
            if not F[b][r]: continue
            f_ = np.concatenate(F[b][r]); o_ = np.concatenate(O[b][r])
            if r > 0 and (o_ >= 1).sum() < MIN_WET: continue
            fq[b, r], oq[b, r] = np.quantile(f_, Q), np.quantile(o_, Q); used[b, r] = True
    for b in range(nb):                                         # fall back to national
        for r in range(1, nreg + 1):
            if not used[b, r]: fq[b, r], oq[b, r] = fq[b, 0], oq[b, 0]
    return fq, oq, used


def apply_map(x, fq, oq):
    xs, idx = np.unique(fq, return_index=True)                # strictly increasing forecast quantiles
    ys = np.maximum.accumulate(oq)[idx]
    y = np.interp(x, xs, ys)
    top = xs[-1]
    return np.where(x > top, x * (ys[-1] / max(top, 1e-6)), np.maximum(y, 0))


def correct(run, qm=None):
    """WN3 run dict -> bias-corrected daily mean [day, y, x] mm (uses data/processed/wn3_qm.npz)."""
    qm = qm if qm is not None else dict(np.load(QMF))
    m = np.nan_to_num(run["mean"]); out = np.zeros_like(m)
    reg = qm["regions"]
    for d in range(m.shape[0]):
        b = band_of(d + 1)
        for r in np.unique(reg):
            k = reg == r
            out[d][k] = apply_map(m[d][k], qm["fq"][b, r], qm["oq"][b, r])
    return out


def rule(daily):
    """Original rain rule on daily totals [day, y, x] -> hazard 0-3."""
    t15, dmax = daily.sum(0), daily.max(0)
    r3 = np.stack([daily[i:i + 3].sum(0) for i in range(daily.shape[0] - 2)]).max(0)
    h = np.zeros(t15.shape, "int8"); h[t15 >= 10] = 1
    h[(dmax >= 10) | (r3 >= 20)] = 2; h[(dmax >= 20) | (r3 >= 40)] = 3
    return h


if __name__ == "__main__":
    wn = load("wn", "11_weathernext_forecast.py"); wn3 = load("wn3", "21_weathernext3_forecast.py")
    rr = load("rr", "14_roads_rain_risk.py")
    P, pmask = wn.load_pardig()
    pdays = pd.to_datetime(P.time.values); obs_end = pdays[-1]
    Pv = P.values.astype("float32"); tidx = {d: i for i, d in enumerate(pdays)}

    def runs_between(a, b):
        out = []
        for f in sorted(glob.glob(os.path.join(wn3.RAW, "wn3_eth_??????????.npz"))):
            t = pd.Timestamp(os.path.basename(f)[8:16])
            if a <= t <= b: out.append(f)
        return out
    train = runs_between(pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31"))
    test = runs_between(pd.Timestamp("2026-11-01"), obs_end)
    print(f"train runs {len(train)} (2025), test runs {len(test)} (Nov 2026); PARDIG {pdays[0]:%Y-%m-%d}..{obs_end:%Y-%m-%d}")
    R0 = dict(np.load(train[0]))
    REG = region_raster(float(R0["x0"]), float(R0["y0"]), float(R0["res"]), int(R0["nx"]), int(R0["ny"]), wn.GEOMS)
    NREG = len(wn.GEOMS)

    def pairs_for(files):
        out = []
        for f in files:
            R = dict(np.load(f)); init = pd.Timestamp(str(R["init"])); m = R["mask"] & pmask
            fm = np.nan_to_num(R["mean"])
            for d in range(fm.shape[0]):
                day = init + pd.Timedelta(days=d)
                if day not in tidx: continue
                out.append((band_of(d + 1), fm[d], np.nan_to_num(Pv[tidx[day]]), m))
        return out

    tr = pairs_for(train)
    fq, oq, used = fit(tr, REG, NREG)
    print("regional mappings used (bands x regions):", used[:, 1:].sum(1).tolist(), "of", NREG)
    qm_train = {"fq": fq, "oq": oq, "regions": REG}

    # --- test on Nov 2026 ------------------------------------------------------------------------
    lat = P.latitude.values; w = np.cos(np.deg2rad(lat))[:, None] * np.ones(P.shape[1:])
    def nat(a, m): return float((a[m] * w[m]).sum() / w[m].sum())
    def csi(f, o, t): a, b = f >= t, o >= t; return (a & b).sum() / max((a | b).sum(), 1)
    import json
    _t = json.load(open(os.path.join(ROOT, "data", "processed", "wn3_hazard_thresholds.json")))
    def rr_wetend(R):
        m_ = np.nan_to_num(R["mean"]); p_ = np.nan_to_num(R["p90"]).max(0)
        t15, r3_ = m_.sum(0), np.stack([m_[i:i + 3].sum(0) for i in range(m_.shape[0] - 2)]).max(0)
        h_ = np.zeros(t15.shape, "int8"); h_[t15 >= 10] = 1
        h_[(p_ >= _t["moderate_p90_day_mm"]) | (r3_ >= 20)] = 2; h_[(p_ >= _t["high_p90_day_mm"]) | (r3_ >= 40)] = 3
        return h_
    rows, hz = [], []
    for f in test:
        R = dict(np.load(f)); init = pd.Timestamp(str(R["init"])); m = R["mask"] & pmask
        raw = np.nan_to_num(R["mean"]); cor = correct(R, qm_train)
        for d in range(raw.shape[0]):
            day = init + pd.Timedelta(days=d)
            if day > obs_end: break
            o = np.nan_to_num(Pv[tidx[day]])
            rows.append({"run": f"{init:%Y-%m-%d}", "lead_day": d + 1, "obs": nat(o, m), "raw": nat(raw[d], m), "cor": nat(cor[d], m),
                         "csi1_raw": csi(raw[d][m], o[m], 1), "csi1_cor": csi(cor[d][m], o[m], 1),
                         "csi10_raw": csi(raw[d][m], o[m], 10), "csi10_cor": csi(cor[d][m], o[m], 10)})
        if init + pd.Timedelta(days=14) <= obs_end:
            o15 = np.nan_to_num(Pv[tidx[init]:tidx[init] + 15]); ho = rule(o15)
            for name, hf in [("raw, wet-end rule", rr_wetend(R)), ("bias-corrected", rule(cor))]:
                for k, lab in [(2, "moderate+"), (3, "high")]:
                    a, b = (hf >= k) & m, (ho >= k) & m
                    hz.append({"run": f"{init:%Y-%m-%d}", "method": name, "class": lab,
                               "fc_share": round(float(a.sum() / m.sum()), 3), "obs_share": round(float(b.sum() / m.sum()), 3),
                               "hit_rate": round(float((a & b).sum() / max(b.sum(), 1)), 3),
                               "false_alarm_ratio": round(float((a & ~b).sum() / max(a.sum(), 1)), 3)})
    T = pd.DataFrame(rows); H = pd.DataFrame(hz)
    G = T.groupby("lead_day").agg(n=("run", "size"), obs=("obs", "sum"), raw=("raw", "sum"), cor=("cor", "sum"),
                                  csi1_raw=("csi1_raw", "mean"), csi1_cor=("csi1_cor", "mean"),
                                  csi10_raw=("csi10_raw", "mean"), csi10_cor=("csi10_cor", "mean"))
    G["ratio_raw"], G["ratio_cor"] = G.raw / G.obs, G.cor / G.obs
    out = G.round(3).reset_index()
    out.to_csv(os.path.join(wn.TB, "eth_wn3_biascorr_test.csv"), index=False)
    H.to_csv(os.path.join(wn.TB, "eth_wn3_biascorr_test_hazard.csv"), index=False)
    print(G[["n", "ratio_raw", "ratio_cor", "csi1_raw", "csi1_cor", "csi10_raw", "csi10_cor"]].round(2).to_string())
    print(H.groupby(["method", "class"])[["fc_share", "obs_share", "hit_rate", "false_alarm_ratio"]].mean().round(3).to_string())

    # --- final mapping on both periods ------------------------------------------------------------
    fq2, oq2, used2 = fit(tr + pairs_for(test), REG, NREG)
    np.savez_compressed(QMF, fq=fq2, oq=oq2, regions=REG, q=Q, bands=np.array(BANDS),
                        trained_on=np.array([os.path.basename(f) for f in train + test]))
    print("wrote", QMF)

    # --- chart ------------------------------------------------------------------------------------------
    INK2, MUTED = wn.INK2, wn.MUTED
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"wspace": 0.62, "width_ratios": [1, 1, 1.15]})
    Gs = G[G.n >= 6]
    a1.plot(Gs.index, Gs.ratio_raw, color="#c95a1c", lw=2, marker="o", ms=4, label="raw WeatherNext 3")
    a1.plot(Gs.index, Gs.ratio_cor, color="#3f8f5f", lw=2, marker="o", ms=4, label="bias-corrected")
    a1.axhline(1, color=INK2, ls="--", lw=1); a1.set_ylim(0, max(1.6, Gs.ratio_cor.max() * 1.1))
    a1.set_xlabel("lead day"); a1.set_ylabel("forecast ÷ observed, national"); a1.legend(frameon=False, fontsize=9)
    wn.title(a1, "Amount", "Nov 2026 runs; mapping trained on Oct–Dec 2025 only")
    a2.plot(Gs.index, Gs.csi1_raw, color="#c95a1c", lw=2, label="raw, ≥ 1 mm"); a2.plot(Gs.index, Gs.csi1_cor, color="#3f8f5f", lw=2, label="corrected, ≥ 1 mm")
    a2.plot(Gs.index, Gs.csi10_raw, color="#c95a1c", lw=2, ls="--", label="raw, ≥ 10 mm"); a2.plot(Gs.index, Gs.csi10_cor, color="#3f8f5f", lw=2, ls="--", label="corrected, ≥ 10 mm")
    a2.set_ylim(0, 0.7); a2.set_xlabel("lead day"); a2.set_ylabel("critical success index"); a2.legend(frameon=False, fontsize=8.5)
    wn.title(a2, "Where it rains", "daily, 0.1° cells")
    S = H.groupby(["method", "class"], sort=False)[["hit_rate", "false_alarm_ratio", "fc_share", "obs_share"]].mean().reset_index()
    y = np.arange(len(S))[::-1]
    a3.barh(y + 0.19, S.hit_rate, 0.36, color="#3f8f5f", label="hit rate"); a3.barh(y - 0.19, S.false_alarm_ratio, 0.36, color="#b8b6ae", label="false alarm ratio")
    for yi, (_, r) in zip(y, S.iterrows()):
        a3.text(1.01, yi, f"flags {r.fc_share:.0%} vs {r.obs_share:.0%} observed", va="center", fontsize=8, color=INK2, transform=a3.get_yaxis_transform())
    a3.set_yticks(y, [f"{m_} · {c_}" for m_, c_ in zip(S.method, S["class"])], fontsize=8.5); a3.set_xlim(0, 1)
    a3.grid(axis="y", visible=False); a3.legend(frameon=False, fontsize=8.5, loc="lower right")
    wn.title(a3, "Rain hazard classes", "runs of 1–4 Nov 2026, against PARDIG")
    wn.foot(fig, "Quantile mapping of WeatherNext 3 daily ensemble-mean rain to PARDIG, per lead band (1–3, 4–7, 8–15 days) and region; trained on 00Z runs "
                 "every 3rd day, 22 Oct – 20 Dec 2025, tested on the Nov 2026 runs it never saw. Lead days with fewer than 6 test runs left out.")
    wn.save(fig, os.path.join(wn.OUT, "ETH_wn3_biascorrection.png"))
