"""
22_calibrate_wn3_hazard.py -- choose WeatherNext 3 wet-end thresholds so the rain hazard classes match the original
WeatherNext 2 member-based rule on runs both models made from the same start time.

WN2 rule (64 members):  high     >= 50% of members have a day >= 20 mm, or ensemble-mean wettest 3 days >= 40 mm
                        moderate >= 50% of members have a day >= 10 mm, or wettest 3 days >= 20 mm
                        low      15-day mean >= 10 mm
WN3 publishes no members, so the member tests are replaced by the wettest day of the wet end (hourly p90 summed per
day, p90max):           high     p90max >= H  or mean wettest 3 days >= 40 mm
                        moderate p90max >= M  or mean wettest 3 days >= 20 mm
                        low      15-day mean >= 10 mm  (unchanged)
M and H are set so that, pooled over the paired runs, the share of Ethiopia at >= moderate and at high equals the
WN2 member rule's share (area-weighted). Agreement cell by cell is reported on the WN2 0.25 deg grid.

Writes data/processed/tables/wn3_hazard_calibration.csv and data/processed/wn3_hazard_thresholds.json (read by 14_).

    python scripts\\22_calibrate_wn3_hazard.py [--runs 2026-11-01 2026-11-16 2026-11-17]
"""
import argparse, importlib.util, json, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("wn3", os.path.join(ROOT, "scripts", "21_weathernext3_forecast.py"))
wn3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(wn3)
OUTJ = os.path.join(ROOT, "data", "processed", "wn3_hazard_thresholds.json")
TB = os.path.join(ROOT, "data", "processed", "tables")


def wn2_member_classes(arr):
    mean15 = arr.sum(1).mean(0); dmax = arr.max(1)
    p20, p10 = (dmax >= 20).mean(0), (dmax >= 10).mean(0)
    r3 = np.stack([arr[:, i:i + 3].sum(1) for i in range(arr.shape[1] - 2)], 1).max(1).mean(0)
    h = np.zeros(mean15.shape, int); h[mean15 >= 10] = 1
    h[(p10 >= .5) | (r3 >= 20)] = 2; h[(p20 >= .5) | (r3 >= 40)] = 3
    return h


def wn3_parts(run):
    m = np.nan_to_num(run["mean"]); p90 = np.nan_to_num(run["p90"])
    r3 = np.stack([m[i:i + 3].sum(0) for i in range(m.shape[0] - 2)]).max(0)
    return m.sum(0), r3, p90.max(0)


def wn3_classes(mean15, r3, p90max, M, H):
    h = np.zeros(mean15.shape, int); h[mean15 >= 10] = 1
    h[(p90max >= M) | (r3 >= 20)] = 2; h[(p90max >= H) | (r3 >= 40)] = 3
    return h


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", default=["2026-11-01", "2026-11-16", "2026-11-17"])
    a = ap.parse_args()
    pairs = []
    for d in a.runs:
        z2 = np.load(os.path.join(ROOT, "data", "raw", "weathernext", f"wn2_eth_{d.replace('-', '')}00.npz"))
        r3 = wn3.load_run(d)
        lat2 = 15.125 - 0.25 * (np.arange(z2["mask"].shape[0]) + 0.5)
        w2 = np.cos(np.deg2rad(lat2))[:, None] * np.ones(z2["mask"].shape)
        lat3 = float(r3["y0"]) - float(r3["res"]) * (np.arange(int(r3["ny"])) + 0.5)
        w3 = np.cos(np.deg2rad(lat3))[:, None] * np.ones(r3["mask"].shape)
        pairs.append(dict(run=d, c2=wn2_member_classes(z2["precip"]), m2=z2["mask"], w2=w2, parts=wn3_parts(r3), m3=r3["mask"], w3=w3, r3=r3))

    def share(c, m, w, k): return float((w[m] * (c[m] >= k)).sum() / w[m].sum())
    t_mod = np.mean([share(p["c2"], p["m2"], p["w2"], 2) for p in pairs])
    t_high = np.mean([share(p["c2"], p["m2"], p["w2"], 3) for p in pairs])
    print(f"WN2 member rule, pooled share of Ethiopia: >= moderate {t_mod:.2%}, high {t_high:.2%}")

    grid = np.arange(5, 120.5, 0.5)
    def pooled(M, H, k): return np.mean([share(wn3_classes(*p["parts"], M, H), p["m3"], p["w3"], k) for p in pairs])
    # moderate first (H out of reach), then high with M fixed
    M = min(grid, key=lambda x: abs(pooled(x, 1e9, 2) - t_mod))
    H = min(grid[grid >= M], key=lambda x: abs(pooled(M, x, 3) - t_high))
    print(f"calibrated: M = {M:.1f} mm, H = {H:.1f} mm (wettest day, hourly p90 summed)")

    rows = []
    for p in pairs:
        c3 = wn3_classes(*p["parts"], M, H); c3_mean = wn3_classes(*p["parts"], 1e9, 1e9)
        # agreement on the WN2 grid: WN3 class at each WN2 cell centre (nearest 0.1 deg cell)
        r3 = p["r3"]; ny2, nx2 = p["m2"].shape
        yy = 15.125 - 0.25 * (np.arange(ny2) + 0.5); xx = 32.875 + 0.25 * (np.arange(nx2) + 0.5)
        ri = np.clip(((float(r3["y0"]) - yy) / float(r3["res"])).astype(int), 0, int(r3["ny"]) - 1)
        ci = np.clip(((xx - float(r3["x0"])) / float(r3["res"])).astype(int), 0, int(r3["nx"]) - 1)
        s3 = c3[np.ix_(ri, ci)]; m = p["m2"]
        rows.append({"run": p["run"],
                     **{f"wn2_share_ge{k}": round(share(p["c2"], p["m2"], p["w2"], k), 4) for k in (1, 2, 3)},
                     **{f"wn3_calibrated_share_ge{k}": round(share(c3, p["m3"], p["w3"], k), 4) for k in (1, 2, 3)},
                     **{f"wn3_mean_only_share_ge{k}": round(share(c3_mean, p["m3"], p["w3"], k), 4) for k in (2, 3)},
                     "cell_agreement_ge2": round(float(((s3[m] >= 2) == (p["c2"][m] >= 2)).mean()), 3)})
    T = pd.DataFrame(rows); print(T.to_string(index=False))
    T.to_csv(os.path.join(TB, "wn3_hazard_calibration.csv"), index=False)
    json.dump({"moderate_p90_day_mm": float(M), "high_p90_day_mm": float(H), "calibrated_on": a.runs,
               "target_share_ge_moderate": round(t_mod, 4), "target_share_high": round(t_high, 4)},
              open(OUTJ, "w"), indent=1)
    print("wrote", OUTJ)
