"""
29_calibrate_intensity.py -- thresholds for the intense-rain hazard: when does WeatherNext 3's hourly wet end signal a
real downpour?

Forecast feature (per 0.1 deg cell and forecast day): wettest hour of the hourly 90th percentile (27_).
Observed event: PARDIG wettest hour on that UTC day (28_) >= 10 mm ("intense") or >= 20 mm ("very intense").
For each lead band (days 1-3, 4-7, 8-15) the forecast threshold is chosen so the forecast flags as many cell-days as
were observed (frequency bias 1); its hit rate, false-alarm ratio and CSI are reported.

Writes data/processed/wn3_intensity_thresholds.json (read by 14_) and data/processed/tables/eth_wn3_intensity_calib.csv.

    python scripts\\29_calibrate_intensity.py
"""
import glob, importlib.util, json, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name, file):
    s = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", file))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


OUTJ = os.path.join(ROOT, "data", "processed", "wn3_intensity_thresholds.json")
BANDS = [(1, 3), (4, 7), (8, 15)]
EVENTS = {"intense": 10.0, "very_intense": 20.0}


def daily_max_hour(h):
    """[360 hours, y, x] -> [15 days, y, x] wettest hour per forecast day."""
    return h[:360].reshape(15, 24, *h.shape[1:]).max(1)


if __name__ == "__main__":
    wn3 = load("wn3", "21_weathernext3_forecast.py"); h27 = load("h27", "27_wn3_hourly.py")
    ob = np.load(sorted(glob.glob(os.path.join(ROOT, "data", "raw", "pardig", "pardig_hourly_max_*.npz")))[-1])
    odays = pd.to_datetime(ob["dates"]); M1 = ob["max1h"]; oidx = {d: i for i, d in enumerate(odays)}
    mask = dict(np.load(sorted(glob.glob(os.path.join(wn3.RAW, "wn3_eth_202611??00.npz")))[0]))["mask"]
    F = {b: [] for b in range(3)}; O = {b: [] for b in range(3)}
    for f in sorted(glob.glob(os.path.join(h27.RAW, "wn3h_eth_*.npz"))):
        z = np.load(f); init = pd.Timestamp(str(z["init"]))
        fx = daily_max_hour(z["p90"].astype("float32") / 10)
        for d in range(15):
            day = init + pd.Timedelta(days=d)
            if day not in oidx: continue
            b = next(i for i, (a, e) in enumerate(BANDS) if a <= d + 1 <= e)
            F[b].append(fx[d][mask]); O[b].append(M1[oidx[day]][mask])
    rows, thr = [], {}
    for b, (a, e) in enumerate(BANDS):
        if not F[b]: continue
        f_, o_ = np.concatenate(F[b]), np.concatenate(O[b])
        thr[f"days_{a}_{e}"] = {}
        for name, t in EVENTS.items():
            base = (o_ >= t).mean()
            T = float(np.quantile(f_, 1 - base)) if base > 0 else float("inf")
            fl, ev = f_ >= T, o_ >= t
            hit = (fl & ev).sum() / max(ev.sum(), 1); far = (fl & ~ev).sum() / max(fl.sum(), 1)
            csi_ = (fl & ev).sum() / max((fl | ev).sum(), 1); rnd = base / (2 - base)     # CSI of a random flag at the same rate
            thr[f"days_{a}_{e}"][name] = round(T, 3)
            rows.append({"lead": f"days {a}-{e}", "event": f"wettest hour >= {t:.0f} mm", "cell_days": len(o_),
                         "observed_rate": round(float(base), 4), "forecast_threshold_p90_mm_per_h": round(T, 2),
                         "hit_rate": round(float(hit), 3), "false_alarm_ratio": round(float(far), 3), "csi": round(float(csi_), 3),
                         "csi_random": round(float(rnd), 3)})
    # --- how the map uses it: per forecast window, with ~20 km tolerance ---------------------------------
    from scipy.ndimage import maximum_filter
    wrows = []
    for f in sorted(glob.glob(os.path.join(h27.RAW, "wn3h_eth_*.npz"))):
        z = np.load(f); init = pd.Timestamp(str(z["init"]))
        fx = daily_max_hour(z["p90"].astype("float32") / 10)
        for b, (a, e) in enumerate(BANDS):
            dd = [init + pd.Timedelta(days=d) for d in range(a - 1, e)]
            if not all(d in oidx for d in dd): continue
            T = thr[f"days_{a}_{e}"]["intense"]
            fw = (fx[a - 1:e] >= T).any(0)
            ow = (np.stack([M1[oidx[d]] for d in dd]) >= EVENTS["intense"]).any(0)
            ow_nb = maximum_filter(ow.astype("uint8"), size=5) > 0           # observed within +-2 cells (~20 km)
            fw_nb = maximum_filter(fw.astype("uint8"), size=5) > 0
            m_ = mask
            wrows.append({"lead": f"days {a}-{e}", "run": f"{init:%Y-%m-%d}",
                          "flagged_share": fw[m_].mean(), "observed_share": ow[m_].mean(),
                          "hit_rate_20km": (fw_nb & ow)[m_].sum() / max(ow[m_].sum(), 1),
                          "false_alarm_ratio_20km": (fw & ~ow_nb)[m_].sum() / max(fw[m_].sum(), 1)})
    WT = pd.DataFrame(wrows).groupby("lead").mean(numeric_only=True).round(3)
    print("per window, ~20 km tolerance (what the map shows):"); print(WT.to_string())
    WT.to_csv(os.path.join(ROOT, "data", "processed", "tables", "eth_wn3_intensity_window_skill.csv"))
    T_ = pd.DataFrame(rows); print(T_.to_string(index=False))
    T_.to_csv(os.path.join(ROOT, "data", "processed", "tables", "eth_wn3_intensity_calib.csv"), index=False)
    json.dump({"bands": BANDS, "thresholds": thr, "events_mm_per_h": EVENTS,
               "note": "forecast = wettest hour of WN3 hourly p90 per day; frequency-matched to PARDIG 1-18 Nov 2026"},
              open(OUTJ, "w"), indent=1)
    print("wrote", OUTJ)
