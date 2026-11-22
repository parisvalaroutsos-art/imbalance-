"""Συνθετικά δεδομένα για δοκιμή του κώδικα (ΔΕΝ είναι πραγματικά δεδομένα αγοράς)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic(start="2024-01-01", end="2026-11-22", seed: int = 7, tz: str = "Europe/Athens") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="15min", tz=tz, inclusive="left")
    n = len(idx)
    hour = (idx.hour + idx.minute / 60).values
    doy = idx.dayofyear.values

    # ΑΠΕ: φωτοβολταϊκά + αιολικά (AR(1))
    season = 0.6 + 0.4 * np.sin(2 * np.pi * (doy - 80) / 365)
    solar = np.clip(np.sin(np.pi * (hour - 6) / 13), 0, None) * 5000 * season
    wind = np.empty(n)
    wind[0] = 1500
    eps = rng.normal(0, 40, n)
    for i in range(1, n):
        wind[i] = 1500 + 0.995 * (wind[i - 1] - 1500) + eps[i]
    wind = np.clip(wind, 50, 4500)
    res_act = solar + wind

    load_act = (
        5500
        + 1200 * np.cos(2 * np.pi * (doy - 200) / 365) ** 2
        + 900 * np.sin(np.pi * (hour - 6) / 16).clip(0)
        - 500 * (idx.dayofweek.values >= 5)
        + rng.normal(0, 60, n)
    )

    # Προβλέψεις ΑΔΜΗΕ = πραγματικά + σφάλμα (αυτοσυσχετισμένο ανά ημέρα)
    day_id = (idx.normalize().tz_localize(None) - pd.Timestamp(start)).days.values
    day_err = rng.normal(0, 1, day_id.max() + 1)
    res_err = 0.08 * res_act * day_err[day_id] + rng.normal(0, 120, n)
    load_err = rng.normal(0, 1, day_id.max() + 1)[day_id] * 120 + rng.normal(0, 50, n)
    res_fc = np.clip(res_act - res_err, 0, None)
    load_fc = load_act - load_err

    gas = 35 + 10 * np.sin(2 * np.pi * day_id / 400) + np.cumsum(rng.normal(0, 0.3, day_id.max() + 1))[day_id]
    resid_fc = load_fc - res_fc
    dam = np.clip(2.2 * gas + 0.03 * (resid_fc - 3000) + rng.normal(0, 8, n), -5, 400)
    dam[resid_fc < 800] = np.clip(dam[resid_fc < 800] * 0.2, -5, None)

    # Σφάλμα συστήματος (+ = έλλειμμα): περισσότερο φορτίο/λιγότερες ΑΠΕ από την πρόβλεψη
    sys_short = (load_act - load_fc) - (res_act - res_fc)
    # IDA1 «βλέπει» μέρος του σφάλματος
    ida1 = dam + 0.015 * (0.5 * sys_short + rng.normal(0, 150, n)) + rng.normal(0, 4, n)
    ida2 = ida1 + 0.01 * (0.3 * sys_short) + rng.normal(0, 3, n)
    ida3 = np.where(hour >= 12, ida2 + 0.01 * (0.3 * sys_short) + rng.normal(0, 3, n), np.nan)

    # Αργά μεταβαλλόμενη μεροληψία του συστήματος (ανά εβδομάδα) -> αξία των ιστορικών lags
    week_bias = np.repeat(rng.normal(0, 12, n // (96 * 7) + 2), 96 * 7)[:n]
    week_bias = pd.Series(week_bias).rolling(96 * 14, min_periods=1).mean().values
    stability = rng.normal(0, 1, n)  # π.χ. δείκτης ευστάθειας/επάρκειας γνωστός D-1
    spread = (
        0.06 * sys_short
        + week_bias
        + 6 * stability
        + 10 * (solar > 2500) * -1
        + rng.standard_t(3, n) * 15
    )
    spikes = rng.random(n) < 0.003
    spread[spikes] += rng.choice([-1, 1], spikes.sum()) * rng.uniform(100, 400, spikes.sum())
    imbalance = dam + spread

    df = pd.DataFrame(
        {
            "dam": dam,
            "ida1": ida1,
            "ida2": ida2,
            "ida3": ida3,
            "imbalance": imbalance,
            "res_forecast": res_fc,
            "load_forecast": load_fc,
            "stability": stability,
        },
        index=idx,
    )
    df.index.name = "timestamp"
    return df
