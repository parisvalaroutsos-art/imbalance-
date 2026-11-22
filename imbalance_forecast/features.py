"""Κατασκευή features χωρίς διαρροή πληροφορίας (as-of).

Για κάθε 15λεπτο της ημέρας D:
  * issue_date = D - issue_lead_days
  * cutoff     = τελευταία ημέρα με δημοσιευμένο imbalance τη στιγμή του issue
                 (κανόνας Πέμπτης, βλ. availability.py)
  * Features από ADMIE (ΑΠΕ, φορτίο, extra), DAM, διαθέσιμες IDA -> γνωστά στο issue.
  * Ιστορικά features imbalance/spread υπολογίζονται ΜΟΝΟ με δεδομένα <= cutoff.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .availability import published_through
from .config import Config

SPREAD = "spread"
TARGET = "y"


# ---------------------------------------------------------------- αργίες
def _orthodox_easter(year: int) -> pd.Timestamp:
    a, b, c = year % 4, year % 7, year % 19
    d = (19 * c + 15) % 30
    e = (2 * a + 4 * b - d + 34) % 7
    month = (d + e + 114) // 31
    day = (d + e + 114) % 31 + 1
    return pd.Timestamp(year, month, day) + pd.Timedelta(days=13)  # Ιουλιανό -> Γρηγοριανό


def greek_holidays(years) -> set[pd.Timestamp]:
    out = set()
    for y in years:
        for m, d in [(1, 1), (1, 6), (3, 25), (5, 1), (8, 15), (10, 28), (12, 25), (12, 26)]:
            out.add(pd.Timestamp(y, m, d))
        e = _orthodox_easter(y)
        for off in (-48, -2, 0, 1, 50):  # Καθαρά Δευτέρα, Μ. Παρασκευή, Πάσχα, Δευτ. Πάσχα, Αγ. Πνεύματος
            out.add(e + pd.Timedelta(days=off))
    return out


# ---------------------------------------------------------------- βοηθητικά
def _within_day_diff(s: pd.Series, date: pd.Series, periods: int) -> pd.Series:
    return s.groupby(date.values).diff(periods)


def _add_exog(f: pd.DataFrame, df: pd.DataFrame, cfg: Config) -> list[str]:
    cols = []
    date = f["date"]
    res = df.get(cfg.col_res)
    load = df.get(cfg.col_load)
    dam = df.get(cfg.col_dam)

    base = {}
    if res is not None:
        base["res"] = res
    if load is not None:
        base["load"] = load
    if res is not None and load is not None:
        base["resid_load"] = load - res
        base["res_share"] = res / load.replace(0, np.nan)
    if dam is not None:
        base["dam"] = dam

    for name, s in base.items():
        f[name] = s.values
        f[f"{name}_dmean"] = f.groupby("date")[name].transform("mean")
        f[f"{name}_dev"] = f[name] - f[f"{name}_dmean"]
        f[f"{name}_ramp1h"] = _within_day_diff(f[name], date, 4)
        cols += [name, f"{name}_dmean", f"{name}_dev", f"{name}_ramp1h"]
    if "dam" in f:
        f["dam_dmax"] = f.groupby("date")["dam"].transform("max")
        f["dam_dmin"] = f.groupby("date")["dam"].transform("min")
        cols += ["dam_dmax", "dam_dmin"]

    # IDA: μόνο όσες είναι διαθέσιμες στο issue. Το (IDA - DAM) δείχνει προς τα πού
    # κινήθηκαν οι θέσεις των συμμετεχόντων μετά το DAM -> ισχυρό σήμα για την κατεύθυνση.
    ida_last = None  # η πιο πρόσφατη διαθέσιμη IDA τιμή ανά 15λεπτο
    for c in cfg.ida_cols:
        if c in cfg.ida_available and c in df:
            f[c] = df[c].values
            cols.append(c)
            if dam is not None:
                f[f"{c}_m_dam"] = f[c] - f["dam"]
                cols.append(f"{c}_m_dam")
            ida_last = f[c] if ida_last is None else f[c].combine_first(ida_last)
    if ida_last is not None and dam is not None:
        f["ida_last"] = ida_last
        f["ida_last_m_dam"] = f["ida_last"] - f["dam"]
        cols += ["ida_last", "ida_last_m_dam"]

    # επιπλέον στήλες χρήστη (π.χ. ευστάθεια/επάρκεια, εφεδρείες, διασυνδέσεις)
    known = {cfg.col_dam, cfg.col_imb, cfg.col_res, cfg.col_load, *cfg.ida_cols}
    for c in df.columns:
        if c not in known:
            f[f"x_{c}"] = df[c].values
            cols.append(f"x_{c}")
    return cols


def _lag_tables(df: pd.DataFrame, cfg: Config, cutoffs: pd.DatetimeIndex):
    """Στατιστικά imbalance/spread ανά (cutoff, ώρα) και (cutoff, ημέρα εβδ., ώρα)."""
    if cfg.col_imb not in df or cfg.col_dam not in df:
        return None, None
    loc = df.index.tz_convert(cfg.tz)
    h = pd.DataFrame(
        {
            "date": loc.tz_localize(None).normalize(),
            "hour": loc.hour,
            "spr": (df[cfg.col_imb] - df[cfg.col_dam]).values,
            "imb": df[cfg.col_imb].values,
        }
    )
    spr = h.pivot_table(index="date", columns="hour", values="spr", aggfunc="mean")
    imb = h.pivot_table(index="date", columns="hour", values="imb", aggfunc="mean")
    if spr.empty:
        return None, None
    all_days = pd.date_range(spr.index.min(), spr.index.max(), freq="D")
    spr, imb = spr.reindex(all_days), imb.reindex(all_days)
    hours = spr.columns

    rows_h, rows_dh = [], []
    for c in cutoffs.unique():
        w7 = spr.loc[c - pd.Timedelta(days=6): c]
        w28 = spr.loc[c - pd.Timedelta(days=27): c]
        if w28.dropna(how="all").empty:
            continue
        last = spr.loc[c] if c in spr.index else pd.Series(np.nan, index=hours)
        t = pd.DataFrame(
            {
                "cutoff": c,
                "hour": hours,
                "spr_h_last": last.values,
                "spr_h_m7": w7.mean().values,
                "spr_h_m28": w28.mean().values,
                "spr_h_med28": w28.median().values,
                "spr_h_sd28": w28.std().values,
                "spr_h_pos7": (w7 > 0).where(w7.notna()).mean().values,
                "spr_h_pos28": (w28 > 0).where(w28.notna()).mean().values,
                "imb_h_m7": imb.loc[w7.index].mean().values,
                "spr_m7": np.nanmean(w7.values),
                "spr_sd7": np.nanstd(w7.values),
                "spr_m28": np.nanmean(w28.values),
            }
        )
        rows_h.append(t)
        # ίδια ημέρα εβδομάδας, τελευταίες 4 εβδομάδες
        w = spr.loc[c - pd.Timedelta(days=34): c]
        for dow in range(7):
            sub = w[w.index.dayofweek == dow].tail(4)
            rows_dh.append(
                pd.DataFrame({"cutoff": c, "dow": dow, "hour": hours, "spr_dh_m4": sub.mean().values})
            )
    if not rows_h:
        return None, None
    return pd.concat(rows_h, ignore_index=True), pd.concat(rows_dh, ignore_index=True)


def build_features(df: pd.DataFrame, cfg: Config, known_through: pd.Timestamp | None = None):
    """Επιστρέφει (frame, feature_cols). Το frame περιέχει date, cutoff, dam, spread, y."""
    loc = df.index.tz_convert(cfg.tz)
    f = pd.DataFrame(index=df.index)
    f["date"] = loc.tz_localize(None).normalize()
    f["hour"] = loc.hour
    f["qh"] = loc.hour * 4 + loc.minute // 15
    f["dow"] = loc.dayofweek
    f["month"] = loc.month
    doy = loc.dayofyear
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    hol = greek_holidays(range(loc.year.min() - 1, loc.year.max() + 2))
    f["holiday"] = f["date"].isin(hol).astype(int)
    f["offday"] = ((f["dow"] >= 5) | (f["holiday"] == 1)).astype(int)
    cal_cols = ["hour", "qh", "dow", "month", "doy_sin", "doy_cos", "holiday", "offday"]

    exog_cols = _add_exog(f, df, cfg)

    # ---- as-of cutoff
    issue = f["date"] - pd.Timedelta(days=cfg.issue_lead_days)
    f["cutoff"] = published_through(issue, cfg.pub_weekday, cfg.pub_available_same_day)
    if known_through is not None:
        f["cutoff"] = f["cutoff"].clip(upper=pd.Timestamp(known_through).normalize())
    f["gap_days"] = (f["date"] - f["cutoff"]).dt.days

    lag_h, lag_dh = _lag_tables(df, cfg, pd.DatetimeIndex(f["cutoff"]))
    lag_cols = ["gap_days"]
    if lag_h is not None:
        idx = f.index
        f = f.merge(lag_h, on=["cutoff", "hour"], how="left").merge(lag_dh, on=["cutoff", "dow", "hour"], how="left")
        f.index = idx
        lag_cols += [c for c in lag_h.columns if c not in ("cutoff", "hour")] + ["spr_dh_m4"]
        if "ida_last_m_dam" in f:
            f["ida_m_dam_x_pos"] = f["ida_last_m_dam"] * f["spr_h_pos28"]
            lag_cols.append("ida_m_dam_x_pos")

    # ---- στόχος
    if cfg.col_imb in df and cfg.col_dam in df:
        f[SPREAD] = (df[cfg.col_imb] - df[cfg.col_dam]).values
    else:
        f[SPREAD] = np.nan
    f["imbalance"] = df[cfg.col_imb].values if cfg.col_imb in df else np.nan
    f[TARGET] = f[SPREAD] if cfg.target == "spread" else f["imbalance"]
    if "dam" not in f:
        f["dam"] = df[cfg.col_dam].values if cfg.col_dam in df else np.nan

    feats = cal_cols + exog_cols + lag_cols
    return f, feats
