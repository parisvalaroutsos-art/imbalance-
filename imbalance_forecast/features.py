"""Κατασκευή features χωρίς διαρροή πληροφορίας (as-of).

Για κάθε 15λεπτο της ημέρας D:
  * issue_date  = D - issue_lead_days  (προεπιλογή: D-1, ΠΡΙΝ τη DAM)
  * cutoff      = τελευταία ημέρα με δημοσιευμένο imbalance (κανόνας Πέμπτης)
  * stab_cutoff = τελευταία ημέρα με γνωστό ιστορικό ευστάθειας (issue - stab_lag_days)
  * Γνωστά στο issue: πρόβλεψη ΑΔΜΗΕ (ΑΠΕ/φορτίο), DAM (θεωρείται γνωστή), διαθέσιμες IDA.
  * Η ευστάθεια της D ΔΕΝ είναι γνωστή -> προβλέπεται (stab_prob, βλ. model.py) από τα
    «φυσικά» της αίτια (υψηλό μερίδιο ΑΠΕ, χαμηλό καθαρό φορτίο) και το πρόσφατο ιστορικό της.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .availability import published_through
from .config import ALL_STAGE_COLS, Config

SPREAD = "spread"
TARGET = "y"
STAB_FLAG = "stab_flag"
STAB_PROB = "stab_prob"
CURT = "curt"
P_CURT = "p_curt"


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


# ---------------------------------------------------------------- exog (γνωστά στο issue)
def _add_exog(f: pd.DataFrame, df: pd.DataFrame, cfg: Config) -> list[str]:
    cols = []
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

    g = f.groupby("date")
    for name, s in base.items():
        f[name] = s.values
        f[f"{name}_dmean"] = g[name].transform("mean")
        f[f"{name}_dev"] = f[name] - f[f"{name}_dmean"]
        f[f"{name}_ramp1h"] = f.groupby("date")[name].diff(4)
        cols += [name, f"{name}_dmean", f"{name}_dev", f"{name}_ramp1h"]
    g = f.groupby("date")
    if "dam" in f:
        f["dam_dmax"] = g["dam"].transform("max")
        f["dam_dmin"] = g["dam"].transform("min")
        cols += ["dam_dmax", "dam_dmin"]
    # οδηγοί ευστάθειας: πόσο «στριμωγμένο» είναι το σύστημα σε θερμικές μονάδες
    if "resid_load" in f:
        f["resid_load_dmin"] = g["resid_load"].transform("min")
        f["res_share_dmax"] = g["res_share"].transform("max")
        f["hrs_res_share_gt70"] = g["res_share"].transform(lambda s: (s > 0.7).sum() / 4)
        cols += ["resid_load_dmin", "res_share_dmax", "hrs_res_share_gt70"]

    # IDA: μόνο όσες είναι διαθέσιμες στο issue
    ida_last = None
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

    # ενδοημερήσια πληροφορία της ίδιας ημέρας (ISP1/ISP2/ISP3), μόνο όση είναι γνωστή στο στάδιο
    for c in cfg.same_day_cols:
        if c not in df:
            continue
        n = f"sd_{c}"
        f[n] = df[c].values
        f[f"{n}_dmean"] = f.groupby("date")[n].transform("mean")
        cols += [n, f"{n}_dmean"]
        if c == cfg.col_stab or c.startswith("stab_"):
            f[f"{n}_pos"] = (f[n] > cfg.stab_threshold).astype(float).where(f[n].notna())
            f[f"{n}_hrs"] = f.groupby("date")[f"{n}_pos"].transform("sum") / 4
            cols += [f"{n}_pos", f"{n}_hrs"]
    for sfx in ("isp2", "isp3"):
        r, l_ = f"res_{sfx}", f"load_{sfx}"
        if r in cfg.same_day_cols and r in df and l_ in df:
            f[f"sd_resid_{sfx}"] = (df[l_] - df[r]).values
            cols.append(f"sd_resid_{sfx}")
            if "res" in f:  # αναθεώρηση πρόβλεψης ΑΠΕ από το ISP1 (pre-DAM)
                f[f"sd_res_rev_{sfx}"] = f[f"sd_{r}"] - f["res"]
                cols.append(f"sd_res_rev_{sfx}")

    # διαθεσιμότητα μονάδων: η πιο πρόσφατη γνωστή έκδοση (ISP3 > ISP2 > ISP1)
    ua = [k for k in ("ua3", "ua2", "ua1") if f"{k}_thermal" in cfg.same_day_cols and f"{k}_thermal" in df]
    if ua:
        k = ua[0]
        avail = df[f"{k}_thermal"].fillna(0) + df[f"{k}_hydro"].fillna(0)
        f["ua_avail"] = avail.where(df[f"{k}_thermal"].notna()).values
        cols.append("ua_avail")
        if "resid_load" in f:  # περιθώριο: διαθέσιμη ελεγχόμενη ισχύς - καθαρό φορτίο
            f["ua_margin"] = f["ua_avail"] - f["resid_load"]
            f["ua_margin_dmin"] = f.groupby("date")["ua_margin"].transform("min")
            cols += ["ua_margin", "ua_margin_dmin"]
        if k != "ua1" and "ua1_thermal" in df:  # νέες απώλειες μετά το ISP1
            f["ua_rev"] = (df[f"{k}_thermal"] - df["ua1_thermal"]).values
            f["ua_forced_rev"] = (df[f"{k}_n_forced"] - df["ua1_n_forced"]).values
            cols += ["ua_rev", "ua_forced_rev"]

    # επιπλέον στήλες χρήστη (πρέπει να είναι γνωστές στο issue, π.χ. προβλέψεις)
    known = {cfg.col_dam, cfg.col_imb, cfg.col_res, cfg.col_load, cfg.col_stab, cfg.col_curt, *cfg.ida_cols,
             *ALL_STAGE_COLS}
    for c in df.columns:
        if c not in known:
            f[f"x_{c}"] = df[c].values
            cols.append(f"x_{c}")
    return cols


# ---------------------------------------------------------------- lags
def _hourly_pivot(f: pd.DataFrame, col: str) -> pd.DataFrame:
    p = f.pivot_table(index="date", columns="hour", values=col, aggfunc="mean")
    return p.reindex(pd.date_range(p.index.min(), p.index.max(), freq="D"))


def _stack(tables: dict[str, pd.DataFrame], key: str) -> pd.DataFrame:
    """{name: date x hour} -> long frame (key, hour, name...)."""
    parts = []
    for name, t in tables.items():
        s = t.stack(future_stack=True)
        s.name = name
        parts.append(s)
    out = pd.concat(parts, axis=1)
    out.index.names = [key, "hour"]
    return out.reset_index()


def _imbalance_lags(f: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]] | None:
    """Στατιστικά spread/imbalance έως κάθε ημερομηνία c (περιλαμβάνεται), ανά ώρα."""
    if f[SPREAD].notna().sum() == 0:
        return None
    spr = _hourly_pivot(f, SPREAD)
    imb = _hourly_pivot(f, "imbalance")
    pos = (spr > 0).where(spr.notna())
    r7, r28 = spr.rolling(7, min_periods=1), spr.rolling(28, min_periods=3)
    tabs = {
        "spr_h_last": spr,
        "spr_h_m7": r7.mean(),
        "spr_h_m28": r28.mean(),
        "spr_h_med28": r28.median(),
        "spr_h_sd28": r28.std(),
        "spr_h_pos7": pos.rolling(7, min_periods=1).mean(),
        "spr_h_pos28": pos.rolling(28, min_periods=3).mean(),
        "imb_h_m7": imb.rolling(7, min_periods=1).mean(),
    }
    lag_h = _stack(tabs, "cutoff")

    # ημερήσια (scalar ανά cutoff)
    d = f.groupby("date")[SPREAD].agg(["sum", "count"]).reindex(spr.index)
    d2 = (f[SPREAD] ** 2).groupby(f["date"]).sum().reindex(spr.index)
    s7, n7, q7 = d["sum"].rolling(7, 1).sum(), d["count"].rolling(7, 1).sum(), d2.rolling(7, 1).sum()
    s28, n28 = d["sum"].rolling(28, 1).sum(), d["count"].rolling(28, 1).sum()
    daily = pd.DataFrame({"spr_m7": s7 / n7, "spr_sd7": np.sqrt((q7 / n7 - (s7 / n7) ** 2).clip(0)), "spr_m28": s28 / n28})
    # τρέχον «κατώτατο» imbalance (αλλάζει με τους κανόνες της αγοράς) και χαμηλό ποσοστημόριο
    dmin = f.groupby("date")["imbalance"].min().reindex(spr.index)
    daily["imb_min28"] = dmin.rolling(28, 1).min()
    daily["imb_q05_28"] = f.groupby("date")["imbalance"].quantile(0.05).reindex(spr.index).rolling(28, 1).median()
    # spread υπό συνθήκη ευστάθειας (28 ημ.): πόσο «πέφτει» το imbalance όταν ο ΑΔΜΗΕ κάνει ευστάθεια
    if STAB_FLAG in f and f[STAB_FLAG].notna().any():
        fl = f[STAB_FLAG]
        ok = f[SPREAD].notna() & fl.notna()
        a = pd.DataFrame(
            {
                "s1": (f[SPREAD] * fl).where(ok),
                "n1": fl.where(ok),
                "s0": (f[SPREAD] * (1 - fl)).where(ok),
                "n0": (1 - fl).where(ok),
            }
        ).groupby(f["date"]).sum().reindex(spr.index).rolling(28, 1).sum()
        daily["spr_stab_m28"] = a["s1"] / a["n1"].replace(0, np.nan)
        daily["spr_nostab_m28"] = a["s0"] / a["n0"].replace(0, np.nan)
    # περικοπή ΑΔΜΗΕ (Curt_Flag): συχνότητα ανά ώρα και P(περικοπή | πλεόνασμα ISP), 28 ημ.
    curt_tabs = {}
    if CURT in f and f[CURT].notna().any():
        cp = _hourly_pivot(f, CURT).reindex(spr.index)
        curt_tabs["curt_h_m28"] = cp.rolling(28, min_periods=3).mean()
        cd = f.groupby("date")[CURT].mean().reindex(spr.index)
        daily["curt_d_m28"] = cd.rolling(28, 3).mean()
        if STAB_FLAG in f:
            ok = f[CURT].notna() & f[STAB_FLAG].notna()
            a = pd.DataFrame({"cs": (f[CURT] * f[STAB_FLAG]).where(ok), "s": f[STAB_FLAG].where(ok)})
            a = a.groupby(f["date"]).sum().reindex(spr.index).rolling(28, 1).sum()
            daily["curt_given_stab_m28"] = a["cs"] / a["s"].replace(0, np.nan)
    if curt_tabs:
        lag_h = lag_h.merge(_stack(curt_tabs, "cutoff"), on=["cutoff", "hour"], how="left")
    daily.index.name = "cutoff"
    daily = daily.reset_index()

    # ίδια ημέρα εβδομάδας: μέσος των 4 τελευταίων ίδιων ημερών <= c, για κάθε dow
    rows = []
    for dow in range(7):
        sub = spr[spr.index.dayofweek == dow].rolling(4, min_periods=1).mean()
        sub = sub.reindex(spr.index).ffill(limit=6)  # για κάθε c: τελευταία ίδια ημέρα <= c
        t = _stack({"spr_dh_m4": sub}, "cutoff")
        t["dow"] = dow
        rows.append(t)
    lag_dh = pd.concat(rows, ignore_index=True)
    names = list(tabs) + list(curt_tabs) + [c for c in daily.columns if c != "cutoff"] + ["spr_dh_m4"]
    return lag_h.merge(daily, on="cutoff", how="left"), lag_dh, names


def _stab_lags(f: pd.DataFrame) -> tuple[pd.DataFrame, list[str]] | None:
    """Συχνότητα/ένταση ευστάθειας έως κάθε ημερομηνία c, ανά ώρα."""
    if STAB_FLAG not in f or f[STAB_FLAG].notna().sum() == 0:
        return None
    fl = _hourly_pivot(f, STAB_FLAG)
    mw = _hourly_pivot(f, "stab_mw")
    day = fl.mean(axis=1)
    tabs = {
        "stab_h_last": fl,
        "stab_h_m7": fl.rolling(7, min_periods=1).mean(),
        "stab_h_m28": fl.rolling(28, min_periods=3).mean(),
        "stab_mw_h_m7": mw.rolling(7, min_periods=1).mean(),
    }
    t = _stack(tabs, "stab_cutoff")
    daily = pd.DataFrame({"stab_d_last": day, "stab_d_m7": day.rolling(7, 1).mean(), "stab_d_m28": day.rolling(28, 3).mean()})
    daily.index.name = "stab_cutoff"
    t = t.merge(daily.reset_index(), on="stab_cutoff", how="left")
    return t, list(tabs) + list(daily.columns)


# ---------------------------------------------------------------- κύρια συνάρτηση
def build_features(
    df: pd.DataFrame,
    cfg: Config,
    known_through: pd.Timestamp | None = None,
    stab_known_through: pd.Timestamp | None = None,
):
    """Επιστρέφει (frame, features κύριου μοντέλου, features μοντέλου ευστάθειας)."""
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
    f = f.copy()  # συμπαγές DataFrame μετά από πολλές προσθήκες στηλών
    if "dam" not in f:
        f["dam"] = df[cfg.col_dam].values if cfg.col_dam in df else np.nan

    # ---- στόχοι / ιστορικά (ΔΕΝ μπαίνουν ως features της ίδιας γραμμής)
    has_imb = cfg.col_imb in df and cfg.col_dam in df
    f["imbalance"] = df[cfg.col_imb].values if cfg.col_imb in df else np.nan
    f[SPREAD] = (df[cfg.col_imb] - df[cfg.col_dam]).values if has_imb else np.nan
    f[TARGET] = f[SPREAD] if cfg.target == "spread" else f["imbalance"]
    if cfg.col_curt in df:
        f[CURT] = df[cfg.col_curt].values
    if cfg.col_stab in df:
        f["stab_mw"] = df[cfg.col_stab].values
        f[STAB_FLAG] = (f["stab_mw"] > cfg.stab_threshold).astype(float).where(f["stab_mw"].notna())

    # ---- as-of cutoffs
    issue = f["date"] - pd.Timedelta(days=cfg.issue_lead_days)
    f["cutoff"] = published_through(issue, cfg.pub_weekday, cfg.pub_available_same_day)
    if known_through is not None:
        f["cutoff"] = f["cutoff"].clip(upper=pd.Timestamp(known_through).normalize())
    f["stab_cutoff"] = issue - pd.Timedelta(days=cfg.stab_lag_days)
    if stab_known_through is not None:
        f["stab_cutoff"] = f["stab_cutoff"].clip(upper=pd.Timestamp(stab_known_through).normalize())
    f["gap_days"] = (f["date"] - f["cutoff"]).dt.days

    idx = f.index
    lag_cols = ["gap_days"]
    lags = _imbalance_lags(f) if has_imb else None
    if lags is not None:
        lag_h, lag_dh, names = lags
        f = f.merge(lag_h, on=["cutoff", "hour"], how="left").merge(lag_dh, on=["cutoff", "dow", "hour"], how="left")
        lag_cols += names
        if "ida_last_m_dam" in f:
            f["ida_m_dam_x_pos"] = f["ida_last_m_dam"] * f["spr_h_pos28"]
            lag_cols.append("ida_m_dam_x_pos")

    stab_cols = []
    st = _stab_lags(f)
    if st is not None:
        t, stab_cols = st
        f = f.merge(t, on=["stab_cutoff", "hour"], how="left")
    f.index = idx

    stab_feats = cal_cols + [c for c in exog_cols if not c.startswith("ida")] + stab_cols
    feats = cal_cols + exog_cols + lag_cols + stab_cols
    if st is not None:
        f[STAB_PROB] = np.nan  # συμπληρώνεται από το μοντέλο ευστάθειας
        feats += [STAB_PROB]
        if "curt_given_stab_m28" in f:
            f[P_CURT] = np.nan  # = stab_prob * P(περικοπή | πλεόνασμα)
            feats += [P_CURT]
    if cfg.target_hours is not None:
        h0, h1 = cfg.target_hours
        f = f[(f["hour"] >= h0) & (f["hour"] < h1)]
    return f, feats, stab_feats
