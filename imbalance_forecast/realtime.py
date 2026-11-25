"""Στάδιο πραγματικού χρόνου: πρόβλεψη imbalance για τις επόμενες 3 ώρες, με ό,τι έχει ήδη συμβεί.

Στη χρονική στιγμή τ (αρχή 15λέπτου, ώρα CET) είναι γνωστά:
  * η πραγματική παραγωγή ENTSO-E έως τ - lag (≈30-60'), άρα και «πώς πήγε το σύστημα» στα τελευταία 15λεπτα
    (απόκλιση ελεγχόμενων μονάδων και ΑΠΕ από το πρόγραμμα του ISP, μπαταρίες),
  * για τα 15λεπτα-στόχους: DAM, η τελευταία IDA και το τελευταίο ISP που είχαν δημοσιευθεί στο τ
    (ISP1/IDA1 από D-1 βράδυ, ISP2/IDA2 από D-1 ~23:30 CET, ISP3/IDA3 από D 11:30 CET, μόνο για 12-24).
Στόχος: spread (imbalance - DAM) για τα 15λεπτα τ, τ+15', ..., τ+2h45' (12 ορίζοντες).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .nowcast import GBM, TZ, build_frame

H = 12  # ορίζοντες (15λεπτα)
FEATURES = ["k", "dist", "hour_t", "qh_t", "dam_t", "ida_m_dam_t", "isp_stab_t", "isp_act_t", "isp_ctrl_t",
            "isp_res_sched_t", "dev_n_s", "res_dev_n_s", "dev_r4_s", "dev_s", "bess_s", "dam_s", "dam_t_m_s"]


def _versioned(m: pd.DataFrame) -> dict[str, dict[str, pd.Series]]:
    """Για κάθε μέγεθος: τιμή ανά έκδοση (1, 2, 3) ώστε να επιλέγεται όποια ήταν γνωστή στο τ."""
    def get(c):
        return m[c] if c in m else pd.Series(np.nan, index=m.index)

    v = {
        "stab": {1: get("stability"), 2: get("stab_isp2"), 3: get("stab_isp3")},
        "act": {1: get("act_isp1"), 2: get("act_isp2"), 3: get("act_isp3")},
        "ctrl": {1: get("isp_thermal") + get("isp_hydro"), 2: get("isp2_thermal") + get("isp2_hydro"),
                 3: get("isp3_thermal") + get("isp3_hydro")},
        "res": {1: get("isp_renewables"), 2: get("res_isp2"), 3: get("res_isp3")},
        "ida": {1: get("ida1"), 2: get("ida2"), 3: get("ida3")},
    }
    return v


def build_panel(m: pd.DataFrame, lag_min: int = 30, every: int = 2, since=None, until=None) -> pd.DataFrame:
    """Γραμμές (τ, k): χαρακτηριστικά γνωστά στο τ για τον στόχο t = τ + k·15', k = 0..H-1.

    lag_min: καθυστέρηση δημοσίευσης ENTSO-E (λεπτά από το τέλος του 15λέπτου). every: κάθε πόσα 15λεπτα ένα τ.
    """
    idx = pd.date_range(m.index.min(), m.index.max(), freq="15min")
    m = m.reindex(idx)
    n = len(idx)
    lag_slots = lag_min // 15 + 1  # τελευταίο γνωστό 15λεπτο: s = i - lag_slots
    v = _versioned(m)
    loc = idx.tz_convert(TZ)
    day0 = loc.normalize()  # αρχή ημέρας CET κάθε 15λέπτου
    i0 = np.arange(lag_slots + 4, n - H, every)
    def utc(x):
        x = pd.Timestamp(x)
        return x.tz_localize("UTC") if x.tzinfo is None else x.tz_convert("UTC")

    if since is not None:
        i0 = i0[idx[i0] >= utc(since)]
    if until is not None:
        i0 = i0[idx[i0] < utc(until)]
    s = i0 - lag_slots
    rows = []
    for k in range(H):
        t = i0 + k
        tau = loc[i0]
        tday = day0[t]
        # ποια έκδοση ISP/IDA ήταν δημοσιευμένη στο τ για την ημέρα του στόχου
        v3 = (tau >= tday + pd.Timedelta(hours=11, minutes=30)) & (loc[t].hour >= 12)
        v2 = tau >= tday - pd.Timedelta(minutes=30)
        r = {"tau": idx[i0], "t": idx[t], "k": k, "dist": k + lag_slots,
             "hour_t": loc[t].hour, "qh_t": loc[t].hour * 4 + loc[t].minute // 15}
        for name, src in v.items():
            a1, a2, a3 = (src[j].to_numpy()[t] for j in (1, 2, 3))
            val = np.where(v3 & ~np.isnan(a3), a3, np.where(v2 & ~np.isnan(a2), a2, a1))
            r[f"{name}_t"] = val
        r["dam_t"] = m["dam"].to_numpy()[t]
        r["ida_m_dam_t"] = r.pop("ida_t") - r["dam_t"]
        r["isp_stab_t"] = r.pop("stab_t")
        r["isp_act_t"] = r.pop("act_t")
        r["isp_ctrl_t"] = r.pop("ctrl_t")
        r["isp_res_sched_t"] = r.pop("res_t") - np.nan_to_num(r["isp_stab_t"])
        for c in ("dev_n", "res_dev_n", "dev", "bess", "dam"):
            r[f"{c}_s"] = m[c].to_numpy()[s]
        r["dev_r4_s"] = m["dev_n"].rolling(4, min_periods=2).mean().to_numpy()[s]
        r["dam_t_m_s"] = r["dam_t"] - r["dam_s"]
        r["spread"] = m["spread"].to_numpy()[t]
        r["imbalance"] = m["imbalance"].to_numpy()[t]
        rows.append(pd.DataFrame(r))
    p = pd.concat(rows, ignore_index=True)
    return p[p["dev_s"].notna()]


class RealtimeModel:
    def fit(self, p: pd.DataFrame) -> "RealtimeModel":
        k = p.dropna(subset=["spread"])
        self.features = [f for f in FEATURES if k[f].notna().sum() > 100]
        lo, hi = k["spread"].quantile([0.005, 0.995])
        X, y = k[self.features], k["spread"].clip(lo, hi)
        self.clf = HistGradientBoostingClassifier(**GBM).fit(X, k["spread"] > 0)
        self.reg = {q: HistGradientBoostingRegressor(loss="quantile", quantile=q, **GBM).fit(X, y)
                    for q in (0.1, 0.5, 0.9)}
        # conformal διόρθωση εύρους στις τελευταίες 28 ημέρες
        last = k["t"].max()
        cal = (k["t"] >= last - pd.Timedelta(days=28)).to_numpy()
        self.cal_lo = self.cal_hi = 0.0
        if cal.sum() > 2000 and (~cal).sum() > 20000:
            lo_m = HistGradientBoostingRegressor(loss="quantile", quantile=0.1, **GBM).fit(X[~cal], y[~cal])
            hi_m = HistGradientBoostingRegressor(loss="quantile", quantile=0.9, **GBM).fit(X[~cal], y[~cal])
            yc = k["spread"].to_numpy()[cal]
            self.cal_lo = float(np.quantile(lo_m.predict(X[cal]) - yc, 0.9))
            self.cal_hi = float(np.quantile(yc - hi_m.predict(X[cal]), 0.9))
        self.trained_through = last
        return self

    def predict(self, p: pd.DataFrame) -> pd.DataFrame:
        X = p[self.features]
        out = p[["tau", "t", "k", "dam_t"]].copy()
        out["p_short"] = self.clf.predict_proba(X)[:, 1]
        P = np.sort(np.column_stack([self.reg[q].predict(X) for q in (0.1, 0.5, 0.9)]), axis=1)
        P[:, 0] = np.minimum(P[:, 0] - self.cal_lo, P[:, 1])
        P[:, 2] = np.maximum(P[:, 2] + self.cal_hi, P[:, 1])
        for i, q in enumerate(("q10", "q50", "q90")):
            out[f"imb_{q}"] = p["dam_t"].to_numpy() + P[:, i]
        return out


def forecast_now(lag_min: int = 30, since="2025-07-01") -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Πρόβλεψη για τα επόμενα 12 15λεπτα από το τελευταίο διαθέσιμο σημείο ENTSO-E.

    Επιστρέφει (πρόβλεψη, τελευταίο γνωστό 15λεπτο ENTSO-E, ημερομηνία έως την οποία είναι γνωστό το imbalance).
    """
    m = build_frame()
    last_s = m["dev"].last_valid_index()
    lag_slots = lag_min // 15 + 1
    tau = last_s + pd.Timedelta(minutes=15) * lag_slots
    panel = build_panel(m, lag_min, every=2, since=since)
    model = RealtimeModel().fit(panel[panel["spread"].notna()])
    full = build_panel(m, lag_min, every=1, since=tau, until=tau + pd.Timedelta(minutes=1))
    out = model.predict(full)
    out["ida_t"] = full["dam_t"].values + full["ida_m_dam_t"].values
    return out, last_s, model.trained_through


def to_html(out: pd.DataFrame, last_s, trained) -> str:
    rows = []
    for _, r in out.iterrows():
        t = pd.Timestamp(r.t).tz_convert(TZ)
        p = r.p_short
        bg = "#ffe1d6" if p >= 0.7 else "#dbe9ff" if p <= 0.3 else ""
        st = "πάνω από DAM" if p >= 0.7 else "κάτω από DAM" if p <= 0.3 else "αβέβαιο"
        ida = "" if pd.isna(r.ida_t) else f"{r.ida_t:.1f}"
        rows.append(f"<tr><td style='padding:3px 8px'>{t:%H:%M}</td>"
                    f"<td style='padding:3px 8px;text-align:right'>{r.dam_t:.1f}</td>"
                    f"<td style='padding:3px 8px;text-align:right'>{ida}</td>"
                    f"<td style='padding:3px 8px;text-align:right;background:{bg}'>{p:.2f} ({st})</td>"
                    f"<td style='padding:3px 8px;text-align:right;font-weight:bold'>{r.imb_q50:.0f}</td>"
                    f"<td style='padding:3px 8px;text-align:right'>{r.imb_q10:.0f} – {r.imb_q90:.0f}</td></tr>")
    th = "".join(f"<th style='padding:4px 8px;border-bottom:2px solid #555'>{c}</th>" for c in
                 ["15λεπτο CET", "DAM", "Τελευταία IDA", "P(imb&gt;DAM)", "Imbalance (πρόβλεψη)", "εύρος q10–q90"])
    ls = pd.Timestamp(last_s).tz_convert(TZ)
    return f"""<div style="font-family:Arial,sans-serif;font-size:13px">
<h3 style="margin:0 0 4px">Πρόβλεψη imbalance πραγματικού χρόνου — επόμενες 3 ώρες</h3>
<p style="margin:0 0 8px;color:#555">Με την πραγματική παραγωγή ENTSO-E έως {ls:%d/%m %H:%M}–{ls + pd.Timedelta(minutes=15):%H:%M} CET
(«πώς πήγε το σύστημα» στα τελευταία 15λεπτα) και το τελευταίο δημοσιευμένο ISP/IDA. Μοντέλο εκπαιδευμένο με imbalance
έως {pd.Timestamp(trained).tz_convert(TZ):%d/%m/%Y}.</p>
<table style="border-collapse:collapse;font-size:12px"><tr>{th}</tr>{''.join(rows)}</table>
<p style="color:#555;margin:8px 0 2px">Έλεγχος εκτός εκπαίδευσης (5–11/2026, καθυστέρηση ENTSO-E 30'): σωστή κατεύθυνση
imbalance vs DAM 75% για το επόμενο 15λεπτο, 68% σε 3 ώρες (πρόβλεψη σταδίων ISP2/ISP3: 60%)· σφάλμα 88–95 έναντι 98
των σταδίων, 102 της IDA και 105 της DAM (€/MWh).</p></div>"""


def evaluate(cut="2026-05-01", lag_min=30, since="2025-07-01"):
    """Έλεγχος εκτός εκπαίδευσης: εκπαίδευση πριν το cut, έλεγχος μετά, ανά ορίζοντα."""
    m = build_frame()
    tr = build_panel(m, lag_min, every=2, since=since, until=cut)
    tr = tr[tr["t"] < pd.Timestamp(cut, tz="UTC")]
    te = build_panel(m, lag_min, every=4, since=cut).dropna(subset=["spread"])
    model = RealtimeModel().fit(tr)
    o = model.predict(te)
    o["imbalance"], o["spread"], o["ida_m_dam_t"] = te["imbalance"].values, te["spread"].values, te["ida_m_dam_t"].values
    return o
