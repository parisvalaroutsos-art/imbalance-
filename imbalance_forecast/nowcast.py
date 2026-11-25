"""«Πώς πήγε το σύστημα»: εκτίμηση imbalance για ώρες που ΠΕΡΑΣΑΝ, πριν τη δημοσίευση (IMBABE, Πέμπτη).

Πραγματική παραγωγή ανά τεχνολογία (ENTSO-E A75, ~30-60' καθυστέρηση) σε σύγκριση με το πρόγραμμα του
τελευταίου ISP (ISP3 > ISP2 > ISP1) δείχνει τι ενεργοποίησε ο ΑΔΜΗΕ:
  * ελεγχόμενη παραγωγή (λιγνίτης/αέριο/υδρο) ΠΑΝΩ από το ISP  -> ενεργοποίηση προς τα πάνω (σύστημα «κοντό»)
  * ελεγχόμενη παραγωγή ΚΑΤΩ από το ISP                         -> προς τα κάτω (σύστημα «μακρύ»)
Ένα μοντέλο (gradient boosting) μαθαίνει από το ιστορικό πώς αυτά μεταφράζονται σε τιμή imbalance.

Έλεγχος εκτός εκπαίδευσης 5/2026-11/2026: σωστή κατεύθυνση imbalance vs DAM 82%, σφάλμα 78,9 έναντι
105,4 €/MWh της DAM. Τα ιστορικά δεδομένα ENTSO-E είναι οριστικά· οι πρώτες τιμές σε πραγματικό χρόνο
μπορεί να αναθεωρηθούν, οπότε η ακρίβεια σε ζωντανή χρήση μπορεί να είναι λίγο μικρότερη.
"""
from __future__ import annotations

import html

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .entsoe import controllable

TZ = "Europe/Berlin"
FEATURES = ["dev", "res_dev", "dev_n", "res_dev_n", "bess", "ctrl", "isp_ctrl", "isp_stab", "isp_act", "dev_l1", "dev_l2", "dev_r4",
            "hour", "qh", "dam", "ida_m_dam"]
GBM = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=80, random_state=0)


def _read(path) -> pd.DataFrame:
    d = pd.read_csv(path, index_col=0)
    d.index = pd.to_datetime(d.index, utc=True)
    return d


def build_frame(gen_csv="data/entsoe_gen.csv", admie_csv="data/admie.csv", extra_csv="data/admie_extra.csv",
                prices_csv="data/prices.csv") -> pd.DataFrame:
    g, a, x, p = _read(gen_csv), _read(admie_csv), _read(extra_csv), _read(prices_csv)
    xc = [c for c in ("isp_thermal", "isp_hydro", "isp2_thermal", "isp2_hydro", "isp3_thermal", "isp3_hydro",
                      "isp_renewables") if c in x]
    idx = a.index.union(g.index)
    m = a.reindex(idx).join(x[xc]).join(p)
    m["ctrl"] = controllable(g).reindex(idx)
    m["res_act"] = (g.get("solar", 0).fillna(0) + g.get("wind", 0).fillna(0)).reindex(idx)
    m["bess"] = (g.get("storage", pd.Series(0, index=g.index)).fillna(0)
                 - g.get("storage_cons", pd.Series(0, index=g.index)).fillna(0)).reindex(idx)

    def latest(*cols):
        out = pd.Series(np.nan, index=m.index)
        for c in cols:
            if isinstance(c, tuple):
                v = m[c[0]] + m[c[1]] if set(c) <= set(m.columns) else None
            else:
                v = m[c] if c in m else None
            if v is not None:
                out = out.combine_first(v)
        return out

    m["isp_ctrl"] = latest(("isp3_thermal", "isp3_hydro"), ("isp2_thermal", "isp2_hydro"), ("isp_thermal", "isp_hydro"))
    m["isp_res"] = latest("res_isp3", "res_isp2", "isp_renewables")
    m["isp_stab"] = latest("stab_isp3", "stab_isp2", "stability")
    m["isp_act"] = latest("act_isp3", "act_isp2", "act_isp1")
    m["dev"] = m["ctrl"] - m["isp_ctrl"]          # + = οι μονάδες έδωσαν περισσότερο από το πρόγραμμα
    # Η γραμμή «Renewables» του ISP είναι η ΔΙΑΘΕΣΙΜΗ ΑΠΕ· όταν υπάρχει πλεόνασμα (Energy Surplus < 0), το
    # πρόγραμμα αγοράς των ΑΠΕ είναι διαθέσιμη - πλεόνασμα (π.χ. 7.000 - 2.000 = 5.000 MW). Σύγκριση με αυτό.
    m["isp_res_sched"] = m["isp_res"] - m["isp_stab"].fillna(0)
    m["res_dev"] = m["res_act"] - m["isp_res_sched"]  # + = περισσότερες ΑΠΕ από το πρόγραμμα αγοράς
    loc = m.index.tz_convert(TZ)
    m["hour"], m["qh"] = loc.hour, loc.hour * 4 + loc.minute // 15
    # Συστηματικές διαφορές ορισμών ENTSO-E vs ISP (π.χ. ΦΒ διανομής εκτός ENTSO-E: έως -1.700 MW το μεσημέρι).
    # «Κανονικοποιημένη» απόκλιση = απόκλιση - διάμεσος ίδιας ώρας των 28 προηγούμενων ημερών (χωρίς τη σημερινή).
    date = pd.Series(loc.tz_localize(None).normalize(), index=m.index)
    for c in ("dev", "res_dev"):
        piv = m.groupby([date, m["hour"]])[c].mean().unstack()
        base = piv.shift(1).rolling(28, min_periods=7).median()
        key = pd.MultiIndex.from_arrays([date, m["hour"]])
        m[f"{c}_n"] = m[c].values - base.stack().reindex(key).values
    m["dev_l1"], m["dev_l2"], m["dev_r4"] = m["dev_n"].shift(1), m["dev_n"].shift(2), m["dev_n"].rolling(4).mean()
    ida = latest("ida3", "ida2", "ida1")
    m["ida_m_dam"] = ida - m["dam"]
    m["spread"] = m["imbalance"] - m["dam"]
    return m


class Nowcaster:
    def fit(self, m: pd.DataFrame, since="2025-07-01") -> "Nowcaster":
        k = m[(m.index >= pd.Timestamp(since, tz="UTC"))].dropna(subset=["spread", "dev"])
        self.features = [f for f in FEATURES if k[f].notna().sum() > 100]
        lo, hi = k["spread"].quantile([0.005, 0.995])
        X, y = k[self.features], k["spread"].clip(lo, hi)
        self.clf = HistGradientBoostingClassifier(**GBM).fit(X, k["spread"] > 0)
        self.reg = {q: HistGradientBoostingRegressor(loss="quantile", quantile=q, **GBM).fit(X, y)
                    for q in (0.1, 0.5, 0.9)}
        self.trained_through = k.index.max()
        # conformal διόρθωση του εύρους: τα άκρα χωρίς τις τελευταίες 28 ημέρες, μέτρηση κάλυψης σε αυτές
        cal = k.index >= self.trained_through - pd.Timedelta(days=28)
        self.cal_lo = self.cal_hi = 0.0
        if cal.sum() > 500 and (~cal).sum() > 5000:
            lo_m = HistGradientBoostingRegressor(loss="quantile", quantile=0.1, **GBM).fit(X[~cal], y[~cal])
            hi_m = HistGradientBoostingRegressor(loss="quantile", quantile=0.9, **GBM).fit(X[~cal], y[~cal])
            yc = k["spread"].to_numpy()[cal]
            self.cal_lo = float(np.quantile(lo_m.predict(X[cal]) - yc, 0.9))
            self.cal_hi = float(np.quantile(yc - hi_m.predict(X[cal]), 0.9))
        return self

    def predict(self, m: pd.DataFrame) -> pd.DataFrame:
        k = m.dropna(subset=["dev"])
        X = k[self.features]
        out = pd.DataFrame(index=k.index)
        out["dam"] = k["dam"]
        for c in ("ctrl", "isp_ctrl", "dev", "res_dev", "dev_n", "res_dev_n"):
            out[c] = k[c]
        out["p_short"] = self.clf.predict_proba(X)[:, 1]  # P(imbalance > DAM) = σύστημα «κοντό»
        P = np.sort(np.column_stack([self.reg[q].predict(X) for q in (0.1, 0.5, 0.9)]), axis=1)
        P[:, 0] = np.minimum(P[:, 0] - self.cal_lo, P[:, 1])
        P[:, 2] = np.maximum(P[:, 2] + self.cal_hi, P[:, 1])
        for i, q in enumerate(("q10", "q50", "q90")):
            out[f"imb_{q}"] = k["dam"].values + P[:, i]
        out["imbalance_actual"] = k["imbalance"]  # αν έχει ήδη δημοσιευθεί
        return out


def nowcast_day(day, **paths) -> tuple[pd.DataFrame, pd.Timestamp]:
    m = build_frame(**paths)
    model = Nowcaster().fit(m)
    d = pd.Timestamp(day)
    loc = m.index.tz_convert(TZ).tz_localize(None).normalize()
    return model.predict(m[loc == d]), model.trained_through


def direction_label(p: float) -> tuple[str, str]:
    """Κατεύθυνση imbalance ως προς τη DAM με την πιθανότητα ΑΥΤΗΣ της κατεύθυνσης (p = P(imbalance > DAM))."""
    if p >= 0.6:
        return f"↑ πάνω από DAM — {p:.0%}", "#ffe1d6" if p >= 0.7 else "#fff0e8"
    if p <= 0.4:
        return f"↓ κάτω από DAM — {1 - p:.0%}", "#dbe9ff" if p <= 0.3 else "#eef5ff"
    return f"αβέβαιο (πάνω {p:.0%} / κάτω {1 - p:.0%})", ""


def _status(p: float) -> tuple[str, str]:
    if p >= 0.8:
        return "κοντό (ενεργ. ↑)", "#ffe1d6"
    if p >= 0.6:
        return "μάλλον κοντό", "#fff0e8"
    if p > 0.4:
        return "ισορροπημένο", ""
    if p > 0.2:
        return "μάλλον μακρύ", "#eef5ff"
    return "μακρύ (ενεργ. ↓)", "#dbe9ff"


def to_html(out: pd.DataFrame, day, trained_through) -> str:
    loc = out.index.tz_convert(TZ)
    h = out.groupby(loc.floor("h")).agg(
        dam=("dam", "mean"), dev=("dev_n", "mean"), res_dev=("res_dev_n", "mean"), p_short=("p_short", "mean"),
        q10=("imb_q10", "mean"), q50=("imb_q50", "mean"), q90=("imb_q90", "mean"),
        actual=("imbalance_actual", "mean"), n=("dev", "size"))
    rows = []
    for t, r in h.iterrows():
        st, bg = _status(r.p_short)
        act = "" if pd.isna(r.actual) else f"{r.actual:.0f}"
        rows.append(
            f"<tr><td style='padding:3px 8px'>{t:%H}-{t + pd.Timedelta(hours=1):%H}</td>"
            f"<td style='padding:3px 8px;text-align:right'>{r.dam:.1f}</td>"
            f"<td style='padding:3px 8px;text-align:right'>{r.dev:+.0f}</td>"
            f"<td style='padding:3px 8px;text-align:right'>{r.res_dev:+.0f}</td>"
            f"<td style='padding:3px 8px;background:{bg}'>{st}</td>"
            f"<td style='padding:3px 8px;background:{direction_label(r.p_short)[1]}'>{direction_label(r.p_short)[0]}</td>"
            f"<td style='padding:3px 8px;text-align:right;font-weight:bold'>{r.q50:.0f}</td>"
            f"<td style='padding:3px 8px;text-align:right'>{r.q10:.0f} – {r.q90:.0f}</td>"
            f"<td style='padding:3px 8px;text-align:right'>{act}</td>"
            f"<td style='padding:3px 8px;text-align:right;color:#888'>{int(r.n)}/4</td></tr>")
    th = "".join(f"<th style='padding:4px 8px;border-bottom:2px solid #555'>{html.escape(c)}</th>" for c in [
        "Ώρα CET", "DAM", "Μονάδες: σήμερα vs ISP (MW)*", "ΑΠΕ: σήμερα vs ISP (MW)*", "Σύστημα", "Imbalance vs DAM (πιθανότητα)",
        "Imbalance (εκτίμηση)", "εύρος q10–q90", "Πραγματικό", "15λεπτα"])
    last = out.index.max().tz_convert(TZ)
    return f"""<div style="font-family:Arial,sans-serif;font-size:13px">
<h3 style="margin:0 0 4px">Πώς πήγε το σύστημα — {pd.Timestamp(day):%d/%m/%Y}</h3>
<p style="margin:0 0 8px;color:#555">Πραγματική παραγωγή ENTSO-E έως {last:%H:%M} CET σε σύγκριση με το τελευταίο ISP.
Μοντέλο εκπαιδευμένο με imbalance έως {trained_through.tz_convert(TZ):%d/%m/%Y}.</p>
<table style="border-collapse:collapse;font-size:12px"><tr>{th}</tr>{''.join(rows)}</table>
<p style="color:#555;margin:8px 0 2px">* Αφορούν τη ΣΗΜΕΡΙΝΗ ημέρα: πραγματική παραγωγή σήμερα (ENTSO-E) μείον το
σημερινό πρόγραμμα του ISP. Από αυτό αφαιρείται μόνο μια σταθερή διαφορά μέτρησης της ίδιας ώρας (η συνηθισμένη απόκλιση,
διάμεσος 28 ημερών), επειδή το ENTSO-E δεν μετρά ακριβώς ό,τι το ISP (π.χ. δεν περιλαμβάνει όλα τα ΦΒ διανομής: το
μεσημέρι δείχνει συστηματικά 800–1.700 MW λιγότερες ΑΠΕ). Παράδειγμα: σήμερα −327 MW, συνήθως −840 MW → +513 MW, δηλαδή
οι ΑΠΕ βγήκαν ~500 MW περισσότερες απ' ό,τι συνήθως σε σχέση με το πρόγραμμα.
«Μονάδες»: ελεγχόμενη παραγωγή (λιγνίτης, αέριο, υδροηλεκτρικά) μείον το
πρόγραμμα του ISP. Θετικό = ο ΑΔΜΗΕ ζήτησε περισσότερη ενέργεια (σύστημα κοντό → imbalance συνήθως πάνω από τη DAM).
Αρνητικό = λιγότερη (σύστημα μακρύ → imbalance συνήθως κάτω, έως το κατώτατο όριο). «ΑΠΕ»: πραγματικές ΑΠΕ
μείον πρόγραμμα αγοράς ΑΠΕ του ISP (διαθέσιμη ΑΠΕ μείον το πλεόνασμα που προβλέπεται να περικοπεί· αρνητικό = λιγότερες ΑΠΕ, συχνή αιτία ενεργοποίησης προς τα πάνω). Έλεγχος εκτός εκπαίδευσης:
σωστή κατεύθυνση 80%, σφάλμα 80 έναντι 105 €/MWh της DAM.</p></div>"""
