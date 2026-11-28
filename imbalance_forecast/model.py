"""Μοντέλα:

1. Ευστάθεια: ταξινομητής P(ο ΑΔΜΗΕ βάζει αέριο/κόβει ΑΠΕ στο 15λεπτο) -> stab_prob.
   Στην εκπαίδευση του κύριου μοντέλου το stab_prob είναι out-of-fold (cross-fitting ανά
   χρονικά μπλοκ), ώστε να μοιάζει με αυτό που θα έχουμε στην πράξη.
2. Κύριο: quantile gradient boosting για spread (ή imbalance) + ταξινομητής P(imbalance > DAM).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

from .config import Config
from .features import P_CURT, STAB_FLAG, STAB_PROB, TARGET


def align_to_direction(point, dam, p_up, thr: float = 0.6):
    """Συνέπεια εκτίμησης και κατεύθυνσης: αν η πιθανότητα δίνει ↑ (≥ thr) αλλά η εκτίμηση βγαίνει κάτω από τη DAM,
    ή ↓ (≤ 1 - thr) αλλά βγαίνει πάνω, η εκτίμηση γίνεται ίση με τη DAM.
    Backtest 2026 (στάδια ISP2/ISP1/DAM): σφάλμα 86,9 → 86,1, 88,9 → 87,9, 94,3 → 93,2 €/MWh· στις ασυνεπείς
    περιπτώσεις η κατεύθυνση ήταν σωστή στο 66–67%."""
    point = np.asarray(point, dtype=float).copy()
    dam = np.asarray(dam, dtype=float)
    p_up = np.asarray(p_up, dtype=float)
    up = (p_up >= thr) & (point < dam)
    dn = (p_up <= 1 - thr) & (point > dam)
    fix = (up | dn) & np.isfinite(dam)
    point[fix] = dam[fix]
    return point


class ImbalanceModel:
    def __init__(self, cfg: Config, features: list[str], stab_features: list[str] | None = None):
        self.cfg = cfg
        self.features = features
        self.stab_features = stab_features or []
        self.reg = {}
        self.clf = None
        self.clf_iso = None  # βαθμονόμηση πιθανότητας κατεύθυνσης (isotonic)
        self.clf_up = None
        self.cal_lo = 0.0  # διεύρυνση κάτω άκρου (conformal), στη μονάδα του στόχου
        self.cal_hi = 0.0
        self.stab_clf = None

    # ------------------------------------------------------------ ευστάθεια
    def _new_stab_clf(self):
        return HistGradientBoostingClassifier(**self.cfg.gbm_params)

    def _stab_ok(self, stab_train) -> bool:
        return (
            STAB_PROB in self.features
            and stab_train is not None
            and STAB_FLAG in stab_train
            and stab_train[STAB_FLAG].dropna().nunique() == 2
        )

    def _oof_stab(self, train: pd.DataFrame, stab_train: pd.DataFrame) -> np.ndarray:
        dates = np.sort(train["date"].unique())
        blocks = np.array_split(dates, max(2, self.cfg.stab_cv_folds))
        out = np.full(len(train), np.nan)
        for blk in blocks:
            in_blk = train["date"].isin(blk).values
            fit_rows = stab_train[~stab_train["date"].isin(blk)]
            if fit_rows[STAB_FLAG].nunique() < 2:
                continue
            k = self._recent(fit_rows)
            m = self._new_stab_clf().fit(fit_rows.loc[k, self.stab_features], fit_rows.loc[k, STAB_FLAG])
            out[in_blk] = m.predict_proba(train.loc[in_blk, self.stab_features])[:, 1]
        return out

    def stab_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.stab_clf is None:
            return np.full(len(frame), np.nan)
        return self.stab_clf.predict_proba(frame[self.stab_features])[:, 1]

    def _recent(self, d: pd.DataFrame) -> np.ndarray:
        """Βάρη εγγύτητας ως τυχαία υποδειγματοληψία: κάθε γραμμή κρατιέται με πιθανότητα
        0.5^(ηλικία/half-life). Ισοδύναμο με sample_weight, αλλά πολύ ταχύτερο στο scikit-learn."""
        hl = self.cfg.recency_halflife_days
        if not hl:
            return np.ones(len(d), bool)
        age = (d["date"].max() - d["date"]).dt.days.to_numpy(dtype=float)
        return np.random.default_rng(0).random(len(d)) < 0.5 ** (age / hl)

    @staticmethod
    def _add_p_curt(frame: pd.DataFrame) -> None:
        """P(περικοπή) ≈ P(πλεόνασμα ISP) × ιστορικό P(περικοπή | πλεόνασμα)."""
        if P_CURT in frame and "curt_given_stab_m28" in frame:
            frame[P_CURT] = frame[STAB_PROB] * frame["curt_given_stab_m28"]

    # ------------------------------------------------------------ fit / predict
    def fit(self, frame: pd.DataFrame, stab_train: pd.DataFrame | None = None) -> "ImbalanceModel":
        d = frame.dropna(subset=[TARGET]).copy()
        if self._stab_ok(stab_train):
            st = stab_train.dropna(subset=[STAB_FLAG])
            d[STAB_PROB] = self._oof_stab(d, st)
            k = self._recent(st)
            self.stab_clf = self._new_stab_clf().fit(st.loc[k, self.stab_features], st.loc[k, STAB_FLAG])
        elif STAB_PROB in d:
            d[STAB_PROB] = np.nan
        self._add_p_curt(d)

        X = d[self.features]
        y = d[TARGET]
        lo, hi = y.quantile(list(self.cfg.winsor))
        y_w = y.clip(lo, hi)
        keep = self._recent(d)
        for q in self.cfg.quantiles:
            m = HistGradientBoostingRegressor(loss="quantile", quantile=q, **self.cfg.gbm_params)
            self.reg[q] = m.fit(X[keep], y_w[keep])
        self._calibrate(d, X, y, y_w, keep)
        sign = (d["spread"] > 0).astype(int)
        if sign.nunique() == 2:
            self.clf = HistGradientBoostingClassifier(**self.cfg.gbm_params).fit(X[keep], sign[keep])
            self._calibrate_prob(d, X, sign, keep)
        up = (d["spread"] > self.cfg.spike_up).astype(int)  # ανοδικό «σπάικ» imbalance
        if up.nunique() == 2:
            self.clf_up = HistGradientBoostingClassifier(**self.cfg.gbm_params).fit(X[keep], up[keep])
        return self

    def _calibrate(self, d, X, y, y_w, keep) -> None:
        """Conformal βαθμονόμηση του εύρους (CQR): εκπαίδευση των άκρων χωρίς τις τελευταίες cal_days ημέρες,
        μέτρηση πόσο έξω από το εύρος πέφτει η πραγματική τιμή σε αυτές, και διεύρυνση κάθε άκρου ώστε
        q_low..q_high να καλύπτει πράγματι (q_high - q_low) των περιπτώσεων (π.χ. 80%)."""
        qs = sorted(self.cfg.quantiles)
        if not self.cfg.calibrate_intervals or len(qs) < 2:
            return
        dates = np.sort(d["date"].unique())
        if len(dates) < self.cfg.cal_days + self.cfg.min_train_days // 2:
            return
        cal = d["date"].isin(dates[-self.cfg.cal_days:]).to_numpy()
        fit_rows = keep & ~cal
        q_lo, q_hi = qs[0], qs[-1]
        lo_hi = []
        for q in (q_lo, q_hi):
            m = HistGradientBoostingRegressor(loss="quantile", quantile=q, **self.cfg.gbm_params)
            lo_hi.append(m.fit(X[fit_rows], y_w[fit_rows]).predict(X[cal]))
        yc = y.to_numpy()[cal]
        # μη-κάλυψη ανά άκρο: θέλουμε P(y < low - δ_lo) = q_lo και P(y > high + δ_hi) = 1 - q_hi
        self.cal_lo = float(np.quantile(lo_hi[0] - yc, 1 - q_lo))
        self.cal_hi = float(np.quantile(yc - lo_hi[1], q_hi))

    def _calibrate_prob(self, d, X, sign, keep) -> None:
        """Βαθμονόμηση P(imbalance > DAM): ταξινομητής χωρίς τις τελευταίες prob_cal_days ημέρες, σύγκριση
        των πιθανοτήτων του με το τι έγινε σε αυτές, και isotonic «χάρτης διόρθωσης» για την τελική πιθανότητα,
        ώστε «70%» να σημαίνει πράγματι ~7 στις 10."""
        days = self.cfg.prob_cal_days
        if not days:
            return
        dates = np.sort(d["date"].unique())
        if len(dates) < days + self.cfg.min_train_days // 2:
            return
        cal = d["date"].isin(dates[-days:]).to_numpy()
        fit_rows = keep & ~cal
        if sign[fit_rows].nunique() < 2 or sign[cal].nunique() < 2:
            return
        m = HistGradientBoostingClassifier(**self.cfg.gbm_params).fit(X[fit_rows], sign[fit_rows])
        raw = m.predict_proba(X[cal])[:, 1]
        self.clf_iso = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds="clip").fit(raw, sign[cal])

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        out = pd.DataFrame(index=frame.index)
        if STAB_PROB in self.features:
            frame[STAB_PROB] = self.stab_proba(frame)
            out[STAB_PROB] = frame[STAB_PROB]
            self._add_p_curt(frame)
            if P_CURT in frame:
                out[P_CURT] = frame[P_CURT]
        X = frame[self.features]
        qs = sorted(self.cfg.quantiles)
        P = np.sort(np.column_stack([self.reg[q].predict(X) for q in qs]), axis=1)  # χωρίς διασταύρωση
        if len(qs) >= 2:  # βαθμονομημένο εύρος (τα άκρα δεν περνούν τη διάμεσο)
            mid = P[:, len(qs) // 2]
            P[:, 0] = np.minimum(P[:, 0] - self.cal_lo, mid)
            P[:, -1] = np.maximum(P[:, -1] + self.cal_hi, mid)
        dam = frame["dam"].values
        for i, q in enumerate(qs):
            tag = f"q{int(round(q * 100)):02d}"
            if self.cfg.target == "spread":
                out[f"spread_{tag}"] = P[:, i]
                out[f"imb_{tag}"] = dam + P[:, i]
            else:
                out[f"imb_{tag}"] = P[:, i]
                out[f"spread_{tag}"] = P[:, i] - dam
        p = self.clf.predict_proba(X)[:, 1] if self.clf is not None else np.full(len(X), np.nan)
        if self.clf_iso is not None:
            p = self.clf_iso.predict(p)
        out["p_imb_gt_dam"] = p
        out["p_imb_lt_dam"] = 1 - p
        out["p_spike_up"] = self.clf_up.predict_proba(X)[:, 1] if self.clf_up is not None else np.nan
        # σημειακή πρόβλεψη: DAM, ή το τρέχον κατώτατο όριο όταν η περικοπή είναι πιθανή
        point = pd.Series(dam, index=frame.index, dtype=float)
        if "ida_last" in frame:  # τελευταία γνωστή τιμή αγοράς
            point = frame["ida_last"].astype(float).fillna(point)
        if self.cfg.point_mode == "q50":
            # isp2/isp3: η διάμεσος του μοντέλου κερδίζει DAM και IDA στο backtest
            point = out["imb_q50"].copy()
            point[:] = align_to_direction(point.values, dam, p)
        elif P_CURT in out and "imb_min28" in frame:
            hit = (out[P_CURT] >= self.cfg.point_curt_threshold) & frame["imb_min28"].notna()
            point[hit] = np.minimum(frame.loc[hit, "imb_min28"], point[hit])
        out["imb_point"] = point
        return out
