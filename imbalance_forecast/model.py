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

from .config import Config
from .features import P_CURT, STAB_FLAG, STAB_PROB, TARGET


class ImbalanceModel:
    def __init__(self, cfg: Config, features: list[str], stab_features: list[str] | None = None):
        self.cfg = cfg
        self.features = features
        self.stab_features = stab_features or []
        self.reg = {}
        self.clf = None
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
        sign = (d["spread"] > 0).astype(int)
        if sign.nunique() == 2:
            self.clf = HistGradientBoostingClassifier(**self.cfg.gbm_params).fit(X[keep], sign[keep])
        return self

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
        out["p_imb_gt_dam"] = p
        out["p_imb_lt_dam"] = 1 - p
        return out
