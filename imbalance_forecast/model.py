"""Μοντέλο: quantile gradient boosting για spread (ή imbalance) + ταξινομητής κατεύθυνσης."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .config import Config
from .features import TARGET


class ImbalanceModel:
    def __init__(self, cfg: Config, features: list[str]):
        self.cfg = cfg
        self.features = features
        self.reg = {}
        self.clf = None

    def fit(self, frame: pd.DataFrame) -> "ImbalanceModel":
        d = frame.dropna(subset=[TARGET])
        X = d[self.features]
        y = d[TARGET]
        lo, hi = y.quantile(list(self.cfg.winsor))
        y_w = y.clip(lo, hi)
        for q in self.cfg.quantiles:
            m = HistGradientBoostingRegressor(loss="quantile", quantile=q, **self.cfg.gbm_params)
            self.reg[q] = m.fit(X, y_w)
        # κατεύθυνση: imbalance > DAM ;
        sign = (d["spread"] > 0).astype(int)
        if sign.nunique() == 2:
            self.clf = HistGradientBoostingClassifier(**self.cfg.gbm_params).fit(X, sign)
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = frame[self.features]
        qs = sorted(self.cfg.quantiles)
        P = np.column_stack([self.reg[q].predict(X) for q in qs])
        P = np.sort(P, axis=1)  # αποφυγή διασταύρωσης quantiles
        out = pd.DataFrame(index=frame.index)
        dam = frame["dam"].values
        for i, q in enumerate(qs):
            tag = f"q{int(round(q * 100)):02d}"
            if self.cfg.target == "spread":
                out[f"spread_{tag}"] = P[:, i]
                out[f"imb_{tag}"] = dam + P[:, i]
            else:
                out[f"imb_{tag}"] = P[:, i]
                out[f"spread_{tag}"] = P[:, i] - dam
        out["p_imb_gt_dam"] = self.clf.predict_proba(X)[:, 1] if self.clf is not None else np.nan
        return out
