"""Walk-forward backtest και παραγωγική πρόβλεψη."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .availability import published_through
from .config import Config
from .data import last_known_imbalance_date
from .features import TARGET, build_features
from .model import ImbalanceModel


def _train_slice(frame: pd.DataFrame, cfg: Config, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Γραμμές εκπαίδευσης: μόνο ημέρες με imbalance γνωστό έως το cutoff."""
    start = cutoff - pd.Timedelta(days=cfg.train_window_days - 1)
    m = (frame["date"] <= cutoff) & (frame["date"] >= start) & frame[TARGET].notna()
    return frame[m]


def backtest(df: pd.DataFrame, cfg: Config, start, end=None, verbose: bool = True) -> pd.DataFrame:
    """Επαναλαμβανόμενη εκπαίδευση κάθε εβδομάδα (με κάθε νέα δημοσίευση imbalance).

    Κάθε ημέρα-στόχος D προβλέπεται από μοντέλο εκπαιδευμένο μόνο με ό,τι ήταν
    διαθέσιμο την ημέρα έκδοσης D - issue_lead_days.
    """
    frame, feats = build_features(df, cfg)
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize() if end is not None else frame["date"].max()
    test = frame[(frame["date"] >= start) & (frame["date"] <= end)]
    results = []
    model, trained_at = None, None
    for cutoff, grp in test.groupby("cutoff"):
        due = trained_at is None or (cutoff - trained_at).days >= 7 * cfg.retrain_every_weeks
        if due:
            train = _train_slice(frame, cfg, cutoff)
            if train["date"].nunique() < cfg.min_train_days:
                continue
            model, trained_at = ImbalanceModel(cfg, feats).fit(train), cutoff
        pred = model.predict(grp)
        res = grp[["date", "cutoff", "dam", "imbalance", "spread"]].join(pred)
        res = res.join(baselines(grp))
        results.append(res)
        if verbose:
            d0, d1 = grp["date"].min().date(), grp["date"].max().date()
            tag = f"νέο μοντέλο ({train['date'].nunique()} ημ.)" if due else f"μοντέλο της {trained_at.date()}"
            print(f"  cutoff {cutoff.date()} | {tag} | test {d0}..{d1}")
    if not results:
        raise ValueError("Δεν υπάρχουν αρκετά δεδομένα εκπαίδευσης για το backtest.")
    return pd.concat(results).sort_index()


def baselines(frame: pd.DataFrame) -> pd.DataFrame:
    """Απλές μέθοδοι αναφοράς για σύγκριση."""
    b = pd.DataFrame(index=frame.index)
    b["bl_dam"] = frame["dam"]  # imbalance = DAM (spread 0)
    if "spr_h_m28" in frame:
        b["bl_hist28"] = frame["dam"] + frame["spr_h_m28"].fillna(0)  # DAM + μέσο spread ώρας (28 ημ.)
    if "ida_last" in frame:
        b["bl_ida"] = frame["ida_last"].fillna(frame["dam"])  # τελευταία IDA τιμή
    return b


def metrics(res: pd.DataFrame) -> pd.DataFrame:
    """MAE/RMSE στην τιμή imbalance, ακρίβεια κατεύθυνσης, pinball, κάλυψη διαστήματος."""
    r = res.dropna(subset=["imbalance"])
    y = r["imbalance"]
    rows = {}
    cands = {"model_q50": r["imb_q50"]} | {c: r[c] for c in r.columns if c.startswith("bl_")}
    for name, p in cands.items():
        e = y - p
        spr_hat = p - r["dam"]
        rows[name] = {
            "MAE": e.abs().mean(),
            "RMSE": np.sqrt((e**2).mean()),
            "bias": e.mean(),
            # σωστή πρόβλεψη πρόσημου του spread (όπου spread_hat != 0)
            "dir_acc": (
                ((np.sign(spr_hat) == np.sign(r["spread"])) & (spr_hat != 0)).sum() / (spr_hat != 0).sum()
                if (spr_hat != 0).any()
                else np.nan
            ),
        }
    out = pd.DataFrame(rows).T
    if "p_imb_gt_dam" in r and r["p_imb_gt_dam"].notna().any():
        out.loc["model_q50", "dir_acc_clf"] = ((r["p_imb_gt_dam"] > 0.5) == (r["spread"] > 0)).mean()
    if {"imb_q10", "imb_q90"} <= set(r.columns):
        out.loc["model_q50", "cover_80"] = ((y >= r["imb_q10"]) & (y <= r["imb_q90"])).mean()
        pin = 0.0
        for q, c in [(0.1, "imb_q10"), (0.5, "imb_q50"), (0.9, "imb_q90")]:
            d = y - r[c]
            pin += np.maximum(q * d, (q - 1) * d).mean()
        out.loc["model_q50", "pinball_avg"] = pin / 3
    return out


def forecast(df: pd.DataFrame, cfg: Config, target_date, issue_date=None) -> pd.DataFrame:
    """Πρόβλεψη για την ημέρα target_date (τοπική) με ό,τι είναι γνωστό σήμερα.

    Ο περιορισμός διαθεσιμότητας = min(κανόνας Πέμπτης, τελευταία πλήρης ημέρα στα δεδομένα).
    """
    target_date = pd.Timestamp(target_date).normalize()
    issue_date = (
        pd.Timestamp(issue_date).normalize()
        if issue_date is not None
        else target_date - pd.Timedelta(days=cfg.issue_lead_days)
    )
    lead = (target_date - issue_date).days
    cfg_run = Config(**{**cfg.__dict__, "issue_lead_days": lead})
    kt = last_known_imbalance_date(df, cfg_run)
    rule = published_through(issue_date, cfg.pub_weekday, cfg.pub_available_same_day)
    cutoff = min(rule, kt) if kt is not None else rule

    frame, feats = build_features(df, cfg_run, known_through=kt)
    train = _train_slice(frame, cfg_run, cutoff)
    if train["date"].nunique() < cfg.min_train_days:
        raise ValueError(f"Λίγα δεδομένα εκπαίδευσης ({train['date'].nunique()} ημέρες έως {cutoff.date()}).")
    tgt = frame[frame["date"] == target_date]
    if tgt.empty:
        raise ValueError(f"Δεν υπάρχουν γραμμές (DAM/ΑΠΕ/φορτίο) για {target_date.date()}.")
    model = ImbalanceModel(cfg_run, feats).fit(train)
    out = tgt[["dam"]].join(model.predict(tgt))
    out.attrs.update(issue_date=str(issue_date.date()), imbalance_known_through=str(cutoff.date()))
    return out
