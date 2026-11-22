"""Φόρτωση και ευθυγράμμιση δεδομένων.

Αναμενόμενη μορφή CSV (ένα ή περισσότερα αρχεία, ενώνονται στη στήλη timestamp):

    timestamp,dam,ida1,ida2,ida3,imbalance,res_forecast,load_forecast[,...extra]

* timestamp: αρχή του διαστήματος. Αν δεν έχει ζώνη ώρας θεωρείται Europe/Athens.
* Ωριαίες σειρές (π.χ. DAM πριν τον Οκτ. 2025) γεμίζουν προς τα εμπρός στα 15λεπτα.
* imbalance: κενό όπου δεν έχει δημοσιευθεί ακόμα.
* Οποιαδήποτε επιπλέον αριθμητική στήλη (π.χ. δείκτες ευστάθειας/επάρκειας, ανάγκες
  εφεδρειών, διασυνδέσεις, πρόβλεψη αιολικών/φωτοβολταϊκών χωριστά) χρησιμοποιείται
  αυτόματα ως feature — με την προϋπόθεση ότι είναι γνωστή τη στιγμή της πρόβλεψης.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Config


def _read_one(path: str | Path, tz: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts_col = next((c for c in df.columns if c.lower() in ("timestamp", "datetime", "time", "date")), df.columns[0])
    raw = df[ts_col].astype(str)
    has_offset = raw.str.contains(r"(?:[+-]\d{2}:?\d{2}|Z)$", regex=True).all()
    ts = pd.to_datetime(raw, utc=has_offset)  # με offset (+02:00/+03:00) -> UTC -> τοπική
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
    df.index = pd.DatetimeIndex(ts).tz_convert(tz)
    df = df.drop(columns=[ts_col])
    df.columns = [c.strip().lower() for c in df.columns]
    return df[~df.index.duplicated(keep="last")].sort_index()


def load_data(paths, cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or Config()
    frames = [_read_one(p, cfg.tz) for p in ([paths] if isinstance(paths, (str, Path)) else paths)]
    df = frames[0]
    for f in frames[1:]:
        df = df.combine_first(f) if set(f.columns) & set(df.columns) else df.join(f, how="outer")
    return align(df, cfg)


def align(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Μετατροπή σε κανονικό πλέγμα MTU (15') με τοπική ώρα."""
    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize(cfg.tz, ambiguous="infer", nonexistent="shift_forward")
    df = df.tz_convert("UTC")
    df = df.select_dtypes("number")
    grid = pd.date_range(df.index.min().floor("D"), df.index.max().ceil("h"), freq=cfg.mtu, tz="UTC")
    # ωριαίες τιμές -> επανάληψη στα 4 τέταρτα (έως 3 βήματα), όχι πέρα από αυτό
    out = df.reindex(grid.union(df.index)).sort_index()
    out = out.ffill(limit=3).reindex(grid)
    # τιμή imbalance δεν γεμίζεται: ό,τι λείπει θεωρείται μη δημοσιευμένο
    for c in (cfg.col_imb, cfg.col_curt):
        if c in df.columns:
            out[c] = df[c].reindex(grid)
    out.index = out.index.tz_convert(cfg.tz)
    out.index.name = "timestamp"
    return out


def last_known_date(df: pd.DataFrame, col: str, cfg: Config) -> pd.Timestamp | None:
    """Τελευταία ημερομηνία (τοπική) με πλήρη ημέρα τιμών στη στήλη col."""
    s = df[col].dropna() if col in df else pd.Series(dtype=float)
    if s.empty:
        return None
    dates = s.index.tz_convert(cfg.tz).tz_localize(None).normalize()
    counts = pd.Series(1, index=dates).groupby(level=0).sum()
    full = counts[counts >= counts.max() * 0.9]
    return full.index.max()


def last_known_imbalance_date(df: pd.DataFrame, cfg: Config) -> pd.Timestamp | None:
    return last_known_date(df, cfg.col_imb, cfg)
