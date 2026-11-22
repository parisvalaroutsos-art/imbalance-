"""Κανόνας διαθεσιμότητας τιμών imbalance.

Τα δεδομένα imbalance δημοσιεύονται κάθε Πέμπτη και καλύπτουν την προηγούμενη
εβδομάδα Δευτέρα–Κυριακή. Άρα τη στιγμή που βγάζουμε μια πρόβλεψη (issue date)
γνωρίζουμε imbalance μόνο έως την Κυριακή πριν από την τελευταία Πέμπτη δημοσίευσης.
Παράδειγμα: issue Τρίτη 24/11 -> τελευταία δημοσίευση Πέμπτη 19/11 -> γνωστά έως Κυριακή 15/11.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def published_through(issue_dates, pub_weekday: int = 3, same_day: bool = False):
    """Τελευταία ημερομηνία (Κυριακή) με γνωστό imbalance για κάθε issue date.

    Δέχεται Timestamp ή DatetimeIndex/Series (naive ημερομηνίες τοπικής ώρας).
    """
    scalar = isinstance(issue_dates, (pd.Timestamp, str))
    d = pd.DatetimeIndex([pd.Timestamp(issue_dates)] if scalar else pd.DatetimeIndex(issue_dates)).normalize()
    delta = (d.weekday.values - pub_weekday) % 7
    if not same_day:
        delta = np.where(delta == 0, 7, delta)
    last_pub = d - pd.to_timedelta(delta, unit="D")
    # Η δημοσίευση της Πέμπτης X καλύπτει Δευτέρα X-10 .. Κυριακή X-4
    out = last_pub - pd.Timedelta(days=4)
    return out[0] if scalar else out
