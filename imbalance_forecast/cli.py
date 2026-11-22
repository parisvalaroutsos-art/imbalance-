"""Γραμμή εντολών.

    python -m imbalance_forecast demo
    python -m imbalance_forecast backtest --data data/prices.csv data/admie.csv --start 2025-06-01
    python -m imbalance_forecast predict  --data data/prices.csv data/admie.csv --date 2026-11-23
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import Config
from .data import align, last_known_imbalance_date, load_data
from .pipeline import backtest, forecast, metrics


def _cfg(a) -> Config:
    cfg = Config(
        target=a.target,
        issue_lead_days=a.lead,
        ida_available=tuple(a.ida) if a.ida else (),
        pub_available_same_day=a.pub_same_day,
        train_window_days=a.window,
        retrain_every_weeks=getattr(a, "retrain_every", 1),
    )
    return cfg


def _common(p):
    p.add_argument("--target", choices=["spread", "imbalance"], default="spread")
    p.add_argument("--lead", type=int, default=1, help="ημέρες πριν την ημέρα-στόχο που εκδίδεται η πρόβλεψη")
    p.add_argument("--ida", nargs="*", default=["ida1"], help="διαθέσιμες IDA στο issue (π.χ. ida1 ida2)")
    p.add_argument("--pub-same-day", action="store_true", help="τα imbalance της Πέμπτης διαθέσιμα ήδη την Πέμπτη")
    p.add_argument("--window", type=int, default=365, help="παράθυρο εκπαίδευσης (ημέρες)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="imbalance_forecast")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="backtest σε συνθετικά δεδομένα")
    _common(d)
    d.add_argument("--start", default="2025-09-01")
    d.add_argument("--out", default="backtest_demo.csv")
    d.add_argument("--retrain-every", type=int, default=4, help="επανεκπαίδευση κάθε N εβδομάδες")

    b = sub.add_parser("backtest", help="walk-forward backtest")
    _common(b)
    b.add_argument("--data", nargs="+", required=True)
    b.add_argument("--start", required=True)
    b.add_argument("--end")
    b.add_argument("--out", default="backtest.csv")
    b.add_argument("--retrain-every", type=int, default=1, help="επανεκπαίδευση κάθε N εβδομάδες")

    p = sub.add_parser("predict", help="πρόβλεψη μίας ημέρας")
    _common(p)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--date", required=True, help="ημέρα-στόχος (τοπική ώρα)")
    p.add_argument("--issue-date", help="ημέρα έκδοσης (προεπιλογή: date - lead)")
    p.add_argument("--out", default=None)

    a = ap.parse_args(argv)
    cfg = _cfg(a)
    pd.set_option("display.width", 160)

    if a.cmd in ("demo", "backtest"):
        if a.cmd == "demo":
            from .synthetic import make_synthetic

            print("Δημιουργία συνθετικών δεδομένων (όχι πραγματικά)...")
            df = align(make_synthetic(), cfg)
            end = None
        else:
            df = load_data(a.data, cfg)
            end = a.end
        print(f"Δεδομένα: {df.index.min()} .. {df.index.max()} | στήλες: {list(df.columns)}")
        res = backtest(df, cfg, a.start, end)
        res.to_csv(a.out)
        print("\nΑποτελέσματα (τιμή imbalance, €/MWh):")
        print(metrics(res).round(3).to_string())
        print(f"\nΑποθηκεύτηκε: {Path(a.out).resolve()}")
    else:
        df = load_data(a.data, cfg)
        kt = last_known_imbalance_date(df, cfg)
        out = forecast(df, cfg, a.date, a.issue_date)
        print(f"Issue: {out.attrs['issue_date']} | imbalance γνωστό έως: {out.attrs['imbalance_known_through']}"
              f" (στα δεδομένα: {kt.date() if kt is not None else '-'})")
        cols = ["dam", "spread_q10", "spread_q50", "spread_q90", "imb_q50", "p_imb_gt_dam"]
        print(out[cols].round(2).to_string())
        if a.out:
            out.to_csv(a.out)
            print(f"Αποθηκεύτηκε: {Path(a.out).resolve()}")


if __name__ == "__main__":
    main()
