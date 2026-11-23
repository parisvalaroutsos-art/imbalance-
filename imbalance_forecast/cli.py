"""Γραμμή εντολών.

    python -m imbalance_forecast demo
    python -m imbalance_forecast admie --start 2024-01-01 --out data/admie.csv
    python -m imbalance_forecast enex  --start 2024-01-01 --out data/prices.csv
    python -m imbalance_forecast backtest --data data/prices.csv data/admie.csv --start 2025-10-01
    python -m imbalance_forecast predict  --data data/prices.csv data/admie.csv --date 2026-11-23
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import STAGES, stage_config
from .data import align, load_data


def _cfg(a) -> Config:
    over = {}
    if a.lead is not None:
        over["issue_lead_days"] = a.lead
    if a.ida is not None:
        over["ida_available"] = tuple(a.ida)
    return stage_config(
        a.stage,
        **over,
        target=a.target,
        pub_available_same_day=a.pub_same_day,
        train_window_days=a.window,
        retrain_every_weeks=getattr(a, "retrain_every", 1),
        stab_threshold=a.stab_threshold,
    )


def _common(p):
    p.add_argument("--stage", choices=list(STAGES), default="pre_dam",
                   help="pre_dam: D-1 πριν τη DAM | isp2: D ~00:30 (IDA1-2, ISP1-2) | isp3: D ~12:30 (IDA3, ISP3, 12-24h)")
    p.add_argument("--target", choices=["spread", "imbalance"], default="spread")
    p.add_argument("--lead", type=int, default=None, help="ημέρες πριν την ημέρα-στόχο (προεπιλογή: από το στάδιο)")
    p.add_argument("--ida", nargs="*", default=None, help="διαθέσιμες IDA στο issue (προεπιλογή: από το στάδιο)")
    p.add_argument("--pub-same-day", action="store_true", help="τα imbalance της Πέμπτης διαθέσιμα ήδη την Πέμπτη")
    p.add_argument("--window", type=int, default=365, help="παράθυρο εκπαίδευσης (ημέρες)")
    p.add_argument("--stab-threshold", type=float, default=0.0, help="MW περικοπής πάνω από τα οποία = ευστάθεια")


def _print_backtest(res, out, cfg):
    from .pipeline import metrics, spike_report, stability_report

    res.to_csv(out)
    print("\nΑποτελέσματα (τιμή imbalance, €/MWh):")
    print(metrics(res).round(3).to_string())
    rep = stability_report(res)
    if rep is not None:
        print("\nΕυστάθεια -> imbalance (ανά προβλεπόμενη πιθανότητα ευστάθειας):")
        print(rep.round(3).to_string())
    sp = spike_report(res, cfg.spike_up)
    if sp is not None:
        print("\nΑνοδικό σπάικ (imbalance > DAM + spike_up) ανά προβλεπόμενη πιθανότητα p_spike_up:")
        print(sp.round(3).to_string())
    print(f"\nΑποθηκεύτηκε: {Path(out).resolve()}")


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

    g = sub.add_parser("admie", help="κατέβασμα αρχείων ΑΔΜΗΕ σε CSV")
    g.add_argument("--start", required=True)
    g.add_argument("--end", default=str((pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).date()))
    g.add_argument("--out", default="data/admie.csv")
    g.add_argument("--raw-dir", default="data/raw/admie")
    g.add_argument("--only", nargs="*", help="μόνο αυτές οι πηγές: load_forecast res_forecast isp1 imbabe isp2 isp3")
    g.add_argument("--sleep", type=float, default=1.0, help="παύση (s) ανάμεσα στα αιτήματα")
    g.add_argument("--dam-gate", default="12:00", help="ώρα (τοπική, D-1) πριν την οποία πρέπει να έχει βγει η πρόβλεψη")

    ge = sub.add_parser("enex", help="κατέβασμα τιμών DAM/IDA από το ΕΧΕ (HEnEx) σε CSV")
    ge.add_argument("--start", required=True)
    ge.add_argument("--end", default=str((pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).date()))
    ge.add_argument("--out", default="data/prices.csv")
    ge.add_argument("--raw-dir", default="data/raw/enex")
    ge.add_argument("--sleep", type=float, default=1.0, help="παύση (s) ανάμεσα στα αιτήματα")

    gl = sub.add_parser("admie-list", help="τι αρχεία επιστρέφει το API ΑΔΜΗΕ για μια κατηγορία")
    gl.add_argument("--category", required=True)
    gl.add_argument("--date", required=True)
    sub.add_parser("admie-types", help="κατάλογος κατηγοριών αρχείων ΑΔΜΗΕ")

    a = ap.parse_args(argv)
    pd.set_option("display.width", 200)

    if a.cmd == "admie":
        from .admie import fetch_admie

        fetch_admie(a.start, a.end, a.out, a.raw_dir, only=a.only, sleep=a.sleep, dam_gate=a.dam_gate)
        return
    if a.cmd == "enex":
        from .enex import fetch_enex

        fetch_enex(a.start, a.end, a.out, a.raw_dir, sleep=a.sleep)
        return
    if a.cmd == "admie-list":
        from .admie import list_files

        print(json.dumps(list_files(a.category, a.date, a.date), indent=2, ensure_ascii=False))
        return
    if a.cmd == "admie-types":
        from .admie import list_types

        for t in list_types():
            print(f"{t['filetype']:42s} {t['process'][:25]:25s} {t['data_type'][:40]:40s} {t['time_gate']}")
        return

    from .pipeline import backtest, forecast

    cfg = _cfg(a)
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
        print(f"Στάδιο {cfg.stage} | issue D-{cfg.issue_lead_days} | IDA: {list(cfg.ida_available) or 'καμία'}"
              + (f" | ώρες {cfg.target_hours[0]}-{cfg.target_hours[1]}" if cfg.target_hours else ""))
        _print_backtest(backtest(df, cfg, a.start, end), a.out, cfg)
    else:
        df = load_data(a.data, cfg)
        out = forecast(df, cfg, a.date, a.issue_date)
        print(
            f"Issue: {out.attrs['issue_date']} | imbalance γνωστό έως: {out.attrs['imbalance_known_through']}"
            f" | ευστάθεια γνωστή έως: {out.attrs['stability_known_through']}"
        )
        cols = [c for c in ["dam", "stab_prob", "p_curt", "spread_q10", "spread_q50", "spread_q90", "imb_q50", "imb_point", "p_imb_lt_dam", "p_spike_up"]
                if c in out]
        h = out[cols].groupby(out.index.hour).mean().round(2)
        h.index.name = "ώρα"
        print("Ωριαίοι μέσοι όροι (πλήρης 15λεπτη ανάλυση στο --out):\n" + h.to_string())
        if a.out:
            out.to_csv(a.out)
            print(f"Αποθηκεύτηκε: {Path(a.out).resolve()}")


if __name__ == "__main__":
    main()
