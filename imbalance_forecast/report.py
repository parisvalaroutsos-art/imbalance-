"""Αναφορά πρόβλεψης για αποστολή (π.χ. email): έλεγχος διαθεσιμότητας δεδομένων και ωριαίος πίνακας
σε ώρα Ελλάδας."""
from __future__ import annotations

import html

import pandas as pd

from .config import Config

LOCAL_TZ = "Europe/Athens"

# Τι πρέπει να έχει δημοσιευθεί για την ημέρα-στόχο D σε κάθε στάδιο (στήλη: ελάχιστα μη κενά 15λεπτα)
REQUIRED = {
    "pre_dam": {"dam": 92, "res_forecast": 92, "load_forecast": 92},
    "isp1": {"dam": 92, "ida1": 92, "stability": 92},
    "isp2": {"dam": 92, "ida2": 92, "stab_isp2": 92},
    "isp3": {"ida3": 40, "stab_isp3": 40},
}
STAGE_TITLE = {
    "pre_dam": "Πριν τη DAM / μετά τη DAM",
    "isp1": "Μετά τα αποτελέσματα ISP1 (D-1 βράδυ)",
    "isp2": "Μετά το ISP2 (D ~00:30)",
    "isp3": "Μετά το ISP3 (D ~12:30, ώρες 13-01 Ελλάδας)",
}
COLS = [
    ("dam", "DAM", "{:.1f}"),
    ("imb_point", "Imbalance (εκτίμηση)", "{:.1f}"),
    ("imb_q10", "q10", "{:.0f}"),
    ("imb_q90", "q90", "{:.0f}"),
    ("stab_prob", "P(ευστάθεια)", "{:.2f}"),
    ("p_curt", "P(περικοπή)", "{:.2f}"),
    ("p_imb_lt_dam", "P(imb<DAM)", "{:.2f}"),
    ("p_spike_up", "P(imb>DAM+100)", "{:.2f}"),
]


def missing(df: pd.DataFrame, stage: str, target_date, tz: str) -> dict[str, int]:
    """Στήλες που δεν έχουν ακόμα δημοσιευθεί για την ημέρα-στόχο -> {στήλη: πλήθος τιμών}."""
    d = pd.Timestamp(target_date).normalize()
    loc = df.index.tz_convert(tz).tz_localize(None).normalize()
    day = df[loc == d]
    out = {}
    for c, need in REQUIRED[stage].items():
        n = int(day[c].notna().sum()) if c in day else 0
        if n < need:
            out[c] = n
    return out


def hourly_table(out: pd.DataFrame) -> pd.DataFrame:
    """Ωριαίοι μέσοι όροι σε ώρα Ελλάδας (στήλη «ώρα» = έναρξη ώρας)."""
    cols = [c for c, _, _ in COLS if c in out]
    loc = out.index.tz_convert(LOCAL_TZ)
    h = out[cols].groupby(loc.floor("h")).mean()
    h.index = [f"{t:%H}-{(t + pd.Timedelta(hours=1)):%H}" for t in h.index]
    return h


def _cell_style(col: str, v: float) -> str:
    if pd.isna(v):
        return ""
    if col == "p_curt" and v >= 0.3 or col == "stab_prob" and v >= 0.9:
        return "background:#dbe9ff"
    if col == "p_spike_up" and v >= 0.2:
        return "background:#ffe1d6"
    if col == "imb_point" and v < 0:
        return "background:#dbe9ff;font-weight:bold"
    return ""


def to_html(out: pd.DataFrame, cfg: Config, target_date, run_label: str, notes: list[str]) -> str:
    h = hourly_table(out)
    names = {c: n for c, n, _ in COLS}
    fmts = {c: f for c, _, f in COLS}
    th = "".join(f"<th style='padding:4px 8px;border-bottom:2px solid #555'>{html.escape(names[c])}</th>"
                 for c in h.columns)
    rows = []
    for idx, r in h.iterrows():
        tds = "".join(
            f"<td style='padding:3px 8px;text-align:right;{_cell_style(c, r[c])}'>"
            f"{'' if pd.isna(r[c]) else fmts[c].format(r[c])}</td>"
            for c in h.columns
        )
        rows.append(f"<tr><td style='padding:3px 8px'>{idx}</td>{tds}</tr>")
    d = pd.Timestamp(target_date)
    info = (f"Issue: {out.attrs.get('issue_date')} | imbalance γνωστό έως {out.attrs.get('imbalance_known_through')}"
            f" | IDA: {', '.join(cfg.ida_available) or 'καμία'}")
    notes_html = "".join(f"<li>{html.escape(n)}</li>" for n in notes)
    return f"""<div style="font-family:Arial,sans-serif;font-size:13px">
<h3 style="margin:0 0 4px">Πρόβλεψη imbalance {d:%d/%m/%Y} — {html.escape(run_label)}</h3>
<p style="margin:0 0 8px;color:#555">{html.escape(STAGE_TITLE.get(cfg.stage, cfg.stage))}. {html.escape(info)}</p>
<table style="border-collapse:collapse;font-size:12px">
<tr><th style='padding:4px 8px;border-bottom:2px solid #555'>Ώρα Ελλάδας</th>{th}</tr>
{''.join(rows)}
</table>
<p style="color:#555;margin:8px 0 2px">€/MWh, ωριαίοι μέσοι όροι των 15λέπτων. Μπλε: πιθανή ευστάθεια/περικοπή
(imbalance προς το κατώτατο όριο). Πορτοκαλί: αυξημένος κίνδυνος ανοδικού σπάικ.</p>
<ul style="color:#555;margin:2px 0">{notes_html}</ul>
</div>"""
