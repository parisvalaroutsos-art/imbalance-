"""Αυτόματο κατέβασμα αρχείων ΑΔΜΗΕ (IPTO) και μετατροπή σε CSV του μοντέλου.

Ο ΑΔΜΗΕ δίνει λίστα αρχείων ανά κατηγορία και εύρος ημερομηνιών:

    https://www.admie.gr/getOperationMarketFilewRange?dateStart=YYYY-MM-DD&dateEnd=YYYY-MM-DD&FileCategory=<cat>

Η απάντηση είναι JSON λίστα με στοιχεία που περιέχουν (μεταξύ άλλων) το `file_path` (URL του xlsx).

ΣΗΜΑΝΤΙΚΟ: τα ονόματα κατηγοριών και η διάταξη των xlsx δεν έχουν επαληθευτεί από αυτό
το περιβάλλον (δεν υπήρχε πρόσβαση στο admie.gr). Γι' αυτό:
  * όλα ορίζονται στο DEFAULT_SPEC και μπορούν να αλλάξουν με JSON (--spec),
  * `python -m imbalance_forecast admie-list --category X --date D` δείχνει τι επιστρέφει το API,
  * `python -m imbalance_forecast admie-inspect --file F.xlsx` δείχνει τις γραμμές/στήλες ενός αρχείου,
    ώστε να ρυθμιστεί το `label` (regex) κάθε σειράς.
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

API = "https://www.admie.gr/getOperationMarketFilewRange"
BASE = "https://www.admie.gr"
UA = {"User-Agent": "Mozilla/5.0 (imbalance-forecast)"}

# name -> {category, label (regex γραμμής/στήλης στο xlsx), sheet (προαιρετικό), agg}
# Για πρόβλεψη ΠΡΙΝ τη DAM χρειάζονται οι προβλέψεις ΑΔΜΗΕ που δημοσιεύονται D-1 πρωί.
DEFAULT_SPEC: dict[str, dict] = {
    "load_forecast": {
        "category": "ISP1DayAheadLoadForecast",
        "label": r"(?i)load\s*forecast|φορτ",
    },
    "res_forecast": {
        "category": "ISP1DayAheadRESForecast",
        "label": r"(?i)res\s*forecast|^\s*res\b|απε",
    },
    # Ευστάθεια: ποσότητα ΑΠΕ που περικόπτει ο ΑΔΜΗΕ στο ISP (MW). Αν το αρχείο δίνει
    # προγραμματισμένες ΑΠΕ αντί για περικοπή, ορίστε "derive": "res_forecast_minus".
    "stability": {
        "category": "ISP1ISPResults",
        "label": r"(?i)curtail|περικοπ",
    },
    # Τιμές απόκλισης (εβδομαδιαία δημοσίευση)
    "imbalance": {
        "category": "IMBABE",
        "label": r"(?i)imbalance\s*price|τιμή\s*απόκλισης|imbp",
    },
}


# ---------------------------------------------------------------- HTTP
def _get(url: str, retries: int = 4, timeout: int = 60) -> bytes:
    err = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # δίκτυο / 5xx
            err = e
            time.sleep(2**i)
    raise RuntimeError(f"Αποτυχία λήψης {url}: {err}")


def list_files(category: str, start, end) -> list[dict]:
    q = urllib.parse.urlencode(
        {
            "dateStart": pd.Timestamp(start).strftime("%Y-%m-%d"),
            "dateEnd": pd.Timestamp(end).strftime("%Y-%m-%d"),
            "FileCategory": category,
        }
    )
    data = json.loads(_get(f"{API}?{q}").decode("utf-8"))
    return data if isinstance(data, list) else data.get("data", [])


def _file_url(item: dict) -> str:
    p = item.get("file_path") or item.get("filePath") or item.get("url") or ""
    return p if p.startswith("http") else urllib.parse.urljoin(BASE, p)


def _file_date(item: dict, url: str) -> pd.Timestamp | None:
    """Ημερομηνία παράδοσης (target day) από το όνομα αρχείου: YYYYMMDD_..."""
    m = re.search(r"(20\d{2})(\d{2})(\d{2})", Path(urllib.parse.urlparse(url).path).name)
    if m:
        return pd.Timestamp(f"{m[1]}-{m[2]}-{m[3]}")
    for k in ("file_fromdate", "fromdate", "file_date"):
        if item.get(k):
            return pd.to_datetime(item[k], dayfirst=True, errors="coerce")
    return None


def download(category: str, start, end, raw_dir: Path) -> list[tuple[pd.Timestamp, Path]]:
    """Κατεβάζει (με cache) όλα τα αρχεία μιας κατηγορίας. Επιστρέφει [(ημέρα, path)]."""
    raw_dir = Path(raw_dir) / category
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = []
    # σε κομμάτια του μήνα για να μην είναι τεράστια η απάντηση
    for s in pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="MS").union([pd.Timestamp(start)]):
        e = min(s + pd.offsets.MonthEnd(0), pd.Timestamp(end))
        for item in list_files(category, s, e):
            url = _file_url(item)
            if not url:
                continue
            name = Path(urllib.parse.urlparse(url).path).name
            dest = raw_dir / name
            if not dest.exists():
                dest.write_bytes(_get(url))
            out.append((_file_date(item, url), dest))
    # κράτα την τελευταία έκδοση ανά ημέρα (π.χ. _01, _02 αναθεωρήσεις)
    best: dict = {}
    for d, p in sorted(out, key=lambda t: t[1].name):
        if d is not None:
            best[d] = p
    return sorted(best.items())


# ---------------------------------------------------------------- xlsx parsing
def _day_slots(day: pd.Timestamp, tz: str) -> int:
    t0 = pd.Timestamp(day).tz_localize(tz)
    t1 = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(tz)
    return int((t1 - t0) / pd.Timedelta(hours=1))  # 23 / 24 / 25


def _numeric_run(values) -> np.ndarray:
    v = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    idx = np.flatnonzero(~np.isnan(v))
    if len(idx) == 0:
        return v[:0]
    return v[idx[0]: idx[-1] + 1]


def extract_series(path: Path, label: str, day: pd.Timestamp, tz: str = "Europe/Athens", sheet=None, agg="sum"):
    """Βρίσκει τη σειρά με ετικέτα label (γραμμή ή στήλη) και την αντιστοιχίζει σε χρονοσειρά.

    Αν βρεθούν πολλές γραμμές (π.χ. ανά ζώνη/μονάδα) αθροίζονται (agg="sum") ή μέσος (agg="mean").
    """
    sheets = pd.read_excel(path, sheet_name=sheet if sheet is not None else None, header=None)
    if isinstance(sheets, pd.DataFrame):
        sheets = {sheet: sheets}
    rx = re.compile(label)
    hours = _day_slots(day, tz)
    def valid(v):
        return len(v) > 0 and len(v) % hours == 0 and len(v) // hours in (1, 2, 4)

    found = []
    for _, sh in sheets.items():
        sh = sh.dropna(how="all").dropna(axis=1, how="all")
        cand = []
        # 1) γραμμή: ετικέτα σε μία από τις πρώτες στήλες, τιμές δεξιά
        for i in range(len(sh)):
            row = sh.iloc[i]
            for j in range(min(3, len(row))):
                if isinstance(row.iloc[j], str) and rx.search(row.iloc[j]):
                    cand.append(_numeric_run(row.iloc[j + 1:].values))
                    break
        cand = [v for v in cand if valid(v)]
        # 2) στήλη: ετικέτα σε κεφαλίδα, τιμές από κάτω
        if not cand:
            for j in range(sh.shape[1]):
                col = sh.iloc[:, j]
                for i in range(min(10, len(col))):
                    if isinstance(col.iloc[i], str) and rx.search(col.iloc[i]):
                        cand.append(_numeric_run(col.iloc[i + 1:].values))
                        break
            cand = [v for v in cand if valid(v)]
        found += cand
    if not found:
        return None
    n = len(found[0])
    found = [v for v in found if len(v) == n]
    vals = np.nansum(found, axis=0) if agg == "sum" else np.nanmean(found, axis=0)
    step = pd.Timedelta(hours=1) / (n // hours)
    t0 = pd.Timestamp(day).tz_localize(tz).tz_convert("UTC")
    idx = pd.date_range(t0, periods=n, freq=step).tz_convert(tz)
    return pd.Series(vals, index=idx)


def inspect(path: Path, max_rows: int = 40) -> str:
    """Σύνοψη ενός xlsx: φύλλα, πρώτες γραμμές/στήλες, για ρύθμιση των labels."""
    out = []
    for name, sh in pd.read_excel(path, sheet_name=None, header=None).items():
        sh = sh.dropna(how="all").dropna(axis=1, how="all")
        out.append(f"=== φύλλο '{name}' {sh.shape}")
        out.append(sh.iloc[:max_rows, :6].to_string())
    return "\n".join(out)


# ---------------------------------------------------------------- κύρια ροή
def fetch_admie(start, end, out_csv, raw_dir="data/raw/admie", spec: dict | None = None,
                only: list[str] | None = None, tz: str = "Europe/Athens") -> pd.DataFrame:
    """Κατεβάζει και ενοποιεί τις σειρές του spec σε ένα CSV (timestamp + μία στήλη ανά σειρά)."""
    spec = spec or DEFAULT_SPEC
    cols = {}
    for name, sp in spec.items():
        if only and name not in only:
            continue
        print(f"[{name}] κατηγορία {sp['category']} ...")
        try:
            files = download(sp["category"], start, end, Path(raw_dir))
        except Exception as e:
            print(f"  ! αποτυχία λίστας/λήψης: {e}")
            continue
        parts, bad = [], []
        for day, path in files:
            try:
                s = extract_series(path, sp["label"], day, tz, sp.get("sheet"), sp.get("agg", "sum"))
            except Exception as e:  # κατεστραμμένο/άλλο format
                s, _ = None, e
            (parts.append(s) if s is not None else bad.append(path.name))
        print(f"  {len(files)} αρχεία, {len(parts)} ημέρες ΟΚ" + (f", {len(bad)} χωρίς ταίριασμα (π.χ. {bad[0]})" if bad else ""))
        if parts:
            s = pd.concat(parts)
            cols[name] = s[~s.index.duplicated(keep="last")]
    if not cols:
        raise RuntimeError("Δεν ανακτήθηκε καμία σειρά. Ελέγξτε κατηγορίες/labels με admie-list / admie-inspect.")
    df = pd.DataFrame(cols).sort_index()
    for name, sp in spec.items():
        if sp.get("derive") == "res_forecast_minus" and name in df and "res_forecast" in df:
            # το αρχείο δίνει προγραμματισμένες ΑΠΕ -> περικοπή = πρόβλεψη - πρόγραμμα
            df[name] = (df["res_forecast"].reindex(df.index) - df[name]).clip(lower=0)
    df.index.name = "timestamp"
    out_csv = Path(out_csv)
    if out_csv.exists():  # σταδιακή ενημέρωση: νέα δεδομένα υπερισχύουν
        old = pd.read_csv(out_csv)
        old.index = pd.to_datetime(old.pop(old.columns[0]), utc=True).dt.tz_convert(tz)
        df = df.combine_first(old).sort_index()
        df.index.name = "timestamp"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv)
    print(f"Αποθηκεύτηκε {out_csv}: {df.index.min()} .. {df.index.max()} | {list(df.columns)}")
    return df
