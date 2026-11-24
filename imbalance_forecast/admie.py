"""Αυτόματο κατέβασμα αρχείων ΑΔΜΗΕ (IPTO) και μετατροπή σε CSV του μοντέλου.

API (επαληθευμένο):
    https://www.admie.gr/getOperationMarketFilewRange?dateStart=YYYY-MM-DD&dateEnd=YYYY-MM-DD&FileCategory=<cat>
    -> JSON λίστα με file_path, file_fromdate, file_todate, file_published ("19.11.2026 09:25"), ...
    https://www.admie.gr/getFiletypeInfoEN -> κατάλογος κατηγοριών

Πηγές που χρησιμοποιούνται:
  * ISP1DayAheadLoadForecast / ISP1DayAheadRESForecast: πρόβλεψη φορτίου/ΑΠΕ ανά 15'. Βγαίνουν
    σε εκδόσεις (v1 ~09:20, v2 ~13:20 της D-1). Για πρόβλεψη ΠΡΙΝ τη DAM κρατάμε την τελευταία
    έκδοση που δημοσιεύθηκε πριν το DAM gate (version="pre_dam"), και για το ιστορικό, ώστε
    εκπαίδευση και παραγωγή να βλέπουν ίδιας ποιότητας πρόβλεψη.
  * ISP1ISPResults (D-1 ~18:45, δηλαδή ΜΕΤΑ τη DAM): «Energy Surplus» < 0 = πλεόνασμα, όταν ο ΑΔΜΗΕ
    κρατά θερμικές μονάδες για ευστάθεια/εφεδρεία και πρέπει να περικοπούν ΑΠΕ.
    -> stability = max(0, -Energy Surplus) σε MW. Για την D είναι άγνωστο πριν τη DAM (προβλέπεται),
       αλλά το ιστορικό έως και την D-1 είναι γνωστό.
  * IMBABE (εβδομαδιαίο, Πέμπτη για προηγ. Δευ–Κυρ): Imbalance Price, Curt_Flag, απόκλιση συστήματος, mFRR.
  * ISP2ISPResults (D ~00:25, όλη η ημέρα) / ISP3ISPResults (D ~12:25, 12:00-24:00): για πρόβλεψη
    ίδιας ημέρας -> stab_isp2/res_isp2/load_isp2 και stab_isp3/res_isp3/load_isp3.

Έξοδοι:
  * admie.csv:       timestamp, load_forecast, res_forecast, stability, imbalance, curt_flag  (για το μοντέλο)
  * admie_extra.csv: υπόλοιπα πεδία ISP/IMBABE για ανάλυση. ΔΕΝ δίνεται στο μοντέλο ως έχει, γιατί
                     αυτά δεν είναι γνωστά για την ημέρα-στόχο πριν τη DAM.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

API = "https://www.admie.gr/getOperationMarketFilewRange"
TYPES_API = "https://www.admie.gr/getFiletypeInfoEN"
UA = {"User-Agent": "Mozilla/5.0 (imbalance-forecast research script)"}
TZ = "Europe/Berlin"  # ώρα αγοράς (CET/CEST), βλ. config.tz

SOURCES: dict[str, dict] = {
    "load_forecast": {"category": "ISP1DayAheadLoadForecast", "parser": "forecast", "version": "pre_dam"},
    "res_forecast": {"category": "ISP1DayAheadRESForecast", "parser": "forecast", "version": "pre_dam"},
    "isp1": {"category": "ISP1ISPResults", "parser": "isp_results", "version": "latest",
             "rename": {"act": "act_isp1", "isp_price_up": "pup_isp1", "isp_price_down": "pdn_isp1"}},
    "imbabe": {"category": "IMBABE", "parser": "imbabe", "version": "latest"},
    # Ενδοημερήσια στάδια (πρόβλεψη ίδιας ημέρας): ISP2 ~00:25 της D (όλη η ημέρα),
    # ISP3 ~12:25 της D (μόνο 12:00-24:00). Στήλες με κατάληξη _isp2 / _isp3.
    "isp2": {"category": "ISP2ISPResults", "parser": "isp_results", "version": "latest", "suffix": "isp2"},
    "isp3": {"category": "ISP3ISPResults", "parser": "isp_results", "version": "latest", "suffix": "isp3"},
    # Διαθεσιμότητα μονάδων: ISP1 (D-1 ~09:40, πριν τη DAM), ISP2 (D-1 ~21:10), ISP3 (D ~09:10)
    "ua1": {"category": "ISP1UnitAvailabilities", "parser": "unit_avail", "version": "pre_dam", "prefix": "ua1"},
    "ua2": {"category": "ISP2UnitAvailabilities", "parser": "unit_avail", "version": "latest", "prefix": "ua2"},
    "ua3": {"category": "ISP3UnitAvailabilities", "parser": "unit_avail", "version": "latest", "prefix": "ua3"},
}
UA_FIELDS = ("thermal", "hydro", "n_out", "n_forced")
UA_COLS = tuple(f"ua{i}_{f}" for i in (1, 2, 3) for f in UA_FIELDS)
# ISP2/ISP3: στήλες μοντέλου (πλεόνασμα, ΑΠΕ και φορτίο του ISP)
STAGE_RENAME = {"stability": "stab", "isp_renewables": "res", "isp_load": "load", "act": "act",
                "isp_price_up": "pup", "isp_price_down": "pdn"}
MODEL_COLS = ["load_forecast", "res_forecast", "stability", "imbalance", "curt_flag",
              "stab_isp2", "res_isp2", "load_isp2", "stab_isp3", "res_isp3", "load_isp3", *UA_COLS,
              "act_isp1", "pup_isp1", "pdn_isp1", "act_isp2", "pup_isp2", "pdn_isp2",
              "act_isp3", "pup_isp3", "pdn_isp3", "net_act"]


# ---------------------------------------------------------------- HTTP
def _get(url: str, retries: int = 4, timeout: int = 90) -> bytes:
    err = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # δίκτυο / 5xx
            err = e
            time.sleep(2 ** (i + 1))
    raise RuntimeError(f"Αποτυχία λήψης {url}: {err}")


def list_types() -> list[dict]:
    return json.loads(_get(TYPES_API).decode("utf-8"))


def list_files(category: str, start, end) -> list[dict]:
    q = urllib.parse.urlencode(
        {
            "dateStart": pd.Timestamp(start).strftime("%Y-%m-%d"),
            "dateEnd": pd.Timestamp(end).strftime("%Y-%m-%d"),
            "FileCategory": category,
        }
    )
    data = json.loads(_get(f"{API}?{q}").decode("utf-8"))
    return data if isinstance(data, list) else []


def _meta(item: dict) -> dict:
    url = item["file_path"]
    name = Path(urllib.parse.urlparse(url).path).name
    m = re.search(r"_(\d+)\.xlsx?$", name)
    return {
        "url": url,
        "name": name,
        "day": pd.to_datetime(item.get("file_fromdate"), format="%d.%m.%Y", errors="coerce"),
        "published": pd.to_datetime(item.get("file_published"), format="%d.%m.%Y %H:%M", errors="coerce"),
        "version": int(m[1]) if m else 0,
    }


def select_versions(items: list[dict], policy: str, dam_gate: str = "12:00") -> list[dict]:
    """Μία έκδοση ανά ημέρα. pre_dam: η τελευταία που δημοσιεύθηκε πριν (D-1, dam_gate)."""
    by_day: dict = {}
    for it in map(_meta, items):
        if pd.notna(it["day"]):
            by_day.setdefault(it["day"], []).append(it)
    gate = dt.time.fromisoformat(dam_gate)
    out = []
    for day, its in sorted(by_day.items()):
        its.sort(key=lambda x: (x["version"], x["published"] if pd.notna(x["published"]) else pd.Timestamp.min))
        if policy == "pre_dam":
            limit = pd.Timestamp.combine((day - pd.Timedelta(days=1)).date(), gate)
            ok = [x for x in its if pd.notna(x["published"]) and x["published"] <= limit]
            out.append(ok[-1] if ok else its[0])  # αν δεν υπάρχει, η πρώτη (πιο κοντά σε pre-DAM)
        else:
            out.append(its[-1])
    return out


def download(category: str, start, end, raw_dir: Path, policy: str, sleep: float = 1.0,
             dam_gate: str = "12:00") -> list[dict]:
    """Κατεβάζει (με cache) μία έκδοση ανά ημέρα. Επιστρέφει metadata + path."""
    raw = Path(raw_dir) / category
    raw.mkdir(parents=True, exist_ok=True)
    items = []
    s = pd.Timestamp(start)
    while s <= pd.Timestamp(end):  # λίστες ανά μήνα
        e = min(s + pd.offsets.MonthEnd(0), pd.Timestamp(end))
        items += list_files(category, s, e)
        s = e + pd.Timedelta(days=1)
        time.sleep(sleep)
    chosen = select_versions(items, policy, dam_gate)
    for it in chosen:
        it["path"] = raw / it["name"]
        if not it["path"].exists():
            it["path"].write_bytes(_get(it["url"]))
            time.sleep(sleep)  # ευγένεια προς τον server
    return chosen


# ---------------------------------------------------------------- parsers
def _hours(day: pd.Timestamp, tz: str = TZ) -> int:
    t0 = pd.Timestamp(day).tz_localize(tz)
    t1 = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(tz)
    return int((t1 - t0) / pd.Timedelta(hours=1))


def _day_index(day: pd.Timestamp, n: int, tz: str = TZ) -> pd.DatetimeIndex | None:
    """n διαστήματα ξεκινώντας από τοπικά μεσάνυχτα (σωστό και σε ημέρες 23/25 ωρών)."""
    t0 = pd.Timestamp(day).tz_localize(tz)
    t1 = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(tz)
    hours = int((t1 - t0) / pd.Timedelta(hours=1))
    if n % hours or n // hours not in (1, 2, 4):
        return None
    step = pd.Timedelta(hours=1) / (n // hours)
    return pd.date_range(t0.tz_convert("UTC"), periods=n, freq=step).tz_convert(tz)


def _wallclock_series(day: pd.Timestamp, vals, tz: str = TZ) -> pd.Series | None:
    """Ημέρα αλλαγής ώρας με «κανονικό» πλήθος τιμών (24/48/96): αντιστοίχιση σε ώρα ρολογιού.

    Άνοιξη: η ώρα που δεν υπάρχει αγνοείται. Φθινόπωρο: η επαναλαμβανόμενη ώρα παίρνει την ίδια τιμή.
    """
    n = len(vals)
    if n not in (24, 48, 96):
        return None
    step = pd.Timedelta(hours=24) / n
    wall = pd.date_range(pd.Timestamp(day), periods=n, freq=step)
    loc = wall.tz_localize(tz, ambiguous=np.ones(n, dtype=bool), nonexistent="NaT")
    s = pd.Series(np.asarray(vals, dtype=float), index=loc)
    s = s[s.index.notna()]
    t0 = pd.Timestamp(day).tz_localize(tz).tz_convert("UTC")
    t1 = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(tz).tz_convert("UTC")
    grid = pd.date_range(t0, t1, freq=step, inclusive="left").tz_convert(tz)
    return s.reindex(grid).ffill(limit=int(pd.Timedelta(hours=1) / step))


def _is_num(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) and not pd.isna(v)


def parse_forecast(path: Path, day: pd.Timestamp, tz: str = TZ) -> pd.Series | None:
    """ISP1DayAhead{Load,RES}Forecast: γραμμή [ετικέτα, ετικέτα, ημερομηνία, v1..v96]."""
    sh = pd.read_excel(path, sheet_name=0, header=None)
    for _, row in sh.iterrows():
        if isinstance(row.iloc[0], str):
            vals = [float(v) for v in row.iloc[1:] if _is_num(v)]
            idx = _day_index(day, len(vals), tz)
            if idx is not None:
                return pd.Series(vals, index=idx)
            ws = _wallclock_series(day, vals, tz) if _hours(day, tz) != 24 else None
            if ws is not None:
                return ws
    return None


ISP_ROWS = {
    "isp_energy_surplus": "Energy Surplus",
    "isp_thermal": "Thermal Units",
    "isp_renewables": "Renewables",
    "isp_hydro": "Hydro Units",
    "isp_load": "Greece Load",
    "isp_price_up": "Price Energy Up",
    "isp_price_down": "Price Energy Down",
    "isp_afrr_up": "aFRR Up",
    "isp_mfrr_up": "mFRR Up",
    # σχεδιασμένη ενεργοποίηση εξισορρόπησης του ISP (MW)
    "isp_cleared_up": "Cleared Energy Up",
    "isp_cleared_down": "Cleared Energy Down",
}


def parse_isp_results(path: Path, day: pd.Timestamp, tz: str = TZ) -> pd.DataFrame | None:
    """ISP1ISPResults, πρώτο φύλλο: γραμμές με ετικέτα στη στήλη 0 και τιμές ανά 15'."""
    sh = pd.read_excel(path, sheet_name=0, header=None)
    out = {}
    ts_cols: list[int] = []
    first_stamps = None
    for _, row in sh.iterrows():
        lab = row.iloc[0]
        if not isinstance(lab, str):
            continue
        # γραμμή-επικεφαλίδα: χρονοσφραγίδες ανά 15' (+ στήλη TOTAL που αγνοείται)
        stamps = pd.to_datetime(pd.Series(row.iloc[1:].values, dtype=object).astype(str), errors="coerce",
                                format="%Y-%m-%d %H:%M:%S")
        if stamps.notna().sum() >= 20:
            ts_cols = [j + 1 for j in np.flatnonzero(stamps.notna().values)]
            if first_stamps is None:
                first_stamps = pd.DatetimeIndex(stamps.dropna().values)
            continue
        for key, want in ISP_ROWS.items():
            if key in out or lab.strip() != want or not ts_cols:
                continue
            vals = [float(row.iloc[j]) if _is_num(row.iloc[j]) else np.nan for j in ts_cols]
            if not np.all(np.isnan(vals)):
                out[key] = vals
    if not out:
        return None
    n = max(len(v) for v in out.values())
    starts_midnight = first_stamps is None or (first_stamps[0].hour == 0 and first_stamps[0].minute == 0)
    idx = _day_index(day, n, tz) if starts_midnight else None
    if idx is None and first_stamps is not None and len(first_stamps) == n:
        # μερική ημέρα (π.χ. ISP3: 12:00-24:00) -> ώρες της επικεφαλίδας (ώρα αγοράς)
        idx = first_stamps.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
    if idx is None:
        return None
    df = pd.DataFrame({k: v for k, v in out.items() if len(v) == n}, index=idx)
    if "isp_energy_surplus" in df:
        df["stability"] = (-df["isp_energy_surplus"]).clip(lower=0) + 0.0
    if {"isp_cleared_up", "isp_cleared_down"} <= set(df.columns):
        df["act"] = df["isp_cleared_up"].fillna(0) - df["isp_cleared_down"].fillna(0)  # καθαρή, + = προς τα πάνω
    return df


IMBABE_COLS = {
    "imbalance": r"^imbalance price",
    "curt_flag": r"^curt_flag",
    "system_deviation": r"^system deviation",
    "mfrr_price_up": r"^mfrr price up",
    "mfrr_price_down": r"^mfrr price down",
    "bal_energy_up": r"^total activated balancing energy up",
    "bal_energy_down": r"^total activated balancing energy down",
}


def parse_imbabe(path: Path, day=None, tz: str = TZ) -> pd.DataFrame | None:
    """IMBABE: πίνακας με STARTDATE (τοπική ώρα) και στήλες τιμών."""
    df = pd.read_excel(path, sheet_name=0)
    norm = {c: re.sub(r"\s+", " ", str(c).replace("\xa0", " ")).strip().lower() for c in df.columns}
    start_col = next((c for c, n in norm.items() if n == "startdate"), None)
    if start_col is None:
        return None
    ts = pd.to_datetime(df[start_col], errors="coerce")
    ok = ts.notna()
    df, ts = df[ok], ts[ok]
    idx = pd.DatetimeIndex(ts).tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
    out = pd.DataFrame(index=idx)
    for key, rx in IMBABE_COLS.items():
        col = next((c for c, n in norm.items() if re.search(rx, n)), None)
        if col is not None:
            out[key] = pd.to_numeric(df[col], errors="coerce").values
    return out


HYDRO_RX = re.compile(
    r"KREMASTA|KASTRAKI|STRATOS|ILARIONAS|POLYFYTO|SFIKIA|ASOMATA|THESAVROS|PLATANOVRYSI|POURNARI|P_AOOU|"
    r"LADONAS|PLASTIRAS|EDESSAIOS|AGRAS|MESOCHORA|GIONA|PIGES|AOOS|MAKROCHORI|GKIOFYRI", re.I)
FORCED_RX = re.compile(r"major outage|forced|fault|damage|tripp|βλάβη|unplanned", re.I)


def parse_unit_avail(path: Path, day: pd.Timestamp, tz: str = TZ) -> pd.DataFrame | None:
    """ISPxUnitAvailabilities: διαθεσιμότητα ανά μονάδα για την ημέρα D -> ημερήσια μεγέθη σε κάθε 15λεπτο.

    ua_thermal / ua_hydro: εκτιμώμενη μέγιστη διαθέσιμη ισχύς (MW) για τη D,
    ua_n_out: μονάδες με 0 MW, ua_n_forced: μονάδες με βλάβη (π.χ. «Major Outage»).
    """
    sh = pd.read_excel(path, sheet_name=0, header=None)
    rows = []
    for _, r in sh.iterrows():
        if not _is_num(r.iloc[0]) or not isinstance(r.iloc[1], str):
            continue
        est = pd.to_numeric(r.iloc[3], errors="coerce") if len(r) > 3 else np.nan
        com = str(r.iloc[4]) if len(r) > 4 and pd.notna(r.iloc[4]) else ""
        rows.append((r.iloc[1].strip(), est, com))
    if not rows:
        return None
    u = pd.DataFrame(rows, columns=["unit", "mw", "comment"])
    hyd = u["unit"].str.contains(HYDRO_RX)
    vals = {
        "ua_thermal": u.loc[~hyd, "mw"].sum(),
        "ua_hydro": u.loc[hyd, "mw"].sum(),
        "ua_n_out": int((u.loc[~hyd, "mw"].fillna(0) <= 0).sum()),
        "ua_n_forced": int(u["comment"].str.contains(FORCED_RX).sum()),
    }
    idx = _day_index(day, _hours(day, tz) * 4, tz)
    return pd.DataFrame({k: float(v) for k, v in vals.items()}, index=idx)


PARSERS = {"forecast": parse_forecast, "isp_results": parse_isp_results, "imbabe": parse_imbabe,
           "unit_avail": parse_unit_avail}


# ---------------------------------------------------------------- κύρια ροή
def _merge_csv(df: pd.DataFrame, path: Path, tz: str) -> pd.DataFrame:
    if path.exists():
        old = pd.read_csv(path)
        old.index = pd.to_datetime(old.pop(old.columns[0]), utc=True).dt.tz_convert(tz)
        cols = list(old.columns) + [c for c in df.columns if c not in old.columns]
        df = df.combine_first(old)[cols]  # σταθερή σειρά στηλών -> μικρά diffs στο git
    df = df.sort_index()
    df.index.name = "timestamp"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def fetch_admie(start, end, out_csv="data/admie.csv", raw_dir="data/raw/admie", extra_csv=None,
                only: list[str] | None = None, sleep: float = 1.0, dam_gate: str = "12:00",
                tz: str = TZ) -> pd.DataFrame:
    """Κατεβάζει τις πηγές και γράφει admie.csv (+ admie_extra.csv)."""
    out_csv = Path(out_csv)
    extra_csv = Path(extra_csv) if extra_csv else out_csv.with_name(out_csv.stem + "_extra.csv")
    frames = []
    for name, src in SOURCES.items():
        if only and name not in only:
            continue
        print(f"[{name}] {src['category']} ({src['version']}) ...", flush=True)
        try:
            files = download(src["category"], start, end, Path(raw_dir), src["version"], sleep, dam_gate)
        except Exception as e:
            print(f"  ! αποτυχία: {e}")
            continue
        parts, bad = [], []
        for f in files:
            try:
                p = PARSERS[src["parser"]](f["path"], f["day"], tz)
            except Exception as e:
                p = None
                bad.append(f"{f['name']} ({e.__class__.__name__})")
            if p is None:
                bad.append(f["name"]) if not bad or not bad[-1].startswith(f["name"]) else None
                continue
            parts.append(p.to_frame(name) if isinstance(p, pd.Series) else p)
        late = [f["name"] for f in files if src["version"] == "pre_dam" and pd.notna(f["published"])
                and f["published"] > pd.Timestamp.combine((f["day"] - pd.Timedelta(days=1)).date(),
                                                          dt.time.fromisoformat(dam_gate))]
        print(f"  {len(files)} αρχεία, {len(parts)} OK"
              + (f", {len(bad)} απέτυχαν (π.χ. {bad[0]})" if bad else "")
              + (f", {len(late)} χωρίς έκδοση πριν τη DAM" if late else ""))
        if parts:
            d = pd.concat(parts)
            if src.get("rename"):
                d = d.rename(columns=src["rename"])
            if src.get("prefix"):
                d = d.rename(columns=lambda c: f"{src['prefix']}_{c.removeprefix('ua_')}")
            if src.get("suffix"):
                sfx = src["suffix"]
                d = d.rename(columns=lambda c: f"{STAGE_RENAME[c]}_{sfx}" if c in STAGE_RENAME
                             else f"{sfx}_{c.removeprefix('isp_')}")
            frames.append(d[~d.index.duplicated(keep="last")])
    if not frames:
        raise RuntimeError("Δεν ανακτήθηκαν δεδομένα.")
    allf = pd.concat(frames, axis=1).sort_index()
    if {"bal_energy_up", "bal_energy_down"} <= set(allf.columns):  # πραγματική καθαρή ενεργοποίηση (IMBABE)
        ok = allf["bal_energy_up"].notna() | allf["bal_energy_down"].notna()
        allf["net_act"] = (allf["bal_energy_up"].fillna(0) - allf["bal_energy_down"].fillna(0)).where(ok)
    model = allf[[c for c in MODEL_COLS if c in allf]]
    extra = allf[[c for c in allf.columns if c not in MODEL_COLS]]
    m = _merge_csv(model, out_csv, tz)
    print(f"Αποθηκεύτηκε {out_csv}: {m.index.min()} .. {m.index.max()} | {list(m.columns)}")
    if not extra.empty:
        x = _merge_csv(extra, extra_csv, tz)
        print(f"Αποθηκεύτηκε {extra_csv}: {list(x.columns)}")
    return m
