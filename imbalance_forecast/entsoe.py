"""ENTSO-E Transparency: πραγματική παραγωγή ανά τεχνολογία (A75) για την Ελλάδα, ανά 15'.

Το κλειδί API (security token) ΔΕΝ υπάρχει στον κώδικα: ο proxy του περιβάλλοντος το προσθέτει στο
header SECURITY_TOKEN. Εκτός τέτοιου περιβάλλοντος, ορίστε τη μεταβλητή ENTSOE_TOKEN.

Δημοσιεύεται με ~30-60 λεπτά καθυστέρηση, οπότε σε συνδυασμό με το πρόγραμμα του ISP δείχνει τι
ενεργοποίησε ο ΑΔΜΗΕ (ελεγχόμενη παραγωγή πάνω/κάτω από το πρόγραμμα) πολύ πριν το IMBABE.
"""
from __future__ import annotations

import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

API = "https://web-api.tp.entsoe.eu/api"
GR = "10YGR-HTSO-----Y"
NS = {"n": "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"}
PSR = {
    "B02": "lignite", "B04": "gas", "B06": "oil", "B10": "hydro_pumped", "B11": "hydro_ror",
    "B12": "hydro_res", "B16": "solar", "B19": "wind", "B01": "biomass", "B25": "storage", "B20": "other",
}
STEP = {"PT15M": pd.Timedelta(minutes=15), "PT60M": pd.Timedelta(hours=1), "PT30M": pd.Timedelta(minutes=30)}


def _get(params: dict, retries: int = 4) -> bytes:
    q = urllib.parse.urlencode(params)
    headers = {"User-Agent": "imbalance-forecast research script"}
    if os.environ.get("ENTSOE_TOKEN"):
        headers["SECURITY_TOKEN"] = os.environ["ENTSOE_TOKEN"]
    err = None
    for i in range(retries):
        try:
            req = urllib.request.Request(f"{API}?{q}", headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:  # δίκτυο / 5xx / 429
            err = e
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"ENTSO-E: αποτυχία ({err})")


def parse_a75(xml: bytes) -> pd.DataFrame:
    """GL_MarketDocument -> DataFrame (UTC, 15') με στήλες ανά τεχνολογία· κατανάλωση ως <τεχνολογία>_cons."""
    root = ET.fromstring(xml)
    cols: dict[str, list[pd.Series]] = {}
    for ts in root.findall("n:TimeSeries", NS):
        code = ts.find(".//n:psrType", NS).text
        name = PSR.get(code, code)
        if ts.find("n:outBiddingZone_Domain.mRID", NS) is not None:  # κατανάλωση (άντληση, φόρτιση)
            name += "_cons"
        for per in ts.findall("n:Period", NS):
            start = pd.Timestamp(per.find("n:timeInterval/n:start", NS).text)
            end = pd.Timestamp(per.find("n:timeInterval/n:end", NS).text)
            step = STEP[per.find("n:resolution", NS).text]
            idx = pd.date_range(start, end, freq=step, inclusive="left")
            pts = {int(p.find("n:position", NS).text): float(p.find("n:quantity", NS).text)
                   for p in per.findall("n:Point", NS)}
            # καμπύλη A03: λείπουσες θέσεις = ίδια τιμή με την προηγούμενη, έως το τέλος της περιόδου
            # (για την τρέχουσα ημέρα το τέλος της περιόδου = τελευταίο δημοσιευμένο 15λεπτο)
            s = pd.Series([pts.get(i + 1) for i in range(len(idx))], index=idx, dtype=float).ffill()
            if step != STEP["PT15M"]:
                s = s.resample("15min").ffill()
            cols.setdefault(name, []).append(s)
    if not cols:
        return pd.DataFrame()
    df = pd.DataFrame({k: pd.concat(v).groupby(level=0).last() for k, v in cols.items()})
    df.index.name = "timestamp"
    return df.sort_index()


def fetch_generation(start, end, out_csv="data/entsoe_gen.csv", sleep: float = 1.0) -> pd.DataFrame:
    """Πραγματική παραγωγή ανά τεχνολογία [start, end) σε κομμάτια ενός μήνα· συγχώνευση με το υπάρχον CSV."""
    s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    parts = []
    while s < e:
        t = min(s + pd.offsets.MonthBegin(1), e)
        xml = _get({"documentType": "A75", "processType": "A16", "in_Domain": GR,
                    "periodStart": s.strftime("%Y%m%d%H%M"), "periodEnd": t.strftime("%Y%m%d%H%M")})
        if b"<TimeSeries>" in xml:
            parts.append(parse_a75(xml))
        print(f"  {s.date()} .. {t.date()}: {len(parts[-1]) if parts and b'<TimeSeries>' in xml else 0} 15λεπτα",
              flush=True)
        s = t
        time.sleep(sleep)
    if not parts:
        raise RuntimeError("ENTSO-E: δεν επιστράφηκαν δεδομένα")
    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    out = Path(out_csv)
    if out.exists():
        old = pd.read_csv(out, index_col=0)
        old.index = pd.to_datetime(old.index, utc=True)
        df = df.combine_first(old)[list(dict.fromkeys(list(old.columns) + list(df.columns)))]
    df.index.name = "timestamp"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print(f"Αποθηκεύτηκε {out}: {df.index.min()} .. {df.index.max()} | {list(df.columns)}")
    return df


def controllable(df: pd.DataFrame) -> pd.Series:
    """Ελεγχόμενη παραγωγή (MW): λιγνίτης + αέριο + πετρέλαιο + υδροηλεκτρικά (χωρίς άντληση)."""
    cols = [c for c in ("lignite", "gas", "oil", "hydro_res", "hydro_pumped", "hydro_ror") if c in df]
    return df[cols].fillna(0).sum(axis=1).where(df[cols].notna().any(axis=1))


BAL_NS = "urn:iec62325.351:tc57wg16:451-6:balancingdocument:4:1"


def parse_a84(xml: bytes) -> pd.DataFrame:
    """Τιμές ενεργοποίησης εξισορρόπησης (A84). Κρατά mFRR (businessType A97): mfrr_up / mfrr_dn σε €/MWh ανά 15'.

    Λείπουσα θέση = καμία ενεργοποίηση σε αυτή την κατεύθυνση (NaN, όχι συμπλήρωση).
    """
    root = ET.fromstring(xml)
    ns = {"n": root.tag.split("}")[0][1:]}
    cols: dict[str, list[pd.Series]] = {}
    for ts in root.findall("n:TimeSeries", ns):
        bt = ts.find("n:businessType", ns)
        if bt is not None and bt.text != "A97":
            continue
        d = ts.find("n:flowDirection.direction", ns).text
        name = {"A01": "mfrr_up", "A02": "mfrr_dn"}.get(d)
        if name is None:
            continue
        for per in ts.findall("n:Period", ns):
            start = pd.Timestamp(per.find("n:timeInterval/n:start", ns).text)
            step = STEP[per.find("n:resolution", ns).text]
            vals = {}
            for p in per.findall("n:Point", ns):
                a = p.find("n:activation_Price.amount", ns)
                if a is not None:
                    vals[start + step * (int(p.find("n:position", ns).text) - 1)] = float(a.text)
            if vals:
                cols.setdefault(name, []).append(pd.Series(vals, dtype=float))
    if not cols:
        return pd.DataFrame()
    df = pd.DataFrame({k: pd.concat(v).groupby(level=0).last() for k, v in cols.items()}).sort_index()
    df.index.name = "timestamp"
    return df


def fetch_mfrr_prices(start, end, out_csv="data/entsoe_mfrr.csv", sleep: float = 1.0) -> pd.DataFrame:
    """Τιμές ενεργοποίησης mFRR Ελλάδας [start, end) από το ENTSO-E· συγχώνευση με το υπάρχον CSV."""
    s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    parts = []
    while s < e:
        t = min(s + pd.Timedelta(days=7), e)
        xml = _get({"documentType": "A84", "processType": "A16", "controlArea_Domain": GR,
                    "periodStart": s.strftime("%Y%m%d%H%M"), "periodEnd": t.strftime("%Y%m%d%H%M")})
        if b"Balancing_MarketDocument" in xml:
            p = parse_a84(xml)
            if not p.empty:
                parts.append(p)
        s = t
        time.sleep(sleep)
    df = pd.concat(parts) if parts else pd.DataFrame(columns=["mfrr_up", "mfrr_dn"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    out = Path(out_csv)
    if out.exists():
        old = pd.read_csv(out, index_col=0)
        old.index = pd.to_datetime(old.index, utc=True)
        df = df.combine_first(old)
    df.index.name = "timestamp"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print(f"Αποθηκεύτηκε {out}: {len(df)} 15λεπτα με ενεργοποίηση mFRR")
    return df
