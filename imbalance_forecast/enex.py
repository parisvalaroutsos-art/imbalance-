"""Κατέβασμα τιμών DAM και IDA από το Ελληνικό Χρηματιστήριο Ενέργειας (ΕΧΕ / HEnEx).

Πηγές (www.enexgroup.gr, επαληθευμένες):
  * Ετήσια αρχεία:   /web/guest/dam-idm-archive  -> "YYYY_EL-DAM-IDAs_ResultsSummary" (zip με ημερήσια xlsx
    DAM, IDA1-3 ή CRIDA1-3 για παλιότερα έτη)
  * Τρέχον έτος:     /web/guest/markets-publications-el-day-ahead-market  (DAM)
                     /web/guest/markets-publications-el-intraday-market   (IDA1, IDA2, IDA3)
    σελιδοποιημένες λίστες (7 αρχεία/σελίδα) με συνδέσμους /c/document_library/get_file?uuid=...

Αρχείο YYYYMMDD_EL-<MARKET>_ResultsSummary_EN_vNN.xlsx: γραμμή «Market Clearing Price» και από κάτω
«Greece Mainland (15min MCP)» (από 1/10/2025) ή «Greece Mainland» (ωριαία, παλιότερα).

Έξοδος: prices.csv με timestamp, dam, ida1, ida2, ida3 (CRIDA1-3 -> ida1-3). Η πρόσβαση γίνεται με παύση
ανάμεσα στα αιτήματα (--sleep) και τα αρχεία μένουν αποθηκευμένα τοπικά (cache).
"""
from __future__ import annotations

import html as htmlmod
import io
import re
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://www.enexgroup.gr"
ARCHIVE_PAGE = "/web/guest/dam-idm-archive"
PUB_PAGES = {
    "DAM": "/web/guest/markets-publications-el-day-ahead-market",
    "IDA1": "/web/guest/markets-publications-el-intraday-market",
    "IDA2": "/web/guest/markets-publications-el-intraday-market",
    "IDA3": "/web/guest/markets-publications-el-intraday-market",
}
COL = {"DAM": "dam", "IDA1": "ida1", "IDA2": "ida2", "IDA3": "ida3",
       "CRIDA1": "ida1", "CRIDA2": "ida2", "CRIDA3": "ida3"}
UA = {"User-Agent": "Mozilla/5.0 (imbalance-forecast research script)"}
TZ = "Europe/Athens"
FILE_RX = re.compile(r"(\d{8})_EL-(DAM|IDA[123]|CRIDA[123])_ResultsSummary_EN_v(\d+)")
PORTLET = "com_liferay_asset_publisher_web_portlet_AssetPublisherPortlet_INSTANCE_"


def _get(url: str, retries: int = 4, timeout: int = 600) -> bytes:
    url = url if url.startswith("http") else BASE + url
    err = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception as e:
            err = e
            time.sleep(2 ** (i + 1))
    raise RuntimeError(f"Αποτυχία λήψης {url}: {err}")


# ---------------------------------------------------------------- parsing
def _day_index(day: pd.Timestamp, n: int, step: pd.Timedelta, tz: str = TZ) -> pd.DatetimeIndex:
    t0 = pd.Timestamp(day).tz_localize(tz).tz_convert("UTC")
    return pd.date_range(t0, periods=n, freq=step).tz_convert(tz)


def _hours_in_day(day: pd.Timestamp, tz: str = TZ) -> int:
    t0 = pd.Timestamp(day).tz_localize(tz)
    t1 = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(tz)
    return int((t1 - t0) / pd.Timedelta(hours=1))


def parse_results_summary(data: bytes, day: pd.Timestamp, tz: str = TZ) -> pd.Series | None:
    """Τιμή εκκαθάρισης (Greece Mainland) ανά MTU από ένα ResultsSummary xlsx."""
    sh = pd.read_excel(io.BytesIO(data), sheet_name=0, header=None)
    # ανάλυση: στις πρώτες γραμμές, απόσταση στηλών ανάμεσα στους αριθμούς 1 και 2 (ωρών/MTU)
    spacing = None
    for i in range(min(4, len(sh))):
        row = list(sh.iloc[i, 1:])
        pos = {v: j for j, v in enumerate(row) if isinstance(v, (int, float)) and v in (1, 2) and v == v}
        if 1 in pos and 2 in pos:
            spacing = pos[2] - pos[1]
            break
    per_hour = 4 if spacing == 4 else 1
    n = _hours_in_day(day, tz) * per_hour

    labels = sh.iloc[:, 0].map(lambda v: v if isinstance(v, str) else "").str.replace(r"\s+", " ", regex=True).str.strip()
    start = labels[labels.str.contains("Market Clearing Price", case=False)].index
    if len(start) == 0:
        return None
    cand = labels.loc[start[0] + 1: start[0] + 4]
    row_i = next((i for i, l in cand.items() if "15min MCP" in l), None)
    if row_i is None:
        row_i = next((i for i, l in cand.items() if l.startswith("Greece Mainland")), None)
    if row_i is None:
        return None
    vals = pd.to_numeric(sh.iloc[row_i, 1: 1 + n], errors="coerce").to_numpy(dtype=float)
    if len(vals) < n or np.all(np.isnan(vals)):
        return None
    step = pd.Timedelta(minutes=15) if per_hour == 4 else pd.Timedelta(hours=1)
    return pd.Series(vals, index=_day_index(day, n, step, tz))


# ---------------------------------------------------------------- πηγές
def _parse_name(name: str):
    m = FILE_RX.search(name)
    return (pd.Timestamp(m[1]), m[2], int(m[3])) if m else None


def archive_links() -> dict[int, str]:
    """{έτος: url} για τα ετήσια zip ResultsSummary."""
    s = htmlmod.unescape(_get(ARCHIVE_PAGE).decode("utf-8", "replace"))
    out = {}
    for url, text in re.findall(r'href="([^"]*get_file\?uuid=[^"]+)"[^>]*>\s*(?:<[^>]+>\s*)*([^<]+)', s):
        m = re.match(r"\s*(20\d\d)_EL-[A-Za-z_-]*ResultsSummary", text)
        if m and "PrelimResults" not in text:
            out[int(m[1])] = url
    return out


def _sections(page_html: str) -> dict[str, tuple[str, str]]:
    """{MARKET: (instance, html τμήματος)} για τα τμήματα ResultsSummary μιας σελίδας δημοσιεύσεων."""
    out = {}
    for m in re.finditer(rf'id="p_p_id_{PORTLET}([A-Za-z0-9]+)_"', page_html):
        seg = page_html[m.end(): m.end() + 120000]
        nxt = seg.find(f"p_p_id_{PORTLET}")
        seg = seg[: nxt if nxt > 0 else None]
        kinds = set(re.findall(r"\d{8}_EL-(DAM|IDA[123])_ResultsSummary_EN", seg))
        if len(kinds) == 1:
            out[kinds.pop()] = (m.group(1), seg)
    return out


def _links(seg: str) -> list[tuple[str, str]]:
    seg = htmlmod.unescape(seg)
    return re.findall(r'href="([^"]*get_file\?uuid=[^"]+)"[^>]*>\s*(?:<[^>]+>\s*)*(?:&nbsp;|\s)*([^<\s]+)', seg)


def current_files(market: str, start: pd.Timestamp, sleep: float = 1.0, max_pages: int = 80) -> list[tuple]:
    """Αρχεία της σελίδας δημοσιεύσεων για μια αγορά, από τα νεότερα προς τα παλιότερα, έως το start."""
    page = PUB_PAGES[market]
    first = _get(page).decode("utf-8", "replace")
    secs = _sections(first)
    if market not in secs:
        return []
    inst, seg = secs[market]
    out = []
    for cur in range(1, max_pages + 1):
        if cur > 1:
            time.sleep(sleep)
            q = f"?p_p_id={PORTLET}{inst}&p_p_lifecycle=0&_{PORTLET}{inst}_cur={cur}"
            seg = _sections(_get(page + q).decode("utf-8", "replace")).get(market, (inst, ""))[1]
        got = [(u, t) for u, t in _links(seg) if _parse_name(t)]
        if not got:
            break
        out += got
        if min(_parse_name(t)[0] for _, t in got) < start:
            break
    return out


# ---------------------------------------------------------------- κύρια ροή
def fetch_enex(start, end, out_csv="data/prices.csv", raw_dir="data/raw/enex", sleep: float = 1.0,
               tz: str = TZ) -> pd.DataFrame:
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    raw = Path(raw_dir)
    (raw / "archive").mkdir(parents=True, exist_ok=True)
    (raw / "daily").mkdir(parents=True, exist_ok=True)
    best: dict[tuple, tuple[int, bytes]] = {}  # (day, column) -> (version, xlsx bytes)

    def offer(name: str, getter):
        p = _parse_name(name)
        if not p or not (start <= p[0] <= end):
            return
        day, mkt, ver = p
        key = (day, COL[mkt])
        if key not in best or ver > best[key][0]:
            best[key] = (ver, getter())

    # 1) ετήσια αρχεία
    links = archive_links()
    for year in range(start.year, end.year + 1):
        if year not in links:
            continue
        zpath = raw / "archive" / f"{year}_ResultsSummary.zip"
        if not zpath.exists() or year >= pd.Timestamp.today().year:
            print(f"[ΕΧΕ] ετήσιο αρχείο {year} ...", flush=True)
            zpath.write_bytes(_get(links[year]))
            time.sleep(sleep)
        z = zipfile.ZipFile(zpath)
        for name in z.namelist():
            if name.endswith(".xlsx"):
                offer(name, lambda n=name, z=z: z.read(n))

    # 2) τρέχουσες δημοσιεύσεις (ό,τι δεν καλύπτουν τα ετήσια)
    for mkt in ("DAM", "IDA1", "IDA2", "IDA3"):
        have = [d for (d, c) in best if c == COL[mkt]]
        since = max(start, max(have) - pd.Timedelta(days=3)) if have else start
        if since > end:
            continue
        print(f"[ΕΧΕ] {mkt}: δημοσιεύσεις από {since.date()} ...", flush=True)
        for url, name in current_files(mkt, since, sleep):
            f = raw / "daily" / f"{name}.xlsx"

            def getter(url=url, f=f):
                if not f.exists():
                    f.write_bytes(_get(url))
                    time.sleep(sleep)
                return f.read_bytes()

            offer(name, getter)

    # 3) ανάγνωση
    cols: dict[str, list] = {}
    bad = 0
    for (day, col), (_, data) in sorted(best.items()):
        try:
            s = parse_results_summary(data, day, tz)
        except Exception:
            s = None
        if s is None:
            bad += 1
            continue
        cols.setdefault(col, []).append(s)
    df = pd.DataFrame({c: pd.concat(v) for c, v in cols.items()})
    # ωριαίες τιμές (πριν το 15λεπτο) -> 15λεπτα
    grid = pd.date_range(df.index.min().tz_convert("UTC"), df.index.max().tz_convert("UTC") + pd.Timedelta(minutes=45),
                         freq="15min").tz_convert(tz)
    df = df.reindex(df.index.union(grid)).sort_index().tz_convert("UTC")  # UTC: χωρίς ασάφεια αλλαγής ώρας
    df = df.groupby(df.index.floor("h")).transform(lambda s: s.ffill()).tz_convert(tz).reindex(grid)
    df.index.name = "timestamp"
    print(f"[ΕΧΕ] {len(best)} ημερήσια αρχεία, {bad} χωρίς ανάγνωση | "
          + ", ".join(f"{c}: {df[c].notna().sum() // 96} ημ." for c in df.columns))

    out_csv = Path(out_csv)
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        old.index = pd.to_datetime(old.pop(old.columns[0]), utc=True).dt.tz_convert(tz)
        df = df.combine_first(old).sort_index()
        df.index.name = "timestamp"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv)
    print(f"Αποθηκεύτηκε {out_csv}: {df.index.min()} .. {df.index.max()} | {list(df.columns)}")
    return df
