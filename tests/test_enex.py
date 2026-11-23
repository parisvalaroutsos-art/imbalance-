"""Tests πάνω σε πραγματικά αρχεία ΕΧΕ (tests/fixtures, δημόσια δεδομένα enexgroup.gr)."""
from pathlib import Path

import pandas as pd

from imbalance_forecast.enex import _parse_name, parse_results_summary

FX = Path(__file__).parent / "fixtures"


def _load(name):
    day, mkt, ver = _parse_name(name)
    return parse_results_summary((FX / name).read_bytes(), day)


def test_dam_hourly_2024():
    s = _load("20241120_EL-DAM_ResultsSummary_EN_v01.xlsx")
    assert len(s) == 24 and s.iloc[0] == 107.78 and s.iloc[-1] == 60.0
    assert str(s.index[0]) == "2024-11-20 00:00:00+01:00"


def test_dam_15min_dst_day():
    s = _load("20251026_EL-DAM_ResultsSummary_EN_v01.xlsx")  # 25ωρη ημέρα -> 100 τέταρτα
    assert len(s) == 100 and s.notna().all()
    assert s.iloc[0] == 93.75 and s.iloc[-1] == 3.47
    assert str(s.index[-1]) == "2025-10-26 23:45:00+01:00"


def test_ida3_half_day():
    s = _load("20251120_EL-IDA3_ResultsSummary_EN_v01.xlsx")
    assert len(s) == 96
    assert s[s.index.hour < 12].isna().all() and s[s.index.hour >= 12].notna().all()


def test_parse_name():
    assert _parse_name("20240115_EL-CRIDA1_ResultsSummary_EN_v02.xlsx") == (pd.Timestamp("2024-01-15"), "CRIDA1", 2)
