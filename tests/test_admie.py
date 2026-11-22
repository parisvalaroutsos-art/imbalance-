"""Tests πάνω σε πραγματικά αρχεία ΑΔΜΗΕ (tests/fixtures, δημόσια δεδομένα admie.gr)."""
from pathlib import Path

import pandas as pd
import pytest

from imbalance_forecast.admie import parse_forecast, parse_imbabe, parse_isp_results, select_versions

FX = Path(__file__).parent / "fixtures"
DAY = pd.Timestamp("2026-11-20")


def test_parse_load_forecast():
    s = parse_forecast(FX / "20261120_ISP1DayAheadLoadForecast_01.xlsx", DAY)
    assert len(s) == 96
    assert str(s.index[0]) == "2026-11-20 00:00:00+02:00"
    assert s.iloc[0] == 4200 and s.iloc[1] == 4140


def test_parse_res_forecast():
    s = parse_forecast(FX / "20261120_ISP1DayAheadRESForecast_01.xlsx", DAY)
    assert len(s) == 96 and s.iloc[0] == 3818


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_parse_isp_results():
    df = parse_isp_results(FX / "20261120_ISP1ISPResults_01.xlsx", DAY)
    assert len(df) == 96  # χωρίς τη στήλη TOTAL
    t = pd.Timestamp("2026-11-20 10:00", tz="Europe/Athens")
    assert df.loc[t, "isp_energy_surplus"] == pytest.approx(-2062.362)
    assert df.loc[t, "stability"] == pytest.approx(2062.362)
    assert df.loc[t, "isp_renewables"] == 9228 and df.loc[t, "isp_thermal"] == pytest.approx(1187.9)
    assert df.iloc[0]["stability"] == 0


def test_parse_imbabe():
    df = parse_imbabe(FX / "20261109_IMBABE_01.xlsx")
    assert len(df) == 7 * 96
    assert str(df.index[0]) == "2026-11-09 00:00:00+02:00"
    assert df["imbalance"].iloc[0] == 550.0
    assert df["curt_flag"].sum() == 35
    assert (df.loc[df["curt_flag"] == 1, "imbalance"] == -100).all()


def test_select_versions_pre_dam():
    base = "https://www.admie.gr/x/20261120_ISP1DayAheadLoadForecast_0{}.xlsx"
    items = [
        {"file_path": base.format(2), "file_fromdate": "20.11.2026", "file_published": "19.11.2026 13:25"},
        {"file_path": base.format(1), "file_fromdate": "20.11.2026", "file_published": "19.11.2026 09:25"},
    ]
    assert select_versions(items, "pre_dam")[0]["version"] == 1
    assert select_versions(items, "latest")[0]["version"] == 2
