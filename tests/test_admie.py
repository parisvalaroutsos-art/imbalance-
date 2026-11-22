import numpy as np
import pandas as pd

from imbalance_forecast.admie import _file_date, extract_series


def test_extract_row_layout(tmp_path):
    """Ετικέτα στην πρώτη στήλη, 96 τιμές δεξιά, με γραμμή επικεφαλίδας πριν."""
    p = tmp_path / "20261123_ISP1DayAheadLoadForecast_01.xlsx"
    rows = [["Date", "23/11/2026"] + [None] * 95,
            ["Period"] + list(range(1, 97)),
            ["Load Forecast"] + list(np.arange(96) * 10.0)]
    pd.DataFrame(rows).to_excel(p, header=False, index=False)
    s = extract_series(p, r"(?i)load\s*forecast", pd.Timestamp("2026-11-23"))
    assert len(s) == 96 and s.iloc[5] == 50.0
    assert str(s.index[0]) == "2026-11-23 00:00:00+02:00"


def test_extract_column_layout_dst_hourly_sum(tmp_path):
    """Στήλες ανά ζώνη, ωριαίες τιμές σε ημέρα 25 ωρών -> άθροισμα των στηλών που ταιριάζουν."""
    p = tmp_path / "x.xlsx"
    df = pd.DataFrame({"Hour": range(1, 26), "RES North": [1.0] * 25, "RES South": [2.0] * 25})
    df.to_excel(p, index=False)
    s = extract_series(p, r"^RES", pd.Timestamp("2026-10-25"))
    assert len(s) == 25 and (s == 3.0).all()
    assert s.index[-1].hour == 23


def test_file_date():
    assert _file_date({}, "https://www.admie.gr/x/20261123_ISP1ISPResults_01.xlsx") == pd.Timestamp("2026-11-23")
