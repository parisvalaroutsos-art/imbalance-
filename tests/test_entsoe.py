"""ENTSO-E A75 (πραγματική παραγωγή ανά τεχνολογία): πραγματικό απόσπασμα 24/11/2026 έως 20:45 UTC."""
from pathlib import Path

import pandas as pd

from imbalance_forecast.entsoe import controllable, parse_a75

FX = Path(__file__).parent / "fixtures"


def test_parse_a75_partial_day_and_compressed_curve():
    df = parse_a75((FX / "entsoe_a75_20261124_partial.xml").read_bytes())
    assert df.index.tz is not None and str(df.index.tz) == "UTC"
    assert df.index.max() == pd.Timestamp("2026-11-24 20:30", tz="UTC")  # ως το τέλος της περιόδου, όχι πέρα
    assert (df.index.to_series().diff().dropna() == pd.Timedelta(minutes=15)).all()
    # καμπύλη A03: ο λιγνίτης δίνεται με ένα μόνο σημείο (0) και ισχύει για όλη την περίοδο
    assert df["lignite"].notna().all() and (df["lignite"] == 0).all()
    assert df.loc["2026-11-24 20:30+00:00", "gas"] == 3838
    assert "hydro_pumped_cons" in df and "storage_cons" in df  # κατανάλωση χωριστά
    c = controllable(df)
    assert c.loc["2026-11-24 20:30+00:00"] == 3838 + 63 + 545  # αέριο + αντλησιοταμίευση + ταμιευτήρες
