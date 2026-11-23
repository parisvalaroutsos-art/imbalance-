"""Ρυθμίσεις μοντέλου / Model configuration."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # --- Χρόνος / time ---
    # Ζώνη ώρας ΑΓΟΡΑΣ: η ημέρα DAM/ISP/imbalance ορίζεται σε CET/CEST (00:00-24:00 CET = 01:00-01:00
    # ώρα Ελλάδας). Επαληθεύτηκε: κορυφή ΦΒ στα αρχεία ΑΔΜΗΕ/ΕΧΕ στις ~11:25 της θέσης = ηλιακό μεσημέρι
    # σε CET, και οι ημέρες αλλαγής ώρας στο IMBABE/ISP1 έχουν επανάληψη/κενό στις 02:00 (κανόνας CET).
    tz: str = "Europe/Berlin"
    mtu: str = "15min"  # χρονική ανάλυση (ISP/imbalance MTU)

    # --- Χρονισμός πρόβλεψης / forecast timing ---
    # Η πρόβλεψη για την ημέρα D εκδίδεται την ημέρα D - issue_lead_days
    # (π.χ. 1 = D-1, μετά το DAM και το IDA1).
    issue_lead_days: int = 1
    # Ποιες IDA δημοπρασίες είναι γνωστές τη στιγμή της πρόβλεψης.
    # Προεπιλογή: πρόβλεψη ΠΡΙΝ τη DAM -> καμία IDA. Η τιμή DAM της D θεωρείται γνωστή
    # (δική σας εκτίμηση ή πραγματική) και δίνεται στη στήλη `dam`.
    # Ελλάδα (CRIDA): IDA1 D-1 απόγευμα, IDA2 D-1 βράδυ, IDA3 ημέρα D (μόνο μέρος της ημέρας).
    ida_available: tuple[str, ...] = ()
    # Στάδιο πρόβλεψης (βλ. STAGES): pre_dam | isp2 | isp3. Ορίζει τι είναι γνωστό στο issue.
    stage: str = "pre_dam"
    # Στήλες γνωστές για την ίδια την ημέρα-στόχο στο issue (πέρα από πρόβλεψη ΑΠΕ/φορτίου, DAM, IDA)
    same_day_cols: tuple[str, ...] = ()
    # Ώρες (αγοράς) που προβλέπονται, [από, έως). None = όλη η ημέρα.
    target_hours: tuple[int, int] | None = None

    # --- Καθυστέρηση δημοσίευσης imbalance ---
    # Τα imbalance δημοσιεύονται Πέμπτη (weekday=3) για την προηγούμενη Δευτέρα-Κυριακή.
    pub_weekday: int = 3
    # True αν τα δεδομένα της Πέμπτης είναι ήδη διαθέσιμα όταν βγάζουμε πρόβλεψη την ίδια Πέμπτη.
    pub_available_same_day: bool = False

    # --- Στόχος / target ---
    target: str = "spread"  # "spread" (= imbalance - DAM) ή "imbalance"
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    winsor: tuple[float, float] = (0.005, 0.995)  # περικοπή ακραίων τιμών μόνο στην εκπαίδευση

    # --- Εκπαίδευση / training ---
    train_window_days: int = 365
    min_train_days: int = 56
    # Στο backtest: επανεκπαίδευση κάθε N εβδομάδες (τα lag features ενημερώνονται πάντα εβδομαδιαία)
    retrain_every_weeks: int = 1
    # Βάρη χρονικής εγγύτητας στην εκπαίδευση (half-life σε ημέρες, None = ίσα βάρη). Η αγορά αλλάζει
    # κανόνες (π.χ. κατώτατο imbalance: χωρίς αρνητικά έως 6/2025, -50 από 7/2025, -100 από 7/2026).
    recency_halflife_days: float | None = 60.0
    gbm_params: dict = field(
        default_factory=lambda: dict(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=80,
            l2_regularization=1.0,
            random_state=0,
        )
    )

    # Στήλες (ονόματα στα CSV)
    col_dam: str = "dam"
    col_imb: str = "imbalance"
    col_res: str = "res_forecast"
    col_load: str = "load_forecast"
    ida_cols: tuple[str, ...] = ("ida1", "ida2", "ida3")

    # --- Ευστάθεια (εντολές ΑΔΜΗΕ σε μονάδες αερίου + περικοπή ΑΠΕ για εφεδρεία/ευστάθεια) ---
    # Στήλη με το ιστορικό: MW περικοπής ΑΠΕ (ή 0/1) ανά 15λεπτο. Για την ημέρα-στόχο ΔΕΝ είναι
    # γνωστή πριν τη DAM, οπότε προβλέπεται από ξεχωριστό μοντέλο (stab_prob).
    col_stab: str = "stability"
    stab_threshold: float = 0.0  # γεγονός ευστάθειας όταν stability > threshold
    # Το ιστορικό ευστάθειας είναι γνωστό έως issue_date - stab_lag_days.
    # ISP1ISPResults της ημέρας X δημοσιεύεται X-1 ~18:45 -> στο issue (D-1 πρωί) γνωστό έως και D-1 -> 0.
    stab_lag_days: int = 0
    # Σημαία περικοπής ΑΔΜΗΕ (IMBABE Curt_Flag): έρχεται μαζί με τα imbalance (εβδομαδιαία καθυστέρηση).
    # Χρησιμοποιείται για να μάθει το μοντέλο πόσο συχνά το πλεόνασμα ISP γίνεται περικοπή (-> imbalance -100).
    col_curt: str = "curt_flag"
    # Σημειακή πρόβλεψη imbalance (imb_point). Στάδια isp2/isp3: = imb_q50. Πριν τη DAM: DAM, εκτός αν p_curt >= κατώφλι -> κατώτατο όριο
    # (ελάχιστο imbalance τελευταίων 28 γνωστών ημερών). Backtest 11/2025-11/2026: MAE 84,3 έναντι 87,0 της DAM.
    point_curt_threshold: float = 0.4
    # "floor": τελευταία γνωστή τιμή (IDA ή DAM) ή κατώτατο όριο όταν p_curt >= κατώφλι | "q50": διάμεσος
    point_mode: str = "floor"
    # Ανοδικό σπάικ: imbalance > DAM + spike_up (€/MWh) -> πιθανότητα p_spike_up
    spike_up: float = 100.0
    stab_cv_folds: int = 4  # cross-fitting για out-of-fold stab_prob στην εκπαίδευση


# Στάδια πρόβλεψης. Ώρες δημοσίευσης (ώρα Ελλάδας, περίπου):
#   pre_dam: D-1 πρωί, πριν τη DAM. Πρόβλεψη ΑΔΜΗΕ ISP1 (ΑΠΕ/φορτίο). Ευστάθεια της D προβλέπεται.
#   isp1:    D-1 ~19:00. + IDA1 (D-1 απόγ.) και αποτελέσματα ISP1 της D (D-1 ~18:45): πλεόνασμα γνωστό.
#   isp2:    D ~00:30. + IDA1 (D-1 απόγ.), IDA2 (D-1 βράδυ), ISP1 αποτελέσματα της D (D-1 ~18:45),
#            ISP2 αποτελέσματα της D (D ~00:25): πλεόνασμα/ΑΠΕ/φορτίο όλης της ημέρας.
#   isp3:    D ~12:30. + IDA3 (D ~10:00) και ISP3 (D ~12:25), μόνο για 12:00-24:00 CET.
_UA = ("thermal", "hydro", "n_out", "n_forced")
UA1 = tuple(f"ua1_{f}" for f in _UA)  # διαθεσιμότητα μονάδων ISP1: D-1 ~09:40 (πριν τη DAM)
UA2 = tuple(f"ua2_{f}" for f in _UA)  # ISP2: D-1 ~21:10
UA3 = tuple(f"ua3_{f}" for f in _UA)  # ISP3: D ~09:10
STAGE_COLS_ISP2 = ("stability", "stab_isp2", "res_isp2", "load_isp2")
STAGE_COLS_ISP3 = ("stab_isp3", "res_isp3", "load_isp3")
STAGES: dict[str, dict] = {
    "pre_dam": dict(issue_lead_days=1, ida_available=(), same_day_cols=UA1, target_hours=None),
    "isp1": dict(issue_lead_days=1, ida_available=("ida1",), same_day_cols=("stability",) + UA1,
                 target_hours=None, point_mode="floor"),
    "isp2": dict(issue_lead_days=0, ida_available=("ida1", "ida2"), same_day_cols=STAGE_COLS_ISP2 + UA1 + UA2,
                 target_hours=None, point_mode="q50"),
    "isp3": dict(issue_lead_days=0, ida_available=("ida1", "ida2", "ida3"),
                 same_day_cols=STAGE_COLS_ISP2 + STAGE_COLS_ISP3 + UA1 + UA2 + UA3, target_hours=(12, 24),
                 point_mode="q50"),
}
ALL_STAGE_COLS = STAGE_COLS_ISP2 + STAGE_COLS_ISP3 + UA1 + UA2 + UA3


def stage_config(stage: str = "pre_dam", **overrides) -> Config:
    """Config με τις προεπιλογές του σταδίου (τα overrides υπερισχύουν)."""
    if stage not in STAGES:
        raise ValueError(f"Άγνωστο στάδιο {stage!r}: {list(STAGES)}")
    base = dict(STAGES[stage])
    if not overrides.pop("use_unit_avail", True):  # σύγκριση χωρίς διαθεσιμότητα μονάδων
        base["same_day_cols"] = tuple(c for c in base["same_day_cols"] if not c.startswith("ua"))
    return Config(**{**base, "stage": stage, **overrides})
