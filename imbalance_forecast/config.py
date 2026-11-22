"""Ρυθμίσεις μοντέλου / Model configuration."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # --- Χρόνος / time ---
    tz: str = "Europe/Athens"
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
    # Το ιστορικό ευστάθειας (από ISP/ΑΔΜΗΕ) είναι γνωστό έως issue_date - stab_lag_days.
    stab_lag_days: int = 1
    stab_cv_folds: int = 4  # cross-fitting για out-of-fold stab_prob στην εκπαίδευση
