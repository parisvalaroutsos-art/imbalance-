# Εκτίμηση τιμής Imbalance / spread (Imbalance − DAM)

Κώδικας Python που εκτιμά ανά 15λεπτο την τιμή απόκλισης (imbalance) ή το spread
`imbalance − DAM` για την ελληνική αγορά, **πριν τη DAM** (η τιμή DAM της ημέρας-στόχου
θεωρείται γνωστή: πραγματική ή δική σας εκτίμηση). Είσοδοι:

* πρόβλεψη ΑΠΕ και φορτίου ΑΔΜΗΕ (αυτόματο κατέβασμα),
* ιστορικό **ευστάθειας**: ο ΑΔΜΗΕ βάζει μονάδες αερίου και περικόπτει ΑΠΕ για να έχει εφεδρεία,
* τιμές DAM (και IDA, αν η πρόβλεψη γίνεται αργότερα),
* ιστορικό imbalance (δημοσιεύεται με καθυστέρηση).

## Χρονισμός: τι είναι γνωστό τη στιγμή της πρόβλεψης

Προεπιλογή: η πρόβλεψη για την ημέρα D βγαίνει στη D-1 πριν τη DAM.

| Πληροφορία | Γνωστή έως | Πώς χρησιμοποιείται |
|---|---|---|
| Πρόβλεψη ΑΠΕ/φορτίου ΑΔΜΗΕ για τη D | D (ολόκληρη) | άμεσα |
| DAM της D | D (ολόκληρη) | στη στήλη `dam` (πραγματική ή εκτίμηση) |
| IDA | – | δεν χρησιμοποιούνται, εκτός αν δοθεί `--ida ida1 ...` |
| Ευστάθεια | D-2 (`stab_lag_days=1`) | ιστορικό + **πρόβλεψη** για τη D (`stab_prob`) |
| Imbalance | Κυριακή πριν την τελευταία Πέμπτη | ιστορικό spread (8–15 ημέρες πίσω) |

Τα imbalance δημοσιεύονται **κάθε Πέμπτη για την προηγούμενη Δευτέρα–Κυριακή**:

| Ημέρα έκδοσης | Τελευταία δημοσίευση | Imbalance γνωστό έως |
|---|---|---|
| Κυρ. 22/11 | Πέμ. 19/11 | Κυρ. 15/11 |
| Πέμ. 19/11 (πριν δημοσιευθούν) | Πέμ. 12/11 | Κυρ. 8/11 |
| Πέμ. 19/11 με `--pub-same-day` | Πέμ. 19/11 | Κυρ. 15/11 |

Οι κανόνες αυτοί εφαρμόζονται και στο backtest. Τα tests `test_no_leakage` και
`test_stability_history_lag` ελέγχουν ότι δεν περνά μελλοντική πληροφορία στις εισόδους του μοντέλου.

## Ευστάθεια

Όταν οι ΑΠΕ είναι πολλές και το καθαρό φορτίο χαμηλό, ο ΑΔΜΗΕ κρατά μονάδες αερίου
σε λειτουργία για εφεδρεία/ευστάθεια και περικόπτει ΑΠΕ. Τότε το σύστημα βγαίνει «μακρύ»
και το imbalance πέφτει κάτω από τη DAM.

Πριν τη DAM δεν ξέρουμε αν θα γίνει ευστάθεια τη D, οπότε ο κώδικας χρησιμοποιεί **δύο μοντέλα**:

1. **Μοντέλο ευστάθειας** (ταξινομητής) → `stab_prob` = P(ευστάθεια στο 15λεπτο). Εισόδους έχει
   το μερίδιο ΑΠΕ (και το ημερήσιο μέγιστό του), το καθαρό φορτίο (και το ημερήσιο ελάχιστό του),
   τις ώρες με μερίδιο ΑΠΕ πάνω από 70%, το ημερολόγιο και το πρόσφατο ιστορικό ευστάθειας
   (ίδια ώρα 7/28 ημερών, χθεσινή ημέρα κ.λπ.). Εκπαιδεύεται με ιστορικό έως D-2, που είναι
   πολύ πιο πρόσφατο από τα imbalance.
2. **Κύριο μοντέλο** (quantile gradient boosting) για το spread. Παίρνει το `stab_prob` ως είσοδο
   μαζί με το ιστορικό μέσο spread *σε περιόδους ευστάθειας* και *εκτός ευστάθειας* (28 ημερών).
   Στην εκπαίδευση το `stab_prob` υπολογίζεται out-of-fold (cross-fitting), ώστε να μοιάζει
   με την ακρίβεια που θα έχει στην πράξη.

Στήλη εισόδου `stability`: MW περικοπής ΑΠΕ ανά 15λεπτο (ή 0/1). Γεγονός = τιμή > `--stab-threshold`.

Το backtest τυπώνει πίνακα **ευστάθεια → imbalance**: ανά κλάση `stab_prob`, πόσο συχνά έγινε
πράγματι ευστάθεια, το μέσο πραγματικό spread και πόσο συχνά ισχύει imbalance < DAM.
Έτσι ελέγχεται η υπόθεση στα πραγματικά δεδομένα.

## Αυτόματο κατέβασμα ΑΔΜΗΕ

```bash
python -m imbalance_forecast admie --start 2024-01-01 --out data/admie.csv
```

Χρησιμοποιεί το API του ΑΔΜΗΕ `getOperationMarketFilewRange` (λίστα αρχείων ανά κατηγορία και
ημερομηνία) και κατεβάζει τα xlsx, με cache στο `data/raw/admie/`. Βρίσκει τη σειρά μέσα σε κάθε
xlsx με regex ετικέτας (σε γραμμή ή σε στήλη· ωριαία, μισάωρα ή 15λεπτα· ημέρες 23/25 ωρών), και
ενημερώνει σταδιακά το `data/admie.csv`.

> ⚠️ Τα ονόματα κατηγοριών και οι ετικέτες στο `DEFAULT_SPEC` (`imbalance_forecast/admie.py`)
> **δεν έχουν επαληθευτεί** σε πραγματικά αρχεία. Πρώτη φορά, ελέγξτε τα έτσι:
>
> ```bash
> python -m imbalance_forecast admie-list --category ISP1DayAheadLoadForecast --date 2026-11-20
> python -m imbalance_forecast admie --start 2026-11-20 --end 2026-11-20 --only load_forecast
> python -m imbalance_forecast admie-inspect --file data/raw/admie/ISP1DayAheadLoadForecast/<αρχείο>.xlsx
> ```
>
> και διορθώστε με ένα JSON (`--spec my_spec.json`), π.χ.:
>
> ```json
> {
>   "load_forecast": {"category": "ISP1DayAheadLoadForecast", "label": "(?i)load forecast"},
>   "res_forecast":  {"category": "ISP1DayAheadRESForecast",  "label": "(?i)res"},
>   "stability":     {"category": "ISP1ISPResults", "label": "(?i)res", "derive": "res_forecast_minus"},
>   "imbalance":     {"category": "<κατηγορία τιμών απόκλισης>", "label": "(?i)imbalance price"}
> }
> ```
>
> Το `"derive": "res_forecast_minus"` σημαίνει ότι το αρχείο δίνει τις προγραμματισμένες ΑΠΕ του ISP,
> και η περικοπή υπολογίζεται ως πρόβλεψη ΑΠΕ − πρόγραμμα ISP.
> Για πρόβλεψη πριν τη DAM, οι προβλέψεις φορτίου/ΑΠΕ πρέπει να είναι εκδόσεις που **δημοσιεύονται
> πριν τη DAM**. Αν το ISP1 βγαίνει μετά τη DAM, διαλέξτε την αντίστοιχη πρωινή κατηγορία.

## Μορφή δεδομένων

Ένα ή περισσότερα CSV με κοινή στήλη `timestamp` (αρχή διαστήματος· χωρίς ζώνη = ώρα Ελλάδας·
με offset `+02:00/+03:00` = σωστά και στις αλλαγές ώρας).

```
timestamp,dam,ida1,ida2,ida3,imbalance
timestamp,res_forecast,load_forecast,stability
```

* Οι ωριαίες σειρές επαναλαμβάνονται αυτόματα στα 15λεπτα.
* Ό,τι δεν έχει δημοσιευθεί (`imbalance`, `stability`) μένει κενό.
* Για την ημέρα-στόχο βάλτε στη στήλη `dam` την τιμή που γνωρίζετε/εκτιμάτε.
* Κάθε επιπλέον αριθμητική στήλη γίνεται feature `x_<όνομα>`. Πρέπει να είναι γνωστή στην έκδοση,
  δηλαδή πρόβλεψη και όχι πραγματική τιμή.

## Χρήση

```bash
pip install -r requirements.txt

python -m imbalance_forecast demo                     # συνθετικά (ψεύτικα) δεδομένα

python -m imbalance_forecast admie --start 2024-01-01 --out data/admie.csv

python -m imbalance_forecast backtest --data data/prices.csv data/admie.csv \
    --start 2025-10-01 --out backtest.csv

# πρόβλεψη για αύριο, πριν τη DAM (προεπιλογή)
python -m imbalance_forecast predict --data data/prices.csv data/admie.csv \
    --date 2026-11-23 --out forecast_2026-11-23.csv

# αργότερα, μετά το IDA1
python -m imbalance_forecast predict --data ... --date 2026-11-23 --ida ida1
```

Επιλογές: `--target spread|imbalance`, `--lead N`, `--ida ...`, `--pub-same-day`, `--window`,
`--stab-threshold`, `--retrain-every N` (backtest).

Έξοδος ανά 15λεπτο: `stab_prob`, `spread_q10/q50/q90`, `imb_q10/q50/q90` (= DAM + spread),
`p_imb_gt_dam`, `p_imb_lt_dam`.

## Δομή

```
imbalance_forecast/
  config.py        ρυθμίσεις (χρονισμός, IDA, ευστάθεια, μοντέλο)
  availability.py  κανόνας «Πέμπτη -> προηγούμενη Δευ–Κυρ»
  admie.py         κατέβασμα/ανάγνωση αρχείων ΑΔΜΗΕ
  data.py          φόρτωση CSV, ζώνες ώρας, πλέγμα 15'
  features.py      features χωρίς διαρροή (as-of cutoffs)
  model.py         μοντέλο ευστάθειας + quantile GBM + ταξινομητής κατεύθυνσης
  pipeline.py      backtest, metrics, stability_report, forecast
  synthetic.py     συνθετικά δεδομένα για δοκιμές
tests/             pytest
```
