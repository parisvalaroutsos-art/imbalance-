# Εκτίμηση τιμής Imbalance / spread (Imbalance − DAM)

Κώδικας Python που εκτιμά την τιμή απόκλισης (imbalance) ή το spread `imbalance − DAM`
ανά 15λεπτο, για την ελληνική αγορά, με είσοδο:

* πρόβλεψη ΑΠΕ και φορτίου ΑΔΜΗΕ,
* προαιρετικούς δείκτες (π.χ. ευστάθεια/επάρκεια, εφεδρείες, διασυνδέσεις),
* τιμές DAM και IDA (IDA1/IDA2/IDA3),
* ιστορικό τιμών imbalance.

## Η καθυστέρηση των τιμών imbalance

Τα imbalance δημοσιεύονται **κάθε Πέμπτη για την προηγούμενη Δευτέρα–Κυριακή**. Ο κώδικας
το εφαρμόζει αυστηρά, τόσο στην παραγωγική πρόβλεψη όσο και στο backtest:

| Ημέρα έκδοσης πρόβλεψης | Τελευταία δημοσίευση | Imbalance γνωστό έως |
|---|---|---|
| Κυρ. 22/11 | Πέμ. 19/11 | Κυρ. 15/11 |
| Πέμ. 19/11 (πριν δημοσιευθούν) | Πέμ. 12/11 | Κυρ. 8/11 |
| Πέμ. 19/11 με `--pub-same-day` | Πέμ. 19/11 | Κυρ. 15/11 |

Άρα για την ημέρα-στόχο D το «κενό» προς τα τελευταία γνωστά imbalance είναι 8–15 ημέρες.
Όλα τα ιστορικά features (`spr_h_*`, `spr_dh_m4`, `imb_h_m7`) υπολογίζονται μόνο με
δεδομένα έως αυτό το cutoff, και το `gap_days` περνάει στο μοντέλο ως feature.
Το test `test_no_leakage` ελέγχει ότι αλλαγές στα imbalance μετά το cutoff δεν επηρεάζουν
τα features.

## Μοντέλο

* **Στόχος:** spread = imbalance − DAM (προεπιλογή) ή κατευθείαν η τιμή imbalance (`--target imbalance`).
  Το spread είναι πιο σταθερό στατιστικά, επειδή το DAM «απορροφά» το επίπεδο τιμών (φυσικό αέριο, CO₂).
* **Quantile gradient boosting** (scikit-learn `HistGradientBoostingRegressor`) για P10 / P50 / P90,
  καθώς και ταξινομητής για την **πιθανότητα imbalance > DAM** (`p_imb_gt_dam`), που είναι χρήσιμη
  για αποφάσεις θέσης.
* Οι ακραίες τιμές (spikes) περικόπτονται (winsorize) μόνο στην εκπαίδευση.
* **Features**
  * ημερολόγιο: ώρα, 15λεπτο, ημέρα εβδομάδας, εποχικότητα, ελληνικές αργίες (με ορθόδοξο Πάσχα),
  * ΑΔΜΗΕ: ΑΠΕ, φορτίο, καθαρό φορτίο (φορτίο − ΑΠΕ), μερίδιο ΑΠΕ, ράμπες, απόκλιση από τον μέσο όρο της ημέρας,
  * τιμές: DAM (επίπεδο, ημερήσιο min/max/μέσος, ράμπα), IDA, **IDA − DAM**. Η μετακίνηση της τιμής μετά το DAM
    δείχνει προς τα πού «γέρνει» το σύστημα, γι' αυτό είναι συνήθως ισχυρό σήμα για την κατεύθυνση,
  * ιστορικό spread ανά ώρα (τελευταία γνωστή ημέρα, μέσος/διάμεσος/τυπική απόκλιση 7 και 28 ημερών,
    ποσοστό θετικών), ίδια ημέρα εβδομάδας (4 εβδομάδες),
  * κάθε επιπλέον αριθμητική στήλη των CSV, ως `x_<όνομα>` (π.χ. `x_stability`).
* **Walk-forward backtest:** κάθε εβδομάδα το μοντέλο ξαναεκπαιδεύεται μόνο με ό,τι ήταν διαθέσιμο,
  και συγκρίνεται με baselines:
  `bl_dam` (imbalance = DAM), `bl_hist28` (DAM + μέσο spread ώρας 28 ημερών), `bl_ida` (τελευταία IDA).

## Μορφή δεδομένων

Ένα ή περισσότερα CSV με κοινή στήλη `timestamp`, που είναι η αρχή του διαστήματος. Αν το timestamp
δεν έχει ζώνη ώρας, θεωρείται τοπική ώρα Ελλάδας· αν έχει offset (`+02:00`/`+03:00`), διαβάζεται σωστά
και τις ημέρες αλλαγής ώρας.

```
timestamp,dam,ida1,ida2,ida3,imbalance
2026-11-01 00:00:00+02:00,110.2,108.5,109.1,,131.7
...
timestamp,res_forecast,load_forecast,stability
2026-11-01 00:00:00+02:00,2310,4980,0.4
```

* Οι ωριαίες σειρές (π.χ. DAM πριν τα 15λεπτα) επαναλαμβάνονται αυτόματα στα 4 τέταρτα.
* Τα `imbalance` που δεν έχουν δημοσιευθεί μένουν κενά.
* Οι στήλες IDA που δεν είναι γνωστές τη στιγμή της πρόβλεψης δεν χρησιμοποιούνται: ρυθμίζονται με `--ida`.
* **Προσοχή:** επιπλέον στήλες πρέπει να είναι *προβλέψεις* γνωστές στην έκδοση, όχι πραγματικές τιμές
  (αλλιώς το backtest θα είναι υπεραισιόδοξο).

## Χρήση

```bash
pip install -r requirements.txt

# Demo σε συνθετικά (ψεύτικα) δεδομένα
python -m imbalance_forecast demo

# Backtest στα δικά σας δεδομένα
python -m imbalance_forecast backtest --data data/prices.csv data/admie.csv \
    --start 2025-10-01 --out backtest.csv

# Πρόβλεψη για αύριο (έκδοση D-1 μετά το IDA1)
python -m imbalance_forecast predict --data data/prices.csv data/admie.csv \
    --date 2026-11-23 --out forecast_2026-11-23.csv

# Πρόβλεψη μετά το IDA2 ή για χρήση εντός ημέρας (lead 0, με IDA3)
python -m imbalance_forecast predict --data ... --date 2026-11-23 --ida ida1 ida2
python -m imbalance_forecast predict --data ... --date 2026-11-23 --lead 0 --ida ida1 ida2 ida3
```

Κύριες επιλογές: `--target spread|imbalance`, `--lead N` (ημέρες πριν τη D), `--ida ...`,
`--pub-same-day`, `--window` (ημέρες εκπαίδευσης), `--retrain-every N` (backtest).

Έξοδος ανά 15λεπτο: `spread_q10/q50/q90`, `imb_q10/q50/q90` (= DAM + spread) και `p_imb_gt_dam`.

Από Python:

```python
from imbalance_forecast import Config, load_data, backtest, metrics, forecast
cfg = Config(ida_available=("ida1",))
df = load_data(["data/prices.csv", "data/admie.csv"], cfg)
res = backtest(df, cfg, start="2025-10-01")
print(metrics(res))
fc = forecast(df, cfg, "2026-11-23")
```

## Δομή

```
imbalance_forecast/
  config.py        ρυθμίσεις (χρονισμός, IDA, κανόνας δημοσίευσης, μοντέλο)
  availability.py  κανόνας «Πέμπτη -> προηγούμενη Δευ–Κυρ»
  data.py          φόρτωση CSV, ζώνες ώρας, πλέγμα 15'
  features.py      features χωρίς διαρροή (as-of cutoff)
  model.py         quantile GBM + ταξινομητής κατεύθυνσης
  pipeline.py      backtest, metrics, forecast
  synthetic.py     συνθετικά δεδομένα για δοκιμές
tests/             pytest
```

## Ιδέες για βελτίωση

* Χωριστή πρόβλεψη αιολικών/φωτοβολταϊκών και διαδοχικές εκδόσεις (vintages) της πρόβλεψης ΑΔΜΗΕ,
  με τη διαφορά μεταξύ εκδόσεων ως ένδειξη αβεβαιότητας.
* Ανάγκες εφεδρειών, ζήτηση και τιμές aFRR/mFRR, διαθεσιμότητα μονάδων, ροές στις διασυνδέσεις.
* Τιμές γειτονικών αγορών (IT-GR, BG) και IDA spreads.
