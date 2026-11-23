#!/usr/bin/env bash
# Χρήση: scripts/run_report.sh <stage> <ημέρα-στόχος YYYY-MM-DD> "<ετικέτα>" <φάκελος εξόδου>
# Ενημερώνει δεδομένα ΑΔΜΗΕ/ΕΧΕ, ελέγχει αν έχουν δημοσιευθεί όσα χρειάζεται το στάδιο και
# γράφει <out>/<ημέρα>_<stage>.html (+ .csv). Exit 3 = δεν έχουν δημοσιευθεί ακόμα.
set -euo pipefail
STAGE=$1; DAY=$2; LABEL=$3; OUT=${4:-reports}
cd "$(dirname "$0")/.."
mkdir -p "$OUT"
PREV=$(date -d "$DAY -1 day" +%F)
python3 -W ignore -m imbalance_forecast admie --start "$PREV" --end "$DAY" --sleep 0.5 > "$OUT/fetch_admie.log" 2>&1 || true
python3 -W ignore -m imbalance_forecast enex --start "$PREV" --end "$DAY" --sleep 0.5 > "$OUT/fetch_enex.log" 2>&1 || true
python3 -W ignore -m imbalance_forecast report --stage "$STAGE" --data data/prices.csv data/admie.csv \
    --date "$DAY" --label "$LABEL" --html "$OUT/${DAY}_${STAGE}.html" --csv "$OUT/${DAY}_${STAGE}.csv"
