#!/usr/bin/env bash
# Χρήση: scripts/run_nowcast.sh [ημέρα YYYY-MM-DD, προεπιλογή σήμερα CET] [φάκελος εξόδου]
# Ενημερώνει ENTSO-E / ISP / IDA / IMBABE και γράφει <out>/<ημέρα>_nowcast.html (+ .csv):
# «πώς πήγε το σύστημα» για τα 15λεπτα που πέρασαν.
set -euo pipefail
DAY=${1:-$(TZ=Europe/Berlin date +%F)}; OUT=${2:-reports}
cd "$(dirname "$0")/.."
mkdir -p "$OUT"
OMP_NUM_THREADS=4 python3 -W ignore -m imbalance_forecast nowcast --date "$DAY" --fetch \
    --html "$OUT/${DAY}_nowcast.html" --csv "$OUT/${DAY}_nowcast.csv" > "$OUT/nowcast.log" 2>&1
tail -n 30 "$OUT/nowcast.log" | grep -v "^  "
