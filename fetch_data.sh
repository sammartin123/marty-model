#!/usr/bin/env bash
# Download the free nflverse data the model needs into ./data (about 150 MB). Safe to rerun: it always refreshes.
set -euo pipefail
cd "$(dirname "$0")"; mkdir -p data; cd data
B=https://github.com/nflverse/nflverse-data/releases/download
Y=${1:-$(date +%Y)}; [ "$(date +%m)" -lt 3 ] && [ -z "${1:-}" ] && Y=$((Y-1))   # Jan/Feb belong to the prior season
for y in $(seq 2014 "$Y"); do
  if [ "$y" -lt "$Y" ] && [ -s "pbp_$y.parquet" ]; then continue; fi
  curl -fsSL --retry 3 --max-time 180 -o "pbp_$y.parquet" "$B/pbp/play_by_play_$y.parquet" &
  [ $(( y % 4 )) -eq 0 ] && wait
done; wait
curl -fsSL --retry 3 -o players.csv "$B/players/players.csv"
for y in $(seq $((Y-3)) "$Y"); do curl -fsSL --retry 3 -o "roster_weekly_$y.csv" "$B/weekly_rosters/roster_weekly_$y.csv"; done
curl -fsSL --retry 3 -o "depth_charts_$Y.csv" "$B/depth_charts/depth_charts_$Y.csv"
curl -fsSL --retry 3 -o "injuries_$Y.csv" "$B/injuries/injuries_$Y.csv" || echo "no injuries file for $Y yet"
curl -fsSL --retry 3 -o games.csv https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv
echo "data ready: $(ls | wc -l) files, $(du -sh . | cut -f1)"
