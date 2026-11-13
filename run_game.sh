#!/usr/bin/env bash
# Full run for one game: refresh data, rebuild tables, simulate 100,000 games, build the page.
# usage: ./run_game.sh games/PIT_CIN.json
set -euo pipefail
cd "$(dirname "$0")"
CFG=${1:?usage: ./run_game.sh games/AWAY_HOME.json}
./fetch_data.sh
python3 src/prep.py
python3 src/run_game.py "$CFG"
OUT=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['out_json'])" "$CFG")
python3 src/build_page.py "$OUT" site/index.html
