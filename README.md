# MartyMakesMoneyModel

NFL game simulator for Sam Martin. One game at a time: 100,000 simulated games, final score, anytime TD
and yardage props compared with sportsbook prices, shown on one page.

Live page (private artifact): https://claude.ai/artifact/G1Lc9453gKaZtNvn6oUVg7
Artifact slug: `79852994-f850-4c0e-876b-fdb91bc130c4`

## Run a game

```bash
pip install -r requirements.txt            # add --break-system-packages if pip asks
./fetch_data.sh                            # free nflverse data into ./data (about 150 MB)
python3 src/make_config.py PIT CIN         # AWAY HOME -> games/PIT_CIN.json (skeleton)
# fill in odds, weather and injuries in the config (see below), then:
./run_game.sh games/PIT_CIN.json           # refresh data, simulate, write site/index.html
```

Publish `site/index.html` to the artifact above (same slug, so the link never changes).

## Run a whole slate (every Sunday game)

```bash
./fetch_data.sh && python3 src/prep.py
python3 src/run_slate.py slates/2026-W10.json     # every game in the file -> out/slate.json
python3 src/build_page.py out/slate.json site/index.html
```

A slate file (`slates/<season>-W<week>.json`) lists games as `"AWAY_HOME"` keys. Teams, kickoff, venue, roof,
injuries (Out and Doubtful removed, Questionable flagged) and starting quarterbacks (current depth chart QB1,
skipping anyone ruled out) are built automatically. Each game may override `market`, `td_odds`, `yard_lines`,
`weather`, `wx_model`, `out_names`, `qb_names`, `notes` and `sources`.

The page only makes a call when all of these hold: a real sportsbook price was found, the final injury report
is in (6 PM ET two days before the game), the player is not Questionable, and neither team's quarterback is
flagged low confidence (new starter with under 60 attempts for the team, or a starter who has not practised).

## What goes in a game config

`make_config.py` fills in teams, week, kickoff, venue, starting QBs (depth chart QB1) and the schedule's
listed spread, total and moneyline. Before running, complete these by hand or by web search:

- `market`: spread, total and moneyline from a named book. Sam prefers bet365 or Stake, then ESPN (DraftKings).
- `td_odds`: best anytime TD price per player, as `{"Player Name": {"price": 190, "book": "...", "all": [["Book", 175]]}}`.
  Names must match nflverse display names. Never invent a price: leave a player out if none is found.
- `yard_lines`: posted yardage lines, as `{"Player Name": [{"stat": "rec|rush|pass|recs", "line": 25.5, "over": -114, "book": "..."}]}`.
- `out`: gsis ids of skill players ruled out. Check Friday's final injury report. Change `qb` if a starter is out.
- `weather` and `wx_model`: use the **hourly** forecast for the kickoff window of outdoor games, not the daily
  summary. Measured effects per team from 2014 to 2025 (same team, same season):
  cold under 40F: pass yards -4%, TDs -3%; wind 15+ mph: pass yards -9%, TDs -8%; rain: pass yards -10%, TDs -14%.
  Weight the rain effect by its probability. If rain chance is 30% or more, set `precip` accordingly: the page
  then warns off receiving-yard overs (especially secondary targets) toward rushing volume, receptions or unders.
  Domes: leave `wx_model` neutral.
- `sources`: a label and URL for every place a price or forecast came from.

## How the model works

- `src/prep.py` turns play-by-play into drive, player-game and team-game tables.
- `src/model.py`
  - Team strength: opponent-adjusted TD and field-goal rates per drive (ridge logistic, recency weighted).
  - Game sim: possession by possession with a clock, score effects (trailing teams press) and overtime.
  - Who scores: usage shares (carries, goal-line carries, targets, air yards, red-zone and end-zone targets)
    weighted by how well each predicted TDs from 2015 to 2025.
  - Yardage: pass and rush volume tied to the simulated game script, then targets, catches and yards per player.
  - Position vs defence and head-to-head history are included at low weight. In testing neither improved accuracy.
- `src/run_game.py` runs 100,000 sims for one config and writes the JSON the page reads.
- `src/build_page.py` drops that JSON into `src/template.html`.
- `src/backtest.py` replays past games blind. `work/bt_report.json` holds the latest result, shown on the page.

## Results tracker

`src/ledger.py` snapshots every call before kickoff (`ledger/pending/`), then grades it once the game is in the
data (`ledger/graded.json`): anytime TD by whether the player scored, yardage against the line, leans against the
final score. One unit per call at the logged price. Only calls that were live before kickoff count toward the
record shown on the page. A player with no touch in the game is graded void.

## Backtest (2024, 2025 and 2026 through Week 9: 681 games)

- Winners picked: 64.5% (book favourite: 68.3%). Against the closing spread: 320-350. Totals: 339-339.
- Anytime TD probabilities are well calibrated across 16,176 player-games.
- No archive of past prop odds was available, so prop edges are untested against real prices.

Game-line calls are capped at Lean on the page for that reason.

## Call thresholds on the page

- Anytime TD: Play = sim beats the best price by 3+ points and sim is at least 10%; Lean = 1.5 to 3; Fade = 5+ under.
- Yardage calculator: Play = 4+ points over break-even; Lean = 2 to 4.
- Spread, total, moneyline: Lean at most.

Refresh the backtest a few times a season: `cd src && python3 backtest.py 2024,2025,2026 2500`.
