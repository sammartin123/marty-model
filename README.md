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
  - Team strength: opponent-adjusted TD and field-goal rates per drive (ridge logistic, recency weighted),
    with a separate rating for each regular quarterback so a change of starter moves the numbers.
  - Game sim: possession by possession with a clock, score effects (trailing teams press) and overtime.
  - Who scores: usage shares (carries, goal-line carries, targets, air yards, red-zone and end-zone targets)
    weighted by how well each predicted TDs from 2015 to 2025, adjusted for snap-share trend, with a
    fitted boost for quarterback rushing TDs.
  - Player props use a second sim whose team scoring is anchored to the sportsbook's implied team totals
    (from the spread and total in the config). Winner, spread and total stay the model's own view.
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

- Winners picked: 66.5% (book favourite: 68.3%). Against the closing spread: 332-338. Totals: 339-339.
- Anytime TD probabilities are well calibrated across 16,898 player-games.
- Known soft spots: the two goal-line rushing quarterbacks (Josh Allen, Jalen Hurts) still score more often than
  the model says, and teams that have just changed quarterback score touchdowns a little less often than it says.
  The page never fades a quarterback and makes no calls on games with an unproven starter.
- No archive of past prop odds was available, so prop edges are untested against real prices.

Game-line calls are capped at Lean on the page for that reason.

## Research log

- **Four upgrades (Nov 2026): tested, adopted.** Each was compared with the previous model on the same
  player-games across the 681 backtest games (`research/compare_bt.py`, bootstrap by game), and kept only
  because it measurably helped.
  1. *Props anchored to the game line* (`anchor`). Team scoring for player props is shifted to match the
     sportsbook's implied team totals; the model then only splits the yards and touchdowns. Winner, spread and
     total numbers stay the model's own view. TD log loss improved, most of all for teams with a new quarterback.
  2. *Quarterback effects* (`use_qb`, `qb_kappa`). Every regular quarterback has his own rating in the team
     strength fit, new or rarely used starters share one, and passing efficiency moves when the starter differs
     from whoever has been throwing lately. Win-probability Brier 0.221 to 0.215 (book 0.208); in games with a
     quarterback change 0.231 to 0.212. Passing-yard error in those games fell from 67 to 60 yards.
  3. *Snap counts* (`snap_gamma`). Every player with an offensive snap gets a row, and a rising or falling snap
     share nudges his usage. TD log loss improved; yardage unchanged.
  4. *Quarterback rushing TDs* (`qb_td_fix`). Quarterbacks score about 1.5 times as many rushing touchdowns as
     their carry shares imply (sneaks, scrambles). One fitted multiplier fixed most of a clear under-prediction
     (quarterbacks: predicted 10.8%, actual 16.2%; now 14.5%).
  All together against the old model on shared player-games: TD log loss 0.3958 to 0.3892, TD Brier 0.1234
  to 0.1215, pass-yard error 57.8 to 56.5 (all clearly better); receiving and rushing yardage error no clear
  difference; win-probability Brier 0.2214 to 0.2150 (book 0.2075).
- **Carries left by injured players go to the running backs (Nov 14 2026): tested, adopted** (`vacate_rb`).
  When a lead back was ruled out, his share used to be spread pro rata over everyone left, which could hand a
  quarterback 40% of the carries (Mahomes with Kenneth Walker III out). Now the vacated share goes to the
  remaining backs. Paired backtest: rushing-yard error 23.55 to 23.40 and squared error both clearly better,
  TD and every other metric no clear difference.
- **Quarterback hold is per team for player calls (Nov 16 2026).** Backtest, players at 15%+ to score: when a
  player's own team changed quarterback the model ran hot (predicted 27.7%, actual 24.0%); when only the
  opponent changed, it did not (30.3% vs 32.1%). So a low-confidence quarterback holds his own team's player
  calls and the game-line calls, and the other team's player calls stay live.
- **Thursday final report timing fixed (Nov 18 2026).** `make_config` treated a Thursday game's injury report
  as final from Tuesday evening; it now waits for Wednesday evening, like the NFL's final Thursday report.
  `inputs.hold_lines` (a reason string) holds game-line calls when the model is far from the market for a
  reason it cannot see, such as the Colts' deadline sell-off.
- **Targets left by absent players go out by recent snaps (Nov 19 2026): tested, adopted** (`vacate_tgt=1.0`).
  When receivers are traded or ruled out, their target share used to go pro rata to whoever already had the
  most targets (Tyler Warren jumped to 34% after the Colts traded Allen and Downs). Now it goes by recent snap
  share. Paired backtest: receiving-yard MAE 24.07 to 23.98, squared error and median error all clearly better,
  and better again in quarterback-change games; everything else no clear difference. Half-weight (0.5) helped
  less.
- **Yardage props vs posted lines (Nov 19 2026).** Every player with a posted line now shows it beside his
  projection with the sim's chance of clearing it. The sim's ranges are well calibrated in the backtest
  (receiving: 9.1% below the 10th percentile, 10.3% above the 90th, 48.8% over the median; rushing and passing
  similar), but with no archive of past lines the edges against real lines are untested.
- **Also tested, not adopted:** a player's own share of his team's past touchdowns as a usage feature
  (`td_hist`, no difference) and a yards-per-attempt ratio for backup quarterbacks (`qb_eff_pow`, made passing worse).
- **Coverage matchups (Nov 2026): tested, not adopted.** `research/coverage_test.py` checks whether a player's
  split against MOFC vs MOFO coverage (and man vs zone), combined with how often the opposing defence plays
  each, improves receiving predictions. Walk-forward on 2024 and 2025 (4,343 WR and TE player-games, coverage
  charting from nflverse `pbp_participation`). Result: no improvement in yards or targets for either position,
  and in the strongest predicted matchups the coverage model pointed the right way under half the time.
  Player splits barely persist year to year (correlation about 0.16, against 0.56 for overall yards per route).
  Defensive tendencies do persist, but the league-wide effect is only a few percent. Results are saved in
  `research/coverage_test_results_*.json`. Current-season coverage data is not free, so this is not worth buying.

## Call thresholds on the page

- Anytime TD: Play = sim beats the best price by 3+ points and sim is at least 10%; Lean = 1.5 to 3; Fade = 5+ under.
- Yardage calculator: Play = 4+ points over break-even; Lean = 2 to 4.
- Spread, total, moneyline: Lean at most.

Refresh the backtest a few times a season: `cd src && python3 backtest.py 2024,2025,2026 2500`.
