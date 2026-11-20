# MartyMakesMoneyModel: instructions for Claude

This repo is Sam Martin's NFL game simulator. Read README.md for how the model works. This file is the
runbook for the weekly run and the rules that always apply.

## Always

- Never invent a price, line, stat, injury status or forecast. If something cannot be found, leave it out
  and say so in the summary. A missing price shows on the page as "none found".
- Never place bets, create sportsbook accounts or sign in to a sportsbook.
- The page is one private artifact. Always republish to the same one so Sam's link never changes:
  slug `79852994-f850-4c0e-876b-fdb91bc130c4` (https://claude.ai/artifact/G1Lc9453gKaZtNvn6oUVg7).
  Get its current version first, then publish `site/index.html` as `index.html` with that base version.
- Results tracker: every run saves its calls to `ledger/pending/` before kickoff and grades finished games into
  `ledger/graded.json`, which feeds the Track record section of the page. This only survives if it is committed,
  so always commit with `git add -A` (never just the config files) and push. Never edit ledger files by hand.
- Writing for Sam: concise, paragraphs over bullets, no em dashes.
- Keep the honest framing on the page (the "How far to trust it" section). Game lines never get more than a Lean.

## Weekly run (Thursday morning, unattended)

1. `pip install --break-system-packages -r requirements.txt`, then `./fetch_data.sh`.
2. Pick the game: tonight's Thursday night game (the game dated today in `data/games.csv` with `home_score`
   empty; on Thanksgiving, the one with the latest kickoff). If Sam named a different game in the task or in
   `games/NEXT.txt` (format `AWAY HOME`), use that instead and clear the file afterwards. If there is no game
   today, stop and tell Sam there was no Thursday game this week.
3. `python3 src/make_config.py AWAY HOME` to start `games/AWAY_HOME.json`. If the file already exists from an earlier run
   this week, it keeps that run's lines, TD prices, weather, notes and sources and rebuilds only injuries and
   quarterbacks: recheck the prices, re-add any late `out` players the final report does not cover, and
   rerun. Use `--fresh` to start over.
4. Research and fill in the config (see README "What goes in a game config"):
   - Injuries: the final report for both teams (Wednesday for a Thursday game, Friday for a Sunday game). Put ruled-out skill players' gsis ids in `out`
     (look ids up in `data/roster_weekly_<season>.csv`). If a starting QB is out, set `qb` to the replacement's id.
     List notable statuses in `injuries` as short strings.
   - Odds: spread, total and moneyline from a named book (bet365 or Stake preferred, then ESPN/DraftKings).
     Anytime TD prices for every player you can find, keeping the best price and its book. Yardage lines if posted.
   - Weather: confirm outdoors, dome or retractable roof first (the schedule often leaves it blank).
     Outdoor games only: use the hourly forecast for the kickoff window, not the daily summary.
     Set `weather` (summary, precip %, wind, note, effect) and `wx_model` using the measured effects in the README.
     Rain chance of 30% or more is a real factor: say so and steer away from receiving-yard overs.
   - Latest trends: check the last week of news for role changes (new starter, committee change, snap or carry
     share shifts, trades). The model already weights recent games, so only act on changes the data cannot see yet:
     note them in the summary and, if a player is clearly out of the rotation, add him to `out`.
   - `sources`: label and URL for every page a number came from.
5. `./run_game.sh games/AWAY_HOME.json`. Sanity-check the console output: win probability within about
   15 points of the market, no player with an absurd share, both starting QBs correct.
   If the gap is bigger and the cause is news the data cannot see yet (a roster sell-off, a mid-week trade),
   set `inputs.hold_lines` to a one-sentence reason: game-line calls are then held, player props stay live.
6. Open `site/index.html` once to confirm it renders, then publish it to the artifact above.
7. Commit the new game config and push to `main`.
8. Send Sam a short summary: the game, model vs book on the winner, any Play or Lean calls, key injuries,
   weather, and anything you could not find.

## Changing the model

Only adopt a change that beats the current model in a paired comparison on the same player-games
(`python3 research/compare_bt.py <old bt_players.parquet> <new bt_players.parquet>`; run the backtest with
`tag=NAME` to keep both). Log what was tested in the README research log, including what did not help.
Player props depend on the config's `market` spread and total (they set team scoring), so a wrong line now
shifts every prop: double-check both numbers.

After any change to `src/model.py`, rerun the backtest (`cd src && python3 backtest.py 2024,2025,2026 2500`)
and commit the refreshed `work/bt_report.json`. Report results honestly, including when a change does not help.
Calls blend model and book (`CALL_W` in src/run_game.py). Recheck it against `ledger/graded.json` every few
weeks: if the model's probabilities start beating the book's on graded props, the weight can go up.

## Sunday slate (every Sunday game on one page)

Sam wants every Sunday game on the page, with calls only where the data is fully confirmed. See README
"Run a whole slate". For a new week: copy the latest file in `slates/`, replace the games, then for each game
fill in `market` (ESPN/DraftKings lines are fine), `td_odds` only for players whose price you actually found
from a named book, `weather` for outdoor venues, and `sources`. The data's injury file is the Friday report, so search for Saturday downgrades and additions (one
search like "NFL Week N injury downgrades ruled out Saturday", plus each team's questionable skill players)
and add them to the game as `out_names` or `q_names` with a note and source. Check every low-confidence quarterback flag
the run prints against the news and add a short `notes` line explaining it. Never add a TD price you did not
see quoted. Publish to the same artifact and push.
