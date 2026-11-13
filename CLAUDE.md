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
- Writing for Sam: concise, paragraphs over bullets, no em dashes.
- Keep the honest framing on the page (the "How far to trust it" section). Game lines never get more than a Lean.

## Weekly run (Saturday morning, unattended)

1. `pip install --break-system-packages -r requirements.txt`, then `./fetch_data.sh`.
2. Pick the game: the coming Sunday night game (the Sunday game with the latest kickoff in `data/games.csv`,
   `home_score` empty). If Sam named a different game in the task or in `games/NEXT.txt` (format `AWAY HOME`),
   use that instead and clear the file afterwards. If there is no game in the next 7 days (off-season), stop and say so.
3. `python3 src/make_config.py AWAY HOME` to start `games/AWAY_HOME.json`.
4. Research and fill in the config (see README "What goes in a game config"):
   - Injuries: the final Friday report for both teams. Put ruled-out skill players' gsis ids in `out`
     (look ids up in `data/roster_weekly_<season>.csv`). If a starting QB is out, set `qb` to the replacement's id.
     List notable statuses in `injuries` as short strings.
   - Odds: spread, total and moneyline from a named book (bet365 or Stake preferred, then ESPN/DraftKings).
     Anytime TD prices for every player you can find, keeping the best price and its book. Yardage lines if posted.
   - Weather: outdoor games only. Use the hourly forecast for the kickoff window, not the daily summary.
     Set `weather` (summary, precip %, wind, note, effect) and `wx_model` using the measured effects in the README.
     Rain chance of 30% or more is a real factor: say so and steer away from receiving-yard overs.
   - Latest trends: check the last week of news for role changes (new starter, committee change, snap or carry
     share shifts, trades). The model already weights recent games, so only act on changes the data cannot see yet:
     note them in the summary and, if a player is clearly out of the rotation, add him to `out`.
   - `sources`: label and URL for every page a number came from.
5. `./run_game.sh games/AWAY_HOME.json`. Sanity-check the console output: win probability within about
   15 points of the market, no player with an absurd share, both starting QBs correct.
6. Open `site/index.html` once to confirm it renders, then publish it to the artifact above.
7. Commit the new game config and push to `main`.
8. Send Sam a short summary: the game, model vs book on the winner, any Play or Lean calls, key injuries,
   weather, and anything you could not find.

## Changing the model

After any change to `src/model.py`, rerun the backtest (`cd src && python3 backtest.py 2024,2025,2026 2500`)
and commit the refreshed `work/bt_report.json`. Report results honestly, including when a change does not help.
