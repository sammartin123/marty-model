"""Start a game config from the schedule, depth charts and listed lines.
usage: python3 src/make_config.py AWAY HOME [season]   ->  games/AWAY_HOME.json
Fill in td_odds, yard_lines, weather, wx_model, out (injured player ids) and sources before running the sim."""
import sys, os, json, datetime as dt
import pandas as pd
from teams import TEAMS
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
away, home = sys.argv[1].upper(), sys.argv[2].upper()
g = pd.read_csv(os.path.join(ROOT, 'data', 'games.csv'))
season = int(sys.argv[3]) if len(sys.argv) > 3 else int(g.season.max())
r = g[(g.season == season) & (g.home_team == home) & (g.away_team == away) & g.home_score.isna()]
if r.empty:
    sys.exit(f'No upcoming {away} at {home} game found in {season}')
r = r.iloc[0]
dc = pd.read_csv(os.path.join(ROOT, 'data', f'depth_charts_{season}.csv'))
dc = dc[dc.dt == dc.dt.max()]
qb = {}
for t in (home, away):
    q = dc[(dc.team == t) & (dc.pos_abb == 'QB')].sort_values('pos_rank')
    qb[t] = q.gsis_id.iloc[0]
d = dt.datetime.strptime(r.gameday, '%Y-%m-%d')
hh, mm = map(int, str(r.gametime).split(':'))
kick = f"{d.strftime('%a %b')} {d.day}, {d.year} · {((hh - 1) % 12) + 1}:{mm:02d} {'PM' if hh >= 12 else 'AM'} ET"
last = g[(g.season == season) & g.home_score.notna()]
cfg = dict(home=home, away=away, season=season, week=int(r.week), asof=r.gameday, n_sims=100000, seed=int(d.strftime('%Y%m%d')),
           qb=qb, out=[], out_json=f'out/{away}_{home}.json',
           meta=dict(title=f'{TEAMS[away][1]} at {TEAMS[home][1]}', home_name=TEAMS[home][1], away_name=TEAMS[away][1],
                     home_city=TEAMS[home][0], away_city=TEAMS[away][0], kickoff=kick, venue=str(r.stadium),
                     generated=dt.date.today().strftime('%a %b %d, %Y').replace(' 0', ' '), roof=str(r.roof),
                     data_through=f"Week {int(last.week.max())}" if len(last) else 'last season'),
           market=dict(spread_home=-float(r.spread_line), total=float(r.total_line), ml_home=int(r.home_moneyline),
                       ml_away=int(r.away_moneyline), book='nflverse schedule line (replace with a named book)', others=[]),
           weather=dict(summary='Not checked yet', precip=0, wind=None, rain_flag=False, note='Fill in the hourly forecast for the kickoff window.'),
           wx_model=dict(pass_mult=1.0, cr_mult=1.0, td_shift=0.0, fg_shift=0.0),
           td_odds={}, yard_lines={}, sources=[dict(label='Play-by-play, rosters, depth charts, injuries, schedule (nflverse)',
                                                    url='https://github.com/nflverse/nflverse-data')], injuries=[])
p = os.path.join(ROOT, 'games', f'{away}_{home}.json')
json.dump(cfg, open(p, 'w'), indent=1)
print('wrote', p, '| QBs', qb, '| line', cfg['market'])
