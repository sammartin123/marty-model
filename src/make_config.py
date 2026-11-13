"""Start a game config from the schedule, depth charts, injury file and listed lines.
usage: python3 src/make_config.py AWAY HOME [season]   ->  games/AWAY_HOME.json
Then fill in odds (td_odds, yard_lines, market), weather and sources before running the sim."""
import sys, os, json, datetime as dt
import pandas as pd
from teams import TEAMS, venue
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
SKILL = ['QB', 'RB', 'WR', 'TE', 'FB']

def build(away, home, season=None):
    g = pd.read_csv(os.path.join(ROOT, 'data', 'games.csv'))
    season = season or int(g.season.max())
    r = g[(g.season == season) & (g.home_team == home) & (g.away_team == away) & g.home_score.isna()]
    if r.empty:
        raise SystemExit(f'No upcoming {away} at {home} game found in {season}')
    r = r.iloc[0]
    week = int(r.week)
    # ---- injuries: Out / Doubtful are removed, Questionable is flagged
    out, status, notes = [], {}, []
    ip = os.path.join(ROOT, 'data', f'injuries_{season}.csv')
    if os.path.exists(ip):
        inj = pd.read_csv(ip)
        inj = inj[(inj.week == week) & inj.team.isin([home, away])]
        for x in inj[inj.position.isin(SKILL) & inj.report_status.notna()].itertuples():
            status[x.gsis_id] = x.report_status
            notes.append(f"{x.team} {x.full_name} ({x.position}): {x.report_status}" + (f", {x.report_primary_injury}" if isinstance(x.report_primary_injury, str) else ''))
            if x.report_status in ('Out', 'Doubtful'):
                out.append(x.gsis_id)
    # ---- starting QBs: depth chart QB1 if active and not ruled out, else the next one down
    dc = pd.read_csv(os.path.join(ROOT, 'data', f'depth_charts_{season}.csv'))
    dc = dc[dc.dt == dc.dt.max()]
    ros = pd.read_csv(os.path.join(ROOT, 'data', f'roster_weekly_{season}.csv'), low_memory=False)
    ros['team'] = ros.team.replace({'LAR': 'LA'})
    pgp = os.path.join(ROOT, 'work', 'player_games.parquet')
    if not os.path.exists(pgp) or os.path.getmtime(pgp) < os.path.getmtime(os.path.join(ROOT, 'data', 'games.csv')):
        import subprocess
        subprocess.run([sys.executable, os.path.join(ROOT, 'src', 'prep.py')], check=True, stdout=subprocess.DEVNULL)
    pg = pd.read_parquet(pgp)
    pg = pg[pg.season == season]
    dnp = set(inj[inj.practice_status.fillna('').str.contains('Did Not')].gsis_id) if os.path.exists(ip) else set()
    qb, qbn, low_conf = {}, {}, {}
    for t in (home, away):
        rt = ros[ros.team == t]
        active = set(rt[(rt.week == rt.week.max()) & (rt.status == 'ACT')].gsis_id) - set(out)
        q = dc[(dc.team == t) & (dc.pos_abb == 'QB') & dc.gsis_id.isin(active)].sort_values('pos_rank')
        qb[t], qbn[t] = q.gsis_id.iloc[0], q.player_name.iloc[0]
        att = int(pg[(pg.pid == qb[t]) & (pg.team == t)].pass_att.sum())
        if att < 60:
            low_conf[t] = f"{qbn[t]} has {att} pass attempts for the {TEAMS[t][1]} this season, so their numbers are low confidence."
        elif qb[t] in dnp:
            low_conf[t] = f"{qbn[t]} has not practised this week, so the {TEAMS[t][1]} quarterback is not certain."
    # injury report counts as final from 6 PM ET two days before the game (Friday for Sunday, Wednesday for Thursday)
    final_at = dt.datetime.strptime(r.gameday, '%Y-%m-%d') - dt.timedelta(days=2) + dt.timedelta(hours=23)
    report_final = dt.datetime.utcnow() >= final_at and os.path.exists(ip)
    d = dt.datetime.strptime(r.gameday, '%Y-%m-%d')
    hh, mm = map(int, str(r.gametime).split(':'))
    kick = f"{d.strftime('%a %b')} {d.day}, {d.year} · {((hh - 1) % 12) + 1}:{mm:02d} {'PM' if hh >= 12 else 'AM'} ET"
    la, lo, roof, neutral, place = venue(home, str(r.stadium))
    last = g[(g.season == season) & g.home_score.notna()]
    has_line = pd.notna(r.spread_line) and pd.notna(r.total_line) and pd.notna(r.home_moneyline)
    cfg = dict(home=home, away=away, season=season, week=week, asof=r.gameday, n_sims=100000, seed=int(d.strftime('%Y%m%d')),
               neutral=neutral, qb=qb, out=out, status=status, out_json=f'out/{away}_{home}.json',
               meta=dict(title=f'{TEAMS[away][1]} at {TEAMS[home][1]}', home_name=TEAMS[home][1], away_name=TEAMS[away][1],
                         home_city=TEAMS[home][0], away_city=TEAMS[away][0], kickoff=kick, time=f"{((hh - 1) % 12) + 1}:{mm:02d} {'PM' if hh >= 12 else 'AM'}",
                         venue=f"{r.stadium}, {place}", generated=dt.date.today().strftime('%a %b %d, %Y').replace(' 0', ' '), roof=roof,
                         lat=la, lon=lo, data_through=f"Week {int(last.week.max())}" if len(last) else 'last season'),
               market=dict(spread_home=-float(r.spread_line), total=float(r.total_line), ml_home=int(r.home_moneyline),
                           ml_away=int(r.away_moneyline), book='schedule feed line', others=[]) if has_line else None,
               weather=dict(summary='Indoors' if roof == 'dome' else 'Not checked yet', precip=0, wind=None, rain_flag=False,
                            note='Fixed roof, so weather is not a factor.' if roof == 'dome' else 'Fill in the hourly forecast for the kickoff window.',
                            effect='No weather adjustment.'),
               wx_model=dict(pass_mult=1.0, cr_mult=1.0, td_shift=0.0, fg_shift=0.0),
               td_odds={}, yard_lines={}, injuries=notes,
               inputs=dict(qbs=qbn, report_final=bool(report_final), low_conf=low_conf),
               sources=[dict(label='Play-by-play, rosters, depth charts, injuries, schedule (nflverse)',
                             url='https://github.com/nflverse/nflverse-data')])
    return cfg

if __name__ == '__main__':
    away, home = sys.argv[1].upper(), sys.argv[2].upper()
    cfg = build(away, home, int(sys.argv[3]) if len(sys.argv) > 3 else None)
    p = os.path.join(ROOT, 'games', f'{away}_{home}.json')
    json.dump(cfg, open(p, 'w'), indent=1)
    print('wrote', p, '| QBs', cfg['inputs']['qbs'], '| out', cfg['out'], '| line', cfg['market'])
