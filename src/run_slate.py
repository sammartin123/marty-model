"""Run every game in a slate file and write one JSON for the multi-game page.
usage: python3 src/run_slate.py slates/2026-W10.json   ->  out/slate.json (+ games/<AWAY_HOME>.json per game)

A slate file lists the games as "AWAY_HOME" keys. Each may override market, td_odds, yard_lines, weather,
wx_model, out, qb and sources; everything else (teams, kickoff, venue, injuries, starting QBs) is built
automatically from the data files. "use_existing" points at a hand-finished game config to reuse."""
import sys, os, json, datetime as dt
import pandas as pd
import model as M
from make_config import build
from run_game import run
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

def main(path):
    S = json.load(open(path))
    db = M.DB()
    games = []
    sched = pd.read_csv(os.path.join(ROOT, 'data', 'games.csv'))
    for key, ov in S['games'].items():
        away, home = key.split('_')
        G = build(away, home, S.get('season'))
        if ov.get('use_existing'):
            E = json.load(open(os.path.join(ROOT, ov['use_existing'])))
            for k in ('market', 'td_odds', 'yard_lines', 'weather', 'wx_model', 'sources'):
                G[k] = E[k]
            G['meta']['td_note'] = E['meta'].get('td_note', 'Market = the implied chance from the best anytime TD price found for each player across several books on Nov 12 to 13. Confirm the price in your own book first.')
        for k in ('market', 'td_odds', 'yard_lines', 'weather', 'wx_model'):
            if k in ov:
                G[k] = ov[k]
        G['sources'] = G['sources'][:1] + ov.get('sources', G['sources'][1:])
        names = {v: k for k, v in db.pnames.items()}
        ros = db.pg[db.pg.season == G['season']]
        def pid_of(nm, team):
            x = ros[(ros.name == nm) & (ros.team == team)]
            return x.pid.iloc[0] if len(x) else names.get(nm)
        for nm, team in ov.get('out_names', []):
            G['out'].append(pid_of(nm, team)); G['injuries'].append(f"{team} {nm}: Out (from news reports)")
        for team, nm in ov.get('qb_names', {}).items():
            G['qb'][team] = pid_of(nm, team); G['inputs']['qbs'][team] = nm
            G['inputs']['low_conf'].pop(team, None)
        G['inputs']['low_conf'].update(ov.get('low_conf', {}))
        G['inputs']['notes'] = ov.get('notes', [])
        G['out'] = sorted(set(G['out']) | set(ov.get('out', [])))
        G['meta'].setdefault('td_note', S.get('td_note', ''))
        if G['market'] is None:
            print('SKIP (no line):', key); continue
        json.dump(G, open(os.path.join(ROOT, 'games', f'{key}.json'), 'w'), indent=1)
        o = run(G, db, quiet=True, record=False)
        gm = o['game']; top = sorted(o['players'], key=lambda p: -p['p_td'])[:3]
        print(' ', G['inputs']['qbs'], 'low_conf', list(G['inputs']['low_conf']), 'out', len(G['out']))
        print(f"{away:>3} @ {home:<3} {home} {gm['p_home']:.3f} (book {gm['mkt_nv_home']:.3f})  {gm['mean_away']:.1f}-{gm['mean_home']:.1f}  tot {gm['med_total']:.0f} (book {G['market']['total']})  "
              + ', '.join(f"{p['name']} {p['p_td']:.2f}" for p in top) + (f"  UNMATCHED {o['inputs']['unmatched']}" if o['inputs'].get('unmatched') else ''))
        games.append(o)
    import ledger
    record = ledger.grade(db)
    out = dict(meta=dict(label=S['label'], week=S['week'], season=S['season'],
                         generated=dt.datetime.now().strftime('%a %b %d, %Y').replace(' 0', ' ')), games=games, record=record)
    os.makedirs(os.path.join(ROOT, 'out'), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, 'out', 'slate.json'), 'w'))
    print('wrote out/slate.json', len(games), 'games')
    print('RESULTS TRACKER UPDATED: commit everything with git add -A (ledger/ included) and push.')

if __name__ == '__main__':
    main(sys.argv[1])
