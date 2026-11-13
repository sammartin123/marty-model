"""Results tracker. Every run snapshots its calls before kickoff (ledger/pending/<game_id>.json);
once a game is final the snapshot is graded and moved into ledger/graded.json.

Call thresholds here must match the page (src/template.html, tdCall / lineCall / leans)."""
import os, json, glob, datetime as dt
from zoneinfo import ZoneInfo
import pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
LED = os.path.join(ROOT, 'ledger')
PEND = os.path.join(LED, 'pending')
GRADED = os.path.join(LED, 'graded.json')

def imp(o):
    return -o / (-o + 100) if o < 0 else 100 / (o + 100)

def payout(price):
    return price / 100 if price > 0 else 100 / -price

def kickoff_utc(season, away, home):
    g = pd.read_csv(os.path.join(ROOT, 'data', 'games.csv'))
    r = g[(g.season == season) & (g.away_team == away) & (g.home_team == home)].sort_values('gameday').iloc[-1]
    t = dt.datetime.strptime(f"{r.gameday} {r.gametime}", '%Y-%m-%d %H:%M').replace(tzinfo=ZoneInfo('America/New_York'))
    return r.game_id, t.astimezone(dt.timezone.utc)

def log_game(G, o):
    """Snapshot this game's calls. Skipped once the game has kicked off, so the last pre-game run is what gets graded."""
    gid, ko = kickoff_utc(G['season'], G['away'], G['home'])
    if dt.datetime.now(dt.timezone.utc) >= ko:
        return
    I = o.get('inputs', {})
    final = I.get('report_final', True); lowc = bool(I.get('low_conf'))
    E = []
    for p in o['players']:
        gate = 'QB unclear' if lowc else ('Questionable' if p.get('status') == 'Questionable' else (None if final else 'Injuries not final'))
        if p.get('td_mkt'):
            e = p['td_edge']
            base = 'play' if (e >= .03 and p['p_td'] >= .10) else 'lean' if (e >= .015 and p['p_td'] >= .10) else 'fade' if e <= -.05 else 'pass'
            E.append(dict(kind='td', pid=p['id'], name=p['name'], team=p['team'], price=p['td_mkt']['price'], book=p['td_mkt']['book'],
                          sim=round(p['p_td'], 4), mkt=round(p['td_mkt']['implied'], 4), call=base, live=gate is None and base != 'pass', gate=gate))
        for L in p.get('lines', []):
            side = 'over' if L['p_over'] >= .5 else 'under'
            pr = max(L['p_over'], 1 - L['p_over']); price = L.get(side)
            edge = pr - (imp(price) if price is not None else .524)
            base = 'play' if edge >= .04 else 'lean' if edge >= .02 else 'pass'
            g2 = gate or (None if price is not None else 'No price for that side')
            E.append(dict(kind='yards', pid=p['id'], name=p['name'], team=p['team'], stat=L['stat'], line=L['line'], side=side, price=price,
                          book=L.get('book'), sim=round(pr, 4), call=base, live=g2 is None and base != 'pass', gate=g2))
    gm, mk = o['game'], o['market']; sp = mk['spread_home']; H, A = G['home'], G['away']
    def lean(kind, ok, **kw):
        E.append(dict(kind=kind, call='lean' if ok else 'pass', live=bool(ok) and not lowc, gate='QB unclear' if lowc else None, **kw))
    side_home = gm['p_home_cover'] >= gm['p_away_cover']
    lean('spread', max(gm['p_home_cover'], gm['p_away_cover']) >= .54, team=H if side_home else A, line=sp if side_home else -sp, price=-110,
         sim=round(max(gm['p_home_cover'], gm['p_away_cover']), 4))
    over = gm['p_over'] >= gm['p_under']
    lean('total', max(gm['p_over'], gm['p_under']) >= .54, side='over' if over else 'under', line=mk['total'], price=-110, sim=round(max(gm['p_over'], gm['p_under']), 4))
    eh, ea = gm['p_home'] - imp(mk['ml_home']), gm['p_away'] - imp(mk['ml_away'])
    lean('ml', max(eh, ea) >= .03, team=H if eh > ea else A, price=mk['ml_home'] if eh > ea else mk['ml_away'], sim=round(gm['p_home'] if eh > ea else gm['p_away'], 4))
    os.makedirs(PEND, exist_ok=True)
    json.dump(dict(game_id=gid, season=G['season'], week=G['week'], away=A, home=H, kickoff_utc=ko.isoformat(), book=mk.get('book'),
                   logged_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec='minutes'), entries=E),
              open(os.path.join(PEND, gid + '.json'), 'w'), indent=1)

def grade(db):
    """Grade every pending snapshot whose game is final and in the play-by-play data."""
    graded = json.load(open(GRADED)) if os.path.exists(GRADED) else []
    done = {g['game_id'] for g in graded}
    tdp = os.path.join(ROOT, 'work', 'tds.parquet')
    tds = pd.read_parquet(tdp) if os.path.exists(tdp) else pd.DataFrame(columns=['game_id', 'pid', 'tds'])
    for f in sorted(glob.glob(os.path.join(PEND, '*.json'))):
        S = json.load(open(f)); gid = S['game_id']
        r = db.games[db.games.game_id == gid]
        pg = db.pg[db.pg.game_id == gid]
        if gid in done:
            os.remove(f); continue
        if r.empty or pd.isna(r.iloc[0].home_score) or pg.empty:
            continue
        r = r.iloc[0]; hs, as_ = int(r.home_score), int(r.away_score)
        S['final'] = {S['home']: hs, S['away']: as_}
        rows = pg.set_index('pid'); t = tds[tds.game_id == gid].set_index('pid').tds
        for e in S['entries']:
            res = None
            if e['kind'] == 'td':
                scored = int(t.get(e['pid'], 0)) > 0
                res = 'void' if (e['pid'] not in rows.index and not scored) else ('win' if scored else 'loss')
                e['actual'] = int(t.get(e['pid'], 0))
                if e['call'] == 'fade':               # a fade is right when the player does not score
                    res = res if res == 'void' else ('win' if not scored else 'loss')
            elif e['kind'] == 'yards':
                if e['pid'] not in rows.index:
                    res = 'void'
                else:
                    x = rows.loc[e['pid']]
                    val = float({'rec': x.rec_yds, 'rush': x.rush_yds, 'pass': x.pass_yds, 'recs': x.rec}[e['stat']])
                    e['actual'] = val
                    res = 'push' if val == e['line'] else ('win' if (val > e['line']) == (e['side'] == 'over') else 'loss')
            elif e['kind'] == 'spread':
                m = (hs - as_) if e['team'] == S['home'] else (as_ - hs)
                res = 'push' if m + e['line'] == 0 else ('win' if m + e['line'] > 0 else 'loss')
            elif e['kind'] == 'total':
                tot = hs + as_
                res = 'push' if tot == e['line'] else ('win' if (tot > e['line']) == (e['side'] == 'over') else 'loss')
            elif e['kind'] == 'ml':
                w = S['home'] if hs > as_ else (S['away'] if as_ > hs else None)
                res = 'push' if w is None else ('win' if w == e['team'] else 'loss')
            e['result'] = res
            can_bet = e.get('price') is not None and e['call'] != 'fade'
            e['units'] = 0.0 if (res in ('void', 'push') or not can_bet) else (round(payout(e['price']), 3) if res == 'win' else -1.0)
        graded.append(S); done.add(gid); os.remove(f)
    os.makedirs(LED, exist_ok=True)
    json.dump(graded, open(GRADED, 'w'), indent=1)
    return summary(graded)

def summary(graded=None):
    """Compact record for the page. Only calls that were live before kickoff count toward the record."""
    if graded is None:
        graded = json.load(open(GRADED)) if os.path.exists(GRADED) else []
    def bucket():
        return dict(w=0, l=0, p=0, units=0.0)
    rec = {k: bucket() for k in ['td_play', 'td_lean', 'td_fade', 'yards', 'spread', 'total', 'ml']}
    recent, sq_m, sq_b, n = [], 0.0, 0.0, 0
    for S in graded:
        for e in S['entries']:
            if e['kind'] == 'td' and e.get('result') in ('win', 'loss'):
                scored = 1.0 if e.get('actual', 0) > 0 else 0.0
                sq_m += (e['sim'] - scored) ** 2; sq_b += (e['mkt'] - scored) ** 2; n += 1
            if not e.get('live') or e.get('result') in (None, 'void'):
                continue
            key = ('td_' + e['call']) if e['kind'] == 'td' else e['kind']
            if key not in rec:
                continue
            b = rec[key]
            b['w' if e['result'] == 'win' else 'l' if e['result'] == 'loss' else 'p'] += 1
            b['units'] = round(b['units'] + e.get('units', 0.0), 3)
            label = (e['name'] + ' anytime TD' + (' (fade)' if e['call'] == 'fade' else '')) if e['kind'] == 'td' else \
                    (f"{e['name']} {e['side']} {e['line']} {e['stat']} yds" if e['kind'] == 'yards' else
                     f"{e['team']} {'+' if e['line'] > 0 else ''}{e['line']}" if e['kind'] == 'spread' else
                     f"{e['side'].title()} {e['line']}" if e['kind'] == 'total' else f"{e['team']} to win")
            recent.append(dict(week=S['week'], game=f"{S['away']} at {S['home']}", label=label, call=e['call'], price=e.get('price'),
                               result=e['result'], units=e.get('units', 0.0)))
    pend = 0; pend_games = 0
    for f in glob.glob(os.path.join(PEND, '*.json')):
        P = json.load(open(f)); pend_games += 1
        pend += sum(1 for e in P['entries'] if e.get('live'))
    weeks = sorted({(S['season'], S['week']) for S in graded})
    return dict(buckets=rec, recent=recent[-14:][::-1], games_graded=len(graded), weeks=len(weeks),
                priced=dict(n=n, brier_model=round(sq_m / n, 4) if n else None, brier_book=round(sq_b / n, 4) if n else None),
                pending_calls=pend, pending_games=pend_games)
