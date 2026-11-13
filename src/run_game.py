"""Run MartyMakesMoneyModel for one game and write the JSON the page reads.
usage: python3 run_game.py <game_config.json>"""
import os, sys, json, numpy as np, pandas as pd
import model as M

cfgfile = sys.argv[1]
G = json.load(open(cfgfile))
N = G.get('n_sims', 100000)
home, away = G['home'], G['away']
asof = pd.Timestamp(G['asof'])
cfg = dict(M.CFG)
db = M.DB()
rng = np.random.default_rng(G.get('seed', 20261115))

# ------------------------------------------------------------ rosters
ros = pd.read_csv(os.path.join(M.DATA, f"roster_weekly_{G['season']}.csv"), low_memory=False)
ros['team'] = ros.team.replace({'LAR': 'LA'})
wk = ros[ros.team.isin([home, away])].week.max()
ros = ros[(ros.week == wk) & (ros.status == 'ACT') & ros.position.isin(['QB', 'RB', 'WR', 'TE', 'FB'])]
out_ids = set(G.get('out', []))
tabs, vols = [], []
for team, opp in [(home, away), (away, home)]:
    ids = set(ros[ros.team == team].gsis_id) - out_ids
    t = M.player_table(db, team, opp, asof, ids, qb_id=G['qb'][team], cfg=cfg)
    tabs.append(t); vols.append(M.team_volume(db, team, asof))

fit = M.fit_team(db, asof, cfg)
par = M.team_params(fit, home, away)
wx = G.get('wx_model', dict(pass_mult=1.0, td_shift=0.0, fg_shift=0.0, cr_mult=1.0))
o = M.simulate(db, par, tabs, vols, N, rng, cfg, wx=wx, keep=True)
pts = o['pts']; mg = pts[:, 0] - pts[:, 1]; tot = pts.sum(1)

def american(p):
    p = float(np.clip(p, 1e-4, 1 - 1e-4))
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))

def implied(ml):
    return -ml / (-ml + 100) if ml < 0 else 100 / (ml + 100)

def q(a, ps):
    return [round(float(x), 1) for x in np.quantile(a, ps)]

QS = np.linspace(0, 1, 101)

# ------------------------------------------------------------ game level
mk = G['market']
p_home = float((mg > 0).mean() + 0.5 * (mg == 0).mean())
sp = mk['spread_home']            # e.g. -3.5 means home favoured by 3.5
p_home_cover = float((mg + sp > 0).mean()); p_push = float((mg + sp == 0).mean())
p_over = float((tot > mk['total']).mean()); p_tpush = float((tot == mk['total']).mean())
ih, ia = implied(mk['ml_home']), implied(mk['ml_away'])
nv_home = ih / (ih + ia)
hist_m = {int(k): int(v) for k, v in zip(*np.unique(np.clip(mg, -35, 35), return_counts=True))}
hist_t = {int(k): int(v) for k, v in zip(*np.unique(np.clip(tot, 0, 100), return_counts=True))}
game = dict(p_home=p_home, p_away=1 - p_home, mean_home=float(pts[:, 0].mean()), mean_away=float(pts[:, 1].mean()),
            med_margin=float(np.median(mg)), mean_margin=float(mg.mean()), med_total=float(np.median(tot)),
            mean_total=float(tot.mean()), sd_margin=float(mg.std()), hist_margin=hist_m, hist_total=hist_t,
            p_home_cover=p_home_cover, p_away_cover=1 - p_home_cover - p_push, p_over=p_over, p_under=1 - p_over - p_tpush,
            mkt_nv_home=nv_home, fair_ml_home=american(p_home), fair_ml_away=american(1 - p_home),
            p_ot=float(o['ot'].mean()), drives=float(o['nd'].mean()))

# ------------------------------------------------------------ typical game
m_star = int(np.round(np.median(mg)))
if m_star == 0:
    m_star = 1 if mg.mean() > 0 else -1
cand = (mg == m_star) & (~o['ot']) & (np.abs(tot - np.median(tot)) <= 4)
pairs, cnt = np.unique(pts[cand], axis=0, return_counts=True)
best = pairs[cnt.argmax()]
idx = np.where(cand & (pts[:, 0] == best[0]) & (pts[:, 1] == best[1]))[0]
# closest to the median stat lines for the main players
dist = np.zeros(len(idx))
for t in (0, 1):
    s = o['stats'][t]
    for arr, w in [(s['pass_yds'], 1 / 60.0)]:
        dist += np.abs(arr[idx] - np.median(arr)) * w
    top = np.argsort(-s['ryds'].mean(0))[:3]
    for j in top:
        dist += np.abs(s['ryds'][idx, j] - np.median(s['ryds'][:, j])) / 30.0
    topr = np.argsort(-s['uyds'].mean(0))[:2]
    for j in topr:
        dist += np.abs(s['uyds'][idx, j] - np.median(s['uyds'][:, j])) / 25.0
    # every touchdown must have a plausible owner in the box score
    dist += 50 * ((s['ctd'][idx] > s['recs'][idx]).any(1) | (s['rtd'][idx] > s['cars'][idx]).any(1))
i = int(idx[dist.argmin()])
sc = pd.read_parquet(os.path.join(M.WORK, 'scores.parquet'))
sc = sc[sc.season >= 2021]
rush_len = sc[(sc.rush_touchdown == 1)].yards_gained.dropna().values
pass_len = sc[(sc.pass_touchdown == 1)].yards_gained.dropna().values
fg_len = sc[(sc.field_goal_result == 'made')].kick_distance.dropna().values
trng = np.random.default_rng(i)
ev = o['ev']
teams = [home, away]
scorers = []
for t in (0, 1):
    s = o['stats'][t]; df = tabs[t]
    lst = []
    for j in range(len(df)):
        lst += [('rush', j)] * int(s['rtd'][i, j]) + [('rec', j)] * int(s['ctd'][i, j])
    trng.shuffle(lst)
    scorers.append(lst)
events = []; score = [0, 0]
for k in range(ev['team'].shape[1]):
    t = int(ev['team'][i, k]); kind = int(ev['kind'][i, k])
    if t < 0 or kind == 0:
        continue
    half = int(ev['half'][i, k]); tend = int(ev['tend'][i, k])
    qtr = half * 2 + (1 if tend > 900 else 2); clock = tend - 900 if tend > 900 else tend
    e = dict(q=qtr, clock=f"{clock // 60}:{clock % 60:02d}")
    pat = int(ev['pat'][i, k])
    if kind == 1:
        typ, j = scorers[t].pop()
        df = tabs[t]; nm = df.name[j]
        yd = int(trng.choice(rush_len if typ == 'rush' else pass_len)); yd = max(1, min(yd, 80))
        score[t] += 6 + pat
        qbn = df.name[df.is_qb.values].iloc[0] if df.is_qb.any() else 'QB'
        e.update(team=teams[t], type='TD', how=typ, player=nm, yards=yd, passer=qbn if typ == 'rec' else None,
                 pat={0: 'PAT no good', 1: 'PAT good', 2: 'Two-point try good'}[pat])
    elif kind == 2:
        score[t] += 3
        e.update(team=teams[t], type='FG', yards=int(trng.choice(fg_len)))
    else:
        score[1 - t] += 6 + min(pat, 1)
        e.update(team=teams[1 - t], type='DTD', how=str(trng.choice(['Interception return', 'Fumble return', 'Punt return'], p=[.55, .3, .15])),
                 yards=int(trng.integers(18, 70)), pat='PAT good' if pat >= 1 else 'PAT no good')
    e['score'] = {home: score[0], away: score[1]}
    events.append(e)
assert score[0] == pts[i, 0] and score[1] == pts[i, 1], (score, pts[i])

def box(t):
    s = o['stats'][t]; df = tabs[t]
    rows = dict(passing=[], rushing=[], receiving=[])
    qbj = int(np.where(df.is_qb.values)[0][0])
    rows['passing'].append(dict(name=df.name[qbj], comp=int(s['comp'][i]), att=int(s['PA'][i]), yds=int(round(s['pass_yds'][i])),
                                td=int(s['pass_td'][i]), ints=int(s['ints'][i])))
    for j in np.argsort(-s['uyds'][i]):
        if s['cars'][i, j] > 0:
            rows['rushing'].append(dict(name=df.name[j], car=int(s['cars'][i, j]), yds=int(round(s['uyds'][i, j])), td=int(s['rtd'][i, j])))
    for j in np.argsort(-s['ryds'][i]):
        if s['recs'][i, j] > 0:
            rows['receiving'].append(dict(name=df.name[j], rec=int(s['recs'][i, j]), tgt=int(s['tgts'][i, j]),
                                          yds=int(round(s['ryds'][i, j])), td=int(s['ctd'][i, j])))
    return dict(passing=rows['passing'], rushing=rows['rushing'][:4], receiving=rows['receiving'][:6])
typical = dict(score={home: int(pts[i, 0]), away: int(pts[i, 1])}, events=events, box={home: box(0), away: box(1)},
               share=float((pts == best).all(1).mean()))

# ------------------------------------------------------------ defence vs position (display)
tp = db.tpos[db.tpos.season == G['season']]
dvp_rows = {}
labels = {'recyds_WR': 'WR receiving', 'recyds_TE': 'TE receiving', 'recyds_RB': 'RB receiving', 'rushyds_RB': 'RB rushing',
          'passyds': 'QB passing', 'rushyds_QB': 'QB rushing'}
raw = tp.groupby('opp')[list(labels) + ['rectd_WR', 'rectd_TE', 'rectd_RB', 'rushtd_RB', 'rushtd_QB']].mean()
rank = raw.rank(method='min')
for d in (home, away):
    m = M.dvp_mult(db, d, asof, cfg)
    dvp_rows[d] = [dict(key=k, label=lab, per_game=round(float(raw.loc[d, k]), 1), rank=int(rank.loc[d, k]),
                        adj=round(m[k], 3)) for k, lab in labels.items()]
    dvp_rows[d + '_td'] = {k: dict(per_game=round(float(raw.loc[d, k]), 2), rank=int(rank.loc[d, k]))
                           for k in ['rectd_WR', 'rectd_TE', 'rectd_RB', 'rushtd_RB', 'rushtd_QB']}

# ------------------------------------------------------------ players
odds = G.get('td_odds', {}); ylines = G.get('yard_lines', {})
pg = db.pg
players = []
for t, (team, opp) in enumerate([(home, away), (away, home)]):
    s = o['stats'][t]; df = tabs[t]
    anytd = (s['rtd'] + s['ctd']) > 0
    for j in range(len(df)):
        pid = df.pid[j]; pos = df.pos[j]
        p_td = float(anytd[:, j].mean())
        m_rec = float(s['ryds'][:, j].mean()); m_rush = float(s['uyds'][:, j].mean())
        if p_td < 0.01 and m_rec < 4 and m_rush < 4 and not df.is_qb[j]:
            continue
        cur = pg[(pg.pid == pid) & (pg.season == G['season'])]
        season = dict(g=int(len(cur)), tgt=int(cur.targets.sum()), rec=int(cur.rec.sum()), rec_yds=int(cur.rec_yds.sum()),
                      rec_td=int(cur.rec_td.sum()), car=int(cur.carries.sum()), rush_yds=int(cur.rush_yds.sum()),
                      rush_td=int(cur.rush_td.sum()), pass_yds=int(cur.pass_yds.sum()), pass_td=int(cur.pass_td.sum()),
                      att=int(cur.pass_att.sum()), comp=int(cur.comp.sum()), car_in5=int(cur.car_in5.sum()),
                      tgt_rz=int(cur.tgt_rz.sum()), tgt_ez=int(cur.tgt_ez.sum()))
        h = pg[(pg.pid == pid) & (pg.opp == opp) & (pg.game_date < asof)].sort_values('game_date', ascending=False).head(5)
        h2h = [dict(date=str(r.game_date.date()), tgt=int(r.targets), rec=int(r.rec), rec_yds=int(r.rec_yds), car=int(r.carries),
                    rush_yds=int(r.rush_yds), td=int(r.rush_td + r.rec_td), pass_yds=int(r.pass_yds), pass_td=int(r.pass_td),
                    att=int(r.pass_att), comp=int(r.comp)) for r in h.itertuples()]
        posk = pos if pos in ('WR', 'TE', 'RB') else 'WR'
        dv = {x['key']: x for x in dvp_rows[opp]}
        d = dict(id=pid, name=df.name[j], pos=pos, team=team, opp=opp, is_qb=bool(df.is_qb[j]),
                 p_td=p_td, p_2td=float(((s['rtd'] + s['ctd'])[:, j] >= 2).mean()), exp_td=float((s['rtd'] + s['ctd'])[:, j].mean()),
                 p_rush_td=float((s['rtd'][:, j] > 0).mean()), p_rec_td=float((s['ctd'][:, j] > 0).mean()),
                 fair_td=american(p_td), tgt_share=float(df.tgt_share[j]), car_share=float(df.car_share[j]),
                 gl_rush_w=float(df.p_rush_td[j]), rz_rec_w=float(df.p_rec_td[j]),
                 rec=dict(mean=m_rec, q=q(s['ryds'][:, j], QS), n=float(s['recs'][:, j].mean()), tgt=float(s['tgts'][:, j].mean())),
                 rush=dict(mean=m_rush, q=q(s['uyds'][:, j], QS), n=float(s['cars'][:, j].mean())),
                 recs=dict(mean=float(s['recs'][:, j].mean()), q=q(s['recs'][:, j], QS)),
                 season=season, h2h=h2h, h2h_n=int(df.h2h_n[j]), h2h_rec=float(df.h2h_rec[j]), h2h_rush=float(df.h2h_rush[j]),
                 dvp_rec=dict(rank=dv['recyds_' + posk]['rank'], per_game=dv['recyds_' + posk]['per_game'], adj=float(df.m_rec[j]), label=posk),
                 dvp_rush=dict(rank=dv['rushyds_QB' if pos == 'QB' else 'rushyds_RB']['rank'], adj=float(df.m_rush[j])))
        if df.is_qb[j]:
            d['pass'] = dict(mean=float(s['pass_yds'].mean()), q=q(s['pass_yds'], QS), att=float(s['PA'].mean()),
                             comp=float(s['comp'].mean()), td=float(s['pass_td'].mean()),
                             p_2ptd=float((s['pass_td'] >= 2).mean()))
            d['dvp_pass'] = dict(rank=dv['passyds']['rank'], per_game=dv['passyds']['per_game'])
        if len(cur) >= 4:
            key, lab = ('pass_yds', 'pass yds') if df.is_qb[j] else (('rush_yds', 'rush yds') if cur.rush_yds.sum() > cur.rec_yds.sum() else ('rec_yds', 'rec yds'))
            cs = cur.sort_values('game_date')
            if cs[key].mean() >= 15:
                d['form'] = dict(label=lab, season=float(cs[key].mean()), last3=float(cs[key].tail(3).mean()), games=int(len(cs)))
        if df.name[j] in odds:
            od = odds[df.name[j]]
            d['td_mkt'] = dict(price=od['price'], book=od['book'], implied=implied(od['price']), all=od.get('all', []))
            d['td_edge'] = p_td - implied(od['price'])
        if df.name[j] in ylines:
            d['lines'] = []
            for L in ylines[df.name[j]]:
                arr = {'rec': s['ryds'][:, j], 'rush': s['uyds'][:, j], 'pass': s['pass_yds'], 'recs': s['recs'][:, j]}[L['stat']]
                po = float((arr > L['line']).mean())
                e = dict(L); e['p_over'] = po
                if 'over' in L:
                    e['imp_over'] = implied(L['over'])
                d['lines'].append(e)
        players.append(d)

# ------------------------------------------------------------ extras
gm = db.games
h2 = gm[(((gm.home_team == home) & (gm.away_team == away)) | ((gm.home_team == away) & (gm.away_team == home))) &
        gm.home_score.notna() & (gm.gameday < asof)].sort_values('gameday', ascending=False).head(6)
meet = [dict(date=str(r.gameday.date()), home=r.home_team, away=r.away_team, hs=int(r.home_score), as_=int(r.away_score))
        for r in h2.itertuples()]
samp = rng.choice(N, 3000, replace=False)
nT = len(M.TEAMS)
def rk(vec, t, rev):
    s = pd.Series(vec, index=M.TEAMS).rank(ascending=not rev, method='min')
    return int(s[t])
strength = {}
for t in (home, away):
    strength[t] = dict(off_td_rank=rk(fit['td'][2:2 + nT], t, True), def_td_rank=rk(fit['td'][2 + nT:], t, False),
                       off_fg_rank=rk(fit['fg'][2:2 + nT], t, True))
bt = json.load(open(os.path.join(M.WORK, 'bt_report.json')))
outj = dict(meta=G['meta'] | dict(home=home, away=away, n_sims=N, asof=str(asof.date()), season=G['season'], week=G['week']),
            market=mk, game=game, typical=typical, players=players, dvp=dvp_rows, meetings=meet,
            sample=[[int(a), int(b)] for a, b in pts[samp]], strength=strength, backtest=bt, weather=G.get('weather', {}),
            sources=G.get('sources', []), injuries=G.get('injuries', []),
            per_drive=dict(td=[float(x) for x in M.sigmoid(par['td'])], fg=[float(x) for x in M.sigmoid(par['fg'])],
                           rush_share=[float(x) for x in M.sigmoid(par['rush'])]))
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
out_path = G['out_json'] if os.path.isabs(G['out_json']) else os.path.join(ROOT, G['out_json'])
os.makedirs(os.path.dirname(out_path), exist_ok=True)
json.dump(outj, open(out_path, 'w'))
# ------------------------------------------------------------ console summary
print(f"{away} @ {home}: {home} win {p_home:.3f} | mean {pts[:,1].mean():.1f}-{pts[:,0].mean():.1f} | med margin {np.median(mg)} | med total {np.median(tot)} | sd {mg.std():.1f}")
print('market nv home', round(nv_home, 3), 'cover', round(p_home_cover, 3), 'over', round(p_over, 3))
print('typical', typical['score'], 'share', round(typical['share'], 4))
for e in events:
    print(' ', e)
for d in sorted(players, key=lambda x: -x['p_td']):
    mk_ = d.get('td_mkt')
    print(f"{d['name']:22s}{d['pos']:3s}{d['team']:4s} TD {d['p_td']:.3f} mkt {mk_['implied'] if mk_ else float('nan'):.3f} edge {d.get('td_edge', float('nan')):+.3f} | rec {d['rec']['mean']:5.1f} rush {d['rush']['mean']:5.1f} tgt% {d['tgt_share']:.2f} car% {d['car_share']:.2f} dvpRec {d['dvp_rec']['adj']:.2f} h2h {d['h2h_n']}")
for d in players:
    if d['is_qb']:
        print(d['name'], 'pass', round(d['pass']['mean'], 1), 'att', round(d['pass']['att'], 1), 'ptd', round(d['pass']['td'], 2))
print(json.dumps(dvp_rows[home][:6]))
print(json.dumps(dvp_rows[away][:6]))
