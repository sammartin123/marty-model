"""Run MartyMakesMoneyModel for one game and write the JSON the page reads.
usage: python3 src/run_game.py games/<AWAY_HOME>.json"""
import os, sys, json, numpy as np, pandas as pd
import model as M

# Weight on the model when it is set against the book for a call. Across the first 87 graded props the
# book's probabilities were more accurate than the model's (TD log loss 0.514 vs 0.534; yardage 0.670 vs 0.822),
# so a call needs the model to disagree strongly. Revisit once the tracker has a few hundred graded props.
CALL_W = 0.4


_REAL = {}
def REAL_SCORES():
    """How often each (winner, loser) final score happened in real regular-season games since 2015."""
    if not _REAL:
        g = pd.read_csv(os.path.join(M.DATA, 'games.csv'))
        g = g[(g.season >= 2015) & g.home_score.notna() & (g.game_type == 'REG')]
        w = np.maximum(g.home_score, g.away_score).astype(int); l = np.minimum(g.home_score, g.away_score).astype(int)
        for a_, b_ in zip(w, l): _REAL[(a_, b_)] = _REAL.get((a_, b_), 0) + 1
        _REAL['_n'] = len(g)
    return _REAL

def run(G, db=None, quiet=False, record=True):
    N = G.get('n_sims', 100000)
    home, away = G['home'], G['away']
    asof = pd.Timestamp(G['asof'])
    cfg = dict(M.CFG)
    db = db or M.DB()
    rng = np.random.default_rng(G.get('seed', 20261115))

    # ------------------------------------------------------------ rosters
    ros = pd.read_csv(os.path.join(M.DATA, f"roster_weekly_{G['season']}.csv"), low_memory=False)
    ros['team'] = ros.team.replace({'LAR': 'LA'})
    wk = ros[ros.team.isin([home, away])].week.max()
    ros = ros[(ros.week == wk) & (ros.status == 'ACT') & ros.position.isin(['QB', 'RB', 'WR', 'TE', 'FB'])]
    out_ids = set(G.get('out', []))
    tabs, vols = [], []
    for team, opp in [(home, away), (away, home)]:
        # players activated from IR or elevated from the practice squad after the weekly roster file (slate 'add_names')
        ids = (set(ros[ros.team == team].gsis_id) | {pid for pid, tm in G.get('add', []) if tm == team}) - out_ids
        t = M.player_table(db, team, opp, asof, ids, qb_id=G['qb'][team], cfg=cfg)
        tabs.append(t); vols.append(M.team_volume(db, team, asof))

    fit = M.fit_team(db, asof, cfg)
    par = M.team_params(fit, home, away, neutral=bool(G.get('neutral')), qbs=(G['qb'][home], G['qb'][away]))
    for t, team in enumerate([home, away]):
        vols[t]['qb_delta'] = M.qb_delta(db, fit, team, G['qb'][team], asof)
    wx = G.get('wx_model', dict(pass_mult=1.0, td_shift=0.0, fg_shift=0.0, cr_mult=1.0))
    # Game-level numbers (winner, spread, total, the typical game) are the model's own view.
    o = M.simulate(db, par, tabs, vols, N, rng, cfg, wx=wx, keep=True)
    pts = o['pts']; mg = pts[:, 0] - pts[:, 1]; tot = pts.sum(1)
    # Player props: team scoring is matched to the sportsbook's implied team totals (the book predicts game
    # scores better than the model does), so the model only decides who gets the yards and touchdowns.
    mk0 = G.get('market') or {}
    anchor = None; op = o
    if cfg.get('anchor') and mk0.get('spread_home') is not None and mk0.get('total') is not None:
        th, ta = M.implied_totals(mk0['spread_home'], mk0['total'])
        par_a = M.anchor(db, par, th, ta, cfg, wx=wx)
        op = M.simulate(db, par_a, tabs, vols, N, np.random.default_rng(G.get('seed', 20261115) + 1), cfg, wx=wx, keep=True)
        anchor = dict(home=float(th), away=float(ta), sim_home=float(op['pts'][:, 0].mean()), sim_away=float(op['pts'][:, 1].mean()),
                      model_home=float(pts[:, 0].mean()), model_away=float(pts[:, 1].mean()))

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
    unmatched = None
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

    # ------------------------------------------------------------ projected game (box score + one replay)
    # Built from the model's own simulations (the same ones behind the win probability and the predicted
    # score). The final score is the closest realistic NFL score to the predicted score, with the model's
    # favourite winning, and the replay never hinges on a missed extra point.
    T0 = o
    TP = T0['pts']; tmg = TP[:, 0] - TP[:, 1]; ttot = TP.sum(1)
    th_, ta_ = float(TP[:, 0].mean()), float(TP[:, 1].mean())
    fav = 0 if p_home >= 0.5 else 1
    kd0 = T0['ev']['kind']; pt0 = T0['ev']['pat']
    clean0 = ((((kd0 == 1) | (kd0 == 3)) & (pt0 != 1)).sum(1) == 0)   # every extra point made, no two-point tries
    win = (TP[:, fav] > TP[:, 1 - fav]) & (~T0['ot']) & clean0
    pairs, cnt = np.unique(TP[win], axis=0, return_counts=True)
    dist_s = np.abs(pairs[:, 0] - th_) + np.abs(pairs[:, 1] - ta_)
    # Realism: an average like 23.4-22.4 is not a real final score. Prefer final scores that real NFL games
    # actually end on (regular season since 2015), so a one-point game built on a missed kick or an odd
    # number like 22 loses to a common nearby score such as 23-20. Each point of distance from the model's
    # predicted score costs 1; a final score that is e times rarer costs 1.5.
    fh = REAL_SCORES()
    rare = np.array([-np.log((fh.get((max(a_, b_), min(a_, b_)), 0) + 0.5) / fh['_n']) for a_, b_ in pairs])
    ok = cnt >= 3                                # the replay needs a few games at that score to choose from
    if not ok.any(): ok = cnt >= 1
    cost = np.where(ok, dist_s + 1.5 * rare, 1e9)
    best = pairs[int(cost.argmin())]

    # projections every replay is measured against (from the full first batch)
    ref = []
    for t in (0, 1):
        s = T0['stats'][t]; df = tabs[t]
        tdj = np.minimum(s['rtd'] + s['ctd'], 4).astype(int)
        pk = np.stack([(tdj == k).mean(0) for k in range(5)])
        lp = -np.log(np.clip(pk, 1e-4, 1))
        items = [('pass_yds', None, s['pass_yds'])]
        for j in range(len(df)):
            if s['ryds'][:, j].mean() >= 8: items.append(('ryds', j, s['ryds'][:, j])); items.append(('recs', j, s['recs'][:, j]))
            if s['uyds'][:, j].mean() >= 8: items.append(('uyds', j, s['uyds'][:, j])); items.append(('cars', j, s['cars'][:, j]))
        sc_ = []
        for key, j, arr in items:
            q25, q50, q75 = np.quantile(arr, [.25, .5, .75])
            sc_.append((key, j, q50, max(q75 - q25, 3.0 if key in ('recs', 'cars') else 15.0)))
        ntd = (s['rtd'] + s['ctd']).sum(1)
        vals, cts = np.unique(ntd[(TP[:, 0] == best[0]) & (TP[:, 1] == best[1]) & clean0], return_counts=True)
        ref.append(dict(items=sc_, lp=lp, ntd=int(vals[cts.argmax()]) if len(vals) else None))

    def pick(T):
        P_ = T['pts']
        idx = np.where((P_[:, 0] == best[0]) & (P_[:, 1] == best[1]) & (~T['ot']))[0]
        # realism: the replay never hinges on a missed extra point or a two-point try
        kd = T['ev']['kind'][idx]; pt = T['ev']['pat'][idx]
        odd = (((kd == 1) | (kd == 3)) & (pt != 1)).sum(1)
        if (odd == 0).any(): idx = idx[odd == 0]; odd = odd[odd == 0]
        for t in (0, 1):
            s = T['stats'][t]
            if ref[t]['ntd'] is not None:
                keep_ = (s['rtd'][idx] + s['ctd'][idx]).sum(1) == ref[t]['ntd']
                if keep_.any(): idx = idx[keep_]
        if not len(idx):
            return np.inf, None
        dist = np.zeros(len(idx))
        kd = T['ev']['kind'][idx]; pt = T['ev']['pat'][idx]
        dist += 1000.0 * (((kd == 1) | (kd == 3)) & (pt != 1)).sum(1)
        for t in (0, 1):
            s = T['stats'][t]
            for key, j, med, scale in ref[t]['items']:
                x = s[key][idx] if j is None else s[key][idx, j]
                dist += np.abs(x - med) / scale
            tdj = np.minimum(s['rtd'][idx] + s['ctd'][idx], 4).astype(int)
            lp = ref[t]['lp']
            dist += 2.5 * (lp[tdj, np.arange(tdj.shape[1])] - lp.min(0)).sum(1)
            dist += 50 * ((s['ctd'][idx] > s['recs'][idx]).any(1) | (s['rtd'][idx] > s['cars'][idx]).any(1))
        dist += 3.0 * (T['ev']['kind'][idx] == 3).sum(1)
        k = int(dist.argmin())
        return float(dist[k]), int(idx[k])

    # look through extra batches so the replay can match the projections closely
    extra = int(G.get('typical_batches', 2))
    best_d, i = pick(T0); T = T0; pool = int(((TP[:, 0] == best[0]) & (TP[:, 1] == best[1])).sum())
    for b_ in range(extra):
        Tb = M.simulate(db, par, tabs, vols, N, np.random.default_rng(G.get('seed', 20261115) + 11 + b_), cfg, wx=wx, keep=True)
        d_, i_ = pick(Tb); pool += int(((Tb['pts'][:, 0] == best[0]) & (Tb['pts'][:, 1] == best[1])).sum())
        if i_ is not None and d_ < best_d:
            best_d, i, T = d_, i_, Tb
        else:
            del Tb
    pts_t = T['pts']

    sc = pd.read_parquet(os.path.join(M.WORK, 'scores.parquet'))
    sc = sc[sc.season >= 2021]
    rush_len = sc[(sc.rush_touchdown == 1)].yards_gained.dropna().values
    pass_len = sc[(sc.pass_touchdown == 1)].yards_gained.dropna().values
    fg_len = sc[(sc.field_goal_result == 'made')].kick_distance.dropna().values
    trng = np.random.default_rng(i)
    ev = T['ev']
    teams = [home, away]
    scorers = []
    for t in (0, 1):
        s = T['stats'][t]; df = tabs[t]
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
    assert score[0] == pts_t[i, 0] and score[1] == pts_t[i, 1], (score, pts_t[i])

    # projected box score: every number is from all of the first batch's simulations
    def line_for(name, stat):
        for L in (G.get('yard_lines') or {}).get(name, []) if isinstance(G.get('yard_lines'), dict) else []:
            if L.get('stat') == stat: return L.get('line')
        return None
    def proj_box(t):
        s = T0['stats'][t]; df = tabs[t]
        qbj = int(np.where(df.is_qb.values)[0][0]); nmq = df.name[qbj]
        md = lambda a: int(round(float(np.median(a))))
        passing = [dict(name=nmq, comp=round(float(s['comp'].mean()), 1), att=round(float(s['PA'].mean()), 1), yds=md(s['pass_yds']),
                        td=round(float(s['pass_td'].mean()), 1), ints=round(float(s['ints'].mean()), 1), line=line_for(nmq, 'pass'))]
        rushing, receiving = [], []
        for j in np.argsort(-s['cars'].mean(0)):
            if s['cars'][:, j].mean() >= 1.5:
                rushing.append(dict(name=df.name[j], car=round(float(s['cars'][:, j].mean()), 1), yds=md(s['uyds'][:, j]),
                                    td=round(float((s['rtd'][:, j] > 0).mean()), 3), line=line_for(df.name[j], 'rush')))
        for j in np.argsort(-np.median(s['ryds'], 0) - 0.01 * s['ryds'].mean(0)):
            if s['tgts'][:, j].mean() >= 1.5:
                receiving.append(dict(name=df.name[j], rec=round(float(s['recs'][:, j].mean()), 1), tgt=round(float(s['tgts'][:, j].mean()), 1),
                                      yds=md(s['ryds'][:, j]), td=round(float((s['ctd'][:, j] > 0).mean()), 3), line=line_for(df.name[j], 'rec')))
        # anyone who scores in the replay is always listed, so the TD badge has a row to sit on
        sc_r = {e['player'] for e in events if e.get('team') == teams[t] and e['type'] == 'TD' and e.get('how') == 'rush'}
        sc_c = {e['player'] for e in events if e.get('team') == teams[t] and e['type'] == 'TD' and e.get('how') == 'rec'}
        rush_out = rushing[:4] + [r for r in rushing[4:] if r['name'] in sc_r]
        rec_out = receiving[:6] + [r for r in receiving[6:] if r['name'] in sc_c]
        for nm_ in sc_r - {r['name'] for r in rush_out}:
            j = int(np.where(df.name.values == nm_)[0][0])
            rush_out.append(dict(name=nm_, car=round(float(s['cars'][:, j].mean()), 1), yds=md(s['uyds'][:, j]),
                                 td=round(float((s['rtd'][:, j] > 0).mean()), 3), line=line_for(nm_, 'rush')))
        for nm_ in sc_c - {r['name'] for r in rec_out}:
            j = int(np.where(df.name.values == nm_)[0][0])
            rec_out.append(dict(name=nm_, rec=round(float(s['recs'][:, j].mean()), 1), tgt=round(float(s['tgts'][:, j].mean()), 1),
                                yds=md(s['ryds'][:, j]), td=round(float((s['ctd'][:, j] > 0).mean()), 3), line=line_for(nm_, 'rec')))
        return dict(passing=passing, rushing=rush_out, receiving=rec_out,
                    pts=round(float(TP[:, t].mean()), 1))
    typical = dict(score={home: int(pts_t[i, 0]), away: int(pts_t[i, 1])}, events=events,
                   box={home: proj_box(0), away: proj_box(1)},
                   share=float((TP == best).all(1).mean()), pool=pool, n_searched=int(N * (1 + extra)),
                   anchored=False, target={home: round(th_, 1), away: round(ta_, 1)}, proj_margin=float(np.median(tmg)), proj_total=float(np.median(ttot)))

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
    def norm(n):
        n = n.lower().replace('.', '').replace("'", '').replace('-', ' ')
        return ' '.join(w for w in n.split() if w not in ('jr', 'sr', 'ii', 'iii', 'iv'))
    odds = {norm(k): v for k, v in G.get('td_odds', {}).items()}
    ylines = {norm(k): v for k, v in G.get('yard_lines', {}).items()}
    matched = set()
    pg = db.pg
    players = []
    for t, (team, opp) in enumerate([(home, away), (away, home)]):
        s = op['stats'][t]; df = tabs[t]
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
            d = dict(id=pid, name=df.name[j], pos=pos, team=team, opp=opp, is_qb=bool(df.is_qb[j]), status=G.get('status', {}).get(pid), low_conf=team in G.get('inputs', {}).get('low_conf', {}),
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
            if norm(df.name[j]) in odds:
                od = odds[norm(df.name[j])]; matched.add(norm(df.name[j]))
                d['td_mkt'] = dict(price=od['price'], book=od['book'], implied=implied(od['price']), all=od.get('all', []))
                d['td_edge'] = p_td - implied(od['price'])
                # calls use a blend of model and book (see CALL_W): the book has been the more accurate of the two
                d['p_td_call'] = CALL_W * p_td + (1 - CALL_W) * implied(od['price'])
                d['call_edge'] = d['p_td_call'] - implied(od['price'])
            if norm(df.name[j]) in ylines:
                d['lines'] = []
                matched.add(norm(df.name[j]))
                for L in ylines[norm(df.name[j])]:
                    arr = {'rec': s['ryds'][:, j], 'rush': s['uyds'][:, j], 'pass': s['pass_yds'], 'recs': s['recs'][:, j]}[L['stat']]
                    po = float((arr > L['line']).mean())
                    e = dict(L); e['p_over'] = po
                    if 'over' in L:
                        e['imp_over'] = implied(L['over'])
                    if 'under' in L:
                        e['imp_under'] = implied(L['under'])
                    # book's own over chance with the vig removed (0.5 when only one side is quoted)
                    m = e['imp_over'] / (e['imp_over'] + e['imp_under']) if ('imp_over' in e and 'imp_under' in e) else 0.5
                    e['p_mkt_over'] = m
                    e['p_call_over'] = CALL_W * po + (1 - CALL_W) * m
                    d['lines'].append(e)
            players.append(d)

    # ------------------------------------------------------------ correlated yardage calls
    # Overs (or unders) on one team's passing game all need the same game script, so only the strongest one
    # in each group is a call; the rest are marked 'Correlated'. Rushing is its own group.
    groups = {}
    for d in players:
        for e in d.get('lines', []):
            pc = e['p_call_over']; side = 'over' if pc >= .5 else 'under'
            price = e.get(side)
            edge = max(pc, 1 - pc) - (implied(price) if price is not None else .524)
            fam = 'rush' if e['stat'] == 'rush' else 'pass'
            groups.setdefault((d['team'], fam, side), []).append((edge, e))
    for g_ in groups.values():
        g_.sort(key=lambda x: -x[0])
        for edge, e in g_[1:]:
            e['corr'] = True

    # ------------------------------------------------------------ extras
    gm = db.games
    h2 = gm[(((gm.home_team == home) & (gm.away_team == away)) | ((gm.home_team == away) & (gm.away_team == home))) &
            gm.home_score.notna() & (gm.gameday < asof)].sort_values('gameday', ascending=False).head(6)
    meet = [dict(date=str(r.gameday.date()), home=r.home_team, away=r.away_team, hs=int(r.home_score), as_=int(r.away_score))
            for r in h2.itertuples()]
    samp = rng.choice(N, 1500, replace=False)
    nT = len(M.TEAMS)
    def rk(vec, t, rev):
        s = pd.Series(vec, index=M.TEAMS).rank(ascending=not rev, method='min')
        return int(s[t])
    strength = {}
    for t in (home, away):
        strength[t] = dict(off_td_rank=rk(fit['td'][2:2 + nT], t, True), def_td_rank=rk(fit['td'][2 + nT:2 + 2 * nT], t, False),
                           off_fg_rank=rk(fit['fg'][2:2 + nT], t, True))
    qb_info = {}
    if 'qb_index' in fit:
        qi, base = fit['qb_index'], fit['qb_base']
        for t, team in enumerate([home, away]):
            q_ = G['qb'][team]
            qb_info[team] = dict(effect=float(fit['td'][base + qi.get(q_, len(qi))]), own=bool(q_ in qi),
                                 delta=float(vols[t]['qb_delta']))
    bt = json.load(open(os.path.join(M.WORK, 'bt_report.json')))
    unmatched = sorted((set(odds) | set(ylines)) - matched)
    outj = dict(meta=G['meta'] | dict(home=home, away=away, n_sims=N, asof=str(asof.date()), season=G['season'], week=G['week']),
                market=mk, game=game, typical=typical, players=players, dvp=dvp_rows, meetings=meet,
                sample=[[int(a), int(b)] for a, b in pts[samp]], strength=strength, backtest=bt, weather=G.get('weather', {}),
                sources=G.get('sources', []), injuries=G.get('injuries', []), inputs=dict(G.get('inputs', {}), unmatched=unmatched),
                per_drive=dict(td=[float(x) for x in M.sigmoid(par['td'])], fg=[float(x) for x in M.sigmoid(par['fg'])],
                               rush_share=[float(x) for x in M.sigmoid(par['rush'])]),
                anchor=anchor, qb_model=qb_info)
    import ledger
    outj['call_w'] = CALL_W
    ledger.log_game(G, outj)
    if record:
        outj['record'] = ledger.grade(db)
    ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    out_path = G['out_json'] if os.path.isabs(G['out_json']) else os.path.join(ROOT, G['out_json'])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(outj, open(out_path, 'w'))
    if not quiet:
        # ------------------------------------------------------------ console summary
        print(f"{away} @ {home}: {home} win {p_home:.3f} | mean {pts[:,1].mean():.1f}-{pts[:,0].mean():.1f} | med margin {np.median(mg)} | med total {np.median(tot)} | sd {mg.std():.1f}")
        print('market nv home', round(nv_home, 3), 'cover', round(p_home_cover, 3), 'over', round(p_over, 3))
        print('typical', typical['score'], 'share', round(typical['share'], 4))
        print('props anchored to', anchor, '| qb', qb_info)
        for e in events:
            print(' ', e)
        for d in sorted(players, key=lambda x: -x['p_td']):
            mk_ = d.get('td_mkt')
            print(f"{d['name']:22s}{d['pos']:3s}{d['team']:4s} TD {d['p_td']:.3f} mkt {mk_['implied'] if mk_ else float('nan'):.3f} edge {d.get('td_edge', float('nan')):+.3f} | rec {d['rec']['mean']:5.1f} rush {d['rush']['mean']:5.1f} tgt% {d['tgt_share']:.2f} car% {d['car_share']:.2f} dvpRec {d['dvp_rec']['adj']:.2f} h2h {d['h2h_n']}")
        for d in players:
            if d['is_qb']:
                print(d['name'], 'pass', round(d['pass']['mean'], 1), 'att', round(d['pass']['att'], 1), 'ptd', round(d['pass']['td'], 2))
    return outj


if __name__ == '__main__':
    run(json.load(open(sys.argv[1])))
    print('RESULTS TRACKER UPDATED: commit everything with git add -A (ledger/ included) and push.')
