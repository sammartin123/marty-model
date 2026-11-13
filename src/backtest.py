"""Walk-forward backtest: every game is simulated using only data from before kickoff."""
import os, sys, json, time, numpy as np, pandas as pd
import model as M

def load_rosters(years):
    fr = []
    for y in years:
        r = pd.read_csv(os.path.join(M.DATA, f'roster_weekly_{y}.csv'), low_memory=False,
                        usecols=['season', 'team', 'position', 'status', 'gsis_id', 'week', 'game_type', 'full_name'])
        fr.append(r)
    r = pd.concat(fr)
    r['team'] = r.team.replace({'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA', 'LAR': 'LA'})
    return r[r.position.isin(['QB', 'RB', 'WR', 'TE', 'FB']) & r.gsis_id.notna()]

def run(seasons, N=3000, cfg=None, seed=7, verbose=True, max_week=99):
    cfg = cfg or M.CFG
    db = run.db
    ros = run.ros
    rng = np.random.default_rng(seed)
    g = db.games[db.games.season.isin(seasons) & (db.games.game_type == 'REG') & db.games.home_score.notna() &
                 (db.games.week <= max_week)].sort_values('gameday')
    fits = {}
    G, P = [], []
    for _, gm in g.iterrows():
        d = gm.gameday
        if d not in fits:
            fits = {d: M.fit_team(db, d, cfg)}
        fit = fits[d]
        par = M.team_params(fit, gm.home_team, gm.away_team, neutral=(gm.location == 'Neutral'))
        tabs, vols, ok = [], [], True
        for team, opp in [(gm.home_team, gm.away_team), (gm.away_team, gm.home_team)]:
            ids = ros[(ros.season == gm.season) & (ros.week == gm.week) & (ros.team == team) & (ros.status == 'ACT')].gsis_id
            act = db.pg[(db.pg.game_id == gm.game_id) & (db.pg.team == team)]
            qb = act.sort_values('pass_att', ascending=False).pid.iloc[0] if len(act) else None
            ids = set(ids) | ({qb} if qb else set())
            t = M.player_table(db, team, opp, d, ids, qb_id=qb, cfg=cfg)
            if t is None or len(t) < 4:
                ok = False; break
            tabs.append(t); vols.append(M.team_volume(db, team, d))
        if not ok:
            continue
        o = M.simulate(db, par, tabs, vols, N, rng, cfg)
        pts = o['pts']; mg = pts[:, 0] - pts[:, 1]
        G.append(dict(game_id=gm.game_id, season=gm.season, week=gm.week, home=gm.home_team, away=gm.away_team,
                      p_home=(mg > 0).mean() + 0.5 * (mg == 0).mean(), m_margin=mg.mean(), sd_margin=mg.std(),
                      m_total=pts.sum(1).mean(), sd_total=pts.sum(1).std(), m_home=pts[:, 0].mean(), m_away=pts[:, 1].mean(),
                      result=gm.result, total=gm.total, spread=gm.spread_line, total_line=gm.total_line,
                      ml_home=gm.home_moneyline, ml_away=gm.away_moneyline))
        for ti, team in enumerate([gm.home_team, gm.away_team]):
            t = tabs[ti]; s = o['stats'][ti]
            act = db.pg[(db.pg.game_id == gm.game_id) & (db.pg.team == team)].set_index('pid')
            td = (s['rtd'] + s['ctd']) > 0
            for i, pid in enumerate(t.pid.values):
                a = act.loc[pid] if pid in act.index else None
                P.append(dict(game_id=gm.game_id, season=gm.season, team=team, pid=pid, name=t.name[i], pos=t.pos[i],
                              is_qb=bool(t.is_qb[i]), p_td=td[:, i].mean(),
                              m_rec=s['ryds'][:, i].mean(), q10_rec=np.quantile(s['ryds'][:, i], .1), q90_rec=np.quantile(s['ryds'][:, i], .9),
                              med_rec=np.median(s['ryds'][:, i]),
                              m_rush=s['uyds'][:, i].mean(), q10_rush=np.quantile(s['uyds'][:, i], .1), q90_rush=np.quantile(s['uyds'][:, i], .9),
                              m_pass=s['pass_yds'].mean() if t.is_qb[i] else np.nan,
                              q10_pass=np.quantile(s['pass_yds'], .1) if t.is_qb[i] else np.nan,
                              q90_pass=np.quantile(s['pass_yds'], .9) if t.is_qb[i] else np.nan,
                              a_td=float((a.rush_td + a.rec_td) > 0) if a is not None else 0.0,
                              a_rec=float(a.rec_yds) if a is not None else 0.0,
                              a_rush=float(a.rush_yds) if a is not None else 0.0,
                              a_pass=float(a.pass_yds) if a is not None else np.nan,
                              played=a is not None,
                              base_rec=float(t.sh_rec[i]), tgt_share=float(t.tgt_share[i]), car_share=float(t.car_share[i])))
    return pd.DataFrame(G), pd.DataFrame(P)

def implied(ml):
    ml = np.asarray(ml, float)
    return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))

def report(G, P, label=''):
    G = G.copy()
    y = (G.result > 0).astype(float).values + 0.5 * (G.result == 0).values
    ph, pa = implied(G.ml_home), implied(G.ml_away)
    pm = ph / (ph + pa)
    out = {}
    out['games'] = len(G)
    out['brier_model'] = float(((G.p_home - y) ** 2).mean()); out['brier_market'] = float(((pm - y) ** 2).mean())
    out['acc_model'] = float(((G.p_home > .5) == (y > .5)).mean()); out['acc_market'] = float(((pm > .5) == (y > .5)).mean())
    out['mae_margin_model'] = float((G.m_margin - G.result).abs().mean()); out['mae_margin_market'] = float((G.spread - G.result).abs().mean())
    out['mae_total_model'] = float((G.m_total - G.total).abs().mean()); out['mae_total_market'] = float((G.total_line - G.total).abs().mean())
    out['rmse_margin'] = float(np.sqrt(((G.m_margin - G.result) ** 2).mean())); out['sim_sd_margin'] = float(G.sd_margin.mean())
    out['rmse_total'] = float(np.sqrt(((G.m_total - G.total) ** 2).mean())); out['sim_sd_total'] = float(G.sd_total.mean())
    out['bias_total'] = float((G.m_total - G.total).mean()); out['bias_margin'] = float((G.m_margin - G.result).mean())
    # against the spread when the model disagrees with the market
    for thr in [0, 2, 3]:
        diff = G.m_margin - G.spread
        pick = diff.abs() >= thr
        cover = np.sign(G.result - G.spread)
        w = ((np.sign(diff) == cover) & pick & (cover != 0)).sum(); l = ((np.sign(diff) == -cover) & pick & (cover != 0)).sum()
        out[f'ats_{thr}'] = [int(w), int(l)]
        dt = G.m_total - G.total_line
        pick = dt.abs() >= thr
        ou = np.sign(G.total - G.total_line)
        w = ((np.sign(dt) == ou) & pick & (ou != 0)).sum(); l = ((np.sign(dt) == -ou) & pick & (ou != 0)).sum()
        out[f'ou_{thr}'] = [int(w), int(l)]
    # win-probability calibration
    bins = [0, .3, .4, .5, .6, .7, 1.01]
    cal = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (G.p_home >= lo) & (G.p_home < hi)
        if m.sum():
            cal.append([round(float(G.p_home[m].mean()), 3), round(float(y[m.values].mean()), 3), int(m.sum())])
    out['cal_win'] = cal
    # anytime TD
    Q = P[~P.is_qb | True]
    out['td_n'] = len(Q); out['td_brier'] = float(((Q.p_td - Q.a_td) ** 2).mean())
    out['td_brier_base'] = float(((Q.a_td.mean() - Q.a_td) ** 2).mean())
    eps = 1e-4
    out['td_logloss'] = float(-(Q.a_td * np.log(Q.p_td.clip(eps, 1 - eps)) + (1 - Q.a_td) * np.log((1 - Q.p_td).clip(eps, 1 - eps))).mean())
    cal = []
    bins = [0, .05, .1, .15, .2, .3, .4, .5, .6, 1.01]
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (Q.p_td >= lo) & (Q.p_td < hi)
        if m.sum():
            cal.append([round(float(Q.p_td[m].mean()), 3), round(float(Q.a_td[m].mean()), 3), int(m.sum())])
    out['cal_td'] = cal
    # yardage (players who actually played, meaningful roles)
    R = P[P.played & (P.m_rec >= 20)]
    out['rec_n'] = len(R); out['rec_mae'] = float((R.m_rec - R.a_rec).abs().mean()); out['rec_bias'] = float((R.m_rec - R.a_rec).mean())
    out['rec_cov80'] = float(((R.a_rec >= R.q10_rec) & (R.a_rec <= R.q90_rec)).mean())
    out['rec_over_med'] = float((R.a_rec > R.med_rec).mean())
    U = P[P.played & (P.m_rush >= 20)]
    out['rush_n'] = len(U); out['rush_mae'] = float((U.m_rush - U.a_rush).abs().mean()); out['rush_bias'] = float((U.m_rush - U.a_rush).mean())
    out['rush_cov80'] = float(((U.a_rush >= U.q10_rush) & (U.a_rush <= U.q90_rush)).mean())
    Qb = P[P.is_qb & P.played & P.a_pass.notna()]
    out['pass_n'] = len(Qb); out['pass_mae'] = float((Qb.m_pass - Qb.a_pass).abs().mean()); out['pass_bias'] = float((Qb.m_pass - Qb.a_pass).mean())
    out['pass_cov80'] = float(((Qb.a_pass >= Qb.q10_pass) & (Qb.a_pass <= Qb.q90_pass)).mean())
    return out

if __name__ == '__main__':
    run.db = M.DB(); run.ros = load_rosters([2023, 2024, 2025, 2026])
    t = time.time()
    seasons = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [2024, 2025, 2026]
    cfg = dict(M.CFG)
    for kv in sys.argv[3:]:
        k, v = kv.split('='); cfg[k] = (v == 'True') if v in ('True', 'False') else float(v)
    G, P = run(seasons, N=int(sys.argv[2]) if len(sys.argv) > 2 else 3000, cfg=cfg)
    print('time', round(time.time() - t, 1))
    G.to_parquet(os.path.join(M.WORK, 'bt_games.parquet')); P.to_parquet(os.path.join(M.WORK, 'bt_players.parquet'))
    r = report(G, P)
    json.dump(r, open(os.path.join(M.WORK, 'bt_report.json'), 'w'))
    print(json.dumps({k:(round(v,4) if isinstance(v,float) else v) for k,v in r.items()}))
