"""MartyMakesMoneyModel core: team strength, player usage, and the Monte Carlo game simulator."""
import os, numpy as np, pandas as pd
from scipy.optimize import nnls

WORK = os.path.join(os.path.dirname(__file__), '..', 'work')
DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
TEAMS = ['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
         'LA', 'LAC', 'LV', 'MIA', 'MIN', 'NE', 'NO', 'NYG', 'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WAS']
TI = {t: i for i, t in enumerate(TEAMS)}

CFG = dict(
    team_halflife_days=150.0,   # recency weighting of drive results
    team_window_days=520,
    ridge=6.0,                  # shrinkage on team effects (TD/FG models)
    s_game=0.12,                # game-day random effect on drive TD logit (per team)
    s_common=0.15,              # shared game effect (pace / shootout vs slog)
    cut_fg=0.30, dur_scale=1.0, td_delta=0.0,
    player_hl=4.0,              # half-life (games) for usage shares
    eff_hl=10.0,                # half-life (games) for per-touch efficiency
    team_hl=6.0,                # half-life (games) for team volume and defence-vs-position
    dvp_k_yds=10.0, dvp_k_td=20.0,
    dir_tgt=80.0, dir_car=22.0,  # Dirichlet concentration: game-to-game share noise
    k_rec=1.8, k_rush_shift=3.0, sd_carry=6.0,
    sig_eff_pass=0.09, sig_eff_rush=0.11,
    use_dvp=True, use_h2h=True, cr_cal=0.975, rush_cal=0.99,
)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def logit(p):
    return np.log(p / (1 - p))

# ------------------------------------------------------------------ data
class DB:
    def __init__(self):
        self.drives = pd.read_parquet(os.path.join(WORK, 'drives.parquet'))
        self.pg = pd.read_parquet(os.path.join(WORK, 'player_games.parquet'))
        self.tg = pd.read_parquet(os.path.join(WORK, 'team_games.parquet'))
        self.games = pd.read_csv(os.path.join(DATA, 'games.csv'))
        for c in ['home_team', 'away_team']:
            self.games[c] = self.games[c].replace({'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA'})
        self.games['gameday'] = pd.to_datetime(self.games['gameday'])
        self._prep_team_drive()
        self._prep_player_states()
        self._prep_team_states()
        self._fit_alloc()
        self._fit_volume()
        self._league_consts()
        self._fit_script()

    # ---- drive-level aggregates per team-game
    def _prep_team_drive(self):
        d = self.drives
        d = d.assign(c=1 - d.last_in_half, ctd=d.td * (1 - d.last_in_half), cfg=d.fg * (1 - d.last_in_half))
        a = d.groupby(['game_id', 'posteam', 'defteam', 'season', 'week', 'game_date', 'is_home']).agg(
            n_all=('td', 'size'), n=('c', 'sum'), td=('ctd', 'sum'), fg=('cfg', 'sum'), rtd=('rush_td', 'sum'), fg_all=('fg', 'sum'),
            td_all=('td', 'sum'), opp=('opp_td', 'sum')).reset_index()
        a['o'] = a.posteam.map(TI); a['d'] = a.defteam.map(TI)
        self.tdrive = a.dropna(subset=['o', 'd']).copy()
        self.tdrive[['o', 'd']] = self.tdrive[['o', 'd']].astype(int)

    # ---- exponentially weighted player usage/efficiency states
    def _prep_player_states(self):
        pg = self.pg.sort_values(['game_date', 'game_id']).reset_index(drop=True)
        team_tot = pg.groupby(['game_id', 'team']).agg(
            T_car=('carries', 'sum'), T_in5=('car_in5', 'sum'), T_in10=('car_in10', 'sum'),
            T_tgt=('targets', 'sum'), T_rz=('tgt_rz', 'sum'), T_in10t=('tgt_in10', 'sum'),
            T_ez=('tgt_ez', 'sum'), T_air=('air', 'sum'), T_rtd=('rush_td', 'sum'),
            T_ctd=('rec_td', 'sum')).reset_index()
        pg = pg.merge(team_tot, on=['game_id', 'team'], how='left')
        use = ['carries', 'car_in5', 'car_in10', 'targets', 'tgt_rz', 'tgt_in10', 'tgt_ez', 'air',
               'T_car', 'T_in5', 'T_in10', 'T_tgt', 'T_rz', 'T_in10t', 'T_ez', 'T_air', 'pass_att']
        eff = ['rec', 'rec_yds', 'rush_yds']
        g = pg.groupby('pid', sort=False)
        a1 = 1 - 0.5 ** (1 / CFG['player_hl']); a2 = 1 - 0.5 ** (1 / CFG['eff_hl'])
        e1 = g[use].transform(lambda s: s.ewm(alpha=a1, adjust=True).mean())
        e1.columns = ['u_' + c for c in use]
        e2 = g[eff + ['targets', 'carries']].transform(lambda s: s.ewm(alpha=a2, adjust=True).mean())
        e2.columns = ['e_' + c for c in eff + ['targets', 'carries']]
        n = g.cumcount() + 1
        pg = pd.concat([pg, e1, e2], axis=1)
        pg['W2'] = (1 - (1 - a2) ** n) / a2
        pg['gp'] = n
        post = [c for c in pg.columns if c.startswith('u_') or c.startswith('e_')] + ['W2', 'gp']
        prior = pg.groupby('pid', sort=False)[post].shift(1)
        prior.columns = ['p_' + c for c in post]
        self.pg = pd.concat([pg, prior], axis=1)
        self.post_cols = post
        # positional priors for efficiency
        r = self.pg[self.pg.season >= 2021]
        self.pri = {}
        for pos in ['QB', 'RB', 'WR', 'TE', 'OTH']:
            x = r[r.pos == pos]
            self.pri[pos] = dict(cr=x.rec.sum() / max(x.targets.sum(), 1), ypr=x.rec_yds.sum() / max(x.rec.sum(), 1),
                                 ypc=x.rush_yds.sum() / max(x.carries.sum(), 1))
        self.pri['OTH'] = self.pri['WR']

    # ---- team volume states and defence-vs-position states
    def _prep_team_states(self):
        pg = self.pg
        k = ['game_id', 'team', 'opp', 'game_date', 'season', 'week']
        pos = pg.pos.where(pg.pos.isin(['QB', 'RB', 'WR', 'TE']), 'WR')
        parts = {}
        for P in ['WR', 'TE', 'RB']:
            m = pos == P
            parts[f'recyds_{P}'] = pg.rec_yds.where(m, 0)
            parts[f'rectd_{P}'] = pg.rec_td.where(m, 0)
            parts[f'tgt_{P}'] = pg.targets.where(m, 0)
            parts[f'rec_{P}'] = pg.rec.where(m, 0)
        parts['rushyds_RB'] = pg.rush_yds.where(pos == 'RB', 0)
        parts['rushyds_QB'] = pg.rush_yds.where(pos == 'QB', 0)
        parts['rushtd_RB'] = pg.rush_td.where(pos == 'RB', 0)
        parts['rushtd_QB'] = pg.rush_td.where(pos == 'QB', 0)
        parts['passyds'] = pg.pass_yds
        df = pd.concat([pg[k], pd.DataFrame(parts)], axis=1)
        tp = df.groupby(k).sum().reset_index().sort_values(['game_date', 'game_id']).reset_index(drop=True)
        self.dvp_stats = list(parts.keys())
        a = 1 - 0.5 ** (1 / CFG['team_hl'])
        # offence expectation = that offence's prior EW mean
        off_prior = tp.groupby('team')[self.dvp_stats].transform(
            lambda s: s.ewm(alpha=a, adjust=True).mean().shift(1))
        off_prior.columns = ['x_' + c for c in self.dvp_stats]
        tp = pd.concat([tp, off_prior], axis=1)
        self.dvp_league = tp[tp.season >= 2021][self.dvp_stats].mean().to_dict()
        for c in self.dvp_stats:
            tp['x_' + c] = tp['x_' + c].fillna(self.dvp_league[c])
        gd = tp.groupby('opp', sort=False)
        da = gd[self.dvp_stats].transform(lambda s: s.ewm(alpha=a, adjust=True).mean())
        da.columns = ['da_' + c for c in self.dvp_stats]
        de = gd[['x_' + c for c in self.dvp_stats]].transform(lambda s: s.ewm(alpha=a, adjust=True).mean())
        de.columns = ['de_' + c for c in self.dvp_stats]
        n = gd.cumcount() + 1
        tp = pd.concat([tp, da, de], axis=1)
        tp['dW'] = (1 - (1 - a) ** n) / a
        self.tpos = tp
        # team volume states
        tg = self.tg.sort_values(['game_date', 'game_id']).reset_index(drop=True)
        gt = tg.groupby('team', sort=False)
        for c in ['pass_att', 'rush_att', 'margin', 'drives']:
            tg['ew_' + c] = gt[c].transform(lambda s: s.ewm(alpha=a, adjust=True).mean())
            tg['pew_' + c] = tg.groupby('team')['ew_' + c].shift(1)
        go = tg.groupby('opp', sort=False)
        for c in ['drives']:
            tg['dew_' + c] = go[c].transform(lambda s: s.ewm(alpha=a, adjust=True).mean())
            tg['pdew_' + c] = tg.groupby('opp')['dew_' + c].shift(1)
        self.tg = tg

    # ---- which usage shares predict who scores (trained on history)
    def _alloc_frame(self, lo, hi):
        pg = self.pg[(self.pg.season >= lo) & (self.pg.season <= hi) & self.pg.p_gp.notna()].copy()
        return pg

    def _fit_alloc(self, lo=2015, hi=2025):
        pg = self._alloc_frame(lo, hi)
        self.beta = {}
        for kind, feats, tot, y in [
            ('rush', [('p_u_carries', 'p_u_T_car'), ('p_u_car_in10', 'p_u_T_in10'), ('p_u_car_in5', 'p_u_T_in5')],
             'T_rtd', 'rush_td'),
            ('rec', [('p_u_targets', 'p_u_T_tgt'), ('p_u_air', 'p_u_T_air'), ('p_u_tgt_rz', 'p_u_T_rz'),
                     ('p_u_tgt_in10', 'p_u_T_in10t'), ('p_u_tgt_ez', 'p_u_T_ez')], 'T_ctd', 'rec_td')]:
            X = []
            grp = pg.groupby(['game_id', 'team'])
            for num, den in feats:
                s = (pg[num] / pg[den].clip(lower=0.25)).clip(0, 1.5)
                tot_s = s.groupby([pg.game_id, pg.team]).transform('sum').clip(lower=1e-6)
                X.append((s / tot_s).values)
            X.append((1.0 / grp[y].transform('size')).values)
            X = np.array(X).T
            T = pg[tot].values
            m = T > 0
            b, _ = nnls(X[m] * T[m][:, None], pg[y].values[m])
            self.beta[kind] = b / b.sum()

    # ---- pass/rush volume vs game script
    def _fit_volume(self):
        tg = self.tg[(self.tg.season >= 2016) & (self.tg.season <= 2025) & self.tg.pew_pass_att.notna()].copy()
        self.vol = {}
        res = {}
        for c in ['pass_att', 'rush_att']:
            X = np.column_stack([np.ones(len(tg)), tg['pew_' + c], tg.margin - tg.pew_margin])
            b, *_ = np.linalg.lstsq(X, tg[c].values, rcond=None)
            r = tg[c].values - X @ b
            self.vol[c] = dict(b=b, sd=r.std())
            res[c] = r
        self.vol['corr'] = float(np.corrcoef(res['pass_att'], res['rush_att'])[0, 1])
        self.tgt_frac = float(self.tg[self.tg.season >= 2021].targets.sum() / self.tg[self.tg.season >= 2021].pass_att.sum())

    def _league_consts(self):
        g = self.games[(self.games.season >= 2021) & self.games.home_score.notna()]
        td = self.tdrive[self.tdrive.season >= 2021]
        tg = self.tg[self.tg.season >= 2021]
        pts = (g.home_score.sum() + g.away_score.sum())
        off_pts = 6.96 * td.td_all.sum() + 3 * td.fg_all.sum()
        # everything that is not an offensive TD or FG (returns, safeties) expressed as TD-equivalents per drive
        self.p_other = max((pts - off_pts) / 6.96 / td.n_all.sum(), 0.004)
        self.int_rate = float(tg.ints.sum() / tg.pass_att.sum())
        self.mean_drives = float(td.n_all.mean())
        d = self.drives[(self.drives.season >= 2018) & (self.drives.last_in_half == 0) & (self.drives.dur > 0)]
        kind = np.where(d.td == 1, 1, np.where(d.fg == 1, 2, np.where(d.opp_td == 1, 3, 0)))
        self.dur_mean = np.array([d.dur[kind == k].mean() for k in range(4)])
        self.dur_sd = np.array([d.dur[kind == k].std() for k in range(4)])

    BUCKETS = np.array([-16.5, -8.5, -0.5, 0.5, 8.5, 16.5])

    def _fit_script(self):
        """How drive success changes with the scoreboard (trailing teams press, leading teams sit)."""
        d = self.drives[(self.drives.season >= 2016) & (self.drives.season <= 2025) & (self.drives.last_in_half == 0) &
                        self.drives.half.isin(['Half1', 'Half2']) & self.drives.diff0.notna()].copy()
        g = self.games.set_index('game_id')
        sp = g.spread_line.reindex(d.game_id).values; tl = g.total_line.reindex(d.game_id).values
        d['exp_pts'] = tl / 2 + np.where(d.is_home == 1, sp, -sp) / 2
        d = d[np.isfinite(d.exp_pts)]
        b = np.digitize(d.diff0.values, self.BUCKETS)
        h = (d.half == 'Half2').astype(int).values
        cell = h * 7 + b
        X = np.zeros((len(d), 2 + 14)); X[:, 0] = 1; X[:, 1] = (d.exp_pts.values - 22.5) / 5
        X[np.arange(len(d)), 2 + cell] = 1
        pen = np.r_[0, 0, np.full(14, 1.0)]
        one = np.ones(len(d))
        freq = np.bincount(cell, minlength=14) / len(d)
        bt = ridge_binom(X, d.td.values.astype(float), one, one, pen)
        m = d.td.values == 0
        bf = ridge_binom(X[m], d.fg.values[m].astype(float), one[m], one[m], pen)
        fq_f = np.bincount(cell[m], minlength=14) / m.sum()
        self.rb_td = (bt[2:] - (bt[2:] * freq).sum()).reshape(2, 7)
        self.rb_fg = (bf[2:] - (bf[2:] * fq_f).sum()).reshape(2, 7)


# ------------------------------------------------------------------ team strength (ridge binomial)
def ridge_binom(X, y, n, w, pen, iters=30):
    b = np.zeros(X.shape[1])
    p0 = (w * y).sum() / (w * n).sum()
    b[0] = logit(p0)
    P = np.diag(pen)
    for _ in range(iters):
        eta = X @ b
        mu = sigmoid(eta)
        W = w * n * mu * (1 - mu)
        grad = X.T @ (w * (y - n * mu)) - P @ b
        H = (X * W[:, None]).T @ X + P
        step = np.linalg.solve(H, grad)
        b += step
        if np.abs(step).max() < 1e-7:
            break
    return b

def fit_team(db, asof, cfg=CFG):
    """Opponent-adjusted offence/defence effects as of a date (uses only earlier games)."""
    asof = pd.Timestamp(asof)
    a = db.tdrive[(db.tdrive.game_date < asof) & (db.tdrive.game_date >= asof - pd.Timedelta(days=cfg['team_window_days']))]
    days = (asof - a.game_date).dt.days.values
    w = 0.5 ** (days / cfg['team_halflife_days'])
    nT = len(TEAMS)
    X = np.zeros((len(a), 2 + 2 * nT))
    X[:, 0] = 1; X[:, 1] = a.is_home.values
    X[np.arange(len(a)), 2 + a.o.values] = 1
    X[np.arange(len(a)), 2 + nT + a.d.values] = 1
    pen = np.r_[0, 0, np.full(2 * nT, cfg['ridge'])]
    out = {}
    out['td'] = ridge_binom(X, a.td.values, a.n.values, w, pen)
    out['fg'] = ridge_binom(X, a.fg.values, (a.n - a.td).values, w, pen)
    m = a.td_all.values > 0
    out['rush'] = ridge_binom(X[m], a.rtd.values[m], a.td_all.values[m], w[m], np.r_[0, 0, np.full(2 * nT, cfg['ridge'] * 0.6)])
    # pace: drives per game, ridge linear
    yv = a.n_all.values.astype(float)
    mu = (w * yv).sum() / w.sum()
    Xd = X[:, 2:]
    A = (Xd * w[:, None]).T @ Xd + np.eye(2 * nT) * cfg['ridge'] * 4
    bd = np.linalg.solve(A, Xd.T @ (w * (yv - mu)))
    out['drives'] = (mu, bd)
    out['asof'] = asof
    return out

def team_params(fit, home, away, neutral=False):
    nT = len(TEAMS)
    h, a = TI[home], TI[away]
    par = {}
    for k in ['td', 'fg', 'rush']:
        b = fit[k]
        hfa = 0.0 if neutral else b[1]
        par[k] = np.array([b[0] + hfa + b[2 + h] + b[2 + nT + a], b[0] + b[2 + a] + b[2 + nT + h]])
    mu, bd = fit['drives']
    par['mu_drives'] = mu + 0.5 * (bd[h] + bd[nT + a] + bd[a] + bd[nT + h])
    return par


# ------------------------------------------------------------------ player usage as of a date
RUSH_F = [('u_carries', 'u_T_car'), ('u_car_in10', 'u_T_in10'), ('u_car_in5', 'u_T_in5')]
REC_F = [('u_targets', 'u_T_tgt'), ('u_air', 'u_T_air'), ('u_tgt_rz', 'u_T_rz'), ('u_tgt_in10', 'u_T_in10t'),
         ('u_tgt_ez', 'u_T_ez')]

def dvp_mult(db, defteam, asof, cfg=CFG):
    t = db.tpos[(db.tpos.opp == defteam) & (db.tpos.game_date < asof)]
    out = {}
    if len(t) == 0:
        return {c: 1.0 for c in db.dvp_stats}
    r = t.iloc[-1]
    for c in db.dvp_stats:
        k = cfg['dvp_k_td'] if 'td' in c else cfg['dvp_k_yds']
        L = db.dvp_league[c]
        out[c] = float((r.dW * r['da_' + c] + k * L) / (r.dW * r['de_' + c] + k * L))
    return out

def player_table(db, team, opp, asof, ids, qb_id=None, cfg=CFG, h2h=True):
    """Usage shares and efficiency for the active skill players of `team` as of a date."""
    asof = pd.Timestamp(asof)
    pg = db.pg
    hist = pg[(pg.game_date < asof) & pg.pid.isin(ids)]
    last = hist.groupby('pid', sort=False).tail(1).copy()
    last = last[(asof - last.game_date).dt.days <= 420]
    if qb_id is None:
        q = last[last.pos == 'QB'].sort_values('u_pass_att', ascending=False)
        qb_id = q.pid.iloc[0] if len(q) else None
    last = last[(last.pos != 'QB') | (last.pid == qb_id)].copy()
    if len(last) == 0:
        return None
    df = last[['pid', 'name', 'pos', 'gp']].copy()
    for kind, feats in [('rush', RUSH_F), ('rec', REC_F)]:
        S = []
        for num, den in feats:
            s = (last[num] / last[den].clip(lower=0.25)).clip(0, 1.5).values
            S.append(s / max(s.sum(), 1e-6))
        S.append(np.full(len(last), 1.0 / len(last)))
        S = np.array(S).T
        df['w_' + kind] = S @ db.beta[kind]
        df['sh_' + kind] = S[:, 0]
    # availability: fringe players only show up in the data when they touch the ball, so scale their
    # share by how often they appeared in the team's recent games (regular contributors are unaffected)
    tgs = db.tg[(db.tg.team == team) & (db.tg.game_date < asof)].game_id.values[-6:]
    cnt = hist[hist.game_id.isin(tgs) & (hist.team == team)].groupby('pid').size()
    rate = df.pid.map(cnt).fillna(0).values / max(len(tgs), 1)
    touches = (last.u_targets + last.u_carries).values
    adj = rate + (1 - rate) * np.minimum(1.0, touches / 5.0)
    adj = np.where(df.pid.values == qb_id, 1.0, adj)
    for c in ['w_rush', 'sh_rush', 'w_rec', 'sh_rec']:
        df[c] = df[c].values * adj
    df['avail'] = adj
    pri = [db.pri[p] for p in df.pos]
    W = last.W2.values
    tgt = last.e_targets.values * W; rec = last.e_rec.values * W; ry = last.e_rec_yds.values * W
    car = last.e_carries.values * W; ruy = last.e_rush_yds.values * W
    df['cr'] = (rec + 25 * np.array([x['cr'] for x in pri])) / (tgt + 25)
    df['ypr'] = (ry + 15 * np.array([x['ypr'] for x in pri])) / (rec + 15)
    df['ypc'] = (ruy + 50 * np.array([x['ypc'] for x in pri])) / (car + 50)
    df['is_qb'] = (df.pid == qb_id).values
    # ---- defence vs position
    m = dvp_mult(db, opp, asof, cfg) if cfg['use_dvp'] else {c: 1.0 for c in db.dvp_stats}
    posk = df.pos.where(df.pos.isin(['WR', 'TE', 'RB']), 'WR')
    df['m_rec'] = [m['recyds_' + p] for p in posk]
    df['m_rush'] = [m['rushyds_QB'] if p == 'QB' else m['rushyds_RB'] for p in df.pos]
    df['m_rectd'] = [m['rectd_' + p] for p in posk]
    df['m_rushtd'] = [m['rushtd_QB'] if p == 'QB' else m['rushtd_RB'] for p in df.pos]
    # ---- head to head (small, shrunk)
    df['h2h_rec'] = 1.0; df['h2h_rush'] = 1.0; df['h2h_n'] = 0
    if cfg['use_h2h'] and h2h:
        recent = pg[(pg.game_date < asof) & (pg.game_date >= asof - pd.Timedelta(days=365 * 3 + 120)) & pg.pid.isin(df.pid)]
        vs = recent[recent.opp == opp].groupby('pid').agg(n=('game_id', 'size'), ry=('rec_yds', 'mean'), ru=('rush_yds', 'mean'))
        al = recent.groupby('pid').agg(ry=('rec_yds', 'mean'), ru=('rush_yds', 'mean'))
        for i, pid in enumerate(df.pid.values):
            if pid in vs.index:
                n = vs.loc[pid, 'n']; wgt = n / (n + 10.0)
                for col, key in [('h2h_rec', 'ry'), ('h2h_rush', 'ru')]:
                    base = al.loc[pid, key]
                    if base >= 15:
                        ratio = np.clip(vs.loc[pid, key] / base, 0.5, 1.6)
                        df.iloc[i, df.columns.get_loc(col)] = float(np.clip(1 + wgt * (ratio - 1), 0.95, 1.05))
                df.iloc[i, df.columns.get_loc('h2h_n')] = int(n)
    # TD allocation weights with defence-vs-position tilt
    wr = df.w_rush.values * df.m_rushtd.values; wc = df.w_rec.values * df.m_rectd.values
    wc = np.where(df.is_qb.values, wc * 0.15, wc)
    df['p_rush_td'] = wr / wr.sum(); df['p_rec_td'] = wc / wc.sum()
    sr = df.sh_rush.values; sc = np.where(df.is_qb.values, 0.0, df.sh_rec.values)
    df['car_share'] = sr / max(sr.sum(), 1e-9); df['tgt_share'] = sc / max(sc.sum(), 1e-9)
    return df.reset_index(drop=True)

def team_volume(db, team, asof):
    t = db.tg[(db.tg.team == team) & (db.tg.game_date < asof)]
    if len(t) == 0:
        return dict(pa=33.0, ra=26.0, margin=0.0)
    r = t.iloc[-1]
    return dict(pa=float(r.ew_pass_att), ra=float(r.ew_rush_att), margin=float(r.ew_margin))


# ------------------------------------------------------------------ simulator
MAXH = 24

def simulate(db, par, players, vols, N, rng, cfg=CFG, wx=None, keep=False):
    """par: team_params; players: [home_df, away_df]; vols: [home, away] volume dicts.
    wx: optional dict(pass_mult, td_shift, fg_mult). Index 0 = home, 1 = away."""
    wx = wx or dict(pass_mult=1.0, td_shift=0.0, fg_shift=0.0, cr_mult=1.0)
    s = cfg['s_game']
    ar = np.arange(N)
    z = rng.normal(0, s, (N, 2)) + rng.normal(0, cfg.get('s_common', 0.0), (N, 1))
    pts = np.zeros((N, 2), dtype=np.int32); nTD = np.zeros((N, 2), dtype=np.int32)
    nFG = np.zeros((N, 2), dtype=np.int32); nOth = np.zeros((N, 2), dtype=np.int32); nd = np.zeros((N, 2), dtype=np.int32)
    first = rng.integers(0, 2, N)
    scale = cfg.get('dur_scale', 1.0) * db.mean_drives / par['mu_drives']
    shp = (db.dur_mean / db.dur_sd) ** 2; scl = db.dur_sd ** 2 / db.dur_mean * scale
    if keep:
        ev = dict(team=np.full((N, 2 * MAXH), -1, np.int8), kind=np.zeros((N, 2 * MAXH), np.int8),
                  tend=np.zeros((N, 2 * MAXH), np.int16), half=np.zeros((N, 2 * MAXH), np.int8),
                  pat=np.zeros((N, 2 * MAXH), np.int8))
    step = 0
    for half in (0, 1):
        t = np.full(N, 1800.0)
        poss = first if half == 0 else 1 - first
        for j in range(MAXH):
            act = t > 0
            if not act.any():
                break
            diff = pts[ar, poss] - pts[ar, 1 - poss]
            bk = np.digitize(diff, db.BUCKETS)
            zz = z[ar, poss]
            ptd = sigmoid(par['td'][poss] + zz + db.rb_td[half][bk] + wx['td_shift'] + cfg.get('td_delta', 0.0))
            pfg = sigmoid(par['fg'][poss] + 0.5 * zz + db.rb_fg[half][bk] + wx['fg_shift'])
            u = rng.random(N)
            kind = np.where(u < ptd, 1, np.where(u < ptd + (1 - ptd) * pfg, 2, np.where(u > 1 - db.p_other, 3, 0)))
            dur = rng.gamma(shp[kind], scl[kind])
            over = dur > t
            # drive cut short by the clock: mostly ends in a field goal try or nothing
            f = np.minimum(1.0, t / 110.0)
            u2 = rng.random(N)
            cut = np.where(u2 < 0.035 * f * ptd / 0.24, 1, np.where(u2 < (0.035 * ptd / 0.24 + cfg.get('cut_fg', 0.30)) * f, 2, 0))
            kind = np.where(over, cut, kind)
            kind = np.where(act, kind, -1)
            r = rng.random(N)
            pat = np.where(r < 0.09 * 0.48, 2, np.where(r < 0.09, 0, np.where(r < 0.09 + 0.91 * 0.945, 1, 0)))
            is_td = kind == 1; is_fg = kind == 2; is_ot = kind == 3
            np.add.at(pts, (ar, poss), np.where(is_td, 6 + pat, 0) + np.where(is_fg, 3, 0))
            np.add.at(pts, (ar, 1 - poss), np.where(is_ot, 6 + np.minimum(pat, 1), 0))
            np.add.at(nTD, (ar, poss), is_td.astype(np.int32)); np.add.at(nFG, (ar, poss), is_fg.astype(np.int32))
            np.add.at(nOth, (ar, 1 - poss), is_ot.astype(np.int32)); np.add.at(nd, (ar, poss), act.astype(np.int32))
            t = np.where(act, t - np.minimum(dur, t), t)
            if keep:
                ev['team'][:, step] = np.where(act, poss, -1); ev['kind'][:, step] = np.maximum(kind, 0)
                ev['tend'][:, step] = np.maximum(t, 0).astype(np.int16); ev['half'][:, step] = half
                ev['pat'][:, step] = pat
            poss = 1 - poss
            step += 1
    # overtime
    tie = pts[:, 0] == pts[:, 1]
    edge = (par['td'][0] - par['td'][1]) * 1.2
    ot_winner = np.where(rng.random(N) < sigmoid(edge), 0, 1)
    ot_pts = np.where(rng.random(N) < 0.55, 3, 6)
    ot_dec = tie & (rng.random(N) < 0.93)
    pts[ar[ot_dec], ot_winner[ot_dec]] += ot_pts[ot_dec]
    out = dict(pts=pts, nTD=nTD, nFG=nFG, nOth=nOth, ot=ot_dec, ot_winner=ot_winner, ot_pts=ot_pts, nd=nd)
    if keep:
        out['ev'] = ev
    margin = pts[:, 0] - pts[:, 1]
    if players is None:
        return out
    pr = sigmoid(par['rush'])
    stats = []
    for t in (0, 1):
        df = players[t]
        P = len(df)
        n_r = rng.binomial(nTD[:, t], pr[t])
        n_c = nTD[:, t] - n_r
        rtd = rng.multinomial(n_r, df.p_rush_td.values)
        ctd = rng.multinomial(n_c, df.p_rec_td.values)
        # volume
        mg = margin if t == 0 else -margin
        v = vols[t]
        bp = db.vol['pass_att']['b']; br = db.vol['rush_att']['b']
        c = db.vol['corr']
        e1 = rng.normal(0, 1, N); e2 = c * e1 + np.sqrt(1 - c * c) * rng.normal(0, 1, N)
        PA = np.rint(bp[0] + bp[1] * v['pa'] + bp[2] * (mg - v['margin']) + db.vol['pass_att']['sd'] * e1).clip(12, 65).astype(int)
        RA = np.rint(br[0] + br[1] * v['ra'] + br[2] * (mg - v['margin']) + db.vol['rush_att']['sd'] * e2).clip(8, 55).astype(int)
        st = np.sqrt(s ** 2 + cfg.get('s_common', 0.0) ** 2)
        effp = np.exp(cfg['sig_eff_pass'] * z[:, t] / st - cfg['sig_eff_pass'] ** 2 / 2) * wx['pass_mult']
        effr = np.exp(cfg['sig_eff_rush'] * z[:, t] / st - cfg['sig_eff_rush'] ** 2 / 2) * wx.get('rush_mult', 1.0)
        # targets
        T = rng.binomial(PA, db.tgt_frac)
        sh = df.tgt_share.values
        gsh = rng.gamma(np.maximum(cfg['dir_tgt'] * sh, 1e-3)[None, :], 1.0, (N, P))
        gsh /= gsh.sum(1, keepdims=True)
        tgts = rng.multinomial(T, gsh)
        cr = np.clip(df.cr.values * wx['cr_mult'] * cfg.get('cr_cal', 1.0), 0.2, 0.95)
        recs = rng.binomial(tgts, cr[None, :])
        k = cfg['k_rec']
        mean_ypr = (df.ypr.values * df.m_rec.values * df.h2h_rec.values)[None, :] * effp[:, None]
        ryds = rng.gamma(np.maximum(recs * k, 1e-6), mean_ypr / k)
        ryds = np.where(recs > 0, ryds, 0.0)
        # carries
        shc = df.car_share.values
        gc = rng.gamma(np.maximum(cfg['dir_car'] * shc, 1e-3)[None, :], 1.0, (N, P))
        gc /= gc.sum(1, keepdims=True)
        cars = rng.multinomial(RA, gc)
        sh0 = cfg['k_rush_shift']
        mean_c = (df.ypc.values * df.m_rush.values * df.h2h_rush.values * cfg.get('rush_cal', 1.0))[None, :] * effr[:, None] + sh0
        kk = (mean_c / cfg['sd_carry']) ** 2
        uyds = rng.gamma(np.maximum(cars * kk, 1e-6), mean_c / kk) - sh0 * cars
        uyds = np.where(cars > 0, uyds, 0.0)
        ints = rng.binomial(PA, db.int_rate)
        stats.append(dict(rtd=rtd, ctd=ctd, tgts=tgts, recs=recs, ryds=ryds, cars=cars, uyds=uyds, PA=PA, RA=RA,
                          pass_yds=ryds.sum(1), comp=recs.sum(1), pass_td=n_c, ints=ints))
    out['stats'] = stats
    return out
