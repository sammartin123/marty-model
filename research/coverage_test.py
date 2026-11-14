"""Does coverage-shell matchup data improve receiving predictions?  (study, 2023-2025)

Idea under test: a pass catcher who is more productive against middle-of-field-closed coverage (MOFC:
Cover 1, Cover 3) should do better against defences that play more MOFC, and the reverse for
middle-of-field-open (MOFO: Cover 0, 2, 2-Man, 4, 6, 9).

Routes are approximated by dropbacks with the player on the field. Every prediction uses only earlier games.
To isolate the matchup effect from playing time, each model predicts yards (and targets) per route and is
scored on the player's actual routes in the game.

usage: python3 research/coverage_test.py <dir with pbp_participation_YYYY.parquet> [shell|manzone]   (pbp comes from ./data)
"""
import sys, os, json, numpy as np, pandas as pd, pyarrow.parquet as pq
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PART = sys.argv[1]
MOFC = {'COVER_1', 'COVER_3'}; MOFO = {'COVER_0', 'COVER_2', '2_MAN', 'COVER_4', 'COVER_6', 'COVER_9'}
YEARS = [2023, 2024, 2025]
pos = pd.read_csv(os.path.join(ROOT, 'data', 'players.csv'), low_memory=False, usecols=['gsis_id', 'position', 'display_name']).dropna(subset=['gsis_id'])
POS = dict(zip(pos.gsis_id, pos.position)); NAME = dict(zip(pos.gsis_id, pos.display_name))

plays = []
for y in YEARS:
    t = pq.read_table(os.path.join(PART, f'part_{y}.parquet'), columns=['nflverse_game_id', 'play_id', 'offense_players', 'defense_coverage_type', 'defense_man_zone_type']).to_pandas()
    p = pq.read_table(os.path.join(ROOT, 'data', f'pbp_{y}.parquet'), columns=['game_id', 'play_id', 'season', 'week', 'game_date', 'season_type', 'posteam', 'defteam', 'qb_dropback', 'receiver_player_id', 'receiving_yards', 'pass_attempt', 'sack', 'two_point_attempt']).to_pandas()
    m = p[(p.qb_dropback == 1) & (p.two_point_attempt != 1) & (p.season_type == 'REG')].merge(t, left_on=['game_id', 'play_id'], right_on=['nflverse_game_id', 'play_id'], how='inner')
    plays.append(m)
P = pd.concat(plays, ignore_index=True)
P['shell'] = np.where(P.defense_coverage_type.isin(MOFC), 'C', np.where(P.defense_coverage_type.isin(MOFO), 'O', 'U'))
MODE = sys.argv[2] if len(sys.argv) > 2 else 'shell'
if MODE == 'manzone':      # same test with man ('C') vs zone ('O') instead of MOFC vs MOFO
    P['shell'] = np.where(P.defense_man_zone_type == 'MAN_COVERAGE', 'C', np.where(P.defense_man_zone_type == 'ZONE_COVERAGE', 'O', 'U'))
P['game_date'] = pd.to_datetime(P.game_date)
P['yds'] = P.receiving_yards.fillna(0)
print('dropbacks', len(P), 'shell mix', P.shell.value_counts(normalize=True).round(3).to_dict())

# one row per skill player on the field per dropback
X = P[['game_id', 'play_id', 'season', 'week', 'game_date', 'posteam', 'defteam', 'shell', 'receiver_player_id', 'yds', 'offense_players']].copy()
X['pid'] = X.offense_players.str.split(';')
X = X.explode('pid'); X = X[X.pid.notna() & (X.pid != '')]
X['pos'] = X.pid.map(POS); X = X[X.pos.isin(['WR', 'TE'])]
X['tgt'] = (X.receiver_player_id == X.pid).astype(int); X['y'] = X.yds * X.tgt
for s in 'CO':
    X['r' + s] = (X.shell == s).astype(int); X['t' + s] = X.tgt * X['r' + s]; X['y' + s] = X.y * X['r' + s]
X['r'] = 1
G = X.groupby(['pid', 'pos', 'game_id', 'season', 'week', 'game_date', 'posteam', 'defteam'], as_index=False)[['r', 'tgt', 'y', 'rC', 'tC', 'yC', 'rO', 'tO', 'yO']].sum()
G = G.sort_values(['game_date', 'game_id']).reset_index(drop=True)

# defence: share of classified dropbacks in MOFC, per game, and prior (as-of) rate
D = P[P.shell != 'U'].groupby(['defteam', 'game_id', 'season', 'game_date'], as_index=False).agg(n=('shell', 'size'), c=('shell', lambda s: (s == 'C').sum()))
D = D.sort_values(['game_date', 'game_id'])
LEAGUE_C = D.c.sum() / D.n.sum()
half = 0.5 ** (1 / 6)                                      # defensive tendencies: 6-game half-life
def asof_rate(g):
    c = n = 0.0; out = []
    for ci, ni in zip(g.c.values, g.n.values):
        out.append((c + 60 * LEAGUE_C) / (n + 60)); c = c * half + ci; n = n * half + ni
    return pd.Series(out, index=g.index)
D['prior_c'] = D.groupby('defteam', group_keys=False)[['c', 'n']].apply(asof_rate)
D['actual_c'] = D.c / D.n
G = G.merge(D[['defteam', 'game_id', 'prior_c', 'actual_c']], on=['defteam', 'game_id'], how='left')

# player as-of (prior) cumulative sums
cum = G.groupby('pid')[['r', 'tgt', 'y', 'rC', 'tC', 'yC', 'rO', 'tO', 'yO']].cumsum() - G[['r', 'tgt', 'y', 'rC', 'tC', 'yC', 'rO', 'tO', 'yO']]
cum.columns = ['p_' + c for c in cum.columns]
G = pd.concat([G, cum], axis=1)

def evaluate(stat, kO, kS, test):
    """stat: 'y' (yards) or 't' (targets). kO: shrink routes for overall rate. kS: shrink routes for the shell split."""
    num = {'y': ('y', 'yC', 'yO'), 't': ('tgt', 'tC', 'tO')}[stat]
    T = G[test].copy()
    out = {}
    for ps in ['WR', 'TE']:
        tr = G[G.pos == ps]                                  # league constants from everything (tiny leakage, same for all models)
        base = tr[num[0]].sum() / tr.r.sum()
        ratioC = (tr[num[1]].sum() / tr.rC.sum()) / base; ratioO = (tr[num[2]].sum() / tr.rO.sum()) / base
        t = T[T.pos == ps]
        overall = (t['p_' + num[0]] + kO * base) / (t.p_r + kO)
        sC = (t['p_' + num[1]] + kS * overall * ratioC) / (t.p_rC + kS)
        sO = (t['p_' + num[2]] + kS * overall * ratioO) / (t.p_rO + kS)
        preds = {
            'baseline': overall,                                                        # player's own rate, no coverage info
            'defence_only': overall * (t.prior_c * ratioC + (1 - t.prior_c) * ratioO),      # league shell effect x defence tendency
            'player_split': t.prior_c * sC + (1 - t.prior_c) * sO,                          # the idea being tested
            'split_if_shell_known': t.actual_c.fillna(LEAGUE_C) * sC + (1 - t.actual_c.fillna(LEAGUE_C)) * sO,  # upper bound: we knew the defence's calls in advance
        }
        actual = t[num[0]].values
        res = {}
        for k, v in preds.items():
            pr = v.values * t.r.values
            res[k] = dict(mae=float(np.abs(pr - actual).mean()), rmse=float(np.sqrt(((pr - actual) ** 2).mean())))
        # bootstrap by game: is player_split better than baseline?
        e0 = np.abs(preds['baseline'].values * t.r.values - actual); e1 = np.abs(preds['player_split'].values * t.r.values - actual)
        gids = t.game_id.values; ug, inv = np.unique(gids, return_inverse=True)
        d = np.bincount(inv, weights=e1 - e0); rng = np.random.default_rng(1)
        bs = [d[rng.integers(0, len(ug), len(ug))].sum() / len(t) for _ in range(2000)]
        res['split_minus_base_mae'] = dict(diff=float((e1 - e0).mean()), lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)))
        res['n'] = int(len(t)); res['ratioC'] = float(ratioC); res['ratioO'] = float(ratioO)
        out[ps] = res
    return out

test = (G.season >= 2024) & (G.r >= 15) & (G.p_r >= 100)
R = {}
for stat in ('y', 't'):
    best = None
    for kO in (100, 200):
        for kS in (100, 300, 1000):
            r = evaluate(stat, kO, kS, test)
            score = r['WR']['player_split']['mae'] + r['TE']['player_split']['mae']
            if best is None or score < best[0]:
                best = (score, kO, kS, r)
    R[stat] = dict(kO=best[1], kS=best[2], res=best[3])

# persistence checks
d24 = D[D.season == 2024].groupby('defteam').agg(c=('c', 'sum'), n=('n', 'sum')); d25 = D[D.season == 2025].groupby('defteam').agg(c=('c', 'sum'), n=('n', 'sum'))
yoy = float(np.corrcoef((d24.c / d24.n).reindex(d25.index), d25.c / d25.n)[0, 1])
wk = D[D.season >= 2024]; wk_corr = float(np.corrcoef(wk.prior_c, wk.actual_c)[0, 1])
spread = float((d25.c / d25.n).std()); rng_ = (float((d25.c / d25.n).min()), float((d25.c / d25.n).max()))
# player split persistence: (yards/route vs MOFC - vs MOFO), 2024 vs 2025, players with 150+ routes vs each shell both years
def split(season):
    s = G[G.season == season].groupby(['pid', 'pos'])[['rC', 'yC', 'rO', 'yO', 'tC', 'tO']].sum()
    s = s[(s.rC >= 150) & (s.rO >= 150)]
    return (s.yC / s.rC - s.yO / s.rO).rename('d'), (s.tC / s.rC - s.tO / s.rO).rename('dt')
a, at = split(2024); b, bt = split(2025)
j = pd.concat([a, b.rename('d2')], axis=1).dropna(); jt = pd.concat([at, bt.rename('d2')], axis=1).dropna()
persist = dict(n=int(len(j)), yprr_split_corr=float(np.corrcoef(j.d, j.d2)[0, 1]), tprr_split_corr=float(np.corrcoef(jt.dt, jt.d2)[0, 1]))
# overall yards-per-route persistence for the same players, for scale
ov = lambda season: (lambda s: (s.y / s.r)[s.r >= 300])(G[G.season == season].groupby('pid')[['y', 'r']].sum())
o = pd.concat([ov(2024).rename('a'), ov(2025).rename('b')], axis=1).dropna()
persist['overall_yprr_corr'] = float(np.corrcoef(o.a, o.b)[0, 1]); persist['overall_n'] = int(len(o))
out = dict(results=R, league_mofc=float(LEAGUE_C), defence=dict(year_to_year_corr=yoy, prior_vs_game_corr=wk_corr, team_sd=spread, team_range=rng_), player_split=persist)
json.dump(out, open(os.path.join(ROOT, 'research', f'coverage_test_results_{MODE}.json'), 'w'), indent=1)
for stat, lab in (('y', 'YARDS'), ('t', 'TARGETS')):
    print(f"\n== {lab} per game, given actual routes (shrink overall={R[stat]['kO']}, split={R[stat]['kS']})")
    for ps in ('WR', 'TE'):
        r = R[stat]['res'][ps]
        print(f" {ps} n={r['n']}  MOFC ratio {r['ratioC']:.3f} MOFO ratio {r['ratioO']:.3f}")
        for k in ('baseline', 'defence_only', 'player_split', 'split_if_shell_known'):
            print(f"    {k:22s} MAE {r[k]['mae']:.4f}  RMSE {r[k]['rmse']:.4f}")
        d = r['split_minus_base_mae']; print(f"    split minus baseline MAE: {d['diff']:+.4f}  (95% range {d['lo']:+.4f} to {d['hi']:+.4f})")
print('\ndefence MOFC rate: league', round(LEAGUE_C, 3), '| team sd', round(spread, 3), 'range', [round(x, 3) for x in rng_], '| 2024->2025 corr', round(yoy, 3), '| prior vs actual game corr', round(wk_corr, 3))
print('player split persistence 2024->2025:', {k: round(v, 3) if isinstance(v, float) else v for k, v in persist.items()})

# ---- robustness: the 10% of player-games where the coverage model disagrees most with the baseline
print('\n== biggest-disagreement decile (yards)')
extra = {}
for ps in ('WR', 'TE'):
    t = G[test & (G.pos == ps)]
    tr = G[G.pos == ps]; base = tr.y.sum() / tr.r.sum(); rC = (tr.yC.sum() / tr.rC.sum()) / base; rO = (tr.yO.sum() / tr.rO.sum()) / base
    ov = (t.p_y + 200 * base) / (t.p_r + 200)
    for kS in (100, 300):
        sC = (t.p_yC + kS * ov * rC) / (t.p_rC + kS); sO = (t.p_yO + kS * ov * rO) / (t.p_rO + kS)
        sp = (t.prior_c * sC + (1 - t.prior_c) * sO) * t.r; b = ov * t.r
        gap = (sp - b); cut = gap.abs().quantile(.9); m = gap.abs() >= cut
        side = np.sign(gap[m]); act = t.y[m] - b[m]
        extra[f'{ps}_k{kS}'] = dict(n=int(m.sum()), mean_gap=float(gap[m].abs().mean()), mae_base=float((t.y[m] - b[m]).abs().mean()),
                                    mae_split=float((t.y[m] - sp[m]).abs().mean()), right_direction=float((np.sign(act) == side).mean()),
                                    actual_shift=float((act * side).mean()))
        print(' ', ps, 'shrink', kS, {k: round(v, 3) if isinstance(v, float) else v for k, v in extra[f'{ps}_k{kS}'].items()})
out['largest_disagreements'] = extra
json.dump(out, open(os.path.join(ROOT, 'research', f'coverage_test_results_{MODE}.json'), 'w'), indent=1)
