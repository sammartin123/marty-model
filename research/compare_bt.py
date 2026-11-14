"""Paired comparison of two backtest runs on the player-games and games they share.
usage: python3 research/compare_bt.py <base players.parquet> <new players.parquet> [label]"""
import sys, numpy as np, pandas as pd
a = pd.read_parquet(sys.argv[1]); b = pd.read_parquet(sys.argv[2]); lab = sys.argv[3] if len(sys.argv) > 3 else ''
m = a.merge(b, on=['game_id', 'pid'], suffixes=('_a', '_b'))
eps = 1e-4
def ll(p, y): p = p.clip(eps, 1 - eps); return -(y * np.log(p) + (1 - y) * np.log(1 - p))
def boot(d, gids, n=1500):
    ug, inv = np.unique(gids, return_inverse=True); s = np.bincount(inv, weights=d); c = np.bincount(inv)
    rng = np.random.default_rng(0); r = []
    for _ in range(n):
        i = rng.integers(0, len(ug), len(ug)); r.append(s[i].sum() / c[i].sum())
    return np.quantile(r, [.025, .975])
def line(name, ea, eb, gids):
    d = eb - ea; lo, hi = boot(d.values, gids.values)
    verdict = 'BETTER' if hi < 0 else ('WORSE' if lo > 0 else 'no clear difference')
    print(f"  {name:26s} n={len(d):6d}  base {ea.mean():.4f}  new {eb.mean():.4f}  diff {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  {verdict}")
print(f"== {lab}: {len(m)} shared player-games")
y = m.a_td_a
line('TD log loss', ll(m.p_td_a, y), ll(m.p_td_b, y), m.game_id)
line('TD Brier', (m.p_td_a - y) ** 2, (m.p_td_b - y) ** 2, m.game_id)
t = m[m.p_td_a >= .15]; line('TD Brier (15%+ players)', (t.p_td_a - t.a_td_a) ** 2, (t.p_td_b - t.a_td_a) ** 2, t.game_id)
r = m[m.played_a & m.played_b & (m.m_rec_a >= 20)]; line('Rec yards MAE', (r.m_rec_a - r.a_rec_a).abs(), (r.m_rec_b - r.a_rec_a).abs(), r.game_id)
u = m[m.played_a & m.played_b & (m.m_rush_a >= 20)]; line('Rush yards MAE', (u.m_rush_a - u.a_rush_a).abs(), (u.m_rush_b - u.a_rush_a).abs(), u.game_id)
q = m[m.is_qb_a & m.is_qb_b & m.played_a & m.a_pass_a.notna()]; line('Pass yards MAE', (q.m_pass_a - q.a_pass_a).abs(), (q.m_pass_b - q.a_pass_a).abs(), q.game_id)
sq = lambda p, y: (p - y) ** 2
line('Rec yards sq error', sq(r.m_rec_a, r.a_rec_a), sq(r.m_rec_b, r.a_rec_a), r.game_id)
line('Rush yards sq error', sq(u.m_rush_a, u.a_rush_a), sq(u.m_rush_b, u.a_rush_a), u.game_id)
line('Pass yards sq error', sq(q.m_pass_a, q.a_pass_a), sq(q.m_pass_b, q.a_pass_a), q.game_id)
if 'med_rec_a' in m:
    line('Rec yards MAE (median)', (r.med_rec_a - r.a_rec_a).abs(), (r.med_rec_b - r.a_rec_a).abs(), r.game_id)
print(f"  bias (new): rec {(r.m_rec_b - r.a_rec_a).mean():+.2f}  rush {(u.m_rush_b - u.a_rush_a).mean():+.2f}  pass {(q.m_pass_b - q.a_pass_a).mean():+.2f}   | base: rec {(r.m_rec_a - r.a_rec_a).mean():+.2f} rush {(u.m_rush_a - u.a_rush_a).mean():+.2f} pass {(q.m_pass_a - q.a_pass_a).mean():+.2f}")
col = 'qb_change_b' if 'qb_change_b' in m else ('qb_change' if 'qb_change' in m else None)
if col:
    c = m[m[col] == True]
    if len(c):
        print(f"  -- teams with a quarterback change ({c.game_id.nunique()} games)")
        line('TD log loss', ll(c.p_td_a, c.a_td_a), ll(c.p_td_b, c.a_td_a), c.game_id)
        r2 = c[c.played_a & c.played_b & (c.m_rec_a >= 20)]; line('Rec yards MAE', (r2.m_rec_a - r2.a_rec_a).abs(), (r2.m_rec_b - r2.a_rec_a).abs(), r2.game_id)
        q2 = c[c.is_qb_a & c.is_qb_b & c.played_a & c.a_pass_a.notna()]; line('Pass yards MAE', (q2.m_pass_a - q2.a_pass_a).abs(), (q2.m_pass_b - q2.a_pass_a).abs(), q2.game_id)
        print(f"     pass bias in those games: base {(q2.m_pass_a - q2.a_pass_a).mean():+.1f}  new {(q2.m_pass_b - q2.a_pass_a).mean():+.1f}")
# games
try:
    ga = pd.read_parquet(sys.argv[1].replace('players', 'games')); gb = pd.read_parquet(sys.argv[2].replace('players', 'games'))
    g = ga.merge(gb, on='game_id', suffixes=('_a', '_b')); yy = (g.result_a > 0) + .5 * (g.result_a == 0)
    line('Win prob Brier', (g.p_home_a - yy) ** 2, (g.p_home_b - yy) ** 2, g.game_id)
    line('Margin MAE', (g.m_margin_a - g.result_a).abs(), (g.m_margin_b - g.result_a).abs(), g.game_id)
    if 'qb_change' in gb:
        c = g[g.qb_change == True] if 'qb_change' in g else g[g.qb_change_b == True]
        yc = (c.result_a > 0) + .5 * (c.result_a == 0)
        print(f"  -- games with a quarterback change ({len(c)})"); line('Win prob Brier', (c.p_home_a - yc) ** 2, (c.p_home_b - yc) ** 2, c.game_id); line('Margin MAE', (c.m_margin_a - c.result_a).abs(), (c.m_margin_b - c.result_a).abs(), c.game_id)
except Exception as e:
    print('  (no game-level comparison:', str(e)[:80], ')')
