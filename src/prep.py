"""Build compact modelling tables from nflverse play-by-play (every pbp_YYYY.parquet in ./data).

Outputs (parquet, in ../work):
  drives.parquet       one row per offensive drive
  player_games.parquet one row per player per game (usage + production)
  team_games.parquet   one row per team per game (volume, points, weather)
  scores.parquet       scoring-play details (TD length, FG distance) for the replay
"""
import os, glob, numpy as np, pandas as pd, pyarrow.parquet as pq

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
WORK = os.path.join(os.path.dirname(__file__), '..', 'work')
os.makedirs(WORK, exist_ok=True)

COLS = ['game_id', 'season', 'week', 'season_type', 'game_date', 'home_team', 'away_team', 'posteam', 'defteam',
        'play_type', 'fixed_drive', 'fixed_drive_result', 'roof', 'temp', 'wind', 'weather', 'rush_attempt',
        'pass_attempt', 'sack', 'two_point_attempt', 'td_team', 'rush_touchdown', 'pass_touchdown', 'touchdown',
        'td_player_id', 'rusher_player_id', 'receiver_player_id', 'passer_player_id', 'rushing_yards',
        'receiving_yards', 'passing_yards', 'complete_pass', 'interception', 'air_yards', 'yardline_100',
        'field_goal_result', 'kick_distance', 'qtr', 'yards_gained', 'qb_kneel', 'qb_scramble',
        'score_differential', 'half_seconds_remaining', 'game_half']

frames = []
for f in sorted(glob.glob(os.path.join(DATA, 'pbp_*.parquet'))):
    p = pq.read_table(f, columns=COLS).to_pandas()
    frames.append(p)
p = pd.concat(frames, ignore_index=True)
p['game_date'] = pd.to_datetime(p['game_date'])
p = p[p.posteam.notna() & (p.posteam != '')].copy()
for c in ['posteam', 'defteam', 'home_team', 'away_team', 'td_team']:
    p[c] = p[c].replace({'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA'})
p['game_id_raw'] = p['game_id']
p['is_home'] = (p.posteam == p.home_team).astype(int)
p['reg'] = (p.season_type == 'REG').astype(int)

games = pd.read_csv(os.path.join(DATA, 'games.csv'))
for c in ['home_team', 'away_team']:
    games[c] = games[c].replace({'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA'})

# ---------------- drives ----------------
real = p[p.play_type.isin(['run', 'pass', 'qb_spike', 'field_goal', 'punt', 'qb_kneel'])]
g = real.groupby(['game_id', 'posteam', 'fixed_drive'], sort=False)
drives = g.agg(season=('season', 'first'), week=('week', 'first'), reg=('reg', 'first'),
               game_date=('game_date', 'first'), defteam=('defteam', 'first'), is_home=('is_home', 'first'),
               result=('fixed_drive_result', 'first'), n_plays=('play_type', 'size'),
               n_kneel=('qb_kneel', 'sum'), qtr=('qtr', 'first'), diff0=('score_differential', 'first'),
               half=('game_half', 'first'), t0=('half_seconds_remaining', 'max')).reset_index()
drives = drives.sort_values(['game_id', 'half', 't0'], ascending=[True, True, False])
nxt = drives.groupby(['game_id', 'half']).t0.shift(-1)
drives['last_in_half'] = nxt.isna().astype(int)
drives['dur'] = drives.t0 - nxt.fillna(0)
off_td = p[(p.touchdown == 1) & (p.td_team == p.posteam) & (p.two_point_attempt != 1)]
tdtype = off_td.groupby(['game_id', 'posteam', 'fixed_drive']).agg(
    rush_td=('rush_touchdown', 'max'), pass_td=('pass_touchdown', 'max')).reset_index()
drives = drives.merge(tdtype, on=['game_id', 'posteam', 'fixed_drive'], how='left')
drives['td'] = ((drives.result == 'Touchdown') & ((drives.rush_td == 1) | (drives.pass_td == 1))).astype(int)
drives['rush_td'] = ((drives.td == 1) & (drives.rush_td == 1)).astype(int)
drives['fg'] = (drives.result == 'Field goal').astype(int)
drives['opp_td'] = (drives.result == 'Opp touchdown').astype(int)
# drop pure kneel-down drives (end of half/game) so drive counts mean real possessions
drives = drives[~((drives.n_kneel == drives.n_plays))].copy()
drives.to_parquet(os.path.join(WORK, 'drives.parquet'))

# ---------------- player games ----------------
pl = p[(p.two_point_attempt != 1) & p.play_type.isin(['run', 'pass', 'qb_kneel'])].copy()
key = ['game_id', 'season', 'week', 'reg', 'game_date', 'posteam', 'defteam', 'is_home']

ru = pl[(pl.rush_attempt == 1) & pl.rusher_player_id.notna()].copy()
ru['in5'] = (ru.yardline_100 <= 5).astype(int)
ru['in10'] = (ru.yardline_100 <= 10).astype(int)
ru['kneel'] = (ru.play_type == 'qb_kneel').astype(int)
ru['car'] = 1 - ru['kneel']
ru['rtd'] = ((ru.rush_touchdown == 1) & (ru.td_player_id == ru.rusher_player_id)).astype(int)
rush = ru.groupby(key + ['rusher_player_id']).agg(
    carries=('car', 'sum'), rush_yds=('rushing_yards', 'sum'), rush_td=('rtd', 'sum'),
    car_in5=('in5', 'sum'), car_in10=('in10', 'sum')).reset_index().rename(columns={'rusher_player_id': 'pid'})

pa = pl[(pl.pass_attempt == 1) & (pl.sack != 1) & (pl.play_type == 'pass')].copy()
tg = pa[pa.receiver_player_id.notna()].copy()
tg['rz'] = (tg.yardline_100 <= 20).astype(int)
tg['in10'] = (tg.yardline_100 <= 10).astype(int)
tg['ez'] = (tg.air_yards >= tg.yardline_100).astype(int)
tg['ay'] = tg.air_yards.fillna(0).clip(lower=0)
tg['ctd'] = ((tg.pass_touchdown == 1) & (tg.td_player_id == tg.receiver_player_id)).astype(int)
rec = tg.groupby(key + ['receiver_player_id']).agg(
    targets=('pass_attempt', 'sum'), rec=('complete_pass', 'sum'), rec_yds=('receiving_yards', 'sum'),
    rec_td=('ctd', 'sum'), tgt_rz=('rz', 'sum'), tgt_in10=('in10', 'sum'), tgt_ez=('ez', 'sum'),
    air=('ay', 'sum')).reset_index().rename(columns={'receiver_player_id': 'pid'})

qb = pa[pa.passer_player_id.notna()].groupby(key + ['passer_player_id']).agg(
    pass_att=('pass_attempt', 'sum'), comp=('complete_pass', 'sum'), pass_yds=('passing_yards', 'sum'),
    pass_td=('pass_touchdown', 'sum'), ints=('interception', 'sum')).reset_index().rename(
    columns={'passer_player_id': 'pid'})

pg = rush.merge(rec, on=key + ['pid'], how='outer').merge(qb, on=key + ['pid'], how='outer').fillna(0)
pg = pg.rename(columns={'posteam': 'team', 'defteam': 'opp'})
players = pd.read_csv(os.path.join(DATA, 'players.csv'), low_memory=False)
pcol = 'gsis_id'
players = players[[pcol, 'display_name', 'position']].dropna(subset=[pcol]).drop_duplicates(pcol)
pg = pg.merge(players.rename(columns={pcol: 'pid', 'display_name': 'name', 'position': 'pos'}), on='pid', how='left')
pg['pos'] = pg.pos.replace({'FB': 'RB', 'HB': 'RB'}).fillna('OTH')
pg.loc[~pg.pos.isin(['QB', 'RB', 'WR', 'TE']), 'pos'] = 'OTH'
pg.to_parquet(os.path.join(WORK, 'player_games.parquet'))

# ---------------- team games ----------------
tgm = pg.groupby(['game_id', 'season', 'week', 'reg', 'game_date', 'team', 'opp', 'is_home']).agg(
    pass_att=('pass_att', 'sum'), rush_att=('carries', 'sum'), pass_yds=('pass_yds', 'sum'),
    rush_yds=('rush_yds', 'sum'), targets=('targets', 'sum'), comp=('comp', 'sum'), ints=('ints', 'sum')).reset_index()
gm = games[['game_id', 'home_team', 'away_team', 'home_score', 'away_score', 'spread_line', 'total_line',
            'home_moneyline', 'away_moneyline', 'roof', 'temp', 'wind']]
tgm = tgm.merge(gm, on='game_id', how='left')
tgm['pts'] = np.where(tgm.is_home == 1, tgm.home_score, tgm.away_score)
tgm['opp_pts'] = np.where(tgm.is_home == 1, tgm.away_score, tgm.home_score)
tgm['margin'] = tgm.pts - tgm.opp_pts
wx = p.groupby('game_id').weather.first().reset_index()
wx['rain'] = wx.weather.fillna('').str.lower().str.contains('rain|shower|drizzle').astype(int)
wx['snow'] = wx.weather.fillna('').str.lower().str.contains('snow|flurr').astype(int)
tgm = tgm.merge(wx[['game_id', 'rain', 'snow']], on='game_id', how='left')
dcount = drives.groupby(['game_id', 'posteam']).agg(drives=('td', 'size'), off_td=('td', 'sum'), fgs=('fg', 'sum'),
                                                    rush_tds=('rush_td', 'sum')).reset_index().rename(
    columns={'posteam': 'team'})
tgm = tgm.merge(dcount, on=['game_id', 'team'], how='left')
tgm.to_parquet(os.path.join(WORK, 'team_games.parquet'))

# ---------------- scoring details ----------------
sc = p[((p.touchdown == 1) & (p.td_team == p.posteam) & (p.two_point_attempt != 1)) |
       (p.field_goal_result == 'made')][['season', 'play_type', 'rush_touchdown', 'pass_touchdown', 'yards_gained',
                                         'kick_distance', 'field_goal_result']].copy()
sc.to_parquet(os.path.join(WORK, 'scores.parquet'))
fga = p[p.play_type == 'field_goal'][['season', 'kick_distance', 'field_goal_result', 'wind', 'roof']]
fga.to_parquet(os.path.join(WORK, 'fga.parquet'))
print('drives', len(drives), 'player_games', len(pg), 'team_games', len(tgm))
print(drives.groupby('season')[['td', 'fg', 'opp_td', 'rush_td']].mean().round(3).tail(4))
print(tgm.groupby('season')[['drives', 'pass_att', 'rush_att', 'pts', 'off_td']].mean().round(2).tail(4))
