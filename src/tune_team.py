import numpy as np, pandas as pd, itertools, model as M, sys
db=M.DB()
TGs=db.tg.groupby('game_id')[['off_td','fgs','drives']].sum()
TG={(g,c):v for c in TGs.columns for g,v in TGs[c].items()}
def ev(cfg, seasons=(2022,2023,2024,2025), N=1500):
    rng=np.random.default_rng(3)
    g=db.games[db.games.season.isin(seasons)&(db.games.game_type=='REG')&db.games.home_score.notna()].sort_values('gameday')
    rows=[]; fd=None; fit=None
    for _,gm in g.iterrows():
        if gm.gameday!=fd: fit=M.fit_team(db,gm.gameday,cfg); fd=gm.gameday
        par=M.team_params(fit,gm.home_team,gm.away_team,neutral=(gm.location=='Neutral'))
        o=M.simulate(db,par,None,None,N,rng,cfg)
        mg=o['pts'][:,0]-o['pts'][:,1]; tot=o['pts'].sum(1)
        rows.append((mg.mean(),mg.std(),tot.mean(),tot.std(),(mg>0).mean()+.5*(mg==0).mean(),gm.result,gm.total,gm.spread_line,gm.total_line,o['nTD'].mean()*2,o['nFG'].mean()*2,o['nd'].mean()*2,TG.get((gm.game_id,'off_td'),np.nan),TG.get((gm.game_id,'fgs'),np.nan),TG.get((gm.game_id,'drives'),np.nan)))
    r=np.array(rows,float); y=(r[:,5]>0)+.5*(r[:,5]==0)
    return dict(mae_m=np.abs(r[:,0]-r[:,5]).mean(), mae_mkt=np.abs(r[:,7]-r[:,5]).mean(), rmse_m=np.sqrt(((r[:,0]-r[:,5])**2).mean()), sd_m=r[:,1].mean(),
                mae_t=np.abs(r[:,2]-r[:,6]).mean(), mae_tmkt=np.abs(r[:,8]-r[:,6]).mean(), rmse_t=np.sqrt(((r[:,2]-r[:,6])**2).mean()), sd_t=r[:,3].mean(), brier=((r[:,4]-y)**2).mean(), bias_t=(r[:,2]-r[:,6]).mean(),
                slope=np.polyfit(r[:,0],r[:,5],1)[0], sTD=r[:,9].mean(), aTD=np.nanmean(r[:,12]), sFG=r[:,10].mean(), aFG=np.nanmean(r[:,13]), sD=r[:,11].mean(), aD=np.nanmean(r[:,14]), corr_mkt=np.corrcoef(r[:,0],r[:,7])[0,1])
if __name__=='__main__':
    for ridge,hl,win in itertools.product([4,8,14],[90,150,240],[520,800]):
        cfg=dict(M.CFG,ridge=ridge,team_halflife_days=hl,team_window_days=win,s_game=0.2)
        r=ev(cfg,N=600)
        print(ridge,hl,win,{k:round(float(v),3) for k,v in r.items()},flush=True)
