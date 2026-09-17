"""Why doesn't the futures benchmark tie out? It has no options in it.

Sharpe is leverage-invariant, so if leverage explains the drawdown gap but not
the Sharpe gap, they are two separate problems and should be reported as such.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np, pandas as pd
from volsurface import data, engine, ivydb, metrics
from volsurface.engine import Costs
from volsurface.strategies import cs_replacement

pd.set_option("display.width", 170, "display.float_format", lambda x: f"{x:,.4f}")
START, END, CAPITAL = "2017-01-03", "2026-09-14", 1_000_000.0

conn = ivydb.connect()
secid = ivydb.securityid(conn, "SPX")
px = ivydb.load_prices(conn, secid, START, END)
surface = data.OptionMetricsSurface(
    ivydb.load_surface(conn, secid, START, END), px,
    rates=ivydb.load_rates(conn, START, END, 30), divs=ivydb.load_divs(conn, secid, START, END))
conn.close()

dates = surface.dates[260:]
engine.use_costs(Costs.zero())
trades, daily, _ = cs_replacement.build(surface, dates, stage="futures")

print("=== futures leg, full metrics ===")
print(pd.Series(metrics.summary(daily, CAPITAL, trades)).to_string())

# Realised leverage: how much notional is actually live on an average day?
live = pd.Series(0.0, index=dates)
for _, r in trades.iterrows():
    live.loc[(live.index >= r["entry"]) & (live.index <= r["exit"])] += cs_replacement.Params().daily_notional
print(f"\ndeployed notional: mean ${live.mean():,.0f}  peak ${live.max():,.0f}  "
      f"(capital used for max_dd: ${CAPITAL:,.0f})")
print(f"=> mean leverage {live.mean()/CAPITAL:.2f}x, peak {live.max()/CAPITAL:.2f}x")

# What did SPX itself do over the same window? This is the honest benchmark
# for the benchmark: a 1x long should roughly match it.
spot = pd.Series({d: surface.spot(d) for d in dates})
ret = spot.pct_change().dropna()
spx_sharpe = float(ret.mean() / ret.std() * np.sqrt(252))
spx_dd = float((spot / spot.cummax() - 1.0).min())
print(f"\nSPX itself {dates[0].date()}->{dates[-1].date()}: "
      f"price return {(spot.iloc[-1]/spot.iloc[0])**(252/len(spot))-1:.2%} p.a., "
      f"vol {ret.std()*np.sqrt(252):.2%}, Sharpe {spx_sharpe:.4f}, maxDD {spx_dd:.2%}")

print(f"\nstrategy ann_vol {metrics.ann_vol(daily, CAPITAL):.2%} vs SPX {ret.std()*np.sqrt(252):.2%}"
      f"  -> implied leverage {metrics.ann_vol(daily, CAPITAL)/(ret.std()*np.sqrt(252)):.2f}x")
print(f"max_dd on $1mm: {metrics.max_drawdown(daily, CAPITAL):.2%}   "
      f"on PEAK DEPLOYED (${live.max():,.0f}): {metrics.max_drawdown(daily, live.max()):.2%}")
