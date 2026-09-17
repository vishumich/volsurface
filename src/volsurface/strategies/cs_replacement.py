"""Idea #2 - call-spread replacement for long equity beta (SPX).

JPM section 3.3.2, built in four stages so each increment is attributable:

  futures   long 1M futures, 1/22 notional per day, held to maturity
  baseline  long 1M 50d/10d call spread, 2x leverage, held to maturity
            reported: Sharpe 0.97 vs 0.73, maxDD -13.6% vs -26.1%
  +trend    skip entry when trend > 0.75; instead overlay a delta-hedged
            3M 10-delta call at 25% notional
            reported: Sharpe 1.13, maxDD -12.7%
  +premcap  size to a 1.5% premium cap rather than fixed notional (2.37x)
            reported: Sharpe 1.01 standalone
  +ptake    when the 50d leg has drifted to 70d, add a 1M 50d/25d put spread
            reported final: Sharpe 1.30, maxDD -7.0%, ~9.6% p.a.

The single most transportable claim here is fixed-premium vs fixed-notional
sizing, which is what removes the 2022 drawdown. `stage="baseline_premcap"`
isolates it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .. import engine, signals
from ..engine import Leg, Structure

STAGES = ("futures", "baseline", "trend", "premcap", "final")


@dataclass
class Params:
    tenor_days: int = 30
    long_delta: float = 0.50
    short_delta: float = 0.10
    leverage: float = 2.0
    premium_cap: float = 0.015           # 1.5% of spot
    premcap_leverage: float = 2.37
    trend_threshold: float = 0.75
    gamma_overlay_tenor_days: int = 91
    gamma_overlay_delta: float = 0.10
    gamma_overlay_size: float = 0.25
    ptake_trigger_delta: float = 0.70    # long leg has drifted to this delta
    ptake_long_delta: float = 0.50
    ptake_short_delta: float = 0.25
    daily_notional: float = 1_000_000.0 / 22.0


def build(surface, dates=None, params: Params | None = None, stage: str = "final"):
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    p = params or Params()
    dates = pd.DatetimeIndex(dates if dates is not None else surface.dates)
    spot = pd.Series({d: surface.spot(d) for d in dates})
    trend = signals.trend_score(spot).reindex(dates)

    if stage == "futures":
        return _futures_benchmark(surface, dates, p)

    use_premcap = stage in ("premcap", "final")
    use_trend = stage in ("trend", "final")
    use_ptake = stage == "final"

    cs = Structure(
        legs=(
            Leg(cp=+1, target_delta=p.long_delta, qty=+1.0, tenor_days=p.tenor_days),
            Leg(cp=+1, target_delta=p.short_delta, qty=-1.0, tenor_days=p.tenor_days),
        ),
        delta_hedge=False,          # this is a beta replacement, not a vol trade
        notional=p.daily_notional * (p.premcap_leverage if use_premcap else p.leverage),
        premium_cap=p.premium_cap if use_premcap else None,
        label="call_spread",
    )

    gamma_overlay = Structure(
        legs=(Leg(cp=+1, target_delta=p.gamma_overlay_delta, qty=+1.0,
                  tenor_days=p.gamma_overlay_tenor_days),),
        delta_hedge=True,
        notional=p.daily_notional * p.gamma_overlay_size,
        label="trend_gamma_overlay",
    )

    results: list = []
    for d in dates:
        hot = use_trend and bool(trend.get(d, np.nan) > p.trend_threshold)
        if hot:
            r = engine.run_trade(surface, d, gamma_overlay)
        else:
            r = engine.run_trade(surface, d, cs)
        if r is not None:
            results.append(r)

    if use_ptake:
        results += _profit_take_overlay(surface, dates, p)

    trades = _as_frame(results)
    daily = engine.to_daily_pnl(results, dates)
    return trades, daily, results


def _profit_take_overlay(surface, dates, p: Params):
    """Add a 1M 50d/25d put spread on days where a live long leg has drifted to 70d.

    Approximated by a spot-based proxy: a 50-delta call struck `tenor_days`
    ago has roughly reached 70 delta once spot is ~1 ATM-vol-move above the
    strike. Replace this with live per-trade delta tracking once wired to real
    chains - the proxy is the known weak point of this stage.
    """
    spot = pd.Series({d: surface.spot(d) for d in dates})
    T = p.tenor_days / 365.0
    atm_vol = pd.Series({d: surface.iv(d, surface.forward(d, T), T) for d in dates})
    drift = spot / spot.shift(p.tenor_days // 2) - 1.0
    trigger = drift > atm_vol * np.sqrt(T / 2.0)

    ps = Structure(
        legs=(
            Leg(cp=-1, target_delta=p.ptake_long_delta, qty=+1.0, tenor_days=p.tenor_days),
            Leg(cp=-1, target_delta=p.ptake_short_delta, qty=-1.0, tenor_days=p.tenor_days),
        ),
        delta_hedge=False,
        notional=p.daily_notional,
        label="profit_take_put_spread",
    )
    out = []
    for d in dates[trigger.reindex(dates).fillna(False)]:
        r = engine.run_trade(surface, d, ps)
        if r is not None:
            out.append(r)
    return out


def _futures_benchmark(surface, dates, p: Params):
    """Long 1M forward, 1/22 per day, held to maturity - the thing to beat."""
    rows, daily = [], pd.Series(0.0, index=dates)
    T = p.tenor_days / 365.0
    for i, d in enumerate(dates):
        # CALENDAR days to maturity, not `i + tenor_days` rows of the business
        # index. The row-offset version held ~42 calendar days instead of 30,
        # so ~30 trades were live at once instead of ~21 and the benchmark ran
        # at 1.40x rather than 1x - which is most of why its drawdown came in
        # at -49% against SPX's own -34%. Same bug class as the one fixed in
        # engine.run_trade; this path did not share that code.
        seg = dates[(dates >= d) & (dates <= d + pd.Timedelta(days=int(p.tenor_days)))]
        if len(seg) < 2:
            continue
        F0 = surface.forward(d, T)
        units = p.daily_notional / F0
        fwd = pd.Series({x: surface.forward(x, max(T - (x - d).days / 365.0, 1e-8)) for x in seg})
        inc = fwd.diff().fillna(0.0) * units
        daily = daily.add(inc.reindex(dates).fillna(0.0), fill_value=0.0)
        rows.append({"entry": d, "exit": seg[-1], "label": "futures",
                     "pnl": float(inc.sum()), "cost": 0.0,
                     "notional": float(p.daily_notional)})
    return pd.DataFrame(rows), daily, []


def _as_frame(results):
    if not results:
        return pd.DataFrame(columns=["entry", "exit", "label", "pnl", "cost"])
    return pd.DataFrame(
        [
            {"entry": r.entry, "exit": r.exit, "label": r.label, "pnl": r.pnl,
             "option_pnl": r.option_pnl, "hedge_pnl": r.hedge_pnl, "cost": r.cost,
             "entry_premium": r.entry_premium, "scale": r.scale,
             "notional": r.notional}
            for r in results
        ]
    )
