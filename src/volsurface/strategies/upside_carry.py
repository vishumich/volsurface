"""Idea #1 - regime-conditioned short upside wings (SPX).

JPM "safer carry" construction, section 3.1.6:

  base     short 10DTE 5-delta calls, entered daily, delta-hedged at close,
           held to expiry
  overlay1 when the 1M/3M term structure INVERTS, add long 10DTE 25-delta
           calls, delta-hedged
  overlay2 when the trend signal is extreme positive (>0.75), add long 1M
           10-delta calls, delta-hedged
  overlay3 during earnings windows, +30% participation on the base leg

Reported: Sharpe 0.91 -> 1.23, worst DD -1.04%, all before transaction costs.

Every cutoff below is a `Params` field rather than a literal, because the
cutoffs are the part of this most likely to be fitted in-sample. Run
`sensitivity.sweep` over them before believing any single configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .. import engine, signals
from ..engine import Leg, Structure


@dataclass
class Params:
    base_tenor_days: int = 10
    base_delta: float = 0.05
    ts_overlay_tenor_days: int = 10
    ts_overlay_delta: float = 0.25
    trend_overlay_tenor_days: int = 30
    trend_overlay_delta: float = 0.10
    ts_inversion_threshold: float = 0.0     # vol points; TS below this = inverted
    trend_extreme_threshold: float = 0.75
    earnings_participation: float = 0.30    # +30% notional in earnings windows
    notional: float = 1_000_000.0


def build(surface, dates=None, params: Params | None = None, overlays=True):
    """Run the strategy. Returns (trades_df, daily_pnl, legs_dict)."""
    p = params or Params()
    dates = pd.DatetimeIndex(dates if dates is not None else surface.dates)
    spot = pd.Series({d: surface.spot(d) for d in dates})

    ts = signals.term_structure(surface, dates).reindex(dates).ffill()
    trend = signals.trend_score(spot).reindex(dates)
    earn = signals.earnings_window(dates)

    inverted = ts < p.ts_inversion_threshold
    extreme_up = trend > p.trend_extreme_threshold

    legs: dict[str, list] = {}

    # --- base: short upside wing -------------------------------------------
    base = Structure(
        legs=(Leg(cp=+1, target_delta=p.base_delta, qty=-1.0, tenor_days=p.base_tenor_days),),
        delta_hedge=True,
        notional=p.notional,
        label="short_wing",
    )
    base_sizer = (lambda d: 1.0 + p.earnings_participation if bool(earn.get(d, False)) else 1.0) if overlays else None
    legs["base"] = _collect(surface, dates, base, base_sizer)

    if overlays:
        # --- overlay 1: long gamma when the curve inverts -------------------
        o1 = Structure(
            legs=(Leg(cp=+1, target_delta=p.ts_overlay_delta, qty=+1.0, tenor_days=p.ts_overlay_tenor_days),),
            delta_hedge=True,
            notional=p.notional,
            label="ts_inversion_overlay",
        )
        legs["ts_overlay"] = _collect(
            surface, dates[inverted.reindex(dates).fillna(False)], o1, None
        )

        # --- overlay 2: long gamma into extreme positive trend --------------
        o2 = Structure(
            legs=(Leg(cp=+1, target_delta=p.trend_overlay_delta, qty=+1.0, tenor_days=p.trend_overlay_tenor_days),),
            delta_hedge=True,
            notional=p.notional,
            label="trend_overlay",
        )
        legs["trend_overlay"] = _collect(
            surface, dates[extreme_up.reindex(dates).fillna(False)], o2, None
        )

    trades = pd.concat([_as_frame(v) for v in legs.values()], ignore_index=True)
    daily = engine.to_daily_pnl([t for v in legs.values() for t in v], dates)
    return trades, daily, legs


def _collect(surface, entries, struct, sizer):
    out = []
    for d in entries:
        w = 1.0 if sizer is None else float(sizer(d))
        if w == 0.0:
            continue
        from dataclasses import replace
        r = engine.run_trade(surface, d, replace(struct, notional=struct.notional * w))
        if r is not None:
            out.append(r)
    return out


def _as_frame(results):
    if not results:
        return pd.DataFrame(columns=["entry", "exit", "label", "pnl", "cost"])
    return pd.DataFrame(
        [
            {"entry": r.entry, "exit": r.exit, "label": r.label, "pnl": r.pnl,
             "option_pnl": r.option_pnl, "hedge_pnl": r.hedge_pnl, "cost": r.cost,
             "entry_vol": r.entry_vol, "entry_delta": r.entry_delta,
             "tenor_days": r.tenor_days,
             "half_spread_vol_points": r.half_spread_vol_points,
             "notional": r.notional}
            for r in results
        ]
    )
