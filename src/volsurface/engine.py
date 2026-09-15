"""Trade lifecycle engine: open a structure, mark it daily, delta hedge, close.

Design notes that matter for the results:

1. Strikes are fixed at entry and the daily mark reads FIXED-STRIKE vol off
   the surface. Re-striking each day would quietly convert a short-gamma
   position into a constant-delta one and remove the path dependency that the
   JPM paper argues is the whole story.

2. Delta hedging is at the close, in the forward, once per day. Every JPM
   equity backtest we are replicating hedges daily at close, so this matches.
   Intraday hedging is a separate experiment (`hedge_times` is a stub).

3. Transaction costs are ON by default. The JPM equity backtests are footnoted
   "assumes no transaction costs", and for a daily-rolled 5-delta structure
   with daily hedging that is not a rounding error. Set `costs=Costs.zero()`
   to reproduce their headline numbers, then turn them back on.

P&L is reported per unit of `notional` (default $1mm), in dollars.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from . import blackscholes as bs


@dataclass(frozen=True)
class Costs:
    """Half-spreads paid on entry and exit."""

    vol_points: float = 0.25   # vol points of option half-spread, e.g. 0.25 = 0.25 vol
    spot_bps: float = 0.5      # bps of notional per unit of hedge turnover

    @staticmethod
    def zero() -> "Costs":
        return Costs(0.0, 0.0)


@dataclass(frozen=True)
class Leg:
    cp: int                 # +1 call, -1 put
    target_delta: float     # absolute delta at entry, e.g. 0.05
    qty: float              # +1 long, -1 short; ratios are just non-unit qty
    tenor_days: int


@dataclass(frozen=True)
class Structure:
    legs: tuple[Leg, ...]
    delta_hedge: bool = True
    notional: float = 1_000_000.0
    hold_days: int | None = None      # None -> hold to the shortest leg's expiry
    premium_cap: float | None = None  # cap net premium as a % of notional; rescales qty
    unwind_at_days: int | None = None # early unwind with this many days left
    label: str = "trade"


@dataclass
class TradeResult:
    entry: pd.Timestamp
    exit: pd.Timestamp
    label: str
    pnl: float
    option_pnl: float
    hedge_pnl: float
    cost: float
    entry_premium: float
    entry_vol: float
    scale: float
    daily: pd.Series = field(repr=False, default_factory=pd.Series)


def _fixed_strikes(surface, date, legs):
    out = []
    for leg in legs:
        T = leg.tenor_days / 365.0
        F = surface.forward(date, T)
        K = bs.strike_from_delta(F, T, leg.target_delta, leg.cp, lambda K: surface.iv(date, K, T))
        out.append((leg, K, surface.iv(date, K, T)))
    return out


def _mark(surface, date, struck, entry_date):
    """Value the structure and return (value, net_delta) per 1.0 of structure."""
    value, net_delta = 0.0, 0.0
    for leg, K, _ in struck:
        T = max((leg.tenor_days - (date - entry_date).days) / 365.0, 1e-8)
        F = surface.forward(date, T)
        df = surface.discount(date, T)
        sig = surface.iv(date, K, T)
        value += leg.qty * float(bs.price(F, K, T, sig, leg.cp, df))
        net_delta += leg.qty * float(bs.delta(F, K, T, sig, leg.cp, df))
    return value, net_delta


def run_trade(surface, entry_date: pd.Timestamp, struct: Structure) -> TradeResult | None:
    """Simulate one structure from entry to exit. Returns None if it cannot be struck."""
    dates = surface.dates
    if entry_date not in dates:
        return None
    try:
        struck = _fixed_strikes(surface, entry_date, struct.legs)
    except (RuntimeError, KeyError, ValueError):
        return None

    S0 = surface.spot(entry_date)
    min_tenor = min(l.tenor_days for l in struct.legs)
    horizon = struct.hold_days if struct.hold_days is not None else min_tenor
    if struct.unwind_at_days is not None:
        horizon = min(horizon, min_tenor - struct.unwind_at_days)
    if horizon <= 0:
        return None

    entry_value, net_delta = _mark(surface, entry_date, struck, entry_date)

    # Premium cap: rescale the whole structure so |net premium| <= cap * notional.
    scale = 1.0
    if struct.premium_cap is not None and abs(entry_value) > 1e-12:
        cap_dollars = struct.premium_cap * S0
        scale = min(1.0, cap_dollars / abs(entry_value)) if abs(entry_value) > cap_dollars else 1.0
    units = struct.notional / S0 * scale

    # Entry cost: vol half-spread valued through vega, plus spot cost on the initial hedge.
    cost = 0.0
    for leg, K, sig in struck:
        T = leg.tenor_days / 365.0
        F = surface.forward(entry_date, T)
        v = float(bs.vega(F, K, T, sig, surface.discount(entry_date, T)))
        cost += abs(leg.qty) * v * (struct_costs.vol_points / 100.0) * units

    idx = dates.searchsorted(entry_date)
    path = dates[idx : idx + horizon + 1]
    if len(path) < 2:
        return None

    hedge_pos = -net_delta * units if struct.delta_hedge else 0.0
    if struct.delta_hedge:
        cost += abs(hedge_pos) * S0 * struct_costs.spot_bps / 1e4

    prev_value = entry_value
    prev_spot = S0
    option_pnl = hedge_pnl = 0.0
    daily = {}

    for d in path[1:]:
        value, net_delta = _mark(surface, d, struck, entry_date)
        S = surface.spot(d)
        option_pnl += (value - prev_value) * units
        hedge_pnl += hedge_pos * (S - prev_spot)
        if struct.delta_hedge:
            target = -net_delta * units
            turnover = abs(target - hedge_pos)
            cost += turnover * S * struct_costs.spot_bps / 1e4
            hedge_pos = target
        daily[d] = option_pnl + hedge_pnl - cost
        prev_value, prev_spot = value, S

    # Exit cost on the option legs (waived if held to expiry - it settles).
    held_to_expiry = struct.unwind_at_days is None and struct.hold_days is None
    if not held_to_expiry:
        for leg, K, _ in struck:
            T = max((leg.tenor_days - (path[-1] - entry_date).days) / 365.0, 1e-8)
            F = surface.forward(path[-1], T)
            sig = surface.iv(path[-1], K, T)
            v = float(bs.vega(F, K, T, sig, surface.discount(path[-1], T)))
            cost += abs(leg.qty) * v * (struct_costs.vol_points / 100.0) * units

    total = option_pnl + hedge_pnl - cost
    return TradeResult(
        entry=entry_date,
        exit=path[-1],
        label=struct.label,
        pnl=total,
        option_pnl=option_pnl,
        hedge_pnl=hedge_pnl,
        cost=cost,
        entry_premium=entry_value * units,
        entry_vol=float(np.mean([s for _, _, s in struck])),
        scale=scale,
        daily=pd.Series(daily, dtype=float),
    )


# Module-level cost setting, swapped by `use_costs`. Keeps the hot path free of
# threading a Costs object through every helper.
struct_costs = Costs()


def use_costs(costs: Costs):
    global struct_costs
    struct_costs = costs


def run_schedule(surface, entries, struct: Structure, sizer=None) -> pd.DataFrame:
    """Run one structure repeatedly over an entry schedule.

    `sizer(date) -> float` scales notional per entry (0.0 skips the date), which
    is how the regime filters and participation overlays are applied.
    """
    rows = []
    for d in entries:
        w = 1.0 if sizer is None else float(sizer(d))
        if w == 0.0:
            continue
        r = run_trade(surface, d, replace(struct, notional=struct.notional * w))
        if r is not None:
            rows.append(r)
    if not rows:
        return pd.DataFrame(columns=["entry", "exit", "label", "pnl", "option_pnl", "hedge_pnl", "cost"])
    return pd.DataFrame(
        [
            {
                "entry": r.entry, "exit": r.exit, "label": r.label, "pnl": r.pnl,
                "option_pnl": r.option_pnl, "hedge_pnl": r.hedge_pnl, "cost": r.cost,
                "entry_premium": r.entry_premium, "entry_vol": r.entry_vol, "scale": r.scale,
            }
            for r in rows
        ]
    )


def to_daily_pnl(trades: list[TradeResult], dates: pd.DatetimeIndex) -> pd.Series:
    """Aggregate overlapping trades into one daily P&L series.

    Overlap is the point: these strategies enter every day and hold to expiry,
    so up to `tenor` trades are live at once. Summing trade-level P&L instead
    of daily marks understates the drawdown badly.
    """
    total = pd.Series(0.0, index=dates)
    for t in trades:
        inc = t.daily.diff()
        if len(t.daily):
            inc.iloc[0] = t.daily.iloc[0]
        total = total.add(inc.reindex(dates).fillna(0.0), fill_value=0.0)
    return total
