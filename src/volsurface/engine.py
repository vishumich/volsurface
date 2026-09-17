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
    """Half-spreads paid on entry and exit.

    `vol_points` is the ASSUMED option half-spread, used when the surface cannot
    quote - which is the only option on the standardised OptionMetrics surface.
    When the surface implements `quote()` (see `chain.OptionChainSurface`) the
    real quoted half-spread is charged instead and `vol_points` is ignored;
    `quoted_spread_mult` scales it, so 1.0 means paying the quoted market, 0.5
    means getting filled halfway to mid, and 2.0 means paying through.

    Set `quoted_spread_mult=None` to force the assumed model even on a quoted
    surface. That is what makes an apples-to-apples comparison against the
    standardised-surface run possible.
    """

    vol_points: float = 0.25   # vol points of option half-spread, e.g. 0.25 = 0.25 vol
    spot_bps: float = 0.5      # bps of notional per unit of hedge turnover
    quoted_spread_mult: float | None = 1.0

    @staticmethod
    def zero() -> "Costs":
        # quoted_spread_mult=None matters: "costs off" has to mean off on a
        # quoted surface too, not "charge whatever the chain quoted".
        return Costs(0.0, 0.0, quoted_spread_mult=None)

    @staticmethod
    def assumed(vol_points: float = 0.25, spot_bps: float = 0.5) -> "Costs":
        """Ignore quotes even where they exist, and charge a flat half-spread."""
        return Costs(vol_points, spot_bps, quoted_spread_mult=None)


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
    # Only meaningful on a surface that resolves listed expiries/strikes.
    # tenor_tolerance_days=None takes whatever expiry is nearest, however far;
    # set it (e.g. 2) to skip days when no weekly near the target tenor existed,
    # instead of quietly substituting a 17-day option for a 10-day one.
    tenor_tolerance_days: int | None = None
    max_delta_error: float = 0.03     # skip if no listed strike is this close


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
    # Realised, not requested. On a listed chain these drift away from the
    # Structure's targets and the gap is the whole point of reading the chain:
    # `entry_delta` says which strike you really got, `tenor_days` which expiry,
    # `half_spread_vol_points` what the market really charged.
    entry_delta: float = float("nan")
    tenor_days: int = 0
    half_spread_vol_points: float = float("nan")


class _NoListing(Exception):
    """The chain had no expiry or strike we could actually have traded that day."""


def _fixed_strikes(surface, date, legs, tenor_tol=None, max_delta_error=0.03):
    """Resolve each leg to (leg, strike, entry_vol, days_to_expiry).

    Both resolution steps defer to the surface when it can answer:

      resolve_tenor      a listed expiry, whose true DTE is usually not the
                         requested tenor. Carrying it is what lets the mark and
                         the holding period run to the real expiry.
      strike_for_delta   a listed strike. Absent, solve for a continuous one on
                         the surface - exact, but not a strike anyone could
                         have traded.
    """
    resolve = getattr(surface, "resolve_tenor", None)
    select = getattr(surface, "strike_for_delta", None)
    out = []
    for leg in legs:
        dte = leg.tenor_days if resolve is None else resolve(date, leg.tenor_days, tenor_tol)
        if dte is None or dte <= 0:
            raise _NoListing(f"no listed expiry near {leg.tenor_days}d on {date}")
        T = dte / 365.0
        if select is None:
            F = surface.forward(date, T)
            K = bs.strike_from_delta(
                F, T, leg.target_delta, leg.cp, lambda K: surface.iv(date, K, T)
            )
        else:
            K = select(date, T, leg.target_delta, leg.cp, max_delta_error)
        out.append((leg, K, surface.iv(date, K, T), dte))
    return out


def _mark(surface, date, struck, entry_date):
    """Value the structure and return (value, net_delta) per 1.0 of structure."""
    value, net_delta = 0.0, 0.0
    elapsed = (date - entry_date).days
    for leg, K, _, dte in struck:
        T = max((dte - elapsed) / 365.0, 1e-8)
        F = surface.forward(date, T)
        df = surface.discount(date, T)
        sig = surface.iv(date, K, T)
        value += leg.qty * float(bs.price(F, K, T, sig, leg.cp, df))
        net_delta += leg.qty * float(bs.delta(F, K, T, sig, leg.cp, df))
    return value, net_delta


def _option_cost(surface, date, leg, K, sig, dte, units):
    """Cost of crossing one leg, in dollars. Real quote if there is one.

    Returns (dollars, effective_half_spread_in_vol_points). The second value is
    the diagnostic worth keeping: it is what the flat `Costs.vol_points` is
    trying to guess, measured.
    """
    T = max(dte / 365.0, 1e-8)

    mult = struct_costs.quoted_spread_mult
    if mult is not None:
        getq = getattr(surface, "quote", None)
        q = None if getq is None else getq(date, K, T, leg.cp)
        if q is not None:
            return abs(leg.qty) * q.half_spread * mult * units, q.half_spread_vol_points * mult

    # No quote available: fall back to the assumed vol half-spread through vega.
    v = float(bs.vega(surface.forward(date, T), K, T, sig, surface.discount(date, T)))
    return abs(leg.qty) * v * (struct_costs.vol_points / 100.0) * units, struct_costs.vol_points


def run_trade(surface, entry_date: pd.Timestamp, struct: Structure) -> TradeResult | None:
    """Simulate one structure from entry to exit. Returns None if it cannot be struck."""
    dates = surface.dates
    if entry_date not in dates:
        return None
    try:
        struck = _fixed_strikes(
            surface, entry_date, struct.legs,
            struct.tenor_tolerance_days, struct.max_delta_error,
        )
    except (RuntimeError, KeyError, ValueError, _NoListing):
        return None

    S0 = surface.spot(entry_date)
    # Holding period is measured in CALENDAR days against the resolved expiry,
    # not in rows of the business-date index. Those differ by ~40%, so counting
    # rows held a 10-day option four calendar days past expiry and kept
    # delta-hedging it at intrinsic.
    min_tenor = min(dte for _, _, _, dte in struck)
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

    # Entry cost: quoted half-spread where the chain has one, otherwise the
    # assumed vol half-spread through vega. Plus spot cost on the initial hedge.
    cost = 0.0
    paid = []
    for leg, K, sig, dte in struck:
        c, vp = _option_cost(surface, entry_date, leg, K, sig, dte, units)
        cost += c
        paid.append(vp)

    last_day = entry_date + pd.Timedelta(days=int(horizon))
    path = dates[(dates >= entry_date) & (dates <= last_day)]
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
        elapsed = (path[-1] - entry_date).days
        for leg, K, _, dte in struck:
            left = max(dte - elapsed, 0)
            T = max(left / 365.0, 1e-8)
            sig = surface.iv(path[-1], K, T)
            c, vp = _option_cost(surface, path[-1], leg, K, sig, left, units)
            cost += c
            paid.append(vp)

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
        entry_vol=float(np.mean([s for _, _, s, _ in struck])),
        scale=scale,
        daily=pd.Series(daily, dtype=float),
        entry_delta=float(np.mean([
            abs(float(bs.delta(
                surface.forward(entry_date, dte / 365.0), K, dte / 365.0, sig, leg.cp,
                surface.discount(entry_date, dte / 365.0),
            )))
            for leg, K, sig, dte in struck
        ])),
        tenor_days=int(min(dte for _, _, _, dte in struck)),
        half_spread_vol_points=float(np.mean(paid)) if paid else 0.0,
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
                "entry_delta": r.entry_delta, "tenor_days": r.tenor_days,
                "half_spread_vol_points": r.half_spread_vol_points,
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
