"""Regime signals used as filters and overlays.

Three of the JPM signals are proprietary (Cross Asset Trend, Mean Reversion
Score, CARV). The substitutes here are documented so results are comparable in
shape but nobody mistakes them for the originals:

  trend_score      ~ JPM Cross Asset Trend. Blended multi-horizon momentum
                     z-score squashed to [-1, 1]. Their cutoffs (0.75 extreme
                     positive, 0.35 mild) are on the same scale by construction
                     but will not map one-for-one.
  mr_score         ~ JPM Mean Reversion Score. Short-horizon reversal z-score.

The other three are fully observable and should replicate exactly:
  term_structure, vix_regime, earnings_window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def term_structure(surface, dates, short_days: int = 30, long_days: int = 91) -> pd.Series:
    """ATM 3M vol minus ATM 1M vol, in vol points. Negative = inverted."""
    out = {}
    for d in dates:
        try:
            Ts, Tl = short_days / 365.0, long_days / 365.0
            s = surface.iv(d, surface.forward(d, Ts), Ts)
            l = surface.iv(d, surface.forward(d, Tl), Tl)
            out[d] = (l - s) * 100.0
        except (KeyError, ValueError):
            continue
    return pd.Series(out, dtype=float)


def ts_regime(ts: pd.Series) -> pd.Series:
    """JPM's three buckets: peaceful (TS>1), nervous (0<TS<1), distressed (TS<0)."""
    return pd.cut(
        ts, bins=[-np.inf, 0.0, 1.0, np.inf], labels=["distressed", "nervous", "peaceful"]
    ).astype(object)


def vix_regime(vix: pd.Series, low: float = 15.0, high: float = 21.0) -> pd.Series:
    return pd.cut(vix, bins=[-np.inf, low, high, np.inf], labels=["low", "mid", "high"]).astype(object)


def vix_notional_weight(vix: pd.Series, long_below=15.0, flat_at=21.0, max_short_at=30.0) -> pd.Series:
    """JPM's long-dated-vega sizing rule, as a signed weight in [-1, 1].

    100% long below `long_below`, ramping linearly to flat at `flat_at`, then
    ramping short to -100% at `max_short_at`.
    """
    w = pd.Series(index=vix.index, dtype=float)
    w[vix <= long_below] = 1.0
    mid = (vix > long_below) & (vix <= flat_at)
    w[mid] = 1.0 - (vix[mid] - long_below) / (flat_at - long_below)
    hi = vix > flat_at
    w[hi] = -np.clip((vix[hi] - flat_at) / (max_short_at - flat_at), 0.0, 1.0)
    return w


def trend_score(spot: pd.Series, windows=(21, 63, 126, 252)) -> pd.Series:
    """Proxy for the JPM Cross Asset Trend signal, squashed to [-1, 1]."""
    parts = []
    for w in windows:
        r = spot.pct_change(w)
        parts.append((r - r.rolling(252, min_periods=60).mean()) / r.rolling(252, min_periods=60).std())
    z = pd.concat(parts, axis=1).mean(axis=1)
    return np.tanh(z).rename("trend")


def mr_score(spot: pd.Series, window: int = 10) -> pd.Series:
    """Proxy for the JPM Mean Reversion Score. Positive = stretched, expect reversal."""
    r = spot.pct_change()
    z = (spot / spot.rolling(window).mean() - 1.0) / r.rolling(63, min_periods=20).std()
    return np.tanh(z).rename("mr")


def earnings_window(dates, start_month_day=15, months=(1, 4, 7, 10), weeks: int = 5) -> pd.Series:
    """JPM's definition: trades from the 15th of Jan/Apr/Jul/Oct, for five weeks."""
    dates = pd.DatetimeIndex(dates)
    flag = pd.Series(False, index=dates)
    for year in sorted({d.year for d in dates}):
        for m in months:
            try:
                s = pd.Timestamp(year=year, month=m, day=start_month_day)
            except ValueError:
                continue
            flag.loc[(dates >= s) & (dates < s + pd.Timedelta(weeks=weeks))] = True
    return flag


def drawdown_from_peak(spot: pd.Series) -> pd.Series:
    return spot / spot.cummax() - 1.0
