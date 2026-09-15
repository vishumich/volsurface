"""Market data interfaces.

Everything downstream talks to a `VolSurface`, never to a vendor file
directly. Swapping OptionMetrics for OPRA chains or an internal mark
should mean writing one adapter and changing nothing else.

Two adapters ship here:

  OptionMetricsSurface - the standardised volatility-surface file
      (securityid, date, days, delta, cp_flag, impl_volatility). This is a
      DELTA grid, not a strike grid. Reading vol off it at a fixed strike
      requires converting delta -> strike per tenor and interpolating in
      log-moneyness; see `_build_day`.

  SyntheticSurface - a parametric skewed surface used by the tests and the
      smoke script so the whole pipeline runs with no vendor data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from . import blackscholes as bs


class VolSurface(Protocol):
    """Read-only view of the market on a set of dates."""

    dates: pd.DatetimeIndex

    def spot(self, date: pd.Timestamp) -> float: ...

    def forward(self, date: pd.Timestamp, T: float) -> float: ...

    def discount(self, date: pd.Timestamp, T: float) -> float: ...

    def iv(self, date: pd.Timestamp, K: float, T: float) -> float:
        """Implied vol at a FIXED strike and tenor. This is the mark used for
        daily P&L, so it must interpolate in strike space, not delta space."""
        ...


# --------------------------------------------------------------------------
# Synthetic surface (tests / smoke runs)
# --------------------------------------------------------------------------


@dataclass
class SyntheticSurface:
    """Skewed surface: iv(k, T) = atm(T) + skew*k/sqrt(T) + curv*k^2, k = ln(K/F).

    `skew` is negative for an equity-like downside skew. The ATM level follows
    a supplied path so tests can drive vol regimes deliberately.
    """

    dates: pd.DatetimeIndex
    spot_path: pd.Series
    atm_path: pd.Series
    skew: float = -0.35
    curv: float = 0.60
    term_slope: float = 0.02  # ATM vol added per sqrt(year) of tenor
    r: float = 0.04
    q: float = 0.015

    def spot(self, date):
        return float(self.spot_path.loc[date])

    def discount(self, date, T):
        return float(np.exp(-self.r * T))

    def forward(self, date, T):
        return self.spot(date) * float(np.exp((self.r - self.q) * T))

    def _atm(self, date, T):
        return float(self.atm_path.loc[date]) + self.term_slope * np.sqrt(max(T, 1e-8))

    def iv(self, date, K, T):
        F = self.forward(date, T)
        k = float(np.log(K / F))
        sig = self._atm(date, T) + self.skew * k / np.sqrt(max(T, 1e-8)) + self.curv * k**2
        return float(np.clip(sig, 0.03, 3.0))


def make_synthetic(
    n_days: int = 1500, seed: int = 0, start: str = "2017-01-03"
) -> SyntheticSurface:
    """GBM spot with a mean-reverting, spot-correlated vol path.

    Not calibrated to anything - its only job is to exercise the code paths
    and give the unit tests a surface with real skew on it.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    v = np.empty(n_days)
    s = np.empty(n_days)
    v[0], s[0] = 0.15, 3000.0
    for i in range(1, n_days):
        z = rng.standard_normal()
        v[i] = np.clip(v[i - 1] + 0.05 * (0.16 - v[i - 1]) / 252 - 0.9 * 0.02 * z + 0.015 * rng.standard_normal(), 0.06, 1.2)
        s[i] = s[i - 1] * np.exp((0.06 - 0.5 * v[i - 1] ** 2) / 252 + v[i - 1] * z / np.sqrt(252))
    return SyntheticSurface(
        dates=dates,
        spot_path=pd.Series(s, index=dates),
        atm_path=pd.Series(v, index=dates),
    )


# --------------------------------------------------------------------------
# OptionMetrics adapter
# --------------------------------------------------------------------------


class OptionMetricsSurface:
    """Adapter over the OptionMetrics standardised vol surface + security price files.

    Expected frames (column names as delivered by WRDS):

      surface: date, days, delta, cp_flag, impl_volatility
               `delta` is signed x100 (e.g. -20 is a 20-delta put)
      prices:  date, close
      rates:   date, days, rate   (continuously compounded, annual)
      divs:    date, rate         (continuous dividend yield)

    Interpolation is linear in log-moneyness within a tenor and linear in
    sqrt(T) across tenors. Outside the delta grid the surface is held flat -
    deliberately, because extrapolating wings is how you manufacture P&L that
    is not in the data.
    """

    def __init__(self, surface: pd.DataFrame, prices: pd.DataFrame, rates=None, divs=None):
        self._px = prices.set_index("date")["close"].sort_index()
        self.dates = pd.DatetimeIndex(self._px.index)
        self._r = 0.04 if rates is None else rates
        self._q = 0.015 if divs is None else divs
        self._days = {}
        for (d, days), grp in surface.groupby(["date", "days"], sort=True):
            self._days.setdefault(pd.Timestamp(d), {})[int(days)] = self._build_day(
                pd.Timestamp(d), int(days), grp
            )

    def _flat_rate(self, x, date, T):
        if isinstance(x, (int, float)):
            return float(x)
        return float(x.set_index("date")["rate"].asof(date))

    def spot(self, date):
        return float(self._px.asof(date))

    def discount(self, date, T):
        return float(np.exp(-self._flat_rate(self._r, date, T) * T))

    def forward(self, date, T):
        r = self._flat_rate(self._r, date, T)
        q = self._flat_rate(self._q, date, T)
        return self.spot(date) * float(np.exp((r - q) * T))

    def _build_day(self, date, days, grp):
        """Convert one (date, tenor) delta slice into a log-moneyness -> vol curve."""
        T = days / 365.0
        F = self.forward(date, T)
        ks, vols = [], []
        for _, row in grp.iterrows():
            sig = float(row["impl_volatility"])
            cp = 1 if str(row["cp_flag"]).upper().startswith("C") else -1
            tgt = abs(float(row["delta"])) / 100.0
            if not 0.0 < tgt < 1.0 or not np.isfinite(sig):
                continue
            # Invert this grid point's own vol to its strike.
            K = bs.strike_from_delta(F, T, tgt, cp, lambda _K, s=sig: s)
            ks.append(np.log(K / F))
            vols.append(sig)
        order = np.argsort(ks)
        return T, np.asarray(ks)[order], np.asarray(vols)[order]

    def iv(self, date, K, T):
        day = self._days.get(pd.Timestamp(date))
        if not day:
            raise KeyError(f"no surface for {date}")
        tenors = sorted(day)
        sqrtT = np.sqrt(max(T, 1e-8))
        vals = []
        for d in tenors:
            Ti, ks, vols = day[d]
            Fi = self.forward(date, Ti)
            k = np.log(K / Fi)
            vals.append(float(np.interp(k, ks, vols)))  # flat outside the grid
        grid = np.sqrt([day[d][0] for d in tenors])
        return float(np.interp(sqrtT, grid, vals))
