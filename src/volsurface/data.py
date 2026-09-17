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
    """Read-only view of the market on a set of dates.

    Three methods below are OPTIONAL and duck-typed rather than required; the
    engine checks for them with `getattr` and falls back to a continuous model
    when they are absent. They exist so a quoted chain can answer questions a
    smoothed surface cannot, without every adapter having to implement them:

      quote(date, K, T, cp) -> Quote | None
          The real two-sided market on that contract. When present the engine
          charges the quoted half-spread instead of `Costs.vol_points`.

      strike_for_delta(date, T, target_delta, cp, max_delta_error) -> float
          Snap to a LISTED strike. Absent, the engine solves for a continuous
          strike on the surface, which is exact but was never tradeable. Raise
          to decline the day; the engine treats that as "not tradeable" and
          skips it.

      resolve_tenor(date, tenor_days, tol_days) -> int | None
          Snap to a LISTED expiry, returning its true calendar days. Returning
          None means no suitable expiry existed and the trade is skipped.

    `chain.OptionChainSurface` implements all three. `SyntheticSurface` and
    `OptionMetricsSurface` implement none, and behave exactly as before.
    """

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
        return float(self.atm_path.loc[date]) + self.term_slope * np.sqrt(np.maximum(T, 1e-8))

    def iv(self, date, K, T):
        """Scalar in, scalar out; arrays broadcast. The array path exists so
        `chain.make_synthetic_chain` can price a whole listed grid at once."""
        F = self.forward(date, T)
        k = np.log(np.asarray(K, dtype=float) / F)
        sqrtT = np.sqrt(np.maximum(T, 1e-8))
        sig = np.clip(self._atm(date, T) + self.skew * k / sqrtT + self.curv * k**2, 0.03, 3.0)
        return float(sig) if np.ndim(sig) == 0 else sig


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

    def __init__(self, surface: pd.DataFrame, prices: pd.DataFrame, rates=None, divs=None,
                 forwards: pd.DataFrame | None = None):
        # `forwards` (date, dte, forward) is OptionMetrics' own published curve.
        # Prefer it: their IVs and deltas were computed on THAT forward, so any
        # drift between it and a reconstructed S*exp((r-q)T) lands in the P&L.
        self._fwd = self._prep_forwards(forwards)
        px = prices.set_index("date")["close"].sort_index()
        px.index = pd.DatetimeIndex(px.index)
        self._px = px
        self.dates = pd.DatetimeIndex(px.index)
        # Prepared ONCE. The previous version re-ran set_index("date") inside
        # every rate lookup, which is O(n) per call and invisible on a scalar
        # rate - with real (date, rate) frames it dominated the whole load.
        self._r = self._prep_rate(0.04 if rates is None else rates)
        self._q = self._prep_rate(0.015 if divs is None else divs)
        self._spot_cache: dict = {}
        self._days = {}
        for (d, days), grp in surface.groupby(["date", "days"], sort=True):
            self._days.setdefault(pd.Timestamp(d), {})[int(days)] = self._build_day(
                pd.Timestamp(d), int(days), grp
            )

    @staticmethod
    def _prep_rate(x):
        """Scalar stays a float; a (date, rate) frame becomes a sorted Series."""
        if isinstance(x, (int, float, np.floating)):
            return float(x)
        s = x.set_index("date")["rate"].sort_index()
        s.index = pd.DatetimeIndex(s.index)
        return s

    def _flat_rate(self, x, date, T):
        if isinstance(x, float):
            return x
        return float(x.asof(date))

    def spot(self, date):
        hit = self._spot_cache.get(date)
        if hit is None:
            hit = float(self._px.asof(date))
            self._spot_cache[date] = hit
        return hit

    def discount(self, date, T):
        return float(np.exp(-self._flat_rate(self._r, date, T) * T))

    @staticmethod
    def _prep_forwards(forwards):
        """(date, dte, forward) -> {date: (dte_array, forward_array)}, sorted."""
        if forwards is None or len(forwards) == 0:
            return None
        out = {}
        for d, g in forwards.sort_values(["date", "dte"]).groupby("date", sort=False):
            out[pd.Timestamp(d)] = (
                g["dte"].to_numpy(dtype=float), g["forward"].to_numpy(dtype=float),
            )
        return out

    def forward(self, date, T):
        if self._fwd is not None:
            hit = self._fwd.get(pd.Timestamp(date))
            if hit is not None:
                # Linear in days-to-expiry, flat outside the quoted curve.
                return float(np.interp(T * 365.0, hit[0], hit[1]))
        r = self._flat_rate(self._r, date, T)
        q = self._flat_rate(self._q, date, T)
        return self.spot(date) * float(np.exp((r - q) * T))

    def _build_day(self, date, days, grp):
        """Convert one (date, tenor) delta slice into a log-moneyness -> vol curve.

        Each grid point carries its OWN vol, so inverting delta->strike is a
        constant-vol solve with a closed form; `strike_from_delta_flat` does it
        vectorised. The brentq path this replaced was ~10^6 iterative solves for
        a decade of surface, and less accurate (9.6e-7 vs 2.2e-15 delta error).
        """
        T = days / 365.0
        F = self.forward(date, T)
        sig = pd.to_numeric(grp["impl_volatility"], errors="coerce").to_numpy(dtype=float)
        cp = np.where(
            grp["cp_flag"].astype(str).str.upper().str.startswith("C"), 1.0, -1.0
        )
        tgt = np.abs(pd.to_numeric(grp["delta"], errors="coerce").to_numpy(dtype=float)) / 100.0

        good = np.isfinite(sig) & np.isfinite(tgt) & (tgt > 0.0) & (tgt < 1.0)
        if not good.any():
            return T, np.zeros(0), np.zeros(0)
        sig, cp, tgt = sig[good], cp[good], tgt[good]

        K = np.asarray(bs.strike_from_delta_flat(F, T, tgt, cp, sig), dtype=float)
        ks = np.log(K / F)
        order = np.argsort(ks)
        return T, ks[order], sig[order]

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
