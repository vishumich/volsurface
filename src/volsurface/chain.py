"""Full option-chain adapter: real quoted bid-ask instead of an assumed half-spread.

`OptionMetricsSurface` in `data.py` reads the *standardised* surface - a smoothed
delta grid with no quotes on it, so the only way to charge a transaction cost
there is to assume one (`Costs.vol_points`). That assumption is doing real work:
for a 5-delta wing rolled daily it is plausibly the whole P&L. This module reads
the raw chain instead, so three things stop being assumptions:

1. **Cost.** The half-spread is whatever was quoted on that strike that day, not
   a constant. Wing spreads are several times ATM spreads and widen in stress -
   exactly when a short-wing book wants to de-risk - so a constant understates
   cost precisely where it matters most.

2. **Strike.** A listed chain has discrete strikes. A "5-delta call" is the
   *listed* strike nearest 5 delta, which on SPX 25-point strikes at 10DTE can be
   4.1 or 6.4 delta. The standardised surface hands you an exact 5.00 that was
   never tradeable.

3. **Expiry.** 10DTE has to be a real listed expiry. Assuming one always exists
   back-fills a weekly grid that did not exist pre-2022, which is the single
   easiest way to manufacture a result out of liquidity that was not there.

The adapter satisfies the same `VolSurface` protocol, plus the optional hooks the
engine looks for (`quote`, `strike_for_delta`, `resolve_tenor`) when the surface
can answer them. Nothing downstream changes.

Vols are re-implied from quoted prices with this package's own forward-based
Black-Scholes rather than taken from OptionMetrics' `impl_volatility` column.
That is deliberate: the engine marks with `bs.price` on forwards, so the vol it
marks with must be the one that reproduces the quote under *that* model. OM
solves a binomial on spot with discrete dividends; mixing the two puts a small
persistent wedge into every mark. Pass `reimply=False` to trust the vendor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import blackscholes as bs

# Columns the adapter works in, after normalisation.
_COLS = ["date", "expiry", "cp", "K", "bid", "ask", "volume", "open_interest"]


# --------------------------------------------------------------------------
# Quote
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Quote:
    """One listed contract on one date, as quoted."""

    date: pd.Timestamp
    expiry: pd.Timestamp
    K: float
    T: float
    cp: int
    bid: float
    ask: float
    bid_vol: float
    ask_vol: float
    volume: float
    open_interest: float

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def mid_vol(self) -> float:
        return 0.5 * (self.bid_vol + self.ask_vol)

    @property
    def half_spread(self) -> float:
        """Cost of crossing, in price terms, per 1.0 of contract."""
        return 0.5 * (self.ask - self.bid)

    @property
    def half_spread_vol_points(self) -> float:
        """The same cost in vol points, directly comparable to `Costs.vol_points`.
        This is the number the 0.25 assumption is guessing at."""
        return 0.5 * (self.ask_vol - self.bid_vol) * 100.0

    @property
    def rel_spread(self) -> float:
        m = self.mid
        return float("inf") if m <= 0 else (self.ask - self.bid) / m


# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ChainFilters:
    """What counts as a tradeable quote.

    Defaults are permissive except for `require_two_sided`. A zero bid is not a
    cheap option, it is the absence of a market: OptionMetrics carries those rows
    with `best_bid = 0` and a nominal offer, and keeping them lets a backtest
    sell wings into a bid that was never there. Everything else should be
    tightened per underlying and the result re-run - if a strategy only survives
    at `min_open_interest=0`, that is itself a finding.
    """

    require_two_sided: bool = True      # best_bid > 0
    min_price: float = 0.05             # below the SPX tick, treat as no market
    min_open_interest: float = 0.0
    min_volume: float = 0.0
    max_rel_spread: float = 2.0         # (ask-bid)/mid; wings legitimately run wide
    iv_bounds: tuple[float, float] = (0.01, 3.0)
    min_dte: int = 1
    max_dte: int = 400

    @staticmethod
    def liquid() -> "ChainFilters":
        """Tighter screen: only strikes with real size behind them."""
        return ChainFilters(min_open_interest=100.0, min_volume=1.0, max_rel_spread=0.5)


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------


def normalize_optionmetrics(df: pd.DataFrame) -> pd.DataFrame:
    """Map a WRDS `opprcd` frame onto the adapter's column names.

    The trap is `strike_price`, which OptionMetrics delivers in **tenths of a
    cent** - a 3000 strike is stored as 3000000. Dividing by 1000 is not optional,
    and getting it wrong does not raise: it just prices a different option.
    """
    out = pd.DataFrame(index=df.index)
    out["date"] = pd.to_datetime(df["date"])
    out["expiry"] = pd.to_datetime(df["exdate"])
    out["cp"] = np.where(df["cp_flag"].astype(str).str.upper().str.startswith("C"), 1, -1)
    strike = df["strike_price"].astype(float)
    # Guard against double-dividing a frame someone already rescaled.
    out["K"] = strike / 1000.0 if strike.max() > 100_000 else strike
    out["bid"] = df["best_bid"].astype(float)
    out["ask"] = df["best_offer"].astype(float)
    zeros = pd.Series(0.0, index=df.index)
    out["volume"] = df.get("volume", zeros).astype(float)
    out["open_interest"] = df.get("open_interest", zeros).astype(float)
    if "impl_volatility" in df:
        out["vendor_iv"] = pd.to_numeric(df["impl_volatility"], errors="coerce")
    return out


def normalize_ivydb(df: pd.DataFrame) -> pd.DataFrame:
    """Map an on-prem IvyDB `option_price` frame onto the adapter's columns.

    IvyDB's own SQL Server distribution does NOT use the WRDS column spellings,
    so `normalize_optionmetrics` will not read it: `securityid` not `secid`,
    `bestbid`/`bestoffer` not `best_bid`/`best_offer`, `expiration` not `exdate`,
    `callput` not `cp_flag`, `openinterest` not `open_interest`. What does carry
    over is the trap - `strike` is still in tenths of a cent.
    """
    out = pd.DataFrame(index=df.index)
    out["date"] = pd.to_datetime(df["date"])
    out["expiry"] = pd.to_datetime(df["expiration"])
    out["cp"] = np.where(df["callput"].astype(str).str.upper().str.startswith("C"), 1, -1)
    strike = df["strike"].astype(float)
    out["K"] = strike / 1000.0 if strike.max() > 100_000 else strike
    out["bid"] = df["bestbid"].astype(float)
    out["ask"] = df["bestoffer"].astype(float)
    zeros = pd.Series(0.0, index=df.index)
    out["volume"] = df.get("volume", zeros).astype(float)
    out["open_interest"] = df.get("openinterest", zeros).astype(float)
    if "impliedvolatility" in df:
        out["vendor_iv"] = pd.to_numeric(df["impliedvolatility"], errors="coerce")
    return out


# --------------------------------------------------------------------------
# Adapter
# --------------------------------------------------------------------------


class OptionChainSurface:
    """Adapter over a full EOD option chain.

    `chain` columns (see `normalize_optionmetrics` for the WRDS mapping):
        date, expiry, cp (+1/-1), K, bid, ask, volume, open_interest
        optional: vendor_iv
    `prices`: date, close
    `rates` / `divs`: scalar, or a frame with (date, rate); continuously compounded.
    """

    def __init__(
        self,
        chain: pd.DataFrame,
        prices: pd.DataFrame,
        rates=None,
        divs=None,
        filters: ChainFilters | None = None,
        reimply: bool = True,
        forwards: pd.DataFrame | None = None,
    ):
        # (date, dte, forward): the vendor's OWN forward curve. Their quotes were
        # struck against it, so prefer it to reconstructing S*exp((r-q)T).
        self._fwd = self._prep_forwards(forwards)
        self.filters = filters or ChainFilters()
        px = prices.set_index("date")["close"].sort_index()
        px.index = pd.DatetimeIndex(px.index)
        self._px = px
        self._r = self._prep_rate(0.04 if rates is None else rates)
        self._q = self._prep_rate(0.015 if divs is None else divs)
        self._spot_cache: dict = {}

        df = self._prepare(chain, reimply)
        self.quotes = df
        self.dates = pd.DatetimeIndex(sorted(df["date"].unique()))
        self._by_date = {d: g for d, g in df.groupby("date", sort=True)}
        self._expiries = {
            d: np.asarray(sorted(g["expiry"].unique())) for d, g in self._by_date.items()
        }
        # (date, expiry) -> slice, built lazily: chains are large and a given
        # backtest touches only a small fraction of the pairs.
        self._slices: dict[tuple, pd.DataFrame] = {}

    # -- rates -------------------------------------------------------------

    @staticmethod
    def _prep_rate(x):
        """Scalar stays a float; a (date, rate) frame becomes a sorted Series."""
        if isinstance(x, (int, float, np.floating)):
            return float(x)
        s = x.set_index("date")["rate"].sort_index()
        s.index = pd.DatetimeIndex(s.index)
        return s

    def _rate_at(self, x, dates):
        """Prepared scalar or Series -> array aligned to `dates`."""
        if isinstance(x, float):
            return np.full(len(dates), x)
        return x.reindex(pd.DatetimeIndex(dates), method="ffill").to_numpy(dtype=float)

    def _scalar_rate(self, x, date):
        # Prepared in __init__, never re-indexed here: this is called once per
        # leg per marking day, and rebuilding the index inside it is O(n) each
        # time - the cost is invisible on a scalar rate and dominates on a frame.
        if isinstance(x, float):
            return x
        return float(x.asof(pd.Timestamp(date)))

    def spot(self, date):
        date = pd.Timestamp(date)
        hit = self._spot_cache.get(date)
        if hit is None:
            hit = float(self._px.asof(date))
            self._spot_cache[date] = hit
        return hit

    def discount(self, date, T):
        return float(np.exp(-self._scalar_rate(self._r, date) * T))

    @staticmethod
    def _prep_forwards(forwards):
        """(date, dte, forward) -> {date: (dte_array, forward_array)}, sorted."""
        if forwards is None or len(forwards) == 0:
            return None
        return {
            pd.Timestamp(d): (g["dte"].to_numpy(dtype=float),
                              g["forward"].to_numpy(dtype=float))
            for d, g in forwards.sort_values(["date", "dte"]).groupby("date", sort=False)
        }

    def _forward_curve(self, dates, dte):
        """Vectorised forward lookup for `_prepare`. NaN where the date is absent,
        so the caller can fall back to the r/q construction on those rows."""
        out = np.full(len(dte), np.nan)
        idx = pd.DataFrame({"date": pd.DatetimeIndex(dates)}).groupby("date").indices
        for d, rows in idx.items():
            hit = self._fwd.get(pd.Timestamp(d))
            if hit is not None:
                out[rows] = np.interp(dte[rows], hit[0], hit[1])
        return out

    def forward(self, date, T):
        if self._fwd is not None:
            hit = self._fwd.get(pd.Timestamp(date))
            if hit is not None:
                return float(np.interp(T * 365.0, hit[0], hit[1]))
        r = self._scalar_rate(self._r, date)
        q = self._scalar_rate(self._q, date)
        return self.spot(date) * float(np.exp((r - q) * T))

    # -- build -------------------------------------------------------------

    def _prepare(self, chain: pd.DataFrame, reimply: bool) -> pd.DataFrame:
        missing = [c for c in _COLS if c not in chain.columns]
        if missing:
            raise ValueError(f"chain is missing columns {missing}; see normalize_optionmetrics")

        df = chain.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["expiry"] = pd.to_datetime(df["expiry"])
        df["dte"] = (df["expiry"] - df["date"]).dt.days
        f = self.filters
        df = df[(df["dte"] >= f.min_dte) & (df["dte"] <= f.max_dte)]

        # Quote hygiene, before any pricing maths touches these rows.
        df = df[np.isfinite(df["bid"]) & np.isfinite(df["ask"])]
        df = df[df["ask"] > df["bid"]]                      # crossed/locked = bad tick
        if f.require_two_sided:
            df = df[df["bid"] > 0]
        mid = 0.5 * (df["bid"] + df["ask"])
        df = df[mid >= f.min_price]
        if f.min_open_interest > 0:
            df = df[df["open_interest"] >= f.min_open_interest]
        if f.min_volume > 0:
            df = df[df["volume"] >= f.min_volume]
        mid = 0.5 * (df["bid"] + df["ask"])
        df = df[((df["ask"] - df["bid"]) / mid) <= f.max_rel_spread]
        if df.empty:
            raise ValueError("every quote was filtered out; loosen ChainFilters")

        df = df.reset_index(drop=True)
        df["T"] = df["dte"] / 365.0
        S = self._px.reindex(pd.DatetimeIndex(df["date"]), method="ffill").to_numpy(dtype=float)
        r = self._rate_at(self._r, df["date"])
        q = self._rate_at(self._q, df["date"])
        T = df["T"].to_numpy()
        F = S * np.exp((r - q) * T)
        if self._fwd is not None:
            # Vendor curve where we have it, reconstruction only as a fallback.
            from_curve = self._forward_curve(df["date"], df["dte"].to_numpy(dtype=float))
            F = np.where(np.isfinite(from_curve), from_curve, F)
        df["F"] = F
        df["df"] = np.exp(-r * T)
        df["mid"] = 0.5 * (df["bid"] + df["ask"])
        df["k"] = np.log(df["K"].to_numpy() / df["F"].to_numpy())

        if reimply or "vendor_iv" not in df:
            args = (
                df["F"].to_numpy(), df["K"].to_numpy(), T,
                df["cp"].to_numpy(), df["df"].to_numpy(),
            )
            df["bid_vol"] = bs.implied_vol_vec(df["bid"].to_numpy(), *args)
            df["ask_vol"] = bs.implied_vol_vec(df["ask"].to_numpy(), *args)
            df["mid_vol"] = bs.implied_vol_vec(df["mid"].to_numpy(), *args)
        else:
            # Vendor vol carries no bid/ask, so the spread still has to come from
            # prices: widen the vendor mid by the price half-spread through vega.
            df["mid_vol"] = df["vendor_iv"]
            v = bs.vega(df["F"], df["K"], T, df["mid_vol"], df["df"])
            hs = np.where(
                v > 1e-10, 0.5 * (df["ask"] - df["bid"]) / np.maximum(v, 1e-10), np.nan
            )
            df["bid_vol"] = df["mid_vol"] - hs
            df["ask_vol"] = df["mid_vol"] + hs

        lo, hi = f.iv_bounds
        good = (
            np.isfinite(df["bid_vol"]) & np.isfinite(df["ask_vol"]) & np.isfinite(df["mid_vol"])
            & (df["mid_vol"] >= lo) & (df["mid_vol"] <= hi)
            & (df["ask_vol"] >= df["bid_vol"])
        )
        df = df[good].reset_index(drop=True)
        if df.empty:
            raise ValueError("no quote survived implied-vol inversion; check strike units (x1000?)")

        df["delta"] = np.abs(
            bs.delta(df["F"], df["K"], df["T"], df["mid_vol"], df["cp"], df["df"])
        )
        return df.sort_values(["date", "expiry", "cp", "K"]).reset_index(drop=True)

    # -- expiry / strike resolution ---------------------------------------

    def expiries(self, date) -> np.ndarray:
        return self._expiries.get(pd.Timestamp(date), np.array([], dtype="datetime64[ns]"))

    def resolve_expiry(self, date, target_days: float, tol_days: float | None = None):
        """Nearest LISTED expiry to `target_days`, or None if none is close enough.

        `tol_days=None` accepts whatever is nearest. Setting it is how you stop
        the backtest silently substituting a 17-day option for the 10-day one it
        asked for, on dates when no weekly was listed.
        """
        exp = self.expiries(date)
        if len(exp) == 0:
            return None
        d0 = np.datetime64(pd.Timestamp(date).to_datetime64(), "D")
        dte = (exp.astype("datetime64[D]") - d0).astype(int)
        err = np.abs(dte - target_days)
        j = int(np.argmin(err))
        if tol_days is not None and err[j] > tol_days:
            return None
        return pd.Timestamp(exp[j])

    def resolve_tenor(self, date, tenor_days: int, tol_days: float | None = None):
        """Engine hook: actual calendar days to the listed expiry we would really trade."""
        e = self.resolve_expiry(date, tenor_days, tol_days)
        if e is None:
            return None
        return int((e - pd.Timestamp(date)).days)

    def _slice(self, date, expiry, cp=None) -> pd.DataFrame:
        key = (pd.Timestamp(date), pd.Timestamp(expiry))
        g = self._slices.get(key)
        if g is None:
            day = self._by_date.get(key[0])
            g = (
                day[day["expiry"] == key[1]] if day is not None
                else pd.DataFrame(columns=self.quotes.columns)
            )
            self._slices[key] = g
        return g if cp is None else g[g["cp"] == cp]

    def strike_for_delta(self, date, T, target_delta, cp, max_delta_error: float = 0.03):
        """Engine hook: the LISTED strike closest to `target_delta`.

        Raises if the chain has nothing within `max_delta_error` of the target.
        `run_trade` catches that and skips the day, which is the honest outcome
        when the wing you wanted was not listed.
        """
        date = pd.Timestamp(date)
        expiry = self.resolve_expiry(date, T * 365.0)
        if expiry is None:
            raise KeyError(f"no listed expiry near {T * 365:.1f}d on {date.date()}")
        g = self._slice(date, expiry, cp)
        if g.empty:
            side = "call" if cp > 0 else "put"
            raise KeyError(f"no {side} quotes for {expiry.date()} on {date.date()}")
        err = (g["delta"] - target_delta).abs().to_numpy()
        j = int(np.argmin(err))
        if err[j] > max_delta_error:
            raise RuntimeError(
                f"closest listed strike to {target_delta:.2f}-delta is "
                f"{g['delta'].iloc[j]:.3f}-delta ({expiry.date()}, {date.date()}) - "
                f"beyond max_delta_error={max_delta_error}"
            )
        return float(g["K"].iloc[j])

    def quote(self, date, K, T, cp) -> Quote | None:
        """The actual listed contract, or None if that strike is not quoted."""
        date = pd.Timestamp(date)
        expiry = self.resolve_expiry(date, T * 365.0, tol_days=1.0)
        if expiry is None:
            return None
        g = self._slice(date, expiry, cp)
        if g.empty:
            return None
        j = int(np.argmin(np.abs(g["K"].to_numpy() - K)))
        row = g.iloc[j]
        if abs(float(row["K"]) - K) > 1e-6:
            return None                       # fixed-strike mark: exact or nothing
        return Quote(
            date=date, expiry=expiry, K=float(row["K"]), T=float(row["T"]), cp=int(row["cp"]),
            bid=float(row["bid"]), ask=float(row["ask"]),
            bid_vol=float(row["bid_vol"]), ask_vol=float(row["ask_vol"]),
            volume=float(row["volume"]), open_interest=float(row["open_interest"]),
        )

    # -- VolSurface ---------------------------------------------------------

    def _smile(self, date, expiry):
        """(T, log-moneyness, mid_vol) for the OTM instrument at each listed strike.

        OTM only: the ITM side of a listed chain is wide, thinly quoted and
        carries pin and early-exercise noise. The OTM quote at each strike is
        what a desk actually marks off.
        """
        g = self._slice(date, expiry)
        if g.empty:
            return None
        F = float(g["F"].iloc[0])
        otm = g[((g["cp"] > 0) & (g["K"] >= F)) | ((g["cp"] < 0) & (g["K"] < F))]
        if len(otm) < 2:
            otm = g
        otm = otm.sort_values("k")
        return float(g["T"].iloc[0]), otm["k"].to_numpy(), otm["mid_vol"].to_numpy()

    def iv(self, date, K, T):
        """Fixed-strike mid vol. Linear in log-moneyness, linear in sqrt(T).

        Held flat outside the quoted strike range, same as the standardised
        adapter and for the same reason: extrapolating a wing manufactures P&L
        that is not in the data.
        """
        date = pd.Timestamp(date)
        exp = self.expiries(date)
        if len(exp) == 0:
            raise KeyError(f"no chain for {date.date()}")
        smiles = [s for s in (self._smile(date, e) for e in exp) if s is not None]
        if not smiles:
            raise KeyError(f"no usable quotes on {date.date()}")

        vals, grid = [], []
        for Ti, ks, vols in smiles:
            Fi = self.forward(date, Ti)
            vals.append(float(np.interp(np.log(K / Fi), ks, vols)))
            grid.append(np.sqrt(max(Ti, 1e-8)))
        order = np.argsort(grid)
        return float(
            np.interp(np.sqrt(max(T, 1e-8)), np.asarray(grid)[order], np.asarray(vals)[order])
        )

    # -- diagnostics --------------------------------------------------------

    def spread_profile(self, by: str = "delta", bins=(0.0, 0.05, 0.10, 0.25, 0.50, 1.0)):
        """What the chain actually charges, bucketed by delta.

        Run this before trusting any constant `Costs.vol_points` - it is the
        empirical answer to "what do we pay", which the standardised surface
        cannot give you at all.
        """
        q = self.quotes
        out = pd.DataFrame({
            by: q[by],
            "half_spread_vol_points": 0.5 * (q["ask_vol"] - q["bid_vol"]) * 100.0,
            "rel_spread": (q["ask"] - q["bid"]) / q["mid"],
            "open_interest": q["open_interest"],
        })
        return out.groupby(pd.cut(out[by], bins), observed=False).agg(
            n=("half_spread_vol_points", "size"),
            median_vol_points=("half_spread_vol_points", "median"),
            p90_vol_points=("half_spread_vol_points", lambda s: s.quantile(0.90)),
            median_rel_spread=("rel_spread", "median"),
            median_oi=("open_interest", "median"),
        )


# --------------------------------------------------------------------------
# Synthetic chain (tests / smoke runs)
# --------------------------------------------------------------------------


def make_synthetic_chain(
    surface,
    dates=None,
    strike_step: float = 25.0,
    band: float = 0.14,
    expiry_weekdays: tuple[int, ...] = (0, 2, 4),
    weekly_horizon_days: int = 35,
    min_dte: int = 2,
    max_dte: int = 45,
    atm_half_spread_vol: float = 0.0020,
    wing_spread_mult: float = 8.0,
    tick: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Discretise a continuous surface onto a listed chain, with quotes.

    Exists so the chain code path runs in tests and the smoke script with no
    vendor data. Three things about it are not cosmetic:

    - **Fixed strike ladder.** Strikes are absolute levels on a global grid, as
      on a real exchange - not a fresh grid around each day's spot. That is what
      makes the delta of a "5-delta" trade wander day to day.
    - **Tick rounding.** Quotes are rounded out to `tick`, so a $0.05/$0.15 wing
      market has a 100% relative spread. This is the effect that actually kills
      cheap-wing strategies and it is completely invisible to a constant
      vol-point cost assumption.
    - **Spreads widen into the wings** as `atm * (1 + mult * |k| / sqrt(T))`,
      because that is the shape of a real surface's liquidity.

    Defaults are SPX-flavoured (25-point strikes, Mon/Wed/Fri weeklies, $0.05
    tick) but nothing is calibrated - it is a plumbing fixture, not a market.
    """
    rng = np.random.default_rng(seed)
    dates = pd.DatetimeIndex(dates if dates is not None else surface.dates)

    # Listed expiry calendar, extended past the last date so late entries still
    # have a chain in front of them. Weeklies near-dated, monthlies (third
    # Friday) beyond - which is how an index chain is actually listed, and the
    # reason a "1M" trade and a "10D" trade face very different grids.
    cal = pd.bdate_range(dates[0], dates[-1] + pd.Timedelta(days=max_dte + 7))
    weeklies = np.asarray(
        sorted(d for d in cal if d.weekday() in expiry_weekdays), dtype="datetime64[D]"
    )
    fridays = [d for d in cal if d.weekday() == 4]
    monthlies = np.asarray(
        sorted({
            d for d in fridays
            if sum(1 for x in fridays if x.year == d.year and x.month == d.month and x <= d) == 3
        }),
        dtype="datetime64[D]",
    )

    spots = np.array([surface.spot(d) for d in dates], dtype=float)
    ladder = np.arange(
        np.floor(spots.min() * (1 - band - 0.05) / strike_step) * strike_step,
        np.ceil(spots.max() * (1 + band + 0.05) / strike_step) * strike_step + strike_step,
        strike_step,
    )

    frames = []
    for d in dates:
        d64 = np.datetime64(d.to_datetime64(), "D")
        w_dte = (weeklies - d64).astype(int)
        m_dte = (monthlies - d64).astype(int)
        listed = np.union1d(
            weeklies[(w_dte >= min_dte) & (w_dte <= min(weekly_horizon_days, max_dte))],
            monthlies[(m_dte >= min_dte) & (m_dte <= max_dte)],
        )
        for expiry in listed:
            dte = int((expiry - d64).astype(int))
            T = dte / 365.0
            F = surface.forward(d, T)
            k_all = np.log(ladder / F)
            sel = np.abs(k_all) <= band
            if sel.sum() < 4:
                continue
            K = ladder[sel]
            k = k_all[sel]
            mid_vol = np.asarray(surface.iv(d, K, T), dtype=float)
            df_ = surface.discount(d, T)

            # OTM instrument at each strike: calls above the forward, puts below.
            cp = np.where(K >= F, 1, -1)
            hs = atm_half_spread_vol * (1.0 + wing_spread_mult * np.abs(k) / np.sqrt(max(T, 1e-8)))
            mid_px = bs.price(F, K, T, mid_vol, cp, df_)
            bid_px = bs.price(F, K, T, np.maximum(mid_vol - hs, 1e-4), cp, df_)
            ask_px = bs.price(F, K, T, mid_vol + hs, cp, df_)

            # Round out to the tick, then guarantee at least a one-tick market.
            bid = np.maximum(np.floor(bid_px / tick) * tick, 0.0)
            ask = np.ceil(ask_px / tick) * tick
            ask = np.maximum(ask, bid + tick)

            # Size concentrates around the money and in the front expiry.
            oi = 5000.0 * np.exp(-0.5 * (k / 0.06) ** 2) * (1.0 + 2.0 * np.exp(-dte / 10.0))
            oi = np.round(oi * rng.uniform(0.6, 1.4, size=len(K)))
            vol = np.round(oi * rng.uniform(0.0, 0.3, size=len(K)))

            frames.append(pd.DataFrame({
                "date": d, "expiry": pd.Timestamp(expiry), "cp": cp, "K": K,
                "bid": bid, "ask": ask, "volume": vol, "open_interest": oi,
                "true_mid_vol": mid_vol,
            }))

    if not frames:
        raise ValueError("synthetic chain came out empty; check dates / band / dte bounds")
    return pd.concat(frames, ignore_index=True)
