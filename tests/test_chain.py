"""Tests for the full-chain adapter.

The ones worth reading are the three that pin down why the chain exists at all:

  test_listed_strike_is_not_the_requested_delta   strike discretisation
  test_spread_profile_separates_wings_from_body   the 0.25 vol-point guess
  test_engine_charges_the_quoted_spread           it actually reaches the P&L

Everything else is guarding a specific way the adapter can silently lie.
"""

import numpy as np
import pandas as pd
import pytest

from volsurface import blackscholes as bs
from volsurface import chain, data, engine
from volsurface.chain import ChainFilters, OptionChainSurface
from volsurface.engine import Costs, Leg, Structure


@pytest.fixture(scope="module")
def surf():
    return data.make_synthetic(n_days=60, seed=5)


@pytest.fixture(scope="module")
def raw(surf):
    # 5-point strikes: SPX lists those near the money on weeklies, and a
    # coarser ladder leaves the 5-delta wing further than max_delta_error
    # from any listed strike on a third of days - real, but it makes this
    # fixture test the skip path instead of the pricing path.
    return chain.make_synthetic_chain(surf, dates=surf.dates, strike_step=5.0)


@pytest.fixture(scope="module")
def prices(surf):
    return pd.DataFrame({"date": surf.dates, "close": [surf.spot(d) for d in surf.dates]})


@pytest.fixture(scope="module")
def cs(raw, prices, surf):
    return OptionChainSurface(raw, prices, rates=surf.r, divs=surf.q)


# -- normalisation ---------------------------------------------------------


def test_optionmetrics_strikes_are_divided_by_1000():
    """OM delivers strike_price in tenths of a cent. Getting this wrong does not
    raise - it prices a different option."""
    om = pd.DataFrame({
        "date": ["2024-01-02"] * 2, "exdate": ["2024-01-19"] * 2,
        "cp_flag": ["C", "P"], "strike_price": [4800000.0, 4700000.0],
        "best_bid": [10.0, 8.0], "best_offer": [10.5, 8.4],
        "volume": [100, 50], "open_interest": [1000, 900],
    })
    out = chain.normalize_optionmetrics(om)
    assert list(out["K"]) == [4800.0, 4700.0]
    assert list(out["cp"]) == [1, -1]


def test_already_scaled_strikes_are_not_divided_twice():
    om = pd.DataFrame({
        "date": ["2024-01-02"], "exdate": ["2024-01-19"], "cp_flag": ["C"],
        "strike_price": [4800.0], "best_bid": [10.0], "best_offer": [10.5],
    })
    assert chain.normalize_optionmetrics(om)["K"].iloc[0] == 4800.0


# -- quote hygiene ---------------------------------------------------------


def test_zero_bid_rows_are_dropped(raw, prices, surf, cs):
    """A zero bid is the absence of a market, not a cheap option. Keeping those
    rows lets a backtest sell wings into a bid that was never there.

    Note they are dropped twice over: `require_two_sided` screens them out, and
    a zero bid is below intrinsic so it would fail vol inversion anyway. The
    explicit filter still earns its place - with `reimply=False` the vendor IV
    column would carry those rows straight through.
    """
    poisoned = raw.copy()
    poisoned.loc[poisoned.index[:200], "bid"] = 0.0
    dirty = OptionChainSurface(poisoned, prices, rates=surf.r, divs=surf.q)
    assert (dirty.quotes["bid"] > 0).all()
    assert len(dirty.quotes) < len(cs.quotes)


def test_crossed_quotes_are_dropped(raw, prices, surf):
    poisoned = raw.copy()
    i = poisoned.index[:50]
    poisoned.loc[i, "ask"] = poisoned.loc[i, "bid"] - 0.05   # locked/crossed tick
    cs = OptionChainSurface(poisoned, prices, rates=surf.r, divs=surf.q)
    assert (cs.quotes["ask"] > cs.quotes["bid"]).all()


def test_open_interest_filter_bites(raw, prices, surf):
    loose = OptionChainSurface(raw, prices, rates=surf.r, divs=surf.q)
    tight = OptionChainSurface(
        raw, prices, rates=surf.r, divs=surf.q, filters=ChainFilters.liquid()
    )
    assert len(tight.quotes) < len(loose.quotes)
    assert tight.quotes["open_interest"].min() >= 100.0


# -- inversion -------------------------------------------------------------

def test_reimplied_vol_recovers_the_generating_surface(cs, raw):
    """Round trip: vol -> price -> tick rounding -> vol. Tolerance is loose
    because the tick rounding is real information loss, not solver error."""
    m = raw.merge(cs.quotes[["date", "expiry", "K", "cp", "mid_vol"]],
                  on=["date", "expiry", "K", "cp"])
    err = (m["mid_vol"] - m["true_mid_vol"]).abs()
    assert err.median() < 1e-3
    assert err.quantile(0.99) < 2e-2


def test_implied_vol_vec_matches_the_scalar_solver():
    F, K, T, cp = 3000.0, 3200.0, 0.08, 1.0
    for sig in (0.08, 0.15, 0.40, 1.10):
        px = float(bs.price(F, K, T, sig, cp, 0.997))
        vec = float(bs.implied_vol_vec(px, F, K, T, cp, 0.997))
        assert vec == pytest.approx(bs.implied_vol(px, F, K, T, cp, 0.997), abs=1e-8)
        assert vec == pytest.approx(sig, abs=1e-8)


def test_implied_vol_vec_is_nan_below_intrinsic():
    # Call worth less than F-K cannot be inverted; must not return a number.
    assert np.isnan(float(bs.implied_vol_vec(1.0, 3000.0, 2000.0, 0.5, 1.0)))


# -- the point of the whole module ----------------------------------------


def test_listed_strike_is_not_the_requested_delta(cs):
    """A '5-delta call' on a real chain is the nearest LISTED strike, whose delta
    is not 5. The standardised surface hands back an exact 5.00 that was never
    tradeable, and that gap is the fidelity this adapter buys."""
    T = 10 / 365
    errs = []
    for d in cs.dates[5:45]:
        K = cs.strike_for_delta(d, T, 0.05, +1)
        q = cs.quote(d, K, cs.resolve_tenor(d, 10) / 365, +1)
        got = abs(float(bs.delta(cs.forward(d, q.T), q.K, q.T, q.mid_vol, +1)))
        errs.append(abs(got - 0.05))
    errs = np.array(errs)
    assert errs.max() > 1e-3, "listed strikes should not land exactly on 5 delta"
    assert errs.max() < 0.03, "but should stay inside max_delta_error"


def test_spread_profile_separates_wings_from_body(cs):
    """`spread_profile` must resolve a real difference between wing and body cost.

    NOTE the assertion is relative, deliberately. An earlier version of this test
    asserted `wing > 0.25` - that the wings cost more than the flat default - and
    it passed only because THIS FIXTURE IS NOT CALIBRATED: tick rounding on a
    $0.35 synthetic option manufactures a 57% spread. Measured on real SPX the
    0-5 delta median is 0.18 vol points, BELOW the 0.25 default. Do not reinstate
    an absolute threshold here; the fixture cannot support one.
    """
    prof = cs.spread_profile()
    wing = prof.loc[prof.index[0], "median_vol_points"]      # 0-5 delta bucket
    body = prof.loc[prof.index[3], "median_vol_points"]      # 25-50 delta bucket
    assert wing > body * 2


def test_spread_widens_monotonically_into_the_wings(cs):
    med = cs.spread_profile()["median_vol_points"].dropna()
    assert med.is_monotonic_decreasing, f"expected wings widest, got\n{med}"


# -- engine integration ----------------------------------------------------


def _wing(**kw):
    return Structure(legs=(Leg(+1, 0.05, -1.0, 10),), delta_hedge=True, label="wing", **kw)


def test_engine_charges_the_quoted_spread(cs):
    """Same trade, two cost models. The quoted one must differ - if it matched
    the assumption we would have wired the chain in and changed nothing."""
    d = cs.dates[10]
    engine.use_costs(Costs(vol_points=0.25, spot_bps=0.5))
    quoted = engine.run_trade(cs, d, _wing())
    engine.use_costs(Costs.assumed(0.25, 0.5))
    flat = engine.run_trade(cs, d, _wing())
    engine.use_costs(Costs())

    assert quoted is not None and flat is not None
    assert quoted.cost != pytest.approx(flat.cost, rel=1e-6)
    # > 0.25 holds on THIS fixture (uncalibrated wings); on real SPX the 5-delta
    # median is 0.18. The point of the assertion is that the quoted path reports a
    # measured number at all, not that wings are always dearer than the default.
    assert quoted.half_spread_vol_points > 0.25
    assert flat.half_spread_vol_points == pytest.approx(0.25)


def test_quoted_spread_multiplier_scales_cost(cs):
    d = cs.dates[10]
    engine.use_costs(Costs(quoted_spread_mult=1.0))
    full = engine.run_trade(cs, d, _wing())
    engine.use_costs(Costs(quoted_spread_mult=0.5))
    half = engine.run_trade(cs, d, _wing())
    engine.use_costs(Costs())
    # Hedge cost is unaffected by the option-spread multiplier, so this is not
    # an exact halving - only the option component scales.
    assert half.cost < full.cost


def test_costs_zero_is_actually_zero_on_a_quoted_surface(cs):
    """The trap: Costs.zero() must not fall through to 'charge whatever the chain
    quoted'. Every no-cost reproduction of the published numbers depends on it."""
    d = cs.dates[10]
    engine.use_costs(Costs.zero())
    r = engine.run_trade(cs, d, _wing())
    engine.use_costs(Costs())
    assert r.cost == pytest.approx(0.0, abs=1e-9)


def test_trade_exits_on_the_listed_expiry(cs):
    """Held-to-expiry must mean the real expiry date, in calendar days."""
    for d in cs.dates[5:25]:
        r = engine.run_trade(cs, d, _wing())
        if r is None:
            continue
        assert (r.exit - r.entry).days <= r.tenor_days
        assert r.tenor_days == cs.resolve_tenor(d, 10)


def test_holding_period_is_calendar_days_not_index_rows(surf):
    """Regression: the path used to be sliced as `dates[i:i+tenor+1]`, i.e. N
    BUSINESS days, so a 10-day option was marked and hedged four calendar days
    past its own expiry at intrinsic."""
    engine.use_costs(Costs.zero())
    r = engine.run_trade(surf, surf.dates[30], _wing())
    engine.use_costs(Costs())
    assert (r.exit - r.entry).days <= 10


def test_no_listed_expiry_in_tolerance_skips_the_day(surf, prices):
    """With only monthly expiries, a 10DTE strategy must decline most days rather
    than substitute a 30-day option and back-fill liquidity that never existed."""
    monthly = chain.make_synthetic_chain(
        surf, dates=surf.dates, expiry_weekdays=(4,), min_dte=2, max_dte=60,
        strike_step=5.0,
    )
    # Keep only one expiry per month, so most days have no weekly nearby.
    keep = monthly["expiry"].dt.day <= 7
    cs = OptionChainSurface(monthly[keep], prices, rates=surf.r, divs=surf.q)

    loose = [engine.run_trade(cs, d, _wing()) for d in cs.dates[5:40]]
    strict = [engine.run_trade(cs, d, _wing(tenor_tolerance_days=2)) for d in cs.dates[5:40]]
    assert sum(r is not None for r in strict) < sum(r is not None for r in loose)


def test_unlistable_wing_is_skipped_not_approximated(cs):
    """Ask for a delta no listed strike is near; the day must be skipped."""
    d = cs.dates[10]
    impossible = Structure(
        legs=(Leg(+1, 0.001, -1.0, 10),), delta_hedge=True,
        max_delta_error=0.0005, label="impossible",
    )
    assert engine.run_trade(cs, d, impossible) is None


def test_chain_surface_satisfies_the_protocol(cs):
    """Nothing downstream should be able to tell the two adapters apart."""
    d = cs.dates[10]
    assert isinstance(cs.spot(d), float)
    assert cs.forward(d, 0.1) > 0
    assert 0 < cs.discount(d, 0.1) <= 1
    assert 0.01 < cs.iv(d, cs.spot(d), 10 / 365) < 3.0


def test_iv_is_flat_outside_the_quoted_strike_range(cs):
    """Extrapolating a wing manufactures P&L that is not in the data."""
    d = cs.dates[10]
    F = cs.forward(d, 10 / 365)
    far = cs.iv(d, F * 5.0, 10 / 365)
    edge = cs.iv(d, F * 1.30, 10 / 365)
    assert far == pytest.approx(edge)


# -- IvyDB dialect ---------------------------------------------------------


def _ivydb_rows():
    """A raw IvyDB `option_price` frame, in ITS spelling - not the WRDS one."""
    return pd.DataFrame({
        "date": ["2026-09-14"] * 2,
        "expiration": ["2026-09-22"] * 2,
        "callput": ["C", "P"],
        "strike": [7820000.0, 7400000.0],      # tenths of a cent, as delivered
        "bestbid": [2.35, 1.10],
        "bestoffer": [2.45, 1.25],
        "volume": [138, 44],
        "openinterest": [55, 900],
        "impliedvolatility": [0.102206, 0.1551],
    })


def test_ivydb_dialect_is_not_the_wrds_dialect():
    """The two vendors' own spellings differ; the WRDS mapper must not silently
    half-read an IvyDB frame."""
    raw = _ivydb_rows()
    out = chain.normalize_ivydb(raw)
    assert list(out["K"]) == [7820.0, 7400.0]          # /1000 applied
    assert list(out["cp"]) == [1, -1]
    assert list(out["bid"]) == [2.35, 1.10]
    assert list(out["ask"]) == [2.45, 1.25]
    assert list(out["open_interest"]) == [55.0, 900.0]
    assert "vendor_iv" in out

    with pytest.raises(KeyError):
        chain.normalize_optionmetrics(raw)   # wrong mapper must fail loudly


def test_ivydb_strikes_not_divided_twice():
    raw = _ivydb_rows()
    raw["strike"] = [7820.0, 7400.0]         # already rescaled
    assert list(chain.normalize_ivydb(raw)["K"]) == [7820.0, 7400.0]


def test_ivydb_frame_drives_the_adapter_end_to_end(surf):
    """A normalised IvyDB frame must be accepted by OptionChainSurface unchanged."""
    raw = chain.make_synthetic_chain(surf, dates=surf.dates[:10], strike_step=5.0)
    ivy = raw.rename(columns={                # pretend it arrived in IvyDB spelling
        "expiry": "expiration", "K": "strike", "bid": "bestbid", "ask": "bestoffer",
        "open_interest": "openinterest",
    })
    ivy["callput"] = np.where(ivy["cp"] > 0, "C", "P")
    ivy["strike"] = ivy["strike"] * 1000.0
    back = chain.normalize_ivydb(ivy)
    assert back["K"].equals(raw["K"])
    assert back["cp"].tolist() == raw["cp"].tolist()
