import numpy as np
import pandas as pd
import pytest

from volsurface import blackscholes as bs
from volsurface import data, engine, metrics, signals
from volsurface.engine import Costs, Leg, Structure


@pytest.fixture(scope="module")
def surf():
    return data.make_synthetic(n_days=600, seed=7)


def test_put_call_parity():
    F, K, T, sig = 100.0, 95.0, 0.25, 0.20
    c = float(bs.price(F, K, T, sig, +1))
    p = float(bs.price(F, K, T, sig, -1))
    assert c - p == pytest.approx(F - K, abs=1e-8)


def test_delta_bounds_and_sign():
    F, K, T, sig = 100.0, 100.0, 0.25, 0.2
    assert 0 < float(bs.delta(F, K, T, sig, +1)) < 1
    assert -1 < float(bs.delta(F, K, T, sig, -1)) < 0


def test_implied_vol_roundtrip():
    F, K, T, sig = 100.0, 110.0, 0.5, 0.23
    px = float(bs.price(F, K, T, sig, +1))
    assert bs.implied_vol(px, F, K, T, +1) == pytest.approx(sig, abs=1e-6)


def test_strike_from_delta_uses_surface_not_flat_vol(surf):
    """On a skewed surface the solved strike must differ from the flat-vol answer.

    This is the bug that silently mis-targets the wings: if the solver ignores
    the surface, a '5-delta put' is struck at the wrong place entirely.
    """
    d = surf.dates[100]
    T = 0.25
    F = surf.forward(d, T)
    K_surf = bs.strike_from_delta(F, T, 0.05, -1, lambda K: surf.iv(d, K, T))
    atm = surf.iv(d, F, T)
    K_flat = bs.strike_from_delta(F, T, 0.05, -1, lambda K: atm)
    assert abs(K_surf / K_flat - 1.0) > 0.005
    # and the solved strike really does have 5 delta on the surface
    got = abs(float(bs.delta(F, K_surf, T, surf.iv(d, K_surf, T), -1)))
    assert got == pytest.approx(0.05, abs=1e-4)


def test_delta_hedged_short_call_pnl_is_not_pure_premium(surf):
    """A delta-hedged short option must have path-dependent P&L, not just theta."""
    engine.use_costs(Costs.zero())
    s = Structure(legs=(Leg(+1, 0.25, -1.0, 30),), delta_hedge=True, label="t")
    res = [engine.run_trade(surf, d, s) for d in surf.dates[50:150:10]]
    res = [r for r in res if r]
    assert len(res) >= 8
    pnls = np.array([r.pnl for r in res])
    assert pnls.std() > 0
    assert not np.allclose(pnls, pnls[0])


def test_costs_reduce_pnl_monotonically(surf):
    s = Structure(legs=(Leg(+1, 0.05, -1.0, 10),), delta_hedge=True, label="t")
    d = surf.dates[200]
    engine.use_costs(Costs.zero())
    free = engine.run_trade(surf, d, s).pnl
    engine.use_costs(Costs(vol_points=0.5, spot_bps=1.0))
    charged = engine.run_trade(surf, d, s)
    assert charged.cost > 0
    assert charged.pnl < free
    engine.use_costs(Costs())


def test_premium_cap_scales_position(surf):
    engine.use_costs(Costs.zero())
    d = surf.dates[300]
    legs = (Leg(+1, 0.50, +1.0, 30), Leg(+1, 0.10, -1.0, 30))
    uncapped = engine.run_trade(surf, d, Structure(legs=legs, delta_hedge=False, label="u"))
    capped = engine.run_trade(
        surf, d, Structure(legs=legs, delta_hedge=False, premium_cap=0.001, label="c")
    )
    assert capped.scale < 1.0
    assert abs(capped.entry_premium) < abs(uncapped.entry_premium)


def test_daily_aggregation_matches_trade_totals(surf):
    """Overlapping trades: summed daily increments must equal summed trade P&L."""
    engine.use_costs(Costs.zero())
    s = Structure(legs=(Leg(+1, 0.10, -1.0, 10),), delta_hedge=True, label="t")
    trades = [engine.run_trade(surf, d, s) for d in surf.dates[100:160]]
    trades = [t for t in trades if t]
    daily = engine.to_daily_pnl(trades, surf.dates)
    assert daily.sum() == pytest.approx(sum(t.pnl for t in trades), rel=1e-9)


def test_vix_notional_weight_ramp():
    v = pd.Series([10, 15, 18, 21, 25, 30, 40], dtype=float)
    w = signals.vix_notional_weight(v)
    assert w.iloc[0] == 1.0 and w.iloc[1] == 1.0
    assert 0 < w.iloc[2] < 1
    assert w.iloc[3] == pytest.approx(0.0, abs=1e-12)
    assert w.iloc[4] < 0 and w.iloc[6] == pytest.approx(-1.0)
    assert w.is_monotonic_decreasing


def test_earnings_window_covers_five_weeks():
    dates = pd.bdate_range("2024-01-01", "2024-12-31")
    e = signals.earnings_window(dates)
    assert e.loc[pd.Timestamp("2024-01-16")]
    assert not e.loc[pd.Timestamp("2024-03-01")]
    assert e.loc[pd.Timestamp("2024-07-16")]


def test_max_drawdown_sign_and_scale():
    pnl = pd.Series([1.0, -5.0, 1.0, 1.0])
    assert metrics.max_drawdown(pnl, capital=100.0) == pytest.approx(-0.05)
