"""End-to-end smoke run on the synthetic surface.

Proves the pipeline works without vendor data. The NUMBERS ARE MEANINGLESS -
the synthetic surface has no volatility risk premium in it by construction, so
carry strategies should score near zero. What this checks is that trades get
struck, marked, hedged, costed and aggregated, and that the sensitivity and
cost harnesses run.

    python scripts/run_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from volsurface import chain, data, engine, metrics, sensitivity  # noqa: E402
from volsurface.engine import Costs  # noqa: E402
from volsurface.strategies import cs_replacement, upside_carry  # noqa: E402

pd.set_option("display.width", 160, "display.float_format", lambda x: f"{x:,.4f}")

CAPITAL = 1_000_000.0


def main():
    surf = data.make_synthetic(n_days=900, seed=11)
    dates = surf.dates[260:-100]  # burn-in for the 252d signal lookbacks

    print("=" * 78)
    print("IDEA #1  regime-conditioned short upside wings")
    print("=" * 78)
    engine.use_costs(Costs.zero())
    base_t, base_d, _ = upside_carry.build(surf, dates, overlays=False)
    full_t, full_d, legs = upside_carry.build(surf, dates, overlays=True)

    print(metrics.summary_table({
        "base (no costs)": (base_d, CAPITAL, base_t),
        "with overlays (no costs)": (full_d, CAPITAL, full_t),
    }))

    print("\nP&L contribution by leg:")
    print(full_t.groupby("label")["pnl"].agg(["count", "sum", "mean"]))

    print("\nCost ladder - where does the edge die?")
    print(sensitivity.cost_ladder(
        run=lambda: upside_carry.build(surf, dates, overlays=True),
        set_costs=lambda vp, sb: engine.use_costs(Costs(vol_points=vp, spot_bps=sb)),
        capital=CAPITAL,
    )[["sharpe", "ann_return", "max_dd", "total_costs"]])

    print("\nSensitivity to the term-structure inversion cutoff:")
    engine.use_costs(Costs())
    tbl = sensitivity.sweep(
        run=lambda p: upside_carry.build(surf, dates, params=p, overlays=True),
        params=upside_carry.Params(),
        field="ts_inversion_threshold",
        values=[-1.0, -0.5, 0.0, 0.5, 1.0],
        capital=CAPITAL,
    )
    print(tbl[["sharpe", "ann_return", "max_dd"]])
    print("monotonicity:", sensitivity.monotonicity(tbl))

    print("\n" + "=" * 78)
    print("IDEA #2  call-spread replacement for equity beta")
    print("=" * 78)
    engine.use_costs(Costs.zero())
    out = {}
    for stage in ("futures", "baseline", "trend", "premcap", "final"):
        t, d, _ = cs_replacement.build(surf, dates, stage=stage)
        out[stage] = (d, CAPITAL, t)
    print(metrics.summary_table(out)[["sharpe", "ann_return", "max_dd", "var95"]])
    print("\n(synthetic surface => no risk premium => these are plumbing checks, not results)")

    chain_section()


def chain_section():
    """Same strategy, standardised surface vs a quoted chain.

    The comparison is the deliverable. Everything that moves between the two
    columns is something the standardised surface was quietly assuming.
    """
    print("\n" + "=" * 78)
    print("FULL-CHAIN ADAPTER  standardised surface vs quoted chain")
    print("=" * 78)

    surf = data.make_synthetic(n_days=250, seed=11)
    raw = chain.make_synthetic_chain(
        surf, dates=surf.dates, strike_step=10.0, band=0.10, max_dte=100,
    )
    prices = pd.DataFrame({"date": surf.dates, "close": [surf.spot(d) for d in surf.dates]})
    cs = chain.OptionChainSurface(raw, prices, rates=surf.r, divs=surf.q)
    print(f"chain: {len(raw):,} rows in, {len(cs.quotes):,} quotes survived filtering")

    print("\nWhat the chain actually charges (the 0.25 vol-point default is a guess):")
    print(cs.spread_profile()[["n", "median_vol_points", "p90_vol_points", "median_rel_spread"]])

    dates = cs.dates[60:-40]

    engine.use_costs(Costs.zero())
    cont_t, cont_d, _ = upside_carry.build(surf, dates, overlays=False)
    chain_t, chain_d, _ = upside_carry.build(cs, dates, overlays=False)

    print("\nStrike and expiry discretisation, costs OFF on both sides:")
    print(pd.DataFrame({
        "standardised": {
            "trades": len(cont_t),
            "mean entry delta": cont_t["entry_delta"].mean(),
            "mean tenor (d)": cont_t["tenor_days"].mean(),
            "sharpe": metrics.sharpe(cont_d),
        },
        "quoted chain": {
            "trades": len(chain_t),
            "mean entry delta": chain_t["entry_delta"].mean(),
            "mean tenor (d)": chain_t["tenor_days"].mean(),
            "sharpe": metrics.sharpe(chain_d),
        },
    }))
    print(f"days the chain could not strike a 5-delta wing: "
          f"{len(dates) - len(chain_t)}/{len(dates)}")

    print("\nCosts ON: assumed 0.25 vol points vs the quoted market")
    engine.use_costs(Costs.assumed(0.25, 0.5))
    ass_t, ass_d, _ = upside_carry.build(cs, dates, overlays=False)
    engine.use_costs(Costs(spot_bps=0.5, quoted_spread_mult=1.0))
    qt_t, qt_d, _ = upside_carry.build(cs, dates, overlays=False)
    print(pd.DataFrame({
        "assumed 0.25vp": {
            "half-spread paid (vp)": ass_t["half_spread_vol_points"].mean(),
            "total cost": ass_t["cost"].sum(),
            "sharpe": metrics.sharpe(ass_d),
        },
        "quoted": {
            "half-spread paid (vp)": qt_t["half_spread_vol_points"].mean(),
            "total cost": qt_t["cost"].sum(),
            "sharpe": metrics.sharpe(qt_d),
        },
    }))

    print("\nExecution ladder - how much of the quoted spread must you avoid paying?")
    print(sensitivity.quoted_cost_ladder(
        run=lambda: upside_carry.build(cs, dates, overlays=False),
        set_mult=lambda m: engine.use_costs(Costs(spot_bps=0.5, quoted_spread_mult=m)),
        capital=CAPITAL,
    )[["sharpe", "ann_return", "max_dd", "total_costs", "paid_vol_points"]])
    engine.use_costs(Costs())


if __name__ == "__main__":
    main()
