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

from volsurface import data, engine, metrics, sensitivity  # noqa: E402
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


if __name__ == "__main__":
    main()
