"""Parameter sensitivity and walk-forward harness.

This is the part that is not in the JPM report and is the reason to build this
rather than take the numbers on trust. Their regime cutoffs (VIX 15/21/30,
trend 0.75, 35th-percentile correlation, 3% correlation pop) read as chosen
with the full sample in hand.

Two checks:

  sweep()        Run the strategy across a grid of one parameter and report the
                 metric surface. The question is NOT "which value is best" but
                 "is performance monotone / plateaued in this parameter". A
                 sharp peak at the published value is the signature of fitting.

  walk_forward() Refit the parameter on a rolling in-sample window and score it
                 out of sample. If the OOS Sharpe collapses toward the
                 unfiltered base case, the filter is not carrying information.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import numpy as np
import pandas as pd

from . import metrics


def sweep(
    run: Callable[[object], tuple[pd.DataFrame, pd.Series, object]],
    params,
    field: str,
    values,
    capital: float,
) -> pd.DataFrame:
    """Vary one parameter, holding the rest fixed. Returns a metrics table."""
    rows = []
    for v in values:
        trades, daily, _ = run(replace(params, **{field: v}))
        rows.append({field: v, **metrics.summary(daily, capital, trades)})
    return pd.DataFrame(rows).set_index(field)


def monotonicity(table: pd.DataFrame, metric: str = "sharpe") -> dict:
    """Score how peaked the metric is in the swept parameter.

    `peak_share` is the best value's excess over the median, normalised by the
    spread. Above ~0.6 with the peak sitting exactly on the published default
    is a red flag: the result depends on the specific cutoff rather than on the
    filter's economics.
    """
    x = table[metric].dropna()
    if len(x) < 3:
        return {}
    rho = float(pd.Series(x.values).corr(pd.Series(range(len(x))), method="spearman"))
    spread = float(x.max() - x.min())
    return {
        "spearman_rank_corr": rho,
        "argmax": x.idxmax(),
        "peak_share": float((x.max() - x.median()) / spread) if spread > 0 else np.nan,
        "range": spread,
    }


def walk_forward(
    run: Callable[[object, pd.DatetimeIndex], tuple[pd.DataFrame, pd.Series, object]],
    params,
    field: str,
    values,
    dates: pd.DatetimeIndex,
    capital: float,
    train_years: int = 3,
    test_years: int = 1,
) -> pd.DataFrame:
    """Rolling refit of one parameter, scored out of sample."""
    dates = pd.DatetimeIndex(dates)
    rows = []
    start = dates[0]
    while True:
        tr_end = start + pd.DateOffset(years=train_years)
        te_end = tr_end + pd.DateOffset(years=test_years)
        if te_end > dates[-1]:
            break
        tr = dates[(dates >= start) & (dates < tr_end)]
        te = dates[(dates >= tr_end) & (dates < te_end)]

        best, best_s = None, -np.inf
        for v in values:
            _, daily, _ = run(replace(params, **{field: v}), tr)
            s = metrics.sharpe(daily)
            if np.isfinite(s) and s > best_s:
                best, best_s = v, s

        trades_oos, daily_oos, _ = run(replace(params, **{field: best}), te)
        rows.append({
            "train_start": tr[0], "test_start": te[0], "chosen": best,
            "is_sharpe": best_s, **{f"oos_{k}": v for k, v in
                                    metrics.summary(daily_oos, capital, trades_oos).items()},
        })
        start = start + pd.DateOffset(years=test_years)
    return pd.DataFrame(rows)


def cost_ladder(
    run: Callable[[], tuple[pd.DataFrame, pd.Series, object]],
    set_costs: Callable[[float, float], None],
    vol_points=(0.0, 0.1, 0.25, 0.5, 1.0),
    spot_bps: float = 0.5,
    capital: float = 1_000_000.0,
) -> pd.DataFrame:
    """Sharpe as a function of the option half-spread.

    Answers the question the report leaves open: at what bid-ask does each
    strategy stop working? Daily-rolled 5-delta structures with daily hedging
    are the most cost-sensitive thing in the paper.

    Only meaningful on a surface that cannot quote. On a chain the engine
    charges the real spread and ignores `vol_points`, so this ladder comes back
    flat unless `set_costs` builds its `Costs` with `quoted_spread_mult=None`
    (i.e. `Costs.assumed`). Use `quoted_cost_ladder` there instead.
    """
    rows = []
    for vp in vol_points:
        set_costs(vp, spot_bps)
        trades, daily, _ = run()
        rows.append({"vol_points": vp, **metrics.summary(daily, capital, trades)})
    return pd.DataFrame(rows).set_index("vol_points")


def quoted_cost_ladder(
    run: Callable[[], tuple[pd.DataFrame, pd.Series, object]],
    set_mult: Callable[[float], None],
    mults=(0.0, 0.25, 0.5, 0.75, 1.0, 1.5),
    capital: float = 1_000_000.0,
) -> pd.DataFrame:
    """`cost_ladder`'s counterpart for a quoted chain.

    On the standardised surface the only question you can ask is "at what
    assumed bid-ask does this die", because there is no quote to compare to.
    Once the chain is wired in, the spread is a measurement and the open
    question becomes execution: how much of the quoted spread do you have to
    avoid paying for the strategy to work.

    `mult=1.0` is lifting the offer / hitting the bid. `0.5` is mid. `0.0` is
    the report's no-cost assumption. If a strategy needs `mult < 0.5` it is
    claiming to get filled better than mid on a daily roll of thousands of
    wing contracts, which is a claim about the desk, not about the surface.
    """
    rows = []
    for m in mults:
        set_mult(m)
        trades, daily, _ = run()
        row = {"spread_mult": m, **metrics.summary(daily, capital, trades)}
        if "half_spread_vol_points" in getattr(trades, "columns", []):
            row["paid_vol_points"] = float(trades["half_spread_vol_points"].mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("spread_mult")
