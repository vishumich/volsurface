"""Reconcile the JPM strategies against real OptionMetrics data.

Costs OFF throughout: every equity backtest in the report is footnoted "assumes
no transaction costs", so this is the like-for-like comparison. Turning costs on
is a separate question and a later step.

WHICH SOURCE FOR WHICH IDEA is forced by the data, not chosen:

  Idea #2 (1M 50d/10d call spreads) reads the STANDARDISED surface. Both deltas
  and the 30d tenor sit on the delta grid, which is what JPM themselves would
  have used.

  Idea #1 (10DTE 5-delta wings) CANNOT use the standardised surface: its delta
  grid stops at 10 delta. A 5-delta strike falls outside it, where the adapter
  holds flat rather than extrapolate, so the "5-delta" trade would silently be
  struck and marked at the 10-delta vol. It needs the chain, which is a much
  bigger pull - run it with --idea1.

    python scripts/reconcile_jpm.py                # Idea #2, standardised surface
    python scripts/reconcile_jpm.py --idea1        # Idea #1, chain (slow)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from volsurface import chain, data, engine, ivydb, metrics, sensitivity  # noqa: E402
from volsurface.engine import Costs  # noqa: E402
from volsurface.strategies import cs_replacement, upside_carry  # noqa: E402

pd.set_option("display.width", 170, "display.float_format", lambda x: f"{x:,.4f}")

START, END = "2017-01-03", "2026-09-14"   # the paper's window

# What the report claims, for Idea #2, pre-cost.
JPM_IDEA2 = {
    "futures":  {"sharpe": 0.73, "max_dd": -0.261},
    "baseline": {"sharpe": 0.97, "max_dd": -0.136},
    "trend":    {"sharpe": 1.13, "max_dd": -0.127},
    "final":    {"sharpe": 1.30, "max_dd": -0.070},
}


def _t(msg, t0):
    print(f"  [{time.time() - t0:5.1f}s] {msg}", flush=True)


def load_reference(conn, secid):
    t0 = time.time()
    px = ivydb.load_prices(conn, secid, START, END)
    rates = ivydb.load_rates(conn, START, END, tenor_days=30)
    divs = ivydb.load_divs(conn, secid, START, END, tenor_days=30)
    # OptionMetrics' own forward curve. Their IVs and deltas were computed on it,
    # so using it removes a whole class of forward error (and answers reviewer
    # question #1: don't model r/q for the forward at all, read theirs).
    fwd = ivydb.load_forwards(conn, secid, START, END)
    _t(f"prices {len(px):,}, rates {len(rates):,}, divs {len(divs):,}, forwards {len(fwd):,}", t0)
    return px, rates, divs, fwd


def idea2(conn, secid):
    print("=" * 96)
    print("IDEA #2  call-spread replacement for equity beta  |  standardised surface, costs OFF")
    print("=" * 96)
    px, rates, divs, fwd = load_reference(conn, secid)

    t0 = time.time()
    surf_df = ivydb.load_surface(conn, secid, START, END)
    _t(f"volatility_surface {len(surf_df):,} rows", t0)

    t0 = time.time()
    surface = data.OptionMetricsSurface(surf_df, px, rates=rates, divs=divs, forwards=fwd)
    _t(f"surface built, {len(surface.dates):,} dates", t0)

    dates = surface.dates[260:]          # burn-in for the 252d trend lookback
    print(f"\nrunning {dates[0].date()} -> {dates[-1].date()} ({len(dates):,} entry dates)")

    engine.use_costs(Costs.zero())

    # ONE base for every stage, and it is the FUTURES leg's deployed notional.
    # Measured rather than assumed - but a per-stage base would be worse than
    # the flat $1mm it replaces: the premcap and final stages deploy 2.4-3.3x
    # the benchmark, so dividing each by its own notional would show the call
    # spread "cutting drawdown by 71%" when most of that is just a bigger
    # denominator. The claim under test is call-spread VERSUS futures for the
    # same equity exposure, so both sides must share the benchmark's base.
    fut_trades, _, _ = cs_replacement.build(surface, dates, stage="futures")
    base = engine.deployed_notional(fut_trades, dates)
    print(f"common capital base = futures leg peak deployed ${base.max():,.0f}")

    out, rows = {}, []
    for stage in cs_replacement.STAGES:
        t0 = time.time()
        trades, daily, _ = cs_replacement.build(surface, dates, stage=stage)
        live = engine.deployed_notional(trades, dates)
        out[stage] = (daily, base, trades)
        s = metrics.summary(daily, base, trades)
        s["peak_deployed"] = float(live.max())     # this stage's OWN leverage
        tgt = JPM_IDEA2.get(stage)
        rows.append({
            "stage": stage, "trades": len(trades),
            "sharpe": s["sharpe"], "jpm_sharpe": tgt["sharpe"] if tgt else None,
            "max_dd": s["max_dd"], "jpm_max_dd": tgt["max_dd"] if tgt else None,
            "ann_return": s["ann_return"], "peak_deployed": s["peak_deployed"],
        })
        _t(f"{stage}: {len(trades):,} trades", t0)

    print("\n--- RECONCILIATION vs JPM section 3.3.2 (pre-cost) ---")
    tbl = pd.DataFrame(rows).set_index("stage")
    tbl["sharpe_gap"] = tbl["sharpe"] - tbl["jpm_sharpe"]
    print(tbl.to_string())
    return tbl


def idea1(conn, secid, start=START, max_dte=18, band=0.09):
    print("=" * 96)
    print("IDEA #1  short upside wings  |  FULL CHAIN (5 delta is off the standardised grid)")
    print("=" * 96)
    px, rates, divs, fwd = load_reference(conn, secid)

    # Bounds matter here, and not just for speed. The base case trades one
    # 10DTE 5-delta call, so DTE>18 and strikes beyond ~9% are rows we pay to
    # move and then never read. Over the full 2017-2026 window the looser
    # DTE<=40 / +-12% bounds are ~16M quotes, which does not fit in memory once
    # the adapter adds its derived columns.
    # Cache the pull. It is ~10 minutes over the paper's window, and a crash in
    # anything downstream should not cost that twice.
    cache = Path(__file__).resolve().parents[1] / ".cache"
    cache.mkdir(exist_ok=True)
    key = cache / f"chain_{secid}_{start}_{END}_{max_dte}_{band}.pkl"
    t0 = time.time()
    if key.exists():
        raw = pd.read_pickle(key)
        _t(f"chain {len(raw):,} quotes (CACHED {key.name})", t0)
    else:
        raw = ivydb.load_chain(conn, secid, start, END, max_dte=max_dte, moneyness_band=band)
        raw.to_pickle(key)
        _t(f"chain {len(raw):,} quotes (dte<={max_dte}, band +-{band:.0%}), cached", t0)

    t0 = time.time()
    cs = chain.OptionChainSurface(raw, px, rates=rates, divs=divs, forwards=fwd)
    _t(f"adapter built, {len(cs.quotes):,} surviving quotes, {len(cs.dates):,} dates", t0)

    print("\n--- What the wings ACTUALLY cost (measured, not assumed) ---")
    print(cs.spread_profile()[["n", "median_vol_points", "p90_vol_points", "median_rel_spread"]])

    # No 252d burn-in needed here: the base case uses no trend overlay, so the
    # only warm-up required is enough chain to strike the first trade.
    dates = cs.dates[5:]

    runs = {
        "costs OFF (JPM basis)":   Costs.zero(),
        "assumed 0.25 vol pts":    Costs.assumed(0.25, 0.5),
        "QUOTED spreads":          Costs(spot_bps=0.5, quoted_spread_mult=1.0),
    }
    rows = {}
    for label, c in runs.items():
        engine.use_costs(c)
        t0 = time.time()
        trades, daily, _ = upside_carry.build(cs, dates, overlays=False)
        rows[label] = {
            "trades": len(trades),
            "entry_delta": trades["entry_delta"].mean(),
            "tenor_d": trades["tenor_days"].mean(),
            "paid_vol_pts": trades["half_spread_vol_points"].mean(),
            **metrics.summary(daily, engine.deployed_notional(trades, dates), trades),
        }
        _t(f"{label}: {len(trades):,} trades", t0)

    print("\n--- Idea #1 base, 10DTE 5-delta calls daily (JPM: Sharpe ~0.91, DD -1.04%) ---")
    keep = ["trades", "entry_delta", "tenor_d", "paid_vol_pts", "peak_deployed",
            "sharpe", "ann_return", "ann_vol", "max_dd", "total_costs", "cost_drag_pct_of_gross"]
    print(pd.DataFrame(rows).reindex(keep).to_string())

    print("\n--- Execution ladder: how much of the quoted spread can you avoid paying? ---")
    print(sensitivity.quoted_cost_ladder(
        run=lambda: upside_carry.build(cs, dates, overlays=False),
        set_mult=lambda m: engine.use_costs(Costs(spot_bps=0.5, quoted_spread_mult=m)),
        capital=engine.deployed_notional(
            upside_carry.build(cs, dates, overlays=False)[0], dates),
    )[["sharpe", "ann_return", "max_dd", "total_costs", "paid_vol_points"]])
    engine.use_costs(Costs())
    return rows


CAPITAL = 1_000_000.0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--idea1", action="store_true", help="run Idea #1 on the chain (slow)")
    ap.add_argument("--chain-start", default=START)
    ap.add_argument("--max-dte", type=int, default=18)
    ap.add_argument("--band", type=float, default=0.09)
    args = ap.parse_args()

    conn = ivydb.connect()
    secid = ivydb.securityid(conn, "SPX")
    print(f"IvyDB connected. SPX securityid={secid}. Window {START} -> {END}.\n")
    try:
        if args.idea1:
            idea1(conn, secid, args.chain_start, args.max_dte, args.band)
        else:
            idea2(conn, secid)
    finally:
        conn.close()
