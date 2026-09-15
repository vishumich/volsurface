# Backtest harness for two JPM cross-asset vol strategies

Branch: `research/jpm-vol-surface-replication`

## What this does

Adds a self-contained harness to replicate and stress-test two strategies from JPM's *Cross Asset
Volatility* report (15 Sep 2026):

1. **Regime-conditioned short upside wings** (§3.1.6) — short 10DTE 5-delta SPX calls daily,
   delta-hedged at close, with three regime overlays. Reported 0.91 → 1.23 Sharpe, −1.04% worst DD.
2. **Call-spread replacement for equity beta** (§3.3.2) — 1M 50d/10d call spreads instead of
   futures, staged through trend filter, fixed-premium sizing and put-spread profit-taking.
   Reported 0.73 → 1.30 Sharpe, −26.1% → −7.0% maxDD.

## Why these two first

Highest ratio of claim strength to implementation cost. Both need only the OptionMetrics
standardised surface plus a spot series, both are fully specified in the report, and #2's
fixed-premium-vs-fixed-notional finding generalises well beyond this trade.

## What's deliberately different from the source

Every equity backtest in the report is footnoted *"assumes no transaction costs."* For structures
entered daily at 5 delta with daily delta hedging, that is not a rounding error. So:

- **Costs are on by default** and modelled as an option half-spread in vol points plus spot
  slippage on hedge turnover. `Costs.zero()` reproduces the published numbers.
- **`sensitivity.cost_ladder`** answers the question the report leaves open: at what bid-ask does
  each strategy stop working.
- **`sensitivity.sweep` + `monotonicity`** exist because the regime cutoffs (VIX 15/21/30, trend
  0.75, TS inversion at 0) read as chosen with the full sample in hand. The test is not "which
  cutoff is best" but "is performance monotone or plateaued in this parameter." A sharp peak on the
  published value is the signature of fitting; `peak_share` flags it.
- **`sensitivity.walk_forward`** refits one parameter on 3y and scores 1y OOS.

## Structure

```
src/volsurface/
  blackscholes.py          pricing, greeks, delta→strike inversion ON the surface
  data.py                  VolSurface protocol; OptionMetrics + synthetic adapters
  engine.py                trade lifecycle, daily delta hedging, costs, overlap aggregation
  signals.py               term structure, VIX regime, trend/MR proxies, earnings windows
  metrics.py               Sharpe / maxDD / hit ratio on JPM conventions
  sensitivity.py           parameter sweep, monotonicity, walk-forward, cost ladder
  strategies/
    upside_carry.py        Idea #1
    cs_replacement.py      Idea #2
tests/test_engine.py       11 tests, no vendor data required
scripts/run_smoke.py       end-to-end on a synthetic surface
```

## Testing

`pytest` — 11 passing. `python scripts/run_smoke.py` runs both strategies end to end on a synthetic
skewed surface, including the cost ladder and a cutoff sweep. The synthetic surface has no vol risk
premium by construction, so the numbers are plumbing checks, not results.

The test worth reading is `test_strike_from_delta_uses_surface_not_flat_vol`. Solving for a 5-delta
strike at a flat vol instead of on the skewed surface moves the strike by more than 0.5% and
silently changes which part of the surface you're testing — the most likely way to get a plausible
but wrong replication.

## Not in scope

No vendor data is committed. `OptionMetricsSurface` expects WRDS frames; see README for the field
list and coverage requirements. The full-chain adapter (real bid-ask rather than an assumed
half-spread) is the recommended follow-up once the base result survives step 2 of the testing order.

## Reviewer questions

1. Is flat r/q acceptable for v1, or should this pull the real curve from the start? Immaterial at
   10DTE, starts to matter for the 1Y variants.
2. Cost assumption: 0.25 vol points half-spread on SPX wings, 0.5bp on hedge turnover. Sanity-check
   against what we actually pay.
3. Capital base for drawdown on a daily-entry overlapping book — peak deployed notional, or
   something margin-based?
