# Full-chain adapter: quoted bid-ask, listed strikes, listed expiries

Branch: `research/full-chain-adapter`
Follows: `research/jpm-vol-surface-replication`

## What this does

The base branch reads the OptionMetrics *standardised* surface — a smoothed delta grid with no
quotes on it. It was listed there as the recommended follow-up, and it is the one that decides
whether Idea #1 is real:

> The full-chain adapter (real bid-ask rather than an assumed half-spread) is the recommended
> follow-up once the base result survives step 2 of the testing order.

`src/volsurface/chain.py` is that adapter. It sits behind the same `VolSurface` protocol, so
`upside_carry`, `cs_replacement` and the whole sensitivity harness are untouched.

## Why it changes the answer

Three things the standardised surface was assuming, measured instead.

**1. Cost.** `Costs.vol_points=0.25` is one number standing in for a spread that is neither
constant across the surface nor constant through time. Measured on real SPX quotes, 2017-01-03 to
2026-09-14 (the paper's window), 4.69M surviving quotes:

| delta bucket | median half-spread | p90 | median (ask-bid)/mid |
|---|---|---|---|
| 0-5 | **0.25 vol pts** | **0.88** | 20.0% |
| 5-10 | 0.10 | 0.20 | 5.0% |
| 10-25 | 0.08 | 0.18 | 2.8% |
| 25-50 | 0.08 | 0.19 | 1.4% |

**The 0.25 default turns out to be a good estimate of the 5-delta bucket median (0.2547) and a bad
model of what the strategy actually pays (0.1362).** Both halves matter. A daily 10DTE wing selects
the most liquid listed strike near its target, which sits at the tight end of its own bucket - so a
flat charge calibrated to the bucket overcharges this trade by 53% ($137,955 vs $90,204 over
2017-2026). The p90 of 0.88 is the other half: spreads widen exactly when a short-wing book would
want to stop selling, and a constant cannot express that either.

Two corrections worth recording. An earlier draft reported 0.80 vol points and 57% relative spreads
from the synthetic fixture - an artefact of tick rounding, since a $0.05 tick is 14% of a $0.35
synthetic option while the real 8DTE 5-delta SPX call trades ~$2.40 on a $0.10 market. And a first
measurement on 2022-2026 alone gave 0.18, which was a benign-period artefact; 2018-2021 and COVID
carry materially wider markets. Measure on the window you intend to trade.

**2. Strike.** `strike_for_delta` snaps to the nearest *listed* strike. On real SPX the selection
is tight - a "5-delta" book comes in at a mean **0.0502** delta over 2,414 trades. `TradeResult.entry_delta` now reports what you actually got, rather than the
exact 5.00 the interpolator hands back for a strike nobody could trade.

**3. Expiry.** `resolve_tenor` snaps to a listed expiry and returns its true DTE, which the engine
carries through the mark and the holding period. On real SPX a "10DTE" book averages **9.85 days**
over the full 2017-2026 window.

## Does it survive? (the question the adapter exists to answer)

Idea #1, 10DTE 5-delta SPX calls entered daily and delta-hedged at close, 2017-01-03 to 2026-09-14,
2,414 trades. JPM report **Sharpe 0.91 pre-cost**.

| | costs OFF | flat 0.25 assumption | **real quoted spreads** |
|---|---|---|---|
| Sharpe | **1.137** | 0.828 | **0.934** |
| ann return | 5.28% | 3.85% | 4.35% |
| paid half-spread (vol pts) | - | 0.250 | **0.136** |
| total costs | - | $137,955 | **$90,204** |
| cost drag (% of gross) | - | 27.1% | 17.7% |

**It survives.** The strategy clears JPM's published pre-cost Sharpe *after* paying real quoted
spreads, which their number excludes. The execution ladder says the same thing - 1.056 at mid,
0.934 at the quoted market, 0.873 paying 1.5x through - so this does not depend on good fills.

Two caveats on that, both real:

- **Beating a published number deserves more scrutiny than missing one.** 1.137 pre-cost against
  their 0.91 is not a reassuring direction. The most likely explanation is that we model no market
  impact: crossing the half-spread is assumed to fill the whole clip. At the $1mm default notional
  that is fine (SPX's $100 multiplier makes it ~1.3 contracts against OI in the tens), but it stops
  being fine two orders of magnitude up, and the cost numbers here should be read as a floor.
- **Drawdown still uses the wrong capital base.** -8.6% is on $1mm against a book running ~7
  overlapping trades; on peak deployed it is nearer -1.2%, against JPM's -1.04%. The engine should
  report deployed notional so this stops being arithmetic done by hand - see reviewer question 4.

## What it refuses to do

Both new skip paths return `None` rather than approximating, which is the whole point of reading a
chain:

- `Structure.tenor_tolerance_days` — decline days when no expiry was listed near the target tenor,
  instead of trading a 17-day option as a 10-day one. Pre-2022 the SPX weekly grid is sparse, and
  assuming a 10DTE option always existed back-fills liquidity that did not.
- `Structure.max_delta_error` — decline when no listed strike is near the target delta.
- `ChainFilters.require_two_sided` — a zero bid is the absence of a market, not a cheap option.
  OptionMetrics carries those rows with `best_bid = 0`; keeping them lets a backtest sell wings into
  a bid that was never there.

The skip counts are themselves a result. A high rate means the earlier standardised-surface run was
trading options that did not exist.

## Also in here

**A bug in the base branch.** The holding period was sliced as `dates[i : i + tenor + 1]` — N
*business* days against a tenor in *calendar* days. A 10-day option was therefore marked and
delta-hedged for four calendar days past its own expiry, pinned at intrinsic. Now measured against
the resolved expiry date. This moves the existing synthetic-surface numbers (Idea #1 base Sharpe
−3.62 → −3.44 on the smoke run); it had to be fixed here because a listed expiry the engine then
holds past is not worth resolving.

**`blackscholes.implied_vol_vec`.** Re-implying a chain is ~10⁵–10⁶ inversions. Bisection, not
Newton, despite the asymptotics: half a listed chain is cheap wings quoted near the tick where vega
underflows, Newton stalls on its seed there, and falling back to scalar `brentq` for that 39% cost
79s against 1.0s for the vectorised bisection on the same input.

**`sensitivity.quoted_cost_ladder`.** `cost_ladder` sweeps an assumption; with quotes the spread is a
measurement, so the open question becomes execution — how much of the quoted spread you have to
avoid paying. Needing `spread_mult < 0.5` is a claim about the desk, not about the surface.

**`make_synthetic_chain`.** Discretises any surface onto a listed chain — fixed strike ladder,
weeklies then third-Friday monthlies, tick-rounded quotes that widen into the wings — so all of the
above runs in tests and in `run_smoke.py` with no vendor data. It is a plumbing fixture and is
**not calibrated**: see the cost note above for how far its wing spreads are from a real market.

**`src/volsurface/ivydb.py`** — loaders for the on-prem OptionMetrics IvyDB-US server, plus
`normalize_ivydb` for its column dialect, which is NOT the WRDS spelling (`securityid`, `bestbid`,
`bestoffer`, `expiration`, `callput`; `strike` still ×1000). The chain loader bounds tenor *and*
moneyness server-side — the unbounded SPX chain over the paper's window is ~16M quotes, which does
not survive the adapter's derived columns in memory.

Two things it changed elsewhere:

- **Forwards now come from OptionMetrics' published `forward_price`** rather than
  `S·e^(r−q)T`. Their IVs and deltas were computed on that forward, so any drift between it and a
  reconstruction lands straight in the P&L. This retires reviewer question #1 from the base branch
  ("is flat r/q acceptable?") — don't model r/q for the forward at all. Worth knowing:
  OptionMetrics carries **q = 0 at the short end** by convention (their 10-day SPX forward implies
  exactly the 10-day zero rate), so a rolled-forward leg pays full carry and will not match spot.
- **Two more latent bugs, both invisible on synthetic data.** `_flat_rate` rebuilt its pandas index
  on every call — O(n) per lookup, fine for a scalar rate, fatal for a real `(date, rate)` frame.
  And `_build_day` ran a `brentq` per surface grid point; each point carries its own vol, so it is a
  constant-vol solve with a closed form (`strike_from_delta_flat`, exact to 2.2e-15 against brentq's
  9.6e-7). Together these took the surface build from minutes to 12s.

**`scripts/reconcile_jpm.py`** runs both ideas against the published numbers. It found a third bug:
`cs_replacement._futures_benchmark` computed its exit as `i + tenor_days`, an offset into the
*business-day* index, so a 30-calendar-day forward was held ~42 days and the benchmark ran at 1.40x
rather than 1x — most of why its drawdown came in at −49% against SPX's own −34%.

## Testing

`pytest` — 34 passing (23 new in `tests/test_chain.py`), none needing a database or vendor data.
`python scripts/run_smoke.py` gains a section running Idea #1 on the standardised surface and
the chain side by side; `python scripts/reconcile_jpm.py` runs the real reconciliation against
IvyDB (add `--idea1` for the chain).

The three tests worth reading are `test_listed_strike_is_not_the_requested_delta`,
`test_spread_profile_separates_wings_from_body`, and `test_engine_charges_the_quoted_spread` — they
pin the three claims above. `test_costs_zero_is_actually_zero_on_a_quoted_surface` guards a trap
worth knowing about: `Costs.zero()` must not fall through to "charge whatever the chain quoted", or
every no-cost reproduction of the published numbers is silently wrong.

## Not in scope

**Depth.** Every `Quote` carries `open_interest` and `volume`, and `ChainFilters` screens on them,
but crossing the half-spread is still assumed to fill the whole clip. For a daily 5-delta roll at
real size that is optimistic, and it is the next thing to model.

No vendor data is committed, and none needs to be: `src/volsurface/ivydb.py` reads Saba's on-prem
OptionMetrics **IvyDB-US** SQL Server directly (daily feed, 122,398 securities, history to
1996-01-04). The measured numbers above come from there, not from the fixture.

**`ChainFilters` still screens on quotes alone.** Nothing yet excludes a strike on the grounds that
the size behind it could not absorb the clip, so the cost figures are a lower bound at real size.

## Reviewer questions

1. `ChainFilters` defaults are permissive apart from `require_two_sided`. Should `liquid()`
   (OI ≥ 100, volume ≥ 1, rel-spread ≤ 0.5) be the default instead? It is the honest screen, but it
   will cut the wing sample hard and I would rather that be a deliberate choice than a default.
2. Vols are re-implied from quotes with our own forward-based BS rather than taken from OM's
   `impl_volatility` (binomial on spot, discrete divs). Consistent with how the engine marks, but it
   will not tie out to anything else quoting OM's IV. Worth a reconciliation column?
3. `max_delta_error=0.03` and the OTM-mid marking convention are both judgement calls. Sanity-check
   them against how the desk actually marks a wing.
4. `max_drawdown` divides by a capital base the caller supplies, and every drawdown above is on a
   flat $1mm, which is wrong for an overlapping daily-entry book. Should `TradeResult` aggregation
   emit a deployed-notional series so the right denominator is available rather than estimated?
5. Idea #1 clears its costs at the $1mm notional tested. At what size does it stop? That needs the
   depth model above, and it is the question that decides whether this is tradeable or merely true.
