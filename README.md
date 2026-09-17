# volsurface

Replication harness for two strategies from JPM *Cross Asset Volatility: A Practical Guide to
Volatility Trading Across Asset Classes* (Hou et al., 15 Sep 2026):

- **Idea #1** — regime-conditioned short upside wings, SPX (§3.1.6)
- **Idea #2** — call-spread replacement for long equity beta, SPX (§3.3.2)

Both are pre-transaction-cost in the source. The point of this harness is to reproduce them, then
put costs and parameter sensitivity on top, because those are the two places the published numbers
are most likely to break.

```bash
pip install -e .
pytest                       # 38 unit tests, no vendor data needed
python scripts/run_smoke.py  # end-to-end on a synthetic surface
```

---

## Data requirements

### Minimum viable (both ideas)

| Item | Source | Fields | Why |
|---|---|---|---|
| SPX standardised vol surface, daily | OptionMetrics `vsurfd` (WRDS) | `date, days, delta, cp_flag, impl_volatility` | Strike selection and daily fixed-strike marks. Delta grid ±10/20/…/80, tenors 10–730d. |
| SPX close | OptionMetrics `secprd` or Bloomberg | `date, close` | Delta hedging, forward construction. |
| Zero curve | OptionMetrics `zerocd` | `date, days, rate` | Forwards and discounting. Flat 4% is an acceptable v1. |
| Index dividend yield | OptionMetrics `idxdvd` | `date, rate` | Forwards. Flat 1.5% is an acceptable v1. |
| VIX close | CBOE / Bloomberg | `date, close` | Regime cutoffs. Only needed for the VIX-sizing variants. |

Coverage needed: **2017-01 to present** matches the paper's window.

**In practice none of the above needs a WRDS pull.** Saba runs an on-prem OptionMetrics **IvyDB-US**
SQL Server, refreshed daily and covering 122,398 securities back to **1996-01-04** - so the 2008
sample, the dot-com unwind and LTCM are all in reach, not just 2017. `src/volsurface/ivydb.py` has
the loaders. Note IvyDB's column names are NOT the WRDS spellings quoted in the tables above
(`securityid`/`bestbid`/`bestoffer`/`expiration`/`callput`), and prefer its published
`forward_price` curve to reconstructing `S*exp((r-q)T)`.

### Preferred (materially better fidelity)

The standardised surface is smoothed and interpolated. For a 5-delta wing traded daily, that
smoothing is doing real work — the interpolation error at 5 delta can exceed the entire per-trade
edge. If the base result survives, re-run on:

| Item | Source | Why it matters |
|---|---|---|
| Full SPX option chain, EOD | OptionMetrics `opprcd` (`best_bid`, `best_offer`, `volume`, `open_interest`) | Real bid-ask instead of an assumed half-spread; lets you drop strikes with no quoted size. |
| 10DTE listing calendar | OPRA / CBOE | 10DTE means a real listed expiry. Pre-2022 the weekly grid is sparser, and assuming a 10-day option always exists back-fills liquidity that wasn't there. |

Both are now implemented — see **The chain adapter** below. `opprcd` is the only extra pull needed;
the listing calendar comes free with it, since a chain only contains expiries that were listed.

### Not needed
Nothing intraday for these two. Idea #7 (SOXX momentum) needs minute bars; that's a separate branch.

---

## The chain adapter

`src/volsurface/chain.py` reads the raw chain behind the same `VolSurface` protocol, so strategies
and the sensitivity harness are unchanged. Three things stop being assumptions.

**1. Cost is measured, not guessed.** `Costs.vol_points` is one number standing in for a spread
that is constant neither across the surface nor through time. Measured on real SPX quotes
(2017-01-03 to 2026-09-14, the paper's window, 4.69M surviving quotes):

| delta bucket | median half-spread | p90 | median (ask−bid)/mid |
|---|---|---|---|
| 0–5 | **0.25 vol pts** | **0.88** | 20.0% |
| 5–10 | 0.10 | 0.20 | 5.0% |
| 10–25 | 0.08 | 0.18 | 2.8% |
| 25–50 | 0.08 | 0.19 | 1.4% |

The 0.25 default is a good estimate of the 5-delta *bucket* (median 0.2547) and a bad model of what
this *trade* pays (0.1362) — a daily 10DTE wing selects the most liquid listed strike near its
target, at the tight end of its own bucket, so a bucket-calibrated flat charge overcharges it by
53%. The p90 of 0.88 is the other half: spreads widen precisely when a short-wing book would want
to stop selling, and a constant expresses neither. Measure with `spread_profile()` on the window
you intend to trade — 2022–2026 alone gives 0.18, a benign-period artefact.

**2. Strikes are listed strikes.** `strike_for_delta` snaps to the nearest quoted strike instead of
solving for a continuous one, and `TradeResult.entry_delta` reports what you actually got. On real
SPX the selection is tight: a "5-delta" book comes in at a mean 0.0502 delta over 2,414 trades. Ask for a
delta nothing is listed near and the day is skipped, not approximated — `Structure.max_delta_error`.

**3. Expiries are listed expiries.** `resolve_tenor` snaps to a real expiry and returns its true DTE,
which the engine then carries through the mark and the holding period. On real SPX a "10DTE" book
averages 9.85 days over the full 2017-2026 window. Set `Structure.tenor_tolerance_days` to skip days when nothing was listed near the target
rather than silently trading a 17-day option as if it were a 10-day one — this is the control that
stops a pre-2022 backtest inventing a weekly grid.

```python
from volsurface import chain, engine
from volsurface.engine import Costs

cs = chain.OptionChainSurface(
    chain.normalize_optionmetrics(opprcd),      # handles strike_price / 1000
    prices, rates=zerocd, divs=idxdvd,
    filters=chain.ChainFilters.liquid(),        # OI >= 100, real two-sided markets
)
print(cs.spread_profile())                      # what do we actually pay?

engine.use_costs(Costs(spot_bps=0.5))           # quoted spreads, since cs can quote
engine.use_costs(Costs.assumed(0.25, 0.5))      # force the flat model, for comparison
engine.use_costs(Costs.zero())                  # off everywhere, to reproduce the paper
```

`sensitivity.quoted_cost_ladder` replaces `cost_ladder` once quotes are real: the open question is
no longer "at what assumed bid-ask does this die" but "how much of the quoted spread do you have to
avoid paying". Needing `spread_mult < 0.5` means claiming better-than-mid fills on a daily roll of
thousands of wing contracts — a claim about the desk, not about the surface.

`make_synthetic_chain` builds a listed chain off any surface (fixed strike ladder, weeklies then
third-Friday monthlies, tick-rounded quotes that widen into the wings) so all of this runs in tests
with no vendor data.

## Testing order

Sequenced so each step can kill the thesis before you spend on the next.

**1. Reproduce the base cases, costs off.** Short 10DTE 5-delta calls daily, hedged at close, held
to expiry → target Sharpe ≈ 0.91. Long 1M 50d/10d call spread at 2x vs 1M futures → target
0.97 vs 0.73, maxDD −13.6% vs −26.1%. If you land materially off with costs disabled, it's a
strike-selection or marking bug, not a data difference — check `strike_from_delta` is solving on
the surface, not at a flat vol (`test_strike_from_delta_uses_surface_not_flat_vol`).

**2. Turn costs on before adding a single overlay.** `sensitivity.cost_ladder` sweeps the option
half-spread from 0 to 1.0 vol points. Idea #1 enters daily at 5 delta and hedges daily; the
question is whether it survives 0.25 vol, not whether it works at zero. Do this *before*
enhancements so you're not tuning overlays to rescue a strategy that costs already killed.

**2b. Re-run step 2 on the chain.** `cost_ladder` sweeps an assumption; once `opprcd` is loaded the
spread is a measurement, so switch to `sensitivity.quoted_cost_ladder` and read
`cs.spread_profile()` first. On SPX 2017-2026 the 5-delta median is 0.25 vol points - the default
is well calibrated to the bucket - but the trade itself pays 0.136, and the p90 is 0.88. So the
flat charge is simultaneously too high for this trade and too low for a bad day. If Idea #1 needs `spread_mult` below 0.5 it needs
better-than-mid fills every day at 5 delta, and the conversation moves to the execution desk. Check the skip rate here too: a large
count of days with no listed 10DTE expiry, or no strike within `max_delta_error` of 5 delta, means
the earlier standardised-surface run was trading options that did not exist.

**3. Add overlays one at a time and attribute.** `build(..., overlays=True)` returns per-leg P&L.
The claim is 0.91 → 1.23. Check which leg delivers it. If the whole increment is the earnings
participation bump, that's a seasonality trade, not a regime filter.

**4. Sweep every cutoff.** `sensitivity.sweep` + `monotonicity`. Cutoffs to test: TS inversion
threshold, trend extreme threshold, base delta (5d vs 10d vs 25d), base tenor (10d vs 30d),
earnings participation. **A sharp peak sitting on the published value is the failure signal.**
The `peak_share` statistic flags it; above ~0.6 treat the result as fitted.

**5. Walk forward.** `sensitivity.walk_forward` refits one parameter on 3y and scores 1y out of
sample. If OOS Sharpe collapses back toward the unfiltered base, the filter carries no information.

**6. Only then, stage the enhancements on Idea #2.** Run `stage=` through
`futures → baseline → trend → premcap → final`. The single most transportable claim is
fixed-premium vs fixed-notional sizing (`premcap`) — it's what removes the 2022 drawdown and it
generalises to anything you buy systematically. Test that in isolation even if the rest fails.

**7. Substitute the proprietary signals honestly.** `trend_score` and `mr_score` are documented
proxies, not the JPM Cross Asset Trend / Mean Reversion Score. Re-run step 4 on whichever internal
trend signal you'd actually trade; if the result only works with one specific momentum
construction, it isn't a vol-surface result.

---

## Conventions

- Deltas are **absolute forward deltas**; a 5-delta call and 5-delta put are both `0.05`.
- Strikes are **fixed at entry**; daily marks read fixed-strike vol. Re-striking daily would
  remove the path dependency the paper's whole argument rests on.
- Delta hedging is **daily at close, in the forward** — matching the source.
- P&L is in **dollars per unit of `notional`** (default $1mm).
- Sharpe is computed on the **daily aggregated** series, not per-trade, because these strategies
  enter every day and hold to expiry, so up to `tenor` trades are live at once. Summing trade P&L
  understates drawdown.
- `max_drawdown(daily, capital)` takes a float **or** the `engine.deployed_notional` series - pass
  the series. A daily-entry book holds ~`tenor` trades at once (measured: 9x for the 10DTE wing
  book, 1.05x for the 1M futures benchmark), and dividing by one trade's notional overstates
  drawdown by exactly that factor. It reported -8.6% where the real figure is -1.01%.
  `summary` then reports `peak_deployed` next to the drawdown so the base is never in doubt.
- **Comparing strategies needs ONE base across all of them.** Giving each stage its own deployed
  notional makes the stage table meaningless - a 2.4x-levered stage looks like it cut drawdown when
  it only enlarged the denominator. `reconcile_jpm.py` uses the futures benchmark's base for every
  Idea #2 stage, because the claim under test is call-spread versus futures at the same exposure.
- Costs are **on by default** (`Costs(vol_points=0.25, spot_bps=0.5)`). `Costs.zero()` reproduces
  the paper. On a surface that can quote (`chain.OptionChainSurface`) the **real half-spread is
  charged and `vol_points` is ignored**; `Costs.assumed(...)` forces the flat model back on for an
  apples-to-apples comparison.

## Known limitations

- The profit-taking overlay in Idea #2 uses a spot-drift proxy for "the 50d leg has drifted to
  70d" rather than tracking live per-trade delta. Fix when wired to real chains — this is the
  weakest part of stage `final`.
- `SyntheticSurface` has no volatility risk premium by construction, so carry strategies score
  near zero (or negative) on it. That's expected; it's a plumbing check, not a result.
- No early-exercise handling. Fine for SPX (European), wrong for single names — do not reuse this
  engine for the Mag 7 ideas (#4) without adding it.
- Interest rates and dividends default to flat. Immaterial for 10DTE, starts to matter at 1Y.
- The chain adapter marks off the **OTM mid** at each listed strike. It models no size: crossing
  the quoted half-spread is assumed to fill the whole clip, which for a daily 5-delta roll at real
  size is optimistic. Depth is the next thing to model, and `open_interest` is already carried on
  every `Quote` to do it with.
- `make_synthetic_chain` is a fixture, not a market, and it is **not calibrated**: tick rounding
  makes its wing spreads ~4x too wide (a $0.05 tick is 14% of a $0.35 synthetic option; the real
  8DTE 5-delta SPX call trades ~$2.40 on a $0.10 market). Use it to exercise code paths, never to
  size a cost assumption.

## Source

`JPM_Cross_Asset_Volatili_2026-09-15_5447282.pdf`, licensed to Saba Capital Management LP for
internal use. No text from it is reproduced here; section numbers are cited for traceability.
