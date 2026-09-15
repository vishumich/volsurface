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
pytest                       # 11 unit tests, no vendor data needed
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

Coverage needed: **2017-01 to present** matches the paper's window. Push back to 2007 if you want
the 2008 sample, which the report only reaches for the post-crash strategy.

### Preferred (materially better fidelity)

The standardised surface is smoothed and interpolated. For a 5-delta wing traded daily, that
smoothing is doing real work — the interpolation error at 5 delta can exceed the entire per-trade
edge. If the base result survives, re-run on:

| Item | Source | Why it matters |
|---|---|---|
| Full SPX option chain, EOD | OptionMetrics `opprcd` (`best_bid`, `best_offer`, `volume`, `open_interest`) | Real bid-ask instead of an assumed half-spread; lets you drop strikes with no quoted size. |
| 10DTE listing calendar | OPRA / CBOE | 10DTE means a real listed expiry. Pre-2022 the weekly grid is sparser, and assuming a 10-day option always exists back-fills liquidity that wasn't there. |

`OptionMetricsSurface` in `src/volsurface/data.py` takes the standardised file. Wire the chain
version as a second adapter behind the same `VolSurface` protocol; nothing downstream changes.

### Not needed
Nothing intraday for these two. Idea #7 (SOXX momentum) needs minute bars; that's a separate branch.

---

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
- `max_drawdown(daily, capital)` divides by a **capital base you supply**. For a daily-entry
  overlapping book, pass peak deployed notional, not $1mm, or the figure will exceed 100%.
- Costs are **on by default** (`Costs(vol_points=0.25, spot_bps=0.5)`). `Costs.zero()` reproduces
  the paper.

## Known limitations

- The profit-taking overlay in Idea #2 uses a spot-drift proxy for "the 50d leg has drifted to
  70d" rather than tracking live per-trade delta. Fix when wired to real chains — this is the
  weakest part of stage `final`.
- `SyntheticSurface` has no volatility risk premium by construction, so carry strategies score
  near zero (or negative) on it. That's expected; it's a plumbing check, not a result.
- No early-exercise handling. Fine for SPX (European), wrong for single names — do not reuse this
  engine for the Mag 7 ideas (#4) without adding it.
- Interest rates and dividends default to flat. Immaterial for 10DTE, starts to matter at 1Y.

## Source

`JPM_Cross_Asset_Volatili_2026-09-15_5447282.pdf`, licensed to Saba Capital Management LP for
internal use. No text from it is reproduced here; section numbers are cited for traceability.
