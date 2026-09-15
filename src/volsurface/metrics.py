"""Performance statistics.

Matched to the JPM conventions so numbers are comparable:
  - Sharpe is mean daily P&L / std of daily P&L, annualised by sqrt(252),
    computed on the DAILY series (not per-trade) because trades overlap.
  - Max drawdown is on the cumulative P&L path, expressed as a % of the
    capital base you pass in, not of peak equity.
  - Returns are stated as non-compounded averages where the paper does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe(daily_pnl: pd.Series, periods: int = 252) -> float:
    x = daily_pnl.dropna()
    if len(x) < 2 or x.std() == 0:
        return np.nan
    return float(x.mean() / x.std() * np.sqrt(periods))


def max_drawdown(daily_pnl: pd.Series, capital: float) -> float:
    cum = daily_pnl.fillna(0.0).cumsum()
    return float((cum - cum.cummax()).min() / capital)


def ann_return(daily_pnl: pd.Series, capital: float, periods: int = 252) -> float:
    return float(daily_pnl.mean() * periods / capital)


def ann_vol(daily_pnl: pd.Series, capital: float, periods: int = 252) -> float:
    return float(daily_pnl.std() * np.sqrt(periods) / capital)


def var95(daily_pnl: pd.Series, capital: float) -> float:
    return float(np.nanpercentile(daily_pnl.dropna(), 5) / capital)


def hit_ratio(trades: pd.DataFrame) -> float:
    if trades.empty:
        return np.nan
    return float((trades["pnl"] > 0).mean())


def summary(daily_pnl: pd.Series, capital: float, trades: pd.DataFrame | None = None) -> dict:
    out = {
        "sharpe": sharpe(daily_pnl),
        "ann_return": ann_return(daily_pnl, capital),
        "ann_vol": ann_vol(daily_pnl, capital),
        "max_dd": max_drawdown(daily_pnl, capital),
        "var95": var95(daily_pnl, capital),
    }
    if trades is not None and not trades.empty:
        out |= {
            "n_trades": int(len(trades)),
            "hit_ratio": hit_ratio(trades),
            "avg_pnl_per_trade": float(trades["pnl"].mean()),
            "total_costs": float(trades["cost"].sum()),
            "cost_drag_pct_of_gross": float(
                trades["cost"].sum() / max(abs(trades["pnl"].sum() + trades["cost"].sum()), 1e-9)
            ),
        }
    return out


def summary_table(named: dict[str, tuple[pd.Series, float, pd.DataFrame | None]]) -> pd.DataFrame:
    return pd.DataFrame({k: summary(*v) for k, v in named.items()}).T
