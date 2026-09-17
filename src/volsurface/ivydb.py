"""Loaders for OptionMetrics IvyDB (on-prem SQL Server).

Saba's OptionMetrics is an internal IvyDB-US SQL Server, not the WRDS Postgres
service every tutorial assumes. Two consequences shape this module:

1. **The column names are the raw IvyDB dialect**, not the WRDS spellings the
   `chain.normalize_optionmetrics` mapper expects - `securityid` not `secid`,
   `bestbid` not `best_bid`, `expiration` not `exdate`, `callput` not `cp_flag`.
   Porting a WRDS query verbatim does not compile. `normalize_ivydb` in
   `chain.py` handles the rename; `strike` is still x1000.

2. **`option_price` is a multi-decade full chain** (SPX alone: 1996 to present).
   Every query here filters `securityid` and a bounded `[date]` range, and the
   chain loader additionally bounds tenor and moneyness server-side. An
   unbounded aggregate over this table does not return.

The credential is never handled here: `read_password` resolves it from the
environment or a file, the same way the existing `debt-ev` app does, so it goes
straight into the connection string and nothing else ever sees it.

WHICH SOURCE FOR WHICH STRATEGY - this is not a preference, it is a constraint:

  The standardised `volatility_surface` delta grid runs 10..90 by 5. A 5-delta
  wing is OUTSIDE it. `OptionMetricsSurface` holds flat outside the grid rather
  than extrapolate, so a 5-delta strategy read off the standardised surface is
  silently reading the 10-delta vol. Anything below 10 delta must use the chain.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

DEFAULT_SERVER = "SAB-CT-L3-DEV01"
DEFAULT_DATABASE = "IvyDB-US"
DEFAULT_USERNAME = "IvyDB-User"
DEFAULT_DRIVER = "ODBC Driver 18 for SQL Server"

SPX_SECURITYID = 108105

# The standardised surface's grid, so callers can check a target is ON it
# before trusting a number read off it.
SURFACE_TENOR_DAYS = (10, 30, 60, 91, 122, 152, 182, 273, 365, 547, 730)
SURFACE_MIN_ABS_DELTA = 10
SURFACE_MAX_ABS_DELTA = 90


def read_password(path: str | None = None) -> str:
    """Resolve the IvyDB password without it passing through the caller.

    Same resolution order as the `debt-ev` app, reimplemented here so this
    package does not import from a sibling repo on H:.
    """
    env = os.getenv("IVYDB_PASSWORD", "").strip()
    if env:
        return env
    candidates: list[Path] = []
    configured = path or os.getenv("IVYDB_PASSWORD_FILE", "").strip()
    if configured:
        candidates.append(Path(configured))
    profile = os.getenv("USERPROFILE", "")
    if profile:
        candidates.append(Path(profile) / "Desktop" / "ivydb_password.txt")
        candidates.append(
            Path(profile) / "OneDrive - Saba Capital Management LP"
            / "Desktop" / "ivydb_password.txt"
        )
    for c in candidates:
        try:
            if c.exists():
                value = c.read_text(encoding="utf-8").strip()
                if value:
                    return value
        except OSError:
            continue
    raise ValueError(
        "IvyDB password not found. Set IVYDB_PASSWORD, or IVYDB_PASSWORD_FILE, "
        "or save Desktop\\ivydb_password.txt."
    )


def connect(server=DEFAULT_SERVER, database=DEFAULT_DATABASE,
            username=DEFAULT_USERNAME, driver=DEFAULT_DRIVER,
            password: str | None = None, timeout: int = 120):
    """Open a pyodbc connection. Imported lazily so the package installs without it."""
    import pyodbc

    cs = ";".join([
        f"Driver={{{driver}}}", f"Server={server}", f"Database={database}",
        f"UID={username}", f"PWD={password or read_password()}",
        "Encrypt=yes", "TrustServerCertificate=yes",
    ])
    return pyodbc.connect(cs, timeout=timeout)


def securityid(conn, ticker: str = "SPX") -> int:
    cur = conn.cursor()
    cur.execute("SELECT TOP 1 [securityid] FROM [security] WHERE RTRIM([ticker]) = ?", ticker)
    row = cur.fetchone()
    if row is None:
        raise KeyError(f"no securityid for ticker {ticker!r}")
    return int(row[0])


# --------------------------------------------------------------------------
# Reference series
# --------------------------------------------------------------------------


def load_prices(conn, secid: int, start, end) -> pd.DataFrame:
    """date, close from `security_price`."""
    df = pd.read_sql(
        "SELECT [date], closeprice FROM dbo.security_price "
        "WHERE securityid = ? AND [date] >= ? AND [date] <= ? ORDER BY [date]",
        conn, params=[secid, str(start), str(end)],
    )
    return df.rename(columns={"date": "date", "closeprice": "close"}).assign(
        date=lambda d: pd.to_datetime(d["date"])
    )


def load_rates(conn, start, end, tenor_days: int = 30) -> pd.DataFrame:
    """date, rate - the zero curve collapsed to ONE tenor.

    `OptionMetricsSurface` and `OptionChainSurface` both take a flat per-date
    rate, so the curve is sampled at the tenor nearest `tenor_days` rather than
    interpolated per option. At 10DTE the difference is immaterial; at 1Y it is
    not, and that is a known limitation rather than a hidden one.
    Rates are stored in PERCENT and returned as a decimal.
    """
    df = pd.read_sql(
        "SELECT [date], days, rate FROM dbo.zero_curve "
        "WHERE [date] >= ? AND [date] <= ? ORDER BY [date], days",
        conn, params=[str(start), str(end)],
    )
    df["date"] = pd.to_datetime(df["date"])
    df["gap"] = (df["days"] - tenor_days).abs()
    pick = df.sort_values(["date", "gap"]).groupby("date", as_index=False).first()
    return pick[["date", "rate"]].assign(rate=lambda d: d["rate"] / 100.0)


def load_divs(conn, secid: int, start, end, tenor_days: int = 91) -> pd.DataFrame:
    """date, rate - the index dividend yield at `tenor_days`, as a decimal.

    `index_dividend` is a term structure keyed by expiration and the short end
    reads exactly ZERO: on SPX 2026-09-14 it is 0.0% out to ~60 days, 0.44% at
    95d, 0.60% at 368d.

    That zero is OptionMetrics' CONVENTION, not missing data - confirmed against
    their own `forward_price`, where the 10-day forward of 7628.75 on a spot of
    7619.98 implies (r-q) = 4.196%, exactly the 10-day zero rate. So q really is
    0 at the short end in the forward they used to compute their IVs and deltas.

    The consequence is still worth knowing: a long forward leg therefore pays
    full r with no dividend offset, which is a real carry drag against spot
    (~1.4%/yr over 2018-2026) and is why a rolled-forward benchmark does not
    match SPX's own Sharpe. That is the vendor's convention showing through, not
    an error to correct - but if you want the forward to agree with theirs
    exactly, pass `load_forwards` to the adapter instead of reconstructing it.

    Taking the nearest expiration for every tenor was wrong regardless: a 1Y leg
    would get the 0-day yield. Sample at the tenor you actually trade.
    """
    df = pd.read_sql(
        "SELECT [date], expiration, rate FROM dbo.index_dividend "
        "WHERE securityid = ? AND [date] >= ? AND [date] <= ? ORDER BY [date], expiration",
        conn, params=[secid, str(start), str(end)],
    )
    df["date"] = pd.to_datetime(df["date"])
    df["dte"] = (pd.to_datetime(df["expiration"]) - df["date"]).dt.days
    df["gap"] = (df["dte"] - tenor_days).abs()
    pick = df.sort_values(["date", "gap"]).groupby("date", as_index=False).first()
    return pick[["date", "rate"]].assign(rate=lambda d: d["rate"] / 100.0)


# --------------------------------------------------------------------------
# Surfaces
# --------------------------------------------------------------------------


def load_surface(conn, secid: int, start, end) -> pd.DataFrame:
    """The standardised delta-grid surface, renamed for `data.OptionMetricsSurface`.

    Remember the grid stops at 10 delta (`SURFACE_MIN_ABS_DELTA`). Reading a
    5-delta wing off this returns the 10-delta vol, silently.
    """
    df = pd.read_sql(
        "SELECT [date], days, delta, callput, impliedvolatility "
        "FROM dbo.volatility_surface "
        "WHERE securityid = ? AND [date] >= ? AND [date] <= ? ORDER BY [date], days, delta",
        conn, params=[secid, str(start), str(end)],
    )
    df["date"] = pd.to_datetime(df["date"])
    return df.rename(columns={"callput": "cp_flag", "impliedvolatility": "impl_volatility"})


def load_chain(conn, secid: int, start, end,
               max_dte: int = 45, min_dte: int = 0,
               moneyness_band: float = 0.15,
               two_sided_only: bool = True) -> pd.DataFrame:
    """The full chain, bounded server-side, in `chain.OptionChainSurface` columns.

    The bounds are not optional tuning: the unbounded SPX chain is tens of
    millions of rows per decade. `moneyness_band` is a fraction of the close
    (0.15 keeps strikes within +/-15% of spot), applied in SQL via a join to
    `security_price` so the rows never leave the server.
    """
    sql = """
        SELECT op.[date], op.expiration, op.callput, op.strike,
               op.bestbid, op.bestoffer, op.volume, op.openinterest,
               op.impliedvolatility, op.delta
        FROM dbo.option_price AS op
        JOIN dbo.security_price AS sp
          ON sp.securityid = op.securityid AND sp.[date] = op.[date]
        WHERE op.securityid = ?
          AND op.[date] >= ? AND op.[date] <= ?
          AND DATEDIFF(day, op.[date], op.expiration) BETWEEN ? AND ?
          AND op.strike BETWEEN sp.closeprice * 1000 * (1 - ?) AND sp.closeprice * 1000 * (1 + ?)
    """
    params = [secid, str(start), str(end), min_dte, max_dte, moneyness_band, moneyness_band]
    if two_sided_only:
        sql += " AND op.bestbid > 0"
    df = pd.read_sql(sql, conn, params=params)

    from .chain import normalize_ivydb
    return normalize_ivydb(df)


def load_forwards(conn, secid: int, start, end, am_settlement: bool = False) -> pd.DataFrame:
    """date, dte, forward - OptionMetrics' OWN implied forward curve.

    Prefer this to reconstructing S*exp((r-q)T). Verified on SPX 2026-09-14:
    spot 7619.98 with a 10-day forward of 7628.75 implies (r-q) = 4.196%, which
    is exactly the 10-day zero rate - i.e. OptionMetrics carries q = 0 at the
    short end, by convention rather than by accident. Reconstruction only agrees
    with them if you reproduce that convention exactly, and any drift between
    your forward and theirs lands straight in the P&L, because their IVs and
    deltas were computed on THEIR forward.

    SPX has both PM-settled (`amsettlement = 0`, the weeklies) and AM-settled
    (the third-Friday) contracts, which carry different forwards on the same
    expiry date. Pick one convention and stay on it.
    """
    df = pd.read_sql(
        "SELECT [date], expiration, forwardprice FROM dbo.forward_price "
        "WHERE securityid = ? AND [date] >= ? AND [date] <= ? AND amsettlement = ? "
        "ORDER BY [date], expiration",
        conn, params=[secid, str(start), str(end), 1 if am_settlement else 0],
    )
    df["date"] = pd.to_datetime(df["date"])
    df["dte"] = (pd.to_datetime(df["expiration"]) - df["date"]).dt.days
    return df.rename(columns={"forwardprice": "forward"})[["date", "dte", "forward"]]
