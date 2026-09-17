"""Black-Scholes on forwards. All functions vectorise over numpy arrays.

Convention throughout the package: options are quoted and struck off the
forward F, tenors T are in years, and `delta` is always the absolute
forward delta (so a "5-delta call" and a "5-delta put" are both 0.05).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def d1_d2(F, K, T, sigma):
    F, K, T, sigma = map(np.asarray, (F, K, T, sigma))
    T = np.maximum(T, 1e-8)
    sigma = np.maximum(sigma, 1e-8)
    vsqrt = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / vsqrt
    return d1, d1 - vsqrt


def price(F, K, T, sigma, cp, df=1.0):
    """Undiscounted-forward price times df. cp=+1 call, -1 put."""
    d1, d2 = d1_d2(F, K, T, sigma)
    cp = np.asarray(cp)
    return df * cp * (F * norm.cdf(cp * d1) - K * norm.cdf(cp * d2))


def delta(F, K, T, sigma, cp, df=1.0):
    """Forward delta. Signed: positive for calls, negative for puts."""
    d1, _ = d1_d2(F, K, T, sigma)
    cp = np.asarray(cp)
    return df * cp * norm.cdf(cp * d1)


def gamma(F, K, T, sigma, df=1.0):
    d1, _ = d1_d2(F, K, T, sigma)
    T = np.maximum(np.asarray(T), 1e-8)
    return df * norm.pdf(d1) / (np.asarray(F) * np.asarray(sigma) * np.sqrt(T))


def vega(F, K, T, sigma, df=1.0):
    """Vega per 1.00 of vol (i.e. per 100 vol points)."""
    d1, _ = d1_d2(F, K, T, sigma)
    return df * np.asarray(F) * norm.pdf(d1) * np.sqrt(np.maximum(T, 1e-8))


def theta_1d(F, K, T, sigma, cp, df=1.0):
    """One calendar-day theta by finite difference on T."""
    dt = 1.0 / 365.0
    return price(F, K, np.maximum(T - dt, 1e-8), sigma, cp, df) - price(F, K, T, sigma, cp, df)


def strike_from_delta(F, T, target_delta, cp, surface_iv, tol=1e-6, max_iter=60):
    """Solve for the strike whose delta equals `target_delta` ON the given surface.

    `surface_iv` is a callable K -> implied vol. Solving on the surface rather
    than at a flat vol matters: on a skewed surface the 5-delta put strike
    moves several percent once its own (higher) vol is fed back in, and getting
    this wrong silently changes which part of the surface you are testing.
    """
    if not 0.0 < target_delta < 1.0:
        raise ValueError(f"target_delta must be in (0,1), got {target_delta}")

    def f(logk):
        K = F * np.exp(logk)
        sig = float(surface_iv(K))
        return abs(float(delta(F, K, T, sig, cp))) - target_delta

    # Delta is monotone in strike, so bracket wide and bisect.
    lo, hi = -3.0, 3.0
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        raise RuntimeError(
            f"cannot bracket {target_delta:.3f}-delta (cp={cp}, T={T:.4f}); "
            f"f(lo)={flo:.4f} f(hi)={fhi:.4f} - check surface bounds"
        )
    logk = brentq(f, lo, hi, xtol=tol, maxiter=max_iter)
    return float(F * np.exp(logk))


def implied_vol(px, F, K, T, cp, df=1.0):
    """Invert price to vol. Returns nan outside [1e-4, 5.0]."""
    intrinsic = max(float(cp) * (F - K), 0.0) * df
    if px <= intrinsic + 1e-12:
        return np.nan
    try:
        return brentq(lambda s: float(price(F, K, T, s, cp, df)) - px, 1e-4, 5.0, xtol=1e-8)
    except ValueError:
        return np.nan


def implied_vol_vec(px, F, K, T, cp, df=1.0, lo=1e-4, hi=5.0, iters=52):
    """Vectorised price -> vol inversion. Returns nan where no solution exists.

    Re-implying a full option chain is ~10^5-10^6 inversions per backtest, so the
    scalar `implied_vol` brentq path is not usable there.

    Bisection rather than Newton, despite Newton's better asymptotics. Half a
    listed chain is cheap wings quoted near the tick, where vega underflows and
    Newton either stalls on its seed or steps wildly; falling back to a scalar
    solver for those turned out to cost ~40x the whole vectorised pass. Price is
    strictly monotone in sigma, so bisection cannot fail, and 52 halvings of
    [1e-4, 5] is already below double precision.
    """
    px, F, K, T, cp = (np.asarray(x, dtype=float) for x in (px, F, K, T, cp))
    px, F, K, T, cp = np.broadcast_arrays(px, F, K, T, cp)
    df_a = np.broadcast_to(np.asarray(df, dtype=float), px.shape)

    intrinsic = np.maximum(cp * (F - K), 0.0) * df_a
    upper = np.where(cp > 0, F, K) * df_a          # call <= F*df, put <= K*df
    ok = np.isfinite(px) & (px > intrinsic + 1e-12) & (px < upper - 1e-12)

    out = np.full(px.shape, np.nan)
    if not ok.any():
        return out

    Fi, Ki, Ti, cpi, dfi, pxi = (
        F[ok], K[ok], np.maximum(T[ok], 1e-8), cp[ok], df_a[ok], px[ok],
    )
    a = np.full(pxi.shape, float(lo))
    b = np.full(pxi.shape, float(hi))
    for _ in range(iters):
        m = 0.5 * (a + b)
        too_cheap = price(Fi, Ki, Ti, m, cpi, dfi) < pxi
        a = np.where(too_cheap, m, a)
        b = np.where(too_cheap, b, m)

    out[ok] = 0.5 * (a + b)
    return out


def strike_from_delta_flat(F, T, target_delta, cp, sigma):
    """Closed-form strike at `target_delta` for a CONSTANT vol. Vectorised.

    K = F * exp(0.5*sigma^2*T - cp*Phi^-1(target)*sigma*sqrt(T))

    `strike_from_delta` brentqs because the vol moves with the strike it is
    solving for. On a delta-GRID surface each point already carries its own
    vol, so the vol is constant for that solve and the root is analytic -
    identical answer, no iteration. Loading a decade of the standardised
    surface is ~10^6 of these, which is minutes of brentq and instant here.
    """
    F, T, target_delta, cp, sigma = map(np.asarray, (F, T, target_delta, cp, sigma))
    T = np.maximum(T, 1e-8)
    sigma = np.maximum(sigma, 1e-8)
    vsqrt = sigma * np.sqrt(T)
    return F * np.exp(0.5 * sigma**2 * T - cp * norm.ppf(target_delta) * vsqrt)
