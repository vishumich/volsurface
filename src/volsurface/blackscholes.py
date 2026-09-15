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
