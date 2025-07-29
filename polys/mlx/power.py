#!/usr/bin/env python

#
# power method
#

import numpy as np
import argparse
from scipy.optimize import linear_sum_assignment  # Hungarian
import scipy.linalg as la
import time

verbose = False
do_scale = False
max_polish = 10

def max_pairing_error(a, b):
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return np.inf
    D = np.abs(a[:, None] - b[None, :])
    row, col = linear_sum_assignment(D)
    return D[row, col].max()

def random_poly(n: int) -> np.ndarray:
    rp = np.random.randn(n)
    ip = np.random.randn(n)
    return rp + 1j * ip

def safe_norm(x):
    m = np.abs(x).max()
    if m == 0 or not np.isfinite(m):
        return m # 0, inf, or nan – caller must handle
    return m * np.linalg.norm(x / m)

def rayleigh(A, x):
    m = np.abs(x).max()
    if m == 0 or not np.isfinite(m):
        return np.nan
    y  = A @ (x / m)
    return (np.vdot(x / m, y)) / (np.vdot(x / m, x / m))

def shifted_solve(A, lam, x, *, blow_up=1e100):
    n = A.shape[0]
    try:
        y = np.linalg.solve(A - lam*np.eye(n, dtype=A.dtype), x)
    except np.linalg.LinAlgError:
        return x, False # outright singular
    nrm = safe_norm(y)
    if nrm == 0 or not np.isfinite(nrm) or nrm > blow_up:
        return x, False # numerically useless
    return y / nrm, True

def dominant_radius(coeffs):
    n = len(coeffs) - 1
    mags = np.abs(coeffs[:-1])
    if n == 0 or mags.max() == 0:
        return 1.0
    k = np.arange(n - 1, -1, -1)
    return 2.0 * np.power(mags, 1.0 / (n - k)).max()

def safe_power(base, max_exp):
    """
    Computes an array of descending powers of a complex base: [base**max_exp, base**(max_exp-1), ..., base**0]
    in a numerically stable manner to avoid overflow.
    
    Parameters:
    - base: complex, the base value.
    - max_exp: int, the highest exponent (must be >= 0).
    
    Returns:
    - v: np.ndarray, array of powers.
    """
    if not isinstance(max_exp, int) or max_exp < 0:
        raise ValueError("max_exp must be a non-negative integer")
    length = max_exp + 1
    v = np.empty(length, dtype=np.complex128)
    r = np.abs(base)
    
    if r == 0:
        v[:] = 0
        v[-1] = 1.0 + 0j
        return v
    
    if r == 1:
        # Special case |base|==1, all |powers|==1, compute directly
        exponents = np.arange(max_exp, -1, -1)
        v = base ** exponents
    elif r <= 1:
        # Start from lowest power, multiply by base (|base|<=1)
        v[-1] = 1.0 + 0j
        for i in range(length - 2, -1, -1):
            v[i] = base * v[i + 1]
    else:
        # Start from 'highest' power (scaled), multiply by 1/base (|1/base|<1)
        v[0] = 1.0 + 0j
        inv_base = 1.0 / base
        for i in range(1, length):
            v[i] = inv_base * v[i - 1]
        # Apply phase correction to align with the direct computation
        unit_base = base / r
        phase_correction = unit_base ** max_exp
        v *= phase_correction
    
    return v

def seed_vector(coeffs, m = 8):
    r = dominant_radius(coeffs)       
    
    thetas  = 2*np.pi*np.arange(m)/m
    zcands  = r * np.exp(1j*thetas)
    errs    = np.abs(np.polyval(coeffs, zcands))
    z0      = zcands[errs.argmin()]
    n = len(coeffs) - 1
    v0 = safe_power(z0, n-1)
    v0 = v0 / np.linalg.norm(v0)
    return v0, z0

def scaled_monic(coeffs, threshold=50):
    coeffs = coeffs.astype(np.complex128)
    if coeffs[0] != 1:
        coeffs = coeffs / coeffs[0]
    mags = np.abs(coeffs[1:])
    if mags.min() == 0 or mags.max() / mags.min() <= threshold:
        return coeffs, 1.0
    n = len(coeffs) - 1
    powers = np.arange(n, -1, -1, dtype=float)
    scale  = (mags ** (1.0 / powers[:-1])).max()
    if scale ** n > 1e12 or scale ** n < 1e-12:
        return coeffs, 1.0
    return coeffs / scale**powers, scale

def companion_matrix(coeffs: np.ndarray) -> np.ndarray:
    coeffs = np.atleast_1d(coeffs).astype(np.complex128)
    n       = coeffs.size - 1
    coeffs  = coeffs / coeffs[0]          # make the polynomial monic

    C       = np.zeros((n, n), dtype=np.complex128)
    if n > 1:
        C[:-1, 1:] = np.eye(n-1)          # super‑diagonal ones
    C[-1, :] = -coeffs[:0:-1]             # ***reverse order here***
    return C

def synthetic_division(coeffs: np.ndarray, root: complex):
    n  = len(coeffs) - 1          # degree
    q  = np.empty(n, dtype=coeffs.dtype)
    q[0] = coeffs[0]              # = 1 if coeffs is monic
    for k in range(1, n):
        q[k] = coeffs[k] + root * q[k-1]
    rem = coeffs[-1] + root * q[-1]
    return q, rem

def safe_horner(coeffs, z):
    """Return (p(z), p′(z)) with overflow guards."""
    a = coeffs.astype(np.complex128)
    p, dp, scale = a[0], 0.0 + 0j, 1.0
    for c in a[1:]:
        if abs(p) > 1e200:                   # rescale to avoid overflow
            p  *= 1e-200
            dp *= 1e-200
            scale *= 1e200
        dp = dp * z + p
        p  = p  * z + c
    return p * scale, dp * scale


def polish_root(coeffs, z0, *, tol=1e-14, max_iter=5):
    z = z0
    p0, _ = safe_horner(coeffs, z)
    err0  = abs(p0)
    if not np.isfinite(err0):
        return z0, False

    for _ in range(max_iter):
        p, dp = safe_horner(coeffs, z)
        if abs(p) < tol * np.max(np.abs(coeffs)):
            return z, True           # residual tiny – done
        if dp == 0 or not np.isfinite(dp):
            break                    # derivative hopeless
        dz = -p / dp
        # step‑size safeguard
        if abs(dz) > 1.0:
            dz = dz / abs(dz)        # cap at 1.0
        z += dz
        if abs(dz) < tol * max(1.0, abs(z)):
            return z, True           # tiny correction
    # accept only if residual improved by ≥ four orders
    p1, _ = safe_horner(coeffs, z)
    if abs(p1) < 1e-4 * err0:
        return z, True
    return z0, False

def power_root(coeffs, *, max_iter=1000, tol=1e-12):
    coeffs = np.asarray(coeffs, np.complex128) / coeffs[0]
    A      = companion_matrix(coeffs)
    n      = A.shape[0]
    x, lam = seed_vector(coeffs)
    lam_old = lam
    for i in range(max_iter):
        # 1) plain power step
        x = A @ x
        nrm = safe_norm(x)
        if nrm == 0 or not np.isfinite(nrm):
            # fallback: random restart
            x = np.random.randn(n) + 1j*np.random.randn(n)
            x /= np.linalg.norm(x)
        else:
            x /= nrm
        lam = rayleigh(A, x)
        if not np.isfinite(lam):
            # restart with new random seed
            x, lam = seed_vector(coeffs)
            lam_old = lam
            continue
        # 2) shifted inverse step (one line turned safe)
        y, ok = shifted_solve(A, lam, x)
        if not ok:
            # drop the inverse step this round
            pass
        else:
            x = y / safe_norm(y)
        lam = rayleigh(A, x)
        if not np.isfinite(lam):
            x, lam = seed_vector(coeffs)
            lam_old = lam
            continue
        # 3) convergence test
        err = abs(lam - lam_old)
        if err < tol * abs(lam):
            break
        lam_old = lam
    return lam, i, err


def power_roots(coeffs, *, max_iter=1000, tol=1e-12):
    coeffs = np.asarray(coeffs, np.complex128) / coeffs[0]
    roots  = []
    if coeffs[0] != 1:
        coeffs = coeffs / coeffs[0]
    if do_scale:
        coeffs_scaled, scale   = scaled_monic(coeffs)
    else:
        coeffs_scaled, scale   = coeffs, 1 

    coeffs_fwd = coeffs_scaled.copy()                 # p_k(x)
    coeffs_rev = coeffs_scaled[::-1].copy() / coeffs_scaled[-1]  # q_k(x), always monic
    want_small = False
    iter = 0; imax = 0; err=0
    while len(coeffs_fwd) > 2:
        if want_small:
            R , iter, err = power_root(coeffs_rev, max_iter=max_iter, tol=tol)
            r = 1 / R
        else:
            r , iter, err = power_root(coeffs_fwd, max_iter=max_iter, tol=tol)

        r, problems = polish_root(coeffs_fwd, r, max_iter=max_polish)
        roots.append(r)
        coeffs_fwd, _ = synthetic_division(coeffs_fwd, r)
        coeffs_rev, _ = synthetic_division(coeffs_rev, 1/r)

        want_small = not want_small            # switch next time
        imax = max(iter,imax)

    if len(coeffs_fwd) == 2:
        roots.append(-coeffs_fwd[1])
    elif len(coeffs_fwd) == 3:
        a, b, c = coeffs_fwd
        disc = np.sqrt(b*b - 4*a*c)
        roots.extend([(-b+disc)/(2*a), (-b-disc)/(2*a)])

    return np.asarray(roots)/scale, imax, err



if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="palette maker")
    parser.add_argument('outfile',nargs="?",type=str, default="palette.png", help="outfile")
    parser.add_argument('-n','--deg', type=int, default=10, help="resolution")
    parser.add_argument('--maxi', type=int, default=1000, help="max iteration")
    parser.add_argument('--maxp', type=int, default=10, help="max polish")
    parser.add_argument('--tol', type=float, default=1e-12, help="max iteration")
    parser.add_argument('--verbose',action='store_true',help="verbose")
    parser.add_argument('--scale',action='store_true',help="verbose")
    args = parser.parse_args()
    verbose = args.verbose
    do_scale = args.scale
    max_polish = args.maxp
    p=random_poly(args.deg)
    start = time.perf_counter()
    r0 = np.roots(p)
    print(f"np.roots: {round((time.perf_counter() - start) * 1000)} ms")
    start = time.perf_counter()
    r2, max_iter, err = power_roots(p,max_iter=args.maxi,tol=args.tol)
    print(f"power_roots: {round((time.perf_counter() - start) * 1000)} ms")
    print(f"r0: {r0.size}")
    print(f"r2: {r2.size}")
    
    print(f"degree: {args.deg} error: {max_pairing_error(r0, r2)} iter: {max_iter} err: {err}")
