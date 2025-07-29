#!/usr/bin/env python
#
# bunch of poly solvers
#
import numpy as np
import mps 
import argparse
import timeit
from numba import njit, prange

def weierstrass_gs(p_coeffs, x0, max_iter=100, tol=1e-12, verbose=False):
    x = np.array(x0, dtype=np.complex128, copy=True)   # isolate from caller
    n = x.size
    ones = np.ones(n, dtype=np.complex128)
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            diff = x[i] - x
            diff[i] = 1.0                               
            denom = diff.prod()
            if denom == 0:                             
                continue
            numer = np.polyval(p_coeffs, x[i])
            x[i] -= numer / denom                       
        delta = np.abs(x - x_old).max()
        if verbose:
            print(f"iter {k:2d}: max |Δx| = {delta:.2e}")
        if delta < tol:
            break
    return x, k

@njit
def horner(p, x):
    s = 0.0 + 0j
    for c in p:
        s = s * x + c
    return s

@njit
def weierstrass_gs_numba(p_coeffs, x0, max_iter=100, tol=1e-12):
    x = x0.copy()
    n = x.size
    for k in range(max_iter):
        delta = 0.0
        for i in range(n):
            denom = 1.0 + 0j
            xi = x[i]
            for j in range(n):
                if j != i:
                    denom *= xi - x[j]
            if denom != 0:
                numer = horner(p_coeffs, xi)
                new_xi = xi -  numer / denom
                delta = max(delta, abs(new_xi - xi))
                x[i] = new_xi
        if delta < tol:
            break
    return x, k

def weierstrass_jcb(p_coeffs, x0, max_iter=100, tol=1e-12, verbose=False):
    x = np.asarray(x0, dtype=np.complex128, copy=True)
    n = x.size
    diag_idx = np.diag_indices(n)
    for k in range(max_iter):
        x_old = x.copy()
        diff = x[:, None] - x[None, :]
        diff[diag_idx] = 1.0
        denom = diff.prod(axis=1)
        numer = np.polyval(p_coeffs, x)
        nz = denom != 0
        x[nz] -= numer[nz] / denom[nz]
        delta = np.abs(x - x_old).max()
        if verbose:
            print(f"iter {k:2d}: max |Δx| = {delta:.2e}")
        if delta < tol:
            break
    return x, k

def weierstrass_jcb_sor(p_coeffs, x0, *,
                              omega=0.5,      # 1.0 → plain Jacobi
                              max_iter=100, tol=1e-12,
                              verbose=False):
    x = np.asarray(x0, dtype=np.complex128, copy=True)
    n = x.size
    diag = np.diag_indices(n)

    for k in range(max_iter):
        diff = x[:, None] - x[None, :]
        diff[diag] = 1.0
        denom = diff.prod(axis=1)
        numer = np.polyval(p_coeffs, x)

        dx = np.where(denom != 0, -numer / denom, 0.0)
        x_new = x + omega * dx           # relaxation

        if verbose:
            print(f"iter {k:2d}: max |Δx| = {np.abs(dx).max():.2e}")

        if np.abs(dx).max() < tol:
            return x_new, k
        x = x_new

    return x, k

@njit
def polyval_vec(coeffs, x):
    """
    Evaluate a polynomial with complex coefficients at all points in x
    using Horner's rule, entirely inside nopython mode.
    """
    y = np.zeros_like(x)
    for c in coeffs:               # highest degree first
        y = y * x + c
    return y                       # shape == x.shape


# ----------  damped Jacobi / SOR kernel  ----------
@njit(fastmath=True)
def weierstrass_jcb_sor_numba(p_coeffs, x0,omega=0.5,max_iter=100,tol=1e-12):
    """
    Numba‑accelerated damped‑Jacobi / SOR Weierstrass root finder.
    Parameters are identical to the pure‑NumPy version.
    """
    x = x0.copy()
    n = x.size
    denom = np.empty(n, dtype=np.complex128)
    dx = np.empty(n, dtype=np.complex128)
    numer = np.empty(n, dtype=np.complex128)

    for k in range(max_iter):
        # outer loop can be parallelised; each iteration is independent
        for i in range(n):
            xi = x[i]
            di = 1.0 + 0j
            for j in range(n):
                if i != j:
                    di *= (xi - x[j])
            denom[i] = di
            numer[i] = horner(p_coeffs, xi)

        # polynomial values p(x_i)
        #numer = polyval_vec(p_coeffs, x)

        # Jacobi update with relaxation
       
        max_delta = 0.0
        for i in range(n):
            if denom[i] != 0.0:
                dx_i = -numer[i] / denom[i]
            else:
                dx_i = 0.0         # extremely rare: two iterates coincide
            dx[i] = dx_i
            if abs(dx_i) > max_delta:
                max_delta = abs(dx_i)

        x += omega * dx

        if max_delta < tol:
            break

    return x,k

def aberth(p_coeffs, x0, max_iter=100, tol=1e-12, verbose=False):
    x = np.asarray(x0, dtype=np.complex128, copy=True)
    n = len(x)
    p_der = np.polyder(p_coeffs)
    for k in range(max_iter):
        x_old = x.copy()
        pvals = np.polyval(p_coeffs, x)
        pdervals = np.polyval(p_der, x)
        N = np.zeros_like(x)
        np.divide(pvals, pdervals, out=N, where=(pdervals != 0))
        
        # Compute sum_{j != i} 1/(x_i - x_j)
        X = x[:, np.newaxis]
        D = X - x[np.newaxis, :]
        mask = np.eye(n, dtype=bool)
        inv_D = np.zeros_like(D)
        np.divide(1.0, D, out=inv_D, where=~mask)
        sums = np.sum(inv_D, axis=1)
        
        denoms = 1.0 - N * sums
        corrections = np.zeros_like(x)
        np.divide(N, denoms, out=corrections, where=(denoms != 0))
        x -= corrections
        
        delta = np.max(np.abs(x - x_old))
        if verbose:
            print(f"iter {k}, max Δx = {delta:.2e}")
        if delta < tol:
            break
    return x, k

@njit(parallel=True, fastmath=True)
def aberth_numba(p_coeffs, p_prime_coeffs, x0, max_iter=100, tol=1e-12):
    x = x0.copy()
    n = x.size
    for k in range(max_iter):
        # p, p'  in one Horner pass (evaluate both at once)
        p  = polyval_vec(p_coeffs,  x)
        pp = polyval_vec(p_prime_coeffs, x)

        # denominator term Σ_j≠i 1/(xi - xj)
        sigma = np.empty(n, dtype=np.complex128)
        for i in prange(n):
            s = 0.0 + 0j
            xi = x[i]
            for j in range(n):
                if i != j:
                    s += 1.0 / (xi - x[j])
            sigma[i] = s

        dx = -p / pp / (1.0 - p / pp * sigma)
        x += dx
        if np.abs(dx).max() < tol:
            break
    return x,k

@njit(fastmath=True)
def horner_pair(p, pp, x):
    """
    Evaluate p(x) and p'(x) together (complex128).
    p  : coeffs of p, length m+1
    pp : coeffs of p', length m
    x  : scalar
    """
    y  = p[0]
    yp = pp[0]
    for k in range(1, p.size):
        y  = y  * x + p[k]
        if k < pp.size:
            yp = yp * x + pp[k]
    return y, yp


@njit(fastmath=True)
def aberth_numba_fast(p_coeffs, x0, max_iter=100, tol=1e-12):
    """
    Faster Aberth–Ehrlich with Numba; tuned for small/medium n (≤ 200).
    """
    n          = x0.size
    x          = x0.copy()
    pprime     = p_coeffs[:-1] * np.arange(p_coeffs.size - 1, 0, -1)
    sigma      = np.empty(n, dtype=np.complex128)
    pvals      = np.empty(n, dtype=np.complex128)
    pprimevals = np.empty(n, dtype=np.complex128)

    for iter in range(max_iter):
        # --- evaluate p and p' ---
        max_delta = 0.0
        for i in range(n):
            pvals[i], pprimevals[i] = horner_pair(p_coeffs, pprime, x[i])

        # --- build σ_i = Σ_{j≠i} 1/(x_i - x_j) ---
        for i in range(n):
            s   = 0.0 + 0j
            xi  = x[i]
            for j in range(n):
                if i != j:
                    s += 1.0 / (xi - x[j])
            sigma[i] = s

        # --- correction step ---
        for i in range(n):
            r      = pvals[i] / pprimevals[i]
            dx     = -r / (1.0 - r * sigma[i])
            x[i]  += dx
            if abs(dx) > max_delta:
                max_delta = abs(dx)

        if max_delta < tol:
            break

    return x,iter

def scale_poly_pow2(coeffs):
    """
    Return (scaled_coeffs, scale_factor), where
      q(y) = coeffs_scaled[0] * y**n + ... + coeffs_scaled[n]
      q(y) = 0   ↆ  p(z)=0 with z = s*y
    """
    coeffs = coeffs.astype(np.complex128, copy=True)
    coeffs /= coeffs[0]                       # make monic
    n  = coeffs.size - 1
    m  = np.max(np.abs(coeffs))
    s  = 2.0 ** np.round(np.log2(m) / n)      # power‑of‑two scale

    # p(s*y) = Σ a_k (s*y)^{n-k} = Σ (a_k * s^{n-k}) y^{n-k}
    powers = s ** np.arange(n, -1, -1)        # s^n, s^{n-1}, …, s^0
    coeffs_scaled = coeffs * powers           # q(y) coefficients
    return coeffs_scaled, s

def circle_guess(n, radius=1.0):
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    return radius * np.exp(1j * theta)

def random_poly(n):
    roots = np.random.randn(n)+ 1j * np.random.randn(n)
    return np.poly(roots)

def random_coefs(n):
    cf = np.random.randn(n)+ 1j * np.random.randn(n)
    return cf

def err(cf,x):
    return np.max(np.abs(np.polyval(cf,x)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="safe power / RQI roots")
    parser.add_argument("-p", "--poly",  type=str, default="1,2,3,4",  help="polynomial")
    parser.add_argument("-d", "--degree",  type=int, default=None,  help="polynomial")
    parser.add_argument("--maxi",      type=int, default=1000, help="max iterations")
    parser.add_argument("--tol",       type=float, default=1e-12, help="tolerance")
    parser.add_argument("--verbose",   action="store_true")
    args = parser.parse_args()

    if args.degree is None:
        cf = np.array([complex(x) for x in args.poly.split(',')])
    else:
        cf = random_poly(args.degree)

    bench = 10
    cf = cf / cf[0]
    #cf, s = scale_poly_pow2(cf)
    print(f"leftmost coeff: {cf[0]}")
    cfd = np.polyder(cf)
    cff = np.flip(cf)
    x0 = circle_guess(cf.size-1)
    #x0 = np.roots(cf)+0.000000001

    print("-"*40)
    print(f"initial guess error: {np.max(np.abs(np.polyval(cf,x0)))}")
    print("-"*40)
    print(f"np.roots() error: {err(cf,np.roots(cf))}")
    tnp = timeit.timeit("np.roots(cf)", globals=globals(), number=bench)
    print(f"np.roots execution time: {tnp:.6f} seconds ({tnp/tnp:.2f}x)")
    print("-"*40)
    gs_roots, i = weierstrass_gs(cf, x0,max_iter=100,tol=1e-12)
    print(f"gauss-siedel error: {err(cf,gs_roots)} [{i}]")
    tgs = timeit.timeit("weierstrass_gs(cf, x0,max_iter=100,tol=1e-12)", globals=globals(), number=bench)
    print(f"gauss-siedel time: {tgs:.6f} seconds  ({tnp/tgs:.2f}x)")
    print("-"*40)
    jcb_roots, i = weierstrass_jcb(cf, x0, max_iter=100, tol=1e-12)
    print(f"jacobi error: {err(cf,jcb_roots)} [{i}]")
    tjcb = timeit.timeit("weierstrass_jcb(cf, x0, max_iter=100, tol=1e-12)", globals=globals(), number=bench)
    print(f"jacobi time: {tjcb:.6f} seconds ({tnp/tjcb:.2f}x)")
    print("-"*40)
    jcbd_roots, i = weierstrass_jcb_sor(cf, x0, omega=0.5,max_iter=100,tol=1e-12)
    print(f"jacobi sor error: {err(cf,jcbd_roots)} [{i}]")
    tjcbsor = timeit.timeit("weierstrass_jcb_sor(cf, x0, omega=0.5,max_iter=100,tol=1e-12)", globals=globals(), number=bench)
    print(f"jacobi sor time: {tjcbsor:.6f} seconds ({tnp/tjcbsor:.2f}x)")
    print("-"*40)
    aberth_roots, i = aberth(cf, x0)
    print(f"aberth error: {err(cf,aberth_roots)} [{i}]")
    taberth = timeit.timeit("aberth(cf, x0)", globals=globals(), number=bench)
    print(f"aberth time: {taberth:.6f} seconds ({tnp/taberth:.2f}x)")
    print("-"*40)
    numba_roots,i = weierstrass_gs_numba(cf, x0, max_iter=100, tol=1e-12)
    print(f"numba gauss-siedel error: {err(cf,numba_roots)} [{i}]")
    tgsn = timeit.timeit("weierstrass_gs_numba(cf, x0, max_iter=100, tol=1e-12)", globals=globals(), number=bench)
    print(f"numba gauss-siedel time: {tgsn:.6f} seconds ({tnp/tgsn:.2f}x)")
    print("-"*40)
    numba_roots,i = weierstrass_jcb_sor_numba(cf, x0, omega=0.5, max_iter=100, tol=1e-12)
    print(f"numba jacobi sor error: {err(cf,numba_roots)}  [{i}]")
    tjcbsorn = timeit.timeit("weierstrass_jcb_sor_numba(cf, x0, omega=0.5, max_iter=100, tol=1e-12)", globals=globals(), number=bench)
    print(f"numba jacobi sor time: {tjcbsorn:.6f} seconds ({tnp/tjcbsorn:.2f}x)")
    print("-"*40)
    numba_roots,i = aberth_numba(cf, cfd, x0, max_iter=100, tol=1e-12)
    print(f"numba aberth error: {err(cf,numba_roots)}  [{i}]")
    taberthn = timeit.timeit("aberth_numba(cf, cfd, x0, max_iter=100, tol=1e-12)", globals=globals(), number=bench)
    print(f"numba aberth time: {taberthn:.6f} seconds ({tnp/taberthn:.2f}x)")
    print("-"*40)
    numba_roots,i = aberth_numba_fast(cf, x0, max_iter=100, tol=1e-12)
    print(f"numba aberth fast error: {err(cf,numba_roots)} [{i}]")
    taberthfn = timeit.timeit("aberth_numba_fast(cf, x0, max_iter=100, tol=1e-12)", globals=globals(), number=bench)
    print(f"numba aberth fast time: {taberthfn:.6f} seconds ({tnp/taberthfn:.2f}x)")
    print("-"*40)
    mps_roots = mps.roots(cff)
    print(f"mps error: {err(cf,mps_roots)}")
    tmps= timeit.timeit("mps.roots(cff)", globals=globals(), number=bench)
    print(f"mps time: {tmps:.6f} seconds ({tnp/tmps:.2f}x)")

   


