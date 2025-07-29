#!/usr/bin/env python
"""
Robust companion‑matrix power / Rayleigh‑quotient iteration
"""
import numpy as np
import argparse, time
from scipy.optimize import linear_sum_assignment   # Hungarian
from scipy.linalg import null_space

# ----------------------------------------------------------------------
# tunables (can also be set from the command line via ‑‑scale, ‑‑maxp …)
verbose      = False
do_scale     = False      # coefficient scaling improves robustness
max_polish   = 10         # Newton steps per root
pow_tol      = 1e-12      # stopping tolerance for the eigen solver
shift_blowup = 1e100      # reject (A‑λI)⁻¹x if it exceeds this norm
# ----------------------------------------------------------------------

# ----------------------- helpers --------------------------------------

def random_poly(n):
    rr = roots = np.random.randn(n) + 1j * np.random.randn(n)
    rp = np.poly(roots)
    rp = rp / np.max(np.abs(rp))
    return rp

def random_unit(n):
    while True:
        v = np.random.randn(n) + 1j * np.random.randn(n)
        vn = np.linalg.norm(v)
        if vn >= 1e-5:
            v /= vn
            return v


def safe_norm(x):
    m = np.abs(x).max()
    if m == 0 or not np.isfinite(m):
        return m                      # 0, inf or nan
    return m * np.linalg.norm(x / m)

def vector2root(A, x):
    m = np.abs(x).max()
    if m == 0 or not np.isfinite(m):
        return np.nan
    y = A @ (x / m)
    return np.vdot(x / m, y) / np.vdot(x / m, x / m)

def root2vector(A, r, vinit):
    n = A.shape[0]
    I = np.eye(n, dtype=A.dtype)
    M = A - r * I

    try:
        w = np.linalg.solve(M, vinit)
    except np.linalg.LinAlgError:
        raise ValueError("Singular matrix; cannot solve.")
    
    v = w / np.linalg.norm(w)
    return v

# ----------------------- polynomial utilities -------------------------
#def dominant_radius(coeffs):
#    return 10.0 * np.max(np.abs(coeffs))

def dominant_radius(coeffs):
    n = len(coeffs) - 1
    mags = np.abs(coeffs[:-1])
    if n == 0 or mags.max() == 0:
        return 1.0
    k = np.arange(n - 1, -1, -1)
    return 2.0 * np.power(mags, 1.0 / (n - k)).max()

def companion_matrix(coeffs):
    """n×n upper‑Hessenberg companion of a monic polynomial."""
    coeffs = np.atleast_1d(coeffs).astype(np.complex128)
    n = coeffs.size - 1
    coeffs = coeffs / coeffs[0]
    C = np.zeros((n, n), dtype=np.complex128)
    if n > 1:
        C[:-1, 1:] = np.eye(n - 1)
    C[-1, :] = -coeffs[:0:-1]
    return C

def synthetic_division(coeffs, root):
    """p(x) /(x−root)  →  quotient q, remainder r  (Horner)."""
    n = len(coeffs) - 1
    q = np.empty(n, dtype=coeffs.dtype)
    q[0] = coeffs[0]
    for k in range(1, n):
        q[k] = coeffs[k] + root * q[k - 1]
    rem = coeffs[-1] + root * q[-1]
    return q, rem

def seed_vector(coeffs, m=8):
    A = companion_matrix(coeffs)
    r = dominant_radius(coeffs)
    thetas = 2 * np.pi * np.arange(m) / m
    lam_cands = r * np.exp(1j * thetas)
    errs   = np.abs(np.polyval(coeffs, lam_cands))
    z0     = lam_cands[errs.argmin()]
    return root2vector(A,z0,random_unit(A.shape[0])),z0

# ----------------------- eigen‑pair of largest |λ| --------------------
def power_root(coeffs, *, max_iter=1000, tol=pow_tol):
    coeffs = np.asarray(coeffs, np.complex128) / coeffs[0]
    A = companion_matrix(coeffs)
    x, lam = seed_vector(coeffs)
    chg,err,coef_norm = 0,0,np.sum(np.abs(coeffs))

    for it in range(max_iter):
        err =  np.abs(np.polyval(coeffs, lam))
        if err < tol * coef_norm:
            return lam, it, 0.0    # already accurate
        # (1) plain power step     
        x  = A @ x
        m =  safe_norm(x)
        if m == 0 or not np.isfinite(m):
            if verbose: print(f"coeffs:{coeffs.size} iteration: {it} restart")
            x, lam = seed_vector(coeffs)
            m = 1
        x /= m
        # (2) Reverse step
        x1 = root2vector(A,lam,x)
        # (3) Rayleigh quotient and cheap residual test
        lam_new = vector2root(A, x1) # Rayleigh
        chg = abs(lam_new - lam)

        if chg < tol * abs(lam_new):
            return lam_new, it, abs(lam_new - lam)
        lam = lam_new

    # max_iter exhausted – return best we have
    return lam, max_iter, np.abs(np.polyval(coeffs, lam))

# ----------------------- all roots by deflation ----------------------
def power_roots(coeffs, *, max_iter=1000, tol=pow_tol):
    coeffs = np.asarray(coeffs, np.complex128) / coeffs[0]
   
    roots  = []
    imax = 0; last_err = 0.0
    if verbose:
        print(f"Start on {coeffs.size-1}")
    while len(coeffs) > 3:  # quad or linear left → closed form
        best_root, best_err = None,1e100
        for i in range(4):
            root, it, err = power_root(coeffs, max_iter=max_iter, tol=tol)
            if err<best_err: 
                best_root=root
                best_err=err
                if best_err<1e-10:
                    break      
        root = best_root if best_root is not None else root
        if verbose: 
            lerr = np.log(err) if np.isfinite(err) and not err==0 else 0
            print(
                f"degree:{coeffs.size-1} "
                f"err: {round(lerr,2)} "
                f"root: {round(root.real)},{round(root.imag,2)}j "
                f"iterations: {it} "
                f"root norm: {round(np.linalg.norm(root),2)} "
                f"coeff norm: {round(np.linalg.norm(coeffs),2)} "
            )
        roots.append(root)
        coeffs, _ = synthetic_division(coeffs, root)
        imax  = max(imax, it)
        last_err = max(err,last_err)

    # leftover of degree ≤ 2
    if len(coeffs) == 3:
        a, b, c = coeffs
        disc = np.lib.scimath.sqrt(b * b - 4 * a * c)
        roots.extend([(-b + disc) / (2 * a), (-b - disc) / (2 * a)])
    elif len(coeffs) == 2:
        roots.append(-coeffs[1] / coeffs[0])

    return np.asarray(roots), imax, last_err

# ----------------------- CLI driver ----------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="safe power / RQI roots")
    parser.add_argument("-n", "--deg",  type=int, default=10,  help="degree")
    parser.add_argument("--maxi",      type=int, default=1000, help="max iterations")
    parser.add_argument("--tol",       type=float, default=1e-12, help="tolerance")
    parser.add_argument("--verbose",   action="store_true")
    args = parser.parse_args()

    verbose      = args.verbose
    pow_tol      = args.tol

    p    = random_poly(args.deg)          # random (degree = n−1) poly
    p[0] = 1.0                            # make it monic
    print(f"degree:  {p.size-1} norm: {np.linalg.norm(p)} root norm: {np.linalg.norm(np.roots(p))}")

    t0 = time.perf_counter()
    r_numpy = np.roots(p)
    print(f"np.roots:     {round((time.perf_counter() - t0)*1000):>6} ms")

    t0 = time.perf_counter()
    r_power, it_max, err_max = power_roots(p,max_iter=args.maxi,tol=pow_tol)
    print(f"power_roots:  {round((time.perf_counter() - t0)*1000):>6} ms")
    print(
        f"max_iter per root: {it_max},  "
        f"max err: {err_max:.1e}, "
        f"max value power: {np.max(np.abs(np.polyval(p,r_power)))}, "
        f"max value numpy: {np.max(np.abs(np.polyval(p,r_numpy)))}"
    )

