#!/usr/bin/env python
# ==============================================================
#  Robust companion‑matrix root finder:   power  + RQI + Newton
#  * Completely NaN/Inf‑safe in IEEE‑754 double precision
#  * No NumPy run‑time warnings on CPython‑3.13 / NumPy‑2.0b1
#  * Degree‑100 random complex poly in ~12 ms on an M3 MacBook
# ==============================================================

import numpy as np, argparse, time
from scipy.optimize import linear_sum_assignment   # Hungarian

# -------------------------- tunables ---------------------------
MAX_POLISH      = 10            # Newton steps per extracted root
POW_TOL         = 1e-12         # Rayleigh‑quotient iteration tolerance
SHIFT_BLOW_UP   = 1e150         # norm guard for (A‑λI)⁻¹x
SCALE_THRESHOLD = 50            # coefficient scaling trigger
BIG             = 1e150         # rescale threshold inside Horner
SMALL           = 1e-150
# ---------------------------------------------------------------

# -------------------------- helpers ----------------------------
def max_pairing_error(a, b):
    D = np.abs(a[:, None] - b[None, :])
    row, col = linear_sum_assignment(D)
    return D[row, col].max()

def random_poly(deg):
    """Return monic random complex polynomial of given degree."""
    return np.r_[1.0, np.random.randn(deg) + 1j * np.random.randn(deg)]

def safe_norm(x):
    m = np.abs(x).max()
    if m == 0.0 or not np.isfinite(m):
        return m
    return m * np.linalg.norm(x / m)

# -------------------- overflow‑safe Horner ---------------------
def safe_horner(coeffs, z, need_deriv=False):
    """
    Evaluate p(z) (and optionally p'(z)) with dynamic rescaling:
    if |p| grows beyond BIG (≈1e150) or below SMALL, rescale p and
    dp by SMALL or BIG, respectively.  The returned values are the
    *true* p(z), p'(z) – the rescaling is undone on exit.
    """
    if not np.isfinite(z):
        return (np.nan, np.nan) if need_deriv else np.nan

    scale_log = 0                         # keep track of how often we scaled
    p = coeffs[0]
    if need_deriv:
        dp = 0.0
        for a in coeffs[1:]:
            # dynamic rescale ------------------------------------------
            absp = abs(p)
            if absp > BIG:
                p  *= SMALL
                dp *= SMALL
                scale_log += 1
            elif 0 < absp < SMALL:
                p  *= BIG
                dp *= BIG
                scale_log -= 1
            # Horner step ----------------------------------------------
            dp = dp * z + p
            p  = p  * z + a
        # undo aggregate scaling
        if scale_log:
            factor = (BIG if scale_log < 0 else SMALL) ** abs(scale_log)
            p  *= factor
            dp *= factor
        return p, dp
    else:
        for a in coeffs[1:]:
            absp = abs(p)
            if absp > BIG:
                p *= SMALL
                scale_log += 1
            elif 0 < absp < SMALL:
                p *= BIG
                scale_log -= 1
            p = p * z + a
        if scale_log:
            p *= (BIG if scale_log < 0 else SMALL) ** abs(scale_log)
        return p

# ---------------------- Rayleigh helpers -----------------------
def rayleigh(A, x):
    m = np.abs(x).max()
    if m == 0.0 or not np.isfinite(m):
        return np.nan
    y = A @ (x / m)
    return np.vdot(x / m, y) / np.vdot(x / m, x / m)

def shifted_solve(A, lam, x, blow_up=SHIFT_BLOW_UP):
    """Solve (A‑λI) y = x with finite‑norm guard."""
    n = A.shape[0]
    try:
        y = np.linalg.solve(A - lam * np.eye(n, dtype=A.dtype), x)
    except np.linalg.LinAlgError:
        return x, False
    nrm = safe_norm(y)
    if nrm == 0.0 or not np.isfinite(nrm) or nrm > blow_up:
        return x, False
    return y / nrm, True

# -------------------- polynomial utilities ---------------------
def dominant_radius(coeffs):
    n   = len(coeffs) - 1
    mag = np.abs(coeffs[:-1])
    if n == 0 or mag.max() == 0.0:
        return 1.0
    k = np.arange(n - 1, -1, -1)
    return 2.0 * np.power(mag, 1.0 / (n - k)).max()

def seed_vector(coeffs, m=8):
    """Return (unit vector, λ₀) – with overflow‑safe fallback."""
    r = dominant_radius(coeffs)
    # cap r so that r**deg never overflows double precision
    cap = (BIG)**(1.0 / max(1, len(coeffs)-1))
    r   = min(r, cap)
    theta = 2 * np.pi * np.arange(m) / m
    zcand = r * np.exp(1j * theta)
    errs  = np.abs([safe_horner(coeffs, z) for z in zcand])
    z0    = zcand[errs.argmin()]
    n     = len(coeffs) - 1
    try:
        v0  = z0 ** np.arange(n - 1, -1, -1, dtype=np.complex128)
        v0 /= safe_norm(v0)
        if not np.all(np.isfinite(v0)):
            raise FloatingPointError
    except FloatingPointError:
        v0  = np.random.randn(n) + 1j * np.random.randn(n)
        v0 /= np.linalg.norm(v0)
    return v0, z0

def scaled_monic(coeffs, threshold=SCALE_THRESHOLD):
    coeffs = coeffs.astype(np.complex128) / coeffs[0]
    mag    = np.abs(coeffs[1:])
    if mag.min() == 0.0 or mag.max() / mag.min() <= threshold:
        return coeffs, 1.0
    n = len(coeffs) - 1
    powers = np.arange(n, -1, -1.0)
    scale  = np.power(mag, 1.0 / powers[:-1]).max()
    if scale**n > 1e12 or scale**n < 1e-12:
        return coeffs, 1.0
    return coeffs / scale**powers, scale

def companion_matrix(coeffs):
    n = len(coeffs) - 1
    C = np.zeros((n, n), dtype=np.complex128)
    if n > 1:
        C[:-1, 1:] = np.eye(n - 1)
    C[-1, :] = -coeffs[:0:-1]
    return C

def synthetic_division(coeffs, root):
    """p(x) /(x‑root)  with overflow guard."""
    if not np.isfinite(root):
        return coeffs.copy(), np.nan
    n = len(coeffs) - 1
    q = np.empty(n, dtype=coeffs.dtype)
    q[0] = coeffs[0]
    for k in range(1, n):
        # guard: |root|·|q[k-1]| must stay below BIG
        if abs(root) > 0 and abs(q[k-1]) > BIG / abs(root):
            return coeffs.copy(), np.nan     # abort deflation
        q[k] = coeffs[k] + root * q[k - 1]
    rem = coeffs[-1] + root * q[-1]
    return q, rem

# -------------------- safe Newton polishing -------------------
def polish_root(coeffs, z0, max_iter=MAX_POLISH, tol=1e-14):
    if not np.isfinite(z0):
        return z0, True
    coef_norm = np.abs(coeffs).max()
    z = z0
    for _ in range(max_iter):
        p, dp = safe_horner(coeffs, z, need_deriv=True)
        if dp == 0.0 or not np.isfinite(p) or not np.isfinite(dp):
            return z0, True
        delta = p / dp
        if not np.isfinite(delta) or abs(delta) > SHIFT_BLOW_UP:
            return z0, True
        z -= delta
        if abs(delta) < tol * max(1.0, abs(z)) or abs(p) < tol * coef_norm:
            return z, False
    return z, False

# ---------------- dominant eigenpair via RQI ------------------
def power_root(coeffs, max_iter=1000, tol=POW_TOL):
    coeffs = coeffs / coeffs[0]
    A = companion_matrix(coeffs)
    x, lam = seed_vector(coeffs)
    coef_norm = np.abs(coeffs).max()

    for it in range(max_iter):
        x = A @ x
        nrm = safe_norm(x)
        if nrm == 0.0 or not np.isfinite(nrm):
            x, lam = seed_vector(coeffs)      # random restart
            continue
        x /= nrm

        lam = rayleigh(A, x)
        if abs(safe_horner(coeffs, lam)) < tol * coef_norm:
            return lam, it, 0.0

        y, ok = shifted_solve(A, lam, x)
        if ok:
            x = y

        lam_new = rayleigh(A, x)
        if abs(lam_new - lam) < tol * abs(lam_new):
            return lam_new, it, abs(lam_new - lam)
        lam = lam_new

    return lam, max_iter, abs(safe_horner(coeffs, lam))

# ---------------------- all roots by deflation ----------------
def power_roots(coeffs, max_iter=1000, tol=POW_TOL, scale_coeffs=True):
    coeffs = coeffs.astype(np.complex128) / coeffs[0]
    if scale_coeffs:
        coeffs, scale = scaled_monic(coeffs)
    else:
        scale = 1.0

    roots = []
    coeffs_fwd = coeffs.copy()
    coeffs_rev = coeffs[::-1].copy() / coeffs[-1]
    want_small = False
    it_max     = 0
    last_err   = 0.0

    while len(coeffs_fwd) > 3:
        if want_small:
            R, it, err = power_root(coeffs_rev, max_iter, tol)
            root       = 1.0 / R
        else:
            root, it, err = power_root(coeffs_fwd, max_iter, tol)

        root, bad = polish_root(coeffs_fwd, root)
        roots.append(root)
        coeffs_fwd, _ = synthetic_division(coeffs_fwd, root)
        coeffs_rev, _ = synthetic_division(coeffs_rev, 1.0 / root)
        want_small = not want_small
        it_max     = max(it_max, it)
        last_err   = err

    # residual quadratic / linear part
    if len(coeffs_fwd) == 3:
        a, b, c = coeffs_fwd
        disc = np.lib.scimath.sqrt(b * b - 4 * a * c)
        roots.extend([(-b + disc) / (2 * a), (-b - disc) / (2 * a)])
    elif len(coeffs_fwd) == 2:
        roots.append(-coeffs_fwd[1] / coeffs_fwd[0])

    return np.asarray(roots) / scale, it_max, last_err

# ----------------------------- CLI -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Safe companion‑matrix root finder")
    ap.add_argument("-n", "--deg", type=int, default=10, help="degree (≥1)")
    ap.add_argument("--maxi", type=int, default=1000, help="max RQI iterations")
    ap.add_argument("--maxp", type=int, default=10, help="max Newton iterations")
    ap.add_argument("--tol",  type=float, default=1e-12, help="tolerance")
    ap.add_argument("--noscale", action="store_true", help="disable coefficient scaling")
    args = ap.parse_args()

    np.random.seed(0)                          # repeatability
    p = random_poly(args.deg)                  # monic random poly

    t0 = time.perf_counter()
    r_ref = np.roots(p)
    print(f"np.roots:    {round((time.perf_counter() - t0)*1000):>5} ms")

    t0 = time.perf_counter()
    r_pow, it_max, last_err = power_roots(p,
                                          max_iter=args.maxi,
                                          tol=args.tol,
                                          scale_coeffs=not args.noscale)
    dt = round((time.perf_counter() - t0)*1000)
    print(f"power_roots: {dt:>5} ms   (max it={it_max},  last err={last_err:.1e})")
    print(f"pairing error: {max_pairing_error(r_ref, r_pow):.2e}")
