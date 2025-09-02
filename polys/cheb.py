import argparse
import timeit
import numpy as np
import pandas as pd
from numba.typed import List
from numba import njit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  

def load(): 
    exec(open("cheb.py").read(), globals())
    print("loaded cheb.py")

#
# poly operations
#

@njit
def _poly_add(p, q):
    m, n = p.shape[0], q.shape[0]
    r = np.zeros(max(m, n), dtype=p.dtype)
    r[:m] += p
    r[:n] += q
    return r

@njit
def _poly_sub(p, q):
    m, n = p.shape[0], q.shape[0]
    r = np.zeros(max(m, n), dtype=p.dtype)
    r[:m] += p
    r[:n] -= q
    return r

@njit
def _poly_scale(p, s):
    r = p.copy()
    r *= s
    return r

@njit
def _poly_mul_linear(p, alpha, beta):
    """
    Multiply p(y) by (beta + alpha*y). Ascending coeffs.
    """
    m = p.shape[0]
    r = np.zeros(m+1, dtype=p.dtype)
    r[0] = beta * p[0]
    for j in range(1, m):
        r[j] = alpha * p[j-1] + beta * p[j]
    r[m] = alpha * p[m-1]
    return r

@njit
def _x_times_cheb(c):
    """Multiply Chebyshev series c(y) by y: returns d(y)=y*c(y). c,d len n+1."""
    n = c.shape[0] - 1
    d = np.zeros_like(c)
    if n >= 1:
        d[0] = 0.5 * c[1]
        d[1] = c[0] + 0.5 * (c[2] if n >= 2 else 0.0)
        for j in range(2, n):
            d[j] = 0.5 * (c[j-1] + c[j+1])
        d[n] = 0.5 * c[n-1]
    else:  # n == 0
        d[1-1] = 0.0  # no-op; scalar series times y is 0 in degree-0 truncation
    return d

@njit
def poly_mul(a, b):
    """Ascending power coeffs multiply: returns conv(a,b)."""
    m = a.shape[0]; n = b.shape[0]
    out = np.zeros(m+n-1, dtype=np.complex128)
    for i in range(m):
        ai = a[i]
        for j in range(n):
            out[i+j] += ai * b[j]
    return out

#
# monomial basis -> cheb basis
#
@njit
def power_to_cheb(p):
    """
    Convert power basis p(y) = sum_{m=0}^n p[m]*y^m (ascending) to Chebyshev coeffs a_k.
    Returns a of length n+1 such that p(y) = sum a[k] T_k(y).
    """
    n = p.shape[0] - 1
    a = np.zeros(n+1, dtype=p.dtype)
    # X = Chebyshev series for y^m; start with m=0 => 1 = T0
    X = np.zeros(n+1, dtype=p.dtype)
    X[0] = 1.0 + 0j
    a += p[0] * X
    for m in range(1, n+1):
        X = _x_times_cheb(X)
        a += p[m] * X
    return a

#
# cheb basis -> monomial basis
#
@njit
def cheb_to_power(a):
    """
    Convert Chebyshev coeffs a(y) to power basis (ascending). Works by building y^m via
    the inverse of _x_times_cheb through a recurrence (stable for modest n).
    """
    n = a.shape[0] - 1
    # Build T_k(y) in power basis progressively and accumulate a_k * T_k(y)
    # T0 = 1 ; T1 = y ; T_{k+1} = 2y T_k - T_{k-1}
    P_prev = np.zeros(1, dtype=a.dtype)  # T_{-1} (unused placeholder)
    P0 = np.zeros(1, dtype=a.dtype)      # T0
    P0[0] = 1.0 + 0j
    poly = a[0] * P0.copy()
    if n >= 1:
        P1 = np.zeros(2, dtype=a.dtype)  # T1 = y
        P1[1] = 1.0 + 0j
        # poly += a1*T1
        poly = _poly_add(poly, _poly_scale(P1, a[1]))
        for k in range(1, n):
            # T_{k+1} = 2y T_k - T_{k-1}
            Pkm1 = P0
            Pk   = P1
            Pk_y = _poly_mul_linear(Pk, alpha=1.0, beta=0.0)  # multiply by y
            Pkp1 = _poly_sub(_poly_scale(Pk_y, 2.0), Pkm1)
            poly = _poly_add(poly, _poly_scale(Pkp1, a[k+1]))
            P0, P1 = Pk, Pkp1
    return poly

#
# horner, but for cheb polys, y is supposed to be [0,1]
#
@njit
def cheb_eval_y(a, y):
    """
    Clenshaw for Chebyshev series sum a[k] T_k(y) at scalar y (complex ok).
    """
    n = a.shape[0] - 1
    b1 = 0.0 + 0j
    b2 = 0.0 + 0j
    for k in range(n, -1, -1):
        b0 = 2.0 * y * b1 - b2 + a[k]
        b2 = b1
        b1 = b0
    return b0 - y * b2

#
# horner, but for cheb polys
# [A,B] is domain of x
# map [A,B] -> [0,1] then evaluate
#
@njit
def cheb_eval_on_domain(a, x, A, B):
    """
    Evaluate Chebyshev series on domain [A,B]. a are coeffs wrt T_k(y), y = affine(x).
    """
    y = (2.0 * x - (A + B)) / (B - A)
    return cheb_eval_y(a, y)

#
# np.polyder equivalent
#
@njit
def cheb_derivative_coeffs_y(a):
    """
    Derivative wrt y: if f(y)=sum a_k T_k(y), returns coeffs d such that f'(y)=sum d_k T_k(y).
    """
    n = a.shape[0] - 1
    if n == 0:
        return np.zeros_like(a)
    d = np.zeros_like(a)
    d[n-1] = 2.0 * n * a[n]
    for k in range(n-2, -1, -1):
        # d[k] = d[k+2] + 2*(k+1)*a[k+1]
        d[k] = (d[k+2] if (k+2) <= (n-1) else 0.0) + 2.0 * (k+1) * a[k+1]
    d[0] *= 0.5
    return d

#
# evaluate both p and p' at x
#
@njit
def cheb_eval_deriv_on_domain_pre(a, ap_y, x, A, B):
    """
    Evaluate f(x) and f'(x) for Chebyshev coeffs a on domain [A,B], using precomputed ap_y.
    """
    y = (2.0*x - (A + B)) / (B - A)
    fy = cheb_eval_y(a, y)
    df_dy = cheb_eval_y(ap_y, y)
    df_dx = (2.0 / (B - A)) * df_dy
    return fy, df_dx

@njit(fastmath=True)
def cheb_eval_deriv_batch(a, ap_y, z, A, B, out_p, out_dp):
    """
    Compute p(z_j) and p'(z_j) for all j at once, using two Clenshaw recurrences.
    a: Chebyshev coeffs on [A,B], ap_y: derivative coeffs wrt y.
    out_p, out_dp: preallocated complex128 arrays len m.
    """
    m = z.shape[0]
    n = a.shape[0] - 1
    # y mapping
    y = np.empty(m, dtype=np.complex128)
    s = 2.0 / (B - A)
    c = (A + B) / (B - A)
    for j in range(m):
        y[j] = s * z[j] - c

    # Clenshaw for p
    b1 = np.zeros(m, dtype=np.complex128)
    b2 = np.zeros(m, dtype=np.complex128)
    # Clenshaw for dp/dy
    d1 = np.zeros(m, dtype=np.complex128)
    d2 = np.zeros(m, dtype=np.complex128)

    for k in range(n, -1, -1):
        ak = a[k]
        apk = ap_y[k] if k <= n-1 else 0.0j

        for j in range(m):
            yj = y[j]
            b0 = 2.0*yj*b1[j] - b2[j] + ak
            d0 = 2.0*yj*d1[j] - d2[j] + apk
            b2[j] = b1[j]; b1[j] = b0
            d2[j] = d1[j]; d1[j] = d0

    for j in range(m):
        out_p[j]  = b1[j] - y[j]*b2[j]
        df_dy     = d1[j] - y[j]*d2[j]
        out_dp[j] = (2.0/(B - A)) * df_dy   # chain rule


#
# np.poly equivalent
#
@njit
def poly_from_roots(roots):
    """
    Build ascending power coeffs for p(x)=prod (x - r_i). Complex-safe, no np.convolve.
    """
    m = len(roots)
    c = np.zeros(m+1, dtype=np.complex128)
    c[0] = 1.0 + 0j
    deg = 0
    for r in roots:
        # multiply by (x - r)
        d = np.zeros(deg+2, dtype=np.complex128)
        d[0] = -r * c[0]
        for j in range(1, deg+1):
            d[j] = c[j-1] - r * c[j]
        d[deg+1] = c[deg]
        c = d
        deg += 1
    return c

@njit
def poly_from_roots_tree(roots):
    """
    Balanced product tree to build p(y)=prod (y - r_i), ascending coeffs.
    Much more stable than sequential multiply.
    """
    L = List()
    for r in roots:
        p = np.empty(2, dtype=np.complex128)
        p[0] = -r
        p[1] = 1.0 + 0.0j
        L.append(p)
    if len(L) == 0:
        return np.array([1.0+0.0j], dtype=np.complex128)
    while len(L) > 1:
        nxt = List()
        k = 0
        while k + 1 < len(L):
            nxt.append(poly_mul(L[k], L[k+1]))
            k += 2
        if k < len(L):
            nxt.append(L[k])
        L = nxt
    return L[0]


@njit
def _poly_mul_linear(p, alpha, beta):
    """
    Multiply p(y) by (beta + alpha*y). Ascending coeffs.
    """
    m = p.shape[0]
    r = np.zeros(m+1, dtype=p.dtype)
    r[0] = beta * p[0]
    for j in range(1, m):
        r[j] = alpha * p[j-1] + beta * p[j]
    r[m] = alpha * p[m-1]
    return r

@njit
def compose_linear(p, alpha, beta):
    """
    q(y) = p(alpha*y + beta), where p is ascending power coeffs in x.
    Correct Horner: start from top coeff, then n multiplies.
    """
    n = p.shape[0] - 1
    q = np.array([p[n]], dtype=p.dtype)  # start with highest coeff
    for i in range(n-1, -1, -1):
        q = _poly_mul_linear(q, alpha, beta)  # q <- (beta + alpha*y)*q
        q[0] += p[i]
    return q

@njit
def cheb_from_power_on_domain(p_x, A, B):
    """
    Given power coeffs p_x (ascending in x), return Chebyshev coeffs a for domain [A,B].
    """
    alpha = 0.5 * (B - A)
    beta  = 0.5 * (A + B)
    q_y = compose_linear(p_x, alpha, beta)  # q(y) = p(alpha*y + beta)
    a = power_to_cheb(q_y)
    return a

@njit
def cheb_on_domain_to_power(a, A, B):
    # a: Chebyshev coeffs on [A,B] in y; return power coeffs in x (ascending)
    qy = cheb_to_power(a)                          # power in y (ascending)
    alpha = 2.0 / (B - A)
    beta  = -(A + B) / (B - A)                     # y = alpha*x + beta
    px = compose_linear(qy, alpha, beta)           # power in x (ascending)
    return px

#
# residual
#
@njit
def max_residual(a, A, B, roots):
    mx = 0.0
    for z in roots:
        y = (2.0*z - (A + B)) / (B - A)
        v = cheb_eval_y(a, y)
        absv = abs(v)
        if absv > mx:
            mx = absv
    return mx


#
# root pattern coefficients
#

@njit
def wilkinson_cheb(n, pad=0.5):
    """
    Chebyshev coeffs (arrays only) for Wilkinson's poly with roots 1..n on [A,B]=[1-pad, n+pad].
    """
    roots = np.arange(1, n+1, dtype=np.float64)
    p_x = poly_from_roots(roots)  # ascending power coeffs in x
    A, B = 1.0 - pad, n + pad
    a = cheb_from_power_on_domain(p_x, A, B)
    return a, A, B

@njit
def chessboard_roots(n: int) -> np.ndarray:
    """n×n grid centered at 0, flattened, dtype complex128."""
    o = 0.5 * (n - 1)
    out = np.empty(n*n, dtype=np.complex128)
    idx = 0
    for i in range(n):
        xi = float(i) - o
        for j in range(n):
            yj = float(j) - o
            out[idx] = xi + 1j*yj
            idx += 1
    return out

@njit
def chessboard_domain(n: int, pad_frac: float = 0.25):
    """Symmetric [A,B] that encloses the complex grid safely."""
    m = 0.5 * (n - 1)
    R = np.sqrt(2.0) * m          # max |x+iy|
    M = (1.0 + pad_frac) * R
    return -M, M

@njit
def chessboard_cheb(n: int, pad_frac: float = 0.25):
    """
    Build Chebyshev coeffs for the chessboard roots.
    Returns (a, A, B, roots) where:
      a: complex128 (n^2+1,) Chebyshev coeffs on [A,B]
      A,B: float64 domain
      roots: complex128 (n^2,) roots vector
    """
    roots = chessboard_roots(n)
    A, B  = chessboard_domain(n, pad_frac)
    p_x   = poly_from_roots(roots)  
    a     = cheb_from_power_on_domain(p_x, A, B)
    return a, A, B

@njit
def chessboard_cheb_scaled(n: int, pad_frac: float = 0.25):
    # 1) chessboard roots in x
    o = 0.5 * (n - 1)
    roots = np.empty(n*n, dtype=np.complex128); k = 0
    for i in range(n):
        xi = float(i) - o
        for j in range(n):
            yj = float(j) - o
            roots[k] = xi + 1j*yj
            k += 1

    # 2) domain [-M, M] and scaling
    M = (1.0 + pad_frac) * np.sqrt(2.0) * o
    A, B = -M, M

    # 3) build power polynomial in y from scaled roots
    yroots = roots / M                          # |yroots| ≤ 1/(1+pad)
    p_y = poly_from_roots(yroots)               # ascending in y

    # 4) convert power-in-y -> Chebyshev-in-y (on [-1,1])
    a = power_to_cheb(p_y)

    # optional: normalize to keep magnitudes friendly (scale-invariant for Aberth)
    s = 0.0
    for i in range(a.shape[0]):
        ai = abs(a[i])
        if ai > s: s = ai
    if s > 0:
        a = a / s

    return a, A, B

@njit
def chessboard_cheb_scaled_tree(n: int, pad_frac: float = 0.25):
    # chessboard roots in x
    o = 0.5*(n - 1)
    roots = np.empty(n*n, dtype=np.complex128); idx = 0
    for i in range(n):
        xi = float(i) - o
        for j in range(n):
            yj = float(j) - o
            roots[idx] = xi + 1j*yj; idx += 1

    # domain [-M, M], so y = x/M
    M = (1.0 + pad_frac) * np.sqrt(2.0) * o   # pad ~ 25%
    A, B = -M, M

    # build polynomial in y from scaled roots
    yroots = roots / M
    p_y = poly_from_roots_tree(yroots)        # ascending in y

    # convert power-in-y -> Chebyshev-in-y (your routine)
    a = power_to_cheb(p_y)

    # normalize (scale-invariant for Aberth)
    smax = 0.0
    for k in range(a.shape[0]):
        ak = abs(a[k])
        if ak > smax: smax = ak
    if smax > 0.0:
        a = a / smax

    return a, A, B

#
# initial guess
#

@njit
def cheb_nodes_initial_guess(n, A, B, jitter=1e-12):
    """
    Chebyshev nodes on [A,B] with tiny imaginary jitter to break symmetry.
    Returns complex128 array of length n.
    """
    k = np.arange(1, n+1, dtype=np.float64)
    nodes = 0.5*(A+B) + 0.5*(B-A)*np.cos((2.0*k - 1.0)*np.pi/(2.0*n))
    z0 = nodes.astype(np.complex128)
    z0 += 1j*jitter*(1.0 + 0.1*k)
    return z0

@njit
def cauchy_circle_from_cheb(a, A, B):
    """
    Return (c, R) where all roots lie in |z - c| <= R.
    Uses simple Cauchy bound from power coeffs in x.
    """
    px = cheb_on_domain_to_power(a, A, B)   # ascending: [a0,...,an]
    n  = px.shape[0] - 1
    an = px[n]
    c  = 0.0 + 0.0j
    if n >= 1 and an != 0:
        c = -px[n-1] / an                     # mean root
    # Cauchy radius: 1 + max_{k<n} |a_k/a_n|
    maxratio = 0.0
    if an != 0:
        for k in range(n):
            rk = abs(px[k] / an)
            if rk > maxratio:
                maxratio = rk
    R = 1.0 + maxratio
    return c, R

@njit
def circle_initial_guess_from_cheb(a, A, B):
    n = a.shape[0] - 1
    c, R = cauchy_circle_from_cheb(a, A, B)
    z0 = np.empty(n, dtype=np.complex128)
    for j in range(n):
        theta = 2.0*np.pi * (j + 0.5) / n
        z0[j] = c + R * np.exp(1j * theta)
    return z0

#
# aberth chebyshev solver
#

@njit(fastmath=True)
def aberth_cheb(a, A, B, z0, maxiter=200, tol=1e-5,
                      step_cap_factor=0.25, eps=1e-30):
    """
    Pure residual stopping: stop when max_j |p(z_j)| < tol.

    a : (n+1,) Chebyshev coeffs on [A,B]
    A,B : domain endpoints
    z0 : (n,) initial guesses (complex128)
    tol : absolute residual tolerance (NOT step-based anymore)
    """
    n = a.shape[0] - 1
    if n <= 0 or z0 is None or z0.shape[0] != n:
        return np.empty(0, dtype=np.complex128), np.int64(0), np.float64(0)

    z = z0.copy()
    ap_y = cheb_derivative_coeffs_y(a)
    W = B - A
    cap = step_cap_factor * W

    # work arrays
    pz  = np.empty(n, dtype=np.complex128)
    dpz = np.empty(n, dtype=np.complex128)
    S   = np.empty(n, dtype=np.complex128)
    dz  = np.empty(n, dtype=np.complex128)

    for iters in range(maxiter):

        cheb_eval_deriv_batch(a, ap_y, z, A, B, pz, dpz)

        # --- PURE RESIDUAL TEST (pre-step) ---
        res_max = 0.0
        for j in range(n):
            m = abs(pz[j])
            if m > res_max:
                res_max = m
        if res_max < tol:
            break

        # tiny-derivative guard
        for j in range(n):
            if abs(dpz[j]) < eps * max(1.0, abs(pz[j])):
                dpz[j] = dpz[j] + eps

        # Newton ratios (reuse pz buffer for eta = p/dp)
        for j in range(n):
            pz[j] = pz[j] / dpz[j]

        # harmonic sums S_j = sum_{k≠j} 1/(z_j - z_k)
        for j in range(n):
            zj = z[j]
            ssum = 0.0 + 0.0j
            for k in range(n):
                if k != j: 
                    ssum += 1.0 / (zj - z[k])
            S[j] = ssum

        # Aberth–Ehrlich step with cap
        for j in range(n):
            denom = 1.0 - pz[j] * S[j]
            step = pz[j] / denom
            mj = abs(step)
            if mj > cap:
                step *= (cap / mj)
            dz[j] = step

        # update
        for j in range(n):
            z[j] -= dz[j]

    return z, iters, res_max

#
# pertubators
#

@njit
def perturb_0(cf,s,t):
    perturbed_cf = cf.copy()
    return perturbed_cf

@njit
def perturb_1(cf,s,t):
    PI2 = 2 * np.pi
    perturbed_cf = cf.copy()
    perturbed_cf[-2]  *= np.exp(1j * PI2 * t)
    perturbed_cf[-3]  *= np.exp(1j * PI2 * s * t)
    perturbed_cf[-6]  *= np.exp(1j * PI2 * (s - t))
    perturbed_cf[-10]  *= np.exp(1j * PI2 * (s + t))
    perturbed_cf[0] *= s * np.exp(1j * PI2 * t)
    return perturbed_cf

@njit
def perturb_1_np(cf,s,t):
    PI2 = 2 * np.pi
    perturbed_cf = cf.copy()
    perturbed_cf[1]  *= np.exp(1j * PI2 * t)
    perturbed_cf[2]  *= np.exp(1j * PI2 * s * t)
    perturbed_cf[5]  *= np.exp(1j * PI2 * (s - t))
    perturbed_cf[9]  *= np.exp(1j * PI2 * (s + t))
    perturbed_cf[-1] *= s * np.exp(1j * PI2 * t)
    return perturbed_cf

@njit
def perturb_5(cf,s,t):
    PI2 = 2 * np.pi
    perturbed_cf = cf.copy()
    perturbed_cf[-2] *= np.exp(1j * PI2 * (s+t))
    perturbed_cf[-5] *= np.exp(1j * PI2 * (s-t))
    perturbed_cf[-7] *= np.exp(1j * PI2 * (s*s - t*t))
    perturbed_cf[-11] *= np.exp(1j * PI2 * (s*s - t*t))
    perturbed_cf[0] *= np.exp(1j * PI2 * t * s)
    return perturbed_cf

@njit
def perturb_11(cf,s,t):
    e1 = 2e-5*(2.0*s - 1.0j)
    e2 = 3e-5*(2.0*t - 1.0j)
    y   = _x_times_cheb(cf)       # adds odd degrees
    y2  = _x_times_cheb(y)        # richer coupling
    return cf + e1*y*1j + e2*y2*t

#
# aberth-cheb tile scanner
#

@njit
def normalize_cheb(a):
    m = 0.0
    for i in range(a.shape[0]):
        ai = abs(a[i])
        if ai > m: m = ai
    if m > 0.0: return a / m
    return a


@njit
def scan(cf, A, B, s_start, s_end, t_start, t_end, n_points, perturb, guess0):
    degree = cf.shape[0] - 1
    steps  = int(np.sqrt(n_points))   # assume perfect square
    s_step = (s_end - s_start) / steps
    t_step = (t_end - t_start) / steps
    out = np.empty((n_points * degree, 6), dtype=np.float64)
    guess = guess0
    prev_guess = guess.copy()
    idx = 0
    i = 0
    j = 0
    dj = 1
    tot_niter = 0
    for k in range(n_points):
        s = s_start + i * s_step
        t = t_start + j * t_step
        local_cf = normalize_cheb(perturb(cf, s, t))                     # must be @njit
        roots, niter, res_max = aberth_cheb(local_cf, A, B, guess)
        tot_niter += niter
        for r in range(degree):
            z = roots[r]
            out[idx, 0] = z.real
            out[idx, 1] = z.imag
            out[idx, 2] = s
            out[idx, 3] = t
            out[idx, 4] = float(niter)
            out[idx, 5] = res_max
            idx += 1
        guess = roots + (roots - prev_guess)
        prev_guess = roots.copy()
        j += dj
        if j == steps or j < 0:
            i += 1
            dj = -dj
            j += dj
            guess = roots.copy()
            prev_guess = guess.copy()
    return out

#
# np.roots tile scanner
#
def tile_scan_np(cf, s_start, s_end, t_start, t_end, n_points, perturb):
    """
    steps x steps grid (steps = sqrt(n_points)).
    Returns (n_points*degree, 4): [Re, Im, s, t]
    """
    degree = len(cf) - 1
    steps  = int(np.sqrt(n_points))
    if steps * steps != n_points: raise ValueError("n_points must be a perfect square")
    s_step = (s_end - s_start) / steps
    t_step = (t_end - t_start) / steps
    out = np.empty((n_points * degree, 4), dtype=np.float64)
    idx = 0
    for i in range(steps):
        s = s_start + i * s_step
        for j in range(steps):
            t = t_start + j * t_step
            local_cf = perturb(cf, s, t) 
            roots = np.roots(local_cf)
            for z in roots:
                out[idx, 0] = z.real
                out[idx, 1] = z.imag
                out[idx, 2] = s
                out[idx, 3] = t
                idx += 1
    return out

#
#
# 
def show_roots(roots):
    roots_sorted = sorted(roots, key=lambda z: z.real,reverse=True)
    df = pd.DataFrame({
        "Real Part": [round(r.real, 3) for r in roots_sorted],
        "Imag Part": [round(r.imag, 15) for r in roots_sorted]
    })
    print(df.to_string(index=False))


def plot_manifold(mat, title, path):
    re, im = mat[:, 0], mat[:, 1]
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.scatter(re, im, s=0.5, marker=".", linewidths=0, alpha=0.6)
    xmin, xmax = re.min(), re.max()
    ymin, ymax = im.min(), im.max()
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    span = max(xmax - xmin, ymax - ymin)
    pad = 0.05 * span
    half = span / 2.0 + pad
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect('equal', adjustable='box')
    try:
        ax.set_box_aspect(1)  # mpl >= 3.3; keeps axes box square
    except Exception:
        pass
    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    fig.savefig(path, dpi=500)
    plt.close(fig)
   
import numpy as np

# --- 1) collapse per-root rows into per-tile grids ---------------------------

def tile_scan_to_grids(res, s_col=2, t_col=3, niter_col=4, err_col=5, atol=1e-12):
    """
    Convert tile_scan output (rows = roots) into per-tile grids.
    Returns:
      s_vals : (S,) unique s coordinates (sorted)
      t_vals : (T,) unique t coordinates (sorted)
      niter_grid : (T,S) float, iterations per tile
      err_grid   : (T,S) float, residual per tile (NaN if absent)
    """
    res = np.asarray(res)
    nrows, ncols = res.shape
    if ncols < 5:
        raise ValueError("Expected at least 5 columns: [Re, Im, s, t, niter, (err)].")

    # infer degree (multiplicity) from the first tile's (s,t)
    s0, t0 = res[0, s_col], res[0, t_col]
    mask0 = (np.abs(res[:, s_col] - s0) <= atol) & (np.abs(res[:, t_col] - t0) <= atol)
    degree = int(np.count_nonzero(mask0))
    if degree <= 0 or nrows % degree != 0:
        raise ValueError(f"Could not infer degree from first tile; degree={degree}, nrows={nrows}.")

    # take the first row of each tile (because per-tile metrics repeat across roots)
    tiles = res[::degree, :]  # shape: (n_points, ncols)
    s_vals = np.unique(tiles[:, s_col])
    t_vals = np.unique(tiles[:, t_col])
    S, T = s_vals.size, t_vals.size

    # compute uniform steps safely (assume regular grid)
    ds = s_vals[1] - s_vals[0] if S > 1 else 1.0
    dt = t_vals[1] - t_vals[0] if T > 1 else 1.0
    s0, t0 = s_vals[0], t_vals[0]

    niter_grid = np.full((T, S), np.nan, dtype=float)
    err_grid   = np.full((T, S), np.nan, dtype=float) if (ncols > err_col) else None

    # map each tile row to grid indices
    for row in tiles:
        s, t = row[s_col], row[t_col]
        si = 0 if S == 1 else int(round((s - s0) / ds))
        ti = 0 if T == 1 else int(round((t - t0) / dt))
        if 0 <= si < S and 0 <= ti < T:
            niter_grid[ti, si] = row[niter_col]
            if err_grid is not None:
                err_grid[ti, si] = row[err_col]

    return s_vals, t_vals, niter_grid, err_grid

# --- 2) plot & save a heatmap (non-blocking, PNG) ----------------------------

def save_heatmap_grid(s_vals, t_vals, grid, path, title, cbar_label=None, log10=False):
    """
    Save a heatmap for a (T,S) grid over coordinates s_vals (S) and t_vals (T).
    Non-blocking; writes a PNG to 'path'.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)  # non-interactive backend
    import matplotlib.pyplot as plt

    g = np.array(grid, dtype=float)
    if log10:
        # avoid log of zero/negative
        g = np.log10(np.maximum(g, 1e-300))
        if cbar_label is None:
            cbar_label = "log10(value)"

    if cbar_label is None:
        cbar_label = "value"

    # extent makes axes reflect the (s,t) range nicely
    extent = [s_vals.min(), s_vals.max(), t_vals.min(), t_vals.max()]

    fig, ax = plt.subplots()
    im = ax.imshow(g, origin="lower", aspect="equal", extent=extent, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    ax.set_xlabel("s")
    ax.set_ylabel("t")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# --- Convenience wrappers -----------------------------------------------------

def save_niter_map(res, out_png):
    s_vals, t_vals, niter_grid, _ = tile_scan_to_grids(res)
    save_heatmap_grid(s_vals, t_vals, niter_grid, out_png,
                      title="Iterations per tile", cbar_label="iterations", log10=False)

def save_residual_map(res, out_png, log10=True):
    s_vals, t_vals, _, err_grid = tile_scan_to_grids(res)
    if err_grid is None:
        raise ValueError("No residual column found in results (need 6th column).")
    save_heatmap_grid(s_vals, t_vals, err_grid, out_png,
                      title=("Residual per tile" + (" (log10)" if log10 else "")),
                      cbar_label=("max residual" if not log10 else "log10 max residual"),
                      log10=log10)
    

#
# 
#



#if __name__ == "__main__":
if False:
    # Example: Wilkinson n=20 on a safely scaled domain
    n = 20

    # cheb solver
    cp, A, B = wilkinson_cheb_array(n, pad=0.5)
    guess = cheb_nodes_initial_guess(len(cp)-1, A, B)
    cr, k = aberth_cheb_array(cp, A, B, guess)
    # cb should be ~ [1,2,...,20] (imag parts ~ 1e-14)
    show_roots(cr)

    # np.roots() solver
    pp = np.poly(np.arange(1, n+1, dtype=np.complex128)).astype(np.complex128)
    pr = np.roots(pp)
    show_roots(pr)

    tpp=timeit.timeit("np.roots(pp)", globals=globals(), number=100)
    tcp=timeit.timeit("aberth_cheb_array(cp, A, B, guess)", globals=globals(), number=100)

    vcr, niter = tile_job(cp,A,B,0.0,1.0,0.0,1.0,4096,perturb_1)
    vpr = tile_job_np(pp,0.0,1.0,0.0,1.0,4096,perturb_1_np)

    plot_manifold(vcr, "Aberth (Chebyshev)", "manifold_cheb.png")
    plot_manifold(vpr, "np.roots", "manifold_numpy.png")

    tjcp=timeit.timeit("tile_job(cp,A,B,0.0,1.0,0.0,1.0,4096,perturb_1)", globals=globals(), number=3)
    tjpp=timeit.timeit("tile_job_np(pp,0.0,1.0,0.0,1.0,4096,perturb_1_np)", globals=globals(), number=3)

