
# scan_aberth.py
# Fast polynomial-manifold scanner using the Aberth–Ehrlich method (monomial basis).
# Author: ChatGPT (GPT-5 Thinking)
# Date: 2025-08-29
#
# Overview
# --------
# This module provides a high-performance implementation of the Aberth–Ehrlich root finder
# for *monomial* polynomials along a 2D parameter grid ("manifold"), with warm starts.
# It is designed to mirror the scanning behavior you likely used in cheb.py (Chebyshev basis),
# but here we expect *monomial* coefficients in NumPy convention (highest power first).
#
# Key features
# ------------
# - Numba-accelerated core (njit) for tight loops and O(n^2) Aberth updates.
# - Warm start across the grid in a "snake" order, reusing solutions from neighbors to reduce iterations.
# - Robust Horner evaluation of p(z) and p'(z) for monomial polynomials (highest-first coefficients).
# - Simple, dependency-free API: provide a (ny, nx, deg+1) coefficient grid; get roots, steps, residuals.
#
# API
# ---
# scan_aberth(coeffs_grid, *, tol=1e-12, max_iters=100, snake=True, per_root_tol=False,
#             newton_fallback=False, verbose=False)
#
# Inputs:
#   coeffs_grid : complex128 ndarray of shape (ny, nx, deg+1)
#       Monomial coefficients per grid point, NumPy convention: [a0, a1, ..., a_deg]
#       representing p(z) = a0*z**deg + a1*z**(deg-1) + ... + a_deg.
#
# Parameters:
#   tol              : float, convergence tolerance. If per_root_tol=False, it's a global threshold
#                      on max |delta_i|; otherwise each root requires |delta_i| <= tol*(1+|z_i|).
#   max_iters        : int, maximum Aberth iterations per polynomial.
#   snake            : bool, if True, scan rows in alternating (snake) order for better warm starts.
#   per_root_tol     : bool, if True, check per-root step tolerance; else use max across roots.
#   newton_fallback  : bool, if True, when p'(z_i) ~ 0 applies a damped Newton step as fallback.
#   verbose          : bool, print simple progress (Python-level, minimal overhead).
#
# Returns:
#   roots    : complex128 ndarray of shape (ny, nx, deg)
#   steps    : int32 ndarray of shape (ny, nx)  (number of Aberth iterations used)
#   residual : float64 ndarray of shape (ny, nx) (max |p(z)| over roots at convergence or last iter)
#
# Utility:
#   prepare_coeffs(func, A, B) -> coeffs_grid
#       Helper to build a (ny, nx, deg+1) array from a Python callable func(a,b)->coeffs.
#
# Notes
# -----
# - All grid polynomials are assumed to have the same degree (constant deg across the grid).
# - If you want different scan orders, you can modify _scan_order().
# - Aberth step: z_i <- z_i - w_i / (1 - w_i * S_i), where w_i = p(z_i)/p'(z_i)
#                and S_i = sum_{j != i} 1/(z_i - z_j).
# - Initial guesses: if no warm start available, we use a circle with Cauchy radius bound.
#
# License: MIT
#
from __future__ import annotations

import numpy as np
from numba import njit, prange

# -----------------------
# Utilities (Numba-safe)
# -----------------------

@njit(cache=True, fastmath=True)
def _horner_and_deriv(a: np.ndarray, z: complex) -> (complex, complex):
    """
    Evaluate p(z) and p'(z) for monomial coefficients 'a' in highest-first order.
    a[0]*z^n + a[1]*z^(n-1) + ... + a[n]
    """
    n = a.size - 1
    p = a[0]
    dp = 0.0 + 0.0j
    for k in range(1, n + 1):
        dp = dp * z + p
        p = p * z + a[k]
    return p, dp


@njit(cache=True, fastmath=True)
def _cauchy_radius(a: np.ndarray) -> float:
    """Cauchy bound: R <= 1 + max_{k>=1} |a[k]/a0|, assumes a0 != 0."""
    a0 = a[0]
    if a0 == 0:
        return 1.0  # degenerate; caller should avoid or pre-normalize
    inv = 1.0 / abs(a0)
    m = 0.0
    for k in range(1, a.size):
        v = abs(a[k]) * inv
        if v > m:
            m = v
    return 1.0 + m


@njit(cache=True, fastmath=True)
def _init_circle(n: int, radius: float, phase: float = 0.0) -> np.ndarray:
    """Uniform points on a circle of 'radius' with a phase offset (deterministic)."""
    z0 = np.empty(n, dtype=np.complex128)
    two_pi = 2.0 * np.pi
    for i in range(n):
        ang = two_pi * (i + 0.5) / n + phase  # half-step to avoid symmetry troubles
        z0[i] = radius * (np.cos(ang) + 1j * np.sin(ang))
    return z0


@njit(cache=True, fastmath=True)
def _aberth_once(a: np.ndarray, z: np.ndarray, newton_fallback: bool) -> (np.ndarray, float):
    """
    One Aberth iteration (Jacobi-style): returns new z and max step size.
    """
    n = z.size
    p = np.empty(n, dtype=np.complex128)
    dp = np.empty(n, dtype=np.complex128)

    # Evaluate p and p' at all z
    for i in range(n):
        pi, dpi = _horner_and_deriv(a, z[i])
        p[i] = pi
        dp[i] = dpi

    # Compute corrections
    z_new = np.empty_like(z)
    max_step = 0.0
    tiny = 1e-300

    for i in range(n):
        zi = z[i]
        dpi = dp[i]
        if abs(dpi) < tiny:
            # Derivative too small: fallback (optionally Newton with damping)
            wi = p[i] / (dpi + 1e-16)  # safeguard
            if newton_fallback:
                step = wi  # Newton step
                # damping if huge
                mag = abs(step)
                if mag > 1.0:
                    step = step / mag
            else:
                step = wi  # still try
        else:
            wi = p[i] / dpi

            # Compute S_i = sum_{j != i} 1/(z_i - z_j)
            s = 0.0 + 0.0j
            for j in range(n):
                if j != i:
                    dz = zi - z[j]
                    # Guard coincident guesses
                    if dz == 0:
                        dz = dz + (1e-16 + 1e-16j)
                    s += 1.0 / dz
            denom = 1.0 - wi * s
            if denom == 0:
                step = wi  # avoid division by zero; fallback to Newton
            else:
                step = wi / denom

        zi_new = zi - step
        z_new[i] = zi_new

        ds = abs(step)
        if ds > max_step:
            max_step = ds

    return z_new, max_step


@njit(cache=True, fastmath=True)
def _aberth_solve(a: np.ndarray,
                  z0: np.ndarray,
                  tol: float,
                  max_iters: int,
                  per_root_tol: bool,
                  newton_fallback: bool) -> (np.ndarray, int, float):
    """
    Solve one polynomial with coefficients 'a' starting from z0.
    Returns: roots, steps_used, max_abs_p
    """
    z = z0.copy()
    n = z.size
    steps = 0
    max_abs_p = 0.0

    for it in range(max_iters):
        z_new, max_step = _aberth_once(a, z, newton_fallback)

        # Convergence check
        if per_root_tol:
            # Per-root relative step criterion
            all_ok = True
            for i in range(n):
                rel = abs(z_new[i] - z[i]) / (1.0 + abs(z_new[i]))
                if rel > tol:
                    all_ok = False
                    break
            z = z_new
            steps = it + 1
            if all_ok:
                break
        else:
            z = z_new
            steps = it + 1
            if max_step <= tol:
                break

    # Compute residual (max |p(zi)|)
    for i in range(n):
        pi, dpi = _horner_and_deriv(a, z[i])
        ap = abs(pi)
        if ap > max_abs_p:
            max_abs_p = ap

    return z, steps, max_abs_p


@njit(cache=True, fastmath=True)
def _scan_order(ny: int, nx: int, snake: bool) -> np.ndarray:
    """
    Return an array of linear indices (ny*nx) that encodes scan order.
    If snake=True, rows alternate direction to maximize warm-start continuity.
    """
    order = np.empty(ny * nx, dtype=np.int64)
    idx = 0
    for r in range(ny):
        if snake and (r % 2 == 1):
            # right-to-left
            for c in range(nx - 1, -1, -1):
                order[idx] = r * nx + c
                idx += 1
        else:
            # left-to-right
            for c in range(nx):
                order[idx] = r * nx + c
                idx += 1
    return order


@njit(cache=True, fastmath=True)
def _scan_aberth_core(coeffs_grid: np.ndarray,
                      tol: float,
                      max_iters: int,
                      snake: bool,
                      per_root_tol: bool,
                      newton_fallback: bool) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Core scanner over a grid of monomial coefficients.
    coeffs_grid: (ny, nx, deg+1), complex128, highest-first
    Returns (roots, steps, residual):
      roots:    (ny, nx, deg)
      steps:    (ny, nx)
      residual: (ny, nx)
    """
    ny, nx, L = coeffs_grid.shape
    n = L - 1  # degree
    roots = np.empty((ny, nx, n), dtype=np.complex128)
    steps = np.zeros((ny, nx), dtype=np.int32)
    residual = np.zeros((ny, nx), dtype=np.float64)

    order = _scan_order(ny, nx, snake)

    # Initialize first tile
    first_lin = order[0]
    r0 = first_lin // nx
    c0 = first_lin % nx
    a = coeffs_grid[r0, c0]
    R = _cauchy_radius(a)
    z0 = _init_circle(n, R, phase=0.0)
    z, s, res = _aberth_solve(a, z0, tol, max_iters, per_root_tol, newton_fallback)
    roots[r0, c0, :] = z
    steps[r0, c0] = s
    residual[r0, c0] = res

    # Walk the rest with warm starts
    for k in range(1, order.size):
        pct = int(100.0 * k / order.size)
        if k % (order.size//100)==0:
            print("pct:",pct)
        lin = order[k]
        r = lin // nx
        c = lin % nx
        a = coeffs_grid[r, c]

        # Warm start from neighbor along scan direction
        prev_lin = order[k - 1]
        pr = prev_lin // nx
        pc = prev_lin % nx
        z0 = roots[pr, pc, :].copy()

        # Safety tweak: if the new leading coeff is zero, perturb slightly (rare/degenerate)
        if a[0] == 0:
            eps = 1e-12
            a = a.copy()
            a[0] = eps

        z, s, res = _aberth_solve(a, z0, tol, max_iters, per_root_tol, newton_fallback)
        roots[r, c, :] = z
        steps[r, c] = s
        residual[r, c] = res

    return roots, steps, residual


# -----------------------
# Public API (Python)
# -----------------------

def scan_aberth(coeffs_grid: np.ndarray,
                *,
                tol: float = 1e-12,
                max_iters: int = 100,
                snake: bool = True,
                per_root_tol: bool = False,
                newton_fallback: bool = False,
                verbose: bool = False):
    """
    Scan a 2D grid of monomial polynomials using Aberth with warm starts.

    Parameters
    ----------
    coeffs_grid : np.ndarray (ny, nx, deg+1), complex128
        Monomial coefficients in numpy convention (highest-first) for each grid point.
    tol : float
        Convergence tolerance (step-based by default). If per_root_tol=True, it's per-root rel tol.
    max_iters : int
        Maximum Aberth iterations per polynomial.
    snake : bool
        If True, rows are scanned alternately left->right then right->left for better warm starts.
    per_root_tol : bool
        If True, require each root to satisfy |Δz|/(1+|z|) <= tol; otherwise use max step over roots.
    newton_fallback : bool
        If True, when p'(z) is tiny, apply damped Newton as a fallback (helps in rare cases).
    verbose : bool
        Print minimalist progress (ny*nx count).

    Returns
    -------
    roots : np.ndarray (ny, nx, deg) complex128
    steps : np.ndarray (ny, nx) int32
    residual : np.ndarray (ny, nx) float64
    """
    coeffs_grid = np.asarray(coeffs_grid, dtype=np.complex128)
    if coeffs_grid.ndim != 3:
        raise ValueError("coeffs_grid must have shape (ny, nx, deg+1).")
    ny, nx, L = coeffs_grid.shape
    if L < 2:
        raise ValueError("deg+1 must be at least 2.")

    if verbose:
        print(f"[scan_aberth] grid={ny}x{nx}, degree={L-1}, tol={tol}, max_iters={max_iters}, snake={snake}")

    roots, steps, residual = _scan_aberth_core(coeffs_grid, tol, max_iters, snake, per_root_tol, newton_fallback)

    if verbose:
        tot = ny * nx
        print(f"[scan_aberth] done {tot} tiles. median steps={np.median(steps)}, max residual={residual.max():.3e}")
    return roots, steps, residual


def prepare_coeffs(func, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Convenience helper: build coeff grid by calling 'func(a, b) -> coeffs' at each (a,b).
    The callable runs in Python (not jitted); the scan itself is jitted.

    Parameters
    ----------
    func : callable
        Returns a 1D array-like of complex coeffs in highest-first order.
    A, B : same-shape ndarrays
        Parameter grids (e.g., from np.meshgrid). For each (i,j), we call func(A[i,j], B[i,j]).

    Returns
    -------
    coeffs_grid : np.ndarray (ny, nx, deg+1) complex128
    """
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape.")
    ny, nx = A.shape

    # Probe degree once
    sample = np.asarray(func(A.flat[0], B.flat[0]), dtype=np.complex128)
    L = sample.size
    coeffs_grid = np.empty((ny, nx, L), dtype=np.complex128)
    coeffs_grid.ravel()[:L] = sample  # first one

    # Fill the rest
    idx = L
    flat = coeffs_grid.ravel()
    for t in range(1, ny * nx):
        ai = A.flat[t]
        bi = B.flat[t]
        c = np.asarray(func(ai, bi), dtype=np.complex128)
        if c.size != L:
            raise ValueError("All polynomials must have the same degree across the grid.")
        start = t * L
        flat[start:start + L] = c
    return coeffs_grid



# --- Rasterization utilities (PIL) ---
def rasterize_points(re, im, llx, lly, urx, ury, width, height, *, x_is_im=True, flip_y=True, dtype=np.uint8, binary=False, scale="log", max_count=None):
    """
    Rasterize points to a 2D numpy array image.

    Parameters
    ----------
    re, im : 1D real arrays of same length (roots' real/imag parts)
    llx,lly,urx,ury : float
        Bounds of the complex plane region to rasterize (x=minRe, y=minIm unless x_is_im=True).
    width,height : int
        Output resolution in pixels.
    x_is_im : bool
        If True, map x := imag and y := real (common in your plots). Otherwise x:=real, y:=imag.
    flip_y : bool
        If True, put y=ury at the top row (image coordinates). If False, mathematical y-up.
    dtype : numpy dtype
        Output dtype (np.uint8 recommended for PNG 'L').
    binary : bool
        If True, output is 0/255 mask. If False, accumulate counts then map by `scale` to 0..255.
    scale : {'linear','sqrt','log'}
        Mapping from counts to 0..255 if binary=False.
    max_count : int or None
        Optional clamp for counts before scaling. If None, use observed max.

    Returns
    -------
    img : (height, width) array of dtype `dtype` (e.g., uint8)
    """
    re = np.asarray(re)
    im = np.asarray(im)

    # choose x/y from convention
    x = im if x_is_im else re
    y = re if x_is_im else im

    m = np.isfinite(x) & np.isfinite(y)
    if not np.any(m):
        return np.zeros((height, width), dtype=dtype)

    x = x[m]; y = y[m]

    span_x = max(urx - llx, np.finfo(float).eps)
    span_y = max(ury - lly, np.finfo(float).eps)

    # map to pixel indices in [0, width-1] × [0, height-1]
    ix = np.floor((x - llx) / span_x * width).astype(np.int64)
    iy = np.floor((y - lly) / span_y * height).astype(np.int64)

    # clamp
    ix = np.clip(ix, 0, width  - 1)
    iy = np.clip(iy, 0, height - 1)

    # accumulate counts
    counts = np.zeros((height, width), dtype=np.int32)
    np.add.at(counts, (iy, ix), 1)

    if flip_y:
        counts = np.flipud(counts)

    if binary:
        img = (counts > 0).astype(dtype) * np.array(255, dtype=dtype)
        return img

    # intensity mapping
    cmax = int(counts.max()) if max_count is None else int(max_count)
    if cmax <= 0:
        return np.zeros((height, width), dtype=dtype)

    counts = counts.astype(np.float64)

    if scale == "linear":
        norm = counts / cmax
    elif scale == "sqrt":
        norm = np.sqrt(counts / cmax)
    elif scale == "log":
        # log1p for stability; normalize so max->1
        norm = np.log1p(counts) / np.log1p(cmax)
    else:
        norm = np.where(counts>0,1.0,0.0)

    img = (np.clip(norm, 0.0, 1.0) * 255.0).astype(dtype)
    return img
# -----------------------
# Simple CLI (optional)
# -----------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Scan a polynomial manifold with Aberth (monomial basis).")
    ap.add_argument("--ny", type=int, default=8, help="Grid rows")
    ap.add_argument("--nx", type=int, default=8, help="Grid cols")
    ap.add_argument("--deg", type=int, default=16, help="Polynomial degree")
    ap.add_argument("--tol", type=float, default=1e-12, help="Convergence tolerance")
    ap.add_argument("--max-iters", type=int, default=100, help="Maximum iterations per polynomial")
    ap.add_argument("--snake", action="store_true", help="Use snake-order scan")
    ap.add_argument("--no-snake", dest="snake", action="store_false")
    ap.set_defaults(snake=True)
    ap.add_argument("--seed", type=int, default=0, help="Deterministic parameterization seed")
    ap.add_argument("--png", type=str, default=None, help="Path to save a PNG of the root locus via PIL (no matplotlib)")
    ap.add_argument("--width", type=int, default=4096, help="PNG width")
    ap.add_argument("--height", type=int, default=4096, help="PNG height")
    ap.add_argument("--mode", type=str, default="L", help="PIL image mode (default 'L')")
    ap.add_argument("--scale", type=str, default="log", choices=["linear","sqrt","log","thresh"], help="Count-to-intensity mapping")
    ap.add_argument("--binary", action="store_true", help="Binary 0/255 mask instead of counts")
    ap.add_argument("--x-is-im", action="store_true", help="Map x := imag, y := real (on)")
    ap.add_argument("--x-is-re", dest="x_is_im", action="store_false", help="Map x := real, y := imag")
    ap.set_defaults(x_is_im=True)
    ap.add_argument("--flip-y", action="store_true", help="Flip vertical axis to image coords (on)")
    ap.add_argument("--no-flip-y", dest="flip_y", action="store_false")
    ap.set_defaults(flip_y=True)
    ap.add_argument("--llx", type=float, default=None, help="BBox: min x")
    ap.add_argument("--lly", type=float, default=None, help="BBox: min y")
    ap.add_argument("--urx", type=float, default=None, help="BBox: max x")
    ap.add_argument("--ury", type=float, default=None, help="BBox: max y")
    args = ap.parse_args()

    # Example parameterization: p(z) = z^deg + a*z + b  (a,b from grid)
    # You can replace this with your own func(a,b).
    rng = np.random.default_rng(args.seed)
    y = np.linspace(0, 1.0, args.ny)
    x = np.linspace(0.0, 1.0, args.nx)
    B, A = np.meshgrid(x, y)  # (ny, nx)

    def demo_func(a, b):
        c = np.zeros(args.deg + 1, dtype=np.complex128)
        c[0] = 1.0  # monic
        c[1] = a+b
        c[2] = a*1j+b
        c[3] = b*np.exp(1j*2*np.pi*a)
        c[4] = a*np.exp(1j*2*np.pi*b)
        c[-10] = 100*a
        c[-1] = b
        return c

    CG = prepare_coeffs(demo_func, A, B)
    roots, steps, residual = scan_aberth(CG, tol=args.tol, max_iters=args.max_iters, snake=args.snake, verbose=True)
    print("roots shape:", roots.shape)
    print("steps stats: mean min/median/max:", steps.mean(),steps.min(), np.median(steps), steps.max())
    print("residual max:", residual.max())

    if args.png is not None:
        # Rasterize all roots to a high-res PNG using PIL (no matplotlib)
        from PIL import Image

        pts = roots.reshape(-1)  # complex array of all roots

        # Determine bbox: either auto from data with small margin, or user-provided
        if hasattr(args, "llx") and args.llx is not None:
            llx, lly, urx, ury = args.llx, args.lly, args.urx, args.ury
        else:
            re = pts.real
            im = pts.imag
            # pad 5%
            rx = re.max() - re.min() if re.size else 1.0
            ry = im.max() - im.min() if im.size else 1.0
            pad_x = 0.05 * (rx if rx > 0 else 1.0)
            pad_y = 0.05 * (ry if ry > 0 else 1.0)
            llx = re.min() - pad_x
            urx = re.max() + pad_x
            lly = im.min() - pad_y
            ury = im.max() + pad_y

        W = getattr(args, "width", 4096)
        H = getattr(args, "height", 4096)
        scale = getattr(args, "scale", "log")
        binary = getattr(args, "binary", False)
        x_is_im = getattr(args, "x_is_im", True)
        flip_y = getattr(args, "flip_y", True)
        mode = getattr(args, "mode", "L")

        img_arr = rasterize_points(pts.real, pts.imag, llx, lly, urx, ury, W, H,
                                x_is_im=x_is_im, flip_y=flip_y, dtype=np.uint8,
                                binary=binary, scale=scale)

        img = Image.fromarray(img_arr, mode=mode)
        img.save(args.png, optimize=True)
        print(f"Saved root locus PNG (PIL) to {args.png} [{W}x{H}, mode={mode}, scale={scale}, binary={binary}]")
