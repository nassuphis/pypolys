
# scan_aberth.py (tile-only, multiprocessing)
# Polynomial manifold scanner using Aberth–Ehrlich, monomial basis.
# - Parallelized over tiles (each worker computes its own coefficients).
# - Warm starts within each tile with snake order + extrapolated guesses.
# - Optional PNG output via PIL by rasterizing all roots.
# Author: ChatGPT (GPT-5 Thinking) — 2025-08-29

from __future__ import annotations

import numpy as np
from numba import njit
from typing import Tuple, Callable
import multiprocessing as mproc

# =======================
# polys
# =======================

@njit(cache=True, fastmath=True)
def p1_1(a: float, b:float)-> np.ndarray:
    c = np.zeros(21, dtype=np.complex128)
    c[0] = 1.0
    c[1] = a + b
    c[2] = a*1j + b
    c[3] = b*np.exp(1j*2*np.pi*a)
    c[4] = a*np.exp(1j*2*np.pi*b)
    c[-10] = 100*a
    c[-1] = b
    return c


# =====================
# Numba core utilities
# =====================

@njit(cache=True, fastmath=True)
def _horner_and_deriv(a: np.ndarray, z: complex) -> Tuple[complex, complex]:
    """Evaluate p(z) and p'(z) for monomial coefficients 'a' in highest-first order."""
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
        return 1.0
    inv = 1.0 / abs(a0)
    m = 0.0
    for k in range(1, a.size):
        v = abs(a[k]) * inv
        if v > m:
            m = v
    return 1.0 + m

@njit(cache=True, fastmath=True)
def _init_circle(n: int, radius: float, phase: float = 0.0) -> np.ndarray:
    z0 = np.empty(n, dtype=np.complex128)
    two_pi = 2.0 * np.pi
    for i in range(n):
        ang = two_pi * (i + 0.5) / n + phase
        z0[i] = radius * (np.cos(ang) + 1j * np.sin(ang))
    return z0

@njit(cache=True, fastmath=True)
def _aberth_once(a: np.ndarray, z: np.ndarray, newton_fallback: bool) -> Tuple[np.ndarray, float]:
    n = z.size
    p = np.empty(n, dtype=np.complex128)
    dp = np.empty(n, dtype=np.complex128)
    for i in range(n):
        pi, dpi = _horner_and_deriv(a, z[i])
        p[i] = pi
        dp[i] = dpi

    z_new = np.empty_like(z)
    max_step = 0.0
    tiny = 1e-300
    for i in range(n):
        zi = z[i]
        dpi = dp[i]
        if abs(dpi) < tiny:
            wi = p[i] / (dpi + 1e-16)
            if newton_fallback:
                step = wi
                mag = abs(step)
                if mag > 1.0:
                    step = step / mag
            else:
                step = wi
        else:
            wi = p[i] / dpi
            s = 0.0 + 0.0j
            for j in range(n):
                if j != i:
                    dz = zi - z[j]
                    if dz == 0:
                        dz = dz + (1e-16 + 1e-16j)
                    s += 1.0 / dz
            denom = 1.0 - wi * s
            step = wi if denom == 0 else wi / denom
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
                  newton_fallback: bool) -> Tuple[np.ndarray, int, float]:
    z = z0.copy()
    n = z.size
    steps = 0
    max_abs_p = 0.0
    for it in range(max_iters):
        z_new, max_step = _aberth_once(a, z, newton_fallback)
        if per_root_tol:
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
    for i in range(n):
        pi, _ = _horner_and_deriv(a, z[i])
        ap = abs(pi)
        if ap > max_abs_p:
            max_abs_p = ap
    return z, steps, max_abs_p

# =======================
# Tile-based multiprocessing
# =======================

@njit
def scan(s_start, s_end, t_start, t_end, n_points, func):
    cf = func(0.0,0.0)
    degree = cf.shape[0] - 1
    steps  = int(np.sqrt(n_points))   # assume perfect square
    s_step = (s_end - s_start) / steps
    t_step = (t_end - t_start) / steps
    out = np.empty((n_points * degree, 6), dtype=np.float64)
    guess =  np.roots(cf)
    prev_guess = guess.copy()
    idx = 0
    i = 0
    j = 0
    dj = 1
    tot_niter = 0
    for k in range(n_points):
        s = s_start + i * s_step
        t = t_start + j * t_step
        local_cf = func(s, t)  # must be @njit
        roots, niter, err  = _aberth_solve(local_cf, guess,1e-12,80,False,False)
        tot_niter += niter
        for r in range(degree):
            z = roots[r]
            out[idx, 0] = z.real
            out[idx, 1] = z.imag
            out[idx, 2] = s
            out[idx, 3] = t
            out[idx, 4] = float(niter)
            out[idx, 5] = err
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


# Top-level worker (picklable on macOS spawn)
# z, s_count, _ = _aberth_solve(a0, guess, tol, max_iters, per_root_tol, newton_fallback)

def scan_mp(func: Callable[[float, float], np.ndarray], N:int):
    """
    func(a,b)->coeffs must be a top-level picklable function (highest-first complex128 length deg+1).
    Returns
    roots : (ny, nx, deg) complex128, with ny=nx=tiles_per_side*sqrt(points_per_worker)
    """
    ctx = mproc.get_context("spawn")
   
    tiles = int(mproc.cpu_count()**0.5)**2 # square tiles
    tiles_per_side = int(tiles**0.5)
    points_per_worker = int((N // tiles)**0.5)**2
    assert tiles_per_side**2 == tiles, "tiles must be a square number"
    tile_size = 1.0 / tiles_per_side
    points_per_worker = int((N // tiles)**0.5)**2
    args = []
    for id in range(mproc.cpu_count()): 
        tx = id % tiles_per_side
        ty = id // tiles_per_side
        s_start = tx * tile_size
        s_end   = s_start + tile_size
        t_start = ty * tile_size
        t_end   = t_start + tile_size
        print(f"{id} [{tx},{ty}] : ({s_start},{t_start})-({s_end},{t_end}) : {points_per_worker}")
        args.append((id, s_start, s_end, t_start, t_end, points_per_worker,func))

    with ctx.Pool(processes=len(args)) as pool: 
            roots = pool.map(scan, args)


    return roots

# =======================
# Rasterizer (PIL)
# =======================

def rasterize_points(
        re, 
        im, 
        llx, 
        lly, urx, ury, 
        width, height, 
        *,
        x_is_im=True, flip_y=True, dtype=np.uint8,
        binary=False, scale="log", max_count=None
    ):
    re = np.asarray(re); im = np.asarray(im)
    x = im if x_is_im else re
    y = re if x_is_im else im
    m = np.isfinite(x) & np.isfinite(y)
    if not np.any(m):
        return np.zeros((height, width), dtype=dtype)
    x = x[m]; y = y[m]
    span_x = max(urx - llx, np.finfo(float).eps)
    span_y = max(ury - lly, np.finfo(float).eps)
    ix = np.floor((x - llx) / span_x * width).astype(np.int64)
    iy = np.floor((y - lly) / span_y * height).astype(np.int64)
    ix = np.clip(ix, 0, width - 1); iy = np.clip(iy, 0, height - 1)
    counts = np.zeros((height, width), dtype=np.int64)
    np.add.at(counts, (iy, ix), 1)
    if flip_y:
        counts = np.flipud(counts)
    if binary:
        return (counts > 0).astype(dtype) * np.array(255, dtype=dtype)
    cmax = int(counts.max()) if max_count is None else int(max_count)
    if cmax <= 0:
        return np.zeros((height, width), dtype=dtype)
    counts = counts.astype(np.float64)
    if scale == "linear":
        norm = counts / cmax
    elif scale == "sqrt":
        norm = np.sqrt(counts / cmax)
    elif scale == "log":
        norm = np.log1p(counts) / np.log1p(cmax)
    else:  # 'thresh'
        norm = (counts > 0).astype(np.float64)
    return (np.clip(norm, 0.0, 1.0) * 255.0).astype(dtype)


# =======================
# CLI
# =======================

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Tile-only Aberth manifold scan (monomial).")

    # Tile domain
    ap.add_argument("--times", type=int, default=1025, help="Points per worker tile")
    ap.add_argument("--verbose", action="store_true")

    # PNG
    ap.add_argument("--png", type=str, default=None, help="Output PNG path (grayscale)")
    ap.add_argument("--width", type=int, default=4096); ap.add_argument("--height", type=int, default=4096)
    ap.add_argument("--mode", type=str, default="L")
    ap.add_argument("--scale", type=str, default="log", choices=["linear","sqrt","log","thresh"])
    ap.add_argument("--binary", action="store_true")
    ap.add_argument("--x-is-im", action="store_true"); ap.add_argument("--x-is-re", dest="x_is_im", action="store_false"); ap.set_defaults(x_is_im=True)
    ap.add_argument("--flip-y", action="store_true"); ap.add_argument("--no-flip-y", dest="flip_y", action="store_false"); ap.set_defaults(flip_y=True)
    ap.add_argument("--llx", type=float, default=None); ap.add_argument("--lly", type=float, default=None)
    ap.add_argument("--urx", type=float, default=None); ap.add_argument("--ury", type=float, default=None)

    args = ap.parse_args()

    # Example picklable coefficient function: monic degree 'deg' with a few complex terms
    

    roots = scan_mp(p1_1,args.times)

    print("roots shape:", roots.shape)

    if args.png is not None:
        from PIL import Image
        pts = roots.reshape(-1)
        if args.llx is not None:
            llx, lly, urx, ury = args.llx, args.lly, args.urx, args.ury
        else:
            re = pts.real; im = pts.imag
            rx = re.max() - re.min() if re.size else 1.0
            ry = im.max() - im.min() if im.size else 1.0
            pad_x = 0.05 * (rx if rx > 0 else 1.0); pad_y = 0.05 * (ry if ry > 0 else 1.0)
            llx = re.min() - pad_x; urx = re.max() + pad_x
            lly = im.min() - pad_y; ury = im.max() + pad_y

        img_arr = rasterize_points(
            pts.real, 
            pts.imag, 
            llx, lly, urx, ury,
            args.width, args.height,
            x_is_im=args.x_is_im, flip_y=args.flip_y,
            dtype=np.uint8, binary=args.binary, scale=args.scale
        )
        img = Image.fromarray(img_arr, mode=args.mode)
        img.save(args.png, optimize=True)
        print(f"Saved PNG to {args.png} [{args.width}x{args.height} mode={args.mode} scale={args.scale}]")
