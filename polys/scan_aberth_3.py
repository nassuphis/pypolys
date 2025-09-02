# scan_aberth.py (tile-only, multiprocessing)
# Polynomial manifold scanner using Aberth–Ehrlich, monomial basis.
# - Parallelized over tiles (each worker computes its own coefficients).
# - Warm starts within each tile with snake order + extrapolated guesses.
# - Optional PNG output via PIL by rasterizing all roots.
# Author: ChatGPT (GPT-5 Thinking) — 2025-08-29

from __future__ import annotations

import numpy as np
import math
from numba import njit
from typing import Tuple, Callable
import multiprocessing as mproc
from multiprocessing.shared_memory import SharedMemory
import time

# =======================
# parameters
# =======================

use_aberth = False
# poly_giga_19 @ 2000*2000 evals
# 356,000,000 roots
# aberth: 177 sec
# np.roots: 2461 sec
# aberth does 13x speedup over np.roots
# for degree-90 poly
# speedup goes up with degree

# =======================
# shared memory
# =======================

def make_shm(rows,cols,type):
    size = rows * cols * np.dtype(type).itemsize
    shm = SharedMemory( create=True, size = size )
    array = np.ndarray((rows,cols), dtype=type, buffer=shm.buf)
    array[:] = 0
    return (shm,array)

def get_shm(name,rows,cols,type):
    shm = SharedMemory(name=name)
    array = np.ndarray((rows, cols), dtype=type, buffer=shm.buf)
    return(shm,array)

# =======================
# polys (example coeff gen)
# =======================

@njit(cache=True, fastmath=True)
def p1_1(a: float, b: float) -> np.ndarray:
    # returns monomial coeffs (highest-first) for degree=20 (length 21)
    c = np.zeros(21, dtype=np.complex128)
    c[0] = 1.0
    c[1] =(a + b)*np.exp(1j*2*np.pi*a*b)
    c[2] = a*1j + b
    c[3] = b*np.exp(1j*2*np.pi*a)
    c[4] = a*np.exp(1j*2*np.pi*b)
    c[-10] = 100*a
    c[-3] = a * 1j
    c[-2] = b
    c[-1] = 1.0
    return c

@njit(cache=True, fastmath=True)
def poly_giga_5(s:float, t:float):
    t1 = np.exp(1j*2*np.pi*s)
    t2 = np.exp(1j*2*np.pi*t)
    cf = np.zeros(25, dtype=np.complex128)
    cf[0]  = 1.0 + 0j 
    cf[4]  = 4.0 + 0j
    cf[12] = 4.0 + 0j 
    cf[19] = -9 + 0j
    cf[20] = -1.9 + 0j
    cf[24] = 0.2 + 0j
    cf[6] = 100j * t2**3 + 100j * t2**2 - 100j * t2 - 100j
    cf[8] = 100j * t1**3 + 100j * t1**2 + 100j * t2 - 100j
    cf[14] = 100j * t2**3 - 100j * t2**2 + 100j * t2 - 100j
    return np.flip(cf)

@njit(cache=True, fastmath=True)  
def poly_giga_10(s: float, t: float) -> np.ndarray:

    n = 120
    t1 = np.exp(1j*2*np.pi*s)
    t2 = np.exp(1j*2*np.pi*t)

    cf = np.zeros(n, dtype=np.complex128)

    re1, im1 = np.real(t1), np.imag(t1)
    re2, im2 = np.real(t2), np.imag(t2)

    k = np.arange(1, n + 1, dtype=np.float64)          # 1..120
    term1 = (100.0 * (re1 + im2) * (k / 10.0) ** 2) * np.exp(1j * (re2 * k / 20.0))
    term2 = (50.0 * (im1 - re2) * np.sin(k * 0.1 * im2)) * np.exp(-1j * k * 0.05 * re1)
    cf[:] = term1 + term2

    cf[29] += 1j * 1000.0         # cf[30] in R
    cf[59] += -500.0              # cf[60] in R
    cf[89] += 250.0 * np.exp(1j * (t1 * t2))  # cf[90] in R

    return np.flip(cf)

@njit(cache=True, fastmath=True) 
def poly_giga_19(t1: float, t2: float) -> np.ndarray:

    n = 90
    cf = np.zeros(n, dtype=np.complex128)
    cf[0] = t1 - t2
    for k in range(1, n):  # Python 1..89 ↔ R 2..90
        v = np.sin((k + 1) * cf[k - 1]) + np.cos((k + 1) * t1)
        av = np.abs(v)
        if np.isfinite(av) and av > 1e-10:
            cf[k] = 1j * v / av
        else:
            cf[k] = t1 + t2

    return np.flip(cf)

@njit(cache=True, fastmath=True)
def p7f(t1, t2):
    pi2  =  2 * np.pi
    n    =  23
    tt1  =  np.exp(1j * 2 * np.pi * t1)
    ttt1 =  np.exp(1j * 2 * np.pi * tt1)
    v  =  np.linspace(np.real(tt1), np.real(ttt1), n)
    if t2 < 0.1:
        f = 10 * t1 * np.exp(1j * np.sin(11 * pi2 * v))
    elif 0.1 <= t2 < 0.2:
        f =  100 * np.exp(1j * np.sin(17 * pi2 * v))
    elif 0.2 <= t2 < 0.3:
        f =  599 * np.exp(1j * np.cos(83 * pi2 * v))
    elif 0.3 <= t2 < 0.4:
        f =  443 * np.exp(1j * np.sin(179 * pi2 * v))
    elif 0.4 <= t2 < 0.5:
        f =  293 * np.exp(1j * np.sin(127 * pi2 * v))
    elif 0.5 <= t2 < 0.6:
        f =  541 * np.exp(1j * np.sin(103 * pi2 * v))
    elif 0.6 <= t2 < 0.7:
        f =  379 * np.exp(1j * np.sin(283 * pi2 * v))
    elif 0.7 <= t2 < 0.8:
        f =  233 * np.exp(1j * np.sin(3 * pi2 * v))
    elif 0.8 <= t2 < 0.9:
        f =  173 * np.exp(1j * np.sin(5 * pi2 * v))
    else:
        f =  257 * np.exp(1j * np.sin(23 * pi2 * v))

    f[n-1] +=  211 * np.exp(1j * pi2 * (1/7) * t2 )

    return f
    
# =====================
# Numba core utilities
# =====================

@njit(cache=True, fastmath=True)
def _horner_and_deriv(a: np.ndarray, z: complex) -> Tuple[complex, complex]:
    n = a.size - 1
    p = a[0]
    dp = 0.0 + 0.0j
    for k in range(1, n + 1):
        dp = dp * z + p
        p = p * z + a[k]
    return p, dp

@njit(cache=True, fastmath=True)
def _cauchy_radius(a: np.ndarray) -> float:
    a0 = a[0]
    if a0 == 0:
        return 1.0
    inv = 1.0 / abs(a0)
    m = 0.0
    for k in range(1, a.size):
        v = abs(a[k]) * inv
        if v > m: m = v
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
        p[i]  = pi
        dp[i] = dpi

    z_new = np.empty_like(z)
    max_step = 0.0
    tiny = 1e-300
    for i in range(n):
        zi = z[i]
        dpi = dp[i]
        if abs(dpi) < tiny:
            wi = p[i] / (dpi + 1e-16)
            step = wi
            if newton_fallback:
                mag = abs(step)
                if mag > 1.0:
                    step = step / mag
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
def _aberth_solve(
    a: np.ndarray,
    z0: np.ndarray,
    tol: float,
    max_iters: int,
    per_root_tol: bool,
    newton_fallback: bool
) -> Tuple[np.ndarray, int, float]:
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
# Tile worker (no @njit) + MP driver
# =======================


def serpentine_grid(
        n: int, 
        x0=0.0, x1=1.0, 
        y0=0.0, y1=1.0
    ):
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows

    X, Y = np.meshgrid(xs, ys)

    # serpentine: flip every other row (odd rows)
    X[1::2] = X[1::2, ::-1]
    Y[1::2] = Y[1::2, ::-1]

    coords = np.column_stack((X.ravel(), Y.ravel()))[:n]

    # scale to target ranges
    coords[:, 0] = x0 + coords[:, 0] * (x1 - x0)
    coords[:, 1] = y0 + coords[:, 1] * (y1 - y0)
    return coords


def tile_worker(args):
    (
        tile_id, 
        s_start, s_end, t_start, t_end,
        n_points, 
        func, 
        tol, max_iters, per_root_tol, newton_fallback, verbose,
        result_name, result_rows
    ) = args

    if verbose:
        print(
            f"worker {tile_id:2d} "
            f"({s_start:.6f},{t_start:.6f})–({s_end:.6f},{t_end:.6f}) "
            f"{n_points}"
        )

    shm_result, result = get_shm(result_name,result_rows,6,np.float64)

    # degree from a sample coeff vector
   
    pts = serpentine_grid(n_points,s_start,s_end,t_start,t_end)
    cf = func(pts[0,0], pts[0,1])
    guess = np.roots(cf).astype(np.complex128)
    idx,tot_niter = tile_id*n_points*len(guess),0

    for k in range(n_points):

        if tile_id == 15 and k % (n_points//100) == 0: 
            print(
                f"worker {tile_id} : "
                f"done: {100*k/n_points:.0f}% [{k:,}/{n_points:,}] "
                f"s: {pts[k,0]:.3f} t: {pts[k,1]:.3f}] "
            )
        
        cf = func(pts[k,0], pts[k,1])
        if use_aberth:
            roots, niter, err = _aberth_solve(cf, guess, tol, max_iters, per_root_tol, newton_fallback)
        else:
            roots, niter, err = np.roots(cf), 1, 1.0
    
        tot_niter += niter

        # write rows
        for r in range(len(roots)):
            z = roots[r]
            if np.isfinite(z):
                result[idx, :6] = (z.real,z.imag,pts[k,0],pts[k,1],float(niter),err)
                idx += 1

        # warm start
        guess = roots 

    shm_result.close() 
    return tot_niter

def scan(
        func: Callable[[float, float], np.ndarray],
        N: int,
        *,
        tol: float = 1e-12,
        max_iters: int = 80,
        per_root_tol: bool = False,
        newton_fallback: bool = False,
        verbose: bool = True
    ) -> np.ndarray:
    """
    func(a,b)->coeffs must be top-level (picklable). Returns stacked float64 array:
    columns [Re, Im, s, t, n_iter, err].
    """
    cf = func(0.1, 0.1)
    guess = np.roots(cf).astype(np.complex128)
    shm_result, result = make_shm(N*len(guess),6,np.float64)

    ctx = mproc.get_context("spawn")
    ncpu = mproc.cpu_count()

    tiles = int(ncpu**0.5)**2                    # square tiles
    tiles_per_side = int(tiles**0.5)
    assert tiles_per_side**2 == tiles, "tiles must be a square number"

    points_per_worker = int((N // tiles)**0.5)**2  # perfect square per worker
    if points_per_worker < 1:
        points_per_worker = 1
    steps_side = int(np.sqrt(points_per_worker))
    N_effective = tiles * (steps_side * steps_side)
    if verbose:
        print(
            f"[mp] ncpu={ncpu} "
            f"tiles={tiles} "
            f"({tiles_per_side}×{tiles_per_side}) "
            f"ppw={points_per_worker} "
            f"(steps_side={steps_side}) N_effective={N_effective}"
        )

    tile_size = 1.0 / tiles_per_side

    args = []
    for wid in range(ncpu):
        tx = wid % tiles_per_side
        ty = wid // tiles_per_side
        s_start = tx * tile_size
        s_end   = s_start + tile_size
        t_start = ty * tile_size
        t_end   = t_start + tile_size
        if verbose:
            print(f"worker {wid:2d} [{tx},{ty}] : ({s_start:.6f},{t_start:.6f})–({s_end:.6f},{t_end:.6f}) : ppw={points_per_worker}")
        args.append((
            wid, 
            s_start, s_end, t_start, t_end,
            points_per_worker, 
            func, 
            tol, max_iters, per_root_tol, newton_fallback, verbose,
            shm_result.name, N*len(guess)
        ))

    # Run
    with ctx.Pool(processes=len(args)) as pool:
        chunks = pool.map(tile_worker, args)

    out = np.copy(result)
    shm_result.close()
    shm_result.unlink()

    return out

# =======================
# Rasterizer (PIL)
# =======================

def view( re:np.ndarray, im:np.ndarray, args):
    rx = re.max() - re.min() if re.size else 1.0
    ry = im.max() - im.min() if im.size else 1.0
    pad_x = args.pad * (rx if rx > 0 else 1.0)
    pad_y = args.pad * (ry if ry > 0 else 1.0)
    llx = args.llx if args.llx is not None else re.min() - pad_x
    urx = args.urx if args.urx is not None else re.max() + pad_x
    lly = args.lly if args.lly is not None else im.min() - pad_y
    ury = args.ury if args.ury is not None else im.max() + pad_y
    return llx, lly, urx, ury

@njit(cache=True, fastmath=True)
def rasterize_points(
    re: np.ndarray,
    im: np.ndarray,
    llx: float, lly: float, urx: float, ury: float,
    pixels: int
):
    counts = np.zeros((pixels, pixels), dtype=np.int32)

    # map (x,y) = (im,re) into pixel grid
    span_x = urx - llx
    span_y = ury - lly
    if span_x <= 0.0:
        span_x = 1e-10
    if span_y <= 0.0:
        span_y = 1e-10
    sx = pixels / span_x
    sy = pixels / span_y

    n = re.size
    for i in range(n):
        x = im[i]  # your choice: x <- im, y <- re
        y = re[i]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue

        ix = int(math.floor((x - llx) * sx))
        iy = int(math.floor((y - lly) * sy))

         # skip if out of range
        if 0 <= ix < pixels and 0 <= iy < pixels:
            counts[iy, ix] += 1

    img = np.zeros((pixels, pixels), dtype=np.uint8)
    for i in range(pixels):
        for j in range(pixels):
            if counts[i, j] > 0:
                img[pixels - 1 - i, j] = 255  # flip vertically
    return img


# =======================
# CLI
# =======================

if __name__ == "__main__":
    import argparse
    from PIL import Image

    ap = argparse.ArgumentParser(description="Tile-only Aberth manifold scan (monomial).")
    ap.add_argument("--pps", type=int, default=1_000, help="Points per side")
    ap.add_argument("--tol", type=float, default=1e-12, help="Convergence tolerance")
    ap.add_argument("--max-iters", type=int, default=100, help="Maximum Aberth iterations per point")
    ap.add_argument("--per-root-tol", action="store_true", help="Use per-root relative step tolerance")
    ap.add_argument("--newton-fallback", action="store_true", help="Damped Newton fallback if p'(z) ~ 0")
    ap.add_argument("--verbose", action="store_true")

    # PNG
    ap.add_argument("--png", type=str, default=None, help="Output PNG path (grayscale)")
    ap.add_argument("--pixels", type=int, default=4096)
    ap.add_argument("--llx", type=float, default=None)
    ap.add_argument("--lly", type=float, default=None)
    ap.add_argument("--urx", type=float, default=None)
    ap.add_argument("--ury", type=float, default=None)
    ap.add_argument("--pad", type=float, default=0.05)

    args = ap.parse_args()


    N = args.pps*args.pps
    t0 = time.perf_counter()
    roots_mat = scan(
        p7f, N,
        tol=args.tol,
        max_iters=args.max_iters,
        per_root_tol=args.per_root_tol,
        newton_fallback=args.newton_fallback,
        verbose=args.verbose
    )
    print(f"scan time: {time.perf_counter() - t0:.3f}s")
    # roots_mat shape: (M, 6) with columns [Re, Im, s, t, n_iter, err]
    print(f"rows: {roots_mat.shape[0]:,} cols: {roots_mat.shape[1]:,}")
    print(f"min real: {np.min(roots_mat[:,0]):.2f} min imag: {np.min(roots_mat[:,1]):.2f}")
    print(f"max real: {np.max(roots_mat[:,0]):.2f} max imag: {np.max(roots_mat[:,1]):.2f}")
    print(f"min s: {np.min(roots_mat[:,2]):.2f} min t: {np.min(roots_mat[:,3]):.2f}")
    print(f"max s: {np.max(roots_mat[:,2]):.2f} max t: {np.max(roots_mat[:,3]):.2f}")

    if args.png is not None:

        re = roots_mat[:, 0]
        im = roots_mat[:, 1]
        llx, lly, urx, ury = view(re,im,args)

        print(f"llx: {llx:.2f} lly: {lly:.2f}")
        print(f"urx: {urx:.2f} ury: {ury:.2f}")
 
        img_arr = rasterize_points( re, im, llx, lly, urx, ury, args.pixels )
        img = Image.fromarray(img_arr, mode="L")
        img.save(args.png, optimize=True)
        print(f"Saved PNG to {args.png} [{args.pixels}x{args.pixels}]")


