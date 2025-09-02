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
from multiprocessing.shared_memory import SharedMemory

# =======================
# shared memory
# =======================

def make_shm(rows,cols,type):
    shm = SharedMemory(
        create=True, 
        size = rows * cols * np.dtype(type).itemsize
    )
    array = np.ndarray(
        (rows,cols), 
        dtype=type, 
        buffer=shm.buf
    )
    array[:] = 0
    return (shm,array)

def get_shm(name,rows,cols,type):
    shm = SharedMemory(name=name)
    array = np.ndarray(
        (rows, cols), 
        dtype=type, 
        buffer=shm.buf
    )
    return(shm,array)

# =======================
# polys (example coeff gen)
# =======================

@njit(cache=True, fastmath=True)
def p1_1(a: float, b: float) -> np.ndarray:
    # returns monomial coeffs (highest-first) for degree=20 (length 21)
    c = np.zeros(21, dtype=np.complex128)
    c[0] = 1.0
    c[1] = a + b
    c[2] = a*1j + b
    c[3] = b*np.exp(1j*2*np.pi*a)
    c[4] = a*np.exp(1j*2*np.pi*b)
    c[-10] = 100*a
    c[-2] = a * 1j
    c[-2] = b
    c[-1] = 1.0
    return c

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
# Tile worker (no @njit) + MP driver
# =======================

def tile_worker(args):
    """
    Unpack args and scan one tile. Returns an array of shape (n_points*degree, 6):
    [Re(z), Im(z), s, t, n_iter, err]
    """
    (
        tile_id, 
        s_start, s_end, t_start, t_end,
        n_points, 
        func, 
        tol, max_iters, per_root_tol, newton_fallback, verbose_first,
        result_name, result_rows
    ) = args

    shm_result, result = get_shm(result_name,result_rows,6,np.float64)

    # degree from a sample coeff vector
    cf0 = func(0.0, 0.0)
    guess = np.roots(cf0).astype(np.complex128)
    prev_guess = guess.copy()
    out = np.empty((n_points * len(guess), 6), dtype=np.float64)

    steps_side = int(np.sqrt(n_points))
    assert steps_side * steps_side == n_points, "n_points must be a perfect square"
    s_step = (s_end - s_start) / steps_side
    t_step = (t_end - t_start) / steps_side

    idx, i, j,dj, tot_niter = 0, 0, 0, 1, 0

    for k in range(n_points):
        s = s_start + i * s_step
        t = t_start + j * t_step

        if tile_id < 1 and k % (n_points//100) == 0: 
                print(
                    f"worker {tile_id} : "
                    f"done: {100*k/n_points:.0f}% [{k:,}/{n_points:,}] "
                )
        
        local_cf = func(s, t)
        roots, niter, err = _aberth_solve(local_cf, guess, tol, max_iters, per_root_tol, newton_fallback)
    
        tot_niter += niter

        # write rows
        for r in range(len(roots)):
            z = roots[r]
            out[idx, 0] = z.real
            out[idx, 1] = z.imag
            out[idx, 2] = s
            out[idx, 3] = t
            out[idx, 4] = float(niter)
            out[idx, 5] = err
            idx += 1

        # extrapolated warm start
        guess = roots + (roots - prev_guess)
        prev_guess = roots.copy()

        # snake advance
        j += dj
        if j == steps_side or j < 0:
            i += 1
            dj = -dj
            j += dj
            guess = roots.copy()
            prev_guess = guess.copy()

    shm_result.close() 
    return tot_niter

def scan_mp(
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
    cf0 = func(0.0, 0.0)
    guess = np.roots(cf0).astype(np.complex128)
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
            print(f"[mp] worker {wid:2d} [{tx},{ty}] : ({s_start:.6f},{t_start:.6f})–({s_end:.6f},{t_end:.6f}) : ppw={points_per_worker}")
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

def rasterize_points(
        re,
        im,
        llx, lly, urx, ury,
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
    from PIL import Image

    ap = argparse.ArgumentParser(description="Tile-only Aberth manifold scan (monomial).")
    ap.add_argument("--pps", type=int, default=1_000, help="Points per side")
    ap.add_argument("--tol", type=float, default=1e-12, help="Convergence tolerance")
    ap.add_argument("--max-iters", type=int, default=80, help="Maximum Aberth iterations per point")
    ap.add_argument("--per-root-tol", action="store_true", help="Use per-root relative step tolerance")
    ap.add_argument("--newton-fallback", action="store_true", help="Damped Newton fallback if p'(z) ~ 0")
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

   

    # Run (using p1_1 as the coeff generator)
    roots_mat = scan_mp(
        func,
        args.pps*args.pps,
        tol=args.tol,
        max_iters=args.max_iters,
        per_root_tol=args.per_root_tol,
        newton_fallback=args.newton_fallback,
        verbose=args.verbose
    )
    # roots_mat shape: (M, 6) with columns [Re, Im, s, t, n_iter, err]
    print(f"rows: {roots_mat.shape[0]:,} cols: {roots_mat.shape[1]:,}")
    print(f"max real: {np.max(roots_mat[:,0]):.2f} max imag: {np.min(roots_mat[:,0]):.2f}")

    shm_result.close()
    shm_result.unlink()

    if args.png is not None:
        re = roots_mat[:, 0]
        im = roots_mat[:, 1]
        if args.llx is not None:
            llx, lly, urx, ury = args.llx, args.lly, args.urx, args.ury
        else:
            rx = re.max() - re.min() if re.size else 1.0
            ry = im.max() - im.min() if im.size else 1.0
            pad_x = 0.05 * (rx if rx > 0 else 1.0)
            pad_y = 0.05 * (ry if ry > 0 else 1.0)
            llx = re.min() - pad_x; urx = re.max() + pad_x
            lly = im.min() - pad_y; ury = im.max() + pad_y

        img_arr = rasterize_points(
            re, im, llx, lly, urx, ury,
            args.width, args.height,
            x_is_im=args.x_is_im, flip_y=args.flip_y,
            dtype=np.uint8, binary=args.binary, scale=args.scale
        )
        img = Image.fromarray(img_arr, mode=args.mode)
        img.save(args.png, optimize=True)
        print(f"Saved PNG to {args.png} [{args.width}x{args.height} mode={args.mode} scale={args.scale}]")


