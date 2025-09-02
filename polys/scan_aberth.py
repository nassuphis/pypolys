# scan_aberth.py (tile-only, multiprocessing)
# Polynomial manifold scanner using Aberth–Ehrlich, monomial basis.
# - Parallelized over tiles (each worker computes its own coefficients).
# - Warm starts within each tile with snake order + extrapolated guesses.
# - Optional PNG output via PIL by rasterizing all roots.
# Author: ChatGPT (GPT-5 Thinking) — 2025-08-29

from __future__ import annotations
import numpy as np
from numba import njit, objmode
from typing import Tuple, Callable
import multiprocessing as mproc
from multiprocessing.shared_memory import SharedMemory
import time
import pyvips as vips
import ast

# =======================
# parameters
# =======================

use_aberth = True
# poly_giga_19 @ 2000*2000 evals
# 356,000,000 roots
# aberth: 177 sec
# np.roots: 2461 sec
# aberth does 13x speedup over np.roots
# for degree-90 poly
# speedup goes up with degree

# scan time: 565.457s rows: 1,000,000 cols: 90
# scan time: 40.010s rows: 1,000,000 cols: 90
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
# tiny guarded math helpers (complex, numba-safe)
# =======================
# sensible clamps for double precision
_EXP_REAL_CLAMP = 20.0    # exp(40) ~ 2.35e17 (safe margin)
_IMAG_CLAMP     = 20.0    # keeps cosh/sinh <= ~1.2e17

@njit(cache=True, fastmath=True)
def _cap_mag(z: complex, max_mag: float) -> complex:
    m = abs(z)
    if m > max_mag:
        z = z * (max_mag / m)
    return z

@njit(cache=True, fastmath=True)
def _clamp_real(x: float, lo: float, hi: float) -> float:
    if x < lo: return lo
    if x > hi: return hi
    return x

@njit(cache=True, fastmath=True)
def _c_log(z: np.complex128, eps: float = 1e-100) -> np.complex128:
    r = abs(z)
    if r < eps:
        if z == 0.0 + 0.0j:
            return np.log(eps) + 0.0j
        z = (z / r) * eps
    return np.log(z)

@njit(cache=True, fastmath=True)
def _c_pow_frac(z: np.complex128, p: float) -> np.complex128:
    w = p * _c_log(z)                 # complex
    u = _clamp_real(w.real, -_EXP_REAL_CLAMP, _EXP_REAL_CLAMP)
    v = w.imag
    # e^{u+iv} = e^u (cos v + i sin v)
    eu = np.exp(u)
    return eu * (np.cos(v) + 1j * np.sin(v))

@njit(cache=True, fastmath=True)
def _c_exp(z: np.complex128) -> np.complex128:
    # exp(x+iy) = exp(x)(cos y + i sin y); clamp x to avoid overflow
    xr = _clamp_real(z.real, -_EXP_REAL_CLAMP, _EXP_REAL_CLAMP)  # exp(40) ~ 2.35e17
    return np.exp(xr) * (np.cos(z.imag) + 1j*np.sin(z.imag))

@njit(cache=True, fastmath=True)
def _c_sin(z: np.complex128) -> np.complex128:
    x = z.real
    y = z.imag
    y = _clamp_real(y, -_IMAG_CLAMP, _IMAG_CLAMP)
    return np.sin(x) * np.cosh(y) + 1j * (np.cos(x) * np.sinh(y))

@njit(cache=True, fastmath=True)
def _c_cos(z: np.complex128) -> np.complex128:
    x = z.real
    y = z.imag
    y = _clamp_real(y, -_IMAG_CLAMP, _IMAG_CLAMP)
    return np.cos(x) * np.cosh(y) - 1j * (np.sin(x) * np.sinh(y))

@njit(cache=True, fastmath=True)
def _safe_div(top: np.complex128, bot: np.complex128, eps: float = 1e-12) -> np.complex128:
    # Tikhonov-regularized division: top / bot ≈ top*conj(bot)/( |bot|^2 + eps^2 )
    br = bot.real; bi = bot.imag
    denom = br*br + bi*bi + eps*eps
    tr = top.real; ti = top.imag
    num_r = tr*br + ti*bi
    num_i = ti*br - tr*bi
    return (num_r/denom) + 1j*(num_i/denom)

@njit(cache=True, fastmath=True)
def _rotate_poly_safe(coeffs: np.ndarray, theta: float, eps: float = 1e-16) -> np.ndarray:
    # Rotate all roots by e^{i theta}; highest-first coeffs in/out
    n = coeffs.size - 1
    out = np.empty_like(coeffs)
    for idx in range(coeffs.size):
        k = n - idx
        out[idx] = coeffs[idx] * np.exp(-1j * theta * k)
    a0 = coeffs[0]
    scale = np.exp(1j * n * theta)
    if abs(a0) > eps:
        scale = scale / a0
    for i in range(out.size):
        out[i] = out[i] * scale
    return out

# =====================
# Numba core utilities
# =====================

@njit(cache=True, fastmath=True)
def rotate_poly(coeffs:np.ndarray, theta:float) -> np.ndarray:
    n = len(coeffs) - 1
    a0 = coeffs[0]
    k = np.arange(n, -1, -1)  # powers: n .. 0
    rotated = coeffs * np.exp(-1j * theta * k)
    rotated *= np.exp(1j * n * theta) / a0
    return rotated

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

def jitter_points(pts, s_start, s_end, t_start, t_end, seed=None):
    n = pts.shape[0]
    steps_side = int(round(np.sqrt(n)))  
    ds = (s_end - s_start) / steps_side
    dt = (t_end - t_start) / steps_side
    rng = np.random.default_rng(seed)
    J = rng.uniform(-0.5, 0.5, size=pts.shape) * np.array([ds, dt])
    out = pts.astype(np.float64, copy=True)
    out += J
    out[:, 0] = np.clip(out[:, 0], s_start, s_end)
    out[:, 1] = np.clip(out[:, 1], t_start, t_end)
    return out

def tile_worker(args):
    (
        tile_id, 
        s_start, s_end, t_start, t_end,
        n_points, 
        func, 
        tol, max_iters, per_root_tol, newton_fallback, verbose,
        result_name, result_rows, result_cols  # <-- extra: cols = 1 + deg
    ) = args

    # complex128 result buffer: shape (result_rows, 1+deg)
    shm_result, result = get_shm(result_name, result_rows, result_cols, np.complex128)

    # serpentine points with or without jitter for this tile
    pts = serpentine_grid(n_points, s_start, s_end, t_start, t_end)
    pts = jitter_points(pts, s_start, s_end, t_start, t_end, seed=None)

    # degree & warm start from a sample

    deg = result_cols -1
    guess = np.zeros(deg,dtype=np.complex128)

    for gi in range(len(pts)):
        cf = func(pts[gi,0], pts[gi,1])
        if np.any(np.isnan(cf)):
            continue
        guess = np.roots(cf).astype(np.complex128)
        if len(guess)==deg:
            break
    if len(guess) != deg:
        raise ValueError(
            f"[worker {tile_id}] : can not find length {deg} guess "
        )

    if verbose:
        print(
            f"worker {tile_id:2d} "
            f"({s_start:.6f},{t_start:.6f})–({s_end:.6f},{t_end:.6f}) "
            f"{n_points} "
            f"deg: {deg} gi: {gi} "
        )
    # this worker owns a contiguous block of rows
    base = tile_id * n_points
    tot_niter = 0
    tot_aberth = 0
    niter = 0
    err = 0
    roots = None

    for k in range(gi,len(pts)):

        if verbose and tile_id == 0 and (n_points >= 100) and (k % (n_points // 100) == 0):
            print(
                f"worker {tile_id} : "
                f"done: {100*k/n_points:.0f}% [{k:,}/{n_points:,}] "
                f"s: {pts[k,0]:.3f} t: {pts[k,1]:.3f}] "
                f"deg: {deg} "
                f"last deg: {len(roots) if not roots is None else 0:,} "
                f"ig: {gi} "
                f"mean niter: {tot_niter/tot_aberth if tot_aberth>0 else max_iters:.2f} "
                f"last niter: {niter} "
                f"max ninter: {max_iters} "
                f"last err: {np.log10(err) if err>0 else 0.0:.2f}"
            )

        cf = func(pts[k,0], pts[k,1]).astype(np.complex128)
        
        niter, err = 1, 1.0
        if not np.any(np.isnan(cf)):
            if use_aberth:
                    roots, niter, err = _aberth_solve(cf, guess, tol, max_iters, per_root_tol, newton_fallback)
                    tot_niter  += niter
                    tot_aberth += 1
                            
            else:
                roots = np.roots(cf)        
        else:
            roots = guess

        roots = roots.astype(np.complex128)
        row_idx = base + k
        result[row_idx, 0] = pts[k,0] + 1j * pts[k,1]

        if len(roots) == deg:
            result[row_idx, 1:] = roots
            guess = roots
        else:
            result[row_idx, 1:] = 0.0+0.0j
            
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
    func(a,b)->coeffs must be top-level (picklable).
    Returns complex128 array of shape (n_points_total, 1+deg):
      col 0: s + 1j*t
      col 1..deg: roots for that (s,t).
    """
    # probe degree
    cf0 = func(0.1, 0.1)
    deg = len(cf0) - 1

    ctx = mproc.get_context("spawn")
    ncpu = mproc.cpu_count()

    tiles = int(ncpu**0.5)**2                    # square tiles
    tiles_per_side = int(tiles**0.5)
    assert tiles_per_side**2 == tiles, "tiles must be a square number"

    points_per_worker = int((N // tiles)**0.5)**2  # perfect square per worker
    if points_per_worker < 1:
        points_per_worker = 1
    steps_side = int(np.sqrt(points_per_worker))

    # actual points computed (since we only launch ncpu workers)
    rows_total = ncpu * points_per_worker
    if verbose:
        print(
            f"[mp] ncpu={ncpu} "
            f"tiles={tiles} "
            f"({tiles_per_side}×{tiles_per_side}) "
            f"ppw={points_per_worker} "
            f"(steps_side={steps_side}) rows_total={rows_total}"
        )

    tile_size = 1.0 / tiles_per_side

    # shared complex result buffer: (rows_total, 1+deg)
    shm_result, result = make_shm(rows_total, 1 + deg, np.complex128)

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
            shm_result.name, rows_total, 1 + deg
        ))

    with ctx.Pool(processes=len(args)) as pool:
        _ = pool.map(tile_worker, args)

    out = np.copy(result)
    shm_result.close()
    shm_result.unlink()
    return out

# =======================
# polys (example coeff gen)
# =======================
@njit(cache=True, fastmath=True)
def uc(s: float, t: float) -> Tuple[np.complex128, np.complex128]:
    t1 = np.exp(1j * 2.0 * np.pi * s)  # safe (s∈[0,1])
    t2 = np.exp(1j * 2.0 * np.pi * t)  # safe (t∈[0,1])
    return( t1, t2 )

@njit(cache=True, fastmath=True)
def coeff2(t1: np.complex128, t2: np.complex128) -> Tuple[np.complex128, np.complex128]:
   return(t1 + t2,t1 * t2)

@njit(cache=True, fastmath=True)
def coeff3(t1: np.complex128, t2: np.complex128) -> Tuple[np.complex128, np.complex128]:
  return(_safe_div(1,t1 + 2),_safe_div(1,t2 + 2))

@njit(cache=True, fastmath=True)
def coeff3u(t1: np.complex128, t2: np.complex128) -> Tuple[np.complex128, np.complex128]:
  return(1/(t1 + 2),1/(t2 + 2))

@njit(cache=True, fastmath=True)
def coeff3a(t1: np.complex128, t2: np.complex128) -> Tuple[np.complex128, np.complex128]:
  return(_safe_div(1,t1 + 1),_safe_div(1,t2 + 1))

@njit(cache=True, fastmath=True)
def coeff4(t1: np.complex128, t2: np.complex128) -> Tuple[np.complex128, np.complex128]:
  return( np.cos(t1),np.sin(t2))

@njit(cache=True, fastmath=True)
def coeff5(t1: np.complex128, t2: np.complex128) -> Tuple[np.complex128, np.complex128]:
  return( t1 + _safe_div( 1.0 + 0.0j, t2 ), t2 + _safe_div( 1.0 + 0.0j, t1 ) )

@njit(cache=True, fastmath=True)
def coeff5u(t1: np.complex128, t2: np.complex128) -> Tuple[np.complex128, np.complex128]:
  return( t1 + (1.0+0.0j)/t2 , t2 + (1.0+0.0j)/t1 )

@njit(cache=True, fastmath=True)
def coeff6(t1: np.complex128, t2: np.complex128) -> Tuple[np.complex128, np.complex128]:
  num1 = t1*t1*t1 + 1j
  den1 = t1*t1*t1 - 1j
  val1 = _safe_div(num1 , den1, eps=1e-12 )
  num2 = t2*t2*t2 + 1j
  den2 = t2*t2*t2 - 1j
  val2 = _safe_div(num2 , den2, eps=1e-12 )
  return ( val1, val2 )

@njit(cache=True, fastmath=True)
def coeff7(t1: np.complex128, t2: np.complex128) -> Tuple[np.complex128, np.complex128]:
  top1  = t1 + np.sin(t1)
  bot1  = t1 + np.cos(t1)
  val1  = _safe_div(top1 , bot1 )
  top2  = t2 + np.sin(t2)
  bot2  = t2 + np.cos(t2)
  val2  = _safe_div(top2 , bot2 )
  return (val1, val2)

@njit(cache=True, fastmath=True)
def coeff8(t1, t2): 
  top1  = t1 + np.sin(t2)
  bot1  = t2 + np.cos(t1)
  val1  = _safe_div(top1 , bot1)
  top2  = t2 + np.sin(t1)
  bot2  = t1 + np.cos(t2)
  val2  = _safe_div(top2 , bot2)
  return (val1, val2)

@njit(cache=True, fastmath=True)
def coeff9(t1: np.complex128, t2: np.complex128) -> Tuple[np.complex128, np.complex128]:
  top1  = t1*t1 + 1j * t2
  bot1  = t1*t1 - 1j * t2
  val1  = _safe_div(top1,bot1, eps=1e-12)
  top2  = t2*t2 + 1j * t1
  bot2  = t2*t2 - 1j * t1
  val2  = _safe_div(top2,bot2, eps=1e-12)
  return (val1, val2)

@njit(cache=True, fastmath=True)
def coeff10(t1: np.complex128, t2: np.complex128) -> Tuple[np.complex128, np.complex128]:
    top1 = t1*t1*t1*t1 - t2
    bot1 = t1*t1*t1*t1 + t2
    val1 = _safe_div(top1,bot1, eps=1e-12)
    top2 = t2*t2*t2*t2 - t1
    bot2 = t2*t2*t2*t2 + t1
    val2 = _safe_div(top2,bot2, eps=1e-12)
    return( val1, val2 )

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
    t1, t2 = uc( s, t )
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
    t1, t2 = uc( s, t )

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
    
@njit(cache=True, fastmath=True)
def poly_giga_42(s: float, t: float) -> np.ndarray:
    # R: length 50, indices 1-based; here: 0-based
    t1, t2 = uc( s, t )
    cf = np.zeros(45, dtype=np.complex128)

    # cf[c(1, 8, 16, 32, 40)] = c(1, -3, 3, -1, 2)
    cf[0]  = 1.0 + 0.0j    # index 1
    cf[7]  = -3.0 + 0.0j   # index 8
    cf[15] = 3.0 + 0.0j    # index 16
    cf[31] = -1.0 + 0.0j   # index 32
    cf[39] = 2.0 + 0.0j    # index 40

    # cf[12] = 100i * exp(t1^2 + t2^2)
    cf[11] = 100.0j * np.exp(t1 * t1 + t2 * t2)

    # cf[20] = 50 * (t1^3 + t2^3)
    cf[19] = 50.0 * (t1 * t1 * t1 + t2 * t2 * t2)

    # cf[25] = exp(1i * (t1 - t2)) + 10 * t1^2
    cf[24] = np.exp(1.0j * (t1 - t2)) + 10.0 * (t1 * t1)

    # cf[45] = 200 * sin(t1 + t2) + 1i * cos(t1 - t2)
    cf[44] = 200.0 * np.sin(t1 + t2) + 1.0j * np.cos(t1 - t2)

    # numpy roots expect highest
    return np.flip(cf)

@njit(cache=True, fastmath=True)
def poly_giga_0142(s: float, t: float) -> np.ndarray:
    tt1, tt2 = uc( s, t )
    t1 = tt1 + tt2
    t2 = tt1 * tt2
    n = 40
    a = abs(t1 + t2) * 0.5
    m = int((5.0 * a) % 21.0) + 3  # in {3, ..., 23}
    v = np.power(np.linspace(0.0, 1.0, n), 0.75) / (m + t1 + t2) 
    uc = np.exp(1j * 50.0 * np.pi * v)                  
    sf = (np.arange(n, dtype=np.float64) % float(m + 10)) 
    cf = (sf * uc).astype(np.complex128)  # (n,) complex128
    return cf[1:]


@njit(cache=True, fastmath=True)
def p11b2_v5(s: float, t: float) -> np.ndarray:
    t1 = s
    t2 = t
    n = 11
    two_pi = 2.0 * np.pi
    x1 = np.linspace(t1, t2, n) 
    x2 = np.linspace(t1+t2,t1*t2, n) 
    v1 = np.exp(1j * two_pi * x1) 
    v2 = np.exp(1j * two_pi * x2)      
    v = v1 + 1j * v2  
    denom = t1 + t2 + 3.0
    ad = abs(denom)
    if ad == 0.0:
        denom = 1.0
    elif ad < 1.0:
        denom = denom / ad 
    u = (n * v) / denom  
    uc = np.exp(1j * np.pi * u) 
    return uc.astype(np.complex128)

# ---------- persistent state held in Python (per process) ----------
_p11b2_guess: np.ndarray | None = None

def _get_p11b2_guess() -> np.ndarray:
    global _p11b2_guess
    if _p11b2_guess is None:
        g0 = np.roots(p11b2_v5(0.01, 0.01)).astype(np.complex128)
        _p11b2_guess = g0
    return _p11b2_guess

def _set_p11b2_guess(new_guess: np.ndarray) -> None:
    global _p11b2_guess
    _p11b2_guess = np.asarray(new_guess, dtype=np.complex128)

@njit(cache=True, fastmath=True)
def p11b2_v5_nproot(t1: float, t2: float) -> np.ndarray:
    cf = p11b2_v5(t1, t2)
    roots = np.roots(cf)
    return roots

@njit(cache=True, fastmath=True)
def p11b2_v5_revnproot(t1: float, t2: float) -> np.ndarray:
    cf = np.flip(p11b2_v5(t1, t2))
    roots = np.roots(cf)
    return roots

@njit(cache=True, fastmath=True)
def p11b2_v5_abroot(t1: float, t2: float) -> np.ndarray:
    cf = p11b2_v5(t1, t2)
    with objmode(g='complex128[:]'): g = _get_p11b2_guess()
    roots, niter, err = _aberth_solve(cf, g, 1e-10, 120, True, True)
    with objmode(): _set_p11b2_guess(roots)
    return roots

@njit(cache=True, fastmath=True)
def p11b2_v5_revabroot(t1: float, t2: float) -> np.ndarray:
    cf = np.flip(p11b2_v5(t1, t2))
    with objmode(g='complex128[:]'): g = _get_p11b2_guess()
    roots, niter, err = _aberth_solve(cf, g, 1e-10, 120, False, False)
    with objmode(): _set_p11b2_guess(roots)
    return roots

@njit(cache=True, fastmath=True)
def poly_cf1p1(s:float, t:float) -> np.ndarray:
    t1, t2 = uc( s, t )
    cf = np.zeros(36, dtype=np.complex128)
    for i in range(1, 37):
        cf[i-1] = np.sin(t1**(i/2)) * np.cos(t2**(i/3)) + (i**2) * t1 * t2 + np.log(np.abs(t1 + t2) + 1) * 1j * i
    cf[10] = t1 * t2 * np.real(cf[6]) + np.imag(cf[18]) * t1**3
    cf[21] = t2 * cf[10] + np.real(cf[34]) * t1**3
    cf[32] = cf[21] - np.real(cf[16]) * t1**2
    return rotate_poly(np.flip(cf),-np.pi/2)

@njit(cache=True, fastmath=True)
def poly_cf3p1(s: float, t: float) -> np.ndarray:
    t1, t2 = uc( s, t )
    t1, t2 = coeff3( t1, t2 )
    cf = np.zeros(36, dtype=np.complex128)
    L = np.log(abs(t1 + t2) + 1.0)
    for i in range(1, 37):
        pw1 = _c_pow_frac(t1, 0.5 * i)        # t1**(i/2)
        pw2 = _c_pow_frac(t2, (1.0/3.0) * i)  # t2**(i/3)
        term_trig = _c_sin(pw1) * _c_cos(pw2)
        term_poly = (i*i) * t1 * t2
        term_log  = (0.0 + 1.0j) * L * i
        cf[i-1] = term_trig + term_poly + term_log
    cf[10] = t1 * t2 * (cf[6].real) + (cf[18].imag) * t1 * t1 * t1
    cf[21] = t2 * cf[10] + (cf[34].real) * t1 * t1 * t1
    cf[32] = cf[21] - (cf[16].real) * t1 * t1
    return rotate_poly(np.flip(cf),-np.pi/2)
 
# np.roots vs aberth difference
@njit(cache=True, fastmath=True)
def poly_cf3p1a(s: float, t: float) -> np.ndarray:
    t1, t2 = uc( s, t )
    t1, t2 = coeff3u( t1, t2 )
    cf = np.zeros(36, dtype=np.complex128)
    L = np.log(abs(t1 + t2) + 1.0)
    for i in range(1, 37):
        cf[i-1] = np.sin(t1**(i/2)) * np.cos(t2**(i/3)) + (i**2) * t1 * t2 + np.log(np.abs(t1 + t2) + 1) * 1j * i
    cf[10] = t1 * t2 * (cf[6].real) + (cf[18].imag) * t1 * t1 * t1
    cf[21] = t2 * cf[10] + (cf[34].real) * t1 * t1 * t1
    cf[32] = cf[21] - (cf[16].real) * t1 * t1
    return rotate_poly(np.flip(cf),-np.pi/2)

@njit(cache=True, fastmath=True)
def poly_cf5p1(s: float, t: float) -> np.ndarray:
    t1, t2 = uc( s, t )
    t1, t2 = coeff5( t1, t2 )
    cf = np.zeros(36, dtype=np.complex128)
    L = np.log(abs(t1 + t2) + 1.0)
    for i in range(1, 37):
        pw1 = _c_pow_frac(t1, 0.5 * i)        # t1**(i/2)
        pw2 = _c_pow_frac(t2, (1.0/3.0) * i)  # t2**(i/3)
        term_trig = _c_sin(pw1) * _c_cos(pw2)
        term_poly = (i*i) * t1 * t2
        term_log  = (0.0 + 1.0j) * L * i
        cf[i-1] = term_trig + term_poly + term_log
    cf[10] = t1 * t2 * (cf[6].real) + (cf[18].imag) * t1 * t1 * t1
    cf[21] = t2 * cf[10] + (cf[34].real) * t1 * t1 * t1
    cf[32] = cf[21] - (cf[16].real) * t1 * t1
    return rotate_poly(np.flip(cf),-np.pi/2)

# np.roots vs aberth difference
@njit(cache=True, fastmath=True)
def poly_cf5p1u(s: float, t: float) -> np.ndarray:
    t1, t2 = uc( s, t )
    t1, t2 = coeff5u( t1, t2 )
    cf = np.zeros(36, dtype=np.complex128)
    L = np.log(abs(t1 + t2) + 1.0)
    for i in range(1, 37):
        cf[i-1] = np.sin(t1**(i/2)) * np.cos(t2**(i/3)) + (i**2) * t1 * t2 + np.log(np.abs(t1 + t2) + 1) * 1j * i
    cf[10] = t1 * t2 * (cf[6].real) + (cf[18].imag) * t1 * t1 * t1
    cf[21] = t2 * cf[10] + (cf[34].real) * t1 * t1 * t1
    cf[32] = cf[21] - (cf[16].real) * t1 * t1
    return rotate_poly(np.flip(cf),-np.pi/2)

@njit(cache=True, fastmath=True)
def poly_8(s:float, t:float)-> np.ndarray:
    t1, t2 = uc( s, t )
    cf = np.zeros(51, dtype=np.complex128)
    cf[0:25] = np.arange(1, 26) * (t1**2 + 1j * t2**3)
    cf[25] = np.abs(t1 + t2)
    cf[26:51] = np.arange(1, 26) * (t2**2 - 1j * t1**3)
    cf[2] = np.sin(t1) * cf[0]**2
    cf[6] = np.log(np.abs(t2) + 1) * cf[4]**3
    cf[32] = cf[6] + cf[2]
    cf[36] = cf[32] - cf[6]
    cf[40] = cf[32] + cf[2]
    cf[49] = np.angle(t1) * np.angle(t2)
    cf[50] = np.abs(cf[40])
    return rotate_poly(np.flip(cf),-np.pi/2)

@njit(cache=True, fastmath=True)
def poly_52(s:float, t:float)-> np.ndarray:
    t1, t2 = uc( s, t )
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1 ** 7 + t2 ** 7
    for k in range(2, 36):
        cf[k - 1] = np.sin(k * np.log(np.abs(t1) + 1) + np.log(np.abs(t2) + 1))
    for k in range(36, 71):
        cf[k - 1] = np.cos(k * np.log(np.abs(t1) + 1) - np.log(np.abs(t2) + 1))
    cf[70] = t1 * t2 - (t1 + t2) ** 2
    return rotate_poly(np.flip(cf),-np.pi/2)
    
@njit(cache=True, fastmath=True)
def poly_123(s:float, t:float)-> np.ndarray:
    t1, t2 = uc( s, t )
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1.real**2 - t2.imag**2
    cf[1] = (t1 + t2)**2 - 7
    cf[2] = t1**2 - t2**2
    cf[3:10] = np.arange(3, 30, 4) * np.abs(t1 + 1j * t2)
    cf[10:20] = (t1 - t2).real * np.arange(11, 21)
    cf[20:30] = 1 / (1 + np.arange(21, 31)) * (t1 + t2).real
    cf[30] = np.angle(t1) * t2.imag
    cf[31:50] = 1000 * (-1)**np.arange(32, 51) * t1 * t2
    cf[50:60] = 2000 * (-1)**np.arange(51, 61) * np.log(np.abs(t1) + 1)
    cf[60:65] = 1j * np.conj(t1 * t2) * np.sqrt(np.arange(61, 66))
    cf[65:70] = np.arange(66, 71) * (np.arange(66, 71) - 1) / (np.abs(t1) + np.abs(t2) + 1)
    cf[70] = np.prod(np.arange(1, 6))
    return rotate_poly(np.flip(cf),-np.pi/2)

@njit(cache=True, fastmath=True)
def poly_184(s:float, t:float)-> np.ndarray:
    t1, t2 = uc( s, t )
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1**5 + 2 * t2**4
    cf[1] = -3 * t1**4 + 4 * t2**3
    cf[2] = 5 * t1**3 - 6 * t2**2
    cf[3] = -7 * t1**2 + 8 * t2
    cf[4] = 9 * t1 - 10 * t2**0
    cf[5:10] = (t1 * t2)**np.arange(1, 6) * np.array([11, -12, 13, -14, 15])
    cf[10:20] = (t1 + t2)**(np.arange(6, 16) / 2) * np.array([-16, 17, -18, 19, -20, 21, -22, 23, -24, 25])
    cf[20:30] = (t1 - t2)**(np.arange(16, 26) / 3) * np.array([26, -27, 28, -29, 30, 31, -32, 33, -34, 35])
    cf[30:40] = (t1 * t2)**(np.arange(26, 36) / 4) * np.array([36, -37, 38, -39, 40, 41, -42, 43, -44, 45])
    cf[40:50] = (t1 + np.conj(t2))**(np.arange(36, 46) / 5) * np.array([-46, 47, -48, 49, -50, 51, -52, 53, -54, 55])
    cf[50:60] = (np.conj(t1) - t2)**(np.arange(46, 56) / 6) * np.array([56, -57, 58, -59, 60, 61, -62, 63, -64, 65])
    cf[60:70] = (np.abs(t1) + np.abs(t2))**(np.arange(56, 66) / 7) * np.array([-66, 67, -68, 69, -70, 71, -72, 73, -74, 75])
    cf[70] = np.log(np.abs(t1) + 1) + np.log(np.abs(t2) + 1)
    return rotate_poly(np.flip(cf),-np.pi/2)
  
@njit(cache=True, fastmath=True)
def poly_532(s:float, t:float)-> np.ndarray:
    t1, t2 = uc( s, t )
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for r in range(1, n + 1):
        j = r % 7 + 1
        k = np.floor(r / 5) + 1
        magnitude = (np.log(np.abs(t1) + 1) * np.cos(r) + np.log(np.abs(t2) + 1) * np.sin(r)) * (1 + r / 10)
        angle = np.angle(t1) * np.sin(r / 2) - np.angle(t2) * np.cos(r / 3) + np.sin(r) * np.cos(r / 4)
        cf[r - 1] = magnitude * (np.cos(angle) + np.sin(angle) * 1j)
    return rotate_poly(np.flip(cf),-np.pi/2)
    
@njit(cache=True, fastmath=True)
def poly_600(s:float, t:float)-> np.ndarray:
    t1, t2 = uc( s, t )
    degree = 25
    cf = np.zeros(degree + 1, dtype=np.complex128)
    rec_step = np.linspace(t1.real, t2.real, num=degree + 1)
    imc_step = np.linspace(t1.imag, t2.imag, num=degree + 1)
    for j in range(1, degree + 2):
        mag_part1 = np.log(np.abs(t1) + j) * np.sin(j * np.pi / 12)
        mag_part2 = np.cos(j * np.pi / 8) * np.log(np.abs(rec_step[j - 1] - imc_step[j - 1]) + 1)
        magnitude = mag_part1 + mag_part2 + j**0.8
        
        angle_part1 = np.angle(t1) * np.sin(j * np.pi / 10)
        angle_part2 = np.angle(t2) * np.cos(j * np.pi / 14)
        angle_part3 = np.sin(j * np.pi / 6) - np.cos(j * np.pi / 9)
        angle = angle_part1 + angle_part2 + angle_part3
        
        cf[j - 1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return rotate_poly(np.flip(cf),-np.pi/2)


@njit(cache=True, fastmath=True)
def poly_597(s:float, t:float)-> np.ndarray:
    t1, t2 = uc( s, t )
    degree = 25
    cf = np.zeros(degree + 1, dtype=np.complex128)
    r1 = t1.real
    i1 = t1.imag
    r2 = t2.real
    i2 = t2.imag
    for j in range(0, degree + 1):
        mag = 0
        angle = 0
        for k in range(1, j + 2):
            term_mag = np.log(np.abs(t1) * k + 1) * np.sin(k * np.pi * r1) + np.cos(k * np.pi * i2)
            term_angle = np.angle(t1) * k**2 - np.angle(t2) * np.sqrt(k)
            mag += term_mag * np.exp(1j * term_angle)
        if j < degree / 3:
            mag *= (j + 1)
        elif j < 2 * degree / 3:
            mag /= (j + 1)
        else:
            mag *= (j + 1)**2
        cf[j] = mag
    cf[0] = (t1.real * t2.real) + 1j * (t1.imag - t2.imag) + np.sin(t1.real) * np.cos(t2.imag)
    return rotate_poly(np.flip(cf),-np.pi/2)



@njit(cache=True, fastmath=True)
def poly_cf10p1(s: float, t: float) -> np.ndarray:

    t1, t2 = uc( s, t )
    t1, t2 = coeff10( t1, t2 )
    cf = np.zeros(36, dtype=np.complex128)

    # real, safe log(|t1+t2|+1)
    L = np.log(abs(t1 + t2) + 1.0)

    for i in range(1, 37):
        pw1 = _c_pow_frac(t1, 0.5 * i)        # t1**(i/2)
        pw2 = _c_pow_frac(t2, (1.0/3.0) * i)  # t2**(i/3)
        term_trig = _c_sin(pw1) * _c_cos(pw2)
        term_poly = (i*i) * t1 * t2
        term_log  = (0.0 + 1.0j) * L * i
        cf[i-1] = term_trig + term_poly + term_log

    # cross terms (guarded powers used)
    cf[10] = t1 * t2 * (cf[6].real) + (cf[18].imag) * t1 * t1 * t1
    cf[21] = t2 * cf[10] + (cf[34].real) * t1 * t1 * t1
    cf[32] = cf[21] - (cf[16].real) * t1 * t1

    # clean non-finite just in case
    for i in range(cf.size):
        z = cf[i]
        if not (np.isfinite(z.real) and np.isfinite(z.imag)):
            cf[i] = 0.0 + 0.0j

    # highest-first + rotate roots by -pi/2
    return rotate_poly(np.flip(cf),-np.pi/2)


@njit(cache=True, fastmath=True)
def poly_cf10p344(s: float, t: float) -> np.ndarray:
    t1, t2 = uc( s, t )
    t1, t2 = coeff10( t1, t2 )
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec_seq = np.linspace(np.real(t1), np.real(t2), n)
    imc_seq = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(n):
        mag = np.log(np.abs(rec_seq[j] * imc_seq[j]) + 1) * (1 + np.sin(j * np.pi / 3)) * (j % 4 + 1)
        ang = np.angle(t1) * np.cos(j * np.pi / 5) + np.angle(t2) * np.sin(j * np.pi / 7) + np.log(np.abs(rec_seq[j] + imc_seq[j]) + 1)
        cf[j] = mag * (np.cos(ang) + 1j * np.sin(ang))
    for i in range(cf.size):
        z = cf[i]
        if not (np.isfinite(z.real) and np.isfinite(z.imag)):
            cf[i] = 0.0 + 0.0j
    return cf
    
@njit(cache=True, fastmath=True)
def poly_cf9p546(s: float, t: float) -> np.ndarray:
    t1, t2 = uc( s, t )
    t1, t2 = coeff9( t1, t2 )
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec1 = t1.real
    imc1 = t1.imag
    rec2 = t2.real
    imc2 = t2.imag
    for j in range(1, n + 1):
        r = (rec1 + rec2) / 2 + np.sin(j * np.pi / 7) * np.cos(j * np.pi / 5)
        theta = (imc1 - imc2) / n * j + np.sin(j * np.pi / 3)
        magnitude = np.log(np.abs(t1) * j + np.abs(t2) * (n - j + 1)) + np.sqrt(j)
        cf[j - 1] = magnitude * (np.cos(theta) + 1j * np.sin(theta))
    return rotate_poly(np.flip(cf),-np.pi/2)

@njit(cache=True, fastmath=True)
def poly_cf9p440(s: float, t: float) -> np.ndarray:
    t1, t2 = uc( s, t )
    t1, t2 = coeff9( t1, t2 )
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        # Calculate real and imaginary parts
        real_part = np.real(t1)**j * np.log(np.abs(j) + 1) + np.sin(j * np.real(t2)) * np.cos(j**2)
        imag_part = np.imag(t1) * j**0.5 + np.cos(j * np.imag(t2)) * np.log(np.abs(t1 + t2) + 1)
        # Assign to complex coefficient with scaling
        cf[j-1] = (real_part+imag_part * 1j) * (1 + 0.1 * j)
    return rotate_poly(np.flip(cf),-np.pi/2)

@njit(cache=True, fastmath=True)
def poly_cf5p371(s: float, t: float) -> np.ndarray:
    t1, t2 = uc( s, t )
    t1, t2 = coeff5( t1, t2 )
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        mag_part = np.log(np.abs(t1) + j) * np.sqrt(j) + np.sin(j * np.angle(t2))**2
        angle_part = np.angle(t1) * np.cos(j) + np.angle(t2) * np.sin(j)
        cf[j - 1] = mag_part * (np.cos(angle_part) + 1j * np.sin(angle_part)) + np.conj(t1)**j - np.log(np.abs(t2) + 1) * np.sin(j)
    return rotate_poly(np.flip(cf),-np.pi/2)

@njit(cache=True, fastmath=True)
def poly_cf6p345(s: float, t: float) -> np.ndarray:
    t1, t2 = uc( s, t )
    t1, t2 = coeff6( t1, t2 )
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(np.real(t1), np.real(t2), n)
    for j in range(n):
        mag_part =  np.log(np.abs(rec[j]) + 1) * np.prod(np.arange(1, j + 1)) / (j + 2)
        angle_part = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j / 3.0)
        cf[j] = mag_part * (np.cos(angle_part) + 1j * np.sin(angle_part))
        cf[j] += np.conj(cf[j]) * np.sin(j * np.pi / 4)
    return _rotate_poly_safe(np.flip(cf),-np.pi/2)

@njit(cache=True, fastmath=True)
def poly_cf6p344(s: float, t: float) -> np.ndarray:
    t1, t2 = uc( s, t )
    t1, t2 = coeff6( t1, t2 )
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec_seq = np.linspace(np.real(t1), np.real(t2), n)
    imc_seq = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(n):
        mag = np.log(np.abs(rec_seq[j] * imc_seq[j]) + 1) * (1 + np.sin(j * np.pi / 3)) * (j % 4 + 1)
        ang = np.angle(t1) * np.cos(j * np.pi / 5) + np.angle(t2) * np.sin(j * np.pi / 7) + np.log(np.abs(rec_seq[j] + imc_seq[j]) + 1)
        cf[j] = mag * (np.cos(ang) + 1j * np.sin(ang))
    return cf

    
@njit(cache=True, fastmath=True)
def poly_cf4p729(s: float, t: float) -> np.ndarray:
    t1, t2 = uc( s, t )
    t1, t2 = coeff4( t1, t2 )
    degree = 8
    cf = np.zeros(degree +1, dtype=np.complex128)
    for j in range(0, degree +1):
        r_part = t1.real * j**2 - t2.real * np.sqrt(j +1)
        im_part = (t1.imag + t2.imag) * np.log(j +2)
        magnitude = np.abs(t1)**(j %3 +1) + np.abs(t2)**(degree -j)
        angle = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j)
        cf[j] = (r_part +1j * im_part) * magnitude * np.exp(1j * angle)
    return _rotate_poly_safe(np.flip(cf),-np.pi/2)
  
@njit(cache=True, fastmath=True)
def poly_cf4p808(s: float, t: float) -> np.ndarray:
    t1, t2 = uc( s, t )
    t1, t2 = coeff4( t1, t2 )
    cf = np.zeros(25, dtype=np.complex128)
    for k in range(25):
        cf[k] = (k + t1) / (k + t2)
    cf[4] += np.log(np.abs(t1 + t2))
    cf[9] += np.sin(np.real(t1)) + np.cos(np.imag(t2))
    cf[14] += np.abs(cf[13]) ** 2 + np.angle(cf[12]) ** 2
    cf[19] += np.abs(np.real(t2) * np.imag(t1))
    cf[24] += np.abs(t1 + np.conj(t2))
    return _rotate_poly_safe(np.flip(cf),-np.pi/2)



@njit(cache=True, fastmath=True)
def poly_cf4p821(s: float, t: float) -> np.ndarray:
    t1, t2 = uc( s, t )
    t1, t2 = coeff4( t1, t2 )
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = 3 * t1 + 5j * t2
    for k in range(1, 25):
        mod_t1 = np.abs(t1)
        arg_t2 = np.angle(t2)
        cf[k] = cf[k-1] * (mod_t1 + arg_t2)
        if cf[k].real < 0 and cf[k].imag < 0:
            cf[k] = np.conj(cf[k])
        if np.abs(cf[k].real) > 10:
            cf[k] = cf[k] / mod_t1
        if np.abs(cf[k].imag) > 10:
            cf[k] = cf[k] / (1j * arg_t2)
    return _rotate_poly_safe(np.flip(cf),-np.pi/2)

@njit(cache=True, fastmath=True)
def poly_cf5p23(s: float, t: float) -> np.ndarray:
    t1, t2 = uc( s, t )
    t1, t2 = coeff5( t1, t2 )
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1 + t2
    cf[1] = 1 + t1 * t2 + np.log(np.abs(t1 + t2) + 1)
    cf[2] = t1 + t2 + np.log(np.abs(1 - t1 * t2) + 1)
    for i in range(3, 72):
        cf[i-1] = i * t1 + (51 - i) * t2 + np.log(np.abs(t1 - t2 * i) + 1)
    cf[10] = cf[0] + cf[9] - np.sin(t1)
    cf[20] = cf[30] + cf[40] - np.cos(t2)
    cf[30] = cf[20] + cf[40] + np.sin(t1)
    cf[40] = cf[30] + cf[20] - np.cos(t2)
    cf[50] = cf[40] + cf[20] + np.sin(t2)
    return _rotate_poly_safe(np.flip(cf),-np.pi/2)


@njit(cache=True, fastmath=True)
def poly_cf5p242(s: float, t: float) -> np.ndarray:
    t1, t2 = uc( s, t )
    t1, t2 = coeff5( t1, t2 )
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(1, 36):
        if j % 5 == 1:
            cf[j - 1] = np.sin(np.abs(t1) * j) + np.cos(np.angle(t2) * j)
        elif j % 5 == 2:
            cf[j - 1] = np.log(np.abs(t1) + 1) * t2**j
        elif j % 5 == 3:
            cf[j - 1] = np.conj(t1)**j - np.real(t2) * j
        elif j % 5 == 4:
            cf[j - 1] = np.imag(t1) + np.abs(t2) * np.sin(j * np.angle(t1))
        else:
            cf[j - 1] = t1 * t2**j + np.cos(j) - np.sin(j)
    cf[6] = 50j * t1**2 - 30j * t2 + 20
    cf[13] = 80 * t1 - 60j * t2**2 + 10
    cf[20] = 40j * t1**3 + 25 * np.conj(t2) - 15
    cf[27] = 70 * np.abs(t1) + 35j * np.angle(t2) + 5
    cf[34] = 90j * t1 * t2 - 45 * np.real(t1) + 22.5
    return _rotate_poly_safe(np.flip(cf),-np.pi/2)

@njit(cache=True, fastmath=True)
def poly_cf5p69(s: float, t: float) -> np.ndarray:
    t1, t2 = uc( s, t )
    t1, t2 = coeff5( t1, t2 )
    cf = np.zeros(71, dtype=np.complex128)
    for i in range(1, 72):
        j = 71 - i
        cf[i - 1] = ((np.real(t1) + np.imag(t1) * j) / np.abs(t2 + i)) * np.sin(np.angle(t1 + t2 * i)) + np.log(np.abs(t1 * t2) + 1) * np.cos(2 * np.pi * i / 71)
    cf[cf == 0] = np.real(t1) ** 2 - np.imag(t1) * np.imag(t2)
    return _rotate_poly_safe(np.flip(cf),-np.pi/2)
        
@njit(cache=True, fastmath=True)
def poly_cf5p6(s: float, t: float) -> np.ndarray:
    t1, t2 = uc( s, t )
    t1, t2 = coeff5( t1, t2 )
    cf = np.zeros(51, dtype=np.complex128)
    for k in range(1, 52):
        cf[k-1] = (t1 + t2) * np.sin(np.log(np.abs(t1 * t2)**k + 1)) + np.cos(np.angle(t1 * t2)**k) * np.conj(t1 - t2)
    return _rotate_poly_safe(np.flip(cf),-np.pi/2)

@njit(cache=True, fastmath=True)
def poly_cf1p530(s: float, t: float) -> np.ndarray:
    t1, t2 = uc( s, t )
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(t1.real, t2.real, num=n)
    imc = np.linspace(t1.imag, t2.imag, num=n)
    for j in range(n):
        r = rec[j]
        m = imc[j]
        term1 = np.sin(r * np.pi / (j + 2)) * np.cos(m * np.pi / (j + 3))
        term2 = np.log(np.abs(r + m) + 1) * (t1.real ** (j + 1))
        term3 = np.prod(np.array([r, m, j + 1])) ** (1 / (j + 1))
        mag = term1 + term2 + term3
        angle = np.angle(t1) * np.sin(m * np.pi / (j + 4)) + np.angle(t2) * np.cos(r * np.pi / (j + 5)) + np.log(j + 2)
        cf[j] = mag * (np.cos(angle) + 1j * np.sin(angle))
    return _rotate_poly_safe(np.flip(cf),-np.pi/2)

@njit(cache=True, fastmath=True)
def poly_cf2p105(s: float, t: float) -> np.ndarray:
    t1, t2 = uc( s, t )
    t1, t2 = coeff2( t1, t2 )
    n = 71
    cf = np.zeros(n, dtype=np.complex128)
    for k in range(1, n+1):
            cf[k - 1] = np.sin(k * (np.real(t1) * np.imag(t2))**3) + np.cos(k * np.log(np.abs(t1 * t2 + 1)) * np.angle(t1 + np.conj(t2)))
    return _rotate_poly_safe(np.flip(cf),-np.pi/2)


@njit(cache=True, fastmath=True)
def poly_cf2p112(s: float, t: float) -> np.ndarray:
    t1, t2 = uc( s, t )
    t1, t2 = coeff2( t1, t2 )
    cf = np.zeros(71, dtype=np.complex128)
    def phi_n(n):
        theta = 2 * np.pi / n
        w = np.exp(1j * theta)
        return w**np.arange(n)
    cf[0:10] = phi_n(10) + (t1**2 + np.abs(t2))
    cf[10:15] = np.exp(np.arange(2, 7) * np.angle(t1)) - np.real(t2)
    cf[15:20] = phi_n(5) * (t1 + 2j * t2)
    cf[20:30] = 100 - np.real(t1**3) + 1j * np.imag(t2**2)
    cf[30:40] = -40 + np.abs(t1 * t2) + 1j * np.angle(t1 - t2)
    cf[40:50] = 1j * phi_n(10)**(np.arange(2, 12)) - (t1 + np.conj(t2))
    cf[50:60] = 2 * np.log(np.abs(np.real(t1) + np.imag(t2)) + 1) * np.arange(2, 12)
    cf[60:70] = ((-1)**np.arange(1, 11)) * phi_n(10)**(np.arange(1, 11) * t1)
    cf[70] = np.sum(cf[0:70]) / 71
    return _rotate_poly_safe(np.flip(cf),-np.pi/2)

@njit(cache=True, fastmath=True)
def poly_cf2p114(s: float, t: float) -> np.ndarray:
    t1, t2 = uc( s, t )
    t1, t2 = coeff2( t1, t2 )
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = np.real(t1**2 + t2**3) - np.imag(t1 * np.conj(t2))
    cf[1:4] = np.array([-827, 221, 653]) * (np.real(t1) + np.imag(t2))
    cf[4] = np.abs(t1 - 2j * t2)**5
    for j in range(6, 29):
        cf[j - 1] = np.cos(j * np.angle(t1 + t2)) * np.sin(j * np.abs(t1**2 + t2)) + j
    cf[28:41] = np.array([89, -233, 144, 377, 610, -987, 1597, -2584, 4181, -6765, 10946, -17711, 28657]) * np.abs(t1 - t2)
    for k in range(42, 62):
        cf[k - 1] = np.log(np.abs(k * t1 * np.conj(t2) + 71))
    cf[62:66] = np.array([3j, 2 - 8j, -6 + 11j, -5.5]) * (t1**3 - t2**3)
    cf[66:71] = np.tan(np.pi / 4) * np.exp(-(np.arange(66, 71)))
    cf[70] = np.exp(t1) - np.exp(t2) 
    return _rotate_poly_safe(np.flip(cf),-np.pi/2)

# np.roots works, aberth not at all
@njit(cache=True, fastmath=True)
def poly_cf2p116(s: float, t: float) -> np.ndarray:
    t1, t2 = uc( s, t )
    t1, t2 = coeff2( t1, t2 )
    cf = np.zeros(71, dtype=np.complex128)
    primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47])
    for k in range(1, 36):
        cf[k - 1] = primes[k % 15] * (t1**k + t2**k) * (-1)**k / (k + 1)
        cf[70 - k] = primes[(k + 11) % 15] * (t1**(71 - k) - t2**(71 - k)) * (-1)**(71 - k) / (71 - k + 1)
    cf[35] = np.sum(primes[:5]) * np.abs(t1 + t2) / (1 + np.abs(t1))
    cf[70] = 1 + 1j
    return _rotate_poly_safe(np.flip(cf),-np.pi/2)


@njit(cache=True, fastmath=True)
def poly_cf2p243(s: float, t: float) -> np.ndarray:
    t1, t2 = uc( s, t )
    t1, t2 = coeff2( t1, t2 )
    cf = np.zeros(35, dtype=np.complex128)
    # Assign base coefficients with fixed values
    i = np.array([0, 5, 9, 14, 21, 27])
    cf[i] = np.array([2, -3 + 2j, 4.5, -5.2j, 3.3, -1.1])
    
    # Loop to assign lower degree coefficients
    for j in range(2, 6):
        cf[j - 1] = (np.real(t1)**j + np.imag(t2)**j) * np.sin(np.angle(t1) * j) / (1 + j)
    
    # Loop to assign middle degree coefficients
    for k in range(7, 15):
        cf[k - 1] = (np.abs(t1)**k * np.cos(np.angle(t2) * k)) + np.conj(t2) * np.log(np.abs(t1 * t2) + 1)
    
    # Loop to assign higher degree coefficients
    for r in range(16, 26):
        cf[r - 1] = (np.real(t1**r) - np.imag(t2**r) * 1j) * np.sin(t1 + t2) + np.cos(t1 * t2)
    
    # Assign coefficients using product and sum
    cf[25] = np.prod(np.array([np.abs(t1), np.abs(t2)])) + np.sum(np.array([np.real(t1), np.imag(t2)])) * np.conj(t1 + t2)
    cf[26] = np.log(np.abs(t1) + 1) + np.log(np.abs(t2) + 1) * 1j
    cf[28] = np.real(t1 * t2) - np.imag(t1 / t2) * 1j
    cf[29] = np.sin(t1**2) + np.cos(t2**3) * 1j
    cf[31] = np.abs(t1 + t2) * np.exp(-np.real(t1 - t2))
    cf[33] = np.angle(t1) + np.angle(t2) * 1j
    
    # Assign the last coefficient with a unique pattern
    cf[34] = (t1**3 + t2**3) / (1 + np.abs(t1) + np.abs(t2))
    return _rotate_poly_safe(np.flip(cf),-np.pi/2)

# np.roots works, aberth not at all
@njit(cache=True, fastmath=True)
def poly_cf2p273(s: float, t: float) -> np.ndarray:
    t1, t2 = uc( s, t )
    t1, t2 = coeff2( t1, t2 )
    cf = np.zeros(35, dtype=np.complex128)
    # Assign base coefficients with fixed values
    i=np.array([0, 5, 11, 17, 23, 29])
    cf[i] = [2, -3 + 1j, 4, -5j, 6 + 2j, -7]
    for j in range(2, 35):
        if cf[j] == 0:
            cf[j] = (np.real(t1)**j - np.imag(t2)**j) + (np.angle(t1) * j + np.abs(t2)) * 1j

    for k in range(3, 34):
        cf[k] += np.sin(t1 * k) * np.cos(t2 / k) + np.log(np.abs(t1) + 1) * np.sin(np.angle(t2)) * 1j

    cf[9] = np.conj(t1) * t2*t2 + np.abs(t2) * 1j
    cf[14] = np.real(t1*t1*t1) + np.imag(t2*t2*t2) * 1j
    cf[19] = np.prod(np.array([np.real(t1), np.real(t2)])) + np.prod(np.array([np.imag(t1), np.imag(t2)])) * 1j
    cf[24] = np.sum(np.array([np.abs(t1), np.abs(t2)])) + np.angle(t1 + t2) * 1j
    cf[27] = np.sin(np.abs(t1)) + np.cos(np.abs(t2)) * 1j
    cf[31] = np.log(np.abs(t1) + 1) + np.log(np.abs(t2) + 1) * 1j
    cf[34] = np.conj(t1 + t2)
    return _rotate_poly_safe(np.flip(cf),-np.pi/2)


@njit(cache=True, fastmath=True)
def poly_cf2p393(s: float, t: float) -> np.ndarray:
    t1, t2 = uc( s, t )
    t1, t2 = coeff2( t1, t2 )
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(1, 36):
        mag = 0
        angle = 0
        for k in range(1, j + 1):
            mag += np.log(np.abs(t1) + k) * np.sin(k * np.real(t2)) / (k + 1)
            angle += np.angle(t1)**k * np.cos(k * np.imag(t2))
        cf[j - 1] = mag * (np.cos(angle) + np.sin(angle) * 1j)
    return _rotate_poly_safe(np.flip(cf),-np.pi/2)

@njit(cache=True, fastmath=True)
def poly_cf2p393rev(s: float, t: float) -> np.ndarray:
    t1, t2 = uc( s, t )
    t1, t2 = coeff2( t1, t2 )
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(1, 36):
        mag = 0
        angle = 0
        for k in range(1, j + 1):
            mag += np.log(np.abs(t1) + k) * np.sin(k * np.real(t2)) / (k + 1)
            angle += np.angle(t1)**k * np.cos(k * np.imag(t2))
        cf[j - 1] = mag * (np.cos(angle) + np.sin(angle) * 1j)
    return cf

@njit(cache=True, fastmath=True)
def poly_cf2p425(s: float, t: float) -> np.ndarray:
    t1, t2 = uc( s, t )
    t1, t2 = coeff2( t1, t2 )
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec_seq = np.linspace(np.real(t1), np.real(t2), n)
    imc_seq = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(1, n +1):
        angle_part = np.sin(j * np.pi /6) * np.cos(j * np.pi /8) + np.angle(t1) * np.log(j +1)
        magnitude_part = np.log(np.abs(t1) + j**2) * np.abs(np.cos(j)) + \
                            np.log(np.abs(t2) + j) * np.abs(np.sin(j / 2))
        cf[j-1] = (magnitude_part + np.real(t1) * np.real(t2) / (j +1)) * \
                    np.exp(1j * angle_part)
        if j %5 ==0:
            cf[j-1] += np.conj(cf[j-1])
        cf[j-1] *= (1 + 0.1 * np.sin(j))
    return _rotate_poly_safe(np.flip(cf),-np.pi/2)

@njit(cache=True, fastmath=True)
def poly_cf2p472(s: float, t: float) -> np.ndarray:
    t1, t2 = uc( s, t )
    t1, t2 = coeff2( t1, t2 )
    n = 71
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        mag = np.log(np.abs(t1) + j) * (1 + np.sin(j * np.pi /7))
        ang = np.angle(t1)*np.cos(j * np.pi /5) + np.angle(t2)*np.sin(j * np.pi /3)
        cf[j-1] = mag * np.exp(1j * ang) + (t1.real + t2.real)/(j +1)
    for k in range(1, n+1):
        cf[k-1] += np.conj(t1)**k - np.conj(t2)**(n -k +1)
    for r in range(1, n+1):
        cf[r-1] *= (1 + 0.1 * np.cos(r * np.angle(t1)) * np.sin(r * np.angle(t2)))
    return _rotate_poly_safe(np.flip(cf),-np.pi/2)

@njit(cache=True, fastmath=True)
def poly_cf2p483(s: float, t: float) -> np.ndarray:
    t1, t2 = uc( s, t )
    t1, t2 = coeff2( t1, t2 )
    n =35
    cf = np.zeros(n, dtype=np.complex128)
    r1 = t1.real
    i1 = t1.imag
    r2 = t2.real
    i2 = t2.imag
    for j in range(1, n+1):
        part1 = r1**j * np.sin(j * np.angle(t2))
        part2 = i2**(n -j) * np.cos(j * np.abs(t1))
        part3 = np.log(np.abs(t1) + np.abs(t2) + j)
        part4 = np.prod(np.array([r1 + j, i2 +j, np.log(np.abs(t1)+1)]))
        magnitude = part1 * part2 + part3 * part4
        angle = np.angle(t1)*np.sin(j) + np.angle(t2)*np.cos(j) + np.log(np.abs(t1)+1)/j
        cf[j-1] = magnitude * np.exp(1j * angle)
    return _rotate_poly_safe(np.flip(cf),-np.pi/2)

@njit(cache=True, fastmath=True)
def poly_cf2p666(s: float, t: float) -> np.ndarray:
    t1, t2 = uc( s, t )
    t1, t2 = coeff2( t1, t2 )
    n = 8
    cf = np.zeros(n, dtype=np.complex128)
    rec1, imc2 = t1.real, t2.imag
    for j in range(1, n+1):
        r_part = np.log(np.abs(rec1 + j) + 1) * np.sin(j * np.pi / 4)
        i_part = np.log(np.abs(imc2 - j) + 1) * np.cos(j * np.pi / 3)
        magnitude = r_part + i_part
        angle = np.angle(t1) * np.cos(j) + np.angle(t2) * np.sin(j)
        cf[j-1] = magnitude * (np.cos(angle) + 1j * np.sin(angle)) 
    return _rotate_poly_safe(np.flip(cf),-np.pi/2)

@njit(cache=True, fastmath=True)
def poly_cf1p11(s: float, t: float) -> np.ndarray:
    t1, t2 = uc( s, t )
    n = 51
    cf = np.zeros(n, dtype=np.complex128)
    cf[0] = t1 * t2 - np.log(np.abs(t1 + t2) + 1)
    for i in range(1, n-1):
        cf[i] = np.sin(i) * (np.real(t1**i) - np.imag(t2**i)) + np.cos(i) * (np.real(t2**i) - np.imag(t1**i))
        cf[i] = cf[i] / (np.abs(cf[i]) + 1e-10)
    cf[n-1] = np.abs(t1) * np.abs(t2) * np.angle(t1 + t2)
    return _rotate_poly_safe(np.flip(cf),-np.pi/2)

@njit(cache=True, fastmath=True)
def poly_cf2p11(s: float, t: float) -> np.ndarray:
    t1, t2 = uc( s, t )
    t1, t2 = coeff2( t1, t2 )
    n = 51
    cf = np.zeros(n, dtype=np.complex128)
    cf[0] = t1 * t2 - np.log(np.abs(t1 + t2) + 1)
    for i in range(1, n-1):
        cf[i] = np.sin(i) * (np.real(t1**i) - np.imag(t2**i)) + np.cos(i) * (np.real(t2**i) - np.imag(t1**i))
        cf[i] = cf[i] / (np.abs(cf[i]) + 1e-10)
    cf[n-1] = np.abs(t1) * np.abs(t2) * np.angle(t1 + t2)
    return _rotate_poly_safe(np.flip(cf),-np.pi/2)

@njit(cache=True, fastmath=True)
def poly_cf3p11(s: float, t: float) -> np.ndarray:
    t1, t2 = uc( s, t )
    t1, t2 = coeff3( t1, t2 )
    n = 51
    cf = np.zeros(n, dtype=np.complex128)
    cf[0] = t1 * t2 - np.log(np.abs(t1 + t2) + 1)
    for i in range(1, n-1):
        cf[i] = np.sin(i) * (np.real(t1**i) - np.imag(t2**i)) + np.cos(i) * (np.real(t2**i) - np.imag(t1**i))
        cf[i] = cf[i] / (np.abs(cf[i]) + 1e-10)
    cf[n-1] = np.abs(t1) * np.abs(t2) * np.angle(t1 + t2)
    return _rotate_poly_safe(np.flip(cf),-np.pi/2)

# =======================
# Rasterizers
# =======================

def view(roots_mat: np.ndarray, args):

    if args.view is not None:
        ll, ur = ast.literal_eval(args.view)
        llx, lly, ury, urx = ll.real, ll.imag, ur.real, ur.imag
        return llx, lly, urx, ury

    zs = roots_mat[:, 1:].ravel()
    if zs.size == 0: return -1.0, -1.0, 1.0, 1.0

    re = zs.real
    im = zs.imag
    rx = re.max() - re.min() if re.size else 1.0
    ry = im.max() - im.min() if im.size else 1.0
    pad_x = args.pad * (rx if rx > 0 else 1.0)
    pad_y = args.pad * (ry if ry > 0 else 1.0)

    llx = re.min() - pad_x
    urx = re.max() + pad_x
    lly = im.min() - pad_y
    ury = im.max() + pad_y

    return llx, lly, urx, ury


def _raster_worker(args):
    (rid, rows_range, shm_roots_name, N, M,
     llx, lly, urx, ury, pixels,
     shm_img_name) = args

    # attach to shared buffers
    shm_roots, R = get_shm(shm_roots_name, N, M, np.complex128)
    shm_img, IMG = get_shm(shm_img_name, pixels, pixels, np.uint8)

    IMG1D = IMG.reshape(-1)

    span_x = urx - llx
    span_y = ury - lly
    if span_x <= 0.0: span_x = 1e-10
    if span_y <= 0.0: span_y = 1e-10
    sx = float(pixels) / span_x
    sy = float(pixels) / span_y

    i0, i1 = rows_range
    step = 1_000_000  # tune

    for a in range(i0, i1, step):
        b = min(a + step, i1)
        Z = R[a:b, 1:].reshape(-1)

        # use float64 for precise mapping at huge resolutions
        xr = Z.real.astype(np.float64, copy=False)
        yi = Z.imag.astype(np.float64, copy=False)
        msk = np.isfinite(xr) & np.isfinite(yi)
        if not msk.any():
            continue
        xr = xr[msk]; yi = yi[msk]

        # 2) cull anything outside the viewport BEFORE scaling
        in_view = (xr >= llx) & (xr < urx) & (yi >= lly) & (yi < ury)
        if not in_view.any():
            continue
        xr = xr[in_view]; yi = yi[in_view]

        # 3) map to pixel space (floats)
        xpf = (xr - llx) * sx
        ypf = (yi - lly) * sy

        # 4) guard against NaN/Inf produced by the multiply
        fOK = np.isfinite(xpf) & np.isfinite(ypf)
        if not fOK.any():
            continue
        xpf = xpf[fOK]; ypf = ypf[fOK]


        # 5) floor then cast to int64
        ix = np.floor(xpf).astype(np.int64, copy=False)
        iy = np.floor(ypf).astype(np.int64, copy=False)

        # 6) final in-bounds mask (defensive)
        inb = (ix >= 0) & (ix < pixels) & (iy >= 0) & (iy < pixels)
        if not inb.any():
            continue
        ix = ix[inb]; iy = iy[inb]


        # bins in 64-bit; also cast to intp for indexing
        bins = (iy * np.int64(pixels) + ix).astype(np.intp, copy=False)

        # de-duplicate to reduce write pressure
        ubins = np.unique(bins)
        IMG1D[ubins] = 1  # multiple writes of 1 are benign

    shm_roots.close()
    shm_img.close()

def rasterize_points(
    roots_mat: np.ndarray,
    llx: float, lly: float, urx: float, ury: float,
    pixels: int,
    nprocs: int | None = None
) -> np.ndarray:
    N, M = roots_mat.shape
    if N == 0 or M <= 1:
        return np.zeros((pixels, pixels), dtype=np.uint8)

    if nprocs is None:
        nprocs = mproc.cpu_count()

    # Share roots
    shm_roots, Rsh = make_shm(N, M, np.complex128)
    Rsh[:] = roots_mat  # one-time copy into shared segment

    # Shared output image (uint8). We'll flip vertically at the end.
    shm_img, IMG = make_shm(pixels, pixels, np.uint8)  # zero-initialized

    # Row ranges per worker
    rows_per = (N + nprocs - 1) // nprocs
    args = []
    for p in range(nprocs):
        i0 = p * rows_per
        i1 = min((p + 1) * rows_per, N)
        if i0 >= i1:
            continue
        args.append((
            p, (i0, i1), shm_roots.name, N, M,
            llx, lly, urx, ury, pixels,
            shm_img.name
        ))

    ctx = mproc.get_context("spawn")
    with ctx.Pool(processes=len(args)) as pool:
        pool.map(_raster_worker, args)

    # copy result out and clean up
    out = np.array(IMG[::-1, :], copy=True)  # flip vertically
    shm_img.close(); shm_img.unlink()
    shm_roots.close(); shm_roots.unlink()
    return (out * 255).astype(np.uint8)

#---------- Hue utilities ----------

def hue_xnorm(xr: np.ndarray, yi: np.ndarray, llx, lly, urx, ury) -> np.ndarray:
    """Hue = normalized x in [llx,urx] -> [0,1]."""
    span_x = max(urx - llx, 1e-12)
    H = (xr - llx) / span_x
    return np.clip(H, 0.0, 1.0).astype(np.float32, copy=False)

def hue_angle(xr: np.ndarray, yi: np.ndarray, llx, lly, urx, ury) -> np.ndarray:
    """Hue = angle around the bbox center mapped to [0,1]."""
    cx = 0.5*(llx + urx)
    cy = 0.5*(lly + ury)
    theta = np.arctan2(yi - cy, xr - cx)  # [-pi, pi]
    H = (theta + np.pi) / (2.0*np.pi)
    return H.astype(np.float32, copy=False)

def resolve_hue_fn(name: str):
    """Top-level resolver (works with multiprocessing if you later choose)."""
    table = {
        "x": hue_xnorm,
        "xnorm": hue_xnorm,
        "angle": hue_angle,
    }
    if name is None: return hue_xnorm
    fn = table.get(name.lower())
    if fn is None:
        raise ValueError(f"Unknown --hue-fn '{name}'. Available: {', '.join(table)}")
    return fn

def _hsv_to_rgb(h, s, v):
    """
    Vectorized HSV->RGB for h,s,v in [0,1]; returns (r,g,b) each in [0,1].
    No seaborn/mpl deps, pure numpy.
    """
    h = np.mod(h, 1.0)
    i = np.floor(h * 6.0).astype(np.int64)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    i_mod = i % 6
    r = np.choose(i_mod, [v, q, p, p, t, v])
    g = np.choose(i_mod, [t, v, v, q, p, p])
    b = np.choose(i_mod, [p, p, t, v, v, q])
    return r, g, b

# ---------- Single-process color rasterizer (safe & fast via bincount) ----------

def rasterize_hsv_color(
    roots_mat: np.ndarray,
    llx: float, lly: float, urx: float, ury: float,
    pixels: int,
    *,
    hue_fn_name: str = "xnorm",
    circular: bool = False,
) -> np.ndarray:
    """
    Returns RGB uint8 image (H from hue_fn, S=1,V=1 on hit; else black).
    Single-process, vectorized with bincount to avoid race artifacts.
    """
    N, M = roots_mat.shape
    if N == 0 or M <= 1:
        return np.zeros((pixels, pixels, 3), dtype=np.uint8)

    span_x = max(urx - llx, 1e-12)
    span_y = max(ury - lly, 1e-12)
    sx = float(pixels) / span_x
    sy = float(pixels) / span_y

    Z = roots_mat[:, 1:].reshape(-1)
    xr = Z.real.astype(np.float64, copy=False)
    yi = Z.imag.astype(np.float64, copy=False)

    # keep only finite and within view
    m = np.isfinite(xr) & np.isfinite(yi) & (xr >= llx) & (xr < urx) & (yi >= lly) & (yi < ury)
    if not m.any():
        return np.zeros((pixels, pixels, 3), dtype=np.uint8)

    xr = xr[m]; yi = yi[m]

    # pixel bins (ix,iy) -> flat index
    ix = np.floor((xr - llx) * sx).astype(np.int64, copy=False)
    iy = np.floor((yi - lly) * sy).astype(np.int64, copy=False)
    inb = (ix >= 0) & (ix < pixels) & (iy >= 0) & (iy < pixels)
    if not inb.any():
        return np.zeros((pixels, pixels, 3), dtype=np.uint8)
    ix = ix[inb]; iy = iy[inb]
    bins = (iy * pixels + ix).astype(np.int64, copy=False)

    # hue values via pluggable function
    hue_fn = resolve_hue_fn(hue_fn_name)
    H = hue_fn(xr[inb], yi[inb], llx, lly, urx, ury)  # float32

    n_pix = pixels * pixels
    if circular:
        # circular mean: avg of exp(i*2πH)
        theta = 2.0 * np.pi * H
        wcos = np.cos(theta).astype(np.float32, copy=False)
        wsin = np.sin(theta).astype(np.float32, copy=False)
        sum_cos = np.bincount(bins, weights=wcos, minlength=n_pix).astype(np.float32, copy=False)
        sum_sin = np.bincount(bins, weights=wsin, minlength=n_pix).astype(np.float32, copy=False)
        cnt     = np.bincount(bins, minlength=n_pix).astype(np.float32, copy=False)

        # avoid zero division; compute angle of resultant vector
        ok = cnt > 0
        Havg = np.zeros_like(sum_cos, dtype=np.float32)
        Havg[ok] = (np.arctan2(sum_sin[ok], sum_cos[ok]) + np.pi) / (2.0 * np.pi)
    else:
        # linear mean
        hsum = np.bincount(bins, weights=H, minlength=n_pix).astype(np.float32, copy=False)
        cnt  = np.bincount(bins, minlength=n_pix).astype(np.float32, copy=False)
        ok = cnt > 0
        Havg = np.zeros_like(hsum, dtype=np.float32)
        Havg[ok] = hsum[ok] / cnt[ok]

    # S,V: 1 on hit, else 0
    S = (cnt > 0).astype(np.float32)
    V = S

    # HSV -> RGB (vectorized)
    h = Havg
    i = np.floor(h * 6.0).astype(np.int64)
    f = (h * 6.0) - i
    p = V * (1.0 - S)
    q = V * (1.0 - f * S)
    t = V * (1.0 - (1.0 - f) * S)
    i_mod = i % 6
    r = np.choose(i_mod, [V, q, p, p, t, V])
    g = np.choose(i_mod, [t, V, V, q, p, p])
    b = np.choose(i_mod, [p, p, t, V, V, q])

    RGB = np.stack([
        (r.reshape(pixels, pixels)[::-1, :] * 255.0 + 0.5).astype(np.uint8),
        (g.reshape(pixels, pixels)[::-1, :] * 255.0 + 0.5).astype(np.uint8),
        (b.reshape(pixels, pixels)[::-1, :] * 255.0 + 0.5).astype(np.uint8),
    ], axis=-1)
    return RGB


def _raster_hue_worker(args):
    """
    Stripe-owned worker: each worker updates only its [x0:x1) columns so there are no write races.
    """
    (wid, rows_range, shm_roots_name, N, M,
     llx, lly, urx, ury, pixels,
     x0, x1,
     shm_sum_name, shm_cnt_name) = args

    # attach to shared buffers
    shm_roots, R = get_shm(shm_roots_name, N, M, np.complex128)
    shm_sum, SUM = get_shm(shm_sum_name, pixels, pixels, np.float32)
    shm_cnt, CNT = get_shm(shm_cnt_name, pixels, pixels, np.uint32)

    SUM1D = SUM.reshape(-1)
    CNT1D = CNT.reshape(-1)

    span_x = urx - llx
    span_y = ury - lly
    if span_x <= 0.0: span_x = 1e-10
    if span_y <= 0.0: span_y = 1e-10
    sx = float(pixels) / span_x
    sy = float(pixels) / span_y

    # precompute stripe mask for quick filtering
    i0, i1 = rows_range
    step = 1_000_000  # tune for memory

    for a in range(i0, i1, step):
        b = min(a + step, i1)
        Z = R[a:b, 1:].reshape(-1)
        xr = Z.real.astype(np.float64, copy=False)
        yi = Z.imag.astype(np.float64, copy=False)
        msk = np.isfinite(xr) & np.isfinite(yi)
        if not msk.any():
            continue
        xr = xr[msk]; yi = yi[msk]

        in_view = (xr >= llx) & (xr < urx) & (yi >= lly) & (yi < ury)
        if not in_view.any():
            continue
        xr = xr[in_view]; yi = yi[in_view]

        # map to pixel coords
        xpf = (xr - llx) * sx
        ypf = (yi - lly) * sy
        fOK = np.isfinite(xpf) & np.isfinite(ypf)
        if not fOK.any():
            continue
        xpf = xpf[fOK]; ypf = ypf[fOK]

        ix = np.floor(xpf).astype(np.int64, copy=False)
        iy = np.floor(ypf).astype(np.int64, copy=False)

        # stripe filter (only update our columns)
        inb = (ix >= x0) & (ix < x1) & (iy >= 0) & (iy < pixels)
        if not inb.any():
            continue
        ix = ix[inb]; iy = iy[inb]

        # hue per point from bbox-normalized x
        H = (xr[fOK][inb] - llx) / span_x
        # de-duplicate identical bins? No — we need proper averaging even with duplicates.
        # We'll aggregate by bin index using np.add.at on *our* own stripe columns (no cross-worker collisions).
        bins = (iy * np.int64(pixels) + ix).astype(np.intp, copy=False)

        # accumulate hue sum and hit count
        np.add.at(SUM1D, bins, H.astype(np.float32, copy=False))
        np.add.at(CNT1D, bins, 1)

    shm_roots.close()
    shm_sum.close()
    shm_cnt.close()


def rasterize_hsv_hue_average(
    roots_mat: np.ndarray,
    llx: float, lly: float, urx: float, ury: float,
    pixels: int,
    nprocs: int | None = None
) -> np.ndarray:
    """
    Returns an RGB uint8 image (pixels x pixels x 3).
    Hue = normalized x-position in [llx,urx].
    S=1,V=1 for pixels hit by ≥1 root; else V=0 (black).
    Averages hue per pixel.
    """
    N, M = roots_mat.shape
    if N == 0 or M <= 1:
        return np.zeros((pixels, pixels, 3), dtype=np.uint8)

    if nprocs is None:
        nprocs = mproc.cpu_count()

    # Share roots
    shm_roots, Rsh = make_shm(N, M, np.complex128)
    Rsh[:] = roots_mat

    # Shared accumulators (owned in stripes → no write races)
    shm_sum, SUM = make_shm(pixels, pixels, np.float32)   # hue sum
    shm_cnt, CNT = make_shm(pixels, pixels, np.uint32)    # hit count

    # Row split same as before
    rows_per = (N + nprocs - 1) // nprocs

    # Column stripes: nprocs stripes across X
    # (you can cap stripes vs procs separately, but this is simple and works)
    stripes = nprocs
    col_per = (pixels + stripes - 1) // stripes

    args = []
    for p in range(nprocs):
        # rows block p:
        i0 = p * rows_per
        i1 = min((p + 1) * rows_per, N)
        if i0 >= i1:
            continue
        # stripe p:
        x0 = p * col_per
        x1 = min((p + 1) * col_per, pixels)
        if x0 >= x1:
            x0, x1 = 0, pixels  # fallback: cover full width if stripes < procs

        args.append((
            p, (i0, i1), shm_roots.name, N, M,
            llx, lly, urx, ury, pixels,
            x0, x1,
            shm_sum.name, shm_cnt.name
        ))

    ctx = mproc.get_context("spawn")
    with ctx.Pool(processes=len(args)) as pool:
        pool.map(_raster_hue_worker, args)

    # finalize image (flip vertically to match your grayscale path)
    SUMa = np.array(SUM[::-1, :], copy=True)
    CNTa = np.array(CNT[::-1, :], copy=True)

    # avoid divide-by-zero
    hit = CNTa > 0
    H = np.zeros_like(SUMa, dtype=np.float32)
    H[hit] = SUMa[hit] / CNTa[hit]  # average hue in [0,1]
    S = np.zeros_like(H, dtype=np.float32)
    V = np.zeros_like(H, dtype=np.float32)
    S[hit] = 1.0
    V[hit] = 1.0

    r, g, b = _hsv_to_rgb(H, S, V)
    RGB = np.stack([
        (r * 255.0 + 0.5).astype(np.uint8),
        (g * 255.0 + 0.5).astype(np.uint8),
        (b * 255.0 + 0.5).astype(np.uint8),
    ], axis=-1)

    # cleanup shared
    shm_sum.close(); shm_sum.unlink()
    shm_cnt.close(); shm_cnt.unlink()
    shm_roots.close(); shm_roots.unlink()

    return RGB

# =======================
# CLI
# =======================

def resolve_poly(name: str):
    """
    Fetch a polynomial coeff generator by name from this module's globals.
    Accepts case-insensitive names. Raises ValueError if not found/callable.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("Empty --poly name")

    # exact first
    cand = globals().get(name)
    if cand is None:
        # case-insensitive fallback
        lname = name.lower()
        for k, v in globals().items():
            if k.lower() == lname:
                cand = v
                break

    if cand is None or not callable(cand):
        # helpful list for the error
        avail = sorted(
            k for k, v in globals().items()
            if callable(v) and k[0].isalpha()
        )
        raise ValueError(f"Unknown --poly '{name}'. Available: {', '.join(avail)}")

    return cand

if __name__ == "__main__":
    import argparse
    from PIL import Image

    ap = argparse.ArgumentParser(description="Tile-only Aberth manifold scan (monomial).")
    ap.add_argument("--poly", type=str, default="p7f",help="Name of polynomial coeff generator")
    ap.add_argument("--pps", type=int, default=1_000, help="Points per side")
    ap.add_argument("--tol", type=float, default=1e-12, help="Convergence tolerance")
    ap.add_argument("--max-iters", type=int, default=100, help="Maximum Aberth iterations per point")
    ap.add_argument("--per-root-tol", action="store_true", help="Use per-root relative step tolerance")
    ap.add_argument("--newton-fallback", action="store_true", help="Damped Newton fallback if p'(z) ~ 0")
    ap.add_argument("--verbose", action="store_true")

    # PNG
    ap.add_argument("--png", type=str, default=None, help="Output PNG path (grayscale)")
    ap.add_argument("--pixels", type=int, default=4096)
    ap.add_argument("--view", type=str, default=None, help="View")
    ap.add_argument("--pad", type=float, default=0.05)
    ap.add_argument("--png-hsv", type=str, default=None, help="Color PNG (HSV hue average)")
    ap.add_argument("--hue-fn", type=str, default="xnorm", help="Hue function: xnorm | angle | <your_fn>")
    ap.add_argument("--hue-circular", action="store_true",help="Average hue on the unit circle (use for angle-like hues)")

    args = ap.parse_args()

    poly_func = resolve_poly(args.poly)
    N = args.pps * args.pps
    t0 = time.perf_counter()
    roots_mat = scan(
        poly_func, N,
        tol = args.tol,
        max_iters = args.max_iters,
        per_root_tol = args.per_root_tol,
        newton_fallback = args.newton_fallback,
        verbose = args.verbose
    )
    print(f"scan time: {time.perf_counter() - t0:.3f}s")
    t0 = time.perf_counter()
    # roots_mat shape: (M, 1+deg) complex128, col0 = s+1j*t
    print(f"rows: {roots_mat.shape[0]:,} cols: {roots_mat.shape[1]:,}")
    print(f"roots: {roots_mat.shape[0]*roots_mat.shape[1]:,}")

    # stats: parameters and roots
    st = roots_mat[:, 0]
    s = st.real
    t = st.imag

    zs = roots_mat[:, 1:].ravel()
    re = np.real(zs)
    im = np.imag(zs)

    if zs.size > 0:
        print(f"len real: {len(re):,} len imag: {len(im):,}")
        print(f"nan real: {np.isnan(re).sum():.2f} nan imag: {np.isnan(im).sum():.2f}")
        print(f"min real: {np.nanmin(re):.2f} min imag: {np.nanmin(im):.2f}")
        print(f"max real: {np.nanmax(re):.2f} max imag: {np.nanmax(im):.2f}")
        print(f"mean real: {np.nanmean(re):.2f} mean imag: {np.nanmean(im):.2f}")
        print(f"median real: {np.nanmedian(re):.2f} mean imag: {np.nanmedian(im):.2f}")
        print(f"q25 real: {np.nanquantile(re,0.25):.2f} q25 imag: {np.nanquantile(im,0.25):.2f}")
        print(f"q75 real: {np.nanquantile(re,0.75):.2f} q75 imag: {np.nanquantile(im,0.75):.2f}")
    else:
        print("no roots in output")

    if st.size > 0:
        print(f"min s: {np.nanmin(s):.2f} min t: {np.nanmin(t):.2f}")
        print(f"max s: {np.nanmax(s):.2f} max t: {np.nanmax(t):.2f}")

    print(f"stats time: {time.perf_counter() - t0:.3f}s")
    if args.png is not None:
        t0 = time.perf_counter()
        llx, lly, urx, ury = view(roots_mat, args)
        print(f"view time: {time.perf_counter() - t0:.3f}s")
        print(f"ll: ({llx:.2f},{lly:.2f}) ur: ({urx:.2f},{ury:.2f})")
        t0 = time.perf_counter()
        img_arr = rasterize_points(roots_mat, llx, lly, urx, ury, args.pixels)
        print(f"rasterize time: {time.perf_counter() - t0:.3f}s")
        print(f"sum nz all: {np.count_nonzero(img_arr):,}  rows: {img_arr.shape[0]:,} cols: {img_arr.shape[1]:,}")
        t0 = time.perf_counter()
        img = vips.Image.new_from_memory(img_arr.data, args.pixels, args.pixels, 1, "uchar")
        img.pngsave(args.png, compression=1, effort=1, filter="none", interlace=False, strip=True, bitdepth=1)
        print(f"image save time: {time.perf_counter() - t0:.3f}s")
        print(f"Saved PNG to {args.png} [{args.pixels}x{args.pixels}]")
    if args.png_hsv is not None:
        t0 = time.perf_counter()
        llx, lly, urx, ury = view(roots_mat, args)
        print(f"[HSV] view time: {time.perf_counter() - t0:.3f}s")
        print(f"[HSV] ll: ({llx:.3f},{lly:.3f}) ur: ({urx:.3f},{ury:.3f})")

        t0 = time.perf_counter()
        rgb = rasterize_hsv_color(
            roots_mat, llx, lly, urx, ury, args.pixels,
            hue_fn_name=args.hue_fn,
            circular=args.hue_circular
        )
        print(f"[HSV] rasterize time: {time.perf_counter() - t0:.3f}s")
        print(f"[HSV] colored pixels: {np.count_nonzero(rgb.any(axis=-1)):,}")

        t0 = time.perf_counter()
        img = vips.Image.new_from_memory(rgb.data, args.pixels, args.pixels, 3, "uchar")
        img.pngsave(args.png_hsv, compression=1, effort=1, interlace=False, strip=True)
        print(f"[HSV] image save time: {time.perf_counter() - t0:.3f}s")
        print(f"Saved color PNG to {args.png_hsv} [{args.pixels}x{args.pixels}]")


