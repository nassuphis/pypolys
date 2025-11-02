#!/usr/bin/env python
# Symmetra Capital Galaxy Logo

import argparse
import numpy as np
import pyvips as vips
from numba import njit, prange
from numba import float64, int64, int32
import sys
import re
import math
import expandspec
import specparser
import time

# --- global render options ---
GROUP_METHOD = "auto"   # "auto" | "sort" | "count"
STAMP_CHUNK = int(32_768)  # top-level constant
np.random.seed(0)

LOGO_KEYS_ORDER = ["N","rng","dth","tda","tdw","tdt","arm","swa","swb","sqs","frt","mrg","pix","drt","lmu","lsig","fos","min"]

LOGO_DEFAULTS = {
    "N":   50_000,
    "rng": 1,
    "dth": 1.0,
    "tda": 1.0,
    "tdw": 0.1,
    "tdt": 1.0,
    "arm": 2,
    "swa": -0.33,
    "swb": 2.0,
    "sqs": 0.5,
    "frt": -0.1,
    "mrg": 0.10,
    "pix": 25_000,
    # dot-size distribution
    "fos": 1e-5,      # baseline fraction of span for size 1.0
    "drt": 3000,      # cap multiplier on fos
    "lmu":  0.0,
    "lsig": 1.125,
    # visibility threshold (integer pixel radius)
    "min":  1,
}

def _cast_like(default_val, s: str):
    if isinstance(default_val, int):
        # ints may appear as floats in spec (e.g., "25000.0") – round safely
        return int(round(float(s)))
    elif isinstance(default_val, float):
        return float(s)
    else:
        return s  # fallback (not used here)

def parse_logo_spec(spec: str, defaults: dict) -> dict:
    """
    Parse 'k:v' comma-separated string into a dict, casting to the type of defaults[k].
    Unknown keys are ignored (but warned).
    """
    out = dict(defaults)
    if not spec:
        return out
    # allow spaces; split on commas not inside braces (future-proof)
    parts = [p.strip() for p in re.split(r",(?![^{}]*\})", spec) if p.strip()]
    for part in parts:
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip()
        v = v.strip()
        if k in out:
            try:
                out[k] = _cast_like(defaults[k], v)
            except Exception:
                print(f"[warn] could not parse {k}:{v}, keeping default {defaults[k]}", file=sys.stderr)
        else:
            print(f"[warn] unknown key '{k}' ignored", file=sys.stderr)
    return out

def logo_dict_to_string(d: dict) -> str:
    """Compact pretty-printer for spec dictionaries."""
    parts = []
    for k in LOGO_KEYS_ORDER:
        v = d[k]
        # integers
        if isinstance(v, int):
            if abs(v) < 100:
                s = str(v)
            else:
                s = f"{v:.0e}".replace("+0", "").replace("+", "")  # 50000 -> 5e4
        # floats
        elif isinstance(v, float):
            s = f"{v:.3g}"        # 3 significant digits, auto-switch to sci
        else:
            s = str(v)
        parts.append(f"{k}:{s}")
    return ",".join(parts)

# ---------- JIT kernels for build_logo ----------

@njit(cache=True, nogil=True)
def _smoothstep_scalar(x: float, w: float) -> float:
    t = (x + w) / (2.0 * w)
    if t < 0.0: t = 0.0
    elif t > 1.0: t = 1.0
    return t * t * (3.0 - 2.0 * t)

@njit(cache=True, nogil=True)
def _teardrop_inplace(x, y, a, w, tail):
    n = x.size
    for i in range(n):
        xi = x[i]; yi = y[i]
        y_pow = yi ** a if yi >= 0.0 else -((-yi) ** a)
        b = _smoothstep_scalar(xi, w)
        y[i] = b * yi + (1.0 - b) * y_pow
        x[i] = xi * (1.0 + tail * (1.0 - b))

@njit(cache=True, nogil=True)
def _swirl_inplace(x, y, swa, swb):
    n = x.size
    # serial rmax (cheap, deterministic)
    rmax = 0.0
    for i in range(n):
        r = (x[i]*x[i] + y[i]*y[i])**0.5
        if r > rmax: rmax = r
    denom = rmax - 1.0
    if denom < 1e-9: denom = 1e-9

    two_pi = 6.283185307179586
    for i in range(n):
        xi = x[i]; yi = y[i]
        r = (xi*xi + yi*yi)**0.5
        if r > 1.0:
            t = (r - 1.0) / denom
            phi = two_pi * swa * (t ** swb)
            c = np.cos(phi); s = np.sin(phi)
            x[i] = xi * c - yi * s
            y[i] = xi * s + yi * c

# -------- fast kernels (parallel + fastmath) --------

@njit(cache=True, nogil=True, parallel=True, fastmath=True)
def _teardrop_inplace_fast(x, y, a, w, tail):
    n = x.size
    for i in prange(n):
        xi = x[i]; yi = y[i]
        y_pow = yi ** a if yi >= 0.0 else -((-yi) ** a)
        # inline smoothstep
        t = (xi + w) / (2.0 * w)
        if t < 0.0: t = 0.0
        elif t > 1.0: t = 1.0
        s = t * t * (3.0 - 2.0 * t)
        y[i] = s * yi + (1.0 - s) * y_pow
        x[i] = xi * (1.0 + tail * (1.0 - s))

@njit(cache=True, nogil=True, fastmath=True)
def _rmax_serial(x, y):
    rmax = 0.0
    for i in range(x.size):
        r = (x[i]*x[i] + y[i]*y[i])**0.5
        if r > rmax: rmax = r
    return rmax

@njit(cache=True, nogil=True, parallel=True, fastmath=True)
def _swirl_inplace_fast(x, y, swa, swb):
    # keep rmax serial for determinism & simplicity
    rmax = _rmax_serial(x, y)
    denom = rmax - 1.0
    if denom < 1e-9: denom = 1e-9
    two_pi = 6.283185307179586
    n = x.size
    for i in prange(n):
        xi = x[i]; yi = y[i]
        r = (xi*xi + yi*yi)**0.5
        if r > 1.0:
            t = (r - 1.0) / denom
            # pow/cos/sin are vectorized-friendly with fastmath
            phi = two_pi * swa * (t ** swb)
            c = np.cos(phi); s = np.sin(phi)
            x[i] = xi * c - yi * s
            y[i] = xi * s + yi * c

@njit(cache=True, nogil=True, fastmath=True)
def apply_n_arms(x, y, m):
    n = x.size
    arm = np.mod(np.arange(n), m)
    angles = (2.0*np.pi/m) * np.arange(m)
    c = np.cos(angles)
    s = np.sin(angles)
    # rotate each point by its arm angle
    xr = x * c[arm] - y * s[arm]
    yr = x * s[arm] + y * c[arm]
    return xr, yr

# ---------- helpers ----------

def smoothstep(x, w):
    t = np.clip((x + w) / (2.0 * w), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def ddth(n, a):
    r = np.random.random(n) ** a
    th = 2.0 * np.pi * np.random.random(n)
    return r * np.exp(1j * th)

def teardrop(z, a=2.0, w=0.05, tail=0.0):
    x = z.real
    y = z.imag
    y_pow = np.sign(y) * (np.abs(y) ** a)
    b = smoothstep(x, w)
    y_new = b * y + (1.0 - b) * y_pow
    x_new = x * (1.0 + tail * (1.0 - b))
    return x_new + 1j * y_new

def swirl(z, a=0.5, b=1.0):
    r = np.abs(z)
    rmax = r.max() if r.size else 1.0
    denom = max(rmax - 1.0, 1e-9)
    t = np.where(r > 1.0, (r - 1.0) / denom, 0.0)
    return z * np.exp(1j * 2.0 * np.pi * a * (t ** b))

def squish(z, factor=0.5):
    return z.real + 1j * (z.imag * factor)

def make_disc_offsets(r):
    r = int(max(1, r))
    yy, xx = np.mgrid[-r:r+1, -r:r+1]
    mask = (xx*xx + yy*yy) <= r*r
    return yy[mask].astype(np.int32), xx[mask].astype(np.int32)

# ---------- process points for stamping ----------

@njit(cache=True, nogil=True)
def bucket_by_radius(r_px: np.ndarray, r_min: int, r_max: int):
    """
    Group integer radii in [r_min, r_max] into contiguous buckets (ascending).

    Returns:
      order   : int64[kept]  permutation indices (grouped by radius, stable)
      r_vals  : int32[k]     unique radii present (ascending)
      starts  : int64[k]     start offset in 'order' for each r in r_vals
      counts  : int64[k]     counts per r in r_vals
    """
    n = r_px.size
    if n == 0 or r_min > r_max:
        return (np.empty(0, np.int64),
                np.empty(0, np.int32),
                np.empty(0, np.int64),
                np.empty(0, np.int64))

    # 1) histogram for r in [r_min, r_max]
    size = r_max + 1  # we index directly by radius value
    counts_full = np.zeros(size, np.int64)
    kept = 0
    for i in range(n):
        r = r_px[i]
        if r_min <= r <= r_max:
            counts_full[r] += 1
            kept += 1

    if kept == 0:
        return (np.empty(0, np.int64),
                np.empty(0, np.int32),
                np.empty(0, np.int64),
                np.empty(0, np.int64))

    # 2) exclusive prefix sum only over [r_min..r_max]
    starts_full = np.zeros(size, np.int64)
    s = 0
    for r in range(r_min, r_max + 1):
        c = counts_full[r]
        starts_full[r] = s
        s += c
    # s == kept

    # 3) scatter indices into 'order' (stable)
    order = np.empty(kept, np.int64)
    write_ptr = starts_full.copy()
    for i in range(n):
        r = r_px[i]
        if r_min <= r <= r_max:
            p = write_ptr[r]
            order[p] = i
            write_ptr[r] = p + 1

    # 4) compact present radii + their starts/counts
    #    (k is usually small, e.g., <= 100)
    k = 0
    for r in range(r_min, r_max + 1):
        if counts_full[r] > 0:
            k += 1

    r_vals  = np.empty(k, np.int32)
    starts  = np.empty(k, np.int64)
    counts  = np.empty(k, np.int64)

    pos = 0
    for r in range(r_min, r_max + 1):
        c = counts_full[r]
        if c > 0:
            r_vals[pos] = np.int32(r)
            starts[pos] = starts_full[r]
            counts[pos] = c
            pos += 1

    return order, r_vals, starts, counts

# ---------- numba stamping kernel ----------

@njit(parallel=True, fastmath=True, cache=True)
def stamp_points(canvas, ys, xs, dy, dx, H, W):
    n = ys.shape[0]
    k = dy.shape[0]
    for i in prange(n):
        y0 = ys[i]
        x0 = xs[i]
        for j in range(k):
            y = y0 + dy[j]
            x = x0 + dx[j]
            if 0 <= y < H and 0 <= x < W:
                canvas[y, x] = 255

# ---------- main pipeline ----------

def build_logo_old(d: dict):
    np.random.seed(0)

    disk = ddth(d["N"], d["dth"])
    td1  = teardrop(disk, a=d["tda"], w=d["tdw"], tail=d["tdt"])
    td2  = td1 * np.exp(1j * 2.0 * np.pi * 0.5)
    td   = np.concatenate([td1, td2])
    logo = swirl(td, a=d["swa"], b=d["swb"])
    logo = squish(logo, d["sqs"])
    logo = logo * np.exp(1j * 2.0 * np.pi * d["frt"])

    # --- lognormal multipliers of fos ---
    z = np.random.normal(0.0, 1.0, size=logo.size)
    mult = np.exp(d["lmu"] + d["lsig"] * z).astype(np.float32)
    np.clip(mult, 1.0, float(d["drt"]), out=mult)   # multiplier of fos

    return logo, mult

def random_points_in_triangle(N, dth):
    """
    Uniform random points inside triangle ABC.
    A, B, C are (x,y) pairs or 2-element arrays.
    Returns (x, y) arrays of length N.
    """
    h = np.sqrt(3) / 2
    A = (-1.0, -h)
    B = ( 1.0, -h)
    C = ( 0.0,  h)
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)

    u = np.random.random(N)
    v = np.random.random(N)

    # reflect across diagonal for u+v>1
    mask = (u + v) > 1.0
    u[mask] = 1.0 - u[mask]
    v[mask] = 1.0 - v[mask]

    # affine combination
    x = A[0] + u * (B[0] - A[0]) + v * (C[0] - A[0])
    y = A[1] + u * (B[1] - A[1]) + v * (C[1] - A[1])
    x = np.sign(x)*np.abs(x)**(2*dth)
    y = np.sign(y)*np.abs(y)**(2*dth)
    return x, y

STRICT_GEOM = False   # <- set True for bit-stable, False for faster

def build_logo(d: dict):
    N   = int(d["N"])
    rng = int(d["rng"])
    dth = float(d["dth"])
    tda = float(d["tda"])
    tdw = float(d["tdw"])
    tdt = float(d["tdt"])
    arm = int(d["arm"])
    swa = float(d["swa"])
    swb = float(d["swb"])
    sqs = float(d["sqs"])
    frt = float(d["frt"])
    lmu = float(d["lmu"])
    lsig= float(d["lsig"])
    drt = float(d["drt"])

    # RNG order preserved
    if rng==1:
        u1 = np.random.random(N)
        u2 = np.random.random(N)
        r  = u1 ** dth
        th = 2.0 * np.pi * u2
        c = np.cos(th); s = np.sin(th)
        x = r * c; y = r * s
    elif rng==2:
        x = np.random.uniform(-1.0, 1.0, size=N).astype(np.float64, copy=False)
        y = np.random.uniform(-1.0, 1.0, size=N).astype(np.float64, copy=False)
        x = np.sign(x)*np.abs(x)**(2*dth)
        y = np.sign(y)*np.abs(y)**(2*dth)
    elif rng == 3:
        # triangle: equilateral of side 2 centered near origin
        x, y = random_points_in_triangle(N, dth)
    elif rng==4:
        theta = 2*np.pi*np.random.random(N)
        r = 2*np.sqrt(np.random.random(N))
        x = r*np.cos(theta)
        y = 0.5*r*np.sin(theta)   # aspect ratio < 1 for flatter core
        x = np.sign(x)*np.abs(x)**(2*dth)
        y = np.sign(y)*np.abs(y)**(2*dth)
    elif rng==5:
        r = np.random.uniform(0.25, 1, N)**dth
        theta = 2*np.pi*np.random.random(N)
        x = r*np.cos(theta)
        y = r*np.sin(theta)
    elif rng==6:
        r = np.random.random(N)**dth
        theta = np.pi*np.random.random(N)   # half-disk
        x = r*np.cos(theta)
        y = r*np.sin(theta)**1.5
    else:
        raise ValueError("Invalid RNG.")

    if STRICT_GEOM:
        _teardrop_inplace(x, y, tda, tdw, tdt)
    else:
        _teardrop_inplace_fast(x, y, tda, tdw, tdt)

    # 180° concat
    #x = np.concatenate((x, -x))
    #y = np.concatenate((y, -y))
    # alternate signs: +1, -1, +1, -1, ...
    #v = np.where(np.arange(x.size) % 2 == 0, 1.0, -1.0)
    #x *= v
    #y *= v
    x, y = apply_n_arms(x,y,arm)

    if STRICT_GEOM:
        _swirl_inplace(x, y, swa, swb)
    else:
        _swirl_inplace_fast(x, y, swa, swb)

    # squish + rotate
    y *= sqs
    rot = 2.0 * np.pi * frt
    rc = np.cos(rot); rs = np.sin(rot)
    xr = x * rc - y * rs
    yr = x * rs + y * rc
    x, y = xr, yr

    # normals after geometry (RNG order preserved)
    z = np.random.normal(0.0, 1.0, size=x.size)
    mult = np.exp(lmu + lsig * z).astype(np.float32, copy=False)
    np.clip(mult, 1.0, float(drt), out=mult)

    logo = x + 1j * y
    return logo, mult




def render_logo(d: dict, verbose: bool = False) -> np.ndarray:
    t0 = time.perf_counter()

    # ---------- build geometry + size multipliers ----------
    logo, mult = build_logo(d)
    t_geom = time.perf_counter()

    # ---------- square frame centered at (0,0) ----------
    rx = np.max(np.abs(logo.real))
    ry = np.max(np.abs(logo.imag))
    half0 = max(rx, ry)
    half  = half0 * (1.0 + 2.0 * d["mrg"])
    span  = 2.0 * half

    W = H = int(d["pix"])
    px_per_logical = (W - 1) / span
   
    # ---------- convert size multipliers → pixel radii + threshold ----------
    r_px = np.rint(mult * d["fos"] * (W - 1)).astype(np.int32)
    r_min = int(d.get("min", 1))
    mask  = r_px >= r_min
    kept  = int(mask.sum())
    N0    = r_px.size
    r_px = r_px[mask]
    x = logo.real[mask]
    y = logo.imag[mask]

    # ---------- map logical → pixel coords ----------
    px = np.rint((x + half) * px_per_logical).astype(np.int32)
    py = np.rint((half - y) * px_per_logical).astype(np.int32)
    px = np.clip(px, 0, W - 1)
    py = np.clip(py, 0, H - 1)
    t_map = time.perf_counter()

    if kept == 0:
        if verbose:
            print(f"[render] N={N0} W=H={W} span={span:.6g} fos={d['fos']} drt={d['drt']} "
                  f"min={r_min}px px/log={px_per_logical:.3f}  kept=0/{N0} (0.0%)  groups=0")
            print(f"[render] build_logo={t_geom - t0:.3f}s  map={t_map - t_geom:.3f}s  "
                  f"radii/threshold={time.perf_counter() - t_map:.3f}s  total={time.perf_counter() - t0:.3f}s")
        return np.zeros((H, W), dtype=np.uint8)

    t_radii = time.perf_counter()

    # ---------- bucket by radius using one sort (contiguous slices) ----------
    method = GROUP_METHOD  # "auto" | "sort" | "count"
    r_max= int(math.ceil(d["drt"] * d["fos"] * (W - 1)))
    if method == "auto":
        use_count = (r_px.size >= 500_000) and ((r_max - r_min) <= 4096)
    elif method == "count":
        use_count = True
    else:  # "sort"
        use_count = False

    t_grp0 = time.perf_counter()
    if use_count:
        # O(N) bucket (njit) -> returns compact order of kept items
        order, r_unique, starts, counts = bucket_by_radius(r_px, r_min, r_max)
        px_sorted = px[order]
        py_sorted = py[order]
        ends = starts + counts
        if verbose:
            t_grp1 = time.perf_counter()
            print(f"[render] group=count (O(N)), points:{r_px.size} rmax={r_max}, groups={r_unique.size}, time={t_grp1 - t_grp0:.3f}s")
    else:
        # argsort reference path
        order     = np.argsort(r_px, kind="stable")
        r_sorted  = r_px[order]
        px_sorted = px[order]
        py_sorted = py[order]
        switch = np.flatnonzero(np.diff(r_sorted)) + 1
        starts = np.concatenate(([0], switch))
        ends   = np.concatenate((switch, [r_sorted.size]))
        r_unique = r_sorted[starts]
        if verbose:
            t_grp1 = time.perf_counter()
            print(f"[render] group=sort (argsort), points:{r_px.size} rmax={r_max}, groups={r_unique.size}, time={t_grp1 - t_grp0:.3f}s")

    t_bucket = time.perf_counter()

    # ---------- verbose summary BEFORE stamping ----------
    if verbose:
        kept_pct = 100.0 * kept / N0
        print(
            f"[render] N={N0}  kept={kept} ({kept_pct:.1f}%)  W=H={W}  span={span:.6g}  "
            f"fos={d['fos']}  drt={d['drt']}  min={r_min}px  px/log={px_per_logical:.3f}  "
            f"rpx[min,max]=[{r_unique.min()},{r_unique.max()}]  groups={r_unique.size}"
        )

    # ---------- rasterize with cached offsets ----------
    offset_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def get_offsets(rr: int):
        rr = int(rr)
        off = offset_cache.get(rr)
        if off is None:
            off = make_disc_offsets(rr)
            offset_cache[rr] = off
        return off

    t1 = time.perf_counter()
    canvas = np.zeros((H, W), dtype=np.uint8)

    for rr, s, e in zip(r_unique, starts, ends):
        if e <= s: continue
        dy, dx = get_offsets(rr)
        for i in range(s, e, STAMP_CHUNK):
            j_end = min(i + STAMP_CHUNK, e)
            stamp_points(canvas, py_sorted[i:j_end], px_sorted[i:j_end], dy, dx, H, W)

    t2 = time.perf_counter()

    if verbose:
        print(f"[render] build_logo={t_geom - t0:.3f}s  map={t_map - t_geom:.3f}s  "
              f"radii/threshold={t_radii - t_map:.3f}s  bucket/sort={t_bucket - t_radii:.3f}s  "
              f"stamp={t2 - t1:.3f}s  total={t2 - t0:.3f}s")

    return canvas

def add_footer_label(
    base: vips.Image,
    text: str,
    *,
    footer_frac: float = 0.02,   # target glyph height ≈ 1.5% of H
    pad_lr_px: int = 40,
    dpi: int = 300,
    align: str = "centre",
    invert: bool = False,
    font_family: str = "Courier New",
    font_weight: str = "Bold",
    min_px: int = 10,             # min glyph height in px
    max_px_frac: float = 0.05,    # cap glyph height to 5% of H
    max_retries: int = 8,         # shrink attempts if render fails/too big
) -> vips.Image:
    H, W = base.height, base.width
    if H <= 0 or W <= 0:
        return base

    bottom_margin_px = max(2, H // 40)
    box_w = max(1, W - 2 * pad_lr_px)

    # initial target height in pixels -> points for vips.text
    target_px = int(max(min_px, min(H * footer_frac, H * max_px_frac)))
    pt = max(6, int(round(target_px * 72.0 / dpi)))
    pt = min(pt, 512)  # hard cap

    # tokens to wrap on (add a space so breaking is clean)
    tokens = [tok.strip() for tok in text.split(",")]
    tokens = [t for t in tokens if t]  # drop empties

    def wrap_lines(font_str: str) -> list[str]:
        """Greedy wrap tokens so each line fits box_w at given font."""
        lines: list[str] = []
        line = ""
        for i, tok in enumerate(tokens):
            piece = tok if not line else f"{line}, {tok}"
            # measure candidate width
            test = vips.Image.text(piece, dpi=dpi, font=font_str, align=align)
            if test.width <= box_w or not line:
                line = piece
            else:
                # commit previous line, start new with current token
                lines.append(line)
                line = tok
        if line:
            lines.append(line)
        return lines

    for _ in range(max_retries):
        font_str = f"{font_family} {font_weight} {pt}"

        try:
            lines = wrap_lines(font_str)
            # render final glyph with explicit newlines (no width param needed now)
            glyph = vips.Image.text("\n".join(lines), dpi=dpi, font=font_str, align=align)
        except vips.Error:
            pt = max(6, int(pt * 0.85))
            continue

        glyph = (glyph > 0).ifthenelse(255, 0, blend=False)

        # if still too big, shrink and retry
        if glyph.height > int(H * max_px_frac * 1.1) or glyph.width > (box_w * 1.02):
            pt = max(6, int(pt * 0.9))
            continue

        gx = pad_lr_px + max(0, (box_w - glyph.width) // 2)
        gy = max(0, H - glyph.height - bottom_margin_px)
        glyph_full = vips.Image.black(W, H).insert(glyph, gx, gy)
        return base | glyph_full if not invert else base & (255 - glyph_full)

    # fallback: no footer if all attempts fail
    return base

def save_png_bilevel(
    canvas: np.ndarray,
    out_path: str,
    invert: bool,
    footer_text: str | None = None,
    *,
    footer_pad_lr_px: int = 48,
    footer_dpi: int = 300,
):
    """
    Save a bilevel (0/255) PNG from a numpy array, optionally adding centered
    footer text rendered with pyvips. Footer color adapts to invert mode.
    """
    if canvas.dtype != np.uint8:
        canvas = canvas.astype(np.uint8, copy=False)
 
    # invert BEFORE converting to vips so the text color logic can see it
    if invert:
        canvas = 255 - canvas

    H, W = canvas.shape
    base = vips.Image.new_from_memory(canvas.data, W, H, 1, "uchar")

    if footer_text:
        base = add_footer_label(
            base,
            footer_text,
            pad_lr_px=footer_pad_lr_px,
            dpi=footer_dpi,
            align="centre",
            invert=invert,  # <- pass inversion flag
        )

    base.write_to_file(
        out_path,
        compression=1,
        effort=1,
        filter="none",
        interlace=False,
        strip=True,
        bitdepth=1,
    )

def np_to_vips_gray_u8(arr: np.ndarray) -> vips.Image:
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    H, W = arr.shape
    return vips.Image.new_from_memory(arr.data, W, H, 1, "uchar")

def pad_to_square(im: vips.Image, px: int) -> vips.Image:
    """Center-pad a 1-band image to (px, px) with black background."""
    dx = max(0, (px - im.width) // 2)
    dy = max(0, (px - im.height) // 2)
    canvas = vips.Image.black(px, px)
    return canvas.insert(im, dx, dy)

def build_mosaic_streaming(
    spec_lines: list[str],
    cols: int,
    gap: int,
    invert: bool,
    footer: bool,
    footer_pad: int,
    footer_dpi: int,
    verbose: bool
) -> vips.Image:
    """
    Stream tiles into a big canvas with draw_image, row-major order.
    All build_logo parameters come from the spec lines.
    """
    if not spec_lines:
        raise ValueError("No specs provided for mosaic.")

    d0 = parse_logo_spec(spec_lines[0], LOGO_DEFAULTS)
    tile_px = int(d0["pix"])

    n = len(spec_lines)
    rows = math.ceil(n / cols)
    W = cols * tile_px + (cols - 1) * gap
    H = rows * tile_px + (rows - 1) * gap
    base = vips.Image.black(W, H)

    for i, spec in enumerate(spec_lines):
        d = parse_logo_spec(spec, LOGO_DEFAULTS)
        canvas = render_logo(d, verbose=verbose)
        vtile  = np_to_vips_gray_u8(canvas)  # VIPS 1-band

        if vtile.width != tile_px or vtile.height != tile_px:
            vtile = pad_to_square(vtile, tile_px)

        if footer:
            vtile = add_footer_label(
                vtile,
                logo_dict_to_string(d),
                pad_lr_px=footer_pad,
                dpi=footer_dpi,
                align="centre",
                invert=False,  # global invert later
            )

        r, c = divmod(i, cols)
        x = c * (tile_px + gap)
        y = r * (tile_px + gap)
        base = base.draw_image(vtile, x, y)

    base = (base > 0).ifthenelse(255, 0)
    if invert:
        base = base ^ 255
    return base


# ---------- JIT everything ----------


def _prejit():
    # strict
    _smoothstep_scalar.compile((float64, float64))
    _teardrop_inplace.compile((float64[:], float64[:], float64, float64, float64))
    _swirl_inplace.compile((float64[:], float64[:], float64, float64))
    # fast
    apply_n_arms.compile(float64[:], float64[:],int64)
    _teardrop_inplace_fast.compile((float64[:], float64[:], float64, float64, float64))
    _rmax_serial.compile((float64[:], float64[:]))
    _swirl_inplace_fast.compile((float64[:], float64[:], float64, float64))
    _ = bucket_by_radius(np.array([1,2,3,2,1], dtype=np.int32), 1,3)
try:
    _prejit()
except Exception:
    pass


# ---------- CLI ----------

def build_parser():
    p = argparse.ArgumentParser(description="Generate a bilevel dotted swirl logo (spec-driven).")
    p.add_argument("--logo", type=str, required=True, help="Logo spec 'k:v,...'. May include expandspec ranges.")
    p.add_argument("--invert", action="store_true")
    p.add_argument("--out", type=str, default="logo.png")
    p.add_argument("--footer", action="store_true", help="Add parameter string as footer text")
    p.add_argument("--footer-dpi", type=int, default=300)
    p.add_argument("--footer-pad", type=int, default=48)
    p.add_argument("--cols", type=int, default=None, help="Columns in mosaic; default = round(sqrt(num_specs))")
    p.add_argument("--gap", type=int, default=20, help="Gap (px) between tiles in mosaic")
    p.add_argument("--verbose", "-v", action="store_true", help="Print progress and diagnostics") 
    p.add_argument("--group", type=str, default="auto",choices=["auto","sort","count"],help="Radius grouping method: auto | sort (argsort) | count (O(N) bucket).")
    return p

def main():
    global GROUP_METHOD
    ap = build_parser()
    args = ap.parse_args()
    GROUP_METHOD = args.group

    # Expand spec (1 → single; >1 → mosaic)
    spec_lines = expandspec.expand_cartesian_lists(args.logo)

    if len(spec_lines) > 1:
        n = len(spec_lines)
        cols = args.cols if args.cols is not None else max(1, int(round(math.sqrt(n))))  # ← auto
        if args.verbose:
            print(f"🧩 Building mosaic from {n} specs with cols={cols} (auto={args.cols is None}), gap={args.gap}")

        mosaic = build_mosaic_streaming(
            spec_lines=spec_lines,
            cols=cols,                 # ← pass computed cols
            gap=args.gap,
            invert=args.invert,
            footer=args.footer,
            footer_pad=args.footer_pad,
            footer_dpi=args.footer_dpi,
            verbose=args.verbose,
        )
        mosaic.write_to_file(
            args.out,
            compression=1, effort=1, filter="none", interlace=False, strip=True, bitdepth=1,
        )
        if args.verbose:
            rows = math.ceil(n / cols)
            print(f"✅ Saved mosaic {args.out}  ({cols}×{rows})")
        else:
            print(f"✅ Saved mosaic {args.out}")
        return

    # Single image
    d = parse_logo_spec(spec_lines[0], LOGO_DEFAULTS)

    print("🎨 Rendering logo with parameters:")
    print(logo_dict_to_string(d))

    canvas = render_logo(d,verbose=args.verbose)

    H, W = canvas.shape
    base = vips.Image.new_from_memory(canvas.data, W, H, 1, "uchar")

    if args.invert:
        base = base ^ 255

    if args.footer:
        base = add_footer_label(
            base,
            logo_dict_to_string(d),
            pad_lr_px=args.footer_pad,
            dpi=args.footer_dpi,
            align="centre",
            invert=args.invert,
        )

    base = (base > 0).ifthenelse(255, 0)
    base.write_to_file(
        args.out,
        compression=1, effort=1, filter="none", interlace=False, strip=True, bitdepth=1,
    )
    print(f"✅ Saved {args.out}")

if __name__ == "__main__":
    main()
