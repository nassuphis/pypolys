import math
import numpy as np
from numba import njit, prange, types
from numba.typed import Dict
import specparser


# ===== random number generator =====

RNG = np.random.default_rng(seed=42)

# ===== registry =====

ALLOWED = {}

# ----- state channels (int8 keys) -----
K_MULT = np.int8(1)   # stores the multipliers vector as complex[:] (real part used)

# ----- tiny helpers -----
def _as_xy(z: np.ndarray):
    return z.real, z.imag  # views, no copies

def _frozen_len(state) -> int:
    mult = state.get(K_MULT)
    return 0 if (mult is None) else mult.size

def _split_head_tail(z, state):
    k = _frozen_len(state)
    return z[:k], z[k:]

def _current_gid(state) -> int:
    mult = state.get(K_MULT)
    if mult is None or mult.size == 0:
        return 1  # first sizing group
    # next group id is always last stored gid + 1
    return int(round(mult.imag[-1] + 1))

# =========================
# RNG ops (return new z)
# =========================



def op_rnorm(z,a,state):
     N   = int(a[0].real) or 1_000_000
     loc  = a[1].real
     scale = a[2].real or 1
     u = RNG.normal(loc=loc,scale=scale,size=N)
     v = RNG.normal(loc=loc,scale=scale,size=N)
     z = np.concatenate((z,u+1j*v))
     return z

ALLOWED["rnorm"] = op_rnorm

#square
def op_runif(z,a,state):
     N   = int(a[0].real) or 1_000_000
     low  = a[1].real or -1
     high  = a[2].real or 1
     u = RNG.uniform(low=low,high=high,size=N)
     v = RNG.uniform(low=low,high=high,size=N)
     z = np.concatenate((z,u+1j*v))
     return z

ALLOWED["runif"] = op_runif


#disk
def op_rud(z, a, state):
    N   = int(a[0].real) or 1_000_000
    dth = float(a[1].real) or 0.5
    u1 = RNG.random(N)
    u2 = RNG.random(N)
    r  = u1 ** dth
    th = np.exp(1j * 2.0 * np.pi * u2)
    z = np.concatenate((z, r*th))
    return z

ALLOWED["rud"] = op_rud

#arc
def op_rua(z, a, state):
    N   = int(a[0].real) or 1_000_000
    start = min(a[1].real,a[1].imag)
    end = max(a[1].real,a[1].imag)
    rmax = a[2].real
    center = a[3]
    dth = float(a[4].real) or 0.5
    u1 = RNG.uniform(low=0,high=rmax,size=N)
    r  = u1 ** dth
    u2 = RNG.uniform(low=start,high=end,size=N)
    th = np.exp(1j * 2.0 * np.pi * u2)
    z = np.concatenate((z, r*th+center)) 
    return z

ALLOWED["rua"] = op_rua

# square
def op_rus(z, a, state):
    N = int(a[0].real) or 1_000_000
    dth = float(a[1].real) or 1.0
    x = RNG.uniform(-1.0, 1.0, size=N).astype(np.float64, copy=False)
    y = RNG.uniform(-1.0, 1.0, size=N).astype(np.float64, copy=False)
    x = np.sign(x)*np.abs(x)**dth
    y = np.sign(y)*np.abs(y)**dth
    z = np.concatenate((z,x + 1j * y))
    return z

ALLOWED["rus"] = op_rus


def op_rtr(z, a, state):
    N = int(a[0].real) or 1_000_000
    dthx = float(a[1].real) or 1.0
    dthy = float(a[2].real) or dthx
    h = np.sqrt(3) / 2
    A = np.asarray((-1.0, -h), dtype=np.float64)
    B = np.asarray(( 1.0, -h), dtype=np.float64)
    C = np.asarray(( 0.0,  h), dtype=np.float64)
    u = RNG.random(N)
    v = RNG.random(N)
    # reflect across diagonal for u+v>1
    mask = (u + v) > 1.0
    u[mask] = 1.0 - u[mask]
    v[mask] = 1.0 - v[mask]
    # affine combination
    x = A[0] + u * (B[0] - A[0]) + v * (C[0] - A[0])
    y = A[1] + u * (B[1] - A[1]) + v * (C[1] - A[1])
    x = np.sign(x)*np.abs(x)**dthx
    y = np.sign(y)*np.abs(y)**dthy
    z = np.concatenate((z,x + 1j*y))
    return z

ALLOWED["rtr"] = op_rtr

def op_rtrd(z, a, state):
    N = int(a[0].real) or 1_000_000
    alpha_C = max(float(a[1].real) or 1.0, 1e-6)
    alpha_A = max(float(a[2].real) or 1.0, 1e-6)
    alpha_B = max(float(a[3].real) or 1.0, 1e-6)
    # Dirichlet via normalized Gamma(α, 1) draws
    gA = RNG.gamma(shape=alpha_A, scale=1.0, size=N)
    gB = RNG.gamma(shape=alpha_B, scale=1.0, size=N)
    gC = RNG.gamma(shape=alpha_C, scale=1.0, size=N)
    s  = gA + gB + gC
    wA = gA / s
    wB = gB / s
    wC = gC / s
    h = np.sqrt(3.0) / 2.0
    shift = h / 3.0        # = √3 / 6
    A = np.array([-1.0, -h+shift], dtype=np.float64)
    B = np.array([ 1.0, -h+shift], dtype=np.float64)
    C = np.array([ 0.0,  h+shift], dtype=np.float64)
    # Linear map preserves triangular outline
    x = wA * A[0] + wB * B[0] + wC * C[0]
    y = wA * A[1] + wB * B[1] + wC * C[1]
    z = np.concatenate((z,x + 1j * y))
    return z

ALLOWED["rtrd"] = op_rtrd

def op_rsqd(z, a, state):
    N = int(a[0].real) or 1_000_000
    alpha_A = max(float(a[1].real) or 1.0, 1e-6)
    alpha_B = max(float(a[2].real) or 1.0, 1e-6)
    alpha_C = max(float(a[3].real) or 1.0, 1e-6)
    alpha_D = max(float(a[4].real) or 1.0, 1e-6)
    gA = RNG.gamma(shape=alpha_A, scale=1.0, size=N)
    gB = RNG.gamma(shape=alpha_B, scale=1.0, size=N)
    gC = RNG.gamma(shape=alpha_C, scale=1.0, size=N)
    gD = RNG.gamma(shape=alpha_D, scale=1.0, size=N)
    s  = gA + gB + gC + gD
    wA = gA / s
    wB = gB / s
    wC = gC / s
    wD = gD / s
    A = np.array([ -1.0, +1.0], dtype=np.float64)
    B = np.array([ +1.0, +1.0], dtype=np.float64)
    C = np.array([ -1.0, -1.0], dtype=np.float64)
    D = np.array([ +1.0, -1.0], dtype=np.float64)
    # Linear map preserves square outline
    x = wA * A[0] + wB * B[0] + wC * C[0] + wD * D[0]
    y = wA * A[1] + wB * B[1] + wC * C[1] + wD * D[1]
    z = np.concatenate((z,x + 1j * y))
    return z

ALLOWED["rsqd"] = op_rsqd

def op_rsq_edge(z, a, state):
    """
    Sample points concentrated on square edges (L∞-unit square [-1,1]^2).
    a[0]=N
    a[1]=beta (edge position shape; 1=uniform along edge, >1 → toward edge middle, <1 → toward corners)
    a[2]=thick (0=exactly on edge; >0 adds inward thickness in L∞ sense, as a fraction of half-width)
    """
    N     = int(a[0].real) or 1_000_000
    beta  = float(a[1].real) or 1.0
    thick = float(a[2].real) 
    thick = max(0.0, min(thick, 1.0))

    # choose an edge: 0=top(y=+1), 1=right(x=+1), 2=bottom(y=-1), 3=left(x=-1)
    edge = (RNG.random(N) * 4).astype(np.int32)

    # position along the edge
    t = RNG.beta(beta, beta, size=N)  # in (0,1)
    u = 2.0 * t - 1.0                 # map to (-1,1)

    # inward offset (0 on boundary → thick toward interior)
    if thick > 0.0:
        # bias toward boundary; d in [0,thick]
        d = thick * (1.0 - RNG.random(N) ** 2.0)
    else:
        d = np.zeros(N, dtype=np.float64)

    x = np.empty(N, dtype=np.float64)
    y = np.empty(N, dtype=np.float64)

    # top: y=+1-d, x=u
    m = (edge == 0)
    x[m] = u[m]
    y[m] = +1.0 - d[m]

    # right: x=+1-d, y=u
    m = (edge == 1)
    x[m] = +1.0 - d[m]
    y[m] = u[m]

    # bottom: y=-1+d, x=u
    m = (edge == 2)
    x[m] = u[m]
    y[m] = -1.0 + d[m]

    # left: x=-1+d, y=u
    m = (edge == 3)
    x[m] = -1.0 + d[m]
    y[m] = u[m]

    return np.concatenate((z, x + 1j*y))

ALLOWED["rsqedge"] = op_rsq_edge

# ---------- deterministic random ----------

@njit(cache=True, nogil=True, fastmath=True)
def op_serp(z, a, state):
    n = int(a[0].real) or 2**20
    n = int(round(math.sqrt(n))**2)
    if n <= 0: return np.empty(0, np.complex128)
    cols = int(math.sqrt(n))
    rows = cols
    zs = np.empty(n, np.complex128)
    for k in range(n):
        r = k // cols
        c = k % cols
        if (r & 1) == 1:
            c_eff = cols - 1 - c
        else:
            c_eff = c
        x = (c_eff + 0.5) / cols -0.5
        y = (r + 0.5) / rows - 0.5
        zs[k] = x + 1j * y
    z=np.concatenate((z,zs))
    return z

ALLOWED["serp"] = op_serp

# ---------- cartesian random ----------


def op_carunif(z, a, state):
    N  = int(a[0].real) or 100
    w  = float(a[1].real) or 0.01
    k  = _frozen_len(state)
    head, base = z[:k], z[k:]
    n = base.size
    if n == 0 or N <= 0:
        return z  # nothing to expand
    u = RNG.uniform(-w, +w, size=(N, n))
    v = RNG.uniform(-w, +w, size=(N, n))
    out = (base[None, :] + u + 1j*v).reshape(N * n)
    return np.concatenate((head, out))

ALLOWED["carunif"] = op_carunif

def op_cardsk(z, a, state):
    N   = int(a[0].real) or 100
    r   = float(a[1].real) or 0.02
    dth = float(a[2].real) or 0.5
    k   = _frozen_len(state)
    head, base = z[:k], z[k:]
    n = base.size
    if n == 0 or N <= 0:
        return z
    u = RNG.random(size=(N, n))
    v = RNG.random(size=(N, n))
    p = r * (u**dth) * np.exp(1j * 2.0 * np.pi * v)
    out = (base[None, :] + p).reshape(N * n)
    return np.concatenate((head, out))

ALLOWED["cardsk"] = op_cardsk

# ---------- diffusion ops ----------

@njit(cache=True, nogil=True)
def _walk_inplace(x, y, scale, steps):
    if steps <= 0:
        return
    n = x.size
    for _ in range(steps):
        # draw per-step vectors
        u = np.random.uniform(-scale, scale, size=n)
        v = np.random.uniform(-scale, scale, size=n)
        x += u
        y += v

@njit(cache=True, nogil=True)
def _pwalk_inplace(x, y, scale, pw, steps):
    if steps <= 0:
        return
    n = x.size
    # fixed amplitude from initial positions (matches your old behavior)
    a = (x * x + y * y) ** 0.5
    if pw != 1.0:
        a = a ** pw
    for _ in range(steps):
        u = np.random.uniform(-scale, scale, size=n)
        v = np.random.uniform(-scale, scale, size=n)
        x += u * a
        y += v * a

@njit(cache=True, nogil=True)
def _disk_diffuse_inplace(x, y, scale, pw, steps):
    if steps <= 0:
        return
    n = x.size
    dist = (x * x + y * y) ** 0.5
    if pw != 1.0:
        dist = dist ** pw
    for _ in range(steps):
        u = np.random.uniform(-scale, scale, size=n)
        v = np.random.uniform(-scale, scale, size=n)
        x += u * dist
        y += v * dist

def op_walk(z, a, state):
    scale = float(a[0].real) if a.size > 0 else 1e-3
    steps = int(a[1].real) if a.size > 1 else 1
    k = _frozen_len(state)
    if k < z.size and steps > 0:
        x, y = _as_xy(z)
        _walk_inplace(x[k:], y[k:], scale, steps)
    return z

ALLOWED["walk"] = op_walk

def op_pwalk(z, a, state):
    scale = float(a[0].real) if a.size > 0 else 1e-3
    pw    = float(a[1].real) if a.size > 1 else 1.0
    steps = int(a[2].real) if a.size > 2 else 1
    k = _frozen_len(state)
    if k < z.size and steps > 0:
        x, y = _as_xy(z)
        _pwalk_inplace(x[k:], y[k:], scale, pw, steps)
    return z

ALLOWED["pwalk"] = op_pwalk

def op_disk_diffuse(z, a, state):
    scale = float(a[0].real) if a.size > 0 else 1e-3
    steps = int(a[1].real) if a.size > 1 else 0
    pw    = float(a[2].real) if a.size > 2 else 1.0
    k = _frozen_len(state)
    if k < z.size and steps > 0:
        x, y = _as_xy(z)
        _disk_diffuse_inplace(x[k:], y[k:], scale, pw, steps)
    return z

ALLOWED["ddiffuse"] = op_disk_diffuse

# ---------- clip ops ----------

def op_dclip(z, a, state):
    r = a[0].real or 1.0
    c = a[1]
    k = _frozen_len(state)
    if k <= 0:
        # no frozen points; clip all
        mask = np.abs(z - c) > r
        return z[mask]
    head, tail = z[:k], z[k:]
    mask_tail = np.abs(tail - c) > r
    return np.concatenate((head, tail[mask_tail]))

ALLOWED["dclip"] = op_dclip

def op_sclip(z, a, state):
    r = a[0].real or 1.0
    c = a[1]
    k = _frozen_len(state)
    if k <= 0:
        # no frozen points; clip all
        dx = z.real - c.real
        dy = z.imag - c.imag
        inside = (np.abs(dx) < r) & (np.abs(dy) < r)
        return z[~inside]

    head, tail = z[:k], z[k:]
    dx = tail.real - c.real
    dy = tail.imag - c.imag
    inside_tail = (np.abs(dx) < r) & (np.abs(dy) < r)
    return np.concatenate((head, tail[~inside_tail]))

ALLOWED["sclip"] = op_sclip

def op_osclip(z, a, state):
    r = a[0].real or 1.0
    c = a[1]
    k = _frozen_len(state)
    if k <= 0:
        # no frozen points; clip all
        dx = z.real - c.real
        dy = z.imag - c.imag
        inside = (np.abs(dx) < r) & (np.abs(dy) < r)
        return z[inside]

    head, tail = z[:k], z[k:]
    dx = tail.real - c.real
    dy = tail.imag - c.imag
    inside_tail = (np.abs(dx) < r) & (np.abs(dy) < r)
    return np.concatenate((head, tail[inside_tail]))

ALLOWED["osclip"] = op_osclip

# ---------- simple ops ----------

def op_add(z, a, state):
    k = _frozen_len(state)
    if k < z.size:
        z[k:] += a[0]
    return z

ALLOWED["add"] = op_add

def op_mul(z, a, state):
    k = _frozen_len(state)
    if k < z.size:
        z[k:] *= a[0]
    return z

ALLOWED["mul"] = op_mul

def op_rmul(z, a, state):
    k = _frozen_len(state)
    if k < z.size:
        zz = z[k:]
        z[k:] = zz.real * a[0] + 1j * zz.imag
    return z

ALLOWED["rmul"] = op_rmul

def op_imul(z, a, state):
    k = _frozen_len(state)
    if k < z.size:
        zz = z[k:]
        z[k:] = zz.real + 1j * zz.imag * a[0]
    return z

ALLOWED["imul"] = op_imul


def op_rot(z, a, state):
    phi = np.exp(1j * 2.0 * np.pi * a[0].real)
    k = _frozen_len(state)
    if k < z.size:
        z[k:] *= phi
    return z
ALLOWED["rot"] = op_rot

def op_rth(z, a, state):
    """
    Radius-from-imag (with exponent), angle-from-real:
      r  = sign(Im(z)) * |Im(z)|^dth
      th = exp(i * 2π * Re(z))
      z_tail := r * th
    Only applies to the unfrozen tail (points without sizes yet).
    """
    mult = state.get(K_MULT)
    k = 0 if (mult is None) else mult.size

    if k < z.size:
        dth = float(a[0].real) if a.size > 0 else 0.5
        tail = z[k:]
        r  = np.sign(tail.imag) * np.abs(tail.imag) ** dth
        th = np.exp(1j * 2.0 * np.pi * tail.real)
        z[k:] = r * th
    return z

ALLOWED["rth"] = op_rth


def op_toline(z, a, state):
    """
    Cayley transform to (imag) line:
      z_tail := i * (1 + z_tail) / (1 - z_tail)
    Only applies to the unfrozen tail (points without sizes yet).
    """
    mult = state.get(K_MULT)
    k = 0 if (mult is None) else mult.size

    if k < z.size:
        tail = z[k:]
        z[k:] = 1j * (1.0 + tail) / (1.0 - tail)
        # (optional) protect exact pole at z=1:
        # denom = (1.0 - tail)
        # denom = np.where(denom == 0, 1e-15 + 0j, denom)
        # z[k:] = 1j * (1.0 + tail) / denom
    return z

ALLOWED["toline"] = op_toline

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

# =========================
# Geometry ops (in-place)
# =========================

def op_td(z, a, state):
    # a[0]=tda, a[1]=tdw, a[2]=tdt
    k = _frozen_len(state)
    if k < z.size:
        x, y = _as_xy(z)
        _teardrop_inplace_fast(x[k:], y[k:], a[0].real, a[1].real, a[2].real)
    return z

ALLOWED["td"] = op_td

def op_arms(z, a, state):
    m = int(a[0].real)
    if m <= 1: return z
    k = _frozen_len(state)
    if k < z.size:
        n_tail = z.size - k
        idx = np.mod(np.arange(n_tail), m)
        ang = (2.0*np.pi/m) * idx
        c = np.cos(ang); s = np.sin(ang)
        x, y = _as_xy(z)
        xr = x[k:] * c - y[k:] * s
        yr = x[k:] * s + y[k:] * c
        x[k:] = xr; y[k:] = yr
    return z

ALLOWED["arms"] = op_arms

def op_swirl(z, a, state):
    # a[0]=swa, a[1]=swb
    k = _frozen_len(state)
    if k < z.size:
        x, y = _as_xy(z)
        _swirl_inplace_fast(x[k:], y[k:], a[0].real, a[1].real)
    return z

ALLOWED["swirl"] = op_swirl


def op_squish(z, a, state):
    k = _frozen_len(state)
    if k < z.size:
        z.imag[k:] *= a[0].real
    return z

ALLOWED["squish"] = op_squish



# =========================
# Sinks
# =========================

@njit(cache=True, nogil=True, parallel=True, fastmath=True)
def _sink_gaussian_inplace_fast(x, y, Cx, Cy, alpha, sigma):
    """
    In-place Gaussian sink:
      x,y : float64[:]
      Cx,Cy : sink location
      alpha : strength (>=0)
      sigma : decay (>0). If sigma<=0, behaves like linear pull with factor alpha.
    """
    n = x.size
    if sigma <= 0.0:
        # linear pull: z' = (1-alpha) z + alpha C
        one_minus = 1.0 - alpha
        for i in prange(n):
            xi = x[i]; yi = y[i]
            x[i] = one_minus * xi + alpha * Cx
            y[i] = one_minus * yi + alpha * Cy
        return

    sig2 = sigma * sigma
    for i in prange(n):
        dx = x[i] - Cx
        dy = y[i] - Cy
        r2 = dx*dx + dy*dy
        f  = alpha * np.exp(- r2 / sig2)   # pull factor in [0, alpha]
        x[i] = x[i] - dx * f
        y[i] = y[i] - dy * f

def op_sink(z, a, state):
    C     = a[0]
    alpha = float(a[1].real) if a.size > 1 else 0.0
    sigma = float(a[2].real) if a.size > 2 else 1.0
    k = 0 if state.get(K_MULT) is None else state[K_MULT].size
    if k < z.size and (alpha != 0.0 or sigma != 0.0):
        x, y = _as_xy(z)
        _sink_gaussian_inplace_fast(x[k:], y[k:], C.real, C.imag, alpha, sigma)
    return z

ALLOWED["sink"] = op_sink

@njit(cache=True, nogil=True, parallel=True, fastmath=True)
def _nsink_inplace_fast(x, y, Cx, Cy, alpha,eps):
    """
    Newtonian (inverse-square) sink/repulsor:
      z' = z - alpha * (z - C) / (|z - C|^2 + eps^2)
    alpha > 0  -> attraction
    alpha < 0  -> repulsion
    """
    n = x.size
    for i in prange(n):
        dx = x[i] - Cx
        dy = y[i] - Cy
        r2 = dx*dx + dy*dy + eps
        inv = 1.0 / r2
        fx = alpha * dx * inv
        fy = alpha * dy * inv
        x[i] = x[i] - fx
        y[i] = y[i] - fy

def op_nsink(z, a, state):
    """
    a[0]=C (complex), a[1]=alpha (real), a[2]=eps (real, small >=0)
    """
    C     = a[0]
    alpha = float(a[1].real) if a.size > 1 else 0.0
    eps   = float(a[2].real) if a.size > 2 else 1e-12
    k = 0 if state.get(K_MULT) is None else state[K_MULT].size
    if k < z.size and alpha != 0.0:
        x, y = _as_xy(z)
        _nsink_inplace_fast(x[k:], y[k:], C.real, C.imag, alpha, eps)
    return z

ALLOWED["nsink"] = op_nsink

# =========================
# Size/mult ops (write to state)
# =========================

def op_dot_lognormal(z, a, state):
    """
    Assign lognormal-distributed dot sizes to the active tail
    and mark them with the current group id (imag part).
    """
    k = _frozen_len(state)
    tail_len = max(z.size - k, 0)
    if tail_len == 0:
        return z

    # parameters
    loc   = float(a[0].real) or 0.5
    scale = float(a[1].real) or 1.25
    drt   = float(a[2].real) or 100.0

    # lognormal tail
    zeta = RNG.normal(loc=loc, scale=scale, size=tail_len)
    mult_tail_real = np.exp(zeta).astype(np.float64)
    np.clip(mult_tail_real, 1.0, drt, out=mult_tail_real)

    # group id: last gid + 1 (or 1 if first)
    gid = _current_gid(state)

    # existing multipliers?
    mult_prev = state.get(K_MULT)
    if mult_prev is None:
        full_real = mult_tail_real
        full_imag = np.full(tail_len, gid, dtype=np.float64)
    else:
        full_real = np.concatenate((mult_prev.real, mult_tail_real))
        full_imag = np.concatenate((mult_prev.imag, np.full(tail_len, gid, dtype=np.float64)))

    # store combined complex vector (real=size, imag=group id)
    state[K_MULT] = (full_real + 1j * full_imag).astype(np.complex128, copy=False)
    return z

ALLOWED["ldot"] = op_dot_lognormal


def op_dot(z, a, state):
    """
    Assign a constant dot size to the active tail and
    mark it with the current group id (imag part).
    """
    k = _frozen_len(state)
    tail_len = max(z.size - k, 0)
    if tail_len == 0:
        return z

    # constant size value
    val = float(a[0].real) if a.size > 0 else 1.0
    mult_tail_real = np.full(tail_len, val, dtype=np.float64)

    # current group id (last gid + 1 or 1 if first)
    gid = _current_gid(state)

    # combine with any existing multipliers
    mult_prev = state.get(K_MULT)
    if mult_prev is None:
        full_real = mult_tail_real
        full_imag = np.full(tail_len, gid, dtype=np.float64)
    else:
        full_real = np.concatenate((mult_prev.real, mult_tail_real))
        full_imag = np.concatenate((mult_prev.imag, np.full(tail_len, gid, dtype=np.float64)))

    # store back
    state[K_MULT] = (full_real + 1j * full_imag).astype(np.complex128, copy=False)
    return z

ALLOWED["dot"] = op_dot

def op_dot_copy(z, a, state):
    """
    Copy size distribution from an existing group onto the active tail.

    a[0] (optional): source gid (real).
        If 0 or omitted, use the previous gid (current_gid - 1).

    Behavior:
      - Finds sizes where imag(mult) == source_gid among the frozen prefix.
      - Samples with replacement to fill the tail.
      - Writes sizes to the tail and tags them with a NEW gid.
    """
    k = _frozen_len(state)
    tail_len = max(z.size - k, 0)
    if tail_len == 0:
        return z

    mult = state.get(K_MULT)
    if mult is None or mult.size == 0:
        # nothing to copy from
        return z

    # determine source gid
    src_gid = int(round(a[0].real)) if a.size > 0 else 0
    if src_gid <= 0:
        src_gid = _current_gid(state) - 1
    if src_gid < 1:
        return z  # no valid source group yet

    # find source samples in the frozen prefix
    src_mask = (mult.imag[:k] == src_gid)
    if not np.any(src_mask):
        return z  # nothing to copy from

    src_sizes = mult.real[:k][src_mask]
    # sample with replacement to fill tail
    tail_sizes = RNG.choice(src_sizes, size=tail_len, replace=True)

    # assign new gid for the copied tail
    gid = _current_gid(state)

    # combine with existing multipliers
    full_real = np.concatenate((mult.real, tail_sizes))
    full_imag = np.concatenate((mult.imag, np.full(tail_len, gid, dtype=np.float64)))

    # store updated sizes + group ids
    state[K_MULT] = (full_real + 1j * full_imag).astype(np.complex128, copy=False)
    return z

ALLOWED["dcopy"] = op_dot_copy

# =========================
# macros
# =========================

def macro(spec:str,z0,state): # spec runner
    #state = Dict.empty(key_type=types.int8, value_type=types.complex128[:])
    names, A = specparser.parse_names_and_args(spec, MAXA=12)
    z=z0
    for k, name in enumerate(names):
        fn = ALLOWED.get(name)
        if fn is None:
            raise ValueError(f"Unknown op '{name}'.")
        z = fn(z, A[k], state)
    z=np.concatenate((z0,z))
    return z

def op_yin(z, a, state):
    N = a[0].real or 1e5
    size = a[1].real or 1
    spec = (
        f"rud:{N},sclip:1:1+0j,"
        f"dclip:0.5:0+0.5j,"
        f"rua:{N/8}:0.75+1.25j:0.25:0-0.5j,"
        f"add:0.5j,mul:{size}"
    )
    return macro(spec,z,state)

ALLOWED["yin"] = op_yin

def op_kirby(z,a,state):
    N = a[0].real or 1e7
    spec = (
        f"rud:{N}:0.1,osclip:0.5,ldot:4:1.5:500"
    )
    return macro(spec,z,state)

ALLOWED["kirby"] = op_kirby

# ===== executor (no JIT needed; kernels inside are Numba-fast) =====
def apply_chain(z0: np.ndarray, names: list[str], A: np.ndarray, state):
    z = z0
    for k, name in enumerate(names):
        fn = ALLOWED.get(name)
        if fn is None:
            raise ValueError(f"Unknown op '{name}'. Allowed: {list(ALLOWED)}")
        z = fn(z, A[k], state)

    mult_c = state.get(K_MULT)
    if mult_c is None:
        mult = np.ones(z.size, dtype=np.float32)
    else:
        k = mult_c.size
        mult = np.ones(z.size, dtype=np.float32)
        mult[:min(k, z.size)] = mult_c.real[:min(k, z.size)].astype(np.float32, copy=False)
    return z, mult

# ===== build_logo from spec =====

def build_logo_from_chain(spec: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Build geometry + multipliers from a pipeline spec string only.
    Returns (z: complex128[:], mult: float32[:]).
    """
    names, A = specparser.parse_names_and_args(spec, MAXA=12)

    # int8 -> complex128[:] typed dict (back-channel)
    state = Dict.empty(key_type=types.int8, value_type=types.complex128[:])

    # start with empty z (RNG op will create it)
    z0 = np.empty(0, dtype=np.complex128)
    z, mult = apply_chain(z0, names, A, state)
    return z, mult

def warmup_numba_geometry():
    # warm up vectorized kernels on tiny arrays
    tmp = np.zeros(8, dtype=np.complex128)
    x, y = _as_xy(tmp)
    _teardrop_inplace_fast(x, y, 0.8, 0.2, 0.1)
    _swirl_inplace_fast(x, y, 0.5, 2.0)
    _walk_inplace(x, y, 1e-3, 1)
    _pwalk_inplace(x, y, 1e-3, 1.0, 1)
    _disk_diffuse_inplace(x, y, 1e-3, 1.0, 1)


