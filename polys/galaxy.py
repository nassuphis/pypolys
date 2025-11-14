import sys
sys.path.insert(0, "/Users/nicknassuphis")
import math
import numpy as np
from numba import njit, prange, types, complex128, int32, float64
from numba.typed import Dict
from specparser import specparser
from specparser import expandspec

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

# ==================================================
# jited calcs
# ==================================================

@njit(cache=True, nogil=True)
def _smoothstep_scalar(x: float, w: float) -> float:
    if w==0.0:
        if x<=0.0: return 0.0
        return 1.0
    t = (x + w) / (2.0 * w)
    if t < 0.0: t = 0.0
    elif t > 1.0: t = 1.0
    return t * t * (3.0 - 2.0 * t)

# ==================================================
# RNG ops (return new z)
# ==================================================

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
    start = min(a[1].real,a[2].real)
    end = max(a[1].real,a[2].real)
    rmax = a[3].real or 1.0
    center = a[4]
    dth = float(a[5].real) or 0.5
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

# triangle
def op_rtr(z, a, state):
    N = int(a[0].real) or 1_000_000
    dthx = float(a[1].real) or 1.0
    dthy = float(a[2].real) or dthx
    h = np.sqrt(3) / 2
    shift = h / 3.0 
    A = np.asarray((-1.0, -h+shift), dtype=np.float64)
    B = np.asarray(( 1.0, -h+shift), dtype=np.float64)
    C = np.asarray(( 0.0,  h+shift), dtype=np.float64)
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

# triangle, diriclet
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

# square diriclet
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

# square 
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

# ==================================================
# deterministic generation
# ==================================================

# serpentine
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
        x = (c_eff + 0.5) / cols - 0.5
        y = (r + 0.5) / rows - 0.5
        zs[k] = x + 1j * y
    z=np.concatenate((z,zs))
    return z

ALLOWED["serp"] = op_serp

# ==================================================
# attractors
# ==================================================

@njit(cache=True, nogil=True, fastmath=True)
def _lorenz_rk4_project(N, dt, sigma, rho, beta,
                        x0, y0, z0, burn, stride, proj, mul):
    """
    Generate N projected complex points from Lorenz flow after burn-in,
    sampling every `stride` steps. Projection:
      proj = 0 -> (x,y), 1 -> (x,z), 2 -> (y,z)
    """
    total = burn + N * stride
    out   = np.empty(N, dtype=np.complex128)

    x = x0
    y = y0
    z = z0
    k = 0
    p = int(proj)  # ensure scalar int

    for i in range(total):
        # --- RK4 step ---
        k1x = sigma * (y - x)
        k1y = x * (rho - z) - y
        k1z = x * y - beta * z

        x2 = x + 0.5 * dt * k1x
        y2 = y + 0.5 * dt * k1y
        z2 = z + 0.5 * dt * k1z
        k2x = sigma * (y2 - x2)
        k2y = x2 * (rho - z2) - y2
        k2z = x2 * y2 - beta * z2

        x3 = x + 0.5 * dt * k2x
        y3 = y + 0.5 * dt * k2y
        z3 = z + 0.5 * dt * k2z
        k3x = sigma * (y3 - x3)
        k3y = x3 * (rho - z3) - y3
        k3z = x3 * y3 - beta * z3

        x4 = x + dt * k3x
        y4 = y + dt * k3y
        z4 = z + dt * k3z
        k4x = sigma * (y4 - x4)
        k4y = x4 * (rho - z4) - y4
        k4z = x4 * y4 - beta * z4

        x += (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        y += (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
        z += (dt / 6.0) * (k1z + 2.0 * k2z + k3z + k4z)

        if i >= burn and ((i - burn) % stride == 0):
            # choose projection as scalars
            if p == 0:       # (x, y)
                rx = x; ry = y
            elif p == 1:     # (x, z)
                rx = x; ry = z
            else:            # (y, z)
                rx = y; ry = z

            # build scalar complex explicitly
            out[k] = (mul * rx) + (mul * ry) * 1j
            k += 1
            if k >= N:
                break

    return out

def op_lorenz(z, a, state):
    """
    Deterministic Lorenz attractor sampler (appends complex points).

    Params (use your “or default” style):
      a[0] = N          (int)   samples to RETURN after burn/stride      [default 200_000]
      a[1] = dt         (real)  time step                                [0.01]
      a[2] = sigma      (real)  Lorenz sigma                              [10]
      a[3] = rho        (real)  Lorenz rho                                [28]
      a[4] = beta       (real)  Lorenz beta                               [8/3]
      a[5] = x0+iy0     (cpx)   initial (x0, y0)                          [0+1j]
      a[6] = z0         (real)  initial z0                                [1.05]
      a[7] = burn       (int)   burn-in steps (not recorded)              [2_000]
      a[8] = stride     (int)   keep every `stride` step after burn       [5]
      a[9] = proj       (real)  0:(x,y) 1:(x,z) 2:(y,z)                   [0]
      a[10]= mul        (real)  scale multiplier                          [1.0]
    """
    N      = int(a[0].real) or 200_000
    sigma  = float(a[1].real) or 10.0
    rho    = float(a[2].real) or 28.0
    beta   = float(a[3].real) or (8.0/3.0)
    dt     = 0.01
    x0     = 0.0
    y0     = 1.0
    z0     = 1.05
    burn   = int(2_000)
    stride = int(5)
    proj   = int(0)
    mul    = 1.0

    # generate
    pts = _lorenz_rk4_project(N, dt, sigma, rho, beta,
                              x0, y0, z0, burn, stride, proj, mul)
    return np.concatenate((z, pts))

ALLOWED["lorenz"] = op_lorenz

@njit(cache=True, nogil=True, fastmath=True)
def _clifford_iter(N, a, b, c, d, x0, y0, burn, stride, mul):
    total = burn + N * stride
    out   = np.empty(N, dtype=np.complex128)
    x = x0; y = y0; k = 0
    for i in range(total):
        xn = np.sin(a * y) + c * np.cos(a * x)
        yn = np.sin(b * x) + d * np.cos(b * y)
        x, y = xn, yn
        if i >= burn and ((i - burn) % stride == 0):
            out[k] = mul * (x + 1j*y)
            k += 1
            if k >= N: break
    return out

def op_clifford(z, a, state):
    N   = int(a[0].real) or 300_000
    A   = a[1].real or -1.4
    B   = a[2].real or  1.6
    C   = a[3].real or  1.0
    D   = a[4].real or  0.7
    burn   = int(a[5].real) or 1_000
    stride = int(a[6].real) or 1
    z0     = a[7]            or (0.1+0.1j)  # x0+iy0
    mul    = a[8].real or 1.0

    pts = _clifford_iter(N, A, B, C, D, z0.real, z0.imag, burn, stride, mul)
    return np.concatenate((z, pts))

ALLOWED["clifford"] = op_clifford

@njit(cache=True, nogil=True, fastmath=True)
def _henon_iter(N, a, b, x0, y0, burn, stride, mul):
    total = burn + N * stride
    out   = np.empty(N, dtype=np.complex128)
    x = x0; y = y0; k = 0
    for i in range(total):
        xn = 1.0 - a * x * x + y
        yn = b * x
        x, y = xn, yn
        if i >= burn and ((i - burn) % stride == 0):
            out[k] = mul * (x + 1j*y)
            k += 1
            if k >= N: break
    return out

def op_henon(z, a, state):
    """
    Hénon map generator (appends complex points).
      a[0]=N       (default 300_000)
      a[1]=a_param (default 1.4)
      a[2]=b_param (default 0.3)
      a[3]=burn    (default 1_000)
      a[4]=stride  (default 1)
      a[5]=z0      (x0+iy0, default 0.0+0.0j)
      a[6]=mul     (default 1.0)
    """
    N      = int(a[0].real) or 300_000
    A      = a[1].real or 1.4
    B      = a[2].real or 0.3
    burn   = int(a[3].real) or 1_000
    stride = int(a[4].real) or 1
    z0     = a[5]  # complex seed
    mul    = a[6].real or 1.0

    pts = _henon_iter(N, A, B, z0.real, z0.imag, burn, stride, mul)
    return np.concatenate((z, pts))

ALLOWED["henon"] = op_henon

@njit(cache=True, nogil=True, fastmath=True)
def _dejong_iter(N, a, b, c, d, x0, y0, burn, stride, mul):
    total = burn + N*stride
    out = np.empty(N, np.complex128)
    x = x0; y = y0; k = 0
    for i in range(total):
        xn = np.sin(a*y) - np.cos(b*x)
        yn = np.sin(c*x) - np.cos(d*y)
        x, y = xn, yn
        if i >= burn and ((i-burn) % stride == 0):
            out[k] = mul*(x + 1j*y); k += 1
            if k >= N: break
    return out

def op_dejong(z, a, state):
    N      = int(a[0].real) or 300_000
    A      = a[1].real or  2.01
    B      = a[2].real or -2.53
    C      = a[3].real or  1.61
    D      = a[4].real or -0.33
    burn   = int(a[5].real) or 1_000
    stride = int(a[6].real) or 1
    z0     = a[7]               # complex seed
    mul    = a[8].real or 1.0
    pts = _dejong_iter(N, A, B, C, D, z0.real, z0.imag, burn, stride, mul)
    return np.concatenate((z, pts))

ALLOWED["dejong"] = op_dejong

@njit(cache=True, nogil=True, fastmath=True)
def _tinker_iter(N, a, b, c, d, x0, y0, burn, stride, mul):
    total = burn + N*stride
    out = np.empty(N, np.complex128)
    x = x0; y = y0; k = 0
    for i in range(total):
        xn = x*x - y*y + a*x + b*y
        yn = 2*x*y + c*x + d*y
        x, y = xn, yn
        if i >= burn and ((i-burn) % stride == 0):
            out[k] = mul*(x + 1j*y); k += 1
            if k >= N: break
    return out

def op_tinker(z, a, state):
    N      = int(a[0].real) or 300_000
    A      = a[1].real or  0.9
    B      = a[2].real or -0.6013
    C      = a[3].real or  2.0
    D      = a[4].real or  0.5
    burn   = int(a[5].real) or 1_000
    stride = int(a[6].real) or 1
    z0     = a[7]               # complex seed
    mul    = a[8].real or 1.0
    pts = _tinker_iter(N, A, B, C, D, z0.real, z0.imag, burn, stride, mul)
    return np.concatenate((z, pts))

ALLOWED["tinker"] = op_tinker

@njit(cache=True, nogil=True, fastmath=True)
def _gm_iter(N, a, b, x0, y0, burn, stride, mul):
    total = burn + N*stride
    out = np.empty(N, np.complex128)
    x = x0; y = y0; k = 0
    for i in range(total):
        f = a*x + (2*(1 - a)*x*x) / (1 + x*x)
        xn = y + f
        yn = -x + b*f
        x, y = xn, yn
        if i >= burn and ((i-burn) % stride == 0):
            out[k] = mul*(x + 1j*y); k += 1
            if k >= N: break
    return out

def op_gumira(z, a, state):
    N      = int(a[0].real) or 300_000
    A      = a[1].real or 0.008
    B      = a[2].real or 0.05
    burn   = int(a[3].real) or 1_000
    stride = int(a[4].real) or 1
    z0     = a[5]               # complex seed
    mul    = a[6].real or 1.0
    pts = _gm_iter(N, A, B, z0.real, z0.imag, burn, stride, mul)
    return np.concatenate((z, pts))

ALLOWED["gumira"] = op_gumira

@njit(cache=True, nogil=True, fastmath=True)
def _ginger_iter(N, x0, y0, burn, stride, mul):
    total = burn + N*stride
    out = np.empty(N, np.complex128)
    x = x0; y = y0; k = 0
    for i in range(total):
        xn = 1.0 - y + abs(x)
        yn = x
        x, y = xn, yn
        if i >= burn and ((i-burn) % stride == 0):
            out[k] = mul*(x + 1j*y); k += 1
            if k >= N: break
    return out

def op_ginger(z, a, state):
    N      = int(a[0].real) or 300_000
    burn   = int(a[1].real) or 1_000
    stride = int(a[2].real) or 1
    z0     = a[3]               # complex seed
    mul    = a[4].real or 1.0
    pts = _ginger_iter(N, z0.real, z0.imag, burn, stride, mul)
    return np.concatenate((z, pts))

ALLOWED["ginger"] = op_ginger

# ==================================================
# self-similar
# ==================================================

def _resample_by_arclength(p, N):
    d = np.abs(np.diff(p))
    s = np.concatenate(([0.0], np.cumsum(d)))
    t = np.linspace(0.0, s[-1], N)
    j = np.searchsorted(s, t, side="right") - 1
    j = np.clip(j, 0, len(p)-2)
    w = (t - s[j]) / (s[j+1] - s[j] + 1e-18)
    return p[j] + w * (p[j+1] - p[j])

def _normalize01(p: np.ndarray) -> np.ndarray:
    xr = p.real; yr = p.imag
    rx = np.ptp(xr); ry = np.ptp(yr)
    if rx > 0: xr = (xr - xr.min()) / rx
    else:      xr = xr - xr.min()
    if ry > 0: yr = (yr - yr.min()) / ry
    else:      yr = yr - yr.min()
    return xr + 1j * yr

def _points_from_mask(mask: np.ndarray, xmin, xmax, ymin, ymax) -> np.ndarray:
    j_idx, i_idx = np.nonzero(mask)
    n = mask.shape[0]
    x = xmin + (xmax - xmin) * (i_idx.astype(np.float64) / (n - 1))
    y = ymin + (ymax - ymin) * (j_idx.astype(np.float64) / (n - 1))
    return x + 1j * y

def _downsample_points(p: np.ndarray, N: int) -> np.ndarray:
    if p.size <= N:
        return p
    idx = np.linspace(0, p.size - 1, N, dtype=np.int64)
    return p[idx]

def hilbert_curve(order: int) -> np.ndarray:
    """
    Generate the Hilbert curve of given order as complex points in [0,1]×[0,1].
    Total points = 4**order.
    """
    n = 2 ** order
    N = n * n
    x = np.zeros(N, np.int64)
    y = np.zeros(N, np.int64)
    for i in range(N):
        xi, yi, t = 0, 0, i
        s = 1
        while s < n:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            if ry == 0:
                if rx == 1:
                    xi, yi = s - 1 - xi, s - 1 - yi
                xi, yi = yi, xi
            xi += s * rx
            yi += s * ry
            t //= 4
            s *= 2
        x[i] = xi
        y[i] = yi
    return (x / (n - 1)) + 1j * (y / (n - 1))

def op_hilbert(z, a, state):
    """
    a[0] = target point count (default 1e3)
    """
    Nreq  = int(a[0].real) or 1000

    # Smallest order with >= Nreq points: 4^k >= Nreq  =>  k >= log4(Nreq)
    k = max(1, int(math.ceil( math.log(Nreq, 4) )))
    pts = hilbert_curve(k)                 # length = 4**k

    if pts.size > Nreq:                    # cap to exactly Nreq (uniform by index)
        pts = _resample_by_arclength(pts, Nreq)

    return np.concatenate((z, pts))

ALLOWED["hilbert"] = op_hilbert

# --- Moore curve via L-system; returns CLOSED polyline (start==end) normalized to [0,1]x[0,1] ---
def moore_curve(order: int) -> np.ndarray:
    # build string with simultaneous rewriting
    seq = "LFL+F+LFL"
    for _ in range(max(0, int(order))):
        out = []
        for ch in seq:
            if ch == "L":
                out.append("-RF+LFL+FR-")
            elif ch == "R":
                out.append("+LF-RFR-FL+")
            else:
                out.append(ch)
        seq = "".join(out)

    pos = 0.0 + 0.0j
    ang = 0.0  # radians; 0 = +x
    pts = [pos]
    for c in seq:
        if c == "F":
            pos += (np.cos(ang) + 1j*np.sin(ang))
            pts.append(pos)
        elif c == "+": ang += np.pi/2
        elif c == "-": ang -= np.pi/2
        # ignore L/R
    arr = np.asarray(pts, dtype=np.complex128)

    # normalize to [0,1] x [0,1]
    rx = np.ptp(arr.real); ry = np.ptp(arr.imag)
    if rx > 0: arr = (arr.real - arr.real.min())/rx + 1j*arr.imag
    if ry > 0: arr = arr.real + 1j*(arr.imag - arr.imag.min())/ry

    # drop duplicate closing point
    if arr.size > 1 and arr[0] == arr[-1]:
        arr = arr[:-1]

    return arr

def op_moore(z, a, state):
    """
    Moore (closed Hilbert) curve.
      a[0] = target points (default 1e3)  -> generates >=N, then resamples to exactly N
    """
    Nreq = int(a[0].real) or 1000

    # For Moore, number of segments per order k equals 4**k (like Hilbert),
    # so closed polyline points = 4**k + 1.
    k = max(1, int(math.ceil(math.log(max(Nreq-1, 1), 4))))
    pts = moore_curve(k)  # closed, normalized

    # resample to exactly Nreq points on the closed path
    if pts.size != Nreq:
        pts = _resample_by_arclength(pts, Nreq)

    return np.concatenate((z, pts))

ALLOWED["moore"] = op_moore

# --- Gosper / flowsnake curve; returns open polyline normalized to [0,1]×[0,1] ---
def gosper_curve(order: int, step: float = 1.0) -> np.ndarray:
    order = max(0, int(order))
    seq = "A"
    for _ in range(order):
        out = []
        for ch in seq:  # simultaneous rewrite
            if ch == "A":
                out.append("A+B++B-A--AA-B+")
            elif ch == "B":
                out.append("-A+BB++B+A--A-B")  # ✅ fixed rule
            else:
                out.append(ch)
        seq = "".join(out)

    # draw using complex direction with 60° rotations
    pos = 0.0 + 0.0j
    dirc = 1.0 + 0.0j
    rot  = np.exp(1j * (np.pi / 3.0))  # 60°
    pts = [pos]
    for ch in seq:
        if ch == "A" or ch == "B":
            pos += step * dirc
            pts.append(pos)
        elif ch == "+": dirc *= rot
        elif ch == "-": dirc /= rot

    arr = np.asarray(pts, dtype=np.complex128)

    # normalize to [0,1]×[0,1]
    xr, yr = arr.real, arr.imag
    rx, ry = np.ptp(xr), np.ptp(yr)
    if rx > 0: xr = (xr - xr.min()) / rx
    else:      xr = xr - xr.min()
    if ry > 0: yr = (yr - yr.min()) / ry
    else:      yr = yr - yr.min()
    out = xr + 1j * yr

    # ensure open
    if out.size > 1 and out[0] == out[-1]:
        out = out[:-1]
    return out

def op_gosper(z, a, state):
    Nreq = int(a[0].real) or 1000
    step = a[1].real or 1.0
    # native points ≈ 7**order + 1
    order = max(1, int(math.ceil(math.log(max(Nreq - 1, 1), 7))))
    pts = gosper_curve(order, step=step)

    # optional: resample to exactly Nreq (keep if your pipeline expects exact N)
    # if pts.size != Nreq:
    #     pts = _resample_by_arclength(pts, Nreq)

    return np.concatenate((z, pts))

ALLOWED["gosper"] = op_gosper

# --- Levy C curve (open polyline), normalized to [0,1]×[0,1] ---
def levy_c_curve(order: int) -> np.ndarray:
    order = max(0, int(order))
    z = np.array([0.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128)
    rot = (1.0 + 1.0j) / 2.0  # rotate +45°, scale 1/√2

    for _ in range(order):
        p0 = z[:-1]            # length m
        p1 = z[1:]             # length m
        a  = p1 - p0
        q  = p0 + a * rot      # length m

        out = np.empty(p0.size * 2 + 1, dtype=np.complex128)  # length 2m+1
        out[0:-1:2] = p0       # fill positions 0,2,4,...,2m-2  (m slots)
        out[1:-1:2] = q        # fill positions 1,3,5,...,2m-1  (m slots)
        out[-1]     = p1[-1]   # final endpoint
        z = out

    # normalize to [0,1]×[0,1]
    rx = np.ptp(z.real); ry = np.ptp(z.imag)
    if rx > 0: z = (z.real - z.real.min())/rx + 1j*z.imag
    if ry > 0: z = z.real + 1j*(z.imag - z.imag.min())/ry
    return z

def op_levyc(z, a, state):
    Nreq = int(a[0].real) or 1000
    order = max(1, int(np.ceil(np.log2(max(Nreq - 1, 1)))))  # points = 2**order + 1
    pts = levy_c_curve(order)

    # optional: resample to exactly Nreq (by arclength) if you use that helper
    # if pts.size != Nreq:
    #     pts = _resample_by_arclength(pts, Nreq)

    return np.concatenate((z, pts))

ALLOWED["levyc"] = op_levyc

def _compact_bits(v: np.ndarray) -> np.ndarray:
    """Remove interleaved zeros: ..0a0b0c0d -> abcd (for 2D morton)."""
    v = v & np.uint64(0x5555555555555555)
    v = (v | (v >> 1))  & np.uint64(0x3333333333333333)
    v = (v | (v >> 2))  & np.uint64(0x0f0f0f0f0f0f0f0f)
    v = (v | (v >> 4))  & np.uint64(0x00ff00ff00ff00ff)
    v = (v | (v >> 8))  & np.uint64(0x0000ffff0000ffff)
    v = (v | (v >> 16)) & np.uint64(0x00000000ffffffff)
    return v.astype(np.uint64, copy=False)

def morton_decode2D(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Given Morton codes m (uint64), return x,y (uint32) on a 2^k grid."""
    m = m.astype(np.uint64, copy=False)
    x = _compact_bits(m)
    y = _compact_bits(m >> 1)
    return x.astype(np.uint32), y.astype(np.uint32)

def zcurve_points(order: int) -> np.ndarray:
    """
    Morton (Z) traversal over a 2^order x 2^order grid.
    Returns complex points normalized to [0,1]×[0,1], open polyline (no repeat).
    """
    order = max(1, int(order))
    n = 1 << order           # grid side
    N = n * n                # total points
    m = np.arange(N, dtype=np.uint64)
    x, y = morton_decode2D(m)
    # normalize to [0,1]
    if n > 1:
        z = (x / (n - 1)) + 1j * (y / (n - 1))
    else:
        z = x.astype(np.float64) + 1j * y.astype(np.float64)
    return z.astype(np.complex128, copy=False)

def op_zcurve(z, a, state):
    """
    Z-curve (Morton order) over a 2^k x 2^k grid.
      a[0] = target point count (default 1e3)
    Behavior: choose smallest k with 4^k >= Nreq, then uniformly index-subsample to exactly Nreq.
    """
    Nreq = int(a[0].real) or 1000
    # 4^k >= Nreq  ->  k >= log4(Nreq)
    k = max(1, int(math.ceil(math.log(max(Nreq,1), 4))))
    pts = zcurve_points(k)  # length = 4**k

    if pts.size > Nreq:
        idx = np.linspace(0, pts.size - 1, Nreq, dtype=np.int64)
        pts = pts[idx]

    return np.concatenate((z, pts))

ALLOWED["zcurve"] = op_zcurve

def dragon_curve(iterations: int, step: float = 1.0) -> np.ndarray:
    """
    Generate the Heighway dragon as complex points.
    """
    seq = "FX"
    for _ in range(iterations):
        seq = seq.replace("X", "X+YF+").replace("Y", "-FX-Y")
    pos = 0.0 + 0.0j
    ang = 0.0
    pts = [pos]
    for c in seq:
        if c == "F":
            pos += step * np.exp(1j * ang)
            pts.append(pos)
        elif c == "+":
            ang += np.pi/2
        elif c == "-":
            ang -= np.pi/2
    arr = np.array(pts, np.complex128)
    arr /= np.max(np.abs(arr))  # normalize to ~unit box
    return arr

def op_dragon(z,a,state):
    order = int(a[0].real) or 12
    step  = a[1].real or 1.0
    pts = dragon_curve(order, step)
    return np.concatenate((z, pts))
ALLOWED["dragon"] = op_dragon

def _koch_once(poly: np.ndarray) -> np.ndarray:
    """
    One refinement step on a closed polyline (complex points).
    Returns a new closed polyline.
    """
    n = poly.size
    out = []
    two_pi_over_3 = 2.0 * np.pi / 3.0
    rot60 = np.cos(np.pi/3.0) + 1j*np.sin(np.pi/3.0)

    for i in range(n - 1):
        p0 = poly[i]
        p1 = poly[i + 1]
        a  = p1 - p0
        pA = p0 + a / 3.0
        pB = pA + (a / 3.0) * rot60
        pC = p0 + 2.0 * a / 3.0
        out.extend([p0, pA, pB, pC])
    out.append(poly[-1])  # close last vertex
    return np.asarray(out, dtype=np.complex128)

def _koch_snowflake(iterations: int, side: float, center: complex, turn: float) -> np.ndarray:
    """
    Build Koch snowflake as a closed complex polyline.
      iterations >= 0
      side: side length of the initial equilateral triangle
      center: complex center of the triangle
      turn: rotation in turns (0..1), 0 = one vertex on +x axis
    """
    # initial equilateral triangle (circumradius R = side / sqrt(3))
    theta0 = 2.0 * np.pi * (turn or 0.0)
    angles = theta0 + np.array([0.0, 2.0*np.pi/3.0, 4.0*np.pi/3.0], dtype=np.float64)
    R = (side or 1.0) / np.sqrt(3.0)
    verts = center + R * (np.cos(angles) + 1j*np.sin(angles))
    poly = np.concatenate([verts, verts[:1]])  # close

    it = int(iterations) if iterations is not None else 0
    for _ in range(max(0, it)):
        poly = _koch_once(poly)
    return poly

def op_koch(z, a, state):
    """
    Koch snowflake generator (appends ~N points).
      a[0] = target N points (default 1e3)
      a[1] = side
      a[2] = center (complex)
      a[3] = turn (0..1)
    """
    N     = int(a[0].real) or 1000
    side  = a[1].real or 1.0
    cen   = a[2]
    turn  = a[3].real or 0.0

    # iterations needed to reach at least N points: 3*4^it + 1 >= N
    # => it >= log4((N-1)/3)
    x  = max((N - 1) / 3.0, 1.0)             # guard small N
    it = int(np.ceil(np.log(x) / np.log(4.0)))

    pts = _koch_snowflake(it, side, cen, turn)   # returns closed polyline

    # Optional: subsample to exactly N points (spread along the path)
    if pts.size > N:
        idx = np.linspace(0, pts.size - 1, N, dtype=np.int64)
        pts = pts[idx]

    return np.concatenate((z, pts))

ALLOWED["koch"] = op_koch

@njit(cache=True, nogil=True, fastmath=True)
def _sierp_tri_iter(N, burn, x0, y0):
    out = np.empty(N, np.complex128)
    # vertices of equilateral triangle
    V = np.array([1+0j, np.cos(2*np.pi/3)+1j*np.sin(2*np.pi/3), 
                        np.cos(4*np.pi/3)+1j*np.sin(4*np.pi/3)], np.complex128)
    z = x0 + 1j*y0
    k = 0
    for i in range(burn + N):
        v = V[int(np.random.random()*3.0)]
        z = 0.5*(z + v)          # midpoint to a random vertex
        if i >= burn:
            out[k] = z; k += 1
    return out

def op_sierptri(z, a, state):
    N    = int(a[0].real) or 300_000
    burn = int(a[1].real) or 200
    z0   = a[2] or (0.1+0.1j)
    pts  = _sierp_tri_iter(N, burn, z0.real, z0.imag)
    return np.concatenate((z, pts))

ALLOWED["sierptri"] = op_sierptri

@njit(cache=True, nogil=True, fastmath=True)
def _sierp_carpet_iter(N, burn, x0, y0):
    out = np.empty(N, np.complex128)
    # eight affine maps: (x,y) -> ( (x+ox)/3, (y+oy)/3 ) with center excluded
    O = np.array([(0,0),(1,0),(2,0),(0,1),(2,1),(0,2),(1,2),(2,2)], np.int64)
    z = x0 + 1j*y0
    k = 0
    for i in range(burn + N):
        ox, oy = O[int(np.random.random()*8.0)]
        z = (z + (ox + 1j*oy)) / 3.0
        if i >= burn:
            out[k] = z; k += 1
    return out

def op_sierpcarpet(z, a, state):
    N    = int(a[0].real) or 300_000
    burn = int(a[1].real) or 200
    z0   = a[2] or (0.1+0.1j)
    pts  = _sierp_carpet_iter(N, burn, z0.real, z0.imag)
    return np.concatenate((z, pts))

ALLOWED["sierpcarpet"] = op_sierpcarpet

@njit(cache=True)
def _peano_points(order: int) -> np.ndarray:
    """
    Peano serpentine fill over a 3^order × 3^order grid.
    Open polyline, points = (3^order)^2.
    """
    order = max(0, int(order))
    n = 1
    for _ in range(order):
        n *= 3

    N = n * n
    xs = np.empty(N, np.float64)
    ys = np.empty(N, np.float64)

    idx = 0
    for j in range(n):
        if (j & 1) == 0:
            # left -> right
            for i in range(n):
                xs[idx] = i
                ys[idx] = j
                idx += 1
        else:
            # right -> left
            for i in range(n-1, -1, -1):
                xs[idx] = i
                ys[idx] = j
                idx += 1

    # normalize to [0,1]×[0,1]
    if n > 1:
        xs = (xs - 0.0) / (n - 1)
        ys = (ys - 0.0) / (n - 1)
    return xs + 1j * ys


def op_peano(z, a, state):
    """
    Peano serpentine (3^k × 3^k raster snake), Numba-backed.
      a[0] = target points (default 1e3)
    """
    Nreq = int(a[0].real) or 1000
    # native points = (3^k)^2 => 9^k, choose smallest k with 9^k >= Nreq
    k = max(1, int(math.ceil(math.log(max(Nreq, 1), 9))))
    pts = _peano_points(k)

    # cap to exactly Nreq (optional)
    if pts.size > Nreq:
        pts = _resample_by_arclength(pts, Nreq)

    return np.concatenate((z, pts))

ALLOWED["peano"] = op_peano

# ---------- core: 3^k × 3^k Peano serpentine path (continuous) ----------
@njit(cache=True)
def _peano_poly_points(order: int) -> np.ndarray:
    """
    Peano serpentine over a 3^order × 3^order grid.
    Returns an OPEN polyline: length = (3^order)^2 points.
    Consecutive points are adjacent grid centers → continuous path.
    """
    order = max(0, int(order))
    n = 1
    for _ in range(order):
        n *= 3

    N = n * n
    xs = np.empty(N, np.float64)
    ys = np.empty(N, np.float64)

    idx = 0
    for j in range(n):
        if (j & 1) == 0:
            # left → right
            for i in range(n):
                xs[idx] = i
                ys[idx] = j
                idx += 1
        else:
            # right → left
            for i in range(n - 1, -1, -1):
                xs[idx] = i
                ys[idx] = j
                idx += 1

    # normalize to [0,1]×[0,1] (grid centers mapped to unit square)
    if n > 1:
        xs = xs / (n - 1)
        ys = ys / (n - 1)
    return xs + 1j * ys

def op_peano_poly(z, a, state):
    """
    Peano serpentine polyline (continuous path).
      a[0] = target points (default 1e3)
    Behavior: choose smallest k with 9^k >= Nreq, generate native path (length 9^k),
              then resample by arclength to exactly Nreq for uniform density.
    """
    Nreq = int(a[0].real) or 1000
    k = max(1, int(math.ceil(math.log(max(Nreq, 1), 9))))  # 9^k >= Nreq

    pts = _peano_poly_points(k)      # native length = 9**k
    if pts.size != Nreq:
        pts = _resample_by_arclength(pts, Nreq)

    return np.concatenate((z, pts))

ALLOWED["peano_poly"] = op_peano_poly

# ---- core: deterministic centers via base-5 digit expansion ----
# mapping for the 5 kept cells in each 3x3: center, +x, -x, +y, -y
# offsets are applied with scales 1/3, 1/3^2, ..., 1/3^order
_OFFS5_X = np.array([0.0,  1.0, -1.0,  0.0,  0.0], dtype=np.float64)
_OFFS5_Y = np.array([0.0,  0.0,  0.0,  1.0, -1.0], dtype=np.float64)

@njit(cache=True)
def _vicsek_points(order: int) -> np.ndarray:
    """
    Vicsek (cross) fractal points at recursion depth 'order'.
    Returns complex128, length = 5**order (open point set, not a polyline).
    """
    order = max(0, int(order))
    # total points = 5**order
    npts = 1
    for _ in range(order):
        npts *= 5

    xs = np.empty(npts, np.float64)
    ys = np.empty(npts, np.float64)

    for i in range(npts):
        t = i
        x = 0.0
        y = 0.0
        scale = 1.0 / 3.0
        for _ in range(order):
            d = t % 5            # which subcell this digit picks
            t //= 5
            x += _OFFS5_X[d] * scale
            y += _OFFS5_Y[d] * scale
            scale /= 3.0
        xs[i] = x
        ys[i] = y

    # normalize to [0,1]×[0,1]
    xr = xs
    yr = ys
    rx = xr.max() - xr.min()
    ry = yr.max() - yr.min()
    if rx > 0.0:
        xr = (xr - xr.min()) / rx
    else:
        xr = xr - xr.min()
    if ry > 0.0:
        yr = (yr - yr.min()) / ry
    else:
        yr = yr - yr.min()

    return xr + 1j * yr

#

# ---- op ----
def op_vicsek(z, a, state):
    """
    Vicsek (cross) fractal as a point set (not a polyline).
      a[0] = target points (default 1e3)
    We choose the smallest 'order' with 5**order >= Nreq and, if necessary,
    downsample to exactly Nreq (uniform by index).
    """
    Nreq = int(a[0].real) or 1000
    order = max(0, int(math.ceil(math.log(max(Nreq, 1), 5))))

    pts = _vicsek_points(order)      # length = 5**order
    if pts.size > Nreq:
        pts = _downsample_points(pts, Nreq)

    return np.concatenate((z, pts))

ALLOWED["vicsek"] = op_vicsek

# ---------------- julia set ----------------

@njit("int32(complex128, complex128, int32, int32, float64)", fastmath=True, cache=True)
def _julia_escape_single(
    z0: np.complex128,
    c: np.complex128,
    eqn: int = 0,
    max_iter: int = 400,
    bailout2: float = 4.0,
) -> np.int32:
    z = z0
    for k in range(max_iter):
        if eqn==0:
            z =  z*z*z*z*z*z - z*z*z*z + c
        elif eqn==1:
            z =  np.exp(1j*2*np.pi*np.abs(z)) + c
        else:
            z = z*z + c
        if np.abs(z) > bailout2:
            return k
    return max_iter

# vectorized, parallel caller
@njit("int32[:](complex128[:], complex128, int32, int32, float64)",
      parallel=True, fastmath=True, cache=True)
def julia_escape_vec(z0, c, eqn, max_iter, bailout2):
    n = z0.size
    out = np.empty(n, np.int32)
    for i in prange(n):
        out[i] = _julia_escape_single(z0[i], c, eqn, max_iter, bailout2)
    return out

@njit("complex128[:](int64,float64,complex128)",fastmath=True, cache=True)
def _points(N,w:float=1,center:complex=0+0j):
    re = -w + 2 * w * np.random.rand(N)
    im = -w + 2 * w * np.random.rand(N)
    return re + 1j*im + center

#
MAX_ITER_CONST = np.int32(1000)
BAILOUT2_CONST = np.float64(4.0)
@njit("complex128[:](int64, complex128, float64, complex128, int32, int32)", fastmath=True, cache=True)
def julia_sample(N, c, w, center, thresh, eqn):
    z0 = _points(N, w,center)
    iters = julia_escape_vec(z0, c, eqn, MAX_ITER_CONST, BAILOUT2_CONST)
    keep = 0
    for i in range(N):
        if iters[i] > thresh:
            keep += 1
    out = np.empty(keep, np.complex128)
    j = 0
    for i in range(N):
        if iters[i] > thresh:
            out[j] = z0[i]
            j += 1
    return out

def julia(N: int, c: np.complex128= -0.8 + 0.156j,maxi:int=200,w:float=1.5,eqn:int=0) -> np.ndarray:
    N = int(N)
    out  = np.zeros(N, np.complex128)
    kept = 0
    # 1) probe sample
    boost = 10
    s = julia_sample(
        np.int64(boost*N), 
        np.complex128(c), 
        np.float64(w), 
        np.complex128(0+0j), 
        np.int32(maxi),
        np.int32(eqn)
    )
    take = min(s.size, N)
    if take:
        out[:take] = s[:take]
        kept += take
    else:
        return out
    p = max(0.01, (s.size / (boost*N)))
    draw = 2*int(math.ceil(N / p))
    print(f"p: {p}")
    if draw < 1: draw = 1
    center = np.mean(out[:kept])
    w = 0.5 * max(
        np.ptp(out[:kept].real),
        np.ptp(out[:kept].imag)
    ) * 1.5
    rounds = 0
    while kept < N:
        rounds += 1
        need = (N - kept)
        s = julia_sample(np.int64(draw), np.complex128(c), np.float64(w),np.complex128(center), np.int32(300),int(eqn))
        take = min(s.size, need)
        if take:
            out[kept:kept + take] = s[:take]
            kept += take
        if rounds > 10 : break
        center = np.mean(out[:kept])
        w = 0.5 * max(
            np.ptp(out[:kept].real),
            np.ptp(out[:kept].imag)
        ) * 1.15
    print(f"kept: {kept}")
    return out[:min(kept,N)]  # exactly N filled unless the defensive break triggered

def op_julia(z, a, state):
    N = int(a[0].real) or 1e6   
    if N<1 : return z
    c = a[1] or   -0.8 + 0.156j
    maxi = int(a[2].real) or 200
    w = a[3].real or 1.5
    eqn = int(a[4].real)
    pts = julia(N,c,maxi,w,eqn)
    return np.concatenate((z, pts))


ALLOWED["julia"] = op_julia


# fern IFS system

@njit(cache=True, nogil=True, fastmath=True)
def _fern_iter(N, burn, x0, y0):
    out = np.empty(N, np.complex128)
    x = x0; y = y0; k = 0
    for i in range(burn + N):
        r = np.random.random()
        if r < 0.01:
            xn = 0.0        
            yn = 0.16*y
        elif r < 0.86:
            xn = 0.85*x + 0.04*y;    yn = -0.04*x + 0.85*y + 1.6
        elif r < 0.93:
            xn = 0.20*x - 0.26*y;    yn = 0.23*x + 0.22*y + 1.6
        else:
            xn = -0.15*x + 0.28*y;   yn = 0.26*x + 0.24*y + 0.44
        x, y = xn, yn
        if i >= burn:
            out[k] = (x + 1j*y) * (1/6)   # gentle scale to fit unit-ish box
            k += 1
    return out



def op_fern(z, a, state):
    N    = int(a[0].real) or 300_000
    burn = int(a[1].real) or 100
    z0   = a[2] or (0.0+0.0j)
    pts  = _fern_iter(N, burn, z0.real, z0.imag)
    return np.concatenate((z, pts))

ALLOWED["fern"] = op_fern

def _dragon_curve(iterations: int, step: float) -> np.ndarray:
    seq = "FX"
    for _ in range(iterations):
        out = []
        for ch in seq:
            if ch == "X":
                out.append("X+YF+")
            elif ch == "Y":
                out.append("-FX-Y")
            else:
                out.append(ch)
        seq = "".join(out)

    pos = 0.0 + 0.0j
    ang = 0.0
    pts = [pos]
    for c in seq:
        if c == "F":
            pos += step * (np.cos(ang) + 1j*np.sin(ang))
            pts.append(pos)
        elif c == "+": ang += np.pi/2
        elif c == "-": ang -= np.pi/2

    arr = np.asarray(pts, np.complex128)
    # normalize to ~unit box (optional)
    arr /= (np.max(arr.real) - np.min(arr.real) + 1e-12)
    return arr

def op_dragon(z, a, state):
    Nreq = int(a[0].real) or 1000
    step = a[1].real or 1.0
    it = int(np.ceil(np.log2(max(1, Nreq - 1))))  # smallest it with ≥ Nreq points
    pts = _dragon_curve(it, step)
    # If you want ≤ Nreq strictly, truncate (keeps path prefix)
    if pts.size > Nreq:
        pts = pts[:Nreq]
    return np.concatenate((z, pts))

ALLOWED["dragon"] = op_dragon

# ---------- cartesian expansion: random and grid----------

# CARtesian Unifrom SQuare: carusq
def op_carusq(z, a, state):
    if z.size<1: return z
    k  = _frozen_len(state)
    if k>=z.size: return z
    head, tail = z[:k], z[k:]
    N  = int(a[0].real) or 100
    m = int(np.ceil(np.sqrt(N)))
    w  = (float(a[1].real) or 1.8)/m
    u = RNG.uniform(-w, +w, size=(m*m, tail.size))
    v = RNG.uniform(-w, +w, size=(m*m, tail.size))
    out = (tail[None, :] + u + 1j*v).reshape(m*m * tail.size)
    return np.concatenate((head, out))

ALLOWED["carusq"] = op_carusq

# CARtesian Grid SQuare: cargsq
def op_cargsq(z, a, state):
    if z.size < 1: return z
    k = _frozen_len(state)
    if k >= z.size: return z
    head, tail = z[:k], z[k:]
    N = int(a[0].real) or 100
    m = int(np.ceil(np.sqrt(N)))
    w = (float(a[1].real) or 1.8) / m
    xs = np.linspace(-w, +w, m, dtype=np.float64)
    ys = np.linspace(-w, +w, m, dtype=np.float64)
    GX, GY = np.meshgrid(xs, ys, indexing="xy")
    grid = (GX + 1j * GY).ravel()[:m*m]  # length n*n, the nearest square
    out = (tail[None, :] + grid[:, None]).reshape(m * m * tail.size)
    return np.concatenate((head, out))

ALLOWED["cargsq"] = op_cargsq

# CARtesian Unifrom DiSK: carudsk
def op_carudsk(z, a, state):
    if z.size < 1: return z
    k = _frozen_len(state)
    if k >= z.size: return z
    head, tail = z[:k], z[k:]
    N   = int(a[0].real) or 100
    if  N <= 0: return z
    r   = float(a[1].real) or 0.141 / math.sqrt(tail.size)
    dth = float(a[2].real) or 0.5
    u = RNG.random(size=(N, tail.size))
    v = RNG.random(size=(N, tail.size))
    p = r * (u**dth) * np.exp(1j * 2.0 * np.pi * v)
    out = (tail[None, :] + p).reshape(N * tail.size)
    return np.concatenate((head, out))

ALLOWED["carudsk"] = op_carudsk

# CARtesian GRid DiSK: cargdsk
def op_cargdsk(z, a, state):
    if z.size < 1:
        return z
    k = _frozen_len(state)
    if k >= z.size:
        return z

    head, tail = z[:k], z[k:]
    N   = int(a[0].real) or 100
    if N <= 0 or tail.size == 0:
        return z
    r   = float(a[1].real) or (0.141 / math.sqrt(tail.size))
    dth = float(a[2].real) or 0.5

    # Special tiny-N cases
    if N == 1:
        grid = np.array([0.0 + 0.0j], dtype=np.complex128)
    else:
        # Choose ring count so total points ~ N with m_i ~ 2π i
        # Sum_{i=1..K} round(2π i) ≈ π K (K+1) ≈ N-1  => K ≈ sqrt((N-1)/π)
        K = int(np.floor(np.sqrt(max(N - 1, 1) / np.pi)))
        if K < 1:
            # Put all points on one ring
            m_last = N - 1
            radii = np.array([r * (1.0)**dth], dtype=np.float64)
            counts = np.array([m_last], dtype=np.int64)
        else:
            i = np.arange(1, K + 1, dtype=np.int64)
            counts = np.round(2.0 * np.pi * i).astype(np.int64)
            counts[counts < 1] = 1
            total = int(counts.sum())

            target = N - 1  # center absorbs 1 point
            if total > target:
                # reduce from outer rings inward, keeping at least 1 per ring
                need = total - target
                for j in range(K - 1, -1, -1):
                    if need == 0:
                        break
                    take = min(need, max(0, counts[j] - 1))
                    counts[j] -= take
                    need -= take
            elif total < target:
                # add any leftover to the outermost ring
                counts[-1] += (target - total)

            radii = r * (i / float(K))**dth

        # Build grid: include center, then each ring with evenly spaced angles
        pieces = [np.array([0.0 + 0.0j], dtype=np.complex128)]
        for rr, m in zip(radii, counts):
            if m <= 0:
                continue
            theta = (2.0 * np.pi / m) * np.arange(m, dtype=np.float64)
            ring = rr * (np.cos(theta) + 1j * np.sin(theta))
            pieces.append(ring.astype(np.complex128, copy=False))

        grid = np.concatenate(pieces, dtype=np.complex128)
        # Safety: trim or pad (should already be exact N)
        if grid.size > N:
            grid = grid[:N]
        elif grid.size < N:
            pad = np.zeros(N - grid.size, dtype=np.complex128)
            grid = np.concatenate([grid, pad], dtype=np.complex128)

    # Add the same disk-grid to every live point and flatten
    out = (tail[None, :] + grid[:, None]).reshape(N * tail.size)
    return np.concatenate((head, out))

ALLOWED["cargdsk"] = op_cargdsk

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

# ============================================================
# "clip in" means remove inside
# "clip out" means remove outside
# ============================================================

def op_clpindisk(z, a, state):
    r = a[0].real or 1.0
    c = a[1]
    k = _frozen_len(state)
    if k >= z.size: return z  # nothing to filter
    head, tail = z[:k], z[k:]
    keep = np.abs(tail - c) > r
    return np.concatenate((head, tail[keep]))

ALLOWED["clpindisk"] = op_clpindisk

def op_clpoutdisk(z, a, state):
    r = a[0].real or 1.0
    c = a[1]
    k = _frozen_len(state)
    if k >= z.size: return z  # nothing to filter
    head, tail = z[:k], z[k:]
    keep = np.abs(tail - c) < r
    return np.concatenate((head, tail[keep]))

ALLOWED["clpoutdisk"] = op_clpoutdisk

def op_clpinsq(z, a, state):
    r = a[0].real or 1.0
    c = a[1]
    k = _frozen_len(state)
    if k >= z.size: return z  # nothing to filter
    head, tail = z[:k], z[k:]
    dx = tail.real - c.real
    dy = tail.imag - c.imag
    inside = (np.abs(dx) < r) & (np.abs(dy) < r)
    keep = ~inside
    return np.concatenate((head, tail[keep]))

ALLOWED["clpinsq"] = op_clpinsq

def op_clpoutsq(z, a, state):
    r = a[0].real or 1.0
    c = a[1]
    k = _frozen_len(state)
    if k >= z.size: return z  # nothing to filter
    head, tail = z[:k], z[k:]
    dx = tail.real - c.real
    dy = tail.imag - c.imag
    inside = (np.abs(dx) < r) & (np.abs(dy) < r)
    keep = inside
    return np.concatenate((head, tail[keep]))

ALLOWED["clpoutsq"] = op_clpoutsq

def op_clpinrect(z, a, state):
    ll = a[0] 
    ur = a[1]
    k = _frozen_len(state)
    if k >= z.size: return z  # nothing to filter
    head, tail = z[:k], z[k:]
    inside_real = (tail.real > ll.real) & (tail.real<ur.real)
    inside_imag = (tail.imag > ll.imag) & (tail.imag<ur.imag)
    inside = inside_real & inside_imag
    keep = ~inside
    return np.concatenate((head, tail[keep]))

ALLOWED["clpinrect"] = op_clpinrect

def op_clpoutrect(z, a, state):
    ll = a[0] 
    ur = a[1]
    k = _frozen_len(state)
    if k >= z.size: return z  # nothing to filter
    head, tail = z[:k], z[k:]
    inside_real = (tail.real > ll.real) & (tail.real<ur.real)
    inside_imag = (tail.imag > ll.imag) & (tail.imag<ur.imag)
    inside = inside_real & inside_imag
    keep = inside
    return np.concatenate((head, tail[keep]))

ALLOWED["clproutrect"] = op_clpoutrect

# ============================================================
# simple transfomation
# ============================================================

def op_add(z, a, state):
    k = _frozen_len(state)
    if k < z.size:
        z[k:] += a[0]
    return z

ALLOWED["add"] = op_add

def op_ibase(z, a, state):
    k = _frozen_len(state)
    if k < z.size:
        fac = a[0].real or 1
        z[k:] += - 1j*np.min(z[k:].imag)*fac
    return z

ALLOWED["ibase"] = op_ibase

def op_itip(z, a, state):
    k = _frozen_len(state)
    if k < z.size:
        z[k:] += - 1j*np.max(z[k:].imag) * (a[0].real or 1)
    return z

ALLOWED["itip"] = op_itip

def op_mul(z, a, state):
    k = _frozen_len(state)
    if k < z.size: z[k:] *= a[0]
    return z
ALLOWED["mul"] = op_mul

def op_rmul(z, a, state):
    k = _frozen_len(state)
    if k < z.size:
        z[k:] = z[k:].real * a[0] + 1j * z[k:].imag
    return z
ALLOWED["rmul"] = op_rmul

def op_loflip(z, a, state):
    k = _frozen_len(state)
    if k < z.size:
        zz = z[k:]
        mask = zz.imag - a[0] < 0
        zz = np.where(mask, -np.conj(zz), zz)
        z[k:] = zz
    return z
ALLOWED["loflip"] = op_loflip

def op_hiflip(z, a, state):
    k = _frozen_len(state)
    if k < z.size:
        zz = z[k:]
        mask = zz.imag - a[0] > 0
        zz = np.where(mask, -np.conj(zz), zz)
        z[k:] = zz
    return z
ALLOWED["hiflip"] = op_hiflip

def op_imul(z, a, state):
    k = _frozen_len(state)
    if k < z.size:
        zz = z[k:]
        z[k:] = zz.real + 1j * zz.imag * a[0]
    return z
ALLOWED["imul"] = op_imul

# center
def op_center(z, a, state):
    k = _frozen_len(state)
    if k < z.size:
        tail = z[k:]
        z[k:] -= np.mean(tail)
    return z
ALLOWED["center"] = op_center

# scale to mult by mult logical
def op_norm(z, a, state):
    k = _frozen_len(state)
    if k < z.size:
        scale = a[0].real or 1.0
        tail = z[k:]
        max_radius = np.max(np.abs(tail))
        sz = scale * tail / max_radius
        z[k:] = sz
    return z
ALLOWED["norm"] = op_norm

# Global normalization (ignores groups):
def op_gnorm(z, a, state):
    """
    Global normalization (ignores groups):
    scale all points so max(|z|) == scale (default 1.0)
    """
    scale = a[0].real or 1.0
    if z.size == 0:
        return z
    R = np.max(np.abs(z)) or 1.0
    if R != 0.0: z *= (scale / R)
    return z
ALLOWED["gnorm"] = op_gnorm

def op_rot(z, a, state):
    k = _frozen_len(state)
    if k < z.size:
        z[k:] *= np.exp(1j * 2.0 * np.pi * a[0].real)
    return z
ALLOWED["rot"] = op_rot

def op_sqrot(z, a, state):
    k = _frozen_len(state)
    if k < z.size:
        squish = a[0].real or 1.0
        phi = np.exp(1j * 2.0 * np.pi * a[1].real)
        tail = z[k:]
        zz = (tail.real + 1j * tail.imag * squish) * phi
        z[k:] = zz
    return z
ALLOWED["sqrot"] = op_sqrot

# global rotate: ignores groups. at the end.
def op_grot(z, a, state):
    z *=  np.exp(1j * 2.0 * np.pi * a[0].real)
    return z
ALLOWED["grot"] = op_grot

def op_rth(z, a, state):
    k = _frozen_len(state)
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
    k = _frozen_len(state)
    if k < z.size:
        z[k:] = 1j * (1.0 + z[k:]) / (1.0 - z[k:])
        # (optional) protect exact pole at z=1:
        # denom = (1.0 - tail)
        # denom = np.where(denom == 0, 1e-15 + 0j, denom)
        # z[k:] = 1j * (1.0 + tail) / denom
    return z

ALLOWED["toline"] = op_toline

def op_csum(z, a, state):
    k = _frozen_len(state)
    if k < z.size:
        z[k:] = np.cumsum(z[k:])
    return z

ALLOWED["csum"] = op_csum

def op_discr(z, a, state):
    mult = state.get(K_MULT)
    k = 0 if (mult is None) else mult.size
    if k < z.size:
        N = np.rint(a[0].real) or 10
        dth = a[1].real or 1.0
        tail = z[k:]
        xdiscr = np.rint((tail.real) * N)/N
        xdpow = np.sign(tail.real)*(np.abs(xdiscr)**dth)
        ydiscr = np.rint((tail.imag) * N)/N
        ydpow = np.sign(tail.imag)*(np.abs(ydiscr)**dth)
        z[k:] = xdpow + 1j * ydpow
    return z

ALLOWED["discr"] = op_discr

def op_rdscr(z, a, state):
    mult = state.get(K_MULT)
    k = 0 if (mult is None) else mult.size
    if k < z.size:
        N = int(a[0].real) or 10
        dth = a[1].real or 1.0
        tail = z[k:]
        discr = np.rint((tail.real) * N)/N
        dpow = np.sign(tail.real)*(np.abs(discr)**dth)
        z[k:] = dpow + 1j * tail.imag
    return z

ALLOWED["rdscr"] = op_rdscr

def op_idscr(z, a, state):
    mult = state.get(K_MULT)
    k = 0 if (mult is None) else mult.size
    if k < z.size:
        N   = int(a[0].real) or 10
        dth = a[1].real or 1.0

        tail = z[k:]
        x = tail.real
        y = tail.imag

        # 1) discretize rows
        y0 = np.rint(y * N) / N

        # 2) power spacing but preserve vertical span
        ymax = np.max(np.abs(y0)) or 1.0
        yn   = np.abs(y0) / ymax
        y1   = np.sign(y0) * (yn ** dth) * ymax

        # 3) keep circular boundary with horizontal lines:
        #    rescale x within each (old y0 -> new y1) row to match circle of radius R
        R  = np.max(np.abs(tail)) or 1.0
        R2 = R * R
        w0 = np.sqrt(np.clip(R2 - y0*y0, 0.0, None))   # half-width before
        w1 = np.sqrt(np.clip(R2 - y1*y1, 0.0, None))   # half-width after
        s  = np.divide(w1, w0, out=np.ones_like(w0), where=w0 > 0)

        x1 = x * s
        z[k:] = x1 + 1j * y1
    return z

ALLOWED["idscr"] = op_idscr

def op_ddscr(z, a, state):
    mult = state.get(K_MULT)
    k = 0 if (mult is None) else mult.size
    if k < z.size:
        N = int(a[0].real) or 10
        dth = a[1].real or 1.0
        tail = z[k:]
        radius = np.abs(tail)
        theta = np.angle(tail)
        dradius = np.rint( radius * N ) / N
        dpow =  dradius**dth
        z[k:] = dpow*np.exp( 1j * 2 * np.pi * theta )
    return z

ALLOWED["ddscr"] = op_ddscr

# ---------- discretization transforms ----------

def op_pixelate(z, a, state):
    if z.size<1: return z
    k = _frozen_len(state)
    if k >= z.size: return z
    nx = int(a[0].real) or 256
    ny = int(a[0].imag) or nx
    thr = int(a[1].real) or 1
    head, tail = z[:k], z[k:]
    x, y = tail.real, tail.imag
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    if not np.isfinite(xmin + xmax + ymin + ymax): return z  
    if xmax == xmin: xmax = xmin + 1e-12
    if ymax == ymin: ymax = ymin + 1e-12
    H, xedges, yedges = np.histogram2d(
        x, y,
        bins=[nx, ny],
        range=[[xmin, xmax], [ymin, ymax]]
    )
    m = H >= thr
    if not m.any(): return head  # nothing passes threshold
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    ii, jj = np.nonzero(m)
    centers = xc[ii] + 1j * yc[jj]
    return np.concatenate((head, centers))

ALLOWED["pixelate"] = op_pixelate

def op_xorelate(z, a, state):
    if z.size<1: return z
    k = _frozen_len(state)
    if k >= z.size: return z
    nx = int(a[0].real) or 256
    ny = int(a[0].imag) or nx
    thr = int(a[1].real) or 2
    head, tail = z[:k], z[k:]
    x, y = tail.real, tail.imag
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    if not np.isfinite(xmin + xmax + ymin + ymax): return z  # bail if degenerate
    if xmax == xmin: xmax = xmin + 1e-12
    if ymax == ymin: ymax = ymin + 1e-12
    H, xedges, yedges = np.histogram2d(
        x, y,
        bins=[nx, ny],
        range=[[xmin, xmax], [ymin, ymax]]
    )
    m = H % thr
    if not m.any(): return head  # nothing passes threshold
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    ii, jj = np.nonzero(m)
    centers = xc[ii] + 1j * yc[jj]
    return np.concatenate((head, centers))

ALLOWED["xorelate"] = op_xorelate

def op_equalate(z, a, state):
    if z.size<1: return z
    k = _frozen_len(state)
    if k >= z.size: return z
    nx = int(a[0].real) or 256
    ny = int(a[0].imag) or nx
    thr = int(a[1].real) or 2
    head, tail = z[:k], z[k:]
    x, y = tail.real, tail.imag
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    if not np.isfinite(xmin + xmax + ymin + ymax): return z  # bail if degenerate
    if xmax == xmin: xmax = xmin + 1e-12
    if ymax == ymin: ymax = ymin + 1e-12
    H, xedges, yedges = np.histogram2d(
        x, y,
        bins=[nx, ny],
        range=[[xmin, xmax], [ymin, ymax]]
    )
    m = H == thr
    if not m.any(): return head  # nothing passes threshold
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    ii, jj = np.nonzero(m)
    centers = xc[ii] + 1j * yc[jj]
    return np.concatenate((head, centers))

ALLOWED["equalate"] = op_equalate

def _life_count(b: np.ndarray):
  n = (
    np.roll(np.roll(b, -1, 0), -1, 1) +  # down-right
    np.roll(np.roll(b, -1, 0),  0, 1) +  # down
    np.roll(np.roll(b, -1, 0),  1, 1) +  # down-left
    np.roll(np.roll(b,  0, 0), -1, 1) +  # right
    np.roll(np.roll(b,  0, 0),  1, 1) +  # left
    np.roll(np.roll(b,  1, 0), -1, 1) +  # up-right
    np.roll(np.roll(b,  1, 0),  0, 1) +  # up
    np.roll(np.roll(b,  1, 0),  1, 1)    # up-left
  )
  return n

def _life_toroidal(board: np.ndarray, steps: int) -> np.ndarray:
    """Conway’s Game of Life with wrap-around (toroidal) boundary."""
    b = board
    for _ in range(steps):
        # neighbor count via 8 rolls
        n = _life_count(b)
        # Life rules (vectorized)
        survive = (b & ((n == 2) | (n == 3)))
        born    = (~b & (n == 3))
        b = survive | born
    return b

def op_life(z, a, state):
    """
    Conway's Game of Life over a discretized tail.
      a[0] = iterations (int, default 100)
      a[1] = nx + i*ny  (complex bins per axis; ny defaults to nx)
      a[2] = thresh     (min count per bin to mark alive; default 1)
    Tail is replaced by centers of alive cells after evolution.
    """
    if z.size<1: return z
    k = _frozen_len(state)
    if k >= z.size: return z
    head, tail = z[:k], z[k:]
    if tail.size == 0: return z
    iterations = int(a[0].real)
    n = int(a[1].real) or 256
    thresh = int(a[2].real) or 1
    x = tail.real
    y = tail.imag
    xmin = x.min(); xmax = x.max()
    ymin = y.min(); ymax = y.max()
    if not np.isfinite(xmin + xmax + ymin + ymax): return z
    if xmax == xmin: xmax = xmin + 1e-12
    if ymax == ymin: ymax = ymin + 1e-12
    H, xedges, yedges = np.histogram2d(
        x, y,
        bins=[n, n],
        range=[[xmin, xmax], [ymin, ymax]]
    )
    board = (H >= thresh).astype(np.uint8, copy=False)
    if not board.any(): return head  # nothing alive → drop the tail entirely
    if iterations>0:
        board = _life_toroidal(board, iterations)
    if not board.any(): return head  # everything died
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    ii, jj = np.nonzero(board)     # ii over x-bins, jj over y-bins
    centers = xc[ii] + 1j * yc[jj]
    return np.concatenate((head, centers.astype(np.complex128)))

ALLOWED["life"] = op_life


# GRADual PIXELization (stochastic): gradpix
def op_gpixsq(z, a, state):
    if z.size < 1:
        return z
    k = _frozen_len(state)
    if k >= z.size:
        return z

    nx = int(a[0].real) or 256
    ny = int(a[0].imag) or nx
    gw = max(0.0, min(1.0, a[1].real))
    gloc = max(0.0, min(a[2].real or 0.5,1.0))
    dth = a[3].real or 0.25
    final_w  = a[4].real or 1.0


    head, tail = z[:k], z[k:]
    x, y = tail.real, tail.imag

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    if not np.isfinite(xmin + xmax + ymin + ymax):
        return z
    if xmax == xmin:
        xmax = xmin + 1e-12
    if ymax == ymin:
        ymax = ymin + 1e-12

    # --- compute 2D histogram (core step) ---
    H, xedges, yedges = np.histogram2d(
        x, y,
        bins=[nx, ny],
        range=[[xmin, xmax], [ymin, ymax]]
    )

    # --- assign each point to its bin ---
    ix = np.searchsorted(xedges, x, side="right") - 1
    iy = np.searchsorted(yedges, y, side="right") - 1
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)
    flat = ix * ny + iy
    order = np.argsort(flat, kind="mergesort")
    flat_sorted = flat[order]

    if flat_sorted.size == 0:
        return z

    cut = np.flatnonzero(np.diff(flat_sorted)) + 1
    starts = np.concatenate(([0], cut))
    ends   = np.concatenate((cut, [flat_sorted.size]))

    # --- stochastic pixelization per bin ---
    for s, e in zip(starts, ends):
        pos_sorted = order[s:e]
        n_bin = pos_sorted.size
        if n_bin == 0: continue

         # bin coordinates
        b_ix = ix[pos_sorted[0]]
        b_iy = iy[pos_sorted[0]]
        x0, x1 = xedges[b_ix], xedges[b_ix + 1]
        y0, y1 = yedges[b_iy], yedges[b_iy + 1]
        cx = (x0+x1)/2
        cy = (y0+y1)/2
        wx = (x1-x0)
        wy = (y1-y0)


        h = _smoothstep_scalar( b_ix/(nx-1) - gloc ,  gw )
 
        m = int(round( h * n_bin))
        if m <= 0: continue

        # random candidate points inside this bin
        rx = (RNG.random(n_bin)-0.5)
        ry = (RNG.random(n_bin)-0.5)
        rx = np.sign(rx)*(np.abs(rx)**dth)
        ry = np.sign(ry)*(np.abs(ry)**dth)
        wfac = ( final_w * h + 1.0 * (1-h) )
        cand_x = cx + rx * wx * wfac
        cand_y = cy + ry * wy * wfac

        # replace g*n_bin points with random ones
        replace_idx = RNG.choice(pos_sorted, size=m, replace=False)
        sample_idx  = RNG.integers(0, n_bin, size=m)
        tail.real[replace_idx] = cand_x[sample_idx]
        tail.imag[replace_idx] = cand_y[sample_idx]

    z[k:] = tail
    return z

ALLOWED["gpixsq"] = op_gpixsq


# GRADual PIXELization (stochastic): gradpix
def op_gpixdsk(z, a, state):
    if a[0].real==0:return z
    if z.size < 1: return z
    k = _frozen_len(state)
    if k >= z.size: return z

    nx = int(a[0].real) or 256
    ny = int(a[0].imag) or nx
    gw = max(0.0, min(1.0, a[1].real))
    gloc = max(0.0, min(a[2].real or 0.5,1.0))
    dth = a[3].real or 0.5
    final_w  = a[4].real or 1.0


    head, tail = z[:k], z[k:]
    x, y = tail.real, tail.imag

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    if not np.isfinite(xmin + xmax + ymin + ymax):
        return z
    if xmax == xmin:
        xmax = xmin + 1e-12
    if ymax == ymin:
        ymax = ymin + 1e-12

    # --- compute 2D histogram (core step) ---
    H, xedges, yedges = np.histogram2d(
        x, y,
        bins=[nx, ny],
        range=[[xmin, xmax], [ymin, ymax]]
    )

    # --- assign each point to its bin ---
    ix = np.searchsorted(xedges, x, side="right") - 1
    iy = np.searchsorted(yedges, y, side="right") - 1
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)
    flat = ix * ny + iy
    order = np.argsort(flat, kind="mergesort")
    flat_sorted = flat[order]

    if flat_sorted.size == 0:
        return z

    cut = np.flatnonzero(np.diff(flat_sorted)) + 1
    starts = np.concatenate(([0], cut))
    ends   = np.concatenate((cut, [flat_sorted.size]))

    # --- stochastic pixelization per bin ---
    for s, e in zip(starts, ends):
        pos_sorted = order[s:e]
        n_bin = pos_sorted.size
        if n_bin == 0: continue

         # bin coordinates
        b_ix = ix[pos_sorted[0]]
        b_iy = iy[pos_sorted[0]]
        x0, x1 = xedges[b_ix], xedges[b_ix + 1]
        y0, y1 = yedges[b_iy], yedges[b_iy + 1]
        cx = (x0+x1)/2
        cy = (y0+y1)/2
        wx = (x1-x0)
        wy = (y1-y0)


        h = _smoothstep_scalar( b_ix/(nx-1) - gloc ,  gw )
 
        m = int(round( h * n_bin))
        if m <= 0: continue

        # random candidate points inside this bin
        rho = RNG.random(n_bin)
        theta = RNG.random(n_bin)
        wfac = ( final_w * h + 1.0 * (1-h) )
        uc = 0.5*(wx+wy)*(wfac*(rho**dth))*np.exp(1j*2*np.pi*theta)        
        cand_x = cx + uc.real
        cand_y = cy + uc.imag

        # replace g*n_bin points with random ones
        replace_idx = RNG.choice(pos_sorted, size=m, replace=False)
        sample_idx  = RNG.integers(0, n_bin, size=m)
        tail.real[replace_idx] = cand_x[sample_idx]
        tail.imag[replace_idx] = cand_y[sample_idx]

    z[k:] = tail
    return z

ALLOWED["gpixdsk"] = op_gpixdsk


# ---------- JIT kernels for build_logo ----------


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
def _swirl_inplace_fast(x, y, swa, swb, mul):
    # keep rmax serial for determinism & simplicity
    rmax = _rmax_serial(x, y)*mul
    denom = rmax - 1.0
    if denom < 1e-9: denom = 1e-9
    two_pi = 6.283185307179586
    n = x.size
    for i in prange(n):
        xi = x[i]*mul
        yi = y[i]*mul
        r = (xi*xi + yi*yi)**0.5
        if r > 1.0:
            t = (r - 1.0) / denom
            # pow/cos/sin are vectorized-friendly with fastmath
            phi = two_pi * swa * (t ** swb)
            c = np.cos(phi); s = np.sin(phi)
            x[i] = xi * c - yi * s
            y[i] = xi * s + yi * c
        else:
            x[i] = xi
            y[i] = yi

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
        swa = a[0].real or 0.0
        swb = a[1].real or 1
        mul = a[2].real or 2.0
        x, y = _as_xy(z)
        _swirl_inplace_fast( x[k:], y[k:], swa, swb, mul )
    return z

ALLOWED["swirl"] = op_swirl


def op_squish(z, a, state):
    k = _frozen_len(state)
    if k < z.size:
        z.imag[k:] *= a[0].real
    return z

ALLOWED["squish"] = op_squish


# ---------- kaleidoscope transforms ----------
def op_archimedean_spiral(z,a,state):
    ap   =  a[0].real or 0.1
    bp   =  a[2].real or 0.1   
    k = _frozen_len(state)
    if k >= z.size: return z
    tail  = z[k:] # view not copy
    n = int(a[0].real)
    if n<0: return z
    if n>z.size-1: return z
    tp = tail.real
    theta = 2 * np.pi * tp
    r = ap + bp * theta
    asp = r * np.exp(1j * theta)
    tail[:] = asp
    return z

ALLOWED["asp"]=op_archimedean_spiral

def op_logarithmic_spiral(z,a,state):
    ap   =  a[0].real or 0.1
    bp   =  a[2].real or 0.1   
    k = _frozen_len(state)
    if k >= z.size: return z
    tail  = z[k:] # view not copy
    n = int(a[0].real)
    if n<0: return z
    if n>z.size-1: return z
    tp = tail.real
    theta = 2 * np.pi * tp
    r = ap + bp * theta
    lsp = ap * np.exp(bp * theta)
    tail[:] = lsp
    return z

ALLOWED["lsp"]=op_logarithmic_spiral

# ---------- kaleidoscope transforms ----------

def op_elkal(z, a, state):
    # ---- params (no a.size checks) ----
    N            = int(a[0].real) or 8
    inner_turns  = float(a[1].real) or 0.5  # rotations for inner ring, in turns
    inner_ratio  = float(a[2].real) or 0.5  # 0..1; 1=circle, 0=line
    r0  = float(a[3].real) or 0.1  # 0..1; 1=circle, 0=line
 
    # ---- tail-only ----
    k = _frozen_len(state)
    if k >= z.size:
        return z
    tail  = z[k:]
    r_raw = tail.real.astype(np.float64, copy=False)
    th_raw= tail.imag.astype(np.float64, copy=False)

    # normalize radius seed to [0,1] across the tail
    rmin = r_raw.min(); rmax = r_raw.max()
    denom = rmax - rmin
    r01 = np.zeros_like(r_raw) if denom <= 0.0 else (r_raw - rmin) / denom

    # ring index and ring radius (midpoints), mapped from r0..1
    ring = np.clip((r01 * N).astype(np.int64), 0, N - 1)
    t_ring = (ring + 0.5) / N                  # 0..1
    rho = r0 + (1.0 - r0) * t_ring             # inner→outer radius

    # angle from fractional part of theta seed
    th = th_raw - np.floor(th_raw)
    ang = 2.0 * np.pi * th

    # per-ring ellipse squish & rotation, interpolated inner->outer
    #   ratio(r)    : inner_ratio  --> 1.0
    #   turns(r)    : inner_turns  --> 0.0
    ratio = inner_ratio + (1.0 - inner_ratio) * t_ring
    turns = inner_turns * (1.0 - t_ring)
    rot   = np.exp(2j * np.pi * turns)         # complex unit rotation per ring

    # base circle on each ring
    cz = rho * (np.cos(ang) + 1j * np.sin(ang))

    # make ellipse by scaling y by 'ratio', then rotate by 'rot'
    ex = cz.real
    ey = cz.imag * ratio
    ez = (ex + 1j * ey) * rot

    tail[:] = ez
    return z

ALLOWED["elkal"] = op_elkal

# ellipse-kaleidoscope: hand-made version
def op_elkal1(z, a, state):
    N             =  int(a[0].real) or 8
    outer_turns   =  float(a[1].real) or 0.0   # rotations at outer ring, in turns
    inner_ratio   =  float(a[2].real) or 1.0   # 0..1; 1=circle, 0=line
    inner_radius  =  float(a[3].real) or 0.1   #
    k = _frozen_len(state)
    if k >= z.size: return z
    tail  = z[k:] # view not copy
    t = np.round((tail.real % 1) * N)/N
    squish = 1*t+inner_ratio*(1.0-t)
    turn = np.exp(1j * 2* np.pi * outer_turns * t)
    R = (1.0 * t + inner_radius * (1.0-t))
    theta = tail.imag % 1
    zz =  R * np.exp(1j * 2 * np.pi * theta)
    ez = (zz.real + 1j * zz.imag * squish) * turn
    tail[:] = ez
    return z

ALLOWED["elkal1"] = op_elkal1

# line-claeidoscope: hand-made version
def op_linkal(z, a, state):
    N             =  int(a[0].real) or 8
    outer_turns   =  float(a[1].real) or 0.5   # rotations at outer ring, in turns
    inner_width   =  float(a[2].real) or 0.5   # 0..1; 1=circle, 0=line
    k = _frozen_len(state)
    if k >= z.size: return z
    tail  = z[k:] # view not copy
    t = np.round((tail.real % 1) * N)/N
    s = tail.imag % 1
    width = 1*t+inner_width*(1.0-t)
    turn = np.exp(1j * 2* np.pi * outer_turns * t)
    zz =  width * (t-0.5) + 1j * s
    ez = zz * turn
    tail[:] = ez
    return z

ALLOWED["linkal"] = op_linkal

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
    if z.size<1: return z
    k = _frozen_len(state)
    if k>=z.size: return z
    C     = a[0]
    alpha = a[1].real or 1.0 
    sigma = a[2].real or 1.0 
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
    if z.size<1: return z
    k = _frozen_len(state)
    if k>=z.size: return z
    C     = a[0]
    alpha = a[1].real or 1
    eps   = a[2].real or 1e-10
    x, y = _as_xy(z)
    _nsink_inplace_fast(x[k:], y[k:], C.real, C.imag, alpha, eps)
    return z

ALLOWED["nsink"] = op_nsink

def op_gsink(z, a, state):
    """
    a[0]=C (complex), a[1]=alpha (real), a[2]=eps (real, small >=0)
    """
    if z.size<1: return z
    k = _frozen_len(state)
    c = a[0]
    f = a[1].real
    head, tail = z[:k],z[k:]
    dist = (c-tail) * f
    z[k:] += dist**2
    return z

ALLOWED["gsink"] = op_gsink

# ==================================================
# Size/mult ops (write to state)
# ==================================================

# add dots to state
def _dot_add(z,state,sizes):
    if z.size<1: return
    k = _frozen_len(state)
    if k>=z.size: return z
    head, tail = z[:k], z[k:]
    if sizes.size != tail.size: raise ValueError(f"sizes.size != tail.size.")
    new_gid = np.full(tail.size, _current_gid(state), dtype=np.float64)
    new_dots = sizes + 1j * new_gid
    prev_dots = state.get(K_MULT)
    if prev_dots is None:
        state[K_MULT] = new_dots.astype(np.complex128, copy=False)
    else:
        state[K_MULT] = np.concatenate((prev_dots, new_dots))
    return

# get mask for a gid
def _dot_gid(state,gid):
    dots = state[K_MULT]
    mask = ( (dots.imag).astype(np.int64) == int(gid) )
    return mask

# set gid to sizes
def _dot_set(z,state,mask,sizes):
    masked = np.sum(mask)
    if masked<1: return
    if sizes.size != masked: return
    dots = state[K_MULT]
    dots[mask] = sizes + 1j * int(dots[mask][0].imag)
    state[K_MULT] = dots 
    return z

#
# densities here
#

# constant
def _const(a,length):
    dot_sizes = np.full(length, a[0].real * a[1].real , dtype=np.float64)
    return dot_sizes

# lognormal
def _lognormal(a,length):
    loc   = a[0].real or 0.5
    scale = a[1].real or 1.25
    drt   = a[2].real or 100.0
    zeta = RNG.normal(loc=loc, scale=scale, size=length)
    dot_sizes = np.exp(zeta).astype(np.float64)
    np.clip(dot_sizes, 1.0, drt, out=dot_sizes)
    return dot_sizes

#
# dots to new group
#

# constant dot
def op_dot(z, a, state):
    if z.size<1: return z
    k = _frozen_len(state)
    if k>=z.size: return z
    head, tail = z[:k], z[k:]
    _dot_add(z,state,_const(a,tail.size))
    return z
ALLOWED["dot"] = op_dot

# lognormal dot
def op_ldot(z, a, state):
    if z.size<1: return z
    k = _frozen_len(state)
    if k>=z.size: return z
    head, tail = z[:k], z[k:]
    _dot_add(z,state,_lognormal(a,tail.size))
    return z
ALLOWED["ldot"] = op_ldot

#
# modify group dot sized
#

# dot set
def op_dotset(z, a, state):
    gid = _current_gid(state)
    if gid == 1: return z # no previous gid
    mask = _dot_gid(state,gid-1)
    _dot_set(z,state,mask, _const(a,sum(mask)))
    return z
ALLOWED["dotset"] = op_dotset

# dot region set 
# FIXME: generalize
# FIXME: optimize to nested

def op_dotrhset(z, a, state):
    gid = _current_gid(state)
    if gid == 1: return z # no previous gid
    mask = ( z.real >0 ) & _dot_gid(state,gid-1)
    _dot_set(z,state,mask, _const(a,sum(mask)))
    return z
ALLOWED["dotrhset"] = op_dotrhset

def op_dotulset(z, a, state):
    gid = _current_gid(state)
    if gid == 1: return z # no previous gid
    mask =  ( z.real >0 ) & ( z.imag >0 ) & _dot_gid(state,gid-1)
    _dot_set(z,state, mask,_const(a,sum(mask)))
    return z
ALLOWED["dotulset"] = op_dotulset

def op_dotlrset(z, a, state):
    gid = _current_gid(state)
    if gid == 1: return z # no previous gid
    mask = ( z.real <0 ) & ( z.imag <0 )  & _dot_gid(state,gid-1)
    _dot_set(z,state, mask,_const(a,sum(mask)))
    return z
ALLOWED["dotlrset"] = op_dotlrset

def op_doturln(z, a, state):
    gid = _current_gid(state)
    if gid == 1: return z # no previous gid
    mask = ( z.real >0 ) & ( z.imag >0 )  & _dot_gid(state,gid-1)
    masked = sum(mask)
    if masked<1: return z
    _dot_set(z,state,mask,_lognormal(a,masked))
    return z
ALLOWED["doturln"] = op_doturln

def op_dotllln(z, a, state):
    gid = _current_gid(state)
    if gid == 1: return z # no previous gid
    mask = ( z.real <0 ) & ( z.imag <0 )  & _dot_gid(state,gid-1)
    masked = np.sum(mask)
    if masked<1: return z # nothing to do
    _dot_set(z,state,mask, _lognormal(a,masked))
    return z
ALLOWED["dotllln"] = op_dotllln

# copy 
def op_dot_copy(z, a, state):
    k = _frozen_len(state)
    tail_len = max(z.size - k, 0)
    if tail_len == 0: return z
    mult = state.get(K_MULT)
    if mult is None or mult.size == 0: return z
    src_gid = int(round(a[0].real)) if a.size > 0 else 0
    if src_gid <= 0: src_gid = _current_gid(state) - 1
    if src_gid < 1:return z  
    src_mask = (mult.imag[:k] == src_gid)
    if not np.any(src_mask): return z 
    src_sizes = mult.real[:k][src_mask]
    tail_sizes = RNG.choice(src_sizes, size=tail_len, replace=True)
    gid = _current_gid(state)
    full_real = np.concatenate((mult.real, tail_sizes))
    full_imag = np.concatenate((mult.imag, np.full(tail_len, gid, dtype=np.float64)))
    state[K_MULT] = (full_real + 1j * full_imag).astype(np.complex128, copy=False)
    return z
ALLOWED["dcopy"] = op_dot_copy

# negate
def op_dneg(z, a, state):
    mult = state.get(K_MULT)
    if mult is None or mult.size == 0: return z
    prev_gid = _current_gid(state) - 1
    if prev_gid < 1: return z
    m = (mult.imag == float(prev_gid))
    if np.any(m):
        mult.real[m] *= -1.0
        state[K_MULT] = mult  # store back (same array, explicit for clarity)
    return z
ALLOWED["dneg"] = op_dneg

# ==================================================
# macro processor
# ==================================================

def macro(spec:str,z0,state): # spec runner
    #state = Dict.empty(key_type=types.int8, value_type=types.complex128[:])
    names, A = specparser.parse_names_and_args(spec, MAXA=12)
    z=z0
    for k, name in enumerate(names):
        fn = ALLOWED.get(name)
        if fn is None:
            raise ValueError(f"Unknown op '{name}'.")
        z = fn(z, A[k], state)
    return z

# ==================================================
# macros
# ==================================================

def op_frame(z,a, state):
    N = a[0].real or 1e6
    border = a[1].real or 0.1
    dotmult = a[2].real or 20
    rot =  np.exp(1j*2*np.pi*0.25)
    ur = (1+1j)-2*(border+1j*border)
    ul = ur * rot
    ll = ul * rot
    lr = ll * rot

    spec = (
        f"gnorm:{1-2*border}," # scale everything to fit inside
        f"rus:{N}:1," # make a square (-1-1j) to (1+1j)
        f"clpinsq:{1-2*border}:0," # clip square 2 borders in
        f"clpindisk:{border}:{ur}," # clip circle on ur radius border
        f"clpindisk:{border}:{ul},"
        f"clpindisk:{border}:{ll},"
        f"clpindisk:{border}:{lr},"
        f"clpinrect:{ll-border}:{ul+border}," # left vertical strip, 2 borders wide and 1-2*border tall
        f"clpinrect:{lr-border}:{ur+border}," # right vertical strip, 2 borders wide and 1-2*border tall
        f"clpinrect:{ll-1j*border}:{lr+1j*border}," # bottom horizontal strip, 2 borders wide and 1-2*border long
        f"clpinrect:{ul-1j*border}:{ur+1j*border},"  # top horizontal strip, 2 borders wide and 1-2*border long
        f"dot:pix:{dotmult}"
    )
    return macro(spec,z,state)

ALLOWED["frame"] = op_frame

def op_frame(z,a, state):
    N = a[0].real or 1e6
    border = a[1].real or 0.1
    dotmult = a[2].real or 20
    rot =  np.exp(1j*2*np.pi*0.25)
    ur = (1+1j)-2*(border+1j*border)
    ul = ur * rot
    ll = ul * rot
    lr = ll * rot

    spec = (
        f"gnorm:{1-2*border}," # scale everything to fit inside
        f"rus:{N}:1," # make a square (-1-1j) to (1+1j)
        f"clpinsq:{1-2*border}:0," # clip square 2 borders in
        f"clpindisk:{border}:{ur}," # clip circle on ur radius border
        f"clpindisk:{border}:{ul},"
        f"clpindisk:{border}:{ll},"
        f"clpindisk:{border}:{lr},"
        f"clpinrect:{ll-border}:{ul+border}," # left vertical strip, 2 borders wide and 1-2*border tall
        f"clpinrect:{lr-border}:{ur+border}," # right vertical strip, 2 borders wide and 1-2*border tall
        f"clpinrect:{ll-1j*border}:{lr+1j*border}," # bottom horizontal strip, 2 borders wide and 1-2*border long
        f"clpinrect:{ul-1j*border}:{ur+1j*border},"  # top horizontal strip, 2 borders wide and 1-2*border long
        f"dot:pix:{dotmult}"
    )
    return macro(spec,z,state)

ALLOWED["frame"] = op_frame

def op_yin(z, a, state):
    N = a[0].real or 1e5
    size = a[1].real or 1
    spec = (
        f"rud:{N},"
        f"clpinsq:1:1+0j,"
        f"clpindisk:0.5:0+0.5j,"
        f"rua:{N/8}:0.75+1.25j:0.25:0-0.5j,"
        f"add:0.5j,mul:{size}"
    )
    return macro(spec,z,state)

ALLOWED["yin"] = op_yin

def op_kirby(z,a,state):
    N = a[0].real or 1e7
    dr = a[1].real or 500
    spec = (
        f"rud:{N}:0.1,osclip:0.5,ldot:4:1.5:{dr}"
    )
    return macro(spec,z,state)

ALLOWED["kirby"] = op_kirby

def op_barred(z,a,state):
    N = a[0].real or 1e7
    swa = a[1].real or -0.33
    swb = a[2].real or 1
    scale = a[3].real or 1
    spec = ( 
        f"rud:{N}:0.5,td:1:0.01:1.25,swirl:{swa}:{swb},arms:2,mul:{scale},dot:pix"
    )
    return macro(spec,z,state)

ALLOWED["barred"] = op_barred


# ==================================================
# build_logo from spec 
# ==================================================

def build_logo_from_chain(spec: str) -> tuple[np.ndarray, np.ndarray]:
    state = Dict.empty(key_type=types.int8, value_type=types.complex128[:])
    z0 = np.empty(0, dtype=np.complex128)
    z = macro(spec,z0,state)
    mult_c = state.get(K_MULT)
    if mult_c is None:
        mult = np.ones(z.size, dtype=np.float32)
    else:
        k = mult_c.size
        mult = np.ones(z.size, dtype=np.float32)
        mult[:min(k, z.size)] = mult_c.real[:min(k, z.size)].astype(np.float32, copy=False)
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


