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
        x = (c_eff + 0.5) / cols - 0.5
        y = (r + 0.5) / rows - 0.5
        zs[k] = x + 1j * y
    z=np.concatenate((z,zs))
    return z

ALLOWED["serp"] = op_serp

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

def op_hilbert(z,a,state):
    order = int(a[0].real) or 6
    pts = hilbert_curve(order)
    return np.concatenate((z, pts))

ALLOWED["hilbert"] = op_hilbert

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
    Koch snowflake generator (appends complex points).
      a[0] = iterations (int, default 6)
      a[1] = side       (real side length of initial triangle, default 1.0)
      a[2] = center     (complex center, default 0+0j)
      a[3] = turn       (real turns, 0..1, default 0.0)
    """
    it   = int(a[0].real) or 6
    side = a[1].real or 1.0
    cen  = a[2]
    turn = a[3].real or 0.0

    pts = _koch_snowflake(it, side, cen, turn)
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
        seq = seq.replace("X", "X+YF+").replace("Y", "-FX-Y")
    pos = 0.0 + 0.0j
    ang = 0.0
    pts = [pos]
    for c in seq:
        if c == "F":
            pos += step * (np.cos(ang) + 1j*np.sin(ang))
            pts.append(pos)
        elif c == "+": ang += np.pi/2
        elif c == "-": ang -= np.pi/2
    arr = np.array(pts, np.complex128)
    # normalize to ~unit box
    arr /= (np.max(arr.real) - np.min(arr.real) + 1e-12)
    return arr

def op_dragon(z, a, state):
    it   = int(a[0].real) or 15
    step = a[1].real or 1.0
    pts  = _dragon_curve(it, step)
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

# ---------- clip ops ----------
# "clip in" means remove inside
# "clip out" means remove outside

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

ALLOWED["cloutdisk"] = op_clpoutdisk

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
    phi = np.exp(1j * 2.0 * np.pi * a[0].real)
    k = _frozen_len(state)
    if k < z.size:
        z[k:] *= phi
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
    phi = np.exp(1j * 2.0 * np.pi * a[0].real)
    z *= phi
    return z
ALLOWED["grot"] = op_grot

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

def op_csum(z, a, state):
    mult = state.get(K_MULT)
    k = 0 if (mult is None) else mult.size
    if k < z.size:
        tail = z[k:]
        z[k:] = np.cumsum(tail)
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

# GRADual PIXELization: gradpix
def op_gradpix(z, a, state):
    if z.size < 1: 
        return z
    k = _frozen_len(state)
    if k >= z.size: 
        return z

    head, tail = z[:k], z[k:]

    nx = int(a[0].real) or 256
    ny = int(a[0].imag) or nx
    g  = float(a[1].real) or 0.0
    # Clamp g to [0, 1] to avoid overshoot
    if g <= 0.0:
        return z
    if g > 1.0:
        g = 1.0

    x = tail.real
    y = tail.imag

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    if not np.isfinite(xmin + xmax + ymin + ymax):
        return z
    if xmax == xmin: 
        xmax = xmin + 1e-12
    if ymax == ymin: 
        ymax = ymin + 1e-12

    # Use the same binning convention as op_pixelate (histogram2d edges)
    H, xedges, yedges = np.histogram2d(
        x, y,
        bins=[nx, ny],
        range=[[xmin, xmax], [ymin, ymax]]
    )

    # Bin indices for each point (vectorized)
    ix = np.searchsorted(xedges, x, side="right") - 1
    iy = np.searchsorted(yedges, y, side="right") - 1
    # Clip to valid bins
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)

    # Bin centers
    cx = 0.5 * (xedges[:-1] + xedges[1:])
    cy = 0.5 * (yedges[:-1] + yedges[1:])

    # Target centers for each point
    x_c = cx[ix]
    y_c = cy[iy]
    centers = x_c + 1j * y_c

    # Move a fraction g toward bin centers
    moved = tail + g * (centers - tail)

    out = np.empty_like(z)
    out[:k] = head
    out[k:] = moved
    return out

ALLOWED["gradpix"] = op_gradpix

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
    loc   = a[0].real or 0.5
    scale = a[1].real or 1.25
    drt   = a[2].real or 100.0
    mul   = a[3].real or 1.0

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
    state[K_MULT] = (full_real * mul + 1j * full_imag).astype(np.complex128, copy=False)
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

    # constant times mult size value so pixels can be specified
    val = float(a[0].real) or 1.0
    mul = float(a[1].real) or 1.0
    mult_tail_real = np.full(tail_len, val*mul, dtype=np.float64)

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

def op_dneg(z, a, state):
    """
    Flip the sizes (real part) of the *previous* group to negative in-place.
    Does not change group ids or geometry.
    """
    mult = state.get(K_MULT)
    if mult is None or mult.size == 0:
        return z

    # previous gid is current_gid - 1
    prev_gid = _current_gid(state) - 1
    if prev_gid < 1:
        return z

    # flip sizes for that gid (across all written multipliers)
    m = (mult.imag == float(prev_gid))
    if np.any(m):
        mult.real[m] *= -1.0
        state[K_MULT] = mult  # store back (same array, explicit for clarity)
    return z

ALLOWED["dneg"] = op_dneg

# =========================
# macro processor
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
    return z

# =========================
# macros
# =========================

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



# ===== build_logo from spec =====

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


