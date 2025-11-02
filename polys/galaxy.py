import numpy as np
from numba import njit, prange, types
from numba.typed import Dict
import specparser

# ===== registry =====

ALLOWED = {}

# ----- state channels (int8 keys) -----
K_MULT = np.int8(1)   # stores the multipliers vector as complex[:] (real part used)

# ----- tiny helper -----
def _as_xy(z: np.ndarray):
    return z.real, z.imag  # views, no copies

# =========================
# RNG ops (return new z)
# =========================

def op_rng_circle(z, a, state):
    """
    a[0]=N, a[1]=dth, a[2]=seed (optional)
    """
    N   = int(a[0].real)
    dth = float(a[1].real if a.size > 1 else 1.0)
    if a.size > 2 and a[2].real != 0.0:
        np.random.seed(int(a[2].real))
    u1 = np.random.random(N)
    u2 = np.random.random(N)
    r  = u1 ** dth
    th = 2.0 * np.pi * u2
    c = np.cos(th); s = np.sin(th)
    x = (r * c).astype(np.float64, copy=False)
    y = (r * s).astype(np.float64, copy=False)
    return x + 1j * y

ALLOWED["rng_circle"] = op_rng_circle

def op_rng_square(z, a, state):
    """
    a[0]=N, a[1]=seed (optional)
    """
    N = int(a[0].real)
    if a.size > 1 and a[1].real != 0.0:
        np.random.seed(int(a[1].real))
    x = np.random.uniform(-1.0, 1.0, size=N).astype(np.float64, copy=False)
    y = np.random.uniform(-1.0, 1.0, size=N).astype(np.float64, copy=False)
    return x + 1j * y

ALLOWED["rng_square"] = op_rng_square

@njit(cache=True, nogil=True, parallel=True, fastmath=True)
def op_rng_circle_njit(N: int, dth: float):
    # Numba RNG uses its own generator; reproducible per process
    x = np.empty(N, np.float64)
    y = np.empty(N, np.float64)
    two_pi = 6.283185307179586
    for i in prange(N):
        u1 = np.random.random()               # Numba's RNG inside njit
        u2 = np.random.random()
        r  = u1 ** dth
        th = two_pi * u2
        c  = np.cos(th); s = np.sin(th)
        x[i] = r * c; y[i] = r * s
    return x + 1j*y

ALLOWED["rng_circle_jit"] = op_rng_circle_njit

@njit(cache=True, nogil=True, parallel=True, fastmath=True)
def op_sizes_lognormal_njit(n: int, lmu: float, lsig: float, drt: float):
    mult = np.empty(n, np.float32)
    for i in prange(n):
        z = np.random.normal()                # Numba RNG
        m = np.exp(lmu + lsig * z)
        if m < 1.0:
            m = 1.0
        elif m > drt:
            m = drt
        mult[i] = m
    return mult

ALLOWED["sizes_lognormal_jit"] = op_sizes_lognormal_njit

def op_rng_triangle(z, a, state):
    """
    a[0]=N, a[1]=seed (optional)
    Equilateral triangle of side ~2 centered near origin.
    """
    N = int(a[0].real)
    if a.size > 1 and a[1].real != 0.0:
        np.random.seed(int(a[1].real))
    h = np.sqrt(3) / 2
    A = (-1.0, -h); B = (1.0, -h); C = (0.0, h)
    u = np.random.random(N); v = np.random.random(N)
    mask = (u + v) > 1.0
    u[mask] = 1.0 - u[mask]
    v[mask] = 1.0 - v[mask]
    x = A[0] + u * (B[0] - A[0]) + v * (C[0] - A[0])
    y = A[1] + u * (B[1] - A[1]) + v * (C[1] - A[1])
    return x + 1j * y

ALLOWED["rng_triangle"] = op_rng_triangle

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
    x, y = _as_xy(z)
    _teardrop_inplace_fast(x, y, a[0].real, a[1].real, a[2].real)
    return z

ALLOWED["td"] = op_td

def op_arms(z, a, state):
    # a[0]=m
    m = int(a[0].real)
    if m <= 1: return z
    n = z.size
    idx = np.mod(np.arange(n), m)
    ang = (2.0*np.pi/m) * idx
    c = np.cos(ang); s = np.sin(ang)
    x, y = _as_xy(z)
    xr = x*c - y*s
    yr = x*s + y*c
    x[:] = xr; y[:] = yr
    return z

ALLOWED["arms"] = op_arms

def op_swirl(z, a, state):
    # a[0]=swa, a[1]=swb
    x, y = _as_xy(z)
    _swirl_inplace_fast(x, y, a[0].real, a[1].real)
    return z

ALLOWED["swirl"] = op_swirl

def op_squish(z, a, state):
    # a[0]=sqs
    z.imag *= a[0].real
    return z

ALLOWED["squish"] = op_squish

def op_rot(z, a, state):
    # a[0]=frt (turns)
    phi = 2.0 * np.pi * a[0].real
    c = np.cos(phi); s = np.sin(phi)
    x, y = _as_xy(z)
    xr = x*c - y*s
    yr = x*s + y*c
    x[:] = xr; y[:] = yr
    return z

ALLOWED["rot"] = op_rot

# =========================
# Size/mult ops (write to state)
# =========================

def op_sizes_lognormal(z, a, state):
    """
    a[0]=lmu, a[1]=lsig, a[2]=drt (clip upper), a[3]=seed (optional)
    Writes multipliers (float) into state[K_MULT] as complex vector (imag=0).
    """
    lmu  = float(a[0].real)
    lsig = float(a[1].real)
    drt  = float(a[2].real) if a.size > 2 else np.inf
    if a.size > 3 and a[3].real != 0.0:
        np.random.seed(int(a[3].real))

    zeta = np.random.normal(0.0, 1.0, size=z.size)
    mult = np.exp(lmu + lsig * zeta).astype(np.float64)
    if np.isfinite(drt):
        np.clip(mult, 1.0, drt, out=mult)

    # store in state as complex[:] (real=mult, imag=0)
    vec = np.empty(mult.size, np.complex128)
    vec.real = mult
    vec.imag = 0.0
    state[K_MULT] = vec
    return z

ALLOWED["sizes_ln"] = op_sizes_lognormal

def op_sizes_const(z, a, state):
    """
    a[0]=value (default 1.0). Fallback if user doesn’t specify sizes.
    """
    val = float(a[0].real) if a.size > 0 else 1.0
    vec = np.empty(z.size, np.complex128)
    vec.real = val; vec.imag = 0.0
    state[K_MULT] = vec
    return z

ALLOWED["sizes"] = op_sizes_const

# ===== executor (no JIT needed; kernels inside are Numba-fast) =====
def apply_chain(z0: np.ndarray, names: list[str], A: np.ndarray, state) -> tuple[np.ndarray, np.ndarray]:
    z = z0
    for k, name in enumerate(names):
        fn = ALLOWED.get(name)
        if fn is None:
            raise ValueError(f"Unknown op '{name}'. Allowed: {list(ALLOWED)}")
        z = fn(z, A[k], state)

    # multipliers: read from state or default 1.0
    mult_c = state.get(K_MULT)
    if mult_c is None or mult_c.size != z.size:
        mult = np.ones(z.size, dtype=np.float32)
    else:
        mult = mult_c.real.astype(np.float32, copy=False)
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
    _ = op_rng_circle_njit(1024, 0.75)
    _ = op_sizes_lognormal_njit(1024, 0.0, 1.0, 3000.0)


