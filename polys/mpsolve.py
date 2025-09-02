import io
import os
import sys
import numpy as np
import ctypes, ctypes.util
import gmpy2
import timeit
import math
import pandas as pd
import inspect
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image,ImageDraw, ImageFont
matplotlib.use("Agg")  

from . import wilkinson

MPS_OUTPUT_GOAL_APPROXIMATE = 1
MPS_ALGORITHM_STANDARD_MPSOLVE = 0
MPS_ALGORITHM_SECULAR_GA    = 1

def load(): 
    exec(open("mpsolve.py").read(), globals())
    print("loaded mpsolve.py")

_mps_path = "/usr/local/lib/libmps.dylib"
if not os.path.exists(_mps_path): 
    raise FileNotFoundError(f"MPSolve not found at {_mps_path}")

# libraries
_libgmp = ctypes.CDLL(ctypes.util.find_library("gmp"))
_libc   = ctypes.CDLL(ctypes.util.find_library("c"))
_mps = ctypes.CDLL(_mps_path, mode=ctypes.RTLD_GLOBAL)

class mpf_t(ctypes.Structure):
    _fields_ = [
        ('_mp_prec', ctypes.c_int),
        ('_mp_size', ctypes.c_int),
        ('_mp_exp', ctypes.c_long),
        ('_mp_d', ctypes.POINTER(ctypes.c_ulong))
    ]

class Cplx(ctypes.Structure): # float complex
    _fields_ = [("real", ctypes.c_double),("imag", ctypes.c_double)]
    def __complex__(self):
        return complex(self.real, self.imag)

class mpc_t(ctypes.Structure):  # MPSolve's custom mpc_t using GMP mpf_t
    _fields_ = [ 
        ("re", mpf_t),
        ("im", mpf_t)
    ]

# --- rdpe_t: "double mantissa + long exponent" --------------------
class RDPE_struct(ctypes.Structure):
    _fields_ = [("m", ctypes.c_double), ("e", ctypes.c_long)]

# In the C headers: typedef __rdpe_struct rdpe_t[1];
# Mirror that "array-of-1" trick in ctypes:
rdpe_t = RDPE_struct * 1

# --- libc mem management helpers
_libc.free.argtypes = [ctypes.c_void_p]
_libc.memmove  # provided by ctypes

# funopen(FILE*) — available on macOS / BSD
FUNOPEN_READ_CB  = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_char), ctypes.c_int)
FUNOPEN_CLOSE_CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)

_libc.funopen.argtypes = [ctypes.c_void_p, FUNOPEN_READ_CB, ctypes.c_void_p, ctypes.c_void_p, FUNOPEN_CLOSE_CB]
_libc.funopen.restype = ctypes.c_void_p

# --- libc FILE* helpers
_libc.fopen.argtypes  = [ctypes.c_char_p, ctypes.c_char_p]
_libc.fopen.restype   = ctypes.c_void_p
_libc.fclose.argtypes = [ctypes.c_void_p]
_libc.fclose.restype  = ctypes.c_int
_libc.fdopen.argtypes  = [ctypes.c_int, ctypes.c_char_p]
_libc.fdopen.restype   = ctypes.c_void_p
# (optional but handy if you try the tmpfile fallback below)
_libc.tmpfile.argtypes = []
_libc.tmpfile.restype  = ctypes.c_void_p
_libc.fwrite.argtypes  = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p]
_libc.fwrite.restype   = ctypes.c_size_t
_libc.fflush.argtypes  = [ctypes.c_void_p]
_libc.fflush.restype   = ctypes.c_int
_libc.fseek.argtypes   = [ctypes.c_void_p, ctypes.c_long, ctypes.c_int]
_libc.fseek.restype    = ctypes.c_int  # SEEK_SET = 0

# Function: void mpf_init2 (mpf_t x, mp_bitcnt_t prec) 
_libgmp.__gmpf_init2.argtypes = [ctypes.POINTER(mpf_t), ctypes.c_ulong] 
# Function: void mpf_set_d (mpf_t rop, double op) 
_libgmp.__gmpf_set_d.argtypes = [ctypes.POINTER(mpf_t), ctypes.c_double]
# Function: void mpf_add (mpf_t rop, const mpf_t op1, const mpf_t op2) : Set rop to op1 + op2
_libgmp.__gmpf_add.argtypes = [ctypes.POINTER(mpf_t), ctypes.POINTER(mpf_t),ctypes.POINTER(mpf_t)]
# Function: void mpf_clear (mpf_t x) : Free the space occupied by x. 
_libgmp.__gmpf_clear.argtypes = [ctypes.POINTER(mpf_t)]

# Function: char * mpf_get_str (char *str, mp_exp_t *expptr, int base, size_t n_digits, const mpf_t op)
#
# Convert op to a string of digits in base base. 
# The base argument may vary from 2 to 62 or from −2 to −36. 
# Up to n_digits digits will be generated. 
# Trailing zeros are not returned. 
# No more digits than can be accurately represented by op are ever generated. 
# If n_digits is 0 then that accurate maximum number of digits are generated.
# For base in the range 2..36, digits and lower-case letters are used; 
# for −2..−36, digits and upper-case letters are used; 
# for 37..62, digits, upper-case letters, and lower-case letters (in that significance order) are used.
# If str is NULL, the result string is allocated using the current 
# allocation function (see Custom Allocation). 
# The block will be strlen(str)+1 bytes, that being exactly enough for the string and null-terminator.
# If str is not NULL, it should point to a block of n_digits + 2 bytes, that being enough for the mantissa, 
# a possible minus sign, and a null-terminator. 
# When n_digits is 0 to get all significant digits, 
# an application won’t be able to know the space required, and str should be NULL in that case.
# The generated string is a fraction, with an implicit radix point 
# immediately to the left of the first digit. 
# The applicable exponent is written through the expptr pointer. 
# For example, the number 3.1416 would be returned as string "31416" and exponent 1.
# When op is zero, an empty string is produced and the exponent returned is 0.
# A pointer to the result string is returned, being either the allocated block or the given str.
_libgmp.__gmpf_get_str.argtypes = [
    ctypes.c_char_p,                      # user buffer (or NULL) for self-allocation
    ctypes.POINTER(ctypes.c_long),        # exponent out
    ctypes.c_int,                         # base
    ctypes.c_size_t,                      # number of digits (0 = all)
    ctypes.POINTER(mpf_t)                 # the mpf_t to convert
]
_libgmp.__gmpf_get_str.restype = ctypes.c_void_p



_mps.mps_context_new.restype  = ctypes.c_void_p   # create context
_mps.mps_context_free.argtypes= [ctypes.c_void_p] # free context

_mps.mps_context_set_input_prec .argtypes = [ctypes.c_void_p, ctypes.c_long]
_mps.mps_context_set_input_prec.restype = None
_mps.mps_context_set_output_prec.argtypes = [ctypes.c_void_p, ctypes.c_long]
_mps.mps_context_set_output_prec.restype = None
_mps.mps_context_set_output_goal.argtypes = [ctypes.c_void_p, ctypes.c_int]
_mps.mps_context_set_output_goal.restype = None

_mps.mps_context_select_algorithm.argtypes = [ctypes.c_void_p, ctypes.c_int]
_mps.mps_context_select_algorithm.restype         = None

_mps.mps_context_set_input_poly.argtypes  = [ctypes.c_void_p, ctypes.c_void_p]
_mps.mps_context_set_input_poly.restype   = None

_mps.mps_context_get_degree.argtypes      = [ctypes.c_void_p]
_mps.mps_context_get_degree.restype       = ctypes.c_int

_mps.mps_mpsolve.argtypes                 = [ctypes.c_void_p]
_mps.mps_mpsolve.restype  = None

_mps.mps_monomial_poly_new.argtypes  = [ctypes.c_void_p, ctypes.c_int] # create poly
_mps.mps_monomial_poly_new.restype   = ctypes.c_void_p 

_mps.mps_monomial_poly_set_coefficient_d.argtypes = (
    ctypes.c_void_p, 
    ctypes.c_void_p, 
    ctypes.c_int,
    ctypes.c_double, 
    ctypes.c_double
)
_mps.mps_monomial_poly_set_coefficient_d.restype  = None

_mps.mps_polynomial_free.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_mps.mps_polynomial_free.restype = None

_mps.mps_monomial_poly_set_coefficient_s.argtypes = (
        ctypes.c_void_p,             # ctx
        ctypes.c_void_p,             # poly
        ctypes.c_int,                # exponent
        ctypes.c_char_p,             # real (ASCII string)
        ctypes.c_char_p              # real (ASCII string)
)  
_mps.mps_monomial_poly_set_coefficient_s.restype  = None

_mps.mps_context_get_roots_d.argtypes = [
    ctypes.c_void_p,# context
    ctypes.POINTER(ctypes.POINTER(Cplx)), # vector of Cplx float structures
    ctypes.c_void_p #ctypes.POINTER(ctypes.POINTER(ctypes.c_double)) # double **radii
] 
_mps.mps_context_get_roots_d.restype  = ctypes.c_int

_mps.mps_context_get_roots_m.argtypes = [
    ctypes.c_void_p, # context
    ctypes.POINTER(ctypes.POINTER(mpc_t)), # vector of GMP complex
    ctypes.POINTER(ctypes.POINTER(rdpe_t)) # vector of radii
]
_mps.mps_context_get_roots_m.restype = ctypes.c_int

# In 3.2.x these enum values are the usual ones:
MPS_STARTING_STRATEGY_DEFAULT   = 0
MPS_STARTING_STRATEGY_RECURSIVE = 1
MPS_STARTING_STRATEGY_FILE      = 2  # <- what we need

# void mps_context_set_root_stream(mps_context*, FILE*);
_mps.mps_context_set_root_stream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_mps.mps_context_set_root_stream.restype  = None

# void mps_context_select_starting_strategy(mps_context*, int);
_mps.mps_context_select_starting_strategy.argtypes = [ctypes.c_void_p, ctypes.c_int]
_mps.mps_context_select_starting_strategy.restype = None

# If you want clean error messages on failure:
_mps.mps_context_has_errors.argtypes = [ctypes.c_void_p]
_mps.mps_context_has_errors.restype  = ctypes.c_int
_mps.mps_context_error_msg.argtypes  = [ctypes.c_void_p]
_mps.mps_context_error_msg.restype   = ctypes.c_void_p  # we'll free it with libc.free


def mpf_to_str(x: mpf_t, base: int = 10) -> str:
    exp = ctypes.c_long()
    raw_ptr = _libgmp.__gmpf_get_str(None,ctypes.byref(exp),base,0,ctypes.byref(x))
    if not raw_ptr: return "0"
    mant_bytes = ctypes.cast(raw_ptr, ctypes.c_char_p).value
    mant_str   = mant_bytes.decode()
    _libc.free(raw_ptr)
    if not mant_str:  # empty string -> treat as zero
        return "0"
    neg = mant_str.startswith('-')
    if neg: mant_str = mant_str[1:]
    if len(mant_str) == 1:
        s = mant_str + ".0"
    else:
        s = mant_str[0] + "." + mant_str[1:]
    s += f"e{exp.value-1}"
    return "-" + s if neg else s

def mpf_t_to_mpfr(x: mpf_t) -> gmpy2.mpfr:
    s = mpf_to_str(x)         # e.g. "3.1415926535…e0"
    return gmpy2.mpfr(s)

# np vec to mpc
def np2mpc(xs: np.ndarray) -> list[gmpy2.mpc]:
    return [gmpy2.mpc(z.real, z.imag) for z in xs.ravel()]

def mpc2np(xs: list[gmpy2.mpc]) -> np.ndarray:
    return np.array([complex(r) for r in xs]).astype(np.complex128)

def r2cf(roots: list[gmpy2.mpc]) -> list[gmpy2.mpc]:
    coeffs = [gmpy2.mpc(1)]
    for r in roots:
        new_coeffs = [gmpy2.mpc(0)] * (len(coeffs) + 1)
        for i, c in enumerate(coeffs):
            new_coeffs[i] += c
            new_coeffs[i+1] -= c * r
        coeffs = new_coeffs
    return coeffs

def horner_mpc(coeffs: list[gmpy2.mpc], x: gmpy2.mpc) -> gmpy2.mpc:
    if not coeffs: return gmpy2.mpc(0)
    result = coeffs[0]
    for a in coeffs[1:]:
        result = result * x + a
    return result

def horner_vec_mpc(coeffs: list[gmpy2.mpc], xs: list[gmpy2.mpc]) -> list[gmpy2.mpc]:
    return [horner_mpc(coeffs, x) for x in xs]

def mpc_abs(xs: list[gmpy2.mpc]) -> list[gmpy2.mpfr]:
    return [gmpy2.sqrt(z.real*z.real + z.imag*z.imag) for z in xs]

def mpfr_max(xs: list[gmpy2.mpfr]) -> gmpy2.mpfr:
    if not xs: return gmpy2.mpfr(0)
    max_val = xs[0]
    for v in xs[1:]:
        if v > max_val: max_val = v
    return max_val

def mpc_t_to_mpc(z: mpc_t) -> gmpy2.mpc: # convert
    re = mpf_t_to_mpfr(z.re)
    im = mpf_t_to_mpfr(z.im)
    return gmpy2.mpc(re, im)

def mpsolve(coeffs0, precision=256, algo=MPS_ALGORITHM_STANDARD_MPSOLVE):

    coeffs=trim_leading_zeros(coeffs0)
    # create context
    ctx = _mps.mps_context_new()
    _mps.mps_context_set_output_prec(ctx, precision)
    _mps.mps_context_set_output_goal(ctx, MPS_OUTPUT_GOAL_APPROXIMATE)
    _mps.mps_context_select_algorithm(ctx, algo)
    _mps.mps_context_select_starting_strategy(ctx, MPS_STARTING_STRATEGY_DEFAULT)

    # create poly
    poly = _mps.mps_monomial_poly_new(ctx, len(coeffs) - 1 )
    for exp, c in enumerate(coeffs[::-1]): # a0 first
        _mps.mps_monomial_poly_set_coefficient_d(ctx, poly, exp, c.real, c.imag)
    _mps.mps_context_set_input_poly(ctx, poly)

    # IMPORTANT: set input precision *after* input poly is set
    _mps.mps_context_set_input_prec(ctx,  precision)

    # solve poly
    _mps.mps_mpsolve(ctx)
    # get results
    n = _mps.mps_context_get_degree(ctx)
    roots_pp = (Cplx*n)()
    roots_ptr = ctypes.cast(roots_pp, ctypes.POINTER(Cplx))
    status = _mps.mps_context_get_roots_d(ctx, ctypes.byref(roots_ptr), None)
    if status != 0 or not bool(roots_pp): 
        raise RuntimeError("mps_context_get_roots_d failed")
        
    roots = [None] * n
    for i in range(n):
        c = roots_pp[i]
        roots[i] = complex(c.real, c.imag)
    _mps.mps_polynomial_free(ctx, poly)
    _mps.mps_context_free(ctx)
    return  np.array(roots)



_SEEK_SET = 0

def _normalize_guess_line(g):
    return f"{float(g.real):.17g} {float(g.imag):.17g}"

def _lines_to_tmpFILE(lines):
    """Write lines to an anonymous tmpfile() and rewind; return FILE*."""
    payload = ("\n".join(line.rstrip("\n") for line in lines) + "\n").encode("utf-8")
    f = _libc.tmpfile()
    if not f:
        raise OSError("tmpfile() failed")
    buf = ctypes.create_string_buffer(payload)
    #wrote = _libc.fwrite(ctypes.byref(buf), 1, len(payload), f)
    wrote = _libc.fwrite(buf, 1, len(payload), f)
    if wrote != len(payload):
        _libc.fclose(f)
        raise OSError("fwrite failed")
    _libc.fflush(f)
    _libc.fseek(f, 0, _SEEK_SET)
    return f, payload  # return payload for debugging on error

def mpsolve_warm(coeffs, roots, precision=256,algo=MPS_ALGORITHM_STANDARD_MPSOLVE):
    """
    coeffs: iterable of complex-like numbers, highest-degree last (same as your mpsolve)
    roots:  list of strings ('(re, im)' or 're im') or complex numbers
    precision: bits of precision for input & output
    """
    # 1) Normalize guesses to the two-column format expected by the file starter
    roots_lines = [_normalize_guess_line(g) for g in roots]

    # 2) Build a seekable FILE* with those lines
    roots_FILE, payload = _lines_to_tmpFILE(roots_lines)

    # 3) Create context and configure
    ctx = _mps.mps_context_new()

    _mps.mps_context_set_output_prec(ctx, precision)
    _mps.mps_context_set_output_goal(ctx, MPS_OUTPUT_GOAL_APPROXIMATE)

    _mps.mps_context_select_algorithm(ctx, algo)
    _mps.mps_context_select_starting_strategy(ctx, MPS_STARTING_STRATEGY_FILE)

    # point the root-stream at our in-memory file
    _mps.mps_context_set_root_stream(ctx, roots_FILE)

    # 4) Build and set the monomial polynomial: a0 first
    poly = _mps.mps_monomial_poly_new(ctx, len(coeffs) - 1)
    for exp, c in enumerate(coeffs[::-1]):
        _mps.mps_monomial_poly_set_coefficient_d(ctx, poly, exp, float(c.real), float(c.imag))
    _mps.mps_context_set_input_poly(ctx, poly)

    # 5) Now that the poly is in the context, set input precision
    _mps.mps_context_set_input_prec(ctx, precision)

    # 6) Solve
    _mps.mps_mpsolve(ctx)

    # Close the file now (MPSolve should have consumed it)
    _libc.fclose(roots_FILE)

    # 7) Error handling: if parsing failed, show the first bytes we fed
    if _mps.mps_context_has_errors(ctx):
        msg_ptr = _mps.mps_context_error_msg(ctx)
        try:
            msg = ctypes.cast(msg_ptr, ctypes.c_char_p).value.decode('utf-8', 'replace') if msg_ptr else "Unknown error"
        finally:
            if msg_ptr:
                _libc.free(msg_ptr)

        # free before raising
        _mps.mps_polynomial_free(ctx, poly)
        _mps.mps_context_free(ctx)

        # small preview of the payload for debugging
        preview = payload[:200].decode('utf-8', 'replace').splitlines()[:5]
        raise RuntimeError(
            "MPSolve error: "
            f"{msg}\n"
            "Roots stream preview (first lines):\n  "
            + "\n  ".join(preview)
        )

    # 8) Collect results (unchanged from your code)
    n = _mps.mps_context_get_degree(ctx)
    roots_pp = (Cplx * n)()
    roots_ptr = ctypes.cast(roots_pp, ctypes.POINTER(Cplx))
    status = _mps.mps_context_get_roots_d(ctx, ctypes.byref(roots_ptr), None)
    
    if status != 0 or not bool(roots_pp):
        _mps.mps_polynomial_free(ctx, poly)
        _mps.mps_context_free(ctx)
        raise RuntimeError("mps_context_get_roots_d failed")

    out_roots = [None] * n
    for i in range(n):
        c = roots_pp[i]
        out_roots[i]=complex(c.real,c.imag)

    _mps.mps_polynomial_free(ctx, poly)
    _mps.mps_context_free(ctx)
    return np.array(out_roots)



def trim_leading_zeros(a, tol=0.0):
    nz = np.flatnonzero(np.abs(a) > tol)
    return a[nz[0]:] if nz.size else a

# conservative bounds for the double path you use in mps_monomial_poly_set_coefficient_d
_DBL_MAX_SAFE = 1e300        # < 1.797e308 to leave headroom
_DBL_MIN_SAFE = 1e-300       # treat tinies as zero

def _poly_ok(a: np.ndarray) -> bool:
    a = np.asarray(a, dtype=np.complex128)

    # 1) finite
    if not np.isfinite(a).all():
        return False

    # 2) non-empty and leading coeff not (absolutely) tiny/zero
    if a.size == 0 or np.abs(a[0]) <= _DBL_MIN_SAFE:
        return False

    # 3) magnitude within safe double range
    maxabs = float(np.max(np.abs(a)))
    if maxabs > _DBL_MAX_SAFE:
        return False

    return True


def scan_cold(
        cf_in, 
        s_start, 
        s_end, 
        t_start, 
        t_end, 
        n_points, 
        perturb,
        precision=1024,
        algo=MPS_ALGORITHM_STANDARD_MPSOLVE
):

    cf = np.asarray(cf_in, dtype=np.complex128)
    degree = cf.shape[0] - 1
    steps  = int(np.sqrt(n_points))
    s_step = (s_end - s_start) / steps
    t_step = (t_end - t_start) / steps
    out   = np.empty((n_points * degree, 4), dtype=np.float64)
    idx, i, j, dj = 0, 0, 0, 1

    for k in range(n_points):

        if n_points >= 100 and (k % (n_points//100) == 0):
            print(f"{100.0*k/n_points:.1f} : {k}/{n_points}")

        s = s_start + i * s_step
        t = t_start + j * t_step
        local_cf = perturb(cf, s, t)

        j += dj
        if j == steps or j < 0:
            i += 1 
            dj = -dj
            j += dj

        if _poly_ok(local_cf):
            roots = mpsolve(local_cf, precision=precision,algo=algo)
            for r in range(len(roots)):
                z = roots[r]
                out[idx, 0] = z.real
                out[idx, 1] = z.imag
                out[idx, 2] = s
                out[idx, 3] = t
                idx += 1

    return out[:idx]


def scan(cf_in, s_start, s_end, t_start, t_end, n_points, perturb, guess0):

    if len(cf_in)!=len(guess0)+1:
        raise RuntimeError("Invalid Guess")
    cf = np.asarray(cf_in, dtype=np.complex128)
    degree = cf.shape[0] - 1
    steps  = int(np.sqrt(n_points))
    s_step = (s_end - s_start) / steps
    t_step = (t_end - t_start) / steps

    out   = np.empty((n_points * degree, 4), dtype=np.float64)
    guess = np.asarray(guess0, dtype=np.complex128)
    idx, i, j, dj, wc = 0, 0, 0, 1, 0

    for k in range(n_points):
        if n_points >= 100 and (k % (n_points//100) == 0):
            print(f"{100.0*k/n_points:.1f} : {k}/{n_points} wc:{wc}")

        s = s_start + i * s_step
        t = t_start + j * t_step
        local_cf = perturb(cf, s, t)

        # --- simple guards ---
        if not _poly_ok(local_cf):
            # skip this iteration
            j += dj
            if j == steps or j < 0:
                i += 1; dj = -dj; j += dj
            continue
        # ----------------------
        roots = mpsolve_warm(local_cf, guess, precision=1024)

        for r in range(len(roots)):
            z = roots[r]
            out[idx, 0] = z.real
            out[idx, 1] = z.imag
            out[idx, 2] = s
            out[idx, 3] = t
            idx += 1

        if len(roots) == degree:
            guess = roots.copy()
            wc += 1

        j += dj
        if j == steps or j < 0:
            i += 1; dj = -dj; j += dj

    return out[:idx]


#
# a test
#

def show_roots(cf,roots):
    roots_sorted = sorted(roots, key=lambda z: z.real,reverse=True)
    df = pd.DataFrame({
        "Real Part": [r.real for r in roots_sorted],
        "Imag Part": [r.imag for r in roots_sorted],
        "Residual" : [np.abs(np.polyval(cf, r)) for r in roots_sorted]
    })
    print(df.to_string(index=False))

def plot_manifold(mat, title, path):
    re, im = mat[:, 0], mat[:, 1]

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.scatter(im, re, s=0.5, marker=".", linewidths=0, alpha=0.6)

    # bounds must match x=im, y=re
    xmin, xmax = im.min(), im.max()
    ymin, ymax = re.min(), re.max()

    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    span = max(xmax - xmin, ymax - ymin)
    pad = 0.05 * span
    half = span / 2.0 + pad

    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)

    ax.set_aspect('equal', adjustable='box')
    try:
        ax.set_box_aspect(1)
    except Exception:
        pass

    ax.set_xlabel("Im(z)")
    ax.set_ylabel("Re(z)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    fig.savefig(path, dpi=500)
    plt.close(fig)

def rasterize_points(re, im, llx, lly, urx, ury, width, height, *, x_is_im=True, flip_y=True, dtype=np.uint8):
    """
    Rasterize points into a binary image.

    re, im : 1D arrays (same length)
    bbox   : llx,lly,urx,ury (x=minX, y=minY, etc.)
    width, height : output image size in pixels
    x_is_im : if True, x := im and y := re (your current convention)
    flip_y : if True, put y=ury at top row (image coordinates)
    """
    re = np.asarray(re)
    im = np.asarray(im)

    # choose x/y from your convention
    x = im if x_is_im else re
    y = re if x_is_im else im

    # finite mask
    m = np.isfinite(x) & np.isfinite(y)
    if not np.any(m):
        return np.zeros((height, width), dtype=dtype)

    x = x[m]; y = y[m]

    # avoid zero-division if bbox has zero span
    span_x = max(urx - llx, np.finfo(float).eps)
    span_y = max(ury - lly, np.finfo(float).eps)

    # map to pixel indices in [0, width-1] × [0, height-1]
    ix = np.floor((x - llx) / span_x * width).astype(np.int64)
    iy = np.floor((y - lly) / span_y * height).astype(np.int64)

    # clamp to valid range (points on the max edge go to last pixel)
    ix = np.clip(ix, 0, width  - 1)
    iy = np.clip(iy, 0, height - 1)

    # accumulate counts
    img = np.zeros((height, width), dtype=np.int32)
    np.add.at(img, (iy, ix), 1)

    # binary threshold → 0/255
    img = (img > 0).astype(dtype) * np.array(255, dtype=dtype)

    # flip vertically if you want y increasing upwards in math space
    if flip_y:
        img = np.flipud(img)

    return img


def text_to_image(
    text: str,
    width: int,
    *,
    font_path="/System/Library/Fonts/SFNSMono.ttf",
    probe_size=100,           # start big, just for measuring
    margin=40,                # px on all sides
    line_spacing=4,           # extra px between lines
    fg=(235,235,235),
    bg=(18,18,18),
    upscale_antialias=2,      # render x2 then downscale for crisp edges
    min_font_size=6,
    max_font_size=None,       # optional cap
):
    """
    Scale font so the longest line fits available width (no wrapping).
    Height grows to fit all lines. Returns a new PIL.Image.
    """
    # Normalize newlines/tabs
    text = text.replace("\r\n", "\n").replace("\r", "\n").expandtabs(4)
    lines = text.split("\n")

    # Hi-DPI canvas for measurement
    scale = max(1, int(upscale_antialias))
    W = width * scale
    avail_w = max(1, W - 2 * margin * scale)

    # Load probe font
    if font_path and os.path.exists(font_path):
        probe_font = ImageFont.truetype(font_path, probe_size * scale)
    else:
        probe_font = ImageFont.load_default()

    # Measure longest line at probe size
    # Use a tiny image for a valid drawing context
    _probe_img = Image.new("RGB", (10,10))
    _draw = ImageDraw.Draw(_probe_img)

    longest_px = 1
    for line in lines:
        px = _draw.textlength(line, font=probe_font)
        if px > longest_px:
            longest_px = px

    # Compute target font size so longest line fits avail_w
    raw_size = (probe_font.size * avail_w) / longest_px
    if max_font_size is not None:
        raw_size = min(raw_size, max_font_size * scale)
    target_px_size = max(min_font_size * scale, int(math.floor(raw_size)))

    # Load final font at target size
    if font_path and os.path.exists(font_path):
        font = ImageFont.truetype(font_path, target_px_size)
    else:
        font = ImageFont.load_default()

    # Measure total block height at final size
    # multiline_textbbox accounts for ascent/descent and spacing
    block_text = "\n".join(lines)
    tmp_img = Image.new("RGB", (W, 10), bg)
    draw = ImageDraw.Draw(tmp_img)
    bbox = draw.multiline_textbbox(
        (0, 0), block_text, font=font, spacing=line_spacing*scale, align="left"
    )
    block_w = bbox[2] - bbox[0]
    block_h = bbox[3] - bbox[1]

    # Final image height = margins + block height
    H = block_h + 2 * margin * scale
    img = Image.new("RGB", (W, H), color=bg)
    draw = ImageDraw.Draw(img)

    # Draw text at top-left inside margins
    x = margin * scale
    y = margin * scale
    draw.multiline_text(
        (x, y),
        block_text,
        font=font,
        fill=fg,
        spacing=line_spacing * scale,
        align="left",
    )

    # Downscale for antialiasing
    if scale > 1:
        img = img.resize((width, H // scale), Image.LANCZOS)

    return img


def chessboard_roots(n: int) -> np.ndarray:
    """n×n grid centered at 0, flattened, dtype complex128."""
    o = 0.5 * (n - 1)
    out = np.empty(n*n, dtype=np.complex128)
    idx = 0
    for i in range(n):
        xi = float(i) - o
        for j in range(n):
            yj = float(j) - o
            out[idx] = xi + 1j*yj
            idx += 1
    return out

def compute_view(re,im):
    xmin, xmax = im.min(), im.max()
    ymin, ymax = re.min(), re.max()
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    span = max(xmax - xmin, ymax - ymin)
    pad = 0.05 * span
    half = span / 2.0 + pad
    llx,urx = cx - half, cx + half
    lly,ury = cy - half, cy + half
    return llx,lly, urx, ury

