#!/usr/bin/env python3
"""
roots_mp_fixed.py  – stable multiprecision root fetch for MPSolve 3.2
  • keeps context alive until every root is converted
  • no access to private MPFR fields
  • works with sizeof(mpfr_t)=32 (Home‑brew default)
"""

import ctypes, ctypes.util, numpy as np
from mpmath import mp

# ── shared libraries ────────────────────────────────────────────────
libmps  = ctypes.CDLL("/usr/local/lib/libmps.dylib", mode=ctypes.RTLD_GLOBAL)
libmpfr = ctypes.CDLL(ctypes.util.find_library("mpfr"))

# ── opaque MPFR / MPC structures (32‑byte mpfr_t on your system) ────
class mpfr_t(ctypes.Structure):
    _fields_ = [('_', ctypes.c_byte * 32)]

class mpc_t(ctypes.Structure):
    _fields_ = [("re", mpfr_t), ("im", mpfr_t)]

# ── MPFR → mpmath converter (needs no field knowledge) ───────────────
libmpfr.mpfr_get_str.restype = ctypes.c_void_p
libmpfr.mpfr_get_str.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_long), ctypes.c_int,
    ctypes.c_size_t, ctypes.POINTER(mpfr_t), ctypes.c_int]
libmpfr.mpfr_free_str.argtypes = [ctypes.c_void_p]
MPFR_RNDN = 0
def mpfr_to_mpf(p):
    exp = ctypes.c_long()
    raw = libmpfr.mpfr_get_str(None, ctypes.byref(exp), 10, 0, p, MPFR_RNDN)
    txt = ctypes.cast(raw, ctypes.c_char_p).value.decode()
    libmpfr.mpfr_free_str(raw)
    if txt == '0':
        return mp.mpf(0)
    mant = txt[0] + ('.' + txt[1:] if len(txt) > 1 else '')
    return mp.mpf(mant) * mp.power(10, exp.value - 1)

# ── minimal subset of libmps API we need ────────────────────────────
libmps.mps_context_new.restype = ctypes.c_void_p
libmps.mps_monomial_poly_new.restype = ctypes.c_void_p
libmps.mps_monomial_poly_new.argtypes = [ctypes.c_void_p, ctypes.c_int]
libmps.mps_monomial_poly_set_coefficient_d.argtypes = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
    ctypes.c_double, ctypes.c_double)
libmps.mps_context_set_input_poly.argtypes=[ctypes.c_void_p, ctypes.c_void_p]
libmps.mps_context_set_input_prec.argtypes=[ctypes.c_void_p, ctypes.c_long]
libmps.mps_context_set_output_prec.argtypes=[ctypes.c_void_p, ctypes.c_long]
libmps.mps_context_set_output_goal.argtypes=[ctypes.c_void_p, ctypes.c_int]
libmps.mps_context_select_algorithm.argtypes=[ctypes.c_void_p, ctypes.c_int]
libmps.mps_mpsolve.argtypes=[ctypes.c_void_p]
libmps.mps_context_get_degree.restype=ctypes.c_int
libmps.mps_context_get_roots_m.argtypes=[
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.POINTER(mpc_t)),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double))]
libmps.mps_polynomial_free.argtypes=[ctypes.c_void_p, ctypes.c_void_p]
libmps.mps_context_free.argtypes=[ctypes.c_void_p]

MPS_OUTPUT_GOAL_APPROXIMATE = 1
MPS_ALGORITHM_SECULAR_GA    = 1

# ── public helper ───────────────────────────────────────────────────
def roots_mp(coeffs, precision=1024):
    coeffs  = [complex(c) for c in coeffs]
    degree  = len(coeffs) - 1
    mp.prec = precision

    ctx = libmps.mps_context_new()
    libmps.mps_context_set_input_prec (ctx, precision)
    libmps.mps_context_set_output_prec(ctx, precision)
    libmps.mps_context_set_output_goal(ctx, MPS_OUTPUT_GOAL_APPROXIMATE)
    libmps.mps_context_select_algorithm(ctx, MPS_ALGORITHM_SECULAR_GA)

    poly = libmps.mps_monomial_poly_new(ctx, degree)
    for k, c in enumerate(reversed(coeffs)):          # a₀ … aₙ
        libmps.mps_monomial_poly_set_coefficient_d(ctx, poly, k, c.real, c.imag)

    libmps.mps_context_set_input_poly(ctx, poly)
    libmps.mps_mpsolve(ctx)

    deg = libmps.mps_context_get_degree(ctx)          # query BEFORE fetch
    roots_pp = ctypes.POINTER(mpc_t)()
    rads_pp  = ctypes.POINTER(ctypes.c_double)()
    libmps.mps_context_get_roots_m(ctx,
        ctypes.byref(roots_pp), ctypes.byref(rads_pp))

    # convert immediately while context is still alive
    roots = []
    rads  = []
    for i in range(deg):
        z = roots_pp[i]
        roots.append(mp.mpc(mpfr_to_mpf(ctypes.byref(z.re)),
                            mpfr_to_mpf(ctypes.byref(z.im))))
        rads.append(mp.mpf(rads_pp[i]) if bool(rads_pp) else None)

    # now it's safe to free
    libmps.mps_polynomial_free(ctx, poly)
    libmps.mps_context_free(ctx)
    return roots, rads

# ── demo / quick self‑check ─────────────────────────────────────────
if __name__ == "__main__":
    poly = np.array([1, 4, 3, 2, 1+1j], dtype=np.complex128)
    rt, rd = roots_mp(poly, precision=1024)
    for z, r in zip(rt, rd):
        print(f"{z}\n   radius ≤ {r}")
    print("\nmax residual:",
          max(abs(np.polyval(poly, [complex(z) for z in rt]))))

