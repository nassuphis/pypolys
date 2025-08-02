# dump_mpfr_layout.py  – NO third‑party deps
import ctypes, ctypes.util, numpy as np, os, sys, struct

libmps  = ctypes.CDLL("/usr/local/lib/libmps.dylib", mode=ctypes.RTLD_GLOBAL)

# -- mpfr_t guess #1 (what I used in v0.7) -----------------------------
class mpfr_t(ctypes.Structure):
    _fields_ = [('_prec', ctypes.c_uint),    # 4  bytes
                ('_sign', ctypes.c_int),     # 4
                ('_exp',  ctypes.c_long),    # 8
                ('_d',    ctypes.POINTER(ctypes.c_ulong))]  # 8
class mpc_t (ctypes.Structure):
    _fields_ = [("re", mpfr_t), ("im", mpfr_t)]

# -- minimal MPSolve proto --------------------------------------------
libmps.mps_context_new.restype  = ctypes.c_void_p
libmps.mps_monomial_poly_new.restype = ctypes.c_void_p
libmps.mps_monomial_poly_new.argtypes= [ctypes.c_void_p, ctypes.c_int]
libmps.mps_context_get_degree.restype= ctypes.c_int
libmps.mps_context_get_degree.argtypes=[ctypes.c_void_p]
libmps.mps_context_get_roots_m.argtypes=[
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.POINTER(mpc_t)),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double))]
libmps.mps_context_get_roots_m.restype = ctypes.c_int
libmps.mps_context_set_output_goal.argtypes=[ctypes.c_void_p, ctypes.c_int]
libmps.mps_context_select_algorithm.argtypes=[ctypes.c_void_p, ctypes.c_int]
libmps.mps_context_set_input_poly.argtypes=[ctypes.c_void_p, ctypes.c_void_p]
libmps.mps_mpsolve.argtypes=[ctypes.c_void_p]

MPS_OUTPUT_GOAL_APPROXIMATE=1
MPS_ALGORITHM_SECULAR_GA   =1

# -- build quadratic  z^2+1  ------------------------------------------
ctx = libmps.mps_context_new()
libmps.mps_context_set_output_goal(ctx, MPS_OUTPUT_GOAL_APPROXIMATE)
libmps.mps_context_select_algorithm(ctx, MPS_ALGORITHM_SECULAR_GA)
poly=libmps.mps_monomial_poly_new(ctx,2)
libmps.mps_monomial_poly_set_coefficient_d(ctx,poly,0,1.0,0.0) # a0
libmps.mps_monomial_poly_set_coefficient_d(ctx,poly,1,0.0,0.0) # a1
libmps.mps_monomial_poly_set_coefficient_d(ctx,poly,2,1.0,0.0) # a2
libmps.mps_context_set_input_poly(ctx,poly)
libmps.mps_mpsolve(ctx)

roots_pp=ctypes.POINTER(mpc_t)()
rad_pp  =ctypes.POINTER(ctypes.c_double)()
libmps.mps_context_get_roots_m(ctx,
    ctypes.byref(roots_pp), ctypes.byref(rad_pp))

root0 = roots_pp[0].re                      # first mpfr_t
raw   = ctypes.string_at(ctypes.addressof(root0), ctypes.sizeof(mpfr_t))

print("ctypes.sizeof(mpfr_t) =", ctypes.sizeof(mpfr_t))
print("raw 4×8‑byte words    =", [hex(struct.unpack_from('Q', raw, 8*i)[0])
                                   for i in range(4)])
