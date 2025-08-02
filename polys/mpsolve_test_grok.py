"""
mpsolve_test.py – minimal MPSolve ctypes wrapper

=================================================

• mpsolve(coeffs [, precision=100])

    low precision double input & output

• mpsolve_double(coeffs [, precision=256])

    high precision with double coeffs

• mpsolve_str(coeffs [, precision=256])

    high precision with string coeffs

• mpsolve_int(coeffs [, precision=256])

    high precision with int coeffs

Both expect `coeffs` in NumPy convention: highest power first.

"""

import ctypes, ctypes.util, numpy as np, os

from mpmath import mp

# ------------------------------------------------------------------------------

# 0. Load libraries

# ------------------------------------------------------------------------------

print("Step 0: Loading libraries...")

_MPS_PATH = "/usr/local/lib/libmps.dylib"

if not os.path.exists(_MPS_PATH):

    raise FileNotFoundError(f"MPSolve not found at {_MPS_PATH}")

_mps = ctypes.CDLL(_MPS_PATH, mode=ctypes.RTLD_GLOBAL)

_libgmp = ctypes.CDLL(ctypes.util.find_library("gmp"))

_libc = ctypes.CDLL(ctypes.util.find_library("c"))

print("Libraries loaded.")

# ------------------------------------------------------------------------------

# 1. C structures

# ------------------------------------------------------------------------------

class Cplx(ctypes.Structure): # for double interface

    _fields_ = [("real", ctypes.c_double),

                ("imag", ctypes.c_double)]

    def __complex__(self):

        return complex(self.real, self.imag)

class mpf_t(ctypes.Structure):

    _fields_ = [('_mp_prec', ctypes.c_int),

                ('_mp_size', ctypes.c_int),

                ('_mp_exp', ctypes.c_long),

                ('_mp_d', ctypes.POINTER(ctypes.c_ulong))]

class mpc_t(ctypes.Structure): # MPSolve's custom mpc_t using GMP mpf_t

    _fields_ = [("re", mpf_t),

                ("im", mpf_t)]

# ------------------------------------------------------------------------------

# 2. GMP helpers (get string & free) with debugging

# ------------------------------------------------------------------------------

_libgmp.__gmpf_get_prec.restype = ctypes.c_ulong

_libgmp.__gmpf_get_prec.argtypes = [ctypes.POINTER(mpf_t)]

_libgmp.__gmpf_get_str.restype = ctypes.c_char_p

_libgmp.__gmpf_get_str.argtypes = [

    ctypes.c_char_p, ctypes.POINTER(ctypes.c_long), ctypes.c_int,

    ctypes.c_size_t, ctypes.POINTER(mpf_t)]

def _mpf_to_mpf(mpf):

    print(" Converting mpf_t to mpf...")

    print(f" Prec (limbs): {mpf._mp_prec}, size: {mpf._mp_size}, exp: {mpf._mp_exp}, d: {mpf._mp_d}")

    if mpf._mp_size == 0:

        print(" Zero value (size=0), returning 0")

        return mp.mpf(0)

    prec_bits = _libgmp.__gmpf_get_prec(ctypes.byref(mpf))

    print(f" Precision in bits: {prec_bits}")

    if prec_bits == 0:

        print(" Invalid precision, returning 0")

        return mp.mpf(0)

    # Calculate number of decimal digits: ceil(prec_bits * log10(2)) + extra

    import math

    num_digits = math.ceil(prec_bits * math.log10(2)) + 10  # extra for safety

    buffer_size = num_digits + 2  # for sign and null

    str_buf = (ctypes.c_char * buffer_size)()

    exp = ctypes.c_long()

    print(" Calling mpf_get_str with pre-allocated buffer...")

    raw = _libgmp.__gmpf_get_str(str_buf, ctypes.byref(exp), 10, num_digits, ctypes.byref(mpf))

    if not raw:

        print(" mpf_get_str returned NULL, returning 0")

        return mp.mpf(0)

    print(" Got string pointer")

    txt = (ctypes.cast(raw, ctypes.c_char_p).value or b'').decode()

    print(f" Decoded string: {txt}, exp: {exp.value}")

    if txt == '0':

        return mp.mpf(0)

    l = len(txt) - (1 if txt.startswith('-') else 0)

    result = mp.mpf(txt) * mp.power(10, exp.value - l)

    print(" Computed mpf value")

    return result

# ------------------------------------------------------------------------------

# 3. ctypes signatures with added string coeff and mpc

# ------------------------------------------------------------------------------

# pointers returned

_mps.mps_context_new.restype = ctypes.c_void_p

_mps.mps_monomial_poly_new.restype = ctypes.c_void_p

_mps.mps_monomial_poly_new.argtypes = [ctypes.c_void_p, ctypes.c_int]

# setters / config

_mps.mps_context_set_input_prec .argtypes = [ctypes.c_void_p, ctypes.c_long]

_mps.mps_context_set_output_prec.argtypes = [ctypes.c_void_p, ctypes.c_long]

_mps.mps_context_set_output_goal.argtypes = [ctypes.c_void_p, ctypes.c_int]

_mps.mps_context_select_algorithm.argtypes = [ctypes.c_void_p, ctypes.c_int]

_mps.mps_monomial_poly_set_coefficient_d.argtypes = (

    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,

    ctypes.c_double, ctypes.c_double)

_mps.mps_monomial_poly_set_coefficient_int.argtypes = (

    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,

    ctypes.c_longlong, ctypes.c_longlong)

_mps.mps_monomial_poly_set_coefficient_s.argtypes = (

    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,

    ctypes.c_char_p, ctypes.c_char_p)

_mps.mps_context_set_input_poly.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

_mps.mps_mpsolve.argtypes = [ctypes.c_void_p]

# getters

_mps.mps_context_get_degree.argtypes = [ctypes.c_void_p]

_mps.mps_context_get_degree.restype = ctypes.c_int

_mps.mps_context_get_roots_d.argtypes = [

    ctypes.c_void_p,

    ctypes.POINTER(ctypes.POINTER(Cplx)),

    ctypes.POINTER(ctypes.POINTER(ctypes.c_double))]

_mps.mps_context_get_roots_d.restype = ctypes.c_int

_mps.mps_context_get_roots_m.argtypes = [

    ctypes.c_void_p,

    ctypes.POINTER(ctypes.POINTER(mpc_t)),

    ctypes.POINTER(ctypes.POINTER(ctypes.c_double))]

_mps.mps_context_get_roots_m.restype = ctypes.c_int

# free

_mps.mps_polynomial_free.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

_mps.mps_context_free.argtypes = [ctypes.c_void_p]

# gmp bindings

_libgmp.__gmpf_clear.argtypes = [ctypes.POINTER(mpf_t)]

# libc free

_libc.free.argtypes = [ctypes.c_void_p]

# ------------------------------------------------------------------------------

# 4. Constants

# ------------------------------------------------------------------------------

MPS_OUTPUT_GOAL_APPROXIMATE = 1

MPS_ALGORITHM_SECULAR_GA = 1

# ------------------------------------------------------------------------------

# 5. Low precision double wrapper

# ------------------------------------------------------------------------------

def mpsolve(coeffs, precision: int = 100):

    print("Starting low precision double solve...")

    coeffs = np.asarray(coeffs, dtype=np.complex128)

    degree = len(coeffs) - 1

    print(f"Degree: {degree}")

    ctx = _mps.mps_context_new()

    print("Context created at address", hex(ctypes.cast(ctx, ctypes.c_void_p).value))

    _mps.mps_context_set_input_prec (ctx, precision)

    print("Input prec set to", precision)

    _mps.mps_context_set_output_prec(ctx, precision)

    print("Output prec set to", precision)

    _mps.mps_context_set_output_goal(ctx, MPS_OUTPUT_GOAL_APPROXIMATE)

    print("Goal set")

    _mps.mps_context_select_algorithm(ctx, MPS_ALGORITHM_SECULAR_GA)

    print("Algorithm set")

    poly = _mps.mps_monomial_poly_new(ctx, degree)

    print("Poly created at address", hex(ctypes.cast(poly, ctypes.c_void_p).value))

    for exp, c in enumerate(coeffs[::-1]): # a0, a1, …

        print(f"Setting coeff {exp}: {c.real} + {c.imag}j using double")

        _mps.mps_monomial_poly_set_coefficient_d(

            ctx, poly, exp, c.real, c.imag)

    _mps.mps_context_set_input_poly(ctx, poly)

    print("Input poly set")

    _mps.mps_mpsolve(ctx)

    print("Solved")

    n = _mps.mps_context_get_degree(ctx)

    print(f"Got degree: {n}")

    roots_p = ctypes.POINTER(Cplx)()

    rad_p = ctypes.POINTER(ctypes.c_double)()

    status = _mps.mps_context_get_roots_d(ctx, ctypes.byref(roots_p), ctypes.byref(rad_p))

    print(f"Got roots double, status: {status}, roots_p address: {hex(ctypes.cast(roots_p, ctypes.c_void_p).value)}, rad_p address: {hex(ctypes.cast(rad_p, ctypes.c_void_p).value)}")

    roots = np.array([complex(roots_p[i]) for i in range(n)], dtype=np.complex128)

    _libc.free(roots_p)

    _libc.free(rad_p)

    _mps.mps_polynomial_free(ctx, poly)

    _mps.mps_context_free(ctx)

    print("Freed resources")

    return roots

# ------------------------------------------------------------------------------

# 6a. High precision with double coeffs

# ------------------------------------------------------------------------------

def mpsolve_double(coeffs, precision: int = 256):

    print(f"Starting high precision solve with double coeffs at {precision} bits")

    coeffs = [complex(c) for c in coeffs]

    degree = len(coeffs) - 1

    mp.prec = precision

    print(f"Degree: {degree}, mp.prec set to {mp.prec}")

    ctx = _mps.mps_context_new()

    print("Context created at address", hex(ctypes.cast(ctx, ctypes.c_void_p).value))

    _mps.mps_context_set_input_prec (ctx, 53)

    print("Input prec set to 53")

    _mps.mps_context_set_output_prec(ctx, precision)

    print("Output prec set to", precision)

    _mps.mps_context_set_output_goal(ctx, MPS_OUTPUT_GOAL_APPROXIMATE)

    print("Goal set")

    _mps.mps_context_select_algorithm(ctx, MPS_ALGORITHM_SECULAR_GA)

    print("Algorithm set")

    poly = _mps.mps_monomial_poly_new(ctx, degree)

    print("Poly created at address", hex(ctypes.cast(poly, ctypes.c_void_p).value))

    for exp, c in enumerate(reversed(coeffs)): # a0 first

        print(f"Setting coeff x^{exp}: {c.real} + {c.imag}j using double")

        _mps.mps_monomial_poly_set_coefficient_d(ctx, poly, exp, c.real, c.imag)

    _mps.mps_context_set_input_poly(ctx, poly)

    print("Input poly set")

    _mps.mps_mpsolve(ctx)

    print("Solved")

    n = _mps.mps_context_get_degree(ctx)

    print(f"Got degree: {n}")

    roots_p = ctypes.POINTER(mpc_t)()

    rad_p = ctypes.POINTER(ctypes.c_double)()

    status = _mps.mps_context_get_roots_m(ctx, ctypes.byref(roots_p), ctypes.byref(rad_p))

    print(f"Got roots MP, status: {status}, roots_p address: {hex(ctypes.cast(roots_p, ctypes.c_void_p).value)}, rad_p address: {hex(ctypes.cast(rad_p, ctypes.c_void_p).value)}")

    roots = np.empty(n, dtype=object)

    radii = np.empty(n, dtype=object)

    for i in range(n):

        print(f"Processing root {i}")

        z = roots_p[i]

        print(" Got z ptr")

        print(f" Re prec (limbs): {z.re._mp_prec}, size: {z.re._mp_size}, exp: {z.re._mp_exp}, d: {z.re._mp_d}")

        print(f" Im prec (limbs): {z.im._mp_prec}, size: {z.im._mp_size}, exp: {z.im._mp_exp}, d: {z.im._mp_d}")

        re = _mpf_to_mpf(z.re)

        print(" Converted re")

        im = _mpf_to_mpf(z.im)

        print(" Converted im")

        roots[i] = mp.mpc(re, im)

        print(" Created mpc")

        radii[i] = mp.mpf(rad_p[i])

        print(" Converted radius")

    for i in range(n):

        _libgmp.__gmpf_clear(ctypes.byref(roots_p[i].re))

        _libgmp.__gmpf_clear(ctypes.byref(roots_p[i].im))

        print(f"Cleared root {i}")

    _libc.free(roots_p)

    _libc.free(rad_p)

    _mps.mps_polynomial_free(ctx, poly)

    _mps.mps_context_free(ctx)

    print("Freed resources")

    return roots, radii

# ------------------------------------------------------------------------------

# 6b. High precision with string coeffs

# ------------------------------------------------------------------------------

def mpsolve_str(coeffs, precision: int = 256):

    print(f"Starting high precision solve with string coeffs at {precision} bits")

    coeffs = [complex(c) for c in coeffs]

    degree = len(coeffs) - 1

    mp.prec = precision

    print(f"Degree: {degree}, mp.prec set to {mp.prec}")

    ctx = _mps.mps_context_new()

    print("Context created at address", hex(ctypes.cast(ctx, ctypes.c_void_p).value))

    _mps.mps_context_set_input_prec (ctx, 0)

    print("Input prec set to 0")

    _mps.mps_context_set_output_prec(ctx, precision)

    print("Output prec set to", precision)

    _mps.mps_context_set_output_goal(ctx, MPS_OUTPUT_GOAL_APPROXIMATE)

    print("Goal set")

    _mps.mps_context_select_algorithm(ctx, MPS_ALGORITHM_SECULAR_GA)

    print("Algorithm set")

    poly = _mps.mps_monomial_poly_new(ctx, degree)

    print("Poly created at address", hex(ctypes.cast(poly, ctypes.c_void_p).value))

    for exp, c in enumerate(reversed(coeffs)): # a0 first

        real_part = str(c.real)  # Use str to support fractions

        imag_part = str(c.imag)

        print(f"Setting coeff x^{exp}: {real_part} + {imag_part}j using string")

        real_str = real_part.encode('ascii')

        imag_str = imag_part.encode('ascii')

        _mps.mps_monomial_poly_set_coefficient_s(ctx, poly, exp, real_str, imag_str)

    _mps.mps_context_set_input_poly(ctx, poly)

    print("Input poly set")

    _mps.mps_mpsolve(ctx)

    print("Solved")

    n = _mps.mps_context_get_degree(ctx)

    print(f"Got degree: {n}")

    roots_p = ctypes.POINTER(mpc_t)()

    rad_p = ctypes.POINTER(ctypes.c_double)()

    status = _mps.mps_context_get_roots_m(ctx, ctypes.byref(roots_p), ctypes.byref(rad_p))

    print(f"Got roots MP, status: {status}, roots_p address: {hex(ctypes.cast(roots_p, ctypes.c_void_p).value)}, rad_p address: {hex(ctypes.cast(rad_p, ctypes.c_void_p).value)}")

    roots_np = np.empty(n, dtype=object)

    radii = np.empty(n, dtype=object)

    for i in range(n):

        print(f"Processing root {i}")

        z = roots_p[i]

        print(" Got z")

        print(f" Re prec (limbs): {z.re._mp_prec}, size: {z.re._mp_size}, exp: {z.re._mp_exp}, d: {z.re._mp_d}")

        print(f" Im prec (limbs): {z.im._mp_prec}, size: {z.im._mp_size}, exp: {z.im._mp_exp}, d: {z.im._mp_d}")

        re = _mpf_to_mpf(z.re)

        print(" Converted re")

        im = _mpf_to_mpf(z.im)

        print(" Converted im")

        roots_np[i] = mp.mpc(re, im)

        print(" Created mpc")

        radii[i] = mp.mpf(rad_p[i])

        print(" Converted radius")

    for i in range(n):

        _libgmp.__gmpf_clear(ctypes.byref(roots_p[i].re))

        _libgmp.__gmpf_clear(ctypes.byref(roots_p[i].im))

        print(f"Cleared root {i}")

    _libc.free(roots_p)

    _libc.free(rad_p)

    _mps.mps_polynomial_free(ctx, poly)

    _mps.mps_context_free(ctx)

    print("Freed resources")

    return roots_np, radii

# ------------------------------------------------------------------------------

# 6c. High precision with int coeffs

# ------------------------------------------------------------------------------

def mpsolve_int(coeffs, precision: int = 256):

    print(f"Starting high precision solve with int coeffs at {precision} bits")

    coeffs = [complex(c) for c in coeffs]

    degree = len(coeffs) - 1

    mp.prec = precision

    print(f"Degree: {degree}, mp.prec set to {mp.prec}")

    ctx = _mps.mps_context_new()

    print("Context created at address", hex(ctypes.cast(ctx, ctypes.c_void_p).value))

    _mps.mps_context_set_input_prec (ctx, 0)

    print("Input prec set to 0")

    _mps.mps_context_set_output_prec(ctx, precision)

    print("Output prec set to", precision)

    _mps.mps_context_set_output_goal(ctx, MPS_OUTPUT_GOAL_APPROXIMATE)

    print("Goal set")

    _mps.mps_context_select_algorithm(ctx, MPS_ALGORITHM_SECULAR_GA)

    print("Algorithm set")

    poly = _mps.mps_monomial_poly_new(ctx, degree)

    print("Poly created at address", hex(ctypes.cast(poly, ctypes.c_void_p).value))

    for exp, c in enumerate(reversed(coeffs)): # a0 first

        real_part = int(c.real)

        imag_part = int(c.imag)

        print(f"Setting coeff x^{exp}: {real_part} + {imag_part}j using int")

        _mps.mps_monomial_poly_set_coefficient_int(ctx, poly, exp, ctypes.c_longlong(real_part), ctypes.c_longlong(imag_part))

    _mps.mps_context_set_input_poly(ctx, poly)

    print("Input poly set")

    _mps.mps_mpsolve(ctx)

    print("Solved")

    n = _mps.mps_context_get_degree(ctx)

    print(f"Got degree: {n}")

    roots_p = ctypes.POINTER(mpc_t)()

    rad_p = ctypes.POINTER(ctypes.c_double)()

    status = _mps.mps_context_get_roots_m(ctx, ctypes.byref(roots_p), ctypes.byref(rad_p))

    print(f"Got roots MP, status: {status}, roots_p address: {hex(ctypes.cast(roots_p, ctypes.c_void_p).value)}, rad_p address: {hex(ctypes.cast(rad_p, ctypes.c_void_p).value)}")

    roots_np = np.empty(n, dtype=object)

    radii = np.empty(n, dtype=object)

    for i in range(n):

        print(f"Processing root {i}")

        z = roots_p[i]

        print(" Got z")

        print(f" Re prec (limbs): {z.re._mp_prec}, size: {z.re._mp_size}, exp: {z.re._mp_exp}, d: {z.re._mp_d}")

        print(f" Im prec (limbs): {z.im._mp_prec}, size: {z.im._mp_size}, exp: {z.im._mp_exp}, d: {z.im._mp_d}")

        re = _mpf_to_mpf(z.re)

        print(" Converted re")

        im = _mpf_to_mpf(z.im)

        print(" Converted im")

        roots_np[i] = mp.mpc(re, im)

        print(" Created mpc")

        radii[i] = mp.mpf(rad_p[i])

        print(" Converted radius")

    for i in range(n):

        _libgmp.__gmpf_clear(ctypes.byref(roots_p[i].re))

        _libgmp.__gmpf_clear(ctypes.byref(roots_p[i].im))

        print(f"Cleared root {i}")

    _libc.free(roots_p)

    _libc.free(rad_p)

    _mps.mps_polynomial_free(ctx, poly)

    _mps.mps_context_free(ctx)

    print("Freed resources")

    return roots_np, radii

# ------------------------------------------------------------------------------

# GMP Test with debugging

# ------------------------------------------------------------------------------

def run_gmp_test():

    print("Starting GMP test...")

    _libgmp.__gmpf_init2.argtypes = [ctypes.POINTER(mpf_t), ctypes.c_ulong]

    _libgmp.__gmpf_set_d.argtypes = [ctypes.POINTER(mpf_t), ctypes.c_double]

    _libgmp.__gmpf_add.argtypes = [ctypes.POINTER(mpf_t), ctypes.POINTER(mpf_t),

                                ctypes.POINTER(mpf_t)]

    _libgmp.__gmpf_clear.argtypes = [ctypes.POINTER(mpf_t)]

    print("GMP functions bound")

    a = mpf_t()

    b = mpf_t()

    c = mpf_t()

    print("GMP vars created")

    _libgmp.__gmpf_init2(ctypes.byref(a), 128)

    print("a initialized")

    _libgmp.__gmpf_init2(ctypes.byref(b), 128)

    print("b initialized")

    _libgmp.__gmpf_init2(ctypes.byref(c), 128)

    print("c initialized")

    _libgmp.__gmpf_set_d(ctypes.byref(a), 1.25)

    print("a set to 1.25")

    _libgmp.__gmpf_set_d(ctypes.byref(b), 2.75)

    print("b set to 2.75")

    _libgmp.__gmpf_add(ctypes.byref(c), ctypes.byref(a), ctypes.byref(b))

    print("Added")

    result = _mpf_to_mpf(c)

    print(f"a + b = {result} (expected 4.0)")

    print(f"c prec (limbs): {c._mp_prec}, size: {c._mp_size}, exp: {c._mp_exp}, d: {c._mp_d}")

    _libgmp.__gmpf_clear(ctypes.byref(a))

    _libgmp.__gmpf_clear(ctypes.byref(b))

    _libgmp.__gmpf_clear(ctypes.byref(c))

    print("Cleared GMP vars")

    print("GMP test done")

# ------------------------------------------------------------------------------

# 7. Self‑test

# ------------------------------------------------------------------------------

if __name__ == "__main__":

    print("MPSolve wrapper version 2.3")

    run_gmp_test()

    cf = np.array([1, 2, 3, 4, 1+1j], dtype=np.complex128)

    print("\nLow precision double roots:")

    low_roots = mpsolve(cf)

    print(low_roots)

    print("\nHigh precision string coeffs roots:")

    roots_s, radii_s = mpsolve_str(cf)

    print(roots_s)

    print("\nHigh precision double coeffs roots:")

    roots_d, radii_d = mpsolve_double(cf)

    print(roots_d)

    print("\nHigh precision int coeffs roots:")

    roots_i, radii_i = mpsolve_int(cf)

    print(roots_i)
    