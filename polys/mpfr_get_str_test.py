"""
mpfr_get_str_test.py - Test harness for mpfr_get_str
"""

import ctypes, ctypes.util

_libmpfr = ctypes.CDLL(ctypes.util.find_library("mpfr"))

class mpfr_t(ctypes.Structure):
    _fields_ = [('_mpfr_prec', ctypes.c_ulong),
                ('_mpfr_sign', ctypes.c_int),
                ('_mpfr_exp',  ctypes.c_long),
                ('_mpfr_d',    ctypes.POINTER(ctypes.c_ulong))]

_libmpfr.mpfr_init2.argtypes  = [ctypes.POINTER(mpfr_t), ctypes.c_ulong]
_libmpfr.mpfr_set_d.argtypes  = [ctypes.POINTER(mpfr_t), ctypes.c_double, ctypes.c_int]
_libmpfr.mpfr_clear.argtypes  = [ctypes.POINTER(mpfr_t)]

_libmpfr.mpfr_get_str.restype = ctypes.c_void_p
_libmpfr.mpfr_get_str.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_long), ctypes.c_int,
    ctypes.c_size_t, ctypes.POINTER(mpfr_t), ctypes.c_int]
_libmpfr.mpfr_free_str.argtypes = [ctypes.c_void_p]

MPFR_RNDN = 0

def test_mpfr_get_str():
    print("Starting mpfr_get_str test...")
    a = mpfr_t()
    _libmpfr.mpfr_init2(ctypes.byref(a), 128)
    print("a initialized")

    _libmpfr.mpfr_set_d(ctypes.byref(a), 3.1415926535, MPFR_RNDN)
    print("a set to pi approx")

    exp = ctypes.c_long()
    raw = _libmpfr.mpfr_get_str(None, ctypes.byref(exp), 10, 0, ctypes.byref(a), MPFR_RNDN)
    if not bool(raw):
        print("mpfr_get_str returned NULL")
        return
    txt = ctypes.cast(raw, ctypes.c_char_p).value.decode()
    print(f"Decoded string: {txt}, exp: {exp.value}")
    _libmpfr.mpfr_free_str(raw)
    print("Freed string")

    _libmpfr.mpfr_clear(ctypes.byref(a))
    print("Cleared a")
    print("Test complete")

if __name__ == "__main__":
    test_mpfr_get_str()

    