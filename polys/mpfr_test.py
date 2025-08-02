# mpfr_test.py  – sanity check
import ctypes, ctypes.util
from mpmath import mp

# --- load libmpfr ----------------------------------------------------
libmpfr = ctypes.CDLL(ctypes.util.find_library("mpfr"))

# --- exact struct on macOS/ARM64 ------------------------------------
class mpfr_t(ctypes.Structure):
    _fields_ = [('_mpfr_prec', ctypes.c_int),          # 32 bit
                ('_mpfr_sign', ctypes.c_int),          # 32 bit
                ('_mpfr_exp',  ctypes.c_long),         # 64 bit
                ('_mpfr_d',    ctypes.POINTER(ctypes.c_ulong))]

# --- prototypes ------------------------------------------------------
libmpfr.mpfr_init2.argtypes  = [ctypes.POINTER(mpfr_t), ctypes.c_ulong]
libmpfr.mpfr_set_d.argtypes  = [ctypes.POINTER(mpfr_t), ctypes.c_double, ctypes.c_int]
libmpfr.mpfr_add.argtypes    = [ctypes.POINTER(mpfr_t), ctypes.POINTER(mpfr_t),
                                ctypes.POINTER(mpfr_t), ctypes.c_int]
libmpfr.mpfr_clear.argtypes  = [ctypes.POINTER(mpfr_t)]

libmpfr.mpfr_get_str.restype = ctypes.c_void_p          # ← key change
libmpfr.mpfr_get_str.argtypes = [ctypes.c_void_p,       # char ** (malloced)
                                 ctypes.POINTER(ctypes.c_long),
                                 ctypes.c_int, ctypes.c_size_t,
                                 ctypes.POINTER(mpfr_t), ctypes.c_int]
libmpfr.mpfr_free_str.argtypes = [ctypes.c_void_p]

MPFR_RNDN = 0

def mpfr_to_mpf(ptr):
    exp = ctypes.c_long()
    raw = libmpfr.mpfr_get_str(None, ctypes.byref(exp), 10, 0, ptr, MPFR_RNDN)
    if not raw:
        return mp.mpf(0)
    txt = ctypes.cast(raw, ctypes.c_char_p).value.decode()
    libmpfr.mpfr_free_str(raw)                  # free original C buffer
    dec = txt[0] + ('.' + txt[1:] if len(txt) > 1 else '')
    return mp.mpf(dec) * mp.power(10, exp.value - 1)

# --- do a + b --------------------------------------------------------
a = mpfr_t(); b = mpfr_t(); c = mpfr_t()
libmpfr.mpfr_init2(ctypes.byref(a), 128)
libmpfr.mpfr_init2(ctypes.byref(b), 128)
libmpfr.mpfr_init2(ctypes.byref(c), 128)

libmpfr.mpfr_set_d(ctypes.byref(a), 1.25, MPFR_RNDN)
libmpfr.mpfr_set_d(ctypes.byref(b), 2.75, MPFR_RNDN)
libmpfr.mpfr_add  (ctypes.byref(c), ctypes.byref(a), ctypes.byref(b), MPFR_RNDN)

print("a + b =", mpfr_to_mpf(ctypes.byref(c)))   # prints 4.0
