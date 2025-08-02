import ctypes
class mpfr_t(ctypes.Structure):
    _fields_ = [('_mpfr_prec', ctypes.c_long),
                ('_mpfr_sign', ctypes.c_int),
                ('_pad', ctypes.c_int),
                ('_mpfr_exp', ctypes.c_long),
                ('_mpfr_d', ctypes.POINTER(ctypes.c_ulong))]
print(ctypes.sizeof(mpfr_t))  # Should be 32