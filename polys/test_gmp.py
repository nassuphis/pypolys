
import ctypes, ctypes.util

_libgmp = ctypes.CDLL(ctypes.util.find_library("gmp"))
_libc   = ctypes.CDLL(ctypes.util.find_library("c"))
_libc.free.argtypes = [ctypes.c_void_p]

class mpf_t(ctypes.Structure):
    _fields_ = [
        ('_mp_prec', ctypes.c_int),
        ('_mp_size', ctypes.c_int),
        ('_mp_exp', ctypes.c_long),
        ('_mp_d', ctypes.POINTER(ctypes.c_ulong))
    ]
    
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
    ctypes.c_char_p,                      # user buffer (or NULL)
    ctypes.POINTER(ctypes.c_long),        # exponent out
    ctypes.c_int,                         # base
    ctypes.c_size_t,                      # number of digits (0 = all)
    ctypes.POINTER(mpf_t)                 # the mpf_t to convert
]
_libgmp.__gmpf_get_str.restype = ctypes.c_void_p

def mpf_to_str(x: mpf_t, base: int = 10) -> str:
    exp = ctypes.c_long()
    raw_ptr = _libgmp.__gmpf_get_str(None,ctypes.byref(exp),base,0,ctypes.byref(x))
    if not raw_ptr: return "0"
    mant_bytes = ctypes.cast(raw_ptr, ctypes.c_char_p).value
    mant_str   = mant_bytes.decode()
    _libc.free(raw_ptr)
    neg = mant_str.startswith('-')
    if neg: mant_str = mant_str[1:]
    if len(mant_str) == 1:
        s = mant_str + ".0"
    else:
        s = mant_str[0] + "." + mant_str[1:]
    s += f"e{exp.value}"
    return "-" + s if neg else s


a = mpf_t()
b = mpf_t()
c = mpf_t()

_libgmp.__gmpf_init2(ctypes.byref(a), 128)
_libgmp.__gmpf_init2(ctypes.byref(b), 128)
_libgmp.__gmpf_init2(ctypes.byref(c), 128)
_libgmp.__gmpf_set_d(ctypes.byref(a), 1.25)
_libgmp.__gmpf_set_d(ctypes.byref(b), 2.75)
_libgmp.__gmpf_add(ctypes.byref(c), ctypes.byref(a), ctypes.byref(b))

print(f"result: {mpf_to_str(a)}+{mpf_to_str(b)}={mpf_to_str(c)}")

_libgmp.__gmpf_clear(ctypes.byref(a))
_libgmp.__gmpf_clear(ctypes.byref(b))
_libgmp.__gmpf_clear(ctypes.byref(c))

