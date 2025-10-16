# ops_coeff.py
# moebius-type transforms
# used pre-solver
import numpy as np
from numba.typed import Dict
from numba import types, njit
import argparse
import ast

@njit(cache=True, fastmath=True)
def _safe_div(top: np.complex128, bot: np.complex128, eps: float = 1e-12) -> np.complex128:
    # Tikhonov-regularized division: top / bot ≈ top*conj(bot)/( |bot|^2 + eps^2 )
    br = bot.real; bi = bot.imag
    denom = br*br + bi*bi + eps*eps
    tr = top.real; ti = top.imag
    num_r = tr*br + ti*bi
    num_i = ti*br - tr*bi
    return (num_r/denom) + 1j*(num_i/denom)


def op_coeff1(z,a,state):
    zz = (z.real).astype(np.complex128)
    return zz

def op_coeff2(z,a,state):
    tt1 = z[0] + z[1]
    tt2 = z[0] * z[1]
    return np.array([tt1,tt2],dtype=np.complex128)

def op_coeff3(z,a,state):
    tt1 = 1 / ( z[0] + 2 )
    tt2 = 1 / ( z[1] + 2 )
    return np.array([tt1,tt2],dtype=np.complex128)

def op_coeff3_safe(z,a,state):
    t1, t2 = z[0], z[1] 
    tt1 = _safe_div( 1, t1 + 2)
    tt2 = _safe_div( 1, t2 + 2 )
    return np.array([tt1,tt2],dtype=np.complex128)

def op_coeff4(z,a,state):
    tt1 = np.cos(z[0])
    tt2 = np.sin(z[1])
    return np.array([tt1,tt2],dtype=np.complex128)

def op_coeff5(z,a,state):
    tt1 = z[0] + (1.0+0.0j) / z[1]
    tt2 = z[1] + (1.0+0.0j) / z[0]
    return np.array([tt1,tt2],dtype=np.complex128)

def op_coeff5_safe(z,a,state):
    t1, t2 = z[0], z[1] 
    tt1 = t1 + _safe_div( 1.0 + 0.0j, t2 )
    tt2 = t2 + _safe_div( 1.0 + 0.0j, t1 )
    return  np.array([tt1,tt2],dtype=np.complex128)

def op_coeff6(z,a,state):
    num1 = z[0]*z[0]*z[0] + 1j
    den1 = z[0]*z[0]*z[0] - 1j
    val1 = num1 / den1
    num2 = z[1]*z[1]*z[1] + 1j
    den2 = z[1]*z[1]*z[1] - 1j
    val2 = num2 / den2
    return np.array([val1,val2],dtype=np.complex128)

def op_coeff7(z,a,state):
    t1, t2 = z[0], z[1]
    top1  = t1 + np.sin(t1)
    bot1  = t1 + np.cos(t1)
    val1  = top1 / bot1 
    top2  = t2 + np.sin(t2)
    bot2  = t2 + np.cos(t2)
    val2  = top2 / bot2 
    return np.array([val1,val2],dtype=np.complex128)

def op_coeff8(z,a,state):
    t1, t2 = z[0], z[1]
    top1  = t1 + np.sin(t2)
    bot1  = t2 + np.cos(t1)
    val1  = top1 / bot1
    top2  = t2 + np.sin(t1)
    bot2  = t1 + np.cos(t2)
    val2  = top2 / bot2
    return np.array([val1,val2],dtype=np.complex128)

def op_coeff9(z,a,state):
    t1, t2 = z[0], z[1]
    top1  = t1*t1 + 1j * t2
    bot1  = t1*t1 - 1j * t2
    val1  = top1 / bot1
    top2  = t2*t2 + 1j * t1
    bot2  = t2*t2 - 1j * t1
    val2  = top2 / bot2
    return np.array([val1,val2],dtype=np.complex128)

def op_coeff10(z,a,state):
    t1, t2 = z[0], z[1]
    top1 = t1*t1*t1*t1 - t2
    bot1 = t1*t1*t1*t1 + t2
    val1 = top1/bot1
    top2 = t2*t2*t2*t2 - t1
    bot2 = t2*t2*t2*t2 + t1
    val2 = top2/bot2
    return np.array([val1,val2],dtype=np.complex128)

def op_coeff11(z,a,state):
    t1, t2 = z[0], z[1]
    val1 = np.log( t1**4 + 2 )
    val2 = np.log( t2**4 + 2 )
    return np.array([val1,val2],dtype=np.complex128)

def op_coeff12(z,a,state):
    t1, t2 = z[0], z[1]
    val1 = 2*t1**4 - 3*t2**3 + 4*t1**2 - 5*t2
    val2 = 2*t2**4 - 3*t1**3 + 4*t2**2 - 5*t1
    return np.array([val1,val2],dtype=np.complex128)


ALLOWED = {
    "cf1":    op_coeff1,
    "cf2":    op_coeff2,
    "cf3":    op_coeff3,
    "cf3s":   op_coeff3_safe,
    "cf4":    op_coeff4,
    "cf5":    op_coeff5,
    "cf5s":   op_coeff5_safe,
    "cf6":    op_coeff6,
    "cf7":    op_coeff7,
    "cf8":    op_coeff8,
    "cf9":    op_coeff9,
    "cf10":   op_coeff10,
    "cf11":   op_coeff11,
    "cf12":   op_coeff12,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fun", type=str, default="nop")
    ap.add_argument("--z", type=str, default="0+0j")
    ap.add_argument("--a", type=str, default="0+0j")
    args = ap.parse_args()
    z = np.array(ast.literal_eval(args.z),dtype=np.complex128)
    a = np.array(ast.literal_eval(args.a),dtype=np.complex128)
    state  = Dict.empty(key_type=types.int8,value_type=types.complex128[:])
    print(f"{args.fun}({args.z},{args.a}) = {ALLOWED[args.fun](z,a,state)}")

if __name__ == "__main__":
    main()


