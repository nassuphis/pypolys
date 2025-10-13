# ops_tfrm.py
#
# transforms: zz = f(z)
# applied term-by-term
import numpy as np
from numba.typed import Dict
from numba import njit, types
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

def op_abs(z,a,state):
    z1 = np.abs(z.real)+np.abs(z.imag)*1j
    return z1

def op_conj(z,a,state):
    z1 = np.conj(z)
    return z1

def op_uc(z,a,state):       
    zz = np.exp(1j*2*np.pi*z)
    return zz

def op_round(z,a,state):
    n=int(a[0].real)
    return np.round(z,n)

def op_plk(z,a,state):
    out = z
    for i in range(z.size):
        zi=z[i]
        if np.isnan(zi.real): continue 
        if np.isnan(zi.imag): continue
        if not np.isfinite(zi.real): continue 
        if not np.isfinite(zi.imag): continue
        if np.abs(zi.imag)>100: continue
        zsin = np.sin(zi)
        if np.isnan(zsin.real): continue 
        if np.isnan(zsin.imag): continue
        if not np.isfinite(zsin.real): continue
        if not np.isfinite(zsin.imag): continue
        if np.abs(zsin.real)>1e10: continue
        if np.abs(zsin.imag)>1e10: continue
        zcos = np.cos(zi)
        if np.isnan(zcos.real): continue 
        if np.isnan(zcos.imag): continue
        if not np.isfinite(zcos.real): continue
        if not np.isfinite(zcos.imag): continue
        if np.abs(zcos.real)<1e-10: continue
        if np.abs(zcos.imag)<1e-10: continue
        if np.abs(zcos.real)>1e10: continue
        if np.abs(zcos.imag)>1e10: continue
        out[i] = zsin/zcos
    return out

def op_pert1(z,a,state):
    n = z.size
    fac = a[0].real
    rr = fac * 2 * (np.random.random(n)-0.5)
    ri = fac * 2 * (np.random.random(n)-0.5)
    rp = rr + ri * 1j
    z1 = z+rp
    return z1

def op_pert2(z,a,state):
    n = z.size
    fac = a[0].real
    rr = fac*2*(np.random.random(n)-0.5)
    ri = fac*2*(np.random.random(n)-0.5)
    rp = rr + ri * 1j
    z1 = z*rp
    return z1

ALLOWED = {
    "abs":    op_abs,
    "conj":   op_conj,
    "round":  op_round,
    "uc":     op_uc,
    "pert1":  op_pert1,
    "pert2":  op_pert2,
    "plk":    op_plk,
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
