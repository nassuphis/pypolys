# ops_xfrm.py
#
# z2 = f(z1)
# z has length 2
# these are parameter
# transforms

import numpy as np
from numba.typed import Dict
from numba import njit, types
import argparse
import ast

def op_zz(z,a,state):
    v = z[0]+z[1]*1j
    return np.array([v,v],dtype=np.complex128)

def op_zz1(z,a,state):
    v1 = z[0]+z[1]*1j
    v2 = z[0]*z[1]+(z[0]+z[1])*1j
    return np.array([v1,v2],dtype=np.complex128)

def op_zz2(z,a,state):
    v1 = z[0]+z[1]*1j
    v2 = z[0]-z[1]*1j
    return np.array([v1,v2],dtype=np.complex128)

def op_zz3(z,a,state):
    v1 = z[0]+z[1]*1j
    v2 = z[1]+z[0]*1j
    return np.array([v1,v2],dtype=np.complex128)


def op_pz(z,a,state):
    z0 = z[0]
    z1 = z[1]
    a0 = a[0]
    a1 = a[1]
    a2 = a[2]
    a3 = a[3]
    p0 = a0+a1*z0+a2*z0**2+a3*z0**3
    p1 = a0+a1*z1+a2*z1**2+a3*z1**3
    return np.array([p0,p1],dtype=np.complex128)

############################################
# Baker's map (mod1 mapping)
############################################
@njit
def bkr1(t):
    x , y = np.real(t), np.imag(t)
    x_fold, y_fold = x % 1 , y % 1  
    x_new = (2 * x_fold) % 1
    shift = np.floor(2 * x_fold)
    y_new = (y_fold + shift) / 2
    return x_new + 1j * y_new

def op_bkr(z,a,state):
  n = int(a[0].real)
  if n==0:
      return z
  out = z
  for i in range(z.size):
    for _ in range(n):
        out[i] = bkr1(out[i]) 
  return  out


ALLOWED = {
    "zz":    op_zz,
    "zz1":   op_zz1,
    "zz2":   op_zz2,
    "zz3":   op_zz3,
    "pz":    op_pz,
    "bkr":   op_bkr,
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
