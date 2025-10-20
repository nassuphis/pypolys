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

def xim(z,a,state):
    v1 = 1j*z[0].real
    v2 = 1j*z[1].real
    return np.array([v1,v2],dtype=np.complex128)

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

#cardioid
def op_crd(z,a,state):
    n = int(a[0].real)
    if n<0: return z
    if n>z.size-1: return z
    size = a[1].real
    v = z.copy()
    t = v[n].real
    theta = 2 * np.pi * t
    r = size * (1 + np.cos(theta)) * np.exp(1j * theta)
    v[n] = r 
    return v

#heart
def op_hrt(z,a,state):
    n = int(a[0].real)
    if n<0: return z
    if n>z.size-1: return z
    size = a[1].real
    rot = np.exp(1j * 2 * np.pi * a[2].real )
    v = z.copy()
    u = v[n].real
    phi = np.pi/2
    t = 2*np.pi*u+phi
    x = 16 * np.sin(t)**3
    y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
    hrt = x/40 + 1j*y/40 + 0.1j
    v[n] = rot*size*hrt
    return v

#spindle
def op_spdl(z,a,state):
    n = int(a[0].real)
    if n<0: return z
    if n>z.size-1: return z
    va = a[1].real or 0.5
    vb = a[2].real or 0.2
    vp = a[3].real or 1.5
    v = z.copy()
    t = v[n].real
    theta = 2 * np.pi * t
    x = va * np.sign(np.cos(theta)) * np.abs(np.cos(theta))**(2/vp)
    y = vb * np.sign(np.sin(theta)) * np.abs(np.sin(theta))**(2/vp)
    v[n] = x + 1j * y
    return v

def limacon(z,a,state):
    n = int(a[0].real)
    if n<0: return z
    if n>z.size-1: return z
    v = z.copy()
    tp = v[n].real
    ap = a[1].real or 0.3
    bp = a[2].real or 0.5
    theta = 2 * np.pi * tp
    r = ap + bp * np.cos(theta)
    v[n] = r * np.exp(1j * theta)
    return v

def rose_curve(z,a,state):
    n = int(a[0].real)
    if n<0: return z
    if n>z.size-1: return z
    v = z.copy()
    tp = v[n].real
    ap = a[1].real or 0.5
    kp = a[2].real or 2
    theta = 2 * np.pi * tp
    r = ap * np.cos(kp * theta)
    v[n] = r * np.exp(1j * theta)
    return v

def lissajous(z,a,state):
    n = int(a[0].real)
    if n<0: return z
    if n>z.size-1: return z
    v = z.copy()
    tp = v[n].real
    Ap = a[1].real or 0.5  
    Bp = a[2].real or 0.5
    ap = a[3].real or 3
    bp = a[4].real or 2
    cp = a[5].real or 0.5
    delta = np.pi * cp
    theta = 2 * np.pi * tp
    x = Ap * np.sin(ap * theta + delta)
    y = Bp * np.sin(bp * theta)
    v[n] = x + 1j * y
    return v

def astroid(z,a,state):
    n = int(a[0].real)
    if n<0: return z
    if n>z.size-1: return z
    v = z.copy()
    tp = v[n].real
    ap = a[1].real or 0.5
    theta = 2 * np.pi * tp
    x = ap * np.cos(theta)**3
    y = ap * np.sin(theta)**3
    v[n] = x + 1j * y
    return v

def archimedean_spiral(z,a,state):
    n = int(a[0].real)
    if n<0: return z
    if n>z.size-1: return z
    v = z.copy()
    tp = v[n].real
    ap = a[1].real or 0.1
    bp = a[2].real or 0.1
    theta = 2 * np.pi * tp
    r = ap + bp * theta
    v[n] = r * np.exp(1j * theta)
    return v

def logarithmic_spiral(z,a,state):
    n = int(a[0].real)
    if n<0: return z
    if n>z.size-1: return z
    v = z.copy()
    tp = v[n].real
    ap = a[1].real or 0.1
    bp = a[2].real or 0.1
    theta = 2 * np.pi * tp
    r = ap * np.exp(bp * theta)
    v[n] = r * np.exp(1j * theta)
    return v

def deltoid(z,a,state):
    n = int(a[0].real)
    if n<0: return z
    if n>z.size-1: return z
    v = z.copy()
    tp = v[n].real
    Rp = a[1].real or 1.0
    theta = 2 * np.pi * tp
    x = Rp * (2 * np.cos(theta) + np.cos(2 * theta)) / 3
    y = Rp * (2 * np.sin(theta) - np.sin(2 * theta)) / 3
    v[n] =  x + 1j * y
    return v

ALLOWED = {
    "xim":   xim,
    "zz":    op_zz,
    "zz1":   op_zz1,
    "zz2":   op_zz2,
    "zz3":   op_zz3,
    "pz":    op_pz,
    "bkr":   op_bkr,
    "crd":   op_crd,
    "hrt":   op_hrt,
    "spdl":  op_spdl,
    "lmc":   limacon,
    "rsc":   rose_curve,
    "lss":   lissajous,
    "ast":   astroid,
    "lsp":   logarithmic_spiral,
    "dlt":   deltoid,
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
