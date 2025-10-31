# ops_rfrm.py
#
# roots2 = f(roots1)
# transforma that work on roots 
#
import numpy as np
from numba.typed import Dict
from numba import types
import argparse
import ast



def rotate_roots(z,a,state):
    theta = np.pi * (a[0].real)
    rotated = z * np.exp(1j * theta )
    return rotated

def pull_unit_circle(z,a,state):
    sigma = a[0].real or 0.75
    alpha = a[1].real or 1.0
    n = z.shape[0]
    out = np.empty_like(z)
    for i in range(n):
        x = z[i].real
        y = z[i].imag
        r = np.hypot(x, y)
        theta = np.arctan2(y, x)
        d = r - 1.0
        rprime = r - alpha * d * np.exp(- (d / sigma) ** 2)
        out[i] = rprime * (np.cos(theta) + 1j * np.sin(theta))
    return out

def push_unit_circle(z, a, state):
    sigma = a[0].real or 0.75
    alpha = a[1].real or 1.0
    n = z.shape[0]
    out = np.empty_like(z)
    for i in range(n):
        x = z[i].real
        y = z[i].imag
        r = np.hypot(x, y)
        d = r - 1.0
        # push: add the Gaussian bump instead of subtracting it
        rprime = r + alpha * d * np.exp(- (d / sigma) ** 2)

        if r > 0.0:
            s = rprime / r
            out[i] = (x * s) + 1j * (y * s)  # preserve angle, avoid trig
        else:
            out[i] = 0.0 + 0.0j
    return out

def pull_towards_center(z, a, state):
    alpha = 1.0
    sigma = 0.75
    n = z.shape[0]
    out = np.empty_like(z)
    for i in range(n):
        x = z[i].real
        y = z[i].imag
        r = np.hypot(x, y)
        # Gaussian falloff centered at r = 0
        shrink = alpha * np.exp(- (r / sigma) ** 2)
        rprime = r * (1.0 - shrink)  # reduce radius smoothly toward 0
        if r > 0.0:
            s = rprime / r
            out[i] = (x * s) + 1j * (y * s)
        else:
            out[i] = 0.0 + 0.0j
    return out

def roots_toline(z,a,state):
   num = 1+z
   den = 1-z
   line = 1j * num/den
   return line

def roots_add(z,a,state):
    return z+a[0]

def roots_mult(z,a,state):
    return z*a[0]

ALLOWED = {
    "rrot":      rotate_roots,
    "radd":      roots_add,
    "rmul":      roots_mult,
    "unitpull":  pull_unit_circle,
    "puc":       pull_unit_circle,
    "centerpull": pull_towards_center,
    "pushuc":    push_unit_circle,
    "toline":    roots_toline,
    "line":      roots_toline,
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
