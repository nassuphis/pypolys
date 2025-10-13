# ops_param.py
#
# parameter generation
# 
import math
import numpy as np
from numba.typed import Dict
from numba import types
import argparse
import ast


# runif
def op_runif(z,a,state):
    t1 = complex(np.random.random(),0)
    t2 = complex(np.random.random(),0)
    u = np.array([t1,t2],dtype=np.complex128)
    return u

# serpentine scan
# generate an equi-distant
# input is step, output is 
# length 2 vector of parameters
# s,t with real part coordinates
# zero imaginary part
# example:
#   serp:runs:0+0j:1+1j
def serp(z,a,state):
    i = z[0]
    i = np.real(i)
    i = int(i)
    n = a[0]
    n = n.real
    n = int(n) 
    ll = a[1]
    ur = a[2]
    i = i % n
    if n <= 0 or i < 0 or i >= n:
        return z
    cols = int(math.ceil(math.sqrt(float(n))))
    rows = int(math.ceil(n / cols))
    r = i // cols              # row
    c = i - r * cols           # col within row (pre serpentine flip)
    if (r & 1) == 1:
        c = cols - 1 - c
    fx = (c + 0.5) / cols
    fy = (r + 0.5) / rows
    llx, lly = ll.real, ll.imag
    urx, ury = ur.real, ur.imag
    x = llx + fx * (urx - llx)
    y = lly + fy * (ury - lly)
    t1 = complex(x,0)
    t2 = complex(y,0)
    u = np.array([t1,t2],dtype=np.complex128)
    return u

# add complex dither to inputs
# dither:runs:0.5
def dither(z,a,state):

    serp_len = a[0].real
    dither_width = a[1].real

    dither_fact = dither_width/math.sqrt(serp_len)

    t1 = dither_fact * (np.random.random()-0.5)
    t1 = complex( t1,0 ) 
    t1 = t1 + z[0]

    t2 = dither_fact * (np.random.random()-0.5)
    t2 = complex( t2,0 ) 
    t2 = t2 + z[1]

    u = np.array([t1,t2],dtype=np.complex128)

    return u

ALLOWED = {
    "runif":     op_runif,
    "dither":    dither,
    "serp":      serp,
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
