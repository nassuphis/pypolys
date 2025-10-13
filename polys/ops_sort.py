# ops_sort.py
#
# cf2 = f(cf1)
# various sorting schemes

import numpy as np
from numba.typed import Dict
from numba import types
import argparse
import ast

def sort_by_abs(z,a,state):
    idx = np.argsort(np.abs(z))
    out = np.empty_like(z)
    for i in range(z.size):
        out[i] = z[idx[i]]
    return out

def sort_moduli_keep_angles(z,a,state):
    angles = np.angle(z)
    sorted_moduli = np.sort(np.abs(z))
    return sorted_moduli * np.exp(1j * angles)

ALLOWED = {
    "sabs":      sort_by_abs,
    "smka":      sort_moduli_keep_angles,
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
