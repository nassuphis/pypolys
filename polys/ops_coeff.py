# ops_basic.py
import numpy as np
from numba.typed import Dict
from numba import types
import argparse
import ast

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
def op_coeff4(z,a,state):
    tt1 = np.cos(z[0])
    tt2 = np.sin(z[1])
    return np.array([tt1,tt2],dtype=np.complex128)
def op_coeff5(z,a,state):
    tt1 = z[0] + (1.0+0.0j) / z[1]
    tt2 = z[1] + (1.0+0.0j) / z[0]
    return np.array([tt1,tt2],dtype=np.complex128)
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
    tt1 = z[0] + z[1]
    tt2 = z[0] * z[1]
    return np.array([tt1,tt2],dtype=np.complex128)
def op_coeff9(z,a,state):
    tt1 = z[0] + z[1]
    tt2 = z[0] * z[1]
    return np.array([tt1,tt2],dtype=np.complex128)
def op_coeff10(z,a,state):
    tt1 = z[0] + z[1]
    tt2 = z[0] * z[1]
    return np.array([tt1,tt2],dtype=np.complex128)
def op_coeff11(z,a,state):
    tt1 = z[0] + z[1]
    tt2 = z[0] * z[1]
    return np.array([tt1,tt2],dtype=np.complex128)
def op_coeff12(z,a,state):
    tt1 = z[0] + z[1]
    tt2 = z[0] * z[1]
    return np.array([tt1,tt2],dtype=np.complex128)


ALLOWED = {
    "coeff1":    op_coeff1,
    "coeff2":    op_coeff2,
    "coeff3":    op_coeff3,
    "coeff4":    op_coeff4,
    "coeff5":    op_coeff5,
    "coeff6":    op_coeff6,
    "coeff7":    op_coeff7,
    "coeff8":    op_coeff8,
    "coeff9":    op_coeff9,
    "coeff10":   op_coeff10,
    "coeff11":   op_coeff11,
    "coeff12":   op_coeff12,
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
