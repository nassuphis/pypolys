#!/usr/bin/env python
import argparse
from functools import partial, reduce
import timeit
import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt
import polys.registry as rgs



poly_fun_init = None
def poly_fun(z,a):
    global poly_fun_init
    if poly_fun_init is None:
        fname = a[0]
        poly_fun_init = rgs.pfun[fname]
    return poly_fun(z)
    

poly_val_init = None
def poly_val(z, a):
    global poly_val_init
    if poly_val_init is None:
        poly_val_init=np.array([complex(x) for x in a])
    return np.polyval(poly_val_init,z)

def poly_giga_6(z,a):
    n = 10
    try:
        t1, t2 = z.real, z.imag
        cf = np.zeros(n, dtype=complex)
        cf[0] = 150 * t2**3 - 150j * t1**2
        cf[1] = 0
        cf[n//2-1] = 100*(t1-t2)**1
        cf[n-2] = 0
        cf[n-1] = 10j
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(n, dtype=np.complex128)
    
sample_init = None
def sample(z,a):
    if len(a)>0: 
        sample_init=np.array([complex(a[0])])
    else:
        sample_init=np.random.random(1)+1j*np.random.random(1)
    return sample_init

multi_sample_init = None
def multi_sample(z,a):
    global multi_sample_init
    if len(a)>0: 
        multi_sample_init=int(a[0])
    n = multi_sample_init
    rv = np.random.random(n)+1j*np.random.random(n)
    return rv

expr_init = None
def expr(z,a):
    global expr_init
    if expr_init is None:
        expr_init=ne.NumExpr(a[0],signature=[('z', complex)])
    return expr_init(z)
    
cf_init = None
def cf(z,a):
    global cf_init
    if cf_init is None:
        cf_init=[ne.NumExpr(expr+"+z*0", signature=[('z', complex)]) for expr in a]
    return np.array([f(z) for f in cf_init]).transpose()

def matrix_roots(z,a):
    coeff_matrix=z
    N, deg_plus_1 = coeff_matrix.shape
    max_deg = deg_plus_1 - 1

    roots = np.full((N, max_deg), np.nan, dtype=np.complex128)

    for i in range(N):
        coeffs = coeff_matrix[i]
        coeffs = np.trim_zeros(coeffs, trim='f')
        if len(coeffs) <= 1:
            continue
        r = np.roots(coeffs)
        roots[i, :len(r)] = r

    return roots

# Dictionary of available operations
ops_dict = {
    'pval'   : poly_val,
    'expr'   : expr,
    'cf'     : cf,
    'roots'  : lambda z,a:np.roots(z),
    'mroots'  : matrix_roots,
    'poly'   : lambda z,a:np.poly(z),
    't'      : sample,
    'ts'     : multi_sample,
    'giga6'  : lambda z,a:poly_giga_6(z,a), 
    'coeff2' : lambda z,a: z.real + z.imag + 1j * (z.real * z.imag),
    'coeff3' : lambda z,a: 1/(z.real + 2)+1j/(z.imag + 2),
    'coeff3a': lambda z,a: 1/(z.real + 1)+1j/(z.imag + 1),
    'coeff4' : lambda z,a:  np.cos(z.real)+1j*np.sin(z.imag),
    'coeff5' : lambda z,a:  (z.real + 1/z.imag)+1j*(z.imag + 1/z.real),
    'rev'    : lambda z,a:  np.flip(z),
    'rep2'   : lambda z,a: np.concatenate([z,z]),
}



def compose_nest(*funcs):
    if not funcs:
        return lambda x: x  # Identity for empty pipeline
    return reduce(lambda f, g: lambda x: f(g(x)), reversed(funcs))

def make_pipeline(txt,dict):
    pipeline_steps = []
    for step in txt.split(','):
        parts = step.split('_')
        op_name = parts[0]
        op_args = parts[1:]
        if op_name not in dict:
            raise ValueError(f"Unknown operation: {op_name}")
        op_func = dict[op_name]
        pipeline_steps.append((op_func, op_args))

    bound_funcs = [partial(op_func, a=op_args) for op_func, op_args in pipeline_steps]
    pipeline = compose_nest(*bound_funcs)
    return pipeline


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--pipeline', required=True, help='Pipeline like op1_arg1_arg2,op2_arg1_arg2,op3_arg1_arg2')
    parser.add_argument('-d','--show', action="store_true", help="show results")
    args = parser.parse_args()

    pipeline = make_pipeline(args.pipeline,ops_dict)
    z = pipeline("start")

    if args.show:
        print(f"{z}")
        exit(0)

    z=z.flatten()
    zd = np.abs(z-np.mean(z))
    qlo,qhi = np.quantile(zd,[0.01,0.99])
    z=z[(zd<qhi) & (zd>qlo)]
    plt.figure(figsize=(6, 6))
    plt.scatter(z.real, z.imag, color='blue', marker='.',s=1)
    # Labels and aspect
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.title("Complex Numbers in the Complex Plane")
    plt.axis('equal')  # Square plot
    plt.show()
 
