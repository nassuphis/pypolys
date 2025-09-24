import argparse
import numpy as np
from numba import njit, complex128
import time
import matplotlib.pyplot as plt
import pandas as pd
import datashader as ds
from datashader.transfer_functions import shade, dynspread
from colorcet import fire

#
# Plain Python Ops
#
def op_nop(z,a):      
    return z
def op_const(z,a):      
    return a
def op_runif(z,a):
    u = np.random.uniform(0.0, 1.0, size=2).astype(np.complex128)   
    return u
def op_round(z,a):
    n=int(a[0].real)
    return np.round(z,n)
def op_uc(z,a):       
    zz = np.exp(1j*2*np.pi*z)
    return zz
def op_coeff1(z,a):
    zz = (z.real).astype(np.complex128)
    return zz
def op_coeff2(z,a):
    tt1 = z[0] + z[1]
    tt2 = z[0] * z[1]
    return np.array([tt1,tt2],dtype=np.complex128)
def op_coeff3(z,a):
    tt1 = 1 / ( z[0] + 2 )
    tt2 = 1 / ( z[1] + 2 )
    return np.array([tt1,tt2],dtype=np.complex128)
def op_coeff4(z,a):
    tt1 = np.cos(z[0])
    tt2 = np.sin(z[1])
    return np.array([tt1,tt2],dtype=np.complex128)
def op_coeff5(z,a):
    tt1 = z[0] + (1.0+0.0j) / z[1]
    tt2 = z[1] + (1.0+0.0j) / z[0]
    return np.array([tt1,tt2],dtype=np.complex128)
def op_coeff6(z,a):
    num1 = z[0]*z[0]*z[0] + 1j
    den1 = z[0]*z[0]*z[0] - 1j
    val1 = num1 / den1
    num2 = z[1]*z[1]*z[1] + 1j
    den2 = z[1]*z[1]*z[1] - 1j
    val2 = num2 / den2
    return np.array([val1,val2],dtype=np.complex128)
def op_coeff7(z,a):
    tt1 = z[0] + z[1]
    tt2 = z[0] * z[1]
    return np.array([tt1,tt2],dtype=np.complex128)
def op_coeff8(z,a):
    tt1 = z[0] + z[1]
    tt2 = z[0] * z[1]
    return np.array([tt1,tt2],dtype=np.complex128)
def op_coeff9(z,a):
    tt1 = z[0] + z[1]
    tt2 = z[0] * z[1]
    return np.array([tt1,tt2],dtype=np.complex128)
def op_coeff10(z,a):
    tt1 = z[0] + z[1]
    tt2 = z[0] * z[1]
    return np.array([tt1,tt2],dtype=np.complex128)
def op_coeff11(z,a):
    tt1 = z[0] + z[1]
    tt2 = z[0] * z[1]
    return np.array([tt1,tt2],dtype=np.complex128)
def op_coeff12(z,a):
    tt1 = z[0] + z[1]
    tt2 = z[0] * z[1]
    return np.array([tt1,tt2],dtype=np.complex128)
def sort_by_abs(z,a):
    idx = np.argsort(np.abs(z))
    out = np.empty_like(z)
    for i in range(z.size):
        out[i] = z[idx[i]]
    return out
def rev(z,a):
    return np.flip(z)
def sort_moduli_keep_angles(z,a):
    angles = np.angle(z,a)
    sorted_moduli = np.sort(np.abs(z))
    return sorted_moduli * np.exp(1j * angles)
def invuc(cf,a):
   sa = np.max(np.abs(cf))
   cf0 = cf / sa
   cf1 = np.exp(1j*2*np.pi*cf0)
   return cf/cf1 
def normalize(cf,a):
   sa = np.max(np.abs(cf))
   return cf / sa
def poly_flip_horizontal(cf,a):
    out = np.empty_like(cf)
    n = cf.shape[0]
    for k in range(n):
        if k & 1:   # odd power
            out[k] = -cf[k]
        else:
            out[k] = cf[k]
    return out
def poly_flip_vertical(cf,a):
    out = np.empty_like(cf)
    n = cf.shape[0]
    for k in range(n):
        out[k] = np.conj(cf[k])
    return out
def rotate_poly(coeffs,a):
    theta = np.pi * (a[0].real)
    n = len(coeffs) - 1
    a0 = coeffs[0]
    k = np.arange(n, -1, -1)  # powers: n .. 0
    rotated = coeffs * np.exp(-1j * theta * k)
    rotated *= np.exp(1j * n * theta) / a0
    return rotated
def roots_toline(rts,a):
   num = 1+rts
   den = 1-rts
   line = 1j * num/den
   return line
def pull_unit_circle(z,a):
    alpha: float = 1.0
    sigma: float = 0.75
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
def swirler(cf,a):
    a = np.abs( cf *100 ) % 1
    b = np.abs( cf *10  ) % 1
    swirled_cf = cf * np.exp( a*a*a*a + b*b*b*b + 1j*2*np.pi*b*a )
    return swirled_cf
def numba_poly(roots,a):
    n = len(roots)
    coeffs = np.zeros(n + 1, dtype=roots.dtype)
    coeffs[0] = 1.0
    for r in roots:
        new_coeffs = np.zeros_like(coeffs)
        for i in range(n):
            new_coeffs[i]     += coeffs[i]
            new_coeffs[i + 1] -= coeffs[i] * r
        coeffs = new_coeffs
    return coeffs
def poly_giga_5(z,a):
    t1, t2 = z[0], z[1] 
    cf = np.zeros(25, dtype=np.complex128)
    cf[0]  = 1.0 + 0j 
    cf[4]  = 4.0 + 0j
    cf[12] = 4.0 + 0j 
    cf[19] = -9 + 0j
    cf[20] = -1.9 + 0j
    cf[24] = 0.2 + 0j
    cf[6] = 100j * t2**3 + 100j * t2**2 - 100j * t2 - 100j
    cf[8] = 100j * t1**3 + 100j * t1**2 + 100j * t2 - 100j
    cf[14] = 100j * t2**3 - 100j * t2**2 + 100j * t2 - 100j
    return cf
def p7f(z, a):
    t1, t2 = z[0].real, z[0].imag 
    pi2  =  2 * np.pi
    n    =  23 # was 23
    tt1  =  np.exp(1j * pi2 * t1)
    ttt1 =  np.exp(1j * pi2 * tt1)
    v  =  np.linspace(np.real(tt1), np.real(ttt1), n)
    if t2 < 0.1:
        f = 10 * t1 * np.exp(1j * np.sin(11 * pi2 * v))
    elif 0.1 <= t2 < 0.2:
        f =  100 * np.exp(1j * np.sin(17 * pi2 * v))
    elif 0.2 <= t2 < 0.3:
        f =  599 * np.exp(1j * np.cos(83 * pi2 * v))
    elif 0.3 <= t2 < 0.4:
        f =  443 * np.exp(1j * np.sin(179 * pi2 * v))
    elif 0.4 <= t2 < 0.5:
        f =  293 * np.exp(1j * np.sin(127 * pi2 * v))
    elif 0.5 <= t2 < 0.6:
        f =  541 * np.exp(1j * np.sin(103 * pi2 * v))
    elif 0.6 <= t2 < 0.7:
        f =  379 * np.exp(1j * np.sin(283 * pi2 * v))
    elif 0.7 <= t2 < 0.8:
        f =  233 * np.exp(1j * np.sin(3 * pi2 * v))
    elif 0.8 <= t2 < 0.9:
        f =  173 * np.exp(1j * np.sin(5 * pi2 * v))
    else:
        f =  257 * np.exp(1j * np.sin(23 * pi2 * v))

    f[n-1] +=  211 * np.exp(1j * pi2 * (1/7) * t2 )

    return f
def op_roots(z,a):     return np.roots(z)

ALLOWED = {
    "nop":       op_nop,
    "const":     op_const,
    "runif":     op_runif,
    "uc":        op_uc,
    "rev":       rev,
    "round":     op_round,
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
    "sort_abs":  sort_by_abs,
    "sort_moduli_keep_angles": sort_moduli_keep_angles,
    "invuc":     invuc,
    "normalize": normalize,
    "flip_horizontal": poly_flip_horizontal,
    "flip_vertical": poly_flip_vertical,
    "rotate":  rotate_poly,
    "swirler": swirler,
    "toline":  roots_toline,
    "unitpull":  pull_unit_circle,
    "poly5":     poly_giga_5,
    "p7f":       p7f,
    "poly":      numba_poly,
    "roots":     op_roots,
}

# Signature: complex vector -> complex vector
SIG = complex128[:](complex128[:],complex128[:])

# Pre-JIT the callees with an explicit signature
JITTED = {
    name: njit(SIG, cache=True, fastmath=True)(fn) for name, fn in ALLOWED.items()
}

ORDERED_NAMES = tuple(sorted(ALLOWED))

NAME2OP = {
    n: np.int8(i) for i, n in enumerate(ORDERED_NAMES)
}

def build_dispatcher_codegen(ordered_names):
    g = {"np": np}  # no need to put njit here
    # expose compiled callees + opcode constants to the generated function
    for i, n in enumerate(ordered_names):
        g[n] = JITTED[n]          # <-- compiled CPUDispatcher
        g[n.upper()] = np.int8(i) # e.g., ADD1 = 1

    lines = ["def _apply_opcode_impl(op, z,a):"]
    for i, n in enumerate(ordered_names):
        kw = "if" if i == 0 else "elif"
        lines += [f"    {kw} op == {n.upper()}:", f"        return {n}(z,a)"]
    lines.append("    return z")
    src = "\n".join(lines)

    # put function into g
    exec(src, g, g)
    _py = g["_apply_opcode_impl"]
    # Important: cache=False because function is from exec("<string>")
    return njit(cache=False, fastmath=True)(_py)

APPLY_OPCODE = build_dispatcher_codegen(ORDERED_NAMES)

@njit(cache=True, fastmath=True)
def apply_program(z,a, opcodes):
    for k in range(opcodes.shape[0]):
        z = APPLY_OPCODE(opcodes[k], z,a[k])
    return z

# ---------- chain parser (names + per-op args) ----------
MAXA = 4  # max number of args per op; adjust as needed

CONST_MAP = {
    'pi':  np.pi,
    'tau': 2*np.pi,
    'e':   np.e,
}

def _parse_scalar(tok: str) -> complex:
    """Parse tok into a complex number without eval(). Supports 1+2j, -0.5j, pi, etc."""
    t = tok.strip().lower()
    if t in CONST_MAP:
        return complex(CONST_MAP[t], 0.0)
    # allow 'i' as imaginary unit too
    t = t.replace('i', 'j')
    # If it looks like a bare real, complex() still works
    try:
        return complex(t)
    except ValueError:
        raise ValueError(f"Bad numeric token: '{tok}'")

def parse_chain_with_args(chain_str: str):
    """
    chain_str like: 'uc:0.12:0.34,coeff5:10,nop,coeff9:1+2j'
    Returns:
      opcodes : int8[ n_ops ]
      args    : complex128[ n_ops, MAXA ]  (unused slots are 0+0j)
    """
    if not chain_str.strip():
        return np.empty(0, dtype=np.int8), np.empty((0, MAXA), dtype=np.complex128)

    items = [s.strip() for s in chain_str.split(',') if s.strip()]
    n_ops = len(items)

    opcodes = np.empty(n_ops, dtype=np.int8)
    args    = np.zeros((n_ops, MAXA), dtype=np.complex128)

    for k, item in enumerate(items):
        parts = item.split(':')
        name = parts[0].lower()
        if name not in NAME2OP:
            raise ValueError(f"Unknown op '{name}'. Allowed: {list(NAME2OP)}")
        opcodes[k] = NAME2OP[name]

        # parse up to MAXA args
        for j, tok in enumerate(parts[1:MAXA+1], start=0):
            args[k, j] = _parse_scalar(tok)

    return opcodes, args

def datashade_complex(M, width=1200, height=1200, how='eq_hist'):
    # Flatten to x,y
    x = M.real.ravel()
    y = M.imag.ravel()

    # Make square bounds (so aspect stays 1:1)
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    cx, cy = (xmin+xmax)/2.0, (ymin+ymax)/2.0
    half = max(xmax-xmin, ymax-ymin) / 2.0 or 1.0
    xr = (cx - half, cx + half)
    yr = (cy - half, cy + half)

    # Datashader pipeline
    df = pd.DataFrame({'x': x, 'y': y})
    cvs = ds.Canvas(plot_width=width, plot_height=height, x_range=xr, y_range=yr)
    agg = cvs.points(df, 'x', 'y',agg=ds.count())             # count per pixel
    img = shade(agg, cmap=fire, how=how)       # 'eq_hist' or 'log' or 'linear'
    img.to_pil().save("out.png")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", type=str, default="nop")
    ap.add_argument("--how", type=str, default="eq_hist")
    ap.add_argument("--runs", type=int, default=100_000)
    ap.add_argument("--px", type=int, default=5_000)
    args = ap.parse_args()

    chain_names_str = args.chain
    opcodes, a = parse_chain_with_args(chain_names_str)
    z=np.array([0],dtype=np.complex128)
    first = apply_program(z.copy(), a, opcodes)
    n = first.size
    M = np.empty(( args.runs, n), dtype=np.complex128)
    M[0] = first
    t0 = time.perf_counter()
    for t in range(1, args.runs):
        z=np.array([t],dtype=np.complex128)
        M[t] = apply_program(z, a, opcodes)
    print(f"sample time: {time.perf_counter() - t0:.3f}s")

    t0 = time.perf_counter()
    datashade_complex(M,width=args.px, height=args.px,how=args.how)
    print(f"shader time: {time.perf_counter() - t0:.3f}s")

    

if __name__ == "__main__":
    main()

