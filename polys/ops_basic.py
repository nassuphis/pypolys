# ops_basic.py
import numpy as np
import math
from numba.typed import Dict
from numba import  njit, types
import argparse
import ast

def op_nop(z,a,state):      
    return z

def op_const(z,a,state):      
    return a


def invuc1(z,a,state):
   sa = np.max(np.abs(z))
   z0 = z / sa
   z1 = np.exp(1j*2*np.pi*z0)
   return z/z1 

def invuc(z, a, state):
    n = z.size
    out = np.empty_like(z)

    # max |z| without creating a float array
    sa = 0.0
    for i in range(n):
        v = z[i]
        mag = (v.real * v.real + v.imag * v.imag) ** 0.5
        if mag > sa:
            sa = mag

    # guard: if all zeros or non-finite, just return z
    if not np.isfinite(sa) or sa <= 0.0:
        for i in range(n):
            out[i] = z[i]
        return out

    two_pi = 2.0 * np.pi

    for i in range(n):
        v = z[i] / sa
        x = v.real
        y = v.imag

        # exp(i*2π*(x + i y)) = exp(-2π y) * (cos(2π x) + i sin(2π x))
        ex = np.exp(-two_pi * y)
        c = np.cos(two_pi * x)
        s = np.sin(two_pi * x)
        z1r = ex * c
        z1i = ex * s

        # divide z by z1 (guard against zero/denormals)
        den = z1r * z1r + z1i * z1i
        if den == 0.0 or not np.isfinite(den):
            out[i] = 0.0 + 0.0j
        else:
            zr = z[i].real
            zi = z[i].imag
            # (zr + i zi) / (z1r + i z1i)
            out[i] = ((zr * z1r + zi * z1i) / den) + 1j * ((zi * z1r - zr * z1i) / den)

        # final safety
        if not np.isfinite(out[i].real) or not np.isfinite(out[i].imag):
            out[i] = 0.0 + 0.0j

    return out

def normalize(cf,a,state):
   sa = np.max(np.abs(cf))
   return cf / sa

def poly_flip_horizontal(z,a,state):
    out = np.empty_like(z)
    n = z.shape[0]
    for k in range(n):
        if k & 1:   # odd power
            out[k] = -z[k]
        else:
            out[k] = z[k]
    return out

def poly_flip_vertical(z,a,state):
    out = np.empty_like(z)
    n = z.shape[0]
    for k in range(n):
        out[k] = np.conj(z[k])
    return out

def rotate_poly(z,a,state):
    theta = np.pi * (a[0].real)
    n = len(z) - 1
    a0 = z[0]
    k = np.arange(n, -1, -1)  # powers: n .. 0
    rotated = z * np.exp(-1j * theta * k)
    rotated *= np.exp(1j * n * theta) / a0
    return rotated

def swirler(z,a,state):
    a = np.abs( z *100 ) % 1
    b = np.abs( z *10  ) % 1
    swirled_cf = z * np.exp( a*a*a*a + b*b*b*b + 1j*2*np.pi*b*a )
    return swirled_cf

def numba_poly(z,a,state):
    n = len(z)
    coeffs = np.zeros(n + 1, dtype=np.complex128)
    coeffs[0] = 1.0
    for r in z:
        new_coeffs = np.zeros_like(coeffs)
        for i in range(n):
            new_coeffs[i]     += coeffs[i]
            new_coeffs[i + 1] -= coeffs[i] * r
        coeffs = new_coeffs
    return coeffs

def op_roots(z,a,state):     
    return np.roots(z)

@njit(cache=True, fastmath=True)
def _horner_and_deriv(cf: np.ndarray, z: complex):
    n = cf.size - 1
    p = cf[0]
    dp = 0.0 + 0.0j
    for k in range(1, n + 1):
        dp = dp * z + p
        p = p * z + cf[k]
    return p, dp

@njit(cache=True, fastmath=True)
def _aberth_once(cf: np.ndarray, guess: np.ndarray, newton_fallback: bool):
    n = guess.size
    p = np.empty(n, dtype=np.complex128)
    dp = np.empty(n, dtype=np.complex128)
    for i in range(n):
        pi, dpi = _horner_and_deriv(cf, guess[i])
        p[i]  = pi
        dp[i] = dpi
    guess_new = np.empty_like(guess)
    max_step = 0.0
    tiny = 1e-300
    for i in range(n):
        gi = guess[i]
        dpi = dp[i]
        if abs(dpi) < tiny:
            wi = p[i] / (dpi + 1e-16)
            step = wi
            if newton_fallback:
                mag = abs(step)
                if mag > 1.0:
                    step = step / mag
        else:
            wi = p[i] / dpi
            s = 0.0 + 0.0j
            for j in range(n):
                if j != i:
                    dz = gi - guess[j]
                    if dz == 0:
                        dz = dz + (1e-16 + 1e-16j)
                    s += 1.0 / dz
            denom = 1.0 - wi * s
            step = wi if denom == 0 else wi / denom
        gi_new = gi - step
        guess_new[i] = gi_new
        ds = abs(step)
        if ds > max_step:
            max_step = ds
    return guess_new, max_step

STATE_ABERTH_GUESS = np.int8(1)
def aberth( z, a, state):
    tol, max_iters, per_root_tol, newton_fallback = 1e-12, 100, False, False
    cf = z.copy()
    if STATE_ABERTH_GUESS in state:
        guess = state[STATE_ABERTH_GUESS]
    else:
        guess = np.roots(cf)
        state[STATE_ABERTH_GUESS] = guess
        return guess
    n = z.size
    for _ in range(max_iters):
        guess_new, max_step = _aberth_once(cf, guess, newton_fallback)
        if per_root_tol:
            all_ok = True
            for i in range(n):
                rel = abs(guess_new[i] - z[i]) / (1.0 + abs(guess_new[i]))
                if rel > tol:
                    all_ok = False
                    break
            guess = guess_new
            if all_ok: break
        else:
            guess = guess_new
            if max_step <= tol: break
    
    state[STATE_ABERTH_GUESS] = guess
    return guess

ALLOWED = {
    "nop":       op_nop,
    "const":     op_const,
    "invuc":     invuc,
    "normalize": normalize,
    "hflip":     poly_flip_horizontal,
    "vflip":     poly_flip_vertical,
    "rotate":    rotate_poly,
    "rot":       rotate_poly,
    "swirler":   swirler,
    "swrl":      swirler,
    "poly":      numba_poly,
    "roots":     op_roots,
    "aberth":    aberth,
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

