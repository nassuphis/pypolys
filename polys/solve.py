################################################
# Poly solvers
################################################

from . import polystate as ps
import numpy as np
from numba import njit 
from . import mpsolve 

# conservative bounds for the double path you use in mps_monomial_poly_set_coefficient_d
_DBL_MAX_SAFE = 1e300        # < 1.797e308 to leave headroom
_DBL_MIN_SAFE = 1e-300       # treat tinies as zero


def _poly_ok(a: np.ndarray) -> bool:
    a = np.asarray(a, dtype=np.complex128)

    # 1) finite
    if not np.isfinite(a).all():
        return False

    # 2) non-empty and leading coeff not (absolutely) tiny/zero
    if a.size == 0 or np.abs(a[0]) <= _DBL_MIN_SAFE:
        return False

    # 3) magnitude within safe double range
    maxabs = float(np.max(np.abs(a)))
    if maxabs > _DBL_MAX_SAFE:
        return False

    return True

def circle_guess(n, radius=1.0):
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    return radius * np.exp(1j * theta)

@njit
def horner(p, x):
    s = 0.0 + 0j
    for c in p:
        s = s * x + c
    return s

def circle_guess(n, radius=1.0):
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    return radius * np.exp(1j * theta)

def wssn(cf):
    x0 = circle_guess(cf.size-1)
    return wssn_jit(cf,x0)
    

@njit
def wssn_jit(p_coeffs,x0):
    max_iter=100 
    tol=1e-12
    p_coeffs = p_coeffs / p_coeffs[0]
    x = x0.copy()
    n = x.size
    for k in range(max_iter):
        delta = 0.0
        for i in range(n):
            denom = 1.0 + 0j
            xi = x[i]
            for j in range(n):
                if j != i:
                    denom *= xi - x[j]
            if denom != 0:
                numer = horner(p_coeffs, xi)
                new_xi = xi -  numer / denom
                delta = max(delta, abs(new_xi - xi))
                x[i] = new_xi
        if delta < tol:
            break
    return x

def solve(cf):
  try:
    rts=np.roots(cf)
    return rts
  except:     
    return np.zeros(len(cf)-1, dtype=complex)

def mps4(cf):
    if  _poly_ok(cf):
        return mpsolve.mpsolve(cf,precision=4)
    return np.zeros(len(cf)-1, dtype=complex)

def mps64(cf):
    if  _poly_ok(cf):
        return mpsolve.mpsolve(cf,precision=64)
    return np.zeros(len(cf)-1, dtype=complex)

def mps128(cf):
    if  _poly_ok(cf):
        return mpsolve.mpsolve(cf,precision=128)
    return np.zeros(len(cf)-1, dtype=complex)

def mps256(cf):
    if  _poly_ok(cf):
        return mpsolve.mpsolve(cf,precision=256)
    return np.zeros(len(cf)-1, dtype=complex)

def mps512(cf):
    if  _poly_ok(cf):
        return mpsolve.mpsolve(cf,precision=512)
    return np.zeros(len(cf)-1, dtype=complex)

def mps1024(cf):
    if  _poly_ok(cf):
        return mpsolve.mpsolve(cf,precision=1024)
    return np.zeros(len(cf)-1, dtype=complex)
  
def none(cf):
  return cf

def safe(cf):
  
        degree = len(cf)

        if degree<2:
           return np.zeros(cf, dtype=complex)
    
        if np.any(np.isnan(cf)) or np.any(np.isinf(cf)):
            return np.zeros(len(cf), dtype=complex)

        sabs = np.sum(np.abs(cf))

        if sabs < 1e-10 or sabs > 1e10:
            return np.zeros(len(cf), dtype=complex)
    
        try:
         
          rts = np.roots(cf)

          if np.any(np.isnan(rts)) or np.any(np.isinf(rts)):
            return np.zeros(len(cf), dtype=complex)
          return rts
    
        except:
            return np.zeros(len(cf), dtype=complex)
        