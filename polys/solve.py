################################################
# Poly solvers
################################################

from . import polystate as ps
import numpy as np
from numba import njit 

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
        