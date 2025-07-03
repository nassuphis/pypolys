################################################
# Poly solvers
################################################

from . import polystate as ps
import numpy as np

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
        