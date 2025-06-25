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

