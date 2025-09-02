#!/usr/bin/env python

import numpy as np
import mpsolve


data = np.load("repro_cf2_z1.npz")
cf2 = data["cf2"]
z1  = data["z1"]

# sanity: warm-start requires len(guesses) == deg
deg = len(cf2) - 1
assert z1.ndim == 1 and z1.dtype == np.complex128 and len(z1) == deg, \
    f"guess length {len(z1)} != degree {deg}"

for i in range(10):
    print(f"i:{i}")
    roots = mpsolve.mpsolve_warm(cf2, z1, precision=1024)
print("Done!")

