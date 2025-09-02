#!/usr/bin/env python

import numpy as np
import mpsolve


#cf = np.poly(np.arange(1,20)).astype(np.complex128)
cf = np.poly(mpsolve.chessboard_roots(10)).astype(np.complex128)

print(f"Computing cf, cold")
for i in range(10):
    print(f"i:{i}")
    roots = mpsolve.mpsolve(cf, precision=1024)
print("Done!")