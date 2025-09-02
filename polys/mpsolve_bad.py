#!/usr/bin/env python

import numpy as np
import mpsolve


cf = np.poly(mpsolve.chessboard_roots(8)).astype(np.complex128)
print(f"Computing cf, no warm")
z = mpsolve.mpsolve(cf, precision=1024)
print(f"Computing cf1, warm")
cf1 = mpsolve.wilkinson.perturb_8(cf, 0.01, 0.0)
z1 =mpsolve.mpsolve_warm(cf1,z, precision=1024)
cf2 = mpsolve.wilkinson.perturb_8(cf, 0.5, 0.1)
np.savez(
    "repro_cf2_z1.npz",
    cf2=np.asarray(cf2, dtype=np.complex128),
    z1 =np.asarray(z1,  dtype=np.complex128)
)
print(f"Computing cf2, warm")
z2 = mpsolve.mpsolve_warm(cf2,z1, precision=1024)
print(f"Done!")