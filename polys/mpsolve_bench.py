import numpy as np
import mpsolve
import timeit


cf = np.poly(mpsolve.chessboard_roots(7)).astype(np.complex128)
z = mpsolve.mpsolve(cf,precision=4096)
print(f"Cold: algo=MPS_ALGORITHM_STANDARD_MPSOLVE")
for p in [64,128,256,512,1024,2048,4096]:
    t=timeit.timeit(f"mpsolve.mpsolve(cf, precision={p},algo=mpsolve.MPS_ALGORITHM_STANDARD_MPSOLVE)", globals=globals(), number=10)
    print(f"{p}:{1000*t/10:.1f} msec")
print(f"Cold: algo=MPS_ALGORITHM_SECULAR_GA")
for p in [64,128,256,512,1024,2048,4096]:
    t=timeit.timeit(f"mpsolve.mpsolve(cf, precision={p},algo=mpsolve.MPS_ALGORITHM_SECULAR_GA)", globals=globals(), number=10)
    print(f"{p}:{1000*t/10:.1f} msec")
print(f"Warm: algo=MPS_ALGORITHM_STANDARD_MPSOLVE")
for p in [64,128,256,512,1024,2048,4096]:
    t=timeit.timeit(f"mpsolve.mpsolve_warm(cf,z, precision={p},algo=mpsolve.MPS_ALGORITHM_SECULAR_GA)", globals=globals(), number=10)
    print(f"{p}:{1000*t/10:.1f} msec")
print(f"Warm: algo=MPS_ALGORITHM_SECULAR_GA")
for p in [64,128,256,512,1024,2048,4096]:
    t=timeit.timeit(f"mpsolve.mpsolve_warm(cf,z, precision={p},algo=mpsolve.MPS_ALGORITHM_SECULAR_GA)", globals=globals(), number=10)
    print(f"{p}:{1000*t/10:.1f} msec")