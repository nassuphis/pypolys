import math
import numpy as np
from numba import njit, prange, types, complex128, int32, float64
from numba.typed import Dict
import specparser
import matplotlib.pyplot as plt


@njit("int32(complex128, complex128, int32, float64)", fastmath=True, cache=True)
def _julia_escape_single(
    z0: np.complex128,
    c: np.complex128,
    max_iter: int,
    bailout2: float = 4.0
) -> np.int32:
    z = z0
    for k in range(max_iter):
        z = z * z + c
        if (z.real * z.real + z.imag * z.imag) > bailout2:
            return k
    return max_iter


# vectorized, parallel caller
@njit("int32[:](complex128[:], complex128, int32, float64)",
      parallel=True, fastmath=True, cache=True)
def julia_escape_vec(z0, c, max_iter, bailout2):
    n = z0.size
    out = np.empty(n, np.int32)
    for i in prange(n):
        out[i] = _julia_escape_single(z0[i], c, max_iter, bailout2)
    return out

@njit("complex128[:](int64,float64)",fastmath=True, cache=True)
def _points(N,w:float=1):
    re = -w + 2 * w * np.random.rand(N)
    im = -w + 2 * w * np.random.rand(N)
    return re + 1j*im

# ---- parameters kept as globals so the public API is (N, w, thresh) ----
C_CONST       = np.complex128(-0.8 + 0.156j)
MAX_ITER_CONST = np.int32(400)
BAILOUT2_CONST = np.float64(4.0)

# 
@njit("complex128[:](int64, float64, int32)", fastmath=True, cache=True)
def julia_sample(N, w, thresh):
    z0 = _points(N, w)
    iters = julia_escape_vec(z0, C_CONST, MAX_ITER_CONST, BAILOUT2_CONST)
    keep = 0
    for i in range(N):
        if iters[i] > thresh:
            keep += 1
    out = np.empty(keep, np.complex128)
    j = 0
    for i in range(N):
        if iters[i] > thresh:
            out[j] = z0[i]
            j += 1
    return out

def julia(N: int) -> np.ndarray:
    """
    Return exactly N Julia points with iter > 300 in window [-1.5, 1.5]^2.
    Strategy: probe once to estimate p, then draw ~need/p to fill.
    """
    N = int(N)
    out  = np.empty(N, np.complex128)
    kept = 0

    # 1) probe sample
    s = julia_sample(np.int64(N), np.float64(1.5), np.int32(300))
    take = min(s.size, N)
    if take:
        out[:take] = s[:take]
        kept += take

    # 2) p = max(0.1, len(sample)/N)
    p = max(0.1, (s.size / N) if N > 0 else 0.1)

    # 3) keep drawing until filled
    #    draw size ~ need/p, ceiling; keep updating p from each draw
    rounds = 0
    while kept < N:
        rounds += 1
        need = (N - kept)
        draw = 2*int(math.ceil(N / p))
        if draw < 1:
            draw = 1

        print(f"kept:{kept} draw:{draw}")

        s = julia_sample(np.int64(draw), np.float64(1.5), np.int32(300))
        take = min(s.size, need)
        if take:
            out[kept:kept + take] = s[:take]
            kept += take

        # refresh p from this batch (avoid collapse below 0.1 per your rule)
        p = max(0.1, (s.size / draw) if draw > 0 else 0.1)

        # ultra-defensive break to avoid pathological loops
        if rounds > 128 and take == 0:
            break

    return out  # exactly N filled unless the defensive break triggered


def plot(x):
    plt.figure(figsize=(10, 10))
    plt.scatter(x.real, x.imag, s=0.01, lw=0)
    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

x=julia(10**6)


