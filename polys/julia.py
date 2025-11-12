import sys
import math
import numpy as np
from numba import njit, prange, types, complex128, int32, float64
import argparse
import specparser
import expandspec
import galaxy_raster
import json
import cv2

@njit
def f(seed):
    np.random.seed(seed)       # sets Numba's RNG
    out = np.empty(5)
    for i in range(5):
        out[i] = np.random.rand()
    return out

@njit("complex128[:](int64,float64,complex128)",fastmath=True, cache=True)
def _points(N,w:float=1,center:complex=0+0j):
    re = -w + 2 * w * np.random.rand(N)
    im = -w + 2 * w * np.random.rand(N)
    return re + 1j*im + center

@njit("complex128(complex128, complex128, int32)", fastmath=True, cache=True)
def julia_equation(z: np.complex128, c:np.complex128, eqn:np.int32):
    if eqn==0:
        return z*z*z*z*z*z - z*z*z*z + c
    elif eqn==1:
        return np.exp(1j*2*np.pi*np.abs(z)) + c
    elif eqn==2:
        return z*z+c
    return z*z + c
    

@njit("int32(complex128, complex128, int32, int32, float64)", fastmath=True, cache=True)
def _julia_escape_single(
    z0: np.complex128,
    c: np.complex128,
    eqn: int = 0,
    max_iter: int = 400,
    bailout2: float = 4.0,
) -> np.int32:
    z = z0
    for k in range(max_iter):
        z = julia_equation(z,c,eqn)
        if (z.real*z.real + z.imag*z.imag) > bailout2: return k
    return max_iter

# vectorized, parallel caller
@njit("int32[:](complex128[:], complex128, int32, int32, float64)",
      parallel=True, fastmath=True, cache=True)
def julia_escape_vec(z0, c, eqn, max_iter, bailout2):
    n = z0.size
    out = np.empty(n, np.int32)
    for i in prange(n):
        out[i] = _julia_escape_single(z0[i], c, eqn, max_iter, bailout2)
    return out



#
MAX_ITER_CONST = np.int32(1000)
BAILOUT2_CONST = np.float64(4.0)
@njit("complex128[:](int64, complex128, float64, complex128, int32, int32)", fastmath=True, cache=True)
def julia_sample(N, c, w, center, thresh, eqn):
    z0 = _points(N, w,center)
    iters = julia_escape_vec(z0, c, eqn, MAX_ITER_CONST, BAILOUT2_CONST)
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

#=========================================
# C sampler
#=========================================

@njit("complex128[:](int64, float64, float64, float64, int32, int32)", fastmath=True, cache=True)
def c_sampler(N:int, lo:float, hi:float, w:float, eqn:int, max_iter:int = 400)-> np.ndarray:
    cs = _points(N,w,0)
    zs = _points(1000,w,0)
    pct_max = np.full(cs.size,0,dtype=np.float64)
    for i, c in enumerate(cs):
        escape_iter = julia_escape_vec(zs,c,eqn,max_iter,4.0)
        pct_max[i] = np.sum(escape_iter==max_iter)/zs.size
    passed = (pct_max>lo) & (pct_max<hi)
    return cs[passed]

# -----------------------------
# mask
# -----------------------------
@njit("float64[:](float64[:], boolean[:])", fastmath=True, cache=True)
def mask_f64(arr, cond):
    """Return arr[cond] — Numba-safe boolean mask for float64 arrays."""
    n = arr.size
    count = 0
    for i in range(n):
        if cond[i]:
            count += 1
    out = np.empty(count, np.float64)
    j = 0
    for i in range(n):
        if cond[i]:
            out[j] = arr[i]
            j += 1
    return out
# -----------------------------
# 2D histogram helper (auto-range)
# -----------------------------
@njit("Tuple((float64[:,:], float64, float64))(complex128[:], int64)", fastmath=True, cache=True)
def hist2d_complex(z, B):
    n = z.size
    if n == 0:
        return np.zeros((B, B), np.float64), 0.0, 0.0

    zc = z - np.mean(z)
    r = np.abs(zc)
    rs = np.sort(r)
    k = int(0.995 * (n - 1))
    rmax = rs[k]
    if rmax <= 1e-15:
        return np.zeros((B, B), np.float64), 0.0, 0.0
    zn = zc / rmax

    re = np.real(zn)
    im = np.imag(zn)
    x = 0.5 * (re + 1.0) * (B - 1)
    y = 0.5 * (im + 1.0) * (B - 1)
    ix = np.minimum(np.maximum(x, 0.0), B - 1.0).astype(np.int64)
    iy = np.minimum(np.maximum(y, 0.0), B - 1.0).astype(np.int64)
    flat = ix * B + iy
    H_flat = np.bincount(flat, minlength=B * B).astype(np.float64)
    H = H_flat.reshape((B, B))
    s = np.sum(H)
    if s > 0:
        H /= s

    # use mask_f64 for entropy and occupancy
    Hflat = H.ravel()
    cond = Hflat > 0.0
    p = mask_f64(Hflat, cond)
    H2D = -np.sum(p * np.log(p))
    H2Dn = H2D / np.log(B * B)
    occ = p.size / float(B * B)

    return H, occ, H2Dn


#=========================================
#
#=========================================
def julia(N: int, c: np.complex128= -0.8 + 0.156j,maxi:int=200,w:float=1.5,eqn:int=1) -> np.ndarray:
    N = int(N)
    out  = np.zeros(N, np.complex128)
    kept = 0
    # 1) probe sample
    boost = 10
    s = julia_sample(
        np.int64(boost*N), 
        np.complex128(c), 
        np.float64(w), 
        np.complex128(0+0j), 
        np.int32(maxi),
        np.int32(eqn)
    )
    take = min(s.size, N)
    if take:
        out[:take] = s[:take]
        kept += take
    else:
        return out
    p = max(0.01, (s.size / (boost*N)))
    draw = 2*int(math.ceil(N / p))
    if draw < 1: draw = 1
    center = np.mean(out[:kept])
    w = 0.5 * max(
        np.ptp(out[:kept].real),
        np.ptp(out[:kept].imag)
    ) * 1.5
    rounds = 0
    while kept < N:
        rounds += 1
        if rounds > 10 : break
        need = (N - kept)
        s = julia_sample(np.int64(draw), np.complex128(c), np.float64(w),np.complex128(center), np.int32(300),int(eqn))
        take = min(s.size, need)
        if take:
            out[kept:kept + take] = s[:take]
            kept += take
        center = np.mean(out[:kept])
        w = 0.5 * max(
            np.ptp(out[:kept].real),
            np.ptp(out[:kept].imag)
        ) * 1.15
    return out[:min(kept,N)]  # exactly N filled unless the defensive break triggered

def dict2julia(d):

    n    = int(d["n"][0].real)     if "n"    in d else 1_000
    c    = d["c"][0]               if "c"    in d else (-0.8 + 0.156j)
    maxi = int(d["maxi"][0].real)  if "maxi" in d else 200
    w    = d["w"][0].real          if "w"    in d else 1.5
    eqn  = int(d["eqn"][0].real)   if "eqn"  in d else 1
    
    z=julia(n,c,maxi,w,eqn)
    return z



#=========================================
# "interesting" image classification
#=========================================

if __name__ == "__main__":
    p = argparse.ArgumentParser("galaxy-cli", description="Julia set renderer")
    p.add_argument("--spec", required=True,help="Julia set specification (can include expandspec braces)")
    p.add_argument("--pix", type=int, default=5000, help="Tile width/height in pixels (default 25000)")
    p.add_argument("--out", type=str, default="julia.png", help="Output PNG path")
    p.add_argument("--cols", type=int, default=None, help="Columns if chain expands to multiple tiles")
    p.add_argument("--rows", type=int, default=None, help="Rows if chain expands to multiple tiles")
    p.add_argument("--invert", action="store_true", help="Invert black/white")
    p.add_argument("--thumb",type=int, default=None,  help="Save thumbnail")
    args = p.parse_args()

    waypoints=[f"{z.real:+5f}{z.imag:+5f}j" for z in c_sampler(10000,0.01,0.05,1.0,1,200)]

    specs = expandspec.expand_cartesian_lists(args.spec,names=waypoints)
   
    dicts = []
    for spec in specs:
        names, A = specparser.parse_names_and_args(spec, MAXA=12)
        d = dict(zip(names, A))
        d["spec"] = spec
        dicts.append(d)
    
    
    canvases = []
    titles = []
    for i,d in enumerate(dicts,start=1):
        print(f"{i}/{len(dicts)} Rendering {d['spec']}")
        j = dict2julia(d)
        canvas = galaxy_raster.render_to_canvas(j, args.pix, 0.1)
        canvases.append(canvas)
        titles.append(f"{d['spec']} ")
    

    n = len(canvases)

    if args.cols:
        cols = args.cols
    elif args.rows:
        cols = int(round(n / args.rows))
    else: 
        cols = max(1, int(round(math.sqrt(n))))

    galaxy_raster.save_mosaic_png_bilevel(
        tiles= canvases, 
        titles=titles,
        cols=cols, 
        gap=20,
        out_path=args.out, 
        invert=args.invert,
        thumbnail=args.thumb,
    )

