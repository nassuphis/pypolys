#!/usr/bin/env python
import mlx.core as mx
import numpy as np
import pyvips
import argparse
import functools
from functools import partial, reduce
import numexpr as ne
import polys
import polys.polystate
import polys.polyutil
import cv2
from scipy import ndimage, spatial
from scipy.stats import rankdata

#######################################
#
# pipeline composition
#
#######################################

def compose_nest(*funcs):
    if not funcs:
        return lambda x: x  # Identity for empty pipeline
    return functools.reduce(lambda f, g: lambda x: f(g(x)), reversed(funcs))

def make_pipeline(txt,dict):
    pipeline_steps = []
    for step in txt.split(','):
        parts = step.split('_')
        op_name = parts[0]
        op_args = parts[1:]
        if op_name not in dict:
            raise ValueError(f"Unknown operation: {op_name}")
        op_func = dict[op_name]
        pipeline_steps.append((op_func, op_args))

    bound_funcs = [functools.partial(op_func, a=op_args) for op_func, op_args in pipeline_steps]
    pipeline = compose_nest(*bound_funcs)
    return pipeline

#######################################
#
# state
#
#######################################

result_fn = 'myresult.npz'
results_pct = 10 # more than 1
sort_unique = False
use_mlx = False
hfrm_pipeline = "clip"
sfrm_pipeline = "one"
vfrm_pipeline = "one"
res = 1000
row_mat, col_mat = np.indices(((res,res)))
H = np.zeros((res,res),dtype=np.float32)
S = np.ones((res,res),dtype=np.float32) 
V = np.ones((res,res),dtype=np.float32) 

params = {
    "rfr":50.0,
    "cfr":50.0,
    "afr":1.0,
    "hmlt":1.0,
    "hoff":0.0,
    "h0":0.0,
    "h1":1.0,
    "xc": 0.0,
    "yc": 0.0,
    "r": 0.5,
    "ccf":1.0,
    "exp": 1.0,
    "cst": 0.0
}

polyres = {
    "i": np.zeros((1000)),
    "j": np.zeros((1000)),
    "x": np.zeros((1000)),
    "y": np.zeros((1000)),
    "z": np.zeros((1000)),
    "r": np.zeros((1000)),
    "gbi" : np.zeros((1000))

}

#######################################
#
# mlx sort_unique
#
#######################################

def mlx_which(x: mx.array,*,stream=mx.gpu):
     if x.size == 0:
        return mx.zeros((0,), dtype=mx.int32, stream=stream)
     xb  = x.astype(mx.int32)
     xb[0] = 0
     ar = mx.arange(x.size, dtype=mx.int32, stream=stream)
     xi = mx.cummax(ar * xb)
     cum = mx.cumsum(xb, stream=stream)
     maxi = cum[-1].item() + 1
     idx = mx.zeros((maxi,), dtype=mx.int32, stream=stream)
     idx[cum]=xi
     if x[0]:
        return idx
     return idx[1:]

def sort_unique_uint16x4_mlx(x: np.array,*,stream=mx.gpu): 
    if  x.ndim != 2 or x.shape[1] != 4:
        raise ValueError("expect (N,4) uint16 array")
    x_u16 = x.view(np.uint16)
    arr = mx.array(x_u16, dtype=mx.uint16)
    a64  = arr.astype(mx.uint64)
    keys = (a64[:, 0] << 48) | (a64[:, 1] << 32) | (a64[:, 2] << 16) |  a64[:, 3]
    order       = mx.argsort(keys, axis=0, stream=stream)
    rows_sorted = arr[order]
    keys_sorted = keys[order]
    first = mx.concatenate(
        [mx.array([True], dtype=mx.bool_), keys_sorted[1:] != keys_sorted[:-1]],
        axis=0
    )
    idx = mlx_which(first, stream=stream)
    unique_rows = rows_sorted[idx]
    mx.eval(unique_rows)
    result = np.array(unique_rows)
    return result

#######################################
#
# utility functions
#
#######################################

def sarg(a,i,v):
    return a[i] if len(a)>i else v

def iarg(a,i,v):
    return int(a[i]) if len(a)>i else v

def farg(a,i,v):
    return float(a[i]) if len(a)>i else v

def carg(a,i,v):
    return complex(a[i]) if len(a)>i else v

def ratio(x,y):
    num = np.where(abs(y)>1e-10,x,0)
    den = np.where(abs(y)>1e-10,y,1)
    return num/den

#######################################
#
# load data
#
#######################################

def load_polyres():
    results_z = np.load(result_fn)
    results = (results_z['arr_0']).astype(np.uint16)
    if results_pct<1:
        end = int(results.shape[0]*results_pct)
        print(f"size: {results.shape[0]:,}, loading: {end:,} {results.dtype}")
        results=results[:end,:]
    if sort_unique:
        if use_mlx:
            print("using MLX de-dupe")
            results = sort_unique_uint16x4_mlx(results)
        else:
            print("using NumPy de-dupe")
            results = np.unique(results,axis=0)
        print(f"de-duped to size: {results.shape[0]:,}")
    #results = results_z['arr_0']
    height = results[:,0].max()+1
    width = results[:,1].max()+1
    polyres["result_height"] = height
    polyres["result_width"] = width
    ifac = (res-1)/(height-1)
    jfac = (res-1)/(width-1)
    pi = (results[:,0]*ifac).astype(np.uint16)
    pj = (results[:,1]*jfac).astype(np.uint16)
    results[:,0] = pi
    results[:,1] = pj

    print(f"{height:,}->{np.max(pi)+1:,}")

    idx = np.lexsort((results[:,1], results[:,0]))
    results=results[idx,:]
    polyres["i"] = results[:,0]
    polyres["j"] = results[:,1]
    polyres["height"]=res
    polyres["width"]=res
    diff = np.diff(polyres["i"]*polyres["width"]+polyres["j"])
    is_new = np.concatenate(([True],diff!=0))
    # group broadcast index group results -> individual locations
    polyres["gbi"] = np.cumsum(is_new)-1 
    polyres["starts"] = np.nonzero(is_new)[0]
    polyres["x"] = results[:,2]
    polyres["y"] = results[:,3]
    polyres["z"] = 2*(norm(polyres["i"])-0.5) + 2j*(norm(polyres["j"])-0.5) # parameters
    polyres["r"] = 2*(norm(polyres["x"])-0.5) + 2j*(norm(polyres["y"])-0.5) # roots
    H = np.zeros((res,res),dtype=np.float32)
    np.add.at(H, (polyres["i"], polyres["j"]), 1.0)
    polyres["haves"] = H > 0.0
    polyres["havemany"] = H > 1.5
    print(f"results: {results.shape[0]:,} by {results.shape[1]:,}")
    print(f"results: x type {polyres["x"].dtype} y type {polyres["y"].dtype}")
    return

def polyres_fill_havenots(x,a):
    # indices of the nearest True cell for every position
    mask_name = a[0] if len(a)>0 else "haves"
    mask = polyres[mask_name]
    idx = ndimage.distance_transform_edt(~mask,return_distances=False,return_indices=True)
    filled = x.copy()
    filled[~mask] = x[tuple(idx[:, ~mask])]
    return filled

def polyres_centroid():
    H = np.zeros((polyres["height"],polyres["width"]),dtype=np.complex64)      
    centroid=polys.polyutil.group_apply(polyres["r"],polyres["starts"],np.mean)       
    H[polyres["i"],polyres["j"]] = centroid[polyres['gbi']]  
    return H

def blurr(H, a):
    sigma = float(a[0]) if len(a)>0 else 3.0
    blurred = cv2.GaussianBlur(H.real.astype(np.float32), (0, 0), sigma)
    return blurred

def direction(F,a):
    sigma = float(a[0]) if len(a)>0 else 3.0
    ur = cv2.GaussianBlur(F.real.astype(np.float32), (0, 0), sigma)
    vr = cv2.GaussianBlur(F.imag.astype(np.float32), (0, 0), sigma)

    # Compute gradients for both components
    Ix_u = cv2.Sobel(ur, cv2.CV_32F, 1, 0, ksize=3)
    Iy_u = cv2.Sobel(ur, cv2.CV_32F, 0, 1, ksize=3)
    Ix_v = cv2.Sobel(vr, cv2.CV_32F, 1, 0, ksize=3)
    Iy_v = cv2.Sobel(vr, cv2.CV_32F, 0, 1, ksize=3)

    # Combine: Jxx = ∂u/∂x² + ∂v/∂x² etc.
    Jxx = cv2.GaussianBlur(Ix_u**2 + Ix_v**2, (0, 0), sigma)
    Jyy = cv2.GaussianBlur(Iy_u**2 + Iy_v**2, (0, 0), sigma)
    Jxy = cv2.GaussianBlur(Ix_u*Iy_u + Ix_v*Iy_v, (0, 0), sigma)

    # Orientation
    orientation = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy)
    return (orientation + np.pi/2) / np.pi

def polyres_centroid_dist_abs():
    H = np.zeros((polyres["height"],polyres["width"]),dtype=np.float32)  
    def dfun(x):
        cnt = np.mean(x)
        diff = x-cnt
        mean_diff = np.mean(np.abs(diff))
        return mean_diff.real
    centroid=polys.polyutil.group_apply(polyres["r"],polyres["starts"],dfun)       
    H[polyres["i"],polyres["j"]] = centroid[polyres['gbi']]  
    return H

def polyres_centroid_dist_abs_max():
    H = np.zeros((polyres["height"],polyres["width"]),dtype=np.float32)  
    def dfun(x):
        cnt = np.mean(x)
        diff = x-cnt
        mean_diff = np.max(np.abs(diff))
        return mean_diff.real
    centroid=polys.polyutil.group_apply(polyres["r"],polyres["starts"],dfun)       
    H[polyres["i"],polyres["j"]] = centroid[polyres['gbi']]  
    return H

def polyres_centroid_dist_abs_min():
    H = np.zeros((polyres["height"],polyres["width"]),dtype=np.float32)  
    def dfun(x):
        cnt = np.mean(x)
        diff = x-cnt
        mean_diff = np.min(np.abs(diff))
        return mean_diff.real
    centroid=polys.polyutil.group_apply(polyres["r"],polyres["starts"],dfun)       
    H[polyres["i"],polyres["j"]] = centroid[polyres['gbi']]  
    return H

def polyres_centroid_dist_angle():
    H = np.zeros((polyres["height"],polyres["width"]),dtype=np.float32)  
    def dfun(x):
        cnt = np.mean(x)
        diff = x-cnt
        mean_diff = np.mean((np.angle(diff)/np.pi+1)/2)
        return mean_diff.real
    centroid=polys.polyutil.group_apply(polyres["r"],polyres["starts"],dfun)       
    H[polyres["i"],polyres["j"]] = centroid[polyres['gbi']]  
    return H

def polyres_centroid_dist_angle_max():
    H = np.zeros((polyres["height"],polyres["width"]),dtype=np.float32)  
    def dfun(x):
        cnt = np.mean(x)
        diff = x-cnt
        mean_diff = np.max((np.angle(diff)/np.pi+1)/2)
        return mean_diff.real
    centroid=polys.polyutil.group_apply(polyres["r"],polyres["starts"],dfun)       
    H[polyres["i"],polyres["j"]] = centroid[polyres['gbi']]  
    return H

def polyres_centroid_dist_angle_min():
    H = np.zeros((polyres["height"],polyres["width"]),dtype=np.float32)  
    def dfun(x):
        cnt = np.mean(x)
        diff = x-cnt
        mean_diff = np.min((np.angle(diff)/np.pi+1)/2)
        return mean_diff.real
    centroid=polys.polyutil.group_apply(polyres["r"],polyres["starts"],dfun)       
    H[polyres["i"],polyres["j"]] = centroid[polyres['gbi']]  
    return H

def polyres_centroid_dist_angle_range():
    H = np.zeros((polyres["height"],polyres["width"]),dtype=np.float32)  
    def dfun(x):
        cnt = np.mean(x)
        diff = x-cnt
        amin = np.min((np.angle(diff)/np.pi+1)/2)
        amax = np.max((np.angle(diff)/np.pi+1)/2)
        diff = amax - amin
        return diff.real
    centroid=polys.polyutil.group_apply(polyres["r"],polyres["starts"],dfun)       
    H[polyres["i"],polyres["j"]] = centroid[polyres['gbi']]  
    return H

def polyres_variance():
    H = np.zeros((polyres["height"],polyres["width"]),dtype=np.float32)   
    def vfun(x):
        r = np.abs(np.mean(x))
        if abs(r)>0:
            return np.mean((np.abs(x-np.mean(x))/r)**2).real
        return 0.0

    variance = polys.polyutil.group_apply(polyres["r"],polyres["starts"],vfun)
    H[polyres["i"],polyres["j"]] = variance[polyres['gbi']]  
    return H

def polyres_count(): # spectral radius
    H = np.zeros((polyres["height"],polyres["width"]),dtype=np.float32)   
    gfun = lambda x:x.size
    gres = polys.polyutil.group_apply(polyres["r"],polyres["starts"],gfun)
    H[polyres["i"],polyres["j"]] = gres[polyres['gbi']]  
    return H

def polyres_mean(): # spectral radius
    H = np.zeros((polyres["height"],polyres["width"]),dtype=np.float32)   
    gfun = lambda x:np.mean(np.abs(x))
    gres = polys.polyutil.group_apply(polyres["r"],polyres["starts"],gfun)
    H[polyres["i"],polyres["j"]] = gres[polyres['gbi']]  
    return H

def polyres_amean(): # spectral radius
    H = np.zeros((polyres["height"],polyres["width"]),dtype=np.float32)   
    gfun = lambda x:np.mean((np.angle(x)/np.pi+1.0)/2.0)
    gres = polys.polyutil.group_apply(polyres["r"],polyres["starts"],gfun)
    H[polyres["i"],polyres["j"]] = gres[polyres['gbi']]  
    return H

def polyres_max(): # spectral radius
    H = np.zeros((polyres["height"],polyres["width"]),dtype=np.float32)   
    gfun = lambda x:np.max(np.abs(x))
    gres = polys.polyutil.group_apply(polyres["r"],polyres["starts"],gfun)
    H[polyres["i"],polyres["j"]] = gres[polyres['gbi']]  
    return H

def polyres_amax(): # spectral radius
    H = np.zeros((polyres["height"],polyres["width"]),dtype=np.float32)   
    gfun = lambda x:np.max((np.angle(x)/np.pi+1.0)/2.0)
    gres = polys.polyutil.group_apply(polyres["r"],polyres["starts"],gfun)
    H[polyres["i"],polyres["j"]] = gres[polyres['gbi']]  
    return H

def polyres_min(): # spectral radius
    H = np.zeros((polyres["height"],polyres["width"]),dtype=np.float32)   
    gfun = lambda x:np.min(np.abs(x))
    gres = polys.polyutil.group_apply(polyres["r"],polyres["starts"],gfun)
    H[polyres["i"],polyres["j"]] = gres[polyres['gbi']]  
    return H

def polyres_amin(): # spectral radius
    H = np.zeros((polyres["height"],polyres["width"]),dtype=np.float32)   
    gfun = lambda x:np.min((np.angle(x)/np.pi+1.0)/2.0)
    gres = polys.polyutil.group_apply(polyres["r"],polyres["starts"],gfun)
    H[polyres["i"],polyres["j"]] = gres[polyres['gbi']]  
    return H

def polyres_range(): # spectral radius
    H = np.zeros((polyres["height"],polyres["width"]),dtype=np.float32)   
    gfun = lambda x:np.max(np.abs(x))-np.min(np.abs(x))
    gres = polys.polyutil.group_apply(polyres["r"],polyres["starts"],gfun)
    H[polyres["i"],polyres["j"]] = gres[polyres['gbi']]  
    return H

def polyres_arange():
    H = np.zeros((polyres["height"],polyres["width"]),dtype=np.float32)   
    gfun = lambda x:np.max(np.angle(x))-np.min(np.angle(x))
    gres = polys.polyutil.group_apply(polyres["r"],polyres["starts"],gfun)
    H[polyres["i"],polyres["j"]] = gres[polyres['gbi']]  
    return H

def approximate_log_discriminant(x, y, bins=8):   
    counts, _, _ = np.histogram2d(x, y, bins=bins)
    count_max = np.max(counts)
    if count_max>0:
        normed = counts / count_max
    else:
        normed = np.zeros(counts.shape)
    return np.sum(normed ** 2.0)

def polyres_discriminant(a):
    bins = ne.evaluate(a[0]) if len(a)>0 else 8
    H = np.zeros((polyres["height"],polyres["width"]),dtype=np.float32)   
    gres = polys.polyutil.group_apply2(
        polyres["x"],
        polyres["y"],
        polyres["starts"],
        lambda x, y: approximate_log_discriminant(x, y, bins=bins)
    )
    H[polyres["i"],polyres["j"]] = gres[polyres['gbi']]  
    return H

def norm(x):
    max_x = np.max(x)
    min_x = np.min(x)
    if max_x-min_x>0:
        nx = ((x-min_x)/(max_x-min_x)).astype(np.float32)
        return nx
    else:
        return x.astype(np.float32)
        
def norm_clip(x,a):
    qlo = float(a[0]) if len(a)>0 else 0
    qhi = float(a[1]) if len(a)>1 else 1
    if qlo<0 or qhi<0:
        xc=x
    else:
        vlo, vhi = np.quantile(x,[qlo,qhi])
        xc = np.clip(x,vlo,vhi)
    max_x = np.max(xc)
    min_x = np.min(xc)
    if max_x-min_x>0:
        nx = ((x-min_x)/(max_x-min_x)).astype(np.float32)
        return nx
    else:
        return x.astype(np.float32)
            
def norm_scale(x,a):
    h0 = float(a[0]) if len(a)>0 else 0
    h1 = float(a[1]) if len(a)>1 else 1
    max_x = np.max(x)
    min_x = np.min(x)
    if max_x-min_x>0:
        nx = ((x-min_x)/(max_x-min_x)).astype(np.float32)
        snx = nx*(h1-h0)+h0
        return snx
    else:
        return x.astype(np.float32)

def greater(H,a):
    val = float(a[0]) if len(a)>0 else 0.0
    dir = float(a[1]) if len(a)>1 else 1.0
    return np.where(dir*(H-val)>0,1.0,0.0)

def equals(H,a):
    val = float(a[0]) if len(a)>0 else 1.0
    return np.where(np.abs(H-val)<1e-10,1.0,0.0)

def nz(H,a):
    true_val = float(a[0]) if len(a)>0 else 1.0
    false_val = float(a[1]) if len(a)>1 else 0.0
    thresh_val = float(a[2]) if len(a)>2 else 0.0
    return np.where(np.abs(H)>thresh_val,true_val,false_val)

def haves(H):
    hr = polyres["haves"]
    return H[hr]
          
def rank(H,a):
    lo = float(a[0]) if len(a)>0 else 0.0
    hi = float(a[1]) if len(a)>1 else 1.0
    bc = int(a[2]) if len(a)>2 else 256
    hist, bins = np.histogram(H.flatten(), bins=bc)
    cdf = np.cumsum(hist>0)
    cdf =  (hi-lo) * cdf / cdf[-1] + lo # normalize to [lo, hi]
    img_eq = np.interp(H.flatten(), bins[:-1], cdf)
    return img_eq.reshape(H.shape)

def rank_haves(H,a):
   
    Hr = haves(H)
    i,j = np.indices(H.shape)

    lo = float(a[0]) if len(a)>0 else 0.0
    hi = float(a[1]) if len(a)>1 else 1.0
    bc = int(a[2]) if len(a)>2 else 256

    hist, bins = np.histogram(Hr, bins=bc)
    cdf = np.cumsum(hist>0) # rank based
    cdf =  (hi-lo) * cdf / cdf[-1] + lo # normalize to [lo, hi]
    img_eq = np.interp(Hr, bins[:-1], cdf)

    res = np.zeros(H.shape)
    res[haves(i),haves(j)] = img_eq
    return res


def heq(H,a):
    lo = float(a[0]) if len(a)>0 else 0.0
    hi = float(a[1]) if len(a)>1 else 1.0
    bc = int(a[2]) if len(a)>2 else 256
    hist, bins = np.histogram(H.flatten(), bins=bc)
    cdf = hist.cumsum()
    cdf = (hi-lo)*cdf / cdf[-1]  + lo # normalize to [lo, hi]
    img_eq = np.interp(H.flatten(), bins[:-1], cdf)
    return img_eq.reshape(H.shape)


def heq_haves(H,a):

    lo = float(a[0]) if len(a)>0 else 0.0
    hi = ne.evaluate(a[1]) if len(a)>1 else 1.0
    bc = int(a[2]) if len(a)>2 else 256

    Hr = haves(H)
    i,j = np.indices(H.shape)

    hist, bins = np.histogram(Hr, bins=bc)
    cdf = hist.cumsum()
    cdf = (hi-lo)*cdf / cdf[-1]  + lo # normalize to [lo, hi]
    img_eq = np.interp(Hr, bins[:-1], cdf)

    res = np.zeros(H.shape)
    res[haves(i),haves(j)] = img_eq
    return res

def hue_clip(H,a):
    h0 = float(a[0]) if len(a)>0 else 0
    h1 = float(a[1]) if len(a)>1 else 0
    cH = norm(H)*(h1-h0)+h0
    return cH



def wavefield(H,a):
    x = 2 * np.pi * norm(H.real) * (float(a[0]) if len(a)>0 else 0)
    y = 2 * np.pi * norm(H.imag) * (float(a[1]) if len(a)>1 else 0)
    e = a[2] if len(a)>2 else "cos(x)+1j*sin(y)"
    v = ne.evaluate(e,{"x":x,"y":y})
    return v

def wavefield3(H,a):
    afr = float(a[0]) if len(a)>0 else 10
    r = np.hypot(norm(H.real) - 0.5, norm(H.imag) - 0.5)
    v = np.sin(2 * np.pi * afr * r)
    return v

def anglefield(H,a):
    x = (norm(H.real) - 0.5) * 2
    y = (norm(H.imag) - 0.5) * 2
    afr = float(a[0]) if len(a)>0 else 10
    e = a[1] if len(a)>1 else "x+1j*y"
    v = ne.evaluate(e,{"x":x,"y":y})
    return ( afr * norm(v) ) % 1

def absfield(H,a):
    afr = float(a[0]) if len(a)>0 else 10
    v0 = np.angle(row_mat + 1j * col_mat)
    return afr*norm(v0)

def xshiftdiff(H):
    return np.roll(H,shift=1,axis=1)-H


def multiply_and_clip_to_one(H,a):
    afr = float(a[0]) if len(a)>0 else 10
    return (afr*H)%1

def map(H,a):
    v = np.array([ne.evaluate(expr).item() for expr in a], dtype=np.float32)
    i = norm(H) * (len(v) - 1)
    indices = np.clip((i).astype(int), 0, len(v) - 1)
    mv = v[indices]
    pH = np.zeros(H.shape)
    pH[:] = mv
    return pH

def bkr(t,a):
    times = int(a[0]) if len(a)>0 else int(1)
    x = np.real(t)
    y = np.imag(t)
    for _ in range(times):
        x_fold = x % 1  # fractional part of x
        y_fold = y % 1  # fractional part of y
        x_new = (2 * x_fold) % 1
        shift = np.floor(2 * x_fold)
        y_new = (y_fold + shift) / 2
        x = x_new
        y = y_new
    return x_new + 1j * y_new

def ebkr(t):
    def to01(x):
        return 1 / (1 + np.exp(-x))
    x = np.real(t)
    y = np.imag(t)
    x01 = to01(x)
    y01 = to01(y)
    x_new = (2 * x01) % 1
    shift = np.floor(2 * x01)
    y_new = (y01 + shift) / 2
    return x_new + 1j * y_new



#######################################
#
# generate constants
#
#######################################

pfrm_constants = {
    "zero": lambda x,a: np.zeros(x.shape),
    "one": lambda x,a: np.ones(x.shape),
    "red":         lambda x,a: np.full_like(x, 0.00),
    "orange":      lambda x,a: np.full_like(x, 0.08),
    "yellow":      lambda x,a: np.full_like(x, 0.17),
    "chartreuse":  lambda x,a: np.full_like(x, 0.25),
    "green":       lambda x,a: np.full_like(x, 0.33),
    "spring":      lambda x,a: np.full_like(x, 0.42),
    "cyan":        lambda x,a: np.full_like(x, 0.50),
    "azure":       lambda x,a: np.full_like(x, 0.58),
    "blue":        lambda x,a: np.full_like(x, 0.67),
    "violet":      lambda x,a: np.full_like(x, 0.75),
    "magenta":     lambda x,a: np.full_like(x, 0.83),
    "rose":        lambda x,a: np.full_like(x, 0.92),
    "value":       lambda x,a: np.full_like(V,float(a[0]) if len(a)>0 else 0.0),
}

#######################################
#
# input values
#
#######################################

pfrm_inputs = {
    "rows": lambda x,a: row_mat,
    "cols": lambda x,a: col_mat,
    "idx": lambda x,a: col_mat + 1j * row_mat,
    "j": lambda x,a: 2*(norm(col_mat)-0.5),
    "i": lambda x,a: 2*(norm(row_mat)-0.5),
    "z": lambda x,a: 2*(norm(col_mat)-0.5)+1j*2*(norm(row_mat)-0.5),
    "r": lambda x,a: polyres["r"],
    "roots": lambda x,a: polyres["r"],
    "H" : lambda x,a : H,
    "S" : lambda x,a : S,
    "V" : lambda x,a : V,
}

#######################################
#
# transforms
#
#######################################

pfrm_transforms_conformal = {
    "wbl": lambda x,a: ratio( x + 1j*np.sin(x), x + 1j*np.cos(x) ),
    "tws": lambda x,a: ratio( np.exp(x) + carg(a,0,-1), np.exp(x) * 1j + carg(a,1,1) ),
    "plk": lambda x,a: ratio( np.sin(1j+x), np.cos(1j+x) ),
    "zzg": lambda x,a: ratio( x +  carg(a,0,2), x + carg(a,1,-2) ),
    "ltl": lambda x,a: ratio( x + carg(a,0,- 2j), x + carg(a,0,2j)),
    "kth": lambda x,a: ratio(1 + x,1j - x),
    "jkw": lambda x,a: x+ratio(1,x),
}

#######################################
#
# poly transforms
#
#######################################

def coeff2(z):
   return z.real + z.imag + 1j*(z.real * z.imag)

def coeff3(z):
  top = np.divide(1, z.real + 2, out=np.zeros_like(z.real), where=(z.real + 2)!=0)
  bot = np.divide(1, z.imag + 2, out=np.zeros_like(z.real), where=(z.imag + 2)!=0)
  return top + 1j * bot

def coeff3a(z):
  top = np.divide(1, z.real + 1, out=np.zeros_like(z.real), where=(z.real + 1)!=0)
  bot = np.divide(1, z.imag + 1, out=np.zeros_like(z.real), where=(z.imag + 1)!=0)
  return top + 1j * bot

def coeff4(z):
  return np.cos(z.real) + 1j * np.sin(z.imag)

def coeff5(z):
  top = np.divide(1, z.real, out=np.zeros_like(z.real), where=(z.real)!=0)
  bot = np.divide(1, z.imag, out=np.zeros_like(z.real), where=(z.imag)!=0)
  return z.real + bot + 1j * ( z.imag + top )

def coeff5a(z):
  top = np.divide(1, z.real, out=np.zeros_like(z.real), where=(z.real)!=0)
  bot = np.divide(1, z.imag, out=np.zeros_like(z).real, where=(z.imag)!=0)
  return z.real + top + 1j * ( z.imag + bot )

def coeff6(z):
  t1 = z.real
  t2 = z.imag
  num1 = t1**3 + 1j
  den1 = t1**3 - 1j
  val1 = np.where(abs(den1)>1e-10,num1,0) / np.where(abs(den1)>1e-10,den1,1)
  num2 = t2**3 + 1j
  den2 = t2**3 - 1j
  val2 = np.where(abs(den2)>1e-10,num2,0) / np.where(abs(den2)>1e-10,den2,1)
  return val1 + 1j *val2 

def coeff7(z):
  t1 = z.real
  t2 = z.imag
  top1  = t1 + np.sin(t1)
  bot1  = t1 + np.cos(t1)
  val1 = np.where(abs(bot1)>1e-10,top1,0) / np.where(abs(bot1)>1e-10,bot1,1)
  top2  = t2 + np.sin(t2)
  bot2  = t2 + np.cos(t2)
  val2 = np.where(abs(bot2)>1e-10,top2,0) / np.where(abs(bot2)>1e-10,bot2,1)
  return val1 + 1j * val2
    
def coeff8(z): 
    t1 = z.real
    t2 = z.imag
    top1  = t1 + np.sin(t2)
    bot1  = t2 + np.cos(t1)
    val1 = np.where(abs(bot1)>1e-10,top1,0) / np.where(abs(bot1)>1e-10,bot1,1)
    top2  = t2 + np.sin(t1)
    bot2  = t1 + np.cos(t2)
    val2 = np.where(abs(bot2)>1e-10,top2,0) / np.where(abs(bot2)>1e-10,bot2,1)
    return val1 + 1j * val2

def coeff9(z):
    t1 = z.real
    t2 = z.imag
    top1  = t1*t1 + 1j * t2
    bot1  = t1*t1 - 1j * t2
    val1 = np.where(abs(bot1)>1e-10,top1,0) / np.where(abs(bot1)>1e-10,bot1,1)
    top2  = t2*t2 + 1j * t1
    bot2  = t2*t2 - 1j * t1
    val2 = np.where(abs(bot2)>1e-10,top2,0) / np.where(abs(bot2)>1e-10,bot2,1)
    return val1 + 1j * val2

def coeff10(z):
    t1 = z.real
    t2 = z.imag
    top1  = t1**4 - t2
    bot1  = t1**4 + t2
    val1 = np.where(abs(bot1)>1e-10,top1,0) / np.where(abs(bot1)>1e-10,bot1,1)
    top2  = t2**4 - t1
    bot2  = t2**4 + t1
    val2 = np.where(abs(bot2)>1e-10,top2,0) / np.where(abs(bot2)>1e-10,bot2,1)
    return val1 + 1j * val2
 

def coeff11(z):
    t1 = z.real
    t2 = z.imag
    val1 = np.log( t1**4 + 2 )
    val2 = np.log( t2**4 + 2 )
    return val1 + 1j * val2

def coeff12(z):
    t1 = z.real
    t2 = z.imag
    val1 = 2*t1**4 - 3*t2**3 + 4*t1**2 - 5*t2
    val2 = 2*t2**4 - 3*t1**3 + 4*t2**2 - 5*t1
    return val1 + 1j * val2

pfrm_transforms_poly = {
    "coeff2": lambda x,a: coeff2(x),
    "coeff3": lambda x,a: coeff3(x),
    "coeff3a": lambda x,a: coeff3a(x),
    "coeff4": lambda x,a: coeff4(x),
    "coeff5": lambda x,a: coeff5(x),
    "coeff5a": lambda x,a: coeff5a(x),
    "coeff6": lambda x,a: coeff6(x),
    "coeff7": lambda x,a: coeff7(x),
    "coeff8": lambda x,a: coeff8(x),
    "coeff9": lambda x,a: coeff9(x),
    "coeff10": lambda x,a: coeff10(x),
    "coeff11": lambda x,a: coeff11(x),
    "coeff12": lambda x,a: coeff12(x),
}

#######################################
#
# clip values
#
#######################################

def above(H,a):
    x = (norm(col_mat) - 0.5)
    xc = float(a[0]) if len(a)>0 else 0
    mask = x > xc
    return np.where(mask,H,0)

def below(H,a):
    x = (norm(col_mat) - 0.5)
    xc = float(a[0]) if len(a)>0 else 0
    mask = x < xc
    return np.where(mask,H,0)

def left(H,a):
    y = (norm(row_mat) - 0.5)
    yc = float(a[0]) if len(a)>0 else 0
    mask = y < yc
    return np.where(mask,H,0)

def right(H,a):
    y = (norm(row_mat) - 0.5)
    yc = float(a[0]) if len(a)>0 else 0
    mask = y > yc
    return np.where(mask,H,0)

def in_circle(H,a):
    y = (norm(H.imag) - 0.5)*2
    x = (norm(H.real) - 0.5)*2
    xc = float(a[0]) if len(a)>0 else 0
    yc = float(a[1]) if len(a)>1 else 0
    r = float(a[2]) if len(a)>2 else 0.25
    cc = (x-xc)**2+(y-yc)**2 - r
    mask =  cc > 0
    return np.where(mask,1,0)

def out_circle(H,a):
    y = (norm(H.imag) - 0.5)*2
    x = (norm(H.real) - 0.5)*2
    xc = float(a[0]) if len(a)>0 else 0
    yc = float(a[1]) if len(a)>1 else 0
    r = float(a[2]) if len(a)>2 else  0.25
    cc = (x-xc)**2+(y-yc)**2 - r
    mask =  cc < 0
    return np.where(mask,H,0)

def in_square(H,a):
    y = (norm(H.imag) - 0.5)*2
    x = (norm(H.real) - 0.5)*2
    xc = float(a[0]) if len(a)>0 else 0
    yc = float(a[1]) if len(a)>1 else 0
    r = ne.evaluate(a[2]) if len(a)>2 else 0.25
    xm = abs(x-xc) < r
    ym = abs(y-yc) < r
    mask =  ym & xm
    return np.where(mask,H,0)

def out_square(H,a):
    y = (norm(row_mat) - 0.5)*2
    x = (norm(col_mat) - 0.5)*2
    xc = float(a[0]) if len(a)>0 else 0
    yc = float(a[1]) if len(a)>1 else 0
    r = float(a[2]) if len(a)>2 else 0.25
    xm = abs(x-xc) > r
    ym = abs(y-yc) > r
    mask =  ym | xm
    return np.where(mask,H,0)

def square_strip(H,a):
    y = (norm(row_mat) - 0.5) * 2
    x = (norm(col_mat) - 0.5) * 2
    sml = float(a[0]) if len(a)>0 else 0.4
    lrg = float(a[1]) if len(a)>1 else 0.5
    sml_square = (abs(x) > sml) | (abs(y) > sml) 
    large_square = (abs(x) < lrg) & (abs(y) < lrg) 
    mask = large_square & sml_square
    return np.where(mask,H,0)

def anulus(H,a):
    y = (norm(row_mat) - 0.5) * 2
    x = (norm(col_mat) - 0.5) * 2
    sml = float(a[0]) if len(a)>0 else 0.4
    lrg = float(a[1]) if len(a)>1 else 0.5
    sml_circle = x*x+y*y > sml*sml
    large_circle = x*x+y*y < lrg*lrg
    mask = large_circle & sml_circle
    return np.where(mask,H,0)

def rescale(H,a):
    h0 = float(a[0]) if len(a)>0 else 0
    h1 = float(a[1]) if len(a)>1 else 1
    max_H = np.max(H)
    min_H = np.min(H)
    if max_x-min_x>0:
        nH = ((H-min_H)/(max_H-min_H)).astype(np.float32)   
        cH = nH*(h1-h0)+h0
        return cH
    return H.astype(np.float32)

pfrm_clip = {
    "rescale": lambda x,a: rescale(x,a),
    "cci": lambda x,a: in_circle(x,a),
    "cco": lambda x,a: out_circle(x,a),
    "csi": lambda x,a: in_square(x,a),
    "cso": lambda x,a: out_square(x,a),
    "css": lambda x,a: square_strip(x,a),
    "ccs": lambda x,a: anulus(x,a),
    "above": lambda x,a: above(x,a),
    "below": lambda x,a: below(x,a),
}

#######################################
#
# stack
#
#######################################
stack = []


stack_pop_funs = {
    "add": lambda x: x+stack.pop(),
    "avg": lambda x: (x+stack.pop())/2,
    "add.rank": lambda x: rank(np.abs(x+stack.pop()),[]),
    "max": lambda x: np.maximum(x,stack.pop()),
    "min": lambda x: np.minimum(x,stack.pop()),
    "sub": lambda x: x-stack.pop(),
    "mul": lambda x: x*stack.pop(),
    "mul.rank": lambda x: rank(np.abs(x*stack.pop()),[]),
    "div": lambda x: ratio(x,stack.pop()),
    "div.rank": lambda x: rank(np.abs(ratio(x,stack.pop())),[]),
    "swp": lambda x: (stack.pop(),stack.append(x))[0],
}

def push_op(x,a):
    stack.append(x)
    return x

def pop_op(x,a):
    if len(a)<1:
        return stack.pop()
    sname = a[0]
    return stack_pop_funs[sname](x) 

pfrm_stack = {
    "push": lambda x,a: push_op(x,a),
    "pop": lambda x,a: pop_op(x,a),
}

#######################################
#
# root group functions
#
#######################################


def principal_axis_lengths(x):
    pts = np.column_stack((x.real, x.imag)).astype(float)
    pts -= pts.mean(axis=0, keepdims=True)
    cov = np.cov(pts, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    order = evals.argsort()[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    if evecs[0, 0] < 0:
        evecs *= -1
    major, minor = np.sqrt(evals)
    angle = np.arctan2(evecs[1, 0], evecs[0, 0])
    return major, minor, angle

def principal_axis_lengths1(x):
    pts = np.column_stack((x.real, x.imag))
    pts -= pts.mean(axis=0)
    cov = np.cov(pts, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)  # always sorted ascending
    lengths = np.sqrt(evals)
    minor, major = lengths
    angle = np.arctan2(evecs[1, 1], evecs[0, 1])  # angle of major axis in radians
    return major, minor, angle

def principal_major_vec(x):
    major, minor, angle = principal_axis_lengths(x)
    principal_major_vector = major * np.exp(1j * angle)
    return principal_major_vector

def principal_minor_vec(x):
    major, minor, angle = principal_axis_lengths(x)
    principal_minor_vector = minor * np.exp(1j * (angle+np.pi/2))
    return principal_minor_vector

def eccentricity(x):
    major, minor, angle = principal_axis_lengths(x)
    if abs(major)<1e-10:
        return 0
    ecc = (1-(minor/major)**2)**0.5
    return ecc

gfuns = {
    "count" : lambda x: x.shape[0],
    "centroid": lambda x: np.mean(x),
    "centroid.angle": lambda x: np.angle(np.mean(x)),
    "centroid.abs": lambda x: np.abs(np.mean(x)),
    "mean.abs": lambda x: np.mean(np.abs(x-np.mean(x))),
    "max.abs": lambda x: np.max(np.abs(x-np.mean(x))),
    "min.abs": lambda x: np.min(np.abs(x-np.mean(x))),
    "range.abs": lambda x: np.ptp(np.abs(x-np.mean(x))),
    "skew.abs": lambda x: np.mean((np.abs(x - np.mean(x)))**3),
    "kurt.abs": lambda x: np.mean((np.abs(x - np.mean(x)))**4),
    "mean.angle": lambda x: np.mean(np.angle(x-np.mean(x))),
    "max.angle": lambda x: np.max(np.angle(x-np.mean(x))),
    "min.angle": lambda x: np.min(np.angle(x-np.mean(x))),
    "range.angle": lambda x: np.ptp(np.angle(x-np.mean(x))),
    "skew.angle": lambda x: np.mean((np.angle(x - np.mean(x)))**3),
    "kurt.angle": lambda x: np.mean((np.angle(x - np.mean(x)))**4),
    "bbox": lambda x: np.ptp(x.real)*np.ptp(x.imag),
    "bbox.ar": lambda x: np.ptp(x.real) / np.ptp(x.imag) if np.ptp(x.imag) != 0 else 0,
    "cor.re.im": lambda x: np.corrcoef(x.real, x.imag)[0, 1],
    "scor.re.im": lambda x: np.corrcoef(np.sign(x.real), np.sign(x.imag))[0, 1],
    "cor.mod.phi": lambda x: np.corrcoef(np.abs(x), np.angle(x))[0, 1],
    "principal.angle": lambda x: principal_axis_lengths(x)[2],
    "principal.major.vec": lambda x: principal_major_vec(x),
    "principal.minor.vec": lambda x: principal_minor_vec(x),
    "principal.major.abs": lambda x: principal_axis_lengths(x)[0],
    "principal.minor.abs": lambda x: principal_axis_lengths(x)[1],
    "principal.angle": lambda x: principal_axis_lengths(x)[2],
    "eccentricity": lambda x: eccentricity(x),
}

def polyres_pack(x):
    if x.shape != polyres["i"].shape:
        raise ValueError(f"polyres_pack: x.shape = {x.shape}, i.shape = {polyres['i'].shape}")
    if x.shape != polyres["j"].shape:
        raise ValueError(f"polyres_pack: x.shape = {x.shape}, j.shape = {polyres['j'].shape}")
    H = np.zeros((polyres["height"],polyres["width"]),dtype=np.complex64)
    H[polyres["i"],polyres["j"]] = x 
    return H

def polyres_ungroup(x):
    if x.shape != polyres["starts"].shape:
        raise ValueError(f"polyres_ungroup: x.shape = {x.shape}, starts.shape = {polyres['starts'].shape}")
    H = np.zeros((polyres["height"],polyres["width"]),dtype=np.complex64)
    H[polyres["i"],polyres["j"]] = x[polyres['gbi']] 
    return H

def polyres_group(x,a):
    if x.shape != polyres["gbi"].shape:
        raise ValueError(f"polyres_group: x.shape = {x.shape}, gbi.shape = {polyres['gbi'].shape}")
    if len(a)<1:
        return polys.polyutil.group_apply(x,polyres["starts"],np.mean) 
    fname = a[0]
    return polys.polyutil.group_apply(x,polyres["starts"],gfuns[fname]) 

pfrm_cast = {
    "pack" : lambda x,a: polyres_pack(x),
    "ungroup": lambda x,a: polyres_ungroup(x),
    "group": lambda x,a: polyres_group(x,a),
}

#######################################
#
# simple functions
#
#######################################

pfrm_fun = {
    "eval": lambda x,a: ne.evaluate(a[0],{"x":x.real,"y":x.imag}),
    "phase": lambda x,a: farg(a,0,1) * (np.angle(x)/np.pi+1.0) /2,
    "angle": lambda x,a: farg(a,0,1) * (np.angle(x)/np.pi+1.0) /2,
    "abs": lambda x,a: np.abs(x),
    "re": lambda x,a: x.real,
    "im": lambda x,a: x.imag,
    "re+im": lambda x,a: x.imag+x.real,
    "re-im": lambda x,a: x.imag-x.real,
    "re*im": lambda x,a: x.imag*x.real,
    "re/im": lambda x,a: x.imag/x.real,
    "sin": lambda x,a: np.sin(x),
    "cos": lambda x,a: np.cos(x),
    "exp": lambda x,a: np.exp(1j*2*np.pi*x),
    "log": lambda x,a: np.log(x),
    "1-x": lambda x,a: 1-x,
    "1j*x": lambda x,a: 1j*x,
    "trans": lambda x,a: np.transpose(x),
    "uc": lambda x,a: x.real*np.exp(1j*2*np.pi*x.imag),
    "rth": lambda x,a: x.imag*np.exp(1j*2*np.pi*x.real),
    "pow": lambda x,a: x**farg(a,0,2.0),
    "times": lambda x,a:  x*farg(a,0,1.0),
    "plus": lambda x,a:  x+farg(a,0,1.0),
}

#######################################
#
# debug
#
#######################################

dfuns = {
    "cnt" : lambda x: x.shape[0],
    "rng" : lambda x: np.ptp(np.abs(x)),
    "min" : lambda x: np.min(np.abs(x)),
    "max" : lambda x: np.max(np.abs(x)),
    "avg" : lambda x: np.mean(np.abs(x)),
    "med" : lambda x: np.median(np.abs(x)),
    "q05" : lambda x: np.quantile(np.abs(x),0.05),
    "q25" : lambda x: np.quantile(np.abs(x),0.25),
    "q75" : lambda x: np.quantile(np.abs(x),0.75),
    "q95" : lambda x: np.quantile(np.abs(x),0.95),
    "shp" : lambda x: x.shape,
    "typ" : lambda x: x.dtype,
    "nz"  : lambda x: np.sum(np.abs(x)>0),
    "fnt" : lambda x: np.sum(np.isfinite(x)>0),
    "z.shp" : lambda x: polyres['z'].shape,
    "z.typ" : lambda x: polyres['z'].dtype,
    "z.rng" : lambda x: np.ptp(polyres['z']),
    "z.max" : lambda x: np.max(polyres['z']),
    "r.shp" : lambda x: polyres['r'].shape,
    "r.typ" : lambda x: polyres['r'].dtype,
    "r.rng" : lambda x: np.ptp(polyres['r']),
    "r.max" : lambda x: np.max(polyres['r']),
    "i.shp" : lambda x: polyres['i'].shape,
    "i.typ" : lambda x: polyres['i'].dtype,
    "i.rng" : lambda x: np.ptp(polyres['i']),
    "i.max" : lambda x: np.max(polyres['i']),
    "j.shp" : lambda x: polyres['j'].shape,
    "j.typ" : lambda x: polyres['j'].dtype,
    "j.rng" : lambda x: np.ptp(polyres['j']),
    "j.max" : lambda x: np.max(polyres['j']),
    "x.shp" : lambda x: polyres['x'].shape,
    "x.typ" : lambda x: polyres['x'].dtype,
    "x.rng" : lambda x: np.ptp(polyres['x']),
    "x.max" : lambda x: np.max(polyres['x']),
    "y.shp" : lambda x: polyres['y'].shape,
    "y.typ" : lambda x: polyres['y'].dtype,
    "y.rng" : lambda x: np.ptp(polyres['y']),
    "y.max" : lambda x: np.max(polyres['y']),
}

def debug(x,a):
    if len(a)<1:
        print(f"mean: {np.mean(x)}")
        return polys.polyutil.group_apply(x,polyres["starts"],np.mean) 
    for i in range(len(a)):
        dname = a[i]
        print(f"{dname} : {dfuns[dname](x)}")
    return x

pfrm_debug = {
    "dbg" : lambda x,a : debug(x,a),
}


#######################################
#
# legacy
#
#######################################

pfrm_other = {
    "none": lambda x,a: x,
    "has.result" : lambda x,a: polyres["haves"],
    "fill" : lambda x,a: polyres_fill_havenots(x,a),
    "get": lambda x,a: np.full_like(x,params[a[0]] if len(a)>0 else 0),
    "centroid": lambda x,a: polyres_centroid(),
    "direction": lambda x,a: direction(x,a),
    "blurr": lambda x,a: blurr(x,a),
    "variance": lambda x,a: polyres_variance(),
    "centroid.dist.abs": lambda x,a: polyres_centroid_dist_abs(),
    "centroid.dist.abs.max": lambda x,a: polyres_centroid_dist_abs_max(),
    "centroid.dist.abs.min": lambda x,a: polyres_centroid_dist_abs_min(),
    "centroid.dist.angle": lambda x,a: polyres_centroid_dist_angle(),
    "centroid.dist.angle.max": lambda x,a: polyres_centroid_dist_angle_max(),
    "centroid.dist.angle.min": lambda x,a: polyres_centroid_dist_angle_min(),
    "centroid.dist.angle.range": lambda x,a: polyres_centroid_dist_angle_range(),
    "count": lambda x,a: polyres_count(),
    "mean": lambda x,a: polyres_mean(),
    "amean": lambda x,a: polyres_amean(),
    "max": lambda x,a: polyres_max(),
    "amax": lambda x,a: polyres_amax(),
    "min": lambda x,a: polyres_min(),
    "amin": lambda x,a: polyres_amin(),
    "range": lambda x,a: polyres_range(),
    "arange": lambda x,a: polyres_arange(),
    "discriminant": lambda x,a: polyres_discriminant(a),
    "mc": lambda x,a: multiply_and_clip_to_one(x,a),
    "norm": lambda x,a: norm_scale(x,a),
    "norm.clip": lambda x,a: norm_clip(x,a),
    "heq": lambda x,a: heq(x,a),
    "heq.haves": lambda x,a: heq_haves(x,a),
    "rank": lambda x,a: rank(x,a),
    "vrank": lambda x,a: norm(rankdata(x,method='average')),
    "rank.haves": lambda x,a: rank_haves(x,a),
    "map": lambda x,a: map(x,a),
    
    
    "bkr": lambda x,a: bkr(x,a),
    "ebkr": lambda x,a: ebkr(x),
   
    
    
    "nz": lambda x,a: nz(x,a),
    "equals": lambda x,a: equals(x,a),
    "greater": lambda x,a: greater(x,a),
    "wf": lambda x,a: wavefield(x,a),
    "wf3": lambda x,a: wavefield3(x,a),
    "nglf": lambda x,a: anglefield(x,a),
    "absf": lambda x,a: absfield(x,a),
   
    "roll": lambda x,a: xshiftdiff(x),

}
  
#######################################
#
# pipeline parser
#
#######################################

pfrm_functions = {}
pfrm_functions.update(pfrm_constants)
pfrm_functions.update(pfrm_inputs)
pfrm_functions.update(pfrm_transforms_conformal)
pfrm_functions.update(pfrm_transforms_poly)
pfrm_functions.update(pfrm_clip)
pfrm_functions.update(pfrm_stack)
pfrm_functions.update(pfrm_cast)
pfrm_functions.update(pfrm_fun)
pfrm_functions.update(pfrm_debug)
pfrm_functions.update(pfrm_other)



def pfrm(H,pipeline):
    tH = polys.polyutil.make_pipeline(pipeline,pfrm_functions)(H)
    return tH


#######################################
#
# RGB conversion
#
#######################################

def hsv_to_rgb_numpy(HSV):
    hsv=np.abs(HSV) # no complex values
    # hsv: (..., 3) array, H in [0,1], S in [0,1], V in [0,1]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6).astype(int)
    f = (h * 6) - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i = i % 6
    conditions = [i == k for k in range(6)]
    rgb = np.zeros(hsv.shape)
    rgb[conditions[0]] = np.stack([v, t, p], axis=-1)[conditions[0]]
    rgb[conditions[1]] = np.stack([q, v, p], axis=-1)[conditions[1]]
    rgb[conditions[2]] = np.stack([p, v, t], axis=-1)[conditions[2]]
    rgb[conditions[3]] = np.stack([p, q, v], axis=-1)[conditions[3]]
    rgb[conditions[4]] = np.stack([t, p, v], axis=-1)[conditions[4]]
    rgb[conditions[5]] = np.stack([v, p, q], axis=-1)[conditions[5]]
    return rgb

def save2rgb(hsv,fn):
    rgb_uint8 = (255*hsv_to_rgb_numpy(hsv)).astype(np.uint8)    
    im = pyvips.Image.new_from_memory(
        rgb_uint8.tobytes(),
        rgb_uint8.shape[1],  # width
        rgb_uint8.shape[0],  # height
        rgb_uint8.shape[2],  # bands (should be 3 for RGB)
        'uchar'
    )            
    im.write_to_file(fn)

#######################################
#
# CLI
#
#######################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="palette maker")
    parser.add_argument('outfile',nargs="?",type=str, default="palette.png", help="outfile")
    parser.add_argument('--verbose',action='store_true',help="verbose")
    parser.add_argument('-r','--res', type=int, default=1000, help="resolution")
    parser.add_argument('-d','--data', type=float, default=10, help="size")
    parser.add_argument('-s','--sort_uniq', action="store_true", help="sort_unique")
    parser.add_argument('-m','--mlx', action="store_true", help="use mlx")
    parser.add_argument('-p','--par', type=str, default=None, help="parameters")
    parser.add_argument('-H','--hfrm', type=str, default="clip", help="hue functions")
    parser.add_argument('-S','--sfrm', type=str, default="none", help="sat functions")
    parser.add_argument('-V','--vfrm', type=str, default="none", help="val functions")

    args = parser.parse_args()    

    hfrm_pipeline = args.hfrm
    sfrm_pipeline = args.sfrm
    vfrm_pipeline = args.vfrm
    #
    res = args.res
    row_mat, col_mat = np.indices(((args.res,args.res)))
    H = np.zeros((res,res),dtype=np.float32)
    S = np.ones((res,res),dtype=np.float32) 
    V = np.ones((res,res),dtype=np.float32) 
    #
    results_pct = args.data
    sort_unique = args.sort_uniq
    use_mlx = args.mlx
    load_polyres()

    params.update(polys.polystate.pu.setf(args.par))
   
    H=pfrm(H,hfrm_pipeline)
    S=pfrm(S,sfrm_pipeline)
    V=pfrm(V,vfrm_pipeline)

    save2rgb(np.stack([H, S, V], axis=-1),args.outfile)

