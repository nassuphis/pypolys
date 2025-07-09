#!/usr/bin/env python

import numpy as np
import pyvips
import argparse
import functools
import numexpr as ne
import polys
import polys.polystate
import polys.polyutil
from itertools import combinations

type_choices = [
        "h",
        "hs",
        "abs",
        "ngl",
        "trigh",
        "trighs",
    ]

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

def load_polyres():
    results_z = np.load('myresult.npz')
    results = results_z['arr_0']

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

    print(f"{height}->{np.max(pi)+1}")

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
    H[polyres["i"],polyres["j"]]=1.0
    polyres["has_result"] = H
    print(f"results: {results.shape}")
    print(f"results: x type {polyres["x"].dtype} y type {polyres["y"].dtype}")
    return

def polyres_roots():
    return polyres["r"]

def polyres_broadcast(x):
    H = np.zeros((polyres["height"],polyres["width"]),dtype=np.complex64)
    H[polyres["i"],polyres["j"]] = x[polyres['gbi']] 
    return H


def polyres_centroid():
    H = np.zeros((polyres["height"],polyres["width"]),dtype=np.complex64)      
    centroid=polys.polyutil.group_apply(polyres["r"],polyres["starts"],np.mean)       
    H[polyres["i"],polyres["j"]] = centroid[polyres['gbi']]  
    return H

def polyres_centroid_dist():
    H = np.zeros((polyres["height"],polyres["width"]),dtype=np.complex64)      
    centroid=polys.polyutil.group_apply(polyres["r"],polyres["starts"],np.mean)       
    H[polyres["i"],polyres["j"]] = polyres["r"] - centroid[polyres['gbi']]  
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
        
def norm_scale(x,a):
    h0 = ne.evaluate(a[0]) if len(a)>0 else 0
    h1 = ne.evaluate(a[1]) if len(a)>1 else 1
    max_x = np.max(x)
    min_x = np.min(x)
    if max_x-min_x>0:
        nx = ((x-min_x)/(max_x-min_x)).astype(np.float32)
        snx = nx*(h1-h0)+h0
        return snx
    else:
        return x.astype(np.float32)

def heq(H,a):
    lo = ne.evaluate(a[0]) if len(a)>0 else 0.0
    hi = ne.evaluate(a[1]) if len(a)>1 else 1.0
    bc = ne.evaluate(a[2]) if len(a)>2 else 256
    hist, bins = np.histogram(H.flatten(), bins=int(bc))
    cdf = hist.cumsum()
    cdf = (hi-lo)*cdf / cdf[-1]  + lo # normalize to [lo, hi]
    img_eq = np.interp(H.flatten(), bins[:-1], cdf)
    return img_eq.reshape(H.shape)

def nz(H,a):
    tval = ne.evaluate(a[0]) if len(a)>0 else 1.0
    fval = ne.evaluate(a[1]) if len(a)>1 else 0.0
    return np.where(np.abs(H)>0,tval,fval)
          
def rank(H,a):
    lo = ne.evaluate(a[0]) if len(a)>0 else 0.0
    hi = ne.evaluate(a[1]) if len(a)>1 else 1.0
    bc = ne.evaluate(a[2]) if len(a)>2 else 256
    hist, bins = np.histogram(H.flatten(), bins=int(bc))
    cdf = np.cumsum(hist>0)
    cdf =  (hi-lo) * cdf / cdf[-1] + lo # normalize to [0, 1]
    img_eq = np.interp(H.flatten(), bins[:-1], cdf)
    return img_eq.reshape(H.shape)


def hue_clip(H,a):
    h0 = ne.evaluate(a[0]) if len(a)>0 else params["h0"]
    h1 = ne.evaluate(a[1]) if len(a)>1 else params["h1"]
    cH = norm(H)*(h1-h0)+h0
    return cH

def in_circle(H,a):
    y = (norm(H.imag) - 0.5)*2
    x = (norm(H.real) - 0.5)*2
    xc = ne.evaluate(a[0]) if len(a)>0 else params["xc"]
    yc = ne.evaluate(a[1]) if len(a)>1 else params["yc"]
    r = ne.evaluate(a[2]) if len(a)>2 else params["r"]
    cc = (x-xc)**2+(y-yc)**2 - r
    mask =  cc > 0
    return np.where(mask,1,0)

def out_circle(H,a):
    y = (norm(H.imag) - 0.5)*2
    x = (norm(H.real) - 0.5)*2
    xc = ne.evaluate(a[0]) if len(a)>0 else params["xc"]
    yc = ne.evaluate(a[1]) if len(a)>1 else params["yc"]
    r = ne.evaluate(a[2]) if len(a)>2 else params["r"]
    cc = (x-xc)**2+(y-yc)**2 - r
    mask =  cc < 0
    return np.where(mask,H,0)

def in_square(H,a):
    y = (norm(H.imag) - 0.5)*2
    x = (norm(H.real) - 0.5)*2
    xc = ne.evaluate(a[0]) if len(a)>0 else params["xc"]
    yc = ne.evaluate(a[1]) if len(a)>1 else params["yc"]
    r = ne.evaluate(a[2]) if len(a)>2 else params["r"]
    xm = abs(x-xc) < r
    ym = abs(y-yc) < r
    mask =  ym & xm
    return np.where(mask,H,0)

def out_square(H,a):
    y = (norm(row_mat) - 0.5)*2
    x = (norm(col_mat) - 0.5)*2
    xc = ne.evaluate(a[0]) if len(a)>0 else params["xc"]
    yc = ne.evaluate(a[1]) if len(a)>1 else params["yc"]
    r = ne.evaluate(a[2]) if len(a)>2 else params["r"]
    xm = abs(x-xc) > r
    ym = abs(y-yc) > r
    mask =  ym | xm
    return np.where(mask,H,0)


def square_strip(H,a):
    y = (norm(row_mat) - 0.5) * 2
    x = (norm(col_mat) - 0.5) * 2
    sml = ne.evaluate(a[0]) if len(a)>0 else 0.4
    lrg = ne.evaluate(a[1]) if len(a)>1 else 0.5
    sml_square = (abs(x) > sml) | (abs(y) > sml) 
    large_square = (abs(x) < lrg) & (abs(y) < lrg) 
    mask = large_square & sml_square
    return np.where(mask,H,0)

def anulus(H,a):
    y = (norm(row_mat) - 0.5) * 2
    x = (norm(col_mat) - 0.5) * 2
    sml = ne.evaluate(a[0]) if len(a)>0 else 0.4
    lrg = ne.evaluate(a[1]) if len(a)>1 else 0.5
    sml_circle = x*x+y*y > sml*sml
    large_circle = x*x+y*y < lrg*lrg
    mask = large_circle & sml_circle
    return np.where(mask,H,0)

def above(H,a):
    x = (norm(col_mat) - 0.5)
    xc = ne.evaluate(a[0]) if len(a)>0 else params["xc"]
    mask = x > xc
    return np.where(mask,H,0)

def below(H,a):
    x = (norm(col_mat) - 0.5)
    xc = ne.evaluate(a[0]) if len(a)>0 else params["xc"]
    mask = x < xc
    return np.where(mask,H,0)

def left(H,a):
    y = (norm(row_mat) - 0.5)
    yc = ne.evaluate(a[0]) if len(a)>0 else params["yc"]
    mask = y < yc
    return np.where(mask,H,0)

def right(H,a):
    y = (norm(row_mat) - 0.5)
    yc = ne.evaluate(a[0]) if len(a)>0 else params["yc"]
    mask = y > yc
    return np.where(mask,H,0)

def wavefield(H,a):
    x = 2 * np.pi * norm(H.real) * arg(a,1,"cfr")
    y = 2 * np.pi * norm(H.imag) * arg(a,0,"rfr")
    e = a[2] if len(a)>2 else "cos(x)+1j*sin(y)"
    v = ne.evaluate(e,{"x":x,"y":y})
    return v

def wavefield3(H,a):
    afr = ne.evaluate(a[0]) if len(a)>0 else params["afr"]
    r = np.hypot(norm(H.real) - 0.5, norm(H.imag) - 0.5)
    v = np.sin(2 * np.pi * afr * r)
    return v

def anglefield(H,a):
    x = (norm(H.real) - 0.5) * 2
    y = (norm(H.imag) - 0.5) * 2
    afr = ne.evaluate(a[0]) if len(a)>0 else params["afr"]
    e = a[1] if len(a)>1 else "x+1j*y"
    v = ne.evaluate(e,{"x":x,"y":y})
    return ( afr * norm(v) ) % 1

def absfield(H,a):
    afr = ne.evaluate(a[0]) if len(a)>0 else params["afr"]
    v0 = np.angle(row_mat + 1j * col_mat)
    return afr*norm(v0)

def xshiftdiff(H):
    return np.roll(H,shift=1,axis=1)-H


stack = []
def push(matrix):
    stack.append(matrix)
    return matrix

def pop():
    return stack.pop()

def add(x):
    return x+stack.pop()

def subtract(x):
    return x-stack.pop()

def mult(x):
    return x+stack.pop()

def multiply_and_clip_to_one(H,a):
    afr = ne.evaluate(a[0]) if len(a)>0 else params["afr"]
    return (afr*H)%1

def map(H,a):
    v = np.array([ne.evaluate(expr).item() for expr in a], dtype=np.float32)
    i = norm(H) * (len(v) - 1)
    indices = np.clip((i).astype(int), 0, len(v) - 1)
    mv = v[indices]
    pH = np.zeros(H.shape)
    pH[:] = mv
    return pH

def bkr(t):
    x = np.real(t)
    y = np.imag(t)
    x_fold = x % 1  # fractional part of x
    y_fold = y % 1  # fractional part of y
    x_new = (2 * x_fold) % 1
    shift = np.floor(2 * x_fold)
    y_new = (y_fold + shift) / 2
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

def coeff6(z):
  t1 = z.real
  t2 = z.imag
  num1 = t1**3 + 1j
  den1 = t1**3 - 1j
  val1 = num1 / den1
  num2 = t2**3 + 1j
  den2 = t2**3 - 1j
  val2 = num2 / den2
  return val1 + 1j *val2 

def coeff7(z):
  t1 = z.real
  t2 = z.imag
  top1  = t1 + np.sin(t1)
  bot1  = t1 + np.cos(t1)
  val1  = top1 / bot1
  top2  = t2 + np.sin(t2)
  bot2  = t2 + np.cos(t2)
  val2  = top2 / bot2
  return val1 + 1j * val2
    
def coeff8(z): 
  t1 = z.real
  t2 = z.imag
  top1  = t1 + np.sin(t2)
  bot1  = t2 + np.cos(t1)
  val1  = top1 / bot1
  top2  = t2 + np.sin(t1)
  bot2  = t1 + np.cos(t2)
  val2  = top2 / bot2
  return val1 + 1j * val2

def coeff9(z):
    t1 = z.real
    t2 = z.imag
    top1  = t1*t1 + 1j * t2
    bot1  = t1*t1 - 1j * t2
    val1  = top1 / bot1
    top2  = t2*t2 + 1j * t1
    bot2  = t2*t2 - 1j * t1
    val2  = top2 / bot2
    return val1 + 1j * val2

def coeff10(z):
    t1 = z.real
    t2 = z.imag
    top1  = t1**4 - t2
    bot1  = t1**4 + t2
    val1  = top1 / bot1
    top2  = t2**4 - t1
    bot2  = t2**4 + t1
    val2  = top2 / bot2
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

def arg(a,i,v):
    return ne.evaluate(a[i]) if len(a)>i else params[v]

def showstats(x, a):
    print(f"result x,y shape : {polyres['x'].shape}, {polyres['y'].shape}")
    print(f"result i,j shape : {polyres['i'].shape}, {polyres['j'].shape}")
    print(f"result z,r shape : {polyres['z'].shape}, {polyres['r'].shape}")
    print(f"result r dtype {polyres['r'].dtype}")
    print(f"result r min {np.min(np.abs(polyres['r']))} - max {np.max(np.abs(polyres['r']))}")
    print(f"result z min {np.min(np.abs(polyres['z']))} - max {np.max(np.abs(polyres['z']))}")
    print(f"result i min {np.min(np.abs(polyres['i']))} - max {np.max(np.abs(polyres['i']))}")
    print(f"result j min {np.min(np.abs(polyres['j']))} - max {np.max(np.abs(polyres['j']))}")
    print(f"result x min {np.min(np.abs(polyres['x']))} - max {np.max(np.abs(polyres['x']))}")
    print(f"result y min {np.min(np.abs(polyres['y']))} - max {np.max(np.abs(polyres['y']))}")
    print(f"input shape: {x.shape}")
    print(f"input dtype: {x.dtype}")
    print(f"input finite: {np.sum(np.isfinite(x))} ({np.round(100 * np.isfinite(x).sum() / x.size,2)}%)")
    print(f"input nz: {np.sum(np.abs(x)>0)}  ({np.round(100 * np.sum(abs(x)>0).sum() / x.size,2)}%)")
    print(f"input: min {np.min(x)} - max: {np.max(x)}")
    print(f"input q01: {np.quantile(x,0.01)}")
    print(f"input q25: {np.quantile(x,0.25)}")
    print(f"input q50: {np.quantile(x,0.50)}")
    print(f"input q75: {np.quantile(x,0.75)}")
    print(f"input q99: {np.quantile(x,0.99)}")
    nz = x[np.abs(x) > 0]
    if nz.size == 0:
        print("No non-zero values.")
    else:
        print(f"{np.min(nz)} - {np.median(nz)} - {np.max(nz)}")
    print(f"{np.sum(np.abs(x) > 0)} / {x.shape[0] * x.shape[1]}")
    return x

hfrm_pipeline = "clip"
sfrm_pipeline = "one"
vfrm_pipeline = "one"
res = 1000
row_mat, col_mat = np.indices(((res,res)))
H = np.zeros((res,res),dtype=np.float32)
S = np.ones((res,res),dtype=np.float32) 
V = np.ones((res,res),dtype=np.float32) 
pfrm_functions = {
    "none": lambda x,a: x,
    "present" : lambda x,a: polyres["has_result"],
    "zero": lambda x,a: np.zeros(x.shape),
    "one": lambda x,a: np.ones(x.shape),
    "value": lambda x,a: np.full_like(V,ne.evaluate(a[0]) if len(a)>0 else 1.0),
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
    "get": lambda x,a: np.full_like(x,params[a[0]] if len(a)>0 else 0),
    "rows": lambda x,a: row_mat,
    "cols": lambda x,a: col_mat,
    "idx": lambda x,a: col_mat + 1j * row_mat,
    "x": lambda x,a: 2*(norm(col_mat)-0.5),
    "y": lambda x,a: 2*(norm(row_mat)-0.5),
    "z": lambda x,a: 2*(norm(col_mat)-0.5)+1j*2*(norm(row_mat)-0.5),
    "centroid": lambda x,a: polyres_centroid(),
    "variance": lambda x,a: polyres_variance(),
    "centroid.dist": lambda x,a: polyres_centroid_dist(),
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
    "heq": lambda x,a: heq(x,a),
    "rank": lambda x,a: rank(x,a),
    "map": lambda x,a: map(x,a),
    "trans": lambda x,a: np.transpose(x),
    "uc": lambda x,a: x.real*np.exp(1j*2*np.pi*x.imag),
    "rth": lambda x,a: x.imag*np.exp(1j*2*np.pi*x.real),
    "pow": lambda x,a: x**(ne.evaluate(a[0]) if len(a)>0 else params["exp"]),
    "sin": lambda x,a: np.sin(x),
    "cos": lambda x,a: np.cos(x),
    "exp": lambda x,a: np.exp(1j*2*np.pi*x),
    "log": lambda x,a: np.log(x),
    "wbl": lambda x,a: (x + np.sin(x)) / (x + np.cos(x)),
    "tws": lambda x,a: (np.exp(x) - 1) / (np.exp(x) + 1),
    "plk": lambda x,a: np.sin(x) / np.cos(x),
    "zzg": lambda x,a: (x + 2) / (x - 2),
    "ltl": lambda x,a: (x - 2j) / (x + 2j),
    "kth": lambda x,a: (1 + x) / (1 - x),
    "jkw": lambda x,a: x + 1/x,
    "bkr": lambda x,a: bkr(x),
    "ebkr": lambda x,a: ebkr(x),
    "coeff2": lambda x,a: (x.real + x.imag) + 1j * ( x.imag * x.real),
    "coeff3": lambda x,a: 1/(x.real+arg(a,0,"xc"))+1j/((1/(x.imag+arg(a,1,"yc")))),
    "coeff5": lambda x,a: (x.real + (1/x.imag)) + 1j * ( x.imag + (1/x.real)),
    "coeff5a": lambda x,a: (x.real + (1/x.real)) + 1j * ( x.imag + (1/x.imag)),
    "coeff6": lambda x,a: coeff6(x),
    "coeff7": lambda x,a: coeff7(x),
    "coeff8": lambda x,a: coeff8(x),
    "coeff9": lambda x,a: coeff9(x),
    "coeff10": lambda x,a: coeff10(x),
    "coeff11": lambda x,a: coeff11(x),
    "coeff12": lambda x,a: coeff12(x),
    "phase": lambda x,a: (np.angle(x)*arg(a,0,"afr")/np.pi+1.0)/2,
    "angle": lambda x,a: (arg(a,0,"afr")*np.angle(x)/np.pi+1.0)/2,
    "abs": lambda x,a: np.abs(x),
    "re": lambda x,a: x.real,
    "im": lambda x,a: x.imag,
    "re+im": lambda x,a: x.imag+x.real,
    "re-im": lambda x,a: x.imag-x.real,
    "re*im": lambda x,a: x.imag*x.real,
    "re/im": lambda x,a: x.imag/x.real,
    "eval": lambda z,a: ne.evaluate(a[0],{"x":z.real,"y":z.imag}),
    "sum": lambda x,a: x.imag+x.real,
    "clip": lambda x,a: hue_clip(x,a),
    "cci": lambda x,a: in_circle(x,a),
    "cco": lambda x,a: out_circle(x,a),
    "csi": lambda x,a: in_square(x,a),
    "cso": lambda x,a: out_square(x,a),
    "css": lambda x,a: square_strip(x,a),
    "ccs": lambda x,a: anulus(x,a),
    "above": lambda x,a: above(x,a),
    "below": lambda x,a: below(x,a),
    "nz": lambda x,a: nz(x,a),
    "wf": lambda x,a: wavefield(x,a),
    "wf3": lambda x,a: wavefield3(x,a),
    "nglf": lambda x,a: anglefield(x,a),
    "absf": lambda x,a: absfield(x,a),
    "inv": lambda x,a: 1-x,
    "push": lambda x,a: push(x),
    "pushv": lambda x,a: push(np.full_like(x,params[a[0]])),
    "pop": lambda x,a: pop(),
    "add": lambda x,a: add(x),
    "subtract": lambda x,a: subtract(x),
    "mult": lambda x,a: mult(x),
    "times1j": lambda x,a: 1j*x,
    "roll": lambda x,a: xshiftdiff(x),
    "showstats" : lambda x,a : showstats(x,a),
    "H" : lambda x,a : H,
    "S" : lambda x,a : S,
    "V" : lambda x,a : V,
}
  

def pfrm(H,pipeline):

    pfuns = []
    pargs = []
    if pipeline is None: 
        return H
    if pipeline.strip()=='': 
        return H
    calls = [name.strip() for name in pipeline.split(',')]

    for call in calls:
        call_parts = call.split("_")
        name = call_parts[0]
        args = call_parts[1:]
        if name not in pfrm_functions:
            raise KeyError(f"pfrm: '{name}' not in pfrm_functions")
        value = pfrm_functions[name]
        if not callable(value):
            raise ValueError(f"pfrm: pfrm_functions value '{name}' not callable")
        pfuns.append(value)
        pargs.append(args)
        print(f"{args}")
    
    tH  = functools.reduce(lambda acc, i: pfuns[i](acc,pargs[i]),range(len(pfuns)),H)
    return tH



def hsv_to_rgb_numpy(hsv):
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="palette maker")
    parser.add_argument('outfile',nargs="?",type=str, default="palette.png", help="outfile")
    parser.add_argument('--verbose',action='store_true',help="verbose")
    parser.add_argument('--type', choices=type_choices,default="none",help="mode")
    parser.add_argument('-r','--res', type=int, default=1000, help="resolution")
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
    load_polyres()

    params.update(polys.polystate.pu.setf(args.par))
   
    H=pfrm(H,hfrm_pipeline)
    S=pfrm(S,sfrm_pipeline)
    V=pfrm(V,vfrm_pipeline)

    save2rgb(np.stack([H, S, V], axis=-1),args.outfile)

