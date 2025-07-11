#!/usr/bin/env python

import numpy as np
import pyvips
import argparse
import functools
import numexpr as ne
import polys
import polys.polystate
import polys.polyutil
import cv2

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
    cdf =  (hi-lo) * cdf / cdf[-1] + lo # normalize to [lo, hi]
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

def times(H,a):
    fac = float(a[0]) if len(a)>0 else 1.0
    return H*fac

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
    return x*stack.pop()

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

def ratio(x,y):
    num = np.where(abs(y)>1e-10,x,0)
    den = np.where(abs(y)>1e-10,y,1)
    return num/den

hfrm_pipeline = "clip"
sfrm_pipeline = "one"
vfrm_pipeline = "one"
res = 1000
row_mat, col_mat = np.indices(((res,res)))
H = np.zeros((res,res),dtype=np.float32)
S = np.ones((res,res),dtype=np.float32) 
V = np.ones((res,res),dtype=np.float32) 

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
    "value":       lambda x,a: np.full_like(V,ne.evaluate(a[0]) if len(a)>0 else 0.0),
}

pfrm_inputs = {
    "rows": lambda x,a: row_mat,
    "cols": lambda x,a: col_mat,
    "idx": lambda x,a: col_mat + 1j * row_mat,
    "j": lambda x,a: 2*(norm(col_mat)-0.5),
    "i": lambda x,a: 2*(norm(row_mat)-0.5),
    "z": lambda x,a: 2*(norm(col_mat)-0.5)+1j*2*(norm(row_mat)-0.5),
    "r": lambda x,a: polyres["r"],
    "roots": lambda x,a: polyres["r"],
}

pfrm_transforms_conformal = {
    "wbl": lambda x,a: ratio(x + 1j*np.sin(x),x + 1j*np.cos(x)),
    "tws": lambda x,a: ratio( np.exp(x) - 1, np.exp(x) * 1j + 1),
    "plk": lambda x,a: ratio(np.sin(1j+x),np.cos(1j+x)),
    "zzg": lambda x,a: ratio(x + 2,x + 2),
    "ltl": lambda x,a: ratio(x - 2j,x + 2j),
    "kth": lambda x,a: ratio(1 + x,1j - x),
    "jkw": lambda x,a: x+ratio(1,x),
}

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

pfrm_clip = {
    "clip": lambda x,a: hue_clip(x,a),
    "cci": lambda x,a: in_circle(x,a),
    "cco": lambda x,a: out_circle(x,a),
    "csi": lambda x,a: in_square(x,a),
    "cso": lambda x,a: out_square(x,a),
    "css": lambda x,a: square_strip(x,a),
    "ccs": lambda x,a: anulus(x,a),
    "above": lambda x,a: above(x,a),
    "below": lambda x,a: below(x,a),
}

pfrm_stack = {
    "push": lambda x,a: push(x),
    "pushv": lambda x,a: push(np.full_like(x,params[a[0]])),
    "pop": lambda x,a: pop(),
    "add": lambda x,a: add(x),
    "subtract": lambda x,a: subtract(x),
    "mult": lambda x,a: mult(x),
}

#######################################
#
# grouping
#
#######################################

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

def principal_axis_lengths(x):
    pts = np.column_stack((x.real, x.imag))
    pts -= pts.mean(axis=0)
    cov = np.cov(pts, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)  # always sorted ascending
    minor, major = np.sqrt(evals)
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
    ecc = (1-ratio(minor,major)**2)**0.5
    return ecc

gfuns = {
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
    "principal.angle": lambda x: principal_axis_lengths(x)[2],
    "principal.major.vec": lambda x: principal_major_vec(x),
    "principal.minor.vec": lambda x: principal_minor_vec(x),
    "principal.major.abs": lambda x: principal_axis_lengths(x)[0],
    "principal.minor.abs": lambda x: principal_axis_lengths(x)[1],
    "principal.angle": lambda x: principal_axis_lengths(x)[2],
    "eccentricity": lambda x: eccentricity(x),
}

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
    "pow": lambda x,a: x**(float(a[0]) if len(a)>0 else 2.0),
    "times": lambda x,a:  x*float(a[0]) if len(a)>0 else 1.0,
    "plus": lambda x,a:  x+float(a[0]) if len(a)>0 else 1.0,
}

#######################################
#
# debug
#
#######################################

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

pfrm_debug = {
    "showstats" : lambda x,a : showstats(x,a),
}

#######################################
#
#
#
#######################################

pfrm_other = {
    "none": lambda x,a: x,
    "present" : lambda x,a: polyres["has_result"],
    
   
    
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
    "times": lambda x,a: times(x,a),
    "norm": lambda x,a: norm_scale(x,a),
    "heq": lambda x,a: heq(x,a),
    "rank": lambda x,a: rank(x,a),
    "map": lambda x,a: map(x,a),
    
    
    
    
    "bkr": lambda x,a: bkr(x),
    "ebkr": lambda x,a: ebkr(x),
   
    "phase": lambda x,a: arg(a,0,"afr")*(np.angle(x)/np.pi+1.0)/2,
    "angle": lambda x,a: arg(a,0,"afr")*(np.angle(x)/np.pi+1.0)/2,
    
    "nz": lambda x,a: nz(x,a),
    "wf": lambda x,a: wavefield(x,a),
    "wf3": lambda x,a: wavefield3(x,a),
    "nglf": lambda x,a: anglefield(x,a),
    "absf": lambda x,a: absfield(x,a),
   
    "roll": lambda x,a: xshiftdiff(x),
   
    "H" : lambda x,a : H,
    "S" : lambda x,a : S,
    "V" : lambda x,a : V,
}
  
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

