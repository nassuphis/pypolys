#!/usr/bin/env python
# ops_xfrm.py
#
# z2 = f(z1)
# z has length 2
# these are parameter
# transforms

import numpy as np
from numba.typed import Dict
from numba import njit, types
import argparse
import ast
import math

ALLOWED = {}

def spc(x, start, end, power=1.0):
    n = x.size
    out = np.empty(n, dtype=np.float64)
    xmin, xmax = x[0], x[0]
    for i in range(n):
        v = x[i]
        if v < xmin: xmin = v
        if v > xmax: xmax = v

    denom = xmax - xmin
    if denom <= 0.0:
        for i in range(n): out[i] = start
        return out

    flip = end < start      
    for i in range(n):
        s = (x[i] - xmin) / denom  
        if flip: s = 1.0 - (1.0 - s)**power
        else: s = s**power
        out[i] = start + s * (end - start)

    return out

def op_circle(z,a,state):
    n = int(a[0].real)
    if n<0: return z
    if n>z.size-1: return z
    v = z.copy()
    t = v[n]
    c = np.exp(1j*2*np.pi*t)
    v[n] = c
    return v

ALLOWED["circ"]=op_circle

def op_rhotheta(z,a,state):
    n = int(a[0].real)
    if n<0: return z
    if n>z.size-1: return z
    zz = z.copy()
    rho = z[0]
    theta = z[1]
    disk = rho* np.exp(1j*2*np.pi*theta)
    zz[n] = disk
    return zz

ALLOWED["rt"]=op_rhotheta

def op_thetarho(z,a,state):
    n = int(a[0].real)
    if n<0: return z
    if n>z.size-1: return z
    zz = z.copy()
    rho = z[1]
    theta = z[0]
    disk = rho* np.exp(1j*2*np.pi*theta)
    zz[n] = disk
    return zz

ALLOWED["tr"]=op_thetarho

def op_rttr(z,a,state):
    v1 = z[0]*np.exp(1j*2*np.pi*z[1])
    v2 = z[1]*np.exp(1j*2*np.pi*z[0])
    return np.array([v1,v2],dtype=np.complex128)

ALLOWED["rttr"]=op_rttr

def op_dot(z,a,state):
    n = int(a[0].real)
    if n<0: return z
    if n>z.size-1: return z
    v = z.copy()
    v[n] = a[1]
    return v

ALLOWED["dot"]=op_dot

def op_xim(z,a,state):
    v1 = 1j*z[0].real
    v2 = 1j*z[1].real
    return np.array([v1,v2],dtype=np.complex128)

ALLOWED["xim"]=op_xim

def op_zz(z,a,state):
    v = z[0]+z[1]*1j
    return np.array([v,v],dtype=np.complex128)

ALLOWED["zz"]=op_zz

def op_zz1(z,a,state):
    v1 = z[0]+z[1]*1j
    v2 = z[0]*z[1]+(z[0]+z[1])*1j
    return np.array([v1,v2],dtype=np.complex128)

ALLOWED["zz1"]=op_zz1

def op_zz2(z,a,state):
    v1 = z[0]+z[1]*1j
    v2 = z[0]-z[1]*1j
    return np.array([v1,v2],dtype=np.complex128)

ALLOWED["zz2"]=op_zz2

def op_zz3(z,a,state):
    v1 = z[0]+z[1]*1j
    v2 = z[1]+z[0]*1j
    return np.array([v1,v2],dtype=np.complex128)

ALLOWED["zz3"]=op_zz3

def op_pz(z,a,state):
    z0 = z[0]
    z1 = z[1]
    a0 = a[0]
    a1 = a[1]
    a2 = a[2]
    a3 = a[3]
    p0 = a0+a1*z0+a2*z0**2+a3*z0**3
    p1 = a0+a1*z1+a2*z1**2+a3*z1**3
    return np.array([p0,p1],dtype=np.complex128)

ALLOWED["pz"]=op_pz

############################################
# Baker's map (mod1 mapping)
############################################
def op_bkr(z,a,state):
  
  def bkr1(t):
    x , y = np.real(t), np.imag(t)
    x_fold, y_fold = x % 1 , y % 1  
    x_new = (2 * x_fold) % 1
    shift = np.floor(2 * x_fold)
    y_new = (y_fold + shift) / 2
    return x_new + 1j * y_new
  
  n = int(a[0].real)

  if n==0: return z
  out = z
  for i in range(z.size):
    for _ in range(n):
        out[i] = bkr1(out[i]) 

  return  out

ALLOWED["bkr"]=op_bkr

#cardioid
def op_crd(z,a,state):
    n = int(a[0].real)
    if n<0: return z
    if n>z.size-1: return z
    size = a[1].real or 1
    v = z.copy()
    t = v[n].real
    theta = 2 * np.pi * t
    r = size * (1 + np.cos(theta)) * np.exp(1j * theta)
    v[n] = r 
    return v

ALLOWED["crd"]=op_crd

#heart
def op_hrt(z,a,state):
    n = int(a[0].real)
    if n<0: return z
    if n>z.size-1: return z
    size = a[1].real or 1
    rot = np.exp(1j * 2 * np.pi * a[2].real )
    v = z.copy()
    u = v[n].real
    phi = np.pi/2
    t = 2*np.pi*u+phi
    x = 16 * np.sin(t)**3
    y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
    hrt = x/40 + 1j*y/40 + 0.1j
    v[n] = rot*size*hrt
    return v

ALLOWED["hrt"]=op_hrt

#spindle
def op_spdl(z,a,state):
    n = int(a[0].real)
    if n<0: return z
    if n>z.size-1: return z
    va = a[1].real or 0.5
    vb = a[2].real or 0.2
    vp = a[3].real or 1.5
    v = z.copy()
    t = v[n].real
    theta = 2 * np.pi * t
    x = va * np.sign(np.cos(theta)) * np.abs(np.cos(theta))**(2/vp)
    y = vb * np.sign(np.sin(theta)) * np.abs(np.sin(theta))**(2/vp)
    v[n] = x + 1j * y
    return v

ALLOWED["spdl"]=op_spdl

def op_limacon(z,a,state):
    n = int(a[0].real)
    if n<0: return z
    if n>z.size-1: return z
    v = z.copy()
    tp = v[n].real
    ap = a[1].real or 0.3
    bp = a[2].real or 0.5
    theta = 2 * np.pi * tp
    r = ap + bp * np.cos(theta)
    v[n] = r * np.exp(1j * theta)
    return v

ALLOWED["lmc"]=op_limacon

def op_rose_curve(z,a,state):
    n = int(a[0].real)
    if n<0: return z
    if n>z.size-1: return z
    v = z.copy()
    tp = v[n].real
    ap = a[1].real or 0.5
    kp = a[2].real or 2
    theta = 2 * np.pi * tp
    r = ap * np.cos(kp * theta)
    v[n] = r * np.exp(1j * theta)
    return v

ALLOWED["rsc"]=op_rose_curve

def lissajous(z,a,state):
    n = int(a[0].real)
    if n<0: return z
    if n>z.size-1: return z
    v = z.copy()
    tp = v[n].real
    Ap = a[1].real or 0.5  
    Bp = a[2].real or 0.5
    ap = a[3].real or 3
    bp = a[4].real or 2
    cp = a[5].real or 0.5
    delta = np.pi * cp
    theta = 2 * np.pi * tp
    x = Ap * np.sin(ap * theta + delta)
    y = Bp * np.sin(bp * theta)
    v[n] = x + 1j * y
    return v

ALLOWED["lss"]=lissajous

def astroid(z,a,state):
    n = int(a[0].real)
    if n<0: return z
    if n>z.size-1: return z
    v = z.copy()
    tp = v[n].real
    ap = a[1].real or 0.5
    theta = 2 * np.pi * tp
    x = ap * np.cos(theta)**3
    y = ap * np.sin(theta)**3
    v[n] = x + 1j * y
    return v

ALLOWED["ast"]=astroid

def archimedean_spiral(z,a,state):
    n = int(a[0].real)
    if n<0: return z
    if n>z.size-1: return z
    v = z.copy()
    tp = v[n].real
    ap = a[1].real or 0.1
    bp = a[2].real or 0.1
    theta = 2 * np.pi * tp
    r = ap + bp * theta
    v[n] = r * np.exp(1j * theta)
    return v

ALLOWED["asp"]=archimedean_spiral

def logarithmic_spiral(z,a,state):
    n = int(a[0].real)
    if n<0: return z
    if n>z.size-1: return z
    v = z.copy()
    tp = v[n].real
    ap = a[1].real or 0.1
    bp = a[2].real or 0.1
    theta = 2 * np.pi * tp
    r = ap * np.exp(bp * theta)
    v[n] = r * np.exp(1j * theta)
    return v

ALLOWED["lst"]=logarithmic_spiral

def deltoid(z,a,state):
    n = int(a[0].real)
    if n<0: return z
    if n>z.size-1: return z
    v = z.copy()
    tp = v[n].real
    Rp = a[1].real or 1.0
    theta = 2 * np.pi * tp
    x = Rp * (2 * np.cos(theta) + np.cos(2 * theta)) / 3
    y = Rp * (2 * np.sin(theta) - np.sin(2 * theta)) / 3
    v[n] =  x + 1j * y
    return v

# =========================
# Regular N-gon (perimeter)
# a[1]=sides (int), a[2]=radius, a[3]=rotation (turns)
# =========================
def op_poly_regular(z,a,state):
    n = int(a[0].real)
    if n<0 or n>z.size-1: return z
    v = z.copy()
    sides = int(a[1].real or 5)
    R     = a[2].real or 0.5
    turns = a[3].real or 0.0
    if sides < 3: 
        v[n] = 0+0j; return v
    rot = np.cos(2*np.pi*turns) + 1j*np.sin(2*np.pi*turns)

    t = v[n].real % 1.0
    segf = t * sides
    seg  = int(np.floor(segf))
    u    = segf - seg

    th0 = 2*np.pi * seg      / sides
    th1 = 2*np.pi * (seg+1)  / sides
    p0 = R * (np.cos(th0) + 1j*np.sin(th0))
    p1 = R * (np.cos(th1) + 1j*np.sin(th1))
    v[n] = rot * ((1-u)*p0 + u*p1)
    return v

# =========================
# Simple star (perimeter)
# a[1]=points p, a[2]=radius R, a[3]=inner_ratio r (0..1)
# (no rotation to stay within 3 params after index)
# =========================
def op_star_simple(z,a,state):
    n = int(a[0].real)
    if n<0 or n>z.size-1: return z
    v = z.copy()
    p  = int(a[1].real or 5)
    R  = a[2].real or 0.6
    r  = a[3].real or 0.5
    if p < 3: 
        v[n]=0+0j; return v

    t = v[n].real % 1.0
    edges = 2*p
    segf = t * edges
    seg  = int(np.floor(segf))
    u    = segf - seg

    def vert(k):
        th = np.pi * k / p
        rad = R if (k % 2)==0 else (r*R)
        return rad * (np.cos(th) + 1j*np.sin(th))

    p0 = vert(seg)
    p1 = vert((seg+1) % edges)
    v[n] = (1-u)*p0 + u*p1
    return v

# =========================
# Rectangle (perimeter)
# a[1]=W, a[2]=H, a[3]=rotation
# =========================
def op_rect(z,a,state):
    n = int(a[0].real)
    if n<0 or n>z.size-1: return z
    v = z.copy()
    W = a[1].real or 1.0
    H = a[2].real or 1.0
    turns = a[3].real or 0.0
    th = 2*np.pi*turns
    rot = np.cos(th) + 1j*np.sin(th)

    t = v[n].real % 1.0
    P = 2*(W+H)
    s = t * P
    if s < W:
        x = -W/2 + s;             y = -H/2
    elif s < W + H:
        x =  W/2;                 y = -H/2 + (s - W)
    elif s < 2*W + H:
        x =  W/2 - (s - (W+H));   y =  H/2
    else:
        x = -W/2;                 y =  H/2 - (s - (2*W+H))
    v[n] = rot * (x + 1j*y)
    return v

# =========================
# Rounded rectangle via superellipse
# a[1]=W, a[2]=H, a[3]=m (roundness; 2=ellipse, big→boxy)
# =========================
def op_roundrect(z,a,state):
    n = int(a[0].real)
    if n<0 or n>z.size-1: return z
    v = z.copy()
    W = a[1].real or 1.0
    H = a[2].real or 0.6
    m = a[3].real or 4.0
    th = 2*np.pi*(v[n].real % 1.0)
    c = np.cos(th); s = np.sin(th)
    cx = np.sign(c) * (np.abs(c)**(2.0/m))
    sy = np.sign(s) * (np.abs(s)**(2.0/m))
    x = (W/2) * cx
    y = (H/2) * sy
    v[n] = x + 1j*y
    return v


def op_ellipse(z,a,state):
    n = int(a[0].real)
    if n<0 or n>z.size-1: return z
    v = z.copy()
    rx = a[1].real or 0.6
    ry = a[2].real or 0.4
    turns = a[3].real or 0.0
    rot = np.cos(2*np.pi*turns) + 1j*np.sin(2*np.pi*turns)

    th = 2*np.pi*(v[n].real % 1.0)
    v[n] = rot * (rx*np.cos(th) + 1j*ry*np.sin(th))
    return v

def op_superellipse(z, a, state):
    """
    a[0]  index in z
    a[1]  A (x half-axis)
    a[2]  B (y half-axis)
    a[3]  m (roundness; 2=ellipse, ↑ boxy, ↓ starry)
    a[4]  rotation (turns)
    a[5]  theta_mul (petal / repetition multiplier)
    a[6]  theta_offset (turns)
    a[7]  ax (x stretch)
    a[8]  ay (y stretch)
    a[9]  skew (x += skew * y)
    a[10] cx (translate x)
    a[11] cy (translate y)
    """
    n = int(a[0].real)
    if n < 0 or n > z.size - 1:
        return z
    v = z.copy()

    # core geometry
    A   = a[1].real  or 0.6
    B   = a[2].real  or 0.4
    m   = a[3].real  or 2.5
    rot = a[4].real  or 0.0     # turns
    km  = a[5].real  or 1.0     # theta multiplier
    th0 = a[6].real  or 0.0     # theta offset (turns)

    # style / deformation
    ax  = a[7].real  or 1.0
    ay  = a[8].real  or 1.0
    skew= a[9].real  or 0.0
    cx  = a[10].real or 0.0
    cy  = a[11].real or 0.0

    # main param angle
    t  = v[n].real % 1.0
    th = 2 * np.pi * (th0 + km * t)

    c = np.cos(th)
    s = np.sin(th)
    # Lamé curve: x = A * sign(c)*|c|^(2/m), y = B * sign(s)*|s|^(2/m)
    px = np.sign(c) * (np.abs(c) ** (2.0 / (m if m != 0 else 1e-9)))
    py = np.sign(s) * (np.abs(s) ** (2.0 / (m if m != 0 else 1e-9)))
    x  = A * px
    y  = B * py

    # apply anisotropy + skew
    x = ax * x + skew * y
    y = ay * y

    # rotation + translation
    R = 2 * np.pi * rot
    xr = x * np.cos(R) - y * np.sin(R) + cx
    yr = x * np.sin(R) + y * np.cos(R) + cy

    v[n] = xr + 1j * yr
    return v


def op_superformula(z, a, state):
    n = int(a[0].real)
    if n < 0 or n > z.size-1: 
        return z
    v = z.copy()
    m   = a[1].real  or 6.0
    n1  = a[2].real  or 0.3
    n2  = a[3].real  or 0.3
    n3  = a[4].real  or 0.3
    ar  = a[5].real  or 1.0
    br  = a[6].real  or 1.0
    S   = a[7].real  or 1.0
    rot = a[8].real  or 0.0       # turns
    th0 = a[9].real  or 0.0       # theta offset (turns)
    km  = a[10].real or 1.0       # theta multiplier
    ay  = a[11].real or 1.0       # y stretch (anisotropy)
    t  = v[n].real % 1.0
    th = 2*np.pi*(th0 + km*t)     # angle
    u  = (m * th) / 4.0
    c = np.cos(u) / ar
    s = np.sin(u) / br
    term = (np.abs(c)**n2) + (np.abs(s)**n3)
    r = S * (term**(-1.0/n1)) if term != 0 else 0.0
    # base point
    x = r * np.cos(th)
    y = r * np.sin(th) * ay       # simple anisotropy on y
    # rotation (turns)
    R = 2*np.pi*rot
    v[n] = (x*np.cos(R) - y*np.sin(R)) + 1j*(x*np.sin(R) + y*np.cos(R))
    return v

def op_epicycloid(z, a, state):
    """
    a[0]  index in z
    a[1]  R     (fixed/base circle radius)
    a[2]  r     (rolling circle radius)
    a[3]  S     (overall scale)
    a[4]  rot   (rotation, turns)
    a[5]  km    (theta multiplier / repetition)
    a[6]  th0   (theta offset, turns)
    a[7]  ax    (x stretch)
    a[8]  ay    (y stretch)
    a[9]  skew  (x += skew*y)
    a[10] cx    (translate x)
    a[11] cy    (translate y)
    """
    n = int(a[0].real)
    if n < 0 or n > z.size - 1:
        return z
    v = z.copy()

    R   = a[1].real  or 0.3
    r   = a[2].real  or 0.1
    S   = a[3].real  or 1.0
    rot = a[4].real  or 0.0
    km  = a[5].real  or 1.0
    th0 = a[6].real  or 0.0
    ax  = a[7].real  or 1.0
    ay  = a[8].real  or 1.0
    skew= a[9].real  or 0.0
    cx  = a[10].real or 0.0
    cy  = a[11].real or 0.0

    t  = v[n].real % 1.0
    th = 2 * np.pi * (th0 + km * t)

    k = (R + r) / r if r != 0 else 0.0
    x = (R + r) * np.cos(th) - r * np.cos(k * th)
    y = (R + r) * np.sin(th) - r * np.sin(k * th)

    # anisotropy + skew
    x = ax * x + skew * y
    y = ay * y

    # rotation, scale, translation
    Rr = 2 * np.pi * rot
    xr = (x * np.cos(Rr) - y * np.sin(Rr)) * S + cx
    yr = (x * np.sin(Rr) + y * np.cos(Rr)) * S + cy

    v[n] = xr + 1j * yr
    return v

def op_hypocycloid(z, a, state):
    """
    a[0]  index in z
    a[1]  R     (fixed/base circle radius)
    a[2]  r     (rolling circle radius)
    a[3]  S     (overall scale)
    a[4]  rot   (rotation, turns)
    a[5]  km    (theta multiplier / repetition)
    a[6]  th0   (theta offset, turns)
    a[7]  ax    (x stretch)
    a[8]  ay    (y stretch)
    a[9]  skew  (x += skew*y)
    a[10] cx    (translate x)
    a[11] cy    (translate y)
    """
    n = int(a[0].real)
    if n < 0 or n > z.size-1:
        return z
    v = z.copy()

    R   = a[1].real  or 0.3
    r   = a[2].real  or 0.1
    S   = a[3].real  or 1.0
    rot = a[4].real  or 0.0
    km  = a[5].real  or 1.0
    th0 = a[6].real  or 0.0
    ax  = a[7].real  or 1.0
    ay  = a[8].real  or 1.0
    skew= a[9].real  or 0.0
    cx  = a[10].real or 0.0
    cy  = a[11].real or 0.0

    t  = v[n].real % 1.0
    th = 2*np.pi*(th0 + km*t)

    k = (R - r)/r if r != 0 else 0.0
    x = (R - r)*np.cos(th) + r*np.cos(k*th)
    y = (R - r)*np.sin(th) - r*np.sin(k*th)

    # anisotropy + skew
    x = ax * x + skew * y
    y = ay * y

    # rotate, scale, translate
    Rr = 2*np.pi*rot
    xr = (x*np.cos(Rr) - y*np.sin(Rr)) * S + cx
    yr = (x*np.sin(Rr) + y*np.cos(Rr)) * S + cy

    v[n] = xr + 1j*yr
    return v

def op_trochoid(z, a, state):
    """
    a[0]  index in z
    a[1]  R     (rolling circle radius)
    a[2]  d     (pen offset; =R→cycloid, <R→curtate, >R→prolate)
    a[3]  S     (overall scale)
    a[4]  rot   (rotation, turns)
    a[5]  km    (theta multiplier / repetition)
    a[6]  th0   (theta offset, turns)
    a[7]  ax    (x stretch)
    a[8]  ay    (y stretch)
    a[9]  skew  (x += skew*y)
    a[10] cx    (translate x)
    a[11] cy    (translate y)
    """
    n = int(a[0].real)
    if n < 0 or n > z.size-1:
        return z
    v = z.copy()

    R   = a[1].real  or 0.2
    d   = a[2].real  or 0.2
    S   = a[3].real  or 1.0
    rot = a[4].real  or 0.0
    km  = a[5].real  or 1.0
    th0 = a[6].real  or 0.0
    ax  = a[7].real  or 1.0
    ay  = a[8].real  or 1.0
    skew= a[9].real  or 0.0
    cx  = a[10].real or 0.0
    cy  = a[11].real or 0.0

    # parameter θ
    t  = v[n].real % 1.0
    th = 2*np.pi*(th0 + km*t)

    # trochoid equations
    x = R * (th - np.sin(th)) + d * np.cos(th)
    y = R * (1.0 - np.cos(th)) + d * np.sin(th)

    # anisotropy + skew
    x = ax * x + skew * y
    y = ay * y

    # rotation, scale, translation
    Rr = 2*np.pi*rot
    xr = (x*np.cos(Rr) - y*np.sin(Rr)) * S + cx
    yr = (x*np.sin(Rr) + y*np.cos(Rr)) * S + cy

    v[n] = xr + 1j*yr
    return v

def op_lemniscate(z, a, state):
    """
    a[0]  index in z
    a[1]  A    (lemniscate scale; r^2 = 2 A^2 cos(2θ))
    a[2]  S    (overall scale)
    a[3]  rot  (rotation, turns)
    a[4]  km   (theta multiplier / repetition)
    a[5]  th0  (theta offset, turns)
    a[6]  ax   (x stretch)
    a[7]  ay   (y stretch)
    a[8]  skew (x += skew*y)
    a[9]  cx   (translate x)
    a[10] cy   (translate y)
    a[11] (reserved)
    """
    n = int(a[0].real)
    if n < 0 or n > z.size-1: 
        return z
    v = z.copy()

    A   = a[1].real  or 0.5
    S   = a[2].real  or 1.0
    rot = a[3].real  or 0.0
    km  = a[4].real  or 1.0
    th0 = a[5].real  or 0.0
    ax  = a[6].real  or 1.0
    ay  = a[7].real  or 1.0
    skew= a[8].real  or 0.0
    cx  = a[9].real  or 0.0
    cy  = a[10].real or 0.0
    # a[11] reserved

    t  = v[n].real % 1.0
    th = 2*np.pi*(th0 + km*t)

    val = 2*(A**2)*np.cos(2*th)
    r = np.sqrt(val) if val > 0.0 else 0.0

    x = r * np.cos(th)
    y = r * np.sin(th)

    # anisotropy + skew
    x = ax * x + skew * y
    y = ay * y

    # rotate, scale, translate
    R  = 2*np.pi*rot
    xr = (x*np.cos(R) - y*np.sin(R)) * S + cx
    yr = (x*np.sin(R) + y*np.cos(R)) * S + cy

    v[n] = xr + 1j*yr
    return v

def op_cassini(z, a, state):
    """
    a[0]  index in z
    a[1]  C   (half-distance between foci; foci at ±C on x-axis)
    a[2]  B   (Cassini parameter; B=C*sqrt(2) gives Bernoulli lemniscate)
    a[3]  S   (overall scale)
    a[4]  rot (rotation, turns)
    a[5]  km  (theta multiplier / repetition)
    a[6]  th0 (theta offset, turns)
    a[7]  ax  (x stretch)
    a[8]  ay  (y stretch)
    a[9]  skew (x += skew*y)
    a[10] cx  (translate x)
    a[11] cy  (translate y)
    """
    n = int(a[0].real)
    if n < 0 or n > z.size-1:
        return z
    v = z.copy()

    # important first
    C   = a[1].real  or 0.30
    B   = a[2].real  or 0.35
    S   = a[3].real  or 1.0
    rot = a[4].real  or 0.0
    km  = a[5].real  or 1.0
    th0 = a[6].real  or 0.0

    # style/deform
    ax  = a[7].real  or 1.0
    ay  = a[8].real  or 1.0
    skew= a[9].real  or 0.0
    cx  = a[10].real or 0.0
    cy  = a[11].real or 0.0

    t  = v[n].real % 1.0
    th = 2*np.pi*(th0 + km*t)

    # solve quadratic in u = r^2:
    # u^2 - 2 C^2 cos(2θ) u + (C^4 - B^4) = 0
    C2 = C*C
    cos2 = np.cos(2*th)
    bb = -2.0 * C2 * cos2
    cc = C2*C2 - B**4
    disc = bb*bb - 4.0*cc

    if disc < 0.0:
        r = 0.0
    else:
        sd = np.sqrt(disc)
        u1 = (-bb + sd) * 0.5
        u2 = (-bb - sd) * 0.5
        u  = max(u1, u2, 0.0)
        r  = np.sqrt(u)

    x = r * np.cos(th)
    y = r * np.sin(th)

    # anisotropy + skew
    x = ax * x + skew * y
    y = ay * y

    # rotate and scale, then translate
    R  = 2*np.pi*rot
    xr = (x*np.cos(R) - y*np.sin(R)) * S + cx
    yr = (x*np.sin(R) + y*np.cos(R)) * S + cy

    v[n] = xr + 1j*yr
    return v

# set i-th state to current z
def op_save(z, a, state):
    i = np.int8(a[0].real)
    state[i] = z.copy()
    return z

# set n-th z value to jth value of ith state
def op_get(z, a, state):
    n = int(a[0].real)
    if n < 0 or n > z.size-1: return z
    zz = z.copy()
    si = np.int8(a[1].real)
    w = state[si]
    j = int(a[2].real)
    if j < 0 or j > w.size-1: return z
    x = w[j]
    zz[n] = x
    return zz

def op_snip(z, a, state):
    n = int(a[0].real)
    m = int(a[1].real)
    if n>m: return z
    if n<0: return z
    if m>z.size: return z
    zz = z[n:m]
    return zz

# add complex dither to inputs
# dither:runs:0.5
def op_dither(z,a,state):
    n = int(a[0].real)
    if n < 0 or n > z.size-1: return z
    serp_len = a[1].real
    dither_width = a[2].real or 0.25
    dither_fact = dither_width/math.sqrt(serp_len)
    dre  = dither_fact * (np.random.random()-0.5)
    drim = dither_fact * (np.random.random()-0.5)
    d = dre + 1j * drim
    zz = z.copy()
    zz[n] = d+z[n]
    return zz

# add complex dither to inputs
# dither:runs:0.5
def op_circle_dither(z,a,state):
    n = int(a[0].real)
    if n < 0 or n > z.size-1: return z
    serp_len = a[1].real
    dither_width = a[2].real or 0.25
    dither_fact = dither_width/math.sqrt(serp_len)
    drad  = dither_fact * np.random.random()
    dngl = np.random.random()
    d = drad * np.exp(1j*2*np.pi*dngl)
    zz = z.copy()
    zz[n] = d+z[n]
    return zz

# uses a[3] as optional sigma multiplier (default 1.0)
def op_normal_dither(z, a, state):
    def _randn2():
        return np.random.normal(), np.random.normal()
    def _fact(serp_len, width):
        return (width or 0.25) / math.sqrt(max(1.0, serp_len))
    n = int(a[0].real)
    if n < 0 or n > z.size-1: return z
    serp_len = a[1].real
    sigma_mul = a[3].real or 1.0
    sigma = sigma_mul * _fact(serp_len, a[2].real)
    dx, dy = _randn2()
    zz = z.copy()
    zz[n] = z[n] + sigma * (dx + 1j*dy)
    return zz

# uses a[3]=inner_frac in [0,1) (default 0.5)
def op_annulus_dither(z, a, state):
    def _fact(serp_len, width):
        return (width or 0.25) / math.sqrt(max(1.0, serp_len))
    n = int(a[0].real)
    if n < 0 or n > z.size-1: return z
    serp_len = a[1].real
    w = _fact(serp_len, a[2].real)
    inner = a[3].real or 0.5
    inner = min(max(inner, 0.0), 0.9999)
    r = math.sqrt(inner*inner + (1.0 - inner*inner) * np.random.random())  # area-uniform
    th = 2.0*math.pi*np.random.random()
    d = (w * r) * (math.cos(th) + 1j*math.sin(th))
    zz = z.copy()
    zz[n] = z[n] + d
    return zz

# uses a[3]=center_angle, a[4]=half_aperture (radians)
def op_sector_dither(z, a, state):
    def _fact(serp_len, width):
        return (width or 0.25) / math.sqrt(max(1.0, serp_len))
    n = int(a[0].real)
    if n < 0 or n > z.size-1: return z
    serp_len = a[1].real
    w = _fact(serp_len, a[2].real)
    cang = a[4].real
    halfap =  math.pi * max(0.0, min( a[3].real or 0.1, 1.0 ) )
    r = math.sqrt(np.random.random())
    th = cang + (np.random.random()*2.0 - 1.0) * halfap
    d = (w * r) * (math.cos(th) + 1j*math.sin(th))
    zz = z.copy()
    zz[n] = z[n] + d
    return zz

# uses a[3]=angle_radians, a[4]=half_length_fraction (default 1 → full width)
def op_line_dither(z, a, state):
    def _fact(serp_len, width):
        return (width or 0.25) / math.sqrt(max(1.0, serp_len))
    n = int(a[0].real)
    if n < 0 or n > z.size-1: return z
    serp_len = a[1].real
    w = _fact(serp_len, a[2].real)
    frac = a[3].real or 1.0
    ang = a[4].real or 0.0
    t = (np.random.random() - 0.5) * 2.0 * 0.5 * frac  # in [-frac/2, frac/2]
    dx = w * (2.0 * t) * math.cos(ang)
    dy = w * (2.0 * t) * math.sin(ang)
    zz = z.copy()
    zz[n] = z[n] + (dx + 1j*dy)
    return zz

def op_cross_dither(z, a, state):
    """
    Cross brush (two perpendicular line segments through the center).
    Params (same layout):
      a[0] = index (int)
      a[1] = serp_len (float)
      a[2] = width (float)  # sets half-length of each arm, scaled by serp_len
    """
    n = int(a[0].real)
    if n < 0 or n > z.size - 1:
        return z

    serp_len = a[1].real
    # scale size like your other brushes: width / sqrt(serp_len)
    w = (a[2].real or 0.25) / math.sqrt(max(1.0, serp_len))

    # pick arm: 0 -> horizontal (real axis), 1 -> vertical (imag axis)
    arm = 0 if np.random.random() < 0.5 else 1
    # pick position along the chosen arm, uniform in [-w, +w]
    t = (np.random.random() * 2.0 - 1.0) * w

    if arm == 0:
        d = t + 0j          # horizontal stroke
    else:
        d = 1j * t          # vertical stroke

    zz = z.copy()
    zz[n] = z[n] + d
    return zz

def op_extend(z,a,state):
    n = int(a[0].real)
    if n < 0 or n > z.size-1: return z
    zz = np.empty(z.size + 1, dtype=np.complex128)
    for i in range(z.size): zz[i] = z[i]
    zz[z.size] = z[n]
    return zz

def op_copy(z,a,state):
    n = int(a[0].real)
    m = int(a[1].real)
    if n < 0 or n > z.size-1: return z
    if m < 0 or m > z.size-1: return z
    zz = z.copy()
    zz[m] = z[n]
    return zz

ALLOWED = {
    "copy":   op_copy,
    "xtnd":   op_extend,
    "snip":   op_snip,
    "save":   op_save,
    "get":    op_get,
    "dot":    op_dot,
    "xim":    op_xim,
    "zz":     op_zz,
    "zz1":    op_zz1,
    "zz2":    op_zz2,
    "zz3":    op_zz3,
    "pz":     op_pz,
    "bkr":    op_bkr,
    "crd":    op_crd,
    "hrt":    op_hrt,
    "spdl":   op_spdl,
    "lmc":    op_limacon,
    "rsc":    op_rose_curve,
    "lss":    lissajous,
    "ast":    astroid,
    "asp":    archimedean_spiral,
    "lsp":    logarithmic_spiral,
    "dlt":    deltoid,
    "rply":   op_poly_regular,
    "star":   op_star_simple,
    "cssn":   op_cassini,
    "lmn":    op_lemniscate,
    "trch":   op_trochoid,
    "hcld":   op_hypocycloid,
    "ecld":   op_epicycloid,
    "supf":   op_superformula,
    "supe":   op_superellipse,
    "eclps":  op_ellipse,
    "rect":   op_rect,
    "rrect":  op_roundrect,
    "circ":   op_circle,
    "rt":     op_rhotheta,
    "tr":     op_thetarho,
    "rttr":   op_rttr,
    "sdth":   op_dither,
    "cdth":   op_circle_dither,
    "ndth":   op_normal_dither,
    "adth":   op_annulus_dither,
    "ldth":   op_line_dither,
    "crdth":  op_cross_dither,
    "scdth":  op_sector_dither,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fun", type=str, default="nop",help="select function")
    ap.add_argument("--plot", type=str, default=None,help="select function")
    ap.add_argument("--z", type=str, default="0+0j",help="set argument")
    ap.add_argument("--a", type=str, default="0+0j",help="set parameter")
    ap.add_argument("--list", action="store_true", help="list functions.")
    args = ap.parse_args()
    if args.list:
        for k in sorted(ALLOWED.keys()):
            print(k)
        return
    z = np.array(ast.literal_eval(args.z),dtype=np.complex128)
    a = np.array(ast.literal_eval(args.a),dtype=np.complex128)
    state  = Dict.empty(key_type=types.int8,value_type=types.complex128[:])
    print(f"{args.fun}({args.z},{args.a}) = {ALLOWED[args.fun](z,a,state)}")

if __name__ == "__main__":
    main()
