# ops_poly.py
import numpy as np
from numba.typed import Dict
from numba import types, njit
import math
import argparse
import ast

@njit(cache=True, fastmath=True)
def _safe_div(top: np.complex128, bot: np.complex128, eps: float = 1e-12) -> np.complex128:
    # Tikhonov-regularized division: top / bot ≈ top*conj(bot)/( |bot|^2 + eps^2 )
    br = bot.real; bi = bot.imag
    denom = br*br + bi*bi + eps*eps
    tr = top.real; ti = top.imag
    num_r = tr*br + ti*bi
    num_i = ti*br - tr*bi
    return (num_r/denom) + 1j*(num_i/denom)

ALLOWED = {}

def g1(z,a,state):
    t1, t2 = z[0], z[1] 
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = 30 * (t1**2 * t2)
    cf[1] = 30 * (t1 * t2**2)
    cf[2] = 40 * (t1**3)
    cf[3] = 40 * (t2**3)
    cf[4] = -25 * (t1**2)
    cf[5] = -25 * (t2**2)
    cf[6] = 10 * (t1 * t2)
    cf[9] = 100 * (t1**4 * t2**4)
    cf[11] = -5 * t1
    cf[13] = 5 * t2
    cf[24] = -10
    return cf

ALLOWED["g1"]=g1

def g2(z,a,state):
    t1, t2 = z[0], z[1] 
    cf = np.zeros(25, dtype=np.complex128)
    cf[9] = 200 * (t1**2 + t2**2)
    cf[15] = 150 * (t1**3 * t2**5)
    cf[7] = -80 * (t1**4 - t2**2)     # z^7 term (index 8 means z^(8-1)=z^7)
    cf[5] = 50 * (t1**3 + t2)         # z^5 term (z^(6-1)=z^5)
    cf[2] = 20 * (t1 - t2)            # z^2
    cf[1] = -10 * (t1 * t2)           # z^1
    cf[0] = -5
    cf[19] = -30 * (t1**5 - t2**5)    # z^19
    cf[24] = 10 * (t1 * t2**3)        # z^24
    return cf
    
ALLOWED["g2"]=g2

def g3(z,a,state):
    t1, t2 = z[0], z[1] 
    n = 25
    cf = np.zeros(n, dtype=np.complex128)
    cf[0] = 1
    cf[n-11] = np.exp(t1-t2)
    cf[n-10] = 0*np.exp(t1+t2)
    cf[n-9] = 1*np.exp(1j*t1)
    cf[n-8] = 1*np.exp(t1)
    cf[n-7] = 1*np.exp(-t1)
    cf[n-6] = 1*np.exp(-1j*t1)
    cf[n-2] = 1*np.exp(1j*t2)
    cf[n-1] = 1+1j
    return cf
    
ALLOWED["g3"]=g3

def g4(z,a,state):
    t1, t2 = z[0], z[1] 
    n = 25
    cf = np.zeros(n, dtype=np.complex128)
    cf[0] = 100
    cf[12] = 100 * t1**3 + 100 * t1**2 + 100 * t1 - 100
    cf[14] = 100 * t1**3 - 100 * t1**2 + 100 * t1 - 100
    cf[16] = 100 * t1**3 + 100 * t1**2 - 100 * t1 - 100
    cf[20] = -10
    cf[24] = np.exp(0.2j * t2)
    return cf
    
ALLOWED["g4"]=g4

def g5(z,a,state):
    t1, t2 = z[0], z[1] 
    cf = np.zeros(25, dtype=np.complex128)
    cf[0]  = 1.0 + 0j 
    cf[4]  = 4.0 + 0j
    cf[12] = 4.0 + 0j 
    cf[19] = -9 + 0j
    cf[20] = -1.9 + 0j
    cf[24] = 0.2 + 0j
    cf[6] = 100j * t2**3 + 100j * t2**2 - 100j * t2 - 100j
    cf[8] = 100j * t1**3 + 100j * t1**2 + 100j * t2 - 100j
    cf[14] = 100j * t2**3 - 100j * t2**2 + 100j * t2 - 100j
    return cf

ALLOWED["g5"]=g5

def g6(z,a,state):
    t1, t2 = z[0], z[1] 
    n = 10
    cf = np.zeros(n, dtype=np.complex128)
    cf[0] = 150 * t2**3 - 150j * t1**2
    cf[1] = 0
    cf[n//2-1] = 100*(t1-t2)**1
    cf[n-2] = 0
    cf[n-1] = 10j
    return cf
    
ALLOWED["g6"]=g6

def g7(z,a,state):
    t1, t2 = z[0], z[1] 
    pi  =  np.pi
    n   =  30
    rec =  np.linspace(np.real(t1), np.real(t2), n)
    imc =  np.linspace(np.imag(t1), np.imag(t2), n)
    f1  =  np.exp(1j * np.sin(10 * pi * imc))
    f2  =  np.exp(1j * np.cos(10 * pi * rec))
    f   =  f1 + f2
    return  f
    
ALLOWED["g7"]=g7

# slightly modified
def g8(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    cf1 = np.flip(np.array([t1**3, -50 * t2, 100 * t1, 10j],dtype=np.complex128))
    roots1 = np.roots(cf1)
    cf2 = np.flip(np.array([1, roots1[0], -np.real(roots1[1]), np.imag(roots1[2])],dtype=np.complex128))
    roots2 = np.roots(cf2)
    cf[0:3] = roots1
    cf[9:12] = roots2
    cf[19] = 50 * t1 * t2 + np.real(roots2[0])
    cf[29] = np.exp(1j * t1) + 50 * t2**3
    cf[34] = 200 * np.exp(1j * t1**3) - np.exp(-1j * t2**2)
    return cf
    
ALLOWED["g8"]=g8

def g9(z,a,state):
    t1, t2 = z[0], z[1]
    n = 20
    re1 = t1.real
    im1 = t1.imag
    re2 = t2.real
    im2 = t2.imag
    rec = np.linspace(re1, re2, n)
    imc = np.linspace(im1, im2, n)
    cf = 100j * imc**9 + 100 * rec**9
    return cf.astype(np.complex128)
    
ALLOWED["g9"]=g9

def g10(z,a,state):
    t1, t2 = z[0], z[1]
    n = 120
    cf = np.empty(n, dtype=np.complex128)
    re1, im1 = t1.real, t1.imag
    re2, im2 = t2.real, t2.imag
    for k in range(n):
        cf[k] = (100 * (re1 + im2) * ((k+1)/10)**2) * np.exp(1j * (re2 * (k+1) / 20)) + \
                (50 * (im1 - re2) * np.sin((k+1) * 0.1 * im2)) * np.exp(-1j * (k+1) * 0.05 * re1)
    cf[29] = cf[29] + 1000j
    cf[59] = cf[59] - 500
    cf[89] = cf[89] + 250 * np.exp(1j * (t1 * t2))
    return cf

ALLOWED["g10"]=g10
    
def g11(z,a,state):
    t1, t2 = z[0], z[1]
    n = 40
    cf = np.zeros(n, dtype=np.complex128)
    m = int(5*abs(t1 + t2) % 17) + 1
    modular_values = np.arange(n) % m
    for k in range(n):
        scale_factor = modular_values[k]
        cf[k] = scale_factor * np.exp(1j * np.pi * (k+1) / (m + t1 + t2))
    return cf

ALLOWED["g11"]=g11

def g12(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = 2
    cf[2] = 50 * (t1**3)
    cf[4] = 50 * (t2**3)
    cf[6] = -30 * (t1**2)
    cf[8] = -30 * (t2**2)
    cf[10] = 100 * (t1 * t2)
    cf[12] = 50 * (t1**2 * t2)
    cf[14] = 50 * (t1 * t2**2)
    cf[19] = -75 * (t1**3 * t2**3)
    cf[20] = 3.5 * t2
    cf[24] = -2 * t1
    return cf
      
ALLOWED["g12"]=g12

def g13(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[1] = 100 * (t1**4)
    cf[3] = 100 * (t2**4)
    cf[5] = 80 * (t1**3 * t2)
    cf[7] = 80 * (t1 * t2**3)
    cf[9] = 1 * t1
    cf[11] = -1 * t2
    cf[13] = 5 * (t1**2 * t2**2)
    cf[17] = -0.5 * (t1**5)
    cf[18] = -0.5 * (t2**5)
    cf[22] = 2.3 * (t1**2 - t2**2)
    cf[24] = 10 * (t1**3 - t2**3)
    return cf
    
ALLOWED["g13"]=g13

def g14(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = 200 * (t1**3 * t2**2)
    cf[4] = 200 * (t1**2 * t2**3)
    cf[6] = 50 * (t1**4)
    cf[8] = 50 * (t2**4)
    cf[10] = -100 * (t1**3)
    cf[12] = -100 * (t2**3)
    cf[14] = 10 * (t1**2 - t2**2)
    cf[16] = 20 * (t1 - t2)
    cf[18] = 0.1 * (t1**5)
    cf[20] = 0.1 * (t2**5)
    cf[22] = 0.05 * (t1 * t2)
    cf[24] = -10
    return cf
    
ALLOWED["g14"]=g14

def g15(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(18, dtype=np.complex128)
    cf[1] = 200 * (t1**5 + t2**5)
    cf[3] = 100 * (t1**4 - t2**4)
    cf[5] = 80 * (t1**6)
    cf[7] = 80 * (t2**6)
    cf[9] = 2 * t1
    cf[11] = -2 * t2
    cf[13] = 5 * (t1**3 * t2**3)
    cf[17] = 5
    return cf

ALLOWED["g15"]=g15

def g16(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(22, dtype=np.complex128)
    cf[12] = 250 * (t1**5 - t2**3)
    cf[17] = 200 * (t1**4 * t2**4)
    cf[8] = 80 * (t1**2 * t2 - t2**2)
    cf[6] = -60 * (t1**3)
    cf[4] = 40 * (t2**3)
    cf[2] = 15 * (t1 - 0.5*t2)
    cf[3] = -20 * (t1*t2)
    cf[1] = 5 * t2
    cf[0] = -10
    cf[21] = -30 * (t1**6 + t2)
    return cf.astype(np.complex128)

ALLOWED["g16"]=g16

def g17(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    theta1 = np.angle(t1)
    theta2 = np.angle(t2)
    scale1 = 1 + 0.5 * np.sin(5*theta1 - 3*theta2)
    scale2 = 1 + 0.3 * np.cos(7*theta1 + 2*theta2)
    high_deg_index = 20 if np.sin(theta1 - theta2) > 0 else 15
    cf[high_deg_index] = 300 * (t1**7 * t2**9) * scale1
    if np.cos(theta1 + theta2) < 0:
        cf[18] = 250 * (t1**4 * t2**7) * scale2
    else:
        cf[12] = 250 * (t1**4 * t2**7) * scale2
    scale3 = 1 + 0.4 * np.sin(theta1 + 2*theta2)
    cf[8] = 80 * (t1**5 - t2) * scale3
    cf[6] = -100 * (t1**2 * t2**2) * scale3
    cf[4] = -20 * (t1**3 + t2) * scale3
    scale4 = 1 + 0.2 * np.cos(2*theta1 - theta2)
    cf[2] = -5 * (t1**2 - t2**2) * scale4
    cf[1] = (t2**3 - t1) * scale4
    cf[0] = -5
    perturb_scale = 0.5 + 0.5 * np.sin(3*theta1) * np.cos(4*theta2)
    cf[24] = (t1**4 - t2**4) * perturb_scale
    cf[22] = -0.5 * (t1**9 + t2**9) * perturb_scale
    return cf.astype(np.complex128)
    
ALLOWED["g17"]=g17

def g18(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    theta1 = np.angle(t1)
    theta2 = np.angle(t2)
    base_scale = 1000 * np.exp(0.5 * np.sin(10*theta1 - 7*theta2))
    secondary_scale = 500 * (np.cos(12*theta1 + 15*theta2))**3
    if np.sin(theta1 - theta2) > 0:
        toggle_scale = 2000 * np.sin(5*theta1)*np.cos(3*theta2)
    else:
        toggle_scale = -2000 * np.cos(4*theta1)*np.sin(2*theta2)
    if np.cos(theta1 + theta2) > 0.5:
        cf[20] = (t1**7 * t2**9) * base_scale * toggle_scale
    elif np.cos(theta1 + theta2) < -0.5:
        cf[18] = (t1**10 - t2**10) * secondary_scale * toggle_scale
    else:
        cf[12] = (t1**4 * t2**7 - t1**5) * base_scale * secondary_scale
    complex_scale = 300 * (np.sin(np.sin(3*theta1 + 4*theta2)))**2
    cf[8] = (t1**5 - t2)*complex_scale
    another_scale = 100 * np.exp(np.sin(theta1)*np.cos(theta2))
    cf[6] = -another_scale * (t1**2 * t2**2)
    sign_flip = 1 if (np.floor((theta1+theta2)*3) % 2) == 0 else -1
    cf[4] = sign_flip * 50 * (t1**3 + t2) * (np.sin(2*theta1 - theta2))
    cf[2] = -5 * (t1**2 - t2**2) * (10 * np.cos(5*theta2))
    cf[1] = (t2**3 - t1) * (200 * np.sin(3*theta1)*np.sin(theta2))
    cf[0] = -5
    cf[24] = (t1**4 - t2**4) * 100 * (np.cos(np.sin(theta1)*theta2)) * np.exp(np.cos(2*theta1 - 3*theta2))
    cf[22] = -10 * (t1**9 + t2**9) * (np.sin(7*theta1 - 8*theta2))**3
    cf[15] = 500 * (t1**6 - t2**3) * np.sin((theta1 + theta2)**2) * np.cos((theta1 - theta2)**2)
    return cf
    
ALLOWED["g18"]=g18

def g19(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(90, dtype=np.complex128)
    cf[0] = t1 - t2
    for k in range(1, len(cf)):
        v = np.sin(k * cf[k-1]) + np.cos(k * t1)
        av = np.abs(v)
        cf[k] = 1j * v / av if np.isfinite(av) and av > 1e-10 else t1 + t2
    return cf
    
ALLOWED["g19"]=g19

def g20(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(90, dtype=np.complex128)
    cf[0] = t1 + 1j * t2
    for k in range(1, len(cf)):
        v = np.sin(k * cf[k-1]) + np.cos(k * t1)
        av = np.abs(v)
        if np.isfinite(av) and av > 1e-10:
            cf[k] = 1j * v / av
        else:
            cf[k] = t1 + t2
    return cf

ALLOWED["g20"]=g20

def g21(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(50, dtype=np.complex128)
    cf[0] = t1 + t2
    for k in range(1, len(cf)):
        v = np.sin(((k+3) % 10) * cf[k-1]) + np.cos(((k+1) % 10) * t1)
        av = np.abs(v)
        if np.isfinite(av) and av > 1e-10:
            cf[k] = v / av
        else:
            cf[k] = t1 + t2
    return cf

ALLOWED["g21"]=g21

def g22(z,a,state):
    t1, t2 = z[0], z[1]
    n = 26
    cf = np.zeros(n, dtype=np.complex128)
    cf[0] = 100
    cf[12] = 100 * t1**3 + 100 * t1**2 + 100 * t1 - 100
    cf[14] = 100 * t2**3 - 100 * t2**2 + 100 * t2 - 100
    cf[16] = 100 * t2**3 + 100 * t2**2 - 100 * t2 - 100
    cf[20] = -10
    cf[24] = 0.2j
    cf[25] = 0
    return cf
    
ALLOWED["g22"]=g22

def g23(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(26, dtype=np.complex128)
    cf[0]=1
    cf[4]=4
    cf[12]=4
    cf[19]=-9
    cf[20]=-1.9
    cf[24]=0.2
    cf[6] = 100 * t2**3 + 100 * t2**2 - 100 * t2 - 100
    cf[8] = 100 * t1**3 + 100 * t1**2 + 100 * t2 - 100
    cf[14] = 100 * t2**3 - 100 * t2**2 + 100 * t2 - 100
    return cf
    
ALLOWED["g23"]=g23

def g24(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(26, dtype=np.complex128)
    cf[0]  = 1.0
    cf[4]  = 4.0
    cf[12] = 4.0
    cf[19] = -9.0
    cf[20] = -1.9
    cf[24] = 0.2
    cf[6] = 100j * t2**3 + 100j * t2**2 - 100 * t2 - 100
    cf[8] = 100j * t1**3 + 100j * t1**2 + 100 * t2 - 100
    cf[14] = 100j * t2**3 - 100j * t2**2 + 100 * t2 - 100
    return cf

ALLOWED["g24"]=g24

def g25(z,a,state):
    t1, t2 = z[0], z[1]
    n = 26
    cf = np.zeros(n, dtype=np.complex128)
    cf[0] = 100
    cf[12] = 100 * t1**3 + 100 * t1**2 + 100 * t1 - 100
    cf[14] = 100 * t2**3 - 100 * t2**2 + 100 * t2 - 100
    cf[16] = 100 * t2**3 + 100 * t2**2 - 100 * t2 - 100
    cf[20] = -10
    cf[24] = 0.2j
    cf[25] = 0
    return cf

ALLOWED["g25"]=g25
    
def g26(z,a,state):
    t1, t2 = z[0], z[1]
    n = 26
    cf = np.zeros(n, dtype=np.complex128)
    cf[0] = 100
    cf[12] = 100 * t1**3 + 100 * t1**2 + 100 * t1 - 100
    cf[14] = 100 * t2**3 - 100 * t2**2 + 100 * t2 - 100
    cf[16] = 100 * t2**3 + 100 * t2**2 - 100 * t2 - 100
    cf[20] = -10
    cf[24] = 0.2
    cf[25] = 0
    return cf

ALLOWED["g26"]=g26

def g27(z,a,state):
    t1, t2 = z[0], z[1]
    n = 12
    cf = np.zeros(n, dtype=np.complex128)
    cf[0:3] = [-100j, -100j, -100j]
    mid_indices = np.array([n//2-2, n//2-1, n//2],dtype=np.intp)
    cf[mid_indices] = 100 * np.roots(np.array([t1, t2, t1, 1],dtype=np.complex128))
    end_indices = np.array([n-1, n-2, n-3],dtype=np.intp)
    cf[end_indices] = 100 * np.roots(np.array([t2, t1, t2, 10j],dtype=np.complex128))
    return cf
    
ALLOWED["g27"]=g27

def g28(z,a,state):
    t1, t2 = z[0], z[1]
    n = 6
    cf = np.zeros(n, dtype=np.complex128)
    cf[0] = 100 * t2**3 + 100j * t1**3
    cf[1] = 0
    # cf[n//2-1] = 150
    cf[int(n/2) - 1] = 150
    cf[n-2] = 0
    cf[n-1] = 40j
    return cf

    
ALLOWED["g28"]=g28

def g29(z,a,state):
    t1, t2 = z[0], z[1]
    n = 10
    cf = np.zeros(n, dtype=np.complex128)
    cf[0] = 150 * t2**3 - 150j * t1**2
    cf[1] = 0
    cf[n//2-1] = 100*(t1-t2)**1
    cf[n-2] = 0
    cf[n-1] = 10j
    return cf
    
ALLOWED["g29"]=g29

def g30(z,a,state):
    t1, t2 = z[0], z[1]
    n = 10
    cf = np.zeros(n, dtype=np.complex128)
    cf[0] = 150j * t2**2 + 100 * t1**3
    cf[n//2-1] = 150 * np.abs(t1 + t2 - 2.5 * (1j + 1))
    cf[n-1] = 100j * t1**3 + 150 * t2**2
    return cf.astype(np.complex128)

ALLOWED["g30"]=g30

def g32(z,a,state):
    t1, t2 = z[0], z[1]
    n = 12
    cf = np.zeros(n, dtype=np.complex128)
    cf[0:3] = [-100j, -100j, -100j]
    mid_indices = np.array([n//2-2, n//2-1, n//2],dtype=np.intp)
    cf1 = np.array([t1, t2, t1, 1], dtype=np.complex128)
    cf[mid_indices] = 100 * np.roots(cf1)
    end_indices = np.array([n-1, n-2, n-3],dtype=np.intp)
    cf2 = np.array([t2, t1, t2, 10j],dtype=np.complex128)
    cf[end_indices] = 100 * np.roots(cf2)
    return cf

ALLOWED["g32"]=g32  


def g33(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(26, dtype=np.complex128)
    cf[0] = 11j * t1**4 + 13j * t1**3 + 17j * t1**2 + 19j * t1 + 23j
    cf[1] = 100j * t1**3 + 100j * t2**2 - 100 * t1 - 100
    cf[2] = 100 * t2**3 + 100j * t1**2 - 100j * t2 - 100j
    cf[3] = 100j * t1**3 + 100 * t2**2 - 100 * t1 - 100j
    cf[4] = -3
    cf[6] = 101 * t2**3 + 103 * t2**2 - 107 * t2 - 109
    cf[8] = 113 * t1**3 + 127 * t1**2 + 131 * t2 - 137
    cf[12] = 5
    cf[14] = 67 * t2**3 - 71 * t2**2 + 73 * t2 - 79
    cf[16] = -7
    cf[20] = 11
    cf[24] = -13
    return cf

ALLOWED["g33"]=g33     

def g34(z,a,state):
    t1, t2 = z[0], z[1]
    n = 120
    cf = np.zeros(n, dtype=np.complex128)
    cf[0] = -1
    cf[n//2-1] = 100 * t1 - 100j * t2
    cf[n-1] = 0.4
    return cf
    
ALLOWED["g34"]=g34

def g35(z,a,state):
    t1, t2 = z[0], z[1]
    n = 120
    cf = np.zeros(n, dtype=np.complex128)
    cf[0] = -1
    cf[n//2-1] = 100 * t1 - 100j * t2
    cf[n-1] = 0.4
    return cf

ALLOWED["g35"]=g35

def g36(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(26, dtype=np.complex128)
    cf[0] = 1
    cf[1] = 100 * t1**3 + 100 * t2**2 - 100 * t1 - 100
    cf[2] = 100 * t2**3 + 100 * t1**2 - 100 * t2 - 100
    cf[4] = 4
    cf[6] = 100 * t2**3 + 100 * t2**2 - 100 * t2 - 100
    cf[8] = 100 * t1**3 + 100 * t1**2 + 100 * t2 - 100
    cf[12] = -8
    cf[14] = 100 * t2**3 - 100 * t2**2 + 100 * t2 - 100
    cf[19] = 16
    cf[20] = -32
    cf[24] = 64
    return cf
    
ALLOWED["g36"]=g36

def g37(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(26, dtype=np.complex128)
    cf[0] = 11j * t1**4 + 13j * t1**3 + 17j * t1**2 + 19j * t1 + 23j
    cf[1] = 100j * t1**3 + 100j * t2**2 - 100 * t1 - 100
    cf[2] = 100 * t2**3 + 100j * t1**2 - 100j * t2 - 100j
    cf[3] = 100j * t1**3 + 100 * t2**2 - 100 * t1 - 100j
    cf[4] = -3
    cf[6] = 101 * t2**3 + 103 * t2**2 - 107 * t2 - 109
    cf[8] = 113 * t1**3 + 127 * t1**2 + 131 * t2 - 137
    cf[12] = 5
    cf[14] = 67 * t2**3 - 71 * t2**2 + 73 * t2 - 79
    cf[16] = -7
    cf[20] = 11
    cf[24] = -13
    return cf

ALLOWED["g37"]=g37

def g38(z,a,state):
    t1, t2 = z[0], z[1]
    n = 26
    cf1 = np.zeros(n, dtype=np.complex128)
    cf1[0] = 100
    cf1[12] = 100 * t1**3 + 100 * t1**2 + 100 * t1 - 100
    cf1[14] = 100 * t1**3 - 100 * t1**2 + 100 * t1 - 100
    cf1[16] = 100 * t1**3 + 100 * t1**2 - 100 * t1 - 100
    cf1[20] = -10
    cf1[24] = np.exp(0.2j * t2)
    cf1[25] = 0
    cf2 = np.zeros(n, dtype=np.complex128)
    cf2[0] = 100
    cf2[12] = 100 * t1**3 + 100 * t1**2 + 100 * t1 - 100
    cf2[14] = 100 * t2**3 - 100 * t2**2 + 100 * t2 - 100
    cf2[16] = 100 * t2**3 + 100 * t2**2 - 100 * t2 - 100
    cf2[20] = -10
    cf2[24] = 0.2j
    cf2[25] = 0
    result = (cf1 - 0.0001 * np.sum(np.abs(cf1))) * (cf2 + 1.5j * np.sum(np.abs(cf2)))
    return result

ALLOWED["g38"]=g38

def g39(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(50, dtype=np.complex128)
    cf[np.array([0, 9, 19, 29, 39, 49])] = np.array([1, 2, -3, 4, -5, 6])
    cf[14] = 100 * (t1**2 + t2**2)
    cf[24] = 50 * (np.sin(t1) + 1j * np.cos(t2))
    cf[34] = 200 * (t1 * t2) + 1j * (t1**3 - t2**3)
    cf[44] = np.exp(1j * (t1 + t2)) + np.exp(-1j * (t1 - t2))
    return cf

ALLOWED["g39"]=g39

def g40(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    cf[np.array([0, 6, 14, 19, 26, 34])] = [1, -2, 3, -4, 5, -6]
    cf[11] = 50j * np.sin(t1**2 - t2**2)
    cf[17] = 100 * (np.cos(t1) + 1j * np.sin(t2))
    cf[24] = 50 * (t1**3 - t2**3 + 1j * t1 * t2)
    cf[29] = 200 * np.exp(1j * t1) + 50 * np.exp(-1j * t2)
    return cf
    
ALLOWED["g40"]=g40

def g41(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(60, dtype=np.complex128)
    cf[np.array([0, 9, 29, 49])] = [1, -5, 10, -20]
    cf[19] = 100 * np.exp(t1 + t2)
    cf[39] = 50 * (t1**2 * t2 + 1j * t2**2)
    cf[54] = np.exp(1j * t1) * np.exp(-1j * t2) + 50 * t1**3
    cf[59] = 300 * np.sin(t1 + t2) + 1j * np.cos(t1 - t2)
    return cf

ALLOWED["g41"]=g41

def g42(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(50, dtype=np.complex128)
    cf[np.array([0, 7, 15, 31, 39])] = [1, -3, 3, -1, 2]
    cf[11] = 100j * np.exp(t1**2 + t2**2)
    cf[19] = 50 * (t1**3 + t2**3)
    cf[24] = np.exp(1j * (t1 - t2)) + 10 * t1**2
    cf[44] = 200 * np.sin(t1 + t2) + 1j * np.cos(t1 - t2)
    return cf.astype(np.complex128)

ALLOWED["g42"]=g42 

def g43(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(40, dtype=np.complex128)
    i = np.array([0, 4, 14, 29],dtype=np.intp)
    cf[i] = np.array([1, -5, 10, -20],dtype=np.complex128)
    cf[19] = 100j * (t1**3 - t2**3)
    cf[9] = 50 * (t1**2 * t2 + 1j * t2**2)
    cf[24] = np.exp(1j * t1) + np.exp(-1j * t2)
    cf[34] = 200 * t1 * t2 * np.sin(t1 + t2)
    return cf
    
ALLOWED["g43"]=g43

def g44(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(30, dtype=np.complex128)
    i = np.array([0, 5, 11, 19],dtype=np.intp)
    cf[i] = np.array([1, 3, -2, 5],dtype=np.complex128)
    cf[9] = 100 * t1**3 + 50 * t2**2
    cf[14] = 50j * (t1.real - t2.imag)
    cf[24] = 200 * t1 * (t2 + 1) - 100j * t2
    cf[29] = np.exp(1j * t1) + t2**3
    return cf


ALLOWED["g44"]=g44

def g45(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(50, dtype=np.complex128)
    cf[0] = 1
    cf[4] = 50 * np.exp(t1)
    cf[9] = 100 * (t2**2 - 1j * t1)
    cf[19] = 200 * np.exp(1j * t1**2) - 50 * np.exp(-1j * t2**3)
    cf[29] = 100 * t1 * t2**2 + 50j * t1**3
    cf[39] = np.exp(1j * (t1 + t2)) - 50 * np.sin((t1 - t2).imag)
    return cf.astype(np.complex128)

ALLOWED["g45"]=g45

def g46(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(40, dtype=np.complex128)
    i = np.array([0, 7, 15, 23, 31],dtype=np.intp) 
    cf[i] = [1, -3, 5, -7, 2]
    cf[4] = 50 * (t1**2 - t2**3)
    cf[11] = 100j * (t1**3 + t2)
    cf[19] = np.exp(1j * t1) + np.exp(-1j * t2**2)
    cf[29] = 200 * np.sin(t1.real + t2.imag) - 50 * np.cos((t1 - t2).imag)
    cf[34] = np.exp(1j * t1**3) + t2**2
    return cf

ALLOWED["g46"]=g46

def g47(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(30, dtype=np.complex128)
    cf1 = np.array([t1**3 - t2**2, 100 * t1, -50 * t2, 10j], dtype=np.complex128)
    cf[0:3] = np.roots(cf1)
    cf2 = np.array([1, t1**2 - 1j * t2, -100], dtype=np.complex128)
    cf[9:11] = np.roots(cf2)
    cf[14] = 50 * t1**3 - 20 * t2
    cf[24] = 200 * np.sin(t1.real + t2.imag) + 1j * np.cos(t1.imag - t2.real)
    cf[29] = np.exp(1j * t1) + t2**3
    return cf

ALLOWED["g47"]=g47
    
def g48(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(40, dtype=np.complex128)
    cf1 = np.array([np.sin(t1) + np.cos(t2), 100 * t1**2, -50 * t2, 10j],dtype=np.complex128) 
    cf[0:4] = cf1
    cf2 = np.array([np.cos(t1.real) + np.sin(t2.imag), -1, t1**3 - t2**2],dtype=np.complex128)
    cf[9:12] = cf2
    cf[19] = 50 * (t1**2 - t2**3)
    cf[29] = np.exp(1j * t1) + t2**2
    cf[34] = 200 * np.sin(t1.real + t2.imag) + 50 * np.cos((t1 - t2).imag)
    cf[39] = np.exp(1j * t1**3) + t2**3
    return cf

ALLOWED["g48"]=g48

#FIXME  cannot assign slice of shape (3,) from input of shape (4,)
def g49(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(30, dtype=np.complex128)
    cf1 = np.array([t1**4 - t2**3, 1j * t1 - t2, 50 * np.sin(t1 + t2), -100],dtype=np.complex128)
    unstable_roots = np.roots(cf1)
    cf[0:4] = unstable_roots
    cf2 = np.array([unstable_roots[0]**2, -unstable_roots[1], 1],dtype=np.complex128)
    cf[7:10] = np.roots(cf2)
    cf[14] = 100j * (t1**2 + t2**3)
    cf[24] = 200 * np.sin(t1.real + t2.imag) + 50 * np.cos((t1 - t2).imag)
    cf[29] = np.exp(1j * t1**3) + t2**3
    return cf.astype(np.complex128)
    
ALLOWED["g49"]=g49

def g50(z,a,state):
    t1, t2 = z[0], z[1]
    n = 1000
    cf = np.zeros(n, dtype=np.complex128)
    w = 100
    p1 = 1 * t2**3 - 1 * t2**2 + 1 * t2 - 1
    p2 = 1 * t2**3 + 1 * t2**2 + 1 * t2 + 1
    p3 = 1 * t1**3 + 1 * t2**2 + 1 * t1 + 1
    p4 = 1 * t1**3 + 1 * t1**2 + 1 * t1 + 1
    p5 = 1 * t1**3 + 1 * t1**2 + 1 * t1 - 1
    cf[0] = 10
    idx_range = np.arange(int(n*0.25) - 5, int(n*0.25) + 1).astype(np.intp)
    cf[idx_range] = 1e8*w * np.array([1j*p1, p2, p1, p1*p3, p1-p3, p1+p3]) * np.abs(t1-t2)
    cf[n//2-2] = w * p1
    cf[n//2-1] = w * p2
    cf[n//2] = w * (p1+p4)
    cf[n//2+1] = w * p4
    cf[n//2+2] = w * 1j * p4
    idx_range = np.arange(int(n*0.75), int(n*0.75) + 6)
    cf[idx_range] = w * np.array([1j*p1, p2, p1, p1*p3, p1-p3, p1+p3]) * np.exp(1j*abs(t1)) * 100
    cf[n-5] = 4 * np.exp(0.3*t1) * np.abs(t2)
    cf[n-4] = 4 * np.exp(0.3*t2) * np.abs(t1-t2)
    cf[n-1] = 10j * np.exp(0.4j*t2) * np.abs(t1)
    cf[n-1] = 8j * np.abs(t2) * (1/(t1-2))
    return cf

ALLOWED["g50"]=g50

def g51(z,a,state):
    t1, t2 = z[0], z[1]
    n = 1000
    cf = np.zeros(n, dtype=np.complex128)
    p1 = 1 * t2**3 - 1 * t2**2 + 1 * t2 - 1
    p2 = 1 * t2**3 + 1 * t2**2 + 1 * t2 + 1
    p3 = 1 * t1**3 + 1 * t2**2 + 1 * t1 + 1
    p4 = 1 * t1**3 + 1 * t1**2 + 1 * t1 + 1
    p5 = 1 * t1**3 + 1 * t1**2 + 1 * t1 - 1
    pp = np.array([1j*p1, p2**2, p4, p1+p2+p3, 1j*p1, p1**9, p2**9, p1*p2*p3*p4, p1, p1*p3, p1-p3, p1+p3])
    for i in range(n):
        k = i % len(pp)
        cf[i] = pp[k]
    cf[0] = -0.1j
    if np.abs(cf[n-1]) < 1e-10:
        cf[n-1] = 1
    return cf

ALLOWED["g51"]=g51

def g52(z,a,state):
    t1, t2 = z[0], z[1]
    n = 100
    cf = np.zeros(n, dtype=np.complex128)
    # Using numpy's roots function instead of R's polyroot
    cf[0] = -100j
    cf[1] = 0
    cf[2] = 0
    cf1 = np.array([1, t1, t2, t1],dtype=np.complex128)
    roots1 = np.roots(cf1)
    cf2 = np.array([10j, t2, t1, t2],dtype=np.complex128)
    roots2 = np.roots(cf2)
    cf[n//2-1:n//2+2] = 100 * np.resize(roots1, 3)
    cf[n-3:n] = 100 * np.resize(roots2, 3)
    return cf

ALLOWED["g52"]=g52

def g53(z,a,state):
    t1, t2 = z[0], z[1]
    n = 10
    cf = np.zeros(n, dtype=np.complex128)
    cf[0] = 100 * np.sin(t1)**3 * np.cos(t2)**2
    cf[1] = 100 * np.exp(1j * (t1 + t2)) - 10 * (t1 - t2)**2
    cf[2] = t1*t2*(t1 - t2) / (np.abs(t1) + np.abs(t2) + 1)
    cf[4] = (t1*t2*np.exp(1j * (t1**2-t2**2)))**3
    cf[6] = np.sqrt(np.abs(t1)) - np.sqrt(np.abs(t2)) + 1j * np.sin(t1*t2)
    cf[7] = 50 * np.abs(t1 - t2) * np.exp(1j * np.abs(t1 + t2))
    cf[8] = t1-abs(t2) if t1.imag > 0 else t2-abs(t1)
    cf[9] = (1j*t1*t2)**(0.1*t1*t2)
    return cf
    
ALLOWED["g53"]=g53

def g54(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(11, dtype=np.complex128)
    cf[0] = t1.real * np.exp(1j * t2)
    cf[2] = (t1 * t2).imag * np.exp(-1j * t2.real)
    cf[4] = (t1.real + t2.imag)**2 + 10j
    cf[6] = (t2.imag**3) / t1.real - 1j
    cf[8] = (t1 * t2).real * np.exp(1j * ((t1 + t2).imag**2))
    cf[9] = np.sum(cf[0:9])
    cf[10] = np.prod(cf[0:10])   
    return cf
    
ALLOWED["g54"]=g54

def g55(z,a,state):
    t1, t2 = z[0], z[1]
    n = 10
    cf = np.zeros(n, dtype=np.complex128)
    cf[0] = np.exp(1j * t1)
    cf[1] = (t1 + t2) * np.cos(t1) + 1j * np.sin(t2)
    cf[2] = t1**3 * t2**2 - 1j * t1**2 * t2**3
    cf[3] = np.log(t1 + 1j*t2)
    cf[4] = t1 * np.cos(t1) + t2 * np.sin(t2)
    cf[5] = t1**2 * t2 - t1 * t2**2
    cf[6] = 1j * t1**3 + t2**3
    cf[7] = (t1 + 1j * t2)**3 - t1 * t2
    cf[8] = t1 * t2 * (t1 - t2) * (t1 + t2)
    cf[9] = t1**3 * t2**2 * np.exp(1j * (t1 - t2))    
    return cf

ALLOWED["g55"]=g55

def g56(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(11, dtype=np.complex128)
    cf[0] = t1 * t2
    cf[1] = t1**2 - t2**2
    cf[2] = t1**3 + t2**3
    cf[3] = np.sin(t1) * np.cos(t2)
    cf[4] = np.exp(1j * (t1 - t2))
    cf[5] = np.log(np.abs(t1 + t2))
    cf[6] = t1**4 + 1j * t2**4
    cf[7] = (t1 * t2)**2
    cf[8] = (t1 + t2)/2
    cf[9] = t1**5 - t2**5
    cf[10] = np.exp(1j * (t1 * t2))
    return cf

ALLOWED["g56"]=g56

def g57(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(10, dtype=np.complex128)
    cf[0] = np.log(np.abs(t1 + t2)) * np.sin(t1 - t2)
    cf[1] = np.exp(t1.real) + np.exp(t2.imag)
    cf[2] = np.sqrt(np.abs(t1)) * np.cos(t2)
    cf[3] = np.sin((t1 + t2).imag**3) * np.exp((t1 - t2).real**2)
    cf[4] = np.tan((t1 * t2).imag) * np.cosh((t1 * t2).real)
    cf[5] = np.abs(t1 - t2) * np.sinh(np.angle(t1 + t2))
    cf[6] = (t1**3 - t2**2).imag * np.tan((t1 * t2).real)
    cf[7] = np.tanh(np.abs(t1) * np.abs(t2)) * np.sin(np.angle(t1 / t2))
    cf[8] = np.sign((t1 - t2).real) * np.cosh(np.angle(t1 * t2))
    cf[9] = np.arctan(1j * t1 / t2) + np.arcsinh(t1 + t2)
    return cf.astype(np.complex128)

ALLOWED["g57"]=g57

def g58(z,a,state):
    t1, t2 = z[0], z[1]
    n = 10
    cf = np.zeros(n, dtype=np.complex128)
    cf[0] = np.log(1j * t1**2 + 1)
    cf[1] = np.exp(1j * t1 * t2) + 1
    cf[2] = np.sin(t1 * t2) + t2
    cf[3] = np.cos(t1**3 + t2**2) * 1j
    cf[4] = t1 * t2 * (t1 - t2)
    cf[5] = np.sqrt(np.abs(t1 * t2)) * (t1 + t2)
    cf[6] = t1**3 * t2**3 * 1j
    cf[7] = (t2 - t1)/(t1 + t2) * 1j
    cf[8] = np.log(t1*t2) + np.sin(t1 + t2) * 1j
    cf[9] = 3**t1 * 2**t2
    return cf.astype(np.complex128)

ALLOWED["g58"]=g58

def g59(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(11, dtype=np.complex128)
    cf[0] = 100 * (t1**4 - t2**4)
    cf[1] = -100j * (t1 * t2 * (t1**2 + t2**2))
    cf[2] = 100 * np.sqrt((t1**2 * t2**2).real)
    cf[3] = 100 * (t1 - t2).imag
    cf[4] = 100j * (t1**2 + t2**2)
    cf[5] = 100 * np.exp(1j * np.abs(t1 - t2))
    cf[6] = 100 * np.sin((t1 + t2).real)
    cf[7] = -100j * np.cos((t1 - t2).imag)
    cf[8] = 100 * (t1*t2 / np.abs(t1*t2))
    cf[9] = 100 * np.sqrt(t1.real * t2.real) - 100j * np.sqrt(t1.imag * t2.imag)
    cf[10] = 100 * np.exp(1j * (np.angle(t1) - np.angle(t2)))
    return cf.astype(np.complex128)
    
ALLOWED["g59"]=g59

def g60(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    for k in range(25):
        cf[k] = np.sin(k+1)*t1/(1+abs(t2)) + np.cos(k+1)*t2/(1+abs(t1)) + np.sqrt(k+1)
    cf[0] = np.abs(t1)*abs(t2)
    cf[4] = np.angle(t1)*abs(t2)
    cf[9] = np.abs(t1)*np.angle(t2)
    cf[14] = np.abs(t1)*t2.real
    cf[19] = np.abs(t1)*t2.imag
    cf[24] = t1.real*abs(t2)
    return cf.astype(np.complex128)

ALLOWED["g60"]=g60

def g61(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = np.abs(t1 + t2)
    cf[1] = 2*t1.real*t2.imag
    cf[2] = np.angle(t1 + t2)
    cf[3] = np.conj(t1)*t2
    cf[4] = np.angle(t1)*np.angle(t2)
    
    for k in range(5, 21):
        cf[k] = np.abs(t1 + (-1)**(k+1)*t1**2/(k+1) + (-1)**(k+1)*t2**2/(k+1))
    
    cf[21] = cf[1] + cf[2] - cf[3] + cf[4]
    cf[22] = np.abs(cf[1]*cf[2]*cf[3]*cf[4])
    cf[23] = 1 + (np.conj(t1)*t2).real
    cf[24] = 1j + (np.conj(t1)*t2).imag
    return cf.astype(np.complex128)
    
ALLOWED["g61"]=g61

def g62(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0:5] = np.array([abs(t1 + t2)**(i+1) for i in range(5)])
    cf[5:10] = ((t1+2j*t2)**3).real * np.log(np.abs(np.conj(t1*t2)))
    cf[10:15] = ((t1-t2)**2).imag / np.angle(t1*t2)
    cf[15:20] = np.abs(cf[5:10])**0.5 + np.angle(cf[0:5])
    cf[20:25] = np.array([abs(t1 * t2)**(i+1) for i in range(5)])
    return cf

ALLOWED["g62"]=g62


def g63(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    for i in range(25):
        numerator = (t1 * (i+1) + t2**((i+1)/2))
        denominator = np.abs(t1 * (i+1) + t2**(i+1))
        if denominator>0 :
            cf[i] =  numerator/denominator 
        else:
            cf[i] = 0
    cf[2] = t1.real + t2.imag
    cf[6] = np.abs(np.exp(1j * np.angle(t1 * t2)))
    cf[10] = (t1 * t2).real + (t1 / t2).imag
    cf[12] = np.angle(t1 + 4*t2) / np.abs(np.conj(t1 - 4*t2))
    cf[16] = np.abs(np.exp(1j * np.angle(t1 - t2)))
    cf[18] = (t1 / t2).real - (t1 * t2).imag
    cf[22] = np.abs(np.exp(1j * np.angle(t1 + t2)))
    return cf

ALLOWED["g63"]=g63

def g64(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = t1 + t2
    cf[1] = np.exp(1j * np.angle(t1 * np.conj(t2)))
    cf[2] = np.abs(t1) * np.abs(t2)
    for k in range(3, 25):
        cf[k] = (cf[k-1].real + 1j * cf[k-1].imag) * np.exp(1j * np.angle(cf[k-2]))
        if cf[k].imag == 0:
            cf[k] = cf[k] + 1e-10
        cf[k] = np.log(np.abs(cf[k])) / 2 + cf[k] * 1j
    return cf

ALLOWED["g64"]=g64

def g65(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = t1 + t2
    cf[1] = t1 - t2
    cf[2] = t1 * t2
    cf[3] = t1 / t2
    cf[4] = np.abs(t1) + np.abs(t2)
    cf[5] = np.abs(t1) - np.abs(t2)
    cf[6] = np.angle(t1) + np.angle(t2)
    cf[7] = np.angle(t1) - np.angle(t2)
    cf[8] = t1**2 + t2**2
    cf[9] = t1**3 + t2**3
    cf[10] = t1**4 + t2**4
    cf[11] = np.log(np.abs(t1)**2 + np.abs(t2)**2 + 1)
    cf[12] = np.exp(np.abs(t1) + np.abs(t2))
    cf[13] = np.conj(t1) * t2
    cf[14] = t1 * np.conj(t2)
    cf[15] = np.conj(t1) * np.conj(t2)
    cf[16] = np.abs(t1 - t2)
    cf[17] = np.abs(t1 + t2)
    cf[18:24] = np.abs(t1 + t2)
    return cf

ALLOWED["g65"]=g65

def g66(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = t1 + t2
    cf[1] = t1 - t2
    cf[2] = t1 * t2
    cf[3] = t1 / t2
    cf[4] = np.abs(t1) + np.abs(t2)
    cf[5] = np.abs(t1) - np.abs(t2)
    cf[6] = np.angle(t1) + np.angle(t2)
    cf[7] = np.angle(t1) - np.angle(t2)
    cf[8] = t1**2 + t2**2
    cf[9] = t1**3 + t2**3
    cf[10] = t1**4 + t2**4
    cf[11] = np.log(np.abs(t1)**2 + np.abs(t2)**2 + 1)
    cf[12] = np.exp(np.abs(t1) + np.abs(t2))
    cf[13] = np.conj(t1) * t2
    cf[14] = t1 * np.conj(t2)
    cf[15] = np.conj(t1) * np.conj(t2)
    cf[16] = np.abs(t1 - t2)
    cf[17] = np.abs(t1 + t2)
    cf[18:25] = np.abs(t1 + t2)
    return cf
    
ALLOWED["g66"]=g66

#FIME
def g67(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = t1 + t2
    cf[1] = t1 * t2
    cf[2] = (t1 + t2)**2
    cf[3] = np.abs(t1) * np.abs(t2)
    for k in range(4, 9):
        r = (t1 + 1j*t2)**(k)
        cf[k] = r / np.abs(r)
    cf[9] = np.log(np.abs(t1)) / np.log(np.abs(t2))
    cf[10] = np.exp(np.angle(t1)) * np.exp(np.angle(t2))
    for k in range(11, 16):
        cf[k] = (np.conj(t1) / np.conj(t2)) * 1j**(k-2)
    cf[16] = t1.real * t2.imag
    cf[17] = t1.imag * t2.real
    for k in range(18, 23):
        z = (t1 + 1j*t2)**(k)
        cf[k] = np.sin(np.angle(z))
    cf[23] = np.cos(np.angle(t1) + np.angle(t2))
    cf[24] = np.tanh(np.abs(t1*t2))
    return cf
    
ALLOWED["g67"]=g67

def g68(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    for k in range(25):
        cf[k] = np.abs(t1)**((k+1)/2) * (np.cos((k+1) * np.angle(t2)) + 1j * np.sin((k+1) * np.angle(t2)))
    cf[4] = cf[4] + (np.log(np.abs(t1)) + np.log(np.abs(t2))) / 2
    cf[9] = cf[9] + np.conj(t1 * t2)
    cf[14] = cf[14] + np.abs(t2 - t1)**2
    cf[19] = cf[19] + (np.sin(np.angle(t1)) / np.cos(np.angle(t2)))**3
    cf[24] = cf[24] + ((1j * t1 - t2)**2 / (1 + np.abs(t1 + t2)**3))**4
    return cf
    
ALLOWED["g68"]=g68

def g69(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    for i in range(25):
        cf[i] = (t1.real**(i+1) + t2.imag**(25-i))/(1 + np.abs(t1 + t2)) * np.exp(1j * np.angle(t1 + t2))
    cf[2] = 3 * np.conj(t1**2 + t2)
    cf[6] = 7 * np.abs(t1 + t2)
    cf[10] = 11 * (t1/t2 + np.conj(t2/t1))
    cf[16] = 17 * (np.abs(t1)*abs(t2))/(np.abs(t1 + t2))**2
    cf[22] = 23 * (np.conj(t1) + t2) / (1 + np.abs(t1 * np.conj(t2)))
    cf[24] = 25 * (np.conj(t1) + np.conj(t2)) / np.abs(t1*t2)
    return cf

ALLOWED["g69"]=g69

def g70(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0:5] = np.real(t1) * np.arange(1, 6) - np.imag(t2) * np.arange(1, 6)
    cf[5] = np.abs(t1) * np.abs(t2)
    cf[6:11] = np.angle(t1+t2) * np.arange(6, 11)
    cf[11] = np.conj(t1) + np.conj(t2)
    cf[12:17] = np.real(t1 + 1j * t2) * np.arange(1, 6)
    cf[17] = np.angle(t1) * np.angle(t2)
    cf[18:23] = np.imag(t1 - 1j * t2) * np.arange(1, 6)
    cf[23] = np.conj(t1 * t2)
    cf[24] = np.abs(cf[11]) + np.angle(cf[17])
    return cf
    
ALLOWED["g70"]=g70

def g71(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = t1 + t2
    for i in range(1, 25):
        cf[i] = np.real(t1)*np.imag(t2)*(np.abs(t1)*np.angle(t2))**(i+1) / (np.conj(t1)*np.conj(t2))**i
        if np.isinf(np.abs(cf[i])) or np.isnan(cf[i]):
            cf[i] = 0
    cf[4] = cf[4] + np.log(np.abs(cf[2]))*cf[1]
    cf[9] = cf[9] + cf[0]*np.conj(cf[3])
    cf[14] = cf[14] + cf[1]*cf[2]
    cf[19] = cf[19] + cf[3]*cf[0]
    cf[24] = cf[24] + cf[4]
    return cf
    
ALLOWED["g71"]=g71

def g72(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = t1 + t2
    for k in range(1, 25):
        v = np.sin(k * cf[k-1] + np.angle(t2**k)) + np.cos(k * np.abs(t1))
        cf[k] = v / (np.abs(v) + 1e-10)
    cf[9] = t1 * t2 - np.abs(t2)**2 + 1j * np.angle(t1)
    cf[14] = np.conj(t1)**3 - np.angle(t2)**3 + 1j * np.abs(t2)
    cf[19] = np.abs(t2)**3 + t1**2 + t2**2 + 1j * np.angle(t2)**2
    cf[24] = np.abs(t1 * t2) + np.angle(t1)**5 + 1j * np.abs(t1)**5
    return cf
    
ALLOWED["g72"]=g72

def g73(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = t1 + t2
    cf[1] = t1 - t2
    cf[2] = t1*t2
    cf[3] = t1/t2
    cf[4] = (t1 + t2)**2
    cf[5] = (t1 - t2)**2
    cf[6] = (t1**2 + t2**2)
    cf[7] = (t1**2 - t2**2)
    cf[8] = (t1**2 + t2**2)**2
    cf[9] = (t1**2 - t2**2)**2
    cf[10] = (t1 + t2 + 1j)**2
    cf[11] = (t1 - t2 - 1j)**2
    cf[12] = (t1 + 1j*t2)**3
    cf[13] = (1j*t1 - t2)**3
    cf[14] = (t1 + t2)**3 + (t1 - t2)**3
    cf[15] = (t1*t2)**3 - 1j*t1*t2
    cf[16] = (t1/t2)**4 + 1j*t1/t2
    cf[17] = (t1*t2 + 1j)**4 - t1*t2
    cf[18] = (t1 + t2 + 1j)**5 - (t1 + t2)
    cf[19] = (t1 - t2 - 1j)**5 + (t1 - t2)
    cf[20] = (t1 + 1j*t2)**6 - 1j*t1*t2
    cf[21] = (1j*t1 - t2)**6 + t1*t2
    cf[22] = (t1 + t2)**7 - (t1 - t2)**7
    cf[23] = (t1*t2)**8 - (t1/t2)**8
    cf[24] = np.log(np.abs(t1 + 1j*t2))
    return cf

ALLOWED["g73"]=g73

def g74(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = t1*t2
    cf[1] = np.real(t1) + 2*np.imag(t2)
    cf[2] = t1/np.abs(t2)
    if np.abs(t2) != 0:
        cf[3] = t2/np.abs(t1)
    cf[4] = np.angle(t1)*np.angle(t2)
    cf[5] = np.abs(np.conj(t1)*t2)
    cf[6] = np.abs(t1-t2)
    cf[7] = np.angle(t1*t2)+np.conj(np.angle(t1*t2))
    cf[8] = np.log(np.abs(t1)) + np.log(np.abs(t2))
    cf[9] = np.abs(t1)*np.imag(t2)
    cf[14] = np.abs(t1)**2 + np.abs(t2)**2
    cf[19] = np.real(t1)**3 + np.imag(t2)**2
    cf[24] = 1j*(np.abs(t1) + np.abs(t2))
    return cf

ALLOWED["g74"]=g74


def g75(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = np.abs(t1) * np.abs(t2)
    cf[1] = np.abs(t1 + t2)
    cf[2] = np.abs(np.conj(t1) + np.conj(t2))
    cf[3] = np.angle(t1) * np.angle(t2)
    cf[4] = np.angle(np.conj(t1) + np.conj(t2))
    cf[5] = np.exp(1j * np.angle(t1 + t2))
    cf[6] = np.exp(1j * np.angle(np.conj(t1) + np.conj(t2)))
    cf[7] = np.sin(np.abs(np.conj(t1)) * np.abs(np.conj(t2)))
    cf[8] = np.cos(np.angle(t1 + t2))
    cf[9] = np.tanh(np.abs(np.conj(t1 + t2)))
    cf[10:15] = np.abs(t1)**((np.arange(11,16)/10)) * np.abs(t2)**((np.arange(15,10,-1)/10))
    cf[15:20] = (np.arange(16,21)) * (np.angle(t1) + np.angle(t2)) / 2
    cf[20:25] = np.real(t1)**2 + np.imag(t2)**2 + np.arange(21,26)
    return cf.astype(np.complex128)

ALLOWED["g75"]=g75

def g76(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = t1 + t2
    for k in range(1, 25):
        v = (t1+t2)**(k+1) + np.sin(k * cf[k-1]) + np.log(np.abs(k * t1)) - np.log(np.abs((k+1) * t2))
        cf[k] = v / np.abs(v)
    return cf

ALLOWED["g76"]=g76

def g77(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0:10] = np.arange(1, 11) * t1 + np.arange(11, 21) * 1j * t2
    cf[10:20] = (t1 + 1j * t2)**2 * np.arange(11, 21)
    cf[20:25] = (np.abs(t1) + np.angle(t2)) * np.arange(1, 6)
    return cf

ALLOWED["g77"]=g77

def g78(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    for k in range(25):
        cf[k] = (k + 1 + t1) / (k + 1 + t2)
    cf[4] = cf[4] + np.log(np.abs(t1 + t2))
    cf[9] = cf[9] + np.sin(np.real(t1)) + np.cos(np.imag(t2))
    cf[14] = cf[14] + np.abs(cf[13])**2 + np.angle(cf[12])**2
    cf[19] = cf[19] + np.abs(np.real(t2) * np.imag(t1))
    cf[24] = cf[24] + np.abs(t1 + np.conj(t2))
    return cf

ALLOWED["g78"]=g78

def g79(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0:10] = np.arange(1, 11) * t1 + np.arange(11, 21) * 1j * t2
    cf[10:20] = (t1 + 1j * t2)**2 * np.arange(11, 21)
    cf[20:25] = (np.abs(t1) + np.angle(t2)) * np.arange(1, 6)
    return cf

ALLOWED["g79"]=g79

def g80(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    for k in range(25):
        cf[k] = (k + 1 + t1) / (k + 1 + t2)
    cf[4] = cf[4] + np.log(np.abs(t1 + t2))
    cf[9] = cf[9] + np.sin(np.real(t1)) + np.cos(np.imag(t2))
    cf[14] = cf[14] + np.abs(cf[13])**2 + np.angle(cf[12])**2
    cf[19] = cf[19] + np.abs(np.real(t2) * np.imag(t1))
    cf[24] = cf[24] + np.abs(t1 + np.conj(t2))
    return cf
    
ALLOWED["g80"]=g80

def g81(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    for i in range(25):
        cf[i] = (t1 * (i+1) + t2**((i+1)/2)) / np.abs(t1 * (i+1) + t2**(i+1))
    cf[2] = np.real(t1) + np.imag(t2)
    cf[6] = np.abs(np.exp(1j * np.angle(t1 * t2)))
    cf[10] = np.real(t1 * t2) + np.imag(t1 / t2)
    cf[12] = np.angle(t1 + 4*t2) / np.abs(np.conj(t1 - 4*t2))
    cf[16] = np.abs(np.exp(1j * np.angle(t1 - t2)))
    cf[18] = np.real(t1 / t2) - np.imag(t1 * t2)
    cf[22] = np.abs(np.exp(1j * np.angle(t1 + t2)))
    return cf

ALLOWED["g81"]=g81

#FIXME crash
def g82(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = t1 + t2
    for k in range(1, 25):
        cf[k] = (1j * cf[k-1]**2 + 2j * t1) / (t2 + 1)
        if np.abs(cf[k]) == 0:
            cf[k] = 1
        if np.isinf(np.abs(cf[k])):
            cf[k] = 1
    
    cf[4] = cf[4] + np.sin(cf[1].real) + np.log(np.abs(t1))
    cf[9] = cf[9] + np.cos(cf[4].imag) + np.log(np.abs(t2))
    cf[14] = cf[14] + np.tan(cf[9].real) + np.log(np.abs(t1 * t2))
    cf[19] = cf[19] + np.arctan(cf[14].imag) + np.log(np.abs(np.conj(t1) * t2))
    cf[24] = cf[24] + np.sin(np.angle(cf[19])) + np.log(np.abs(np.conj(t1) * np.conj(t2)))
    return cf.astype(np.complex128)

ALLOWED["g82"]=g82

# FIXME 
def g83(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = t1.real * t2.real + t1.imag * t2.imag
    cf[1] = np.abs(t1) * np.abs(t2)
    cf[2] = np.angle(t1) + np.angle(t2)
    cf[3] = np.conj(t1).real + np.conj(t2).imag
    
    for k in range(4, 25):
        cf[k] = (cf[k-1] * cf[k-4] + cf[k-3] * cf[k-2]) / np.abs(cf[k-1] * cf[k-4] + cf[k-3] * cf[k-2])
    
    cf[12] = cf[0] / cf[1] + cf[2] / cf[3]
    cf[18] = np.log(np.abs(cf[0] * cf[1])+1) / np.log(np.abs(cf[2] * cf[3])+1)
    cf[24] = cf[4] * cf[9] * cf[14] * cf[19] / np.abs(cf[4] * cf[9] * cf[14] * cf[19])
    return cf
    
ALLOWED["g83"]=g83

def g84(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    primes = np.array([2, 3, 5, 7, 11, 13, 17])
    cf[0:7] = primes * (t1 + t2)
    cf[14] = np.sum(primes[0:5]) + (t1+1j*t2)**2
    cf[24] = np.prod(primes[0:4]) / np.abs(t1 + 1j * t2)
    
    for k in range(7, 14):
        try:
            v = np.log(np.abs(t1)) * np.sin(cf[k-1]) + np.log(np.abs(t2)) * np.cos(cf[k-1])
            if not np.isinf(v):
                cf[k] = v
        except:
            pass
    cf[15:24] = np.real(t1) + np.imag(t2)
    return cf

ALLOWED["g84"]=g84

def g85(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)   
    for k in range(25):
        cf[k] = np.sin(t1)+(1j*np.cos(t2.real**2-t2.imag**2))/(np.sqrt(np.abs(t1)**2+abs(t2)**2))
    cf[2] = cf[6]*cf[10]*cf[18]/cf[22]
    cf[4] = cf[9] + cf[14] + cf[19] - cf[24]
    cf[8] = 1j*t1*t2*(t1 - t2)
    cf[12] = cf[4]*t1/(1+abs(t2))
    cf[16] = np.conj(cf[8])/t2
    cf[20] = np.log(np.abs(cf[4]*t2/(1+abs(t1))))
    return cf

ALLOWED["g85"]=g85

def g86(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = (t1**3).real*(t2**3).real
    cf[1] = (t1**3).imag*(t2**3).imag
    
    for k in range(2, 25):
        if k % 3 == 0:
            cf[k] = (t1+1j*t2)**(k/3) / k
        else:
            cf[k] = np.conj(cf[k-1]) ** 2 + np.abs(t1)*abs(t2)
    return cf.astype(np.complex128)

ALLOWED["g86"]=g86


def g87(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(51, dtype=np.complex128)
    cf[0] = t1 + t2
    cf[1] = 1 + t1*t2 + np.log(np.abs(t1 + t2) + 1)
    cf[2] = t1 + t2 + np.log(np.abs(1 - t1 * t2) + 1)
    for i in range(3, 51):
        cf[i] = i * t1 + (51 - i) * t2 + np.log(np.abs(t1 - t2 * i) + 1)
    cf[10] = cf[0] + cf[9] - np.sin(t1)
    cf[20] = cf[30] + cf[40] - np.cos(t2)
    cf[30] = cf[20] + cf[40] + np.sin(t1)
    cf[40] = cf[30] + cf[20] - np.cos(t2)
    cf[50] = cf[40] + cf[20] + np.sin(t2)
    return cf

ALLOWED["g87"]=g87

def g88(z,a,state):
    t1, t2 = z[0].real, z[0].imag 
    cf = np.zeros(51, dtype=np.complex128)
    primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71])
    cf[0:20] = (t1**primes[0:20] + t2**(primes[0:20] + 1)) / np.abs(t1 - t2)
    for k in range(20, 51):
        cf[k] = (np.cos(k * np.log(np.abs(t1) + 1)) + np.sin(k * np.log(np.abs(t2) + 1))) / k
    cf[50] = np.abs(t1) * np.abs(t2) * (np.cos(np.angle(t1) * np.angle(t2)) - 1j * np.sin(np.angle(t1) * np.angle(t2)))
    return cf.astype(np.complex128)

ALLOWED["g88"]=g88

def g89(z,a,state):
    t1, t2 = z[0].real, z[0].imag 
    cf = np.zeros(51, dtype=np.complex128)
    for i in range(51):
        cf[i] = np.cos(i*t1) + np.sin(i*t2)   
    cf[0] = cf[0]*t1**50
    cf[1] = cf[1]*t2**49
    for i in range(2, 51):
        cf[i] = cf[i]*t1**(51-1j)*t2**(i-2)
    return cf.astype(np.complex128)

ALLOWED["g89"]=g89

def g90(z,a,state):
    t1, t2 = z[0].real, z[0].imag 
    cf = np.zeros(51, dtype=np.complex128)
    cf[0] = t1 + t2
    for i in range(1, 51):
        cf[i] = ((np.cos(t1)/np.sin(t2))**i + (np.sin(t1)/np.cos(t2))**(2j)) * np.angle(t1 + t2)**i
    cf[1:51] = np.log(np.abs(cf[1:51]) + 1) / np.log(i+1)
    cf[4] = cf[4] * np.abs(t1 + t2)
    cf[9] = cf[9] * (t1 * t2.conjugate()).real
    cf[19] = cf[19] * (t2 * t1.conjugate()).imag
    cf[29] = cf[29] * np.angle(t1 + t2)
    cf[39] = cf[39] * np.abs(t1 + t2)
    cf[49] = cf[49] * (t1 * t2.conjugate()).real
    return cf.astype(np.complex128)
    
ALLOWED["g90"]=g90

def g91(z,a,state):
    t1, t2 = z[0].real, z[0].imag 
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1 + t2
    for k in range(1, 71):
        r = np.abs(t1)**k + np.abs(t2)**(71-k)
        theta = np.angle(t1)**k - np.angle(t2)**(71-k)
        cf[k] = r * np.cos(theta) + r * np.sin(theta)*1j
    cf[2:70] = cf[2:70] + np.log(np.abs(t2-t1)+1)
    cf[70] = cf[70] + (t1*t2).conjugate()
    return cf.astype(np.complex128)

ALLOWED["g91"]=g91

#FIXME
def g92(z,a,state):
    t1, t2 = z[0].real, z[0].imag 
    cf = np.zeros(71, dtype=np.complex128)
    for i in range(71):
        cf[i] = (np.sin(t1)*np.cos(t2)**i) + (np.cos(t1)*np.sin(t2)**i) + np.log(np.abs(t1+t2)+1)**i
    cf[0] = cf[0] + t1.real*t2.imag + t1.imag*t2.real
    cf[1] = cf[1] + np.abs(t1) * np.abs(t2)
    cf[2] = cf[2] + np.angle(t1) / np.angle(t2)
    cf[3] = cf[3] + np.angle(t2) / np.angle(t1)
    cf[4] = cf[4] + t1.real / np.abs(t2)
    cf[5] = cf[5] + t2.imag / np.abs(t1)
    cf[70] = cf[70] + (t1+t2).conjugate()
    return cf.astype(np.complex128)

ALLOWED["g92"]=g92

def g93(z,a,state):
    t1, t2 = z[0].real, z[0].imag 
    cf = np.zeros(71, dtype=np.complex128)
    angles = np.linspace(0, 2*np.pi, 35)
    for i in range(35):
        cf[i] = np.cos(angles[i] * t1) + np.sin(angles[i] * t2) / np.abs(t1)
    for i in range(35, 71):
        cf[i] = (t1*1j + t2*(71-1j))**3 / (t1*1j + 1j*t2*(71-1j))**2
    return cf.astype(np.complex128)
    
ALLOWED["g93"]=g93

def g94(z,a,state):
    t1, t2 = z[0].real, z[0].imag 
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1.real * t2.imag
    for k in range(1, 71):
        cf[k] = np.sin(k * cf[k-1]) + np.cos(k * t1)
        cf[k] = cf[k] / np.abs(cf[k])
    cf[30] = np.abs(cf[14])**2 + np.angle(t2)**2
    cf[40] = cf[20] * (np.abs(t1) + np.log(np.abs(t2)+1))
    cf[50] = cf[30] + np.log(np.abs(t1*t2)+1)
    cf[60] = np.abs(t1 + t2) * np.angle(cf[30])
    cf[70] = np.abs(cf[34]) / (t1 + 1j * t2)
    return cf

ALLOWED["g94"]=g94

def g95(z,a,state):
    t1, t2 = z[0].real, z[0].imag 
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1.real * t2.imag
    cf[1] = np.abs(t1) / np.angle(t2)
    cf[2] = np.sin(t1) * np.cos(t2)
    for i in range(3, 71):
        cf[i] = cf[i-1] + cf[i-3] * np.log(np.abs(cf[i-2]+1))
        if i % 2 == 0:
            cf[i] = cf[i] + np.abs(cf[i-1] * t1)
        else:
            cf[i] = cf[i] - np.abs(cf[i-1] * t2.conjugate())
    return cf

ALLOWED["g95"]=g95

def g96(z,a,state):
    t1, t2 = z[0].real, z[0].imag 
    cf = np.zeros(71, dtype=np.complex128)
    for i in range(71):
        cf[i] = ((i+1)**2 + 3j + 1)*(t1**2) + ((i+1)**3 - (i+1)**2 + 1)*(t2**2) + np.sin(t1*1j + t2) + np.log(np.abs(t1*1j - t2)+1)
        if (i+1) % 2 == 0:
            cf[i] = cf[i] + (t1+1j*t2)**2
        elif (i+1) % 3 == 0:
            cf[i] = cf[i] + np.abs(t1+1j*t2)**3
        else:
            cf[i] = cf[i] + (t1+1j*t2).real**4
    cf[0] *= 10000
    cf[1] *= 1000
    cf[2] *= 100
    cf[3] *= 10
    return cf

ALLOWED["g96"]=g96

def g97(z,a,state):
    t1, t2 = z[0].real, z[0].imag 
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1 * t2 * np.abs(t1) * np.abs(t2) + np.log(np.abs(t1)+1) + np.log(np.abs(t2)+1)
    cf[1:3] = (t1 + t2) * np.array([1, -1])
    for i in range(3, 10):
        cf[i] = np.sin(i*t1*t2) / ((i/t1/t2).real)
    for i in range(10, 20):
        cf[i] = np.cos(i*t1*t2) / ((i/t1/t2).imag)
    for i in range(20, 30):
        cf[i] = np.log(np.abs(t1**i+t2**i) + 1)
    cf[30:40] = (t1 + t2) ** np.arange(1, 11)
    for i in range(40, 50):
        cf[i] = np.abs(i/t1) * np.abs(i/t2)
    v = 1.0 + 0.0j
    for i in range(30):
        s = -1.0 if (i & 1) == 0 else 1.0   # (-1, +1, -1, +1, …)
        v *= cf[i] * s
    cf[50:60] = v
    for i in range(60, 70):
        cf[i] = np.abs(t1)**i - np.abs(t2)**i + np.log(np.abs(t1**i+t2**i) + 1) + np.angle(cf[i-10])
    cf[70] = np.sum(cf[:70]).conjugate()
    return cf.astype(np.complex128)

ALLOWED["g97"]=g97

def g98(z,a,state):
    t1, t2 = z[0].real, z[0].imag 
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = np.log(np.abs(t1) + 1) + np.log(np.abs(t2) + 1)
    for i in range(1, 71):
        cf[i] = np.sin(i * np.angle(t1 * t2)) * np.abs(cf[i-1]**(i-1)) * np.cos(np.abs(t1) + np.abs(t2))
    for i in range(34, 55):
        cf[i] = cf[i] * np.abs(t1 + t2) * cf[i-1]
    cf[4:26] = cf[4:26] * cf[0] * (np.abs(t1) * np.abs(t2))**np.arange(1, 23)
    cf[55:71] = cf[55:71] * cf[0] / (np.abs(t1) * np.abs(t2))**np.arange(15, 0, -1)
    indices = np.array([0, 14, 29, 44, 59, 70],dtype=np.intp)
    cf[indices] = cf[indices] + np.conjugate(t1 * t2)
    return cf

ALLOWED["g98"]=g98

def g99(z,a,state):
    t1, t2 = z[0].real, z[0].imag 
    cf = np.zeros(71, dtype=np.complex128)
    complex_val = np.sin(t1 + t2) + np.cos(t1 - t2)
    for k in range(71):
        if (k+1) % 2 == 0:
            cf[k] = complex_val / np.abs(k+1) - np.abs(t1)
        else:
            cf[k] = complex_val * np.abs(k+1) + np.log(np.abs(k+1) + 1) + t2.imag - t1.real
        if (k+1) % 3 == 0 and k > 0:
            cf[k] = cf[k] + 3 * cf[k-1]
        if (k+1) % 5 == 0 and k > 1:
            cf[k] = cf[k] + 5 * cf[k-2]
        if (k+1) % 7 == 0 and k > 2:
            cf[k] = cf[k] + 7 * cf[k-3]
    return cf

ALLOWED["g99"]=g99

def p7f(z,a,state):
    t1, t2 = z[0].real, z[0].imag 
    pi2  =  2 * np.pi
    n    =  23 # was 23
    tt1  =  np.exp(1j * pi2 * t1)
    ttt1 =  np.exp(1j * pi2 * tt1)
    v  =  np.linspace(np.real(tt1), np.real(ttt1), n)
    if t2 < 0.1:
        f = 10 * t1 * np.exp(1j * np.sin(11 * pi2 * v))
    elif 0.1 <= t2 < 0.2:
        f =  100 * np.exp(1j * np.sin(17 * pi2 * v))
    elif 0.2 <= t2 < 0.3:
        f =  599 * np.exp(1j * np.cos(83 * pi2 * v))
    elif 0.3 <= t2 < 0.4:
        f =  443 * np.exp(1j * np.sin(179 * pi2 * v))
    elif 0.4 <= t2 < 0.5:
        f =  293 * np.exp(1j * np.sin(127 * pi2 * v))
    elif 0.5 <= t2 < 0.6:
        f =  541 * np.exp(1j * np.sin(103 * pi2 * v))
    elif 0.6 <= t2 < 0.7:
        f =  379 * np.exp(1j * np.sin(283 * pi2 * v))
    elif 0.7 <= t2 < 0.8:
        f =  233 * np.exp(1j * np.sin(3 * pi2 * v))
    elif 0.8 <= t2 < 0.9:
        f =  173 * np.exp(1j * np.sin(5 * pi2 * v))
    else:
        f =  257 * np.exp(1j * np.sin(23 * pi2 * v))
    f[n-1] +=  211 * np.exp(1j * pi2 * (1/7) * t2 )
    return f
    
ALLOWED["p7f"]=p7f

def g221(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 25
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(1, degree + 1):
        mag = np.log(np.abs(t1) + j**1.3) * np.abs(np.sin(j * np.pi / 4)) + np.abs(t2) * np.cos(j * np.pi / 6)
        angle = np.angle(t1) * np.sin(j / 3) + np.angle(t2) * np.cos(j / 5) + np.sin(j * np.pi / 7)
        cf[j - 1] = mag * (np.cos(angle) + 1j * np.sin(angle))
    for k in range(degree + 1, 36):
        cf[k - 1] = np.log(k + 1) * (np.sin(k * np.angle(t1)) + 1j * np.cos(k / 2))
    return  cf

ALLOWED["g221"]=g221

def g224(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(51, dtype=np.complex128)
    cf[0] = t1 + t2
    for i in range(1, 51):
        sint2 = np.sin(t2)
        cost1 = np.cos(t1)
        sint1 =  np.cos(t1)
        cost2 = np.cos(t2)
        if abs(sint2)>1e-10 and abs(cost2)>1e-10:
            cf[i] = ((np.cos(t1)/np.sin(t2))**i + (np.sin(t1)/np.cos(t2))**(2j)) * np.angle(t1 + t2)**i
        else:
            cf[i] = 0
    cf[1:51] = np.log(np.abs(cf[1:51]) + 1) / np.log(i+1)
    cf[4] = cf[4] * np.abs(t1 + t2)
    cf[9] = cf[9] * (t1 * t2.conjugate()).real
    cf[19] = cf[19] * (t2 * t1.conjugate()).imag
    cf[29] = cf[29] * np.angle(t1 + t2)
    cf[39] = cf[39] * np.abs(t1 + t2)
    cf[49] = cf[49] * (t1 * t2.conjugate()).real
    return  cf

ALLOWED["g224"]=g224

def g227(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0:5] = np.array([abs(t1 + t2)**(i+1) for i in range(5)])
    cf[5:10] = ((t1+2j*t2)**3).real * np.log(np.abs(np.conj(t1*t2)))
    if  np.abs(np.angle(t1*t2))>1e-10:
        cf[10:15] = ((t1-t2)**2).imag / np.angle(t1*t2)
    else:
         cf[10:15] = ((t1-t2)**2).imag
    cf[15:20] = np.abs(cf[5:10])**0.5 + np.angle(cf[0:5])
    cf[20:25] = np.array([abs(t1 * t2)**(i+1) for i in range(5)])
    return  cf

ALLOWED["g227"]=g227

def g230(z,a,state):
    t1, t2 = z[0], z[1]
    n = 10
    cf = np.zeros(n, dtype=np.complex128)
    cf[0] = 100 * np.sin(t1)**3 * np.cos(t2)**2
    cf[1] = 100 * np.exp(1j * (t1 + t2)) - 10 * (t1 - t2)**2
    cf[2] = t1*t2*(t1 - t2) / (np.abs(t1) + np.abs(t2) + 1)
    cf[4] = (t1*t2*np.exp(1j * (t1**2-t2**2)))**3
    cf[6] = np.sqrt(np.abs(t1)) - np.sqrt(np.abs(t2)) + 1j * np.sin(t1*t2)
    cf[7] = 50 * np.abs(t1 - t2) * np.exp(1j * np.abs(t1 + t2))
    cf[8] = t1-abs(t2) if t1.imag > 0 else t2-abs(t1)
    cf[9] = (1j*t1*t2)**(0.1*t1*t2)
    return  cf

ALLOWED["g230"]=g230

def g232(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree +1, dtype=np.complex128)
    for j in range(0, degree +1):
        r_part = t1.real * j**2 - t2.real * np.sqrt(j +1)
        im_part = (t1.imag + t2.imag) * np.log(j +2)
        magnitude = np.abs(t1)**(j %3 +1) + np.abs(t2)**(degree -j)
        angle = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j)
        cf[j] = (r_part +1j * im_part) * magnitude * np.exp(1j * angle)
    return  cf

ALLOWED["g232"]=g232

# serp:runs:zero:one,zz3,cf6,g2863:12,aberth,rrot:0.5
def g2863(z,a,state):
    t1, t2 = z[0], z[1]
    a0 = a[0]
    n =  int(a0.real) # original is 11
    cf = np.zeros(n, dtype=np.complex128)
    for k in range(n):
        freq_t1 = (k+1)*np.angle(t1)
        freq_t2 = (k+1)*np.angle(t2)
        cf[k] = (np.sin(freq_t1)+1j*np.cos(freq_t2)) * np.exp(-abs(t1*t2)*k/n)
    for k in range(1, n-1):
        cf[k] = (cf[k-1] + cf[k+1])*0.5*(t1 + t2)
    cf = cf[np.argsort(np.abs(np.cumsum(cf)))] 
    return cf

ALLOWED["g2863"]=g2863

# n=71
def g2864(z,a,state):
    t1, t2 = z[0], z[1]
    a0 = a[0]
    n =  int(a0.real) # original is 71
    cf = np.zeros(n, dtype=np.complex128)
    for k in range(n):
        a = [t1.real, t1.imag]
        b = [t2.real, t2.imag]
        dot = a[0]*b[0] + a[1]*b[1]
        wedge = a[0]*b[1] - a[1]*b[0]
        gp = dot + 1j*wedge
        cf[k] = gp**(k+1)
    cf[::2] *= -1
    return cf

ALLOWED["g2864"]=g2864

def g2864a(z,a,state):
    t1, t2 = z[0], z[1]
    a0 = a[0]
    n =  int(a0.real) # original is 71
    cf = np.zeros(n, dtype=np.complex128)
    st = t1*t1+t2*t2
    cf = np.empty(n, dtype=np.complex128)
    p = st  
    for k in range(n):
        cf[k] = ((k & 1) * 2 - 1) * p + 1j   # -,+,-,+,...
        p *= st
    return cf

ALLOWED["g2864a"]=g2864a

def p1(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(36, dtype=np.complex128)
    for i in range(1, 37):
        cf[i-1] = np.sin(t1**(i/2)) * np.cos(t2**(i/3)) + (i**2) * t1 * t2 + np.log(np.abs(t1 + t2) + 1) * 1j * i
    cf[10] = t1 * t2 * np.real(cf[6]) + np.imag(cf[18]) * t1**3
    cf[21] = t2 * cf[10] + np.real(cf[34]) * t1**3
    cf[32] = cf[21] - np.real(cf[16]) * t1**2
    return cf

ALLOWED["p1"]=p1

def p2(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(36, dtype=np.complex128)
    cf[0] = t1 + t2
    for k in range(2, 37):
        v = np.sin(k * cf[k-2]) + np.cos(k * t1) + np.real(k * t2) * np.imag(k * cf[k-2])
        cf[k-1] = v / np.abs(v)
    cf[17] = t1**2 + np.real(t1) * t2 - np.imag(t2**2)
    cf[31] = 2 * (t1 + t2) - np.real(t1 * t2) + np.sin(np.real(t1)) * np.cos(np.imag(t2))
    cf[35] = cf[17] * cf[31] + np.sin(np.real(t1 * t2)) - np.cos(np.imag(t1 * t2))
    return cf.astype(np.complex128)
    
ALLOWED["p2"]=p2

def p3(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(36, dtype=np.complex128)
    for k in range(2, 37):
        v = (t1 ** k) / 2 + np.cos(t2 ** (k - 1)) + np.log(np.abs(t1) + 1)
        if k % 2 == 0:
            cf[k-1] = np.real(v)
        else:
            cf[k-1] = np.imag(v)
    cf[0] = 1 + t1 * t2
    cf[17] = cf[2] * cf[4] - cf[3] * cf[1]
    cf[23] = cf[11] / cf[5] + cf[7] * cf[9]
    cf[35] = np.sum(cf[15:20])
    return cf

ALLOWED["p3"]=p3

def p4(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(36, dtype=np.complex128)
    for k in range(1, 37):
        cf[k-1] = (t1 ** (36 - k) + t2 ** (36 - k)) / (k * 1j)
    cf[16] = t1 * t2 + np.log(np.abs(t1) + 1) - np.sin(t2)
    cf[24] = np.real(t1) - np.imag(t1) + 1j * (np.real(t2) + np.imag(t2))
    cf[29] = np.abs(t1)**2 - np.abs(t2)**2 + 1j * np.angle(t1) * np.angle(t2)
    cf[35] = np.conj(t1 * t2)**2 - np.sin(t1 + t2)
    return cf

ALLOWED["p4"]=p4

def p5(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(36, dtype=np.complex128)
    p = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53])
    for k in range(1, 17):
        cf[k-1] = np.sin(p[k-1] * t1) + np.cos(p[k-1] * t2)
    for k in range(17, 33):
        cf[k-1] = np.log(np.abs(p[k-17] * t1 + t2)) / (t1 + t2)
    cf[32] = np.prod(p[0:4]) / (t1 * t2)
    cf[33] = np.sum(p[4:8]) - t1**2 + t2**2
    cf[34] = p[8] * p[9] * (t1 + t2)
    cf[35] = p[10] * p[11] / (t1 - t2)
    return cf

ALLOWED["p5"]=p5

def p6(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(51, dtype=np.complex128)
    for k in range(1, 52):
        cf[k-1] = (t1 + t2) * np.sin(np.log(np.abs(t1 * t2)**k + 1)) + np.cos(np.angle(t1 * t2)**k) * np.conj(t1 - t2)
    return cf

ALLOWED["p6"]=p6


def p7(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(51, dtype=np.complex128)
    for k in range(1, 52):
        cf[k-1] = (
            np.cos(t1 * k)
            + 1j * np.sin(t2 * k)
            + np.log(np.abs(t1) + 1)
            + np.log(np.abs(t2) + 1)
        )
    primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47])
    # Fix the slice here:
    cf[1:16] += primes * (np.real(t1) + np.imag(t2)**2)
    cf[24] += np.sum(primes[0:5]) / (t1 + t2)
    cf[49] *= ( np.real(t1 * t2) + np.imag(t1 * t2) + np.log(np.abs(t1 * t2) + 1) )
    return cf

ALLOWED["p7"]=p7

def p8(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(51, dtype=np.complex128)
    cf[0:25] = np.arange(1, 26) * (t1**2 + 1j * t2**3)
    cf[25] = np.abs(t1 + t2)
    cf[26:51] = np.arange(1, 26) * (t2**2 - 1j * t1**3)
    cf[2] = np.sin(t1) * cf[0]**2
    cf[6] = np.log(np.abs(t2) + 1) * cf[4]**3
    cf[32] = cf[6] + cf[2]
    cf[36] = cf[32] - cf[6]
    cf[40] = cf[32] + cf[2]
    cf[49] = np.angle(t1) * np.angle(t2)
    cf[50] = np.abs(cf[40])
    return cf

ALLOWED["p8"]=p8

def p9(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(51, dtype=np.complex128)
    cf[0] = t1 + t2
    for k in range(2, 52):
        cf[k-1] = (np.abs(t1) * np.sin(k) + np.angle(t2) * np.cos(k)) / np.abs(k + 1j)
    cf[9] = cf[0]**2 - cf[1]**2 + np.log(np.abs(cf[2]) + 1)
    cf[19] = np.sum(cf[0:19]) * t1
    cf[29] = np.prod(cf[0:29]) * t2
    cf[39] = cf[38] * cf[37] / (1 + t1 * t2)
    cf[40:50] = np.real(cf[30:40]) + 1j * np.imag(cf[0:10])
    cf[50] = np.sum(cf[0:50])
    return cf
    
ALLOWED["p9"]=p9

def p10(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(51, dtype=np.complex128)
    cf[0] = np.real(t1 * t2) + np.imag(t2) * np.real(t1)
    cf[1] = np.abs(t1 * t2) * np.cos(np.angle(t1 + t2))
    for i in range(2, 51):
        cf[i] = cf[i - 2] * np.abs(cf[i - 1]) * np.sin(np.angle(t1 + t2))
    cf[50] = np.log(np.abs(t1 * t2)) + cf[0] + cf[1]
    return cf.astype(np.complex128)
    
ALLOWED["p10"]=p10

def p11(z,a,state):
    t1, t2 = z[0], z[1]
    n = 51
    cf = np.zeros(n, dtype=np.complex128)
    cf[0] = t1 * t2 - np.log(np.abs(t1 + t2) + 1)
    for i in range(1, n-1):
        cf[i] = np.sin(i) * (np.real(t1**i) - np.imag(t2**i)) + np.cos(i) * (np.real(t2**i) - np.imag(t1**i))
        cf[i] = cf[i] / (np.abs(cf[i]) + 1e-10)
    cf[n-1] = np.abs(t1) * np.abs(t2) * np.angle(t1 + t2)
    return cf

ALLOWED["p11"]=p11

def p12(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(51, dtype=np.complex128)
    cf[0] = t1 * (2 * (np.imag(t2))**2)
    cf[1] = t2 * (2 * (np.real(t1))**2)
    for k in range(2, 51):
        cf[k] = ((np.abs(t1)**k + np.abs(t2)**(50-k)) / (k**2)) * np.exp(1j * np.angle(t1 * t2))
    cf[22] = np.cos(t1 * t2) * (t1 - 1j * t2)
    cf[34] = np.sin(t1 * t2) * (1j * t2 - t1)
    cf[49] = np.log(np.abs(t1 + t2))**3
    cf[50] = np.conj(t1) * np.conj(t2)
    return cf

ALLOWED["p12"]=p12

def p13(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(51, dtype=np.complex128)
    fib = np.array([1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181])
    for n in range(19):
        cf[n] = fib[n] * t1 * np.cos(np.angle(t2))
        cf[n + 19] = fib[n] * t1 * np.sin(np.angle(t2))
        cf[n + 38] = fib[n] * t2 * np.sin(np.angle(t1))
    cf[19] = np.abs(t1 * t2)
    cf[49] = np.log(np.abs(t1 * t2) + 1)
    cf[50] = np.real(t1) + np.imag(t2)
    return cf.astype(np.complex128)

ALLOWED["p13"]=p13

def p14(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(51, dtype=np.complex128)
    triangleNums = np.cumsum(np.arange(1, 51))
    cf[0] = t1 + 3 * t2
    for k in range(1, 51):
        v1 = triangleNums[k] * (t1 + t2 * np.log(np.abs(t1) + 1))**(k) 
        v2 = triangleNums[k] * (t2 + t1 * np.log(np.abs(t2) + 1))**(k)
        cf[k] = v1 + v2
    cf[42] = np.real(np.abs(t1)) + np.imag(np.abs(t2))
    cf[20] = np.real(np.abs(t2)) + np.imag(np.abs(t1))
    cf[31] = np.real(np.abs(t1 * t2)) + np.imag(np.conj(t1 * t2))
    cf[27] = 2 * np.real(t1 - t2) + 2 * np.imag(t1 - t2)
    return cf

ALLOWED["p14"]=p14

def p14a(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(51, dtype=np.complex128)
    triangleNums = np.cumsum(np.arange(1, 51))
    cf[0] = t1 + 3 * t2
    for k in range(1, 51):
        v1 = triangleNums[k] * (t1 + t2 * np.log(np.abs(t1) + 1))**(k) 
        v2 = triangleNums[k] * (t2 + t1 * np.log(np.abs(t2) + 1))**(k)
        cf[k] = v1 + v2
    return cf

ALLOWED["p14a"]=p14a


def p15(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    primes = np.array([
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 
        89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 
        181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241
    ])
    for i in range(71):
        top =  (primes[i] * t1 + 1j * t2**i)
        bot = (1 + np.abs(t1))**i
        cf[i] = _safe_div( top, bot)
    cf[70] = np.sum(cf[0:70])
    return cf.astype(np.complex128)

ALLOWED["p15"]=p15

def p16(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(51, dtype=np.complex128)
    cf[0] = t1 + t2
    cf[1] = np.real(t1**2 - t2**2)
    primes = np.array([
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
        53, 59, 61, 67, 71, 73, 79, 83, 89, 97
    ])
    for k in range(2, 25):  # Adjusted indices: R is 1-indexed, Python is 0-indexed
        cf[k] = np.imag(cf[k - 1] * primes[k - 2]) * np.angle(t1) * np.abs(t2)
    for k in range(25, 50):
        cf[k] = np.abs(cf[k - 1] * primes[k - 25] ** 2) * np.angle(t2) * np.real(t1)
    cf[50] = np.sum(cf) + np.sin(np.real(t2)) * np.log(np.abs(t1) + 1)
    return cf

ALLOWED["p16"]=p16

def p17(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0:10] = (t1 + t2) * np.arange(1, 11)
    cf[10:20] = np.real(t1 - t2)**3 * np.arange(11, 21)
    cf[20:30] = np.imag(t1 + t2)**2 * np.arange(21, 31)
    cf[30:40] = np.abs(t1 - t2) * np.arange(31, 41)
    cf[40:50] = np.angle(t1 * t2) * np.arange(41, 51)
    cf[50] = np.sin(t1) * np.cos(t2) + np.sin(t2) * np.cos(t1)
    return cf

ALLOWED["p17"]=p17

def p18(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for i in range(1, 72):
        arg = np.angle(t1 * t2 * 1j)
        mod = np.abs((t1 + 1j) * (t2 + 1j))
        cyclotomic = 0
        for k in range(1, i + 1):
            v1 = t1 - np.exp(2j * np.pi * k / i)
            cyclotomic += v1
        cf[i-1] = mod * cyclotomic * arg
    cf[70] = np.log(np.abs(t1 * t2)) + 1
    return cf

ALLOWED["p18"]=p18

def p19(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = np.real(t1) + np.imag(t2)
    cf[1] = np.angle(t1)
    cf[2] = np.abs(t2)
    cf[3] = np.sin(t1) + np.cos(t2)
    cf[4:10] = np.arange(1,7) * 0.2 + 1
    cf[10] = np.log(np.abs(t1) + 1) + np.log(np.abs(t2) + 1)
    for i in range(11, 72):
        cf[i-1] = cf[i-2] * np.sin(i * cf[i-3] + np.abs(cf[i-4])) + cf[i-5]
    return cf

ALLOWED["p19"]=p19

def p20(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for i in range(1, 18):
        cf[i * 3 - 2] = ((t1 + t2)**i + i * (t1 - t2)**i) / (2**i)
    cf[3] = np.abs(t1) * np.sin(np.angle(t1))
    cf[7] = np.abs(t2) * np.cos(np.angle(t2))
    cf[18] = np.log(np.abs(t1 * t2)) * np.cos(np.angle(t1 - t2))
    cf[36] = np.abs(t1 * t2) * np.cos(np.angle(t1 + t2))
    i1 = np.array([20, 24, 28, 32, 36, 40, 44, 48, 50],dtype=np.intp) 
    v1 = np.real(t1) + np.imag(t2)
    cf[i1] = v1
    i2 = np.array([22, 26, 30, 34, 38, 42, 46],dtype=np.intp)
    v2 = np.imag(t1) + np.real(t2)
    cf[i2] = v2
    cf[49] = np.abs(t1)**2 * np.sin(2 * np.angle(t2))
    cf[50] = np.abs(t2)**2 * np.cos(2 * np.angle(t1))
    return cf

ALLOWED["p20"]=p20

#FIXME
def p21(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    roots = np.exp(2 * np.pi * 1j * np.arange(0, 71) / 71)
    for i in range(cf.size):
        v2 = 1.0 + 0.0j
        for j in range(roots.size):
            if j == i: continue
            v2 = v2 * (roots[j] - roots[i])
        v3 = t1 - roots[i]
        v4 = t2 - roots[i]
        cf[i] = v2 / v3 / v4
    v5 = (t1 - roots)
    v6 = (t2 - roots)
    cf1 =  cf * v5 * v6
    return cf1

ALLOWED["p21"]=p21

def p22(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97])
    cf[0:25] = primes[0:25] * (np.real(t1)**2 - np.imag(t1**2) + np.real(t2)**2 - np.imag(t2**2))
    cf[25:50] = cf[0:25] * (np.cos(np.angle(t1 + t2)) + np.sin(np.abs(t1) * np.abs(t2)))
    cf[50] = np.sum(cf[0:50])
    return cf
    
ALLOWED["p22"]=p22

# infinity sign
def p23(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1 + t2
    cf[1] = 1 + t1 * t2 + np.log(np.abs(t1 + t2) + 1)
    cf[2] = t1 + t2 + np.log(np.abs(1 - t1 * t2) + 1)
    for i in range(3, 72):
        cf[i-1] = i * t1 + (51 - i) * t2 + np.log(np.abs(t1 - t2 * i) + 1)
    cf[10] = cf[0] + cf[9] - np.sin(t1)
    cf[20] = cf[30] + cf[40] - np.cos(t2)
    cf[30] = cf[20] + cf[40] + np.sin(t1)
    cf[40] = cf[30] + cf[20] - np.cos(t2)
    cf[50] = cf[40] + cf[20] + np.sin(t2)
    return cf

ALLOWED["p23"]=p23

def p24(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1 + t2
    cf[1] = t1 * t2
    for n in range(3, 72):
        v1 = np.abs(cf[n-2]) + 1j * np.angle(cf[n-3])
        v2 = np.abs(t1 + t2)**(1/n) * (np.cos(n * t2) + 1j * np.sin(n * t1))
        cf[n-1] = v1 + v2
    return cf.astype(np.complex128)

ALLOWED["p24"]=p24

def p25(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = np.real(t1 * t2) + np.imag(t1 / t2)
    for k in range(1, 72):
        cf[k] = np.abs(t1)**k + np.angle(t2)**k + np.sin(t1 + k) + np.cos(t2 + k) - np.log(np.abs(t1 * t2)**k + 1)
    cf[35] = np.real(cf[0] * cf[34]) + np.imag(t1 * t2)
    cf[45] = 0.5 * (t1 + np.conj(cf[44]) + t2)
    cf[50] = cf[0] + cf[34] + cf[44] + np.real(t1) + np.imag(t2)
    return cf
    
ALLOWED["p25"]=p25

def p26(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1 + t2
    cf[1] = t1 * t2
    for k in range(3, 72):
        v = np.sin(k * cf[k-2]) + np.cos(k * cf[k-3])
        cf[k-1] = v / np.abs(v)
    cf[14] = np.abs(t1 - t2) * np.angle(t1 + t2)
    cf[29] = np.log(np.abs(t1 * np.real(t2) + 1)) - np.log(np.abs(t2 * np.imag(t1) + 1))
    cf[49] = np.abs(t1) * np.abs(t2) * np.abs(t1 - t2)
    cf[50] = np.sum(cf[15:29]) * np.sum(cf[30:44]) + t1**2
    return cf

ALLOWED["p26"]=p26

def p27(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0:10] = np.abs(t1) * np.abs(t2) * np.arange(1, 11)
    cf[10:20] = np.abs(t1)**(np.arange(2, 12))
    cf[20:30] = np.abs(t2)**(np.arange(2, 12))
    cf[30] = np.real(t1 * t2)
    cf[31] = np.imag(t1 * t2)
    cf[32:42] = np.real(t1) * np.arange(1, 11) + np.imag(t2) * np.arange(1, 11)
    cf[42:51] = np.real(t2) * np.arange(1, 10) + np.imag(t1) * np.arange(1, 10)
    for i in range(50):
        cf[i] += np.sin(cf[i + 1])
    for i in range(51, 1, -1):
        cf[i - 1] -= np.cos(cf[i - 2])
    cf[50] += np.angle(t1) + np.angle(t2)
    return cf

ALLOWED["p27"]=p27

def p28(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53])
    for k in range(1, 17):
        cf[k-1] = (np.real(t1) + np.imag(t2)) / primes[k-1]
        cf[71-k] = (np.imag(t1) + np.real(t2)) * primes[k-1]
    for k in range(17, 36):
        cf[k-1] = np.sin((np.real(t1) + np.imag(t1))**2) * (np.real(t2) + np.imag(t2))**(2 + k)
    cf[35] = np.log(np.abs(t1) * np.abs(t2) + 1) + np.abs(t2 - t1)
    cf[36:51] = np.angle(t1 + t2) + np.abs(t1 - t2) + np.angle(np.conj(t1 * t2))
    cf[50] = np.sum(cf[0:50])**2
    return cf
    
ALLOWED["p28"]=p28

def p29(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0:5] = np.array([1, t1, t1**2, t1**3, t1**4])
    cf[5:10] = np.array([1, t2, t2**2, t2**3, t2**4])
    cf[10:15] = np.array([1, np.exp(1j * t1), np.exp(2j * t1), np.exp(3j * t1), np.exp(4j * t1)])
    cf[15:20] = np.array([1, np.exp(1j * t2), np.exp(2j * t2), np.exp(3j * t2), np.exp(4j * t2)])
    cf[20:30] = np.array([1, np.real(t1 + t2), np.imag(t1 + t2), np.real(t1 * t2), np.imag(t1 * t2), np.real(t1 + t2)**2, np.imag(t1 + t2)**2, np.real(t1 * t2)**2, np.imag(t1 * t2)**2, np.abs(t1 + t2)])
    cf[30:40] = np.arange(1, 11) * np.abs(t1) * np.abs(t2)
    cf[40:50] = np.array([1, np.log(np.abs(t1) + 1), np.log(np.abs(t2) + 1), np.log(np.abs(t1 + t2) + 1), np.log(np.abs(t1 * t2) + 1), np.angle(t1), np.angle(t2), np.abs(t1), np.abs(t2), np.angle(t1 + t2)])
    cf[50] = np.abs(t1 + t2) * np.angle(t1 * t2)
    return cf.astype(np.complex128)

ALLOWED["p29"]=p29

def p30(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for i in range(1, 72):
        v1 = (np.real(t1) * np.imag(t2) + np.imag(t1) * np.real(t2))**(1/i)
        v2 = np.abs(t2)**(i/50)
        v3 = np.sin(np.angle(t1) * (i/25))
        v4 =  np.cos(np.angle(t2) * (i/50))
        v5 =  np.log(np.abs(t1) + 1)
        v6 = np.log(np.abs(t2) + 1)
        cf[i-1] =  v1 *  v2 * v3 * v4 + v5 + v6
    return cf
    
ALLOWED["p30"]=p30

def p31(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1 + t2
    for i in range(1, 36):
        cf[i] = (np.cos(i * t1) + np.sin(i * t2)) / (np.abs(t1) * np.abs(t2))**i
    for i in range(36, 72):
        cf[i] = (np.cos(t1**i) + np.sin(t2**i)) * np.log(np.abs(t1)**i + 1) * np.log(np.abs(t2)**i + 1)
    return cf.astype(np.complex128)

ALLOWED["p31"]=p31

def p32(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1 + t2
    for k in range(1, 72):
        r = np.abs(t1)**k + np.abs(t2)**(71-k)
        theta = np.angle(t1)**k - np.angle(t2)**(71-k)
        cf[k-1] = r * np.cos(theta) + r * np.sin(theta) * 1j
    cf[2:70] += np.log(np.abs(t2 - t1) + 1)
    cf[70] += np.conj(t1 * t2)
    return cf
    
ALLOWED["p32"]=p32

def p33(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    f = lambda z, n: z**n - 1
    cf[0:35] = [np.real(f(t1, n)) - np.imag(f(t2, n)) for n in range(1, 36)]
    cf[35:70] = [np.log(np.abs(f(t2, n))) + np.angle(f(t1, n)) + np.sin(np.abs(f(t1, n))) + np.cos(np.angle(f(t2, n))) for n in range(1, 36)]
    cf[70] = np.prod(cf[0:70])
    return cf
    
ALLOWED["p33"]=p33

def p34(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for i in range(1, 72):
        cf[i-1] = (t1 * t2) * np.sin(i * t1 + t2) + np.cos(i * t1 - t2) + i * np.log(np.abs(t1 + i * t2) + 1) / (1 + np.abs(t1 + i * t2))
    cf[12] = 3 * (t1**2 - t2**2)
    cf[13] = cf[12] + t1 * t2 * np.sin(np.angle(t1 + t2))
    cf[14] = 2 * cf[13] - t1 * t2 * np.cos(np.angle(t1 - t2))
    cf[15] = 3 * cf[12] - cf[13] + t1 * t2 * np.sin(2 * np.angle(t1 - t2))
    cf[16] = 2 * cf[12] - 3 * cf[13] + cf[14] - t1 * t2 * np.cos(2 * np.angle(t1 + t2))
    cf[69] = 2 * cf[13] - 3 * cf[12] + t1 * t2 * np.sin(np.angle(2 * t1 - t2))
    cf[70] = cf[16] - cf[13] * t1 * t2 * np.cos(np.angle(2 * t1 + t2))
    return cf
    
ALLOWED["p34"]=p34

def p35(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(1, 71):
        z = np.cos(t1) * t1**k - np.sin(t2) * t2**k
        cf[k-1] = np.real(z) + 1j * np.imag(z)
    cf[70] = np.abs(t1 * t2)
    return cf

ALLOWED["p35"]=p35

def p36(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1 + t2 * 1j
    cf[2] = np.abs(t1)**2 + np.angle(t2)**2
    cf[8] = np.sin(t1 + t2)
    for k in range(9, 72):
        cf[k-1] = np.cos(k * np.real(t1 + t2)) + np.sin(k * np.imag(t1 * np.conj(t2)))
    cf[70] = np.log(np.abs(t1 * t2) + 1)
    return cf
    
ALLOWED["p36"]=p36

def p37(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for i in range(1, 72):
        cf[i-1] = (np.sin(i * t1) + np.cos(i * t2)) * i**2
    cf[1] += np.sum(cf[0:2])
    cf[4] += np.prod(cf[0:5])
    cf[11] += np.log(np.abs(cf[10]) + 1)
    cf[24] += np.angle(cf[23])
    i1 = np.array([34, 44, 54, 64],dtype=np.intp)
    cf[i1] += np.abs(t2)**2 + np.real(t1)**3
    i2 = np.array([6, 13, 20, 27, 34, 41, 48, 55, 62, 69],dtype=np.intp)
    cf[i2] += np.sin(t1)**i - np.cos(t2)**i
    return cf

ALLOWED["p37"]=p37

def p38(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(1, 71):
        cf[k-1] = (1 + t1 * t2)**k / (1 + np.real(t1 * t2)**2)
    cf[70] = np.abs(t1) + np.abs(t2) + np.angle(t1 + t2) + np.sin(np.real(t1) + np.imag(t2)) + np.log(np.abs(np.real(t2) + 1))
    return cf

ALLOWED["p38"]=p38

#FIXME
def p39(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1
    cf[1] = t2
    for k in range(3, 72):
        cf[k-1] = np.sin(k * t1) + np.cos(k * t2) + np.log(np.abs(k) + 1) * np.abs(cf[k-2]) * np.abs(cf[k-3]) * np.abs(np.angle(t1 + t2))
    return cf

ALLOWED["p39"]=p39

def p40(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(1, 71):
        cf[k-1] = (np.real(t1)**(k + 1)) * np.sin(np.angle(t2 * k)) + (np.imag(t2)**k) * np.cos(np.angle(t1 / k))
    cf[70] = np.abs(t1) + np.abs(t2)
    return cf
    
ALLOWED["p40"]=p40

def p41(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1 + t2
    cf[1] = (t1 + t2) * (t1 - t2)
    cf[2] = np.abs(np.real(t1))**2 + np.abs(np.imag(t1))**2 + np.abs(np.real(t2))**2 + np.abs(np.imag(t2))**2
    for i in range(3, 72):
        cf[i-1] = cf[i-2] * t1 + cf[i-3] * t2 + cf[i-4]
    return cf

ALLOWED["p41"]=p41

def p42(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0:35] = np.abs(t1) * np.sin((np.arange(1, 36)) * np.angle(t1))
    cf[35:70] = np.real(t2) * np.cos((np.arange(1, 36)) * np.imag(t2))
    cf[70] = t1 * t2 + 1j * np.sum(np.log(np.abs(cf[0:70]) + 1))
    return cf
    
ALLOWED["p42"]=p42

def p43(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1 * t2
    cf[1] = np.log(np.abs(t1) + 1) + np.log(np.abs(t2) + 1)
    for k in range(2, 72):
        cf[k-1] = np.sin(k * cf[k-2]) + np.cos(k * t1) - np.sin(k * cf[k-3]) + np.cos(k * t2)
        cf[k-1] = cf[k-1] / np.abs(cf[k-1])
    cf[34] = np.real(t1)**3 - np.imag(t2)**3
    cf[52] = np.abs(t1 * t2)**2 - np.angle(t1 * t2)
    cf[70] = np.real(t1 * t2) - np.imag(t1 * t2) + np.angle(t1 * t2)
    return cf
    
ALLOWED["p43"]=p43

def p44(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = np.sin(t1) * np.cos(t2)
    cf[1] = np.cos(t1) + np.sin(t2)
    cf[2] = np.abs(t1)**3 - np.abs(t2)**4
    cf[3] = np.angle(t1) - np.angle(t2)
    cf[4] = np.abs(t1 * t2)
    for k in range(6, 36):
        cf[k-1] = np.sin(k * t1) + np.cos(k * t2)
        cf[k + 34] = np.sin((70 - k) * t1) - np.cos((70 - k) * t2)
    cf[35] = np.abs(t1 + t2)
    cf[70] = np.log(np.abs(t1 * t2) + 1)
    return cf
    
ALLOWED["p44"]=p44

def p45(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(1, 72):
        if k % 2 == 0:
            cf[k-1] = k * (t1 + np.real(t2)) * np.sin(np.abs(t1) * k)
        else:
            cf[k-1] = k * (t2 - np.imag(t1)) * np.cos(np.angle(t2) * k)
    for i in range(2, len(cf) // 2):
        cf[i-1] = cf[i-2] * (np.abs(t1) + 0.5) + np.log(np.abs(t2) + 1)
        cf[len(cf) - i] = -cf[len(cf) - i + 1] * (np.abs(t2) + 0.5) - np.log(np.abs(t1) + 1)
    return cf
    
ALLOWED["p45"]=p45

def p46(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    primes = np.array([2, 3, 5, 7, 11, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67])
    cf[0:18] = np.real(primes * (t1 + t2))
    cf[18:36] = np.imag(primes * (t1 - t2))
    cf[36:54] = np.real(primes * (t1 * np.conj(t2)) + np.log(np.abs(primes)))
    cf[54:71] = np.imag(t1**(np.arange(1, 18))) * t2**(np.arange(1, 18)**2)
    cf[70] = np.sum(t1**(np.arange(1, 6)) * t1**(np.arange(1, 6)**2)) + np.sum(t2**(np.arange(1, 11)))**2
    return cf.astype(np.complex128)

ALLOWED["p46"]=p46

def p47(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1 * t2
    cf[1] = np.sin(t1) * np.cos(t2)
    cf[2:6] = np.log(np.abs(t1 + t2) + 1) * np.arange(1, 5)
    cf[6:10] = 2 * np.abs(t1) * np.log(np.abs(t2) + 1) * np.arange(1, 5)
    cf[10:14] = 3 * np.abs(t2) * np.log(np.abs(t1) + 1) * np.arange(1, 5)
    cf[14:18] = np.angle(t1 + t2) * np.arange(1, 5)
    cf[18:26] = np.cos(t1) * np.sin(t2) * np.arange(1, 9)
    cf[26:34] = np.sin(t1) * np.cos(t2) * np.arange(1, 9)
    cf[34:50] = (t1 + t2) / np.arange(1, 17)
    cf[50] = np.prod(np.arange(1, 72))
    cf[51:72] = (t1 - t2) / np.arange(20, 0, -1)
    return cf
    
ALLOWED["p47"]=p47

def p48(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for i in range(1, 72):
        cf[i-1] = np.abs(t1)**(i / (t1 + t2))
    cf[0:10] += np.angle(t2) * 10
    cf[10:20] -= np.angle(t1) * 10
    cf[20:30] += np.real(t1)**2
    cf[30:40] -= np.imag(t2)**2
    cf[40:50] += np.abs(t1) * np.log(np.abs(t1) + 1)
    cf[50:60] -= np.abs(t2) * np.log(np.abs(t2) + 1)
    cf[60:70] += np.sin(t1 + t2)
    cf[70] = np.prod(cf[0:70]) / 70
    return cf

ALLOWED["p48"]=p48

def p49(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for i in range(1, 72):
        cf[i-1] = (i**2 + 3 * 1j + 1) * (t1**2) + (i**3 - i**2 + 1) * (t2**2) + np.sin(t1 * 1j + t2) + np.log(np.abs(t1 * 1j - t2) + 1)
        if i % 2 == 0:
            cf[i-1] += (t1 + 1j * t2)**2
        elif i % 3 == 0:
            cf[i-1] += np.abs(t1 + 1j * t2)**3
        else:
            cf[i-1] += np.real(t1 + 1j * t2)**4
    cf[0] *= 10000
    cf[1] *= 1000
    cf[2] *= 100
    cf[3] *= 10
    return cf

ALLOWED["p49"]=p49

def p50(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    complex_val = np.sin(t1 + t2) + np.cos(t1 - t2)
    for k in range(1, 72):
        if k % 2 == 0:
            cf[k-1] = complex_val / np.abs(k) - np.abs(t1)
        else:
            cf[k-1] = complex_val * np.abs(k) + np.log(np.abs(k) + 1) + np.imag(t2) - np.real(t1)
        if k % 3 == 0:
            cf[k-1] += 3 * cf[k-2]
        if k % 5 == 0:
            cf[k-1] += 5 * cf[k-3]
        if k % 7 == 0:
            cf[k-1] += 7 * cf[k-4]
    return cf
    
ALLOWED["p50"]=p50

#FIXME
def p51(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = 1.0+1.0j
    for k in range(1, 71):
        if k % 2 == 0:
            v1 = np.sin(k * (t1 + t2) ** k)
            v2 = np.cos(k * (t1 - t2) ** k)
            cf[k] = v1 + v2
        else:
            cf[k] = np.real(t1) ** k + np.imag(t2) ** k
    cf[70] = np.abs(t1) ** 3 + np.angle(t2)
    return cf
    
ALLOWED["p51"]=p51

def p52(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1 ** 7 + t2 ** 7
    for k in range(2, 36):
        cf[k - 1] = np.sin(k * np.log(np.abs(t1) + 1) + np.log(np.abs(t2) + 1))
    for k in range(36, 71):
        cf[k - 1] = np.cos(k * np.log(np.abs(t1) + 1) - np.log(np.abs(t2) + 1))
    cf[70] = t1 * t2 - (t1 + t2) ** 2
    return cf

ALLOWED["p52"]=p52

def p53(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1 * t2
    cf[1] = 1j * (t1 + t2)
    cf[2] = np.real(t1) + np.imag(t2)
    cf[3] = np.sin(cf[0]) * np.cos(cf[1])
    cf[4] = np.abs(t1 - t2)
    for k in range(6, 37):
        cf[k - 1] = np.abs(cf[k - 2]) ** 2 - np.log(np.abs(cf[k - 3]) + 1)
    for k in range(36, 72):
        cf[k - 1] = np.angle(cf[k - 2] * cf[k - 4]) * np.abs(cf[k - 3] * cf[k - 6])
    cf[70] = np.real(t1) * np.imag(t2) - np.real(t2) * np.imag(t1)
    return cf

ALLOWED["p53"]=p53

def p54(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for i in range(1, 72):
        z = t1 * np.cos(i * t2 / 15) + t2 * np.sin(i * t1 / 15)
        phi = np.angle(z)
        r = np.abs(z)
        cf[i - 1] = r * np.exp(1j * phi) ** i + (-1) ** (i + 1) * i ** 2
    cf[0:30] = cf[0:30] * (np.abs(t1) * np.abs(t2)) ** np.arange(1, 31)
    return cf

ALLOWED["p54"]=p54

def p55(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1 * t2
    cf[1] = np.real(t1) * np.imag(t2)
    cf[2] = np.real(t2) * np.imag(t1)
    cf[3:10] = np.linspace(np.log(np.abs(cf[0]) + 1), np.log(np.abs(cf[2]) + 1), 7)
    cf[10:30] = [np.cos(cf[i - 1]) + ((t1 + t2) ** i) / (i + 1) for i in range(11, 31)]
    cf[30:50] = [np.sin(cf[i - 1]) + ((t1 - t2) ** i) / (i + 1) for i in range(31, 51)]
    cf[50:70] = np.abs(cf[0:20]) + np.abs(cf[20:40] + t1 + t2)
    cf[70] = np.prod(cf[0:70])
    return cf

ALLOWED["p55"]=p55

def p56(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(1, 72):
        cf[k - 1] = ((t1 + 1j * t2) ** k + (t2 + 1j * t1) ** (71 - k)) / 2
    cf[3:68] = cf[3:68] * (1 + np.sin(np.angle(t1 + 1j * t2)))
    cf[0:3] = cf[0:3] * (1 + np.cos(np.angle(t1 + 1j * t2)))
    cf[68:71] = cf[68:71] * np.abs(t1 + 1j * t2)
    cf[34] = cf[34] * np.log(np.abs(np.imag(t1 + 1j * t2)) + 1)
    return cf

ALLOWED["p56"]=p56

def p57(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(1, 36):
        cf[k - 1] = (t1 * k ** 2 + t2 * (70 - k)) * (1 - (-1) ** k) / 2
    for k in range(36, 71):
        cf[k - 1] = (t1 * np.conj(t2)) ** k * np.abs(t1 - t2)
    cf[70] = np.abs(t1 * np.real(t2)) * np.abs(t2 - t1)
    return cf

ALLOWED["p57"]=p57

def p58(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = 100 * np.abs(t1) * np.abs(t2) - 100
    cf[1] = 100 * t1 ** 3 - t2 ** 3 + 100
    cf[2] = 100 * t2 ** 3 - t1 ** 3 + 100
    cf[3:71] = [np.cos(k * t1) * np.sin(k * t2) / np.log(np.abs(k + 1)) for k in range(1, 69)]
    #root_coeff = np.abs(t1) * np.abs(t2) * np.prod(range(1, 71)) / np.sum(range(1, 71))
    root_coeff = np.abs(t1) * np.abs(t2) * np.float64(np.prod(np.arange(1, 71))) / (70.0 * 71.0 / 2.0)
    cf[4] = root_coeff * np.sum(
        (np.cos(np.arange(1.0, 71.0) * t1) * np.sin(np.arange(1.0, 71.0) * t2))
        / np.log1p(np.arange(1.0, 71.0))
    )
    cf[35] = root_coeff * t1 ** 2
    cf[34] = root_coeff * t2 ** 2
    cf[36:71] = root_coeff * (
        np.cos(2.0 * np.arange(1.0, 36.0) * t1) *
        np.sin(2.0 * np.arange(1.0, 36.0) * t2)
    ) / np.log1p(2.0 * np.arange(1.0, 36.0))
    return cf

ALLOWED["p58"]=p58

def p59(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for i in range(1, 72):
        cf[i - 1] = (t1 / (i + 1)) ** i + (t2 / (i + 1)) * (2j)
    cf[np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19],dtype=np.intp)] *= (t1 + 2 * t2)
    cf[np.array([2, 5, 8, 11, 14, 17, 20, 23, 26, 29],dtype=np.intp)] *= (t1 - 2 * t2)
    cf[4:36] += 2 * t1
    cf[36:67] -= 2 * t2
    cf[67:71] = np.real(np.log(cf[67:71])) + cf[4]
    return cf.astype(np.complex128)

ALLOWED["p59"]=p59

def p60(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(1, 72):
        cf[k - 1] = t1 ** k + np.conj(t2) ** (k - 1) / ((k % 2) + 1)
    cf[0:5] = np.abs(cf[0:5]) ** 2
    cf[5:10] *= np.log(np.abs(t2) + 1)
    cf[10:15] *= np.abs(t1 - t2) ** 3
    cf[15:20] *= np.sin(t1 + t2)
    cf[20:25] *= np.cos(t1 - t2)
    cf[25:30] *= np.abs(t1 + t2) ** 2
    cf[30:35] *= np.sin(t1 - t2)
    cf[35:40] *= np.cos(t1 + t2)
    cf[40:45] *= np.abs(t1 - t2)
    cf[45:50] *= np.sin(t1 + t2)
    cf[50:55] *= np.cos(t1 - t2)
    cf[55:60] *= np.abs(t1 + t2)
    cf[60:65] *= np.sin(t1 - t2)
    cf[65:70] *= np.cos(t1 + t2)
    cf[70] *= np.abs(t1 - t2)
    return cf
    
ALLOWED["p60"]=p60

def p61(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0:35] = np.real(t1) * (np.arange(1, 36) ** 3) + np.imag(t2) * np.sin(np.arange(1, 36))
    cf[35:70] = np.imag(t1) * (np.arange(70, 35, -1) ** 2) + np.real(t2) * np.cos(np.arange(70, 35, -1))
    cf[70] = np.abs(t1) * np.angle(t2) - np.abs(t2) * np.angle(t1)
    return cf
    
ALLOWED["p61"]=p61

#FIXME crash
def p62(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1 ** 5 + t2 ** 5
    for i in range(2, 72):
        if i % 2 == 0:
            cf[i - 1] = i * cf[i - 2] ** 2
        else:
            cf[i - 1] = i * cf[i - 2] ** 2 * (1 + 0.1 * t2)
    cf[0] += 2 * cf[1]
    cf[1] -= 3 * cf[2]
    for i in range(3, 70):
        cf[i] += cf[i + 1] - cf[i + 2]
    cf[69] += cf[70]
    cf[70] = np.abs(t1) ** 2 - np.abs(t2) ** 2 + 2 * np.imag(t1) * np.imag(t2) - np.angle(t2)
    return cf

ALLOWED["p62"]=p62

def p63(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for j in range(1, 72):
        cf[j - 1] = np.sin(t1 ** j) * np.cos(t2 ** (71 - j)) * np.abs(t1 * t2 ** j) * np.log(np.abs(t1 * t2 + 1))
    cf[0:30] += cf[30:60]
    cf[32:72] -= cf[0:40]
    cf[10:60] += np.real(t1) * np.imag(t2) * cf[0:50]
    cf[30:70] -= np.imag(t1) * np.real(t2) * cf[1:41]
    cf[20:40] += np.angle(t1 ** t2) * cf[30:50]
    cf[40:72] -= np.angle(t2 ** t1) * cf[0:32]
    return cf

ALLOWED["p63"]=p63

def p64(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(1, 72):
        cf[k - 1] = (np.abs(t1) ** k + np.abs(t2) ** (71 - k)) * np.cos(np.angle(t1) * k + np.angle(t2) * (71 - k))
    cf[1::2] *= 1j
    cf[2::3] *= -1
    cf[0] *= 100
    cf[70] /= 100
    return cf

ALLOWED["p64"]=p64

def p65(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(1, 72):
        cf[k - 1] = (t1 + t2) ** (2 * k - 1) + np.sin(k * t1) * np.cos(k * t2) + np.log(np.abs(k ** t2) + 1) * np.real(t1 ** t2) + np.abs(np.imag(t1 ** (2 * k + 1) + t2 ** (2 * k)))
        cf[k - 1] = np.conj(cf[k - 1]) * (-1) ** k
        if k % 2 == 0:
            cf[k - 1] /= (k + t1)
    return cf.astype(np.complex128)
    
ALLOWED["p65"]=p65

def p66(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0:25] = np.real(t1) * np.log(np.abs(t2) + 1) * (np.arange(1, 26) ** 2)
    cf[25:50] = np.imag(t2) * np.log(np.abs(t1) + 1) * (np.arange(1, 26) ** 3)
    cf[50:70] = np.abs(t1) * np.abs(t2) * np.log(np.abs(t1 + t2) + 1) * (np.arange(1, 21))
    cf[70] = np.sum(cf[0:70]) * np.angle(t1 + t2)
    return cf.astype(np.complex128)
    
ALLOWED["p66"]=p66

def p67(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1 ** 7 + 1j * t2 ** 8
    cf[1] = 2 * t1 ** 6 - 1j * t2 ** 7
    cf[2] = 3 * t1 ** 5 + 3j * t2 ** 6
    cf[3] = 5 * t1 ** 4 - 5j * t2 ** 5
    cf[4] = 7 * t1 ** 3 + 7j * t2 ** 4
    cf[5] = 11 * t1 ** 2 - 11j * t2 ** 3
    cf[6] = 13 * t1 + 13j * t2
    cf[70] = np.abs(t1 * t2) ** 2 * np.angle(t1 * t2) + np.sin(np.real(t1)) - np.cos(np.imag(t2))
    for k in range(8, 71):
        cf[k] = np.real((t1 + 1j * t2) ** (70 - k)) - np.imag((t1 - 1j * t2) ** (k - 1))
    return cf

ALLOWED["p67"]=p67

def p68(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0:5] = np.abs(t1) ** (np.arange(1, 6))
    for i in range(6, 71):
        cf[i] = (i * t1 + 2 * i * t2) / (i + 1)
    cf[70] = np.abs(t1) * np.abs(t2) * np.angle(t1) * np.angle(t2) * np.sin(np.abs(t1 + t2))
    cf[20:30] += np.log(np.abs(t1 + t2) + 1) * np.exp(1j * np.pi / 10 * np.arange(1, 11))
    cf[50:60] += 1j * (cf[0:10] / np.arange(11, 21))
    cf[60:70] -= np.sin(cf[0:10])
    cf[30:40] += np.cos(t1 + t2) ** np.arange(1, 11)
    return cf.astype(np.complex128)

ALLOWED["p68"]=p68

def p69(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for i in range(1, 72):
        j = 71 - i
        cf[i - 1] = ((np.real(t1) + np.imag(t1) * j) / np.abs(t2 + i)) * np.sin(np.angle(t1 + t2 * i)) + np.log(np.abs(t1 * t2) + 1) * np.cos(2 * np.pi * i / 71)
    cf[cf == 0] = np.real(t1) ** 2 - np.imag(t1) * np.imag(t2)
    return cf

ALLOWED["p69"]=p69

def p70(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for i in range(1, 72):
        cf[i - 1] = np.real(t1) * np.real(t2) * (i ** 2) / np.exp(np.abs(t1) * 1j) + np.imag(t1) * np.imag(t2) * (i ** 3) / np.exp(np.abs(t2) * 1j)
    cf[1::2] *= -1
    p = np.arange(1, 72)
    cf[p ** 2 <= 71] += 1j * np.abs(t1) * np.abs(t2)
    return cf.astype(np.complex128)

ALLOWED["p70"]=p70

def p71(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for i in range(1, 72):
        cf[i - 1] = ((np.real(t1) ** i * np.imag(t2) ** i) + (np.imag(t1) ** i * np.real(t2) ** i)) / (i ** 2 + 1)
    cf[0:5] *= 1000
    cf[8:15] *= -500
    cf[17:25] *= 250
    cf[29:35] *= -125
    cf[36:45] *= 60
    cf[49:54] *= -30
    cf[55:64] *= 15
    cf[69:71] *= -5
    return cf
    
ALLOWED["p71"]=p71

def p72(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1 + t2
    cf[1:71] = np.arange(1, 71) * (t1 - t2 + (np.sin(np.arange(1, 71)) + 1j * np.cos(np.arange(1, 71))))
    roots = np.abs(cf[1:71])
    sorted_roots = np.sort(roots)[::-1]
    cf[1:71] = sorted_roots * (t1 + t2 * 1j * np.arange(1, 71))
    cf = np.real(cf) + np.imag(cf) + (np.sin(np.angle(cf + 1)) + 1j * np.cos(np.angle(cf + 1)))
    return cf

ALLOWED["p72"]=p72

def p73(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    v1 = (t1 + t2) * 1e30 
    cf[0] = v1
    v2 = (t1 - t2) * 10e28
    cf[1] = v2
    v3 = (t1 + t2) * 1e26 
    cf[2] = v3
    for k in range(4, 22):
        cf[k - 1] = 10.0 ** (30 - k) * (np.cos(t1) + np.sin(t2))
    for k in range(22, 32):
        cf[k - 1] = 10.0 ** (k - 21) * (np.cos(t1) - np.sin(t2))
    for k in range(32, 42):
        cf[k - 1] = 10.0 ** (42 - k) * (t1 + t2) * (np.cos(t1 + t2) + np.sin(t1 - t2))
    cf[41] = 10.0 ** 21 * (t1 - t2)
    for k in range(43, 54):
        cf[k - 1] = 10.0 ** (53 - k) * (np.abs(t1 + t2) + np.angle(t1 - t2))
    for k in range(54, 65):
        cf[k - 1] = 10.0 ** (64 - k) * (np.abs(t1 - t2) + np.angle(t1 + t2))
    for k in range(65, 72):
        cf[k - 1] = 10.0 ** (71 - k) * (np.sin(t1) + np.cos(t2))
    return cf
    
ALLOWED["p73"]=p73

def p74(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for i in range(1, 36):
        cf[i - 1] = i * (t1 + i * t2) ** (1 / i)
        cf[70] = np.conj(cf[i - 1])
    cf[35] = 2 * t1 + 3 * np.abs(t2)
    return cf

ALLOWED["p74"]=p74

def p75(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    powers = np.arange(0, 71)
    cf[0] = 100 * t1 ** 3 + 110 * t1 ** 2 + 120 * t2 - 130
    cf[1] = 200 * t2 ** 3 - 210 * t2 ** 2 + 220 * t2 - 230
    cf[4] = np.abs(t1) ** 4
    cf[9] = np.angle(t2) ** 6
    cf[14] = np.log(np.abs(t1 + 1j * t2)) + 1
    cf[np.array([19, 39],dtype=np.intp)] = np.real(1j * cf[4] * t1 * t2)
    cf[np.array([29, 59],dtype=np.intp)] = np.imag(cf[1] * np.conj(cf[0]))
    cf[34] = np.sin(cf[1]) + np.cos(cf[0])
    cf[2] = np.abs(cf[9]) ** 2
    i1 = np.array( [ 2, int(cf[1].real) ], dtype=np.intp )
    cf[3] = np.prod(cf[i1])
    cf[8] = np.sum(cf[np.array([19, 39, 59],dtype=np.intp)])
    cf[15:71] = powers[15:71] * np.abs(t1 - t2)
    cf[70] = np.prod(cf[0:4])
    return cf
    
ALLOWED["p75"]=p75

def p76(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1 ** 7
    cf[1] = t2 ** 7
    cf[2] = t1 ** 6
    cf[3] = t2 ** 6
    cf[4] = t1 ** 5
    cf[5] = t2 ** 5
    cf[6] = t1 ** 4
    cf[7] = t2 ** 4
    for k in range(9, 23):
        cf[k - 1] = np.sin((t1 + t2) / (k - 8)) ** k
    for k in range(23, 37):
        cf[k - 1] = np.cos((t1 - t2) / (k - 22)) ** k
    for k in range(37, 51):
        cf[k - 1] = np.cos((t1 + 1j * t2) / (k - 36)) ** k
    for k in range(51, 65):
        cf[k - 1] = np.sin((t1 - 1j * t2) / (k - 50)) ** k
    for k in range(65, 72):
        cf[k - 1] = (t1 + 1j * t2) ** (k - 64)
    return cf.astype(np.complex128)
    
ALLOWED["p76"]=p76

def p77(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(1, 72):
        if k % 2 == 0:
            cf[k - 1] = (np.log(np.abs(t1) + 1) ** k + np.log(np.abs(t2) + 1) ** (71 - k)) * np.sin(k * t1 + (71 - k) * t2)
        else:
            cf[k - 1] = (np.log(np.abs(t1) + 1) ** k - np.log(np.abs(t2) + 1) ** (71 - k)) * np.cos(k * t1 - (71 - k) * t2)
    r = np.abs(t1) * np.abs(t2)
    for k in range(50, 72):
        cf[k - 1] *= (r ** (k - 50))
    for k in range(15, 36):
        cf[k - 1] *= 2 * (r ** (71 - k))
    cf = np.real(cf) + 1j * np.imag(np.conj(cf))
    return cf.astype(np.complex128)
    
ALLOWED["p77"]=p77

def p78(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0:10] = np.abs(t1) ** (np.arange(1, 11) / 5) * np.log(1 + np.abs(t2))
    cf[10:20] = np.real(t1) ** (np.arange(1, 11)) * np.angle(t2) * (-1) ** (np.arange(1, 11))
    cf[20:30] = np.imag(t1) ** (np.arange(1, 11) / 3) * np.abs(t2) ** (np.arange(1, 11) / 4) * (-1) ** (np.arange(1, 11))
    cf[30:40] = np.abs(t1 * t2) ** (np.arange(1, 11) / 2) * (np.arange(1, 11))
    cf[40:50] = np.real((t1 + t2) ** (np.arange(1, 11) / 2)) * np.cos(np.angle(t1 * t2)) * (-1) ** (np.arange(1, 11))
    cf[50:60] = np.imag((t1 + t2) ** (np.arange(1, 11) / 3)) * np.sin(np.angle(t1 - t2)) * (-1) ** (np.arange(1, 11))
    cf[60:70] = np.real(t1 ** (np.arange(1, 11))) * np.abs(t2 ** (np.arange(1, 11))) * np.log(1 + np.abs(t1 + t2))
    cf[70] = np.abs(t1 - t2) * np.log(1 + np.abs(t1))
    return cf

ALLOWED["p78"]=p78

def p79(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0:35] = (np.arange(1, 36)) * (t1 + t2) * np.abs(t1) ** (np.arange(1, 36))
    cf[35:70] = (np.arange(35, 0, -1)) * (t1 - t2) * np.abs(t2) ** (np.arange(35, 0, -1))
    cf[70] = np.abs(t1) * np.abs(t2) + np.imag(t1 * np.conj(t2))
    return cf

ALLOWED["p79"]=p79

def p80(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1 ** 70 + t2 ** 70
    cf[1:70] = np.sin(t1 * t2 * (np.arange(1, 70))) ** 2
    cf[13:28] = np.log(np.abs(t2) + 1) ** 2 * cf[13:28]
    cf[30:46] *= np.log(np.abs(t1) + 1)
    for i in range(2, 5):
        cf[i * 15] += i * np.abs(t1) * np.abs(t2)
    cf[70] = np.real(t1) ** 3 - np.imag(t2) ** 2
    return cf

ALLOWED["p80"]=p80

def p81(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(1, 72):
        cf[k - 1] = (k + 20) * np.sin(t1 * k) * np.cos(t2 * k) + np.abs(t1) ** k + np.abs(t2) ** k
    cf[np.arange(2, 71, 5)] += np.abs(t1) * np.abs(t2)
    cf[np.arange(3, 70, 7)] += ((-1) ** (np.arange(1, 11))) * np.angle(t1 + t2)
    cf[np.arange(6, 67, 9)] += ((-1) ** (np.arange(1, 8))) * np.log(np.abs(t1 + t2) + 1)
    cf[np.arange(5, 71, 7)] *= np.real(t1 + t2)
    cf[np.arange(7, 64, 11)] *= np.imag(t1 + t2)
    cf[np.arange(1, 72, 7)] *= np.conj(t1 + t2)
    return cf
    
ALLOWED["p81"]=p81

def p82(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = 1 + t1 * t2
    cf[1] = 2 + np.abs(t1) * np.abs(t2)
    cf[2] = 3 + np.abs(t1 + t2)
    for i in range(4, 37):
        cf[i - 1] = i + cf[i - 2] * np.sin(i * t2) + cf[i - 3] * np.cos(i * t1) + cf[i - 4] * np.log(np.abs(i * t1 * t2 + 1))
    for i in range(37, 71):
        cf[i - 1] = 70 - i + cf[70 - min(i, 69)] * np.sin((70 - i) * t1) + cf[69 - min(i, 68)] * np.cos((70 - i) * t2) + cf[68 - min(i, 67)] * np.log(np.abs((70 - i) * t1 * t2 + 1))
    cf[70] = np.sum(cf[0:70]) + np.real(np.angle(t1 - t2)) + np.imag(np.angle(t1 + t2))
    return cf
    
ALLOWED["p82"]=p82

def p83(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for i in range(1, 72):
        cf[i - 1] = ((t1 + t2) ** (i - 1)) * np.sin(i) + ((t1 - t2) ** (70 - i + 1)) * np.cos(i)
    cf[0] = np.real(cf[0]) + 1j * np.imag(cf[70])
    cf[70] = np.real(cf[70]) + 1j * np.imag(cf[0])
    cf[35] += np.log(np.abs(t1 * t2)) ** 2
    return cf
    
ALLOWED["p83"]=p83

def p84(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = np.real(t1) + 1000j
    cf[1] = np.log(1 + np.abs(t1 + t2)) * 1000
    for k in range(3, 36):
        cf[k - 1] = (-1) ** k * (np.real(t1 ** k) + np.imag(t2 ** k)) * 1000 / (k ** 2)
    for k in range(36, 71):
        cf[k - 1] = (-1) ** (k + 1) * (np.abs(t1) ** (70 - k) + np.abs(np.sin(t1 + t2))) / (k ** 2)
    cf[70] = np.abs(t1) + np.cos(np.angle(t2)) * 1000
    return cf

ALLOWED["p84"]=p84

def p85(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(1, 72):
        cf[k - 1] = np.real(t1 ** k + t2 ** (k - 1)) + np.imag((t1 + t2) ** (70 - k))
        if k % 2 == 0:
            cf[k - 1] += np.sin(np.angle(t1 + t2) * k)
        else:
            cf[k - 1] += np.cos(np.angle(t1 + t2) ** k)
        if k % 3 == 0:
            cf[k - 1] *= np.abs(t1 - t2) ** (k / 10)
        if k % 4 == 0:
            cf[k - 1] += np.log(np.abs(t1) + 1) ** k
        if k % 5 == 0:
            cf[k - 1] -= np.log(np.abs(t2) + 1) ** (71 - k)
    cf[35] *= t1 * t2
    cf[65] *= np.conj(t1) * np.conj(t2)
    return cf
    
ALLOWED["p85"]=p85

# FIXME
def p86(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(1, 72):
        cf[k - 1] = np.cos(k * t1) ** k + 1j * np.sin(k * t2) ** k
    cf[1::2] **= -1
    cf[2::3] **= -2
    for r in range(5, 66, 5):
        cf[r - 1] = (t1 * t2) ** r
    cf[70] = (np.abs(t1) ** 2 + 2 * np.real(t1) * np.imag(t2) + 3 * np.abs(t2) ** 2)
    return cf

ALLOWED["p86"]=p86

def p87(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(1, 36):
        cf[k - 1] = (t1 + 1j * t2) ** k + np.log(np.abs(t1 + k * t2) + 1) * np.real(t1 * t2)
        cf[70 - k] = k * (t1 - 1j * t2) ** k - np.log(np.abs(t2 - k * t1) + 1) * np.imag(t1 * t2)
    cf[35] = 100 * np.abs(t1) * np.abs(t2)
    cf[36] = 200 * np.angle(t1) * np.angle(t2)
    cf[37:72] = cf[0:34] - cf[37:72]
    return cf

ALLOWED["p87"]=p87

def p88(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = np.log(np.abs(t1) + 1) * np.log(np.abs(t2) + 1)
    for k in range(2, 37):
        cf[k - 1] = np.sin(k * t1) + np.cos(k * t2) - k ** 2
    for k in range(37, 71):
        cf[k - 1] = np.sin((71 - k) * t1) - np.cos((71 - k) * t2) + (71 - k) ** 2
    cf[70] = np.real(t1 * t2) + np.imag(t1 * t2)
    return cf.astype(np.complex128)
    
ALLOWED["p88"]=p88

def p89(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1 * t2 - 1j * np.abs(t2 - t1)
    for k in range(2, 7):
        cf[k - 1] = cf[k - 2] + np.sin(k * t1) + np.cos(k * t2)
    for k in range(7, 12):
        cf[k - 1] = cf[k - 2] + np.log(np.abs(t1 - k)) - np.log(np.abs(t2 - k))
    for k in range(12, 17):
        cf[k - 1] = cf[k - 2] - np.sin(k * t1) - np.cos(k * t2)
    for k in range(17, 22):
        cf[k - 1] = cf[k - 2] - np.log(np.abs(t1 - k)) + np.log(np.abs(t2 - k))
    for k in range(22, 27):
        cf[k - 1] = cf[k - 2] + np.sin(k * t1) + np.cos(k * t2)
    for k in range(27, 32):
        cf[k - 1] = cf[k - 2] + np.log(np.abs(t1 - k)) - np.log(np.abs(t2 - k))
    for k in range(32, 37):
        cf[k - 1] = cf[k - 2] - np.sin(k * t1) - np.cos(k * t2)
    for k in range(37, 42):
        cf[k - 1] = cf[k - 2] - np.log(np.abs(t1 - k)) + np.log(np.abs(t2 - k))
    for k in range(42, 47):
        cf[k - 1] = cf[k - 2] + np.sin(k * t1) + np.cos(k * t2)
    for k in range(47, 52):
        cf[k - 1] = cf[k - 2] + np.log(np.abs(t1 - k)) - np.log(np.abs(t2 - k))
    for k in range(52, 57):
        cf[k - 1] = cf[k - 2] - np.sin(k * t1) - np.cos(k * t2)
    for k in range(57, 62):
        cf[k - 1] = cf[k - 2] - np.log(np.abs(t1 - k)) + np.log(np.abs(t2 - k))
    for k in range(62, 67):
        cf[k - 1] = cf[k - 2] + np.sin(k * t1) + np.cos(k * t2)
    for k in range(67, 72):
        cf[k - 1] = cf[k - 2] + np.log(np.abs(t1 - k)) - np.log(np.abs(t2 - k))
    return cf.astype(np.complex128)
    
ALLOWED["p89"]=p89

def p90(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1 ** 3 - t2 ** 2
    cf[1] = np.real(3 * t1 ** 2 * t2 - t2 ** 3)
    cf[2] = np.imag(3 * t1 * t2 ** 2 - t1 ** 3)
    cf[3] = 4 * t1 ** 2 - 6 * t1 * t2 + 4 * t2 ** 2
    for k in range(5, 72):
        cf[k - 1] = np.abs(t1 * t2) * np.sin(k * t1 + t2) + np.cos(k * np.conj(t1 + t2))
    return cf
    
ALLOWED["p90"]=p90

def p91(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(1, 71):
        cf[k - 1] = t1 ** k * t2 ** k
    cf[0] = np.log(np.abs(t1 + t2)) + 1
    cf[1] = np.log(np.abs(t1 * t2)) + 1
    cf[3] = np.abs(t1) ** 2 + np.abs(t2) ** 2
    cf[5] = np.abs(t1) ** 3 - np.abs(t2) ** 3
    cf[7] = np.abs(t1) ** 4 + np.abs(t2) ** 4
    cf[9] = t1 ** 5 - t2 ** 5
    cf[19] = np.abs(t1) * np.sin(np.angle(t2))
    cf[29] = np.abs(t2) * np.real(t1)
    cf[39] = np.abs(t1) * np.imag(t2)
    cf[49] = np.abs(t2) * np.angle(t1)
    cf[59] = np.abs(t1 * t2) * np.cos(np.angle(t2))
    cf[69] = np.abs(t2 * t1) * np.real(t1)
    cf[70] = np.sum(cf[0:70])
    return cf
    
ALLOWED["p91"]=p91

def p92(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(1, 72):
        cf[k - 1] = t1 ** k + t2 ** (71 - k)
    cf[14:25] = 1j * cf[14:25]
    cf[29:45] = np.conj(cf[29:45])
    cf[45:70] = -cf[45:70]
    cf[54:71] = cf[54:71] * (1 + 2j)
    cf[70] = np.real(t1) ** 3 + np.imag(t2) ** 3 - np.log(np.abs(t1) * np.abs(t2) + 1)
    return cf
    
ALLOWED["p92"]=p92

def p93(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(1, 36):
        cf[k - 1] = k * (t1 ** (2 * k)) * (t2 ** (2 * (35 - k))) / np.sin(k * np.pi / 180)
        cf[71 - k] = k * (t2 ** (2 * k)) * (t1 ** (2 * (35 - k))) / np.cos(k * np.pi / 180)
    cf[35] = 100 * np.real(t1) * np.imag(t2) + 100 * np.imag(t1) * np.real(t2)
    cf[70] = np.abs(t1 + t2) * np.angle(t1 - t2)
    return cf

ALLOWED["p93"]=p93

def p94(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[3] = -t1 ** 9
    cf[5] = t2 ** 8
    cf[7] = -t1 ** 7
    cf[9] = t2 ** 6
    cf[10] = -t1 ** 5
    cf[12] = t2 ** 4
    cf[14] = -t1 ** 3
    cf[16] = t2 ** 2
    cf[20] = -t1
    cf[30] = 5e5
    cf[40] = -5e6
    cf[50] = 5e7
    cf[60] = -5e8
    cf[70] = 5e9
    multiplier = np.sin(t1) * np.sin(t2) - np.cos(t1) * np.cos(t2)
    cf[3:71] *= multiplier
    return cf.astype(np.complex128)

ALLOWED["p94"]=p94

def p95(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = np.real(t1) ** 3 - np.imag(t1) ** 2 + 2 * np.imag(t1) * np.real(t1) - np.real(t2) + np.imag(t2) ** 2
    cf[1] = np.imag(t1) ** 3 - 5 * np.real(t1) ** 2 + 2 * np.real(t1) * np.imag(t1) + 5 * np.real(t2) - 2 * np.imag(t2) ** 2
    for k in range(3, 72):
        cf[k - 1] = np.abs(np.sin(k * t1)) + np.abs(np.cos(k * t2)) - np.abs(t1 ** k + t2 ** (k - 1))
    cf[29:40] = np.abs(cf[29:40]) / (np.abs(t1 - t2) ** 2 + 1)
    cf[49:60] = -np.abs(cf[49:60]) / (np.abs(t1 + t2) ** 2 + 1)
    cf[64:71] = cf[0:7] * (np.abs(t1) ** 2 + np.abs(t2) ** 2)
    return cf

ALLOWED["p95"]=p95

def p96(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1 ** 5 - t2 ** 4 + t1 ** 2 - t2 ** 2 + np.abs(t1) + np.abs(t2)
    cf[50] = t2 ** 6 - t1 ** 4 + t2 ** 3 - t1 ** 2 + np.angle(t1) + np.sin(t2)
    cf[70] = t1 ** 7 + t2 ** 5 - t1 ** 3 - t2 ** 2 + np.cos(t1) - np.sin(t2)
    for k in range(2, 51):
        cf[k - 1] = k * cf[k - 2] + np.abs(cf[0]) / k
    for r in range(52, 71):
        cf[r - 1] = r * cf[r - 2] + np.abs(cf[50]) / r
    return cf.astype(np.complex128)

ALLOWED["p96"]=p96

def p97(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(1, 72):
        cf[k - 1] = (t1 + 1j * t2) * k ** (-np.abs(t1) * np.log(np.abs(k + 1)))
    for k in range(1, 11):
        cf[k - 1] = (t1 + 1j * t2) * np.abs(k ** 3 * np.cos(np.imag(t1 + k * t2)) - np.sin(np.real(t1 - k * t2)))
    for k in range(61, 72):
        cf[k - 1] = (t1 + 1j * t2) * np.abs(k * cf[k - 2]) / np.abs(k ** 3 * np.cos(np.imag(t1 + k * t2)))
    return cf

ALLOWED["p97"]=p97

def p98(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(1, 36):
        cf[k - 1] = (t1 + t2) * np.sin(t2 * k) / k ** 2
    for k in range(36, 71):
        cf[k - 1] = (t1 - t2) * np.cos(t1 * (71 - k)) / (71 - k) ** 2
    cf[70] = np.real(t1) * np.imag(t2) - np.real(t2) * np.imag(t1)
    return cf
    
ALLOWED["p98"]=p98

def p99(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1 + t2 + 1
    cf[1:10] = np.real(t1) ** 2 + np.imag(t2) ** 2
    cf[10:20] = np.real(t2) ** 2 + np.imag(t1) ** 2
    cf[20:30] = np.abs(t1 * t2) ** 2
    cf[30:40] = np.abs(t1 + t2) ** 2
    cf[40:50] = np.abs(t1) * np.abs(t2)
    cf[50:60] = np.angle(t1) + np.angle(t2)
    cf[60:70] = np.sin(t1 + t2)
    cf[70] = np.cos(t1 - t2)
    return cf
    
ALLOWED["p99"]=p99

def p100(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    iter = complex(1)
    for j in range(1, 72):
        cf[j - 1] = iter
        iter *= (np.log(np.abs(t1 + 1j * t2) + 1) / (71 - j + 1) + np.conj(iter))
    return cf
    
ALLOWED["p100"]=p100

#FIXME
def p101(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0:10] = 1000 * np.arange(1, 11)
    cf[10:71] = 1
    cf[14] = -1 * np.abs(t1)**3
    cf[29] = -1 * np.abs(t2)**4
    cf[44] = np.abs(t1)**5
    cf[59] = np.abs(t2)**6 
    cf[19] = np.abs(t1)**2 * np.sin(np.angle(t1))
    cf[39] = np.abs(t2)**3 * np.cos(np.angle(t2))
    cf[24] = np.log(np.abs(t1) + 1) * np.abs(t1)
    cf[49] = np.log(np.abs(t2) + 1) * np.abs(t2)
    for j in range(1, 36):
        cf[2*j] = cf[2*j] * (np.sin(j * t1) + np.cos(j * t2)) + cf[2*j + 1] * (np.cos(j * t1) + np.sin(j * t2))
    return cf
    
ALLOWED["p101"]=p101

def p102(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = 1000 * (t1 + t2)**2
    for k in range(1, 20):
        cf[k] = (k + 1) * cf[k - 1] + np.sin((k + 1) * t1) + np.cos((k + 1) * t2)
    for k in range(20, 40):
        cf[k] = (k + 1) * cf[k - 1] - np.sin((k + 1) * t1) - np.cos((k + 1) * t2)
    for k in range(40, 60):
        cf[k] = (k + 1) * cf[k - 1] + np.sin((k + 1) * t1 * t2) + np.cos((k + 1) * t1 * t2)
    for k in range(60, 70):
        cf[k] = (k + 1) * cf[k - 1] - np.sin((k + 1) * t1 * t2) - np.cos((k + 1) * t1 * t2)
    cf[70] = np.abs(cf[69]) + np.angle(t1) - np.angle(t2) + np.real(t1 * t2) - np.imag(np.conj(t1) * t2)
    return cf
    
ALLOWED["p102"]=p102

def p103(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t2 * np.log(np.abs(t1) + 1)
    cf[1] = t1 * np.log(np.abs(t2) + 1)
    
    for k in range(2, 32, 2):
        cf[k] = np.sin(k * t1) * np.log(np.abs(t2) + 1)
        cf[k + 1] = np.cos(k * t2) * np.log(np.abs(t1) + 1)
    
    for k in range(32, 52, 2):
        cf[k] = np.cos(k * t1) * np.log(np.abs(t2) + 1)
        cf[k + 1] = np.sin(k * t2) * np.log(np.abs(t1) + 1)
    
    for k in range(52, 72, 2):
        cf[k] = t1 * np.log(np.abs(t2 * (k + 1)) + 1)
        cf[k + 1] = t2 * np.log(np.abs(t1 * (k + 1)) + 1)
    
    mod_cf = (71 - np.arange(1, 72)) * np.abs(cf)
    arg_cf = np.arange(1, 72) / 71 * np.angle(cf)
    cf = mod_cf * np.exp(1j * arg_cf)
    return cf
    
ALLOWED["p103"]=p103

def p104(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(1, 72):
        cf[k - 1] = ((np.abs(t1)**(1/k)) * (np.cos(np.angle(t1)) + 1j * np.sin(np.angle(t1))) + 
                        (np.abs(t2)**(1/k)) * (np.cos(np.angle(t2)) + 1j * np.sin(np.angle(t2)))) / k
    cf[np.arange(0, 71, 3)] *= -1
    cf[np.arange(1, 71, 4)] *= 2
    cf[np.arange(2, 71, 5)] *= 3
    cf[np.arange(3, 71, 6)] *= 4
    cf[np.arange(4, 71, 7)] *= 5
    return cf
    
ALLOWED["p104"]=p104

def p105(z,a,state):
    t1, t2 = z[0], z[1]
    n = 71
    cf = np.zeros(n, dtype=np.complex128)
    for k in range(1, n+1):
            cf[k - 1] = np.sin(k * (np.real(t1) * np.imag(t2))**3) + np.cos(k * np.log(np.abs(t1 * t2 + 1)) * np.angle(t1 + np.conj(t2)))
    return cf

ALLOWED["p105"]=p105

def p106(z,a,state):
    t1, t2 = z[0], z[1]
    n = 71
    cf = np.zeros(n, dtype=np.complex128)
    for k in range(1, 72):
        cf[k - 1] = ((t1**3 + t2**2)**2 + np.abs(t1 - t2) + np.sin(t1 * t2)) * np.abs(t1 + t2)**(1/k)
    cf[0] *= 100
    cf[1] *= 90
    cf[2] *= 80
    cf[3] *= 70
    cf[4] *= 60
    cf[5] *= 50
    cf[6] *= 40
    cf[7] *= 30
    cf[8] *= 20
    cf[9] *= 10
    cf[11] *= 5
    cf[23] *= 4
    cf[35] *= 3
    cf[47] *= 2
    cf[59] *= 1
    for k in range(15, 72):
        cf[k] = -cf[k] * np.log(np.abs(k))
    return cf.astype(np.complex128)
    
ALLOWED["p106"]=p106

def p107(z,a,state):
    t1, t2 = z[0], z[1]
    n = 71
    cf = np.zeros(n, dtype=np.complex128)
    for r in range(1, n+1):
        cf[r - 1] = (100 * (t1 ** (n - r))) * np.sin(0.5 * t1 * r) + \
                        (100 * (t2 ** r)) * np.cos(0.5 * t2 * r)
    cf[14] = 100 * t2**3 - 100 * t2**2 + (100 * t2 - 100)
    cf[29] = 100 * np.log(np.abs(t1 * t2) + 1)
    cf[44] = np.abs(10 * t1 + 0.5 * t2)
    cf[59] = np.angle(0.2 * t1 - 3j * t2)
    cf[70] = np.real(10 * t1 + 0.5 * t2)
    return cf
    
ALLOWED["p107"]=p107

def p108(z,a,state):
    t1, t2 = z[0], z[1]
    n = 71
    cf = np.zeros(n, dtype=np.complex128)
    cf[0] = t1 + t2
    cf[1] = -(t1**2 + t2)
    cf[2] = t1**2 - t2**2 - 1j
    cf[3:10] = [1 - t1, -1 + t2, 2 - t1, -2 + t2, 3 - t1, -3 + t2, 4 - t1]
    cf[10] = 15 * (np.real(t1) + np.imag(t2))
    cf[11] = -17 * np.angle(t1) * np.angle(t2)
    cf[14] = 30 * np.abs(t1) * np.abs(t2)
    cf[16] = -(t1**3 + t2**3)
    cf[18] = (t1**2 - t2**2) * 1j
    cf[19] = 5 + 1j * t1
    cf[24] = 50 * np.abs(t1 - t2)
    cf[29] = -40 * np.real(t1) + 35 * np.imag(t2)
    cf[34] = np.sum(np.array([3, 3, 9, 15, -12],dtype=np.complex128)) * (np.real(t1) - np.imag(t2))
    cf[39] = -t1**4 + t2**4 - 3
    cf[44] = 3 * np.angle(t1) + 4 * np.angle(t2)
    cf[49] = -55 * np.abs(np.abs(t1) - np.abs(t2))
    cf[54] = 33 * np.abs(t1)**3 + np.abs(t2)**2
    cf[59] = t1**5 + t2**5 - 29
    cf[64] = -22 * np.real(t1**2) + 22 * np.imag(t2**2)
    cf[69] = (np.sum(np.arange(1, 6)) * np.imag(t1)) + (np.prod(np.arange(1, 6)) * np.real(t2))
    return cf
    
ALLOWED["p108"]=p108

def p109(z,a,state):
    t1, t2 = z[0], z[1]
    n = 71
    cf = np.zeros(n, dtype=np.complex128)
    cf[0] = t1 + t2
    cf[1] = t1**2 - 2*t2 + 5
    cf[2] = np.conj(t1) * t2 + 7
    cf[3] = t2**2 - t1 + 11
    cf[4] = np.abs(t1 + t2) + 13
    cf[5] = np.angle(t1) * np.angle(t2) + 17
    cf[6] = t1 * t2 - 19
    cf[7] = t1**3 + t2**3 + 23
    cf[8] = np.sin(t1) + np.cos(t2) + 29
    cf[9] = np.log(np.abs(t1 + t2) + 1) + 31
    cf[10] = t1**2 - t2**2 + 37
    cf[11] = np.conj(t2) * t1 + 41
    cf[12] = np.imag(t1) * np.real(t2) - 43
    cf[13] = t1 * np.conj(t2) + 47
    cf[14] = np.abs(t1 - t2) + 53
    cf[15] = t1**4 - t2**4 + 59
    cf[16] = 61 - 5 * t1 * t2
    cf[17] = 67 + np.abs(t1**2 + t2**2)
    cf[18] = 71 + t1**5 + t2**5
    cf[19] = 73 - np.angle(t1) * np.angle(t2)
    cf[20] = 79 + np.abs(t1**3 + t2**3)
    cf[21] = 83 - t1**6 + t2**6
    cf[22] = 89 + np.sin(t1 + t2)
    cf[23] = np.abs(np.real(t1) * np.imag(t2)) + 97
    cf[24] = 101 + t1 * t2**2
    cf[25] = 103 - np.conj(t1) * np.real(t2)
    cf[26] = 107 + t1**7 - t2**7
    cf[27] = 109 + np.abs(np.conj(t1 - t2))
    cf[28] = 113 - np.abs(t1**2 - t2**2)
    cf[29] = 127 + (t1**8 * t2**8)
    cf[30] = t1 - t2 + np.abs(t1 * t2) + 131
    cf[31] = 137 + np.angle(t1**2) - np.angle(t2**2)
    cf[32] = 139 - t1**9 + t2**9
    cf[33] = np.log(np.abs(t1 * t2) + 1) + 149
    cf[34] = 151 + (np.abs(t1) + np.abs(t2))**2
    cf[35] = np.sin(2 * t1) - np.cos(2 * t2) + 157
    cf[36] = np.log(np.abs(t1 - t2) + 1) + 163
    cf[37] = 167 + np.real(t1**3) - np.imag(t2**3)
    cf[38] = 173 - (t1**2 * t2**2)**1.5
    cf[39] = 179 + np.angle(t1 * t2) + 1j
    cf[40] = 181 - np.conj(t1**3 - t2**3)
    cf[41] = 191 + np.abs(t1) * np.abs(t2)
    cf[42] = 193 - np.abs(np.real(t1) + np.imag(t2))
    cf[43] = 197 + np.sin(t1**2 + t2**2)
    cf[44] = 199 - t1 * t2**3
    cf[45] = t1 * np.imag(t2) + 211
    cf[46] = np.abs(t1**4 + t2**4) + 223
    cf[47] = 227 - np.conj(t1**2) * np.conj(t2**2)
    cf[48] = 229 + np.sin(t1 * t2) - np.cos(t1 - t2)
    cf[49] = 233 + t1**9 - t2**9
    cf[50] = 239 - np.abs(np.conj(t1**2 + t2**2))
    cf[51] = 241 + t1**3 + t2**3
    cf[52] = t1**10 + t2**10 + 251
    cf[53] = t1 * t2 * np.real(t1 + t2) - 257
    cf[54] = np.abs(t1 - t2) - 263
    cf[55] = t1**11 - t2**11 + 269
    cf[56] = 271 + np.abs(t1 * t2**2 - t2**3)
    cf[57] = 277 + np.sin(t1**3 - t2**3)
    cf[58] = 281 - np.conj(t1**2 * t2)
    cf[59] = np.conj(t1**5 + t2**5) + 283
    cf[60] = np.angle(t1**3 * t2**3) + 293
    cf[61] = 307 - np.sin(t1 * t2 + 1j)
    cf[62] = np.abs(t1**6 + t2**6) + 311
    cf[63] = 313 - np.cos(t1**3 - t2**3)
    cf[64] = np.angle(t1 * t2) + 317
    cf[65] = np.real(t1**2 - t2**2) - 331
    cf[66] = 337 + np.abs(t1**6 * t2**6)
    cf[67] = 347 - np.abs(t1**4 - t2**4)
    cf[68] = 349 + np.sin(np.conj(t1 - t2))
    cf[69] = 353 - np.cos(t1 + t2**2)
    cf[70] = np.abs((t1 + t2)**3 - 359)
    return cf
    
ALLOWED["p109"]=p109

def p110(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    prime_sequence = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59])
    for k in range(1, 36):
        cf[k - 1] = np.real(t1) * prime_sequence[k % len(prime_sequence)] + np.imag(t2) * k**2
        cf[70 - k] = np.real(t2) * prime_sequence[(70 - k) % len(prime_sequence)] - np.imag(t1) * k**2
    cf[35] = np.sum(prime_sequence) * (np.cos(np.angle(t1)) + 1j * np.sin(np.angle(t2)))
    return cf
    
ALLOWED["p110"]=p110

def p111(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71])
    
    def cyclo(n):
        return np.prod(1 - t1**n * np.exp((np.arange(1, n + 1) - 1) * 2j * np.pi / n))
    
    for r in range(1, 21):
        cf[r - 1] = (primes[r - 1] * (t1**r + t2**r)) / cyclo(r + 1)
    
    for r in range(21, 31):
        cf[r - 1] = cyclo(primes[r - 21]) * (t1 + t2)
    
    for r in range(31, 46):
        theta = np.angle(t1 + t2)
        cf[r - 1] = ((r - 30) * np.abs(t1 - t2) * np.cos(r * theta)) / (1 + np.abs(t1)**r + np.abs(t2)**r)
    
    for r in range(46, 61):
        cf[r - 1] = (np.sin(primes[r - 46] * t1) + np.cos(primes[r - 46] * t2)) * ((-1j) ** (r - 45)) / np.prod(np.arange(1, r + 1))
    
    for r in range(61, 71):
        cf[r - 1] = (np.log(np.abs(t1) + r) * np.real(t2**2 - t1**2)) / (primes[r - 61] * cyclo(r - 50))
    
    cf[70] = np.conj(np.sum(cf[0:35])) + np.prod(cf[35:70])
    
    return cf.astype(np.complex128)
    
ALLOWED["p111"]=p111

def p112(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    def phi_n(n):
        theta = 2 * np.pi / n
        w = np.exp(1j * theta)
        return w**np.arange(n)
    cf[0:10] = phi_n(10) + (t1**2 + np.abs(t2))
    cf[10:15] = np.exp(np.arange(2, 7) * np.angle(t1)) - np.real(t2)
    cf[15:20] = phi_n(5) * (t1 + 2j * t2)
    cf[20:30] = 100 - np.real(t1**3) + 1j * np.imag(t2**2)
    cf[30:40] = -40 + np.abs(t1 * t2) + 1j * np.angle(t1 - t2)
    cf[40:50] = 1j * phi_n(10)**(np.arange(2, 12)) - (t1 + np.conj(t2))
    cf[50:60] = 2 * np.log(np.abs(np.real(t1) + np.imag(t2)) + 1) * np.arange(2, 12)
    cf[60:70] = ((-1)**np.arange(1, 11)) * phi_n(10)**(np.arange(1, 11) * t1)
    cf[70] = np.sum(cf[0:70]) / 71
    return cf

ALLOWED["p112"]=p112

def p113(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    z = t1 + t2 * 1j
    for k in range(1, 36):
        cf[k - 1] = np.cos(np.pi * k / 35) * ((-1)**k) * np.abs(z)**k
        cf[70 - k] = np.sin(np.pi * (35 - k) / 35) * ((-1)**(k + 1)) * np.angle(z)**(35 - k)
    cf[35] = np.exp(np.abs(z))
    return cf
    
ALLOWED["p113"]=p113

def p114(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = np.real(t1**2 + t2**3) - np.imag(t1 * np.conj(t2))
    cf[1:4] = np.array([-827, 221, 653]) * (np.real(t1) + np.imag(t2))
    cf[4] = np.abs(t1 - 2j * t2)**5
    for j in range(6, 29):
        cf[j - 1] = np.cos(j * np.angle(t1 + t2)) * np.sin(j * np.abs(t1**2 + t2)) + j
    cf[28:41] = np.array([89, -233, 144, 377, 610, -987, 1597, -2584, 4181, -6765, 10946, -17711, 28657]) * np.abs(t1 - t2)
    for k in range(42, 62):
        cf[k - 1] = np.log(np.abs(k * t1 * np.conj(t2) + 71))
    cf[62:66] = np.array([3j, 2 - 8j, -6 + 11j, -5.5]) * (t1**3 - t2**3)
    cf[66:71] = np.tan(np.pi / 4) * np.exp(-(np.arange(66, 71)))
    cf[70] = np.exp(t1) - np.exp(t2) 
    return cf

ALLOWED["p114"]=p114

def p115(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    kappa = (1 + t1**2) * (1 - t2)**2 * np.angle(t1) * np.angle(t2)
    offset = np.real(t1) - np.imag(t2) + 1j * (np.imag(t1) + np.real(t2))
    
    cf[0] = kappa - t1**3 + t2**2
    cf[1] = -2 * offset + t1 * t2 + np.abs(t1 + t2)
    cf[2] = (3 + 2 * kappa) * (t1 - t2)
    cf[3] = 0.5 * (offset - 1j * kappa)
    cf[4:10] = np.abs(t1) / (np.arange(1, 7))
    cf[10:20] = -(t2**2) * (np.arange(1, 11))
    
    for k in range(21, 31):
        cf[k - 1] = (t1**k - t2**(k - 1)) / (k**2)
    
    cf[30:51] = np.real(offset) * (np.arange(20, 41)) + 0.1 * np.imag(offset) * (np.arange(1, 22))
    
    for k in range(51, 61):
        cf[k - 1] = np.imag(t1 * t2)**2 / (k**2)
    
    cf[61] = np.abs(offset) + 0.1 * t2**2 - 0.1 * t1**2
    cf[62] = 0.01 * (t1**3 - 2 * t2**3)
    cf[63] = 0.001 * (offset * np.conj(t2))
    cf[64:70] = ((np.arange(64, 70) + 1) * np.real(offset) + (np.arange(64, 70) + 1) * np.imag(offset)**2) / 2
    cf[70] = -t1 + 2j * t2
    cf[71] = (1 + (t1**3 * np.conj(t2))) / 3
    
    return cf

ALLOWED["p115"]=p115

def p116(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47])
    for k in range(1, 36):
        cf[k - 1] = primes[k % 15] * (t1**k + t2**k) * (-1)**k / (k + 1)
        cf[70 - k] = primes[(k + 11) % 15] * (t1**(71 - k) - t2**(71 - k)) * (-1)**(71 - k) / (71 - k + 1)
    cf[35] = np.sum(primes[:5]) * np.abs(t1 + t2) / (1 + np.abs(t1))
    cf[70] = 1 + 1j
    return cf

ALLOWED["p116"]=p116

def p117(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(1, 72):
        cf[k - 1] = (t1 + t2)**(k - 1) + ((-1)**k) * (np.exp(1j * k * np.pi / 71)) * (k**(1/3))
    cf *= (1 + np.log(np.abs(cf) + 1) / (1 + np.abs(t1 * t2)))
    cf[0:10] += (t1**2 + t2**2)**(1/3)
    cf[61:71] *= np.exp(-1j * np.angle(t1))
    return cf
    
ALLOWED["p117"]=p117

def p118(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53])
    f1 = t1 * np.sum(primes) + t2
    f2 = t2 * np.sum(primes[:8]) + np.conj(t1)
    cf[0:16] = primes[:16] * (t1 - t2)
    cf[16:32] = f1**2 - f2**2
    cf[32:48] = (t1**3 - t2**3) * (primes[:16] - f1)
    cf[48:64] = (primes[:16] * t1**2 + t2**3) - t1
    cf[64:70] = np.sin(cf[0:6] * t2) + np.cos(cf[0:6] * t1)
    cf[70] = np.prod(primes[:9])
    return cf
    
ALLOWED["p118"]=p118

def p119(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47])
    cycle = np.array([1, -1, 1j, -1j, 0.5, -0.5, 0.5j, -0.5j])
    for k in range(1, 16):
        cf[k - 1] = primes[k - 1] * (t1 + cycle[k % 8] * t2) / (k + 1)
    for k in range(16, 31):
        cf[k - 1] = np.real(t1)**k - np.imag(t2)**(k % 5 + 1) + cycle[k % 8]
    for k in range(31, 46):
        cf[k - 1] = np.abs(t1 * t2) * np.sin(np.angle(t2)**2) + np.cos(k * t1) * cycle[k % 8]
    for k in range(46, 56):
        cf[k - 1] = 1 / np.abs(t1 + t2) * k**2 + np.sum(primes[:3]) * 1j**k * cycle[k % 8]
    for k in range(56, 66):
        cf[k - 1] = np.log(np.abs(t1 * k + t2 * k + 1)) * cycle[k % 8]
    cf[66] = np.prod(primes[:10]) / (np.abs(t1)**2 + np.abs(t2)**2 + 1)
    cf[67] = np.sum(primes[10:15]) * (t1 - t2)**3
    cf[68] = np.real(t1**2 * t2) - np.imag(t2**2 * t1) + 1j
    cf[69] = (t1 * t2 + 1)**35
    cf[70] = -np.prod(cycle) * (t1 + t2)
    cf[71] = np.abs(t1 * cycle[0]) * np.abs(t2 * cycle[1]) + 1
    return cf.astype(np.complex128)

ALLOWED["p119"]=p119

def p120(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    theta = np.angle(t1) * np.angle(t2)
    mult_factors = (-1)**(np.arange(1, 71) + 1)
    cf[0:10] = (np.arange(1, 11)) * t1**2 - (np.arange(10, 0, -1)) * t2**2
    for k in range(11, 41):
        cf[k - 1] = (k % 2) * np.abs(t1) + (k % 3) * np.abs(t2) * np.exp((k / 5) * theta * 1j)
    cf[40:60] = ((np.arange(41, 61)) + np.log(np.abs(theta) + 1)) * np.conj(t1) * 5 * mult_factors[0:20]
    cf[60:71] = np.arange(60, 71) - np.arange(1, 12) * t2 - np.sum(np.arange(1, 12) * mult_factors[10:21])
    cf[70] = (np.sum(np.arange(1, 36)) + np.sum(np.arange(36, 72))) / np.prod(np.arange(1, 16))
    return cf.astype(np.complex128)
    
ALLOWED["p120"]=p120

def p121(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    w = np.exp(2 * np.pi * 1j / 7)
    for k in range(1, 8):
        cf[k - 1] = (t1**k - (t2 / w)**k) * np.real(w**k)
    cf[7] = -(t1**2+t2**2) + np.real(t1 * t2) + np.imag(t1 * t2)
    for k in range(9, 36):
        z = np.angle(t1) + np.angle(t2)
        cf[k - 1] = np.cos(k * z) + 1j * np.sin(k * z)
    for k in range(36, 71):
        cf[k - 1] = (np.abs(t1) * t2 + t1 * np.imag(t2))**2 / (k + 1)
    cf[70] = np.abs(t1) - np.abs(t2) + np.log(np.abs(t1 + t2 + 1) + 1)
    return cf
    
ALLOWED["p121"]=p121

def p122(z,a,state):
    t1, t2 = z[0], z[1]
    n = int(71)
    m = int(36)
    cf = np.zeros(n, dtype=np.complex128)
    for k in range(1, m):
        cf[k - 1] = (-1)**k * (k**2 + t1 * t2**k + k * np.abs(t1)) * (np.cos(k * np.angle(t2)))
    for k in range(m, n):
        cf[k - 1] = (k**3 + t2 * t1**k + k * np.abs(t2)) * (np.sin(k * np.angle(t1)))
    cf[n-1] = np.sum(np.abs(cf[0:(n-1)]))
    return cf
    
ALLOWED["p122"]=p122

def p123(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1.real**2 - t2.imag**2
    cf[1] = (t1 + t2)**2 - 7
    cf[2] = t1**2 - t2**2
    cf[3:10] = np.arange(3, 30, 4) * np.abs(t1 + 1j * t2)
    cf[10:20] = (t1 - t2).real * np.arange(11, 21)
    cf[20:30] = 1 / (1 + np.arange(21, 31)) * (t1 + t2).real
    cf[30] = np.angle(t1) * t2.imag
    cf[31:50] = 1000 * (-1)**np.arange(32, 51) * t1 * t2
    cf[50:60] = 2000 * (-1)**np.arange(51, 61) * np.log(np.abs(t1) + 1)
    cf[60:65] = 1j * np.conj(t1 * t2) * np.sqrt(np.arange(61, 66))
    cf[65:70] = np.arange(66, 71) * (np.arange(66, 71) - 1) / (np.abs(t1) + np.abs(t2) + 1)
    cf[70] = np.prod(np.arange(1, 6))
    return cf

ALLOWED["p123"]=p123

def p124(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61])
    for j in range(1, 36):
        cf[j - 1] = primes[j % len(primes)] * (t1**j - t2**(71 - j))
    for k in range(36, 72):
        cf[k - 1] = (np.abs(t1) * np.real(t2) - np.imag(t1) * np.angle(t2))**(142 - k) / (1 + np.abs(primes[k % len(primes)]))
    return cf

ALLOWED["p124"]=p124

def p125(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0:15] = [(-1)**j * (j**2) * (np.abs(t1) + np.abs(t2)) for j in range(1, 16)]
    cf[15:30] = [(-1)**(k + 1) * (k**3) * np.angle(t1 + 1j * t2) for k in range(16, 31)]
    cf[30:45] = [(-1)**(r + 1) * np.cos(r * t1) + np.sin(r * t2) for r in range(31, 46)]
    for s in range(46, 61):
        cf[s] = (-1)**s * (s**2) * np.conj(t1) * np.conj(t2)
    cf[61:70] = [n**3 * np.log(np.abs(t1 * t2) + 1) for n in range(61, 70)]
    cf[70] = 1
    return cf.astype(np.complex128)
    
ALLOWED["p125"]=p125

def p126(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for j in range(1, 36):
        cf[j - 1] = ((-1)**j) * (t1**3 + t2**2) * j**2
        cf[70 - j] = ((-1)**(j + 1)) * (t2**3 + t1**2) * j**1.5
    middle_index = 36
    for k in range(1, 6):
        cf[middle_index + k - 1] = np.exp(1j * np.pi * ((-1)**k) * (t1 + t2) / 2)
    cf[middle_index - 1] = np.log(np.abs(t1 + t2) + 1)
    cf[0] = np.sin(t1**3) + np.cos(t2**3)
    cf[70] = np.cos(t1) * np.sin(t2) + t1**2 + t2**2
    return cf

ALLOWED["p126"]=p126
    
def p127(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(0, 71):
        cf[k] = (t1**(70 - k) + np.conj(t2)**k) * (-1)**k * np.log(np.abs(t1 + t2) + k)
    return cf

ALLOWED["p127"]=p127

def p128(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    k = np.arange(1, 72)
    cf[:] = (-1)**k * (t1**k + np.conj(t2)**(71 - k)) * (72 - k)
    return cf
    
ALLOWED["p128"]=p128

def p129(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1**5 + 2 * t2**3
    cf[1] = np.conj(t1) * t2**2 - t1**3
    cf[2] = t1**4 - t2**4 + t1 * t2
    cf[3] = t1**2 * t2 + 3 * t1
    cf[4] = 4 * t2**5 - t1 * t2**2
    cf[5] = t1**3 - 5 * t2**6
    cf[6] = 6 * t1 * t2**3 + t2
    cf[7] = t1**7 - 7 * t2**7
    cf[8] = 8 * t1**2 * t2**4 - t1
    cf[9] = t1**8 + 9 * t2**8
    cf[10] = 10 * t1 * t2**5 - t1**2
    cf[11] = t1**9 - 11 * t2**9
    cf[12] = 12 * t1**2 * t2**6 + t2
    cf[13] = t1**10 - 13 * t2**10
    cf[14] = 14 * t1 * t2**7 - t1**3
    cf[15] = t1**11 + 15 * t2**11
    cf[16] = 16 * t1**2 * t2**8 - t2
    cf[17] = t1**12 - 17 * t2**12
    cf[18] = 18 * t1 * t2**9 + t1
    cf[19] = t1**13 - 19 * t2**13
    cf[20] = 20 * t1**2 * t2**10 - t2**2
    cf[21] = t1**14 + 21 * t2**14
    cf[22] = 22 * t1 * t2**11 - t1**4
    cf[23] = t1**15 - 23 * t2**15
    cf[24] = 24 * t1**2 * t2**12 + t2**3
    cf[25] = t1**16 + 25 * t2**16
    cf[26] = 26 * t1 * t2**13 - t1**5
    cf[27] = t1**17 - 27 * t2**17
    cf[28] = 28 * t1**2 * t2**14 + t2**4
    cf[29] = t1**18 + 29 * t2**18
    cf[30] = 30 * t1 * t2**15 - t1**6
    cf[31] = t1**19 - 31 * t2**19
    cf[32] = 32 * t1**2 * t2**16 + t2**5
    cf[33] = t1**20 + 33 * t2**20
    cf[34] = 34 * t1 * t2**17 - t1**7
    cf[35] = t1**21 - 35 * t2**21
    cf[36] = 36 * t1**2 * t2**18 + t2**6
    cf[37] = t1**22 + 37 * t2**22
    cf[38] = 38 * t1 * t2**19 - t1**8
    cf[39] = t1**23 - 39 * t2**23
    cf[40] = 40 * t1**2 * t2**20 + t2**7
    cf[41] = t1**24 + 41 * t2**24
    cf[42] = 42 * t1 * t2**21 - t1**9
    cf[43] = t1**25 - 43 * t2**25
    cf[44] = 44 * t1**2 * t2**22 + t2**8
    cf[45] = t1**26 + 45 * t2**26
    cf[46] = 46 * t1 * t2**23 - t1**10
    cf[47] = t1**27 - 47 * t2**27
    cf[48] = 48 * t1**2 * t2**24 + t2**9
    cf[49] = t1**28 + 49 * t2**28
    cf[50] = 50 * t1 * t2**25 - t1**11
    cf[51] = t1**29 - 51 * t2**29
    cf[52] = 52 * t1**2 * t2**26 + t2**10
    cf[53] = t1**30 + 53 * t2**30
    cf[54] = 54 * t1 * t2**27 - t1**12
    cf[55] = t1**31 - 55 * t2**31
    cf[56] = 56 * t1**2 * t2**28 + t2**11
    cf[57] = t1**32 + 57 * t2**32
    cf[58] = 58 * t1 * t2**29 - t1**13
    cf[59] = t1**33 - 59 * t2**33
    cf[60] = 60 * t1**2 * t2**30 + t2**12
    cf[61] = t1**34 + 61 * t2**34
    cf[62] = 62 * t1 * t2**31 - t1**14
    cf[63] = t1**35 - 63 * t2**35
    cf[64] = 64 * t1**2 * t2**32 + t2**13
    cf[65] = t1**36 + 65 * t2**36
    cf[66] = 66 * t1 * t2**33 - t1**15
    cf[67] = t1**37 - 67 * t2**37
    cf[68] = 68 * t1**2 * t2**34 + t2**14
    cf[69] = t1**38 + 69 * t2**38
    cf[70] = np.log(np.abs(t1) + 1) + np.real(t2)
    return cf.astype(np.complex128)
    
ALLOWED["p129"]=p129

def p130(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1**10 + 2 * t2**8 + t1 * t2
    cf[1] = 3 * t1**9 - t2**7 + t1**2 * t2
    cf[2] = -2 * t1**8 + 4 * t2**6 + t1 * t2**2
    cf[3] = t1**7 - 3 * t2**5 + 2 * t1**3 * t2
    cf[4] = 5 * t1**6 + t2**4 - t1**2 * t2**3
    cf[5] = -t1**5 + 6 * t2**3 + t1**4 * t2
    cf[6] = 4 * t1**4 - 2 * t2**2 + t1 * t2**4
    cf[7] = -3 * t1**3 + 5 * t2 + t1**3 * t2**2
    cf[8] = 2 * t1**2 - 4 * t2**5 + t1 * t2**5
    cf[9] = -t1 + 3 * t2**4 + t1**2 * t2**2
    cf[10] = t1**5 * t2 - t2**3 + t1 * t2**3
    cf[11] = 4 * t1**4 * t2 + 2 * t2**2 - t1 * t2**4
    cf[12] = -2 * t1**3 * t2**2 + 3 * t2 + t1**3 * t2
    cf[13] = 3 * t1**2 * t2**3 - t2**4 + t1 * t2**2
    cf[14] = -t1 * t2**4 + 2 * t2**2 + t1**2 * t2
    cf[15] = t1**6 + t2**7 - t1 * t2
    cf[16] = -t1**5 + 2 * t2**6 + t1**3 * t2
    cf[17] = 3 * t1**4 - t2**5 + t1**2 * t2**2
    cf[18] = -2 * t1**3 + 4 * t2**4 + t1 * t2**3
    cf[19] = 5 * t1**2 - t2**3 + t1 * t2**4
    cf[20] = -t1 + 3 * t2**2 + t1**2 * t2
    cf[21] = t1 * t2**5 - t2 + t1**3 * t2
    cf[22] = 2 * t1**4 * t2 - t2**2 + t1 * t2**5
    cf[23] = -3 * t1**3 * t2**2 + 4 * t2 + t1**2 * t2**3
    cf[24] = 4 * t1**2 * t2**3 - t2**4 + t1 * t2**4
    cf[25] = -2 * t1 * t2**4 + 3 * t2**2 + t1**2 * t2**2
    cf[26] = t1**5 * t2 - t2**3 + t1 * t2**3
    cf[27] = 3 * t1**4 * t2 + 2 * t2**2 - t1 * t2**4
    cf[28] = -t1**3 * t2**2 + 5 * t2 + t1 * t2**3
    cf[29] = 2 * t1**2 * t2**3 - t2**4 + t1 * t2**4
    cf[30] = -t1 * t2**4 + 4 * t2**2 + t1**2 * t2**2
    cf[31] = t1**5 * t2 - t2**3 + t1 * t2**3
    cf[32] = 4 * t1**4 * t2 + 3 * t2**2 - t1 * t2**4
    cf[33] = -t1**3 * t2**2 + 6 * t2 + t1 * t2**3
    cf[34] = 3 * t1**2 * t2**3 - t2**4 + t1 * t2**4
    cf[35] = -2 * t1 * t2**4 + 5 * t2**2 + t1**2 * t2**2
    cf[36] = t1**5 * t2 - t2**3 + t1 * t2**3
    cf[37] = 5 * t1**4 * t2 + 4 * t2**2 - t1 * t2**4
    cf[38] = -t1**3 * t2**2 + 7 * t2 + t1 * t2**3
    cf[39] = 4 * t1**2 * t2**3 - t2**4 + t1 * t2**4
    cf[40] = -3 * t1 * t2**4 + 6 * t2**2 + t1**2 * t2**2
    cf[41] = t1**5 * t2 - t2**3 + t1 * t2**3
    cf[42] = 6 * t1**4 * t2 + 5 * t2**2 - t1 * t2**4
    cf[43] = -t1**3 * t2**2 + 8 * t2 + t1 * t2**3
    cf[44] = 5 * t1**2 * t2**3 - t2**4 + t1 * t2**4
    cf[45] = -4 * t1 * t2**4 + 7 * t2**2 + t1**2 * t2**2
    cf[46] = t1**5 * t2 - t2**3 + t1 * t2**3
    cf[47] = 7 * t1**4 * t2 + 6 * t2**2 - t1 * t2**4
    cf[48] = -t1**3 * t2**2 + 9 * t2 + t1 * t2**3
    cf[49] = 6 * t1**2 * t2**3 - t2**4 + t1 * t2**4
    cf[50] = -5 * t1 * t2**4 + 8 * t2**2 + t1**2 * t2**2
    cf[51] = t1**5 * t2 - t2**3 + t1 * t2**3
    cf[52] = 8 * t1**4 * t2 + 7 * t2**2 - t1 * t2**4
    cf[53] = -t1**3 * t2**2 + 10 * t2 + t1 * t2**3
    cf[54] = 7 * t1**2 * t2**3 - t2**4 + t1 * t2**4
    cf[55] = -6 * t1 * t2**4 + 9 * t2**2 + t1**2 * t2**2
    cf[56] = t1**5 * t2 - t2**3 + t1 * t2**3
    cf[57] = 9 * t1**4 * t2 + 8 * t2**2 - t1 * t2**4
    cf[58] = -t1**3 * t2**2 + 11 * t2 + t1 * t2**3
    cf[59] = 8 * t1**2 * t2**3 - t2**4 + t1 * t2**4
    cf[60] = -7 * t1 * t2**4 + 10 * t2**2 + t1**2 * t2**2
    cf[61] = t1**5 * t2 - t2**3 + t1 * t2**3
    cf[62] = 10 * t1**4 * t2 + 9 * t2**2 - t1 * t2**4
    cf[63] = -t1**3 * t2**2 + 12 * t2 + t1 * t2**3
    cf[64] = 9 * t1**2 * t2**3 - t2**4 + t1 * t2**4
    cf[65] = -8 * t1 * t2**4 + 11 * t2**2 + t1**2 * t2**2
    cf[66] = t1**5 * t2 - t2**3 + t1 * t2**3
    cf[67] = 11 * t1**4 * t2 + 10 * t2**2 - t1 * t2**4
    cf[68] = -t1**3 * t2**2 + 13 * t2 + t1 * t2**3
    cf[69] = 10 * t1**2 * t2**3 - t2**4 + t1 * t2**4
    cf[70] = -9 * t1 * t2**4 + 12 * t2**2 + t1**2 * t2**2
    return cf.astype(np.complex128)
    
ALLOWED["p130"]=p130

def p131(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(1, 72):
        cf[k - 1] = (-1)**k * (np.real(t1)**k + np.imag(t2)**k) + (np.cos(k * t1) + np.sin(k * t2)) / k
    return cf
    
ALLOWED["p131"]=p131

def p132(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1**10 - t2**8 + np.real(t1 * t2)
    cf[1] = -t1**9 + t2**7 - np.imag(t1 + t2)
    cf[2] = t1**8 - t2**6 + np.real(t1**2 * t2)
    cf[3] = -t1**7 + t2**5 - np.real(t1 * t2**2)
    cf[4] = t1**6 - t2**4 + np.real(t1**3)
    cf[5] = -t1**5 + t2**3 - np.real(t2**3)
    cf[6] = t1**4 - t2**2 + np.real(t1**2 * t2)
    cf[7] = -t1**3 + t2 - np.real(t1 * t2)
    cf[8] = t1**2 - np.real(t2**2)
    cf[9] = -t1 + np.real(t1 * t2)
    for k in range(11, 61):
        cf[k - 1] = ((-1)**k * (np.real(t1) + np.imag(t2))**k) / (k + 1)
    for k in range(61, 71):
        cf[k - 1] = ((-1)**k * (np.real(t2) - np.imag(t1))**(k - 60)) / (k**2)
    cf[70] = np.real(t1 * t2)
    return cf.astype(np.complex128)
    
ALLOWED["p132"]=p132

def p133(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(1, 72):
        cf[k - 1] = (t1**k * np.sin(t2 * k) + t2**k * np.cos(t1 * k)) * (-1)**k / (k + 1)
    return cf.astype(np.complex128)
    
ALLOWED["p133"]=p133

def p134(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(1, 72):
        cf[k-1] = (np.sin(t1 * k) + np.cos(np.conj(t2) * k)) * (-1)**k / (k + 1)
    return cf.astype(np.complex128)

ALLOWED["p134"]=p134

def p135(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(1, 71):
        cf[k-1] = (t1**k * np.sin(k * t2) + (-1)**k * t2**(k-1) * np.cos(k * t1)) / k
    cf[70] = (np.log(np.abs(t1) + np.abs(t2) + 1) + np.sin(t1 * t2)) / 71
    return cf.astype(np.complex128)
    
ALLOWED["p135"]=p135

def p136(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1**5 + 2*t2**3 + 1
    cf[1] = t1*t2 - 3
    cf[2] = np.real(t1) * np.imag(t2) - np.real(t2) * np.imag(t1)
    cf[3] = np.sin(t1) + np.cos(t2)
    cf[4] = t1**2 - t2**2 + t1*t2
    for j in range(6, 36):
        cf[j-1] = ((t1 + (-1)**j * t2)**j) / j
    for k in range(36, 71):
        cf[k-1] = ((t1 - t2)**k) / (k**2) * (-1)**k
    cf[70] = np.log(np.abs(t1) + 1) + np.log(np.abs(t2) + 1) + t1*t2
    return cf.astype(np.complex128)
    
ALLOWED["p136"]=p136

def p137(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1**5 + 2*t2**3 + np.real(t1*t2)
    cf[1] = t1**4 - 3*t2**2 + np.imag(t1 + t2)
    cf[2] = 4*t1**3 + 2*t2**4 - np.sin(t1)
    cf[3] = 5*t1**2 - 2*t2**5 + np.cos(t2)
    cf[4] = 6*t1 - 3*t2**6 + np.real(t1**2 * t2)
    cf[5] = 7 + 4*t2**7 - np.imag(t1 * t2**2)
    cf[6] = 8*t1**2 + 5*t2**8 + np.real(t1**3)
    cf[7] = 9*t1**3 - 6*t2**9 + np.imag(t2**3)
    cf[8] = 10*t1**4 + 7*t2**10 + np.real(t1*t2**3)
    cf[9] = 11*t1**5 - 8*t2**11 + np.imag(t1**2 * t2**2)
    cf[10] = 12*t1**6 + 9*t2**12 + np.real(t2**4)
    cf[11] = 13*t1**7 - 10*t2**13 + np.imag(t1**3 * t2)
    cf[12] = 14*t1**8 + 11*t2**14 + np.real(t1**4)
    cf[13] = 15*t1**9 - 12*t2**15 + np.imag(t2**5)
    cf[14] = 16*t1**10 + 13*t2**16 + np.real(t1*t2**4)
    cf[15] = 17*t1**11 - 14*t2**17 + np.imag(t1**2 * t2**3)
    cf[16] = 18*t1**12 + 15*t2**18 + np.real(t2**6)
    cf[17] = 19*t1**13 - 16*t2**19 + np.imag(t1**3 * t2**2)
    cf[18] = 20*t1**14 + 17*t2**20 + np.real(t1**4 * t2)
    cf[19] = 21*t1**15 - 18*t2**21 + np.imag(t2**7)
    cf[20] = 22*t1**16 + 19*t2**22 + np.real(t1*t2**5)
    cf[21] = 23*t1**17 - 20*t2**23 + np.imag(t1**2 * t2**4)
    cf[22] = 24*t1**18 + 21*t2**24 + np.real(t2**8)
    cf[23] = 25*t1**19 - 22*t2**25 + np.imag(t1**3 * t2**3)
    cf[24] = 26*t1**20 + 23*t2**26 + np.real(t1**4 * t2**2)
    cf[25] = 27*t1**21 - 24*t2**27 + np.imag(t2**9)
    cf[26] = 28*t1**22 + 25*t2**28 + np.real(t1*t2**6)
    cf[27] = 29*t1**23 - 26*t2**29 + np.imag(t1**2 * t2**5)
    cf[28] = 30*t1**24 + 27*t2**30 + np.real(t2**10)
    cf[29] = 31*t1**25 - 28*t2**31 + np.imag(t1**3 * t2**4)
    cf[30] = 32*t1**26 + 29*t2**32 + np.real(t1**4 * t2**3)
    cf[31] = 33*t1**27 - 30*t2**33 + np.imag(t2**11)
    cf[32] = 34*t1**28 + 31*t2**34 + np.real(t1*t2**7)
    cf[33] = 35*t1**29 - 32*t2**35 + np.imag(t1**2 * t2**6)
    cf[34] = 36*t1**30 + 33*t2**36 + np.real(t2**12)
    cf[35] = 37*t1**31 - 34*t2**37 + np.imag(t1**3 * t2**5)
    cf[36] = 38*t1**32 + 35*t2**38 + np.real(t1**4 * t2**4)
    cf[37] = 39*t1**33 - 36*t2**39 + np.imag(t2**13)
    cf[38] = 40*t1**34 + 37*t2**40 + np.real(t1*t2**8)
    cf[39] = 41*t1**35 - 38*t2**41 + np.imag(t1**2 * t2**7)
    cf[40] = 42*t1**36 + 39*t2**42 + np.real(t2**14)
    cf[41] = 43*t1**37 - 40*t2**43 + np.imag(t1**3 * t2**6)
    cf[42] = 44*t1**38 + 41*t2**44 + np.real(t1**4 * t2**5)
    cf[43] = 45*t1**39 - 42*t2**45 + np.imag(t2**15)
    cf[44] = 46*t1**40 + 43*t2**46 + np.real(t1*t2**9)
    cf[45] = 47*t1**41 - 44*t2**47 + np.imag(t1**2 * t2**8)
    cf[46] = 48*t1**42 + 45*t2**48 + np.real(t2**16)
    cf[47] = 49*t1**43 - 46*t2**49 + np.imag(t1**3 * t2**7)
    cf[48] = 50*t1**44 + 47*t2**50 + np.real(t1**4 * t2**6)
    cf[49] = 51*t1**45 - 48*t2**51 + np.imag(t2**17)
    cf[50] = 52*t1**46 + 49*t2**52 + np.real(t1*t2**10)
    cf[51] = 53*t1**47 - 50*t2**53 + np.imag(t1**2 * t2**9)
    cf[52] = 54*t1**48 + 51*t2**54 + np.real(t2**18)
    cf[53] = 55*t1**49 - 52*t2**55 + np.imag(t1**3 * t2**8)
    cf[54] = 56*t1**50 + 53*t2**56 + np.real(t1**4 * t2**7)
    cf[55] = 57*t1**51 - 54*t2**57 + np.imag(t2**19)
    cf[56] = 58*t1**52 + 55*t2**58 + np.real(t1*t2**11)
    cf[57] = 59*t1**53 - 56*t2**59 + np.imag(t1**2 * t2**10)
    cf[58] = 60*t1**54 + 57*t2**60 + np.real(t2**20)
    cf[59] = 61*t1**55 - 58*t2**61 + np.imag(t1**3 * t2**9)
    cf[60] = 62*t1**56 + 59*t2**62 + np.real(t1**4 * t2**8)
    cf[61] = 63*t1**57 - 60*t2**63 + np.imag(t2**21)
    cf[62] = 64*t1**58 + 61*t2**64 + np.real(t1*t2**12)
    cf[63] = 65*t1**59 - 62*t2**65 + np.imag(t1**2 * t2**11)
    cf[64] = 66*t1**60 + 63*t2**66 + np.real(t2**22)
    cf[65] = 67*t1**61 - 64*t2**67 + np.imag(t1**3 * t2**10)
    cf[66] = 68*t1**62 + 65*t2**68 + np.real(t1**4 * t2**9)
    cf[67] = 69*t1**63 - 66*t2**69 + np.imag(t2**23)
    cf[68] = 70*t1**64 + 67*t2**70 + np.real(t1*t2**13)
    cf[69] = 71*t1**65 - 68*t2**71 + np.imag(t1**2 * t2**12)
    cf[70] = np.log(np.abs(t1) + np.abs(t2) + 1)
    return cf.astype(np.complex128)
    
ALLOWED["p137"]=p137

def p138(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1 + t2
    for k in range(2, 72):
        if k % 2 == 0:
            cf[k-1] = (t1**k - t2**k) * (-1)**k / np.log(k + np.abs(t1) + 1)
        else:
            cf[k-1] = (t1**(k//2) + t2**(k//3)) * (1 + np.sin(k * np.angle(t1 + t2)))
    return cf.astype(np.complex128)

ALLOWED["p138"]=p138

def p139(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1**6 + 2*t2**5
    cf[1] = -3*t1**5 + 4*t2**4
    cf[2] = 5*t1**4 - 6*t2**3
    cf[3] = -7*t1**3 + 8*t2**2
    cf[4] = 9*t1**2 - 10*t2
    cf[5] = -11*t1 + 12
    for k in range(7, 71):
        cf[k-1] = (-1)**k * (t1**(k-5) + t2**(k-3)) / (k-6)
    cf[70] = np.log(np.abs(t1) + 1) + np.log(np.abs(t2) + 1)
    return cf.astype(np.complex128)
    
ALLOWED["p139"]=p139

def p140(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1**5 - 2*t2 + 1
    cf[1] = 3*t1**4 + np.conj(t2)**2
    cf[2] = -4*t1**3 + 5*t1*t2
    cf[3] = 6*t2**3 - 7*t1**2*t2
    cf[4] = 8*t1*t2**2 + 9*t1
    cf[5] = -10*t2 + 11*t1**3
    cf[6] = 12*t1**2 - 13*t2**2
    cf[7] = 14*t1*t2 - 15
    cf[8] = -16*t1 + 17*t2**3
    cf[9] = 18*t1**4 - 19*t2
    cf[10] = 20*t1*t2**2 + 21*t1**2
    cf[11] = -22*t2**3 + 23*t1*t2
    cf[12] = 24*t1**3 - 25*t2**2
    cf[13] = 26*t1*t2 - 27
    cf[14] = -28*t1 + 29*t2**3
    cf[15] = 30*t1**4 - 31*t2
    cf[16] = 32*t1*t2**2 + 33*t1**2
    cf[17] = -34*t2**3 + 35*t1*t2
    cf[18] = 36*t1**3 - 37*t2**2
    cf[19] = 38*t1*t2 - 39
    cf[20] = -40*t1 + 41*t2**3
    cf[21] = 42*t1**4 - 43*t2
    cf[22] = 44*t1*t2**2 + 45*t1**2
    cf[23] = -46*t2**3 + 47*t1*t2
    cf[24] = 48*t1**3 - 49*t2**2
    cf[25] = 50*t1*t2 - 51
    cf[26] = -52*t1 + 53*t2**3
    cf[27] = 54*t1**4 - 55*t2
    cf[28] = 56*t1*t2**2 + 57*t1**2
    cf[29] = -58*t2**3 + 59*t1*t2
    cf[30] = 60*t1**3 - 61*t2**2
    cf[31] = 62*t1*t2 - 63
    cf[32] = -64*t1 + 65*t2**3
    cf[33] = 66*t1**4 - 67*t2
    cf[34] = 68*t1*t2**2 + 69*t1**2
    cf[35] = -70*t2**3 + 71*t1*t2
    cf[36] = 72*t1**3 - 73*t2**2
    cf[37] = 74*t1*t2 - 75
    cf[38] = -76*t1 + 77*t2**3
    cf[39] = 78*t1**4 - 79*t2
    cf[40] = 80*t1*t2**2 + 81*t1**2
    cf[41] = -82*t2**3 + 83*t1*t2
    cf[42] = 84*t1**3 - 85*t2**2
    cf[43] = 86*t1*t2 - 87
    cf[44] = -88*t1 + 89*t2**3
    cf[45] = 90*t1**4 - 91*t2
    cf[46] = 92*t1*t2**2 + 93*t1**2
    cf[47] = -94*t2**3 + 95*t1*t2
    cf[48] = 96*t1**3 - 97*t2**2
    cf[49] = 98*t1*t2 - 99
    cf[50] = -100*t1 + 101*t2**3
    cf[51] = 102*t1**4 - 103*t2
    cf[52] = 104*t1*t2**2 + 105*t1**2
    cf[53] = -106*t2**3 + 107*t1*t2
    cf[54] = 108*t1**3 - 109*t2**2
    cf[55] = 110*t1*t2 - 111
    cf[56] = -112*t1 + 113*t2**3
    cf[57] = 114*t1**4 - 115*t2
    cf[58] = 116*t1*t2**2 + 117*t1**2
    cf[59] = -118*t2**3 + 119*t1*t2
    cf[60] = 120*t1**3 - 121*t2**2
    cf[61] = 122*t1*t2 - 123
    cf[62] = -124*t1 + 125*t2**3
    cf[63] = 126*t1**4 - 127*t2
    cf[64] = 128*t1*t2**2 + 129*t1**2
    cf[65] = -130*t2**3 + 131*t1*t2
    cf[66] = 132*t1**3 - 133*t2**2
    cf[67] = 134*t1*t2 - 135
    cf[68] = -136*t1 + 137*t2**3
    cf[69] = 138*t1**4 - 139*t2
    cf[70] = 140*t1*t2**2 + 141*t1**2
    return cf.astype(np.complex128)
    
ALLOWED["p140"]=p140

def p141(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1**5 + 2*t2
    cf[1] = t1**4 - 3*t2**2 + np.conj(t1)
    cf[2] = 4*t1*t2 + 5*np.sin(t1)
    cf[3] = 6*np.cos(t2) - 7*t1**2
    cf[4] = 8*t2**3 + 9*t1*t2
    cf[5] = 10*np.sin(t1 + t2) - 11*t2
    cf[6] = 12*np.cos(t1) + 13*t1*t2
    cf[7] = 14*t1**3 - 15*t2**2
    cf[8] = 16*t1*t2**2 + 17*np.sin(t2)
    cf[9] = 18*np.cos(t1 + t2) - 19*t1
    cf[10] = 20*t2**4 + 21*t1**2*t2
    cf[11] = 22*np.sin(t1) - 23*t2**3
    cf[12] = 24*np.cos(t2) + 25*t1*t2
    cf[13] = 26*t1**4 - 27*t2**2*t1
    cf[14] = 28*np.sin(t1 + t2) + 29*t2
    cf[15] = 30*np.cos(t1) - 31*t1*t2**2
    cf[16] = 32*t2**5 + 33*t1**3
    cf[17] = 34*np.sin(t2) - 35*t1*t2
    cf[18] = 36*np.cos(t1 + t2) + 37*t2**3
    cf[19] = 38*t1**5 - 39*t2**2
    cf[20] = 40*np.sin(t1) + 41*t1**2*t2
    cf[21] = 42*np.cos(t2) - 43*t2**4
    cf[22] = 44*t1**3*t2 + 45*np.sin(t1 + t2)
    cf[23] = 46*np.cos(t1) - 47*t2**3*t1
    cf[24] = 48*t2**6 + 49*t1**4
    cf[25] = 50*np.sin(t2) + 51*t1*t2**2
    cf[26] = 52*np.cos(t1 + t2) - 53*t2**4
    cf[27] = 54*t1**6 - 55*t2**3
    cf[28] = 56*np.sin(t1) + 57*t1**3*t2
    cf[29] = 58*np.cos(t2) - 59*t2**5
    cf[30] = 60*t1**4*t2 + 61*np.sin(t1 + t2)
    cf[31] = 62*np.cos(t1) + 63*t2**4*t1
    cf[32] = 64*t2**7 + 65*t1**5
    cf[33] = 66*np.sin(t2) - 67*t1*t2**3
    cf[34] = 68*np.cos(t1 + t2) + 69*t2**5
    cf[35] = 70*t1**7 - 71*t2**4
    cf[36] = 72*np.sin(t1) + 73*t1**4*t2
    cf[37] = 74*np.cos(t2) - 75*t2**6
    cf[38] = 76*t1**5*t2 + 77*np.sin(t1 + t2)
    cf[39] = 78*np.cos(t1) - 79*t2**5*t1
    cf[40] = 80*t2**8 + 81*t1**6
    cf[41] = 82*np.sin(t2) + 83*t1*t2**4
    cf[42] = 84*np.cos(t1 + t2) - 85*t2**6
    cf[43] = 86*t1**8 - 87*t2**5
    cf[44] = 88*np.sin(t1) + 89*t1**5*t2
    cf[45] = 90*np.cos(t2) - 91*t2**7
    cf[46] = 92*t1**6*t2 + 93*np.sin(t1 + t2)
    cf[47] = 94*np.cos(t1) + 95*t2**6*t1
    cf[48] = 96*t2**9 + 97*t1**7
    cf[49] = 98*np.sin(t2) - 99*t1*t2**5
    cf[50] = 100*np.cos(t1 + t2) + 101*t2**7
    cf[51] = 102*t1**9 - 103*t2**6
    cf[52] = 104*np.sin(t1) + 105*t1**6*t2
    cf[53] = 106*np.cos(t2) - 107*t2**8
    cf[54] = 108*t1**7*t2 + 109*np.sin(t1 + t2)
    cf[55] = 110*np.cos(t1) + 111*t2**7*t1
    cf[56] = 112*t2**10 + 113*t1**8
    cf[57] = 114*np.sin(t2) - 115*t1*t2**6
    cf[58] = 116*np.cos(t1 + t2) + 117*t2**8
    cf[59] = 118*t1**10 - 119*t2**7
    cf[60] = 120*np.sin(t1) + 121*t1**7*t2
    cf[61] = 122*np.cos(t2) - 123*t2**9
    cf[62] = 124*t1**8*t2 + 125*np.sin(t1 + t2)
    cf[63] = 126*np.cos(t1) + 127*t2**8*t1
    cf[64] = 128*t2**11 + 129*t1**9
    cf[65] = 130*np.sin(t2) - 131*t1*t2**7
    cf[66] = 132*np.cos(t1 + t2) + 133*t2**9
    cf[67] = 134*t1**11 - 135*t2**8
    cf[68] = 136*np.sin(t1) + 137*t1**8*t2
    cf[69] = 138*np.cos(t2) - 139*t2**10
    cf[70] = 140*t1**9*t2 + 141*np.sin(t1 + t2) / 200
    return cf.astype(np.complex128)
    
ALLOWED["p141"]=p141

def p142(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1 + 2*t2
    cf[1] = t1**2 - t2
    cf[2] = np.conj(t1) + t2**3
    for k in range(4, 36):
        cf[k-1] = (t1**k + t2**(k-1)) / (k * np.sin(k))
        cf[72 - k - 1] = (t2**k - t1**(k-1)) / (k * np.cos(k))
    for k in range(36, 71):
        cf[k-1] = np.log(np.abs(t1) + 1) * np.sin(k * t2) + np.cos(k * t1)
    cf[70] = (t1**5 + t2**5) / (1 + np.abs(t1*t2))
    return cf.astype(np.complex128)
    
ALLOWED["p142"]=p142

def p143(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1**4 + 2*t2**3 - 3*t1*t2
    cf[1] = 4*t1**3 - 5*t2**2 + 6*np.real(t1)
    cf[2] = -7*t1**2*t2 + 8*np.imag(t2)
    cf[3] = 9*t1*t2**2 - 10*t1**3
    cf[4] = 11*np.abs(t1) + 12*np.angle(t2)
    cf[5] = -13*t1**4 + 14*t2 - 15*np.cos(t1)
    cf[6] = 16*np.sin(t2) + 17*t1*t2**2
    cf[7] = 18*t1**2 - 19*t2**3 + 20*t1
    cf[8] = -21*t1*t2 + 22*np.log(np.abs(t1) + 1) 
    cf[9] = 23*t2**2 - 24*t1**3 + 25*t1*t2
    cf[10] = 26*np.cos(t1*t2) + 27*np.sin(t2)
    cf[11] = -28*t1**2 + 29*t2 - 30*t1*t2**2
    cf[12] = 31*t1**4 - 32*t2**3 + 33*t1**2*t2
    cf[13] = 34*t1*t2 - 35*np.log(np.abs(t2) + 1)
    cf[14] = -36*t1**3 + 37*t2**2 - 38*t1*t2
    cf[15] = 39*np.sin(t1) + 40*np.cos(t2)
    cf[16] = 41*t1**2*t2 - 42*t2**3 + 43*t1
    cf[17] = -44*t1*t2 + 45*np.real(t2)
    cf[18] = 46*t2**2 - 47*t1**4 + 48*t1*t2
    cf[19] = 49*np.imag(t1) + 50*t2**3 - 51*t1*t2**2
    cf[20] = -52*t1**3 + 53*t2 - 54*np.cos(t1*t2)
    cf[21] = 55*t1*t2 + 56*np.log(np.abs(t1) + 1)
    cf[22] = 57*t1**2 - 58*t2**2 + 59*t1*t2
    cf[23] = -60*np.sin(t2) + 61*np.cos(t1)
    cf[24] = 62*t1**4 - 63*t2**3 + 64*t1**2*t2
    cf[25] = 65*t1*t2 - 66*np.real(t1)
    cf[26] = -67*t1**3 + 68*t2**2 - 69*t1*t2
    cf[27] = 70*np.sin(t1) + 71*np.cos(t2)
    cf[28] = 72*t1**2*t2 - 73*t2**3 + 74*t1
    cf[29] = -75*t1*t2 + 76*np.imag(t2)
    cf[30] = 77*t2**2 - 78*t1**4 + 79*t1*t2
    cf[31] = 80*np.real(t1) + 81*t2**3 - 82*t1*t2**2
    cf[32] = -83*t1**3 + 84*t2 - 85*np.sin(t1*t2)
    cf[33] = 86*t1*t2 + 87*np.log(np.abs(t2) + 1)
    cf[34] = 88*t1**2 - 89*t2**2 + 90*t1*t2
    cf[35] = -91*np.cos(t2) + 92*np.sin(t1)
    cf[36] = 93*t1**4 - 94*t2**3 + 95*t1**2*t2
    cf[37] = 96*t1*t2 - 97*np.real(t1)
    cf[38] = -98*t1**3 + 99*t2**2 - 100*t1*t2
    cf[39] = 101*np.sin(t1) + 102*np.cos(t2)
    cf[40] = 103*t1**2*t2 - 104*t2**3 + 105*t1
    cf[41] = -106*t1*t2 + 107*np.imag(t2)
    cf[42] = 108*t2**2 - 109*t1**4 + 110*t1*t2
    cf[43] = 111*np.real(t1) + 112*t2**3 - 113*t1*t2**2
    cf[44] = -114*t1**3 + 115*t2 - 116*np.sin(t1*t2)
    cf[45] = 117*t1*t2 + 118*np.log(np.abs(t1) + 1)
    cf[46] = 119*t1**2 - 120*t2**2 + 121*t1*t2
    cf[47] = -122*np.cos(t2) + 123*np.sin(t1)
    cf[48] = 124*t1**4 - 125*t2**3 + 126*t1**2*t2
    cf[49] = 127*t1*t2 - 128*np.real(t1)
    cf[50] = -129*t1**3 + 130*t2**2 - 131*t1*t2
    cf[51] = 132*np.sin(t1) + 133*np.cos(t2)
    cf[52] = 134*t1**2*t2 - 135*t2**3 + 136*t1
    cf[53] = -137*t1*t2 + 138*np.imag(t2)
    cf[54] = 139*t2**2 - 140*t1**4 + 141*t1*t2
    cf[55] = 142*np.real(t1) + 143*t2**3 - 144*t1*t2**2
    cf[56] = -145*t1**3 + 146*t2 - 147*np.sin(t1*t2)
    cf[57] = 148*t1*t2 + 149*np.log(np.abs(t2) + 1)
    cf[58] = 150*t1**2 - 151*t2**2 + 152*t1*t2
    cf[59] = -153*np.cos(t2) + 154*np.sin(t1)
    cf[60] = 155*t1**4 - 156*t2**3 + 157*t1**2*t2
    cf[61] = 158*t1*t2 - 159*np.real(t1)
    cf[62] = -160*t1**3 + 161*t2**2 - 162*t1*t2
    cf[63] = 163*np.sin(t1) + 164*np.cos(t2)
    cf[64] = 165*t1**2*t2 - 166*t2**3 + 167*t1
    cf[65] = -168*t1*t2 + 169*np.imag(t2)
    cf[66] = 170*t2**2 - 171*t1**4 + 172*t1*t2
    cf[67] = 173*np.real(t1) + 174*t2**3 - 175*t1*t2**2
    cf[68] = -176*t1**3 + 177*t2 - 178*np.sin(t1*t2)
    cf[69] = 179*t1*t2 + 180*np.log(np.abs(t1) + 1)
    cf[70] = 181*t1**2 - 182*t2**2 + 183*t1*t2
    return cf.astype(np.complex128)
    
ALLOWED["p143"]=p143

def p144(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1**5 + 2*t2**3 - t1*t2
    cf[1] = 3*t1**4 - t2**2 + 4*np.real(t1*t2)
    cf[2] = 5*t1**3 + 6*t2**4 - np.imag(t1**2)
    cf[3] = 7*t1**2*t2 - 8*np.abs(t2) + np.angle(t1)
    cf[4] = 9*t1*t2**2 + 10*np.cos(t1) - 11*np.sin(t2)
    cf[5] = 12*t1**3*t2 - 13*t2**3 + 14*np.log(np.abs(t1) + 1)
    cf[6] = 15*t1**2 - 16*t2 + 17*np.exp(t1*t2)
    cf[7] = 18*t1*t2**3 - 19*t1**4 + 20*(t1 + t2)
    cf[8] = 21*t1**5 - 22*t2**2 + 23*(t1 * t2)
    cf[9] = 24*t1**3*t2**2 - 25*t2**4 + 26*np.cos(t1*t2)
    cf[10] = 27*t1**2*t2**3 - 28*t1**4 + 29*np.sin(t2*t1)
    cf[11] = 30*t1*t2**4 - 31*t2**5 + 32*np.real(t1**2*t2)
    cf[12] = 33*t1**3*t2 - 34*t2**3 + 35*np.angle(t1 + t2)
    cf[13] = 36*t1**4 - 37*t2**2*t1 + 38*np.log(np.abs(t2) + 1)
    cf[14] = 39*t1*t2**3 - 40*t1**5 + 41*np.cos(t1 + t2)
    cf[15] = 42*t1**2*t2**2 - 43*t2**4 + 44*np.sin(t1 - t2)
    cf[16] = 45*t1**3*t2 - 46*t2**3*t1 + 47*np.abs(t1*t2)
    cf[17] = 48*t1**4*t2 - 49*t2**5 + 50*np.real(t1 + t2)
    cf[18] = 51*t1*t2**4 - 52*t1**5 + 53*np.angle(t1*t2)
    cf[19] = 54*t1**2*t2**3 - 55*t2**4*t1 + 56*np.log(np.abs(t1*t2) + 1)
    cf[20] = 57*t1**3*t2**2 - 58*t2**5 + 59*np.cos(t1*t2)
    cf[21] = 60*t1**4*t2 - 61*t1**5*t2 + 62*np.sin(t1 + t2)
    cf[22] = 63*t1*t2**5 - 64*t2**6 + 65*np.real(t1**2 + t2**2)
    cf[23] = 66*t1**5*t2 - 67*t2**6 + 68*np.angle(t1**2*t2)
    cf[24] = 69*t1**6 - 70*t2**3 + 71*np.log(np.abs(t1) + np.abs(t2) + 1)
    cf[25] = 72*t1**3*t2**3 - 73*t2**6 + 74*np.cos(t1**2 + t2**2)
    cf[26] = 75*t1**4*t2**2 - 76*t1**6 + 77*np.sin(t1*t2)
    cf[27] = 78*t1**5*t2 - 79*t2**7 + 80*np.real(t1*t2**2)
    cf[28] = 81*t1**6*t2 - 82*t2**7 + 83*np.angle(t1**3*t2)
    cf[29] = 84*t1**7 - 85*t2**4 + 86*np.log(np.abs(t1**2) + np.abs(t2)**2 + 1)
    cf[30] = 87*t1**4*t2**3 - 88*t2**7 + 89*np.cos(t1**3 + t2**3)
    cf[31] = 90*t1**5*t2**2 - 91*t1**7 + 92*np.sin(t1**2*t2)
    cf[32] = 93*t1**6*t2 - 94*t2**8 + 95*np.real(t1**2*t2**2)
    cf[33] = 96*t1**7*t2 - 97*t2**8 + 98*np.angle(t1**4*t2)
    cf[34] = 99*t1**8 - 100*t2**5 + 101*np.log(np.abs(t1**3) + np.abs(t2)**3 + 1)
    cf[35] = 102*t1**5*t2**3 - 103*t2**9 + 104*np.cos(t1**4 + t2**4)
    cf[36] = 105*t1**6*t2**2 - 106*t1**8 + 107*np.sin(t1**3*t2)
    cf[37] = 108*t1**7*t2 - 109*t2**9 + 110*np.real(t1**3*t2**2)
    cf[38] = 111*t1**8*t2 - 112*t2**10 + 113*np.angle(t1**5*t2)
    cf[39] = 114*t1**9 - 115*t2**6 + 116*np.log(np.abs(t1**4) + np.abs(t2)**4 + 1)
    cf[40] = 117*t1**6*t2**3 - 118*t2**10 + 119*np.cos(t1**5 + t2**5)
    cf[41] = 120*t1**7*t2**2 - 121*t1**9 + 122*np.sin(t1**4*t2)
    cf[43] = 123*t1**8*t2 - 124*t2**11 + 125*np.real(t1**4*t2**2)
    cf[44] = 126*t1**9*t2 - 127*t2**11 + 128*np.angle(t1**6*t2)
    cf[45] = 129*t1**10 - 130*t2**7 + 131*np.log(np.abs(t1**5) + np.abs(t2)**5 + 1)
    cf[46] = 132*t1**7*t2**3 - 133*t2**12 + 134*np.cos(t1**6 + t2**6)
    cf[47] = 135*t1**8*t2**2 - 136*t1**10 + 137*np.sin(t1**5*t2)
    cf[48] = 138*t1**9*t2 - 139*t2**12 + 140*np.real(t1**5*t2**2)
    cf[49] = 141*t1**10*t2 - 142*t2**13 + 143*np.angle(t1**7*t2)
    cf[50] = 144*t1**11 - 145*t2**8 + 146*np.log(np.abs(t1**6) + np.abs(t2)**6 + 1)
    cf[51] = 147*t1**8*t2**3 - 148*t2**13 + 149*np.cos(t1**7 + t2**7)
    cf[52] = 150*t1**9*t2**2 - 151*t1**11 + 152*np.sin(t1**6*t2)
    cf[53] = 153*t1**10*t2 - 154*t2**14 + 155*np.real(t1**6*t2**2)
    cf[54] = 156*t1**11*t2 - 157*t2**14 + 158*np.angle(t1**8*t2)
    cf[55] = 159*t1**12 - 160*t2**9 + 161*np.log(np.abs(t1**7) + np.abs(t2)**7 + 1)
    cf[56] = 162*t1**9*t2**3 - 163*t2**15 + 164*np.cos(t1**8 + t2**8)
    cf[57] = 165*t1**10*t2**2 - 166*t1**12 + 167*np.sin(t1**7*t2)
    cf[58] = 168*t1**11*t2 - 169*t2**15 + 170*np.real(t1**7*t2**2)
    cf[59] = 171*t1**12*t2 - 172*t2**16 + 173*np.angle(t1**9*t2)
    cf[60] = 174*t1**13 - 175*t2**10 + 176*np.log(np.abs(t1**8) + np.abs(t2)**8 + 1)
    cf[61] = 177*t1**10*t2**3 - 178*t2**16 + 179*np.cos(t1**9 + t2**9)
    cf[62] = 180*t1**11*t2**2 - 181*t1**13 + 182*np.sin(t1**8*t2)
    cf[63] = 183*t1**12*t2 - 184*t2**17 + 185*np.real(t1**8*t2**2)
    cf[64] = 186*t1**13*t2 - 187*t2**17 + 188*np.angle(t1**10*t2)
    cf[65] = 189*t1**14 - 190*t2**11 + 191*np.log(np.abs(t1**9) + np.abs(t2)**9 + 1)
    cf[66] = 192*t1**11*t2**3 - 193*t2**18 + 194*np.cos(t1**10 + t2**10)
    cf[67] = 195*t1**12*t2**2 - 196*t1**14 + 197*np.sin(t1**9*t2)
    cf[68] = 198*t1**13*t2 - 199*t2**18 + 200*np.real(t1**9*t2**2)
    cf[69] = 201*t1**14*t2 - 202*t2**19 + 203*np.angle(t1**11*t2)
    cf[70] = 204*t1**15 - 205*t2**12 + 206*np.log(np.abs(t1**10) + np.abs(t2)**10 + 1)
    return cf.astype(np.complex128)
    
ALLOWED["p144"]=p144

def p145(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(1, 72):
        cf[k-1] = ((t1**k + np.conj(t2)**k) * (-1)**k) / (k + np.real(t1) + np.real(t2))
    return cf.astype(np.complex128)

ALLOWED["p145"]=p145

def p146(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0:10] = (t1**np.arange(1, 11) + t2**np.arange(1, 11)) * (-1)**np.arange(1, 11)
    cf[10:20] = np.sin(t1 * np.arange(11, 21)) - np.cos(t2 * np.arange(11, 21))
    cf[20:30] = np.log(np.abs(t1) + 1) * np.arange(21, 31) - np.log(np.abs(t2) + 1)
    cf[30:40] = (t1 * t2)**np.arange(31, 41) / (1 + np.arange(31, 41))
    cf[40:71] = np.real(t1) * np.imag(t2) - np.imag(t1) * np.real(t2) + np.angle(t1 + t2) * np.abs(t1 - t2) * np.arange(40, 71)
    cf[70] = np.sum(cf[0:70])
    return cf.astype(np.complex128)
    
ALLOWED["p146"]=p146

def p147(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1**5 + t2**4
    cf[1] = t1**4 - t2**3
    cf[2] = t1**3 * t2 + t1**2
    cf[3] = t1**2 - t2**5
    cf[4] = t2**4 - t1**3
    cf[5] = np.real(t1) * np.imag(t2) + np.sin(t1 * t2)
    cf[6] = np.cos(t2) + t1 * t2**2
    cf[7] = np.log(np.abs(t1) + 1) - np.log(np.abs(t2) + 1)
    cf[8] = t1**6 - t2**6
    cf[9] = t1 * t2**3 - t2 * t1**4
    for k in range(11, 71):
        cf[k-1] = (t1**(71 - k) + t2**(71 - k)) * (-1)**k / (k**2)
    cf[70] = np.real(t1)**2 + np.imag(t2)**2
    return cf.astype(np.complex128)
    
ALLOWED["p147"]=p147

def p148(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1**4 - 3*t2 + np.real(t1*t2)
    cf[1] = np.imag(t1)**2 + np.abs(t2)**3
    cf[2] = np.sin(t1) + np.cos(t2)
    cf[3] = np.angle(t1) * np.angle(t2)
    cf[4] = np.log(np.abs(t1 + t2) + 1)
    for j in range(6, 71):
        cf[j-1] = (t1**j + t2**(71 - j)) / (j - 5) + (-1)**j * np.abs(t1 - t2)
    cf[70] = t1 * t2 / (1 + np.abs(t1)**2 + np.abs(t2)**2)
    return cf.astype(np.complex128)

ALLOWED["p148"]=p148

def p149(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1**4 + 2*t2**3
    cf[1] = -t1**3 + 3*t2**2
    cf[2] = t1**2 - 4*t2
    cf[3] = -t1 + 5*t2**2
    cf[4] = t2**3 - 6*t1**2*t2
    for j in range(6, 36):
        cf[j-1] = (np.real(t1)**j - np.imag(t2)**j) * (-1)**j / j
    for j in range(36, 71):
        cf[j-1] = (np.sin(j * t1) + np.cos(j * t2)) / (j + 1)
    cf[70] = (np.real(t1) + np.imag(t2)) * (t1 - t2)
    return cf.astype(np.complex128)
    
ALLOWED["p149"]=p149

def p150(z,a,state):
    t1, t2 = z[0], z[1]
    j = np.arange(0, 71)
    cf = (t1 + j * t2) * (-1)**j * np.log(np.abs(t1) + np.abs(t2) + 1)**(np.abs(j) % 5 + 1) * (j + 1)
    return cf.astype(np.complex128)

ALLOWED["p150"]=p150

def p151(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = (t1 + t2)**8 + (t1 - t2)**7
    cf[1] = np.sin(t1) * np.cos(t2)**6
    cf[2] = np.log(np.abs(t1 * t2) + 1) * (t1**2 - t2**2)
    cf[3] = np.real(t1) * np.imag(t2) * (t1 + t2)
    cf[4] = np.abs(t1 - t2) * np.angle(t1 + np.conj(t2))
    cf[5] = (t1**3 + t2**3) / (1 + np.abs(t1 + t2))
    cf[6] = (t1 - t2)**4 * (np.sin(t1) - np.cos(t2))
    cf[7] = (np.real(t1)**2 + np.imag(t2)**2) * np.log(np.abs(t1 - t2) + 1)
    cf[8] = (t1 * t2)**5 - (t1 + t2)**5
    cf[9] = np.sin(t1 * t2) + np.cos(t1 + t2)
    for j in range(11, 71):
        cf[j-1] = ((t1**j + t2**j) / (j + 1)) * (-1)**j
    cf[70] = 1
    return cf.astype(np.complex128)
    
ALLOWED["p151"]=p151

def p152(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    k = np.arange(1, 72)
    cf = (t1**k + np.conj(t2)**k) * (-1)**k / (1 + k)
    return cf.astype(np.complex128).astype(np.complex128)

ALLOWED["p152"]=p152

def p153(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(1, 72):
        if k <= 35:
            cf[k-1] = ((np.real(t1)**k + np.imag(t2)**k) * (-1)**k) / (k)
        else:
            cf[k-1] = (np.sin(t1 * k) + np.cos(t2 * k)) * (-1)**k / (71 - k + 1)
    cf[70] = np.log(np.abs(t1) + 1) + np.log(np.abs(t2) + 1)
    return cf.astype(np.complex128)

ALLOWED["p153"]=p153

def p154(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = np.real(t1)**10 + np.real(t2)**10
    cf[1] = np.imag(t1)**9 - np.imag(t2)**9
    for j in range(3, 36):
        cf[j-1] = (-1)**j * (np.real(t1)**j + np.real(t2)**(j-1)) / j
    for j in range(36, 71):
        cf[j-1] = (np.log(np.abs(t1) + 1) * j) - (np.log(np.abs(t2) + 1) / j)
    cf[70] = np.real(t1 * t2) + np.imag(t1 + t2)
    return cf.astype(np.complex128)
    
ALLOWED["p154"]=p154

def p155(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = (t1 + t2)**5 + np.real(t1) * np.imag(t2)
    cf[1] = (t1**2 - t2**2) * np.sin(t1 * t2)
    cf[2] = np.log(np.abs(t1) + 1) + np.cos(t2)
    cf[3] = np.real(t1)**2 - np.imag(t2)**2 + t1 * t2
    cf[4] = (t1 - t2)**3 + np.real(t2) * np.imag(t1)
    cf[5] = np.sin(t1 + t2) * np.cos(t1 - t2)
    cf[6] = np.real(t1 * t2) + np.imag(t1 + t2)
    cf[7] = (t1**3 + t2**3) / (np.abs(t1) + np.abs(t2) + 1)
    cf[8] = np.real(t1)**3 - np.imag(t2)**3 + t1**2 * t2
    cf[9] = np.cos(t1 * t2) + np.sin(t1 + t2)
    cf[10] = (t1 + t2)**4 - np.real(t1**2 - t2**2)
    cf[11] = np.real(t1)**2 * np.imag(t2) + np.real(t2)**2 * np.imag(t1)
    cf[12] = np.log(np.abs(t1) + np.abs(t2) + 1) * np.sin(t1 - t2)
    cf[13] = np.real(t1**2 + t2**2) - np.imag(t1 * t2)
    cf[14] = (t1 - t2)**4 + np.cos(t1**2 + t2**2)
    cf[15] = np.real(t1)**4 - np.imag(t2)**4 + t1**3 * t2
    cf[16] = np.sin(t1**2 - t2**2) * np.cos(t1 + t2)
    cf[17] = (np.real(t1) + np.imag(t2))**2 + (np.real(t2) - np.imag(t1))**2
    cf[18] = np.log(np.abs(t1 * t2) + 1) * (t1 + t2)
    cf[19] = np.real(t1**3 + t2**3) - np.imag(t1**3 - t2**3)
    cf[20] = np.cos(t1**3 - t2**3) + np.sin(t1**2 + t2**2)
    cf[21] = (t1 + t2)**5 - np.real(t1**4 - t2**4)
    cf[22] = np.real(t1**4) * np.imag(t2**2) + np.real(t2**4) * np.imag(t1**2)
    cf[23] = np.log(np.abs(t1**2) + np.abs(t2**2) + 1) * np.sin(t1**2 - t2**2)
    cf[24] = np.real(t1**4 + t2**4) - np.imag(t1**4 - t2**4)
    cf[25] = (t1 - t2)**5 + np.cos(t1**4 + t2**4)
    cf[26] = np.real(t1**5) - np.imag(t2**5) + t1**4 * t2
    cf[27] = np.sin(t1**3 - t2**3) * np.cos(t1**2 + t2**2)
    cf[28] = (np.real(t1**2) + np.imag(t2**2))**2 + (np.real(t2**2) - np.imag(t1**2))**2
    cf[29] = np.log(np.abs(t1**2 * t2**2) + 1) * (t1**2 + t2**2)
    cf[30] = np.real(t1**5 + t2**5) - np.imag(t1**5 - t2**5)
    cf[31] = np.cos(t1**5 - t2**5) + np.sin(t1**4 + t2**4)
    cf[32] = (t1 + t2)**6 - np.real(t1**6 - t2**6)
    cf[33] = np.real(t1**6) * np.imag(t2**3) + np.real(t2**6) * np.imag(t1**3)
    cf[34] = np.log(np.abs(t1**3) + np.abs(t2**3) + 1) * np.sin(t1**3 - t2**3)
    cf[35] = np.real(t1**6 + t2**6) - np.imag(t1**6 - t2**6)
    cf[36] = (t1 - t2)**6 + np.cos(t1**6 + t2**6)
    cf[37] = np.real(t1**7) - np.imag(t2**7) + t1**6 * t2
    cf[38] = np.sin(t1**4 - t2**4) * np.cos(t1**3 + t2**3)
    cf[39] = (np.real(t1**3) + np.imag(t2**3))**2 + (np.real(t2**3) - np.imag(t1**3))**2
    cf[40] = np.log(np.abs(t1**3 * t2**3) + 1) * (t1**3 + t2**3)
    cf[41] = np.real(t1**7 + t2**7) - np.imag(t1**7 - t2**7)
    cf[42] = np.cos(t1**7 - t2**7) + np.sin(t1**6 + t2**6)
    cf[43] = (t1 + t2)**7 - np.real(t1**8 - t2**8)
    cf[44] = np.real(t1**8) * np.imag(t2**4) + np.real(t2**8) * np.imag(t1**4)
    cf[45] = np.log(np.abs(t1**4) + np.abs(t2**4) + 1) * np.sin(t1**4 - t2**4)
    cf[46] = np.real(t1**8 + t2**8) - np.imag(t1**8 - t2**8)
    cf[47] = (t1 - t2)**7 + np.cos(t1**8 + t2**8)
    cf[48] = np.real(t1**9) - np.imag(t2**9) + t1**8 * t2
    cf[49] = np.sin(t1**5 - t2**5) * np.cos(t1**4 + t2**4)
    cf[50] = (np.real(t1**4) + np.imag(t2**4))**2 + (np.real(t2**4) - np.imag(t1**4))**2
    cf[51] = np.log(np.abs(t1**4 * t2**4) + 1) * (t1**4 + t2**4)
    cf[52] = np.real(t1**9 + t2**9) - np.imag(t1**9 - t2**9)
    cf[53] = np.cos(t1**9 - t2**9) + np.sin(t1**8 + t2**8)
    cf[54] = (t1 + t2)**8 - np.real(t1**10 - t2**10)
    cf[55] = np.real(t1**10) * np.imag(t2**5) + np.real(t2**10) * np.imag(t1**5)
    cf[56] = np.log(np.abs(t1**5) + np.abs(t2**5) + 1) * np.sin(t1**5 - t2**5)
    cf[57] = np.real(t1**10 + t2**10) - np.imag(t1**10 - t2**10)
    cf[58] = (t1 - t2)**8 + np.cos(t1**10 + t2**10)
    cf[59] = np.real(t1**11) - np.imag(t2**11) + t1**10 * t2
    cf[60] = np.sin(t1**6 - t2**6) * np.cos(t1**5 + t2**5)
    cf[61] = (np.real(t1**5) + np.imag(t2**5))**2 + (np.real(t2**5) - np.imag(t1**5))**2
    cf[62] = np.log(np.abs(t1**5 * t2**5) + 1) * (t1**5 + t2**5)
    cf[63] = np.real(t1**11 + t2**11) - np.imag(t1**11 - t2**11)
    cf[64] = np.cos(t1**11 - t2**11) + np.sin(t1**10 + t2**10)
    cf[65] = (t1 + t2)**9 - np.real(t1**12 - t2**12)
    cf[66] = np.real(t1**12) * np.imag(t2**6) + np.real(t2**12) * np.imag(t1**6)
    cf[67] = np.log(np.abs(t1**6) + np.abs(t2**6) + 1) * np.sin(t1**6 - t2**6)
    cf[68] = np.real(t1**12 + t2**12) - np.imag(t1**12 - t2**12)
    cf[69] = (t1 - t2)**9 + np.cos(t1**12 + t2**12)
    cf[70] = np.real(t1**13) - np.imag(t2**13) + t1**12 * t2
    return cf.astype(np.complex128)

ALLOWED["p155"]=p155

def p156(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1 + 2 * t2
    cf[1] = np.real(t1) * t2 - np.imag(t2)
    cf[2] = np.sin(t1) + np.cos(t2)
    cf[3] = np.log(np.abs(t1) + 1) + np.angle(t2)
    cf[4] = t1**2 - t2**3
    cf[5] = np.real(t1)**2 + np.imag(t2)**2
    cf[6] = np.sin(t1 * t2) - np.cos(t1)
    cf[7] = np.angle(t1)**3 + np.abs(t2)
    cf[8] = t1 * t2 + np.real(t1 + t2)
    cf[9] = np.real(t1)**3 - np.imag(t2)**3
    cf[10] = np.log(np.abs(t1) * np.abs(t2) + 1)
    cf[11] = t1**4 + t2**4
    cf[12] = np.sin(t1) * np.cos(t2)
    cf[13] = np.real(t1 * t2) - np.imag(t1 / t2)
    cf[14] = np.angle(t1 + t2)**2
    cf[15] = np.abs(t1 + t2)**3
    cf[16] = t1**5 - t2**2
    cf[17] = np.sin(np.real(t1)) + np.cos(np.imag(t2))
    cf[18] = np.log(np.abs(t1) + np.abs(t2))
    cf[19] = np.real(t1**2) * np.imag(t2**2)
    cf[20] = t1 * t2**3 - t1**3 * t2
    cf[21] = np.sin(t1 + t2) + np.cos(t1 - t2)
    cf[22] = np.real(t1)**4 - np.imag(t2)**4
    cf[23] = np.angle(t1)**2 + np.angle(t2)**2
    cf[24] = np.abs(t1)**5 - np.abs(t2)**5
    cf[25] = np.sin(t1**2) + np.cos(t2**2)
    cf[26] = np.log(np.abs(t1 * t2) + 1)
    cf[27] = np.real(t1 + t2)**3
    cf[28] = np.imag(t1 * t2)**2
    cf[29] = t1**6 + t2**6
    cf[30] = np.sin(t1 * t2) - np.cos(t1**2)
    cf[31] = np.real(t1**3) * np.imag(t2**3)
    cf[32] = np.log(np.abs(t1)**2 + np.abs(t2)**2)
    cf[33] = t1**7 - t2**4
    cf[34] = np.sin(np.real(t1 * t2)) + np.cos(np.imag(t1) + np.imag(t2))
    cf[35] = np.real(t1**4) - np.imag(t2**5)
    cf[36] = np.angle(t1 + t2)**3
    cf[37] = np.abs(t1 + t2)**4
    cf[38] = t1**8 - t2**3
    cf[39] = np.sin(t1**3) + np.cos(t2**3)
    cf[40] = np.log(np.abs(t1 * t2**2) + 1)
    cf[41] = np.real(t1**5) * np.imag(t2**4)
    cf[42] = t1**9 + t2**5
    cf[43] = np.sin(t1 * t2**2) - np.cos(t1**4)
    cf[44] = np.real(t1**6) - np.imag(t2**6)
    cf[45] = np.angle(t1 + t2)**4
    cf[46] = np.abs(t1 + t2)**5
    cf[47] = t1**10 - t2**0.5
    cf[48] = np.sin(t1**5) + np.cos(t2**5)
    cf[49] = np.log(np.abs(t1**2 * t2) + 1)
    cf[50] = np.real(t1**7) * np.imag(t2**5)
    cf[51] = t1**11 + t2**6
    cf[52] = np.sin(t1 * t2**3) - np.cos(t1**5)
    cf[53] = np.real(t1**8) - np.imag(t2**7)
    cf[54] = np.angle(t1 + t2)**5
    cf[55] = np.abs(t1 + t2)**6
    cf[56] = t1**12 - t2**0.5
    cf[57] = np.sin(t1**5) + np.cos(t2**5)
    cf[58] = np.log(np.abs(t1**3 * t2) + 1)
    cf[59] = np.real(t1**9) * np.imag(t2**6)
    cf[60] = t1**13 + t2**7
    cf[61] = np.sin(t1 * t2**4) - np.cos(t1**6)
    cf[62] = np.real(t1**10) - np.imag(t2**8)
    cf[63] = np.angle(t1 + t2)**6
    cf[64] = np.abs(t1 + t2)**7
    cf[65] = t1**14 - t2**0.5
    cf[66] = np.sin(t1**6) + np.cos(t2**6)
    cf[67] = np.log(np.abs(t1**4 * t2) + 1)
    cf[68] = np.real(t1**11) * np.imag(t2**7)
    cf[69] = t1**15 + t2**8
    cf[70] = np.sin(t1 * t2**5) - np.cos(t1**7)
    return cf.astype(np.complex128)
    
ALLOWED["p156"]=p156

def p157(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    k = np.arange(71)
    cf = ((np.real(t1) + np.imag(t2))**(70 - k) + (np.real(t1) - np.imag(t2))**k) * (-1)**k / (k + 1)
    return cf.astype(np.complex128).astype(np.complex128)
    
ALLOWED["p157"]=p157

#EMPTY
def p158(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(1, 72):
        cf[k-1] = (np.real(t1) + np.imag(t2) * k)**(1 + k / 10) * (-1)**k + np.sin(k * np.angle(t1 * t2)) + np.cos(k * np.abs(t1 + t2))
    return cf.astype(np.complex128)
    
ALLOWED["p158"]=p158

def p159(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1**5 + 2 * t2**4
    cf[1] = t1**4 - t2**3 + np.real(t1 * t2)
    cf[2] = np.sin(t1) + np.cos(t2) + t1 * t2
    cf[3] = np.log(np.abs(t1) + 1) - np.log(np.abs(t2) + 1)
    cf[4] = t1**3 - 2 * t2**2 + np.real(t1)**2
    for j in range(6, 36):
        cf[j-1] = (t1**(j-1) + (-1)**j * t2**(j-2)) / j
    for j in range(36, 71):
        cf[j-1] = np.real(t1)**j + np.imag(t2)**(j/2) * (-1)**j
    cf[70] = (t1 + t2)**2 / (1 + t1 * t2)
    return cf.astype(np.complex128)
    
ALLOWED["p159"]=p159

def p160(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1**3 - 2 * t2 + np.sin(t1)
    cf[1] = np.conj(t1) * t2 - np.cos(t2)
    cf[2] = t1**2 + t2**2 + np.real(t1 * t2)
    cf[3] = t1 - t2 + np.log(np.abs(t1) + 1)
    cf[4] = t1 * t2 - np.abs(t1) + np.sin(t2)
    cf[5] = np.real(t1)**2 - np.imag(t2)**2 + np.cos(t1)
    cf[6] = t1**4 - t2**3 + np.real(t1 + t2)
    cf[7] = np.imag(t1 * t2) + np.log(np.abs(t2) + 1)
    cf[8] = t1**5 - t1 * t2 + np.sin(t1 * t2)
    cf[9] = np.conj(t1)**2 + t2**4
    cf[10] = t1 * t2**2 - np.real(t2)
    cf[11] = t1**6 + np.imag(t1) - np.cos(t2)
    cf[12] = t1**3 * t2 - np.log(np.abs(t1) - np.abs(t2) + 1)
    cf[13] = t1 + t2**5 + np.sin(t1 + t2)
    cf[14] = np.conj(t1)**3 - t2**2
    cf[15] = np.real(t1 * t2**3) + np.cos(t1 * t2)
    cf[16] = t1**2 - t2**4 + np.sin(t1**2)
    cf[17] = t1 * t2**4 - np.log(np.abs(t1 * t2) + 1)
    cf[18] = t1**7 + np.real(t2**3)
    cf[19] = np.imag(t1**2 * t2) - np.cos(t2**2)
    cf[20] = t1**3 - t2**5 + np.sin(t1 - t2)
    cf[21] = np.conj(t1) * t2**2 + np.log(np.abs(t2) + 1)
    cf[22] = t1**4 + t2**6 - np.real(t1 * t2)
    cf[23] = t1 * t2**5 + np.imag(t1 + t2)
    cf[24] = t1**5 - t2**7 + np.sin(t1 * t2)
    cf[25] = np.real(t1**3 * t2**2) - np.cos(t1)
    cf[26] = t1**6 + t2**8 + np.log(np.abs(t1) + np.abs(t2))
    cf[27] = t1 * t2**6 - np.real(t2**3)
    cf[28] = t1**7 - t2**9 + np.sin(t1**2)
    cf[29] = np.conj(t1**2) * t2**3 + np.cos(t2)
    cf[30] = t1**8 + t2**10 - np.real(t1 * t2**2)
    cf[31] = t1 * t2**7 + np.imag(t1**2)
    cf[32] = t1**9 - t2**11 + np.log(np.abs(t1**2) + 1)
    cf[33] = np.real(t1**4 * t2**3) - np.sin(t2)
    cf[34] = t1**10 + t2**12 + np.cos(t1 * t2)
    cf[35] = t1 * t2**8 - np.real(t2**4)
    cf[36] = t1**11 - t2**13 + np.sin(t1**3)
    cf[37] = np.conj(t1**3) * t2**4 + np.log(np.abs(t2**2) + 1)
    cf[38] = t1**12 + t2**14 - np.real(t1 * t2**3)
    cf[39] = t1 * t2**9 + np.imag(t1**3)
    cf[40] = t1**13 - t2**15 + np.sin(t1 * t2**2)
    cf[41] = np.real(t1**5 * t2**4) - np.cos(t2**2)
    cf[42] = t1**14 + t2**16 + np.log(np.abs(t1**3) + 1)
    cf[43] = t1 * t2**10 - np.real(t2**5)
    cf[44] = t1**15 - t2**17 + np.sin(t1**4)
    cf[45] = np.conj(t1**4) * t2**5 + np.cos(t2**3)
    cf[46] = t1**16 + t2**18 - np.real(t1 * t2**4)
    cf[47] = t1 * t2**11 + np.imag(t1**4)
    cf[48] = t1**17 - t2**19 + np.log(np.abs(t1**4) + 1)
    cf[49] = np.real(t1**6 * t2**5) - np.sin(t2**3)
    cf[50] = t1**18 + t2**20 + np.cos(t1 * t2**2)
    cf[51] = t1 * t2**12 - np.real(t2**6)
    cf[52] = t1**19 - t2**21 + np.sin(t1**5)
    cf[53] = np.conj(t1**5) * t2**6 + np.log(np.abs(t2**3) + 1)
    cf[54] = t1**20 + t2**22 - np.real(t1 * t2**5)
    cf[55] = t1 * t2**13 + np.imag(t1**5)
    cf[56] = t1**21 - t2**23 + np.sin(t1 * t2**3)
    cf[57] = np.real(t1**7 * t2**6) - np.cos(t2**4)
    cf[58] = t1**22 + t2**24 + np.log(np.abs(t1**5) + 1)
    cf[59] = t1 * t2**14 - np.real(t2**7)
    cf[60] = t1**23 - t2**25 + np.sin(t1**6)
    cf[61] = np.conj(t1**6) * t2**7 + np.cos(t2**5)
    cf[62] = t1**24 + t2**26 - np.real(t1 * t2**6)
    cf[63] = t1 * t2**15 + np.imag(t1**6)
    cf[64] = t1**25 - t2**27 + np.log(np.abs(t1**6) + 1)
    cf[65] = np.real(t1**8 * t2**7) - np.sin(t2**5)
    cf[66] = t1**26 + t2**28 + np.cos(t1 * t2**3)
    cf[67] = t1 * t2**16 - np.real(t2**8)
    cf[68] = t1**27 - t2**29 + np.sin(t1**7)
    cf[69] = np.conj(t1**7) * t2**8 + np.log(np.abs(t2**4) + 1)
    cf[70] = t1**28 + t2**30 - np.real(t1 * t2**7)
    return cf.astype(np.complex128)
    
ALLOWED["p160"]=p160

def p161(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    k = np.arange(1, 71)
    cf[:70] = (t1**((k % 4) + 1) * t2**((k % 3) + 1)) + (-1)**k * np.log(np.abs(t1) + 1) * np.sin(k * t2)
    cf[70] = t1 * t2 / (1 + t1**2 + t2**2)
    return cf.astype(np.complex128)
    
ALLOWED["p161"]=p161

def p162(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1**10 + 2 * t2**9
    cf[1] = 3 * t1**8 - 4 * t2**7
    cf[2] = 5 * t1**6 + 6 * t2**5
    cf[3] = 7 * t1**4 - 8 * t2**3
    cf[4] = 9 * t1**2 + 10 * t2
    cf[5] = 11 * np.conj(t1) - 12 * np.conj(t2)
    cf[6] = 13 * np.sin(t1) + 14 * np.cos(t2)
    cf[7] = 15 * np.log(np.abs(t1) + 1) - 16 * np.log(np.abs(t2) + 1)
    cf[8] = 17 * t1 * t2 + 18 * t1**3
    cf[9] = 19 * t2**2 - 20 * t1**4
    cf[10] = 21 * t1**5 + 22 * t2**5
    cf[11] = 23 * t1**6 - 24 * t2**6
    cf[12] = 25 * t1**7 + 26 * t2**7
    cf[13] = 27 * t1**8 - 28 * t2**8
    cf[14] = 29 * t1**9 + 30 * t2**9
    cf[15] = 31 * t1**10 - 32 * t2**10
    for j in range(17, 71):
        cf[j-1] = (33 + j) * t1**(j % 5) * t2**((j + 1) % 5)
    cf[70] = 71 * np.sin(t1 + t2) + 72 * np.cos(t1 - t2)
    return cf.astype(np.complex128)
    
ALLOWED["p162"]=p162

def p163(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1**4 + 2 * t2**3
    cf[1] = -t1**3 + 3 * t1 * t2**2
    cf[2] = 4 * t1**2 - 5 * t2
    cf[3] = -6 * t1 + 7 * t2**2
    cf[4] = 8 - 9 * t1 * t2
    cf[5] = (t1 + np.conj(t2))**2
    cf[6] = -10 * t1**3 + 11 * t2**4
    cf[7] = 12 * t1**2 * t2 - 13 * t2**3
    cf[8] = -14 * t1 * t2**2 + 15
    cf[9] = 16 * t1**4 - 17 * t2
    for j in range(11, 36):
        cf[j-1] = ((-1)**j * (t1 + j * t2)) / (j + 1)
    for j in range(36, 71):
        cf[j-1] = ((j % 2) * t1**2 - (j % 3) * t2**2) / (j + 2)
    cf[70] = t1**3 - t2**3 + 18
    return cf.astype(np.complex128)

ALLOWED["p163"]=p163

def p164(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(1, 72):
        if k % 4 == 1:
            cf[k-1] = t1**k + t2**(k-1) * np.sin(t1)
        elif k % 4 == 2:
            cf[k-1] = (t1 + t2)**k * np.cos(t2)
        elif k % 4 == 3:
            cf[k-1] = np.log(np.abs(t1) + 1) * t2**k
        else:
            cf[k-1] = (t1 - t2)**k * np.sin(t1 * t2)
    cf[0] = t1 + 2 * t2
    cf[70] = t1**35 - t2**35 + 1j * t1 * t2
    return cf.astype(np.complex128)
    
ALLOWED["p164"]=p164

def p165(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1**6 + t2**5
    cf[1] = t1**5 - t2**4
    cf[2] = t1**4 + t2**3
    cf[3] = -t1**3 + t2**2
    cf[4] = t1**2 - t2
    cf[5] = -t1 + 1
    for j in range(7, 72):
        cf[j-1] = (t1 * t2) / j * (-1)**j
    return cf.astype(np.complex128)
    
ALLOWED["p165"]=p165

def p166(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(1, 72):
        cf[k-1] = t1**k + (-1)**k * t2**k / (k + 1) + np.sin(k * t1) + np.cos(k * t2)
    return cf.astype(np.complex128)
    
ALLOWED["p166"]=p166

def p167(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(0, 71):
        j = k + 1
        cf[j-1] = (t1 + t2)**(70 - k) * (t1 - t2)**k * (-1)**k / (k + 1)
    return cf.astype(np.complex128)

ALLOWED["p167"]=p167

def p168(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1 + t2
    cf[1] = t1**2 - t2**2
    cf[2] = np.sin(t1) * np.cos(t2)
    cf[3] = np.log(np.abs(t1) + 1) - np.log(np.abs(t2) + 1)
    cf[4] = t1 * t2
    for k in range(6, 72):
        cf[k-1] = ((t1**(k-5) + t2**(k-5)) * (-1)**k) / (k**0.5)
    return cf.astype(np.complex128)

ALLOWED["p168"]=p168

def p169(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for j in range(1, 72):
        cf[j-1] = (t1**(j % 5 + 1) * np.conj(t2)**(j % 3 + 1)) * (-1)**j / (j + 1)
    return cf.astype(np.complex128)

ALLOWED["p169"]=p169

def p170(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for j in range(0, 71):
        cf[j] = (t1**j + np.conj(t2)**j) * (-1)**j / (j + 2)
    return cf.astype(np.complex128)

ALLOWED["p170"]=p170

def p171(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    k = np.arange(71)
    cf = (t1**k + t2**(70 - k)) * (-1)**k * (71 - k)
    return cf.astype(np.complex128).astype(np.complex128)

ALLOWED["p171"]=p171

def p172(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1**5 + 2 * t2
    cf[1] = np.conj(t1) - 3 * np.conj(t2)
    cf[2] = t1 * t2 + 4 * t1**2
    cf[3] = 5 * np.sin(t1) - 6 * np.cos(t2)
    cf[4] = 7 * t1**3 + 8 * t2**3
    cf[5] = 9 * t1 * t2**2 - 10 * t1**2 * t2
    cf[6] = 11 * np.log(np.abs(t1) + 1) + 12 * np.log(np.abs(t2) + 1)
    cf[7] = 13 * t1**4 - 14 * t2**4
    cf[8] = 15 * np.sin(t1 * t2) + 16 * np.cos(t1 + t2)
    cf[9] = 17 * t1**2 * t2 + 18 * t1 * t2**2
    cf[10] = 19 * np.sin(t1)**2 - 20 * np.cos(t2)**2
    cf[11] = 21 * t1**5 + 22 * t2**5
    cf[12] = 23 * np.conj(t1 * t2) - 24 * t1 * t2
    cf[13] = 25 * t1**3 * t2**2 + 26 * t1**2 * t2**3
    cf[14] = 27 * np.sin(t1**2) - 28 * np.cos(t2**2)
    cf[15] = 29 * t1**6 + 30 * t2**6
    cf[16] = 31 * t1 * t2**4 - 32 * t1**4 * t2
    cf[17] = 33 * np.log(np.abs(t1**2) + 1) + 34 * np.log(np.abs(t2**2) + 1)
    cf[18] = 35 * t1**7 - 36 * t2**7
    cf[19] = 37 * np.sin(t1 * t2**2) + 38 * np.cos(t1**2 + t2)
    cf[20] = 39 * t1**3 * t2**3 - 40 * t1**3 * t2**3
    cf[21] = 41 * np.sin(t1)**3 - 42 * np.cos(t2)**3
    cf[22] = 43 * t1**8 + 44 * t2**8
    cf[23] = 45 * t1 * t2**5 + 46 * t1**5 * t2
    cf[24] = 47 * np.log(np.abs(t1**3) + 1) - 48 * np.log(np.abs(t2**3) + 1)
    cf[25] = 49 * t1**9 - 50 * t2**9
    cf[26] = 51 * np.sin(t1 * t2**3) - 52 * np.cos(t1**3 + t2**2)
    cf[27] = 53 * t1**4 * t2**4 + 54 * t1**4 * t2**4
    cf[28] = 55 * np.sin(t1)**4 - 56 * np.cos(t2)**4
    cf[29] = 57 * t1**10 + 58 * t2**10
    cf[30] = 59 * t1 * t2**6 - 60 * t1**6 * t2
    cf[31] = 61 * np.log(np.abs(t1**4) + 1) + 62 * np.log(np.abs(t2**4) + 1)
    cf[32] = 63 * t1**11 - 64 * t2**11
    cf[33] = 65 * np.sin(t1 * t2**4) + 66 * np.cos(t1**4 + t2**3)
    cf[34] = 67 * t1**5 * t2**5 - 68 * t1**5 * t2**5
    cf[35] = 69 * np.sin(t1)**5 - 70 * np.cos(t2)**5
    cf[36] = 71 * t1**12 + 72 * t2**12
    cf[37] = 73 * t1 * t2**7 + 74 * t1**7 * t2
    cf[38] = 75 * np.log(np.abs(t1**5) + 1) - 76 * np.log(np.abs(t2**5) + 1)
    cf[39] = 77 * t1**13 - 78 * t2**13
    cf[40] = 79 * np.sin(t1 * t2**5) - 80 * np.cos(t1**5 + t2**4)
    cf[41] = 81 * t1**6 * t2**6 + 82 * t1**6 * t2**6
    cf[42] = 83 * np.sin(t1)**6 - 84 * np.cos(t2)**6
    cf[43] = 85 * t1**14 + 86 * t2**14
    cf[44] = 87 * t1 * t2**8 - 88 * t1**8 * t2
    cf[45] = 89 * np.log(np.abs(t1**6) + 1) + 90 * np.log(np.abs(t2**6) + 1)
    cf[46] = 91 * t1**15 - 92 * t2**15
    cf[47] = 93 * np.sin(t1 * t2**6) + 94 * np.cos(t1**6 + t2**5)
    cf[48] = 95 * t1**7 * t2**7 - 96 * t1**7 * t2**7
    cf[49] = 97 * np.sin(t1)**7 - 98 * np.cos(t2)**7
    cf[50] = 99 * t1**16 + 100 * t2**16
    cf[51] = 101 * t1 * t2**9 + 102 * t1**9 * t2
    cf[52] = 103 * np.log(np.abs(t1**7) + 1) - 104 * np.log(np.abs(t2**7) + 1)
    cf[53] = 105 * t1**17 - 106 * t2**17
    cf[54] = 107 * np.sin(t1 * t2**7) - 108 * np.cos(t1**7 + t2**6)
    cf[55] = 109 * t1**8 * t2**8 + 110 * t1**8 * t2**8
    cf[56] = 111 * np.sin(t1)**8 - 112 * np.cos(t2)**8
    cf[57] = 113 * t1**18 + 114 * t2**18
    cf[58] = 115 * t1 * t2**10 - 116 * t1**10 * t2
    cf[59] = 117 * np.log(np.abs(t1**8) + 1) + 118 * np.log(np.abs(t2**8) + 1)
    cf[60] = 119 * t1**19 - 120 * t2**19
    cf[61] = 121 * np.sin(t1 * t2**8) + 122 * np.cos(t1**8 + t2**7)
    cf[62] = 123 * t1**9 * t2**9 - 124 * t1**9 * t2**9
    cf[63] = 125 * np.sin(t1)**9 - 126 * np.cos(t2)**9
    cf[64] = 127 * t1**20 + 128 * t2**20
    cf[65] = 129 * t1 * t2**11 + 130 * t1**11 * t2
    cf[66] = 131 * np.log(np.abs(t1**9) + 1) - 132 * np.log(np.abs(t2**9) + 1)
    cf[67] = 133 * t1**21 - 134 * t2**21
    cf[68] = 135 * np.sin(t1 * t2**9) - 136 * np.cos(t1**9 + t2**8)
    cf[69] = 137 * t1**10 * t2**10 + 138 * t1**10 * t2**10
    cf[70] = 139 * np.sin(t1)**10 - 140 * np.cos(t2)**10
    return cf.astype(np.complex128)
    
ALLOWED["p172"]=p172

def p173(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(1, 72):
        cf[k-1] = (t1**(k % 5 + 1) + np.conj(t2)**(k % 7 + 1)) * (-1)**k * np.log(np.abs(t1) + np.abs(t2) + 1)
    return cf.astype(np.complex128)

ALLOWED["p173"]=p173

def p174(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(1, 72):
        cf[k-1] = (t1**k + np.conj(t2)**k) * (-1)**k / (1 + k**1.2)
    return cf.astype(np.complex128)

ALLOWED["p174"]=p174

def p175(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(1, 72):
        cf[k-1] = ((np.real(t1)**k + np.imag(t2)**(k % 5 + 1)) * (-1)**k) / (1 + k)
    return cf.astype(np.complex128)
    
ALLOWED["p175"]=p175

def p176(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for j in range(1, 72):
        cf[j-1] = ((t1**j + np.conj(t2)**(71 - j)) * (-1)**j) / np.log(j + np.abs(t1) + np.abs(t2) + 1)
    return cf.astype(np.complex128)

ALLOWED["p176"]=p176

def p177(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1**5 - t2**3 + 2 * t1 * t2
    cf[1] = -t1**4 + 3 * t2**2 - t1 * t2**2
    cf[2] = t1**3 - 4 * t2 + 2 * t1**2 * t2
    cf[3] = -t1**2 + 5 * t2**4 - 3 * t1 * t2**3
    cf[4] = t1 - 6 * t2**5 + 4 * t1**2 * t2**4
    for k in range(6, 71):
        cf[k-1] = ((t1**k) * (-1)**k + t2**(k-1)) / (k + 1)
    cf[70] = np.log(np.abs(t1) + 1) + t1 * t2**2
    return cf.astype(np.complex128)

ALLOWED["p177"]=p177

def p178(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for j in range(0, 71):
        cf[j] = ((-1)**j * (t1**j + t2**j)) / ((j + 1)**2)
    return cf.astype(np.complex128)
    
ALLOWED["p178"]=p178

def p179(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = (t1 + t2) * (np.real(t1) - np.imag(t2))
    for j in range(2, 71):
        if j % 2 == 0:
            cf[j-1] = (np.abs(t1)**j - np.abs(t2)**j) / (j + 1) * (-1)**j
        else:
            cf[j-1] = (np.real(t1)**j + np.imag(t2)**j) / (j + 2) * np.sin(j * np.angle(t1 + t2))
    cf70a = (
        (t1**1).real * (t2**1).imag +
        (t1**2).real * (t2**2).imag +
        (t1**3).real * (t2**3).imag +
        (t1**4).real * (t2**4).imag +
        (t1**5).real * (t2**5).imag
    )    
    cf70b =  np.log(np.abs(t1) + np.abs(t2) + 1)   
    cf[70] =  cf70b + cf70a
    return cf.astype(np.complex128)

ALLOWED["p179"]=p179

def p180(z,a,state):
    t1, t2 = z[0], z[1]
    idx = np.arange(71, dtype=np.int64)
    e0 = (idx % 6) + 1
    e1 = (idx % 4) + 1
    cf0 = np.power(t1, e0)                 # works for complex t1
    cf1 = np.power(t2, e1)                 # works for complex t2
    cf2 = (1 - 2 * (idx & 1)).astype(np.float64)  # +1,-1,+1,-1,... without (-1)**idx
    cf3 = np.log(idx + 1.0)                        # ensure float input to log
    cf = ((cf0 + cf1) * cf2 * cf3).astype(np.complex128)
    return cf
    
ALLOWED["p180"]=p180

def p181(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(1, 72):
        cf[k-1] = (-1)**k * (np.abs(t1)**k + np.abs(t2)**k) + (np.sin(k * np.angle(t1)) - np.cos(k * np.angle(t2))) / k
    return cf.astype(np.complex128)
    
ALLOWED["p181"]=p181

def p182(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for j in range(71):
        cf[j] = (t1**(j + 1) * np.conj(t2)**(70 - j)) * (-1)**(j // 5) + np.log(np.abs(t1 + (j + 1) * t2) + 1)
    cf[70] = np.sin(t1) + np.cos(t2) + np.real(t1 * t2)
    return cf.astype(np.complex128)
    
ALLOWED["p182"]=p182

def p183(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1**6 + 2 * t2**5
    cf[1] = 3 * t1**5 - t2**4
    cf[2] = 4 * t1**4 + 2 * t2**3
    cf[3] = 5 * t1**3 - 3 * t2**2
    cf[4] = 6 * t1**2 + 4 * t2
    cf[5] = 7 * t1 - 5
    for j in range(6, 36):
        cf[j] = ((-1)**j) * (t1 + 1)**j + (t2 - 1)**j
    for j in range(36, 71):
        cf[j] = (j - 35) * np.sin(t1 * j) - (j - 34) * np.cos(t2 * j)
    cf[70] = t1 * t1 + t2 * t2 + t1 * t2
    return cf

ALLOWED["p183"]=p183

def p184(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1**5 + 2 * t2**4
    cf[1] = -3 * t1**4 + 4 * t2**3
    cf[2] = 5 * t1**3 - 6 * t2**2
    cf[3] = -7 * t1**2 + 8 * t2
    cf[4] = 9 * t1 - 10 * t2**0
    cf[5:10] = (t1 * t2)**np.arange(1, 6) * np.array([11, -12, 13, -14, 15])
    cf[10:20] = (t1 + t2)**(np.arange(6, 16) / 2) * np.array([-16, 17, -18, 19, -20, 21, -22, 23, -24, 25])
    cf[20:30] = (t1 - t2)**(np.arange(16, 26) / 3) * np.array([26, -27, 28, -29, 30, 31, -32, 33, -34, 35])
    cf[30:40] = (t1 * t2)**(np.arange(26, 36) / 4) * np.array([36, -37, 38, -39, 40, 41, -42, 43, -44, 45])
    cf[40:50] = (t1 + np.conj(t2))**(np.arange(36, 46) / 5) * np.array([-46, 47, -48, 49, -50, 51, -52, 53, -54, 55])
    cf[50:60] = (np.conj(t1) - t2)**(np.arange(46, 56) / 6) * np.array([56, -57, 58, -59, 60, 61, -62, 63, -64, 65])
    cf[60:70] = (np.abs(t1) + np.abs(t2))**(np.arange(56, 66) / 7) * np.array([-66, 67, -68, 69, -70, 71, -72, 73, -74, 75])
    cf[70] = np.log(np.abs(t1) + 1) + np.log(np.abs(t2) + 1)
    return cf

ALLOWED["p184"]=p184

def p185(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = t1**5 + 2 * t2**4
    cf[1] = -3 * t1 * t2 + 4 * np.log(np.abs(t1) + 1)
    cf[2] = 5 * t1**2 - 6 * t2**3
    cf[3] = 7 * np.sin(t1) - 8 * np.cos(t2)
    cf[4] = 9 * t1**3 + 10 * t2**4
    cf[5] = -11 * t1 * t2**2 + 12 * t2 * t1**2
    cf[6] = 13 * t1**2 * t2 - 14 * t2**2 * t1
    cf[7] = 15 * np.sin(t1 * t2) + 16 * np.cos(t1 + t2)
    cf[8] = -17 * t1**4 + 18 * t2**5
    cf[9] = 19 * t1 * t2**3 - 20 * t2 * t1**3
    cf[10] = 21 * t1**3 * t2**2
    cf[11] = -22 * np.sin(t1**2) + 23 * np.cos(t2**2)
    cf[12] = 24 * t1**5 - 25 * t2**6
    cf[13] = 26 * t1**2 * t2**4 - 27 * t2**2 * t1**4
    cf[14] = 28 * np.sin(t1 * t2**2) - 29 * np.cos(t1**2 * t2)
    cf[15] = 30 * t1**6 + 31 * t2**7
    cf[16] = -32 * t1**3 * t2**3 + 33 * t2**3 * t1**3
    cf[17] = 34 * t1 * t2**5 - 35 * t2 * t1**5
    cf[18] = 36 * np.sin(t1**3) + 37 * np.cos(t2**3)
    cf[19] = 38 * t1**7 - 39 * t2**8
    cf[20] = 40 * t1**2 * t2**6 - 41 * t2**2 * t1**6
    cf[21] = 42 * np.sin(t1 * t2**3) - 43 * np.cos(t1**3 * t2)
    cf[22] = 44 * t1**8 + 45 * t2**9
    cf[23] = -46 * t1**4 * t2**4 + 47 * t2**4 * t1**4
    cf[24] = 48 * t1 * t2**7 - 49 * t2 * t1**7
    cf[25] = 50 * np.sin(t1**4) + 51 * np.cos(t2**4)
    cf[26] = 52 * t1**9 - 53 * t2**10
    cf[27] = 54 * t1**2 * t2**8 - 55 * t2**2 * t1**8
    cf[28] = 56 * np.sin(t1 * t2**4) - 57 * np.cos(t1**4 * t2)
    cf[29] = 58 * t1**10 + 59 * t2**11
    cf[30] = -60 * t1**5 * t2**5 + 61 * t2**5 * t1**5
    cf[31] = 62 * t1 * t2**9 - 63 * t2 * t1**9
    cf[32] = 64 * np.sin(t1**5) + 65 * np.cos(t2**5)
    cf[33] = 66 * t1**11 - 67 * t2**12
    cf[34] = 68 * t1**2 * t2**10 - 69 * t2**2 * t1**10
    cf[35] = 70 * np.sin(t1 * t2**5) - 71 * np.cos(t1**5 * t2)
    cf[36] = 72 * t1**12 + 73 * t2**13
    cf[37] = -74 * t1**6 * t2**6 + 75 * t2**6 * t1**6
    cf[38] = 76 * t1 * t2**11 - 77 * t2 * t1**11
    cf[39] = 78 * np.sin(t1**6) + 79 * np.cos(t2**6)
    cf[40] = 80 * t1**13 - 81 * t2**14
    cf[41] = 82 * t1**2 * t2**12 - 83 * t2**2 * t1**12
    cf[42] = 84 * np.sin(t1 * t2**6) - 85 * np.cos(t1**6 * t2)
    cf[43] = 86 * t1**14 + 87 * t2**15
    cf[44] = -88 * t1**7 * t2**7 + 89 * t2**7 * t1**7
    cf[45] = 90 * t1 * t2**13 - 91 * t2 * t1**13
    cf[46] = 92 * np.sin(t1**7) + 93 * np.cos(t2**7)
    cf[47] = 94 * t1**15 - 95 * t2**16
    cf[48] = 96 * t1**2 * t2**14 - 97 * t2**2 * t1**14
    cf[49] = 98 * np.sin(t1 * t2**7) - 99 * np.cos(t1**7 * t2)
    cf[50] = 100 * t1**16 + 101 * t2**17
    cf[51] = -102 * t1**8 * t2**8 + 103 * t2**8 * t1**8
    cf[52] = 104 * t1 * t2**15 - 105 * t2 * t1**15
    cf[53] = 106 * np.sin(t1**8) + 107 * np.cos(t2**8)
    cf[54] = 108 * t1**17 - 109 * t2**18
    cf[55] = 110 * t1**2 * t2**16 - 111 * t2**2 * t1**16
    cf[56] = 112 * np.sin(t1 * t2**8) - 113 * np.cos(t1**8 * t2)
    cf[57] = 114 * t1**18 + 115 * t2**19
    cf[58] = -116 * t1**9 * t2**9 + 117 * t2**9 * t1**9
    cf[59] = 118 * t1 * t2**17 - 119 * t2 * t1**17
    cf[60] = 120 * np.sin(t1**9) + 121 * np.cos(t2**9)
    cf[61] = 122 * t1**19 - 123 * t2**20
    cf[62] = 124 * t1**2 * t2**18 - 125 * t2**2 * t1**18
    cf[63] = 126 * np.sin(t1 * t2**9) - 127 * np.cos(t1**9 * t2)
    cf[64] = 128 * t1**20 + 129 * t2**21
    cf[65] = -130 * t1**10 * t2**10 + 131 * t2**10 * t1**10
    cf[66] = 132 * t1 * t2**19 - 133 * t2 * t1**19
    cf[67] = 134 * np.sin(t1**10) + 135 * np.cos(t2**10)
    cf[68] = 136 * t1**21 - 137 * t2**22
    cf[69] = 138 * t1**2 * t2**20 - 139 * t2**2 * t1**20
    cf[70] = 140 * np.sin(t1 * t2**10) + 141 * np.cos(t1**10 * t2)
    return cf.astype(np.complex128)
    
ALLOWED["p185"]=p185

def p186(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for j in range(71):
        cf[j] = ((-1)**j) * (np.real(t1)**j + np.imag(t2)**j) / (np.log(np.abs(t1 + t2)) + j) * np.sin(j * np.angle(t1 * t2 + 1))
    return cf.astype(np.complex128)

ALLOWED["p186"]=p186

def p187(z,a,state):
    t1, t2 = z[0], z[1]
    k = np.arange(71)
    cf = t1**k * np.sin(t2 * k) + t2**k * np.cos(t1 * k)
    return cf.astype(np.complex128)
    
ALLOWED["p187"]=p187

def p188(z,a,state):
    t1, t2 = z[0], z[1]
    k = np.arange(71)
    cf = (np.real(t1)**k + np.imag(t2) * k) * (-1)**k + np.log(np.abs(t1 + t2 * k) + 1) + np.sin(t1 * k) * np.cos(t2 * k) + (np.angle(t1) * k - np.angle(t2)) * 1j
    return cf.astype(np.complex128)

ALLOWED["p188"]=p188

def p189(z,a,state):
    t1, t2 = z[0], z[1]
    degrees = np.arange(71)
    cf = t1**(70 - degrees) * (np.cos(degrees) + 1j * np.sin(degrees)) + t2**degrees * (np.cos(degrees) - 1j * np.sin(degrees))
    return cf.astype(np.complex128)
    
ALLOWED["p189"]=p189

def p190(z,a,state):
    t1, t2 = z[0], z[1]
    exponents = np.arange(71)
    cf = t1**exponents + (-1)**exponents * t2**(exponents + 1)
    return cf.astype(np.complex128)
    
ALLOWED["p190"]=p190

def p191(z,a,state):
    t1, t2 = z[0], z[1]
    exponents = np.arange(71)
    cf = (t1**exponents) * np.sin(exponents * np.angle(t2)) + (np.conj(t2)**exponents) * np.cos(exponents * np.real(t1))
    return cf.astype(np.complex128).astype(np.complex128)

ALLOWED["p191"]=p191

def p192(z,a,state):
    t1, t2 = z[0], z[1]
    degrees = np.arange(71)
    cf = (t1**degrees) * (np.conj(t2)**(degrees % 7)) * (-1)**(degrees // 6) * (1 + degrees / 70)
    return cf.astype(np.complex128).astype(np.complex128)
    
ALLOWED["p192"]=p192

def p193(z,a,state):
    t1, t2 = z[0], z[1]
    j = np.arange(71)
    cf = (t1**j * t2**(70 - j)) * ((-1)**j + np.real(t1) * np.imag(t2) / (j + 1))
    return cf.astype(np.complex128).astype(np.complex128)
    
ALLOWED["p193"]=p193

def p194(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    exponents = np.arange(71)
    cf = (t1**exponents) * np.real(t2) + (np.conj(t1)**exponents) * np.imag(t2) - np.log(np.abs(t1) + 1)**exponents + np.sin(t1 * exponents) - np.cos(t2 * exponents)
    return cf.astype(np.complex128).astype(np.complex128)
    
ALLOWED["p194"]=p194

def p195(z,a,state):
    t1, t2 = z[0], z[1]
    degrees = np.arange(71)
    cf = np.zeros(71, dtype=np.complex128)
    cf = (t1**degrees) * np.sin(t2 * degrees) + (t2**degrees) * np.cos(t1 * degrees)
    return cf.astype(np.complex128).astype(np.complex128)

ALLOWED["p195"]=p195

def p196(z,a,state):
    t1, t2 = z[0], z[1]
    exponents1 = np.arange(1, 72)
    exponents2 = np.arange(71, 0, -1)
    terms1 = t1**exponents1
    terms2 = (-1)**np.arange(71) * t2**exponents2
    terms3 = np.sin(t1 * np.arange(71)) * np.cos(t2 * np.arange(1, 72))
    cf = terms1 + terms2 + terms3
    return cf.astype(np.complex128)

ALLOWED["p196"]=p196

def p197(z, a, state):
    t1, t2 = z[0], z[1]

    n = 71
    j = np.arange(n)  # int64

    # angles (avoid np.angle for njit robustness)
    ang2 = math.atan2(t2.imag, t2.real)
    ang1 = math.atan2(t1.imag, t1.real)  # used below if needed

    re1 = t1.real
    re2 = t2.real
    im1 = t1.imag
    im2 = t2.imag

    # integer power series via cumulative multiply (njit-safe)
    re1_pow = np.empty(n, dtype=np.float64)
    im1_pow = np.empty(n, dtype=np.float64)
    re2_pow = np.empty(n, dtype=np.float64)
    re1_pow[0] = 1.0
    im1_pow[0] = 1.0
    re2_pow[0] = 1.0
    for k in range(1, n):
        re1_pow[k] = re1_pow[k - 1] * re1
        im1_pow[k] = im1_pow[k - 1] * im1
        re2_pow[k] = re2_pow[k - 1] * re2

    # fractional powers: promote base to complex to allow non-integer exponents safely
    im2_c = np.complex128(im2)
    im2_pow_half = np.empty(n, dtype=np.complex128)
    for k in range(n):
        # complex ** float exponent is supported; equivalent to exp((k/2)*log(im2_c))
        im2_pow_half[k] = im2_c ** (0.5 * k)

    # trig vectors
    s1 = np.empty(n, dtype=np.float64)
    c1 = np.empty(n, dtype=np.float64)
    s2 = np.empty(n, dtype=np.float64)
    c2 = np.empty(n, dtype=np.float64)
    for k in range(n):
        jk = float(k)
        jk2 = float(70 - k)
        s1[k] = math.sin(jk * ang2)
        c1[k] = math.cos(jk * ang2)
        s2[k] = math.sin(jk2 * ang2)
        c2[k] = math.cos(jk2 * ang2)

    # build cf (as complex to be safe)
    cf = np.empty(n, dtype=np.complex128)
    for k in range(n):
        term = (
            re1_pow[k] * s1[k] +
            re2_pow[70 - k] * c2[k] +
            im1_pow[k] * c1[k] -
            im2_pow_half[k] * s2[k]
        )
        cf[k] = np.complex128(term)

    # scalar multiplier (use math.* on scalars)
    m1 = math.log(math.hypot(t1.real, t1.imag) + 1.0)
    m2 = math.log(math.hypot(t2.real, t2.imag) + 1.0)
    cf *= (m1 * m2)

    return cf

ALLOWED["p197"]=p197

def p198(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    k = np.arange(71)
    cf = (np.real(t1) + np.imag(t2))**(70 - k) * (np.abs(t1) + np.abs(t2))**k * np.sin(k * np.angle(t1) - np.angle(t2)) + \
            (np.real(t2) - np.imag(t1))**k * np.cos(k * np.angle(t2) + np.angle(t1)) + np.log(np.abs(t1) + 1) * np.log(np.abs(t2) + 1) / (k + 1)
    return cf.astype(np.complex128)

ALLOWED["p198"]=p198

def p199(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    j = np.arange(71)
    cf = (np.real(t1)**j * np.sin(j * np.real(t2))) + (np.imag(t2)**j * np.cos(j * np.imag(t1))) / (j + 1)
    return cf.astype(np.complex128)
    
ALLOWED["p199"]=p199

def p200(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(71):
        cf[k] = (np.real(t1)**(70 - k) * np.sin(k * np.angle(t1)) + np.imag(t2)**k * np.cos(k * np.angle(t2))) / (1 + k)
    return cf.astype(np.complex128)
    
ALLOWED["p200"]=p200

def p201(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    j = np.arange(71)
    cf = (np.real(t1)**j * np.sin(j * np.angle(t2)) + np.real(t2)**j * np.cos(j * np.angle(t1))) + \
            (np.imag(t1)**j * np.cos(j * np.angle(t2)) - np.imag(t2)**j * np.sin(j * np.angle(t1))) * 1j
    return cf.astype(np.complex128)
    
ALLOWED["p201"]=p201

def p202(z, a, state):
    t1, t2 = z[0], z[1]
    k = np.arange(71)
    cf = (t1**k * np.sin(k * np.angle(t1) + np.real(t2)) + t2**k * np.cos(k * np.angle(t2) - np.real(t1))) * np.log(np.abs(t1 * t2) + 1) / (1 + k**2) + \
            (np.sin(k * np.real(t1)) - np.cos(k * np.imag(t2))) * (np.abs(t1) + np.abs(t2)) / (2 + k)
    return cf.astype(np.complex128)
    
ALLOWED["p202"]=p202

def p203(z, a, state):
    t1, t2 = z[0], z[1]
    k = np.arange(1, 72)
    real_part = np.real(t1)**k * np.sin(k * np.angle(t2)) + np.real(t2)**k * np.cos(k * np.angle(t1)) + np.log(np.abs(t1) + k) + np.real(t1 + t2)**k / (k + 1)
    imag_part = np.imag(t1)**k * np.cos(k * np.angle(t2)) + np.imag(t2)**k * np.sin(k * np.angle(t1)) + np.sin(k) + np.cos(k)
    cf = real_part + 1j * imag_part
    cf = cf * ((-1)**k * np.log(k + np.abs(t1)))
    return cf.astype(np.complex128)

ALLOWED["p203"]=p203

def p204(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    j = np.arange(1, 72)
    cf = np.real(t1)**j + np.real(t2)**(71 - j) * np.cos(j * np.angle(t1) + (71 - j) * np.angle(t2)) + np.log(np.abs(t1) + 1) * np.sin(j * np.angle(t2))
    cf += np.imag(t1)**j - np.imag(t2)**(71 - j) * np.sin(j * np.angle(t1) - (71 - j) * np.angle(t2)) + np.log(np.abs(t2) + 1) * np.cos(j * np.angle(t1))
    return cf.astype(np.complex128)
    
ALLOWED["p204"]=p204

def p205(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for j in range(71):
        cf[j] = (np.real(t1) * (j + 1)**2 + np.imag(t2) * (j + 1)) * np.sin((j + 1) * np.angle(t1 + t2)) + \
                    (np.cos((j + 1) * np.angle(t1)) + np.log(np.abs(t1 * t2) + 1)) * (np.cos(j + 1) + np.sin(j + 1) * (0 + 1j))
    return cf.astype(np.complex128)
    
ALLOWED["p205"]=p205

def p206(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for j in range(71):
        real_part = np.real(t1)**(j + 1) * np.sin((j + 1) * np.angle(t2) + np.log(np.abs(t1) + 1))
        imag_part = np.imag(t2)**(j + 1) * np.cos((j + 1) * np.angle(t1) + np.log(np.abs(t2) + 1))
        cf[j] = real_part + np.imag(t1) * np.real(t2) * imag_part
    return cf.astype(np.complex128)

ALLOWED["p206"]=p206

def p207(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for j in range(71):
        k = (j + 1)**2 * np.real(t1) - np.imag(t2)
        r = np.sin((j + 1) * np.angle(t1)) + np.cos((j + 1) * np.angle(t2))
        magnitude = np.log(np.abs(t1) + 1) * (k + r**2)
        angle = np.abs(t2) * (j + 1) + np.real(t1) * np.sin(j + 1)
        cf[j] = magnitude * (np.cos(angle) + np.sin(angle) * 1j)
    for j in range(71):
        cf[j] += np.conj(t1) * np.conj(t2) * (j + 1)
    return cf.astype(np.complex128)
    
ALLOWED["p207"]=p207

def p208(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for j in range(71):
        mag = np.log(np.abs(t1) * (j + 1) + np.abs(t2) * (j + 1)**2 + 1)
        angle = np.angle(t1) * np.sqrt(j + 1) + np.angle(t2) * np.log(j + 2)
        cf[j] = mag * (np.cos(angle) + np.sin(angle) * 1j)
    return cf.astype(np.complex128)
    
ALLOWED["p208"]=p208

def p209(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for j in range(71):
        mag = 0
        ang = 0
        for k in range(1, min(j + 1, 11)):
            mag += np.real(t1)**k * np.real(t2)**(j - k) * np.log(np.abs(t1) + np.abs(t2) + 1)
            ang += np.angle(t1) * k - np.angle(t2) * (j - k) + np.sin(k) * np.angle(np.conj(t1 + t2))
        cf[j] = mag * (np.cos(ang) + np.sin(ang) * 1j)
    return cf.astype(np.complex128)
    
ALLOWED["p209"]=p209

def p210(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for j in range(71):
        mag = np.log(np.abs(t1)**(j + 1) + np.abs(t2)**(70 - j) + 1) * (1 + np.sin((j + 1) * np.real(t1)) - np.cos((j + 1) * np.imag(t2)))
        angle = np.angle(t1) * (j + 1) + np.angle(t2) * ((j + 1)**2) + np.sin((j + 1) * np.real(t1)) - np.cos((j + 1) * np.imag(t2))
        cf[j] = mag * (np.cos(angle) + np.sin(angle) * 1j)
    return cf.astype(np.complex128)

ALLOWED["p210"]=p210

def p211(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for j in range(71):
        mag = np.log(np.abs(t1) * (j + 1) + np.abs(t2) * np.sqrt(j + 1) + 1) + np.sin((j + 1) * np.real(t1)) * np.cos((j + 1) * np.imag(t2))
        angle = np.angle(t1)**2 / (j + 2) + np.angle(t2) * np.cos(j + 1) + np.real(t1 * t2)
        cf[j] = mag * (np.cos(angle) + np.sin(angle) * 1j)
    return cf.astype(np.complex128)
    
ALLOWED["p211"]=p211

def p212(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for j in range(71):
        cf[j] = (np.real(t1)**j + np.imag(t2)**(70 - j)) * np.cos(j * np.angle(t1 + t2)) + np.sin(j * np.angle(t1 * t2)) + np.log(np.abs(t1) + np.abs(t2) + 1)**j
    return cf.astype(np.complex128)
    
ALLOWED["p212"]=p212

def p213(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for j in range(71):
        k = (j + 1) * 3 + (j // 7)
        r = (j % 5) + 2
        mag = np.abs(t1)**k + np.abs(t2)**r + (np.real(t1)+np.imag(t2)) * np.log(np.abs(t1) + 1.0)
        angle = np.angle(t1) * k - np.angle(t2) * r + np.sin(j + 1) * np.cos(j + 1)
        cf[j] = mag * (np.cos(angle) + np.sin(angle) * 1j)
    return cf.astype(np.complex128)
    
ALLOWED["p213"]=p213

def p214(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    cf[0] = np.real(t1) + np.imag(t2) * 1j
    prev = t1 * t2
    for j in range(70):
        magnitude = np.log(np.abs(prev) + 1) + np.real(prev)**2 - np.imag(prev)**2
        angle = np.angle(prev) + np.sin(np.real(prev)) - np.cos(np.imag(prev))
        cf[j + 1] = magnitude * (np.cos(angle) + np.sin(angle) * 1j)
        prev = prev * t1 - t2 / (j + 1)
    return cf.astype(np.complex128)
    
ALLOWED["p214"]=p214

def p215(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for j in range(71):
        cf[j] = (np.real(t1) + (j + 1)) * np.sin((j + 1) * np.angle(t1)) + np.conj(t2) * (np.imag(t1) + (j + 1)) * np.cos((j + 1) * np.angle(t2)) * 1j
    return cf.astype(np.complex128).astype(np.complex128)
    
ALLOWED["p215"]=p215

def p216(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for j in range(71):
        k = ((j + 1) * 5) % 20 + 1
        r = (j // 6) + 1
        cf[j] = (np.real(t1)**k + np.imag(t2)**r) * np.cos((j + 1) * np.angle(t1)) + np.conj(t2) * np.sin((j + 1) * np.angle(t2)) - np.real(t1 * t2) * np.cos(j + 1)
    return cf.astype(np.complex128)
    
ALLOWED["p216"]=p216

def p217(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for j in range(71):
        angle = np.angle(t1) * (j + 1) + np.angle(t2) * (71 - (j + 1))
        magnitude = np.abs(t1)**(j + 1) * np.abs(t2)**(71 - (j + 1)) + np.log(np.abs(t1) + 1) * np.sin(j + 1) + np.cos(j + 1)
        cf[j] = magnitude * (np.cos(angle) + np.sin(angle) * 1j)
    return cf.astype(np.complex128)
    
ALLOWED["p217"]=p217

def p218(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(71):
        mag = np.sin((k + 1) * np.abs(t1)) + np.cos((k + 1) * np.abs(t2)) + np.log(np.abs(t1) + (k + 1))
        angle = np.angle(t1) * (k + 1) + np.angle(t2) * (71 - (k + 1))
        cf[k] = mag * (np.cos(angle) + np.sin(angle) * 1j)
    return cf.astype(np.complex128)
    
ALLOWED["p218"]=p218

def p219(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for k in range(71):
        angle = np.angle(t1) * (k + 1) - np.angle(t2) * (71 - (k + 1))
        magnitude = np.abs(t1)**(k + 1) + np.log(np.abs(t2) + 1)**(k + 1)
        cf[k] = magnitude * (np.cos(angle) + np.sin(angle) * 1j)
    return cf.astype(np.complex128).astype(np.complex128)
    
ALLOWED["p219"]=p219

def p220(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for j in range(71):
        cf[j] = (np.real(t1) * (j + 1) + np.imag(t2) * (j + 1)**2) * np.sin(np.angle(t1) * (j + 1)) + \
                    np.cos(np.angle(t2) * (j + 1)) * np.log(np.abs(t1) + np.abs(t2) * (j + 1)) + np.real(np.conj(t1) * t2)**(j + 1) - np.imag(t1 * np.conj(t2))**(j + 1)
    return cf.astype(np.complex128)
    
ALLOWED["p220"]=p220

def p221(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for j in range(71):
        cf[j] = (np.real(t1)**(j + 1) * np.cos(np.angle(t2) + (j + 1))) + (np.imag(t2)**(j + 1) * np.sin(np.angle(t1) * (j + 1))) + np.log(np.abs(t1) + (j + 1)) + np.log(np.abs(t2) + 1) + np.conj(t1) * (j + 1) - np.conj(t2)**(j + 1)
    return cf.astype(np.complex128)
    
ALLOWED["p221"]=p221

def p222(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    prev = t1 + t2
    for j in range(71):
        magnitude = np.abs(prev) * np.log(np.abs(prev) + 1)
        angle = np.angle(prev) + np.sin(j + 1) * np.cos(j + 1)
        cf[j] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
        prev = prev * t1 - t2 / (j + 1)
    return cf.astype(np.complex128)
    
ALLOWED["p222"]=p222

def p223(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)
    for j in range(71):
        mag = np.log(np.abs(t1) * (j + 1) + 1) * (1 + np.sin((j + 1) * np.angle(t2)))
        ang = np.angle(t1) * np.sqrt(j + 1) + np.cos((j + 1) * np.angle(t2))
        cf[j] = mag * np.cos(ang) + mag * np.sin(ang) * 1j
    return cf.astype(np.complex128)

ALLOWED["p223"]=p223
    
def p224(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(35):
        if (j + 1) % 4 == 1:
            k = j + 3
            cf[j] = (np.real(t1)**k) + (np.imag(t2)**k) * np.sin(np.angle(t1) * k)
        elif (j + 1) % 4 == 2:
            k = j + 4
            cf[j] = (np.abs(t1 + t2)**k) * np.cos(np.angle(t2) * k) + np.log(np.abs(t1) + 1)
        elif (j + 1) % 4 == 3:
            k = j + 2
            cf[j] = np.conj(t1)**k - np.conj(t2)**k + np.sin(t1 * k) - np.cos(t2 * k)
        else:
            k = j + 1
            cf[j] = np.log(np.abs(t1) + 1)**k + np.log(np.abs(t2) + 1)**(35 - k) + np.real(t1 * t2) * np.imag(t1 + t2)
    # Assign specific intricate coefficients
    i1 = np.array([4, 9, 14, 19, 24, 29, 34],dtype=np.intp)
    cf[i1] = np.array([2 + 3j, -4j, 5 - 6j, -7 + 8j, 9 - 10j, 11 + 12j, -13 + 14j],dtype=np.complex128)
    # More intricate assignments
    cf[7] = 100j * t2**3 + 100j * t2**2 - 100 * t2 - 100
    cf[11] = 150j * t1**3 + 150j * t1**2 + 150 * t2 - 150
    cf[17] = 200j * t2**3 - 200j * t2**2 + 200 * t2 - 200
    cf[21] = 250 * np.sin(t1) + 300j * np.cos(t2) + 50 * np.log(np.abs(t1) + 1)
    cf[27] = 350 * (t1 * t2) + 400j * (t1 + t2)
    cf[32] = 450j * t1 * t2 + 500 * np.conj(t1 - t2)
    return cf.astype(np.complex128)
    
ALLOWED["p224"]=p224

def p225(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    # Fixed coefficients with varied complex values
    fixed_indices = np.array([3, 8, 14, 19, 23, 29],dtype=np.intp)
    fixed_values = np.array([2 - 1j, -3 + 4j, 5 - 2j, -4 + 3j, 1.5 - 0.5j, 3 + 2j],dtype=np.complex128)
    cf[fixed_indices] = fixed_values
    # Loop to assign coefficients with intricate patterns
    for j in range(35):
        if j not in fixed_indices:
            k = (j % 7) + 1
            r = np.real(t1) * np.imag(t2) / (k + j + 1)
            angle = np.angle(t1) + np.angle(t2) * k
            magnitude = np.abs(t1)**k + np.abs(t2)**(7 - k)
            cf[j] = (magnitude * np.cos(angle)) + (r * np.sin(angle)) + np.conj(t1) * t2 / (j + 1)
    
    # Additional intricate assignments
    for k in range(1, 6):
        idx = 5 * k
        if idx <= 35:
            cf[idx] = (np.real(t1)**k - np.imag(t2)**k) * np.cos(k * np.angle(t1)) + (np.abs(t1) + np.abs(t2)) * np.sin(k * np.angle(t2)) + np.conj(t1)**k * np.conj(t2)**k / (k + 1)
    
    # Define specific coefficients with creative combinations
    cf[9] = np.log(np.abs(t1 * t2) + 1) * (np.real(t1) + np.imag(t2)) + 2j * np.real(t1)**2 - 3 * np.imag(t2)**2
    cf[20] = np.conj(t1) * t2**3 + np.real(t2) * np.imag(t1) - 4j * np.abs(t1 + t2)
    cf[35] = np.sum(np.real(cf)) + np.sum(np.imag(cf)) * 1j
    
    return cf.astype(np.complex128)
    
ALLOWED["p225"]=p225

def p226(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    # Assign fixed coefficients
    i1=np.array([0, 6, 13, 20, 27, 34],dtype=np.intp)
    v1=np.array([2, -1 + 3j, 4 - 2j, -3 + 5j, 1.5 - 1.5j, 0.3 + 0.7j],dtype=np.complex128)
    cf[i1] = v1
    
    # Loop to assign other coefficients with intricate calculations
    for j in range(2, 35):
        if j not in [0, 6, 13, 20, 27, 34]:
            r = np.real(t1) + np.imag(t2) * (j + 1)
            theta = np.angle(t1) * (j + 1) - np.angle(t2) / ((j + 1) % 5 + 1)
            magnitude = np.sin(r) * np.cos(r + theta) + np.log(np.abs(t1) + np.abs(t2) + (j + 1))
            phase = theta + np.sin((j + 1) * np.real(t1)) - np.cos((j + 1) * np.imag(t2))
            cf[j] = magnitude * (np.cos(phase) + 1j * np.sin(phase))
    
    # Additional intricate assignments
    cf[9] = np.conj(t1) * t2**2 + np.sin(t1 * t2)
    cf[18] = np.abs(t1 + t2) * np.exp(1j * np.angle(t1 - t2))
    cf[25] = (np.real(t1)+np.imag(t1)+np.real(t2)+np.imag(t2)) + (np.abs(t1) * np.abs(t2))
    
    return cf.astype(np.complex128)
    
ALLOWED["p226"]=p226

def p227(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(35):
        k = (j + 1) + (j // 5)
        magnitude = np.log(np.abs(t1) + 1) * np.sin(j + 1) + np.log(np.abs(t2) + 1) * np.cos(j + 1)
        angle = np.angle(t1) * (j + 1)**0.5 - np.angle(t2) * np.log(j + 2)
        cf[j] = magnitude * (np.cos(angle) + 1j * np.sin(angle)) + np.conj(t1)**k - np.conj(t2)**(35 - j)
    return cf.astype(np.complex128)
    
ALLOWED["p227"]=p227

def p228(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    # Assign fixed coefficients
    i1=np.array([3, 7, 13, 17, 26, 32],dtype=np.intp)
    v1=np.array( [2.5, -3.4, 5.6, -4.2, 3.1, 0.8],dtype=np.complex128)
    cf[i1] = v1
    # Loop through coefficients to assign intricate patterns
    for j in range(35):
        if j not in [3, 7, 13, 17, 26, 32]:
            k = 35 - j
            r = (j % 5) + 1
            angle = np.angle(t1) * (j + 1) + np.angle(t2) * k
            magnitude = np.log(np.abs(t1) + 1) * np.sin(j + 1) + np.log(np.abs(t2) + 1) * np.cos(k)
            cf[j] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    
    # Additional intricate coefficient assignments
    cf[9] = np.conj(t1)**2 * t2 + np.sin(t1 * t2)
    cf[18] = np.abs(t1 + t2) * np.exp(1j * np.angle(t1 - t2))
    cf[25] = (np.real(t1)+np.imag(t1)+np.real(t2)+np.imag(t2)) + (np.abs(t1) * np.abs(t2))
    return cf.astype(np.complex128)
    
ALLOWED["p228"]=p228

def p229(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    # Initialize specific coefficients with unique values
    i1=np.array([2, 6, 13, 18, 21, 27],dtype=np.intp)
    v1=np.array([2 - 3j, -4 + 5j, 1.5 - 2.5j, -3.3 + 4.4j, 0.5 - 1.2j, 3 - 3j],dtype=np.complex128)
    cf[i1] = v1
    # Loop through coefficients and assign values based on intricate calculations
    for j in range(35):
        if j not in [2, 6, 13, 18, 21, 27]:
            angle = np.angle(t1**(j + 1) + t2**(35 - j))
            magnitude = np.abs(t1)**((j % 5) + 1) * np.abs(t2)**((35 - j) % 7 + 1)
            cf[j] = magnitude * (np.cos(angle) + np.sin(angle) * 1j) + np.conj(t1**2 - t2**2)
    j = 35
    # Additional intricate computations for specific coefficients
    cf[9] = np.sin(t1 * j) + np.cos(t2 * j) + np.log(np.abs(t1) + np.abs(t2) + 1)
    cf[17] = np.real(t1)**2 - np.imag(t2)**2 + 2j * np.real(t1) * np.imag(t2)
    cf[25] = (np.abs(t1) * np.abs(t2)) * np.exp(1j * np.angle(t1 + t2))
    cf[30] = (np.abs(t1 + t2)+np.abs(t1 - t2)) + 1j * (np.angle(t1)+np.angle(t2))
    return cf.astype(np.complex128)
    
ALLOWED["p229"]=p229

def p230(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    i1=np.array([2, 7, 11, 16, 22, 27, 31, 34],dtype=np.intp)
    v1=np.array([3, -2, 5, -4, 6, -3, 2, -1],dtype=np.complex128)
    # Assign fixed coefficients
    cf[i1] = v1
    # Loop to assign coefficients with intricate patterns
    for j in range(35):
        if (j + 1) % 4 == 0:
            cf[j] = (np.real(t1)**2 + np.imag(t2)) * (j + 1) + (np.imag(t1) * np.real(t2)) * 1j
        elif (j + 1) % 5 == 1:
            cf[j] = np.sin(t1 * (j + 1)) + np.cos(t2 + (j + 1)) * 1j
        elif (j + 1) % 3 == 2:
            cf[j] = np.log(np.abs(t1) * (j + 1) + 1) + np.angle(t2)**(j + 1) * 1j
        else:
            cf[j] = np.real(t1 + t2) * (j + 1) + np.imag(t1 - t2) * 1j
    
    # Additional intricate assignments using nested loops
    for k in range(1, 6):
        for r in range(1, 8):
            idx = (k * r) % 35
            cf[idx] += (np.real(t1)**k * np.imag(t2)**r) + (np.real(t2)**k * np.imag(t1)**r) * 1j
    
    # Modify certain coefficients with conjugates and products
    for m in range(10, 31, 5):
        cf[m] = cf[m] * np.conj(t1) + (np.abs(t2) * m) * 1j
    
    return cf.astype(np.complex128)

ALLOWED["p230"]=p230

def p231(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(50, dtype=np.complex128)
    for k in range(50):
        magnitude = np.abs(t1)**((k % 5) + 1) + np.abs(t2)**((k % 7) + 1) + np.log(np.abs(t1) + 1) * np.sin(k + 1)
        angle = np.angle(t1) * np.cos(k + 1) + np.angle(t2) * np.sin(k + 1)
        cf[k] = magnitude * (np.cos(angle) + np.sin(angle) * 1j)
    for j in range(2, 51, 3):
        cf[j] += np.conj(t1) * t2**(j % 4)
    for r in range(5, 51, 5):
        cf[r] += np.real(t2) * np.cos(r) + np.imag(t1) * np.sin(r) * 1j
    cf[9] = (np.abs(t1)+np.abs(t2)) + (np.real(t1) * np.real(t2)) * 1j
    cf[19] = np.real(t1)**2 - np.imag(t2)**2 + 2 * np.real(t1) * np.imag(t2) * 1j
    cf[29] = np.log(np.abs(t1) + np.abs(t2) + 1) * (np.sin(np.angle(t1)) + np.cos(np.angle(t2)) * 1j)
    cf[39] = np.abs(t1) * np.abs(t2) * np.exp(1j * (np.angle(t1) - np.angle(t2)))
    cf[49] = np.conj(t1) + np.conj(t2) - t1 * t2 * 1j
    return cf.astype(np.complex128)
    
ALLOWED["p231"]=p231

def p232(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(1, 36):
        k = j % 5 + 1
        r = j // 7 + 1
        cf[j - 1] = (np.real(t1)**k - np.imag(t2)**r) * np.sin(j * np.angle(t1 + t2)) / (np.abs(t1) + np.abs(t2) + 1) + \
            np.conj(t1)**k * np.cos(r * np.angle(t2)) + np.log(np.abs(t1) + 1) * np.log(np.abs(t2) + 1)
    
    i1 = np.array([2, 7, 11, 18, 22, 28, 33],dtype=np.intp)
    cf[i1] = np.array([
        (t1 * t2)**2 - np.conj(t1) * np.sin(t2),
        np.abs(t1) * np.real(t2) + np.imag(t1) * np.imag(t2),
        np.cos(t1) + np.sin(t2),
        np.log(np.abs(t1) + 1) + np.log(np.abs(t2) + 1),
        t1**3 - t2**3 + np.conj(t1 * t2),
        np.real(t1 + t2) * np.imag(t1 - t2),
        np.sin(t1 * t2) + np.cos(np.conj(t1)) * np.cos(np.conj(t2))
    ])
    
    i2 = np.array([4, 9, 14, 19, 24, 29],dtype=np.intp)
    cf[i2] = np.array([
        np.real(t1) ** 2 + np.imag(t2) ** 2,
        np.angle(t1) * np.angle(t2),
        np.abs(t1 + t2) * np.conj(t1 - t2),
        np.sin(np.abs(t1)) * np.cos(np.abs(t2)),
        np.log(np.abs(t1 * t2) + 1),
        (np.real(t1) + np.imag(t2) + np.angle(t1 + t2))
    ])
    
    i3 = np.array([6, 10, 16, 20, 26, 30],dtype=np.intp)
    cf[i3] = np.array([
        t1**2 * t2 - t1 * t2**2,
        np.conj(t1)**2 + np.conj(t2)**2,
        np.sin(t1) * np.cos(t2) + np.cos(t1) * np.sin(t2),
        np.real(t1 * t2) + np.imag(t1 * t2),
        (t1 + t2)**3 - (t1 - t2)**3,
        (np.abs(t1) * np.abs(t2) * np.real(t1 + t2))
    ])
    
    i4 = np.array([8, 12, 17, 23, 27, 31],dtype=np.intp)
    cf[i4] = np.array([
        np.real(t1)**3 - np.imag(t2)**3,
        np.angle(t1)**2 + np.angle(t2)**2,
        np.sin(t1 + t2) - np.cos(t1 - t2),
        np.log(np.abs(t1)**2 + np.abs(t2)**2 + 1),
        np.real(np.conj(t1) * t2),
        np.imag(t1 * np.conj(t2))
    ])
    
    i5 = np.array([3, 5, 15, 20, 21, 25, 32, 34],dtype=np.intp)
    cf[i5] = np.array([
        np.real(t1) * np.real(t2),
        np.imag(t1) * np.imag(t2),
        np.angle(t1 + t2) * np.abs(t1 * t2),
        np.sin(np.real(t1)) + np.cos(np.imag(t2)),
        np.log(np.abs(t1 + t2) + 1),
        np.real(np.conj(t1 + t2)),
        np.sin(np.abs(t1)**2) * np.cos(np.abs(t2)**2),
        np.real(t1)**2 + np.imag(t1)**2 + np.real(t2)**2 + np.imag(t2)**2
    ])
    
    return cf.astype(np.complex128)
    
ALLOWED["p232"]=p232

def p233(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    
    # Assign initial coefficients with direct values
    i1 = np.array([2, 5, 11, 17, 23, 29],dtype=np.intp)
    cf[i1] = np.array([2 - 3j, -4 + 2j, 5 - 1j, -6 + 3j, 7 - 2j, -8 + 4j],dtype=np.complex128)
    
    # Loop to assign intricate coefficients
    for j in range(1, 36):
        if j % 4 == 0:
            cf[j - 1] = (np.real(t1) * j**2 - np.imag(t2) * j) + (np.imag(t1) + np.real(t2)) * 1j
        elif j % 5 == 0:
            cf[j - 1] = (np.abs(t1) ** (j % 3 + 1)) * np.cos(np.angle(t2) * j) + (np.sin(np.angle(t1) * j) * np.abs(t2)) * 1j
        elif j % 3 == 0:
            cf[j - 1] = np.log(np.abs(t1) + 1) * j - np.log(np.abs(t2) + 1) * j * 1j
        else:
            cf[j - 1] = (np.real(t1) + np.real(t2)) * j + (np.imag(t1) - np.imag(t2)) * j * 1j
    
    # Introduce variations with complex operations
    for k in range(1, 36):
        if k % 7 == 0:
            cf[k - 1] = (np.conj(t1) * t2**2) + (np.sin(t1) - np.cos(t2)) * 1j
        if k % 11 == 0:
            cf[k - 1] = (np.real(t1) * np.imag(t2) * k) + (np.abs(t1) + np.abs(t2) + k) * 1j
    
    # Assign specific coefficients to ensure non-symmetry
    cf[4] = 10 * t1 - 5j * t2**2
    cf[9] = 15j * t1**3 + 8 * t2
    cf[14] = 20 * t1**2 - 10j * t2**3
    cf[19] = 25j * t1 - 12 * t2**2
    cf[24] = 30 * t1**4 + 15j * t2
    cf[34] = 35j * t1**2 - 18 * t2**3
    
    return cf.astype(np.complex128)
    
ALLOWED["p233"]=p233

def p242(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(1, 36):
        if j % 5 == 1:
            cf[j - 1] = np.sin(np.abs(t1) * j) + np.cos(np.angle(t2) * j)
        elif j % 5 == 2:
            cf[j - 1] = np.log(np.abs(t1) + 1) * t2**j
        elif j % 5 == 3:
            cf[j - 1] = np.conj(t1)**j - np.real(t2) * j
        elif j % 5 == 4:
            cf[j - 1] = np.imag(t1) + np.abs(t2) * np.sin(j * np.angle(t1))
        else:
            cf[j - 1] = t1 * t2**j + np.cos(j) - np.sin(j)
    cf[6] = 50j * t1**2 - 30j * t2 + 20
    cf[13] = 80 * t1 - 60j * t2**2 + 10
    cf[20] = 40j * t1**3 + 25 * np.conj(t2) - 15
    cf[27] = 70 * np.abs(t1) + 35j * np.angle(t2) + 5
    cf[34] = 90j * t1 * t2 - 45 * np.real(t1) + 22.5
    return cf

ALLOWED["p242"]=p242


def p243(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    # Assign base coefficients with fixed values
    i = np.array([0, 5, 9, 14, 21, 27])
    cf[i] = np.array([2, -3 + 2j, 4.5, -5.2j, 3.3, -1.1])
    
    # Loop to assign lower degree coefficients
    for j in range(2, 6):
        cf[j - 1] = (np.real(t1)**j + np.imag(t2)**j) * np.sin(np.angle(t1) * j) / (1 + j)
    
    # Loop to assign middle degree coefficients
    for k in range(7, 15):
        cf[k - 1] = (np.abs(t1)**k * np.cos(np.angle(t2) * k)) + np.conj(t2) * np.log(np.abs(t1 * t2) + 1)
    
    # Loop to assign higher degree coefficients
    for r in range(16, 26):
        cf[r - 1] = (np.real(t1**r) - np.imag(t2**r) * 1j) * np.sin(t1 + t2) + np.cos(t1 * t2)
    
    # Assign coefficients using product and sum
    cf[25] = np.prod(np.array([np.abs(t1), np.abs(t2)])) + np.sum(np.array([np.real(t1), np.imag(t2)])) * np.conj(t1 + t2)
    cf[26] = np.log(np.abs(t1) + 1) + np.log(np.abs(t2) + 1) * 1j
    cf[28] = np.real(t1 * t2) - np.imag(t1 / t2) * 1j
    cf[29] = np.sin(t1**2) + np.cos(t2**3) * 1j
    cf[31] = np.abs(t1 + t2) * np.exp(-np.real(t1 - t2))
    cf[33] = np.angle(t1) + np.angle(t2) * 1j
    
    # Assign the last coefficient with a unique pattern
    cf[34] = (t1**3 + t2**3) / (1 + np.abs(t1) + np.abs(t2))
    return cf

ALLOWED["p243"]=p243

def p273(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    # Assign base coefficients with fixed values
    i=np.array([0, 5, 11, 17, 23, 29])
    cf[i] = [2, -3 + 1j, 4, -5j, 6 + 2j, -7]
    for j in range(2, 35):
        if cf[j] == 0:
            cf[j] = (np.real(t1)**j - np.imag(t2)**j) + (np.angle(t1) * j + np.abs(t2)) * 1j

    for k in range(3, 34):
        cf[k] += np.sin(t1 * k) * np.cos(t2 / k) + np.log(np.abs(t1) + 1) * np.sin(np.angle(t2)) * 1j

    cf[9] = np.conj(t1) * t2*t2 + np.abs(t2) * 1j
    cf[14] = np.real(t1*t1*t1) + np.imag(t2*t2*t2) * 1j
    cf[19] = np.prod(np.array([np.real(t1), np.real(t2)])) + np.prod(np.array([np.imag(t1), np.imag(t2)])) * 1j
    cf[24] = np.sum(np.array([np.abs(t1), np.abs(t2)])) + np.angle(t1 + t2) * 1j
    cf[27] = np.sin(np.abs(t1)) + np.cos(np.abs(t2)) * 1j
    cf[31] = np.log(np.abs(t1) + 1) + np.log(np.abs(t2) + 1) * 1j
    cf[34] = np.conj(t1 + t2)
    return cf

ALLOWED["p273"]=p273

def p344(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35 # was 35
    cf = np.zeros(n, dtype=np.complex128)
    rec_seq = np.linspace(np.real(t1), np.real(t2), n)
    imc_seq = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(n):
        mag = np.log(np.abs(rec_seq[j] * imc_seq[j]) + 1) * (1 + np.sin(j * np.pi / 3)) * (j % 4 + 1)
        ang = np.angle(t1) * np.cos(j * np.pi / 5) + np.angle(t2) * np.sin(j * np.pi / 7) + np.log(np.abs(rec_seq[j] + imc_seq[j]) + 1)
        cf[j] = mag * (np.cos(ang) + 1j * np.sin(ang))
    return cf

ALLOWED["p344"]=p344

def p345(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(np.real(t1), np.real(t2), n)
    for j in range(n):
        mag_part =  np.log(np.abs(rec[j]) + 1) * np.prod(np.arange(1, j + 1)) / (j + 2)
        angle_part = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j / 3.0)
        cf[j] = mag_part * (np.cos(angle_part) + 1j * np.sin(angle_part))
        cf[j] += np.conj(cf[j]) * np.sin(j * np.pi / 4)
    return cf

ALLOWED["p345"]=p345

def p371(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        mag_part = np.log(np.abs(t1) + j) * np.sqrt(j) + np.sin(j * np.angle(t2))**2
        angle_part = np.angle(t1) * np.cos(j) + np.angle(t2) * np.sin(j)
        cf[j - 1] = mag_part * (np.cos(angle_part) + 1j * np.sin(angle_part)) + np.conj(t1)**j - np.log(np.abs(t2) + 1) * np.sin(j)
    return cf

ALLOWED["p371"]=p371

def p393(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(1, 36):
        mag = 0
        angle = 0
        for k in range(1, j + 1):
            mag += np.log(np.abs(t1) + k) * np.sin(k * np.real(t2)) / (k + 1)
            angle += np.angle(t1)**k * np.cos(k * np.imag(t2))
        cf[j - 1] = mag * (np.cos(angle) + np.sin(angle) * 1j)
    return cf

ALLOWED["p393"]=p393

def p425(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n +1):
        angle_part = np.sin(j * np.pi /6) * np.cos(j * np.pi /8) + np.angle(t1) * np.log(j +1)
        magnitude_part = np.log(np.abs(t1) + j**2) * np.abs(np.cos(j)) + \
                            np.log(np.abs(t2) + j) * np.abs(np.sin(j / 2))
        cf[j-1] = (magnitude_part + np.real(t1) * np.real(t2) / (j +1)) * \
                    np.exp(1j * angle_part)
        if j %5 ==0:
            cf[j-1] += np.conj(cf[j-1])
        cf[j-1] *= (1 + 0.1 * np.sin(j))
    return cf

ALLOWED["p425"]=p425

def p440(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        # Calculate real and imaginary parts
        real_part = np.real(t1)**j * np.log(np.abs(j) + 1) + np.sin(j * np.real(t2)) * np.cos(j**2)
        imag_part = np.imag(t1) * j**0.5 + np.cos(j * np.imag(t2)) * np.log(np.abs(t1 + t2) + 1)
        # Assign to complex coefficient with scaling
        cf[j-1] = (real_part+imag_part * 1j) * (1 + 0.1 * j)
    return cf

ALLOWED["p440"]=p440

def p472(z,a,state):
    t1, t2 = z[0], z[1]
    n = 71
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        mag = np.log(np.abs(t1) + j) * (1 + np.sin(j * np.pi /7))
        ang = np.angle(t1)*np.cos(j * np.pi /5) + np.angle(t2)*np.sin(j * np.pi /3)
        cf[j-1] = mag * np.exp(1j * ang) + (t1.real + t2.real)/(j +1)
    for k in range(1, n+1):
        cf[k-1] += np.conj(t1)**k - np.conj(t2)**(n -k +1)
    for r in range(1, n+1):
        cf[r-1] *= (1 + 0.1 * np.cos(r * np.angle(t1)) * np.sin(r * np.angle(t2)))
    return cf

ALLOWED["p472"]=p472

def p483(z,a,state):
    t1, t2 = z[0], z[1]
    n =35
    cf = np.zeros(n, dtype=np.complex128)
    r1 = t1.real
    i1 = t1.imag
    r2 = t2.real
    i2 = t2.imag
    for j in range(1, n+1):
        part1 = r1**j * np.sin(j * np.angle(t2))
        part2 = i2**(n -j) * np.cos(j * np.abs(t1))
        part3 = np.log(np.abs(t1) + np.abs(t2) + j)
        part4 = np.prod(np.array([r1 + j, i2 +j, np.log(np.abs(t1)+1)]))
        magnitude = part1 * part2 + part3 * part4
        angle = np.angle(t1)*np.sin(j) + np.angle(t2)*np.cos(j) + np.log(np.abs(t1)+1)/j
        cf[j-1] = magnitude * np.exp(1j * angle)
    return cf

ALLOWED["p483"]=p483

def p530(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(t1.real, t2.real, num=n)
    imc = np.linspace(t1.imag, t2.imag, num=n)
    for j in range(n):
        r = rec[j]
        m = imc[j]
        term1 = np.sin(r * np.pi / (j + 2)) * np.cos(m * np.pi / (j + 3))
        term2 = np.log(np.abs(r + m) + 1) * (t1.real ** (j + 1))
        term3 = np.prod(np.array([r, m, j + 1])) ** (1 / (j + 1))
        mag = term1 + term2 + term3
        angle = np.angle(t1) * np.sin(m * np.pi / (j + 4)) + np.angle(t2) * np.cos(r * np.pi / (j + 5)) + np.log(j + 2)
        cf[j] = mag * (np.cos(angle) + 1j * np.sin(angle))
    return cf

ALLOWED["p530"]=p530

def p532(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for r in range(1, n + 1):
        j = r % 7 + 1
        k = np.floor(r / 5) + 1
        magnitude = (np.log(np.abs(t1) + 1) * np.cos(r) + np.log(np.abs(t2) + 1) * np.sin(r)) * (1 + r / 10)
        angle = np.angle(t1) * np.sin(r / 2) - np.angle(t2) * np.cos(r / 3) + np.sin(r) * np.cos(r / 4)
        cf[r - 1] = magnitude * (np.cos(angle) + np.sin(angle) * 1j)
    return cf

ALLOWED["p532"]=p532

def p546(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    imc1 = t1.imag 
    imc2 = t2.imag 
    for j in range(1, n + 1):
        theta = (imc1 - imc2) / n * j + np.sin(j * np.pi / 3)
        magnitude = np.log(np.abs(t1) * j + np.abs(t2) * (n - j + 1)) + np.sqrt(j)
        cf[j - 1] = magnitude * (np.cos(theta) + 1j * np.sin(theta))
    return cf

ALLOWED["p546"]=p546

def p597(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 25
    cf = np.zeros(degree + 1, dtype=np.complex128)
    r1 = t1.real
    i2 = t2.imag
    for j in range(0, degree + 1):
        mag = 0
        angle = 0
        for k in range(1, j + 2):
            term_mag = np.log(np.abs(t1) * k + 1) * np.sin(k * np.pi * r1) + np.cos(k * np.pi * i2)
            term_angle = np.angle(t1) * k**2 - np.angle(t2) * np.sqrt(k)
            mag += term_mag * np.exp(1j * term_angle)
        if j < degree / 3:
            mag *= (j + 1)
        elif j < 2 * degree / 3:
            mag /= (j + 1)
        else:
            mag *= (j + 1)**2
        cf[j] = mag
    cf[0] = (t1.real * t2.real) + 1j * (t1.imag - t2.imag) + np.sin(t1.real) * np.cos(t2.imag)
    return cf

ALLOWED["p597"]=p597

def p600(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 25
    cf = np.zeros(degree + 1, dtype=np.complex128)
    rec_step = np.linspace(t1.real, t2.real, num=degree + 1)
    imc_step = np.linspace(t1.imag, t2.imag, num=degree + 1)
    for j in range(1, degree + 2):
        mag_part1 = np.log(np.abs(t1) + j) * np.sin(j * np.pi / 12)
        mag_part2 = np.cos(j * np.pi / 8) * np.log(np.abs(rec_step[j - 1] - imc_step[j - 1]) + 1)
        magnitude = mag_part1 + mag_part2 + j**0.8
        
        angle_part1 = np.angle(t1) * np.sin(j * np.pi / 10)
        angle_part2 = np.angle(t2) * np.cos(j * np.pi / 14)
        angle_part3 = np.sin(j * np.pi / 6) - np.cos(j * np.pi / 9)
        angle = angle_part1 + angle_part2 + angle_part3
        
        cf[j - 1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf

ALLOWED["p600"]=p600

def p666(z,a,state):
    t1, t2 = z[0], z[1]
    n = 8
    cf = np.zeros(n, dtype=np.complex128)
    rec1, imc2 = t1.real, t2.imag
    for j in range(1, n+1):
        r_part = np.log(np.abs(rec1 + j) + 1) * np.sin(j * np.pi / 4)
        i_part = np.log(np.abs(imc2 - j) + 1) * np.cos(j * np.pi / 3)
        magnitude = r_part + i_part
        angle = np.angle(t1) * np.cos(j) + np.angle(t2) * np.sin(j)
        cf[j-1] = magnitude * (np.cos(angle) + 1j * np.sin(angle)) 
    return cf

ALLOWED["p666"]=p666

def p729(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree +1, dtype=np.complex128)
    for j in range(0, degree +1):
        r_part = t1.real * j**2 - t2.real * np.sqrt(j +1)
        im_part = (t1.imag + t2.imag) * np.log(j +2)
        magnitude = np.abs(t1)**(j %3 +1) + np.abs(t2)**(degree -j)
        angle = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j)
        cf[j] = (r_part +1j * im_part) * magnitude * np.exp(1j * angle)
    return cf

ALLOWED["p729"]=p729

def p808(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    for k in range(25):
        cf[k] = (k + t1) / (k + t2)
    cf[4] += np.log(np.abs(t1 + t2))
    cf[9] += np.sin(np.real(t1)) + np.cos(np.imag(t2))
    cf[14] += np.abs(cf[13]) ** 2 + np.angle(cf[12]) ** 2
    cf[19] += np.abs(np.real(t2) * np.imag(t1))
    cf[24] += np.abs(t1 + np.conj(t2))
    return cf

ALLOWED["p808"]=p808

def p821(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = 3 * t1 + 5j * t2
    for k in range(1, 25):
        mod_t1 = np.abs(t1)
        arg_t2 = np.angle(t2)
        cf[k] = cf[k-1] * (mod_t1 + arg_t2)
        if cf[k].real < 0 and cf[k].imag < 0:
            cf[k] = np.conj(cf[k])
        if np.abs(cf[k].real) > 10:
            cf[k] = cf[k] / mod_t1
        if np.abs(cf[k].imag) > 10:
            cf[k] = cf[k] / (1j * arg_t2)
    return cf

ALLOWED["p821"]=p821

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fun", type=str, default="nop")
    ap.add_argument("--z", type=str, default="0+0j")
    ap.add_argument("--a", type=str, default="0+0j")
    args = ap.parse_args()
    z = np.array(ast.literal_eval(args.z),dtype=np.complex128)
    a = np.array(ast.literal_eval(args.a),dtype=np.complex128)
    state  = Dict.empty(key_type=types.int8,value_type=types.complex128[:])
    print(f"{args.fun}({args.z},{args.a}) = {ALLOWED[args.fun](z,a,state)}")

if __name__ == "__main__":
    main()

