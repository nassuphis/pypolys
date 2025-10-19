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

@njit(cache=True, fastmath=True)
def _poly_eval_cubic(a3: complex, a2: complex, a1: complex, a0: complex, x: complex) -> complex:
    # Horner: (((a3*x) + a2)*x + a1)*x + a0
    return (((a3 * x) + a2) * x + a1) * x + a0

@njit(cache=True, fastmath=True)
def durand_kerner_cubic(a3: complex, a2: complex, a1: complex, a0: complex):
    """
    Return (roots[3], ok) for a3*x^3 + a2*x^2 + a1*x + a0 = 0.
    a3 must be nonzero. JIT-safe, no Python lists/try/except.
    """
    roots = np.empty(3, dtype=np.complex128)

    # Guard: degenerate leading coefficient → fail
    if not np.isfinite(a3.real) or not np.isfinite(a3.imag):
        return roots, False
    if (a3.real == 0.0 and a3.imag == 0.0):
        return roots, False

    # Scale polynomial for numerical stability (optional but cheap)
    # We normalize so that |a3| == 1 (divide all coeffs by a3)
    inv_a3 = 1.0 / a3
    b2 = a2 * inv_a3
    b1 = a1 * inv_a3
    b0 = a0 * inv_a3

    # Initial guesses: 1, w, w^2 with a mild scale based on coefficients
    two_pi_over_3 = 2.0943951023931953  # 2π/3
    w = np.cos(two_pi_over_3) + 1j * np.sin(two_pi_over_3)
    r0 = 1.0 + 0.0j
    r1 = w
    r2 = w * w

    # Optional radius heuristic: |b0|^(1/3) as a first scale
    # Avoid NaNs if b0 = 0
    mag = np.abs(b0)
    if np.isfinite(mag) and mag > 0.0:
        s = mag ** (1.0 / 3.0)
        r0 *= s
        r1 *= s
        r2 *= s

    # Iterate
    max_iter = 60
    tol = 1e-12
    ok = False
    eps = 1e-18

    for _ in range(max_iter):
        # Evaluate normalized p(x) = x^3 + b2 x^2 + b1 x + b0
        p0 = _poly_eval_cubic(1.0 + 0.0j, b2, b1, b0, r0)
        p1 = _poly_eval_cubic(1.0 + 0.0j, b2, b1, b0, r1)
        p2 = _poly_eval_cubic(1.0 + 0.0j, b2, b1, b0, r2)

        # Denominators Π_{j≠k} (rk - rj)
        d0 = (r0 - r1) * (r0 - r2)
        d1 = (r1 - r0) * (r1 - r2)
        d2 = (r2 - r0) * (r2 - r1)

        # Jitter if any denom is ~0 to avoid blow-ups
        if np.abs(d0) < eps or np.abs(d1) < eps or np.abs(d2) < eps:
            r0 += eps
            r1 -= eps
            r2 += 2 * eps
            continue

        # Update
        nr0 = r0 - p0 / d0
        nr1 = r1 - p1 / d1
        nr2 = r2 - p2 / d2

        # Convergence check (max step)
        step = max(np.abs(nr0 - r0), np.abs(nr1 - r1), np.abs(nr2 - r2))
        r0, r1, r2 = nr0, nr1, nr2
        if step < tol:
            ok = True
            break

    roots[0] = r0
    roots[1] = r1
    roots[2] = r2
    return roots, ok

def g31(z,a,state):
    t1, t2 = z[0], z[1]
    n = 100
    cf = np.zeros(n, dtype=np.complex128)
    cf[0] = -100.0j  # cf[1] and cf[2] already 0

    # ---- first cubic: [t1, t2, t1, 1] ----
    roots1, ok1 = durand_kerner_cubic(t1, t2, t1, 1.0 + 0.0j)
    mid = (n // 2) - 2  # slice length 3: mid:mid+3
    if ok1:
        cf[mid]     = 100.0 * roots1[0]
        cf[mid + 1] = 100.0 * roots1[1]
        cf[mid + 2] = 100.0 * roots1[2]
    else:
        # broadcast scalar to slice (Numba supports this)
        cf[mid:mid + 3] = 100.0 + 0.0j

    # ---- second cubic: [t2, t1, t2, 10j] ----
    roots2, ok2 = durand_kerner_cubic(t2, t1, t2, 0.0 + 10.0j)
    end = n - 3
    if ok2:
        cf[end]     = 100.0 * roots2[0]
        cf[end + 1] = 100.0 * roots2[1]
        cf[end + 2] = 100.0 * roots2[2]
    else:
        cf[end:n] = 100.0 + 0.0j

    return cf

ALLOWED["g31"]=g31

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

def g49(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(30, dtype=np.complex128)
    cf1 = np.array([t1**4 - t2**3, 1j * t1 - t2, 50 * np.sin(t1 + t2), -100],dtype=np.complex128)
    unstable_roots = (cf1)
    cf[0:4] = unstable_roots
    cf2 = np.array([unstable_roots[0]**2, -unstable_roots[1], 1],dtype=np.complex128)
    cf[7:10] = (cf2)
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

def g58(z, a, state):
    t1, t2 = z[0], z[1]
    n = 10
    cf = np.zeros(n, dtype=np.complex128)

    EPS = 1e-12

    def safe_logc(x: complex) -> complex:
        # avoid log(0) or non-finite inputs
        if not (np.isfinite(x.real) and np.isfinite(x.imag)):
            return 0.0 + 0.0j
        if abs(x) < EPS:
            return 0.0 + 0.0j
        return np.log(x)

    def safe_divc(num: complex, den: complex) -> complex:
        if not (np.isfinite(den.real) and np.isfinite(den.imag)):
            return 0.0 + 0.0j
        if abs(den) < EPS:
            return 0.0 + 0.0j
        return num / den

    # 3**t1 * 2**t2 but stable for complex exponents
    def pow_real_base_complex_exp(b: float, zc: complex) -> complex:
        # since b>0, log(b) is real; exp(zc*log(b)) is well-defined
        return np.exp(zc * np.log(b))

    # build cf with guards
    cf[0] = safe_logc(1j * t1**2 + 1.0)
    cf[1] = np.exp(1j * t1 * t2) + 1.0
    cf[2] = np.sin(t1 * t2) + t2
    cf[3] = np.cos(t1**3 + t2**2) * 1j
    cf[4] = t1 * t2 * (t1 - t2)
    cf[5] = np.sqrt(np.abs(t1 * t2)) * (t1 + t2)
    cf[6] = (t1**3) * (t2**3) * 1j
    cf[7] = 1j * safe_divc((t2 - t1), (t1 + t2))           # <-- guarded division
    cf[8] = safe_logc(t1 * t2) + 1j * np.sin(t1 + t2)      # <-- guarded log
    cf[9] = pow_real_base_complex_exp(3.0, t1) * pow_real_base_complex_exp(2.0, t2)

    return cf.astype(np.complex128)

ALLOWED["g58"]=g58

def g59(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(11, dtype=np.complex128)

    EPS = 1e-15

    # precompute a few reused things (Numba likes explicit scalars)
    t1r, t1i = t1.real, t1.imag
    t2r, t2i = t2.real, t2.imag

    prod = t1 * t2
    prod_abs = np.abs(prod)  # scalar float

    # 0) 100 * (t1**4 - t2**4)
    cf[0] = 100.0 * (t1**4 - t2**4)

    # 1) -100j * (t1 * t2 * (t1**2 + t2**2))
    cf[1] = -100.0j * (t1 * t2 * (t1**2 + t2**2))

    # 2) 100 * sqrt( Re( (t1**2 * t2**2) ) ), but clamp to >= 0
    v2 = (t1 * t1 * t2 * t2).real
    if not np.isfinite(v2) or v2 < 0.0:
        v2 = 0.0
    cf[2] = 100.0 * np.sqrt(v2)

    # 3) 100 * (t1 - t2).imag
    cf[3] = 100.0 * (t1i - t2i)

    # 4) 100j * (t1**2 + t2**2)
    cf[4] = 100.0j * (t1**2 + t2**2)

    # 5) 100 * exp(i * |t1 - t2|)  -> 100 * (cos(a) + i sin(a)), a real
    a5 = np.abs(t1 - t2)
    cf[5] = 100.0 * (np.cos(a5) + 1j * np.sin(a5))

    # 6) 100 * sin( (t1 + t2).real )
    cf[6] = 100.0 * np.sin(t1r + t2r)

    # 7) -100j * cos( (t1 - t2).imag )
    cf[7] = -100.0j * np.cos(t1i - t2i)

    # 8) 100 * ( (t1*t2) / |t1*t2| )  with guard for |.| ~ 0 or non-finite
    if np.isfinite(prod_abs) and prod_abs > EPS:
        cf[8] = 100.0 * (prod / prod_abs)
    else:
        cf[8] = 0.0 + 0.0j

    # 9) 100 * sqrt(t1.real * t2.real) - 100j * sqrt(t1.imag * t2.imag)
    rpart = t1r * t2r
    ipart = t1i * t2i
    if not np.isfinite(rpart) or rpart < 0.0:
        rpart = 0.0
    if not np.isfinite(ipart) or ipart < 0.0:
        ipart = 0.0
    cf[9] = 100.0 * np.sqrt(rpart) - 100.0j * np.sqrt(ipart)

    # 10) 100 * exp( i * (angle(t1) - angle(t2)) )
    #      angle(z) = atan2(imag, real)
    ang1 = np.arctan2(t1i, t1r)
    ang2 = np.arctan2(t2i, t2r)
    dphi = ang1 - ang2
    cf[10] = 100.0 * (np.cos(dphi) + 1j * np.sin(dphi))

    return cf
    
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

def g67(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)

    EPS = 1e-15

    t1r, t1i = t1.real, t1.imag
    t2r, t2i = t2.real, t2.imag

    # 0..3
    cf[0] = t1 + t2
    cf[1] = t1 * t2
    cf[2] = (t1 + t2) * (t1 + t2)
    cf[3] = np.abs(t1) * np.abs(t2)

    # 4..8 : r / |r| with guard
    for k in range(4, 9):
        r = (t1 + 1j * t2) ** k
        mag = np.abs(r)
        if np.isfinite(mag) and mag > EPS:
            cf[k] = r / mag
        else:
            cf[k] = 0.0 + 0.0j

    # 9 : log(|t1|)/log(|t2|) with guards
    a1 = np.abs(t1)
    a2 = np.abs(t2)
    if not np.isfinite(a1) or a1 <= EPS:
        ln1 = 0.0
    else:
        ln1 = np.log(a1)
    if not np.isfinite(a2) or a2 <= EPS:
        ln2 = 0.0
    else:
        ln2 = np.log(a2)
    if np.abs(ln2) > EPS and np.isfinite(ln1) and np.isfinite(ln2):
        cf[9] = ln1 / ln2
    else:
        cf[9] = 0.0 + 0.0j

    # 10 : exp(angle(t1) + angle(t2))  (real, positive)
    ang1 = np.arctan2(t1i, t1r)
    ang2 = np.arctan2(t2i, t2r)
    cf[10] = np.exp(ang1 + ang2)

    # 11..15 : (conj(t1)/conj(t2)) * (1j)^(k-2) with guard on denom
    den = np.conj(t2)
    denmag = np.abs(den)
    if np.isfinite(denmag) and denmag > EPS:
        base = np.conj(t1) / den
    else:
        base = 0.0 + 0.0j
    for k in range(11, 16):
        p = k - 2
        # 1j**p using p mod 4 to avoid pow edge cases
        m = p & 3
        if m == 0:
            factor = 1.0 + 0.0j
        elif m == 1:
            factor = 0.0 + 1.0j
        elif m == 2:
            factor = -1.0 + 0.0j
        else:
            factor = 0.0 - 1.0j
        cf[k] = base * factor

    # 16..17
    cf[16] = t1r * t2i
    cf[17] = t1i * t2r

    # 18..22 : sin(angle( (t1 + i t2)^k ))
    for k in range(18, 23):
        zz = (t1 + 1j * t2) ** k
        angz = np.arctan2(zz.imag, zz.real)
        cf[k] = np.sin(angz)

    # 23 : cos(angle(t1) + angle(t2))
    cf[23] = np.cos(ang1 + ang2)

    # 24 : tanh(|t1*t2|)
    prod_abs = np.abs(t1 * t2)
    if np.isfinite(prod_abs):
        cf[24] = np.tanh(prod_abs)
    else:
        cf[24] = 0.0 + 0.0j

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

def g69(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)

    EPS = 1e-15

    # Precompute common scalars
    s = t1 + t2
    s_abs = np.abs(s)
    s_den = s_abs + 1.0
    if not np.isfinite(s_den):
        s_den = 1.0

    # unit phasor for angle(s): exp(i * angle(s)) = cos(theta) + i sin(theta)
    theta = np.arctan2(s.imag, s.real)
    phasor = np.cos(theta) + 1j * np.sin(theta)

    # 0..24 baseline fill
    for i in range(25):
        # (t1.real)^(i+1) + (t2.imag)^(25-i) — integer exponents -> safe
        rpart = t1.real
        ipart = t2.imag
        num = (rpart ** (i + 1)) + (ipart ** (25 - i))
        # divide by (1 + |t1+t2|) and multiply by unit phasor
        cf[i] = (num / s_den) * phasor

    # cf[2] = 3 * conj(t1**2 + t2)
    cf[2] = 3.0 * np.conj(t1 * t1 + t2)

    # cf[6] = 7 * |t1 + t2|
    cf[6] = 7.0 * s_abs

    # cf[10] = 11 * (t1/t2 + conj(t2/t1)) with guards
    term1 = 0.0 + 0.0j
    term2 = 0.0 + 0.0j
    a2 = np.abs(t2)
    a1 = np.abs(t1)
    if np.isfinite(a2) and a2 > EPS:
        term1 = t1 / t2
    if np.isfinite(a1) and a1 > EPS:
        term2 = np.conj(t2 / t1)
    cf[10] = 11.0 * (term1 + term2)

    # cf[16] = 17 * (|t1| |t2|) / |t1 + t2|^2  with guard
    denom16 = s_abs * s_abs
    if np.isfinite(denom16) and denom16 > EPS:
        cf[16] = 17.0 * ((np.abs(t1) * np.abs(t2)) / denom16)
    else:
        cf[16] = 0.0 + 0.0j

    # cf[22] = 23 * (conj(t1) + t2) / (1 + |t1 * conj(t2)|)
    prod12 = t1 * np.conj(t2)
    den22 = 1.0 + np.abs(prod12)  # real, >= 1
    if not np.isfinite(den22) or den22 <= 0.0:
        den22 = 1.0
    cf[22] = 23.0 * ((np.conj(t1) + t2) / den22)

    # cf[24] = 25 * (conj(t1) + conj(t2)) / |t1 * t2|  with guard
    prod = t1 * t2
    prod_abs = np.abs(prod)
    if np.isfinite(prod_abs) and prod_abs > EPS:
        cf[24] = 25.0 * ((np.conj(t1) + np.conj(t2)) / prod_abs)
    else:
        cf[24] = 0.0 + 0.0j

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

def g81(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)


    def _safe_angle(z: complex) -> float:
        # arg(z); returns 0 if z is (near) zero or non-finite
        zr, zi = z.real, z.imag
        if not (np.isfinite(zr) and np.isfinite(zi)):
            return 0.0
        if zr == 0.0 and zi == 0.0:
            return 0.0
        return np.arctan2(zi, zr)


    def _unit_phasor(z: complex) -> complex:
        # exp(i*angle(z)); 0 if z is (near) zero or non-finite
        theta = _safe_angle(z)
        return np.cos(theta) + 1j * np.sin(theta)


    def _cpow_real(z: complex, a: float) -> complex:
        # z**a for real a, via polar form; returns 0 if z≈0 or non-finite
        zr, zi = z.real, z.imag
        if not (np.isfinite(zr) and np.isfinite(zi)):
            return 0.0 + 0.0j
        r = np.hypot(zr, zi)  # |z|
        if r == 0.0:
            return 0.0 + 0.0j
        theta = np.arctan2(zi, zr)
        # r^a * exp(i * a * theta)
        ra = np.exp(a * np.log(r))
        ang = a * theta
        return ra * (np.cos(ang) + 1j * np.sin(ang))

    # Precompute for loops
    for i in range(25):
        k1 = float(i + 1)
        # numerator: t1*(i+1) + t2^((i+1)/2)
        term1 = t1 * k1
        term2 = _cpow_real(t2, 0.5 * k1)  # handles half-integers safely
        num = term1 + term2

        # denominator: 1 + | t1*(i+1) + t2^(i+1) |
        # note: integer power is fine under numba
        pow_int = t2
        # fast integer power: repeated multiply (avoid Python pow edge cases)
        # but for small range, ** is OK in numba; still we can do it explicitly
        pow_int = t2 ** (i + 1)
        den = 1.0 + np.abs(term1 + pow_int)

        # unit phasor exp(i*angle(t1+t2))
        ph = _unit_phasor(t1 + t2)

        # safe division (den >= 1 so no zero)
        cf[i] = (num / den) * ph

    # cf[2] = Re(t1) + Im(t2)
    cf[2] = (t1.real + t2.imag) + 0.0j

    # cf[6] = |exp(i*angle(t1*t2))| -> 1 if t1*t2 != 0 and finite, else 0
    prod = t1 * t2
    if (prod.real != 0.0 or prod.imag != 0.0) and np.isfinite(prod.real) and np.isfinite(prod.imag):
        cf[6] = 1.0 + 0.0j
    else:
        cf[6] = 0.0 + 0.0j

    # cf[10] = Re(t1*t2) + Im(t1/t2)
    # guard t1/t2 if t2≈0
    cf[10] = (t1 * t2).real + 0.0j
    a2 = np.abs(t2)
    if np.isfinite(a2) and a2 > 0.0:
        q = t1 / t2
        cf[10] += (q.imag + 0.0j)

    # cf[12] = angle(t1 + 4*t2) / |conj(t1 - 4*t2)| = angle(s1)/|t1-4t2|
    s1 = t1 + 4.0 * t2
    s2 = t1 - 4.0 * t2
    ang = _safe_angle(s1)
    den12 = np.abs(s2)
    if np.isfinite(den12) and den12 > 0.0:
        cf[12] = (ang / den12) + 0.0j
    else:
        cf[12] = 0.0 + 0.0j

    # cf[16] = |exp(i*angle(t1 - t2))| -> 1 if finite & nonzero, else 0
    diff = t1 - t2
    if (diff.real != 0.0 or diff.imag != 0.0) and np.isfinite(diff.real) and np.isfinite(diff.imag):
        cf[16] = 1.0 + 0.0j
    else:
        cf[16] = 0.0 + 0.0j

    # cf[18] = Re(t1/t2) - Im(t1*t2)
    if np.isfinite(a2) and a2 > 0.0:
        q = t1 / t2
        cf[18] = (q.real - (t1 * t2).imag) + 0.0j
    else:
        cf[18] = (0.0 - (t1 * t2).imag) + 0.0j

    # cf[22] = |exp(i*angle(t1 + t2))| -> 1 if finite & nonzero, else 0
    s = t1 + t2
    if (s.real != 0.0 or s.imag != 0.0) and np.isfinite(s.real) and np.isfinite(s.imag):
        cf[22] = 1.0 + 0.0j
    else:
        cf[22] = 0.0 + 0.0j

    return cf

ALLOWED["g81"]=g81

def g82(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)

    # ---- helpers & guards ----
    EPS_DEN   = 1e-12     # division guard
    LOG_EPS   = 1e-300    # log guard
    MAG_CAP   = 1e300     # absolute cap to avoid LAPACK nuking
    TAN_EPS   = 1e-8      # cos proximity guard for tan

    def cabs(w):
        return math.hypot(w.real, w.imag)

    def finite_c(w):
        return math.isfinite(w.real) and math.isfinite(w.imag)

    def safe_log_pos(x):          # x is float
        if not math.isfinite(x) or x <= 0.0:
            return None
        return math.log(x)

    def safe_add_real(idx, val):  # add real to cf[idx], bail on non-finite
        if not math.isfinite(val):
            return False
        v = complex(val, 0.0)
        nv = cf[idx] + v
        if not finite_c(nv):
            return False
        if abs(nv.real) > MAG_CAP or abs(nv.imag) > MAG_CAP:
            return False
        cf[idx] = nv
        return True

    # ---- recurrence core ----
    cf[0] = t1 + t2
    if not finite_c(cf[0]):
        return np.zeros(25, dtype=np.complex128)

    denom = t2 + 1.0
    dmag = cabs(denom)
    if dmag < EPS_DEN or not finite_c(denom):
        return np.zeros(25, dtype=np.complex128)  # bail early

    for k in range(1, 25):
        num = 1j * (cf[k - 1] * cf[k - 1]) + (2.0j * t1)
        if not finite_c(num):
            return np.zeros(25, dtype=np.complex128)
        v = num / denom
        if not finite_c(v):
            return np.zeros(25, dtype=np.complex128)
        mag = cabs(v)
        if mag <= 0.0 or not math.isfinite(mag):
            cf[k] = 1.0 + 0.0j
        else:
            cf[k] = v
        # magnitude cap
        if abs(cf[k].real) > MAG_CAP or abs(cf[k].imag) > MAG_CAP:
            return np.zeros(25, dtype=np.complex128)

    # ---- scalar magnitudes for logs ----
    r1 = cabs(t1)
    r2 = cabs(t2)
    r12 = cabs(t1 * t2)
    if not (math.isfinite(r1) and math.isfinite(r2) and math.isfinite(r12)):
        return np.zeros(25, dtype=np.complex128)

    # cf[4] += sin(Re cf[1]) + log(|t1|)
    add4 = math.sin(cf[1].real)
    log1 = safe_log_pos(max(r1, LOG_EPS))
    if log1 is None:
        return np.zeros(25, dtype=np.complex128)
    if not safe_add_real(4, add4 + log1):
        return np.zeros(25, dtype=np.complex128)

    # cf[9] += cos(Im cf[4]) + log(|t2|)
    add9 = math.cos(cf[4].imag)
    log2 = safe_log_pos(max(r2, LOG_EPS))
    if log2 is None:
        return np.zeros(25, dtype=np.complex128)
    if not safe_add_real(9, add9 + log2):
        return np.zeros(25, dtype=np.complex128)

    # cf[14] += tan(Re cf[9]) + log(|t1*t2|)
    x = cf[9].real
    # tan safety: tan(x)=sin/ cos; guard cos≈0
    c = math.cos(x)
    s = math.sin(x)
    if abs(c) < TAN_EPS or not (math.isfinite(s) and math.isfinite(c)):
        return np.zeros(25, dtype=np.complex128)
    tanx = s / c
    log12 = safe_log_pos(max(r12, LOG_EPS))
    if log12 is None:
        return np.zeros(25, dtype=np.complex128)
    if not safe_add_real(14, tanx + log12):
        return np.zeros(25, dtype=np.complex128)

    # cf[19] += arctan(Im cf[14]) + log(|conj(t1)*t2|) == log(|t1|*|t2|)
    add19 = math.atan(cf[14].imag)
    logc = safe_log_pos(max(r1 * r2, LOG_EPS))
    if logc is None:
        return np.zeros(25, dtype=np.complex128)
    if not safe_add_real(19, add19 + logc):
        return np.zeros(25, dtype=np.complex128)

    # cf[24] += sin(angle(cf[19])) + log(|conj(t1)*conj(t2)|) == log(|t1|*|t2|)
    ang = math.atan2(cf[19].imag, cf[19].real)
    add24 = math.sin(ang)
    if not math.isfinite(add24):
        return np.zeros(25, dtype=np.complex128)
    if not safe_add_real(24, add24 + logc):
        return np.zeros(25, dtype=np.complex128)

    # final sanity: all entries finite & within caps
    for i in range(25):
        if not finite_c(cf[i]):
            return np.zeros(25, dtype=np.complex128)
        if abs(cf[i].real) > MAG_CAP or abs(cf[i].imag) > MAG_CAP:
            return np.zeros(25, dtype=np.complex128)

    return cf

ALLOWED["g82"]=g82

def g83(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    EPS = 1e-15

    # ----- helpers (inline-friendly for njit) -----
    def angle(zc: complex) -> float:
        # safe arg(z)
        return np.arctan2(zc.imag, zc.real)

    def unit_norm(zc: complex) -> complex:
        m = np.abs(zc)
        if np.isfinite(m) and m > EPS:
            return zc / m
        return 0.0 + 0.0j

    # ----- seeds -----
    # cf[0] = t1.real * t2.real + t1.imag * t2.imag   (real)
    cf[0] = (t1.real * t2.real + t1.imag * t2.imag) + 0.0j

    # cf[1] = |t1| * |t2|   (real, >=0)
    cf[1] = (np.abs(t1) * np.abs(t2)) + 0.0j

    # cf[2] = angle(t1) + angle(t2)   (real)
    cf[2] = (angle(t1) + angle(t2)) + 0.0j

    # cf[3] = conj(t1).real + conj(t2).imag
    # conj(t1).real == t1.real ; conj(t2).imag == -t2.imag
    cf[3] = (t1.real + (-t2.imag)) + 0.0j

    # ----- recurrence k = 4..24 -----
    for k in range(4, 25):
        num = cf[k - 1] * cf[k - 4] + cf[k - 3] * cf[k - 2]
        cf[k] = unit_norm(num)

    # ----- post adjustments -----
    # cf[12] = cf[0]/cf[1] + cf[2]/cf[3]   (guard small denominators)
    c1 = 0.0 + 0.0j
    c2 = 0.0 + 0.0j
    m1 = np.abs(cf[1])
    if np.isfinite(m1) and m1 > EPS:
        c1 = cf[0] / cf[1]
    m3 = np.abs(cf[3])
    if np.isfinite(m3) and m3 > EPS:
        c2 = cf[2] / cf[3]
    cf[12] = c1 + c2

    # cf[18] = log(|cf[0]*cf[1]| + 1) / log(|cf[2]*cf[3]| + 1)
    num18 = np.abs(cf[0] * cf[1]) + 1.0
    den18 = np.abs(cf[2] * cf[3]) + 1.0
    # num18, den18 are real >= 1, so logs are >= 0
    ln_num = np.log(num18)
    ln_den = np.log(den18)
    if np.isfinite(ln_den) and ln_den > EPS:
        cf[18] = (ln_num / ln_den) + 0.0j
    else:
        cf[18] = 0.0 + 0.0j

    # cf[24] = cf[4]*cf[9]*cf[14]*cf[19] normalized
    prod = cf[4] * cf[9] * cf[14] * cf[19]
    cf[24] = unit_norm(prod)

    return cf
    
ALLOWED["g83"]=g83

def g84(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)

    EPS = 1e-15

    # constants (avoid building arrays repeatedly inside njit)
    p0 = 2.0;  p1 = 3.0;  p2 = 5.0;  p3 = 7.0;  p4 = 11.0;  p5 = 13.0;  p6 = 17.0
    sum_first5 = p0 + p1 + p2 + p3 + p4        # 2+3+5+7+11 = 28
    prod_first4 = p0 * p1 * p2 * p3            # 2*3*5*7 = 210

    s = t1 + t2

    # cf[0:7] = primes * (t1 + t2)
    cf[0] = p0 * s
    cf[1] = p1 * s
    cf[2] = p2 * s
    cf[3] = p3 * s
    cf[4] = p4 * s
    cf[5] = p5 * s
    cf[6] = p6 * s

    # cf[14] = sum(primes[:5]) + (t1 + 1j*t2)^2
    cf[14] = sum_first5 + (t1 + 1j * t2) * (t1 + 1j * t2)

    # cf[24] = prod(primes[:4]) / |t1 + 1j*t2|   (guard denom)
    d24 = np.abs(t1 + 1j * t2)
    if np.isfinite(d24) and d24 > EPS:
        cf[24] = prod_first4 / d24
    else:
        cf[24] = 0.0 + 0.0j

    # precompute stable logs: log(|t1|), log(|t2|) with guards
    a1 = np.abs(t1)
    a2 = np.abs(t2)
    if np.isfinite(a1) and a1 > EPS:
        ln1 = np.log(a1)
    else:
        ln1 = 0.0
    if np.isfinite(a2) and a2 > EPS:
        ln2 = np.log(a2)
    else:
        ln2 = 0.0

    # for k in 7..13:
    # v = ln|t1| * sin(cf[k-1]) + ln|t2| * cos(cf[k-1])
    # assign only if finite
    for k in range(7, 14):
        x = cf[k - 1]
        sv = np.sin(x)   # complex-safe in numba
        cv = np.cos(x)
        v = ln1 * sv + ln2 * cv
        # finiteness check for complex
        if np.isfinite(v.real) and np.isfinite(v.imag):
            cf[k] = v
        # else leave as 0

    # cf[15:24] = Re(t1) + Im(t2)   (broadcast a complex scalar)
    baseline = (t1.real + t2.imag) + 0.0j
    for i in range(15, 24):
        cf[i] = baseline

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

def g92(z, a, state):
    # Use real scalars: r = Re(z0), s = Im(z0)
    r = z[0].real
    s = z[0].imag

    cf = np.zeros(71, dtype=np.complex128)

    # Precompute base terms (real scalars)
    sin_r = math.sin(r)
    cos_r = math.cos(r)
    cos_s = math.cos(s)
    sin_s = math.sin(s)
    log_base = math.log(abs(r + s) + 1.0)  # >= 0

    # Main series
    for i in range(71):
        # integer powers (safe for i>=0)
        cos_s_i = 1.0
        sin_s_i = 1.0
        log_i   = 1.0
        for _ in range(i):
            cos_s_i *= cos_s
            sin_s_i *= sin_s
            log_i   *= log_base

        val = sin_r * cos_s_i + cos_r * sin_s_i + log_i
        cf[i] = complex(val, 0.0)

    # Helpers for "angle" of a real number: atan2(0, x) -> 0 if x>=0 else pi
    def angle_real(x):
        return math.atan2(0.0, x)

    # Guards for division by zero
    abs_r = abs(r)
    abs_s = abs(s)

    # cf[0] += 2*r*s  (since original tried mixed real/imag cross terms)
    cf[0] = cf[0] + complex(2.0 * r * s, 0.0)

    # cf[1] += |r| * |s|
    cf[1] = cf[1] + complex(abs_r * abs_s, 0.0)

    # cf[2] += angle(r) / angle(s)
    ar = angle_real(r)
    as_ = angle_real(s)
    if as_ != 0.0:
        cf[2] = cf[2] + complex(ar / as_, 0.0)
    else:
        # if denominator angle is 0 (e.g., s >= 0), add 0 to stay safe
        cf[2] = cf[2] + 0.0

    # cf[3] += angle(s) / angle(r)
    if ar != 0.0:
        cf[3] = cf[3] + complex(as_ / ar, 0.0)
    else:
        cf[3] = cf[3] + 0.0

    # cf[4] += r / |s|
    if abs_s > 0.0:
        cf[4] = cf[4] + complex(r / abs_s, 0.0)

    # cf[5] += s / |r|
    if abs_r > 0.0:
        cf[5] = cf[5] + complex(s / abs_r, 0.0)

    # cf[70] += conjugate(r+s) == (r+s) (real)
    cf[70] = cf[70] + complex(r + s, 0.0)

    return cf

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

def g95(z, a, state):
    # Use complex inputs directly
    t1 = z[0]
    t2 = z[1]

    cf = np.zeros(71, dtype=np.complex128)
    EPS = 1e-15

    # cf[0] = t1.real * t2.imag  (real -> complex)
    cf[0] = (t1.real * t2.imag) + 0.0j

    # cf[1] = |t1| / angle(t2)  with guard
    ang_t2 = 0.0
    if (t2.real != 0.0 or t2.imag != 0.0) and np.isfinite(t2.real) and np.isfinite(t2.imag):
        ang_t2 = np.arctan2(t2.imag, t2.real)
    a1 = np.abs(t1)
    if not np.isfinite(a1):  # defensive
        a1 = 0.0
    if np.isfinite(ang_t2) and np.abs(ang_t2) > EPS:
        cf[1] = (a1 / ang_t2) + 0.0j
    else:
        cf[1] = 0.0 + 0.0j

    # cf[2] = sin(t1) * cos(t2)  (Numba supports complex sin/cos)
    cf[2] = np.sin(t1) * np.cos(t2)

    # Recurrence
    for i in range(3, 71):
        # log(|cf[i-2] + 1|)
        base = cf[i - 2] + (1.0 + 0.0j)
        mag = np.abs(base)
        if np.isfinite(mag) and mag > 0.0:
            lnmag = np.log(mag)
        else:
            lnmag = 0.0

        cf[i] = cf[i - 1] + cf[i - 3] * lnmag

        if (i & 1) == 0:
            # even: + |cf[i-1] * t1|
            term = np.abs(cf[i - 1] * t1)
            cf[i] = cf[i] + term
        else:
            # odd:  - |cf[i-1] * conj(t2)|
            term = np.abs(cf[i - 1] * np.conj(t2))
            cf[i] = cf[i] - term

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

def g97(z, a, state):
    # As in your original: take real & imag from z[0]
    t1 = z[0].real  # float64
    t2 = z[0].imag  # float64

    cf = np.zeros(71, dtype=np.complex128)
    EPS = 1e-15

    # cf[0] = t1*t2*|t1|*|t2| + log(|t1|+1) + log(|t2|+1)
    a1 = abs(t1)
    a2 = abs(t2)
    ln1 = np.log(a1 + 1.0)
    ln2 = np.log(a2 + 1.0)
    cf[0] = (t1 * t2 * a1 * a2 + ln1 + ln2) + 0.0j

    # cf[1:3] = (t1 + t2) * [1, -1]
    s = t1 + t2
    cf[1] = ( s) + 0.0j
    cf[2] = (-s) + 0.0j

    # Common guarded denominator for the next two loops
    # Your original used (i/t1/t2).real or .imag; since t1,t2 are real,
    # we use denom = i/(t1*t2) with an EPS guard (same for both loops).
    tt = t1 * t2
    # Loop 3..9
    for i in range(3, 10):
        x = i * tt
        # denom = i/(t1*t2)
        if abs(tt) > EPS:
            denom = i / tt
        else:
            denom = 1.0  # safe fallback
        val = np.sin(x) / denom
        cf[i] = val + 0.0j

    # Loop 10..19
    for i in range(10, 20):
        x = i * tt
        if abs(tt) > EPS:
            denom = i / tt
        else:
            denom = 1.0
        val = np.cos(x) / denom
        cf[i] = val + 0.0j

    # 20..29: log( |t1**i + t2**i| + 1 )
    # (t1,t2 are real so these powers are real)
    for i in range(20, 30):
        p = t1 ** i + t2 ** i
        mag = abs(p)
        cf[i] = np.log(mag + 1.0) + 0.0j

    # 30..39: (t1 + t2) ** k, k = 1..10
    base = t1 + t2
    pk = base  # k=1
    for k in range(10):
        if k > 0:
            pk *= base  # iterative power
        cf[30 + k] = pk + 0.0j

    # 40..49: |i/t1| * |i/t2|
    for i in range(40, 50):
        if abs(t1) > EPS:
            a = abs(i / t1)
        else:
            a = 0.0
        if abs(t2) > EPS:
            b = abs(i / t2)
        else:
            b = 0.0
        cf[i] = (a * b) + 0.0j

    # Accumulate v over cf[0..29] with alternating sign
    v = 1.0 + 0.0j
    for i in range(30):
        sgn = -1.0 if (i & 1) == 0 else 1.0
        v *= cf[i] * sgn

    # 50..59: constant v
    for i in range(50, 60):
        cf[i] = v

    # 60..69:
    # |t1|**i - |t2|**i + log(|t1**i + t2**i| + 1) + angle(cf[i-10])
    for i in range(60, 70):
        # powers
        p1 = a1 ** i
        p2 = a2 ** i
        # log term
        p = (t1 ** i) + (t2 ** i)
        mag = abs(p)
        ln = np.log(mag + 1.0)
        # angle of cf[i-10]
        w = cf[i - 10]
        theta = np.arctan2(w.imag, w.real)
        cf[i] = (p1 - p2 + ln + theta) + 0.0j

    # cf[70] = conj(sum(cf[:70]))
    ssum = 0.0 + 0.0j
    for i in range(70):
        ssum += cf[i]
    cf[70] = np.conj(ssum)

    return cf

ALLOWED["g97"]=g97

def g98(z, a, state):
    # As in your original: take real & imag from z[0]
    t1 = z[0].real  # float64
    t2 = z[0].imag  # float64

    cf = np.zeros(71, dtype=np.complex128)
    EPS = 1e-15

    a1 = abs(t1)
    a2 = abs(t2)
    cf[0] = (np.log(a1 + 1.0) + np.log(a2 + 1.0)) + 0.0j

    # angle(t1 * t2) with arctan2 on a purely real number
    prod = t1 * t2
    theta = np.arctan2(0.0, prod)  # 0 if prod>0, pi if prod<0, 0 if prod==0
    cos_const = np.cos(a1 + a2)

    # 1..70
    for i in range(1, 71):
        # sin(i * angle(t1*t2)) * |cf[i-1]**(i-1)| * cos(|t1| + |t2|)
        # (i-1) is an integer exponent; Numba handles complex**int, but cf[i-1] may be complex.
        # We only need the magnitude.
        pow_mag = np.abs(cf[i - 1] ** (i - 1))
        cf[i] = (np.sin(i * theta) * pow_mag * cos_const) + 0.0j

    # 34..54: cf[i] *= |t1 + t2| * cf[i-1]
    s_abs = abs(t1 + t2)
    for i in range(34, 55):
        cf[i] = cf[i] * s_abs * cf[i - 1]

    # 4..25: cf[i] *= cf[0] * (|t1|*|t2|)^(i-3)
    base = a1 * a2
    for i in range(4, 26):
        k = i - 3  # 1..22
        cf[i] = cf[i] * cf[0] * (base ** k)

    # 55..70: cf[i] *= cf[0] / (|t1|*|t2|)^(15 - (i-55))
    # exponents go 15,14,...,1 across indices 55..70
    for i in range(55, 71):
        k = 15 - (i - 55)
        if base > EPS:
            cf[i] = cf[i] * cf[0] / (base ** k)
        else:
            # if base ~ 0, original would blow up; set to 0 for safety
            cf[i] = 0.0 + 0.0j

    # Add conjugate(t1*t2) to selected positions (no fancy indexing)
    addv = np.conj(prod + 0.0j)
    cf[0]  = cf[0]  + addv
    cf[14] = cf[14] + addv
    cf[29] = cf[29] + addv
    cf[44] = cf[44] + addv
    cf[59] = cf[59] + addv
    cf[70] = cf[70] + addv

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

def gp7f(z,a,state):
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
    
ALLOWED["gp7f"]=gp7f

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

def p13(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(51, dtype=np.complex128)

    # Fibonacci (int64 is fine under njit)
    fib = np.array([1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377,
                    610, 987, 1597, 2584, 4181], dtype=np.int64)

    # Angles (avoid np.angle)
    ang1 = math.atan2(t1.imag, t1.real)
    ang2 = math.atan2(t2.imag, t2.real)

    # First two 19-length blocks
    for n in range(19):
        f = float(fib[n])
        cf[n]      = f * t1 * math.cos(ang2)
        cf[n + 19] = f * t1 * math.sin(ang2)

    # Last block must fit indices 38..50 (13 elems)
    for n in range(13):  # 0..12 -> indices 38..50
        f = float(fib[n])
        cf[n + 38] = f * t2 * math.sin(ang1)

    # Magnitude and logs (use hypot + log)
    prod = t1 * t2
    prod_mag = math.hypot(prod.real, prod.imag)
    cf[19] = np.complex128(prod_mag)              # overwrites earlier cf[19]
    cf[49] = np.complex128(math.log(prod_mag + 1.0))
    cf[50] = np.complex128(t1.real + t2.imag)

    return cf

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

def p53(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)

    COSH_Y_MAX = 20.0      # guard hyperbolics
    ABS_MAX    = 1e300     # clamp to avoid overflow in downstream
    LOG_EPS    = 0.0       # we use log(x+1) with x>=0, so fine

    def finite2(x, y):
        return math.isfinite(x) and math.isfinite(y)

    def abs_c(w):
        r = math.hypot(w.real, w.imag)
        if not math.isfinite(r):
            return None
        return r

    def sin_c(w):
        x, y = w.real, w.imag
        if abs(y) > COSH_Y_MAX:
            return None
        sx = math.sin(x); cx = math.cos(x)
        ch = math.cosh(y); sh = math.sinh(y)
        re = sx * ch
        im = cx * sh
        if not finite2(re, im):
            return None
        return np.complex128(re + 1j * im)

    def cos_c(w):
        x, y = w.real, w.imag
        if abs(y) > COSH_Y_MAX:
            return None
        cx = math.cos(x); sx = math.sin(x)
        ch = math.cosh(y); sh = math.sinh(y)
        re = cx * ch
        im = -sx * sh
        if not finite2(re, im):
            return None
        return np.complex128(re + 1j * im)

    def angle_c(w):
        # returns real angle; guard NaNs
        if not finite2(w.real, w.imag):
            return None
        return math.atan2(w.imag, w.real)

    # seeds
    cf0 = t1 * t2
    if not finite2(cf0.real, cf0.imag):
        return cf
    cf[0] = cf0

    cf1 = 1j * (t1 + t2)
    if not finite2(cf1.real, cf1.imag):
        return cf
    cf[1] = cf1

    cf[2] = np.complex128(t1.real + t2.imag)

    s = sin_c(cf[0]); c = cos_c(cf[1])
    if s is None or c is None:
        return cf
    v3 = s * c
    if not finite2(v3.real, v3.imag):
        return cf
    cf[3] = v3

    a04 = abs_c(t1 - t2)
    if a04 is None:
        return cf
    if a04 > ABS_MAX:
        return cf
    cf[4] = a04 + 0.0j 

    # first recurrence: k = 6..36 -> indices 5..35
    for k in range(6, 37):
        a1 = abs_c(cf[k - 2])
        a2 = abs_c(cf[k - 3])
        if a1 is None or a2 is None:
            return cf
        # clamp squares to avoid overflow
        a1_sq = a1 * a1
        if a1_sq > ABS_MAX:
            return cf
        val = a1_sq - math.log(a2 + 1.0)
        if not math.isfinite(val):
            return cf
        cf[k - 1] = np.complex128(val)

    # second recurrence: k = 36..71 -> indices 35..70
    for k in range(36, 72):
        p = cf[k - 2] * cf[k - 4]
        if not finite2(p.real, p.imag):
            return cf
        ang = angle_c(p)
        if ang is None or not math.isfinite(ang):
            return cf
        q = cf[k - 3] * cf[k - 6]
        if not finite2(q.real, q.imag):
            return cf
        mag = abs_c(q)
        if mag is None or mag > ABS_MAX:
            return cf
        val = ang * mag
        if not math.isfinite(val):
            return cf
        cf[k - 1] = np.complex128(val)

    # final entry
    cross = t1.real * t2.imag - t2.real * t1.imag
    if not math.isfinite(cross):
        return cf
    cf[70] = np.complex128(cross)

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

def p62(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)

    # seed
    cf[0] = t1**5 + t2**5

    # build cf[1..70]
    for i in range(2, 72):  # i -> writes cf[i-1], reads cf[i-2]
        prev = cf[i - 2]
        sq = prev * prev
        MAG_CAP = 1e200
        if abs(sq.real) > MAG_CAP or abs(sq.imag) > MAG_CAP:
            return np.zeros(71, dtype=np.complex128)
        if (i % 2) == 0:
            cf[i - 1] = i * sq
        else:
            cf[i - 1] = i * sq * (1.0 + 0.1 * t2)

    # tweaks
    cf[0] = cf[0] + 2.0 * cf[1]
    cf[1] = cf[1] - 3.0 * cf[2]

    # safe range so i+2 ≤ 70
    for i in range(3, 69):  # writes cf[3..68], reads cf[i+1], cf[i+2]
        cf[i] = cf[i] + cf[i + 1] - cf[i + 2]

    cf[69] = cf[69] + cf[70]

    # final element: replace np.abs / np.angle with hypot / atan2
    r1 = math.hypot(t1.real, t1.imag)
    r2 = math.hypot(t2.real, t2.imag)
    ang2 = math.atan2(t2.imag, t2.real)
    val70 = (r1 * r1) - (r2 * r2) + 2.0 * t1.imag * t2.imag - ang2
    cf[70] = complex(val70, 0.0)

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

def p86(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)

    EPS = 1e-300  # reciprocal guard

    # 1) Fill: cf[k-1] = (cos(k*Re(t1)))^k + i*(sin(k*Re(t2)))^k
    # (bounded inputs ⇒ |cos|,|sin| ≤ 1 ⇒ integer powers well-behaved)
    for k in range(1, 72):
        c = math.cos(k * t1.real)
        s = math.sin(k * t2.real)

        # integer powers by repeated multiply (Numba-safe)
        pc = 1.0
        ps = 1.0
        for _ in range(k):
            pc *= c
            ps *= s

        val = complex(pc, ps) * (0.0 + 1.0j)  # i * s^k in imag part
        # We want: (cos^k) + i*(sin^k)
        cf[k - 1] = complex(pc, 0.0) + complex(0.0, ps)

        if not (math.isfinite(cf[k - 1].real) and math.isfinite(cf[k - 1].imag)):
            return np.zeros(71, dtype=np.complex128)

    # 2) cf[1::2] **= -1  -> reciprocal for odd indices (1,3,...,69)
    for i in range(1, 71, 2):
        v = cf[i]
        denom = v.real * v.real + v.imag * v.imag
        if denom <= EPS or not math.isfinite(denom):
            cf[i] = 0.0 + 0.0j
        else:
            cf[i] = complex(v.real, -v.imag) / denom  # 1/v

    # 3) cf[2::3] **= -2 -> inverse square for (2,5,8,...,68)
    for i in range(2, 71, 3):
        v = cf[i]
        v2 = v * v
        denom = v2.real * v2.real + v2.imag * v2.imag
        if denom <= EPS or not math.isfinite(denom):
            cf[i] = 0.0 + 0.0j
        else:
            cf[i] = complex(v2.real, -v2.imag) / denom  # 1/(v^2)

    # 4) Every 5th: cf[r-1] = (t1*t2)^r  BUT normalize base to keep bounded
    base = t1 * t2
    base_mag = math.hypot(base.real, base.imag)
    base_n = base / (1.0 + base_mag)  # |base_n| < 1
    for r in range(5, 66, 5):
        p = 1.0 + 0.0j
        for _ in range(r):
            p *= base_n
        cf[r - 1] = p
        if not (math.isfinite(p.real) and math.isfinite(p.imag)):
            return np.zeros(71, dtype=np.complex128)

    # 5) Final element (real-valued, then cast)
    r1 = math.hypot(t1.real, t1.imag)
    r2 = math.hypot(t2.real, t2.imag)
    val70 = r1 * r1 + 2.0 * t1.real * t2.imag + 3.0 * r2 * r2
    if not math.isfinite(val70):
        return np.zeros(71, dtype=np.complex128)
    cf[70] = complex(val70, 0.0)

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

def p101(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(71, dtype=np.complex128)

    # Fill cf[0:10] = 1000 * arange(1..10)
    for i in range(10):
        cf[i] = complex(1000.0 * float(i + 1), 0.0)

    # Fill cf[10:71] = 1
    for i in range(10, 71):
        cf[i] = complex(1.0, 0.0)

    # Magnitudes and angles (replace np.abs / np.angle)
    r1 = math.hypot(t1.real, t1.imag)
    r2 = math.hypot(t2.real, t2.imag)
    ang1 = math.atan2(t1.imag, t1.real)
    ang2 = math.atan2(t2.imag, t2.real)

    # Specific entries (replace np.abs and angle uses)
    cf[14] = complex(- (r1 ** 3), 0.0)
    cf[29] = complex(- (r2 ** 4), 0.0)
    cf[44] = complex(r1 ** 5, 0.0)
    cf[59] = complex(r2 ** 6, 0.0)
    cf[19] = complex((r1 ** 2) * math.sin(ang1), 0.0)
    cf[39] = complex((r2 ** 3) * math.cos(ang2), 0.0)
    cf[24] = complex(math.log(r1 + 1.0) * r1, 0.0)
    cf[49] = complex(math.log(r2 + 1.0) * r2, 0.0)

    # Helper: complex sin/cos for z = x + i y
    # sin(z) = sin x cosh y + i cos x sinh y
    # cos(z) = cos x cosh y - i sin x sinh y
    def sin_c(zc):
        x = zc.real
        y = zc.imag
        return complex(math.sin(x) * math.cosh(y), math.cos(x) * math.sinh(y))

    def cos_c(zc):
        x = zc.real
        y = zc.imag
        return complex(math.cos(x) * math.cosh(y), -math.sin(x) * math.sinh(y))

    # Final transform (fix OOB: j in 1..34 so 2*j+1 ≤ 69)
    for j in range(1, 35):
        jf = float(j)
        s1 = sin_c(complex(jf * t1.real, jf * t1.imag))
        c1 = cos_c(complex(jf * t1.real, jf * t1.imag))
        s2 = sin_c(complex(jf * t2.real, jf * t2.imag))
        c2 = cos_c(complex(jf * t2.real, jf * t2.imag))

        i0 = 2 * j
        i1 = 2 * j + 1  # ≤ 69 for j ≤ 34
        cf[i0] = cf[i0] * (s1 + c2) + cf[i1] * (c1 + s2)

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

def p234(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)

    EPS = 1e-12

    def _angle(z: complex) -> float:
        return np.arctan2(z.imag, z.real)

    a1 = np.abs(t1)
    a2 = np.abs(t2)
    ang1 = _angle(t1)
    ang2 = _angle(t2)

    for j in range(1, 36):
        k = j + 2
        r = np.sqrt(float(j))
        mag = (a1**r + a2**(k % 5 + 1)) * np.sin(j) + np.log(a1 + 1.0) * np.cos(r)
        ang = ang1 * np.cos(0.5 * j) - ang2 * np.sin(r)
        val = mag * (np.cos(ang) + 1j * np.sin(ang))
        val += np.conj(t1) * (t2**k) - np.conj(t2) * (t1**(k % 3))
        cf[j - 1] = val

    cf[4]  = (t1.real) + 1j * (t2.imag)
    cf[11] = np.log(a1 + 1.0) * np.log(a2 + 1.0) + np.sin(t1 * t2)
    cf[19] = (a1 - a2) * np.cos(ang1 - ang2) + 2j * np.sin(ang1 + ang2)
    cf[24] = np.conj(t1 + t2) * (t1.real - t2.imag) + 3j
    # sum([a1,a2]) and prod([a1,a2])
    cf[29] = (a1 + a2) + 1j * (a1 * a2)
    cf[34] = _angle(t1 * t2) + 1j * np.abs(t1 + t2)
    return cf

ALLOWED["p234"]=p234

def p235(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)

    EPS = 1e-12

    def _angle(z: complex) -> float:
        return np.arctan2(z.imag, z.real)

    a1 = np.abs(t1)
    a2 = np.abs(t2)
    ang1 = _angle(t1)
    ang2 = _angle(t2)

    # Fixed coefficients (assign individually for njit)
    cf[2]  = 3.0
    cf[6]  = -5.0
    cf[11] = 7.0
    cf[18] = -11.0
    cf[24] = 13.0
    cf[29] = -17.0

    # 1..10
    for j in range(1, 11):
        magnitude = np.log(a1**j + 1.0) + a2**(j % 3 + 1)
        angle = ang1 * j - ang2 * (j % 2)
        cf[j - 1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))

    # 11..20
    for k in range(11, 21):
        magnitude = np.sin(t1.real * k) + np.cos(t2.imag * k)
        angle = np.log(np.abs(t1 + t2) + 1.0) * k
        cf[k - 1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))

    # 21..34
    for r in range(21, 35):
        magnitude = np.log(np.abs(t1 * r) + 1.0) + (t2.real * t2.real)
        angle = _angle(np.conj(t1) + np.conj(t2)) * r
        cf[r - 1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))

    # last coefficient with guard on (t2 + 1)
    den = t2 + 1.0
    term = 0.0
    if np.isfinite(den.real) and np.isfinite(den.imag) and np.abs(den) > EPS:
        term = np.cos(t1 / den)
    cf[34] = np.sin(t1 * t2) + term + 1j * np.log(np.abs(t1 + t2) + 1.0)
    return cf

ALLOWED["p235"]=p235

def p236(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)

    EPS = 1e-12

    def _angle(z: complex) -> float:
        return np.arctan2(z.imag, z.real)

    for j in range(1, 36):
        angle = np.sin(t1.real * j) * np.cos(t2.imag * j) + _angle(t1 + t2) / float(j)
        magnitude = np.log(np.abs(t1) * j + np.abs(t2) + 1.0) * np.sqrt(1.0 + np.sin(j))
        cf[j - 1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))

    # while loop branches rewritten as for loop 1..35
    for k in range(1, 36):
        if (k % 5) == 0:
            cf[k - 1] = cf[k - 1] * np.conj(t1) + (np.abs(t2) ** 2)
        elif (k % 3) == 0:
            cf[k - 1] += (t1**k) - (t2**k)
        else:
            cf[k - 1] += np.sin(t1 * k) * np.cos(t2 * k)

    # r in 2..5, idx = r^2
    for r in range(2, 6):
        idx = r * r
        if idx <= 35:
            cf[idx - 1] += (t1.real * t2.imag) / float(r)

    # add vector at positions [3,9,15,21,27,33] => indices 3,9,15,21,27,33 (0-based)
    # divide by (r+1) where r from last loop ended at 5 -> r+1 = 6
    denom = 6.0
    for pos in (3, 9, 15, 21, 27, 33):
        cf[pos] += 100j * (t1 * t1 - t2 * t2) / denom

    return cf

ALLOWED["p236"]=p236

def p237(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)

    EPS = 1e-12

    def _angle(z: complex) -> float:
        return np.arctan2(z.imag, z.real)

    ang1 = _angle(t1)
    ang2 = _angle(t2)

    for j in range(1, 36):
        if (j % 4) == 1:
            cf[j - 1] = (t1**j * np.sin(j * ang2)) + (np.conj(t2) ** 2)
        elif (j % 4) == 2:
            cf[j - 1] = (t2**j * np.cos(j * ang1)) + (np.conj(t1) ** 2)
        elif (j % 4) == 3:
            cf[j - 1] = (t1.real * t2.imag) * np.log(np.abs(t1) + 1.0) + (t2.real ** j)
        else:
            cf[j - 1] = (t1.imag * t2.real) * np.log(np.abs(t2) + 1.0) + (t2.imag ** j)

        cf[j - 1] += (np.conj(t1) * np.conj(t2)) / float(j + 1)

    # second loop: avoid np.angle(cf[k-1])
    for k in range(1, 36):
        theta = _angle(cf[k - 1])
        if (k % 5) == 0:
            cf[k - 1] = cf[k - 1] * (1.0 + 0.05 * k) + np.sin(k * theta)
        else:
            cf[k - 1] = cf[k - 1] / (1.0 + 0.02 * k) + np.cos(k * theta)
        cf[k - 1] += np.log(np.abs(cf[k - 1]) + 1.0) * (cf[k - 1].real)

    return cf

ALLOWED["p237"]=p237

def p238(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)

    EPS = 1e-12

    def _angle(z: complex) -> float:
        return np.arctan2(z.imag, z.real)

    ang1 = _angle(t1)
    ang2 = _angle(t2)

    for j in range(1, 36):
        angle = ang1 * (j * j) - ang2 * np.sqrt(float(j))
        magnitude = (np.abs(t1) ** j) + (np.abs(t2) ** (35 - j)) + np.log(np.abs(t1) + np.abs(t2) + j)
        cf[j - 1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))

    # uses cf[36-k] -> 1..35 maps to indices 35..1 -> valid
    for k in range(1, 36):
        r = np.sqrt(float(k))  # (not used in calc but keeping consistency if desired)
        left = np.conj(cf[36 - k]) * (t1 ** (k % 5))
        right = np.conj(cf[k - 1]) * (t2 ** (35 - (k % 3)))
        cf[k - 1] += left - right

    cf[4]  += 2.0 * (t1**3) - 3.0 * (t2**2) + 1j * np.sin(t1 * t2)
    cf[9]   = np.conj(cf[9]) * t1 - (cf[9].imag) * t2 + np.log(np.abs(t1 + t2) + 1.0)

    denom14 = np.abs(t2) + 1.0
    adj14 = -cf[14] / denom14
    cf[14] = cf[14] * (t1 * t1) + adj14 + 1j * np.cos(t1 - t2)

    cf[19] = (cf[19].real) + 1j * (cf[19].imag) + (t1 * t2)
    cf[24] = np.abs(t1) * np.abs(t2) + 1j * _angle(t1 + t2)
    cf[29] = np.sin(t1 * t1) + 1j * np.cos(t2 * t2 * t2) - np.log(np.abs(t1 * t2) + 1.0)
    cf[34] = np.conj(cf[34]) + (t1 - t2) + 1j * np.sin(t1 + t2)

    return cf

ALLOWED["p238"]=p238

def p239(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)

    EPS = 1e-12

    def _angle(z: complex) -> float:
        return np.arctan2(z.imag, z.real)

    ang1 = _angle(t1)
    ang2 = _angle(t2)

    for j in range(1, 36):
        m = j % 5
        if m == 1:
            cf[j - 1] = (t1.real ** j) + (t2.imag ** 2)
        elif m == 2:
            cf[j - 1] = (np.abs(t1) * (np.abs(t2) ** j)) * (np.cos(ang1 * j) + 1j * np.sin(ang1 * j))
        elif m == 3:
            cf[j - 1] = np.conj(t1) * (np.sin(t2) ** j) + np.cos(t1 * t2)
        elif m == 4:
            cf[j - 1] = np.log(np.abs(t1) + 1.0) * (t2 ** j) + 1j * np.log(np.abs(t2) + 1.0)
        else:
            cf[j - 1] = (t1 + t2) ** j - (t1 - t2) ** j

    cf[4]  += 2j * t1 * t2
    cf[9]   = (t1.real ** 2) - (t2.imag ** 3) + 3j * np.abs(t1 * t2)
    cf[14]  = np.sin(t1 + t2) * np.cos(t1 - t2) + 1j * np.log(np.abs(t1) + np.abs(t2) + 1.0)
    cf[19]  = (t1 * t2) ** 2 - np.conj(t1) * np.conj(t2) + 2j * _angle(t1 + t2)
    cf[24]  = (t1 * t2).real + (t1.imag ** 2) - (t2.imag ** 2) + 1j * (t1.real - t2.real)
    cf[29]  = (np.abs(t1 + t2) ** 3) * (np.cos(_angle(t1 - t2)) + 1j * np.sin(_angle(t1 - t2)))
    cf[34]  = np.sin(np.abs(t1) * t2) + np.cos(np.abs(t2) * t1) + 1j * (t1.real * t2.real)
    return cf

ALLOWED["p239"]=p239

def p240(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)

    EPS = 1e-12
    def _angle(z: complex) -> float:
        return np.arctan2(z.imag, z.real)
    
    # specific coefficients (assign individually)
    cf[0]  = 1.0 + 0.0j
    cf[4]  = 4.0 + 0.0j
    cf[12] = 4.0 + 0.0j
    cf[19] = -9.0 + 0.0j
    cf[20] = -1.9 + 0.0j
    cf[24] = 0.2 + 0.0j

    ang1 = _angle(t1)
    ang2 = _angle(t2)

    # j=2..34 except 5,13,20,21,25
    for j in range(2, 35):
        if j in (5, 13, 20, 21, 25):
            continue
        mag = np.log(np.abs(t1 + j) + 1.0) * np.sin(j * ang2) + np.cos(j * ang1)
        angle = (ang1 ** j) + np.sin(j * ang2) - np.cos(j)
        cf[j - 1] = mag * np.cos(angle) + 1j * (mag * np.sin(angle))

    cf[34] = np.conj(t1) * np.conj(t2) + np.sin(np.abs(t1) * np.abs(t2)) + 1j * np.log(np.abs(t1) + np.abs(t2) + 1.0)
    return cf

ALLOWED["p240"]=p240

def p241(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)

    EPS = 1e-12

    def _angle(z: complex) -> float:
        return np.arctan2(z.imag, z.real)

    a1 = np.abs(t1)
    a2 = np.abs(t2)

    for j in range(1, 36):
        magnitude = np.log(a1 + a2 + j) * (a1 ** np.sin(j) + a2 ** np.cos(j))
        angle = _angle(t1) * j - _angle(t2) * (35 - j) + np.sin(j) * np.cos(j)
        cf[j - 1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))

    # add conj(t1)*t2**k/(k+1)
    for k in range(1, 36):
        cf[k - 1] += (np.conj(t1) * (t2 ** k)) / float(k + 1)

    # add 50*(Re(t1) - Im(t2))*1j at indices 4,9,14,19,24,29
    delta = 50.0 * (t1.real - t2.imag) * 1j
    for idx in (4, 9, 14, 19, 24, 29):
        cf[idx] += delta

    return cf

ALLOWED["p241"]=p241

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

def p244(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(1, 36):
        k = (j % 6) + 1
        r = (j % 4) + 1
        angle_part = np.sin(j * np.real(t1)) * np.cos(j * np.imag(t2)) + np.angle(t1) / (k + 1)
        mag_part = np.abs(t1)**k * np.abs(t2)**r + np.log(np.abs(t1) + np.abs(t2) + j)
        cf[j - 1] = np.cos(angle_part) * mag_part + np.sin(angle_part) * mag_part * 1j
    return cf.astype(np.complex128)
    
ALLOWED["p244"]=p244

def p245(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    # Assign fixed coefficients with non-symmetric values
    cf[np.array([1, 5, 9, 13, 17, 21, 25, 29, 33],dtype=np.intp)] = np.array([2 + 3j, -3 + 2j, 4 - 1j, -5 + 4j, 6 - 3j, -7 + 5j, 8 - 4j, -9 + 6j, 10 - 5j])
    # Assign coefficients using loop for j indices with intricate calculations
    j_indices = [0, 4, 8, 12, 16, 20, 24, 28, 32]
    for j in j_indices:
        cf[j] = (np.real(t1) * np.imag(t2) + np.imag(t1) * np.real(t2)) + \
                    (np.abs(t1)**2 - np.abs(t2)**2) * np.sin(t1 + t2) + \
                    np.log(np.abs(t1) + 1) * np.cos(t2)
    # Assign coefficients using loop for k indices with complex functions
    k_indices = [2, 6, 10, 14, 18, 22, 26, 30, 34]
    for k in k_indices:
        cf[k] = np.sin(t1 * t2) + np.cos(t1 / (np.abs(t2) + 1)) * np.conj(t2) + \
                    np.angle(t1 + t2) * np.abs(t1 - t2) + \
                    (np.real(t1) * np.imag(t2))
    # Assign coefficients using loop for r indices with mixed parameters
    r_indices = [3, 7, 11, 15, 19, 23, 27, 31]
    for r in r_indices:
        cf[r] = np.real(t1)**3 - np.imag(t2)**3 + np.real(t1 * t2) + np.imag(t1 + t2) + np.log(np.abs(t1 * t2) + 1)
    # Additional intricate assignments for specific coefficients
    cf[18] = 100j * t1**3 + 50j * t2**2 - 75 * t1 * t2 + 25
    cf[22] = 80j * t2**3 - 60j * t1**2 + 40 * np.sin(t1 + t2) - 20
    cf[26] = 90j * t1 * t2**2 - 70 * np.cos(t1) + 50 * np.log(np.abs(t2) + 1)
    cf[30] = 110j * np.sin(t1**2) - 95 * np.abs(t2) * t1 + 85j * np.angle(t1 + t2)
    cf[34] = 120j * np.cos(t1 * t2) - 100 * np.sin(t2) + 75 * np.log(np.abs(t1) + 1)
    # Return the complex coefficient vector
    return cf.astype(np.complex128).astype(np.complex128)
    
ALLOWED["p245"]=p245

def p246(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    # Fixed coefficients with varied complex expressions
    cf[np.array([0, 3, 7, 11, 15, 19, 23, 27, 31],dtype=np.intp)] = np.array([
        2 + 3j,
        np.conj(t1) * np.sin(t2),
        np.log(np.abs(t1) + 1) + np.log(np.abs(t2) + 1) * 1j,
        np.real(t1)**2 - np.imag(t2)**2 + (np.real(t2) * np.imag(t1)) * 1j,
        np.sin(t1 * t2) + np.cos(t1 + t2) * 1j,
        (t1 *t2) + (np.real(t1) * np.imag(t2)) * 1j,
        np.abs(t1)**3 - np.abs(t2)**3 + np.angle(t1) * np.angle(t2) * 1j,
        np.real(t2) * np.sin(np.angle(t1)) + np.imag(t1) * np.cos(np.angle(t2)) * 1j,
        np.real(t1 + t2) + np.imag(t1 - t2) * 1j
    ])
    # Loop to assign remaining coefficients with intricate patterns
    for j in [2, 3, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34]:
        k = j * 3
        r = j % 4
        cf[j] = (np.real(t1)**k + np.imag(t2)**k) * np.sin(k * np.angle(t1)) + \
                    (np.real(t2)**r - np.imag(t1)**r) * np.cos(r * np.angle(t2)) * 1j + \
                    np.log(np.abs(t1) + np.abs(t2) + j) * (1 + 1j)
    
    return cf.astype(np.complex128)

ALLOWED["p246"]=p246

def p247(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(1, 36):
        angle = np.angle(t1)**j + np.angle(t2)**(j % 5 + 1)
        magnitude = np.abs(t1)**(j % 7) * np.abs(t2)**(j // 5 + 1)
        cf[j - 1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    
    k = 1
    while k <= 35:
        cf[k - 1] += np.conj(t1) * t2**(k % 3) - np.log(np.abs(t1 + t2) + 1)
        k += 4
    
    for r in range(2, 35, 3):
        cf[r - 1] *= (np.sin(t1 * r) + np.cos(t2 / (r + 1)))
    
    cf[9] = (np.abs(t1) + np.abs(t2)) * np.exp(1j * np.angle(t1 + t2))
    cf[19] = (np.abs(t1) * np.abs(t2)) / (1 + np.abs(t1 - t2))
    cf[34] = np.real(t1)**3 - np.imag(t2)**2 + 2j * np.real(t2) * np.imag(t1)
    
    return cf.astype(np.complex128)
    
ALLOWED["p247"]=p247

def p248(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    # Initialize specific coefficients
    cf[np.array([1, 6, 12, 18, 24, 30],dtype=np.intp)] = np.array([3, -5, 8, -12, 20, -25])
    
    # Loop to assign coefficients with intricate patterns
    for j in range(1, 36):
        if j % 4 == 1:
            angle = np.angle(t1) * j + np.sin(j * np.angle(t2))
            magnitude = np.log(np.abs(t1) + np.abs(t2) + j) * (j % 3 + 1)
            cf[j - 1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
        elif j % 4 == 2:
            angle = np.angle(t2) * j + np.cos(j * np.angle(t1))
            magnitude = np.abs(t1)**2 + np.abs(t2)**2 + j
            cf[j - 1] = magnitude * (np.cos(angle) - 1j * np.sin(angle))
        elif j % 4 == 3:
            angle = np.sin(j * np.angle(t1 + t2))
            magnitude = np.log(np.abs(t1 * t2) + 1) * (j + 2)
            cf[j - 1] = magnitude * np.exp(1j * angle)
        else:
            angle = np.cos(j * np.angle(t1 - t2))
            magnitude = (np.abs(t1) + np.abs(t2))**j / (j + 1)
            cf[j - 1] = magnitude * (1 + 1j * angle)
    
    # Additional intricate modifications
    for k in range(5, 36, 5):
        cf[k - 1] += (np.real(t1)**k - np.imag(t2)**k) * 1j
    
    for r in range(10, 16):
        cf[r - 1] *= (1 + 0.5j * np.real(t1 + t2))
    
    # Assign non-symmetric, non-circular roots patterns
    cf[19] = np.prod(np.abs(cf[0:10]))**(1/5) * (np.sin(np.angle(t1)) + np.cos(np.angle(t2)))
    cf[33] = np.conj(cf[33]) + t1**3 - t2**3
    
    return cf.astype(np.complex128)

ALLOWED["p248"]=p248

def p249(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(1, 36):
        if j % 5 == 1:
            cf[j - 1] = np.real(t1)**j + np.imag(t2)**(j % 3) * np.sin(j * np.angle(t1))
        elif j % 5 == 2:
            cf[j - 1] = np.conj(t1) * np.cos(j * np.angle(t2)) + np.abs(t2)**2 / (j + 1)
        elif j % 5 == 3:
            cf[j - 1] = np.log(np.abs(t1) + 1) + 1j * np.log(np.abs(t2) + 1) + np.real(t1)**2 - np.imag(t2)**2
        elif j % 5 == 4:
            cf[j - 1] = (np.real(t1) * np.imag(t2))**j + (np.abs(t1) + np.abs(t2)) * np.sin(j)
        else:
            cf[j - 1] = (np.real(t1) + np.imag(t2)) * np.cos(j * np.angle(t1) * np.angle(t2)) + 1j * (np.abs(t1) * np.abs(t2))
    return cf.astype(np.complex128)
    
ALLOWED["p249"]=p249

def p250(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(1, 36):
        if j % 4 == 1:
            cf[j - 1] = (np.real(t1)**j + np.sin(j * np.angle(t1))) + 1j * (np.imag(t1)**j + np.cos(j * np.abs(t1)))
        elif j % 4 == 2:
            cf[j - 1] = np.log(np.abs(t2) + 1) * (np.real(t2)**j - np.imag(t2)**j) + 1j * (np.angle(t2)**j + np.abs(t2)**j)
        elif j % 4 == 3:
            cf[j - 1] = np.sin(t1 * j) * np.cos(t2 * j) + np.conj(t1) * np.conj(t2)
        else:
            cf[j - 1] = np.abs(t1 + t2)**j + 1j * np.angle(t1 - t2)
    for k in range(1, 8):
        idx = k * 5
        if idx <= 35:
            cf[idx - 1] *= (np.sin(k) + 1j * np.cos(k))
    cf[7] = np.sum(np.abs(cf[0:7])) + 1j * np.prod(np.abs(cf[0:7]))
    cf[15] = np.cos(t1 + t2) + 1j * np.sin(t1 - t2)
    cf[23] = np.log(np.abs(t1**2 - t2**2) + 1) + 1j * np.angle(t1 * t2)
    cf[31] = np.conj(t1)**3 + np.conj(t2)**2 + np.sin(t1 * t2)
    cf[34] = np.real(t1) * np.real(t2) + np.imag(t1) * np.imag(t2) + 1j * (np.real(t1) - np.imag(t2))
    return cf.astype(np.complex128)
    
ALLOWED["p250"]=p250

def p251(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(1, 36):
        r = np.real(t1) + np.real(t2) + j
        angle = np.angle(t1) * j - np.angle(t2)
        cf[j - 1] = (np.abs(t1)**j + np.abs(t2)**(35 - j)) * np.exp(1j * angle) * np.sin(j * np.real(t1) - np.imag(t2))
    cf[4] = np.conj(t1) * t2**2 - np.log(np.abs(t1) + 1) + 2j * np.real(t2)
    cf[9] = np.sin(t1) + np.cos(t2) * np.conj(t1)
    cf[14] = (t1 * t2)**3 - np.real(t1)**2 + np.imag(t2)**3
    cf[19] = np.exp(1j * np.angle(t1)) * np.log(np.abs(t2) + 1) + np.abs(t1 + t2)
    cf[24] = np.sin(t1 + t2) * np.cos(t1 - t2) + 1j * (np.real(t1) * np.imag(t2))
    cf[29] = (np.real(t1) * np.imag(t2) * np.abs(t1 + t2)) + (np.real(t2) + np.imag(t1))
    cf[34] = np.conj(t1)**2 + np.conj(t2)**3 - t1 * t2
    return cf.astype(np.complex128)
    
ALLOWED["p251"]=p251

def p252(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(1, 36):
        k = j % 6
        r = j // 6 + 1
        if k == 1:
            cf[j - 1] = (np.log(np.abs(t1) + 1) + np.sin(np.angle(t2))) * t1**r
        elif k == 2:
            cf[j - 1] = (np.cos(np.angle(t1)) - np.sin(np.abs(t2))) * np.conj(t2)**r
        elif k == 3:
            cf[j - 1] = (np.real(t1) * np.imag(t2) + np.real(t2) * np.imag(t1)) * (t1 + t2)**r
        elif k == 4:
            cf[j - 1] = (np.abs(t1)**2 - np.abs(t2)**2) * np.exp(1j * np.angle(t1 * t2)) * r
        elif k == 5:
            cf[j - 1] = (np.sin(t1 * r) + np.cos(t2 / r)) * (t1 - t2)**2
        else:
            cf[j - 1] = (np.log(np.abs(t1 * t2) + 1) + np.angle(t1 + t2)) * (t1 + np.conj(t2))**r
    cf[4] = 100j * t1**4 - 50 * t2**2 + 25j
    cf[11] = 75 * np.conj(t1) - 60j * t2 + 30
    cf[18] = (t1**3 + t2**3) / (np.real(t1) + np.real(t2) + 1)
    cf[25] = np.sin(t1 + t2) * np.cos(t1 - t2) * 1j
    cf[32] = np.log(np.abs(t1 + t2) + 1) * (t1**2 - t2**2)
    cf[34] = np.real(t1 * t2) + np.imag(t1 - t2) * 1j
    return cf.astype(np.complex128)
    
ALLOWED["p252"]=p252

def p253(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(1, 36):
        if j % 4 == 0:
            k = j // 4
            angle = np.angle(t1 + t2) * k
            cf[j - 1] = (np.real(t1)**k + np.imag(t2)**k) * (np.cos(angle) + np.sin(angle) * 1j)
        elif j % 5 == 0:
            r = j // 5
            cf[j - 1] = np.log(np.abs(t1) * r + 1) + np.conj(t2)**r
        elif j % 3 == 1:
            cf[j - 1] = np.sin(t1 * j) + np.cos(t2 * j) * 1j
        else:
            cf[j - 1] = np.real(t1 * t2) + np.imag(t1 / t2) * 1j
    cf[6] = (np.real(t1) * np.imag(t2)) + (np.abs(t1) + np.abs(t2)) * 1j
    cf[13] = t1**3 + t2**2 - 5 * t1 * t2 * 1j
    cf[20] = np.sin(t1 + t2) + np.cos(t1 - t2) * 1j
    cf[27] = np.log(np.abs(t1) + 1) * np.conj(t2) - np.sin(t1 * t2)
    cf[34] = np.real(t1)**2 - np.imag(t2)**2 + 2 * np.real(t1) * np.imag(t2) * 1j
    return cf.astype(np.complex128)
    
ALLOWED["p253"]=p253

def p254(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(1, 36):
        k = j % 5 + 1
        r = j // 5 + 1
        angle = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j)
        magnitude = np.abs(t1)**k + np.abs(t2)**r + np.log(np.abs(t1) + 1) * np.log(np.abs(t2) + 1)
        cf[j - 1] = magnitude * (np.cos(angle) + np.sin(angle) * 1j)
    
    cf[np.array([2, 7, 13, 21, 28],dtype=np.intp)] = np.conj(t1) * t2**2 - t1**2 * np.conj(t2)
    cf[np.array([4, 10, 18, 26, 34],dtype=np.intp)] = np.sin(t1 * t2) + np.cos(t1 + t2) * 1j
    cf[16] = (np.abs(t1) * np.abs(t2)) * np.exp(1j * (np.angle(t1) - np.angle(t2)))
    cf[24] = (np.abs(t1 + t2) + np.real(t1)**2 + np.imag(t2)**2) * (1 + 1j)
    cf[34] = np.log(np.abs(t1) + 1) + np.log(np.abs(t2) + 1) * 1j
    
    return cf.astype(np.complex128)
    
ALLOWED["p254"]=p254

def p255(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    # Initialize specific coefficients with fixed values
    cf[np.array([2, 7, 13, 18, 25, 33],dtype=np.intp)] = [2 + 3j, -4 + 1j, 5 - 2j, -3 + 4j, 1.5 - 0.5j, -2.2 + 2j]
    # Loop to assign intricate coefficients
    for j in range(1, 36):
        if j not in [3, 8, 14, 19, 26, 34]:
            k = j % 7 + 1
            r = j // 5 + 1
            magnitude = np.sin(j * np.angle(t1)) * np.cos(k * np.abs(t2)) + np.log(np.abs(t1) + 1) * r
            angle = np.angle(t2) * k - np.angle(t1) * r + np.sin(j * np.imag(t1))
            cf[j - 1] = magnitude * (np.cos(angle) + np.sin(angle) * 1j)
    # Additional intricate patterns
    for k in range(1, 6):
        idx = 7 * k
        if idx <= 35:
            cf[idx - 1] = (np.conj(t1)**k + t2**k) * np.exp(-k / (np.abs(t1) + np.abs(t2) + 1)) + \
                            (np.sin(t1 * k) + np.cos(t2 * k)) * 1j
    for r in range(1, 4):
        start = 10 * r
        for j in range(start, start + 4):
            if j <= 35:
                cf[j - 1] = (t1 + t2)**r * np.sin(j) + (np.real(t1) - np.imag(t2))**2 * 1j
    return cf.astype(np.complex128)
    
ALLOWED["p255"]=p255

# serp:runs:zero:one,uc,p256,aberth,line
def p256(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(1, 36):
        mag_part = np.log(np.abs(t1 + j) + 1) * (np.abs(t2)**(j % 5 + 1))
        angle_part = np.angle(t1) * np.sin(j) - np.angle(t2) * np.cos(j)
        cf[j - 1] = mag_part * (np.cos(angle_part) + 1j * np.sin(angle_part))
    for k in range(1, 36):
        if k % 4 == 0:
            cf[k - 1] += np.conj(t1)**k * np.sin(t2 * k)
        elif k % 3 == 0:
            cf[k - 1] *= (np.real(t1) + np.imag(t2) * np.log(k + 1))
        else:
            cf[k - 1] += np.abs(t1) * np.abs(t2) / (k + 1)
    for r in range(1, 8):
        idx = r * 5
        if idx <= 35:
            cf[idx - 1] += 100j * t2**r - 50 * t1**r
    cf[9] = np.sum(np.abs(cf[0:9])) * np.sin(np.real(t1)) - np.cos(np.imag(t2))
    cf[19] = np.prod(np.abs(cf[14:19] + 1)) / (1 + np.abs(t1 * t2))
    cf[29] = np.conj(t1) + np.sin(t2) * np.log(np.abs(t1) + 1)
    cf[34] = np.real(t1)**2 - np.imag(t2)**2 + 1j * (np.real(t2) * np.imag(t1))
    return cf.astype(np.complex128)

ALLOWED["p256"]=p256

def p257(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(1, 36):
        k = j % 5 + 1
        r = (j**2 + np.sin(np.real(t1) * j) - np.cos(np.imag(t2) * k)) / (np.log(np.abs(t1) + 1) + 1)
        angle = np.angle(t1)**k + np.angle(t2)**(j % 3)
        cf[j - 1] = r * (np.real(t1)**k + np.imag(t2)**j) * np.exp(1j * angle)
    for k in range(1, 6):
        r = (np.real(t1) + k) - (np.imag(t2) * k)
        angle = np.angle(t1 + k) - np.angle(t2 + k)
        index = 5 * k
        if index <= 35:
            cf[index - 1] = r * np.exp(1j * angle) * (np.sin(t1 * k) + np.cos(t2 * k))
    cf[6] = np.conj(t1) * t2**2 + np.sin(t1 + t2)
    cf[13] = np.log(np.abs(t1) + 1) * np.cos(t2) - 1j * np.sin(t1 * t2)
    cf[20] = np.abs(t1)**3 - np.abs(t2)**2 + 1j * np.angle(t1 * t2)
    cf[27] = np.real(t1**2) + np.imag(t2**3) - 2j * np.real(t1 * t2)
    cf[34] = (np.real(t1) * np.real(t2)) + (np.imag(t1) + np.imag(t2)) * 1j
    return cf.astype(np.complex128)
    
ALLOWED["p257"]=p257

def p258(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(1, 36):
        angle = np.sin(j * np.angle(t1) + np.cos(j * np.angle(t2))) + np.real(t1) * np.imag(t2)
        magnitude = np.log(np.abs(t1) + np.abs(t2) + j) + np.real(t1)**((j % 4) + 1) - np.imag(t2)**((j % 3) + 1) + (np.real(t1) * np.imag(t2) * j)
        cf[j - 1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))

    cf[2] = np.conj(t1) * t2**2 + np.sin(t1 * t2) * np.cos(t1 - t2)
    cf[7] = np.real(t1**2 + t2**2) + 1j * np.imag(t1 * t2)
    cf[14] = np.log(np.abs(t1 + t2) + 1) + 1j * np.angle(t1 - t2)
    cf[21] = np.sin(t1)**3 - np.cos(t2)**3 + 1j * (np.sin(t1) * np.cos(t2))
    cf[28] = np.real(t1 * t2) + np.imag(t1 + t2) * 1j
    cf[34] = (np.abs(t1) * np.abs(t2) * j) + 1j * (np.real(t1) + np.imag(t2))

    return cf.astype(np.complex128)
    
ALLOWED["p258"]=p258

def p259(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    # Initialize coefficients with base patterns
    for j in range(1, 36):
        cf[j - 1] = (np.real(t1)**j * np.imag(t2)**(35 - j)) + np.conj(t1) * np.sin(j * np.angle(t2)) + \
                        np.log(np.abs(t1) + np.abs(t2) + 1) * np.cos(j * np.angle(t1 + t2))
    # Introduce variations using loops
    for k in range(1, 6):
        r = k + 5
        cf[r - 1] += np.abs(t1)**k * np.abs(t2)**(5 - k) * np.exp(1j * (np.angle(t1) - np.angle(t2)))
    for m in range(6, 11):
        cf[m - 1] += np.sin(np.real(t1) * m) + np.cos(np.imag(t2) * m)
    # Assign specific intricate coefficients
    cf[11] = np.real(t1 * t2) + 1j * np.imag(t1 / t2)
    cf[19] = np.log(np.abs(t1 + t2)) + 1j * np.angle(t1 - t2)
    cf[24] = np.conj(t1)**2 - np.conj(t2)**3 + np.sin(t1 * t2)
    cf[29] = np.abs(t1)**3 * np.abs(t2)**2 + np.cos(np.angle(t1) * np.angle(t2))
    cf[34] = (np.abs(t1) * np.abs(t2)) + (np.real(t1) + np.imag(t2)) * 1j

    return cf.astype(np.complex128)

ALLOWED["p259"]=p259

def p260(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(1, len(cf) + 1):
        k = j % 7 + 1
        r = j // 7 + 1
        cf[j - 1] = (np.real(t1)**k - np.imag(t2)**r) * np.cos(np.angle(t1) * j) + np.sin(np.angle(t2) * r) / (np.abs(t1) + np.abs(t2) + j)
    cf[3] = np.conj(t1) * t2**2 + np.log(np.abs(t1) + 1) * np.sin(t2)
    cf[7] = np.real(t1 * t2) + np.imag(t1)**2 - np.cos(t2)
    cf[12] = np.abs(t1 + t2)**2 - np.real(t1)**3 + np.imag(t2)
    cf[16] = np.sin(t1) * np.cos(t2) + np.real(t2)**2 - np.imag(t1)**2
    cf[21] = np.log(np.abs(t1 * t2) + 1) + np.conj(t1) - np.conj(t2)
    cf[25] = np.real(t1)**2 * np.imag(t2) - np.real(t2) * np.imag(t1) + np.sin(np.angle(t1 + t2))
    cf[30] = (np.real(t1) + np.imag(t1)) * (np.real(t2) - np.imag(t2)) + np.cos(np.angle(t1 * t2))
    cf[33] = np.real(t1)**3 - np.imag(t1)**3 + np.real(t2)**3 - np.imag(t2)**3
    cf[34] = (np.real(t1) + np.real(t2) + np.imag(t1) + np.imag(t2)) + (np.abs(t1) * np.abs(t2))
    return cf.astype(np.complex128).astype(np.complex128)

ALLOWED["p260"]=p260

def p261(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(1, 36):
        mag = np.log(1 + np.abs(t1)**j + np.abs(t2)**(35 - j)) + np.sin(j * np.angle(t1) + np.angle(t2))
        angle = np.cos(j * np.angle(t1)) - np.sin((35 - j) * np.angle(t2))
        cf[j - 1] = mag * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p261"]=p261

def p262(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    # Assign fixed coefficients for specific indices
    cf[np.array([2, 7, 15, 23, 29],dtype=np.intp)] = np.array([2 + 1j, -3 + 2j, 4 - 1.5j, -2.2 + 0.8j, 0.6 - 0.4j],dtype=np.complex128)
    # Loop to assign intricate coefficients
    for j in range(1, 36):
        if j not in [3, 8, 16, 24, 30]:
            k = (j * 3) % 7 + 1
            r = (j + 4) % 5 + 1
            mag = np.log(np.abs(t1)**k + np.abs(t2)**r + j)
            ang = np.angle(t1) * k - np.angle(t2) * r + np.sin(j) * np.pi / 6
            cf[j - 1] = mag * (np.cos(ang) + np.sin(ang) * 1j)
    # Add additional intricate coefficients
    cf[11] = np.conj(t1)**2 * t2 - t1 * np.conj(t2)
    cf[18] = np.sin(t1 * t2) + np.cos(t1 + t2) * 1j
    cf[26] = np.log(np.abs(t1 + t2) + 1) + np.angle(t1 - t2) * 1j
    cf[33] = (np.real(t1) + np.imag(t2)) * np.cos(np.angle(t1)) + (np.imag(t1) - np.real(t2)) * np.sin(np.angle(t2)) * 1j
    return cf.astype(np.complex128)
    
ALLOWED["p262"]=p262

def p263(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(1, 36):
        if j % 4 == 0:
            cf[j - 1] = (np.real(t1) * j + np.imag(t2) * (35 - j)) + (np.abs(t1)**0.5 * np.angle(t2)) * 1j
        elif j % 3 == 0:
            cf[j - 1] = np.sin(t1 * j) + np.cos(t2 + j) + np.log(np.abs(t1 * t2) + 1)
        elif j % 5 == 0:
            cf[j - 1] = (np.real(t2)**j - np.imag(t1)**(j % 3)) + np.conj(t1) * np.imag(t2) * 1j
        else:
            cf[j - 1] = np.real(t1)**2 + np.imag(t2)**2 + np.sin(t1 + t2) * np.cos(t1 - t2) * 1j
    for k in range(1, 8):
        index = k * 5
        if index <= 35:
            cf[index - 1] += (t1**k - t2**k) * (k % 2) + np.log(np.abs(t1 + t2) + 1) * 1j
    for r in range(1, 6):
        idx = 7 + r * 6
        if idx <= 35:
            cf[idx - 1] += np.prod(cf[0:r]) * np.sin(t1 * r) + np.cos(t2 * r) * 1j
    cf[9] = 100j * t2**3 + 100j * t2**2 - 100 * t2 - 100
    cf[14] = 100j * t1**3 - 100j * t1**2 + 100 * t2 - 100
    cf[24] = np.real(t1 * t2) + np.imag(t1 + t2) * 1j
    return cf.astype(np.complex128)

ALLOWED["p263"]=p263

def p264(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    for k in range(1, 36):
        r = np.log(np.abs(t1) + 1) * np.sin(np.angle(t2) * k) + np.cos(np.angle(t1) * k)
        theta = np.angle(t1)**2 / (k + 1) + np.angle(t2) * np.log(np.abs(t2) + 1)
        magnitude = (np.real(t1) + np.imag(t2))**k / (k + 2) + (np.real(t2) - np.imag(t1))**(k % 5 + 1)
        cf[k - 1] = magnitude * (np.cos(theta) + np.sin(theta) * 1j)
    for j in range(5, 36, 5):
        cf[j - 1] = np.conj(cf[j - 1]) * t1**2 - t2**3
    for r in range(3, 36, 3):
        cf[r - 1] = np.real(t1) * cf[r - 1] + np.imag(t2) * cf[r - 1]**2
    cf[0] = 1 + t1 - t2
    cf[34] = np.sin(t1 * t2) + np.cos(t1 / (np.abs(t2) + 1))

    return cf.astype(np.complex128)
    
ALLOWED["p264"]=p264

def p265(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(1, 36):
        angle = np.angle(t1) * j + np.angle(t2) * (35 - j)
        magnitude = np.abs(t1)**((j % 5) + 1) + np.abs(t2)**((j % 7) + 1) + np.log(np.abs(t1 * t2) + 1)
        phase = np.sin(j * np.real(t1)) + np.cos(j * np.imag(t2)) + np.angle(t1 + t2)
        cf[j - 1] = magnitude * (np.cos(phase) + 1j * np.sin(phase))
    return cf.astype(np.complex128)

ALLOWED["p265"]=p265

def p266(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    for k in range(1, 36):
        mag = np.sin(np.abs(t1) * (k**2)) + np.cos(np.abs(t2) / k) + np.log(np.abs(t1) + 1) * np.log(np.abs(t2) + 1)
        ang = np.angle(t1) * k + np.angle(t2) * (35 - k) + np.sin(k) * np.cos(k)
        cf[k - 1] = mag * (np.cos(ang) + 1j * np.sin(ang))
    return cf.astype(np.complex128)

ALLOWED["p266"]=p266

def p267(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(1, 36):
        k = (j * 3 + 7) % 35 + 1
        angle = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j)
        magnitude = np.log(np.abs(t1) + 1) * np.real(t2)**0.5 + np.imag(t1)**2 / (j + 1)
        cf[j - 1] = magnitude * (np.cos(angle) + np.sin(angle) * 1j) + np.conj(t1) * t2**j
    cf[4] = np.real(t1) + np.imag(t2) * 1j
    cf[11] = np.abs(t1)**2 - np.abs(t2)**2 + (np.real(t1) * np.imag(t2)) * 1j
    cf[19] = np.sin(t1) + np.cos(t2) * 1j
    for r in range(25, 36):
        cf[r - 1] += (t1 * t2)**r / (r + 1)
    return cf.astype(np.complex128).astype(np.complex128)

ALLOWED["p267"]=p267

def p268(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(1, 36):
        k = j + 2
        r = (j % 5) + 1
        cf[j - 1] = (np.real(t1)**k - np.imag(t2)**r) * np.sin(np.abs(t1) * j) + (np.angle(t2) + j) * np.cos(np.log(np.abs(t1) + 1))
    cf[3] = np.conj(t1) * t2**2 - np.abs(t2) * np.cos(t1)
    cf[6] = np.sin(t1 * t2) + np.cos(t1 + t2) * t1
    cf[9] = np.log(np.abs(t1) + 1) + np.real(t2)**3 - np.imag(t1) * np.imag(t2)
    cf[12] = (t1**2 + t2**2) * np.sin(t1) - np.cos(t2)
    cf[15] = np.real(t1) * np.real(t2) + np.imag(t1) * np.imag(t2) + np.angle(t1 * t2)
    cf[18] = np.abs(t1 + t2) * np.sin(np.angle(t1)) - np.cos(np.abs(t2))
    cf[21] = np.conjugate(t1**3)+t2**3 - np.log(np.abs(t1*t2)+1)
    cf[24] = np.sin(t1**2) + np.cos(t2**2) - np.real(t1 * t2)
    cf[27] = np.imag(t1**2) - np.real(t2**2) - np.real(t1 * t2)
    cf[30] = np.abs(t1)**2 * np.cos(t2) - np.sin(np.abs(t2))
    cf[33] = np.real(t1**3) - np.imag(t2**3) + np.log(np.abs(t1 + t2) + 1)
    return cf.astype(np.complex128)

ALLOWED["p268"]=p268

def p269(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(1, 36):
        if j % 3 == 1:
            cf[j - 1] = (np.real(t1)**j + np.imag(t2)**(j % 5 + 1)) * np.sin(j * np.angle(t1)) + np.cos(j * np.angle(t2))
        elif j % 3 == 2:
            cf[j - 1] = (np.abs(t1) * np.abs(t2))**((j + 1) / 7) + np.log(np.abs(t1) + 1) * np.log(np.abs(t2) + 1)
        else:
            cf[j - 1] = np.conj(t1) * t2**(j % 4) - np.conj(t2) * t1**(j % 3)
    # Override specific coefficients with intricate patterns
    cf[3] = (np.real(t1) + np.imag(t2)) + (np.abs(t1) * np.abs(t2))
    cf[9] = np.sin(t1 * t2) + np.cos(t1 - t2) + np.log(np.abs(t1 + t2) + 1)
    cf[15] = (np.real(t1)**2 - np.imag(t1)**2) + (np.real(t2)**2 - np.imag(t2)**2)
    cf[21] = np.abs(t1 * t2) * np.angle(t1 + t2) + np.conj(t1 - t2)
    cf[27] = np.sin(np.real(t1) * np.imag(t2)) + np.cos(np.imag(t1) * np.real(t2))
    cf[33] = np.log(np.abs(t1)**3 + np.abs(t2)**3 + 1) + np.real(t1 * np.conj(t2))
    return cf.astype(np.complex128)

ALLOWED["p269"]=p269 

def p270(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(1, 36):
        phase1 = np.angle(t1) * j + np.sin(j * np.angle(t2))
        magnitude1 = np.log(np.abs(t1) + j) * np.cos(j * np.pi / 7)
        term1 = magnitude1 * np.exp(1j * phase1)

        phase2 = np.angle(t2) * (35 - j) + np.cos(j * np.angle(t1))
        magnitude2 = np.log(np.abs(t2) + (35 - j)) * np.sin(j * np.pi / 5)
        term2 = magnitude2 * np.exp(1j * phase2)

        cf[j - 1] = term1 + term2 + np.conj(t1)**(j % 5) * np.conj(t2)**(j % 3)

    for k in range(2, 35, 3):
        cf[k - 1] *= (np.sin(np.abs(t1 * k)) + np.cos(np.abs(t2 + k)))

    for r in range(1, 36, 5):
        cf[r - 1] += 1j * np.log(np.abs(t1 + r) + 1) * np.sin(np.angle(t2) * r)

    cf[0] = np.real(t1) + np.real(t2)
    cf[34] = np.imag(t1) - np.imag(t2) + np.conj(t1 * t2)

    return cf.astype(np.complex128).astype(np.complex128)
    
ALLOWED["p270"]=p270

def p271(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(1, 36):
        mag = np.log(np.abs(t1) + j) * np.sin(j * np.angle(t2)) + np.cos(np.abs(t2) * j)
        angle = np.real(t1) * j + np.imag(t2) / (j + 1)
        cf[j - 1] = mag * (np.cos(angle) + np.sin(angle) * 1j)

    return cf.astype(np.complex128)

ALLOWED["p271"]=p271

def p272(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    for k in range(1, 36):
        j = (k + 3) % 6 + 1
        r = k // 4 + 1
        mag_part = np.log(np.abs(t1) + k) * np.sin(j * np.angle(t2)) + np.cos(r * np.angle(t1))
        angle_part = np.angle(t1)**j - np.angle(t2)**r + np.sin(k) * np.cos(k)
        cf[k - 1] = mag_part * (np.cos(angle_part) + 1j * np.sin(angle_part)) + np.conj(t1)**j * np.conj(t2)**r
    return cf.astype(np.complex128)

ALLOWED["p272"]=p272

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

def p274(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(1, 36):
        mag = np.log(np.abs(t1) + 1)**j * np.sin(j * np.angle(t1)) + np.abs(t2)**(j % 4 + 1)
        ang = np.angle(t1) * j + np.angle(t2) * (j % 5)
        cf[j - 1] = mag * (np.cos(ang) + 1j * np.sin(ang))
    for k in range(1, 36, 7):
        cf[k - 1] += 100j * t1**k - 50 * t2**(k % 3)
    for r in range(2, 35):
        cf[r - 1] = cf[r - 1] * (1 + 0.1 * r) + np.conj(t1) * np.sin(r * np.angle(t2))
    cf[0] = 1 + np.real(t1) - np.real(t2)
    cf[34] = 2 - np.imag(t1) + np.imag(t2)
    return cf.astype(np.complex128)

ALLOWED["p274"]=p274

def p275(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(1, 36):
        k = (j * 4) % 8 + 1
        r = j // 5 + 2
        angle = np.angle(t1) * j + np.angle(t2) * k + np.sin(j) * np.cos(k)
        mag = np.abs(t1)**j + np.abs(t2)**k + np.log(np.abs(t1 * t2) + 1) * r
        cf[j - 1] = mag * (np.cos(angle) + 1j * np.sin(angle)) + np.conj(t1)**r * np.conj(t2)**k
    return cf.astype(np.complex128)

ALLOWED["p275"]=p275

def p276(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(1, 36):
        if j % 5 == 1:
            cf[j - 1] = np.real(t1)**j + np.imag(t2)**(j % 3 + 1) * np.conj(t1)
        elif j % 5 == 2:
            cf[j - 1] = np.abs(t1 + t2)**j * np.sin(np.angle(t1) * j) + 1j * np.cos(np.angle(t2) * j)
        elif j % 5 == 3:
            cf[j - 1] = np.log(np.abs(t1) + 1) * np.real(t2)**j - 1j * np.log(np.abs(t2) + 1) * np.imag(t1)**j
        elif j % 5 == 4:
            cf[j - 1] = (t1**2 + t2**3) * np.sin(j) + 1j * (t1 * t2)**2 * np.cos(j)
        else:
            cf[j - 1] = (np.real(t1) * np.imag(t2) * j) + 1j * (np.abs(t1) + np.abs(t2) + j)

    cf[4] = 100 * t1**4 - 50j * t2**2 + 25
    cf[9] = 200j * np.sin(t1) + 150 * np.cos(t2)
    cf[14] = 300 * np.log(np.abs(t1) + 1) + 100j * np.log(np.abs(t2) + 1)
    cf[19] = np.conj(t1) * t2**3 - t1**2 * np.conj(t2)
    cf[24] = np.abs(t1)**3 + np.abs(t2)**2 * 1j
    cf[29] = np.sin(t1 * t2) + np.cos(t1 + t2) * 1j
    cf[34] = np.log(np.abs(t1 * t2) + 1) + 1j * np.angle(t1 + t2)
    return cf.astype(np.complex128)
    
ALLOWED["p276"]=p276

def p277(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(35):  # Python uses 0-based indexing
        if j < 5:
            k = j + 1
            cf[j] = (np.real(t1)**k + np.imag(t2)**k) * np.exp(1j * np.angle(t1 + t2))
        elif j < 10:
            k = j - 4
            cf[j] = (np.abs(t1)**k * np.abs(t2)**5) / (k + 1) + 1j * np.sin(k * np.angle(t1))
        elif j < 15:
            k = j - 9
            cf[j] = np.real(t1 * t2) + 1j * np.imag(t1**k + t2**k)
        elif j < 20:
            k = j - 14
            cf[j] = np.log(np.abs(t1) + 1) * np.cos(k * np.angle(t2)) + 1j * np.log(np.abs(t2) + 1) * np.sin(k * np.angle(t1))
        elif j < 25:
            k = j - 19
            cf[j] = (t1 + np.conj(t2))**k + (np.conj(t1) - t2)**k
        elif j < 30:
            k = j - 24
            cf[j] = np.real(t1)**k * np.imag(t2)**k + 1j * (np.abs(t1 + t2)**k)
        else:
            k = j - 29
            cf[j] = (np.real(t1) * np.imag(t2))**k + np.conj(t1 * t2)**k
    
    cf[11] = 100 * t1**3 - 50j * t2**2 + 25 * t1 * t2
    cf[17] = 200j * np.sin(t1) + 150 * np.cos(t2)
    cf[26] = 300 * np.log(np.abs(t1) + 1) + 100j * np.abs(t2)**2
    cf[33] = 400 * np.real(t1 * t2) - 200j * np.imag(t1 + t2)
    return cf.astype(np.complex128)
    
ALLOWED["p277"]=p277

def p278(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    fixed_indices = np.array([0, 3, 9, 15, 21, 29],dtype=np.intp)  # Adjusted for 0-based indexing
    cf[fixed_indices] = np.array([2, -3, 5 + 2j, -4 + 1j, 3.5, -2.2])
    for j in range(1, 35):
        if j not in fixed_indices:
            angle = np.angle(t1)**0.5 * (j+1) + np.angle(t2)**0.3 * (35 - (j+1))
            magnitude = np.abs(t1)**((j+1) / 3) + np.abs(t2)**(35 - (j+1))/2
            cf[j] = magnitude * (np.cos(angle) + np.sin(angle)*1j)
    cf[6] = (100 * t1**2 - 50 * np.conj(t2)) + (25 * np.sin(t1) + 75 * np.cos(t2))*1j
    cf[13] = (200 * t2**3 + 100 * np.real(t1)) + (50 * np.imag(t2) - 30 * np.log(np.abs(t1)+1))*1j
    cf[20] = (np.abs(t1) + np.abs(t2)) + (np.real(t1) * np.real(t2))*1j
    cf[27] = (np.log(np.abs(t1) + 1) * np.real(t1)) - (np.real(t2)**2) + (np.imag(t1) * np.imag(t2))*1j
    return cf.astype(np.complex128)

ALLOWED["p278"]=p278

def p279(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(35):
        mag = np.log(np.abs(t1)**(j+1) + np.abs(t2)**(35-(j+1)) + 1) * ((j+1) % 7 + 1) * (1 + np.sin(j+1))
        ang = np.angle(t1) * (j+1)**0.5 - np.angle(t2) * (35 - (j+1))**0.3
        cf[j] = mag * (np.cos(ang) + 1j * np.sin(ang))
    
    for k in range(35):
        if (k+1) % 5 == 0:
            cf[k] = cf[k] * np.conj(t1) + np.real(t2)**2
        elif (k+1) % 3 == 0:
            cf[k] = cf[k] + np.imag(t1) * np.imag(t2)
        else:
            cf[k] = cf[k] * np.real(t1 + t2) - np.imag(t1 - t2)
    
    indices = np.array([2, 7, 14, 22, 28, 34],dtype=np.intp)
    cf[indices] = cf[indices] + 100 * t1**2 - 50 * t2
    return cf.astype(np.complex128)
    
ALLOWED["p279"]=p279

def p280(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(35):
        if (j+1) % 6 == 1:
            cf[j] = (t1**(j+1) + np.conj(t2)**(j+1)) * np.log(np.abs(t1) + 1)
        elif (j+1) % 6 == 2:
            cf[j] = (np.sin(t1 * (j+1)) + np.cos(t2 * (j+1))) * ((j+1)**2 + np.real(t1))
        elif (j+1) % 6 == 3:
            cf[j] = (np.real(t1) * np.imag(t2))**(j+1) + np.conj(t1 * t2)
        elif (j+1) % 6 == 4:
            cf[j] = np.log(np.abs(t1 + t2) + 1) * ((j+1)**1.5) * np.angle(t1 + t2)
        elif (j+1) % 6 == 5:
            cf[j] = (np.real(t1)**2 - np.imag(t2)**2) * (j+1) + 1j * (np.imag(t1) + np.real(t2))
        else:
            cf[j] = (np.abs(t1) + np.abs(t2)) * (j+1)**3 * np.sin(np.angle(t1 * t2))
    
    indices = np.array([2, 7, 14, 21, 28, 33],dtype=np.intp)
    cf[indices] = [2 + 3j, -1 + 4j, 0.5 - 2j, 3 + 0j, -2.5j, 1 + 1j]
    cf[9] = 100j * (t1**3) - 50 * t2**2 + 25 * np.conj(t1)
    cf[19] = 75 * t2**3 + 50j * np.conj(t2) - 25 * t1
    cf[24] = 60j * np.sin(t1) * np.cos(t2) + 40 * np.log(np.abs(t1 * t2) + 1)
    cf[34] = 150 * np.real(t1 + t2) - 100j * np.imag(t1 - t2)
    return cf.astype(np.complex128)

ALLOWED["p280"]=p280

def p281(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    cf[np.array([0, 4, 9, 14, 19, 24, 29, 34],dtype=np.intp)] = np.array([
        np.real(t1) + np.imag(t2) * 1j,
        np.abs(t1)**2 - np.abs(t2)**2 * 1j,
        np.sin(np.angle(t1)) + np.cos(np.angle(t2)) * 1j,
        np.log(np.abs(t1) + 1) - np.log(np.abs(t2) + 1) * 1j,
        np.conj(t1) + np.conj(t2) * 1j,
        np.real(t1)**3 - np.imag(t2)**3 * 1j,
        np.abs(t1)**4 + np.abs(t2)**4 * 1j,
        np.sin(np.angle(t1) * 2) - np.cos(np.angle(t2) * 2) * 1j
    ])
    for j in [2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29, 31, 32, 33, 34]:
        k = j * 3
        r = j % 5
        if r == 0:
            cf[j] = (np.real(t1) + np.imag(t2)) * np.sin(k) + (np.real(t2) - np.imag(t1)) * np.cos(k) * 1j
        elif r == 1:
            cf[j] = np.abs(t1 + t2)**k * np.exp(1j * np.angle(t1 - t2))
        elif r == 2:
            cf[j] = np.log(np.abs(t1)**k + 1) + np.log(np.abs(t2)**k + 1) * 1j
        elif r == 3:
            cf[j] = np.conj(t1)**k - np.conj(t2)**k * 1j
        else:
            cf[j] = (np.real(t1) + np.imag(t2)) * (np.abs(t1) * np.abs(t2)) + 1j * (np.imag(t1) + np.real(t2))
    return cf.astype(np.complex128)

ALLOWED["p281"]=p281

def p282(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    cf[np.array([0, 6, 13, 20, 27, 34],dtype=np.intp)] = np.array([2.5, -4.2, 3.8, -16.5, 5.3, 0.6])
    for j in range(2, 35):
        if j % 4 == 0:
            k = j // 2
            cf[j] = (150j * t1**k + 75 * np.conj(t2)) * np.sin(k * np.angle(t1)) - 50 * np.log(np.abs(t2) + 1)
        elif j % 3 == 0:
            k = j % 5
            cf[j] = (200 * np.real(t1 * t2**k) + 100j * np.imag(t1 - t2)) * np.cos(k * np.angle(t2))
        else:
            r = j % 7
            cf[j] = np.conj(t1)**r * t2**j + np.abs(t1**j) * np.abs(t2**r)
    cf[9] = 180j * t1**3 - 120 * t2**2 + 90 * np.sin(t1) * np.cos(t2)
    cf[19] = 220j * t2**4 + 130 * np.real(t1**3) - 100 * np.imag(t2)
    cf[29] = 260j * t1**2 * t2 + 160 * np.log(np.abs(t1 * t2) + 1) - 110 * np.conj(t1)
    return cf.astype(np.complex128)

ALLOWED["p282"]=p282

def p283(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(1, 36):
        angle_part = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j)
        magnitude_part = np.abs(t1)**((j % 5) + 1) + np.abs(t2)**((j % 7) + 1)
        cf[j - 1] = magnitude_part * (np.cos(angle_part) + 1j * np.sin(angle_part))
        if j % 4 == 0:
            cf[j - 1] += np.conj(t1) * t2**2 - np.log(np.abs(t1) + 1)
        if j % 6 == 0:
            cf[j - 1] *= np.sin(t1 * j) + np.cos(t2 / (j + 1))
    specific_indices = [3, 8, 15, 22, 29, 35]
    for k in specific_indices:
        cf[k - 1] += (np.real(t1) + np.imag(t2)) * t1**k - (np.real(t2) - np.imag(t1)) * t2**k
    cf[np.array([4, 11, 18, 25, 32],dtype=np.intp)] = np.array([5, -10, 15, -20, 25]) + 1j * np.array([-5, 10, -15, 20, -25])
    return cf.astype(np.complex128)
    
ALLOWED["p283"]=p283

def p284(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        phase = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j)
        magnitude = np.abs(t1)**((j % 5) + 1) + np.abs(t2)**(j // 7 + 1)
        perturb = np.log(np.abs(t1 + t2) + 1) * np.cos(j * np.pi / 3) + np.sin(j * np.pi / 4)
        cf[j - 1] = magnitude * np.exp(1j * phase) + perturb
    return cf.astype(np.complex128)

ALLOWED["p284"]=p284

def p285(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        real_seq = np.linspace(np.real(t1), np.real(t2), j)
        imag_seq = np.linspace(np.imag(t1), np.imag(t2), j)
        mag_component = np.sum(np.log(np.abs(real_seq) + 1) * np.sin(real_seq * j)) + np.prod(imag_seq + 1)
        angle_component = np.sum(np.cos(imag_seq * j)) - np.sum(np.sin(real_seq / (j + 1)))
        cf[j - 1] = mag_component * (np.cos(angle_component) + 1j * np.sin(angle_component))
    return cf.astype(np.complex128)
    
ALLOWED["p285"]=p285

def p286(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for k in range(1, n + 1):
        a = np.real(t1) * np.sin(k * np.real(t2)) + np.imag(t1) * np.cos(k * np.imag(t2))
        b = np.log(np.abs(t1) + 1) * np.sin(k * np.angle(t2) / (k + 1))
        c = np.abs(t2)**k * np.cos(k * np.real(t1))
        d = np.sin(k * np.imag(t1)) + np.cos(k * np.real(t2))
        angle = np.angle(t1) * np.sin(k) + np.angle(t2) * np.cos(k)
        magnitude = a + b + c + d
        cf[k - 1] = magnitude * np.exp(1j * angle) + np.conj(t1) * np.conj(t2)**k
    return cf.astype(np.complex128)
    
ALLOWED["p286"]=p286

def p287(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec_seq = np.linspace(np.real(t1), np.real(t2), n)
    imc_seq = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(1, n + 1):
        r = rec_seq[j - 1]
        m = imc_seq[j - 1]
        mag_part = np.log(np.abs(r * m) + 1) * (j**2 + np.sin(j) * np.cos(j))
        angle_part = np.angle(t1) * np.sin(j / 3) + np.angle(t2) * np.cos(j / 4) + np.sin(m * np.pi / 5)
        coeff = mag_part * np.exp(1j * angle_part) + np.conj(t1) * np.conj(t2) / (j + 1)
        cf[j - 1] = coeff
    return cf.astype(np.complex128)
    
ALLOWED["p287"]=p287

def p288(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(np.real(t1), np.real(t2), n)
    imc = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(1, n + 1):
        mag_sum = 0
        angle_sum = 0
        for k in range(1, j + 1):
            term_mag = np.log(np.abs(rec[k - 1] * imc[j - 1]) + 1) * np.sin(k * np.pi / n)
            term_angle = np.angle(rec[k - 1] + 1j * imc[j - 1]) + np.cos(k * np.pi / (n + 1))
            mag_sum += term_mag
            angle_sum += term_angle
        magnitude = mag_sum * np.prod(np.repeat(np.abs(t1) + k, j % 3 + 1))
        angle = angle_sum / (j + 1) + np.sin(j * np.pi / (n + 2)) * np.cos(j * np.pi / (n + 3))
        variation = np.sin(j) if j % 2 == 0 else np.cos(j)
        cf[j - 1] = magnitude * np.exp(1j * angle) + variation * np.conj(t2)**j
    return cf.astype(np.complex128)
    
ALLOWED["p288"]=p288

def p289(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec_seq = np.linspace(np.real(t1), np.real(t2), n)
    imc_seq = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(1, n + 1):
        mag_factor = np.log(np.abs(rec_seq[j - 1] + imc_seq[j - 1] * 1j) + 1) * (1 + np.sin(j * np.pi / 4))
        angle_factor = np.angle(rec_seq[j - 1] + 1j * imc_seq[j - 1]) + np.cos(j * np.pi / 3) * np.sin(j * np.pi / 5)
        cf[j - 1] = mag_factor * (np.cos(angle_factor) + 1j * np.sin(angle_factor))
    return cf.astype(np.complex128)
    
ALLOWED["p289"]=p289

def p290(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(np.real(t1), np.real(t2), 35)
    imc = np.linspace(np.imag(t1), np.imag(t2), 35)
    for j in range(1, 36):
        if j % 4 == 1:
            mag = np.log(np.abs(t1) + j**2) + np.sin(j * np.pi / 6) * np.cos(j * np.pi / 4)
            angle = np.angle(t1) * j + np.sin(j * np.pi / 5) - np.cos(j * np.pi / 3)
        elif j % 4 == 2:
            mag = np.log(np.abs(t2) + j) * np.prod(np.arange(1, (j % 5) + 2))
            angle = np.angle(t2) / (j + 1) + np.sin(j * np.pi / 7)
        elif j % 4 == 3:
            mag = np.real(t1) * j - np.imag(t2) + np.log(np.abs(t1 + t2) + 1)
            angle = np.angle(t1 * t2) + np.cos(j * np.pi / 2)
        else:
            mag = np.abs(np.real(t1 - t2)) * j**1.5 + np.sin(j * np.pi / 3)
            angle = np.angle(t1 - t2) + np.sin(j * np.pi / 4)
        cf[j - 1] = mag * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p290"]=p290

def p291(z, a, state):
    t1 = z[0]
    t2 = z[1]
    n = 35

    # Always return same shape
    cf = np.zeros(n, dtype=np.complex128)

    # Early guards -> zero vector (fixed shape)
    if abs(t1 - t2) < 1e-10:
        return cf
    if abs(t1.real - t2.real) < 1e-10:
        return cf

    # Precompute constants/angles/magnitudes (scalar-safe)
    r1 = math.hypot(t1.real, t1.imag)
    r2 = math.hypot(t2.real, t2.imag)
    ang1 = math.atan2(t1.imag, t1.real)
    ang2 = math.atan2(t2.imag, t2.real)
    dreal = abs(t1.real - t2.real)

    # Cumulative sums for angle_part2:
    # S1(j) = sum_{m=1..j} sin(m*pi/6)
    # C2(j) = sum_{m=1..j} cos(m*pi/8)
    S1 = 0.0
    C2 = 0.0

    for j in range(1, n + 1):
        jf = float(j)

        # Update cumulative sums
        S1 += math.sin(jf * math.pi / 6.0)
        C2 += math.cos(jf * math.pi / 8.0)

        # magnitude pieces
        mag_part1 = math.log(r1 + r2 + jf) * (1.0 + math.sin(jf * math.pi / 7.0))
        # sqrt(j!) = exp(0.5 * lgamma(j+1))
        mag_part2 = math.exp(0.5 * math.lgamma(j + 1.0)) / (1.0 + dreal / (jf + 1.0))
        magnitude = mag_part1 * mag_part2 * (1.0 + math.cos(jf * math.pi / 5.0))

        # angle pieces
        angle_part1 = ang1 * math.sin(jf / 3.0) + ang2 * math.cos(jf / 4.0)
        angle_part2 = S1 - C2
        angle = angle_part1 + angle_part2

        # perturbation
        real_component = t1.real * math.cos(jf) - t2.imag * math.sin(jf)
        imag_component = t2.real * math.sin(jf) + t1.imag * math.cos(jf)
        perturbation = math.sin(real_component) + math.cos(imag_component)

        # exp(i*angle) via cos/sin (robust under njit)
        ci = math.cos(angle)
        si = math.sin(angle)
        val_real = magnitude * perturbation * ci
        val_imag = magnitude * perturbation * si

        cf[j - 1] = np.complex128(val_real + 1j * val_imag)

    return cf

ALLOWED["p291"]=p291

def p292(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35 # was 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        k = (j**2 + 3 * j + 1) % n + 1
        r = np.sin(j * np.real(t1)) * np.cos(k * np.imag(t2))
        angle = np.angle(t1) * j - np.angle(t2) * k + np.log(j + 1)
        magnitude = np.abs(t1)**0.5 * np.abs(t2)**0.3 * np.abs(r) + j
        cf[j - 1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p292"]=p292

def p293(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35 # was 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        k = (j * 2 + 5) % 12
        r = j // 6 + 1
        term_re = np.real(t1) * np.sin(j) + np.real(t2) * np.cos(k)
        term_im = np.imag(t1) * np.cos(j / 4) - np.imag(t2) * np.sin(k / 3)
        magnitude = (np.abs(term_re) + np.abs(term_im)) * np.log(1 + j) * (j**0.4)
        angle = np.angle(t1) * np.sin(j / 2) + np.angle(t2) * np.cos(k / 4) + np.log(j + 2)
        cf[j - 1] = magnitude * np.exp(1j * angle)
    return cf.astype(np.complex128)

ALLOWED["p293"]=p293

def p294(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35 # was 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(np.real(t1), np.real(t2), n)
    imc = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(1, n + 1):
        k = (j % 7) + 1
        r = rec[j - 1] * np.cos(j) - imc[j - 1] * np.sin(j)
        i_part = rec[j - 1] * np.sin(j) + imc[j - 1] * np.cos(j)
        mag = np.log(np.abs(r + 1) + np.abs(i_part + 1)) * (1 + np.sin(j * np.pi / k)) * (1 + np.cos(j * np.pi / (k + 1)))
        angle = np.angle(t1) + np.angle(t2) + np.sin(j * np.pi / k) + np.cos(j * np.pi / (k + 2))
        cf[j - 1] = mag * np.exp(1j * angle) + np.conj(t1 * t2) / (j + 2)
    return cf.astype(np.complex128)
    
ALLOWED["p294"]=p294

def p295(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35 # was 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(np.real(t1), np.real(t2), n)
    imc = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(1, n + 1):
        r = rec[j - 1] + imc[j - 1]
        magnitude = np.log(np.abs(r + 1j) + 1) * (j**(np.sin(j) + 1))
        angle = np.angle(r + 1j) + np.sin(j * np.pi / 4) * np.cos(j * np.pi / 3)
        cf[j - 1] = magnitude * np.exp(1j * angle) + np.conj(magnitude * np.exp(1j * (angle / 2))) * np.cos(j * np.pi / 5)
    return cf.astype(np.complex128)

ALLOWED["p295"]=p295

def p296(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35 # was 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(np.real(t1), np.real(t2), 35)
    imc = np.linspace(np.imag(t1), np.imag(t2), 35)
    for k in range(1, 36):
        j = (k + 7) % 12 + 1
        term1 = rec[k - 1] * np.cos(imc[j - 1] * np.pi / 5)
        term2 = imc[k - 1] * np.sin(rec[j - 1] * np.pi / 4)
        conj_part = np.conj(t1) * np.conj(t2)
        angle = np.angle(term1 + term2 + np.angle(conj_part))
        magnitude = np.log(np.abs(term1 + term2) + 1) * (np.abs(t1)**((k % 4) + 1)) * (np.abs(t2)**((j % 3) + 1))
        cf[k - 1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p296"]=p296

def p297(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35 # was 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        mag_part = np.log(np.abs(t1) + j) * np.sin(j * np.angle(t2)) + np.cos(j**2 * np.real(t1))
        angle_part = np.angle(t1) * np.log(j + 1) + np.angle(t2) * np.sqrt(j)
        cf[j - 1] = mag_part * np.exp(1j * angle_part) + np.conj(t1)**j / (1 + np.abs(t2 + j))
    return cf.astype(np.complex128)

ALLOWED["p297"]=p297

def p298(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35 # was 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(np.real(t1), np.real(t2), n)
    imc = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(1, n + 1):
        mag_part = np.log(np.abs(rec[j - 1]) + 1) * np.sin(j * np.pi / 7) + np.cos(j * np.pi / 5)
        angle_part = np.angle(t1) * j**0.5 - np.angle(t2) / (j + 2)
        fluctuation = np.abs(t1) * np.abs(t2) if j % 3 == 0 else np.abs(t1 + t2) / (j + 1)
        cf[j - 1] = (mag_part + fluctuation) * np.exp(1j * angle_part) + np.conj(t1 * t2)**j
    return cf.astype(np.complex128)
    
ALLOWED["p298"]=p298

def p299(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35 # was 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        mag = np.abs(t1)**j + np.abs(t2)**(n - j + 1) + np.sum(np.sin(j * np.pi / (np.arange(1, 6) + 1)))
        ang = np.angle(t1) * np.log(j + 1) + np.angle(t2) * np.arctan(j) + np.sum(np.cos((np.arange(1, 4)) * np.pi / j))
        cf[j - 1] = mag * (np.cos(ang) + 1j * np.sin(ang))
    return cf.astype(np.complex128)

ALLOWED["p299"]=p299

def p300(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35 # was 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        r1 = np.real(t1)
        r2 = np.real(t2)
        i1 = np.imag(t1)
        i2 = np.imag(t2)
        term_mag = np.log(np.abs(t1) + j) * np.abs(r1 * j - i2 / (j + 1)) + np.prod(np.array([r1, i2, j]))
        term_angle = np.angle(t1) * j - np.angle(t2) * (n - j) + np.sin(j * r2) * np.cos(j * i1)
        cf[j - 1] = term_mag * (np.cos(term_angle) + 1j * np.sin(term_angle))
    return cf.astype(np.complex128)
    
ALLOWED["p300"]=p300

def p301(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35 # was 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        mag_part1 = np.log(np.abs(np.real(t1) * j) + 1) * np.sin(j * np.pi / 4)
        mag_part2 = np.cos(j * np.pi / 3) * np.abs(t2)**0.5
        magnitude = mag_part1 + mag_part2 + np.sum(np.arange(1, j + 1)) / np.prod(np.arange(1, min(j, 6) + 1))
        
        angle_part1 = np.angle(t1) * np.sin(j / 2)
        angle_part2 = np.angle(t2) * np.cos(j / 3)
        phase = angle_part1 + angle_part2 + np.sin(j) * np.cos(j / 2)
        
        cf[j - 1] = magnitude * np.exp(1j * phase) + np.conj(t1) * np.sin(j * np.pi / 6) - np.conj(t2) * np.cos(j * np.pi / 5)
    return cf.astype(np.complex128)

ALLOWED["p301"]=p301

def p302(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35 # was 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(np.real(t1), np.real(t2), n)
    imc = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(1, n + 1):
        mag = np.log(np.abs(t1) + j) * (1 + np.sin(rec[j - 1] * imc[j - 1])) + np.prod(np.array([np.real(t1), np.imag(t2)])) / (j + 1)
        if np.abs((imc[j - 1] + 1))<1e-10: return cf
        angle = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j) + np.sin(rec[j - 1] / (imc[j - 1] + 1))
        cf[j - 1] = mag * (np.cos(angle) + 1j * np.sin(angle)) + np.conj(t1) * np.cos(j / 3)
    return cf.astype(np.complex128)
    
ALLOWED["p302"]=p302

def p303(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35 # was 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, 36):
        r1 = np.real(t1)
        i1 = np.imag(t1)
        r2 = np.real(t2)
        i2 = np.imag(t2)
        term1 = np.log(np.abs(r1 * j) + 1) * np.sin(j * np.pi / 3)
        term2 = np.log(np.abs(i2 + j) + 1) * np.cos(j * np.pi / 4)
        term3 = np.real(np.conj(t1) * t2) / (j + 1)
        mag = np.abs(term1 + term2 + term3)
        angle = np.angle(t1) * np.cos(j * np.pi / 6) + np.angle(t2) * np.sin(j * np.pi / 8) + np.log(np.abs(r1 + i1 * i2) + 1) / (j + 2)
        cf[j - 1] = mag * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p303"]=p303

def p304(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35 # was 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(np.real(t1), np.real(t2), n)
    imc = np.linspace(np.imag(t1), np.imag(t2), n)
    for k in range(1, n + 1):
        r = rec[k - 1]
        im = imc[k - 1]
        conj_t1 = np.conj(t1)
        conj_t2 = np.conj(t2)
        mag_component = np.log(np.abs(t1) + np.abs(t2) + 1) * np.sin(k * np.angle(t1 * conj_t2) + np.cos(k * np.pi / 6))
        angle_component = np.angle(t1)**k - np.angle(t2)**(n - k) + np.sin(k * np.pi / 5)
        magnitude = np.abs(mag_component + r * np.prod(imc[:k]) / (k + 1))
        cf[k - 1] = magnitude * (np.cos(angle_component) + 1j * np.sin(angle_component))
    return cf.astype(np.complex128)
    
ALLOWED["p304"]=p304

def p305(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35 # was 35
    cf = np.zeros(n, dtype=np.complex128)
    real_seq = np.linspace(np.real(t1), np.real(t2), n)
    imag_seq = np.linspace(np.imag(t1), np.imag(t2), n)
    for k in range(1, n + 1):
        r = real_seq[k - 1]
        im = imag_seq[k - 1]
        mag_pattern = np.log(np.abs(t1 * t2) + k**2) * (1 + np.sin(k) * np.cos(k / 2))
        angle_pattern = np.angle(t1) * np.sin(k / 3) + np.angle(t2) * np.cos(k / 4) + np.sin(k * np.pi / 5)
        cf[k - 1] = mag_pattern * np.exp(1j * angle_pattern) + np.conj(mag_pattern * np.exp(1j * angle_pattern / 2))
    return cf.astype(np.complex128)
    
ALLOWED["p305"]=p305

def p306(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35 # was 35
    cf = np.zeros(n, dtype=np.complex128)
    rec1 = np.real(t1)
    rec2 = np.real(t2)
    imc1 = np.imag(t1)
    imc2 = np.imag(t2)
    for j in range(1, n + 1):
        k = j % 5 + 1
        r = np.log(np.abs(rec1 * rec2) + 1) + np.log(np.abs(t1 * t2) + 1)
        angle = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j)
        magnitude = np.abs(t1)**k + np.abs(t2)**(n - k) + r * np.sin(j * np.pi / 7)
        phase = angle + np.cos(j * np.pi / 5)
        cf[j - 1] = magnitude * (np.cos(phase) + 1j * np.sin(phase))
        if j % 7 == 0:
            cf[j - 1] = cf[j - 1] * np.conj(t1) + np.conj(t2)
        if j % 3 == 0:
            cf[j - 1] += np.sin(t1 * j) * np.cos(t2 / j)
        if j % 4 == 0:
            cf[j - 1] += np.exp(1j * (rec1 * j - imc2))
    return cf.astype(np.complex128)
    
ALLOWED["p306"]=p306

def p307(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    # Early exits must return same-shaped array
    if abs(t1.real - t2.real) < 1e-10: return cf
    if abs(t1.imag - t2.imag) < 1e-10: return cf
    # Precompute polar bits for t1
    r1   = math.hypot(t1.real, t1.imag)
    ang1 = math.atan2(t1.imag, t1.real)
    # Conjugate of t2 without np.conj (njit-safe)
    t2_conj = np.complex128(t2.real - 1j * t2.imag)
    for j in range(1, n + 1):
        jf = float(j)
        # phase: replace np.angle with atan2, keep trig in math.*
        phase = (math.sin(jf * math.pi / 4.0) +
                 math.cos(jf * math.pi / 3.0) +
                 ang1 * jf / 10.0)
        # magnitude:
        # - log(|t1| + j) via hypot + math.log
        # - sqrt(j!) via exp(0.5 * lgamma(j+1))
        magnitude = (math.log(r1 + jf) * (1.0 + math.sin(jf * math.pi / 6.0))
                     + math.exp(0.5 * math.lgamma(jf + 1.0)) * math.cos(jf * math.pi / 8.0))
        # exp(1j * phase) = cos + i sin (avoid np.exp on complex)
        c = math.cos(phase)
        s = math.sin(phase)
        cf[j - 1] = magnitude * (c + 1j * s) + t2_conj * (j % 5)
    return cf

ALLOWED["p307"]=p307

def p308(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(np.real(t1), np.real(t2), n)
    imc = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(1, n + 1):
        magnitude = (np.abs(t1) * np.log(np.abs(rec[j - 1]) + 1) * j) if j % 2 == 0 else (np.abs(t2) * np.sin(rec[j - 1]) + np.abs(t1 + t2) / (j + 1))
        angle = (np.angle(t1) + np.sin(imc[j - 1] * np.pi / j)) if j <= n / 2 else (np.angle(t2) + np.cos(rec[j - 1] * np.pi / j))
        cf[j - 1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    for k in range(1, n + 1):
        cf[k - 1] += np.conj(cf[k - 1]) * np.exp(1j * (np.sin(k) + np.cos(k)))
    for r in range(1, n + 1):
        cf[r - 1] *= (1 + np.log(np.abs(cf[r - 1]) + 1)) / (1 + r / n)
    return cf.astype(np.complex128)
    
ALLOWED["p308"]=p308

def p309(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec_seq = np.linspace(np.real(t1), np.real(t2), n)
    imc_seq = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(1, n + 1):
        mag = np.log(np.abs(rec_seq[j - 1] + imc_seq[j - 1] * 1j) + 1) * (1 + np.sin(j * np.pi / 4)) * (1 + np.cos(j * np.pi / 5))
        angle = np.sin(j * rec_seq[j - 1]) + np.cos(j * imc_seq[j - 1]) + np.angle(t1 * t2) / (j + 1)
        cf[j - 1] = mag * np.exp(1j * angle) + np.conj(t2) * np.sin(j / n * np.pi)
    return cf.astype(np.complex128)

ALLOWED["p309"]=p309

def p310(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        mag_part = np.log(np.abs(t1) + 1) * np.sin(j * np.pi / 5) + np.log(np.abs(t2) + 1) * np.cos(j * np.pi / 7) + j**1.5
        ang_part = np.angle(t1) * np.sin(j * np.pi / 4) + np.angle(t2) * np.cos(j * np.pi / 6) + np.sin(j / 3)
        cf[j - 1] = mag_part * np.exp(1j * ang_part)
    return cf.astype(np.complex128)
    
ALLOWED["p310"]=p310

def p311(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    if abs(t1.real - t2.real) < 1e-10: return cf
    if abs(t1.imag - t2.imag) < 1e-10: return cf
    rec = np.linspace(np.real(t1), np.real(t2), n)
    imc = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(1, n + 1):
        mag_part = np.log(np.abs(rec[j - 1] * t1 + imc[j - 1] * t2) + 1) * (1 + np.sin(j) * np.cos(j))
        angle_part = np.sin(j * np.pi * imc[j - 1]) + np.cos(j * np.pi * rec[j - 1]) + np.angle(t1) - np.angle(t2)
        cf[j - 1] = mag_part * np.exp(1j * angle_part)
    for k in range(1, n + 1):
        r = np.log(k + 1)
        cf[k - 1] *= (1 + np.sin(r) + 1j * np.cos(r))
    return cf.astype(np.complex128)
    
ALLOWED["p311"]=p311

def p312(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    if abs(t1.real - t2.real) < 1e-10: return cf
    if abs(t1.imag - t2.imag) < 1e-10: return cf
    rec = np.linspace(np.real(t1), np.real(t2), 35)
    imc = np.linspace(np.imag(t1), np.imag(t2), 35)
    for j in range(1, 36):
        magnitude = np.log(np.abs(rec[j - 1] * imc[j - 1]) + 1) * np.sin(j * np.pi / 4) + np.cos(j * np.pi / 3)
        angle = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j) + np.log(np.abs(rec[j - 1] + imc[j - 1]) + 1)
        cf[j - 1] = magnitude * np.exp(1j * angle)
    return cf.astype(np.complex128)
    
ALLOWED["p312"]=p312

def p313(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(np.real(t1), np.real(t2), n)
    imc = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(1, n + 1):
        if j % 4 == 1:
            mag = np.log(np.abs(rec[j - 1] * imc[j - 1]) + 1) * (j**1.3 + np.sqrt(j))
            ang = np.sin(j * np.pi * rec[j - 1]) + np.cos(j**2 * np.pi * imc[j - 1]) + np.angle(t1) * np.real(t2)
        elif j % 4 == 2:
            mag = np.log(np.abs(rec[j - 1] + imc[j - 1]) + 1) * (np.exp(0.05 * j) + j)
            ang = np.cos(j * np.pi * rec[j - 1]) - np.sin(j**1.5 * np.pi * imc[j - 1]) + np.angle(t2) * np.imag(t1)
        elif j % 4 == 3:
            mag = np.log(np.abs(rec[j - 1] - imc[j - 1]) + 1) * (j**2 / (1 + j))
            ang = np.sin(j**2 * np.pi * rec[j - 1]) * np.cos(j * np.pi * imc[j - 1]) + np.angle(t1) * np.angle(t2)
        else:
            mag = np.log(np.abs(rec[j - 1]**2 + imc[j - 1]**2) + 1) * np.sqrt(j) * (1 + np.log(j))
            ang = np.sin(j * np.pi * rec[j - 1] / 2) + np.cos(j**3 * np.pi * imc[j - 1] / 3)
        cf[j - 1] = mag * np.exp(1j * ang)
    return cf.astype(np.complex128)
    
ALLOWED["p313"]=p313

def p314(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for k in range(1, n + 1):
        j = k
        r = (j**2 + np.real(t1) * np.imag(t2)) % 7 + 1
        angle = np.angle(t1) * np.sin(j * np.pi / r) + np.angle(t2) * np.cos(j * np.pi / (r + 1))
        magnitude = np.abs(t1)**(0.5 * j) + np.abs(t2)**(0.3 * (n - j + 1))
        cf[k - 1] = magnitude * np.exp(1j * angle) + np.conj(t1) * np.sin(j) - np.conj(t2) * np.cos(j)
        cf[k - 1] = cf[k - 1] * np.log(np.abs(cf[k - 1]) + 1) + np.prod(np.arange(1 * (j % 5) + 2)) + (j + r)
    return cf.astype(np.complex128)
    
ALLOWED["p314"]=p314

def p315(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        r = np.real(t1) * np.log(j + 1) + np.real(t2) * np.sin(j)
        q = np.imag(t1) * np.cos(j / 2) + np.imag(t2) * np.log(j + np.abs(t1 * t2))
        magnitude = np.abs(r)**(j % 5 + 1) + np.abs(t2)**(j % 3 + 2)
        angle = np.angle(q) * np.sin(j) - np.angle(t2) * np.cos(j / 3)
        cf[j - 1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p315"]=p315

def p316(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)

    # precompute constant angles
    ang1 = np.arctan2(t1.imag, t1.real)
    ang2 = np.arctan2(t2.imag, t2.real)

    fact = 1.0  # running factorial

    for j in range(1, n + 1):
        k = j + 3

        # real/imag linear combos (all real scalars)
        r  = t1.real * np.sin(j * (np.pi / 8.0)) + t2.real * np.cos(j * (np.pi / 5.0))
        im = t1.imag * np.cos(j * (np.pi / 7.0)) - t2.imag * np.sin(j * (np.pi / 9.0))

        # update factorial
        fact *= float(j)

        # magnitude (real)
        mag = np.log(abs(r * im) + 1.0) * (1.0 + np.sin(k * (np.pi / 4.0))) * (fact / float(n))

        # angle (real) using precomputed args
        ang = ang1 * np.cos(k * (np.pi / 6.0)) + ang2 * np.sin(k * (np.pi / 10.0))

        # polar synth
        cf[j - 1] = mag * (np.cos(ang) + 1j * np.sin(ang))

    # normalize once: cf /= sum(|cf|)
    s = 0.0
    for i in range(n):
        s += np.abs(cf[i])

    if np.isfinite(s) and s > 1e-12:
        inv = 1.0 / s
        for i in range(n):
            cf[i] *= inv

    return cf
    
ALLOWED["p316"]=p316

def p317(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(np.real(t1), np.real(t2), n)
    imc = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(1, n + 1):
        mag_part = np.log(np.abs(t1) + j) * np.sin(j) + np.cos(j**2)
        angle_part = np.angle(t1) * np.cos(j) + np.angle(t2) * np.sin(j)
        cf[j - 1] = mag_part * np.exp(1j * angle_part) + np.conj(t1) * np.prod(np.arange(1, j + 1)) / (j + 1)
    for k in range(1, n + 1):
        cf[k - 1] *= (1 + 0.05 * np.cos(k) + 0.03j * np.sin(k))
    return cf.astype(np.complex128)

ALLOWED["p317"]=p317

def p318(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec1 = np.real(t1)
    imc1 = np.imag(t1)
    rec2 = np.real(t2)
    imc2 = np.imag(t2)
    for j in range(1, n + 1):
        angle = np.sin(j * rec1) + np.cos(j * imc2) + np.angle(t1) * np.angle(t2) / (j + 0.1)
        magnitude = np.abs(t1)**j * np.log(np.abs(t2) + j) * (1 + (-1)**j * 0.5)
        cf[j - 1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p318"]=p318

def p319(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        r = np.real(t1) * np.log(j + 1) + np.real(t2) * np.sin(j * np.pi / 7)
        q = np.imag(t1) * np.cos(j * np.pi / 5) - np.imag(t2) * np.log(j + 2)
        magnitude = np.log(np.abs(r + 1j) + 1) * (1 + (j % 4))
        angle = np.angle(q) * np.sin(j) + np.angle(t2) * np.cos(j / 3)
        cf[j - 1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p319"]=p319

def p320(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(np.real(t1), np.real(t2), n)
    imc = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(n):
        mag_part1 = np.sin(rec[j] * np.pi / (j + 1))
        mag_part2 = np.cos(imc[j] * np.pi / (j + 2))
        mag = (mag_part1 + mag_part2) * np.log(np.abs(t1) + np.abs(t2) + j) * (1 + j / n)
        ang_part1 = np.angle(t1) * rec[j] / n
        ang_part2 = np.angle(t2) * imc[j] / n
        angle = ang_part1 - ang_part2 + np.sin(j * np.pi / 5)
        cf[j] = mag * np.exp(1j * angle)
    return cf.astype(np.complex128)
    
ALLOWED["p320"]=p320

def p321(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec_seq = np.linspace(np.real(t1), np.real(t2), n)
    imc_seq = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(n):
        mag_part = (j**1.8 + np.log(np.abs(t1) + np.abs(t2) + j)) * np.abs(np.sin(j * np.real(t1)) + np.cos(j * np.imag(t2)))
        angle_part = np.angle(t1) * np.log(j + 1) + np.angle(t2) * np.sin(j / 3)
        cf[j] = mag_part * (np.cos(angle_part) + 1j * np.sin(angle_part))
    return cf.astype(np.complex128)

ALLOWED["p321"]=p321

def p322(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(np.real(t1), np.real(t2), n)
    imc = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(n):
        r = rec[j]
        d = imc[j]
        mag = np.log(np.abs(r + 1j) + 1) * (1 + np.sin(j * np.pi / 5) * np.cos(j * np.pi / 3))
        angle = np.angle(d) * np.sin(j * np.pi / 4) + np.angle(t2) * np.cos(j * np.pi / 6)
        cf[j] = mag * (np.cos(angle) + np.sin(angle) * 1j)
    return cf.astype(np.complex128)

ALLOWED["p322"]=p322

def p323(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(n):
        mag = np.real(t1)**(j % 5 + 1) * np.log(np.abs(t2) + j) + np.imag(t1) * np.sin(j * np.pi / 7)
        angle = np.angle(t1) * np.cos(j / 3) + np.angle(t2) * np.sin(j / 4)
        cf[j] = mag * np.exp(1j * angle)
    return cf.astype(np.complex128)

ALLOWED["p323"]=p323

def p324(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    if abs(t1.real - t2.real) < 1e-10: return cf
    if abs(t1.imag - t2.imag) < 1e-10: return cf
    rec = np.linspace(np.real(t1), np.real(t2), n)
    imc = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(n):
        r = rec[j]
        d = imc[j]
        mag = np.log(np.abs(r**2) + 1) * (1 + np.sin(2 * np.pi * r * j)) * (1 + np.cos(np.pi * 1j * j))
        ang = np.angle(r + d * 1j) + np.sin(j) * np.log(np.abs(r + 1j)) - np.cos(j) * np.angle(r - 1j)
        cf[j] = mag * (np.cos(ang) + np.sin(ang) * 1j)
    return cf.astype(np.complex128)

ALLOWED["p324"]=p324

def p325(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(np.real(t1), np.real(t2), n)
    imc = np.linspace(np.imag(t1), np.imag(t2), n)
    for k in range(n):
        j = k % 5 + 1
        r = rec[k]
        d = imc[k]
        mag = np.log(np.abs(t1) + k) * np.sin(j * np.pi / 4) + np.prod(rec[max(0, k-3):k])**(1/3)
        angle = np.angle(d) * np.sin(j * np.pi / 6) + np.angle(t2) * np.cos(j * np.pi / 8) + np.imag(t2) / (k + 1)
        cf[k] = mag * np.exp(1j * angle)
    return cf.astype(np.complex128)
    
ALLOWED["p325"]=p325

def p326(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(np.real(t1), np.real(t2), n)
    imc = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(n):
        k = (j % 7) + 1
        r = j + k
        mag = np.log(np.abs(rec[j] * imc[j]) + 1) * np.sin(j / 3) + np.cos(j / 4) * np.abs(t1 + t2)
        angle = np.angle(t1) * np.cos(j * np.pi / 6) + np.angle(t2) * np.sin(j * np.pi / 8) + np.sin(j)
        cf[j] = mag * (np.cos(angle) + 1j * np.sin(angle)) + np.conj(t1) * np.sin(j) - np.conj(t2) * np.cos(j)
    return cf.astype(np.complex128)

ALLOWED["p326"]=p326

def p327(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)

    # precompute angles (Numba-safe)
    ang1 = np.arctan2(t1.imag, t1.real)
    ang2 = np.arctan2(t2.imag, t2.real)

    r1 = t1.real
    r2 = t2.real
    i1 = t1.imag
    i2 = t2.imag

    # safe fractional magnitudes
    part1_scale = np.sqrt(abs(r1))                 # (~ np.real(t1)**0.5 but safe)
    part2_scale = abs(i2) ** 0.3                   # (~ np.imag(t2)**0.3 but safe, real)

    denom = float(n - 1) if n > 1 else 1.0

    for j in range(n):
        # linear interpolation to replace linspace
        t = j / denom
        rec = r1 + (r2 - r1) * t
        imc = i1 + (i2 - i1) * t

        part1 = np.sin(j * (np.pi / 4.0)) * part1_scale
        part2 = np.cos(j * (np.pi / 3.0)) * part2_scale
        part3 = np.log(abs(rec * imc) + 1.0)       # real

        magnitude = part1 + part2 + part3

        # angle from fixed args; j-only trigs are real
        angle = ang1 * np.sin(float(j)) + ang2 * np.cos(float(j))

        # exp(i*angle) safely
        cf[j] = magnitude * (np.cos(angle) + 1j * np.sin(angle))

    return cf

ALLOWED["p327"]=p327

def p328(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(np.real(t1), np.real(t2), n)
    imc = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(n):
        k = j % 5 + 1
        r = (j**2 + np.sin(j) * np.cos(j)) / (np.log(np.abs(t1) + 1) + 1)
        if j <= n / 2:
            mag_variation = r * (1 + np.sin(j * np.pi / 7))
        else:
            mag_variation = r * (1 + np.cos(j * np.pi / 5))
        angle_variation = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j) + np.log(mag_variation + 1)
        cf[j] = mag_variation * np.exp(1j * angle_variation)
    return cf.astype(np.complex128)
    
ALLOWED["p328"]=p328

def p329(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(np.real(t1), np.real(t2), n)
    imc = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(n):
        mag_part1 = np.log(np.abs(rec[j] + imc[j]) + 1)
        mag_part2 = np.sin(j * np.pi / 4) * np.cos(j * np.pi / 3)
        magnitude = mag_part1 * (1 + mag_part2)
        angle_part1 = np.angle(np.conj(t1) * t2) + np.sin(j * np.pi / 5)
        angle_part2 = np.cos(j * np.pi / 7) * np.angle(rec[j] + 1j * imc[j])
        angle = angle_part1 + angle_part2
        cf[j] = magnitude * np.exp(1j * angle)
    return cf.astype(np.complex128)
    
ALLOWED["p329"]=p329

def p330(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec_seq = np.linspace(np.real(t1), np.real(t2), n)
    imc_seq = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(n):
        rec = rec_seq[j]
        imc = imc_seq[j]
        mag = np.log(np.abs(rec * j + imc * j**2) + 1) * np.sin(j * np.pi / 4) + \
                np.cos(j * np.pi / 3) * (np.abs(rec - imc) + 1)
        ang = np.angle(t1) * np.sin(j * np.pi / 6) + np.angle(t2) * np.cos(j * np.pi / 8) + np.log(j + 1)
        cf[j] = mag * (np.cos(ang) + np.sin(ang) * 1j)
    return cf.astype(np.complex128)
    
ALLOWED["p330"]=p330

def p331(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(n):
        mag_part1 = np.log(np.abs(t1) + j)
        mag_part2 = np.log(np.abs(t2) + n - j)
        mag = mag_part1 * mag_part2 + np.sin(j) * np.cos(j / 2)
        angle_part1 = np.angle(t1) * np.sin(j / 3)
        angle_part2 = np.angle(t2) * np.cos(j / 4)
        angle = angle_part1 + angle_part2 + np.sin(j * np.pi / 7)
        cf[j] = mag * (np.cos(angle) + np.sin(angle) * 1j)
    return cf.astype(np.complex128)
    
ALLOWED["p331"]=p331

def p332(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(np.real(t1), np.real(t2), n)
    imc = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(n):
        temp = 0 + 0j
        for k in range(1, 6):
            r = j * k
            temp += ((rec[j] + np.imag(t1) * k)**k * np.conj(t2)**r) / (np.log(np.abs(t1) + k) + 1)
        magnitude = np.log(np.abs(temp) + 1) * np.sin(j * np.angle(t1) + np.cos(j * np.angle(t2)))
        angle = np.angle(temp) + np.sin(j) * np.cos(k)
        cf[j] = magnitude * np.exp(1j * angle) + np.abs(t1)**j - np.abs(t2)**j
    return cf.astype(np.complex128)

ALLOWED["p332"]=p332

def p333(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(n):
        angle = np.sin(j * np.imag(t1)) + np.cos(j * np.real(t2)) + np.angle(t1) * np.angle(t2) / (j+1)
        magnitude = np.log(np.abs(t1) + 1) * (j**1.5) + np.exp(-j / (np.abs(t2) + 1)) * np.sqrt(j)
        cf[j] = magnitude * np.exp(1j * angle) + np.conj(magnitude * np.exp(-1j * angle / (j + 1)))
    return cf.astype(np.complex128)

ALLOWED["p333"]=p333
 
def p334(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    if abs(t1.real - t2.real) < 1e-10: return cf
    if abs(t1.imag - t2.imag) < 1e-10: return cf
    rec = np.linspace(np.real(t1), np.real(t2), n)
    imc = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(n):
        angle_part = np.sin(j * np.pi / 5) + np.cos(j * np.pi / 7 + np.angle(t1))
        magnitude_part = np.log(np.abs(rec[j] + imc[j]) + 1) * (np.abs(t1) + np.abs(t2)) / (j + 1)
        intricate_term = (rec[j]**3 - 2 * imc[j]**2) * np.cos(j * np.pi / 3)
        cf[j] = magnitude_part * np.exp(1j * angle_part) + np.conj(t1) * intricate_term
    return cf.astype(np.complex128)
    
ALLOWED["p334"]=p334

def p335(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec_t1 = np.real(t1)
    imc_t1 = np.imag(t1)
    rec_t2 = np.real(t2)
    imc_t2 = np.imag(t2)
    for j in range(n):
        angle_part = np.sin(j * np.pi / 7) * np.cos(j * np.pi / 5) + np.angle(t1) * np.angle(t2)
        magnitude_part = np.log(np.abs(t1) + 1) * (j**2) / (1 + j) + np.abs(t2)**(1 + np.sin(j))
        phase_shift = np.exp(1j * (angle_part + np.imag(t1) * np.real(t2) / (j+1) ))
        cf[j] = magnitude_part * phase_shift + np.conj(t1) * np.conj(t2) / (j + 1)
    for k in range(n):
        if k % 5 == 0:
            cf[k] *= (1 + 0.5 * np.cos(k * np.pi / 3))
        elif k % 3 == 0:
            cf[k] *= (1 + 0.3 * np.sin(k * np.pi / 4))
        else:
            cf[k] *= (1 + 0.2 * np.log(k + 1))
    cf = cf * np.prod(np.abs(cf))**(1/n) + np.sum(np.real(cf)) + 1j * np.sum(np.imag(cf))
    return cf.astype(np.complex128)

ALLOWED["p335"]=p335

def p336(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(n):
        k = (j % 7) + 1
        r = np.real(t1) * np.sin(j * np.pi / 6) + np.real(t2) * np.cos(j * np.pi / 5)
        s = np.imag(t1) * np.cos(j * np.pi / 4) - np.imag(t2) * np.sin(j * np.pi / 3)
        magnitude = np.log(np.abs(t1) + np.abs(t2) + j) * (1 + np.sin(r)) + np.abs(s)
        angle = np.angle(t1) * np.cos(r) + np.angle(t2) * np.sin(s)
        cf[j] = magnitude * np.exp(1j * angle) + np.conj(t1)**k * np.conj(t2)**k
    return cf.astype(np.complex128)

ALLOWED["p336"]=p336

def p337(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(35):
        phase = np.sin(j * np.real(t1)) + np.cos(j * np.imag(t2)) + np.log(np.real(t1) + 1) * np.angle(t2)
        magnitude = (np.abs(t1)**j + np.abs(t2)**(35 - j)) * (j % 7 + 1) / (j + 1)
        cf[j] = magnitude * (np.cos(phase) + 1j * np.sin(phase))
    return cf.astype(np.complex128)

ALLOWED["p337"]=p337

def p338(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for k in range(1, n + 1):
        real_part = np.real(t1) * np.cos(k * np.pi / 5) + np.real(t2) * np.sin(k * np.pi / 7)
        imag_part = np.imag(t1) * np.sin(k * np.pi / 6) - np.imag(t2) * np.cos(k * np.pi / 8)
        magnitude = np.sqrt(real_part**2 + imag_part**2) * np.log(np.abs(k) + 1) * (1 + np.sin(k))
        angle = np.arctan2(imag_part, real_part) + np.sin(k * np.angle(t1)) * np.cos(k * np.angle(t2))
        cf[k - 1] = magnitude * np.exp(1j * angle)
    for r in range(1, n + 1):
        cf[r - 1] += np.conj(cf[n - r]) * np.sin(r * np.pi / 10)
    return cf.astype(np.complex128)

ALLOWED["p338"]=p338

def p339(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(np.real(t1), np.real(t2), n)
    imc = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(1,n+1):
        angle = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j)
        magnitude = np.log(np.abs(t1) + np.abs(t2) + j) * (np.sin(j * np.pi / 7) + np.cos(j * np.pi / 5)**2)
        cf[j-1] = magnitude * np.exp(1j * angle)
        for k in range(1, 6):
            cf[j-1] += (np.real(t1) * rec[k - 1] - np.imag(t2) * imc[k - 1]) * np.exp(1j * np.sin(k))
        if j % 3 == 0:
            cf[j-1] *= (1 + 1j * np.log(np.abs(rec[j-1] + imc[j-1]) + 1))
        else:
            cf[j-1] += np.conj(rec[j-1] - imc[j-1]) * np.cos(j * np.pi / 6)
    for r in range(1, 5):
        cf *= (1 + 0.1 * r * np.sin(r * np.pi / 4))
    return cf.astype(np.complex128)
    
ALLOWED["p339"]=p339

def p340(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(n):
        mag_part = np.log(np.abs(t1) + j) * np.abs(np.sin(j * np.real(t2))) + np.sqrt(j) * np.abs(np.cos(j * np.imag(t1)))
        angle_part = np.angle(t1) * j + np.sin(j) + np.cos(j / 2)
        cf[j] = mag_part * np.exp(1j * angle_part) + np.conj(t2)**(j % 5 + 1)
    return cf.astype(np.complex128)
    
ALLOWED["p340"]=p340

def p341(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(n):
        mag = (np.abs(t1) * np.log(j + 1) + np.abs(t2) * np.sqrt(j)) / (1 + j**1.3)
        angle = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j / 2) + np.sin(j / 3 * np.pi)
        perturb = np.exp(1j * (np.sin(j / 4 * np.pi) + np.cos(j / 5 * np.pi)))
        cf[j] = mag * np.exp(1j * angle) * perturb
    return cf.astype(np.complex128)
    
ALLOWED["p341"]=p341

def p342(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, 36):
        k = j % 7 + 1
        r = np.real(t1) * np.log(j + 1)
        s = np.imag(t2) * np.sin(k * np.pi / 5)
        theta = np.angle(t1) * np.cos(k * np.pi / 3) + np.sin(k * np.pi / 4)
        magnitude = np.abs(t1)**k + np.log(np.abs(t2) + j)
        cf[j - 1] = (r + s * 1j) * np.exp(1j * theta) + np.conj(t1 + t2)**k * np.cos(j * np.angle(t1))
    return cf.astype(np.complex128)
    
ALLOWED["p342"]=p342

def p343(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec1 = np.real(t1)
    imc1 = np.imag(t1)
    rec2 = np.real(t2)
    imc2 = np.imag(t2)
    for j in range(n):
        mag_part = np.log(np.abs(rec1 * (j + 1) + imc2 / (j + 1))) + (np.sin(rec1 * (j + 1)) * np.cos(imc2 / (j + 1)))
        angle_part = np.angle(t1) * (j**0.5) + np.angle(t2) * np.sqrt(j) + np.sin(j * np.real(t1)) - np.cos(j * np.imag(t2))
        cf[j] = mag_part * (np.cos(angle_part) + 1j * np.sin(angle_part))
    return cf.astype(np.complex128)
    
ALLOWED["p343"]=p343

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

def p346(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(n):
        rec_part = np.real(t1) * np.sin(j / 2) + np.real(t2) * np.cos(j / 3)
        imc_part = np.imag(t1) * np.cos(j / 4) - np.imag(t2) * np.sin(j / 5)
        magnitude = np.log(np.abs(rec_part + imc_part) + 1) * (j**(1.2)) * (1 + np.sin(j * np.pi / 6))
        angle = np.angle(t1) * np.cos(j / 7) + np.angle(t2) * np.sin(j / 8)
        cf[j] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p346"]=p346

def p347(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(n):
        mag = 0
        angle = 0
        for k in range(1, int(np.floor(j / 5)) + 2):
            mag += np.real(t1) * np.sin(j * k) * np.log(k + 1)
            angle += np.imag(t2) * np.cos(j + k) / (k + 1)
        for r in range(1, 4):
            mag *= (1 + np.real(t1) * 0.1 * r)
            angle += np.angle(t2) * 0.05 * r
        cf[j] = mag * (np.cos(angle) + 1j * np.sin(angle))
    cf *= np.exp(1j * np.sin(np.abs(t1) * np.arange(1, n + 1)))
    return cf.astype(np.complex128)
    
ALLOWED["p347"]=p347

def p348(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(np.real(t1), np.real(t2), n)
    imc = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(n):
        magnitude = np.log(np.abs(t1) + j) * np.sin(j * np.pi / 7) + np.cos(j * np.pi / 5) * np.abs(t2)
        angle = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j)
        cf[j] = magnitude * np.exp(1j * angle)
    for k in range(n):
        cf[k] += (np.real(t1)**(k % 5 + 1) - np.imag(t2)**(k % 3 + 1)) * np.exp(1j * (np.sin(k) + np.cos(k)))
    return cf.astype(np.complex128)
    
ALLOWED["p348"]=p348

def p349(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    base_mag = np.abs(t1) * 2 + np.abs(t2)
    base_ang = np.angle(t1) - np.angle(t2)
    for j in range(n):
        mag = base_mag * j**1.2 + np.log(j + 1) * np.sqrt(n - j + 1)
        ang = base_ang * np.sin(j * np.pi / n) + np.cos(j * np.pi / (n / 2))
        cf[j] = mag * (np.cos(ang) + 1j * np.sin(ang))
    for k in range(n):
        increment = (np.real(t1) * np.real(t2) - np.imag(t1) * np.imag(t2)) + \
                        (np.real(t1) * np.imag(t2) + np.imag(t1) * np.real(t2)) * 1j
        angle_mod = np.sin(k) * np.cos(k) * np.angle(t1 + t2)
        cf[k] = cf[k] * (np.cos(angle_mod) + 1j * np.sin(angle_mod)) + np.conj(cf[k]) * np.log(np.abs(cf[k]) + 1)
    for r in range(1, int(np.floor(n / 5)) + 1):
        idx = (r * 7) % n
        adjustment = (np.real(t1)**2 - np.imag(t2)**2) + (2 * np.real(t1) * np.imag(t2)) * 1j
        cf[idx] += adjustment * np.sin(r)
    for m in range(n):
        cf[m] = cf[m] * np.exp(1j * np.sin(m * np.pi / 4)) + np.exp(1j * np.cos(m * np.pi / 3))
    return cf.astype(np.complex128)

ALLOWED["p349"]=p349

def p350(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(35):
        mag_part1 = np.log(np.abs(t1) + j) * np.sin(j * np.pi / 6)
        mag_part2 = np.log(np.abs(t2) + j) * np.cos(j * np.pi / 8)
        magnitude = mag_part1 + mag_part2 + np.sqrt(j)
        angle_part1 = np.angle(t1) * np.sin(j / 3)
        angle_part2 = np.angle(t2) * np.cos(j / 4)
        angle = angle_part1 + angle_part2 + np.sin(j) * np.cos(j / 2)
        cf[j] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p350"]=p350

def p351(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec1 = np.real(t1)
    imc1 = np.imag(t1)
    rec2 = np.real(t2)
    imc2 = np.imag(t2)
    for j in range(n):
        mag_part = np.log(np.abs(rec1 * j**1.5 + imc2 / (j + 2)) + 1) * (1 + np.sin(j * np.pi / 4))
        angle_part = np.angle(t1) * np.sin(j / 3) + np.angle(t2) * np.cos(j / 5) + np.sin(j * np.pi / 7)
        cf[j] = mag_part * (np.cos(angle_part) + 1j * np.sin(angle_part))
    return cf.astype(np.complex128)

ALLOWED["p351"]=p351

def p352(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(n):
        sum_real = 0
        sum_imag = 0
        for k in range(1, j + 1):
            sum_real += np.real(t1)**k * np.cos(k * np.pi / j)
            sum_imag += np.imag(t2)**k * np.sin(k * np.pi / j)
        mag = np.log(np.abs(sum_real) + 1) * np.abs(t1)**(0.5) + np.log(np.abs(sum_imag) + 1) * np.abs(t2)**(0.3)
        angle = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j / 2)
        cf[j] = mag * (np.cos(angle) + 1j * np.sin(angle))
    for r in range(n):
        cf[r] *= (1 + 0.1 * np.sin(r) + 0.1j * np.cos(r))
    return cf.astype(np.complex128)

ALLOWED["p352"]=p352

def p353(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(np.real(t1), np.real(t2), n)
    imc = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(n):
        mag_variation = np.log(np.abs(rec[j] + imc[j]) + 1) * (1 + np.sin(j) + np.cos(j / 2))
        angle_variation = np.angle(rec[j] + 1j * imc[j]) + np.sin(rec[j] * np.pi / (j + 1)) - np.cos(imc[j] * np.pi / (j + 1))
        cf[j] = mag_variation * np.exp(1j * angle_variation)
    return cf.astype(np.complex128)

ALLOWED["p353"]=p353

def p354(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(np.real(t1), np.real(t2), n)
    imc = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(n):
        part1 = np.sin(j * np.pi / 5) * np.cos(j * np.angle(t1)) 
        part2 = np.log(np.abs(t2) + j) * np.sin(j * np.pi / 3)
        part3 = np.cos(j * np.pi / 4) + np.sin(j * np.pi / 6)
        magnitude = np.abs(t1)**(0.5 * j) + np.log(np.abs(j) + 1) * (j % 7 + 1)
        angle = part1 + part2 + part3
        cf[j] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p354"]=p354

def p355(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(35):
        mag = 0
        angle = 0
        for k in range(1, 36):
            mag += (np.real(t1)**k * np.log(np.abs(t2) + k)) / (1 + k**2)
            angle += np.sin(k * np.angle(t1)) * np.cos(k * np.angle(t2))
        cf[j] = mag * (np.cos(angle) + 1j * np.sin(angle))
    for r in range(1, 36):
        cf[r - 1] *= np.exp(1j * (np.real(t1) * r - np.imag(t2) / (r + 1)))
    return cf.astype(np.complex128)
    
ALLOWED["p355"]=p355

def p356(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(35):
        temp = 0
        for k in range(1, j + 1):
            temp += (np.real(t1)**k * np.sin(k * np.angle(t2))) + (np.imag(t2)**k * np.cos(k * np.angle(t1)))
        magnitude = np.log(np.abs(t1) + np.abs(t2) + j) * temp
        angle = np.angle(t1) * np.cos(j) + np.angle(t2) * np.sin(j)
        cf[j] = magnitude * np.exp(1j * angle)
    for r in range(35):
        cf[r] += (np.real(t1) - np.real(t2)) * np.sin(r) + (np.imag(t1) + np.imag(t2)) * np.cos(r)
    return cf.astype(np.complex128)
    
ALLOWED["p356"]=p356

def p357(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(n):
        magnitude = np.log(np.abs(t1) + j) * np.sin(j / 2) + np.sqrt(np.abs(t2)) * np.cos(j / 3)
        angle = np.angle(t1) * j + np.angle(t2) * (n - j)
        cf[j] = magnitude * (np.cos(angle) + np.sin(angle) * 1j)
    for k in range(n):
        cf[k] += np.real(t1) * np.sin(k) - np.imag(t2) * np.cos(k)
    return cf.astype(np.complex128)
    
ALLOWED["p357"]=p357

def p358(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    real_diff = np.real(t2) - np.real(t1)
    im_diff = np.imag(t2) - np.imag(t1)
    for j in range(n):
        mag = 1 + np.abs(j - n / 2) * np.log(j + 1)
        angle = np.angle(t1) + (j / n) * np.angle(t2) + np.sin(j) * np.cos(j / 2)
        for k in range(1, 4):
            mag *= (1 + 0.1 * k * np.sin(j * k / n))
            angle += 0.5 * k * np.cos(j * k / n)
            for r in range(1, 3):
                mag += 0.05 * r * np.log(j + r)
                angle += 0.3 * r * np.sin(j * r / n)
        cf[j] = mag * (np.cos(angle) + np.sin(angle) * 1j)
    return cf.astype(np.complex128)
    
ALLOWED["p358"]=p358

def p359(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    t_conj = np.conj(t1) + np.conj(t2)
    for j in range(n):
        real_part = np.real(t1) * np.sin(j * np.pi / 7) + np.real(t2) * np.cos(j * np.pi / 5)
        imag_part = np.imag(t1) * np.cos(j * np.pi / 6) - np.imag(t2) * np.sin(j * np.pi / 4)
        magnitude = np.log(np.abs(t_conj) + j) * (np.abs(real_part) + np.abs(imag_part))
        angle = np.angle(t1) + np.angle(t_conj) * np.sin(j / 3) - np.angle(t2) * np.cos(j / 4)
        cf[j] = magnitude * (np.cos(angle) + np.sin(angle) * 1j)
    return cf.astype(np.complex128)
    
ALLOWED["p359"]=p359

#empty
def p360(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(np.real(t1), np.real(t2), n)
    imc = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(n):
        magnitude = np.log(np.abs(rec[j] * imc[j]) + 1) * (j**np.sin(j)) * (1 + np.cos(j))
        angle = np.sin(2 * np.pi * rec[j]) + np.cos(3 * np.pi * imc[j]) + np.log(np.abs(rec[j] + imc[j]) + 1)
        cf[j] = magnitude * np.exp(1j * angle)
    return cf.astype(np.complex128)
    
ALLOWED["p360"]=p360

def p361(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)

    a1 = np.abs(t1)
    a2 = np.abs(t2)
    re1, im1 = t1.real, t1.imag
    re2, im2 = t2.real, t2.imag

    half = n // 2
    EPS = 1e-12

    for j in range(n):
        # ----- magnitude term (guard log argument) -----
        if (j & 1) == 0:  # even
            x = a1 + j * re1
            base = x if (np.isfinite(x) and x > EPS) else 1.0
            mag = np.log(base) * (1.0 + np.sin(j * im2))
        else:              # odd
            x = a2 + j * im1
            base = x if (np.isfinite(x) and x > EPS) else 1.0
            mag = np.log(base) * (1.0 + np.cos(j * re2))

        # ----- angle term -----
        if j <= half:
            angle = np.sin(j * (np.pi * re1 / n)) + np.cos(j * (np.pi * im2 / n))
        else:
            angle = np.sin(j * (np.pi * re2 / n)) - np.cos(j * (np.pi * im1 / n))

        # polar synth
        cf[j] = mag * (np.cos(angle) + 1j * np.sin(angle))

    return cf
    
ALLOWED["p361"]=p361

def p362(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(n):
        k = (j * 4 + 5) % n + 1
        r = j // 6 + 1
        mag = np.log(np.abs(t1) + j**2) * np.abs(np.sin(j / 2 + k / 3)) + np.log(j + r)
        angle = np.angle(t1) * np.cos(j / (k + 1)) + np.angle(t2) * np.sin(j / (r + 2)) + np.real(t1) * np.imag(t2) / (j + 1)
        cf[j] = mag * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p362"]=p362

def p363(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec_seq = np.linspace(np.real(t1), np.real(t2), n)
    imc_seq = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(1, n + 1):
        angle = np.sin(j * np.angle(t1)) + np.cos(j * np.angle(t2)) + np.sin(j**2 / n)
        magnitude = (np.log(np.abs(t1) + 1) * j) + (np.abs(t2)**0.5 * np.sqrt(j))
        cf[j - 1] = magnitude * np.exp(1j * angle) + np.conj(t1) * np.conj(t2) / (j + 1)
    for k in range(1, n // 2 + 1):
        cf[k - 1] += (np.real(t1) * np.real(t2) + np.imag(t1) * np.imag(t2)) * np.sin(k)
    for r in range(n - 4, n + 1):
        cf[r - 1] += (np.abs(t1) * np.abs(t2) * r) * np.cos(r)
    return cf.astype(np.complex128)
    
ALLOWED["p363"]=p363

def p364(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(np.real(t1), np.real(t2), n)
    imc = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(1, n + 1):
        mag = np.log(np.abs(rec[j - 1] + imc[j - 1]) + 1) * (j**np.sin(j)) + np.sqrt(j) * np.abs(np.cos(j * np.pi / 3))
        ang = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j)
        cf[j - 1] = mag * (np.cos(ang) + 1j * np.sin(ang))
    return cf.astype(np.complex128)
    
ALLOWED["p364"]=p364

def p365(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    n = len(cf)
    rec = np.linspace(np.real(t1), np.real(t2), n)
    imc = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(1, n + 1):
        mag_part = np.log(np.abs(rec[j - 1] + imc[j - 1]) + 1) * j**(np.sin(j) + 1)
        angle_part = np.angle(t1) * np.cos(j / 3) + np.angle(t2) * np.sin(j / 4)
        cf[j - 1] = mag_part * (np.cos(angle_part) + 1j * np.sin(angle_part))
    for k in range(1, n + 1):
        modifier = np.exp(1j * (np.sin(k * np.pi / 5) + np.cos(k * np.pi / 7)))
        cf[k - 1] = cf[k - 1] * modifier + np.conj(cf[max(0, k - 2)]) / (k + 1)
    for r in range(1, n + 1):
        cf[r - 1] += np.real(t1) * np.imag(t2) / (r + 1) + np.real(t2) * np.imag(t1) / (n - r + 1)
    return cf.astype(np.complex128)

ALLOWED["p365"]=p365

def p366(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, 36):
        r = np.real(t1) * np.sin(j) + np.real(t2) * np.cos(j**2)
        im = np.imag(t1) * np.cos(j) - np.imag(t2) * np.sin(j**2)
        mag = np.log(np.abs(t1) + j) * np.sqrt(j) * (j % 5 + 1)
        angle = np.angle(t1) + np.angle(t2) + np.sin(j) * np.cos(j)
        cf[j - 1] = complex(r * np.cos(angle), im * np.sin(angle)) * mag
    for k in range(1, 36):
        cf[k - 1] += np.conj(cf[((k + 3) % 35)]) * np.sin(k / 2) - np.cos(k)
    return cf.astype(np.complex128)
    
ALLOWED["p366"]=p366

def p367(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        k = (j * 4 + 2) % n + 1
        r = (j + 3) % n + 1
        rec = np.real(t1) * np.sin(j) + np.real(t2) * np.cos(k)
        imc = np.imag(t1) * np.cos(r) - np.imag(t2) * np.sin(k)
        mag = np.log(np.abs(t1) + 1) * np.abs(rec) + np.sin(j) * np.cos(r) + (np.real(t1) * np.imag(t2) * j)
        angle = np.angle(t1) * k - np.angle(t2) * r + np.sin(j * np.pi / n)
        cf[j - 1] = mag * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p367"]=p367

def p368(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        mag = np.log(np.abs(t1) + j) * (1 + np.sin(j * np.angle(t2)))
        angle = np.cos(j * np.real(t1)) + np.sin(j * np.imag(t2))**2
        cf[j - 1] = mag * (np.cos(angle) + 1j * np.sin(angle))
    for k in range(1, n + 1):
        cf[k - 1] += (np.real(t1)**k - np.imag(t2)**k) * np.exp(1j * np.angle(t1 + k * t2))
    for r in range(1, n + 1):
        cf[r - 1] *= (np.abs(t1 + r * t2)**(1 + r / 10)) * np.cos(r * np.angle(t1 * t2))
    return cf.astype(np.complex128)
    
ALLOWED["p368"]=p368

def p369(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, 36):
        k = (j * 2 + 5) % 10 + 1
        r = np.real(t1)**((j % 4) + 1) + np.real(t2) * np.log(np.abs(t1) + j)
        im_part = np.imag(t2) + np.sin(j / 3) * np.cos(k / 2)
        angle = np.angle(t1) * j - np.angle(t2) * k + np.sin(j) * np.cos(k)
        magnitude = np.abs(t1)**(1 + (j % 5)) + np.abs(t2)**(2 + (k % 3))
        cf[j - 1] = magnitude * (np.cos(angle) + 1j * np.sin(angle)) + np.conj(t1 + t2)**k
    return cf.astype(np.complex128)

ALLOWED["p369"]=p369

def p370(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.real(t1) + (np.real(t2) - np.real(t1)) * (np.arange(1, n + 1)) / n
    imc = np.imag(t1) + (np.imag(t2) - np.imag(t1)) * (np.arange(1, n + 1)) / n
    for j in range(1, n + 1):
        mag = np.log(1 + rec[j - 1]**2 + imc[j - 1]**2) * np.sin(j * np.angle(t1) + np.cos(j * np.angle(t2)))
        angle = np.angle(t1) * np.real(t2) / (j + 1) + np.angle(t2) * np.imag(t1) / (j + 2)
        cf[j - 1] = mag * np.exp(1j * angle) + np.conj(t1) * np.prod(np.arange(1, j + 1)) / (j + 3)
    return cf.astype(np.complex128)
    
ALLOWED["p370"]=p370

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

def p372(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(np.real(t1), np.real(t2), n)
    imc = np.linspace(np.imag(t1), np.imag(t2), n)
    for k in range(1, n + 1):
        mag1 = np.log(np.abs(t1 * k) + 1) * np.sin(np.angle(t1) * k)
        mag2 = np.log(np.abs(t2 / k) + 1) * np.cos(np.angle(t2) / (k + 1))
        mag = mag1 + mag2
        angle = np.sin(rec[k - 1] * np.pi / (k + 2)) + np.cos(imc[k - 1] * np.pi / (k + 3))
        cf[k - 1] = mag * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p372"]=p372

def p373(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(np.real(t1), np.real(t2), 35)
    imc = np.linspace(np.imag(t1), np.imag(t2), 35)
    for j in range(1, 36):
        mag_part = np.log(np.abs(t1) + j) * np.sin(j * np.pi / 7) + np.cos(j * np.pi / 11) * np.real(t2)
        ang_part = np.angle(t1) + np.angle(t2) * j + np.sin(j * np.pi / 13)
        cf[j - 1] = (mag_part + np.imag(t1) * np.cos(j * np.pi / 5)) * np.exp(1j * ang_part) + np.conj(t2) * np.sin(j * np.pi / 17)
    return cf.astype(np.complex128)
    
ALLOWED["p373"]=p373

def p374(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        r = np.real(t1) + np.real(t2) * j
        im = np.imag(t1) - np.imag(t2) * j
        mag = np.log(np.abs(t1) + j**2) * (1 + np.sin(j * np.pi / 5) * np.cos(j * np.pi / 7))
        angle = np.angle(t1) * np.cos(j) + np.angle(t2) * np.sin(j)
        cf[j - 1] = mag * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p374"]=p374

def p375(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, 36):
        r = np.real(t1) * j**2 - np.real(t2) / (j + 1)
        im = np.imag(t2) * np.log(j + np.abs(t1)) + np.sin(j * np.angle(t1)) * np.cos(j * np.angle(t2))
        cf[j - 1] = (r + 1j * im) * np.exp(1j * (np.real(t1) * np.sin(j) + np.imag(t2) * np.cos(j)))
    return cf.astype(np.complex128)
    
ALLOWED["p375"]=p375

def p376(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, 36):
        mag = 0
        angle = 0
        for k in range(1, j + 1):
            mag += (np.real(t1) * np.log(k + 1)) / (k**0.5)
            angle += np.sin(k * np.angle(t2)) + np.cos(k * np.real(t1))
        cf[j - 1] = mag * np.exp(1j * angle) + np.conj(t1) * np.sin(j) - np.conj(t2) * np.cos(j)
    return cf.astype(np.complex128)
    
ALLOWED["p376"]=p376

def p377(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        angle = np.angle(t1) * np.log(j + 1) + np.sin(j) * np.angle(t2) / (j + 1)
        magnitude = np.abs(t1)**np.sqrt(j) + np.abs(t2)**(1 + 1/j) + np.log(np.abs(j - n/2) + 1)
        cf[j - 1] = magnitude * (np.cos(angle) + 1j * np.sin(angle)) + np.conj(t1) * np.real(t2) / (j + 1)
    return cf.astype(np.complex128)
    
ALLOWED["p377"]=p377

def p378(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(np.real(t1), np.real(t2), n)
    imc = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(1, n + 1):
        term1 = np.exp(1j * np.sin(5 * np.pi * rec[j - 1]))
        term2 = np.exp(1j * np.cos(3 * np.pi * imc[j - 1]))
        term3 = np.log(np.abs(rec[j - 1] * imc[j - 1]) + 1)
        mag = term3 * (j**2) + np.sum(np.arange(1, j % 4 + 2))
        angle = np.angle(term1) + np.angle(term2) + np.sin(j * np.angle(t1 + t2))
        cf[j - 1] = mag * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p378"]=p378

def p379(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    n = 35
    for j in range(1, n + 1):
        magnitude = np.log(np.abs(t1) + j) * (1 + np.sin(j * np.pi / 6)) + np.cos(j * np.pi / 4) * np.abs(t2)
        angle = np.angle(t1) * np.sin(j / 3) + np.angle(t2) * np.cos(j / 5)
        cf[j - 1] = magnitude * (np.cos(angle) + 1j * np.sin(angle)) + np.conj(t1)**(j % 7) * np.sin(j * np.angle(t2))
    return cf.astype(np.complex128)
    
ALLOWED["p379"]=p379

def p380(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    n = 35
    rec = np.linspace(np.real(t1), np.real(t2), n)
    imc = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(1, n + 1):
        r = rec[j - 1]
        m = imc[j - 1]
        mag = np.log(np.abs(r**2 + m**2) + 1) * (j + 1)**(np.sin(r) + np.cos(m))
        angle = np.sin(j * r) + np.cos(j * m) + np.angle(t1) * np.sin(m) - np.angle(t2) * np.cos(r)
        cf[j - 1] = mag * np.exp(1j * angle)
    return cf.astype(np.complex128)
    
ALLOWED["p380"]=p380

def p381(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec_t1 = np.real(t1)
    rec_t2 = np.real(t2)
    imc_t1 = np.imag(t1)
    imc_t2 = np.imag(t2)
    for j in range(1, n + 1):
        phase = np.sin(j * np.pi / 7) * np.cos(j * np.pi / 5)
        magnitude = np.log(np.abs(t1) + 1) * np.sin(j) + np.log(np.abs(t2) + 1) * np.cos(j)
        angle = np.angle(t1) * j**0.5 + np.angle(t2) / (j + 1)
        cf[j - 1] = magnitude * (np.cos(angle + phase) + 1j * np.sin(angle - phase)) + np.conj(t1)**j * np.real(t2) / (j + 2) + np.imag(t1 + t2) * np.cos(j * np.angle(t1)) * np.sin(j * np.angle(t2))
    return cf.astype(np.complex128)

ALLOWED["p381"]=p381

def p382(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        coeff_real = 0
        coeff_imag = 0
        for k in range(1, 6):
            coeff_real += np.real(t1)**k * np.sin(j * k * np.pi / 4) + np.real(t2)**(k / 2) * np.cos(j * k * np.pi / 3)
            coeff_imag += np.imag(t1)**k * np.cos(j * k * np.pi / 5) - np.imag(t2)**(k / 2) * np.sin(j * k * np.pi / 6)
        for r in range(1, 4):
            coeff_real += np.log(np.abs(t1 + r) + 1) * np.sin(j * r * np.pi / 7) * np.angle(t1 + r)
            coeff_imag += np.log(np.abs(t2 - r) + 1) * np.cos(j * r * np.pi / 8) * np.angle(t2 - r)
        cf[j - 1] = coeff_real + 1j * coeff_imag
    return cf.astype(np.complex128)

ALLOWED["p382"]=p382

def p383(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        mag_part = np.log(np.abs(t1) + j) * np.sin(j * np.real(t2)) + np.prod(np.arange(1, j + 1)) / (j + 1)
        angle_part = np.angle(t1) * np.cos(j * np.imag(t2)) + np.sin(j) * np.angle(t2)
        cf[j - 1] = mag_part * (np.cos(angle_part) + 1j * np.sin(angle_part)) + np.conj(t1)**j - np.conj(t2)**(n - j)
    return cf.astype(np.complex128)
    
ALLOWED["p383"]=p383

def p384(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, 36):
        r = np.real(t1) + (np.real(t2) - np.real(t1)) * j / 35
        im = np.imag(t1) + (np.imag(t2) - np.imag(t1)) * j / 35
        magnitude = np.log(np.abs(t1) + j) * np.abs(np.sin(r * j)) + np.prod(np.arange(1, (j % 5) + 1))
        angle = np.angle(t1) * np.cos(j) + np.angle(t2) * np.sin(im * j)
        cf[j - 1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p384"]=p384

def p385(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for k in range(1, n + 1):
        real_part = np.real(t1)**k + np.real(t2)**(n - k)
        imag_part = np.imag(t1)**(k % 5 + 1) - np.imag(t2)**(k // 3 + 1)
        magnitude = (np.abs(t1) + np.abs(t2))**k * np.log(k + 1)
        angle = np.angle(t1) * np.sin(k) + np.angle(t2) * np.cos(k)
        cf[k - 1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p385"]=p385

def p386(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, 36):
        real_part = np.real(t1) * np.sin(j) + np.real(t2) * np.cos(j / 2)
        imag_part = np.imag(t1) * np.cos(j) - np.imag(t2) * np.sin(j / 2)
        magnitude = np.sqrt(real_part**2 + imag_part**2) * np.log(j + np.abs(t1) + np.abs(t2))
        angle = np.angle(t1) * np.sqrt(j) + np.angle(t2) * np.cos(j)
        cf[j - 1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p386"]=p386

def p387(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for k in range(1, n + 1):
        j = k
        mag_part = np.log(np.abs(t1) + k) * np.sin(j) + np.cos(j) * np.log(np.abs(t2) + 1)
        angle_part = np.angle(t1) * j**0.5 + np.angle(t2) * np.log(j + 1) + np.sin(j * np.real(t1)) - np.cos(j * np.imag(t2))
        cf[j - 1] = mag_part * np.exp(1j * angle_part)
    return cf.astype(np.complex128)

ALLOWED["p387"]=p387

def p388(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        r_part = np.real(t1) * j + np.real(t2) / (j + 1)
        i_part = np.imag(t1) * np.sin(j) + np.imag(t2) * np.cos(j)
        mag = np.log(np.abs(t1) + j) * (j % 5 + 1)
        angle = np.angle(t1) * np.sin(j / 3) + np.angle(t2) * np.cos(j / 4)
        cf[j - 1] = (r_part + 1j * i_part) * np.exp(1j * angle) * mag
    return cf.astype(np.complex128)

ALLOWED["p388"]=p388

def p389(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        coeff_real = 0
        coeff_imag = 0
        for k in range(1, 6):
            coeff_real += np.real(t1)**k * np.cos(k * np.angle(t2) + j)
            coeff_imag += np.imag(t2)**k * np.sin(k * np.angle(t1) + j)
        for r in range(1, 4):
            coeff_real += np.log(np.abs(t1 + t2) + 1) * np.real(t1) / r
            coeff_imag += np.log(np.abs(t1 - t2) + 1) * np.imag(t2) / r
        magnitude = np.sqrt(coeff_real**2 + coeff_imag**2) * (1 + j / n)
        angle = np.arctan2(coeff_imag, coeff_real) + np.sin(j) * np.cos(j / 2)
        cf[j - 1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p389"]=p389

def p390(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)

    r1 = math.hypot(t1.real, t1.imag)
    r2 = math.hypot(t2.real, t2.imag)
    ang1 = math.atan2(t1.imag, t1.real)

    # main coefficients
    for j in range(1, n + 1):
        jf = float(j)
        angle = ang1 * jf + math.sin(jf * math.pi / 4.0) * math.cos(jf * math.pi / 6.0)
        # |t1|**j + log(|t2|+1) * (j%5 + 1)
        magnitude = (r1 ** j) + math.log(r2 + 1.0) * (float(j % 5) + 1.0)
        c = magnitude
        cf[j - 1] = np.complex128(c * math.cos(angle) + 1j * c * math.sin(angle))

    # lightweight deterministic PRNG (LCG) for indices (Numba-safe)
    # seed derived from t1,t2; keep it deterministic across calls
    s = np.uint64(1469598103934665603)  # FNV offset basis
    s ^= np.uint64(np.abs(t1.real) * 1e6 + 3.0)
    s ^= np.uint64(np.abs(t1.imag) * 1e6 + 5.0)
    s ^= np.uint64(np.abs(t2.real) * 1e6 + 7.0)
    s ^= np.uint64(np.abs(t2.imag) * 1e6 + 11.0)
    A = np.uint64(1664525)
    C = np.uint64(1013904223)
    MASK = np.uint64(0xFFFFFFFFFFFFFFFF)  # 64-bit

    # first perturbation block (≈ n//5 picks)
    m1 = n // 5
    for k in range(1, m1 + 1):
        s = (A * s + C) & MASK
        idx = int(s % np.uint64(n))
        # multiply by exp(i*sin(k)) + add conj
        ph = math.sin(float(k))
        co = math.cos(ph)
        si = math.sin(ph)
        v = cf[idx]
        v = v * np.complex128(co + 1j * si)
        # manual conjugate (Numba-safe)
        v_conj = np.complex128(v.real - 1j * v.imag)
        cf[idx] = v + v_conj

    # second perturbation block (≈ n//7 picks)
    m2 = n // 7
    for r in range(1, m2 + 1):
        s = (A * s + C) & MASK
        idx = int(s % np.uint64(n))
        cf[idx] += np.complex128(r1 * math.cos(r * math.pi / 3.0) - 1j * math.sin(r * math.pi / 4.0))

    return cf
    
ALLOWED["p390"]=p390

def gpt1(z, a, state):
    # Interface-compatible: z is complex pair, a/state unused
    t1, t2 = z[0], z[1]

    n = 121  # pick any degree; this gives a nice dense structure
    cf = np.zeros(n, dtype=np.complex128)

    # Basic polar bits (Numba-safe)
    r1 = math.hypot(t1.real, t1.imag)
    r2 = math.hypot(t2.real, t2.imag)
    ang1 = math.atan2(t1.imag, t1.real)
    ang2 = math.atan2(t2.imag, t2.real)

    # Normalized controls in [0,1)
    denom = 1.0 + r1 + r2
    u = r1 / denom
    v = r2 / denom

    # Phase/chirp parameters (quasi-periodic & incommensurate)
    alpha = ang1 + 0.37 * (2.0 * u - 1.0)              # linear term
    beta  = ang2 + 0.53 * (2.0 * v - 1.0)              # quadratic chirp
    gamma = 0.11 + 0.5 * (ang1 - ang2)                 # sinusoidal wobble

    # Envelope shaping
    kappa = 0.6 + 0.4 * math.cos(ang1 - ang2)          # decay/boost
    dr = abs(t1.real - t2.real) + 1e-12                # avoid 0-div

    # Conjugate once (Numba-safe)
    t2_conj = np.complex128(t2.real - 1j * t2.imag)

    # Ensure a proper constant term so the poly isn’t identically zero
    cf[0] = 1.0 + 0.0j

    for j in range(1, n):
        jf = float(j)
        x = jf / float(n)

        # Smooth amplitude envelope: log1p for stability, mild decay by x
        env = (math.sqrt(max(1e-16, math.log1p(r1 + r2 + jf))) /
               (1.0 + 0.3 * dr)) * math.exp(-kappa * 0.35 * x)

        # Slow AM from two incommensurate trigs
        am = (1.0 + 0.28 * math.cos(jf * (0.7 + u)) *
                     (1.0 + 0.22 * math.sin(jf * (0.5 + v))))

        # Chirped phase: linear + quadratic + gentle wobble
        phase = (alpha * jf) + (beta * (jf * jf) / float(n)) + gamma * math.sin((2.0 * jf + 1.0) * 0.5)

        c = env * am
        re = c * math.cos(phase)
        im = c * math.sin(phase)
        cf[j] = np.complex128(re + 1j * im)

        # Sparse “comb” perturbation: inject conjugate spikes every 11th coef
        if (j % 11) == 0:
            cf[j] += t2_conj * (0.12 * (jf / n))

        # Gentle palindromic bias to encourage interesting symmetry
        if j >= 2 and j <= n - 2:
            mirror = n - 1 - j
            # blend a bit with the mirrored index to seed self-inversive flavor
            cf[j] = 0.92 * cf[j] + 0.08 * np.conjugate(cf[mirror])  # conj is njit-safe on scalars

    return cf

ALLOWED["gpt1"]=gpt1

def p391(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)

    # Precompute real and imag interpolation endpoints
    r1, r2 = t1.real, t2.real
    i1, i2 = t1.imag, t2.imag
    denom = float(n - 1) if n > 1 else 1.0

    # Precompute angles safely (Numba-safe)
    ang1 = np.arctan2(t1.imag, t1.real)
    ang2 = np.arctan2(t2.imag, t2.real)

    EPS = 1e-12

    # ----- build rec / imc linearly on the fly -----
    for j in range(1, n + 1):
        # linear interpolation
        t = (j - 1) / denom
        rec = r1 + (r2 - r1) * t
        imc = i1 + (i2 - i1) * t

        # magnitude
        val = abs(rec + imc) + 1.0
        if not np.isfinite(val) or val <= 0.0:
            val = 1.0
        mag = np.log(val) * np.sqrt(float(j)) * (1.2 if (j % 2) == 0 else 0.8)

        # angle
        angle = ang1 * np.sin(j / 3.0) + ang2 * np.cos(j / 4.0) + np.sin(j) * np.cos(j / 2.0)

        cf[j - 1] = mag * (np.cos(angle) + 1j * np.sin(angle))

    # ----- second loop: cumulative product-like term -----
    for k in range(1, n + 1):
        # compute rec[:k]+imc[:k] product manually (simulate np.prod)
        prod_val = 1.0
        for m in range(1, k + 1):
            t = (m - 1) / denom
            rec_m = r1 + (r2 - r1) * t
            imc_m = i1 + (i2 - i1) * t
            prod_val *= (rec_m + imc_m)
            # clamp overflow
            if not np.isfinite(prod_val):
                prod_val = 1.0
                break

        base = abs(prod_val)
        if base <= EPS or not np.isfinite(base):
            base = 1.0

        root_val = base ** (1.0 / float(k))
        # angle of cf[k-1]
        theta = np.arctan2(cf[k - 1].imag, cf[k - 1].real)
        cf[k - 1] += root_val * (np.cos(theta) + 1j * np.sin(theta))

    # ----- third loop: conjugate scaling -----
    for r in range(1, n + 1):
        t = (r - 1) / denom
        rec_r = r1 + (r2 - r1) * t
        imc_r = i1 + (i2 - i1) * t
        num = np.conj(rec_r - imc_r)
        den = 1.0 + abs(rec_r + imc_r)
        if not np.isfinite(den) or den <= EPS:
            den = 1.0
        cf[r - 1] *= num / den

    return cf
    
ALLOWED["p391"]=p391

def p392(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        mag = 0
        ang = 0
        for k in range(1, 6):
            term1 = np.real(t1) * np.log(k + 1) * np.sin(j * k)
            term2 = np.imag(t2) * np.cos(j + k)
            mag += term1**2 + term2**2
            ang += np.angle(t1) * np.sin(k) - np.angle(t2) * np.cos(k)
        for r in range(1, 4):
            mag += np.abs(t1 + r) * np.sqrt(r) / (j + r)
            ang += np.sin(r * np.pi / j) * np.cos(r)
        varied_mag = mag * (1 + j / n)
        varied_ang = ang + np.log(np.abs(t1) + 1) - np.log(np.abs(t2) + 1)
        cf[j - 1] = varied_mag * (np.cos(varied_ang) + 1j * np.sin(varied_ang))
    return cf.astype(np.complex128)
    
ALLOWED["p392"]=p392

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

def p394(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(1, 36):
        k = j
        r = np.real(t1) * np.sin(k) + np.real(t2) * np.cos(k**2)
        im = np.imag(t1) * np.cos(k / 3) - np.imag(t2) * np.sin(k / 4)
        mag = np.log(np.abs(t1) + np.abs(t2) + k) * (1 + np.sin(k) * np.cos(k))
        ang = np.angle(t1) * np.sin(k / 5) + np.angle(t2) * np.cos(k / 7)
        cf[j - 1] = mag * (np.cos(ang) + 1j * np.sin(ang))
    return cf.astype(np.complex128)
    
ALLOWED["p394"]=p394

def p395(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(np.real(t1), np.real(t2), n)
    imc = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(1, n + 1):
        mag_part1 = np.log(np.abs(rec[j - 1] * t1) + 1)
        mag_part2 = (np.cos(rec[j - 1]) + np.sin(imc[j - 1])) / (j ** 0.5)
        mag = mag_part1 * mag_part2 * (1 + np.sin(j * np.pi / 7))
        ang_part1 = np.angle(t1) * np.sin(j / 2)
        ang_part2 = np.angle(t2) * np.cos(j / 3)
        ang = ang_part1 + ang_part2 + np.log(np.abs(j) + 1)
        cf[j - 1] = mag * np.exp(1j * ang) + np.conj(t1 + t2) * np.sin(j) / (j + 1)
    return cf.astype(np.complex128)

ALLOWED["p395"]=p395

def p396(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        r_part = np.real(t1) * np.log(j + 1) + np.imag(t2) * np.sin(j * np.pi / 8)
        i_part = np.real(t2) * np.cos(j * np.pi / 7) - np.imag(t1) * np.sin(j * np.pi / 5)
        mag = np.sqrt(r_part**2 + i_part**2) + np.prod(np.arange(1, j + 1)) / (j + 2)
        temp = r_part + 1j * i_part
        theta = np.angle(temp) + np.cos(j * np.pi / 6)
        cf[j - 1] = mag * (np.cos(theta) + 1j * np.sin(theta))
    return cf.astype(np.complex128)
    
ALLOWED["p396"]=p396

def p397(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(np.real(t1), np.real(t2), n)
    imc = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(1, n + 1):
        magnitude_part = np.log(np.abs(t1)**abs(j - n/2) + np.abs(t2)**(np.abs(j - n/3)) + 1)
        angle_part = np.sin(j * np.angle(t1)) * np.cos(j * np.angle(t2)) + np.sin(np.imag(t1) * np.pi / j)
        variation = (np.cos(np.angle(t1) * j) * np.sin(np.angle(t2) * (n - j)) if j % 3 == 0 
                        else np.sin(np.angle(t1) * (j + 1)) - np.cos(np.angle(t2) * (j + 2)))
        cf[j - 1] = magnitude_part * np.exp(1j * (angle_part + variation))
    return cf.astype(np.complex128)

ALLOWED["p397"]=p397

def p398(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, 36):
        r = np.real(t1) + (np.real(t2) * j) / 35
        d = np.imag(t1) - (np.imag(t2) * j) / 35
        mag = np.log(np.abs(r + 1j) + 1) * np.sin(j * np.pi / 4) + np.cos(j * np.pi / 6)
        ang = np.angle(d) * np.cos(j) + np.angle(t2) * np.sin(j)
        cf[j - 1] = mag * (np.cos(ang) + 1j * np.sin(ang))
    return cf.astype(np.complex128)
    
ALLOWED["p398"]=p398

def p399(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, 36):
        k = j % 7 + 1
        r = j // 5 + 1
        term1 = np.real(t1)**k * np.sin(k * np.pi / 3)
        term2 = np.imag(t2)**r * np.cos(r * np.pi / 4)
        magnitude = term1 + term2 + np.log(np.abs(t1) + np.abs(t2) + j)
        angle = np.angle(t1) * r - np.angle(t2) / k + np.sin(j) * np.cos(j)
        cf[j - 1] = magnitude * np.exp(1j * angle) + np.conj(t1 * t2) * (np.abs(t1) + np.abs(t2)) / j
        if j % 4 == 0:
            cf[j - 1] *= (np.sin(j * np.pi / 5) + np.cos(j * np.pi / 6))
        if j % 6 == 0:
            cf[j - 1] += np.real(t1)**2 - np.imag(t2)**2
    return cf.astype(np.complex128)
    
ALLOWED["p399"]=p399

def p400(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec_seq = np.linspace(np.real(t1), np.real(t2), n)
    imc_seq = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(1, n + 1):
        term1 = np.sin(rec_seq[j - 1] * np.pi / n) * np.cos(imc_seq[j - 1] * np.pi / (2 * n))
        term2 = np.log(np.abs(rec_seq[j - 1] + imc_seq[j - 1]) + 1)
        term3 = np.abs(t1)**j / (1 + j)
        term4 = np.angle(t2) * j
        magnitude = term1 + term2 + term3
        angle = term4 + np.sin(j * imc_seq[j - 1]) - np.cos(j * rec_seq[j - 1])
        cf[j - 1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p400"]=p400

def p401(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec1 = np.real(t1)
    rec2 = np.real(t2)
    imc1 = np.imag(t1)
    imc2 = np.imag(t2)
    for j in range(1, n + 1):
        magnitude = (np.abs(t1) ** (j / 2)) * np.log(np.abs(j) + 1) + (np.abs(t2) ** (np.sqrt(j))) * np.sin(j)
        angle = np.angle(t1) * np.cos(j) + np.angle(t2) * np.sin(j / 2) + np.log(np.abs(j) + 1)
        cf[j - 1] = magnitude * np.exp(1j * angle)
    for k in range(1, n + 1):
        cf[k - 1] += (0.5 * np.real(t1) * np.sin(k)) + (0.3 * np.imag(t2) * np.cos(k))
    return cf.astype(np.complex128)
    
ALLOWED["p401"]=p401

def p402(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        temp_mag = 0
        temp_angle = 0
        for k in range(1, j + 1):
            temp_mag += np.log(np.abs(t1) * k + 1) * np.sin(k * np.real(t2))
            temp_angle += np.cos(k * np.imag(t1)) * np.angle(t2 + k)
        cf[j - 1] = temp_mag * (np.cos(temp_angle) + np.sin(temp_angle) * 1j)
    for r in range(1, n + 1):
        cf[r - 1] = cf[r - 1] * (np.abs(t2)**(r / 2)) + np.conj(t1)**r
    return cf.astype(np.complex128)
    
ALLOWED["p402"]=p402

def p403(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(np.real(t1), np.real(t2), n)
    imc = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(1, n + 1):
        phase = np.sin(j * np.pi / 7) * np.cos(j * np.pi / 5) + np.log(np.abs(t1 + t2) + 1)
        magnitude = (np.real(t1)**j + np.imag(t2)**j) * np.sin(j) + np.cos(j * np.pi / 3)
        cf[j - 1] = magnitude * np.exp(1j * phase) + np.conj(magnitude * np.exp(1j * phase))
    return cf.astype(np.complex128)
    
ALLOWED["p403"]=p403

def p404(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, 36):
        mag_part1 = np.real(t1) * np.log(j + 1)
        mag_part2 = np.abs(t2)**np.sin(j)
        mag_part3 = (1 + (j / 35))
        magnitude = mag_part1 + mag_part2 * mag_part3
        ang_part1 = np.angle(t1) * j
        ang_part2 = np.cos(j * np.pi / 7)
        angle = ang_part1 + ang_part2
        cf[j - 1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p404"]=p404

def p405(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for k in range(1, n + 1):
        phase = np.sin(k * np.angle(t1)) + np.cos(k * np.angle(t2))
        magnitude = np.log(np.abs(t1) + k) * np.exp(-k / (np.abs(t2) + 1)) + np.sqrt(k) * np.abs(t1 - t2)
        cf[k - 1] = magnitude * (np.cos(phase) + np.sin(phase) * 1j) + np.conj(t1) * np.sin(k) * np.cos(k)
    return cf.astype(np.complex128)
    
ALLOWED["p405"]=p405

def p406(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(np.real(t1), np.real(t2), n)
    imc = np.linspace(np.imag(t1), np.imag(t2), n)
    for r in range(1, n + 1):
        term1 = np.sin(r * np.pi / 8) * np.real(t1) * rec[r - 1]
        term2 = np.cos(r * np.pi / 6) * np.imag(t2) * imc[r - 1]
        term3 = np.log(np.abs(rec[r - 1] + imc[r - 1]) + 1)
        mag = term1 + term2 + term3
        angle = np.angle(t1) * np.sin(r * np.pi / 5) + np.angle(t2) * np.cos(r * np.pi / 7) + imc[r - 1] / 3
        cf[r - 1] = mag * (np.cos(angle) + np.sin(angle) * 1j)
    return cf.astype(np.complex128)
    
ALLOWED["p406"]=p406

def p407(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(np.real(t1), np.real(t2), n)
    imc = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(1, n + 1):
        mag = np.abs(t1)**j / (j + 1) + np.abs(t2)**(n - j) * np.sin(j) + np.log(np.abs(rec[j - 1] + imc[j - 1]) + 1)
        angle = np.angle(t1) * np.cos(j) + np.angle(t2) * np.sin(j) + np.sin(j * np.pi / n)
        cf[j - 1] = mag * (np.cos(angle) + 1j * np.sin(angle))
    for k in range(1, n + 1):
        cf[k - 1] += np.conj(cf[(k % n)]) * np.cos(k) - np.real(cf[k - 1]) * np.sin(k)
    for r in range(1, n + 1):
        if r % 3 == 0:
            cf[r - 1] *= (1 + 0.5 * np.sin(r)) 
        else:
            cf[r - 1] += 0.3 * np.imag(cf[r - 1]) * np.cos(r)
    return cf.astype(np.complex128)
    
ALLOWED["p407"]=p407

def p408(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec_t1 = t1.real
    imc_t1 = t1.imag
    rec_t2 = t2.real
    imc_t2 = t2.imag
    for j in range(1, n + 1):
        angle_part = np.sin(j * np.pi * rec_t1) + np.cos(j * np.pi * imc_t2)
        magnitude_part = np.log(np.abs(j * rec_t2 + 1)) * np.sqrt(j) + np.abs(t1) ** 0.5
        phase_shift = np.angle(t1) * np.angle(t2) / j
        cf[j - 1] = (
            magnitude_part * np.exp(1j * (angle_part + phase_shift))
            + np.conj(t1) * np.sin(j * np.angle(t2))
            - np.cos(np.abs(t1))
        )
    for k in range(1, n // 2 + 1):
        idx = k * 2
        if idx <= n:
            perturbation = np.exp(1j * (np.sin(k) + np.cos(k)))
            cf[idx - 1] = cf[idx - 1] * perturbation + np.log(np.abs(cf[idx - 1]) + 1)
    for r in range(1, n + 1):
        scaling_factor = (r ** 2 + np.sqrt(r)) / (np.abs(t1) + np.abs(t2) + 1)
        cf[r - 1] = cf[r - 1] * scaling_factor
    return cf.astype(np.complex128)
    
ALLOWED["p408"]=p408

def p409(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, 36):
        # Calculate the real part
        r_part = np.real(t1) * j**2 - np.real(t2) / (j + 1)
        # Calculate the imaginary part
        i_part = np.imag(t1) * np.log(np.abs(j) + 1) + np.imag(t2) * np.sin(j * np.pi / 7)
        # Calculate magnitude
        magnitude = np.sqrt(r_part**2 + i_part**2) * (1 + 0.1 * j)
        # Calculate angle
        angle = np.angle(t1) + np.angle(t2) + np.sin(j) - np.cos(j / 3)
        # Assign the complex coefficient
        cf[j-1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p409"]=p409

def p410(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, 36):
        # Calculate magnitude parts
        mag_part1 = np.log(np.abs(t1) + j**1.5)
        mag_part2 = np.sin(j * np.real(t2)) * np.cos(j * np.imag(t1))
        magnitude = mag_part1 * (1 + mag_part2**2)
        # Calculate angle parts
        angle_part1 = np.angle(t1) + np.angle(t2) * j
        angle_part2 = np.log(np.abs(t1 * t2) + 1)
        angle = angle_part1 + angle_part2
        # Assign the complex coefficient
        cf[j-1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p410"]=p410

def p411(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, 36):
        # Calculate real and imaginary components
        rec = np.real(t1) * j + np.real(t2) * (35 - j)
        imc = np.imag(t1) - np.imag(t2) * np.sin(j * np.pi / 8)
        # Calculate magnitude
        mag = np.log(np.abs(t1) + 1) * (np.sin(j * np.pi / 5) + np.cos(j * np.pi / 7)) * (1 + j % 6)
        # Calculate angle
        ang = np.angle(t1) * np.sin(j * np.pi / 9) + np.angle(t2) * np.cos(j * np.pi / 11) + np.sin(j * np.pi / 13)
        # Assign the complex coefficient
        cf[j-1] = mag * (np.cos(ang) + 1j * np.sin(ang))
    return cf.astype(np.complex128)

ALLOWED["p411"]=p411 

def p412(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        k = (j % 7) + 1
        r = np.floor(j / 5) + 1
        angle = (np.sin(k * np.real(t1) + r * np.imag(t2)) +
                    np.cos(k * np.imag(t1) - r * np.real(t2)) +
                    np.angle(t1) * np.angle(t2) / j)
        magnitude = (np.log(np.abs(t1) + 1) * (j**0.5 + r) +
                        np.abs(t2) * r**1.2)
        # Assign the complex coefficient using exponential form
        cf[j-1] = magnitude * np.exp(1j * angle)
    return cf.astype(np.complex128)

ALLOWED["p412"]=p412

def p413(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        # Calculate angle and magnitude
        angle = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j / 3)
        magnitude = np.abs(t1)**j * np.log(np.abs(t2) + 1)**(n - j) * (j % 5 + 1)
        # Assign the complex coefficient
        cf[j-1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    # Modify coefficients symmetrically
    for k in range(1, int(n/2) + 1):
        r = k**2 + np.sqrt(k)
        cf[k-1] *= np.exp(1j * r)
        cf[-k] *= np.conj(np.exp(1j * r))
    # Add variations based on index
    for r in range(1, n + 1):
        cf[r-1] += 0.1 * r * np.exp(-1j * r / n)
    return cf.astype(np.complex128)
    
ALLOWED["p413"]=p413

def p414(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for k in range(1, n + 1):
        j = (k % 5) + 1
        r = (np.real(t1) * np.sin(k) + np.real(t2) * np.cos(j)) / (j + k)
        mag = (np.log(np.abs(t1) + np.abs(t2) + k**1.5) * np.sin(r * np.pi / j) +
                np.cos(r * np.pi / (k + 1))**2)
        angle = (np.angle(t1) * np.cos(j / (k + 1)) +
                    np.angle(t2) * np.sin(r))
        # Assign the complex coefficient
        cf[k-1] = mag * (np.cos(angle) + 1j * np.sin(angle)) * (1 + 0.05 * k**2)
    return cf.astype(np.complex128)

ALLOWED["p414"]=p414

def p415(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n +1):
        # Calculate magnitude and angle
        mag = (np.log(np.abs(t1) + np.abs(t2) + j) * (1 + np.sin(j * np.pi / 7)) *
                (1 + np.cos(j * np.pi / 5)))
        ang = np.angle(t1) * np.sin(j / 3) + np.angle(t2) * np.cos(j / 4)
        # Assign the complex coefficient with additional terms
        cf[j-1] = mag * (np.cos(ang) + 1j * np.sin(ang)) + \
                    np.conj(t1) * np.sin(j) - np.conj(t2) * np.cos(j)
    return cf.astype(np.complex128)

ALLOWED["p415"]=p415

def p416(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, 36):
        # Calculate magnitude parts
        mag_part1 = np.log(np.abs(t1) * j + 1)
        mag_part2 = np.sin(j * np.pi / 3) * np.abs(t2)
        mag = mag_part1 + mag_part2 * np.cos(j / 2)
        # Calculate angle parts
        angle_part1 = np.angle(t1) * np.cos(j / 4)
        angle_part2 = np.sin(j * np.pi / 5) * np.angle(t2)
        angle = angle_part1 + angle_part2 + np.sin(j) * np.cos(j / 3)
        # Assign the complex coefficient
        cf[j-1] = mag * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p416"]=p416

def p417(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    r1 = np.real(t1)
    i1 = np.imag(t1)
    r2 = np.real(t2)
    i2 = np.imag(t2)
    for j in range(1, n +1):
        # Calculate magnitude and angle
        mag = (np.log(np.abs(r1 + j) + 1) * (j**1.5 + np.sin(j * r2)) *
                (1 + np.abs(np.cos(j * i1))))
        ang = np.angle(t1) * np.sin(j * r2) + np.angle(t2) * np.cos(j * i1) + np.sin(j * i2)
        # Assign the complex coefficient
        cf[j-1] = mag * (np.cos(ang) + 1j * np.sin(ang))
    return cf.astype(np.complex128)

ALLOWED["p417"]=p417

def p418(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)

    EPS = 1e-12
    SUM_CAP = 1e300     # cap accumulators to stay finite
    BASE_CAP = 8.0      # clamp bases before powering to avoid huge growth

    # endpoints & safe angles
    r1 = t1.real
    r2 = t2.real
    i1 = t1.imag
    i2 = t2.imag

    ang1 = np.arctan2(t1.imag, t1.real)
    ang2 = np.arctan2(t2.imag, t2.real)
    if not np.isfinite(ang1):
        ang1 = 0.0
    if not np.isfinite(ang2):
        ang2 = 0.0

    denom_lin = float(n - 1) if n > 1 else 1.0

    for j in range(1, n + 1):
        sum_re = 0.0
        sum_im = 0.0

        d = np.pi / (j + 1.0)

        for k in range(1, j + 1):
            t = float(k - 1) / denom_lin
            rec_k = r1 + (r2 - r1) * t
            imc_k = i1 + (i2 - i1) * t

            # clamp bases
            if rec_k >  BASE_CAP: rec_k =  BASE_CAP
            if rec_k < -BASE_CAP: rec_k = -BASE_CAP
            if imc_k >  BASE_CAP: imc_k =  BASE_CAP
            if imc_k < -BASE_CAP: imc_k = -BASE_CAP

            # integer power with simple loop (Numba-safe, overflow-resistant with caps)
            pr = 1.0
            pi = 1.0
            for _ in range(k):
                pr *= rec_k
                pi *= imc_k
                if not np.isfinite(pr) or pr >  SUM_CAP: pr =  SUM_CAP
                if pr < -SUM_CAP: pr = -SUM_CAP
                if not np.isfinite(pi) or pi >  SUM_CAP: pi =  SUM_CAP
                if pi < -SUM_CAP: pi = -SUM_CAP

            sum_re += pr * np.cos(k * d)
            sum_im += pi * np.sin(k * d)

            # keep accumulators finite
            if not np.isfinite(sum_re) or sum_re >  SUM_CAP: sum_re =  SUM_CAP
            if sum_re < -SUM_CAP: sum_re = -SUM_CAP
            if not np.isfinite(sum_im) or sum_im >  SUM_CAP: sum_im =  SUM_CAP
            if sum_im < -SUM_CAP: sum_im = -SUM_CAP

        # magnitude via safe logs (arguments >= 1)
        s1 = 1.0 + sum_re * sum_re + sum_im * sum_im
        if not np.isfinite(s1) or s1 < 1.0:
            s1 = 1.0
        s2 = 1.0 + sum_re * sum_im
        if not np.isfinite(s2) or s2 < 1.0:
            s2 = 1.0

        mag = np.log(s1) * np.sin(j * (np.pi / 5.0)) + np.log(s2) * np.cos(j * (np.pi / 7.0))
        if not np.isfinite(mag):
            mag = 0.0

        # angle from real pieces; guard finiteness
        angle = ang1 * sum_re - ang2 * sum_im + np.sin(j * (np.pi / 3.0)) - np.cos(j * (np.pi / 7.0))
        if not np.isfinite(angle):
            angle = 0.0

        cf[j - 1] = mag * (np.cos(angle) + 1j * np.sin(angle))

    # Additional modifications (no divisions by zero here)
    re_re = t1.real * t2.real
    im_sum = t1.imag + t2.imag
    if not np.isfinite(re_re): re_re = 0.0
    if not np.isfinite(im_sum): im_sum = 0.0

    for k in range(1, n + 1):
        denom = float(k + 1)  # > 0
        addv = (re_re / denom) * np.sin(k * (np.pi / 6.0)) + im_sum * np.cos(k * (np.pi / 8.0))
        if not np.isfinite(addv):
            addv = 0.0
        cf[k - 1] += addv

    return cf
    
ALLOWED["p418"]=p418

def p419(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, 36):
        # Calculate magnitude and angle parts
        mag_part = np.log(np.abs(t1) + np.abs(t2) + j) * np.sin(j * np.pi / 7) + np.cos(j * np.pi / 5)
        angle_part = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j / 2)
        # Assign the complex coefficient
        cf[j-1] = mag_part * np.exp(1j * angle_part)
    return cf.astype(np.complex128)
    
ALLOWED["p419"]=p419

def p420(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    r1 = np.real(t1)
    i1 = np.imag(t1)
    r2 = np.real(t2)
    i2 = np.imag(t2)
    for j in range(1, n +1):
        # Calculate angle and magnitude
        angle_component = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j)
        magnitude_component = np.abs(t1)**(j % 5 +1) + np.log(np.abs(t2) +1) * j
        phase_shift = np.sin(j * np.pi / 7) + np.cos(j * np.pi / 5)
        # Assign the complex coefficient with additional terms
        cf[j-1] = (magnitude_component * np.exp(1j * (angle_component + phase_shift)) + 
                    (np.conj(t1) * r2) / (j + 1) + 
                    (i1 + i2) * (j % 3))
    # Introduce region-based variations
    for k in range(1, n +1):
        if k <= n/3:
            cf[k-1] *= (1 + 0.5 * np.sin(k))
        elif k <= 2*n/3:
            cf[k-1] *= (1 + 0.3 * np.cos(k * 2))
        else:
            cf[k-1] *= (1 + 0.2 * np.sin(k * 3) * np.cos(k))
    # Add interactions between coefficients
    for r in range(1, n):
        cf[r-1] += 0.1 * cf[r] * np.exp(1j * np.angle(cf[r-1]))
    return cf.astype(np.complex128)

ALLOWED["p420"]=p420

def p421(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n +1):
        if j <=10:
            # First segment
            mag = np.log(np.abs(t1) + j) * (1 + np.sin(j * np.real(t1)) * np.cos(j * np.imag(t2)))
            angle = np.angle(t1) * np.sin(j / 2) + np.angle(t2) * np.cos(j / 3)
        elif j <=20:
            # Second segment
            mag = np.log(np.abs(t1 * t2) + j) * (1 + np.sin(j)**2 - np.cos(j)**2)
            angle = np.angle(t1 + t2) * np.sin(j / 4) + np.log(j + 1)
        else:
            # Third segment
            mag = np.log(np.abs(t1)**2 + np.abs(t2)**2 + j) * (1 + np.sin(j * np.real(t1) + np.cos(j * np.real(t2))))
            angle = np.angle(t1) * np.cos(j /5) + np.angle(t2) * np.sin(j /6)
        # Assign the complex coefficient
        cf[j-1] = mag * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p421"]=p421

def p422(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, 36):
        # Calculate real and imaginary parts
        r_part = np.real(t1) * np.sin(j * np.pi / 7) + np.real(t2) * np.cos(j * np.pi / 5)
        im_part = np.imag(t1) * np.cos(j * np.pi / 6) - np.imag(t2) * np.sin(j * np.pi / 8)
        # Calculate magnitude and angle
        magnitude = np.log(np.abs(t1) + np.abs(t2) + j) * np.sin(j * np.pi /4) + np.cos(j * np.pi /3)
        angle = np.angle(t1) * np.cos(j /3) + np.angle(t2) * np.sin(j /4)
        # Assign the complex coefficient
        cf[j-1] = magnitude * (np.cos(angle) + 1j * np.sin(angle)) + \
                    np.conj(t1) * np.sin(j) - np.conj(t2) * np.cos(j)
    return cf.astype(np.complex128)

ALLOWED["p422"]=p422

def p423(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, 36):
        if j <=10:
            mag = np.real(t1) * j**2 + np.log(np.abs(t2) + 1) * np.sin(j * np.real(t1))
            angle = np.angle(t1) + np.cos(j * np.imag(t2))
        elif j <=25:
            mag = np.exp(np.real(t2) / (j + 1)) + np.sqrt(j) * (np.real(t1) + np.real(t2))
            angle = np.sin(j) + np.angle(t2) * np.cos(j * np.real(t1))
        else:
            mag = np.log(np.abs(t1) + j) * np.exp(-np.real(t2) / j) + np.sin(j * np.imag(t1))
            angle = np.angle(t1 * t2) / j + np.cos(j * np.imag(t2))
        # Assign the complex coefficient
        cf[j-1] = mag * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p423"]=p423

def p424(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n +1):
        # Calculate magnitude parts
        mag_part1 = np.real(t1) * j**2
        mag_part2 = np.log(np.abs(t2) + j) * np.sin(j * np.angle(t1))
        mag_part3 = np.cos(j * np.real(t2)) * np.sqrt(j)
        magnitude = mag_part1 + mag_part2 + mag_part3
        # Calculate angle parts
        angle_part1 = np.angle(t1) + np.sin(j * np.real(t1))
        angle_part2 = np.cos(j * np.imag(t2)) - np.angle(t2) / j
        angle = angle_part1 + angle_part2
        # Assign the complex coefficient
        cf[j-1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p424"]=p424

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

def p426(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for k in range(1, n +1):
        j = 36 - k
        term1 = np.real(t1) * np.sin(k) * np.cos(j)
        term2 = np.imag(t2) * np.cos(k) * np.sin(j)
        term3 = np.log(np.abs(t1) + k)
        term4 = np.log(np.abs(t2) + j)
        angle = np.angle(t1) * np.sin(k / 2) + np.angle(t2) * np.cos(j / 3)
        magnitude = term1 + term2 + term3 - term4
        # Assign the complex coefficient
        cf[k-1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p426"]=p426

def p427(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n +1):
        # Intermediate real and imaginary sequences
        rec = np.real(t1) + (np.real(t2) - np.real(t1)) * j / n
        imc = np.imag(t1) + (np.imag(t2) - np.imag(t1)) * j / n
        
        # Complex transformations (these variables are calculated but not used directly)
        conj_t1 = np.conj(t1)
        conj_t2 = np.conj(t2)
        
        # Magnitude calculations with intricate patterns
        mag_part1 = np.log(np.abs(rec) +1) * np.sin(j * np.pi /3)
        mag_part2 = np.log(np.abs(imc) +1) * np.cos(j * np.pi /4)
        magnitude = mag_part1 + mag_part2
        
        # Angle calculations with varying functions
        angle = np.angle(t1) * j + np.angle(t2) / (j +1) + np.sin(j * imc) * np.cos(j * rec)
        
        # Assigning the complex coefficient
        cf[j-1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p427"]=p427

def p428(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)

    EPS = 1e-12
    SUM_CAP = 1e300

    r1 = t1.real
    r2 = t2.real
    i1 = t1.imag
    i2 = t2.imag
    ang1 = np.arctan2(t1.imag, t1.real)
    ang2 = np.arctan2(t2.imag, t2.real)
    if not np.isfinite(ang1):
        ang1 = 0.0
    if not np.isfinite(ang2):
        ang2 = 0.0

    denom = float(n - 1) if n > 1 else 1.0

    for j in range(1, n + 1):
        # Linear interpolation (replaces linspace)
        t = float(j - 1) / denom
        rec = r1 + (r2 - r1) * t
        imc = i1 + (i2 - i1) * t

        # magnitude part
        x = abs(rec * imc) + 1.0
        if not np.isfinite(x) or x <= 0.0:
            x = 1.0
        mag_part = np.log(x) * (1.0 + np.sin(j * (np.pi / 5.0)))

        # angle part
        angle_part = ang1 * np.cos(float(j)) + ang2 * np.sin(float(j) / 3.0)

        # base value
        cf[j - 1] = mag_part * (np.cos(angle_part) + 1j * np.sin(angle_part))

        # manual cumulative product of rec[:j] + imc[:j]
        prod_val = 1.0
        for k in range(1, j + 1):
            t2k = float(k - 1) / denom
            rec_k = r1 + (r2 - r1) * t2k
            imc_k = i1 + (i2 - i1) * t2k
            prod_val *= (rec_k + imc_k)
            if not np.isfinite(prod_val):
                prod_val = 1.0
                break
            if prod_val > SUM_CAP:
                prod_val = SUM_CAP
            elif prod_val < -SUM_CAP:
                prod_val = -SUM_CAP

        # safe root
        base = abs(prod_val)
        if base < EPS or not np.isfinite(base):
            base = 1.0
        root_val = base ** (1.0 / float(j))
        cf[j - 1] += root_val

        # every 5th term: multiply by exp(i·angle(cf[j-1])) safely
        if (j % 5) == 0:
            ang_cf = np.arctan2(cf[j - 1].imag, cf[j - 1].real)
            cf[j - 1] *= (np.cos(ang_cf) + 1j * np.sin(ang_cf))

    # final finite check
    for i in range(n):
        if not (np.isfinite(cf[i].real) and np.isfinite(cf[i].imag)):
            cf[i] = 0.0 + 0.0j

    return cf

ALLOWED["p428"]=p428

def p429(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, 36):
        mag_part = np.log(np.abs(t1) + j**2) * np.sin(j * np.real(t2)) + np.cos(j * np.imag(t1))
        angle_part = np.angle(t1) * j + np.angle(t2) * np.sqrt(j) + np.sin(j) * np.angle(t2)
        # Assign the complex coefficient
        cf[j-1] = mag_part * (np.cos(angle_part) + 1j * np.sin(angle_part))
    # Modify coefficients based on additional terms
    for k in range(1, 36):
        r = np.real(t1) * k
        angle = np.angle(t2) * np.cos(k /4) + np.sin(k /6)
        cf[k-1] += (np.log(np.abs(t1 + t2) +1) * np.cos(angle) + 
                    1j * np.log(np.abs(t1 - t2) +1) * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p429"]=p429

def p430(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(np.real(t1), np.real(t2), n)
    imc = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(1, n +1):
        mag = np.log(np.abs(rec[j-1] * imc[j-1]) +1) * (1 + np.sin(j * np.pi /4)) + \
                (1 + np.cos(j * np.pi /3))
        ang = np.angle(t1) + np.angle(t2) + np.sin(j) - np.cos(j)
        # Assign the complex coefficient with additional terms
        cf[j-1] = mag * (np.cos(ang) + 1j * np.sin(ang)) + \
                    np.conj(t1) * np.real(t2) * np.sin(j /2)
    return cf.astype(np.complex128)
    
ALLOWED["p430"]=p430

def p431(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec_seq = np.linspace(np.real(t1), np.real(t2), n)
    imc_seq = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(1, n +1):
        term_real = np.sin(rec_seq[j-1] * j) + np.cos(imc_seq[j-1] * j**1.5)
        term_imag = np.log(np.abs(t1) + j) - np.sin(imc_seq[j-1] * np.pi / (j +1))
        mag = np.sqrt(term_real**2 + term_imag**2) * (1 + 0.1 * j)
        angle = np.angle(t1) * np.log(j +1) + np.cos(j * np.pi /7)
        # Assign the complex coefficient
        cf[j-1] = mag * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p431"]=p431

def p432(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n +1):
        mag_part1 = np.log(np.abs(t1) + j) * np.sin(j * np.real(t2))
        mag_part2 = np.cos(j * np.imag(t1)) * (np.abs(t2)**(1 + j/10))
        magnitude = mag_part1 + mag_part2
        angle_part1 = np.angle(t1) * np.cos(j /5)
        angle_part2 = np.angle(t2) * np.sin(j /3)
        angle = angle_part1 + angle_part2
        # Assign the complex coefficient
        cf[j-1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    for k in range(1, n +1):
        cf[k-1] *= np.exp(1j * (np.real(t1) * k /n + np.imag(t2) * (n -k) /n))
    for r in range(1, n +1):
        cf[r-1] += np.conj(t1) * np.conj(t2) / (r +1)
    return cf.astype(np.complex128)

ALLOWED["p432"]=p432

def p433(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n +1):
        real_component = (np.real(t1)**j) * np.cos(j * np.angle(t2)) + \
                            np.sin(j * np.real(t1)) * np.log(np.abs(t2) + j)
        imag_component = np.imag(t1) * np.log(j +1) + \
                            np.cos(np.imag(t2)) * (np.abs(t1)**0.5)
        cf[j-1] = real_component + 1j * imag_component
    return cf.astype(np.complex128)
    
ALLOWED["p433"]=p433

def p434(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec_seq = np.linspace(np.real(t1), np.real(t2), n)
    imc_seq = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(1, n +1):
        magnitude = (np.log(np.abs(rec_seq[j-1]) + np.abs(imc_seq[j-1]) + j) *
                        (np.log(np.abs(t1) + np.abs(t2))**(j/10)))
        angle_part = np.sin(np.sin(j /2) * np.cos(j /3))
        angle = np.angle(t1) + np.angle(t2) + angle_part
        # Assign the complex coefficient with additional terms
        cf[j-1] = magnitude * np.exp(1j * angle) + \
                    np.conj(t1) * np.sin(j)**2
    return cf.astype(np.complex128)
    
ALLOWED["p434"]=p434

def p435(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n +1):
        mag_part = np.log(np.abs(t1) * j + np.abs(t2)**(n -j) +1)
        angle_part = np.angle(t1) * np.sin(j * np.pi /7) + np.angle(t2) * np.cos(j * np.pi /5)
        # Calculate real_sum and imag_sum
        real_sum = 0
        imag_sum =0
        for k in range(1, j +1):
            real_sum += np.real(t1)**k * np.real(t2)**(j -k)
            imag_sum += np.imag(t1)**k * np.imag(t2)**(j -k)
        intricate_mag = mag_part * (1 + np.sin(real_sum / (j +1)))
        intricate_angle = angle_part + np.cos(imag_sum / (j +1))
        # Assign the complex coefficient
        cf[j-1] = intricate_mag * (np.cos(intricate_angle) + 1j * np.sin(intricate_angle))
    return cf.astype(np.complex128)

ALLOWED["p435"]=p435

def p436(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, 36):
        # Calculate magnitude components
        mag_part1 = np.log(np.abs(t1) + j) * np.sin(j / 3)
        mag_part2 = np.abs(t2)**(1 + (j % 4))
        magnitude = mag_part1 + mag_part2 * np.cos(j / 2)
        
        # Calculate angle components
        angle_part1 = np.angle(t1) * np.cos(j)
        angle_part2 = np.angle(t2) * np.sin(j / 2)
        angle = angle_part1 + angle_part2 + np.real(t1) * np.imag(t2) / j
        
        # Assign complex coefficient
        cf[j-1] = magnitude * (np.cos(angle) + 1j * np.sin(angle)) + np.conj(t1)**((j % 5) + 1)
    return cf.astype(np.complex128)
    
ALLOWED["p436"]=p436

def p437(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, 36):
        # Linear interpolation between real and imaginary parts
        r = np.real(t1) + (np.real(t2) - np.real(t1)) * j / 35
        im = np.imag(t1) + (np.imag(t2) - np.imag(t1)) * j / 35
        
        # Calculate magnitude
        mag = np.log(np.abs(r * im) + 1) * (np.sin(j * np.angle(t1)) + np.cos(j * np.angle(t2))) + np.sqrt(r**2 + im**2)
        
        # Calculate angle
        angle = np.angle(t1) * np.sin(j / 3) - np.angle(t2) * np.cos(j / 4)
        
        # Assign complex coefficient
        cf[j-1] = mag * np.exp(1j * angle) + np.conj(t1) * np.cos(im * np.pi / 5) * np.sin(r * np.pi / 7)
    return cf.astype(np.complex128)
    
ALLOWED["p437"]=p437

def p438(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    # First loop: initial coefficients
    for j in range(1, 36):
        # Calculate angle and magnitude
        angle = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j/2)
        magnitude = np.abs(t1)**(j%4 + 1) + np.log(np.abs(t2) + 1) * np.sqrt(j)
        # Assign to complex coefficient
        cf[j-1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    
    # Second loop: modify coefficients
    for k in range(1, 36):
        cf[k-1] = cf[k-1] + np.conj(t1)**k * np.cos(k) - np.real(t2) * np.sin(k/3) + np.abs(t1 + t2)**(1 + (k % 5))
    
    return cf.astype(np.complex128)
    
ALLOWED["p438"]=p438

def p439(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(np.real(t1), np.real(t2), n)
    imc = np.linspace(np.imag(t1), np.imag(t2), n)
    for j in range(n):
        r = rec[j]
        im = imc[j]
        mag = np.log(np.abs(r + im*1j) + 1) * (1 + np.sin((j+1) * np.pi / 4)) * (1 + np.cos((j+1) * np.pi / 3))
        angle = np.angle(r + im*1j) + np.sin((j+1) * np.pi / 5) - np.cos((j+1) * np.pi / 6)
        cf[j] = mag * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p439"]=p439

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

def p441(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec_seq = np.linspace(t1.real, t2.real, num=n)
    imc_seq = np.linspace(t1.imag, t2.imag, num=n)
    for j in range(n):
        mag_part = np.log(np.abs(rec_seq[j]) + np.abs(imc_seq[j]) + (j + 1)) * (np.abs(t1) + np.abs(t2)) ** ((j + 1) / 10)
        angle_part = np.sin((j + 1) * np.angle(t1)) * np.cos((j + 1) * np.angle(t2)) + np.log(np.abs(t1) + np.abs(t2) + (j + 1))
        cf[j] = mag_part * np.exp(1j * angle_part) + np.conj(mag_part * np.exp(-1j * angle_part))
    return cf.astype(np.complex128)
    
ALLOWED["p441"]=p441

def p442(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        mag_part = np.log(np.abs(t1) + j**2) * np.sin(j * np.pi / 4) + np.cos(j * np.pi / 6)
        angle_part = np.angle(t2) + np.sin(j) * np.cos(j / 2) + np.log(np.abs(t1) + 1)
        cf[j - 1] = mag_part * (np.cos(angle_part) + 1j * np.sin(angle_part))
    for k in range(1, n + 1):
        if k % 2 == 0:
            cf[k - 1] = cf[k - 1] * np.exp(1j * np.angle(t1) * k / 10)
        else:
            cf[k - 1] = cf[k - 1] * np.exp(-1j * np.angle(t2) * k / 15)
    for r in range(1, n + 1):
        cf[r - 1] = cf[r - 1] + np.conj(cf[n - r]) * (np.abs(t1) / (r + 1))
    return cf.astype(np.complex128)

ALLOWED["p442"]=p442

def p443(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    real_seq = np.linspace(t1.real, t2.real, num=n)
    imag_seq = np.linspace(t1.imag, t2.imag, num=n)
    for j in range(1, n + 1):
        mag = np.log(np.abs(real_seq[j - 1] + t2) + 1) * np.sin(j * np.pi / 4) + np.cos(j * np.pi / 5) * np.abs(t1)
        angle = np.angle(t1) * j + np.sin(j * np.pi / 6) - np.cos(j * np.pi / 7)
        cf[j - 1] = mag * np.exp(1j * angle)
        for k in range(1, j + 1):
            cf[j - 1] += (t1.real * t2.real / (k + 1)) * np.exp(1j * (np.sin(k) - np.cos(k)))
    for r in range(1, n + 1):
        cf[r - 1] = cf[r - 1] * (1 + 0.05 * r**2) + np.conj(t2) * np.sin(r * np.pi / 8) - t1.real * np.cos(r * np.pi / 9)
    return cf.astype(np.complex128)
    
ALLOWED["p443"]=p443

def p444(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        mag = np.log(np.abs(t1)**j + np.abs(t2)**(n - j)) + np.sin(j * t1.real) * np.cos(j * t2.imag)
        angle = np.angle(t1) * j - np.angle(t2) * (n - j) + np.sin(j) - np.cos(j)
        cf[j - 1] = mag * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p444"]=p444

def p445(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        k = (j * 5 + 2) % 12 + 1
        r_part = t1.real * np.sin(j * np.pi / k) + t2.real * np.cos(j * np.pi / (k + 1))
        i_part = t1.imag * np.cos(j * np.pi / k) - t2.imag * np.sin(j * np.pi / (k + 1))
        magnitude = np.log(np.abs(t1) + j) * np.abs(np.sin(j * np.pi / 10))
        angle = np.angle(t1) * np.cos(j * np.pi / 8) + np.angle(t2) * np.sin(j * np.pi / 9)
        cf[j - 1] = magnitude * (r_part + 1j * i_part) * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p445"]=p445

def p446(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, 36):
        mag = np.log(np.abs(t1)*j + np.abs(t2)/j + 1) * (1 + np.sin(j * np.pi / 4)) + (t1.real * t2.imag * j)
        angle = np.angle(t1)*j + np.angle(t2)*np.cos(j * np.pi / 5) + np.sin(j)**2
        cf[j-1] = mag * (np.cos(angle) + np.sin(angle)*1j)
    return cf.astype(np.complex128)
    
ALLOWED["p446"]=p446

def p447(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        k = (j % 7) + 1
        r = (j % 5) + 2
        mag_part = np.log(np.abs(t1)*j + np.abs(t2)/k + 1) * np.sin(j * np.pi / r)
        angle_part = np.angle(t1)*np.cos(j / k) + np.angle(t2)*np.sin(j / r) + t1.real * t2.imag / (k + 1)
        cf[j-1] = mag_part * np.exp(1j * angle_part)
    for j in range(1, n+1):
        for r in range(1, j//3 +1):
            k = (j % 7) + 1
            cf[j-1] += (t1.real**r - t2.imag**k) * np.exp(1j * (np.angle(t1)*r - np.angle(t2)*k))
    return cf.astype(np.complex128)
    
ALLOWED["p447"]=p447

def p448(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        rec = t1.real + (t2.real - t1.real)*j / n
        imc = t1.imag + (t2.imag - t1.imag)*j / n
        mag = np.log(np.abs(t1) + np.abs(t2) + j**3) * (1 + np.sin(j * np.pi / 5)) * (1 + np.cos(j * np.pi / 4))
        angle = np.angle(t1)*np.sin(j * np.pi / 3) + np.angle(t2)*np.cos(j * np.pi / 4) + np.sin(j * rec) * np.cos(j * imc)
        cf[j-1] = mag * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p448"]=p448

def p449(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        r = t1.real + j * t2.real
        k = t1.imag - j * t2.imag
        mag = np.log(np.abs(t1) + np.abs(t2) + j) * (1 + np.sin(j * np.pi / 7))
        angle = np.angle(t1)*np.sin(j / 3) + np.angle(t2)*np.cos(j / 5)
        cf[j-1] = mag * (np.cos(angle) + 1j * np.sin(angle))
    for k in range(1, n+1):
        r = t1.real * k
        angle = np.angle(t2)*np.cos(k / 4) + np.sin(k / 6)
        cf[k-1] += (np.log(np.abs(t1 + t2) + 1) * np.cos(angle) + 1j * np.log(np.abs(t1 - t2) + 1) * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p449"]=p449

def p450(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        angle_part = np.sin(j * np.angle(t1) + np.cos(j * np.angle(t2)))
        mag_part = np.log(np.abs(t1) + np.abs(t2) + j) * (1 + 0.1 * j)
        phase_shift = np.angle(t1) * np.cos(j / n * np.pi) - np.angle(t2) * np.sin(j / n * np.pi)
        cf[j-1] = mag_part * (np.cos(angle_part + phase_shift) + 1j * np.sin(angle_part - phase_shift))
    # Introduce variations using conjugates and nonlinear combinations
    for k in range(1, 6):
        idx = n - k
        if idx >=0:
            cf[idx] = cf[idx] * np.conj(t1)**((k % 3) + 1) + np.conj(t2)**(k % 4)
    # Apply modulation based on cumulative product
    cumulative = 1
    for r in range(1, n+1):
        cumulative *= (np.abs(t1) + np.abs(t2) + r)
        cf[r-1] += cumulative / (r + 1)
    return cf.astype(np.complex128)
    
ALLOWED["p450"]=p450

def p451(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        mag_part1 = np.log(np.abs(t1) + j) * np.sqrt(j)
        mag_part2 = np.abs(t2)**(1 + np.sin(j))
        magnitude = mag_part1 + mag_part2
        phase_part1 = np.sin(j * t1.real) + np.cos(j * t2.imag)
        phase_part2 = np.angle(t1)*j - np.angle(t2)*(n - j)
        phase = phase_part1 * phase_part2 + np.sin(j / 3) * np.cos(j / 5)
        cf[j-1] = magnitude * np.exp(1j * phase)
    return cf.astype(np.complex128)
    
ALLOWED["p451"]=p451

def p452(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for k in range(1, n+1):
        temp = 0
        for j in range(1, k+1):
            temp += np.sin(j * t1.real) * np.cos(j * t2.imag) / j
        magnitude = np.log(np.abs(t1)*np.abs(t2) + k) * (1 + np.sin(k / 2) * 3)
        angle = np.angle(t1) + np.angle(t2) * np.log(k + 1) + temp
        cf[k-1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p452"]=p452

def p453(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        r_part = t1.real * j / n + t2.real * (n - j +1) /n
        i_part = np.sin(j * np.pi / n) * t1.imag - np.cos(j * np.pi / n) * t2.imag
        magnitude = np.log(np.abs(t1) + j) * (1 + np.abs(np.sin(j / 3)))
        angle = np.angle(t1)*np.cos(j) + np.angle(t2)*np.sin(j)
        cf[j-1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p453"]=p453

def p454(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)

    EPS = 1e-12
    SUM_CAP = 1e300

    r1, r2 = t1.real, t2.real
    i1, i2 = t1.imag, t2.imag

    ang1 = np.arctan2(t1.imag, t1.real)
    ang2 = np.arctan2(t2.imag, t2.real)
    if not np.isfinite(ang1): ang1 = 0.0
    if not np.isfinite(ang2): ang2 = 0.0

    denom = float(n - 1) if n > 1 else 1.0

    for j in range(1, n + 1):
        # rec_seq[j-1], imc_seq[j-1] via interpolation
        tj = float(j - 1) / denom
        rec_j = r1 + (r2 - r1) * tj
        imc_j = i1 + (i2 - i1) * tj

        # log term (arg >= 1)
        x = abs(rec_j * imc_j) + 1.0
        if not np.isfinite(x) or x <= 0.0:
            x = 1.0
        mag = np.log(x) * (1.0 + np.sin(j * (np.pi / 4.0)))

        # cumulative product prod(rec_seq[:j]) with caps
        prod_val = 1.0
        for k in range(1, j + 1):
            tk = float(k - 1) / denom
            rec_k = r1 + (r2 - r1) * tk
            prod_val *= rec_k
            if not np.isfinite(prod_val):
                prod_val = 1.0
                break
            if prod_val >  SUM_CAP: prod_val =  SUM_CAP
            if prod_val < -SUM_CAP: prod_val = -SUM_CAP

        base = abs(prod_val)
        if not np.isfinite(base) or base < EPS:
            base = 1.0
        mag += base ** (1.0 / float(j))

        # angle parts (Numba-safe)
        angle = (
            ang1 * float(j)
            + np.sin(float(j) * ang2)
            + np.cos(float(j) * imc_j)
        )
        if not np.isfinite(angle):
            angle = 0.0

        # polar synthesis (avoid complex exp)
        cf[j - 1] = mag * (np.cos(angle) + 1j * np.sin(angle))

    # final finite clamp (defensive)
    for i in range(n):
        if not (np.isfinite(cf[i].real) and np.isfinite(cf[i].imag)):
            cf[i] = 0.0 + 0.0j

    return cf

ALLOWED["p454"]=p454

def p455(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, 36):
        mag = np.log(np.abs(t1) + 1) * np.sin(j * t2.real) + np.cos(j * t1.imag)**2
        angle = np.angle(t1)*j + np.angle(t2)/(j +1) + np.sin(j * t1.real * t2.imag)
        cf[j-1] = mag * (np.cos(angle) + 1j * np.sin(angle))
    for k in range(1, 36):
        cf[k-1] = cf[k-1] * (1 + 0.05 * k * t1.real) / (1 + 0.03 * k * t2.imag)
        cf[k-1] += np.conj(cf[35 -k]) * np.exp(-0.1 * k)
    for r in range(1, 36):
        cf[r-1] = np.abs(cf[r-1])**np.sin(r * np.pi /17) * (np.cos(r * np.pi /23) + 1j * np.sin(r * np.pi /23))
    return cf.astype(np.complex128)
    
ALLOWED["p455"]=p455

def p456(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, 36):
        if j <=10:
            mag = np.log(np.abs(t1) + j**2) * np.abs(np.sin(j * t2.real) + np.cos(j * t1.imag))
            ang = np.angle(t1)**j + np.angle(t2)*np.sin(j)
        elif j <=20:
            mag = np.log(np.abs(t2) + j**1.5) * np.abs(np.cos(j * t1.real) - np.sin(j * t2.imag))
            ang = np.angle(t2)**j - np.angle(t1) * np.log(j + 1)
        else:
            mag = np.log(np.abs(t1 * t2) + j) * np.abs(np.sin(j * t1.real + np.cos(j * t2.real)))
            ang = np.angle(t1 + np.conj(t2)) * j + np.log(np.abs(t1 - t2) +1)
        cf[j-1] = mag * np.exp(1j * ang)
    return cf.astype(np.complex128)

ALLOWED["p456"]=p456

def p457(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec_seq = np.linspace(t1.real, t2.real, n)
    imc_seq = np.linspace(t1.imag, t2.imag, n)
    for j in range(1, n+1):
        mag_part1 = np.log(np.abs(rec_seq[j-1]) + 1) * np.sin(j * np.pi /4)
        mag_part2 = np.abs(t2)**((j %5)+1)
        mag = mag_part1 + mag_part2 * np.cos(j * t1.real)
        angle_part1 = np.angle(t1)*np.cos(j /3)
        angle_part2 = np.sin(j * t2.imag) + np.cos(j * np.pi /6)
        angle = angle_part1 + angle_part2
        cf[j-1] = mag * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p457"]=p457

def p458(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1,36):
        k = (j%5)+1
        r = np.sqrt(j)*np.log(np.abs(t1)+np.abs(t2)+1)
        angle = np.angle(t1)*np.sin(k*j)+np.angle(t2)*np.cos(k*j)
        magnitude = ((np.real(t1)**k)+(np.imag(t2)**k))*(1+np.cos(j*np.pi/7))*(1+np.sin(j*np.pi/5))
        cf[j-1] = magnitude*(np.cos(angle)+1j*np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p458"]=p458

def p459(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for k in range(1, n+1):
        temp_real = t1.real * np.log(np.abs(t1)*k + np.abs(t2) +1)
        temp_imag = t2.imag * np.sin(k) + np.cos(k * t1.real)
        temp_angle = (np.angle(t1 + t2)) / (k +1)
        magnitude = temp_real + temp_imag * temp_angle
        angle = np.sin(temp_real) + np.cos(temp_imag) * temp_angle
        cf[k-1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p459"]=p459

def p460(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec_seq = np.linspace(t1.real, t2.real, n)
    imc_seq = np.linspace(t1.imag, t2.imag, n)
    for j in range(1, n+1):
        phase = np.angle(t1)*j + np.angle(t2)/(j +1) + np.sin(j * rec_seq[j-1]) - np.cos(j * imc_seq[j-1])
        magnitude = np.log(np.abs(t1) + np.abs(t2) + j) * (j**2 + np.sin(j)*np.cos(j))
        cf[j-1] = magnitude * (np.cos(phase) + 1j * np.sin(phase))
    return cf.astype(np.complex128)

ALLOWED["p460"]=p460

def p461(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec1 = t1.real
    imc1 = t1.imag
    rec2 = t2.real
    imc2 = t2.imag
    
    for j in range(1, n+1):
        magnitude = np.log(np.abs(t1) + j) * np.sin(j * np.pi /4) + np.cos(j * np.pi /3) * ((j %5) +1)
        angle = np.angle(t1)*np.cos(j/3) + np.angle(t2)*np.sin(j /5) + np.sin(j * np.pi /6)
        cf[j-1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    
    for k in range(1, n+1):
        cf[k-1] += (t1.real - t2.real) * np.sin(k * np.pi /7) + (t1.imag + t2.imag) * np.cos(k * np.pi /8)
    
    for r in range(1, n+1):
        cf[r-1] *= np.exp(1j * (np.sin(r) + np.cos(r)))
    
    return cf.astype(np.complex128)
    
ALLOWED["p461"]=p461

def p462(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec_seq = np.linspace(t1.real, t2.real, n)
    imc_seq = np.linspace(t1.imag, t2.imag, n)
    for j in range(1, n+1):
        mag = np.log(np.abs(rec_seq[j-1]*imc_seq[j-1] +1) +1) * (1 + np.sin(j)*np.cos(j/3))
        angle = np.angle(t1)*np.sin(j * np.pi /4) + np.angle(t2)*np.cos(j * np.pi /5)
        cf[j-1] = mag * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p462"]=p462

def p463(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        mag_part = np.log(np.abs(t1) + j) * np.sin(j * np.angle(t2)) + np.cos(j * np.angle(t1))
        ang_part = np.angle(t1)*j + np.angle(t2)*(n -j) + np.sin(j * t2.real) - np.cos(j * t1.imag)
        cf[j-1] = mag_part * np.exp(1j * ang_part)
    for k in range(1,6):
        for r in range(1, n+1):
            cf[r-1] += (t1.real * np.cos(k*r) + t2.imag * np.sin(k*r)) * np.exp(1j * (t2.real * r - t1.imag * k))
    for j in range(1, n+1):
        cf[j-1] *= (1 + 0.1 * j) / (1 + np.log(j +1))
    return cf.astype(np.complex128)
    
ALLOWED["p463"]=p463

def p464(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec_seq = np.linspace(t1.real, t2.real, n)
    imc_seq = np.linspace(t1.imag, t2.imag, n)
    for k in range(1, n+1):
        mag_part = np.log(np.abs(t1) + k) * np.sin(k * np.pi /7) + np.cos(k * np.pi /5) * np.sqrt(k)
        angle_part = np.angle(t1)*np.sin(k) + np.angle(t2)*np.cos(k) + np.sin(k * t1.real) * np.cos(k * t2.imag)
        cf[k-1] = mag_part * np.exp(1j * angle_part)
    for r in range(1, n+1):
        cf[r-1] += (t1.real * t2.real / (r +1)) + 1j * (t1.imag - t2.imag) * np.sin(r)
    for j in range(1, n+1):
        cf[j-1] *= (1 + 0.1 * j) * np.exp(0.05j * j)
    return cf.astype(np.complex128)

ALLOWED["p464"]=p464

def p465(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        mag_part = np.log(np.abs(t1) + j) * np.sin(j * np.angle(t1)) + np.cos(j * np.angle(t2))
        ang_part = np.angle(t1)*j + np.angle(t2)*(n -j) + np.sin(j * t1.real) - np.cos(j * t2.imag)
        cf[j-1] = mag_part * np.exp(1j * ang_part)
    for k in range(1,6):
        for r in range(1, n+1):
            cf[r-1] += (t1.real * np.cos(k*r) + t2.imag * np.sin(k*r)) * np.exp(1j * (t2.real * r - t1.imag *k))
    for j in range(1, n+1):
        cf[j-1] *= (1 + 0.1 *j)
    return cf.astype(np.complex128)
    
ALLOWED["p465"]=p465

def p466(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        magnitude = np.log(np.abs(t1) + j**2) * np.abs(np.sin(j * np.angle(t1))) + np.sqrt(j) * np.cos(j * np.angle(t2))
        angle = np.angle(t1)*np.log(j +1) - t2.imag / (j +0.5) + np.sin(j * t1.real)*np.cos(j * t2.imag)
        cf[j-1] = magnitude * np.exp(1j * angle)
    return cf.astype(np.complex128)

ALLOWED["p466"]=p466

def p467(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)

    # Numba-safe polar pieces
    r1 = math.hypot(t1.real, t1.imag)   # |t1|
    r2 = math.hypot(t2.real, t2.imag)   # |t2|
    ang2 = math.atan2(t2.imag, t2.real) # angle(t2)

    for j in range(1, n + 1):
        mag_sum = 0.0

        # sum_{k=1..j} log(|t1|+k) * sin(k*angle(t2))
        for k in range(1, j + 1):
            kf = float(k)
            mag_sum += math.log(r1 + kf) * math.sin(kf * ang2)

        # sum_{r=1..(n-j)} log(|t2|+r) * cos(r*angle(t2))
        up = n - j
        for r in range(1, up + 1):
            rf = float(r)
            mag_sum += math.log(r2 + rf) * math.cos(rf * ang2)

        # stable magnitude (original had log(mag_sum+1); guard against negatives)
        mag = math.log(abs(mag_sum) + 1.0)

        # angle = mag_sum/(j+1) + mag_sum/(n-j+1)
        angle = (mag_sum / (float(j) + 1.0)) + (mag_sum / (float(up) + 1.0))

        # exp(1j*angle) via cos/sin
        c = mag
        cf[j - 1] = np.complex128(c * math.cos(angle) + 1j * c * math.sin(angle))

    # post-perturbation (avoid np.conj)
    for j in range(1, n + 1):
        scale = 1.0 + 0.05 * float(j * j)
        v = cf[j - 1]
        v_conj = np.complex128(v.real - 1j * v.imag)
        cf[j - 1] = v * scale + v_conj * 0.02

    return cf
    
ALLOWED["p467"]=p467

def p468(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        if j <=10:
            mag = np.log(np.abs(t1) + j**2) * np.abs(np.sin(j * t2.real)) + np.cos(j * t1.imag)**2
            ang = np.angle(t1)**j + np.angle(t2)*np.sin(j)
        elif j <=20:
            mag = np.log(np.abs(t2) + j**1.5) * np.abs(np.cos(j * t1.real) - np.sin(j * t2.imag))
            ang = np.angle(t2)**j - np.angle(t1)*np.log(j+1)
        else:
            mag = np.log(np.abs(t1 * t2) + j) * np.abs(np.sin(j * t1.real + np.cos(j * t2.real)))
            ang = np.angle(t1 + np.conj(t2))*j + np.log(np.abs(t1 - t2) +1)
        cf[j-1] = mag * np.exp(1j * ang)
    return cf.astype(np.complex128)

ALLOWED["p468"]=p468    

def p469(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        mag = np.log(np.abs(t1) + np.abs(t2) + j) * (1 + np.sin(j * np.angle(t1)) * np.cos(j * np.angle(t2)))
        angle = np.angle(t1)**j - np.angle(t2)**(j %4) + np.sin(j * t1.real) - np.cos(j * t2.imag)
        cf[j-1] = mag * (np.cos(angle) + 1j * np.sin(angle))
        for k in range(1,4):
            mag = mag * (1 + 0.1 * np.sin(k * t1.real + 0.0))  # Adjusted to numpy
            angle = angle + 0.5 * np.angle(t1)**k - 0.3 * np.angle(t2)**k
            cf[j-1] += mag * (np.cos(angle) + 1j * np.sin(angle))
        cf[j-1] = cf[j-1] * (1 + 0.05 * j)
    return cf.astype(np.complex128)
    
ALLOWED["p469"]=p469

def p470(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec_seq = np.linspace(t1.real, t2.real, n)
    imc_seq = np.linspace(t1.imag, t2.imag, n)
    for k in range(1, n+1):
        r = rec_seq[k-1]
        im = imc_seq[k-1]
        mag = np.log(np.abs(r) + 1)*np.abs(t1)**0.5 + np.sin(r * k)*np.cos(im / (k +1)) + (k %3 +1)*np.abs(t2)
        angle = np.angle(t1) * np.sin(k) + np.angle(t2) * np.cos(k * np.pi /4) + np.sin(im * k /2)
        cf[k-1] = mag * (np.cos(angle) + np.sin(angle)*1j)
    return cf.astype(np.complex128)

ALLOWED["p470"]=p470

def p471(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        mag_part = np.log(np.abs(t1) + j) * np.sin(j) + np.cos(j * t2.real)
        angle_part = np.angle(t1)*np.sqrt(j) + t2.imag / (j +1)
        cf[j-1] = mag_part * (np.cos(angle_part) + 1j * np.sin(angle_part))
    for k in range(1, n+1):
        cf[k-1] += (t2.real - t1.imag) * np.exp(1j * np.log(k +1)) * np.cos(k)
    return cf.astype(np.complex128)
    
ALLOWED["p471"]=p471

def p472(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
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

def p473(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec_seq = np.linspace(t1.real, t2.real, n)
    imc_seq = np.linspace(t1.imag, t2.imag, n)
    for j in range(1, n+1):
        mag = np.log(np.abs(rec_seq[j-1]*imc_seq[j-1] +1)**1) * (1 + np.sin(j * np.pi /4)*np.cos(j * np.pi /6))
        angle = np.angle(t1)*np.sin(j /2) + np.angle(t2)*np.cos(j /3) + np.log(np.abs(j) +1)
        cf[j-1] = mag * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p473"]=p473

def p474(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        k = j**2
        r = np.log(np.abs(t1) + j) * np.sin(j * np.angle(t2))
        angle = np.angle(t1)*np.sin(k) - np.angle(t2)*np.cos(k) + np.log(np.abs(t2) +1)
        magnitude = (t1.real * np.cos(k) + t2.imag * np.sin(k)) * (np.abs(t1)**2 / (k +1))
        cf[j-1] = magnitude * (np.cos(angle) + np.sin(angle)*1j)
    return cf.astype(np.complex128)

ALLOWED["p474"]=p474

def p475(z,a,state):
    t1, t2 = z[0], z[1]
    n =40
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        mag = t1.real**j + np.log(np.abs(t2) +j) + np.prod(np.arange(1, j+1))**(1/3)
        angle = np.angle(t1)*np.sin(j * np.pi /6) + np.angle(t2)*np.cos(j * np.pi /4)
        cf[j-1] = mag * np.exp(1j * angle) + np.conj(t1)*np.sin(j/2) - np.conj(t2)*np.cos(j/3)
    return cf.astype(np.complex128)

ALLOWED["p475"]=p475

def p476(z,a,state):
    t1, t2 = z[0], z[1]
    n =35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        mag_part1 = np.log(np.abs(t1) + j) * np.sin(j * t1.real)
        mag_part2 = np.abs(t2)**0.5 * np.cos(j /3)
        magnitude = mag_part1 + mag_part2 + j**2
        angle_part1 = np.angle(t1)*np.cos(j * np.pi /4)
        angle_part2 = np.angle(t2)*np.sin(j * np.pi /5)
        angle = angle_part1 + angle_part2 + np.sin(j)
        cf[j-1] = magnitude * (np.cos(angle) + np.sin(angle)*1j)
    return cf.astype(np.complex128)
    
ALLOWED["p476"]=p476

def p477(z,a,state):
    t1, t2 = z[0], z[1]
    n =40
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        r = t1.real + t2.real * j
        i_part = t1.imag - t2.imag * j
        phase = np.sin(r) * np.cos(i_part) + np.log(np.abs(t1) + j)
        magnitude = np.abs(t1)**0.5 * np.abs(t2)**0.3 * j**np.sin(j) + np.cos(j * np.angle(t2))
        cf[j-1] = magnitude * np.exp(1j * phase)
    return cf.astype(np.complex128)
    
ALLOWED["p477"]=p477

def p478(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)

    # Polar bits (Numba-safe)
    r1 = math.hypot(t1.real, t1.imag)
    ang1 = math.atan2(t1.imag, t1.real)
    ang2 = math.atan2(t2.imag, t2.real)

    # early guard: avoid division by ~0 in cos(j / (t1.imag + 1))
    if abs(t1.imag + 1.0) < 1e-10:
        return cf  # already zeros

    t1_conj = np.complex128(t1.real - 1j * t1.imag)
    t2_conj = np.complex128(t2.real - 1j * t2.imag)

    denom = t1.imag + 1.0  # safe by guard above

    for j in range(1, n + 1):
        jf = float(j)

        # mag_part1: log(|t1| + j)
        mag_part1 = math.log(r1 + jf)

        # mag_part2: sin(j * Re(t1)) * cos(j / (Im(t1) + 1))
        mag_part2 = math.sin(jf * t1.real) * math.cos(jf / denom)

        # sqrt(j!) via exp(0.5 * lgamma(j+1))
        sqrt_fact = math.exp(0.5 * math.lgamma(jf + 1.0))

        magnitude = mag_part1 * mag_part2 + sqrt_fact

        # angle parts: replace np.angle with atan2; keep trig in math
        angle_part1 = ang1 * math.sin(jf) + ang2 * math.cos(jf)
        angle_part2 = math.sin(jf * t1.real) - math.cos(jf * t2.imag)
        angle = angle_part1 + angle_part2

        # exp(i*angle) via cos/sin
        ca = math.cos(angle)
        sa = math.sin(angle)

        base = np.complex128(magnitude * ca + 1j * magnitude * sa)
        cf[j - 1] = base + t1_conj * math.sin(jf) - t2_conj * math.cos(jf)

    return cf
    
ALLOWED["p478"]=p478

def p479(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec_seq = np.linspace(t1.real, t2.real, n)
    imc_seq = np.linspace(t1.imag, t2.imag, n)
    for j in range(1, n+1):
        mag_part1 = np.log(np.abs(rec_seq[j-1]) +1) * np.sin(j * np.pi /4)
        mag_part2 = np.abs(t2)**((j %5) +1)
        mag = mag_part1 + mag_part2 * np.cos(j * t1.real)
        angle_part1 = np.angle(t1)*np.cos(j /3)
        angle_part2 = np.sin(j * t2.imag) + np.cos(j * np.pi /6)
        angle = angle_part1 + angle_part2
        cf[j-1] = mag * (np.cos(angle) + np.sin(angle)*1j)
    return cf.astype(np.complex128)
    
ALLOWED["p479"]=p479

def p480(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)

    EPS = 1e-12

    # angles via atan2 (Numba-safe)
    ang1 = np.arctan2(t1.imag, t1.real)
    ang2 = np.arctan2(t2.imag, t2.real)

    for j in range(1, n + 1):
        # magnitude parts (all real)
        mag_part1 = np.log(np.abs(t1) + j) * np.sin(j * t1.real)
        mag_part2 = np.log(np.abs(t2) + j) * np.cos(j * t1.imag)
        magnitude = mag_part1 + mag_part2 + np.sqrt(float(j))

        # guarded divisions in the angle terms
        d1 = t1.real + 1.0
        d2 = t2.imag + 1.0

        term1 = 0.0
        term2 = 0.0
        if np.isfinite(d1) and abs(d1) > EPS:
            term1 = ang1 * np.cos(j / d1)
        if np.isfinite(d2) and abs(d2) > EPS:
            term2 = ang2 * np.sin(j / d2)

        angle = term1 - term2 + np.sin(j * (np.pi / 6.0))

        # polar synth -> complex
        cf[j - 1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))

    return cf

ALLOWED["p480"]=p480

def p481(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec_seq = np.linspace(t1.real, t2.real, n)
    imc_seq = np.linspace(t1.imag, t2.imag, n)
    for j in range(1, n+1):
        term1 = np.exp(1j * np.sin(5 * np.pi * rec_seq[j-1])) * (t1.real + j)
        term2 = np.exp(1j * np.cos(3 * np.pi * imc_seq[j-1])) * (t2.imag + j**2)
        cf[j-1] = term1 + term2 + np.log(np.abs(t1)*np.abs(t2)+1)
    for k in range(1, n+1):
        cf[k-1] *= (1 + 0.05 * k * np.sin(cf[k-1].real) + 0.05j * k * np.cos(cf[k-1].imag))
    for r in range(1, n+1):
        cf[r-1] += np.conj(cf[r-1]) * np.sin(r * t1.real) * np.cos(r * t2.imag)
    return cf.astype(np.complex128)
    
ALLOWED["p481"]=p481

def p482(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)

    r1   = math.hypot(t1.real, t1.imag)          # |t1|
    ang1 = math.atan2(t1.imag, t1.real)          # angle(t1)
    ang2 = math.atan2(t2.imag, t2.real)          # angle(t2)

    denom = t1.imag + 1.0
    if abs(denom) < 1e-10:
        return cf  # avoid division by ~0 in cos(j/(Im(t1)+1))

    t1_conj = np.complex128(t1.real - 1j * t1.imag)
    t2_conj = np.complex128(t2.real - 1j * t2.imag)

    for j in range(1, n + 1):
        jf = float(j)

        # mag_part1 = log(|t1| + j)
        mag_part1 = math.log(r1 + jf)

        # mag_part2 = sin(j * Re(t2)) * cos(j / (Im(t1) + 1))
        mag_part2 = math.sin(jf * t2.real) * math.cos(jf / denom)

        # sqrt(j!) = exp(0.5 * lgamma(j+1))  (stable, njit-safe)
        sqrt_fact = math.exp(0.5 * math.lgamma(jf + 1.0))

        magnitude = mag_part1 * mag_part2 + sqrt_fact

        # angle parts (replace np.angle with atan2)
        angle_part1 = ang1 * math.sin(jf) + ang2 * math.cos(jf)
        angle_part2 = math.sin(jf * t1.real) - math.cos(jf * t2.imag)
        angle = angle_part1 + angle_part2

        # exp(i*angle) via cos/sin
        ca = math.cos(angle)
        sa = math.sin(angle)

        cf[j - 1] = (
            np.complex128(magnitude * ca + 1j * magnitude * sa)
            + t1_conj * math.sin(jf)
            - t2_conj * math.cos(jf)
        )

    return cf
    
ALLOWED["p482"]=p482

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

def p484(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        mag = np.log(np.abs(t1) + j) * np.sqrt(j) + np.sin(j * t2.real)**2 + np.cos(j * t1.imag / (j +1))
        angle = np.angle(t1)*j + np.sin(j * t1.real * t2.real) - np.cos(j * t2.imag)
        cf[j-1] = mag * (np.cos(angle) +1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p484"]=p484

def p485(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec_seq = np.linspace(t1.real, t2.real, n)
    imc_seq = np.linspace(t1.imag, t2.imag, n)
    for j in range(1, n+1):
        phase = np.sin(j * np.pi /4) * np.cos(j * np.pi /3) + np.log(np.abs(rec_seq[j-1] + imc_seq[j-1]) +1)
        magnitude = np.sqrt(rec_seq[j-1]**2 + imc_seq[j-1]**2)**(1 + 0.1 *j) * np.abs(np.sin(j)) + np.abs(np.cos(j /2))
        cf[j-1] = magnitude * (np.cos(phase) + 1j * np.sin(phase))
    return cf.astype(np.complex128)
    
ALLOWED["p485"]=p485

def p486(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        mag_real = np.log(np.abs(t1) + np.abs(t2) + j) * np.abs(np.sin(j * t1.real) + np.cos(j * t2.imag))
        mag_imag = np.log(np.abs(t1) + np.abs(t2) + j) * np.abs(np.sin(j * t1.imag) - np.cos(j * t2.imag))
        angle_real = np.angle(t1)*np.sin(j /n * np.pi) + np.angle(t2)*np.cos(j /n * np.pi)
        angle_imag = np.angle(t1)*np.cos(j /n * np.pi) - np.angle(t2)*np.sin(j /n * np.pi)
        cf[j-1] = (mag_real * np.cos(angle_real) + mag_imag * np.sin(angle_imag)) +\
                    1j * (mag_real * np.sin(angle_real) - mag_imag * np.cos(angle_imag))
    return cf.astype(np.complex128)
    
ALLOWED["p486"]=p486

def p487(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        rec_seq = np.linspace(t1.real, t2.real, n)
        imc_seq = np.linspace(t1.imag, t2.imag, n)
        mag_part1 = np.log(np.abs(rec_seq[j-1]) +1) * np.sin(j * np.pi /4)
        mag_part2 = np.abs(t2)**((j %5) +1)
        mag = mag_part1 + mag_part2 * np.cos(j * t1.real)
        angle_part1 = np.angle(t1)*np.cos(j /3)
        angle_part2 = np.sin(j * t2.imag) + np.cos(j * np.pi /6)
        angle = angle_part1 + angle_part2
        cf[j-1] = mag * (np.cos(angle) + 1j * np.sin(angle))
    for k in range(1, n+1):
        if k >1 and k <n:
            cf[k-1] += 0.5 * (cf[k-2] * np.conj(cf[k])) * np.cos(k * np.pi /n)
        elif k ==1:
            cf[k-1] += 0.3 * np.conj(cf[k]) * np.sin(k * np.pi /n)
        else:
            cf[k-1] += 0.3 * np.conj(cf[k-2]) * np.sin(k * np.pi /n)
    return cf.astype(np.complex128)
    
ALLOWED["p487"]=p487

def p488(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    r1 = t1.real
    i1 = t1.imag
    r2 = t2.real
    i2 = t2.imag
    for j in range(1, n+1):
        mag = np.log(np.abs(t1) + j) * np.abs(np.sin(j * t1.real)) + np.log(np.abs(t2) + j) * np.abs(np.cos(j * t1.imag / (j +1)))
        angle_part1 = np.angle(t1)*np.sin(j) + np.angle(t2)*np.cos(j)
        angle_part2 = np.sin(j * t1.real) - np.cos(j * t2.imag)
        angle = angle_part1 + angle_part2
        cf[j-1] = mag * np.exp(1j * angle) + np.conj(t1) * np.sin(j) - np.conj(t2) * np.cos(j)
    return cf.astype(np.complex128)
    
ALLOWED["p488"]=p488

def p489(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)

    # Precompute angles safely
    ang1 = np.arctan2(t1.imag, t1.real)
    ang2 = np.arctan2(t2.imag, t2.real)

    for j in range(1, n + 1):
        # accumulate real magnitude contribution
        sum_mag = 0.0

        # k = 1..j term: (t1.real**k) * sin(k*angle(t1))
        base1 = t1.real
        for k in range(1, j + 1):
            # integer power is safe
            p = 1.0
            for _ in range(k):
                p *= base1
            sum_mag += p * np.sin(k * ang1)

        # r = 1..(n-j) term: (t2.imag**r) * cos(r*angle(t2))
        base2 = t2.imag
        up = n - j
        for r in range(1, up + 1):
            q = 1.0
            for _ in range(r):
                q *= base2
            sum_mag += q * np.cos(r * ang2)

        # magnitude: guard log domain
        mag = np.log(1.0 + abs(sum_mag))

        # angle: real scalars; denominators > 0
        angle = (sum_mag / float(j + 1)) + (sum_mag / float(n - j + 1))

        # polar synthesis (avoid complex exp)
        cf[j - 1] = mag * (np.cos(angle) + 1j * np.sin(angle))

    return cf
    
ALLOWED["p489"]=p489

def p490(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        mag = t1.real**((j %5)+1) + np.abs(t2)**(np.floor(j /7)+1) + np.log(j +1)*np.sin(j * np.pi /4)
        angle = np.angle(t1)*np.cos(j * np.pi /6) + np.angle(t2)*np.sin(j * np.pi /8)
        cf[j-1] = mag * np.exp(1j * angle)
    for k in range(1, n+1):
        cf[k-1] += np.conj(t1) * np.cos(k * np.pi /5) + np.conj(t2) * np.sin(k * np.pi /3)
    return cf.astype(np.complex128)
    
ALLOWED["p490"]=p490

def p491(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec_seq = np.linspace(t1.real, t2.real, n)
    imc_seq = np.linspace(t1.imag, t2.imag, n)
    for j in range(1, n+1):
        r = rec_seq[j-1]
        i_part = t1.imag * np.sin(j) + t2.imag * np.cos(j)
        magnitude = np.log(np.abs(t1) + np.abs(t2) + j) * (j**1.5 + np.prod(np.arange(1, (j %5)+2)))
        angle = np.angle(t1)*np.cos(j * np.pi /n) + np.angle(t2)*np.sin(j * np.pi /n)
        cf[j-1] = magnitude * (np.cos(angle) +1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p491"]=p491

def p492(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        real_part = t1.real * np.sin(j * np.pi /7) + t2.real * np.log(j +1)
        imag_part = t1.imag * np.cos(j**2 /5) - t2.imag * np.exp(-j /10)
        magnitude = (np.abs(t1) + np.abs(t2)) * (j**1.5 + (n -j)**1.2)
        angle = np.angle(t1)*np.sqrt(j) + np.angle(t2)*np.sin(j * np.pi /3)
        cf[j-1] = (real_part +1j * imag_part) * (np.cos(angle) + np.sin(angle)*1j) * magnitude
    return cf.astype(np.complex128)
    
ALLOWED["p492"]=p492

def p493(z,a,state):
    t1, t2 = z[0], z[1]
    n =40
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        mag = np.log(np.abs(t1) + np.abs(t2) +j) * (1 + np.sin(j) + np.cos(j /3))
        angle = np.angle(t1)*j + np.angle(t2)*np.sin(j /2)
        cf[j-1] = mag * (np.cos(angle) +1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p493"]=p493

def p494(z,a,state):
    t1, t2 = z[0], z[1]
    n =35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        rec = t1.real * j
        imc = t2.imag / j
        mag = np.log(np.abs(t1) + j) * (1 + np.sin(j * np.pi /7)) * (1 + np.cos(j * np.pi /5))
        angle = np.angle(t1)*np.sin(j /3) + np.angle(t2)*np.cos(j /4) + np.sin(j /2)
        cf[j-1] = mag * (np.cos(angle) +1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p494"]=p494

def p495(z,a,state):
    t1, t2 = z[0], z[1]
    n =35
    cf = np.zeros(n, dtype=np.complex128)
    rec1 = t1.real
    imc1 = t1.imag
    rec2 = t2.real
    imc2 = t2.imag
    for j in range(1, n+1):
        mag = (np.sin(rec1 * j) + np.cos(imc2 * j**1.2)) * np.log(1 +j) + np.abs(t1)**0.5 * np.abs(t2)**0.3
        ang = np.angle(t1)*np.sin(j) + np.angle(t2)*np.cos(j**2)
        cf[j-1] = mag * np.exp(1j * ang)
    return cf.astype(np.complex128)
    
ALLOWED["p495"]=p495

def p496(z,a,state):
    t1, t2 = z[0], z[1]
    n =35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        mag = np.log(np.abs(t1)*j +1) * np.sin(j * np.pi * t2.real / (j +1)) +\
                np.log(np.abs(t2)*(n -j +1) +1)*np.cos(j * np.pi * t1.imag / (j +1))
        ang = np.angle(t1)*np.sin(j /2) + np.angle(t2)*np.cos(j /3) + np.log(j +1)
        cf[j-1] = mag * (np.cos(ang) +1j * np.sin(ang))
    return cf.astype(np.complex128)

ALLOWED["p496"]=p496

def p497(z,a,state):
    t1, t2 = z[0], z[1]
    n =35
    cf = np.zeros(n, dtype=np.complex128)
    rec_seq = np.linspace(t1.real, t2.real, n)
    imc_seq = np.linspace(t1.imag, t2.imag, n)
    for j in range(1, n+1):
        mag_part1 = np.log(np.abs(rec_seq[j-1] + imc_seq[j-1]) +1)
        mag_part2 =1 + np.sin(j * np.pi /6)*np.cos(j * np.pi /4)
        magnitude = mag_part1 * mag_part2 * (1 + (j * rec_seq[j-1] * imc_seq[j-1])**(1/3))
        ang_part1 = np.angle(t1)*np.sin(j) + np.angle(t2)*np.cos(j)
        ang_part2 = np.sin(j * np.pi /5)*np.cos(j * np.pi /7)
        angle = ang_part1 + ang_part2
        cf[j-1] = magnitude * np.exp(1j * angle)
    for k in range(1, n+1):
        if k >1 and k <n:
            cf[k-1] += 0.5 * (cf[k-2] * np.conj(cf[k])) * np.cos(k * np.pi /n)
        elif k ==1:
            cf[k-1] += 0.3 * np.conj(cf[k]) * np.sin(k * np.pi /n)
        else:
            cf[k-1] += 0.3 * np.conj(cf[k-2]) * np.sin(k * np.pi /n)
    return cf.astype(np.complex128)

ALLOWED["p497"]=p497

def p498(z,a,state):
    t1, t2 = z[0], z[1]
    n =35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        magnitude = np.log(np.abs(t1)*j + np.abs(t2)/(j +1) +1)
        angle = np.angle(t1)*np.sin(j) + np.angle(t2)*np.cos(j)
        cf[j-1] = magnitude * (np.cos(angle) +1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p498"]=p498

def p499(z,a,state):
    t1, t2 = z[0], z[1]
    n =35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        mag_part1 = np.log(np.abs(t1) + j**1.5) * np.sin(j * np.pi /6)
        mag_part2 = np.abs(t2)/(j +2) + np.cos(j * np.pi /4)
        magnitude = mag_part1 + mag_part2 * np.exp(-j /10)
        angle_part1 = np.angle(t1)*np.cos(j /3)
        angle_part2 = np.angle(t2)*np.sin(j /5) + np.sin(j**2 /7)
        angle = angle_part1 + angle_part2
        cf[j-1] = magnitude * (np.cos(angle) +1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p499"]=p499

def p500(z,a,state):
    t1, t2 = z[0], z[1]
    n =35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        angle = np.sin(j * np.angle(t1)) * np.cos(j * np.angle(t2)) + np.sin(j**2 * np.angle(t1 + t2))
        magnitude = (np.abs(t1)**j + np.abs(t2)**(n -j)) * np.log(j + np.abs(t1 - t2)) / (1 + (j %5))
        cf[j-1] = magnitude * (np.cos(angle) +1j * np.sin(angle)) + np.conj(t1)*np.sin(j * t2.imag)
    return cf.astype(np.complex128)
    
ALLOWED["p500"]=p500

def p501(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        mag = np.abs(t1)**((j %5)+1) * np.log(np.abs(t2) +j) * np.sin(j *0.5) + np.angle(t2)*np.cos(j *0.3)
        angle = np.angle(t1)*np.sin(j *0.5) + np.angle(t2)*np.cos(j *0.3)
        cf[j-1] = mag * np.exp(1j * angle)
        if j %3 ==1:
            cf[j-1] += np.conj(t1)*np.sin(j * t2.real) - np.conj(t2)*np.cos(j * t1.imag)
        elif j%3 ==2:
            cf[j-1] += t1.real * t2.imag * np.sin(j)
        else:
            cf[j-1] += t2.real * t1.imag * np.cos(j)
    for k in range(1, n+1):
        cf[k-1] = cf[k-1] * (1 +0.05 *k) + 0.02 * np.sin(k * np.angle(t1)) * np.cos(k * np.angle(t2))
    return cf.astype(np.complex128)
    
ALLOWED["p501"]=p501

def p502(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        phase_r = np.sin(j * t1.real) + np.cos(j * t2.real)
        phase_i = np.cos(j * t1.imag) - np.sin(j * t2.imag)
        magnitude = np.log(np.abs(t1)**j + np.abs(t2)**(n -j) +1)
        angle = np.angle(t1)*j - np.angle(t2)*(n -j) + phase_r * phase_i
        cf[j-1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p502"]=p502

def p503(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec1 = t1.real
    imc1 = t1.imag
    rec2 = t2.real
    imc2 = t2.imag
    for j in range(1, n+1):
        mag_part1 = np.log(np.abs(t1) +j) * (1 + np.sin(j)*np.cos(j))
        mag_part2 = np.angle(t1)*j + np.angle(t2)/(j +1) + np.sin(j * rec1) - np.cos(j *imc2)
        magnitude = mag_part1 * mag_part2
        angle = np.angle(t1)*j + np.angle(t2)/(j +1) + np.sin(j * rec1) - np.cos(j * imc2)
        cf[j-1] = magnitude * np.exp(1j * angle) + np.conj(t1)*np.sin(j) + np.conj(t2)*np.cos(j)
    return cf.astype(np.complex128)
    
ALLOWED["p503"]=p503

def p504(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec_seq = np.linspace(t1.real, t2.real, n)
    imc_seq = np.linspace(t1.imag, t2.imag, n)
    for j in range(1, n+1):
        magnitude = np.log(np.abs(rec_seq[j-1] + imc_seq[j-1]) +1) * (np.abs(t1)**j + np.abs(t2)**(n -j))
        angle = np.angle(t1)*np.sin(j) + np.angle(t2)*np.cos(j/2)
        cf[j-1] = magnitude * (np.cos(angle) +1j * np.sin(angle))
    for k in range(1, np.floor(n/2)+1):
        if k <=n:
            cf[k-1] += np.prod(rec_seq[:k]) * np.conj(t2)**k
            cf[n -k] += np.sum(imc_seq[:k]) * np.sin(np.abs(t1)*k) * np.cos(np.abs(t2)/(k +1))
    return cf.astype(np.complex128)
   
ALLOWED["p504"]=p504

def p505(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec1 = t1.real
    imc1 = t1.imag
    rec2 = t2.real
    imc2 = t2.imag
    for j in range(1, n+1):
        mag = (np.sin(rec1 * j) + np.cos(imc2 * j**1.2)) * np.log(1 +j) + np.abs(t1)**0.5 * np.abs(t2)**0.3
        ang = np.angle(t1)*np.sin(j) + np.angle(t2)*np.cos(j**2)
        cf[j-1] = mag * np.exp(1j * ang)
    return cf.astype(np.complex128)

ALLOWED["p505"]=p505

def p506(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        r = t1.real + j * t2.real
        im = t1.imag - j * t2.imag
        mag = np.log(np.abs(t1) +j) * np.abs(np.sin(j * np.angle(t1)) + np.cos(j * np.angle(t2)))
        angle = np.angle(t1)*j + np.angle(t2)*(n -j)
        cf[j-1] = mag * (np.cos(angle) +1j * np.sin(angle))
    for k in range(1, n+1):
        cf[k-1] += np.conj(cf[k-1]) * np.sin(k * t1.real) / (1 +k)
    return cf.astype(np.complex128)

ALLOWED["p506"]=p506

def p507(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec_seq = np.linspace(t1.real, t2.real, n)
    imc_seq = np.linspace(t1.imag, t2.imag, n)
    for j in range(1, n+1):
        mag = np.log(np.abs(rec_seq[j-1] * imc_seq[j-1]) +1) * ((j %4) +1)
        angle = np.sin(j * np.pi * t1.real) + np.cos(j * np.pi * t2.imag) + np.abs(t1.real)**0.5 * np.abs(t2.imag)**0.3
        cf[j-1] = mag * (np.cos(angle) +1j * np.sin(angle))
    for k in range(1, n+1):
        cf[k-1] = cf[k-1] * (1 +0.5 * np.conj(cf[max(0, k -3)]) ) + 0.3 * np.sin(k)*np.cos(k)
    for r in range(1, n+1):
        cf[r-1] += cf[r-1] * (1 + 0.5 * np.conj(cf[max(0, r -2)])) + 0.3 * np.sin(r)*np.cos(r)
    return cf.astype(np.complex128)

ALLOWED["p507"]=p507

def p508(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        k =j**2
        angle_part = np.angle(t1)*np.sin(j * np.pi /4) + np.angle(t2)*np.cos(j * np.pi /6)
        magnitude_part = np.log(np.abs(t1) + np.abs(t2) +j) * (1 + np.sin(j /3)) * (1 + np.cos(j /5))
        perturbation = np.conj(t1)**0.5 * np.sin(j) + np.conj(t2)**0.3 * np.cos(j)
        cf[j-1] = magnitude_part * np.exp(1j * angle_part) + 0.1 * perturbation
    for k in range(1,6):
        for r in range(1,8):
            index = (k -1)*7 + r
            if index <=n:
                cf[index-1] = cf[index-1] * (1 +0.05 * np.sin(k * r)) +0.02 * np.cos(k +r)
    return cf.astype(np.complex128)

ALLOWED["p508"]=p508

def p509(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, 36):
        mag_part = np.log(np.abs(t1) + np.abs(t2) +j) * (1 + np.sin(j /4) * np.cos(j /6))
        angle_part = np.angle(t1)*np.sqrt(j) - np.angle(t2)*np.cos(j /2)
        cf[j-1] = mag_part * (np.cos(angle_part) +1j * np.sin(angle_part)) + np.conj(t1)*t2**j
    return cf.astype(np.complex128)
    
ALLOWED["p509"]=p509

def p510(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    r1 = t1.real
    im1 = t1.imag
    r2 = t2.real
    im2 = t2.imag
    for j in range(1, n+1):
        mag_part1 = np.log(np.abs(r1 +j) +1) * np.sqrt(j)
        mag_part2 = np.abs(t2)*np.sin(j /3)
        magnitude = mag_part1 + mag_part2
        ang_part1 = np.angle(t1) + np.cos(j * np.pi /5)
        ang_part2 = np.sin(j * np.pi /7) * np.angle(t2)
        angle = ang_part1 + ang_part2
        cf[j-1] = magnitude * (np.cos(angle) +1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p510"]=p510

def p511(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        term1 = np.sin(j) * t1.real + np.cos(j *2) * t2.imag
        term2 = np.cos(j /3) * t2.real - np.sin(j /4) * t1.imag
        magnitude = np.log(np.abs(term1 + term2) +1) *j
        angle = np.angle(t1)*np.sin(j /2) + np.angle(t2)*np.cos(j /3) + np.sin(j)**2
        cf[j-1] = magnitude * np.exp(1j * angle) +0.3 * np.exp(1j*(angle /2))
    return cf.astype(np.complex128)

ALLOWED["p511"]=p511

def p512(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        mag = np.abs(t1)**j + np.abs(t2)**(n -j) + np.log(j + np.abs(t1 - t2))
        angle = np.angle(t1)*j - np.angle(t2)*(n -j) + np.sin(j)*np.cos(j)
        cf[j-1] = mag * (np.cos(angle) +1j * np.sin(angle))
    for k in range(1,6):
        cf[k-1] *= np.conj(t1) * np.sin(k)
    for r in range(1,6):
        cf[n -r] *= np.conj(t2) * np.cos(r)
    return cf.astype(np.complex128)
    
ALLOWED["p512"]=p512

def p513(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    real_seq = np.linspace(t1.real, t2.real, n)
    imag_seq = np.linspace(t1.imag, t2.imag, n)
    for j in range(1, n+1):
        magnitude = np.log(np.abs(t1 +j) +1) * np.sin(j * np.pi /7) + np.cos(j * np.pi /5) * np.prod(np.arange(1, (j %5)+2))
        angle = np.angle(t2) + np.sin(j * np.pi /3) * np.cos(j * np.pi /4) + np.tan(j * np.pi /6)
        cf[j-1] = magnitude * (np.cos(angle) +1j * np.sin(angle))
    for k in range(1, (n //2)+1):
        idx = k *2
        if idx <=n:
            cf[idx-1] *= np.exp(1j * real_seq[k-1]/(np.abs(imag_seq[k-1]) +1))
    for r in range(1, (n%3)+2):
        cf[r-1] = cf[r-1]**2 / (1 + np.abs(cf[r-1]))
    return cf.astype(np.complex128)
    
ALLOWED["p513"]=p513

def p514(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec_seq = np.linspace(t1.real, t2.real, n)
    imc_seq = np.linspace(t1.imag, t2.imag, n)
    for r in range(1, n+1):
        mag_part = np.log(np.abs(t1) + np.abs(t2) + r) * (np.sin(rec_seq[r-1] * np.pi / (r +1)) + np.cos(imc_seq[r-1] * np.pi / (r +2)))
        angle_part = np.angle(t1)*np.sin(r /5) + np.angle(t2)*np.cos(r /7)
        intricate_sum =0
        for j in range(1,4):
            intricate_sum += (t1.real**j - t2.imag**j) * np.sin(j * rec_seq[r-1]) * np.cos(j * imc_seq[r-1])
        cf[r-1] = mag_part * np.exp(1j * angle_part) + intricate_sum * np.conj(t1)*t2**r
    return cf.astype(np.complex128)
    
ALLOWED["p514"]=p514

def p515(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        phase = np.sin(j * t1.real + np.cos(j * t2.imag)) + np.cos(j * t2.real - np.sin(j * t1.imag))
        magnitude = np.log(np.abs(t1) +1) + np.sqrt(j)*t2.real - np.abs(t1.imag) + (t1.real * t2.real) / (j +1)
        cf[j-1] = magnitude * np.exp(1j * phase)
    return cf.astype(np.complex128)
    
ALLOWED["p515"]=p515

def p516(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = t1.real
    imc = t2.imag
    for j in range(1, n+1):
        mag_part = np.log(np.abs(t1) +j) * np.sin(j * np.pi /7) + np.sqrt(j)*np.cos(j * np.angle(t2))
        angle_part = np.angle(t1)*j**0.5 + np.angle(t2)*np.log(j +1)
        cf[j-1] = mag_part * np.exp(1j * angle_part) + np.conj(t1)**j * np.cos(j * np.pi /5)
    return cf.astype(np.complex128)
    
ALLOWED["p516"]=p516

def p517(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for r in range(1, n+1):
        mag_part = np.log(np.abs(t1) + np.abs(t2) +r) * np.sin(r * t1.real / (1 +r)) + np.cos(r * t1.imag / (1 +r))
        angle_part = np.angle(t1)*r + np.angle(t2)*(n -r) + np.sin(r * t1.real)*np.cos(r * t2.imag)
        cf[r-1] = mag_part * np.exp(1j * angle_part) + np.conj(mag_part * np.exp(-1j * angle_part))
    return cf.astype(np.complex128)
    
ALLOWED["p517"]=p517

def p518(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    r1 = t1.real
    i1 = t1.imag
    r2 = t2.real
    i2 = t2.imag
    for j in range(1, n+1):
        mag = np.log(np.abs(t1) +1) * (j**(np.sin(j) + np.cos(j))) + np.sin(j * np.pi /7) * np.cos(j * np.pi /5)
        angle = np.angle(t1)*np.sin(j /3) + np.angle(t2)*np.cos(j /4) + np.sin(j * np.pi /6)
        for k in range(1, min(j,5)+1):
            mag += r1**k * i2**(j -k) * np.log(k +1)
            angle += np.angle(t1)**k - np.angle(t2)**(j -k) * np.cos(k * np.pi /8)
        cf[j-1] = mag * np.exp(1j * angle)
    return cf.astype(np.complex128)
    
ALLOWED["p518"]=p518

def p519(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        mag = np.log(np.abs(t1) +j) * np.abs(np.sin(j * np.pi /7)) + np.sqrt(j) * np.cos(j * np.angle(t2))
        angle = np.angle(t1)*np.sin(j) + np.angle(t2)*np.cos(j /3)
        cf[j-1] = mag * (np.cos(angle) +1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p519"]=p519

def p520(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec_seq = np.linspace(t1.real, t2.real, n)
    imc_seq = np.linspace(t1.imag, t2.imag, n)
    for j in range(1, n+1):
        magnitude = np.log(np.abs(rec_seq[j-1] + imc_seq[j-1]) +1) * (1 + np.sin(j * np.pi /5)) * (1 + np.cos(j * np.pi /7))
        angle = np.angle(t1)*np.sin(j /3) + np.angle(t2)*np.cos(j /4) + np.sin(j * t1.real)*np.cos(j * t2.imag)
        cf[j-1] = magnitude * (np.cos(angle) +1j * np.sin(angle))
    for k in range(1, n+1):
        cf[k-1] *= (1 +0.5 * np.conj(cf[max(0, k -2)]) ) +0.3 * np.sin(k)*np.cos(k)
    return cf.astype(np.complex128)
    
ALLOWED["p520"]=p520

def p521(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        mag_part1 = np.log(np.abs(t1) +j)
        mag_part2 = np.sin(j * np.angle(t2)) + np.cos(j /2 * np.angle(t1))
        magnitude = mag_part1 * (1 + mag_part2**2)
        angle_part1 = np.angle(t1 +j)
        angle_part2 = np.cos(j * t2.imag)
        angle = angle_part1 + angle_part2
        cf[j-1] = magnitude * (np.cos(angle) +1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p521"]=p521

def p522(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        mag =0
        angle =0
        for k in range(1, j+1):
            mag += (t1.real**k * t2.real**(j -k) + t1.imag**k * t2.imag**(j -k))
            angle += (np.angle(t1) * np.sin(k) - np.angle(t2) * np.cos(j -k))
        mag *= np.log(np.abs(t1) + np.abs(t2) +j)
        cf[j-1] = mag * (np.cos(angle) +1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p522"]=p522

def p523(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    r1 = t1.real
    i1 = t1.imag
    r2 = t2.real
    i2 = t2.imag
    for j in range(1, n+1):
        mag_part1 = np.log(np.abs(t1) +j) * np.sin(j * np.pi /6)
        mag_part2 = (r2**j + i1**(n -j)) * np.cos(j * np.pi /4)
        mag = mag_part1 + mag_part2
        angle = np.angle(t1)*np.sin(j /3) + np.angle(t2)*np.cos(j /5) + np.sin(j * np.pi /7)
        cf[j-1] = mag * np.exp(1j * angle) + np.conj(t2)*(j %4)
    return cf.astype(np.complex128)

ALLOWED["p523"]=p523

def p524(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        phase = np.sin(j * np.angle(t1)) + np.cos(j * np.angle(t2)) + np.log(np.abs(t1) + np.abs(t2) + j)
        magnitude = (j**2 + np.sqrt(j)) * np.abs(np.sin(j /3)) + np.exp(-j /10) * np.abs(t1 + t2)
        cf[j-1] = magnitude * (np.cos(phase) +1j * np.sin(phase))
    return cf.astype(np.complex128)
    
ALLOWED["p524"]=p524

def p525(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        mag_part = np.log(np.abs(t1) +1) * (j**np.abs(t2.real)) + np.sum(np.arange(1, j+1)) * np.sqrt(j)
        angle_part = np.angle(t1)*np.sin(j) + np.angle(t2)*np.cos(j) + np.sin(j * t1.imag)*np.cos(j * t2.imag)
        coeff = mag_part * np.exp(1j * angle_part)
        for k in range(1,4):
            coeff += (t1.real**k) * (t2.imag**k) * np.sin(k *j) / (k +1)
        cf[j-1] = coeff + np.conj(t2)*t1.real**((j %5)+1)
    return cf.astype(np.complex128)
    
ALLOWED["p525"]=p525

def p526(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        r = t1.real + t2.real *j
        im = t1.imag - t2.imag *j
        mag = np.log(np.abs(r + im*1j) +1) * np.sin(j * np.pi /n) + np.cos(j * np.pi /5)
        angle = np.angle(t1)*np.sin(r * np.pi /7) + np.angle(t2)*np.cos(r * np.pi /4)
        cf[j-1] = mag * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p526"]=p526

def p527(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        angle = np.angle(t1)*j + np.angle(t2)/(j +1)
        magnitude = np.abs(t1)**j + np.abs(t2)**(n -j) + np.log(np.abs(t1) + np.abs(t2) +j)
        phase = np.sin(j * t1.real) * np.cos(j * t2.imag) + np.sin(t1.imag * j /2)
        cf[j-1] = magnitude * (np.cos(angle + phase) +1j * np.sin(angle - phase))
    for k in range(1, n//2 +1):
        cf[k-1] *= np.conj(cf[n -k])
    for r in range(1, n+1):
        cf[r-1] += np.exp(1j * (t1.real * r - t2.imag / (r +1)))
    return cf.astype(np.complex128)

ALLOWED["p527"]=p527

def p528(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n+1):
        k = j**1.5
        r = t1.real * np.log(k +1) + t2.real * np.sqrt(k +1)
        im_part = t1.imag * np.sin(k) + t2.imag * np.cos(k)
        magnitude = np.abs(t1)*k + np.abs(t2)/(k +1)
        angle = np.angle(t1)*np.sin(k /10) + np.angle(t2)*np.cos(k /10)
        cf[j-1] = (r +1j * im_part) * (1 + magnitude) * np.exp(1j * angle)
    for k in range(1,6):
        for r in range(1,8):
            index = (k -1)*7 + r
            if index <=n:
                cf[index-1] *= (1 +0.05 * np.sin(k * r)) +0.02 * np.cos(k +r)
    return cf.astype(np.complex128)
    
ALLOWED["p528"]=p528

def p529(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    r1 = t1.real
    i1 = t1.imag
    r2 = t2.real
    i2 = t2.imag
    for j in range(1, n+1):
        mag = np.log(np.abs(r1 +j)) * np.sin(j * np.pi * i2) + np.sqrt(j) * np.cos(j * np.pi * r2)
        angle = np.angle(t1)*np.log(j +1) + np.angle(t2)*np.sin(j*r1) + np.cos(j *i2)
        cf[j-1] = mag * (np.cos(angle) +1j * np.sin(angle)) + np.conj(t1)*np.sin(j) / (j +1)
    return cf.astype(np.complex128)
    
ALLOWED["p529"]=p529

def p530(z, a, state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)

    # Precompute
    r1, r2 = t1.real, t2.real
    i1, i2 = t1.imag, t2.imag
    ang1 = np.arctan2(t1.imag, t1.real)
    ang2 = np.arctan2(t2.imag, t2.real)
    if not np.isfinite(ang1): ang1 = 0.0
    if not np.isfinite(ang2): ang2 = 0.0

    denom_lin = float(n - 1) if n > 1 else 1.0
    EPS = 1e-12

    for j in range(n):
        # Linear interpolation to replace linspace
        t = float(j) / denom_lin
        r = r1 + (r2 - r1) * t
        m = i1 + (i2 - i1) * t

        # term1
        term1 = np.sin(r * (np.pi / float(j + 2))) * np.cos(m * (np.pi / float(j + 3)))

        # term2: log(|r+m|+1) * (t1.real)^(j+1)
        base_log = abs(r + m) + 1.0
        if not np.isfinite(base_log) or base_log <= 0.0:
            base_log = 1.0
        log_part = np.log(base_log)

        # integer power (Numba-safe)
        pw = 1.0
        rb = t1.real
        for _ in range(j + 1):
            pw *= rb

        term2 = log_part * pw

        # term3: (|r * m * (j+1)|)^(1/(j+1))
        prod_abs = abs(r * m * float(j + 1))
        if not np.isfinite(prod_abs) or prod_abs < EPS:
            prod_abs = 1.0
        term3 = prod_abs ** (1.0 / float(j + 1))

        mag = term1 + term2 + term3

        # angle
        angle = (
            ang1 * np.sin(m * (np.pi / float(j + 4)))
            + ang2 * np.cos(r * (np.pi / float(j + 5)))
            + np.log(float(j + 2))
        )
        if not np.isfinite(angle):
            angle = 0.0

        # polar synthesis
        cf[j] = mag * (np.cos(angle) + 1j * np.sin(angle))

    return cf

ALLOWED["p530"]=p530

def p531(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec1 = t1.real
    imc1 = t1.imag
    rec2 = t2.real
    imc2 = t2.imag

    for r in range(1, n + 1):
        if r % 3 == 1:
            mag = np.log(np.abs(t1 + r) + 1) * np.sin(r / n * np.pi) + np.cos(r * np.pi / 4)
            ang = np.angle(t1) + np.sin(r * np.pi / 6) * np.angle(t2)
        elif r % 3 == 2:
            mag = np.log(np.abs(t2 + r) + 1) * np.cos(r / n * np.pi) + np.sin(r * np.pi / 3)
            ang = np.angle(t2) + np.cos(r * np.pi / 5) * np.angle(t1)
        else:
            mag = np.log(np.abs(t1 * t2 + r) + 1) * np.sin(r / (2 * n) * np.pi) + np.cos(r * np.pi / 2)
            ang = np.angle(t1 * t2) + np.sin(r * np.pi / 4) * np.cos(r * np.pi / 3)
        cf[r - 1] = mag * np.exp(1j * ang)

    for k in range(n):
        if k < n / 3:
            cf[k] = cf[k] * (k + 1)
        elif k < 2 * n / 3:
            cf[k] = cf[k] * (-(k + 1))
        else:
            cf[k] = cf[k] * (1 / (k + 1))
    
    return cf.astype(np.complex128)
    
ALLOWED["p531"]=p531

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

def p533(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for k in range(1, n + 1):
        r = t1.real * np.log(k + 1) + t2.real * np.sin(k)
        im = t1.imag * np.cos(k) + t2.imag * np.log(k + 2)
        mag = np.log(np.abs(t1) + k**2) * (1 + np.sin(k / 3))
        ang = np.angle(t1) * np.cos(k / 4) + np.angle(t2) * np.sin(k / 5)
        cf[k - 1] = (r + 1j * im) * mag * np.exp(1j * ang)
    return cf.astype(np.complex128)
    
ALLOWED["p533"]=p533

def p534(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        r1 = t1.real
        r2 = t2.real
        im1 = t1.imag
        im2 = t2.imag
        mag_part1 = np.log(np.abs(t1) + np.abs(t2) + j) * (1 + np.sin(j * r1) + np.cos(j * im2))
        mag_part2 = np.sin(r2 * j**1.3) * np.cos(im1 * np.sqrt(j))
        magnitude = mag_part1 * mag_part2 + np.log(j + 1)
        angle = np.angle(t1) * np.sin(j / 2) + np.angle(t2) * np.cos(j / 3)
        cf[j - 1] = magnitude * (np.cos(angle) + np.sin(angle) * 1j)
    return cf.astype(np.complex128)
    
ALLOWED["p534"]=p534

def p535(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(t1.real, t2.real, num=n)
    imc = np.linspace(t1.imag, t2.imag, num=n)
    for j in range(1, n + 1):
        mag = np.log(np.abs(rec[j - 1] + imc[j - 1]*1j) + 1) * np.sin(j * np.pi / 5) + np.cos(j * np.pi / (j + 2))
        ang = np.angle(rec[j - 1] + imc[j - 1]*1j) + np.sin(j / n * np.pi * 4) - np.cos(j / n * np.pi * 3)
        cf[j - 1] = mag * (np.cos(ang) + np.sin(ang) * 1j)
    return cf.astype(np.complex128)
    
ALLOWED["p535"]=p535

def p536(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        r = t1.real * np.log(j + 1) + t2.real * np.sin(j / 3)
        im = t1.imag * np.cos(j / 4) - t2.imag * np.log(j + 2)
        magnitude = np.log(np.abs(t1) + j**1.2) * (1 + 0.5 * np.sin(j * np.pi / 6))
        angle = np.angle(t1) * np.cos(j / 5) + np.angle(t2) * np.sin(j / 7) + np.log(np.abs(t2) + 1)
        cf[j - 1] = magnitude * (np.cos(angle) + np.sin(angle) * 1j) + (r + im) * 1j
    return cf.astype(np.complex128)

ALLOWED["p536"]=p536

def p537(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec_seq = np.linspace(t1.real, t2.real, num=n)
    imc_seq = np.linspace(t1.imag, t2.imag, num=n)
    for k in range(1, n + 1):
        magnitude = np.log(np.abs(rec_seq[k - 1] + imc_seq[k - 1]) + 1) * (np.sin(k * np.pi / 7) + np.cos(k * np.pi / 5))
        angle = np.angle(t1 * t2) + np.sin(k) - np.cos(k / 2)
        cf[k - 1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p537"]=p537

def p538(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(t1.real, t2.real, num=n)
    imc = np.linspace(t1.imag, t2.imag, num=n)
    for j in range(1, n + 1):
        magnitude = np.log(np.abs(t1) + j) * (1 + np.sin(j * np.pi / 7)) + np.sqrt(j) * np.cos(j * np.pi / 5)
        angle = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j / 3) + np.sin(j * imc[j - 1]) - np.cos(j * rec[j - 1])
        cf[j - 1] = magnitude * (np.cos(angle) + np.sin(angle) * 1j)
    return cf.astype(np.complex128)
    
ALLOWED["p538"]=p538


def p539(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        k = j**2 + int(np.floor(t1.real))
        r = np.log(np.abs(t1) + j) * (1 + np.sin(j * np.angle(t2)))
        s = np.cos(j * t2.real) * np.sin(j * t1.imag) + np.cos(j * t2.imag)
        magnitude = r + np.log(np.abs(t2) + j)
        angle = s + np.sin(j * t1.real) * np.cos(np.angle(t1))
        cf[j - 1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p539"]=p539

def p540(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, 36):
        r1 = t1.real
        i1 = t1.imag
        r2 = t2.real
        i2 = t2.imag

        if j <= 10:
            mag = np.log(np.abs(t1) + 1) * np.sin(j * r2) + (np.abs(t2)**j) / (j + 1)
            angle = np.angle(t1) + j * np.angle(t2)
        elif j <= 25:
            mag = np.cos(j * i1) * np.log(np.abs(t2) + 1) + np.real(np.conj(t1)) / (j + 2)
            angle = np.angle(t1) * np.sin(j * np.pi / 5) + np.angle(t2) * np.cos(j * np.pi / 7)
        else:
            mag = np.sin(j * r1 + np.cos(j * i2)) * np.log(np.abs(t1) + np.abs(t2) + 1)
            angle = (np.angle(t1)**2) / (j + 3) + (np.angle(t2)**2) / (j + 4)
        cf[j - 1] = mag * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p540"]=p540

def p541(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        mag_part1 = np.log(np.abs(t1) + j) * np.sin(0.3 * j * t2.real)
        mag_part2 = np.log(np.abs(t2) + j) * np.cos(0.2 * j * t1.imag)
        mag = mag_part1 + mag_part2
        angle_part1 = np.angle(t1) + j * 0.1 * np.pi * np.sin(j / 5)
        angle_part2 = np.angle(t2) + j * 0.1 * np.pi * np.cos(j / 3)
        angle = angle_part1 + angle_part2
        cf[j - 1] = mag * (np.cos(angle) + np.sin(angle) * 1j)
    return cf.astype(np.complex128)

ALLOWED["p541"]=p541

def p542(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(t1.real, t2.real, num=n)
    imc = np.linspace(t1.imag, t2.imag, num=n)
    for j in range(1, n + 1):
        mag_sum = 0
        for k in range(1, j + 1):
            mag_sum += np.sin(k * t1.real) * np.cos(k * t2.imag)
        magnitude = np.log(np.abs(rec[j - 1] + imc[j - 1]) + 1) * mag_sum
        angle = 0
        for r in range(1, j + 1):
            angle += np.angle(t1) * np.sin(r) + np.angle(t2) * np.cos(r)
        cf[j - 1] = magnitude * (np.cos(angle) + np.sin(angle) * 1j)
    return cf.astype(np.complex128)
    
ALLOWED["p542"]=p542

def p543(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        mag_part1 = np.log(np.abs(t1) * j + 1)
        mag_part2 = np.log(np.abs(t2) + np.sqrt(j))
        mag_variation = mag_part1 * np.sin(j * t1.real) + mag_part2 * np.cos(j * t2.imag)
        angle_part1 = np.angle(t1) * j**1.3
        angle_part2 = np.angle(t2) / (j + 1)
        angle_variation = angle_part1 - angle_part2 + np.sin(j) * np.cos(j / 2)
        cf[j - 1] = (np.abs(mag_variation) + 1) * np.exp(1j * angle_variation)
    return cf.astype(np.complex128)

ALLOWED["p543"]=p543

def p544(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        mag_part1 = np.log(np.abs(t1) + j) * np.sqrt(np.abs(t2) + j)
        mag_part2 = np.sin(j * t1.real) * np.cos(j * t2.imag)
        magnitude = mag_part1 + mag_part2 * (j % 5 + 1)
        
        angle_part1 = np.angle(t1) * np.sin(j / 3)
        angle_part2 = np.angle(t2) * np.cos(j / 4)
        angle_part3 = np.log(np.abs(t1) + np.abs(t2) + j)
        angle = angle_part1 + angle_part2 + angle_part3
        
        cf[j - 1] = magnitude * (np.cos(angle) + np.sin(angle) * 1j)
    for k in range(1, n + 1):
        cf[k - 1] = cf[k - 1] * np.conj(t1) / (np.abs(t2) + 1)
    return cf.astype(np.complex128)

ALLOWED["p544"]=p544

def p545(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(t1.real, t2.real, num=n)
    imc = np.linspace(t1.imag, t2.imag, num=n)
    for j in range(1, n + 1):
        mag = (rec[j - 1]**3 + imc[j - 1]**2) * np.log(np.abs(t1) + 1) + np.sin(j * np.pi / 4) * np.cos(j * np.pi / 3)
        ang = np.angle(t1) + np.angle(t2) * j + np.sin(j * t1.real * t2.imag)
        cf[j - 1] = mag * np.exp(1j * ang) + np.conj(t1)**(j % 5) * np.cos(j * imc[j - 1])
    for k in range(1, n + 1):
        cf[k - 1] += (np.prod(rec[:k] + imc[:k])) * np.sin(k * np.angle(t2))
    for r in range(1, n + 1):
        cf[r - 1] = cf[r - 1] * (1 + np.abs(t1 - t2) / (r + 1)) + np.log(np.abs(t1 + t2) + 1) * np.cos(r * np.angle(t1))
    return cf.astype(np.complex128)

ALLOWED["p545"]=p545

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

def p547(z, a, state):
    t1 = z[0]
    t2 = z[1]

    n = 35
    cf = np.zeros(n, dtype=np.complex128)

    # Precompute basics
    re1, im1 = t1.real, t1.imag
    re2, im2 = t2.real, t2.imag
    abs_t1 = np.hypot(re1, im1)
    abs_t2 = np.hypot(re2, im2)

    ang_t1 = np.arctan2(im1, re1)
    ang_t2 = np.arctan2(im2, re2)

    # First pass
    for j in range(1, n + 1):
        lj1 = np.log(abs_t1 + j)          # j >= 1, so argument > 0
        lj2 = np.log(abs_t2 + j)

        mag = lj1 * np.sin(j * re1) + lj2 * np.cos(j * im1)
        angle = ang_t1 * (j * j) - ang_t2 * np.sqrt(float(j))

        cf[j - 1] = mag * (np.cos(angle) + 1j * np.sin(angle))

    # Second pass
    for k in range(1, n + 1):
        denom = 1.0 + k * re2
        if denom == 0.0:
            frac = 0.0  # avoid division by zero; neutralize that term
        else:
            frac = (k + re1) / denom

        osc = np.cos(k * ang_t1) + np.sin(k * ang_t2)
        cf[k - 1] *= (frac + osc)

    # Third pass
    for r in range(1, n + 1):
        # power(|t1|, 1/r) — safe since |t1| >= 0
        pr = abs_t1 ** (1.0 / r)
        cf[r - 1] += cf[n - r].conjugate() * pr

    return cf

ALLOWED["p547"]=p547

def p548(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        r = t1.real * np.log(j + 1) + t2.real * np.sqrt(j)
        im = t1.imag * np.sin(j) + t2.imag * np.cos(j * np.pi / 4)
        mag = np.abs(t1)**(j % 5 + 1) + np.abs(t2)**(n - j + 1)
        angle = np.angle(t1) * j + np.angle(t2) / (j + 1)
        cf[j - 1] = (mag * np.exp(1j * angle)) + np.conj(t1) * np.conj(t2) / (j + 1)
    return cf.astype(np.complex128)

ALLOWED["p548"]=p548

def p549(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        mag = np.log(np.abs(t1) * j + 1) + np.sin(j * t2.real)**2 + np.cos(j * t1.imag)
        angle = np.angle(t1) * j + np.angle(t2) / (j + 1) + np.sin(j / 3)
        cf[j - 1] = mag * (np.cos(angle) + np.sin(angle) * 1j)
    return cf.astype(np.complex128)
    
ALLOWED["p549"]=p549

def p550(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(t1.real, t2.real, num=n)
    imc = np.linspace(t1.imag, t2.imag, num=n)
    for j in range(1, n + 1):
        coeff = 0
        for k in range(1, j + 1):
            coeff += (rec[k - 1]**2 - imc[k - 1]**3) * np.exp(1j * np.sin(k * np.pi / n))
            coeff += np.conj(t1) * np.cos(k * np.pi / (j + 1))
        for r in range(1, n - j + 1):
            coeff += np.log(np.abs(rec[j - 1] + imc[j - 1]) + 1) * r * np.sin(r * np.pi / n)
            coeff += np.abs(t2)**r * np.cos(r * np.angle(t1 + t2))
        cf[j - 1] = coeff
    return cf.astype(np.complex128)
    
ALLOWED["p550"]=p550

def p551(z, a, state):
    t1, t2 = z[0], z[1]
    n = 40
    cf = np.zeros(n, dtype=np.complex128)

    # Precompute magnitudes and angles safely
    a1 = np.abs(t1)
    a2 = np.abs(t2)
    ang1 = np.arctan2(t1.imag, t1.real)
    ang2 = np.arctan2(t2.imag, t2.real)

    # fixed scalars
    re1, im1 = t1.real, t1.imag
    re2, im2 = t2.real, t2.imag

    # sin/cos periods
    pi_over_n = np.pi / float(n)
    two_pi_over_n = 2.0 * np.pi / float(n)

    for j in range(1, n + 1):
        jf = float(j)

        # ---- magnitude parts (guarded logs and roots) ----
        # log(|t|+1) is always defined
        mag_part1 = np.log(a1 + 1.0) * np.sin(jf * pi_over_n)
        mag_part2 = np.log(a2 + 1.0) * np.cos(jf * two_pi_over_n)

        # (re1 * im2) could be negative; take absolute before 1/j power
        base_rm = abs(re1 * im2)
        root_val = base_rm ** (1.0 / jf) if base_rm > 0.0 else 0.0

        mag_variation = mag_part1 + mag_part2 + root_val

        # ---- angle parts (Numba-safe) ----
        angle_part1 = ang1 * np.sin(jf / 2.0)
        angle_part2 = ang2 * np.cos(jf / 3.0)
        angle_variation = angle_part1 + angle_part2 + np.sin(jf) * np.cos(jf)

        # unit phasor without complex exp
        comp = np.cos(angle_variation) + 1j * np.sin(angle_variation)

        # ---- assemble coefficient ----
        cf[j - 1] = (
            mag_variation * comp
            + np.conj(t1) * np.sin(jf)
            + np.conj(t2) * np.cos(jf)
        )

    return cf
        
ALLOWED["p551"]=p551

def p552(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        mag_part = np.log(np.abs(t1) + j) * np.sin(j * np.pi / 7) + np.log(np.abs(t2) + 1) * np.cos(j * np.pi / 11)
        angle_part = np.angle(t1) * np.sin(j / 2) + np.angle(t2) * np.cos(j / 3)
        intricate_sum = 0
        for k in range(1, j + 1):
            intricate_sum += (t1.real**k - t2.imag**k) * np.sin(k * np.pi / (j + 1))
        cf[j - 1] = mag_part * np.exp(1j * angle_part) + np.conj(t1) * (t2**(j % 5)) + intricate_sum
    return cf.astype(np.complex128)
    
ALLOWED["p552"]=p552

def p553(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        mag_sum = 0
        ang_sum = 0
        for k in range(1, j + 1):
            term_mag = np.log(np.abs(t1) + k**2) * np.sin(k * t2.real) + np.cos(k * t1.imag)
            term_ang = np.angle(t2) * np.sqrt(k) + np.sin(k / 2)
            mag_sum += term_mag
            ang_sum += term_ang
        cf[j - 1] = (mag_sum * np.exp(1j * ang_sum)) + np.conj(t1) * (t2**j)
    return cf.astype(np.complex128)
    
ALLOWED["p553"]=p553

def p554(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec1 = t1.real
    rec2 = t2.real
    imc1 = t1.imag
    imc2 = t2.imag
    for j in range(1, n + 1):
        term1 = rec1**j * np.sin(j * np.pi / 4)
        term2 = imc2 * np.cos(j * np.pi / 3)
        term3 = np.log(np.abs(t1) + 1) * t2.real**((j % 5) + 1)
        term4 = np.abs(t1 + t2)**(0.5 * j)
        angle = np.angle(t1) + np.angle(t2) * j
        magnitude = np.abs(term1 + term2 + term3) + term4
        cf[j - 1] = magnitude * (np.cos(angle) + np.sin(angle) * 1j)
    for k in range(1, n + 1):
        cf[k - 1] = cf[k - 1] * np.exp(1j * np.sin(k)) + np.conj(cf[(k % n)])
    return cf.astype(np.complex128)
    
ALLOWED["p554"]=p554

def p555(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        real_part = t1.real**j + t2.real**((j % 3) + 1)
        imag_part = t1.imag**((j % 4) + 1) + t2.imag**((j % 5) + 1)
        magnitude = real_part + imag_part + np.log(np.abs(t1 * t2) + 1)
        angle = np.sin(j * np.angle(t1)) + np.cos(j * np.angle(t2))
        cf[j - 1] = magnitude * (np.cos(angle) + np.sin(angle)*1j)
    return cf.astype(np.complex128)

ALLOWED["p555"]=p555

def p556(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(t1.real, t2.real, num=n)
    imc = np.linspace(t1.imag, t2.imag, num=n)
    for j in range(1, n + 1):
        angle = np.sin(j * rec[j - 1]) + np.cos(j * imc[j - 1]) + np.angle(t1 + t2)
        magnitude = np.log(np.abs(rec[j - 1]**2 + imc[j - 1]**2) + 1) * (j**1.5 + np.prod(rec[:j] + imc[:j]))
        cf[j - 1] = magnitude * (np.cos(angle) + np.sin(angle) * 1j)
    return cf.astype(np.complex128)

ALLOWED["p556"]=p556

def p557(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for k in range(1, n + 1):
        mag_sum = 0
        for j in range(1, k + 1):
            mag_sum += np.sin(j * np.pi / 7) * np.cos(j * np.pi / 5) + np.log(np.abs(t1 * j + t2 / (j + 1)) + 1)
        magnitude = mag_sum * (1 + np.abs(t1 - t2) / 10)
        
        angle_sum = 0
        for j in range(1, k + 1):
            angle_sum += np.angle(t1 + j * t2) * np.sin(j * np.pi / 9) - np.angle(t2) * np.cos(j * np.pi / 11)
        angle = angle_sum / k
        
        cf[k - 1] = magnitude * (np.cos(angle) + np.sin(angle) * 1j)
    return cf.astype(np.complex128)
    
ALLOWED["p557"]=p557

def p558(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        r = t1.real * np.log(j + 1) + t2.real * np.sqrt(j)
        theta = np.sin(j * t1.imag) + np.cos(j * t2.imag) + np.angle(t1 + t2)
        for k in range(1, 4):
            r += t1.real * k / (j + 1)
            theta += np.sin(k * np.pi / j)
        cf[j - 1] = r * (np.cos(theta) + 1j * np.sin(theta))
    return cf.astype(np.complex128)
    
ALLOWED["p558"]=p558

def p559(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        temp_real = 0
        temp_imag = 0
        for k in range(1, j + 1):
            temp_real += (t1.real**k) * np.sin(k * t2.real) / (k + 1)
            temp_imag += (t2.imag**(j - k + 1)) * np.cos((j - k + 1) * t1.imag) / (j - k + 2)
        magnitude = np.log(np.abs(temp_real + temp_imag) + 1) * j
        angle = t1.real * t2.imag / j + np.sin(j * np.angle(t1 + t2))
        cf[j - 1] = magnitude * (np.cos(angle) + np.sin(angle) * 1j)
    return cf.astype(np.complex128)
    
ALLOWED["p559"]=p559

def p560(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    re1 = t1.real
    im1 = t1.imag
    re2 = t2.real
    im2 = t2.imag
    for j in range(1, n + 1):
        magnitude = np.log(np.abs(t1) + j) * np.sin(j * np.pi / n) + np.log(np.abs(t2) + np.sqrt(j)) * np.cos(j * np.pi / (n + 1))
        angle = np.angle(t1) * np.sin(j / 3) + np.angle(t2) * np.cos(j / 5)
        cf[j - 1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p560"]=p560

def p561(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        r = t1.real * np.log(j + 1) + t2.imag * np.sin(j * t1.real)
        theta = np.angle(t1) * np.cos(j) - np.angle(t2) * np.sin(j)
        mag_variation = np.abs(t1)**j / (1 + j) + np.abs(t2)**(np.sqrt(j))
        cf[j - 1] = (r + 1j * theta) * (mag_variation + np.sin(j) - np.cos(j))
    return cf.astype(np.complex128)
    
ALLOWED["p561"]=p561

def p562(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        rec = t1.real + (t2.real - t1.real) * j / n
        imc = t1.imag + (t2.imag - t1.imag) * j / n
        mag = np.log(np.abs(rec) + 1) * (j**2 + np.sqrt(n - j + 1)) * np.sin(j) + np.prod(np.arange(1, (j % 5) + 2))
        angle = np.sin(rec * np.pi / 7) + np.cos(imc * np.pi / 5) + np.angle(t1) - np.angle(t2)
        cf[j - 1] = mag * (np.cos(angle) + np.sin(angle) * 1j)
    return cf.astype(np.complex128)
    
ALLOWED["p562"]=p562

def p563(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, 27):
        mag_part = np.log(np.abs(t1) + j) * (j**2 + t2.real * np.sin(j * np.pi / 5))
        angle_part = np.angle(t1) * np.cos(j * np.pi / 7) - np.angle(t2) * np.sin(j * np.pi / 3)
        cf[j - 1] = mag_part * (np.cos(angle_part) + np.sin(angle_part) * 1j)
    return cf.astype(np.complex128)

ALLOWED["p563"]=p563

def p564(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, 27):
        mag_part1 = np.log(np.abs(t1) + j) * np.sin(j * t2.real)
        mag_part2 = np.cos(j * t1.imag) * np.abs(t2 + j)
        mag = mag_part1 + mag_part2
        
        angle_part1 = np.angle(t1) * j**0.5
        angle_part2 = np.sin(j * np.pi / 7) + np.cos(j * np.pi / 11)
        angle = angle_part1 + angle_part2
        
        cf[j - 1] = mag * np.exp(1j * angle)
    return cf.astype(np.complex128)

ALLOWED["p564"]=p564

def p565(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, 27):
        a = t1.real + np.sin(j) * t2.real
        b = t1.imag + np.cos(j) * t2.imag
        c = np.log(np.abs(t1) + np.abs(t2) + 1)
        d = np.angle(t1) * j + np.angle(t2) / (j + 1)
        cf[j - 1] = (a + 1j * b) * c * (np.cos(d) + 1j * np.sin(d)) + np.conj(t1)**j * np.cos(j) - np.conj(t2) * np.sin(j)
    return cf.astype(np.complex128)
    
ALLOWED["p565"]=p565

def p566(z,a,state):
    t1, t2 = z[0], z[1]
    n = 35
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        mag_part1 = np.log(np.abs(t1) + j)
        mag_part2 = np.prod(np.arange(1, (j % 5) + 2))
        magnitude = mag_part1 * (mag_part2 + np.sqrt(j))
        
        angle_part1 = np.sin(j * np.angle(t1))
        angle_part2 = np.cos(np.angle(t2) / (j + 1))
        angle = angle_part1 + angle_part2
        
        cf[j - 1] = magnitude * (np.cos(angle) + np.sin(angle) * 1j)
        
        # Introduce variation based on imaginary components
        cf[j - 1] += (t1.imag - t2.imag) * np.sin(j)**2
    return cf.astype(np.complex128)
    
ALLOWED["p566"]=p566

def p567(z, a, state):
    t1 = z[0]
    t2 = z[1]
    n = 25
    cf = np.zeros(n, dtype=np.complex128)

    re1 = t1.real
    im1 = t1.imag
    re2 = t2.real
    im2 = t2.imag

    for j in range(1, n + 1):
        # safe abs and log
        abs_t1 = np.hypot(re1, im1)
        abs_t2 = np.hypot(re2, im2)
        mag_arg = abs_t1 + abs_t2 + j
        if mag_arg <= 0.0 or np.isnan(mag_arg) or np.isinf(mag_arg):
            cf[j - 1] = 0.0j
            continue

        # safe magnitude
        mag = np.log(mag_arg)
        mag *= 1.0 + np.sin(j * re1) + np.cos(j * im2)
        # safe fractional powers
        re1_safe = np.abs(re1)
        im2_safe = np.abs(im2)
        mag *= np.sqrt(re1_safe) * (im2_safe ** 0.3)

        # safe angle approximation (avoid np.angle)
        angle1 = np.arctan2(im1, re1)
        angle2 = np.arctan2(im2, re2)
        angle = angle1 * j + angle2 * np.sin(j / 3.0) + np.cos(j / 5.0)

        if np.isnan(mag) or np.isnan(angle) or np.isinf(mag) or np.isinf(angle):
            cf[j - 1] = 0.0j
        else:
            cf[j - 1] = mag * np.exp(1j * angle)

    return cf
    
ALLOWED["p567"]=p567

def p568(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 25
    cf = np.zeros(degree + 1, dtype=np.complex128)
    rec = np.linspace(t1.real, t2.real, num=degree + 1)
    imc = np.linspace(t1.imag, t2.imag, num=degree + 1)
    for r in range(1, degree + 2):
        mag = np.log(np.abs(t1 * rec[r - 1] + t2 * imc[r - 1]) + 1) * (1 + np.sin(r * np.pi / 4)) + np.cos(r * np.pi / 5)
        ang = np.angle(t1) * rec[r - 1] + np.angle(t2) * imc[r - 1] + np.sin(r * np.pi / 6)
        cf[r - 1] = mag * (np.cos(ang) + np.sin(ang) * 1j)
    return cf.astype(np.complex128)
    
ALLOWED["p568"]=p568

def p569(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 25
    cf = np.zeros(degree + 1, dtype=np.complex128)
    rec = np.linspace(t1.real, t2.real, num=degree + 1)
    imc = np.linspace(t1.imag, t2.imag, num=degree + 1)
    for j in range(1, degree + 2):
        mag_part = np.log(np.abs(t1) + j) + np.sin(rec[j - 1] * np.pi / (j + 1)) * np.cos(imc[j - 1] * np.pi / (j + 2))
        angle_part = np.angle(t1) * j + np.sin(rec[j - 1] / (j + 1)) - np.cos(imc[j - 1] / (j + 2))
        cf[j - 1] = mag_part * (np.cos(angle_part) + np.sin(angle_part) * 1j)
    return cf.astype(np.complex128)
    
ALLOWED["p569"]=p569

def p570(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 25
    cf = np.zeros(degree + 1, dtype=np.complex128)
    for j in range(1, degree + 2):
        sum_mag = 0
        sum_ang = 0
        for k in range(1, j + 1):
            term = t1 * k + t2 / k
            sum_mag += np.log(np.abs(term) + 1)
            angle_term = t1 * np.sin(k * t2.real) + t2 * np.cos(k * t1.imag)
            sum_ang += np.angle(angle_term)
        cf[j - 1] = sum_mag * (np.cos(sum_ang) + np.sin(sum_ang) * 1j)
    return cf.astype(np.complex128)
    
ALLOWED["p570"]=p570

def p571(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 25
    cf = np.zeros(degree + 1, dtype=np.complex128)
    for j in range(1, degree + 2):
        k = j % 7 + 1
        r = t1.real * np.sin(k * t2.real) + t2.real * np.cos(k * t1.imag)
        i_part = t1.imag * np.cos(k * t2.real) - t2.imag * np.sin(k * t1.imag)
        magnitude = np.log(np.abs(t1) + j) * (j**1.5) / (1 + np.log(j + 1))
        angle = np.angle(t1) * j + np.log(j + 1) * np.angle(t2)
        cf[j - 1] = magnitude * np.exp(1j * angle) * (r + 1j * i_part)
    return cf.astype(np.complex128)
    
ALLOWED["p571"]=p571

def p572(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    for j in range(1, 26):
        summation_mag = 0
        summation_ang = 0
        for k in range(1, j + 1):
            summation_mag += np.log(np.abs(t1 * k + t2) + 1) * np.sin(k * t1.real)
            summation_ang += np.angle(t1 * k - t2) + np.cos(k * t2.imag)
        cf[j - 1] = summation_mag * (np.cos(summation_ang) + np.sin(summation_ang) * 1j)
    return cf.astype(np.complex128)
    
ALLOWED["p572"]=p572

def p573(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 25
    cf = np.zeros(degree + 1, dtype=np.complex128)
    for j in range(1, degree + 2):
        part1 = t1.real * np.sin(j * np.angle(t2)) + t2.real * np.cos(j * np.angle(t1))
        part2 = t1.imag * np.cos(j * t2.real) - t2.imag * np.sin(j * t1.real)
        magnitude = np.log(np.abs(t1) + j) + np.sum(np.sin(np.arange(1, j + 1) * t1.real) * np.cos(np.arange(1, j + 1) * t2.imag))
        angle = np.angle(t1) * j + np.angle(t2) * j**0.5
        cf[j - 1] = magnitude * (np.cos(angle) + np.sin(angle) * 1j)
    return cf.astype(np.complex128)
    
ALLOWED["p573"]=p573

def p574(z,a,state):
    t1, t2 = z[0], z[1]
    deg = 25
    cf = np.zeros(deg + 1, dtype=np.complex128)
    for j in range(1, deg + 2):
        r_part = t1.real**j * np.cos(j * np.angle(t2)) + t2.real**(deg + 1 - j) * np.sin(j * np.angle(t1))
        im_part = t1.imag**j * np.sin(j * np.angle(t2)) - t2.imag**(deg + 1 - j) * np.cos(j * np.angle(t1))
        magnitude = np.log(np.abs(r_part + im_part) + 1) * (j**1.5)
        angle = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j)
        cf[j - 1] = magnitude * (np.cos(angle) + np.sin(angle) * 1j)
    for k in range(1, deg + 2):
        factor = (k % 3) + 1
        cf[k - 1] *= (t1.real**(k % 5) + t2.imag**(k % 4)) * np.sin(k * np.angle(cf[k - 1]))
    for r in range(2, deg):
        cf[r - 1] += (cf[r - 2] * cf[r]) / (1 + np.abs(cf[r - 1]))
    return cf.astype(np.complex128)

ALLOWED["p574"]=p574

def p575(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 25
    cf = np.zeros(degree + 1, dtype=np.complex128)
    for j in range(1, degree + 2):
        k = (j * 3 + 7) % 10
        r = t1.real * np.cos(j) + t2.real * np.sin(k)
        im = t1.imag * np.sin(j) + t2.imag * np.cos(k)
        magnitude = np.log(np.abs(t1) + np.abs(t2) + j) * (1 + j % 5) / (j + 1)
        angle = np.angle(t1) * np.sin(k) + np.angle(t2) * np.cos(j)
        cf[j - 1] = magnitude * (np.cos(angle) + np.sin(angle) * 1j)
    return cf.astype(np.complex128)

ALLOWED["p575"]=p575

def p576(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 25
    cf = np.zeros(degree + 1, dtype=np.complex128)
    for j in range(1, degree + 2):
        r = j / degree
        k = (j**2 + 3*j + 1)
        mag = np.log(np.abs(t1) + np.abs(t2) + r * k) * (1 + np.sin(j) * np.cos(k))
        angle = np.angle(t1) * np.sin(k) + np.angle(t2) * np.cos(r * j)
        cf[j - 1] = mag * (np.cos(angle) + np.sin(angle) * 1j)
    return cf.astype(np.complex128)

ALLOWED["p576"]=p576

def p577(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 25
    cf = np.zeros(degree + 1, dtype=np.complex128)
    rec_seq = np.linspace(t1.real, t2.real, num=degree + 1)
    imc_seq = np.linspace(t1.imag, t2.imag, num=degree + 1)
    for j in range(1, degree + 2):
        r = rec_seq[j - 1]
        im = imc_seq[j - 1]
        mag = np.log(np.abs(r) + np.abs(im) + 1) * np.sin(2 * np.pi * r) + np.cos(3 * np.pi * im)
        ang = np.angle(t1) * j + np.sin(im * np.pi)
        cf[j - 1] = mag * (np.cos(ang) + np.sin(ang) * 1j) + np.conj(t2) * np.cos(j * np.pi / degree)
    return cf.astype(np.complex128)
    
ALLOWED["p577"]=p577

def p578(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    for j in range(1, 26):
        r = t1.real * j + t2.real / (j + 1)
        im = t1.imag * np.sin(j) + t2.imag * np.cos(j / 2)
        magnitude = np.log(np.abs(t1) + j) + np.sin(j * t2.real) * np.cos(j * t1.imag)
        angle = np.angle(t1) * j - np.angle(t2) / (j + 0.5)
        cf[j - 1] = magnitude * np.exp(1j * angle) + np.conj(t1) * t2**(j % 7)
    for k in range(1, 26):
        cf[k - 1] = cf[k - 1] * (1 + 0.05 * np.sin(k * t1.real)) + 0.05j * np.cos(k * t2.imag)
    return cf.astype(np.complex128)

ALLOWED["p578"]=p578

def p579(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 25
    cf = np.zeros(degree + 1, dtype=np.complex128)
    for j in range(1, degree + 2):
        mag_part = np.abs(t1)**j * np.log(np.abs(t2) + 1) + np.abs(t2)**(degree +1 -j) * np.sin(j)
        angle_part = np.angle(t1) * np.cos(j) + np.angle(t2) * np.sin(j)
        cf[j - 1] = mag_part * np.exp(1j * angle_part)
    for k in range(1, degree + 2):
        cf[k - 1] += np.conj(t1) * t2.real / (k + 1)
        cf[k - 1] = cf[k - 1] * (1 + np.sin(k * np.pi / 12)) + np.cos(k * np.pi / 18) * t1.imag
    for r in range(1, degree + 2):
        cf[r - 1] += (np.abs(t1) * np.abs(t2)) / (r + 2)
        cf[r - 1] = cf[r - 1] * np.log(np.abs(cf[r - 1]) + 1) + np.exp(1j * np.sin(r))
    return cf.astype(np.complex128)
    
ALLOWED["p579"]=p579

def p580(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 25
    cf = np.zeros(degree + 1, dtype=np.complex128)
    for j in range(1, degree + 2):
        mag_part1 = np.log(np.abs(t1) + j) * np.sin(2 * np.pi * t1.real / (j + 1))
        mag_part2 = np.log(np.abs(t2) + j) * np.cos(2 * np.pi * t2.imag / (j + 1))
        magnitude = mag_part1 + mag_part2 + (t1.real * t1.imag * j)
        angle = np.angle(t1) * j + np.angle(t2) * (degree +1 - j) + np.sin(j) - np.cos(j)
        cf[j - 1] = magnitude * np.exp(1j * angle)
    return cf.astype(np.complex128)
    
ALLOWED["p580"]=p580

def p581(z,a,state):
    t1, t2 = z[0], z[1]
    n = 26
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        term1 = t1.real * np.sin(j * np.angle(t2)) + t2.real * np.cos(j * t1.imag)
        term2 = t1.imag * np.cos(j * t2.real) - t2.imag * np.sin(j * np.angle(t1))
        magnitude = np.log(np.abs(t1) + j) + np.abs(t2)**j
        angle = term1 + term2
        cf[j - 1] = magnitude * (np.cos(angle) + np.sin(angle) * 1j) + np.conj(t1) * np.conj(t2) / (j + 1)
    return cf.astype(np.complex128)
    
ALLOWED["p581"]=p581

def p582(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(26, dtype=np.complex128)
    for j in range(1, 27):
        r = t1.real**(j % 5 + 1) + t2.real**(j % 7 + 1)
        imc = t1.imag**(j % 3 + 2) - t2.imag**(j % 4 + 1)
        magnitude = np.log(np.abs(t1) + j) * np.sin(r) + np.cos(imc)
        angle = np.angle(t1) * np.cos(j) + np.angle(t2) * np.sin(imc)
        cf[j - 1] = magnitude * np.exp(1j * angle)
    return cf.astype(np.complex128)

ALLOWED["p582"]=p582

def p583(z,a,state):
    t1, t2 = z[0], z[1]
    n = 26
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        mag = np.log(np.abs(t1.real * j**2 + t2.imag / (j + 1)) + 1) * (1 + np.sin(j * np.pi / 4)) * (1 + np.cos(j * np.pi / 5))
        angle = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j**2) + np.log(j + 1)
        cf[j - 1] = mag * (np.cos(angle) + np.sin(angle) * 1j)
    return cf.astype(np.complex128)

ALLOWED["p583"]=p583

def p584(z,a,state):
    t1, t2 = z[0], z[1]
    n = 26
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        r = t1.real * np.sin(j) + t2.real * np.cos(j)
        k = t1.imag * np.cos(j) - t2.imag * np.sin(j)
        magnitude = np.log(np.abs(r) + 1) * (1 + (j % 5)) + np.abs(k)**1.5
        angle = np.angle(t1) * j + np.angle(t2) * np.sqrt(j)
        cf[j - 1] = magnitude * np.exp(1j * angle) + np.conj(t1) * np.conj(t2) / (j + 1)
    return cf.astype(np.complex128)

ALLOWED["p584"]=p584

def p585(z,a,state):
    t1, t2 = z[0], z[1]
    n = 26
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        k = j**2
        r = t1.real * np.sin(k * np.pi / 7) + t2.real * np.cos(k * np.pi / 5)
        s = t1.imag * np.cos(k * np.pi / 3) - t2.imag * np.sin(k * np.pi / 4)
        mag = np.log(np.abs(t1) + np.abs(t2) + k) * (np.abs(r) + np.abs(s) + 1)
        angle = np.angle(t1) * np.log(k + 1) + np.sin(r) - np.cos(s)
        cf[j - 1] = mag * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p585"]=p585

def p586(z,a,state):
    t1, t2 = z[0], z[1]
    n = 26
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        mag_part1 = np.log(np.abs(t1) + j) * np.sin(j * np.pi / 7)
        mag_part2 = np.log(np.abs(t2) + j) * np.cos(j * np.pi / 5)
        magnitude = mag_part1 + mag_part2 + j
        angle_part1 = np.angle(t1) * np.cos(j * np.pi / 4)
        angle_part2 = np.angle(t2) * np.sin(j * np.pi / 3)
        angle = angle_part1 + angle_part2
        cf[j - 1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    for k in range(1, n + 1):
        cf[k - 1] = cf[k - 1] * (1 + 0.05 * k**2) * np.exp(-k / n)
    return cf.astype(np.complex128)
    
ALLOWED["p586"]=p586

def p587(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    n = 25
    for j in range(1, n + 1):
        r1 = t1.real + j * t2.real
        i1 = t1.imag - j * t2.imag
        magnitude = np.log(np.abs(t1) + j) * np.sin(j * np.angle(t2)) + np.cos(j * np.pi / 12)
        angle = np.angle(t1) * np.cos(j) + np.sin(j * np.angle(t2))
        cf[j - 1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p587"]=p587

def p588(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    degree = 25
    cf = np.zeros(35, dtype=np.complex128)
    for j in range(1, degree + 1):
        mag = np.log(np.abs(t1) + j**1.3) * np.abs(np.sin(j * np.pi / 4)) + np.abs(t2) * np.cos(j * np.pi / 6)
        angle = np.angle(t1) * np.sin(j / 3) + np.angle(t2) * np.cos(j / 5) + np.sin(j * np.pi / 7)
        cf[j - 1] = mag * (np.cos(angle) + 1j * np.sin(angle))
    for k in range(degree + 1, 36):
        cf[k - 1] = np.log(k + 1) * (np.sin(k * np.angle(t1)) + 1j * np.cos(k / 2))
    return cf.astype(np.complex128)
    
ALLOWED["p588"]=p588

def p589(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    for j in range(1, 26):
        k = j + 1
        r = t1.real * np.log(np.abs(t2) + 1) / (j + 1)
        theta = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j * np.pi / 8)
        mag = (np.abs(t1)**j + np.abs(t2)**(25 - j)) * (1 + np.sin(j * np.pi / 5))
        cf[j - 1] = mag * np.exp(1j * theta) + np.conj(t1) * np.cos(j * np.pi / 7)
    for k in range(1, 26):
        cf[k - 1] += (1 + 0.05 * np.sin(k * t1.real)) + 0.05j * np.cos(k * t2.imag)
    for r in range(1, 26):
        cf[r - 1] += (np.abs(t1) * np.abs(t2)) / (r + 2)
        cf[r - 1] = cf[r - 1] * np.log(np.abs(cf[r - 1]) + 1) + np.exp(1j * np.sin(r))
    return cf.astype(np.complex128)
    
ALLOWED["p589"]=p589

def p590(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 25
    cf = np.zeros(degree + 1, dtype=np.complex128)
    for j in range(1, degree + 2):
        r_part = t1.real * j + t2.real / (j +1)
        i_part = t1.imag * np.sin(j) + t2.imag * np.cos(j)
        magnitude = np.log(np.abs(r_part) + 1) + np.abs(t1) * np.abs(t2) / (j +1)
        angle = np.angle(t1) * j - np.angle(t2) / (j +1) + np.sin(j * np.pi / 5)
        cf[j - 1] = magnitude * (np.cos(angle) + np.sin(angle) * 1j)
    for k in range(1, degree + 2):
        cf[k - 1] *= (1 + 0.05 * k**2)
    return cf.astype(np.complex128)

ALLOWED["p590"]=p590

def p591(z,a,state):
    t1, t2 = z[0], z[1]
    n = 25
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        k = j * 3 + t1.real - t2.imag
        r = np.log(np.abs(t1) + np.abs(t2) + j) * (np.sin(j * np.angle(t1)) + np.cos(j * np.angle(t2)))
        angle = np.sin(k * np.angle(t1)) + np.cos(k * np.angle(t2))
        cf[j - 1] = r * (np.cos(angle) + np.sin(angle) * 1j) + np.conj(t1)**k * np.sin(j)
    return cf.astype(np.complex128)

ALLOWED["p591"]=p591

def p592(z,a,state):
    t1, t2 = z[0], z[1]
    n = 25
    cf = np.zeros(n + 1, dtype=np.complex128)
    for j in range(1, n + 2):
        sum_val = 0
        prod_val = 1
        for k in range(1, j + 1):
            sum_val += np.sin(k * t1.real) * np.cos(k * t2.imag)
            prod_val *= (t1.real + t2.imag * k)
        magnitude = np.log(np.abs(t1) + j) * sum_val + prod_val
        angle = np.angle(t1) * j - np.angle(t2) * j**2
        cf[j - 1] = magnitude * (np.cos(angle) + np.sin(angle) * 1j) + np.conj(t1) * np.conj(t2)
    return cf.astype(np.complex128)

ALLOWED["p592"]=p592

def p593(z,a,state):
    t1, t2 = z[0], z[1]
    deg = 25
    cf = np.zeros(deg + 1, dtype=np.complex128)
    for j in range(1, deg + 2):
        mag = np.log(np.abs(t1) + j) * (1 + np.sin(j * t2.real)) * (1 + np.cos(j * t1.imag))
        ang = np.angle(t1) * j + np.angle(t2) * np.sqrt(j)
        cf[j - 1] = mag * complex(np.cos(ang), np.sin(ang))
    return cf.astype(np.complex128)

ALLOWED["p593"]=p593

def p594(z,a,state):
    t1, t2 = z[0], z[1]
    n = 26
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        r = t1.real * (j**2 + np.sin(j)) + t2.real * np.log(j + 1)
        im = t1.imag * (np.cos(j / 2) + j) + t2.imag * np.sin(j)
        mag = np.abs(r + im * 1j) * (1 + np.cos(j * np.pi / 5))
        angle = np.angle(r + im * 1j)
        cf[j - 1] = mag * (np.cos(angle) + np.sin(angle) * 1j)
    return cf.astype(np.complex128)

ALLOWED["p594"]=p594

def p595(z,a,state):
    t1, t2 = z[0], z[1]
    n = 26
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        magnitude = np.log(np.abs(t1) + j) * np.sqrt(j) * (1 + np.sin(j)) + np.abs(t2) / (j + 1)
        angle = np.angle(t1) * np.cos(j) + np.angle(t2) * np.sin(j) + np.sin(j * np.pi / 3)
        cf[j - 1] = magnitude * (np.cos(angle) + np.sin(angle) * 1j) + np.conj(t1) * np.cos(j / n)
    return cf.astype(np.complex128)

ALLOWED["p595"]=p595

def p596(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(26, dtype=np.complex128)
    n = 25
    for j in range(1, n + 1):
        k = (j * 3 + 7) % 10
        r = t1.real * np.sin(j) + t2.imag * np.cos(k)
        mag = np.log(np.abs(t1) + j**2) * np.sin(k * np.pi / 4) + np.cos(r)
        angle = np.angle(t1) * np.cos(j) + np.sin(k * np.angle(t2))
        cf[j - 1] = mag * (np.cos(angle) + np.sin(angle) * 1j) + np.conj(t2)**k
    cf[25] = np.sum(np.abs(cf[:n]) * np.cos(np.arange(1, n + 1) * np.pi / 6)) + (np.abs(t1) * np.abs(t2))
    return cf.astype(np.complex128)
    
ALLOWED["p596"]=p596

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

def p598(z,a,state):
    t1, t2 = z[0], z[1]
    n = 26
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        rec1 = t1.real * np.log(j + 1)
        rec2 = t2.real * np.sin(j * np.pi / 7)
        imc1 = t1.imag * np.cos(j * np.pi / 5)
        imc2 = t2.imag * np.sin(j * np.pi / 3)
        mag = np.log(np.abs(t1) + j) * (1 + (j % 2)) + np.abs(t2)**(j / 2)
        ang = np.angle(t1) * np.sin(j / 4) - np.angle(t2) * np.cos(j / 6)
        cf[j - 1] = (rec1 + rec2) + 1j * (imc1 + imc2) + mag * np.exp(1j * ang)
    return cf.astype(np.complex128)
    
ALLOWED["p598"]=p598

def p599(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 25
    cf = np.zeros(degree + 1, dtype=np.complex128)
    for j in range(1, degree + 2):
        part1 = np.sin(j * t1.real) * np.cos(j * t2.imag)
        part2 = np.log(np.abs(t1) + j) + np.log(np.abs(t2) + j)
        part3 = t1.real**j - t2.imag**j
        angle = np.angle(t1) * j + t2.real * np.sin(j)
        cf[j - 1] = (part1 * part2 + part3) * np.exp(1j * angle)
    return cf.astype(np.complex128)
    
ALLOWED["p599"]=p599

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

def p601(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 25
    cf = np.zeros(degree + 1, dtype=np.complex128)
    for j in range(1, degree + 2):
        mag = np.log(1 + np.abs(t1) * j) * (1 + np.sin(j * np.pi / 6)) + np.cos(j * np.pi / 7)
        angle = np.angle(t1) * np.cos(j / 3) + np.angle(t2) * np.sin(j / 4) + np.log(j + 1)
        cf[j - 1] = mag * (np.cos(angle) + 1j * np.sin(angle)) + np.conj(t2)**(j % 5)
    return cf.astype(np.complex128)
    
ALLOWED["p601"]=p601

def p602(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 25
    cf = np.zeros(degree + 1, dtype=np.complex128)
    rec = np.linspace(t1.real, t2.real, num=degree + 1)
    imc = np.linspace(t1.imag, t2.imag, num=degree + 1)
    for j in range(1, degree + 2):
        mag_sum = 0
        angle_sum = 0
        for k in range(1, j + 1):
            mag_sum += np.log(np.abs(rec[k - 1] + imc[k - 1]) + 1) * np.sin(k * np.pi / 4)
            angle_sum += np.angle(rec[k - 1] + imc[k - 1] * 1j) * np.cos(k * np.pi / 3)
        magnitude = mag_sum * np.cos(j * np.pi / 5) + np.abs(t1) / (j + 1)
        angle = angle_sum + np.sin(j * np.angle(t2))
        cf[j - 1] = magnitude * np.exp(1j * angle) + np.conj(t2) * np.cos(j)
    return cf.astype(np.complex128)
    
ALLOWED["p602"]=p602

def p603(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(26, dtype=np.complex128)
    for j in range(1, 27):
        phase = np.sin(j * t1.real) + np.cos(j * t2.imag)
        mag = np.log(np.abs(t1) + np.abs(t2) + j) * (j**1.5) * ((j % 5) + 1)
        real_part = t1.real * np.cos(phase) - t2.imag * np.sin(phase)
        imag_part = t2.real * np.sin(phase) + t1.imag * np.cos(phase)
        cf[j - 1] = (real_part + 1j * imag_part) * mag
    return cf.astype(np.complex128)

ALLOWED["p603"]=p603

def p604(z,a,state):
    t1, t2 = z[0], z[1]
    n = 25
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(t1.real, t2.real, num=n)
    imc = np.linspace(t1.imag, t2.imag, num=n)
    for j in range(1, n + 1):
        magnitude = np.log(np.abs(rec[j - 1]) + 1) * (np.abs(t1)**(j % 5 + 1)) + np.abs(t2)**((n - j) % 7 + 1)
        angle = np.sin(rec[j - 1] * np.pi * j) + np.cos(imc[j - 1] * np.pi / (j + 1)) + np.angle(t1) * np.log(j + 2) - np.angle(t2) * np.sqrt(j)
        cf[j - 1] = magnitude * np.exp(1j * angle)
    # Introduce variation using product and sum
    cf = cf * (np.prod(rec) / (np.sum(imc) + 1)) + np.sum(rec) * np.conj(t1) - np.sum(imc) * np.conj(t2)
    return cf.astype(np.complex128)
    
ALLOWED["p604"]=p604

def p605(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 25
    cf = np.zeros(degree + 1, dtype=np.complex128)
    for j in range(1, degree + 2):
        mag_part1 = np.log(np.abs(t1) + j) * np.sin(2 * np.pi * t1.real / (j + 1))
        mag_part2 = np.log(np.abs(t2) + j) * np.cos(2 * np.pi * t2.imag / (j + 1))
        magnitude = mag_part1 + mag_part2 + (t1.real * t1.imag * j)
        angle = np.angle(t1) * j + np.angle(t2) * (degree + 1 - j) + np.sin(j) - np.cos(j)
        cf[j - 1] = magnitude * np.exp(1j * angle)
    return cf.astype(np.complex128)
    
ALLOWED["p605"]=p605

def p606(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 25
    cf = np.zeros(degree + 1, dtype=np.complex128)
    r1 = t1.real
    i1 = t1.imag
    r2 = t2.real
    i2 = t2.imag
    for j in range(1, degree + 2):
        r_part = r1 * np.sin(j * np.pi / 7) + r2 * np.log(np.abs(j) + 1)
        i_part = i1 * np.cos(j * np.pi / 5) - i2 * np.sin(j * np.pi / 3)
        magnitude = np.log(np.abs(t1) + j**2) * (1 + np.sin(j * np.pi / 4)) + np.cos(j * np.pi / 6)
        angle = np.angle(t1) * j + np.angle(t2) * (degree + 1 - j) + np.sin(j * np.pi / 8)
        cf[j - 1] = magnitude * (np.cos(angle) + np.sin(angle) * 1j)
    return cf.astype(np.complex128)
    
ALLOWED["p606"]=p606

def p607(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 25
    cf = np.zeros(degree + 1, dtype=np.complex128)
    for j in range(1, degree + 2):
        real_part = t1.real**j * np.log(np.abs(t2) + j) + np.cos(j * np.angle(t1 + t2))
        imag_part = np.sin(j * np.angle(t1)) * np.abs(t2)**j + (t1.real + t2.real) / (j + 1)
        cf[j - 1] = real_part + 1j * imag_part
    return cf.astype(np.complex128)
    
ALLOWED["p607"]=p607

def p608(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 25
    cf = np.zeros(degree + 1, dtype=np.complex128)
    rec = np.linspace(t1.real, t2.real, num=degree + 1)
    imc = np.linspace(t1.imag, t2.imag, num=degree + 1)
    for j in range(1, degree + 2):
        mag = np.log(np.abs(rec[j - 1]) + j) * np.sin(2 * np.pi * imc[j - 1]) + np.cos(3 * np.pi * rec[j - 1])
        ang = np.angle(t1) * j + np.sin(np.pi * imc[j - 1]) - np.cos(np.pi * rec[j - 1])
        cf[j - 1] = mag * (np.cos(ang) + np.sin(ang) * 1j)
    return cf.astype(np.complex128)

ALLOWED["p608"]=p608

def p609(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 25
    cf = np.zeros(degree + 1, dtype=np.complex128)
    for j in range(1, degree + 2):
        rec_part = t1.real * np.sin(j * t2.real) + np.cos(j * np.angle(t1))
        imc_part = t2.imag * np.cos(j * t1.imag) + np.sin(j * np.angle(t2))
        mag = np.log(np.abs(t1) + j) * rec_part + np.abs(t2)**0.5 * imc_part
        angle = np.angle(t1) * np.cos(j / 3) + np.angle(t2) * np.sin(j / 4)
        cf[j - 1] = mag * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p609"]=p609

def p610(z,a,state):
    t1, t2 = z[0], z[1]
    n = 26
    cf = np.zeros(n, dtype=np.complex128)
    for j in range(1, n + 1):
        magnitude = np.log(np.abs(t1) + 1) * j**2 + np.log(np.abs(t2) +1) * (n - j +1)**1.5
        angle = np.angle(t1) * np.sin(j / n * np.pi) + np.angle(t2) * np.cos(j / n * np.pi) + np.sin(j) * 0.5
        cf[j - 1] = magnitude * (np.cos(angle) + np.sin(angle) * 1j) + np.conj(t1) * t2**j
    for k in range(1, n + 1):
        cf[k - 1] = cf[k - 1] * np.exp(1j * np.sin(k * t1.real)) + np.exp(1j * np.cos(k * t2.imag))
    return cf.astype(np.complex128)
    
ALLOWED["p610"]=p610

def p611(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(26, dtype=np.complex128)
    for j in range(1, 27):
        mag_part = np.log(np.abs(t1) + j) * np.sin(j * t1.real) + np.cos(j * t2.imag)
        angle_part = np.angle(t1) * j**2 - np.angle(t2) * np.log(j + 1) + np.sin(j) * np.cos(j)
        intricate_sum = 0
        for k in range(1, j + 1):
            intricate_sum += (t1.real**k * np.cos(k)) / (k + 1)
        for r in range(1, int(np.floor(j / 2)) +1):
            intricate_sum += (t2.imag**r * np.sin(r)) / (r + 1)
        cf[j - 1] = (mag_part + intricate_sum * 1j) * np.exp(1j * angle_part)
    return cf.astype(np.complex128)

ALLOWED["p611"]=p611

def p612(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 25
    cf = np.zeros(degree + 1, dtype=np.complex128)
    for j in range(1, degree + 2):
        r_part = t1.real * np.sin(j * np.pi / 6) + t2.real * np.cos(j * np.pi / 7)
        i_part = t1.imag * np.cos(j * np.pi / 5) - t2.imag * np.sin(j * np.pi / 8)
        magnitude = np.log(np.abs(t1) + np.abs(t2) + j) * (1 + np.sin(j * np.pi / 4)) * (1 + np.cos(j * np.pi / 9))
        angle = np.angle(t1) * np.sin(j * np.pi / 10) + np.angle(t2) * np.cos(j * np.pi / 11)
        cf[j - 1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    for j in range(1, degree + 2):
        cf[j - 1] += (t1.real - t2.imag) * np.sin(j * np.pi / 3) * np.cos(j * np.pi /5)
    return cf.astype(np.complex128)

ALLOWED["p612"]=p612

def p613(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    rec = np.linspace(t1.real, t2.real, num=25)
    imc = np.linspace(t1.imag, t2.imag, num=25)
    for j in range(1, 26):
        mag = np.log(np.abs(rec[j -1] + imc[j -1]) +1) * (j**2 + np.sin(j))
        ang = np.sin(rec[j -1] * j) + np.cos(imc[j -1] * j)
        cf[j -1] = mag * np.exp(1j * ang) + np.conj(t1) * t2**j
    return cf.astype(np.complex128)
    
ALLOWED["p613"]=p613

def p614(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree + 1, dtype=np.complex128)
    for j in range(0, degree + 1):
        mag_part1 = t1.real * j**2
        mag_part2 = np.log(np.abs(t1) + np.abs(t2) + 1) * (j + 1)
        mag = mag_part1 + mag_part2 + np.abs(t2.imag)**(j % 3 + 1)
        
        angle_part1 = np.angle(t1) * np.sin(j * np.pi / 4)
        angle_part2 = np.angle(t2) * np.cos(j * np.pi / 3)
        angle = angle_part1 + angle_part2 + np.sin(j)
        
        cf[j] = mag * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p614"]=p614

def p615(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree + 1, dtype=np.complex128)
    for j in range(1, degree + 2):
        mag_part = np.log(np.abs(t1) + j) * np.sin(j * t2.real) + np.log(np.abs(t2) + j) * np.cos(j * t1.imag)
        angle_part = np.angle(t1) * j + np.angle(t2) / (j + 1)
        cf[j -1] = mag_part * np.exp(1j * angle_part)
    for k in range(1, degree + 2):
        cf[k -1] += (t1.real - t2.imag) * np.sin(k * np.angle(t1)) + (t2.real + t1.imag) * np.cos(k * np.angle(t2))
    return cf.astype(np.complex128)
    
ALLOWED["p615"]=p615

def p616(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree + 1, dtype=np.complex128)
    for k in range(0, degree + 1):
        j = k + 1
        r_part = t1.real * np.sin(k * t2.real) + t2.real * np.cos(k * t1.real)
        im_part = t1.imag * np.cos(k * t2.imag) - t2.imag * np.sin(k * t1.imag)
        magnitude = np.log(np.abs(t1 + t2) + 1) * (k + 1) / (1 + k)
        angle = np.angle(t1) * np.sin(k) + np.angle(t2) * np.cos(k)
        cf[k] = magnitude * (np.cos(angle) + np.sin(angle) * 1j)
    return cf.astype(np.complex128)
    
ALLOWED["p616"]=p616

def p617(z, a, state):
    t1 = z[0]
    t2 = z[1]
    cf = np.zeros(9, dtype=np.complex128)

    re1 = t1.real
    im1 = t1.imag
    re2 = t2.real
    im2 = t2.imag

    # Numba-safe angle
    ang_t1 = np.arctan2(im1, re1)
    ang_t2 = np.arctan2(im2, re2)

    for j in range(1, 10):
        real_sum = 0.0
        imag_sum = 0.0

        for k in range(1, j + 1):
            # re1**k via integer multiply (avoids float pow corner cases)
            pk = 1.0
            for _ in range(k):
                pk *= re1

            # sin(k * angle(t1)) computed once per k
            sin_k_ang_t1 = np.sin(k * ang_t1)

            for r in range(1, k + 1):
                real_sum += pk * np.cos(r * ang_t2)

                # (im2)**r via integer multiply (handles negative bases cleanly)
                pr = 1.0
                for _ in range(r):
                    pr *= im2
                imag_sum += pr * sin_k_ang_t1

        # Logs: log1p(x) valid only for x > -1
        x = real_sum
        y = imag_sum
        if (
            np.isnan(x) or np.isinf(x) or x <= -0.999999999999
            or np.isnan(y) or np.isinf(y) or y <= -0.999999999999
        ):
            cf[j - 1] = 0.0j
        else:
            rx = np.log1p(x)
            ry = np.log1p(y)
            if np.isnan(rx) or np.isinf(rx) or np.isnan(ry) or np.isinf(ry):
                cf[j - 1] = 0.0j
            else:
                cf[j - 1] = complex(rx, ry)

    return cf
    
ALLOWED["p617"]=p617

def p618(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree + 1, dtype=np.complex128)
    for j in range(0, degree + 1):
        r = j + 1
        mag = np.log(np.abs(t1) + r**2) * np.sin(r * t2.imag) + np.cos(r * t1.real)
        ang = np.angle(t1) * r + t2.real / (j + 1)
        cf[j] = mag * (np.cos(ang) + np.sin(ang) * 1j)
    return cf.astype(np.complex128)
    
ALLOWED["p618"]=p618

def p619(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    r1 = t1.real
    i1 = t1.imag
    r2 = t2.real
    i2 = t2.imag
    for j in range(1, 10):
        mag = np.log(np.abs(t1 + j)**2 + np.abs(t2 - j)**2) * (1 + np.sin(j * np.angle(t1)) * np.cos(j * np.angle(t2)))
        angle = np.angle(t1) * j**0.5 - np.angle(t2) * (9 - j)**0.5 + np.sin(j * np.pi / 5)
        cf[j -1] = mag * (np.cos(angle) + np.sin(angle) * 1j)
    return cf.astype(np.complex128)

ALLOWED["p619"]=p619

def p620(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    r1 = t1.real
    i1 = t1.imag
    r2 = t2.real
    i2 = t2.imag
    for j in range(1, 10):
        mag_real = np.log(np.abs(t1) + 1) * np.sin(j * np.angle(t1))
        mag_imag = np.log(np.abs(t2) + 1) * np.cos(j * np.angle(t2))
        magnitude = mag_real + mag_imag
        angle = np.angle(t1)**j - np.angle(t2)**j
        cf[j -1] = magnitude * (np.cos(angle) + np.sin(angle) * 1j)
    return cf.astype(np.complex128)
    
ALLOWED["p620"]=p620

def p621(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1, 10):
        mag = (t1.real**j + t2.imag**(j/2)) * np.log(np.abs(t1) + j) + np.sin(j * t2.real) * np.cos(j * t1.imag)
        angle = np.angle(t1) * np.sin(j) - np.angle(t2) * np.cos(j) + np.sin(j * t1.real) * np.cos(j * t2.imag)
        cf[j -1] = mag * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p621"]=p621

def p622(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1, 10):
        mag_part = np.log(np.abs(t1)**j + 1) * (t2.real + j)**2
        angle_part = np.angle(t1) * np.sin(j * np.angle(t2)) + np.cos(j * t2.real)
        cf[j -1] = mag_part * (np.cos(angle_part) + np.sin(angle_part) * 1j)
    return cf.astype(np.complex128)
    
ALLOWED["p622"]=p622

def p623(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for k in range(1, 10):
        j = k
        magnitude = np.log(np.abs(t1) + j) * (np.sin(j * t1.real) + np.cos(j * t1.imag))
        angle = np.angle(t1) * np.sin(j * t1.real) - np.angle(t2) * np.cos(j * t1.imag)
        cf[j -1] = magnitude * np.exp(1j * angle) + np.conj(t1) * (t2**j)
    return cf.astype(np.complex128)

ALLOWED["p623"]=p623

def p624(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1, 10):
        mag_part = np.log(np.abs(t1) + j) * np.sin(j * np.angle(t1)) + np.cos(j * t2.imag)
        angle_part = np.angle(t1)**j + np.angle(t2)**(j/2) + np.log(j + 1)
        mag = np.abs(mag_part) + (t1.real * t2.imag * j)
        angle = angle_part + (np.abs(t1) + np.abs(t2) + j)
        cf[j -1] = mag * (np.cos(angle) + np.sin(angle) * 1j)
    return cf.astype(np.complex128)

ALLOWED["p624"]=p624

def p625(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1, 10):
        mag = np.log(np.abs(t1) + j) * (1 + np.sin(j * np.angle(t1)) + np.cos(j * np.angle(t2)))
        angle = np.angle(t1)**j - np.angle(t2)**(j/2) + np.sin(j * t1.real) * np.cos(j * t2.imag)
        cf[j -1] = mag * (np.cos(angle) + np.sin(angle) * 1j)
    return cf.astype(np.complex128)

ALLOWED["p625"]=p625

def p626(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1, 10):
        mag_part1 = np.sin(j * np.pi * t1.real / (1 + j)) + np.cos(j * np.pi * t2.real / (1 + j))
        mag_part2 = np.log(np.abs(t1) + 1) * np.log(np.abs(t2) + 1)
        magnitude = mag_part1 * mag_part2 + j**2
        angle_part1 = np.angle(t1) + np.angle(t2) * j
        angle_part2 = np.cos(j * t2.real) - np.sin(j * t1.imag)
        angle = angle_part1 + angle_part2
        cf[j -1] = magnitude * np.exp(1j * angle)
    return cf.astype(np.complex128)

ALLOWED["p626"]=p626

def p627(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for k in range(1, 10):
        mag_part1 = np.log(np.abs(t1)**k + 1) * (t2.real + k)**2
        angle_part = np.angle(t1) * np.sin(k * np.angle(t2)) + np.cos(k * t2.real)
        mag_variation = mag_part1 * (np.abs(np.sin(k * t1.real)) + np.abs(np.cos(k * t2.real)))
        angle = angle_part
        cf[k -1] = mag_variation * np.exp(1j * angle)
    return cf.astype(np.complex128)

ALLOWED["p627"]=p627

def p628(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1, 10):
        mag = 0
        angle = 0
        for k in range(1, 5):
            mag += (t1.real**k * np.sin(k * j)) + (t2.imag**k * np.cos(k + j))
            angle += (np.angle(t1) + np.angle(t2)) / (k + j)
        cf[j -1] = mag * np.exp(1j * angle) + np.conj(t1) * np.log(np.abs(t2) + 1)
    return cf.astype(np.complex128)
    
ALLOWED["p628"]=p628

def p629(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1, 10):
        r = t1.real * j
        im = t2.imag / j
        mag = np.log(np.abs(t1) + j) * (np.sin(r) + np.cos(im))
        angle = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j)
        cf[j -1] = mag * (np.cos(angle) + np.sin(angle) * 1j)
    return cf.astype(np.complex128)
    
ALLOWED["p629"]=p629

def p630(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1, 10):
        r_part = t1.real * np.log(np.abs(t2) + j) + np.cos(j * np.angle(t1 + t2))
        im_part = np.sin(j * np.angle(t1)) * np.abs(t2)**j + (t1.real + t2.real) / (j + 1)
        magnitude = np.sqrt(r_part**2 + im_part**2) * j**1.5
        angle = np.arctan2(im_part, r_part) + np.sin(j * np.pi / 3)
        cf[j -1] = magnitude * (np.cos(angle) + np.sin(angle) * 1j)
    return cf.astype(np.complex128)
    
ALLOWED["p630"]=p630

def p631(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree + 1, dtype=np.complex128)
    for j in range(0, degree + 1):
        real_part = t1.real**j * np.log(np.abs(t2) + j) + np.cos(j * np.angle(t1 + t2))
        imag_part = np.sin(j * np.angle(t1)) * np.abs(t2)**j + (t1.real + t2.real) / (j + 1)
        cf[j] = real_part + 1j * imag_part
    return cf.astype(np.complex128)

ALLOWED["p631"]=p631

def p632(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1, 10):
        rec = t1.real + (t2.real - t1.real) * j / 8
        imc = t1.imag + (t2.imag - t1.imag) * j / 8
        mag = np.log(np.abs(rec * imc) + 1) * (j**2 + np.sin(j))
        angle = np.angle(t1) * j - np.angle(t2) * (9 - j) + np.cos(j * np.pi / 4)
        cf[j -1] = mag * (np.cos(angle) + np.sin(angle) * 1j)
    return cf.astype(np.complex128)
  
ALLOWED["p632"]=p632

def p633(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree + 1, dtype=np.complex128)
    for j in range(0, degree + 1):
        r = j + 1
        mag = np.log(np.abs(t1) + r**2) * np.sin(r * t2.imag) + np.cos(r * t1.real)
        ang = np.angle(t1) * r + t2.real / (j + 1)
        cf[j] = mag * (np.cos(ang) + np.sin(ang) * 1j)
    return cf.astype(np.complex128)
    
ALLOWED["p633"]=p633

def p634(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    r1 = t1.real
    im1 = t1.imag
    r2 = t2.real
    im2 = t2.imag
    for j in range(1, 10):
        mag_part1 = np.sin(j * np.pi * r1 / (1 + j)) + np.cos(j * np.pi * r2 / (1 + j))
        mag_part2 = np.log(np.abs(t1) + 1) * np.log(np.abs(t2) + 1)
        magnitude = mag_part1 * mag_part2 + j**2
        angle_part1 = np.angle(t1) + np.angle(t2) * j
        angle_part2 = np.cos(j * t2.real) - np.sin(j * im1)
        angle = angle_part1 + angle_part2
        cf[j -1] = magnitude * np.exp(1j * angle)
    return cf.astype(np.complex128)
    
ALLOWED["p634"]=p634

def p635(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(0, 9):
        mag_part1 = np.log(np.abs(t1.real + j) + 1)
        mag_part2 = np.abs(t2)**(j / 3) + np.sin(j * np.pi / 5)
        angle_part1 = np.sin(j * t1.real) + np.cos(j * t2.imag)
        angle_part2 = np.angle(t1) * np.cos(j) - np.angle(t2) * np.sin(j)
        magnitude = mag_part1 * mag_part2 + (t1.real + t2.imag + j)
        angle = angle_part1 + angle_part2
        cf[j] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
  
ALLOWED["p635"]=p635

def p636(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    r1 = t1.real
    i1 = t1.imag
    r2 = t2.real
    i2 = t2.imag
    for j in range(1, 10):
        mag_part = np.log(np.abs(t1) + j) * np.sin(j * np.angle(t2) / 2) + np.cos(j * np.angle(t1) / 3)
        angle_part = np.angle(t1) * np.cos(j) + np.sin(np.angle(t2))
        cf[j-1] = mag_part * (np.cos(angle_part) + 1j * np.sin(angle_part))
    return cf.astype(np.complex128)

ALLOWED["p636"]=p636

def p637(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1, 10):
        mag = np.log(np.abs(t1 * t2) + 1) * (t1.real**j + t2.imag**(j/2))
        angle = np.angle(t1) * np.sin(j) - np.angle(t2) * np.cos(j)
        cf[j-1] = mag * (np.cos(angle) + 1j * np.sin(angle)) + np.conj(t1) * np.sin(j) - np.conj(t2) * np.cos(j)
    return cf.astype(np.complex128)
    
ALLOWED["p637"]=p637

def p638(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1, 10):
        rec_part = t1.real**j - t2.real**(9-j)
        im_part = t1.imag * t2.imag + np.cos(j * np.angle(t1) + np.sin(j * np.angle(t2)))
        mag = np.log(np.abs(t1) + np.abs(t2) + j) * (1 + np.sin(j * t1.real) * np.cos(j * t2.imag))
        angle = np.angle(t1) * j + np.angle(t2) / (j + 1)
        cf[j-1] = mag * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
   
ALLOWED["p638"]=p638

def p639(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1, 10):
        magnitude = np.log(np.abs(t1) + j) * np.sin(j * np.angle(t1)) + np.log(np.abs(t2) + j**2) * np.cos(j * np.angle(t2))
        angle = np.angle(t1) * np.cos(j) + np.angle(t2) * np.sin(j)
        cf[j-1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
  
ALLOWED["p639"]=p639

def p640(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1, 10):
        k = j * 2
        r = 9 - j
        mag = np.log(np.abs(t1) + j) * (1 + np.sin(j)) + np.abs(t2)**(0.5 + np.cos(j))
        angle = np.angle(t1) * j + np.angle(t2) * k + np.sin(j) * np.cos(r)
        cf[j-1] = mag * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p640"]=p640

def p641(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1, 10):
        r_part = t1.real**j + t2.real**(9-j)
        im_part = t1.imag * np.sin(j) - t2.imag * np.cos(j)
        magnitude = np.log(np.abs(t1) + np.abs(t2) + j) * (1 + np.sin(j) * np.cos(j))
        angle = np.angle(t1) * j - np.angle(t2) / (j + 1) + np.sin(j * np.pi / 4)
        cf[j-1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
   
ALLOWED["p641"]=p641

def p642(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree + 1, dtype=np.complex128)
    for j in range(0, degree + 1):
        k = (j % 3) + 1
        r = (j // 2) + 1
        mag_part1 = np.log(np.abs(t1) + 1) * np.sin(j * np.pi / 4)
        mag_part2 = np.cos(k * np.pi / 3) * np.abs(t2)**r
        magnitude = mag_part1 + mag_part2
        angle_part1 = np.angle(t1) * j
        angle_part2 = np.sin(r * np.pi / 5) + np.angle(t2) * k
        angle = angle_part1 + angle_part2
        cf[j] = magnitude * np.exp(1j * angle)
    return cf.astype(np.complex128)

ALLOWED["p642"]=p642
    
def p643(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for k in range(1, 10):
        mag = t1.real**k + t2.imag**(9 - k) + np.log(np.abs(t1) + np.abs(t2) + 1) * np.sin(k * np.angle(t1) * np.angle(t2))
        angle = np.angle(t1) * np.cos(k * t2.real) - np.angle(t2) * np.sin(k * t1.imag)
        cf[k-1] = mag * np.exp(1j * angle)
    return cf.astype(np.complex128)

ALLOWED["p643"]=p643

def p644(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1, 10):
        part1 = t1.real**j + t2.imag**(j % 4 + 1)
        part2 = np.sin(j * t1.real + np.cos(j * t2.imag))
        part3 = np.log(np.abs(t1) + np.abs(t2) + j)
        magnitude = part1 * part2 + part3
        angle = np.angle(t1)**j + np.angle(t2) * np.sin(j) + np.angle(np.conj(t1)) - np.angle(np.conj(t2))
        cf[j-1] = magnitude * np.exp(1j * angle)
    return cf.astype(np.complex128)

ALLOWED["p644"]=p644

def p645(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree + 1, dtype=np.complex128)
    real_seq = np.linspace(t1.real, t2.real, degree + 1)
    im_seq = np.linspace(t1.imag, t2.imag, degree + 1)
    for j in range(1, degree + 2):
        mag_component = np.log(np.abs(t1) + j) * np.sin(j * np.pi / 4) + np.cos(j * np.pi / 3) * np.abs(t2)
        angle_component = np.angle(t1) * j + np.angle(t2) * (degree + 1 - j)
        intricate_part = np.exp(1j * (np.sin(real_seq[j-1]) + np.cos(im_seq[j-1])))
        cf[j-1] = mag_component * intricate_part * np.conj(t2) + np.prod(np.arange(1, j+1)) * np.sin(j)
    return cf.astype(np.complex128)

ALLOWED["p645"]=p645
    
def p646(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    rec = np.linspace(t1.real, t2.real, 9)
    imc = np.linspace(t1.imag, t2.imag, 9)
    for j in range(1, 10):
        magnitude = np.log(np.abs(t1)**j + np.abs(t2)**(9-j) + 1) * (j**np.sin(j) + j**np.cos(j))
        angle = np.angle(t1) * np.sin(j * np.pi / 4) + np.angle(t2) * np.cos(j * np.pi / 3)
        cf[j-1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    for k in range(1, 10):
        cf[k-1] = cf[k-1] * (1 + 0.1 * k) / (1 + 0.05 * k**2)
    return cf.astype(np.complex128)

ALLOWED["p646"]=p646

def p647(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for k in range(0, 9):
        j = k + 1
        r = t1.real + t2.real * k
        im = t1.imag - t2.imag * k
        angle = np.sin(r) * np.cos(im) + np.angle(t1 * t2) / (k + 1) - np.log(np.abs(r * im) + 1)
        mag = (np.abs(t1) * np.abs(t2))**k + (r + im + k) + (r * im * (k + 1))
        cf[k] = mag * np.exp(1j * angle)
    return cf.astype(np.complex128)

ALLOWED["p647"]=p647

def p648(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree + 1, dtype=np.complex128)
    for j in range(0, degree + 1):
        sum_real = 0
        sum_imag = 0
        for k in range(1, j + 2):
            term_real = np.log(np.abs(t1) * k + 1) * np.sin(k * np.angle(t2))
            term_imag = np.log(np.abs(t2) * (degree - j + k) + 1) * np.cos(k * np.angle(t1))
            sum_real += term_real
            sum_imag += term_imag
        magnitude = sum_real**2 + sum_imag**2
        angle = sum_real / (sum_imag + 1e-8)
        cf[j] = np.sqrt(magnitude) * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p648"]=p648

def p649(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree + 1, dtype=np.complex128)
    for j in range(0, degree + 1):
        k = j + 1
        mag_part1 = np.log(np.abs(t1) + 1) * (j + 1)**1.5
        mag_part2 = np.log(np.abs(t2) + 1) * (degree - j + 1)**1.2
        magnitude = mag_part1 + mag_part2
        angle_part1 = np.sin(j * np.angle(t1)) 
        angle_part2 = np.cos(j * np.angle(t2))
        angle = angle_part1 + angle_part2
        cf[k-1] = magnitude * np.exp(1j * angle) + np.conj(t1) * np.conj(t2) * j
    return cf.astype(np.complex128)
 
ALLOWED["p649"]=p649

def p650(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree + 1, dtype=np.complex128)
    for j in range(1, degree + 2):
        mag_factor = np.log(np.abs(t1) + j) * np.sin(j * t2.real) + np.cos(j * t1.imag)
        angle_factor = np.angle(t1) * np.sqrt(j) - np.angle(t2) / (j + 1) + np.sin(j)
        cf[j-1] = mag_factor * np.exp(1j * angle_factor)
    return cf.astype(np.complex128)

ALLOWED["p650"]=p650
    
def p651(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    rec = np.linspace(t1.real, t2.real, 9)
    imc = np.linspace(t1.imag, t2.imag, 9)
    for j in range(1, 10):
        mag = np.log(np.abs(rec[j-1]) + 1) * np.cos(j) + np.abs(t2)**j
        ang = np.angle(t1) * np.sin(j * np.pi * imc[j-1]) + np.angle(t2) * np.cos(j * np.pi * rec[j-1])
        cf[j-1] = mag * (np.cos(ang) + 1j * np.sin(ang))
    return cf.astype(np.complex128)

ALLOWED["p651"]=p651

def p652(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1, 10):
        angle_part = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j)
        magnitude_part = np.abs(t1)**j + np.log(np.abs(t2) + 1) * j
        phase_variation = np.sin(j * t1.real) + np.cos(j * t2.imag)
        real_component = t1.real * magnitude_part * np.cos(angle_part) + phase_variation
        imag_component = t2.imag * magnitude_part * np.sin(angle_part) + phase_variation
        cf[j-1] = complex(real_component, imag_component)
    return cf.astype(np.complex128)

ALLOWED["p652"]=p652

def p653(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree + 1, dtype=np.complex128)
    rec_seq = np.linspace(t1.real, t2.real, degree + 1)
    imc_seq = np.linspace(t1.imag, t2.imag, degree + 1)
    for k in range(1, degree + 2):
        mag = 0
        ang = 0
        for j in range(1, k + 1):
            mag += np.sin(rec_seq[j-1] * j) * np.cos(imc_seq[j-1] * j)
            ang += np.angle(rec_seq[j-1] + imc_seq[j-1] * 1j) * j
        mag = mag * np.log(np.abs(rec_seq[k-1] + imc_seq[k-1] * 1j) + 1)
        ang = ang / k
        cf[k-1] = mag * (np.cos(ang) + 1j * np.sin(ang))
    return cf.astype(np.complex128)

ALLOWED["p653"]=p653

def p654(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(0, 9):
        k = (j % 3) + 1
        r = np.log(np.abs(t1) + np.abs(t2)*j) * (j**1.5)
        angle = np.angle(t1)**k - np.angle(t2)**j + np.sin(j * np.pi / 5)
        cf[j] = r * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
  
ALLOWED["p654"]=p654

def p655(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1, 10):
        if j % 2 == 0:
            mag = np.log(np.abs(t1) + j**2) * np.sqrt(j)
        else:
            mag = np.abs(t2)**j / (1 + j)
        if j % 3 == 0:
            angle = np.angle(t1) * np.sin(j) + np.cos(j * np.angle(t2))
        else:
            angle = np.sin(j * np.angle(t1)) - np.cos(j * np.angle(t2))
        cf[j-1] = mag * np.exp(1j * angle) * (t1.real + t2.imag / j)
    return cf.astype(np.complex128)
 
ALLOWED["p655"]=p655

def p656(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1, 10):
        mag_part1 = np.log(np.abs(t1) + 1) * np.sin(j * np.angle(t1))
        mag_part2 = np.log(np.abs(t2) + 1) * np.cos(j * np.angle(t2))
        magnitude = mag_part1 + mag_part2 + (j**2) / (np.abs(t1) + np.abs(t2) + 1)
        
        angle_part1 = np.angle(t1) * np.cos(j) 
        angle_part2 = np.angle(t2) * np.sin(j)
        angle_part3 = np.sin(j * t1.real) - np.cos(j * t2.imag)
        angle = angle_part1 + angle_part2 + angle_part3
        
        cf[j-1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p656"]=p656

def p657(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree + 1, dtype=np.complex128)
    for j in range(0, degree + 1):
        r = j / degree
        mag = np.log(np.abs(t1) + np.abs(t2) + r) * (1 + np.sin(j * np.pi / 4))
        angle = np.angle(t1) * r**2 + np.angle(t2) * (1 - r)**2 + np.cos(j * np.pi / 3)
        cf[j] = mag * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p657"]=p657

def p658(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree + 1, dtype=np.complex128)
    for j in range(0, degree + 1):
        mag = np.log(np.abs(t1) + np.abs(t2) + j + 1) * (j + 1)**1.5
        ang = np.sin(j * np.angle(t1)) - np.cos(np.angle(t2))
        cf[j] = mag * np.exp(1j * ang)
    return cf.astype(np.complex128)

ALLOWED["p658"]=p658

def p659(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for r in range(0, 9):
        mag = t1.real**2 * r + np.log(np.abs(t2) + 1) + np.sin(r * t1.real)
        angle = np.angle(t1) * r - np.cos(r * t2.imag)
        cf[r] = mag * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
 
ALLOWED["p659"]=p659

def p660(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree + 1, dtype=np.complex128)
    for j in range(0, degree + 1):
        mag_component = np.log(np.abs(t1) + np.abs(t2) + j + 1) * np.sin(j * np.pi / 5)
        angle_component = np.angle(t1) * np.cos(j) - np.angle(t2) * np.sin(j / 2)
        real_part = t1.real**(j % 3 + 1) + t2.real**(degree - j % 2)
        imag_part = t2.imag * np.cos(j * np.pi / 4)
        cf[j] = (mag_component + real_part) + 1j * (angle_component + imag_part)
    return cf.astype(np.complex128)

ALLOWED["p660"]=p660

def p661(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1, 10):
        r_part = t1.real**j + t2.real**(9-j)
        i_part = t1.imag * np.sin(j) - t2.imag * np.cos(j)
        magnitude = np.log(r_part + 1) * np.abs(t1 + t2) + np.sin(r_part) * np.cos(i_part)
        angle = np.angle(t1)*j**2 - np.angle(t2)/j + np.sin(j * np.angle(t2))
        cf[j-1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p661"]=p661

def p662(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree + 1, dtype=np.complex128)
    for k in range(0, degree + 1):
        j = np.sin(k * t1.real + np.cos(k * t2.imag)) + np.log(np.abs(t1) + np.abs(t2) + 1)
        r = np.cos(k * np.angle(t1)) * np.sin(k * np.angle(t2)) + t1.real * t2.imag
        magnitude = np.sqrt(j**2 + r**2) * (k + 1)
        angle = np.arctan2(r, j) + np.sin(k * t1.real * t2.imag)
        cf[k] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p662"]=p662

def p663(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1, 10):
        r = j
        term1 = (t1.real**r + t2.imag**(r % 5 + 1)) * np.log(np.abs(t1) + 1)
        term2 = (np.abs(t2) * np.cos(r * np.angle(t1))) + (np.sin(r) * t2.real)
        angle = np.angle(t1) * np.sin(r) - np.angle(t2) * np.cos(r)
        cf[j-1] = (term1 + term2) * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p663"]=p663

def p664(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(0,9):
        mag = np.log(np.abs(t1 + j * t2) + 1) * np.sin(j * np.pi / 4) + np.cos(j * np.angle(t1 * t2))
        angle = np.angle(t1)**j - np.angle(t2)**(8 - j) + np.sin(j * np.pi / 3)
        cf[j] = mag * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p664"]=p664

def p665(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(8, dtype=np.complex128)
    for j in range(1, 9):
        mag = np.log(np.abs(t1) + j) * np.sin(j * np.angle(t1))
        angle = np.cos(j * np.angle(t2)) + np.sin(j * t2.real)
        cf[j-1] = mag * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p665"]=p665
    
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

def p667(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree + 1, dtype=np.complex128)
    for j in range(1, degree + 1):
        mag_sum = 0
        angle_sum = 0
        for k in range(1, j + 1):
            mag_sum += np.log(np.abs(t1) + k) * np.sin(k * np.angle(t2) + j)
            angle_sum += np.cos(k * np.pi / (j + 1))
        magnitude = mag_sum * (1 + j)
        angle = angle_sum + np.angle(np.conj(t1) * np.conj(t2)) * j**2
        cf[j] = magnitude * np.exp(1j * angle)
    cf[degree] = np.conj(t1) * np.conj(t2) + np.sum(np.abs(cf[0:degree]))
    return cf.astype(np.complex128)
   
ALLOWED["p667"]=p667

def p668(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1, 10):
        k = j
        r_part = t1.real * np.log(k + 1) + t2.real * np.sin(k * np.pi / 7)
        i_part = t1.imag * np.cos(k * np.pi / 5) + t2.imag * np.log(k + 2)
        magnitude = np.sqrt(r_part**2 + i_part**2) * (1 + 0.1 * j)
        angle = np.arctan2(i_part, r_part) + np.cos(j * t2.real) * np.sin(j * t1.imag)
        cf[j-1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p668"]=p668

def p669(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1, 10):
        term1 = t1.real**j * np.sin(j * np.angle(t2)) + t2.imag**(j % 5) * np.cos(j * t1.real - np.sin(np.abs(t2)))
        term2 = (t2.imag ** (j % 5)) * math.cos(j * t1.real - math.sin(j * np.abs(t2)))
        term3 = np.log(np.abs(t1) + np.abs(t2) + 1) * (t1.real * t2.imag)**(j % 3 + 1)
        magnitude = term1 + term2 + term3
        angle = np.angle(t1) * j - np.angle(t2) * (10 - j) + np.sin(j * t2.real)
        cf[j-1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p669"]=p669

def p670(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1, 10):
        magnitude = np.log(np.abs(t1) + np.abs(t2) + j) * (j**2 + np.sin(j))
        angle = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j**2)
        cf[j-1] = magnitude * (np.cos(angle) + 1j * np.sin(angle)) + np.conj(t1) * np.cos(j) - np.conj(t2) * np.sin(j)
    return cf.astype(np.complex128)

ALLOWED["p670"]=p670

def p671(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree + 1, dtype=np.complex128)
    for j in range(0, degree + 1):
        real_part = t1.real * j + t2.real * (degree - j)
        imag_part = t1.imag * np.sin(j) - t2.imag * np.cos(j)
        magnitude = np.log(np.abs(t1) + np.abs(t2) + j + 1) * (j + 1)
        angle = np.angle(t1) * np.cos(j) + np.angle(t2) * np.sin(j)
        cf[j] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    for k in range(1, degree + 2):
        cf[k-1] = cf[k-1] * np.exp(1j * np.sin(k * np.pi / 4)) + np.conj(cf[degree + 1 - k])
    return cf.astype(np.complex128)
    
ALLOWED["p671"]=p671

def p672(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree + 1, dtype=np.complex128)
    for j in range(1, degree + 2):
        magnitude = np.log(np.abs(t1) + j) * (np.sin(j * np.angle(t1)) + np.cos(j * np.angle(t2)))
        angle = np.angle(t1) * np.sqrt(j) + np.angle(t2) / (j + 1)
        cf[j-1] = magnitude * np.exp(1j * angle) + np.conj(t1) * np.sin(j) - np.conj(t2) * np.cos(j)
    return cf.astype(np.complex128)

ALLOWED["p672"]=p672 

def p673(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree + 1, dtype=np.complex128)
    for j in range(0, degree + 1):
        mag = np.log(np.abs(t1) + j * np.abs(t2) + 1) * (j + 1)**1.5
        angle = np.angle(t1)**j - np.sin(j * np.angle(t2)) + np.cos(j * t1.real * t2.imag)
        cf[j] = mag * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p673"]=p673

def p674(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    degree = 8
    for j in range(0, 9):
        mag_part1 = np.log(np.abs(t1) + 1) * np.sin(j * np.angle(t2))
        mag_part2 = np.cos(j * t1.real) * np.abs(t2)**0.5
        mag_part3 = t1.real * np.cos(t2.real) if j % 2 == 0 else t1.imag + t2.real
        magnitude = mag_part1 + mag_part2 + mag_part3
        angle_part1 = np.angle(t1)**j
        angle_part2 = t2.real * j
        angle_part3 = np.sin(j * np.angle(t1)) if j % 3 == 0 else np.cos(j * np.angle(t2))
        angle = angle_part1 + angle_part2 + angle_part3
        cf[j] = magnitude * np.exp(1j * angle)
    return cf.astype(np.complex128)

ALLOWED["p674"]=p674

def p675(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree + 1, dtype=np.complex128)
    for j in range(1, degree + 2):
        mag_sum = 0
        angle_sum = 0
        for k in range(1, j + 1):
            mag_sum += np.log(np.abs(t1) + k) * np.sin(k * np.angle(t2) + j)
            angle_sum += np.cos(k * np.angle(t1 - t2))
        for r in range(1, (j % 3) + 2):
            mag_sum += (t1.real * t2.imag * r)
            angle_sum += np.angle(np.conj(t1) * np.conj(t2)) * r
        magnitude = mag_sum * (1 + j)
        angle = angle_sum + np.angle(t1) * j - np.angle(t2) * j**2
        cf[j-1] = magnitude * np.exp(1j * angle)
    return cf.astype(np.complex128)
  
ALLOWED["p675"]=p675

def p676(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    r = t1.real
    m = t2.imag
    for j in range(1, 10):
        mag = (r**j + m**(j % 3 + 1)) * np.log(np.abs(t1) + np.abs(t2) + 1)
        angle = np.angle(t1)**j - np.angle(t2) + np.sin(j * r) * np.cos(j * m)
        cf[j-1] = mag * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p676"]=p676

def p677(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree, dtype=np.complex128)
    for j in range(1, degree + 1):
        r = t1.real * j / degree + t2.imag * (degree - j + 1) / degree
        theta = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j)
        mag = np.log(np.abs(t1) + 1) * np.sin(j * np.pi / 4) + np.cos(j * np.pi / (degree + 1))
        cf[j-1] = mag * np.exp(1j * theta)
    return cf.astype(np.complex128)

ALLOWED["p677"]=p677

def p678(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1,10):
        k = j
        angle = np.angle(t1) * np.log(k + 1) + np.angle(t2) * np.sin(k)
        magnitude = np.abs(t1)**k + np.abs(t2)**(9 - k) + np.cos(k * np.pi / 4)
        cf[j-1] = magnitude * np.exp(1j * angle) + np.conj(t1) * np.sin(angle) * np.cos(k)
    return cf.astype(np.complex128)

ALLOWED["p678"]=p678

def p679(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree + 1, dtype=np.complex128)
    for j in range(0, degree +1):
        mag_part = np.log(np.abs(t1) + j + 1) * np.abs(np.sin((j + 1) * np.angle(t1))) + \
                    np.log(np.abs(t2) + degree - j + 1) * np.abs(np.cos((j + 1) * np.angle(t2)))
        angle_part = np.angle(t1) * (j + 1) + np.angle(t2) * (degree - j) + np.sin(j)
        cf[j] = mag_part * (np.cos(angle_part) + 1j * np.sin(angle_part)) + \
                np.conj(t1 * t2) * (j + 1) / (degree + 1)
    for j in range(0, degree +1):
        cf[j] += np.conj(t1) * np.sin(j * t1.real) + np.cos(j * t2.imag)
    return cf.astype(np.complex128)

ALLOWED["p679"]=p679

def p680(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree + 1, dtype=np.complex128)
    for j in range(0, degree + 1):
        k = j + 1
        magnitude = np.log(np.abs(t1) + 1) * np.sin(j * np.angle(t2)) + np.cos(j * np.cos(j + 1))
        angle = np.angle(t1) * j - np.log(np.abs(t2) + 1) * np.cos(j * np.pi / 4)
        real_part = magnitude * np.cos(angle) + t1.real**(j % 3)
        imag_part = magnitude * np.sin(angle) + t2.imag**(j % 2 + 1)
        cf[j] = real_part + 1j * imag_part
    return cf.astype(np.complex128)

ALLOWED["p680"]=p680

def p681(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1,10):
        k = (j * 3 + 2) % 8 +1
        r = np.log(np.abs(t1) + np.abs(t2)) * (j)
        mag = np.abs(t1)**j + np.abs(t2)**k + np.sin(j * np.angle(t1)) * np.cos(k * np.angle(t2))
        ang = np.angle(t1)*j - np.angle(t2)*k + np.sin(j * np.angle(t1)) + np.cos(k * np.angle(t2))
        cf[j-1] = mag * (np.cos(ang) + 1j * np.sin(ang))
    return cf.astype(np.complex128)

ALLOWED["p681"]=p681

def p682(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(8, dtype=np.complex128)
    for j in range(8):  # Python indexing starts at 0
        mag = 0
        ang = 0
        for k in range(1, j + 2):  # 1 to j inclusive in R translates to range(1, j+2) in Python
            mag += np.log(np.abs(t1 * k) + 1) * np.sin(k * np.pi / 4)
            ang += np.angle(t2) * np.cos(k * np.pi / 3)
        cf[j] = mag * np.exp(1j * ang) + np.conj(t1)**(j + 1) * np.imag(t2)
    return cf.astype(np.complex128)

ALLOWED["p682"]=p682

def p683(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree + 1, dtype=np.complex128)
    rec1 = t1.real
    imc1 = t1.imag
    rec2 = t2.real
    imc2 = t2.imag
    for j in range(0, degree +1):
        mag = np.log(np.abs(t1) + j**1.5) * (1 + np.sin(j * np.pi / 3)) + np.cos(j * np.pi /4) * np.abs(t2)
        ang = np.angle(t1) * j + np.sin(j * np.pi /5) - np.cos(j * np.pi /6)
        cf[j] = mag * (np.cos(ang) + 1j * np.sin(ang))
    return cf.astype(np.complex128)

ALLOWED["p683"]=p683

def p684(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1, 10):
        mag = np.log(np.abs(t1) + np.abs(t2) + j**2) * np.sin(j * np.angle(t1)) + np.cos(j * np.angle(t2))
        angle = np.angle(t1) * j - np.angle(t2) / (j +1)
        cf[j-1] = mag * np.cos(angle) + mag * np.sin(angle) * 1j
    return cf.astype(np.complex128)

ALLOWED["p684"]=p684

def p685(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1,10):
        mag = np.log(np.abs(t1) + np.abs(t2) + j) * (1 + np.sin(j * np.pi / 3)) * (1 + np.cos(j * np.pi / 4))
        angle = np.angle(t1) * np.sqrt(j) + np.angle(t2) / (j +1)
        cf[j-1] = mag * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p685"]=p685

def p686(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree + 1, dtype=np.complex128)
    r1 = t1.real
    im1 = t1.imag
    r2 = t2.real
    im2 = t2.imag
    for j in range(1, degree +2):
        mag = np.log(np.abs(t1) + j) + np.sin(j * np.abs(t2)) * np.cos(j) + (t1.real**j) / (1 + j)
        angle = np.sin(j * r1) + np.cos(j * im2) + t2.imag / (j +1)
        cf[j-1] = mag * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p686"]=p686

def p687(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1,10):
        mag = np.log(np.abs(t1)**j + np.abs(t2)**(9-j) + 1)
        ang = np.sin(j * t1.imag) + np.cos((9-j) * t2.real)
        cf[j-1] = mag * (np.cos(ang) + 1j * np.sin(ang))
    return cf.astype(np.complex128)

ALLOWED["p687"]=p687

def p688(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1,10):
        k = (j % 3) + 1
        r = (t1.real * j - t2.imag * k) / (j + k)
        mag = np.log(np.abs(t1) + 1) * np.sin(j * t2.real) + np.cos(k * t1.imag)
        ang = np.angle(t1)**j + np.angle(t2)**k + np.sin(j * k)
        cf[j-1] = mag * (np.cos(ang) + 1j * np.sin(ang))
    return cf.astype(np.complex128)

ALLOWED["p688"]=p688

def p689(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for k in range(1,10):
        mag_part = np.log(np.abs(t1)**k + np.abs(t2)**(9 - k) + 1)
        angle_part = np.sin(k * np.angle(t1)) + np.cos((9 - k) * np.angle(t2))
        cf[k-1] = mag_part * (np.cos(angle_part) + 1j * np.sin(angle_part))
    return cf.astype(np.complex128)

ALLOWED["p689"]=p689

def p690(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree +1, dtype=np.complex128)
    for j in range(0, degree +1):
        mag_part = np.log(np.abs(t1) + np.abs(t2) + j) * np.sin(j * t2.real) + np.cos(j * t1.imag)
        ang_part = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j)
        magnitude = mag_part + (t1.real + t2.imag) / (j + 1)
        angle = ang_part + np.sin(j * np.angle(t1)) * np.cos(j * np.angle(t2))
        cf[j] = magnitude * np.exp(1j * angle)
    return cf.astype(np.complex128)

ALLOWED["p690"]=p690

def p691(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1,10):
        r_part = t1.real * j**np.sin(j) + t2.real / (j + 1)
        im_part = t1.imag * np.cos(j) + t2.imag * np.sin(j / 2)
        mag = np.log(np.abs(t1) + j) * np.abs(np.sin(j)) + np.log(np.abs(t2) + 1)
        angle = np.angle(t1) * j + np.angle(t2) * np.cos(j / 3)
        coeff = (r_part + 1j * im_part) * np.exp(1j * angle) * mag
        cf[j-1] = coeff + np.conj(t1) * np.sin(j) + np.cos(j) * np.sin(j / 2)
    return cf.astype(np.complex128)

ALLOWED["p691"]=p691

def p692(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1,10):
        radius = np.log(np.abs(t1)**j + np.abs(t2)**(9-j) + 1)
        angle = np.sin(j * np.angle(t1)) + np.cos((9 - j) * np.angle(t2))
        cf[j-1] = radius * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p692"]=p692

def p693(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1,9):
        mag = np.log(np.abs(t1) + j**2) * np.sin(j * np.pi / 4) + np.cos(j * np.pi /6)
        ang = np.angle(t1) * np.cos(j) + np.angle(t2) * np.sin(j)
        cf[j-1] = mag * (np.cos(ang) +1j * np.sin(ang))
    cf[8] = t1.real**2 - t2.imag**2 + np.sin(t1.real)
    return cf.astype(np.complex128)

ALLOWED["p693"]=p693

def p694(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1,10):
        mag_part1 = np.log(np.abs(t1) + 1) * np.sin(j * np.pi / 5)
        mag_part2 = np.cos(j * t1.real) * np.abs(t2)**0.5
        mag_part3 = t1.real + t2.imag if j %2 ==0 else t1.real + t2.real
        magnitude = mag_part1 + mag_part2 + mag_part3
        
        angle_part1 = np.angle(t1)**j
        angle_part2 = t2.real * j
        angle_part3 = np.sin(j * np.angle(t1)) if j %3 ==0 else np.cos(j * np.angle(t2))
        angle = angle_part1 + angle_part2 + angle_part3
        
        base = magnitude * (np.cos(angle) + 1j * np.sin(angle))
        
        sum_part = np.sum(t1.real**np.arange(1,j+1)) + np.sum(t2.imag**np.arange(1,(j%2)+2))
        additional = sum_part * np.conj(t1 + t2)
        
        cf[j-1] = base + additional
    for k in range(1,10):
        cf[k-1] = cf[k-1] * (np.sin(k * t1.real) + np.cos(k * t2.imag)) * np.abs(t1 - t2)
    return cf.astype(np.complex128)

ALLOWED["p694"]=p694

def p695(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1,9):
        k = j +1
        mag = np.log(np.abs(t1) + np.abs(t2) + j) * (np.sin(j * t1.real) + np.cos(k * t2.imag))
        angle = np.angle(t1) * j - np.angle(t2) * k
        cf[j-1] = mag * np.exp(1j * angle)
    cf[8] = np.conj(cf[0]) + np.sum(cf[1:8]) * 0.5
    return cf.astype(np.complex128)

ALLOWED["p695"]=p695

def p696(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree +1, dtype=np.complex128)
    for j in range(0, degree +1):
        mag = np.log(np.abs(t1)**j + np.abs(t2)**(degree - j) + 1)
        ang = np.sin(j * np.angle(t1) + np.cos((degree - j) * np.angle(t2)))
        cf[j] = mag * np.exp(1j * ang) + np.conj(t1) * (j + 1)/(degree +1)
    return cf.astype(np.complex128)

ALLOWED["p696"]=p696

def p697(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree +1, dtype=np.complex128)
    for j in range(0, degree +1):
        mag_part = np.log(np.abs(t1) + 1) * (j +1)**np.sin(j) + np.sqrt(j +1) * np.cos(j * np.angle(t2))
        angle_part = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j)
        cf[j] = mag_part * (np.cos(angle_part) + 1j * np.sin(angle_part))
    for k in range(0, degree):
        cf[k+1] = cf[k+1] * np.exp(1j * (np.sin(k +1) + np.cos(k +1)))
    cf[0] = np.abs(t1) + np.abs(t2)
    cf[degree] = np.conj(t1) * np.conj(t2)
    return cf.astype(np.complex128)

ALLOWED["p697"]=p697

def p698(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    angle_part = 0+0j
    for j in range(1,10):
        mag_part = np.log(np.abs(t1)**j + np.abs(t2)**(9-j) + 1)
        angle_part = np.sin(j * np.angle(t1)) + np.cos(angle_part)
        cf[j-1] = mag_part * (np.cos(angle_part) + 1j * np.sin(angle_part))
    return cf.astype(np.complex128)

ALLOWED["p698"]=p698

def p699(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1,9):
        r = t1.real * j + t2.real * (9 - j)
        im = t1.imag / (j +1) - t2.imag / (10 - j)
        mag = np.log(np.abs(t1) + np.abs(t2) + j) * (j**2)
        angle = np.sin(r) + np.cos(im) + np.angle(t1) * np.angle(t2)
        cf[j-1] = mag * (np.cos(angle) + 1j * np.sin(angle))
    cf[8] = np.conj(t1) + np.conj(t2) + np.sin(t1.real * t2.real)
    return cf.astype(np.complex128)

ALLOWED["p699"]=p699

def p700(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1,10):
        magnitude = np.log(np.abs(t1) + np.abs(t2) + j) * (j**2 + np.sin(j))
        angle = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j**2)
        cf[j-1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)
  
ALLOWED["p700"]=p700

def p701(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree +1, dtype=np.complex128)
    for j in range(0, degree +1):
        r_part = t1.real * np.log(j + 2) + t2.real * np.sqrt(j +1)
        i_part = t1.imag * np.sin(j * np.pi /4) + t2.imag * np.cos(j * np.pi /3)
        magnitude = np.abs(t1)**(j %3 +1) + np.abs(t2)**(degree - j %2 +1)
        angle = np.angle(t1) * j + np.angle(t2) * (degree - j)
        cf[j] = (r_part +1j * i_part) * np.exp(1j * angle) + np.log(np.abs(magnitude) +1)
    return cf.astype(np.complex128)

ALLOWED["p701"]=p701

def p702(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    rec = np.linspace(t1.real, t2.real, 9)
    imc = np.linspace(t1.imag, t2.imag, 9)
    for j in range(1,10):
        magnitude = np.log(np.abs(rec[j-1] + imc[j-1]) + 1) * np.sin(j * np.pi /4) + np.cos(j * np.pi /6)
        angle = np.angle(t1) * np.sin(2 * np.pi * j /9) + np.angle(t2) * np.cos(4 * np.pi * j /9)
        cf[j-1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p702"]=p702

def p703(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1,10):
        mag = np.log(np.abs(t1) + np.abs(t2) + j) * (j**2 + np.sin(j * t1.real) + np.cos(t2.imag))
        angle = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j)
        cf[j-1] = mag * (np.cos(angle) + 1j * np.sin(angle)) + np.conj(t1) * t2.real * j
    return cf.astype(np.complex128)

ALLOWED["p703"]=p703

def p704(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree +1, dtype=np.complex128)
    for j in range(0, degree +1):
        r = j +1
        mag = np.log(np.abs(t1) * j +1) * (np.abs(t2)**(degree - j +1)) + np.sin(j * t1.real)**2
        ang = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j)
        real_part = mag * np.cos(ang) + t1.real**(j %3)
        imag_part = mag * np.sin(ang) + t2.imag**(j %2 +1)
        cf[j] = mag * np.exp( 1j * ang ) + ( np.conj(t1)**j ) * np.cos( j * np.real(t2) )
    return cf.astype(np.complex128)

ALLOWED["p704"]=p704

def p705(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1,10):
        mag_variation = np.log(np.abs(t1) + np.abs(t2) + j) * (1 + np.sin(j * t1.real) * np.cos(j * t2.imag))
        angle_variation = np.angle(t1) * np.sqrt(j) - np.angle(t2) / (j +1)
        cf[j-1] = mag_variation * np.exp(1j * angle_variation) + np.conj(t1) * np.sin(j) - np.conj(t2) * np.cos(j)
    return cf.astype(np.complex128)

ALLOWED["p705"]=p705

def p706(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1,10):
        mag = 0
        angle = 0
        for k in range(1,j+1):
            mag += np.log(np.abs(t1.real + k)) * np.sin(k * t2.real)
            angle += np.cos(k * t1.imag) * np.angle(t2)**k
        cf[j-1] = mag * (np.cos(angle) + 1j * np.sin(angle)) + np.conj(t1)**j + np.conj(t2)**j
    return cf.astype(np.complex128)

ALLOWED["p706"]=p706

def p707(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    rec = np.linspace(t1.real, t2.real, 9)
    imc = np.linspace(t1.imag, t2.imag, 9)
    for j in range(1,10):
        mag = np.exp(np.sin(5 * np.pi * imc[j-1])) + np.log(np.abs(rec[j-1]) + 1) * np.cos(3 * np.pi * rec[j-1])
        ang = np.angle(t1) * np.sin(2 * np.pi * j /9) + np.angle(t2) * np.cos(4 * np.pi * j /9)
        cf[j-1] = mag * (np.cos(ang) +1j * np.sin(ang))
    return cf.astype(np.complex128)

ALLOWED["p707"]=p707

def p708(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1,10):
        k = j**2
        r = t1.real * np.cos(k * np.angle(t2)) + np.sin(k * t1.imag)
        im = t2.imag * np.log(k + np.abs(t1)) + np.cos(k * t1.real)
        mag = np.sqrt(r**2 + im**2) * (1 + j)
        angle = np.arctan2(im, r) + np.sin(j * np.angle(t1 + t2))
        cf[j-1] = mag * (np.cos(angle) +1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p708"]=p708

def p709(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree +1, dtype=np.complex128)
    for j in range(0, degree +1):
        mag_part = np.log(np.abs(t1) + np.abs(t2) +1) * (j +1)**np.sin(j * t1.real) + np.cos(j * t2.real)
        angle_part = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j)
        cf[j] = mag_part * np.exp(1j * angle_part)
    for k in range(0, degree +1):
        cf[k] = cf[k] * (1 + 0.5 * np.sin(k * t1.imag) - 0.3 * np.cos(k * t2.imag))
    cf = cf * (1 + 0.1 * t1.real) / (1 + 0.1 * t2.imag)
    return cf.astype(np.complex128)

ALLOWED["p709"]=p709

def p710(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree +1, dtype=np.complex128)
    for j in range(1, degree +2):
        sum_part = 0
        for k in range(1, j +1):
            sum_part += np.cos(k * t1.real) * np.sin(k * t2.imag)
        mag = np.log(np.abs(t1) + np.abs(t2) + sum_part +1)
        angle = (np.angle(t1)**0.5 * j) + (np.angle(t2)**0.3 * (degree - j +1))
        cf[j-1] = mag * (np.cos(angle) +1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p710"]=p710

def p711(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(8, dtype=np.complex128)
    for k in range(1,9):
        r_part = t1.real**k + np.log(np.abs(t2))**2
        i_part = np.cos(k * t2.imag) + np.sin(k * np.angle(t1))
        angle = np.angle(t1) * k / 2 + np.sin(k) * np.pi /3
        cf[k-1] = (r_part + 1j * i_part) * (np.cos(angle) +1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p711"]=p711

def p712(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree +1, dtype=np.complex128)
    rec = np.linspace(t1.real, t2.real, degree +1)
    imc = np.linspace(t1.imag, t2.imag, degree +1)
    for j in range(1, degree +2):
        angle_component = np.sin(j * np.pi /3) * np.cos(j * np.pi /4)
        magnitude_component = np.log(np.abs(rec[j-1] * imc[j-1]) +1) + t1.real**j - t2.imag**(degree +1 -j)
        cf[j-1] = magnitude_component * np.exp(1j * (angle_component + np.angle(t1) * j))
    return cf.astype(np.complex128)

ALLOWED["p712"]=p712

def p713(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(8, dtype=np.complex128)
    for j in range(1,9):
        k = j * 2
        r = t1.real**j + np.log(np.abs(t2) +1) * np.sin(j * np.angle(t1))
        i_part = t2.imag**k - np.log(np.abs(t1) +1) * np.cos(j * np.angle(t2))
        mag = np.log(np.abs(t1)*np.abs(t2) +1) * j
        ang = np.angle(t1) + np.angle(t2) *j + np.sin(j) - np.cos(k)
        cf[j-1] = (r +1j * i_part) * mag * (np.cos(ang) +1j * np.sin(ang)) + np.conj(t1)**j * np.conj(t2)**k
    return cf.astype(np.complex128)

ALLOWED["p713"]=p713

def p714(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1,10):
        r = t1.real * j**2 - t2.real / (j +1)
        im = t1.imag + t2.imag * np.sin(j)
        mag = np.log(np.abs(t1) + np.abs(t2) + j) * (1 + np.abs(np.sin(j * np.pi /3)))
        ang = np.angle(t1) * np.cos(j) + np.angle(t2) * np.sin(j * np.pi /4)
        cf[j-1] = mag * np.exp(1j * ang) + np.conj(t1) * np.sin(j) - np.conj(t2) * np.cos(j)
    return cf.astype(np.complex128)

ALLOWED["p714"]=p714

def p715(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1,10):
        rj = t1.real + j * t2.real
        ij = t1.imag - j * t2.imag
        mag = np.log(np.abs(rj + ij) +1) * (1 + np.sin(j * np.pi / 4 )) * (j**1.5)
        ang = np.angle(t1) * np.cos(j /3) + np.angle(t2) * np.sin(j /5)
        cf[j-1] = mag * np.exp(1j * ang)
    return cf.astype(np.complex128)
 
ALLOWED["p715"]=p715

def p716(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1,10):
        mag = np.log(np.abs(t1) + j) * (np.sin(j * np.angle(t1)) + np.cos(j * np.angle(t2)))
        angle = np.angle(t1) * np.sqrt(j) + np.angle(t2) / (j +1)
        cf[j-1] = mag * np.exp(1j * angle)
    return cf.astype(np.complex128)

ALLOWED["p716"]=p716

def p717(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1,10):
        mag = np.log(np.abs(t1)**j + np.abs(t2)**(j/2)) + (t1.real * t2.imag)
        angle = np.angle(t1) * np.cos(j) + np.angle(t2) * np.sin(j) + np.sin(j * t1.real) * np.cos(j * t2.imag)
        cf[j-1] = mag * np.exp(1j * angle) + np.conj(t1 + t2) / (j +1)
    return cf.astype(np.complex128)

ALLOWED["p717"]=p717

def p718(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1,10):
        k = j + 2
        r = t1.real * np.cos(j) - t2.imag * np.sin(j)
        im = t2.real * np.sin(j) + t1.imag * np.cos(j)
        magnitude = np.log(np.abs(t1) + np.abs(t2) + j) * (1 + np.sin(k))+np.cos(k)
        angle = np.angle(t1) * np.sin(k) + np.angle(t2) * np.cos(k)
        cf[j-1] = magnitude * (np.cos(angle) +1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p718"]=p718

def p719(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree +1, dtype=np.complex128)
    for j in range(0, degree +1):
        k = j +1
        r = t1.real * np.sin(j * t1.real) + t2.imag * np.cos(k * t2.real)
        mag = np.log(np.abs(t1) + np.abs(t2) + j +1) * (j +1)**2
        angle = np.sin(r) + np.cos(k * np.pi /4)
        cf[j] = mag * (np.cos(angle) +1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p719"]=p719

def p720(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree +1, dtype=np.complex128)
    for k in range(0, degree +1):
        mag_part = np.abs(t1)**k * np.abs(t2)**(degree -k) + np.log(np.abs(t1) + np.abs(t2) +1)
        angle_part = np.angle(t1) * np.sin(k) + np.angle(t2) * np.cos(k)
        variation = np.sin(k * np.pi /3) + np.cos(k * np.pi /4)
        cf[k] = (mag_part * variation) * (np.cos(angle_part) +1j * np.sin(angle_part))
    return cf.astype(np.complex128)

ALLOWED["p720"]=p720

def p721(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree +1, dtype=np.complex128)
    for j in range(1, degree +2):
        angle = np.sin(j * t1.real) + np.cos(j * t2.imag) + np.angle(t1) * j
        mag = np.log(j +1) * np.abs(t1)**j + np.abs(t2)**(degree +1 -j)
        cf[j-1] = mag * (np.cos(angle) +1j * np.sin(angle))
    for k in range(1, degree +2):
        cf[k-1] = cf[k-1] * np.conj(t1) / (1 + k)
    return cf.astype(np.complex128)

ALLOWED["p721"]=p721

def p722(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree +1, dtype=np.complex128)
    for j in range(1,10):
        mag_part = np.log(np.abs(t1)**j + np.abs(t2)**(9-j) + 1)
        angle_part = np.sin(j * np.angle(t1)) + np.cos((9-j) * np.angle(t2))
        real_part = mag_part * np.cos(angle_part) + t1.real * np.abs(t2)**j
        imag_part = mag_part * np.sin(angle_part) + t2.imag * np.abs(t2)**j
        cf[j-1] = complex(real_part, imag_part)
    return cf.astype(np.complex128)

ALLOWED["p722"]=p722

def p723(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1,10):
        mag_part = np.log(np.abs(t1) + j) * np.sin(j * np.angle(t2)) + np.cos(j * t1.imag)
        angle_part = np.angle(t1)**j - np.angle(t2)**(j %3) + np.sin(j * t1.real)
        cf[j-1] = mag_part * np.exp(1j * angle_part)
    return cf.astype(np.complex128)

ALLOWED["p723"]=p723

def p724(z,a,state):
    t1, t2 = z[0], z[1]
    degree = 8
    cf = np.zeros(degree +1, dtype=np.complex128)
    for j in range(1, degree +2):
        mag_part = np.abs(t1)**j * np.log(np.abs(t2) +1) + np.abs(t2)**(degree -j +1) * np.sin(j * t1.real)
        angle_part = np.angle(t1) * np.cos(j * t2.real) + np.angle(t2) * np.sin(j * t1.real)
        cf[j-1] = mag_part * np.exp(1j * angle_part) + np.conj(t1 * t2) / (j +1)
    return cf.astype(np.complex128)

ALLOWED["p724"]=p724

def p725(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1,10):
        k = j %5 +1
        r = np.log(np.abs(t1) + j) * (1 + np.sin(j * t2.real) + np.cos(k * t1.imag))
        angle = np.angle(t1) * np.cos(j) + np.angle(t2) * np.sin(k)
        cf[j-1] = r * (np.cos(angle) +1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p725"]=p725

def p726(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for k in range(1,10):
        mag = np.log(np.abs(t1) + k**2) * np.sin(k * np.angle(t2)) + np.cos(k * t1.real)
        angle = np.angle(t1) * np.log(np.abs(t2) +1) + np.sin(k * t2.imag)
        cf[k-1] = mag * (np.cos(angle) +1j * np.sin(angle)) * (-1)**k
    return cf.astype(np.complex128)

ALLOWED["p726"]=p726

def p727(z,a,state):
    t1, t2 = z[0], z[1]
    n =  9
    cf = np.zeros(n, dtype=np.complex128)
    rec = np.linspace(t1.real, t2.real, n)
    imc = np.linspace(t1.imag, t2.imag, n)
    for k in range(1, n+1):
        mag = np.log( np.abs(t1) + np.abs(t2) + k ) * (k**2)
        angle = np.angle(t1) * np.sin(k) + np.angle(t2) * np.cos(k)
        cf[k-1] = mag * (np.cos(angle) +1j * np.sin(angle))
    return cf.astype(np.complex128)
    
ALLOWED["p727"]=p727

def p728(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1,10):
        mag = np.log(np.abs(t1) + j) * np.abs(np.sin(j * t1.real)) + np.sqrt(np.abs(t2.imag) + j)
        ang = np.angle(t1) * np.cos(j * np.angle(t2)) + np.sin(j * t2.real)
        cf[j-1] = mag * np.exp(1j * ang)
    return cf.astype(np.complex128)

ALLOWED["p728"]=p728 
    
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

def p730(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1,10):
        mag_part = np.log(np.abs(t1) +1) * (j +1)**np.sin(j * t1.real) + np.cos(j * t2.real)
        angle_part = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j)
        cf[j-1] = mag_part * np.exp(1j * angle_part)
    return cf.astype(np.complex128)

ALLOWED["p730"]=p730

def p731(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1,10):
        mag_part = np.log(np.abs(t1) * j +1) * np.abs(np.sin(t1.real * j) + np.cos(t2.imag / (j +1)))
        angle_part = np.angle(t1) * np.sqrt(j) + np.angle(t2) / (j +2)
        cf[j-1] = mag_part * (np.cos(angle_part) +1j * np.sin(angle_part))
    return cf.astype(np.complex128)

ALLOWED["p731"]=p731

def p732(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for k in range(0,9):
        r_part = t1.real * np.log(k +2) + t2.real * np.sqrt(k +1)
        i_part = t1.imag * np.sin(k) - t2.imag * np.cos(k)
        angle = np.sin(r_part) + np.cos(i_part)
        magnitude = np.log(np.abs(t1) + np.abs(t2) +k +1) * (k +1)
        cf[k] = magnitude * (np.cos(angle) +1j * np.sin(angle)) + np.conj(t1) * np.conj(t2)**k
    return cf.astype(np.complex128)

ALLOWED["p732"]=p732 

def p733(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for k in range(1,10):
        tmp = 0 +0j
        for j in range(1,k+1):
            tmp += (t1.real**j / (j +1)) * np.exp(1j * np.sin(j * t2.real))
        for r in range(1, (k %3)+2):
            tmp += (t2.imag**r / (r +2)) * np.exp(1j * np.cos(r * t1.imag))
        cf[k-1] = tmp
    return cf.astype(np.complex128)

ALLOWED["p733"]=p733

def p734(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1,10):
        mag = np.log(np.abs(t1) + j**2) * np.sin(j * np.angle(t2)) + np.cos(j * t1.real * t2.imag)
        angle = np.angle(t1) * np.cos(j * t2.real) + np.sin(j) * np.log(np.abs(t2) +1)
        cf[j-1] = mag * np.exp(1j * angle) + np.conj(t1) * t2**j
    return cf.astype(np.complex128)

ALLOWED["p734"]=p734

def p735(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1,10):
        real_part = t1.real**j + np.log(np.abs(t2) +1) *j
        imag_part = t2.imag * np.sin(j * np.angle(t1)) + np.cos(j * t2.real)
        magnitude = np.log(np.abs(t1 + t2) +j) * (1 +j**2)
        angle = np.angle(t1) * np.cos(j) + np.sin(j * np.angle(t2)) - np.cos(j * t1.imag)
        cf[j-1] = magnitude * (np.cos(angle) +1j * np.sin(angle))
    return cf.astype(np.complex128)

ALLOWED["p735"]=p735

def p736(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    for j in range(1,10):
        mag_part = np.log(np.abs(t1) + j) * np.sin(j * np.angle(t2)) + np.cos(j * t1.real)
        ang_part = np.angle(t1) * np.cos(j) + np.sin(j * np.angle(t2))
        cf[j-1] = mag_part * np.exp(1j * ang_part)
    return cf.astype(np.complex128)

ALLOWED["p736"]=p736

def p737(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(9, dtype=np.complex128)
    rec = np.linspace(t1.real, t2.real, 9)
    imc = np.linspace(t1.imag, t2.imag, 9)
    for k in range(1,10):
        mag = np.log(np.abs(t1) + np.abs(t2) +k) * (1 + np.sin(3 * np.pi * rec[k-1]) + np.cos(2 * np.pi * imc[k-1]))
        ang = np.angle(t1) * np.sin(5 * np.pi * imc[k-1]) - np.angle(t2) * np.cos(4 * np.pi * rec[k-1])
        cf[k-1] = mag * (np.cos(ang) +1j * np.sin(ang))
    return cf.astype(np.complex128)

ALLOWED["p737"]=p737

def p738(z,a,state):
    t1, t2 = z[0], z[1]
    r = t1.real * t2.imag
    i = t1.imag * t2.real
    
    cf = np.zeros(10, dtype=np.complex128)
    cf[0] = i * np.exp(1j * r) 
    cf[1] = 100j * np.sin(r) * np.cos(i)
    cf[2] = 1j * r + r * 1j
    cf[3] = np.abs(r * i) * np.exp(1j * np.arctan2(r, i))
    cf[4] = 100 / (1 + np.exp(-r))
    cf[5] = r * np.abs(i) + i * np.abs(r)
    cf[6] = np.sqrt(np.abs(r * i)**(1/3)) * np.exp(1j * (np.arctan2(r, i) + np.pi/4))
    cf[7] = r**2 + i**2
    cf[8] = np.abs(r +1j*i)**1.5 * np.exp(1j * (np.arctan2(r, i) + np.pi/4))
    cf[9] = r * i * np.abs(r - i)
    return cf.astype(np.complex128)

ALLOWED["p738"]=p738

def p739(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(11, dtype=np.complex128)
    cf[0] = np.log(t1 + t2)
    cf[1] = np.exp(1j * t1)
    cf[2] = np.exp(-1j * t2)
    cf[3] = t1**2 - t2**2
    cf[4] = 1j * t1 * t2
    cf[5] = np.sin(t1) + np.cos(t2)
    cf[6] = np.cos(t1) - np.sin(t2)
    cf[7] = (t1 +1j*t2)**2
    cf[8] = (t2 -1j*t1)**3
    cf[9] = np.sqrt(t1**2 + t2**2)
    cf[10] = t1*t2*(t1 - t2)*(t1 +1j*t2)*(t2 -1j*t1)
    return cf.astype(np.complex128)

ALLOWED["p739"]=p739

def p740(z, a, state):
    t1 = z[0]
    t2 = z[1]

    r1 = t1.real
    r2 = t2.real
    i1 = t1.imag
    i2 = t2.imag

    # Safe square-roots (avoid NaNs for negative inputs)
    sr1 = np.sqrt(np.abs(r1))
    sr2 = np.sqrt(np.abs(r2))
    si1 = np.sqrt(np.abs(i1))
    si2 = np.sqrt(np.abs(i2))

    cf = np.zeros(11, dtype=np.complex128)

    cf[0]  = i1**3 - i2**3 + r1**2 - r2**2
    cf[1]  = (r1 * i1 * r2 * i2) * (r1 - 1j * i1)
    cf[2]  = i1**2 + r2**2 - i2**2 + (r2 - i1)
    cf[3]  = r1 * (r1**2 * i1**2 - r2**2 * i2**2)
    cf[4]  = r1**3 + r1**2 + i2**3 + i1**2 - 10.0
    cf[5]  = i1 * i2 * (i1**2 * r2**2 - i2**2 * r1**2)
    cf[6]  = sr1 - si2 + sr2 - si1                  # was r1**0.5 - i2**0.5 + r2**0.5 - i1**0.5
    cf[7]  = r1 * i1**2 - r2 * i2 * (r1 - 1j * i1)
    cf[8]  = i1**3 - i2**3 + r2**2 - r1**2
    cf[9]  = (r1 * i1 * r2 * i2) * (i2 - r1)
    cf[10] = i1**4 + i2**4 + r1**4 + r2**4

    return cf

ALLOWED["p740"]=p740
    
def p741(z,a,state):
    t1, t2 = z[0], z[1]
    n = 10
    cf = np.zeros(n, dtype=np.complex128)
    cf[0] = np.real((t1 +7j)**2) + np.imag((t2 +5j)**3)
    cf[1] = np.exp(1j*t1*t2)
    cf[2:6] = np.log(1j * np.array([2,3,4,5])) + np.exp(1j*t1) + np.exp(-1j*t2)
    cf[6:10] = cf[0:4][::-1]
    return cf.astype(np.complex128)

ALLOWED["p741"]=p741

def p742(z, a, state):
    t1, t2 = z[0], z[1]
    n = 10
    cf = np.zeros(n, dtype=np.complex128)

    base = t1 + 1j * t2          # complex base
    p    = np.complex128(1.0)    # base**0
    fact = 1.0                   # 0! = 1

    for k in range(1, n + 1):
        p *= base                # base**k (iterative, njit-friendly)
        fact *= float(k)         # k!
        phase = math.sin(float(k) * t2.real)
        ei_phase = np.complex128(math.cos(phase) + 1j * math.sin(phase))
        cf[k - 1] = (p / fact) * ei_phase

    # Overwrites (keep them after loop)
    cf[0] = t1**3 - 1j * (t1**2) + (t2**2) - 1j * t2
    cf[4] = t2.real * t1.imag - 1j * (t2**3)

    theta = t1.real + t2.real
    eitheta = np.complex128(math.cos(theta) + 1j * math.sin(theta))
    cf[9] = (t1.real**2) * (t2.real**2) * eitheta

    return cf

ALLOWED["p742"]=p742

def p743(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(11, dtype=np.complex128)
    cf[0] = t1 + t2
    cf[1] = np.sin(t1 * t2) * np.exp(1j * t1)
    cf[2] = t1**2 + t2**2
    cf[3] = np.exp(1j * t1**2) * np.cos(t2)
    cf[4] = (t1 +1j*t2) * (t1 -1j*t2)
    cf[5] = (t2 +1j*t1)**3 - (t1 -1j*t2)**3
    cf[6] = np.exp(-1j * t2**2) * np.sin(t1)
    cf[7] = np.real(t1**2 * t2**2 * np.exp(1j * (t1.real + t2.imag)))
    cf[8] = 100 * np.sin(t1) * np.cos(t2) - 100j * np.sin(t2) * np.cos(t1)
    cf[9] = 1j * (t1**3 - t2**3) + (100 -1j)
    cf[10] = (t1 + t2) * np.exp(2j * t1 * t2)
    return cf.astype(np.complex128)

ALLOWED["p743"]=p743

def p744(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(11, dtype=np.complex128)
    cf[0] = t1**3 + t2**3
    cf[1] = 11*(t1 + t2)**9
    cf[2] = 0 +1j
    cf[3] = np.exp(1j*t1)
    cf[4] = 100 * np.sin(t2)
    cf[5] = t1.real -1j * t2.imag
    cf[6] = 11j * (t2.real / np.abs(t1.imag +0.1))
    cf[7] = t1.real / (np.abs(t1.real + t2.real) +0.125)
    cf[8] = np.exp(1j * t1.real * t2.real)
    cf[9] = np.abs(t1 * t2) * np.exp(1j*(np.angle(t1) - np.angle(t2)))
    cf[10] = t1.real * t2.imag +10j
    return cf.astype(np.complex128)

ALLOWED["p744"]=p744

def p745(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(10, dtype=np.complex128)
    cf[9] = t1.real * t2.imag
    cf[4] = t1.imag * t2.real
    cf[0] = np.exp(1j*(t1 + t2))
    m = np.abs(t1 + t2)
    cf[2] = 1 / (m +1)
    polar_coordinates = np.sqrt(t1.real**2 + t1.imag**2) * np.sqrt(t2.real**2 + t2.imag**2)
    cf[6] = np.exp(1j * polar_coordinates)
    cf[8] = np.sum(np.arange(1,10)**2) * t1.real
    cf[1] = t2**10 - cf[9]**10
    cf[3] = np.angle(t1) * cf[1]
    cf[5] = cf[2] + cf[6] * cf[8]
    cf[7] = np.conj(cf[3]) * cf[5]
    return cf.astype(np.complex128)

ALLOWED["p745"]=p745

def p746(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(11, dtype=np.complex128)
    coeff_sequence = np.linspace(-1 + 2 * np.real(t1), 1 - 2 * np.imag(t2), num=11)
    for i in range(11):
        cf[i] = np.exp(1j * (i + 1) * t1) + np.sum(coeff_sequence[:i + 1] * np.log(1 + np.abs(t2))**2)
    cf[10] += np.sqrt(cf[0] * cf[1] * t1)
    cf[0] -= np.sqrt(cf[9] * cf[10] * t2)
    cf[5] = np.sum(cf) / 11
    cf[2] *= cf[7] / cf[5]
    cf[7] = cf[7]**2 - cf[4] + cf[8]
    return cf.astype(np.complex128)

ALLOWED["p746"]=p746

def p747(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(11, dtype=np.complex128)
    for k in range(11):
        cf[k] = np.exp((t1 + t2 * 1j) ** (k + 1))
        if k % 2 == 1:
            cf[k] *= np.cos(k + np.imag(t2))
        else:
            cf[k] *= np.sin(k + np.real(t1))
        cf[k] += t1 * t2 * 1j * (k + 1) ** 2
        if cf[k] == 0:
            cf[k] = -1j
    return cf.astype(np.complex128)

ALLOWED["p747"]=p747

def p748(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(11, dtype=np.complex128)
    cf[0] = (t1 + t2) ** 2 - np.sqrt(np.abs(t1 * t2))
    cf[1] = np.log(1 + np.abs(t2)) * np.exp(1j * t1)
    cf[2] = np.real(t1) ** 2 - np.imag(t2) ** 2
    cf[3] = np.exp(1j * (np.real(t1) + np.real(t2)))
    cf[4] = 100 * np.exp(-np.abs(t1 - t2))
    cf[5] = (np.imag(t1) ** 3 - np.real(t2) ** 3) * np.exp(1j * (np.real(t1) + np.imag(t2)))
    cf[6] = np.sin(t1) - np.cos(t2)
    cf[7] = t2 ** 2 - t1 ** 2
    cf[8] = -10 * np.exp(1j * (t1 - t2) ** 2)
    cf[9] = np.exp(1j * t1 * t2) - np.sin(t1 * t2)
    cf[10] = (t1 + t2) ** 3 * np.exp(1j * (t1 - t2))
    return cf.astype(np.complex128)

ALLOWED["p748"]=p748

def p749(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(10, dtype=np.complex128)
    cf[0] = np.exp(1j * (t1 + t2))
    cf[1] = np.sin(t1) * np.cos(t2) - np.cos(t1) * np.sin(t2)
    for j in range(2, 9):
        cf[j] = np.exp((j + 1) / 3) * np.sin(t1 + t2) * np.exp(-1j * (t1 - t2) / (j + 1))
    cf[9] = np.sqrt(t1 ** 2 + t2 ** 2) - np.log10(np.abs(t1) + np.abs(t2))
    return cf.astype(np.complex128)

ALLOWED["p749"]=p749

def p750(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(11, dtype=np.complex128)
    cf[0] = np.abs(t1 + t2) + 100j
    for k in range(1, 10):
        cf[k] = cf[k - 1] * (-1j * t1 + 1 * t2) ** (k + 1) / (k + 1)
    cf[10] = cf[0] * np.exp(1j * np.abs(np.sum(cf[:-1])))
    return cf.astype(np.complex128)

ALLOWED["p750"]=p750

def p751(z,a,state):
    t1, t2 = z[0], z[1]
    angle = np.linspace(0, 2 * np.pi, num=11)
    cf = 10 * np.exp(1j * angle)
    cf[0] = np.abs(t1 + t2)
    cf[5] = np.sin(t1) * (np.cos(t2) ** 2)
    cf[7] = -np.log(np.abs(t1) + 1) + 1j * np.log(np.abs(t2) + 1)
    cf[9] = (t1 ** 2) / (t2 + 1j)
    cf[10] = np.sqrt(t1 * t2)
    return cf.astype(np.complex128)

ALLOWED["p751"]=p751

def p752(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(11, dtype=np.complex128)
    cf[0] = np.exp(1j * np.abs(t1)) + np.cos(np.real(t2) ** 2)
    cf[1] = 1j * np.sin(np.imag(t1 ** 2)) - t2
    cf[2] = (np.abs(t1) * np.abs(t2)) / 3
    cf[3] = (t1 + 1j * t2) / 2
    cf[4] = (t1 - 1j * t2) ** 2
    cf[5] = np.exp(1j * np.real(t1) * np.imag(t2)) - t1
    cf[6] = 1j * np.abs(t2 ** 3) + t1 ** 2
    cf[7] = np.exp(np.real(t1) ** 3 - 1j * np.imag(t2) ** 3)
    cf[8] = np.abs(t1 - t2) * (t1 + t2)
    cf[9] = t1 * np.abs(t2) - 1j * t1 * t2
    cf[10] = (t1 + t2) / (1 + t1 ** 2 + t2 ** 2)
    return cf.astype(np.complex128)

ALLOWED["p752"]=p752

def p753(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(10, dtype=np.complex128)
    cf[0] = np.sin(t1 + t2) + np.cos(t1 - t2)
    cf[1] = np.exp(1j * (np.abs(t1 + t2) ** 2))
    prime_numbers = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
    for k in range(2, 8):
        cf[k] = prime_numbers[k] * ((t1 / (k + 1)) ** 2) * np.exp(1j * (t2 / (k + 1)))
    fibonacci_sequence = np.array([0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55])
    cf[8] = fibonacci_sequence[10] * t1 * t2 * np.exp(1j * (t1 - t2))
    cf[9] = ((t1 ** 3) + (t2 ** 3) - 1) * np.exp(1j * (t1 + t2))
    return cf.astype(np.complex128)

ALLOWED["p753"]=p753

def p754(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(10, dtype=np.complex128)
    cf[0] = 1j * t1 ** 2
    cf[1] = 1 / 2 * np.exp(-t2)
    cf[2] = t1 * t2 - t2 ** 3
    cf[3] = np.cos(t1) + np.cos(t2)
    cf[4] = np.sin(t1) * np.cos(t2)
    cf[5] = np.log(np.abs(t1 - t2) + 1)
    cf[6] = (t1 + t2) ** 2
    cf[7] = np.real(t1) ** 3 + np.imag(t2) ** 3
    cf[8] = np.abs(t1 * t2)
    cf[9] = np.exp(1j * (t1 * t2))
    return cf.astype(np.complex128)

ALLOWED["p754"]=p754

def p755(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(11, dtype=np.complex128)
    cf[0] = (t1 * np.conj(t2)) ** 3 + t1 - t2
    cf[10] = t2 + 1j * t1
    snd = np.linspace(np.sin(2 * np.pi * np.real(t1)), np.sin(2 * np.pi * np.imag(t2)), num=9)
    csi = np.linspace(np.cos(2 * np.pi * np.imag(t1)), np.cos(2 * np.pi * np.real(t2)), num=9)
    stat = t1 * csi ** 2 + t2 * snd ** 2 + 1j * (t1 * snd ** 2 + t2 * csi ** 2)
    cf[1:10] = stat
    return cf.astype(np.complex128)

ALLOWED["p755"]=p755

def p756(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(10, dtype=np.complex128)
    cf[0] = np.exp(1j * t1 * t2)
    cf[1] = 100 * np.sin(t1 * t2)
    cf[2] = 100j * np.cos(t1 * t2)
    cf[3] = 1j * t1 ** 3 - 2j * t1 * t2 ** 2
    cf[4] = t1 ** 5 + t2 ** 5
    cf[5] = 10j * t1 ** 4 - 10j * t2 ** 4
    for k in range(6, 10):
        cf[k] = cf[k - 1] * 1j * 0.8 ** (k + 1)
    return cf.astype(np.complex128)

ALLOWED["p756"]=p756

def p757(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(11, dtype=np.complex128)
    cf[0] = 0
    cf[1] = np.sqrt(np.abs(t1))
    cf[2] = np.cos(t1) + np.sin(t2)
    cf[3] = t2 / (np.abs(t1) + 1)
    cf[4] = np.exp(1j * np.angle(t1)) - np.exp(-1j * np.angle(t2))
    cf[5] = cf[2] ** 2 - cf[3] ** 2
    cf[6] = (t1 * t2 * cf[1]) / (1 + cf[1] ** 2)
    cf[7] = np.real(t1) + np.imag(t2) - (np.real(t2) + np.imag(t1))
    cf[8] = cf[7] / (1 + np.abs(cf[5]))
    cf[9] = cf[4] * cf[5] * cf[6]
    cf[10] = cf[8] ** 3 + cf[9] ** 2
    return cf.astype(np.complex128)
 
ALLOWED["p757"]=p757

def p758(z, a, state):
    t1 = z[0]
    t2 = z[1]
    cf = np.zeros(10, dtype=np.complex128)

    # precompute re/im
    re1, im1 = t1.real, t1.imag
    re2, im2 = t2.real, t2.imag

    # safe magnitudes and angles
    abs_t1 = np.hypot(re1, im1)
    abs_t2 = np.hypot(re2, im2)
    abs_t1t2 = np.hypot(re1 + re2, im1 + im2)
    abs_t1mt2 = np.hypot(re1 - re2, im1 - im2)
    ang_t1t2 = np.arctan2(im1 + im2, re1 + re2)
    ang_t1mt2 = np.arctan2(im1 - im2, re1 - re2)
    ang_t1t2_prod = np.arctan2(im1 * re2 + im2 * re1, re1 * re2 - im1 * im2)

    # cf[0]
    cf[0] = (t1**2 + t2**2) * np.exp(1j * ang_t1t2)

    # cf[1]
    cf[1] = 100j * (t1**3 - t2**2) * np.cos(ang_t1mt2)

    # cf[2]
    real_part = (re1 - re2) ** 2
    imag_part = (im1 + im2) ** 2
    cf[2] = real_part + imag_part - 100.0

    # cf[3]  safe logs
    val1 = np.log1p(abs_t1)
    val2 = np.log1p(abs_t2)
    cf[3] = 42.0 * (val1 + val2) * np.exp(1j * np.pi / 4)

    # cf[4]
    mag = np.sqrt(abs_t1 + abs_t2)
    cf[4] = mag * np.exp(1j * (ang_t1t2_prod - np.pi / 3.0))

    # cf[5]
    cf[5] = np.sinh(0.1 * (re1 + re2)) + np.cosh(0.1 * (im1 - im2))

    # cf[6]  logistic form, safe exponent
    e1 = np.exp(-abs_t1) if abs_t1 < 700.0 else 0.0
    e2 = np.exp(-abs_t2) if abs_t2 < 700.0 else 0.0
    cf[6] = 1.0 / (1.0 + e1) + 1j / (1.0 + e2)

    # cf[7]  safe arctan
    denom = abs_t1t2
    if denom == 0.0 or np.isnan(denom) or np.isinf(denom):
        cf[7] = 0.0j
    else:
        cf[7] = 1j * np.arctan(1.0 / denom)

    # cf[8]
    cf[8] = 1j * t1**3 - (t2 * 2.0) - 200.0

    # cf[9]  safe division
    denom2 = 1.0 + t1 + t2
    if np.abs(denom2) < 1e-12:
        cf[9] = 0.0j
    else:
        cf[9] = np.exp(1j * t1 * t2) / denom2

    return cf

ALLOWED["p758"]=p758

def p759(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(11, dtype=np.complex128)
    cf[0] = np.exp(t1) * np.cos(t2)
    cf[1] = -np.exp(1j * t1)
    cf[2] = np.log(np.abs(t2))
    cf[3] = (t1 + t2) / (1j)
    cf[4] = t1 ** 3 - t2 ** 3
    cf[5] = -2 * np.exp(t2) * np.sin(t1)
    cf[6] = np.sqrt(np.abs(t1 - t2))
    cf[7] = np.exp(1j * (t1 + t2) ** 2)
    cf[8] = np.log1p(np.abs(t1 * t2))
    cf[9] = (t1 - t2) / (1j)
    cf[10] = np.exp(t1 * t2) / t1
    return cf.astype(np.complex128)

ALLOWED["p759"]=p759

def p760(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(11, dtype=np.complex128)

    # Helpers
    r1 = math.hypot(t1.real, t1.imag)  # |t1|
    r2 = math.hypot(t2.real, t2.imag)  # |t2|

    # cf[0]
    denom0 = 1.0 + (math.hypot((t1 + t2).real, (t1 + t2).imag) ** 2)
    cf[0] = np.complex128(
        math.sin((t1.real + 2.0 * t2.imag) / denom0)
    )

    # cf[1]
    # abs(cos(Re t2))**2.1 and abs(sin(Im t2))**2.1
    term_c = math.pow(abs(math.cos(t2.real)), 2.1)
    term_s = math.pow(abs(math.sin(t2.imag)), 2.1)
    cf[1] = np.complex128(100.0 * t1.real * t1.imag * (term_c - term_s))

    # cf[2] -- 2x2 inverse sum without np.linalg
    # M = [[a,b],[c,d]] = [[Re t2, Im t2], [-Im t1, Re t1]]
    a = t2.real
    b = t2.imag
    c = -t1.imag
    d = t1.real
    det = a * d - b * c  # = Re2*Re1 + Im1*Im2
    if abs(det) < 1e-10:
        vv = 0.0
    else:
        # inv(M) = (1/det) * [[d, -b], [-c, a]]
        vv = (d - b - c + a) / det  # sum of all entries of inv(M)
    cf[2] = np.complex128(1000.0 * vv)

    # cf[3] -- sum of inverse FFT of [t1, t2] (length-2) equals t1
    # ifft([x0,x1]) = [(x0+x1)/2, (x0-x1)/2]; sum = x0
    cf[3] = np.complex128(t1)

    # cf[4] -- exp(i*arg(t1)) * |t1| using cos/sin + atan2 + hypot
    theta1 = math.atan2(t1.imag, t1.real)
    cf[4] = np.complex128(
        10.0 * (math.cos(theta1) + 1j * math.sin(theta1)) * r1
    )

    # cf[5] -- median of four reals (Numba-safe)
    a1 = t1.real
    a2 = t1.imag
    a3 = t2.real
    a4 = t2.imag
    # sort 4 numbers (simple network)
    if a1 > a2: a1, a2 = a2, a1
    if a3 > a4: a3, a4 = a4, a3
    if a1 > a3: a1, a3 = a3, a1
    if a2 > a4: a2, a4 = a4, a2
    if a2 < a3: a2, a3 = a3, a2
    med = 0.5 * (a2 + a3)  # median of 4 = mean of middle pair
    cf[5] = np.complex128(1000.0 * med * med + 500.0 * med * med * med)

    # cf[6]
    cf[6] = np.complex128(t2.real + 1j * (2.0 * math.sqrt(abs(t2.imag))))

    # cf[7] -- log10(|Re t2| + |Im t2|), guard <=0
    s7 = abs(t2.real) + abs(t2.imag)
    if s7 <= 0.0:
        cf[7] = np.complex128(-1e300)  # or 0.0, depending on your preference
    else:
        cf[7] = np.complex128(100.0 * (math.log(s7) / math.log(10.0)))

    # cf[8] -- sqrt(|t1*t2|) = sqrt(|t1|*|t2|)
    cf[8] = np.complex128(math.sqrt(r1 * r2))

    # cf[9] -- 1000 * Im(t1)^3 - 3 * Im(t1) * Re(t1)^2 (real)
    cf[9] = np.complex128(1000.0 * (t1.imag * t1.imag * t1.imag
                                    - 3.0 * t1.imag * t1.real * t1.real))

    # cf[10]
    cf[10] = np.complex128(2.0j * (t1.real - t1.imag) + 2.0 * (t1.real + t1.imag))

    return cf

ALLOWED["p760"]=p760

def p761(z,a,state):
    t1, t2 = z[0], z[1]
    n = 10
    cf = np.zeros(n, dtype=np.complex128)
    for k in range(n):
        cf[k] = (100 * np.sin(np.real(t1) * (k + 1)) * np.cos(np.imag(t2) * (k + 1))) * np.exp(1j * (np.real(t2) * (k + 1) / 100))
    cf[0] *= 10
    cf[4] *= np.sin(t1 * t2)
    cf[9] += np.cos(t1 * t2) * np.exp(1j * (np.real(t1) + np.real(t2)))
    return cf.astype(np.complex128)

ALLOWED["p761"]=p761

def p762(z, a, state):
    t1, t2 = z[0], z[1]

    cf = np.zeros(10, dtype=np.complex128)

    # complex primes as constants (Numba-friendly)
    primes = np.array([2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0, 29.0], dtype=np.complex128)

    cf[0] = t1 + t2

    s = t1 * t1 + t2 * t2
    cf[1:9] = primes[1:9] * (s + 1j)

    # sum of first 9 primes (avoid np.sum on complex in nopython)
    sp = 0.0 + 0.0j
    for i in range(9):
        sp += primes[i]

    # |s| using hypot(re, im) to avoid np.abs(complex) issues
    mag_s = math.hypot(s.real, s.imag)

    cf[9] = sp + mag_s

    return cf

ALLOWED["p762"]=p762

def p763(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(10, dtype=np.complex128)
    cf[0] = (t1 ** 2 + t2 ** 2) * 1j
    cf[1] = 10
    cf[4] = np.abs(t1 * 100) - 0.5
    cf[5] = np.abs(t2 * 100) - 0.5
    cf[7] = -10
    cf[9] = (t1 ** 2 + t2 ** 2)
    return cf.astype(np.complex128)

ALLOWED["p763"]=p763

def p764(z,a,state):
    t1, t2 = z[0], z[1]
    n = 10
    cf = np.zeros(n, dtype=np.complex128)
    cf[0] = np.exp(1j * t1)
    cf[1] = (t1 + t2) * np.cos(t1) + 1j * np.sin(t2)
    cf[2] = t1 ** 3 * t2 ** 2 - 1j * t1 ** 2 * t2 ** 3
    cf[3] = np.log(t1 + 1j * t2)
    cf[4] = t1 * np.cos(t1) + t2 * np.sin(t2)
    cf[5] = t1 ** 2 * t2 - t1 * t2 ** 2
    cf[6] = 1j * t1 ** 3 + t2 ** 3
    cf[7] = (t1 + 1j * t2) ** 3 - t1 * t2
    cf[8] = t1 * t2 * (t1 - t2) * (t1 + t2)
    cf[9] = t1 ** 3 * t2 ** 2 * np.exp(1j * (t1 - t2))
    return cf.astype(np.complex128)

ALLOWED["p764"]=p764

def p765(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(11, dtype=np.complex128)
    cf[0] = t1 * t2
    cf[1] = 100 * np.real(t1 ** t2)
    cf[2] = 10 * np.real(t1 * t2 ** 3)
    cf[3] = 200 * np.imag(t1 ** t2)
    cf[4] = np.exp(1j * t1)
    cf[5] = np.sign(np.real(t1)) * np.abs(t2)
    cf[6] = (np.real(t1) ** 2 + np.imag(t2) ** 2) * np.exp(1j * np.angle(t1))
    cf[7] = np.where(np.real(t1) > np.imag(t2), 1 + 1j * t2, 100j + t1)
    cf[8] = t1 + 1j
    cf[9] = 10 * (0.1 + np.exp(1j * t2))
    cf[10] = 0.001 + 1j * t2 ** 3
    return cf.astype(np.complex128)

ALLOWED["p765"]=p765

def p766(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(11, dtype=np.complex128)

    # magnitudes and simple helpers
    r1 = math.hypot(t1.real, t1.imag)

    # cf[0] = 100 * exp(i*|t1|)
    cf[0] = np.complex128(100.0 * (math.cos(r1) + 1j * math.sin(r1)))

    # cf[1] = 10 * (Re(t1)^2 - Im(t2)^2)
    cf[1] = np.complex128(10.0 * (t1.real * t1.real - t2.imag * t2.imag))

    # cf[2] = i * (Re(t2)+Im(t1)) * (Re(t1)-Im(t2))
    cf[2] = 1j * np.complex128((t2.real + t1.imag) * (t1.real - t2.imag))

    # cf[3] = (t1+t2)^2 - (t1-t2)^2 = 4*t1*t2
    cf[3] = 4.0 * t1 * t2

    # cf[4] = 100*(t1+t2)/(1+|t1*t2|)
    prod = t1 * t2
    denom4 = 1.0 + math.hypot(prod.real, prod.imag)
    cf[4] = (100.0 * (t1 + t2)) / denom4

    # cf[5] = sum_{k=1..5} (Re(t1^k) * Im(t2^k))
    s = 0.0
    p1 = np.complex128(t1)  # t1^1
    p2 = np.complex128(t2)  # t2^1
    for k in range(1, 6):
        if k > 1:
            p1 *= t1
            p2 *= t2
        s += p1.real * p2.imag
    cf[5] = np.complex128(s)

    # cf[6] = sqrt(|t1^2 - t2^2|)
    diff = t1 * t1 - t2 * t2
    mag_diff = math.hypot(diff.real, diff.imag)
    cf[6] = np.complex128(math.sqrt(mag_diff))

    # cf[7] = t1*t2 / (1 + |t1 - t2|)
    d = t1 - t2
    denom7 = 1.0 + math.hypot(d.real, d.imag)
    cf[7] = (t1 * t2) / denom7

    # cf[8] = (integer products) + i*(integer products)
    # Implement math.prod over integer ranges manually (Numba-safe)
    a1 = int(math.floor(t1.real))
    b1 = int(math.floor(t2.imag))
    if a1 <= b1:
        p_real = 1
        for x in range(a1, b1 + 1):
            p_real *= x
    else:
        p_real = 1
        for x in range(a1, b1 - 1, -1):
            p_real *= x

    a2 = int(math.floor(t2.real))
    b2 = int(math.floor(t1.imag))
    if a2 <= b2:
        p_imag = 1
        for x in range(a2, b2 + 1):
            p_imag *= x
    else:
        p_imag = 1
        for x in range(a2, b2 - 1, -1):
            p_imag *= x

    cf[8] = np.complex128(float(p_real)) + 1j * np.complex128(float(p_imag))

    # cf[9] = exp(i*(Re(t1) - Im(t2)))
    ang9 = t1.real - t2.imag
    cf[9] = np.complex128(math.cos(ang9) + 1j * math.sin(ang9))

    # cf[10] = exp(-i*(Re(t2) - Im(t1)))
    ang10 = -(t2.real - t1.imag)
    cf[10] = np.complex128(math.cos(ang10) + 1j * math.sin(ang10))

    return cf

ALLOWED["p766"]=p766

def p767(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(11, dtype=np.complex128)
    cf[0] = np.real(t1) * np.imag(t2) + t1 * t2 + 1j
    cf[1] = 3j * t1 ** 2 + 2j * t2 ** 2 - 2 * t2 + 1j
    cf[2] = np.sin(t1 ** 3 + t2 ** 3) + np.cos(t1 * t2) + 2
    cf[3] = np.exp(1j * t1 * t2) + 2j * t1 - t2 + 2
    cf[4] = t1 ** 3 - 3 * t2 + np.imag(t1 + t2) ** 2 + 1j
    cf[5] = np.real(t1) * np.sin(t2) + t1 / (t2 + 1) + 2j
    cf[6] = np.cos(t1) / (1 + np.abs(t2)) + np.sin(t1 + t2) + 2
    cf[7] = np.exp(1j * t1) - np.exp(1j * t2) + np.sqrt(np.abs(t1 + t2)) + 1j
    cf[8] = np.abs(t1) * np.abs(t2) - np.sin(np.real(t1)) * np.cos(np.imag(t2)) + 1j
    cf[9] = t1 ** 2 * (t2 + 1) - (t1 + 1) * t2 ** 2 + 1j
    cf[10] = 1j * t2 + np.real(t1)
    return cf.astype(np.complex128)

ALLOWED["p767"]=p767

def p768(z,a,state):
    t1, t2 = z[0], z[1]
    n = 10
    cf = np.zeros(n, dtype=np.complex128)
    cf[0] = 100 * (t1 ** 4 - t2 ** 4) * np.exp(1j * (t1 - t2) ** 4)
    cf[1] = 50j * (t1 + t2 ** 3)
    cf[2] = 100 * (t1 - t2) * np.exp(-1j * t1 ** 2) + 50 * (t1 + t2) ** 2
    cf[3] = -25j * (t1 - 1j * t2) ** 3
    cf[4] = (t1 + 3j * t2) ** 4 + 50j * (t1 + 2j * t2) ** 2
    cf[5] = 75 * (t1 - 1j) ** 4 + 100 * (2j * t2 + t1) ** 2
    cf[6] = -100j * np.exp(1j * np.abs(t1 * t2))
    cf[7] = 50j * (t1 ** 2 + t2 ** 2) ** 2 + 25 * (t1 - 1j) ** 4
    cf[8] = 75 * np.exp(-1j * t1 ** 2) + 100j * t2 ** 4
    cf[9] = 100 * (t1 * t2) ** 4 - 25j * ((t1 + t2) ** 3 + t1 ** 2 * t2 ** 2)
    return cf.astype(np.complex128)

ALLOWED["p768"]=p768

def p769(z,a,state):
    t1, t2 = z[0], z[1]
    n = 10
    cf = np.zeros(n, dtype=np.complex128)
    cf[0] = 100 * np.sin(t1) ** 3 * np.cos(t2) ** 2
    cf[1] = 100 * np.exp(1j * (t1 + t2)) - 10 * (t1 - t2) ** 2
    cf[2] = t1 * t2 * (t1 - t2) / (np.abs(t1) + np.abs(t2) + 1)
    cf[4] = (t1 * t2 * np.exp(1j * (t1 ** 2 - t2 ** 2))) ** 3
    cf[6] = np.sqrt(np.abs(t1)) - np.sqrt(np.abs(t2)) + 1j * np.sin(t1 * t2)
    cf[7] = 50 * np.abs(t1 - t2) * np.exp(1j * np.abs(t1 + t2))
    cf[8] = np.where(np.imag(t1) > 0, t1 - np.abs(t2), t2 - np.abs(t1))
    cf[9] = (1j * t1 * t2) ** (0.1 * t1 * t2)
    return cf.astype(np.complex128)

ALLOWED["p769"]=p769

def p770(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(11, dtype=np.complex128)
    cf[0] = t1 * t2
    cf[1] = t1 ** 2 - t2 ** 2
    cf[2] = t1 ** 3 + t2 ** 3
    cf[3] = np.sin(t1) * np.cos(t2)
    cf[4] = np.exp(1j * (t1 - t2))
    cf[5] = np.log(np.abs(t1 + t2))
    cf[6] = t1 ** 4 + 1j * t2 ** 4
    cf[7] = (t1 * t2) ** 2
    cf[8] = (t1 + t2) / 2
    cf[9] = t1 ** 5 - t2 ** 5
    cf[10] = np.exp(1j * (t1 * t2))
    return cf.astype(np.complex128)

ALLOWED["p770"]=p770

def p771(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(11, dtype=np.complex128)
    cf[0] = 100 * t1 ** 3 + 200j * t2 ** 2 - 100 * t1 - 55j
    cf[1] = 1j * (t1 ** 2 + t2 ** 2) ** 3 + 2j * t2 ** 5
    cf[2] = (t1 + t2) ** 3 - (t1 - t2) ** 2 + np.sin(t1 + t2)
    cf[3] = np.exp(1j * t1 ** 2 * t2 ** 3)
    cf[4] = np.log(1 + np.abs(t1 * t2)) - 50 * t1 * t2
    cf[5] = np.abs(t1 - t2) * (t1 + t2) * np.log(1 + np.abs(t1 + t2))
    cf[6] = np.exp(-1j * t1 ** 2) * (t1 + t2) ** 3 - (t1 - t2) ** 3
    cf[7] = 100 * t2 ** 3 + 200j * t1 ** 2 - 100 * t2 - 55j
    cf[8] = (t1 ** 2 + t2 ** 2) * np.sin(t1 * t2)
    cf[9] = 2 * (np.abs(t1) + np.abs(t2)) ** 4 + 1j * (t1 ** 2 - t2 ** 2)
    cf[10] = 50 * (t1 + t2) ** 2 + 50 * t1 * t2 - 50j * (t1 - t2)
    return cf.astype(np.complex128)

ALLOWED["p771"]=p771

def p772(z, a, state):
    t1, t2 = z[0], z[1]
    m = 10
    cf = np.zeros(m + 1, dtype=np.complex128)

    # Scalar assignments (fancy indexing isn't reliable in nopython mode)
    cf[0] = t1
    cf[m] = t2
    cf[1] = 1j * t1
    cf[m - 1] = -1j * t2

    # cos(t1*t2) for complex argument:
    # cos(x + i y) = cos x * cosh y - i sin x * sinh y
    prod = t1 * t2
    ax = prod.real
    by = prod.imag
    cos_prod = np.complex128(
        math.cos(ax) * math.cosh(by) - 1j * (math.sin(ax) * math.sinh(by))
    )

    # Fill inner coefficients:
    # cf[x+2] = |cf[x+1]*cf[m-x+1]| * (cos_prod)^(x+1)
    # Use iterative power to avoid ** on complex in loops
    # We need exponent p = x+1 for x=1..m-2 -> p = 2..(m-1)
    # Initialize cpow = (cos_prod)^2
    cpow = cos_prod * cos_prod
    for x in range(1, m - 1):
        # |complex| via hypot
        u = cf[x + 1] * cf[m - x + 1]
        mag = math.hypot(u.real, u.imag)

        # Set value
        cf[x + 2] = np.complex128(mag) * cpow

        # next exponent: multiply by cos_prod to increment power by 1
        cpow = cpow * cos_prod

    # Multiply by exp(i * theta_i) where theta_i = 2π*i/m
    for i in range(m + 1):
        theta = (2.0 * math.pi * i) / m
        rot = np.complex128(math.cos(theta) + 1j * math.sin(theta))
        cf[i] = cf[i] * rot

    return cf

ALLOWED["p772"]=p772

def p773(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(11, dtype=np.complex128)
    cf[0] = np.sin(t1) + 1j * np.cos(t2)
    cf[1] = -1j * t1 + t2
    cf[2] = np.exp(1j * (t1 - t2)) + np.log(1 + np.abs(t1) + np.abs(t2))
    cf[3] = (t1 * t2) ** 2 - 1j * (t1 - t2) ** 2
    cf[4] = 100j * np.sin(t2) + 100 * np.cos(t1)
    cf[5] = (t1 + t2) ** 3 - 1j * (t1 - t2) ** 3
    cf[6] = np.exp(1j * (t1 * t2))
    cf[7] = np.log(1 + np.abs(t1 * t2)) + 1j * (t1 - t2)
    cf[8] = (t1 ** 2 + t2 ** 2)
    cf[9] = (t1 + 1j * t2) ** 3
    cf[10] = np.prod(np.cos(np.linspace(1, 10, num=100)) + 1j * np.sin(np.linspace(1, 10, num=100))) * t1 * t2
    return cf.astype(np.complex128)

ALLOWED["p773"]=p773

def p774(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(11, dtype=np.complex128)
    complex_sum = t1 + t2
    complex_product = t1 * t2
    complex_diff = t1 - t2
    cf[0] = np.exp(1j * np.angle(complex_sum)) * np.abs(complex_sum)
    cf[1] = np.real(complex_product) + 1j * np.imag(complex_diff)
    cf[2] = np.abs(complex_product)
    cf[3] = cf[0] + cf[1] + cf[2]
    cf[4] = cf[0] * cf[2] - cf[1] * cf[3]
    cf[5] = 10 + np.sin(cf[4]) + np.sin(cf[3]) + np.cos(cf[2]) + np.cos(cf[1]) + np.sin(cf[0])
    cf[6] = 2j * cf[0] + cf[1] / cf[5]
    cf[7] = cf[1] * cf[2] * cf[3] + cf[0]
    cf[8] = 2 * cf[7] - cf[4] * cf[6]
    cf[9] = cf[0] * cf[1] * cf[2] * cf[3] * cf[4] * cf[5] * cf[6] * cf[7] * cf[8]
    cf[10] = (cf[0] + cf[2] - cf[4] + cf[6] - cf[8]) / (cf[1] - cf[3] + cf[5] - cf[7] + cf[9])
    return cf.astype(np.complex128)

ALLOWED["p774"]=p774

def p775(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(11, dtype=np.complex128)
    cf[0] = t1 * t2
    cf[1] = (t1 ** 3 + t2 ** 2) * 1j
    cf[2] = t1 ** 2 * np.cos(np.abs(t2))
    cf[3] = t2 ** 3 * np.sin(np.angle(t1))
    cf[4] = np.real(np.exp(1j * t1) * np.exp(-1j * t2))
    cf[5] = np.imag((t1 + t2) ** 2 * 1j)
    cf[6] = np.abs(t1 - t2) * np.cos(t1 + t2)
    cf[7] = np.angle(t1 * t2) * np.sin(t1 * t2) * 1j
    cf[8] = (t1 + t2) ** 3 - np.real(t1 ** 2 * t2 ** 3 * 1j)
    cf[9] = np.abs(t1 ** 2 - t2 ** 2) * np.exp(1j * np.angle(t1 + t2))
    cf[10] = (np.real(t1) + np.imag(t2)) ** 2 * np.sin(np.abs(np.real(t1) - np.imag(t2)))
    return cf.astype(np.complex128)

ALLOWED["p775"]=p775

def p776(z,a,state):
    t1, t2 = z[0], z[1]
    n = 10
    cf = np.zeros(n, dtype=np.complex128)
    cf[0] = 1 + np.sin(t1 + t2)
    cf[1] = t1 * t2 * np.exp(-1j * t2)
    cf[2] = 100 * np.where(np.abs(t1 - t2) < 1, t1, t2)
    cf[3] = 200 * (np.real(t1) - np.imag(t1))
    cf[4] = -100j * t2 + 100 * t1 ** 2
    cf[5] = np.tan(t1 / (1 + np.abs(t2)))
    cf[6] = 100j * np.real(t1 * t2 * np.exp(1j * (t1 - t2)))
    cf[7] = -100 * np.sin(t2) ** 3 + t1 ** 2 * np.cos(t2) * np.sin(t1)
    cf[8] = (t1 + t2) ** 4 - (t1 - t2) ** 4
    cf[9] = cf[1] * cf[7] - cf[0] * cf[8]
    return cf.astype(np.complex128)

ALLOWED["p776"]=p776

def p777(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[1] = t1 + t2
    for k in range(2, 26):
        v = np.exp(1j * np.angle(cf[k - 1] + t2)) * np.abs(k * cf[k - 1] + t1)
        cf[k] = v
    cf[9] = np.real(t1) + np.imag(t2)
    cf[14] = 1j * (np.real(t1) + np.imag(t2))
    cf[19] = np.real(t1 * t2) * (1 + 1j)
    cf[24] = np.abs(t1) ** 2 + np.abs(t2) ** 2
    return cf.astype(np.complex128)

ALLOWED["p777"]=p777

def p778(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = np.abs(t1 + t2)
    cf[1] = 2 * np.real(t1) * np.imag(t2)
    cf[2] = np.angle(t1 + t2)
    cf[3] = np.conj(t1) * t2
    cf[4] = np.angle(t1) * np.angle(t2)
    for k in range(5, 22):
        cf[k] = np.abs(t1 + (-1) ** k * t1 ** 2 / (k + 1) + (-1) ** k * t2 ** 2 / (k + 1))
    cf[21] = cf[1] + cf[2] - cf[3] + cf[4]
    cf[22] = np.abs(cf[1] * cf[2] * cf[3] * cf[4])
    cf[23] = 1 + np.real(np.conj(t1) * t2)
    cf[24] = 1j + np.imag(np.conj(t1) * t2)
    return cf.astype(np.complex128)

ALLOWED["p778"]=p778

def p779(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    for k in range(25):
        cf[k] = np.abs(t1) ** (k + 1) + np.angle((t2 + 1j) ** (k + 1)) + np.log(np.abs(np.sin(k * t1) * np.cos(k * t2)))
    cf[4] = t1 ** t2 + np.conj(t2) ** 3
    cf[14] = np.cos(np.real(t1)) * np.sin(np.imag(t2)) + np.log(np.abs(1j * t2))
    cf[24] = np.real(np.conj(t2) ** t1) - np.imag(1j * t1 ** 3)
    return cf.astype(np.complex128)

ALLOWED["p779"]=p779

def p780(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)

    # seed
    cf[0] = t1 * t1 + t2 * t2 - t1 * t2

    # recurrence: cf[k] = cf[k-1] * (t1+t2) / (1 + |cf[k-1]|)
    s = t1 + t2
    for k in range(1, 25):
        prev = cf[k - 1]
        denom = 1.0 + math.hypot(prev.real, prev.imag)  # |prev|
        cf[k] = (prev * s) / denom

    # scatter adds: replace fancy indexing with loops
    plus_idx = (2, 5, 8, 11, 14, 17, 20, 23)
    add_val = t1 + 1j * t2
    for i in plus_idx:
        cf[i] = cf[i] + add_val

    minus_idx = (1, 4, 7, 10, 13, 16, 19, 22, 24)
    sub_val = t2 + 1j * t1
    for i in minus_idx:
        cf[i] = cf[i] - sub_val

    return cf

ALLOWED["p780"]=p780

def p781(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = t1 + t2
    cf[1] = t1 ** 2 + t2 ** 2
    for k in range(2, 13):
        cf[k] = (cf[k - 1] * cf[k - 2]) / (1 + np.abs(t1) + np.abs(t2))
    for l in range(13, 25):
        cf[l] = (cf[l - 1] + cf[l - 13]) / (1 + np.real(t1) ** 2 + np.imag(t2) ** 2)
    return cf.astype(np.complex128)

ALLOWED["p781"]=p781
        
def p781v1(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = t1 + t2
    cf[1] = t1 ** 2 + t2 ** 2
    for k in range(3, 14):
        cf[k] = (cf[k - 1] * cf[k - 2]) / (1 + np.abs(t1) + np.abs(t2))
    for l in range(14, 25):
        cf[l] = (cf[l - 1] + cf[l - 13]) / (1 + np.real(t1) ** 2 + np.imag(t2) ** 2)
    return cf.astype(np.complex128)

ALLOWED["p781v1"]=p781v1
        
def p782(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = t1 + t2
    cf[1] = np.exp(1j * np.angle(t1 * np.conj(t2)))
    cf[2] = np.abs(t1) * np.abs(t2)
    for k in range(3, 26):
        cf[k] = (np.real(cf[k - 1]) + 1j * np.imag(cf[k - 1])) * np.exp(1j * np.angle(cf[k - 2]))
        if np.imag(cf[k]) == 0:
            cf[k] += 1e-10
        cf[k] = np.log(np.abs(cf[k])) / 2 + cf[k] * 1j
    return cf.astype(np.complex128)

ALLOWED["p782"]=p782

def p783(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = np.abs(t1 * t2) * np.exp(1j * np.angle(t1 + t2))
    cf[1] = np.angle(t1 * t2) * np.exp(1j * np.abs(t1 - t2))
    for k in range(2, 25):
        cf[k] = np.abs(t1 + t2 * 1j ** (k + 1)) * np.exp(1j * np.angle(cf[k - 1] + t1 * cf[k - 2]))
    return cf.astype(np.complex128)

ALLOWED["p783"]=p783

def p784(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = np.real(t1 + t2) + np.conj(t1 * t2)
    for k in range(1, 25):
        n = np.abs(t1 * cf[k - 1])
        if n != 0:
            cf[k] = np.sin(np.angle(t1)) * np.log(np.abs(n)) + np.cos(np.angle(t2)) * np.log(np.abs(1 / n))
        else:
            cf[k] = 0
    return cf.astype(np.complex128)

ALLOWED["p784"]=p784

def p785(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[:5] = np.array([1, t1, t2, t1 * t2, np.abs(t1 + t2)]) ** 2
    for k in range(5, 25):
        cf[k] = 3 * cf[k - 1] + 2 * cf[k - 5] + 5 * (k + 1)
    cf[9:15] = np.conj(cf[:6])
    cf[19:25] = np.exp(1j * np.angle(cf[:6]))
    return cf.astype(np.complex128)

ALLOWED["p785"]=p785

def p786(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    for k in range(25):
        cf[k] = np.abs(t1) ** (k / 2) * (np.cos(k * np.angle(t2)) + 1j * np.sin(k * np.angle(t2)))
    cf[4] += (np.log(np.abs(t1)) + np.log(np.abs(t2))) / 2
    cf[9] += np.conj(t1 * t2)
    cf[14] += np.abs(t2 - t1) ** 2
    cf[19] += (np.sin(np.angle(t1)) / np.cos(np.angle(t2))) ** 3
    cf[24] += ((1j * t1 - t2) ** 2 / (1 + np.abs(t1 + t2) ** 3)) ** 4
    return cf.astype(np.complex128)

ALLOWED["p786"]=p786

def p787(z, a, state):
    t1 = z[0]
    t2 = z[1]
    w  = t1 - t2  # complex exponent used in the loop

    cf = np.zeros(25, dtype=np.complex128)

    # cf[0] = Re(t1 + t2) + i * Im(t1 * t2)
    cf[0] = (t1 + t2).real + 1j * (t1 * t2).imag

    for i in range(1, 24):
        prev = cf[i - 1]
        re = prev.real
        im = prev.imag

        # angle(prev)
        theta = np.arctan2(im, re)

        # r = |prev|
        r = np.hypot(re, im)

        # Compute |prev**w| safely:
        # If r==0 → magnitude is 0
        if r == 0.0:
            mag_pow = 0.0
        else:
            # ln(prev) = ln r + i*theta
            ln_r = np.log(r)
            # x = Re(w * (ln r + i*theta)) = Re(w)*ln r - Im(w)*theta
            x = w.real * ln_r - w.imag * theta

            # Clamp x to avoid overflow in exp(x)
            if x > 700.0:
                mag_pow = np.inf
            elif x < -700.0:
                mag_pow = 0.0
            else:
                mag_pow = np.exp(x)

        val_real = mag_pow
        val_imag = theta

        # Guard finiteness
        if (not np.isfinite(val_real)) or (not np.isfinite(val_imag)):
            cf[i] = 0.0 + 1.0j
        else:
            cf[i] = val_real + 1j * val_imag

    # Tail expression; avoid np.conj to be extra safe in nopython
    cf[24] = cf[0]**3 + cf[23]**2 + cf[22] - cf[21] + cf[20].conjugate()

    return cf

ALLOWED["p787"]=p787

def p788(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = (t1 + 1j * t2)
    cf[1] = 2 * np.real((t1 + 1j * t2)) ** 2
    cf[2] = 3 * np.imag((t1 + 1j * t2)) ** 3
    for k in range(3, 15):
        cf[k] = np.abs((cf[k - 1] ** (k + 1) + cf[k - 2] ** k)) / (k ** 2 + 1) + (np.angle(t1) + np.angle(t2))
    cf[15:20] = cf[10:15] + cf[:5]
    cf[20] = np.abs(t1) ** 2 - np.abs(t2) ** 2
    cf[21] = np.angle(t1) + np.angle(t2)
    cf[22] = np.real(t1 ** 3 - t2 ** 3)
    cf[23] = np.imag(t1 * t2 * (t1 - t2))
    cf[24] = np.abs(t1 * t2 * (t1 - t2)) ** 0.5
    return cf.astype(np.complex128)

ALLOWED["p788"]=p788

def p789(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[:5] = np.real(t1) * np.arange(1, 6) - np.imag(t2) * np.arange(1, 6)
    cf[5] = (np.abs(t1) * np.abs(t2))
    cf[6:11] = np.angle(t1 + t2) * np.arange(6, 11)
    cf[12] = np.conj(t1) + np.conj(t2)
    cf[13:18] = np.real(t1 + 1j * t2) * np.arange(1, 6)
    cf[18] = (np.angle(t1) * np.angle(t2))
    cf[19:24] = np.imag(t1 - 1j * t2) * np.arange(1, 6)
    cf[24] = np.conj(t1 * t2)
    cf[25] = np.abs(cf[12]) + np.angle(cf[18])
    return cf.astype(np.complex128)

ALLOWED["p789"]=p789

def p790(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = t1 + t2
    for k in range(1, 25):
        v = np.sin(k * cf[k - 1] + np.angle(t2 ** k)) + np.cos(k * np.abs(t1))
        cf[k] = v / (np.abs(v) + 1e-10)
    cf[9] = t1 * t2 - np.abs(t2) ** 2 + 1j * np.angle(t1)
    cf[14] = np.conj(t1) ** 3 - np.angle(t2) ** 3 + 1j * np.abs(t2)
    cf[19] = np.abs(t2) ** 3 + t1 ** 2 + t2 ** 2 + 1j * np.angle(t2) ** 2
    cf[24] = np.abs(t1 * t2) + np.angle(t1) ** 5 + 1j * np.abs(t1) ** 5
    return cf.astype(np.complex128)

ALLOWED["p790"]=p790

def p791(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[:10] = np.arange(1, 11) + t1 ** 2 + t2 ** 2
    for i in range(10, 20):
        cf[i] = 2 * cf[i - 1] * np.sin(t1 + t2)
    cf[20:22] = np.array([np.sum(cf[:20]), np.prod(cf[:20])]) * (t1 + t2)
    cf[22] = np.log(np.abs(t1 * t2)) + np.angle(t1 ** 2 + t2 ** 2) * cf[21]
    cf[23] = np.abs(t1 - t2) / (np.abs(cf[22]) + 1)
    cf[24] = np.conj(cf[23] * cf[21])
    return cf.astype(np.complex128)

ALLOWED["p791"]=p791

def p792(z, a, state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)

    # ===== numerics guards =====
    EXP_X_MAX   = 700.0   # exp(>~709) overflows in float64
    COSH_Y_MAX  = 20.0    # cosh/sinh explode beyond ~20
    def finite2(x, y):
        return math.isfinite(x) and math.isfinite(y)

    # log|z| with guard
    def logabs(zc):
        r = math.hypot(zc.real, zc.imag)
        if r <= 0.0 or not math.isfinite(r):
            return None
        return math.log(r)

    # sin(x+iy) with guard
    def sin_c(zc):
        x, y = zc.real, zc.imag
        if abs(y) > COSH_Y_MAX:
            return None
        sx = math.sin(x); cx = math.cos(x)
        ch = math.cosh(y); sh = math.sinh(y)
        re = sx * ch
        im = cx * sh
        if not finite2(re, im):
            return None
        return np.complex128(re + 1j * im)

    # cos(x+iy) with guard
    def cos_c(zc):
        x, y = zc.real, zc.imag
        if abs(y) > COSH_Y_MAX:
            return None
        cx = math.cos(x); sx = math.sin(x)
        ch = math.cosh(y); sh = math.sinh(y)
        re = cx * ch
        im = -sx * sh
        if not finite2(re, im):
            return None
        return np.complex128(re + 1j * im)

    # exp(x+iy) with guard
    def exp_c(zc):
        x, y = zc.real, zc.imag
        if x > EXP_X_MAX or x < -EXP_X_MAX:
            return None
        ex = math.exp(x)
        re = ex * math.cos(y)
        im = ex * math.sin(y)
        if not finite2(re, im):
            return None
        return np.complex128(re + 1j * im)

    # ----- cf[0] = sin(t1+t2) + i*cos(t1-t2) -----
    s = t1 + t2
    d = t1 - t2
    s_val = sin_c(s)
    if s_val is None:
        return cf  # zero vector on error
    d_val = cos_c(d)
    if d_val is None:
        return cf
    cf[0] = s_val + 1j * d_val

    # real offset: log|t1| - log|t2|
    la1 = logabs(t1)
    if la1 is None:
        return cf
    la2 = logabs(t2)
    if la2 is None:
        return cf
    off = la1 - la2  # real scalar

    # ----- recurrence -----
    for k in range(1, 25):
        prev = cf[k - 1]
        w = np.complex128(float(k) * prev)
        ev = exp_c(w)
        if ev is None:
            return cf
        v = ev + off  # off is real; promotes safely to complex
        # final finite check
        if not finite2(v.real, v.imag):
            return cf
        cf[k] = v

    return cf

ALLOWED["p792"]=p792

def p793(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = t1 + t2
    cf[1] = np.abs(t1) * t2
    cf[2] = np.conj(t1) + np.real(t2)
    cf[3] = np.abs(t1) * np.imag(t2)
    cf[4] = np.angle(t1) * np.conj(t2)
    for k in range(5, 26):
        cf[k] = np.abs(cf[k - 1] * t1) + np.angle(cf[k - 2] * t2)
    return cf.astype(np.complex128)

ALLOWED["p793"]=p793

def p794(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    for k in range(25):
        cf[k] = np.abs(t1) ** k + np.angle(t2) ** k + t1 * (1j ** k) + np.conj(t2) ** ((k + 1) / 2)
    cf[9] += np.log(np.abs(t1 * t2))
    if not np.isinf(cf[9]) and not np.isnan(cf[9]):
        cf[9] /= np.abs(t1 + t2 + 1j)
    cf[19] += (t1 + 1j * t2) ** 2
    if not np.isinf(cf[19]) and not np.isnan(cf[19]):
        cf[19] /= np.abs(t1 + t2 + 1j) ** 2
    cf[24] = (cf[4] ** 2 + cf[5] ** 2 + cf[6] ** 2) ** 0.5
    return cf.astype(np.complex128)

ALLOWED["p794"]=p794

def p795(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    for i in range(25):
        cf[i] = (i ** 2 + t1) * np.exp((1 + 0j) * i * t2) / (1 + np.abs(t1 * t2))
    cf[4:15] = np.real(cf[4:15]) * np.cos(np.imag(cf[4:15]))
    cf[16:25] = np.abs(cf[16:25]) * np.exp(1j * np.angle(t1 + t2))
    cf[2] = (t1 + 1j * t2) ** 3 - np.conj(t1 + 1j * t2) ** 3
    cf[6] = cf[22] = (np.abs(t1) ** 3 + np.abs(t2) ** 3) * np.exp(1j * np.angle(t2 - t1))
    cf[18] = np.where(np.abs(t2) > 1, np.log(np.abs(t2)), 0)
    return cf.astype(np.complex128)

ALLOWED["p795"]=p795

def p796(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    for k in range(25):
        phase = np.angle(t1) * np.angle(t2)
        modulus = np.abs(t1 + t2)
        cf[k] = modulus ** (k + 1) * np.exp(1j * phase / (k + 1))
    cf[0] = t1 ** 5 + t2 ** 5
    cf[2] = np.real(t1) + np.imag(t2)
    cf[4] = t1 * t2 * (1 + 1j)
    cf[14] = np.abs(t1 + t2 + 1j) ** 3
    cf[19] = np.conj(t1) + np.real(t2)
    cf[24] = (np.abs(t1) ** 2 + np.abs(t2) ** 2) ** 0.5
    return cf.astype(np.complex128)

ALLOWED["p796"]=p796

def p797(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = 1 + t1 * t2
    cf[1] = -(t1 + t2) - 1j
    for k in range(2, 25):
        cf[k] = cf[k - 1] * cf[1] / cf[k - 2]
        cf[k] += t2 ** (k + 1) + 1j * t1 ** (k + 1)
        cf[k] *= np.exp(1j * np.angle(cf[k - 2]))
        cf[k] /= np.abs(cf[k])
    return cf.astype(np.complex128)

ALLOWED["p797"]=p797

def p798(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[1] = np.abs(t1) * np.sin(np.angle(t2))
    cf[2] = np.abs(t2) * np.cos(np.angle(t1))
    for k in range(3, 26):
        cf[k] = np.abs(cf[k - 1]) * np.sin(np.angle(cf[k - 2])) + t1
        if np.abs(cf[k]) > 10000:
            cf[k] /= np.abs(cf[k])
    cf[0] = cf[24]
    return cf.astype(np.complex128)

ALLOWED["p798"]=p798

def p799(z, a, state):
    t1 = z[0]
    t2 = z[1]
    cf = np.zeros(25, dtype=np.complex128)

    # Precompute basics
    re1, im1 = t1.real, t1.imag
    re2, im2 = t2.real, t2.imag
    prod = t1 * t2
    re_p, im_p = prod.real, prod.imag

    abs_t1 = np.hypot(re1, im1)
    abs_t2 = np.hypot(re2, im2)
    ang_p  = np.arctan2(im_p, re_p)

    # Powers of |t1|, |t2| built iteratively (stable, njit-friendly)
    p1 = abs_t1
    p2 = abs_t2
    for i in range(25):
        term1 = (2.0 * re_p + 3.0 * im_p) * p1
        term2 = (2.0 * ang_p - 3.0 * abs_t2) * p2
        val_real = term1 + term2

        # store as complex
        cf[i] = val_real + 0.0j

        # finite guard on real/imag separately
        if not (np.isfinite(cf[i].real) and np.isfinite(cf[i].imag)):
            cf[i] = 0.0 + 0.0j

        # prepare next powers
        p1 *= abs_t1
        p2 *= abs_t2

    # cf[0] += conj(t1*t2)
    cf0 = cf[0] + prod.conjugate()
    if np.isfinite(cf0.real) and np.isfinite(cf0.imag):
        cf[0] = cf0
    else:
        cf[0] = 0.0 + 0.0j

    # cf[12] *= log(|t1 + t2|)
    sum_re, sum_im = re1 + re2, im1 + im2
    abs_sum = np.hypot(sum_re, sum_im)
    if abs_sum > 0.0:
        scale = np.log(abs_sum)
        tmp = cf[12] * scale
        if np.isfinite(tmp.real) and np.isfinite(tmp.imag):
            cf[12] = tmp
        else:
            cf[12] = 0.0 + 0.0j
    else:
        cf[12] = 0.0 + 0.0j

    # cf[24] -= cf[24] * t1*t2 / (t1 + t2)
    denom = t1 + t2
    if np.hypot(denom.real, denom.imag) > 1e-12:
        correction = cf[24] * prod / denom
        tmp = cf[24] - correction
        if np.isfinite(tmp.real) and np.isfinite(tmp.imag):
            cf[24] = tmp
        else:
            cf[24] = 0.0 + 0.0j
    else:
        cf[24] = 0.0 + 0.0j

    return cf

ALLOWED["p799"]=p799

def p800(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = t1 * t2
    cf[1:10] = np.abs(t1) * np.angle(1j * t2) ** np.arange(1, 10)
    cf[14] = np.real(t1) * np.imag(t2) + np.real(t2) * np.imag(t1)
    cf[15:20] = np.abs(t1 - t2 + 1j) ** np.arange(1, 6)
    cf[20] = np.imag(t1) * np.real(t2) + np.real(t1) * np.imag(t2)
    cf[21:24] = np.abs(cf[14] + t1 * t2) ** np.arange(1, 4)
    cf[24] = np.abs(cf[23]) + np.log(np.abs(t1 + t2 + 0.5j))
    return cf.astype(np.complex128)

ALLOWED["p800"]=p800

def p801(z,a,state):
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
    cf[10:15] = np.abs(t1) ** (np.arange(11, 16) / 10) * np.abs(t2) ** (np.arange(15, 10, -1) / 10)
    cf[15:20] = (np.arange(16, 21) * (np.angle(t1) + np.angle(t2))) / 2
    cf[20:25] = np.real(t1) ** 2 + np.imag(t2) ** 2 + np.arange(21, 26)
    return cf.astype(np.complex128)

ALLOWED["p801"]=p801

def p802(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0:3] = [2*t1 + 3*t2, 3*t1 - 2*t2, t1**2 - t2**2]
    cf[3:5] = [np.real(t1 * t2), np.imag(t1 * t2)]
    for k in range(5, 20):  # indices 5 to 19 inclusive (6:20 in R)
        cf[k] = np.sin(cf[k-1]) + np.cos(cf[k-2])
        mod_cf = np.abs(cf[k])
        if mod_cf != 0:
            cf[k] = cf[k] / mod_cf
        else:
            cf[k] = 1
    return cf.astype(np.complex128)

ALLOWED["p802"]=p802

def p803(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = t1 + t2
    cf[1] = np.abs(t1) * np.abs(t2)
    cf[2] = np.real(t1 * t2)
    cf[3] = np.imag(t1 * t2)
    cf[4] = np.angle(t1 + t2)
    for k in range(5, 26):
        cf[k] = cf[k - 1] + np.cos(k * t1) + np.sin(k * np.abs(t2)) * 1j
    return cf.astype(np.complex128)

ALLOWED["p803"]=p803

def p804(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0:10] = np.arange(1, 11) * t1 + np.arange(11, 21) * 1j * t2
    cf[10:20] = (t1 + 1j * t2) ** 2 * np.arange(11, 21)
    cf[20:25] = (np.abs(t1) + np.angle(t2)) * np.arange(1, 6)
    return cf.astype(np.complex128)

ALLOWED["p804"]=p804

def p805(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = t1 ** 2 + t2 ** 2
    cf[1] = t1 * t2
    cf[2:8] = t1 * t2 ** np.arange(1, 7)
    cf[8:14] = (t1 ** 2 + t2 ** 2) / np.arange(1, 7)
    cf[14:20] = t1 ** 3 * t2 ** np.arange(1, 7)
    cf[20:25] = (t1 + t2) ** np.arange(1, 6)
    return cf.astype(np.complex128)

ALLOWED["p805"]=p805

def p806(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cylotomic_coefs = np.array([np.exp(2j * np.pi / 3), np.exp(-2j * np.pi / 3)])
    for k in range(25):
        cf[k] = np.sin(k * t1) * cylotomic_coefs[0] + np.cos(k * t2) * cylotomic_coefs[1]
        if np.imag(cf[k]) == 0:
            cf[k] /= np.real(cf[k])
        else:
            cf[k] /= np.imag(cf[k])
    cf[7] = t1 ** 2 + t2 ** 2
    cf[24] = np.abs(t1 + t2)
    return cf.astype(np.complex128)

ALLOWED["p806"]=p806

def p807(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = t1 * t2
    cf[1] = t1**2 + 2 * t2**2
    cf[2] = t1**4 - t2**4
    for k in range(3, 14):  # indices shifted by -1
        cf[k] = (t1 * t2) / (k + 1) + (1j * cf[k - 1])
    cf[14] = cf[13].real + cf[9].imag
    for k in range(15, 25):
        arg_val = np.angle(cf[k - 1] + 1j * t2**2)
        cf[k] = np.floor(arg_val)
    return cf.astype(np.complex128)
    
ALLOWED["p807"]=p807

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

def p809(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = t1 + t2
    for k in range(1, 25):
        cf[k] = np.cos(k * np.angle(cf[k - 1])) + np.sin(k * np.abs(t1)) + np.conj(t2) / np.abs(1 + t1)
        if np.isinf(cf[k]) or np.isnan(cf[k]):
            cf[k] = cf[k - 1]
    cf[9] = cf[0] ** 3 + cf[1] ** 2 - cf[0] * cf[1]
    if np.isinf(cf[9]) or np.isnan(cf[9]):
        cf[9] = cf[8]
    cf[14] = np.log(np.abs(cf[13])) - t1 ** 2 + t2 ** 2
    if np.isinf(cf[14]) or np.isnan(cf[14]):
        cf[14] = cf[13]
    cf[19] = cf[0] * (t1 + t2) ** 2 - cf[2] / (1 + np.abs(t1 * t2))
    if np.isinf(cf[19]) or np.isnan(cf[19]):
        cf[19] = cf[18]
    cf[24] = (t1 + t2) ** 3 - cf[23]
    if np.isinf(cf[24]) or np.isnan(cf[24]):
        cf[24] = cf[23]
    return cf.astype(np.complex128)

ALLOWED["p809"]=p809

def p810(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    for i in range(25):
        if i % 2 == 1:
            cf[i] = ((i * t1 + 3 * i * t2) / (i + 1) ** 2) ** i
        else:
            cf[i] = (t1 + np.conj(t2)) ** i
    cf[cf == np.inf] = 1e10
    cf[cf == -np.inf] = -1e10
    cf[np.isnan(cf)] = 0
    return cf.astype(np.complex128)

ALLOWED["p810"]=p810

def p811(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = t1 * t2
    cf[1] = (t1 + t2) * np.conj(t1 - t2)
    for k in range(2, 26):
        cf[k] = np.abs(t1) * np.abs(t2) * np.sin(np.angle(t1 + 1j * t2) ** k) + np.log(np.abs(t1 ** k / (1 + t2)))
    return cf.astype(np.complex128)

ALLOWED["p811"]=p811

def p812(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = t1 * t2
    for k in range(1, 26):
        cf[k] = (cf[k - 1] ** 2 + np.real(t1) + np.imag(1j * t2)) / (1 + np.abs(cf[k - 1]))
        if np.abs(cf[k]) > 1e6 or np.isnan(cf[k]) or np.isinf(cf[k]):
            cf[k] = cf[k - 1]
    return cf.astype(np.complex128)

ALLOWED["p812"]=p812

def p813(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = t1 * t2
    for k in range(1, 26):
        v = (np.sin(k * cf[k - 1]) + np.cos(k * cf[k - 1])) * np.real(t1 + t2)
        if np.abs(v) != 0:
            cf[k] = np.log(np.abs(v)) + np.conj(t1 * t2)
        else:
            cf[k] = 0
    cf[24] = np.sum(cf[:24]) + np.abs(t1 - t2)
    return cf.astype(np.complex128)

ALLOWED["p813"]=p813

def p814(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = np.angle(t1) * t2
    cf[1] = t1 * t2
    for i in range(2, 26):
        v = cf[i - 1] + cf[i - 2] + np.conj(t1 * t2)
        if np.abs(v) != 0:
            cf[i] = np.log(np.abs(v))
        else:
            cf[i] = 0
    cf[24] = np.abs(t1 - t2) + np.sum(cf[:24])
    return cf.astype(np.complex128)

ALLOWED["p814"]=p814

def p815(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    for k in range(1, 26):
        cf[k-1] = np.sin(k)*t1/(1+abs(t2)) + np.cos(k)*t2/(1+abs(t1)) + np.sqrt(k)
    
    cf[0] = np.abs(t1)*abs(t2)
    cf[4] = np.angle(t1)*abs(t2)
    cf[9] = np.abs(t1)*np.angle(t2)
    cf[14] = np.abs(t1)*t2.real
    cf[19] = np.abs(t1)*t2.imag
    cf[24] = t1.real*abs(t2)
    return cf.astype(np.complex128)

ALLOWED["p815"]=p815

def p816(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = t1*t2
    cf[1] = np.abs(t1)**2 + np.abs(t2)**2
    for k in range(2, 24):
        cf[k] = (cf[k-1] / cf[k-2]) + np.conj(t1) - t2
        if np.isnan(cf[k]) or np.isinf(cf[k]):
            cf[k] = 0
    cf[24] = cf[23] + cf[22] - np.conj(t1 + t2)
    return cf.astype(np.complex128)

ALLOWED["p816"]=p816

def p817(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = (t1**3).real*(t2**3).real
    cf[1] = (t1**3).imag*(t2**3).imag
    for k in range(2, 25):
        if (k+1)%3 == 0:
            cf[k] = (t1+1j*t2)**((k+1)/3) / (k+1)
        else:
            cf[k] = np.conj(cf[k-1])**2 + np.abs(t1)*abs(t2)
    return cf.astype(np.complex128)

ALLOWED["p817"]=p817

def p818(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    for k in range(1, 26):
        cf[k-1] = (np.abs(t1 + 1j*t2 + 0.5 + k/25)*np.cos(np.angle(t1 + 1j*t2)**(k-1)) + 
                    1j*abs(t2 + 1j*t1 + 0.5 + k/25)*np.sin(np.angle(t2 + 1j*t1)**(k-1)))
        if np.isnan(cf[k-1]) or np.isinf(cf[k-1]):
            cf[k-1] = 0
    return cf.astype(np.complex128)

ALLOWED["p818"]=p818

def p819(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = t1*t2
    for i in range(1, 25):
        cf[i] = (cf[i-1].real**2 - cf[i-1].imag**2)*t1*t2 + 1j*2*cf[i-1].real*cf[i-1].imag
        if np.isnan(cf[i]):
            cf[i] = 1.0
        if np.abs(cf[i]) < 1e-10:
            cf[i] = 1
    cf[24] = cf[23] + t1 + t2
    return cf.astype(np.complex128)

ALLOWED["p819"]=p819

def p820(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0] = t1 + t2
    cf[4] = 5 * np.real(t1 * t2)
    cf[8] = 9 * np.sin(np.angle(t1 * np.conj(t2)))
    for i in range(1, 4):
        cf[i] = cf[i-1]**2 + cf[4]
    for j in range(5, 8):
        cf[j] = np.abs(t1)**(cf[j-1]) + cf[0]
    for k in range(9, 25):
        cf[k] = np.log(np.abs(cf[k-1])+1) + cf[8]
    return cf.astype(np.complex128)

ALLOWED["p820"]=p820

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

def moth1(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(35, dtype=np.complex128)  
    cf[0] = t1 + t2
    for k in range(1, 35):
        v = np.sin(k * cf[k-1]) + np.cos(k * t1)
        mag = np.abs(v)
        if mag == 0:
            cf[k] = 0 + 0j
        else:
            cf[k] = v / mag
    return cf.astype(np.complex128)
    
ALLOWED["moth1"]=moth1

def moth2(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(90, dtype=np.complex128)  
    cf[0] = t1 - t2
    for k in range(1, 90):
        v = np.sin(k * cf[k-1]) + np.cos(k * t1)
        mag = np.abs(v)
        if mag == 0:
            cf[k] = t1 + t2
        else:
            cf[k] = 1j * v / mag
    return cf.astype(np.complex128)
    
ALLOWED["moth2"]=moth2

def moth4(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(50, dtype=np.complex128)
    cf[0] = t1 + t2
    for k in range(1, len(cf)):
        v = np.sin(((k + 4) % 10) * cf[k - 1]) + np.cos((k % 10) * t1)
        av = np.abs(v)
        if np.isfinite(av) and av > 1e-10:
            cf[k] = v / av
        else:
            cf[k] = t1 + t2
    return cf.astype(np.complex128)

ALLOWED["moth4"]=moth4

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

