# ops_poly.py
import numpy as np
from numba.typed import Dict
from numba import types
import argparse
import ast

def poly_giga_1(z,a,state):
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

def poly_giga_2(z,a,state):
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
    
# FIXME comes out with missing edges
def poly_giga_3(z,a,state):
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
    
def poly_giga_4(z,a,state):
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
    
def poly_giga_5(z,a,state):
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

def poly_giga_6(z,a,state):
    t1, t2 = z[0], z[1] 
    n = 10
    cf = np.zeros(n, dtype=np.complex128)
    cf[0] = 150 * t2**3 - 150j * t1**2
    cf[1] = 0
    cf[n//2-1] = 100*(t1-t2)**1
    cf[n-2] = 0
    cf[n-1] = 10j
    return cf
    
def poly_giga_7(z,a,state):
    t1, t2 = z[0], z[1] 
    pi  =  np.pi
    n   =  30
    rec =  np.linspace(np.real(t1), np.real(t2), n)
    imc =  np.linspace(np.imag(t1), np.imag(t2), n)
    f1  =  np.exp(1j * np.sin(10 * pi * imc))
    f2  =  np.exp(1j * np.cos(10 * pi * rec))
    f   =  f1 + f2
    return  f
    
# slightly modified
def poly_giga_8(z,a,state):
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
    
# FIXME missing some parts
# could it be the rasterizer ?
def poly_giga_9(z,a,state):
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
    
def poly_giga_10(z,a,state):
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
    
def poly_giga_11(z,a,state):
    t1, t2 = z[0], z[1]
    n = 40
    cf = np.zeros(n, dtype=np.complex128)
    m = int(5*abs(t1 + t2) % 17) + 1
    modular_values = np.arange(n) % m
    for k in range(n):
        scale_factor = modular_values[k]
        cf[k] = scale_factor * np.exp(1j * np.pi * (k+1) / (m + t1 + t2))
    return cf

def poly_giga_12(z,a,state):
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
      
def poly_giga_13(z,a,state):
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
    
def poly_giga_14(z,a,state):
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
    
def poly_giga_15(z,a,state):
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

def poly_giga_16(z,a,state):
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

def poly_giga_17(z,a,state):
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
    
def poly_giga_18(z,a,state):
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
    

def poly_giga_19(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(90, dtype=np.complex128)
    cf[0] = t1 - t2
    for k in range(1, len(cf)):
        v = np.sin(k * cf[k-1]) + np.cos(k * t1)
        av = np.abs(v)
        cf[k] = 1j * v / av if np.isfinite(av) and av > 1e-10 else t1 + t2
    return cf
    
def poly_giga_47(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(30, dtype=np.complex128)
    cf[0:3] = np.roots([t1**3 - t2**2, 100 * t1, -50 * t2, 10j])
    cf[9:11] = np.roots([1, t1**2 - 1j * t2, -100])
    cf[14] = 50 * t1**3 - 20 * t2
    cf[24] = 200 * np.sin(t1.real + t2.imag) + 1j * np.cos(t1.imag - t2.real)
    cf[29] = np.exp(1j * t1) + t2**3
    return cf

    
def poly_giga_62(z,a,state):
    t1, t2 = z[0], z[1]
    cf = np.zeros(25, dtype=np.complex128)
    cf[0:5] = np.array([abs(t1 + t2)**(i+1) for i in range(5)])
    cf[5:10] = ((t1+2j*t2)**3).real * np.log(np.abs(np.conj(t1*t2)))
    cf[10:15] = ((t1-t2)**2).imag / np.angle(t1*t2)
    cf[15:20] = np.abs(cf[5:10])**0.5 + np.angle(cf[0:5])
    cf[20:25] = np.array([abs(t1 * t2)**(i+1) for i in range(5)])
    return cf

def poly_giga_87(z,a,state):
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
    

def giga_221(z,a,state):
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

ALLOWED = {
    "poly_giga_1":    poly_giga_1,
    "poly_giga_2":    poly_giga_2,
    "poly_giga_3":    poly_giga_3,
    "poly_giga_4":    poly_giga_4,
    "poly_giga_5":    poly_giga_5,
    "poly_giga_6":    poly_giga_6,
    "poly_giga_7":    poly_giga_7,
    "poly_giga_8":    poly_giga_8,
    "poly_giga_9":    poly_giga_9,
    "poly_giga_10":   poly_giga_10,
    "poly_giga_11":   poly_giga_11,
    "poly_giga_12":   poly_giga_12,
    "poly_giga_13":   poly_giga_13,
    "poly_giga_14":   poly_giga_14,
    "poly_giga_15":   poly_giga_15,
    "poly_giga_16":   poly_giga_16,
    "poly_giga_17":   poly_giga_17,
    "poly_giga_18":   poly_giga_18,
    "poly_giga_19":   poly_giga_19,
    "poly_giga_62":   poly_giga_62,
    "poly_giga_87":   poly_giga_87,
    "p7f":            p7f,
    "giga_221":       giga_221,
}

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

