import numpy as np
import math
import cmath
from scipy.special import sph_harm
import polystate as ps
import letters
import zfrm

def poly_giga_1(t1, t2):
    n = 25
    try:
        cf = np.zeros(n, dtype=complex)
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
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(n, dtype=complex)
    
def poly_giga_2(t1, t2):
    n = 25
    try:
        cf = np.zeros(n, dtype=complex)
        cf[0] = -5
        cf[1] = -10 * (t1 * t2)      
        cf[2] = 20 * (t1 - t2)      
        cf[5] = 50 * (t1**3 + t2)   
        cf[7] = -80 * (t1**4 - t2**2)     
        cf[9] = 200 * (t1**2 + t2**2)
        cf[15] = 150 * (t1**3 * t2**5)
        cf[19] = -30 * (t1**5 - t2**5)    
        cf[24] = 10 * (t1 * t2**3)        
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(n, dtype=complex)

def poly_giga_3(t1, t2):
    n = 25
    try:
        cf = np.zeros(n, dtype=complex)
        cf[0] = 1
        cf[14] = np.exp(t1-t2)
        cf[15] = 0*np.exp(t1+t2)
        cf[16] = 1*np.exp(1j*t1)
        cf[17] = 1*np.exp(t1)
        cf[18] = 1*np.exp(-t1)
        cf[19] = 1*np.exp(-1j*t1)
        cf[23] = 1*np.exp(1j*t2)
        cf[24] = 1+1j
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(n, dtype=complex)
    
def poly_giga_4(t1, t2):
    n = 25
    try:
        cf = np.zeros(n, dtype=complex)
        cf[0] = 100
        cf[12] = 100 * t1**3 + 100 * t1**2 + 100 * t1 - 100
        cf[14] = 100 * t1**3 - 100 * t1**2 + 100 * t1 - 100
        cf[16] = 100 * t1**3 + 100 * t1**2 - 100 * t1 - 100
        cf[20] = -10
        cf[24] = np.exp(0.2j * t2)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(n, dtype=complex)
    
def poly_giga_5(t1, t2):
    n = 26
    try:
        cf = np.zeros(n, dtype=complex)
        cf[[0, 4, 12, 19, 20, 24]] = [1, 4, 4, -9, -1.9, 0.2]
        cf[6] = 100j * t2**3 + 100j * t2**2 - 100j * t2 - 100j
        cf[8] = 100j * t1**3 + 100j * t1**2 + 100j * t2 - 100j
        cf[14] = 100j * t2**3 - 100j * t2**2 + 100j * t2 - 100j
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(n, dtype=complex)
    
def poly_giga_6(t1, t2):
    n = 10
    try:
        cf = np.zeros(n, dtype=complex)
        cf[0] = 150 * t2**3 - 150j * t1**2
        cf[1] = 0
        cf[n//2-1] = 100*(t1-t2)**1
        cf[n-2] = 0
        cf[n-1] = 10j
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(n, dtype=complex)
    

def poly_giga_7(t1, t2):
    n = 30
    try:
        pi  =  np.pi
        rec =  np.linspace(np.real(t1), np.real(t2), n)
        imc =  np.linspace(np.imag(t1), np.imag(t2), n)
        f1  =  np.exp(1j * np.sin(10 * pi * imc))
        f2  =  np.exp(1j * np.cos(10 * pi * rec))
        f   =  f1 + f2
        return  f
    except Exception:
        return np.zeros(n, dtype=complex)


def poly_giga_7a(t1, t2):
    pi  =  np.pi
    n   =  30
    try:    
        tt1 =  np.exp(1j * 2 * pi * t1)
        tt2 =  np.exp(1j * 2 * pi * t2) 
        rec =  np.linspace(np.real(tt1), np.real(tt2), n)
        imc =  np.linspace(np.imag(tt1), np.imag(tt2), n)
        f1  =  np.exp(1j * np.sin(10 * pi * imc))
        f2  =  np.exp(1j * np.cos(10 * pi * rec))
        f   =  f1 + f2
        return f
    except:
        return np.zeros(30, dtype=complex)
    

def poly_giga_7b(t1, t2):
    pi  =  np.pi
    n   =  23
    try:
        tt1 =  np.exp(1j * 2 * pi * t1)
        ttt1 =  np.exp(1j * 2 * pi * np.real(tt1))
        tttt1 =  np.exp(1j * 2 * pi * np.imag(ttt1))
        tt2 =  np.exp(1j * 2 * pi * t2)
        ttt2 =  np.exp(1j * 2 * pi * np.imag(tt2))
        tttt2 =  np.exp(1j * 2 * pi * np.real(ttt2)) 
        rec =  np.linspace(np.real(tt1), np.real(tttt2), n)
        imc =  np.linspace(np.imag(tt2), np.imag(tttt1), n)
        f1  =  np.exp(1j * np.sin(10 * pi * imc))
        f2  =  np.exp(1j * np.cos(10 * pi * rec))
        f   =  f1 + f2
        return f
    except:
        return np.zeros(23, dtype=complex)


def poly_giga_7c(t1, t2):
    pi   =  np.pi
    n    =  23
    try:
        tt1  =  np.exp(1j * 2 * pi * t1)
        ttt1 =  np.exp(1j * 2 * pi * tt1)
        rec  =  np.linspace(np.real(tt1), np.real(ttt1), n)
        imc  =  np.linspace(np.imag(tt1), np.imag(ttt1), n)
        f1   =  np.exp(1j * np.sin(11 * pi * imc))
        f2   =  np.exp(1j * np.cos(13 * pi * rec))
        f    =  f1 + f2
        return f
    except:
        return np.zeros(23, dtype=complex)


def poly_giga_7d(t1, t2):
    pi   =  np.pi
    n    =  61
    try:
        tt1  =  np.exp(1j * 2 * pi * t1)
        ttt1 =  np.exp(1j * 2 * pi * np.real(tt1))
        tttt1 =  np.exp(1j * 2 * pi * np.real(ttt1))
        rec  =  np.linspace(np.real(tt1), np.real(ttt1), n)
        imc  =  np.linspace(np.imag(ttt1), np.imag(tttt1), n)
        f1   =  np.exp(1j * np.sin(11 * pi * imc))
        f2   =  np.exp(1j * np.sin(67 * pi * rec))
        f    =  f1 + f2
        f[0] = f[0] + np.exp(1j * 2 * pi * t2)
        return f
    except:
        return np.zeros(61, dtype=complex)
    

def poly_giga_8(t1, t2):
    n = 35
    try:
        cf = np.zeros(n, dtype=complex)

        roots1 = np.roots([1, t1**3, -50 * t2, 100 * t1, 10j])
        roots2 = np.roots([1, roots1[0], -np.real(roots1[1]), np.imag(roots1[2])])
        cf[0:4] = roots1
        cf[9:13] = roots2
        cf[19] = 50 * t1 * t2 + np.real(roots2[0])
        cf[29] = np.exp(1j * t1) + 50 * t2**3
        cf[34] = 200 * np.exp(1j * t1**3) - np.exp(-1j * t2**2)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(n, dtype=complex)

def poly_giga_9(t1, t2):
    n = 20
    try:

        re1 = t1.real
        im1 = t1.imag
        re2 = t2.real
        im2 = t2.imag
        rec = np.linspace(re1, re2, n)
        imc = np.linspace(im1, im2, n)
        cf = 100j * imc**9 + 100 * rec**9
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(n, dtype=complex)


def poly_giga_10(t1, t2):
    n = 30
    try:
        k = np.arange(1, n + 1)
        cf = k * np.exp(1j * k * t1) + k**2 * np.exp(-1j * k * t2)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(n, dtype=complex)


def poly_giga_11(t1, t2):
    n = 25
    try:
        k = np.arange(1, n + 1)
        base = t1 + t2
        cf = base**k / k
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(n, dtype=complex)


def poly_giga_12(t1, t2):
    n = 20
    try:
        k = np.arange(1, n + 1)
        cf = np.sin(k * t1) + 1j * np.cos(k * t2)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(n, dtype=complex)


def poly_giga_13(t1, t2):
    n = 15
    try:
        cf = np.zeros(n, dtype=complex)
        cf[::2] = np.exp(1j * (np.arange(0, n, 2) + 1) * t1)
        cf[1::2] = np.exp(-1j * (np.arange(0, n, 2) + 1) * t2)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(n, dtype=complex)


def poly_giga_14(t1, t2):
    n = 35
    try:
        k = np.arange(1, n + 1)
        cf = ((t1 - t2) ** (k % 5)) / (k + 1) + np.exp(1j * k * (t1 + t2))
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(n, dtype=complex)


def poly_giga_15(t1, t2):
    n = 40
    try:
        k = np.arange(1, n + 1)
        cf = np.exp(1j * t1 * k) * np.cos(t2 * k)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(n, dtype=complex)


def poly_giga_16(t1, t2):
    n = 12
    try:
        cf = np.log(np.abs(t1 + t2) + np.arange(1, n + 1))
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(n, dtype=complex)


def poly_giga_17(t1, t2):
    n = 20
    try:
        k = np.arange(1, n + 1)
        cf = (t1 + 1j * t2) ** k / (k + 1)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(n, dtype=complex)


def poly_giga_18(t1, t2):
    n = 30
    try:
        k = np.arange(1, n + 1)
        cf = np.exp(1j * k * (t1 - t2)) * (t1 + t2) ** (k % 3)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(n, dtype=complex)


def poly_giga_19(t1, t2):
    n = 10
    try:
        k = np.arange(1, n + 1)
        cf = np.exp(-1j * k * (t1 + t2)) / (k)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(n, dtype=complex)

def poly_giga_20(t1, t2):
    try:
        cf = np.zeros(90, dtype=complex)
        cf[0] = t1 + 1j * t2
        for k in range(1, len(cf)):
            v = np.sin(k * cf[k-1]) + np.cos(k * t1)
            av = np.abs(v)
            if np.isfinite(av) and av > 1e-10:
                cf[k] = 1j * v / av
            else:
                cf[k] = t1 + t2
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_21(t1, t2):
    try:
        cf = np.zeros(50, dtype=complex)
        cf[0] = t1 + t2
        for k in range(1, len(cf)):
            v = np.sin(((k+3) % 10) * cf[k-1]) + np.cos(((k+1) % 10) * t1)
            av = np.abs(v)
            if np.isfinite(av) and av > 1e-10:
                cf[k] = v / av
            else:
                cf[k] = t1 + t2
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_22(t1, t2):
    try:
        n = 26
        cf = np.zeros(n, dtype=complex)
        cf[0] = 100
        cf[12] = 100 * t1**3 + 100 * t1**2 + 100 * t1 - 100
        cf[14] = 100 * t2**3 - 100 * t2**2 + 100 * t2 - 100
        cf[16] = 100 * t2**3 + 100 * t2**2 - 100 * t2 - 100
        cf[20] = -10
        cf[24] = 0.2j
        cf[25] = 0
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_23(t1, t2):
    try:
        cf = np.zeros(26, dtype=complex)
        indices = [0, 4, 12, 19, 20, 24]
        values = [1, 4, 4, -9, -1.9, 0.2]
        cf[indices] = values
        cf[6] = 100 * t2**3 + 100 * t2**2 - 100 * t2 - 100
        cf[8] = 100 * t1**3 + 100 * t1**2 + 100 * t2 - 100
        cf[14] = 100 * t2**3 - 100 * t2**2 + 100 * t2 - 100
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_24(t1, t2):
    try:
        cf = np.zeros(26, dtype=complex)
        indices = [0, 4, 12, 19, 20, 24]
        values = [1, 4, 4, -9, -1.9, 0.2]
        cf[indices] = values
        cf[6] = 100j * t2**3 + 100j * t2**2 - 100 * t2 - 100
        cf[8] = 100j * t1**3 + 100j * t1**2 + 100 * t2 - 100
        cf[14] = 100j * t2**3 - 100j * t2**2 + 100 * t2 - 100
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_25(t1, t2):
    try:
        n = 26
        cf = np.zeros(n, dtype=complex)
        cf[0] = 100
        cf[12] = 100 * t1**3 + 100 * t1**2 + 100 * t1 - 100
        cf[14] = 100 * t2**3 - 100 * t2**2 + 100 * t2 - 100
        cf[16] = 100 * t2**3 + 100 * t2**2 - 100 * t2 - 100
        cf[20] = -10
        cf[24] = 0.2j
        cf[25] = 0
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_26(t1, t2):
    try:
        n = 26
        cf = np.zeros(n, dtype=complex)
        cf[0] = 100
        cf[12] = 100 * t1**3 + 100 * t1**2 + 100 * t1 - 100
        cf[14] = 100 * t2**3 - 100 * t2**2 + 100 * t2 - 100
        cf[16] = 100 * t2**3 + 100 * t2**2 - 100 * t2 - 100
        cf[20] = -10
        cf[24] = 0.2
        cf[25] = 0
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_27(t1, t2):
    try:
        n = 12
        cf = np.zeros(n, dtype=complex)
        cf[0:3] = [-100j, -100j, -100j]
        mid_indices = [n//2-2, n//2-1, n//2]
        cf[mid_indices] = 100 * np.roots([t1, t2, t1, 1])
        end_indices = [n-1, n-2, n-3]
        cf[end_indices] = 100 * np.roots([t2, t1, t2, 10j])
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_28(t1, t2):
    try:
        n = 6
        cf = np.zeros(n, dtype=complex)
        cf[0] = 100 * t2**3 + 100j * t1**3
        cf[1] = 0
        # cf[n//2-1] = 150
        cf[int(n/2) - 1] = 150
        cf[n-2] = 0
        cf[n-1] = 40j
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_29(t1, t2):
    try:
        n = 10
        cf = np.zeros(n, dtype=complex)
        cf[0] = 150 * t2**3 - 150j * t1**2
        cf[1] = 0
        cf[n//2-1] = 100*(t1-t2)**1
        cf[n-2] = 0
        cf[n-1] = 10j
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_30(t1, t2):
    try:
        n = 10
        cf = np.zeros(n, dtype=complex)
        cf[0] = 150j * t2**2 + 100 * t1**3
        cf[n//2-1] = 150 * np.abs(t1 + t2 - 2.5 * (1j + 1))
        cf[n-1] = 100j * t1**3 + 150 * t2**2
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_31(t1, t2):
    try:
        n = 100
        cf = np.zeros(n, dtype=complex)
        cf[0:3] = [-100j, 0, 0]
        pr1 = safe_polyroot([t1, t2, t1, 1])
        if len(pr1) == 3:
            cf[n//2-2:n//2+1] = 100 * pr1
        else:
            cf[n//2-2:n//2+1] = 100
        pr2 = safe_polyroot([t2, t1, t2, 10j])
        if len(pr2) == 3:
            cf[n-3:n] = 100 * pr2
        else:
            cf[n-3:n] = 100
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_32(t1, t2):
    try:
        n = 12
        cf = np.zeros(n, dtype=complex)
        cf[0:3] = [-100j, -100j, -100j]
        mid_indices = [n//2-2, n//2-1, n//2]
        cf[mid_indices] = 100 * np.roots([t1, t2, t1, 1])
        end_indices = [n-1, n-2, n-3]
        cf[end_indices] = 100 * np.roots([t2, t1, t2, 10j])
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_33(t1, t2):
    try:
        cf = np.zeros(26, dtype=complex)
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
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_34(t1, t2):
    try:
        n = 120
        cf = np.zeros(n, dtype=complex)
        cf[0] = -1
        cf[n//2-1] = 100 * t1 - 100j * t2
        cf[n-1] = 0.4
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_35(t1, t2):
    try:
        n = 120
        cf = np.zeros(n, dtype=complex)
        cf[0] = -1
        cf[n//2-1] = 100 * t1 - 100j * t2
        cf[n-1] = 0.4
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_36(t1, t2):
    try:
        cf = np.zeros(26, dtype=complex)
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
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)
    

def poly_giga_37(t1, t2):
    try:
        cf = np.zeros(26, dtype=complex)
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
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_38(t1, t2):
    try:
        n = 26
        cf1 = np.zeros(n, dtype=complex)
        cf1[0] = 100
        cf1[12] = 100 * t1**3 + 100 * t1**2 + 100 * t1 - 100
        cf1[14] = 100 * t1**3 - 100 * t1**2 + 100 * t1 - 100
        cf1[16] = 100 * t1**3 + 100 * t1**2 - 100 * t1 - 100
        cf1[20] = -10
        cf1[24] = np.exp(0.2j * t2)
        cf1[25] = 0

        cf2 = np.zeros(n, dtype=complex)
        cf2[0] = 100
        cf2[12] = 100 * t1**3 + 100 * t1**2 + 100 * t1 - 100
        cf2[14] = 100 * t2**3 - 100 * t2**2 + 100 * t2 - 100
        cf2[16] = 100 * t2**3 + 100 * t2**2 - 100 * t2 - 100
        cf2[20] = -10
        cf2[24] = 0.2j
        cf2[25] = 0

        result = (cf1 - 0.0001 * np.sum(np.abs(cf1))) * (cf2 + 1.5j * np.sum(np.abs(cf2)))
        return np.flip(result)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_39(t1, t2):
    try:
        cf = np.zeros(50, dtype=complex)
        cf[np.array([0, 9, 19, 29, 39, 49])] = np.array([1, 2, -3, 4, -5, 6])
        cf[14] = 100 * (t1**2 + t2**2)
        cf[24] = 50 * (np.sin(t1) + 1j * np.cos(t2))
        cf[34] = 200 * (t1 * t2) + 1j * (t1**3 - t2**3)
        cf[44] = np.exp(1j * (t1 + t2)) + np.exp(-1j * (t1 - t2))
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_40(t1, t2):
    try:
        cf = np.zeros(35, dtype=complex)
        cf[np.array([0, 6, 14, 19, 26, 34])] = [1, -2, 3, -4, 5, -6]
        cf[11] = 50j * np.sin(t1**2 - t2**2)
        cf[17] = 100 * (np.cos(t1) + 1j * np.sin(t2))
        cf[24] = 50 * (t1**3 - t2**3 + 1j * t1 * t2)
        cf[29] = 200 * np.exp(1j * t1) + 50 * np.exp(-1j * t2)
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_41(t1, t2):
    try:
        cf = np.zeros(60, dtype=complex)
        cf[np.array([0, 9, 29, 49])] = [1, -5, 10, -20]
        cf[19] = 100 * np.exp(t1 + t2)
        cf[39] = 50 * (t1**2 * t2 + 1j * t2**2)
        cf[54] = np.exp(1j * t1) * np.exp(-1j * t2) + 50 * t1**3
        cf[59] = 300 * np.sin(t1 + t2) + 1j * np.cos(t1 - t2)
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_42(t1, t2):
    try:
        cf = np.zeros(50, dtype=complex)
        cf[np.array([0, 7, 15, 31, 39])] = [1, -3, 3, -1, 2]
        cf[11] = 100j * np.exp(t1**2 + t2**2)
        cf[19] = 50 * (t1**3 + t2**3)
        cf[24] = np.exp(1j * (t1 - t2)) + 10 * t1**2
        cf[44] = 200 * np.sin(t1 + t2) + 1j * np.cos(t1 - t2)
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_43(t1, t2):
    try:
        cf = np.zeros(40, dtype=complex)
        cf[np.array([0, 4, 14, 29])] = [1, -5, 10, -20]
        cf[19] = 100j * (t1**3 - t2**3)
        cf[9] = 50 * (t1**2 * t2 + 1j * t2**2)
        cf[24] = np.exp(1j * t1) + np.exp(-1j * t2)
        cf[34] = 200 * t1 * t2 * np.sin(t1 + t2)
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_44(t1, t2):
    try:
        cf = np.zeros(30, dtype=complex)
        cf[np.array([0, 5, 11, 19])] = [1, 3, -2, 5]
        cf[9] = 100 * t1**3 + 50 * t2**2
        cf[14] = 50j * (t1.real - t2.imag)
        cf[24] = 200 * t1 * (t2 + 1) - 100j * t2
        cf[29] = np.exp(1j * t1) + t2**3
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_45(t1, t2):
    try:
        cf = np.zeros(50, dtype=complex)
        cf[0] = 1
        cf[4] = 50 * np.exp(t1)
        cf[9] = 100 * (t2**2 - 1j * t1)
        cf[19] = 200 * np.exp(1j * t1**2) - 50 * np.exp(-1j * t2**3)
        cf[29] = 100 * t1 * t2**2 + 50j * t1**3
        cf[39] = np.exp(1j * (t1 + t2)) - 50 * np.sin((t1 - t2).imag)
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_46(t1, t2):
    try:
        cf = np.zeros(40, dtype=complex)
        cf[np.array([0, 7, 15, 23, 31])] = [1, -3, 5, -7, 2]
        cf[4] = 50 * (t1**2 - t2**3)
        cf[11] = 100j * (t1**3 + t2)
        cf[19] = np.exp(1j * t1) + np.exp(-1j * t2**2)
        cf[29] = 200 * np.sin(t1.real + t2.imag) - 50 * np.cos((t1 - t2).imag)
        cf[34] = np.exp(1j * t1**3) + t2**2
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_47(t1, t2):
    try:
        cf = np.zeros(30, dtype=complex)
        cf[0:4] = pr([t1**3 - t2**2, 100 * t1, -50 * t2, 10j])
        cf[9:12] = pr([1, t1**2 - 1j * t2, -100])
        cf[14] = 50 * t1**3 - 20 * t2
        cf[24] = 200 * np.sin(t1.real + t2.imag) + 1j * np.cos(t1.imag - t2.real)
        cf[29] = np.exp(1j * t1) + t2**3
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_48(t1, t2):
    try:
        cf = np.zeros(40, dtype=complex)
        cf[0:4] = pr([np.sin(t1) + np.cos(t2), 100 * t1**2, -50 * t2, 10j])
        cf[9:12] = pr([np.cos(t1.real) + np.sin(t2.imag), -1, t1**3 - t2**2])
        cf[19] = 50 * (t1**2 - t2**3)
        cf[29] = np.exp(1j * t1) + t2**2
        cf[34] = 200 * np.sin(t1.real + t2.imag) + 50 * np.cos((t1 - t2).imag)
        cf[39] = np.exp(1j * t1**3) + t2**3
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_49(t1, t2):
    try:
        cf = np.zeros(30, dtype=complex)
        unstable_roots = pr([t1**4 - t2**3, 1j * t1 - t2, 50 * np.sin(t1 + t2), -100])
        cf[0:4] = unstable_roots
        cf[7:10] = pr([unstable_roots[0]**2, -unstable_roots[1], 1])
        cf[14] = 100j * (t1**2 + t2**3)
        cf[24] = 200 * np.sin(t1.real + t2.imag) + 50 * np.cos((t1 - t2).imag)
        cf[29] = np.exp(1j * t1**3) + t2**3
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)



def poly_giga_50(t1, t2):
    try:
        n = 1000
        cf = np.zeros(n, dtype=complex)
        w = 100
        p1 = 1 * t2**3 - 1 * t2**2 + 1 * t2 - 1
        p2 = 1 * t2**3 + 1 * t2**2 + 1 * t2 + 1
        p3 = 1 * t1**3 + 1 * t2**2 + 1 * t1 + 1
        p4 = 1 * t1**3 + 1 * t1**2 + 1 * t1 + 1
        p5 = 1 * t1**3 + 1 * t1**2 + 1 * t1 - 1
        
        cf[0] = 10
        idx_range = np.arange(int(n*0.25) - 5, int(n*0.25) + 1)
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
        
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_51(t1, t2):
    try:
        n = 1000
        cf = np.zeros(n, dtype=complex)
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
            
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_52(t1, t2):
    try:
        n = 100
        cf = np.zeros(n, dtype=complex)
        
        # Using numpy's roots function instead of R's polyroot
        cf[[0, 1, 2]] = [-100j, 0, 0]
        roots1 = np.roots([1, t1, t2, t1])
        roots2 = np.roots([10j, t2, t1, t2])
        
        cf[n//2-1:n//2+2] = 100 * np.resize(roots1, 3)
        cf[n-3:n] = 100 * np.resize(roots2, 3)
        
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_53(t1, t2):
    try:
        n = 10
        cf = np.zeros(n, dtype=complex)
        cf[0] = 100 * np.sin(t1)**3 * np.cos(t2)**2
        cf[1] = 100 * np.exp(1j * (t1 + t2)) - 10 * (t1 - t2)**2
        cf[2] = t1*t2*(t1 - t2) / (np.abs(t1) + np.abs(t2) + 1)
        cf[4] = (t1*t2*np.exp(1j * (t1**2-t2**2)))**3
        cf[6] = np.sqrt(np.abs(t1)) - np.sqrt(np.abs(t2)) + 1j * np.sin(t1*t2)
        cf[7] = 50 * np.abs(t1 - t2) * np.exp(1j * np.abs(t1 + t2))
        cf[8] = t1-abs(t2) if t1.imag > 0 else t2-abs(t1)
        cf[9] = (1j*t1*t2)**(0.1*t1*t2)
        
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_54(t1, t2):
    try:
        cf = np.zeros(11, dtype=complex)
        cf[0] = t1.real * np.exp(1j * t2)
        cf[2] = (t1 * t2).imag * np.exp(-1j * t2.real)
        cf[4] = (t1.real + t2.imag)**2 + 10j
        cf[6] = (t2.imag**3) / t1.real - 1j
        cf[8] = (t1 * t2).real * np.exp(1j * ((t1 + t2).imag**2))
        cf[9] = np.sum(cf[0:9])
        cf[10] = np.prod(cf[0:10])
        
        return cf.astype(np.complex128)
    except:
        return np.zeros(11, dtype=complex)


def poly_giga_55(t1, t2):
    try:
        n = 10
        cf = np.zeros(n, dtype=complex)
        
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
        
        return cf.astype(np.complex128)
    except:
        return np.zeros(10, dtype=complex)


def poly_giga_56(t1, t2):
    try:
        cf = np.zeros(11, dtype=complex)
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
        
        return cf.astype(np.complex128)
    except:
        return np.zeros(11, dtype=complex)


def poly_giga_57(t1, t2):
    try:
        cf = np.zeros(10, dtype=complex)
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
    except:
        return np.zeros(10, dtype=complex)


def poly_giga_58(t1, t2):
    try:
        n = 10
        cf = np.zeros(n, dtype=complex)
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
    except:
        return np.zeros(10, dtype=complex)


def poly_giga_59(t1, t2):
    try:
        cf = np.zeros(11, dtype=complex)
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
    except:
        return np.zeros(10, dtype=complex)


def poly_giga_60(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        for k in range(25):
            cf[k] = np.sin(k+1)*t1/(1+abs(t2)) + np.cos(k+1)*t2/(1+abs(t1)) + np.sqrt(k+1)
        cf[0] = np.abs(t1)*abs(t2)
        cf[4] = np.angle(t1)*abs(t2)
        cf[9] = np.abs(t1)*np.angle(t2)
        cf[14] = np.abs(t1)*t2.real
        cf[19] = np.abs(t1)*t2.imag
        cf[24] = t1.real*abs(t2)
        return cf.astype(np.complex128)
    except:
        return np.zeros(25, dtype=complex)


def poly_giga_61(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
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
    except:
        return np.zeros(25, dtype=complex)


def poly_giga_62(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        cf[0:5] = np.array([abs(t1 + t2)**(i+1) for i in range(5)])
        cf[5:10] = ((t1+2j*t2)**3).real * np.log(np.abs(np.conj(t1*t2)))
        cf[10:15] = ((t1-t2)**2).imag / np.angle(t1*t2)
        cf[15:20] = np.abs(cf[5:10])**0.5 + np.angle(cf[0:5])
        cf[20:25] = np.array([abs(t1 * t2)**(i+1) for i in range(5)])
        return cf.astype(np.complex128)
    except:
        return np.zeros(25, dtype=complex)


def poly_giga_63(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
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
        return cf.astype(np.complex128)
    except:
        return np.zeros(25, dtype=complex)


def poly_giga_64(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        cf[0] = t1 + t2
        cf[1] = np.exp(1j * np.angle(t1 * np.conj(t2)))
        cf[2] = np.abs(t1) * np.abs(t2)
        for k in range(3, 25):
            cf[k] = (cf[k-1].real + 1j * cf[k-1].imag) * np.exp(1j * np.angle(cf[k-2]))
            if cf[k].imag == 0:
                cf[k] = cf[k] + 1e-10
            cf[k] = np.log(np.abs(cf[k])) / 2 + cf[k] * 1j
        return cf.astype(np.complex128)
    except:
        return np.zeros(25, dtype=complex)


def poly_giga_65(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
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
        return cf.astype(np.complex128)
    except:
        return np.zeros(25, dtype=complex)


def poly_giga_66(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
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
        return cf.astype(np.complex128)
    except:
        return np.zeros(25, dtype=complex)


def poly_giga_67(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
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
        return cf.astype(np.complex128)
    except:
        return np.zeros(25, dtype=complex)


def poly_giga_68(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        for k in range(25):
            cf[k] = np.abs(t1)**((k+1)/2) * (np.cos((k+1) * np.angle(t2)) + 1j * np.sin((k+1) * np.angle(t2)))
        cf[4] = cf[4] + (np.log(np.abs(t1)) + np.log(np.abs(t2))) / 2
        cf[9] = cf[9] + np.conj(t1 * t2)
        cf[14] = cf[14] + np.abs(t2 - t1)**2
        cf[19] = cf[19] + (np.sin(np.angle(t1)) / np.cos(np.angle(t2)))**3
        cf[24] = cf[24] + ((1j * t1 - t2)**2 / (1 + np.abs(t1 + t2)**3))**4
        return cf.astype(np.complex128)
    except:
        return np.zeros(25, dtype=complex)


def poly_giga_69(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        for i in range(25):
            cf[i] = (t1.real**(i+1) + t2.imag**(25-i))/(1 + np.abs(t1 + t2)) * np.exp(1j * np.angle(t1 + t2))
        cf[2] = 3 * np.conj(t1**2 + t2)
        cf[6] = 7 * np.abs(t1 + t2)
        cf[10] = 11 * (t1/t2 + np.conj(t2/t1))
        cf[16] = 17 * (np.abs(t1)*abs(t2))/(np.abs(t1 + t2))**2
        cf[22] = 23 * (np.conj(t1) + t2) / (1 + np.abs(t1 * np.conj(t2)))
        cf[24] = 25 * (np.conj(t1) + np.conj(t2)) / np.abs(t1*t2)
        return cf.astype(np.complex128)
    except:
        return np.zeros(25, dtype=complex)
    

def poly_giga_70(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        cf[0:5] = np.real(t1) * np.arange(1, 6) - np.imag(t2) * np.arange(1, 6)
        cf[5] = np.abs(t1) * np.abs(t2)
        cf[6:11] = np.angle(t1+t2) * np.arange(6, 11)
        cf[11] = np.conj(t1) + np.conj(t2)
        cf[12:17] = np.real(t1 + 1j * t2) * np.arange(1, 6)
        cf[17] = np.angle(t1) * np.angle(t2)
        cf[18:23] = np.imag(t1 - 1j * t2) * np.arange(1, 6)
        cf[23] = np.conj(t1 * t2)
        cf[24] = np.abs(cf[11]) + np.angle(cf[17])
        return cf.astype(np.complex128)
    except:
        return np.zeros(25, dtype=complex)


def poly_giga_71(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
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
        return cf.astype(np.complex128)
    except:
        return np.zeros(25, dtype=complex)


def poly_giga_72(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        cf[0] = t1 + t2
        for k in range(1, 25):
            v = np.sin(k * cf[k-1] + np.angle(t2**k)) + np.cos(k * np.abs(t1))
            cf[k] = v / (np.abs(v) + 1e-10)
        cf[9] = t1 * t2 - np.abs(t2)**2 + 1j * np.angle(t1)
        cf[14] = np.conj(t1)**3 - np.angle(t2)**3 + 1j * np.abs(t2)
        cf[19] = np.abs(t2)**3 + t1**2 + t2**2 + 1j * np.angle(t2)**2
        cf[24] = np.abs(t1 * t2) + np.angle(t1)**5 + 1j * np.abs(t1)**5
        return cf.astype(np.complex128)
    except:
        return np.zeros(25, dtype=complex)


def poly_giga_73(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
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
        return cf.astype(np.complex128)
    except:
        return np.zeros(25, dtype=complex)


def poly_giga_74(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
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
        return cf.astype(np.complex128)
    except:
        return np.zeros(25, dtype=complex)


def poly_giga_75(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
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
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_76(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        cf[0] = t1 + t2
        for k in range(1, 25):
            v = (t1+t2)**(k+1) + np.sin(k * cf[k-1]) + np.log(np.abs(k * t1)) - np.log(np.abs((k+1) * t2))
            cf[k] = v / np.abs(v)
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_77(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        cf[0:10] = np.arange(1, 11) * t1 + np.arange(11, 21) * 1j * t2
        cf[10:20] = (t1 + 1j * t2)**2 * np.arange(11, 21)
        cf[20:25] = (np.abs(t1) + np.angle(t2)) * np.arange(1, 6)
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_78(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        for k in range(25):
            cf[k] = (k + 1 + t1) / (k + 1 + t2)
        cf[4] = cf[4] + np.log(np.abs(t1 + t2))
        cf[9] = cf[9] + np.sin(np.real(t1)) + np.cos(np.imag(t2))
        cf[14] = cf[14] + np.abs(cf[13])**2 + np.angle(cf[12])**2
        cf[19] = cf[19] + np.abs(np.real(t2) * np.imag(t1))
        cf[24] = cf[24] + np.abs(t1 + np.conj(t2))
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_79(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        cf[0:10] = np.arange(1, 11) * t1 + np.arange(11, 21) * 1j * t2
        cf[10:20] = (t1 + 1j * t2)**2 * np.arange(11, 21)
        cf[20:25] = (np.abs(t1) + np.angle(t2)) * np.arange(1, 6)
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_80(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        for k in range(25):
            cf[k] = (k + 1 + t1) / (k + 1 + t2)
        cf[4] = cf[4] + np.log(np.abs(t1 + t2))
        cf[9] = cf[9] + np.sin(np.real(t1)) + np.cos(np.imag(t2))
        cf[14] = cf[14] + np.abs(cf[13])**2 + np.angle(cf[12])**2
        cf[19] = cf[19] + np.abs(np.real(t2) * np.imag(t1))
        cf[24] = cf[24] + np.abs(t1 + np.conj(t2))
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_81(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        for i in range(25):
            cf[i] = (t1 * (i+1) + t2**((i+1)/2)) / np.abs(t1 * (i+1) + t2**(i+1))
        cf[2] = np.real(t1) + np.imag(t2)
        cf[6] = np.abs(np.exp(1j * np.angle(t1 * t2)))
        cf[10] = np.real(t1 * t2) + np.imag(t1 / t2)
        cf[12] = np.angle(t1 + 4*t2) / np.abs(np.conj(t1 - 4*t2))
        cf[16] = np.abs(np.exp(1j * np.angle(t1 - t2)))
        cf[18] = np.real(t1 / t2) - np.imag(t1 * t2)
        cf[22] = np.abs(np.exp(1j * np.angle(t1 + t2)))
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_82(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
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
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_83(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        cf[0] = t1.real * t2.real + t1.imag * t2.imag
        cf[1] = np.abs(t1) * np.abs(t2)
        cf[2] = np.angle(t1) + np.angle(t2)
        cf[3] = np.conj(t1).real + np.conj(t2).imag
        
        for k in range(4, 25):
            cf[k] = (cf[k-1] * cf[k-4] + cf[k-3] * cf[k-2]) / np.abs(cf[k-1] * cf[k-4] + cf[k-3] * cf[k-2])
        
        cf[12] = cf[0] / cf[1] + cf[2] / cf[3]
        cf[18] = np.log(np.abs(cf[0] * cf[1])+1) / np.log(np.abs(cf[2] * cf[3])+1)
        cf[24] = cf[4] * cf[9] * cf[14] * cf[19] / np.abs(cf[4] * cf[9] * cf[14] * cf[19])
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_84(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
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
        return cf.astype(np.complex128)
    except:
        return np.zeros(25, dtype=complex)


def poly_giga_85(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        
        for k in range(25):
            cf[k] = np.sin(t1)+(1j*np.cos(t2.real**2-t2.imag**2))/(np.sqrt(np.abs(t1)**2+abs(t2)**2))
        
        cf[2] = cf[6]*cf[10]*cf[18]/cf[22]
        cf[4] = cf[9] + cf[14] + cf[19] - cf[24]
        cf[8] = 1j*t1*t2*(t1 - t2)
        cf[12] = cf[4]*t1/(1+abs(t2))
        cf[16] = np.conj(cf[8])/t2
        cf[20] = np.log(np.abs(cf[4]*t2/(1+abs(t1))))
        return cf.astype(np.complex128)
    except:
        return np.zeros(25, dtype=complex)


def poly_giga_86(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        cf[0] = (t1**3).real*(t2**3).real
        cf[1] = (t1**3).imag*(t2**3).imag
        
        for k in range(2, 25):
            if k % 3 == 0:
                cf[k] = (t1+1j*t2)**(k/3) / k
            else:
                cf[k] = np.conj(cf[k-1]) ** 2 + np.abs(t1)*abs(t2)
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_87(t1, t2):
    try:
        cf = np.zeros(51, dtype=complex)
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
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_88(t1, t2):
    try:
        cf = np.zeros(51, dtype=complex)
        primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71])
        cf[0:20] = (t1**primes[0:20] + t2**(primes[0:20] + 1)) / np.abs(t1 - t2)
        
        for k in range(20, 51):
            cf[k] = (np.cos(k * np.log(np.abs(t1) + 1)) + np.sin(k * np.log(np.abs(t2) + 1))) / k
        
        cf[50] = np.abs(t1) * np.abs(t2) * (np.cos(np.angle(t1) * np.angle(t2)) - 1j * np.sin(np.angle(t1) * np.angle(t2)))
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_89(t1, t2):
    try:
        cf = np.zeros(51, dtype=complex)
        for i in range(51):
            cf[i] = np.cos(i*t1) + np.sin(i*t2)
        
        cf[0] = cf[0]*t1**50
        cf[1] = cf[1]*t2**49
        
        for i in range(2, 51):
            cf[i] = cf[i]*t1**(51-1j)*t2**(i-2)
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_90(t1, t2):
    try:
        cf = np.zeros(51, dtype=complex)
        cf[0] = t1 + t2
        for i in range(1, 51):
            cf[i] = ((np.cos(t1)/np.sin(t2))**i + (np.sin(t1)/np.cos(t2))**(2j)) * cmath.phase(t1 + t2)**i
        cf[1:51] = np.log(np.abs(cf[1:51]) + 1) / np.log(i+1)
        cf[4] = cf[4] * np.abs(t1 + t2)
        cf[9] = cf[9] * (t1 * t2.conjugate()).real
        cf[19] = cf[19] * (t2 * t1.conjugate()).imag
        cf[29] = cf[29] * cmath.phase(t1 + t2)
        cf[39] = cf[39] * np.abs(t1 + t2)
        cf[49] = cf[49] * (t1 * t2.conjugate()).real
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_91(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = t1 + t2
        for k in range(1, 71):
            r = np.abs(t1)**k + np.abs(t2)**(71-k)
            theta = cmath.phase(t1)**k - cmath.phase(t2)**(71-k)
            cf[k] = r * np.cos(theta) + r * np.sin(theta)*1j
        cf[2:70] = cf[2:70] + np.log(np.abs(t2-t1)+1)
        cf[70] = cf[70] + (t1*t2).conjugate()
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_92(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for i in range(71):
            cf[i] = (np.sin(t1)*np.cos(t2)**i) + (np.cos(t1)*np.sin(t2)**i) + np.log(np.abs(t1+t2)+1)**i
        cf[0] = cf[0] + t1.real*t2.imag + t1.imag*t2.real
        cf[1] = cf[1] + np.abs(t1) * np.abs(t2)
        cf[2] = cf[2] + cmath.phase(t1) / cmath.phase(t2)
        cf[3] = cf[3] + cmath.phase(t2) / cmath.phase(t1)
        cf[4] = cf[4] + t1.real / np.abs(t2)
        cf[5] = cf[5] + t2.imag / np.abs(t1)
        cf[70] = cf[70] + (t1+t2).conjugate()
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_93(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        angles = np.linspace(0, 2*np.pi, 35)
        for i in range(35):
            cf[i] = np.cos(angles[i] * t1) + np.sin(angles[i] * t2) / np.abs(t1)
        for i in range(35, 71):
            cf[i] = (t1*1j + t2*(71-1j))**3 / (t1*1j + 1j*t2*(71-1j))**2
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_94(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = t1.real * t2.imag
        for k in range(1, 71):
            cf[k] = np.sin(k * cf[k-1]) + np.cos(k * t1)
            cf[k] = cf[k] / np.abs(cf[k])
        cf[30] = np.abs(cf[14])**2 + cmath.phase(t2)**2
        cf[40] = cf[20] * (np.abs(t1) + np.log(np.abs(t2)+1))
        cf[50] = cf[30] + np.log(np.abs(t1*t2)+1)
        cf[60] = np.abs(t1 + t2) * cmath.phase(cf[30])
        cf[70] = np.abs(cf[34]) / (t1 + 1j * t2)
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_95(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = t1.real * t2.imag
        cf[1] = np.abs(t1) / cmath.phase(t2)
        cf[2] = np.sin(t1) * np.cos(t2)
        for i in range(3, 71):
            cf[i] = cf[i-1] + cf[i-3] * np.log(np.abs(cf[i-2]+1))
            if i % 2 == 0:
                cf[i] = cf[i] + np.abs(cf[i-1] * t1)
            else:
                cf[i] = cf[i] - np.abs(cf[i-1] * t2.conjugate())
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_96(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_97(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
        cf[50:60] = sum(np.prod(cf[:30] * np.array([-1, 1])))
        for i in range(60, 70):
            cf[i] = np.abs(t1)**i - np.abs(t2)**i + np.log(np.abs(t1**i+t2**i) + 1) + cmath.phase(cf[i-10])
        cf[70] = np.sum(cf[:70]).conjugate()
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_98(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = np.log(np.abs(t1) + 1) + np.log(np.abs(t2) + 1)
        for i in range(1, 71):
            cf[i] = np.sin(i * cmath.phase(t1 * t2)) * np.abs(cf[i-1]**(i-1)) * np.cos(np.abs(t1) + np.abs(t2))
        for i in range(34, 55):
            cf[i] = cf[i] * np.abs(t1 + t2) * cf[i-1]
        cf[4:26] = cf[4:26] * cf[0] * (np.abs(t1) * np.abs(t2))**np.arange(1, 23)
        cf[55:71] = cf[55:71] * cf[0] / (np.abs(t1) * np.abs(t2))**np.arange(15, 0, -1)
        indices = [0, 14, 29, 44, 59, 70]
        cf[indices] = cf[indices] + (t1 * t2).conjugate()
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_99(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)
def poly_giga_100(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for i in range(71):
            z = t1 * np.cos((i+1)*t2/15) + t2 * np.sin((i+1)*t1/15)
            phi = cmath.phase(z)
            r = np.abs(z)
            cf[i] = r * np.exp(1j * phi) ** (i+1) + (-1)**(i+2) * (i+1)**2
        
        cf[:30] = cf[:30] * (np.abs(t1) * np.abs(t2)) ** np.arange(1, 31)
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_101(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for i in range(71):
            cf[i] = (t1/(i+2))**(i+1) + (t2/(i+2))*(2j*(i+1))
        
        even_indices = np.array([1,3,5,7,9,11,13,15,17,19])
        cf[even_indices] = cf[even_indices]*(t1+2*t2)
        
        third_indices = np.array([2,5,8,11,14,17,20,23,26,29])
        cf[third_indices] = cf[third_indices]*(t1-2*t2)
        
        cf[4:35] = cf[4:35] + 2*t1
        cf[35:66] = cf[35:66] - 2*t2
        cf[66:] = np.log(np.abs(cf[66:])).real + np.sum(cf[4])
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_102(t1, t2):
    try:
        n = int(ps.poly.get("n") or 71)
        m = int(np.floor(n/2)+1)
        cf = np.zeros(n, dtype=complex)
        if np.abs(t2)>0:
            cf[:(m-1)] = np.arange(1, m) * np.sin(t1) / np.abs(t2)
        else:
             cf[:(m-1)] = np.arange(1, m) * np.sin(t1)
        cf[m-1] = np.log(np.abs(t1 + t2) + 1)
        if np.abs(t1)>0:
            cf[m:] = np.arange(m-1, 0, -1) * np.cos(t2) / np.abs(t1)
        else:
            cf[m:] = np.arange(m-1, 0, -1) * np.cos(t2)
        nested_pattern = (-1)**np.arange(1, n+1)
        cf = cf * nested_pattern
        return cf.astype(np.complex128)
    except Exception as e:
        print(f"poly_giga_102 error: {e}")
        return np.zeros(n, dtype=complex)


def poly_giga_103(t1, t2):
    try:
        n = int(ps.poly.get("n") or 71)
        cf = np.zeros(n, dtype=complex)
        for k in range(n):
            cf[k] = ((-1)**(k+1)) * ((k+1) / (t1 * np.abs(t1))) * np.abs(np.sin((k+1) * t2))
            if (k+1) % 2 == 0:
                cf[k] = cf[k] + np.abs(t1)**(k+1) * np.cos(t2)**(k+1)
            else:
                cf[k] = cf[k] + np.abs(t2)**(k+1) * np.sin(t1)**(k+1)
            if (k+1) % 3 == 0:
                cf[k] = cf[k] * cmath.phase(t1 + k+1)
            elif (k+1) % 5 == 0:
                cf[k] = cf[k] * cmath.phase(t2 - (k+1))
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_104(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        powers = ((np.arange(1, 71)**2 + t1 * 1j * t2)).real
        cf[1:] = 1 / (1 + powers)
        cf[0] = cf[1] + 100 * np.abs(t1 + t2)
        cf[2::2] = cf[2::2] * np.conj(t1 + t2)
        cf[3::2] = cf[3::2] * np.abs(t1 - t2)
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_105(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        powerValues = np.abs(t1 * t2) * np.arange(1, 71)
        cf[0] = (t1.real*t1.imag)/(t2.real*t2.imag) - (cmath.phase(t1)/cmath.phase(t2)) + np.log(np.abs(t1*t2+1))
        cf[1] = (t1.real*t2.imag)/(t2.real*t1.imag) + (cmath.phase(t1)/cmath.phase(t2)) - np.log(np.abs(t1*t2+1))
        
        for n in range(2, 71):
            cf[n] = np.sin(powerValues[n-2])*np.cos(powerValues[n-2]) + cf[n-1] + cf[n-2]
        
        cf[24:50] = 1 * np.abs(t1) * np.arange(51, 25, -1)
        cf[50:] = -1 * np.abs(t2) * np.arange(1, 22)
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_106(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = np.abs(t1 + t2) ** 2
        cf[1] = t1.real * t2.imag - t2.real * t1.imag
        cf[2] = np.abs(t1) * np.abs(t2) ** 2 - np.abs(t1 - t2) ** 2
        
        for k in range(3, 36):
            cf[k] = (np.sin(k * t1 * t2) + np.cos(k * (t1 + t2))) / np.abs(np.sin(((k + 1) / 2) * (t1 - t2)) + np.cos(((k + 1) / 2) * (t1 + t2))) ** 2
        
        cf[36] = np.log(np.abs(t1) + 1) + np.log(np.abs(t2) + 1)
        
        for k in range(37, 71):
            cf[k] = (t1 * t2).real + (t1 * t2).imag / np.abs(t1 - t2) + np.log(np.abs(t1 * t2) + 1) / np.abs(t1 + t2)
        
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_107(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = t1 * t2 - 1
        cf[4] = (t1**5 - t2**5) / 5
        cf[12] = (t1**7 + t2**7) / np.abs(t1*t2)
        cf[16] = np.log(np.abs(t1 + t2) + 1) * np.sin(t1*t2)
        
        for i in [22,30,42,52]:
            cf[i] = ((t1**i - t2**(70-1j))/i).real
            
        cf[66] = ((np.cos(t1)* np.sin(t2)**2)*((np.abs(t1)*abs(t2))**0.5)).imag - t1
        cf[70] = ((np.cos(t2)**2 * np.sin(t1)) / (np.log(np.abs(t1*t2)+1))).real
        
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_108(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = 100 * (t1 + t2**2)
        cf[1] = 90 * (t1**2 + t2)
        cf[2] = 80 * (t1**3 - t2**2)
        cf[3] = 70 * (t1**4 + t2**3)
        cf[4] = 60 * (t1**5 - t2**4)
        cf[5] = 50 * (t1**6 + t2**5)
        cf[6] = 40 * (t1**7 - t2**6)
        cf[7] = 30 * (t1**8 + t2**7)
        cf[8] = 20 * (t1**9 - t2**8)
        cf[9] = 10 * (t1**10 + t2**9)
        
        for k in range(10, 71):
            v = np.sin(k * np.log(np.abs(cf[k-1]+1))) + np.cos(k * np.log(np.abs(t1*t2+1)))
            cf[k] = np.conj(v) / np.abs(v)
            
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_109(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = 100 * (t1 + t2)
        cf[1] = t1.real ** 2 + t2.imag ** 2
        cf[2] = cmath.phase(t1) + cmath.phase(t2)
        cf[3] = np.abs(t1) / np.abs(t2)
        cf[4] = np.abs(t1 - t2)
        cf[5] = np.sin(t1) * np.cos(t2)
        
        for i in range(6, 70):
            cf[i] = np.abs(cf[i-1] / (i+1)) + cf[i-6] + np.log(np.abs(t1 + t2) + 1)
            
        cf[70] = np.prod(cf[np.array([0,10,20,30,40,50,60])]) / (np.abs(t1 + t2) + 1)
        
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_110(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(71):
            cf[k] = np.sin((k+1)*t1)*np.cos((k+1)*t2) / (np.abs((k+1)*t1)+1)**(1/(k+1))
        
        cf[:10] = cf[:10]**3
        cf[60:] = cf[60:] / cf[10:21]
        cf[30:40] = cf[30:40] * np.conj(t2)
        cf[40:50] = cf[40:50] * np.conj(t1)
        
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_111(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = t1**2 + 2*t2**2 + 1
        cf[1] = t1*t2/2 + 2*t1**3 - 3*t2 + 1j
        
        for k in range(2, 71):
            cf[k] = (7*t1 + t2)/((k+1)*t1 + (k+1)*t2) + np.sin((k+1)*t1) - np.cos((k+1)*t2) + np.abs(t1) * (k+1)**2 - np.abs(t2) * (k+1)**3 + 1
            
        cf[39:50] = cf[39:50]*np.cos(t1+t2) + 1j*np.sin(t1-t2)
        cf[59:] = cf[59:]*np.log(np.abs(t1+t2)+1) - 1j*np.log(np.abs(t1-t2)+1)
        
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)
    

def poly_giga_112(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for i in range(71):
            j = 70 - i
            cf[i] = ((np.real(t1) + np.imag(t1) * j) / np.abs(t2 + (i+1))) * np.sin(np.angle(t1 + t2 * (i+1))) + np.log(np.abs(t1 * t2) + 1) * np.cos(2 * np.pi * (i+1) / 71)
        
        cf[cf == 0] = np.real(t1) ** 2 - np.imag(t1) * np.imag(t2)
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_113(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = (t1**2 + t2**2) / ((t1 + t2)**2)
        cf[1] = 1j * (np.real(t1) + np.real(t2)) / ((np.imag(t1) + np.imag(t2))**2)
        cf[2:10] = [(t1*t2)**(i+1) / (np.abs(t1 - t2) * (i+1)) for i in range(8)]
        cf[10:21] = np.sin((t1 + t2)**2) * (np.real(t1)**3 + np.imag(t2)**3)
        cf[21:31] = np.cos((t1 - t2)**2) * (np.imag(t1)**3 + np.real(t2)**3)
        
        for j in range(31, 43):
            cf[j] = cf[j-1] / np.abs(cf[j-1] + cf[j-2])
            
        cf[43:50] = [np.log(np.abs(t1 + 1j * t2)) * (i+1)**2 for i in range(7)]
        cf[50:55] = [(t1 + t2)**5 / (np.abs(t1 - t2) * (5-i))**2 for i in range(5)]
        cf[55:60] = [(t1 - t2)**4 / (np.abs(t1 * 1j * t2) * (5-i))**2 for i in range(5)]
        cf[60:65] = [(t1 * 1j * t2)**3 / (np.abs(t1 - t2) * (5-i)) for i in range(5)]
        cf[65:70] = [(t1**2 - t2**2) * (i+1)**2 / (np.abs(t1)**4 + np.abs(t2)**4) for i in range(5)]
        cf[70] = (t1 * t2 * (t1 - t2)) / (np.abs(t1 + t2) * np.abs(t1 - t2))
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_114(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for i in range(71):
            cf[i] = np.real(t1) * np.real(t2) * ((i+1)**2)/(np.exp(np.abs(t1)*1j)) + np.imag(t1) * np.imag(t2) * ((i+1)**3)/(np.exp(np.abs(t2)*1j))
        
        cf[1::2] = cf[1::2] * (-1)
        p = np.arange(1, 72)
        cf[p**2 <= 71] = cf[p**2 <= 71] + 1j*abs(t1)*abs(t2)
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_115(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0:35] = np.sin(t1+1j) * np.arange(1, 36) * np.abs(t2)**2
        cf[35:71] = np.log(np.abs(t1)) * np.arange(36, 72) * np.abs(t2)
        
        cf[0:10] = cf[0:10] * t1 + t2
        cf[60:71] = cf[60:71] * (t1 + 1j*t2)
        cf[10:20] = cf[10:20]/t1**2
        cf[20:30] = cf[20:30] * np.conj(t2)**3
        cf[30:40] = cf[30:40] * t1 + 2*np.real(t2)
        cf[40:50] = cf[40:50]/(t1 + 1j*t2 - 1)
        cf[50:60] = cf[50:60] * (3*t1 - 1j*t2)
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_116(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for i in range(71):
            cf[i] = (np.abs(t1) + np.abs(t2)) * np.sin((i+1) / (t1 * t2)) * (i+1)**(1/3)
        
        cf[0:23] = cf[0:23] * ((t1 + t2) / np.abs(t1 - t2))**2
        cf[23:47] = cf[23:47] * np.log(np.abs((t1 + t2)**3) + 1)
        cf[47:71] = cf[47:71] * np.log(np.abs((t1 - t2)**3) + 1)
        
        cf[0] *= 2
        cf[22] *= 2
        cf[46] *= 2
        cf[70] *= 2
        
        cf[1:22] = cf[1:22] / np.real(t1)
        cf[23:46] = cf[23:46] / np.imag(t2)
        cf[2:22] = cf[2:22] / np.abs(cf[2:22])
        cf[24:46] = cf[24:46] / np.abs(cf[24:46])
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_117(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = 1000 * t1**2 * t2**2 + 500 * t1 + 100
        cf[1] = 500 * t1**2 - 1000 * t2 + 300
        cf[2] = 300 * t1**3 - 200 * t2**2 + 100 * t1 - 200
        cf[3] = 200 * t1**4 + 100 * t2**3 - 10 * t1**2 + 20 * t2
        
        for i in range(4, 71):
            cf[i] = ((-1)**(i+1)) * (t1**(i+3) + 2 * t2**(i+2)) / np.log(np.abs(t1)+1) / (i+1)
        
        for j in range(5, 71, 5):
            cf[j] = cf[j] + np.sin(t1 * t2) * 200
            
        for k in [6, 20, 34, 48, 62]:
            cf[k] = cf[k] + np.cos(t1 * t2)**2 * 500
            
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_118(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for i in range(35):
            cf[i] = (i+1)*(t1 + (i+1)*1j*t2)**(1/(i+1))
            cf[70] = np.conj(cf[i])
        
        cf[35] = 2*t1 + 3*abs(t2)
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_119(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        
        for n in range(1, 71):
            cf[n] = ((n+1)**4 * (np.abs(t1)*abs(t2))**((n+1)/70) * np.sin(np.angle(t1)*np.angle(t2))*(n+1)) / (np.abs(t1+t2)**2 * (np.log(np.abs(t1+t2)+1))**(1/(n+1)))
        
        cf[0] = (np.abs(t1*t2)**3 - np.abs(t1+t2)**3 + np.abs(t1-t2)**3) / ((np.log(np.abs(t1+t2)+1))**3 * np.angle(t1+t2))
        
        cf[2:5] = cf[2:5] * np.conj(t1)
        cf[5:10] = cf[5:10] * np.conj(t2)
        cf[12:19] = cf[12:19] * np.abs(t1) * np.abs(t2)
        cf[19:30] = cf[19:30] * (np.imag(np.sin(t1*t2)) + np.real(np.cos(t1*t2))) / (np.abs(t1+t2))
        cf[30:50] = cf[30:50] * (np.imag(t1)*np.log(np.abs(t2)+1) - np.real(t2)*np.log(np.abs(t1)+1)) * np.abs(t1+t2)
        cf[50:71] = cf[50:71] * (np.real(t1)*np.log(np.abs(t2)+1) - np.imag(t2)*np.log(np.abs(t1)+1)) / (np.abs(t1+t2)**2)
        
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)
    

def poly_giga_120(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(71):
            if (k + 1) % 2 == 0:
                cf[k] = (np.log(np.abs(t1)+1)**(k+1) + np.log(np.abs(t2)+1)**(71-(k+1))) * np.sin((k+1)*t1 + (71-(k+1))*t2)
            else:
                cf[k] = (np.log(np.abs(t1)+1)**(k+1) - np.log(np.abs(t2)+1)**(71-(k+1))) * np.cos((k+1)*t1 - (71-(k+1))*t2)
        
        r = np.abs(t1) * np.abs(t2)
        for k in range(49, 71):
            cf[k] = cf[k] * (r ** (k+1-50))
        
        for k in range(14, 35):
            cf[k] = cf[k] * 2 * (r ** (71 - (k+1)))
        
        cf = cf.real + 1j * np.conj(cf).imag
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_121(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = t1.real + 1000j
        cf[1] = np.log(1+abs(t1+t2)) * 1000
        
        for k in range(2, 35):
            cf[k] = (-1)**k * (np.real(t1**k) + np.imag(t2**k)) * 1000 / (k ** 2)
        
        for k in range(35, 70):
            cf[k] = (-1)**(k+1) * (np.abs(t1)**(70-k) + np.abs(np.sin(t1+t2))) / (k ** 2)
        
        cf[70] = np.abs(t1) + np.cos(np.angle(t2)) * 1000
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_122(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(35):
            cf[k] = (t1 + 1j * t2) ** (k+1) + np.log(np.abs(t1+(k+1)*t2)+1) * np.real(t1 * t2)
            cf[70 - k] = (k+1) * (t1 - 1j * t2) ** (k+1) - np.log(np.abs(t2-(k+1)*t1)+1) * np.imag(t1 * t2)
        
        cf[35] = 100 * np.abs(t1) * np.abs(t2)
        cf[36] = 200 * np.angle(t1) * np.angle(t2)
        cf[37:71] = cf[0:34] - cf[37:71]
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_123(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = t1*t2 - 1j * np.abs(t2 - t1)
        
        for ranges in [(1,6), (6,11), (11,16), (16,21), (21,26), (26,31), 
                      (31,36), (36,41), (41,46), (46,51), (51,56), (56,61),
                      (61,66), (66,71)]:
            start, end = ranges
            for k in range(start, end):
                if k % 20 < 5:
                    cf[k] = cf[k-1] + np.sin((k+1) * t1) + np.cos((k+1) * t2)
                elif k % 20 < 10:
                    cf[k] = cf[k-1] + np.log(np.abs(t1 - (k+1))) - np.log(np.abs(t2 - (k+1)))
                elif k % 20 < 15:
                    cf[k] = cf[k-1] - np.sin((k+1) * t1) - np.cos((k+1) * t2)
                else:
                    cf[k] = cf[k-1] - np.log(np.abs(t1 - (k+1))) + np.log(np.abs(t2 - (k+1)))
        
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_124(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(70):
            cf[k] = t1**(k+1) * t2**(k+1)
        
        cf[0] = np.log(np.abs(t1 + t2)) + 1
        cf[1] = np.log(np.abs(t1 * t2)) + 1
        cf[3] = np.abs(t1)**2 + np.abs(t2)**2
        cf[5] = np.abs(t1)**3 - np.abs(t2)**3
        cf[7] = np.abs(t1)**4 + np.abs(t2)**4
        cf[9] = t1**5 - t2**5
        cf[19] = np.abs(t1) * np.sin(np.angle(t2))
        cf[29] = np.abs(t2) * np.real(t1)
        cf[39] = np.abs(t1) * np.imag(t2)
        cf[49] = np.abs(t2) * np.angle(t1)
        cf[59] = np.abs(t1 * t2) * np.cos(np.angle(t2))
        cf[69] = np.abs(t2 * t1) * np.real(t1)
        cf[70] = np.sum(cf[:70])
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_125(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = t1 + t2 + 1
        cf[1:10] = np.real(t1)**2 + np.imag(t2)**2
        cf[10:20] = np.real(t2)**2 + np.imag(t1)**2
        cf[20:30] = np.abs(t1 * t2)**2
        cf[30:40] = np.abs(t1 + t2)**2
        cf[40:50] = np.abs(t1) * np.abs(t2)
        cf[50:60] = np.angle(t1) + np.angle(t2)
        cf[60:70] = np.sin(t1 + t2)
        cf[70] = np.cos(t1 - t2)
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_126(t1, t2):
    try:
        cf = np.zeros(51, dtype=complex)
        cf[0] = np.real(t1) + np.imag(t2)
        cf[1] = np.angle(t1)
        cf[2] = np.abs(t2)
        cf[3] = np.sin(t1) + np.cos(t2)
        cf[4:10] = np.linspace(1, 2, 6)
        cf[10] = np.log(np.abs(t1) + 1) + np.log(np.abs(t2) + 1)
        
        for i in range(11, 51):
            cf[i] = cf[i-1] * np.sin(i * cf[i-2] + np.abs(cf[i-3])) + cf[i-4]
        
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_127(t1, t2):
    try:
        cf = np.zeros(51, dtype=complex)
        primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79])
        
        for i in range(len(primes)):
            cf[i] = t1**primes[i] / (t2 + np.abs(t1) * np.abs(t2))
            cf[i+len(primes)] = t2**primes[i] / (t1 + np.abs(t1) * np.abs(t2))
        
        cf[39] = np.sin(t1)/t2 + np.sin(t2)/t1
        cf[40:43] = np.log(np.abs(t1+t2))+1
        cf[43] = t1*t2 + np.imag(t1)*np.imag(t2)
        cf[44] = t1*t2 - np.real(t1)*np.real(t2)
        cf[45:50] = np.real(t1)*np.imag(t2) - np.imag(t1)*np.real(t2)
        cf[50] = np.abs(t1)*abs(t2)
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_128(t1, t2):
    try:
        cf = np.zeros(51, dtype=complex)
        triangleNums = np.cumsum(np.arange(1, 51))
        cf[0] = t1 + 3 * t2
        
        for k in range(1, 51):
            cf[k] = triangleNums[k-1] * (t1 + t2 * np.log(np.abs(t1) + 1))**(k) + \
                    triangleNums[k-1] * (t2 + t1 * np.log(np.abs(t2) + 1))**(k)
        
        cf[42] = np.real(np.abs(t1)) + np.imag(np.abs(t2))
        cf[20] = np.real(np.abs(t2)) + np.imag(np.abs(t1))
        cf[31] = np.real(np.abs(t1*t2)) + np.imag(np.conj(t1*t2))
        cf[27] = 2 * np.real(t1 - t2) + 2 * np.imag(t1 - t2)
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_129(t1, t2):
    try:
        cf = np.zeros(51, dtype=complex)
        n = np.arange(1, 51)
        cf[0] = t1 + t2
        cf[1] = np.sin(t1) + np.cos(t2)
        cf[2:51] = n * np.log(np.abs(t1 + t2 * np.real(np.conj(t1 * t2)))) * \
                   (np.abs(t1 * t2)**n) * (t1 * np.real(np.conj(t2)))**n
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_130(t1, t2):
    try:
        cf = np.zeros(51, dtype=complex)
        cf[0] = t1 * t2
        for i in range(1, 51):
            cf[i] = (cf[i-1] + (i+1) * t1) / (1 + (i+1) * t2)
        for j in range(25):
            cf[2*j+1] = np.abs(cf[2*j+1] * t1**(j+1) / t2**(j+1))
        cf[29] = np.real(t1 + t2) + np.imag(t1 - t2)
        cf[39] = np.angle(t1) * np.angle(t2)
        cf[49] = np.log(np.abs(t1*t2) + 1)
        cf[50] = cf[0] + np.real(t1**2 + t2**2) - np.imag(t1**2 - t2**2)
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_131(t1, t2):
    try:
        cf = np.zeros(51, dtype=complex)
        cf[0] = 1
        for i in range(1, 51):
            cf[i] = np.sin(i * t1) * np.cos(i**2 * t2) + np.log(np.abs(cf[i-1])**(i+1) * (i+1))
        cf[24] = t1 / t2 + np.abs(t1) * np.abs(t2)
        for i in range(29, 50):
            cf[i] = cf[i] + t1 + (t2 / (i+1)) - t1 * (i+1) - t2 * np.abs(cf[24]) / (t1 + 0.5 * t2)
        cf[49] = np.sum(cf[24:49]) - t2
        cf[50] = np.sum(cf[48:50]) + t1
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_132(t1, t2):
    try:
        cf = np.zeros(51, dtype=complex)
        primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241])
        for i in range(50):
            cf[i] = (primes[i]*t1 + 1j*t2**(i+1))/(1 + np.abs(t1))**(i+1)
        cf[50] = np.sum(cf[:50])
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_133(t1, t2):
    try:
        cf = np.zeros(51, dtype=complex)
        cf[0] = t1 + t2
        cf[1] = np.real(t1**2 - t2**2)
        primes = np.array([2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97])
        for k in range(2, 25):
            cf[k] = np.imag(cf[k-1] * primes[k-2]) * np.angle(t1) * np.abs(t2)
        for k in range(25, 50):
            cf[k] = np.abs(cf[k-1] * primes[k-25]**2) * np.angle(t2) * np.real(t1)
        cf[50] = np.sum(cf) + np.sin(np.real(t2))*np.log(np.abs(t1)+1)
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_134(t1, t2):
    try:
        cf = np.zeros(51, dtype=complex)
        cf[0] = np.real(t1 * t2) + np.imag(t2) * np.real(t1)
        cf[1] = np.abs(t1 * t2) * np.cos(np.angle(t1 + t2))
        for i in range(2, 51):
            cf[i] = cf[i-2] * np.abs(cf[i-1]) * np.sin(np.angle(t1 + t2))
        cf[50] = np.log(np.abs(t1 * t2)) + cf[0] + cf[1]
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_135(t1, t2):
    try:
        cf = np.zeros(51, dtype=complex)
        cf[0:5] = [1, t1, t1**2, t1**3, t1**4]
        cf[5:10] = [1, t2, t2**2, t2**3, t2**4]
        
        cf[10:15] = [1, np.exp(1j * t1), np.exp(2j * t1), np.exp(3j * t1), np.exp(4j * t1)]
        cf[15:20] = [1, np.exp(1j * t2), np.exp(2j * t2), np.exp(3j * t2), np.exp(4j * t2)]
        
        cf[20:30] = [1, np.real(t1+t2), np.imag(t1+t2), np.real(t1*t2), np.imag(t1*t2), 
                     np.real(t1+t2)**2, np.imag(t1+t2)**2, np.real(t1*t2)**2, np.imag(t1*t2)**2, np.abs(t1+t2)]
        cf[30:40] = np.arange(1, 11) * np.abs(t1) * np.abs(t2)
        
        cf[40:50] = [1, np.log(np.abs(t1) + 1), np.log(np.abs(t2) + 1), np.log(np.abs(t1 + t2) + 1),
                     np.log(np.abs(t1 * t2) + 1), np.angle(t1), np.angle(t2), np.abs(t1), np.abs(t2), np.angle(t1 + t2)]
        cf[50] = np.abs(t1+t2) * np.angle(t1*t2)
        
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)


def poly_giga_136(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
        cf[16] = 61 - 5*t1*t2
        cf[17] = 67 + np.abs(t1**2 + t2**2)
        cf[18] = 71 + t1**5 + t2**5
        cf[19] = 73 - np.angle(t1) * np.angle(t2)
        cf[20] = 79 + np.abs(t1**3 + t2**3)
        cf[21] = 83 - t1**6 + t2**6
        cf[22] = 89 + np.sin(t1 + t2)
        cf[23] = np.abs(np.real(t1) * np.imag(t2)) + 97
        cf[24] = 101 + t1*t2**2
        cf[25] = 103 - np.conj(t1) * np.real(t2)
        cf[26] = 107 + t1**7 - t2**7
        cf[27] = 109 + np.abs(np.conj(t1-t2))
        cf[28] = 113 - np.abs(t1**2 - t2**2)
        cf[29] = 127 + (t1**8 * t2**8)
        cf[30] = t1 - t2 + np.abs(t1*t2) + 131
        cf[31] = 137 + np.angle(t1**2) - np.angle(t2**2)
        cf[32] = 139 - t1**9 + t2**9
        cf[33] = np.log(np.abs(t1*t2) + 1) + 149
        cf[34] = 151 + (np.abs(t1) + np.abs(t2))**2
        cf[35] = np.sin(2*t1) - np.cos(2*t2) + 157
        cf[36] = np.log(np.abs(t1-t2) + 1) + 163
        cf[37] = 167 + np.real(t1**3) - np.imag(t2**3)
        cf[38] = 173 - (t1**2 * t2**2)**1.5
        cf[39] = 179 + np.angle(t1*t2) + 1j
        cf[40] = 181 - np.conj(t1**3 - t2**3)
        cf[41] = 191 + np.abs(t1) * np.abs(t2)
        cf[42] = 193 - np.abs(np.real(t1) + np.imag(t2))
        cf[43] = 197 + np.sin(t1**2 + t2**2)
        cf[44] = 199 - t1*t2**3
        cf[45] = t1*np.imag(t2) + 211
        cf[46] = np.abs(t1**4 + t2**4) + 223
        cf[47] = 227 - np.conj(t1**2) * np.conj(t2**2)
        cf[48] = 229 + np.sin(t1*t2) - np.cos(t1-t2)
        cf[49] = 233 + t1**9 - t2**9
        cf[50] = 239 - np.abs(np.conj(t1**2+t2**2))
        cf[51] = 241 + t1**3 + t2**3
        cf[52] = t1**10 + t2**10 + 251
        cf[53] = t1*t2*np.real(t1+t2) - 257
        cf[54] = np.abs(t1-t2) - 263
        cf[55] = t1**11 - t2**11 + 269
        cf[56] = 271 + np.abs(t1*t2**2 - t2**3)
        cf[57] = 277 + np.sin(t1**3 - t2**3)
        cf[58] = 281 - np.conj(t1**2*t2)
        cf[59] = np.conj(t1**5 + t2**5) + 283
        cf[60] = np.angle(t1**3 * t2**3) + 293
        cf[61] = 307 - np.sin(t1*t2 + 1j)
        cf[62] = np.abs(t1**6 + t2**6) + 311
        cf[63] = 313 - np.cos(t1**3-t2**3)
        cf[64] = np.angle(t1*t2) + 317
        cf[65] = np.real(t1**2-t2**2) - 331
        cf[66] = 337 + np.abs(t1**6 * t2**6)
        cf[67] = 347 - np.abs(t1**4 - t2**4)
        cf[68] = 349 + np.sin(np.conj(t1-t2))
        cf[69] = 353 - np.cos(t1+t2**2)
        cf[70] = np.abs((t1+t2)**3 - 359)
        return cf.astype(np.complex128)
    except Exception:
        size = locals().get("n")
        if size is None:
            size = len(locals().get("cf", []))
        return np.zeros(size, dtype=complex)



       

