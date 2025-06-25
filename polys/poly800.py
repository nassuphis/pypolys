from . import polystate as ps
import math
import cmath
import numpy as np
from scipy.special import sph_harm
from . import letters
from . import zfrm

pi = math.pi

def poly_701(t1, t2):
    try:
        degree = 8
        cf = np.zeros(degree +1, dtype=complex)
        for j in range(0, degree +1):
            r_part = t1.real * np.log(j + 2) + t2.real * np.sqrt(j +1)
            i_part = t1.imag * np.sin(j * np.pi /4) + t2.imag * np.cos(j * np.pi /3)
            magnitude = np.abs(t1)**(j %3 +1) + np.abs(t2)**(degree - j %2 +1)
            angle = np.angle(t1) * j + np.angle(t2) * (degree - j)
            cf[j] = (r_part +1j * i_part) * np.exp(1j * angle) + np.log(np.abs(magnitude) +1)
        return cf.astype(np.complex128)
    except:
        return np.zeros(71, dtype=complex)

def poly_702(t1, t2):
    try:
        cf = np.zeros(9, dtype=complex)
        rec = np.linspace(t1.real, t2.real, 9)
        imc = np.linspace(t1.imag, t2.imag, 9)
        for j in range(1,10):
            magnitude = np.log(np.abs(rec[j-1] + imc[j-1]) + 1) * np.sin(j * np.pi /4) + np.cos(j * np.pi /6)
            angle = np.angle(t1) * np.sin(2 * np.pi * j /9) + np.angle(t2) * np.cos(4 * np.pi * j /9)
            cf[j-1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
        return cf.astype(np.complex128)
    except:
        return np.zeros(9, dtype=complex)

def poly_703(t1, t2):
    try:
        cf = np.zeros(9, dtype=complex)
        for j in range(1,10):
            mag = np.log(np.abs(t1) + np.abs(t2) + j) * (j**2 + np.sin(j * t1.real) + np.cos(t2.imag))
            angle = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j)
            cf[j-1] = mag * (np.cos(angle) + 1j * np.sin(angle)) + np.conj(t1) * t2.real * j
        return cf.astype(np.complex128)
    except:
        return np.zeros(9, dtype=complex)

def poly_704(t1, t2):
    try:
        degree = 8
        cf = np.zeros(degree +1, dtype=complex)
        for j in range(0, degree +1):
            r = j +1
            mag = np.log(np.abs(t1) * j +1) * (np.abs(t2)**(degree - j +1)) + np.sin(j * t1.real)**2
            ang = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j)
            real_part = mag * np.cos(ang) + t1.real**(j %3)
            imag_part = mag * np.sin(ang) + t2.imag**(j %2 +1)
            cf[j] = mag * np.exp( 1j * ang ) + ( np.conj(t1)**j ) * np.cos( j * np.real(t2) )
        return cf.astype(np.complex128)
    except:
        return np.zeros(71, dtype=complex)

def poly_705(t1, t2):
    try:
        cf = np.zeros(9, dtype=complex)
        for j in range(1,10):
            mag_variation = np.log(np.abs(t1) + np.abs(t2) + j) * (1 + np.sin(j * t1.real) * np.cos(j * t2.imag))
            angle_variation = np.angle(t1) * np.sqrt(j) - np.angle(t2) / (j +1)
            cf[j-1] = mag_variation * np.exp(1j * angle_variation) + np.conj(t1) * np.sin(j) - np.conj(t2) * np.cos(j)
        return cf.astype(np.complex128)
    except:
        return np.zeros(9, dtype=complex)

def poly_706(t1, t2):
    try:
        cf = np.zeros(9, dtype=complex)
        for j in range(1,10):
            mag = 0
            angle = 0
            for k in range(1,j+1):
                mag += np.log(np.abs(t1.real + k)) * np.sin(k * t2.real)
                angle += np.cos(k * t1.imag) * np.angle(t2)**k
            cf[j-1] = mag * (np.cos(angle) + 1j * np.sin(angle)) + np.conj(t1)**j + np.conj(t2)**j
        return cf.astype(np.complex128)
    except:
        return np.zeros(9, dtype=complex)

def poly_707(t1, t2):
    try:
        cf = np.zeros(9, dtype=complex)
        rec = np.linspace(t1.real, t2.real, 9)
        imc = np.linspace(t1.imag, t2.imag, 9)
        for j in range(1,10):
            mag = np.exp(np.sin(5 * np.pi * imc[j-1])) + np.log(np.abs(rec[j-1]) + 1) * np.cos(3 * np.pi * rec[j-1])
            ang = np.angle(t1) * np.sin(2 * np.pi * j /9) + np.angle(t2) * np.cos(4 * np.pi * j /9)
            cf[j-1] = mag * (np.cos(ang) +1j * np.sin(ang))
        return cf.astype(np.complex128)
    except:
        return np.zeros(9, dtype=complex)

def poly_708(t1, t2):
    try:
        cf = np.zeros(9, dtype=complex)
        for j in range(1,10):
            k = j**2
            r = t1.real * np.cos(k * np.angle(t2)) + np.sin(k * t1.imag)
            im = t2.imag * np.log(k + np.abs(t1)) + np.cos(k * t1.real)
            mag = np.sqrt(r**2 + im**2) * (1 + j)
            angle = np.arctan2(im, r) + np.sin(j * np.angle(t1 + t2))
            cf[j-1] = mag * (np.cos(angle) +1j * np.sin(angle))
        return cf.astype(np.complex128)
    except:
        return np.zeros(9, dtype=complex)

def poly_709(t1, t2):
    try:
        degree = 8
        cf = np.zeros(degree +1, dtype=complex)
        for j in range(0, degree +1):
            mag_part = np.log(np.abs(t1) + np.abs(t2) +1) * (j +1)**np.sin(j * t1.real) + np.cos(j * t2.real)
            angle_part = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j)
            cf[j] = mag_part * np.exp(1j * angle_part)
        for k in range(0, degree +1):
            cf[k] = cf[k] * (1 + 0.5 * np.sin(k * t1.imag) - 0.3 * np.cos(k * t2.imag))
        cf = cf * (1 + 0.1 * t1.real) / (1 + 0.1 * t2.imag)
        return cf.astype(np.complex128)
    except:
        return np.zeros(71, dtype=complex)

def poly_710(t1, t2):
    try:
        degree = 8
        cf = np.zeros(degree +1, dtype=complex)
        for j in range(1, degree +2):
            sum_part = 0
            for k in range(1, j +1):
                sum_part += np.cos(k * t1.real) * np.sin(k * t2.imag)
            mag = np.log(np.abs(t1) + np.abs(t2) + sum_part +1)
            angle = (np.angle(t1)**0.5 * j) + (np.angle(t2)**0.3 * (degree - j +1))
            cf[j-1] = mag * (np.cos(angle) +1j * np.sin(angle))
        return cf.astype(np.complex128)
    except:
        return np.zeros(71, dtype=complex)

def poly_711(t1, t2):
    try:
        cf = np.zeros(8, dtype=complex)
        for k in range(1,9):
            r_part = t1.real**k + np.log(np.abs(t2))**2
            i_part = np.cos(k * t2.imag) + np.sin(k * np.angle(t1))
            angle = np.angle(t1) * k / 2 + np.sin(k) * np.pi /3
            cf[k-1] = (r_part + 1j * i_part) * (np.cos(angle) +1j * np.sin(angle))
        return cf.astype(np.complex128)
    except:
        return np.zeros(8, dtype=complex)

def poly_712(t1, t2):
    try:
        degree = 8
        cf = np.zeros(degree +1, dtype=complex)
        rec = np.linspace(t1.real, t2.real, degree +1)
        imc = np.linspace(t1.imag, t2.imag, degree +1)
        for j in range(1, degree +2):
            angle_component = np.sin(j * np.pi /3) * np.cos(j * np.pi /4)
            magnitude_component = np.log(np.abs(rec[j-1] * imc[j-1]) +1) + t1.real**j - t2.imag**(degree +1 -j)
            cf[j-1] = magnitude_component * np.exp(1j * (angle_component + np.angle(t1) * j))
        return cf.astype(np.complex128)
    except:
        return np.zeros(71, dtype=complex)

def poly_713(t1, t2):
    try:
        cf = np.zeros(8, dtype=complex)
        for j in range(1,9):
            k = j * 2
            r = t1.real**j + np.log(np.abs(t2) +1) * np.sin(j * np.angle(t1))
            i_part = t2.imag**k - np.log(np.abs(t1) +1) * np.cos(j * np.angle(t2))
            mag = np.log(np.abs(t1)*np.abs(t2) +1) * j
            ang = np.angle(t1) + np.angle(t2) *j + np.sin(j) - np.cos(k)
            cf[j-1] = (r +1j * i_part) * mag * (np.cos(ang) +1j * np.sin(ang)) + np.conj(t1)**j * np.conj(t2)**k
        return cf.astype(np.complex128)
    except:
        return np.zeros(8, dtype=complex)

def poly_714(t1, t2):
    try:
        cf = np.zeros(9, dtype=complex)
        for j in range(1,10):
            r = t1.real * j**2 - t2.real / (j +1)
            im = t1.imag + t2.imag * np.sin(j)
            mag = np.log(np.abs(t1) + np.abs(t2) + j) * (1 + np.abs(np.sin(j * np.pi /3)))
            ang = np.angle(t1) * np.cos(j) + np.angle(t2) * np.sin(j * np.pi /4)
            cf[j-1] = mag * np.exp(1j * ang) + np.conj(t1) * np.sin(j) - np.conj(t2) * np.cos(j)
        return cf.astype(np.complex128)
    except:
        return np.zeros(9, dtype=complex)


def poly_715(t1, t2):
    try:
        cf = np.zeros(9, dtype=complex)
        for j in range(1,10):
            rj = t1.real + j * t2.real
            ij = t1.imag - j * t2.imag
            mag = np.log(np.abs(rj + ij) +1) * (1 + np.sin(j * np.pi / 4 )) * (j**1.5)
            ang = np.angle(t1) * np.cos(j /3) + np.angle(t2) * np.sin(j /5)
            cf[j-1] = mag * np.exp(1j * ang)
        return cf.astype(np.complex128)
    except:
        return np.zeros(9, dtype=complex)

def poly_716(t1, t2):
    try:
        cf = np.zeros(9, dtype=complex)
        for j in range(1,10):
            mag = np.log(np.abs(t1) + j) * (np.sin(j * np.angle(t1)) + np.cos(j * np.angle(t2)))
            angle = np.angle(t1) * np.sqrt(j) + np.angle(t2) / (j +1)
            cf[j-1] = mag * np.exp(1j * angle)
        return cf.astype(np.complex128)
    except:
        return np.zeros(9, dtype=complex)

def poly_717(t1, t2):
    try:
        cf = np.zeros(9, dtype=complex)
        for j in range(1,10):
            mag = np.log(np.abs(t1)**j + np.abs(t2)**(j/2)) + np.sum(t1.real * t2.imag)
            angle = np.angle(t1) * np.cos(j) + np.angle(t2) * np.sin(j) + np.sin(j * t1.real) * np.cos(j * t2.imag)
            cf[j-1] = mag * np.exp(1j * angle) + np.conj(t1 + t2) / (j +1)
        return cf.astype(np.complex128)
    except:
        return np.zeros(9, dtype=complex)

def poly_718(t1, t2):
    try:
        cf = np.zeros(9, dtype=complex)
        for j in range(1,10):
            k = j + 2
            r = t1.real * np.cos(j) - t2.imag * np.sin(j)
            im = t2.real * np.sin(j) + t1.imag * np.cos(j)
            magnitude = np.log(np.abs(t1) + np.abs(t2) + j) * (1 + np.sin(k))+np.cos(k)
            angle = np.angle(t1) * np.sin(k) + np.angle(t2) * np.cos(k)
            cf[j-1] = magnitude * (np.cos(angle) +1j * np.sin(angle))
        return cf.astype(np.complex128)
    except:
        return np.zeros(9, dtype=complex)

def poly_719(t1, t2):
    try:
        degree = 8
        cf = np.zeros(degree +1, dtype=complex)
        for j in range(0, degree +1):
            k = j +1
            r = t1.real * np.sin(j * t1.real) + t2.imag * np.cos(k * t2.real)
            mag = np.log(np.abs(t1) + np.abs(t2) + j +1) * (j +1)**2
            angle = np.sin(r) + np.cos(k * np.pi /4)
            cf[j] = mag * (np.cos(angle) +1j * np.sin(angle))
        return cf.astype(np.complex128)
    except:
        return np.zeros(71, dtype=complex)

def poly_720(t1, t2):
    try:
        degree = 8
        cf = np.zeros(degree +1, dtype=complex)
        for k in range(0, degree +1):
            mag_part = np.abs(t1)**k * np.abs(t2)**(degree -k) + np.log(np.abs(t1) + np.abs(t2) +1)
            angle_part = np.angle(t1) * np.sin(k) + np.angle(t2) * np.cos(k)
            variation = np.sin(k * np.pi /3) + np.cos(k * np.pi /4)
            cf[k] = (mag_part * variation) * (np.cos(angle_part) +1j * np.sin(angle_part))
        return cf.astype(np.complex128)
    except:
        return np.zeros(71, dtype=complex)

def poly_721(t1, t2):
    try:
        degree = 8
        cf = np.zeros(degree +1, dtype=complex)
        for j in range(1, degree +2):
            angle = np.sin(j * t1.real) + np.cos(j * t2.imag) + np.angle(t1) * j
            mag = np.log(j +1) * np.abs(t1)**j + np.abs(t2)**(degree +1 -j)
            cf[j-1] = mag * (np.cos(angle) +1j * np.sin(angle))
        for k in range(1, degree +2):
            cf[k-1] = cf[k-1] * np.conj(t1) / (1 + k)
        return cf.astype(np.complex128)
    except:
        return np.zeros(71, dtype=complex)

def poly_722(t1, t2):
    try:
        degree = 8
        cf = np.zeros(degree +1, dtype=complex)
        for j in range(1,10):
            mag_part = np.log(np.abs(t1)**j + np.abs(t2)**(9-j) + 1)
            angle_part = np.sin(j * np.angle(t1)) + np.cos((9-j) * np.angle(t2))
            real_part = mag_part * np.cos(angle_part) + t1.real * np.abs(t2)**j
            imag_part = mag_part * np.sin(angle_part) + t2.imag * np.abs(t2)**j
            cf[j-1] = complex(real_part, imag_part)
        return cf.astype(np.complex128)
    except:
        return np.zeros(71, dtype=complex)

def poly_723(t1, t2):
    try:
        cf = np.zeros(9, dtype=complex)
        for j in range(1,10):
            mag_part = np.log(np.abs(t1) + j) * np.sin(j * np.angle(t2)) + np.cos(j * t1.imag)
            angle_part = np.angle(t1)**j - np.angle(t2)**(j %3) + np.sin(j * t1.real)
            cf[j-1] = mag_part * np.exp(1j * angle_part)
        return cf.astype(np.complex128)
    except:
        return np.zeros(9, dtype=complex)

def poly_724(t1, t2):
    try:
        degree = 8
        cf = np.zeros(degree +1, dtype=complex)
        for j in range(1, degree +2):
            mag_part = np.abs(t1)**j * np.log(np.abs(t2) +1) + np.abs(t2)**(degree -j +1) * np.sin(j * t1.real)
            angle_part = np.angle(t1) * np.cos(j * t2.real) + np.angle(t2) * np.sin(j * t1.real)
            cf[j-1] = mag_part * np.exp(1j * angle_part) + np.conj(t1 * t2) / (j +1)
        return cf.astype(np.complex128)
    except:
        return np.zeros(71, dtype=complex)

def poly_725(t1, t2):
    try:
        cf = np.zeros(9, dtype=complex)
        for j in range(1,10):
            k = j %5 +1
            r = np.log(np.abs(t1) + j) * (1 + np.sin(j * t2.real) + np.cos(k * t1.imag))
            angle = np.angle(t1) * np.cos(j) + np.angle(t2) * np.sin(k)
            cf[j-1] = r * (np.cos(angle) +1j * np.sin(angle))
        return cf.astype(np.complex128)
    except:
        return np.zeros(9, dtype=complex)

def poly_726(t1, t2):
    try:
        cf = np.zeros(9, dtype=complex)
        for k in range(1,10):
            mag = np.log(np.abs(t1) + k**2) * np.sin(k * np.angle(t2)) + np.cos(k * t1.real)
            angle = np.angle(t1) * np.log(np.abs(t2) +1) + np.sin(k * t2.imag)
            cf[k-1] = mag * (np.cos(angle) +1j * np.sin(angle)) * (-1)**k
        return cf.astype(np.complex128)
    except:
        return np.zeros(9, dtype=complex)
    
#function(t1, t2) {
#  n = 9
#  cf = complex(n)
#  rec = seq(Re(t1), Re(t2), length.out = n)
#  imc = seq(Im(t1), Im(t2), length.out = n)
#  for (k in 1:n) {
#    mag = log(Mod(t1) + Mod(t2) + k) * (k^2)
#    angle = Arg(t1) * sin(k) + Arg(t2) * cos(k)
#    cf[k] = mag * (cos(angle) + 1i * sin(angle))
#  }
#  cf

def bat(t1, t2):
    n = ps.poly.get('bat') or 11
    cf = np.arange(1, n + 1)
    mag = np.log(np.abs(t1) + np.abs(t2) + cf) * cf * cf
    angle = np.angle(t1) * np.sin(cf) + np.angle(t2) * np.cos(cf)
    return mag * (np.cos(angle) + 1j * np.sin(angle))

def batman(t1, t2, n=9):
    cf = np.arange(1, n + 1)
    mag = np.log(np.abs(t1) + np.abs(t2) + cf) * cf * cf
    angle = np.angle(t1) * np.sin(cf) + np.angle(t2) * np.cos(cf)
    return mag * (np.cos(angle) + 1j * np.sin(angle))

def batman10(t1, t2, n=10):
    cf = np.arange(1, n + 1)
    mag = np.log(np.abs(t1) + np.abs(t2) + cf) * cf * cf
    angle = np.angle(t1) * np.sin(cf) + np.angle(t2) * np.cos(cf)
    return mag * (np.cos(angle) + 1j * np.sin(angle))

def batman11(t1, t2, n=11):
    cf = np.arange(1, n + 1)
    mag = np.log(np.abs(t1) + np.abs(t2) + cf) * cf * cf
    angle = np.angle(t1) * np.sin(cf) + np.angle(t2) * np.cos(cf)
    return mag * (np.cos(angle) + 1j * np.sin(angle))

def batman12(t1, t2, n=12):
    cf = np.arange(1, n + 1)
    mag = np.log(np.abs(t1) + np.abs(t2) + cf) * cf * cf
    angle = np.angle(t1) * np.sin(cf) + np.angle(t2) * np.cos(cf)
    return mag * (np.cos(angle) + 1j * np.sin(angle))

def batman31(t1, t2, n=31):
    cf = np.arange(1, n + 1)
    mag = np.log(np.abs(t1) + np.abs(t2) + cf) * cf * cf
    angle = np.angle(t1) * np.sin(cf) + np.angle(t2) * np.cos(cf)
    return mag * (np.cos(angle) + 1j * np.sin(angle))

def batman51(t1, t2, n=51):
    cf = np.arange(1, n + 1)
    mag = np.log(np.abs(t1) + np.abs(t2) + cf) * cf * cf
    angle = np.angle(t1) * np.sin(cf) + np.angle(t2) * np.cos(cf)
    return mag * (np.cos(angle) + 1j * np.sin(angle))


def poly_727_old(t1, t2):
        try:
            n = 9
            cf = np.zeros(n, dtype=complex)
            rec = np.linspace(t1.real, t2.real, n)
            imc = np.linspace(t1.imag, t2.imag, n)
            for k in range(1, n+1):
                mag = np.log( np.abs(t1) + np.abs(t2) + k ) * (k**2)
                angle = np.angle(t1) * np.sin(k) + np.angle(t1) * np.cos(k)
                cf[k-1] = mag * (np.cos(angle) +1j * np.sin(angle))
            return cf.astype(np.complex128)
        except:
            return np.zeros(9, dtype=complex)

def poly_727(t1, t2):
    try:
        n = ps.poly.get('n') or 9
        cf = np.zeros(n, dtype=complex)
        rec = np.linspace(t1.real, t2.real, n)
        imc = np.linspace(t1.imag, t2.imag, n)
        for k in range(1, n+1):
            mag = np.log( np.abs(t1) + np.abs(t2) + k ) * (k**2)
            angle = np.angle(t1) * np.sin(k) + np.angle(t2) * np.cos(k)
            cf[k-1] = mag * (np.cos(angle) +1j * np.sin(angle))
        return cf.astype(np.complex128)
    except:
        return np.zeros(9, dtype=complex)

def poly_727_v1(t1, t2):
    try:
        n = 9
        cf = np.zeros(n, dtype=complex)
        rec = np.linspace(t1.real, t2.real, n)
        imc = np.linspace(t1.imag, t2.imag, n)
        for k in range(n):
            mag = np.log( np.abs(t1) + np.abs(t2) + k ) * (k**2)
            angle = np.angle(t1) * np.sin(k) + np.angle(t2) * np.cos(k)
            cf[k] = mag * (np.cos(angle) +1j * np.sin(angle))
        return cf.astype(np.complex128)
    except:
        return np.zeros(9, dtype=complex)
        
def poly_727a(t1, t2):
    try:
        n = 9
        cf = np.zeros(n, dtype=complex)
        rec = np.linspace(t1.real, t2.real, n)
        imc = np.linspace(t1.imag, t2.imag, n)
        for k in range(1, n+1):
            mag = np.log( np.abs(t1) + np.abs(t2) + k ) * (k**3)
            angle = np.angle(t1) * np.sin(k) + np.angle(t1) * np.cos(k)
            cf[k-1] = mag * (np.cos(angle) +1j * np.sin(angle))
        return cf.astype(np.complex128)
    except:
        return np.zeros(9, dtype=complex)    
    
def poly_727b(t1, t2):
    try:
        n = 13
        cf = np.zeros(n, dtype=complex)
        rec = np.linspace(t1.real, t2.real, n)
        imc = np.linspace(t1.imag, t2.imag, n)
        for k in range(1, n+1):
            mag = np.log( np.abs(t1) + np.abs(t2) + k ) * (k**3)
            angle = np.angle(t1) * np.sin(k) + np.angle(t1) * np.cos(k)
            cf[k-1] = mag * (np.cos(angle) +1j * np.sin(angle))
        return cf.astype(np.complex128)
    except:
        return np.zeros(9, dtype=complex)  

def poly_727c(t1, t2):
    try:
        n = 29
        cf = np.zeros(n, dtype=complex)
        rec = np.linspace(t1.real, t2.real, n)
        imc = np.linspace(t1.imag, t2.imag, n)
        for k in range(1, n+1):
            mag = np.log( np.abs(t1) + np.abs(t2) + k ) * (k**3)
            angle = np.angle(t1) * np.sin(k) + np.angle(t1) * np.cos(k)
            cf[k-1] = mag * (np.cos(angle) +1j * np.sin(angle))
        return cf.astype(np.complex128)
    except:
        return np.zeros(9, dtype=complex)

def poly_727d(t1, t2):
    try:
        n = 29
        cf = np.zeros(n, dtype=complex)
        rec = np.linspace(t1.real, t2.real, n)
        imc = np.linspace(t1.imag, t2.imag, n)
        for k in range(1, n+1):
            mag = np.log( np.abs(t1) + np.abs(t2) + k ) * (k**3)
            angle = np.angle(t1) * np.sin(k) + np.angle(t1) * np.cos(k)
            cf[k-1] = mag * (np.cos(angle) +1j * np.sin(angle))
        return cf.astype(np.complex128)
    except:
        return np.zeros(9, dtype=complex)  
            
def poly_727_alt(t1, t2):
    try:
        n = 9
        cf = np.zeros(n, dtype=complex)
        rec = np.linspace(t1.real, t2.real, n)
        imc = np.linspace(t1.imag, t2.imag, n)
        for k in range(1, n+1):
            mag = np.log( np.abs(t1) + np.abs(t2) + k ) * (k**2)
            angle = np.sin(k * np.angle(t1)) + np.cos(k)
            cf[k-1] = mag * (np.cos(angle) +1j * np.sin(angle))
        return cf.astype(np.complex128)
    except:
        return np.zeros(9, dtype=complex)

def poly_728(t1, t2):
    try:
        cf = np.zeros(9, dtype=complex)
        for j in range(1,10):
            mag = np.log(np.abs(t1) + j) * np.abs(np.sin(j * t1.real)) + np.sqrt(np.abs(t2.imag) + j)
            ang = np.angle(t1) * np.cos(j * np.angle(t2)) + np.sin(j * t2.real)
            cf[j-1] = mag * np.exp(1j * ang)
        return cf.astype(np.complex128)
    except:
        return np.zeros(9, dtype=complex)

def poly_729(t1, t2):
    try:
        degree = 8
        cf = np.zeros(degree +1, dtype=complex)
        for j in range(0, degree +1):
            r_part = t1.real * j**2 - t2.real * np.sqrt(j +1)
            im_part = (t1.imag + t2.imag) * np.log(j +2)
            magnitude = np.abs(t1)**(j %3 +1) + np.abs(t2)**(degree -j)
            angle = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j)
            cf[j] = (r_part +1j * im_part) * magnitude * np.exp(1j * angle)
        return cf.astype(np.complex128)
    except:
        return np.zeros(71, dtype=complex)

def poly_730(t1, t2):
    try:
        cf = np.zeros(9, dtype=complex)
        for j in range(1,10):
            mag_part = np.log(np.abs(t1) +1) * (j +1)**np.sin(j * t1.real) + np.cos(j * t2.real)
            angle_part = np.angle(t1) * np.sin(j) + np.angle(t2) * np.cos(j)
            cf[j-1] = mag_part * np.exp(1j * angle_part)
        return cf.astype(np.complex128)
    except:
        return np.zeros(9, dtype=complex)

def poly_731(t1, t2):
    try:
        cf = np.zeros(9, dtype=complex)
        for j in range(1,10):
            mag_part = np.log(np.abs(t1) * j +1) * np.abs(np.sin(t1.real * j) + np.cos(t2.imag / (j +1)))
            angle_part = np.angle(t1) * np.sqrt(j) + np.angle(t2) / (j +2)
            cf[j-1] = mag_part * (np.cos(angle_part) +1j * np.sin(angle_part))
        return cf.astype(np.complex128)
    except:
        return np.zeros(9, dtype=complex)

def poly_732(t1, t2):
    try:
        cf = np.zeros(9, dtype=complex)
        for k in range(0,9):
            r_part = t1.real * np.log(k +2) + t2.real * np.sqrt(k +1)
            i_part = t1.imag * np.sin(k) - t2.imag * np.cos(k)
            angle = np.sin(r_part) + np.cos(i_part)
            magnitude = np.log(np.abs(t1) + np.abs(t2) +k +1) * (k +1)
            cf[k] = magnitude * (np.cos(angle) +1j * np.sin(angle)) + np.conj(t1) * np.conj(t2)**k
        return cf.astype(np.complex128)
    except:
        return np.zeros(9, dtype=complex)

def poly_733(t1, t2):
    try:
        cf = np.zeros(9, dtype=complex)
        for k in range(1,10):
            tmp = 0 +0j
            for j in range(1,k+1):
                tmp += (t1.real**j / (j +1)) * np.exp(1j * np.sin(j * t2.real))
            for r in range(1, (k %3)+2):
                tmp += (t2.imag**r / (r +2)) * np.exp(1j * np.cos(r * t1.imag))
            cf[k-1] = tmp
        return cf.astype(np.complex128)
    except:
        return np.zeros(9, dtype=complex)

def poly_734(t1, t2):
    try:
        cf = np.zeros(9, dtype=complex)
        for j in range(1,10):
            mag = np.log(np.abs(t1) + j**2) * np.sin(j * np.angle(t2)) + np.cos(j * t1.real * t2.imag)
            angle = np.angle(t1) * np.cos(j * t2.real) + np.sin(j) * np.log(np.abs(t2) +1)
            cf[j-1] = mag * np.exp(1j * angle) + np.conj(t1) * t2**j
        return cf.astype(np.complex128)
    except:
        return np.zeros(9, dtype=complex)

def poly_735(t1, t2):
    try:
        cf = np.zeros(9, dtype=complex)
        for j in range(1,10):
            real_part = t1.real**j + np.log(np.abs(t2) +1) *j
            imag_part = t2.imag * np.sin(j * np.angle(t1)) + np.cos(j * t2.real)
            magnitude = np.log(np.abs(t1 + t2) +j) * (1 +j**2)
            angle = np.angle(t1) * np.cos(j) + np.sin(j * np.angle(t2)) - np.cos(j * t1.imag)
            cf[j-1] = magnitude * (np.cos(angle) +1j * np.sin(angle))
        return cf.astype(np.complex128)
    except:
        return np.zeros(9, dtype=complex)

def poly_736(t1, t2):
    try:
        cf = np.zeros(9, dtype=complex)
        for j in range(1,10):
            mag_part = np.log(np.abs(t1) + j) * np.sin(j * np.angle(t2)) + np.cos(j * t1.real)
            ang_part = np.angle(t1) * np.cos(j) + np.sin(j * np.angle(t2))
            cf[j-1] = mag_part * np.exp(1j * ang_part)
        return cf.astype(np.complex128)
    except:
        return np.zeros(9, dtype=complex)

def poly_737(t1, t2):
    try:
        cf = np.zeros(9, dtype=complex)
        rec = np.linspace(t1.real, t2.real, 9)
        imc = np.linspace(t1.imag, t2.imag, 9)
        for k in range(1,10):
            mag = np.log(np.abs(t1) + np.abs(t2) +k) * (1 + np.sin(3 * np.pi * rec[k-1]) + np.cos(2 * np.pi * imc[k-1]))
            ang = np.angle(t1) * np.sin(5 * np.pi * imc[k-1]) - np.angle(t2) * np.cos(4 * np.pi * rec[k-1])
            cf[k-1] = mag * (np.cos(ang) +1j * np.sin(ang))
        return cf.astype(np.complex128)
    except:
        return np.zeros(9, dtype=complex)

def poly_738(t1, t2):
    try:
        r = t1.real * t2.imag
        i = t1.imag * t2.real
        
        cf = np.zeros(10, dtype=complex)
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
    except:
        return np.zeros(10, dtype=complex)

def poly_739(t1, t2):
    try:
        cf = np.zeros(11, dtype=complex)
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
    except:
        return np.zeros(11, dtype=complex)

def poly_740(t1, t2):
    try:
        cf = np.zeros(11, dtype=complex)
        r_1 = t1.real
        r_2 = t2.real
        i_1 = t1.imag
        i_2 = t2.imag
        
        cf[0] = i_1**3 - i_2**3 + r_1**2 - r_2**2
        cf[1] = r_1 * i_1 * r_2 * i_2 * (r_1 -1j*i_1) 
        cf[2] = i_1**2 + r_2**2 - i_2**2 + (r_2 - i_1)
        cf[3] = r_1 * (r_1**2 * i_1**2 - r_2**2 * i_2**2)
        cf[4] = r_1**3 + r_1**2 + i_2**3 + i_1**2 - 10
        cf[5] = i_1 * i_2 * (i_1**2 * r_2**2 - i_2**2 * r_1**2)
        cf[6] = r_1**0.5 - i_2**0.5 + r_2**0.5 - i_1**0.5
        cf[7] = r_1 * i_1**2 - r_2 * i_2 * (r_1 -1j*i_1)
        cf[8] = i_1**3 - i_2**3 + r_2**2 - r_1**2
        cf[9] = r_1 * i_1 * r_2 * i_2 * (i_2 - r_1) 
        cf[10] = i_1**4 + i_2**4 + r_1**4 + r_2**4
        return cf.astype(np.complex128)
    except:
        return np.zeros(11, dtype=complex)

def poly_741(t1, t2):
    try:
        n = 10
        cf = np.zeros(n, dtype=complex)
        cf[0] = np.real((t1 +7j)**2) + np.imag((t2 +5j)**3)
        cf[1] = np.exp(1j*t1*t2)
        cf[2:6] = np.log(1j * np.array([2,3,4,5])) + np.exp(1j*t1) + np.exp(-1j*t2)
        cf[6:10] = cf[0:4][::-1]
        return cf.astype(np.complex128)
    except:
        return np.zeros(10, dtype=complex)

def poly_742(t1, t2):
    try:
        n = 10
        cf = np.zeros(n, dtype=complex)
        for k in range(1, n+1):
            cf[k-1] = ((t1 +1j*t2)**k / np.math.factorial(k)) * np.exp(1j * np.sin(k*t2.real))
        cf[0] = t1**3 -1j*t1**2 + t2**2 -1j*t2
        cf[4] = t2.real * t1.imag -1j*t2**3
        cf[9] = t1.real**2 * t2.real**2 * np.exp(1j * (t1.real + t2.real))
        return cf.astype(np.complex128)
    except:
        return np.zeros(10, dtype=complex)

def poly_743(t1, t2):
    try:
        cf = np.zeros(11, dtype=complex)
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
    except:
        return np.zeros(11, dtype=complex)

def poly_744(t1, t2):
    try:
        cf = np.zeros(11, dtype=complex)
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
    except:
        return np.zeros(11, dtype=complex)

def poly_745(t1, t2):
    try:
        cf = np.zeros(10, dtype=complex)
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
    except:
        return np.zeros(10, dtype=complex)

def poly_746(t1, t2):
    try:
        cf = np.zeros(11, dtype=complex)
        coeff_sequence = np.linspace(-1 + 2 * np.real(t1), 1 - 2 * np.imag(t2), num=11)
        for i in range(11):
            cf[i] = np.exp(1j * (i + 1) * t1) + np.sum(coeff_sequence[:i + 1] * np.log(1 + np.abs(t2))**2)
        cf[10] += np.sqrt(cf[0] * cf[1] * t1)
        cf[0] -= np.sqrt(cf[9] * cf[10] * t2)
        cf[5] = np.sum(cf) / 11
        cf[2] *= cf[7] / cf[5]
        cf[7] = cf[7]**2 - cf[4] + cf[8]
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(11, dtype=complex)

def poly_747(t1, t2):
    try:
        cf = np.zeros(11, dtype=complex)
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
    except Exception:
        return np.zeros(11, dtype=complex)

def poly_748(t1, t2):
    try:
        cf = np.zeros(11, dtype=complex)
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
    except Exception:
        return np.zeros(11, dtype=complex)

def poly_749(t1, t2):
    try:
        cf = np.zeros(10, dtype=complex)
        cf[0] = np.exp(1j * (t1 + t2))
        cf[1] = np.sin(t1) * np.cos(t2) - np.cos(t1) * np.sin(t2)
        for j in range(2, 9):
            cf[j] = np.exp((j + 1) / 3) * np.sin(t1 + t2) * np.exp(-1j * (t1 - t2) / (j + 1))
        cf[9] = np.sqrt(t1 ** 2 + t2 ** 2) - np.log10(np.abs(t1) + np.abs(t2))
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(10, dtype=complex)

def poly_750(t1, t2):
    try:
        cf = np.zeros(11, dtype=complex)
        cf[0] = np.abs(t1 + t2) + 100j
        for k in range(1, 10):
            cf[k] = cf[k - 1] * (-1j * t1 + 1 * t2) ** (k + 1) / (k + 1)
        cf[10] = cf[0] * np.exp(1j * np.abs(np.sum(cf[:-1])))
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(11, dtype=complex)

def poly_751(t1, t2):
    try:
        angle = np.linspace(0, 2 * np.pi, num=11)
        cf = 10 * np.exp(1j * angle)
        cf[0] = np.abs(t1 + t2)
        cf[5] = np.sin(t1) * (np.cos(t2) ** 2)
        cf[7] = -np.log(np.abs(t1) + 1) + 1j * np.log(np.abs(t2) + 1)
        cf[9] = (t1 ** 2) / (t2 + 1j)
        cf[10] = np.sqrt(t1 * t2)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(71, dtype=complex)

def poly_752(t1, t2):
    try:
        cf = np.zeros(11, dtype=complex)
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
    except Exception:
        return np.zeros(11, dtype=complex)

def poly_753(t1, t2):
    try:
        cf = np.zeros(10, dtype=complex)
        cf[0] = np.sin(t1 + t2) + np.cos(t1 - t2)
        cf[1] = np.exp(1j * (np.abs(t1 + t2) ** 2))
        prime_numbers = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
        for k in range(2, 8):
            cf[k] = prime_numbers[k] * ((t1 / (k + 1)) ** 2) * np.exp(1j * (t2 / (k + 1)))
        fibonacci_sequence = np.array([0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55])
        cf[8] = fibonacci_sequence[10] * t1 * t2 * np.exp(1j * (t1 - t2))
        cf[9] = ((t1 ** 3) + (t2 ** 3) - 1) * np.exp(1j * (t1 + t2))
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(10, dtype=complex)

def poly_754(t1, t2):
    try:
        cf = np.zeros(10, dtype=complex)
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
    except Exception:
        return np.zeros(10, dtype=complex)

def poly_755(t1, t2):
    try:
        cf = np.zeros(11, dtype=complex)
        cf[0] = (t1 * np.conj(t2)) ** 3 + t1 - t2
        cf[10] = t2 + 1j * t1
        snd = np.linspace(np.sin(2 * np.pi * np.real(t1)), np.sin(2 * np.pi * np.imag(t2)), num=9)
        csi = np.linspace(np.cos(2 * np.pi * np.imag(t1)), np.cos(2 * np.pi * np.real(t2)), num=9)
        stat = t1 * csi ** 2 + t2 * snd ** 2 + 1j * (t1 * snd ** 2 + t2 * csi ** 2)
        cf[1:10] = stat
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(11, dtype=complex)

def poly_756(t1, t2):
    try:
        cf = np.zeros(10, dtype=complex)
        cf[0] = np.exp(1j * t1 * t2)
        cf[1] = 100 * np.sin(t1 * t2)
        cf[2] = 100j * np.cos(t1 * t2)
        cf[3] = 1j * t1 ** 3 - 2j * t1 * t2 ** 2
        cf[4] = t1 ** 5 + t2 ** 5
        cf[5] = 10j * t1 ** 4 - 10j * t2 ** 4
        for k in range(6, 10):
            cf[k] = cf[k - 1] * 1j * 0.8 ** (k + 1)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(10, dtype=complex)

def poly_757(t1, t2):
    try:
        cf = np.zeros(11, dtype=complex)
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
    except Exception:
        return np.zeros(11, dtype=complex)

def poly_758(t1, t2):
    try:
        cf = np.zeros(10, dtype=complex)
        cf[0] = (t1 ** 2 + t2 ** 2) * np.exp(1j * np.angle(t1 + t2))
        cf[1] = 100j * (t1 ** 3 - t2 ** 2) * np.cos(np.angle(t1 - t2))
        cf[2] = np.real(t1 - t2) ** 2 + np.imag(t1 + t2) ** 2 - 100
        cf[3] = 42 * (np.log(1 + np.abs(t1)) + np.log(1 + np.abs(t2))) * np.exp(1j * np.pi / 4)
        cf[4] = np.sqrt(np.abs(t1) + np.abs(t2)) * np.exp(1j * (np.angle(t1 * t2) - np.pi / 3))
        cf[5] = np.sinh(0.1 * np.real(t1 + t2)) + np.cosh(0.1 * np.imag(t1 - t2))
        cf[6] = 1 / (1 + np.exp(-np.abs(t1))) + 1j / (1 + np.exp(-np.abs(t2)))
        cf[7] = np.arctan(1 / np.abs(t1 + t2)) * 1j
        cf[8] = 1j * t1 ** 3 - t2 * 2 - 200
        cf[9] = np.exp(1j * t1 * t2) / (1 + t1 + t2)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(10, dtype=complex)

def poly_759(t1, t2):
    try:
        cf = np.zeros(11, dtype=complex)
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
    except Exception:
        return np.zeros(11, dtype=complex)

def poly_760(t1, t2):
    try:
        cf = np.zeros(11, dtype=complex)
        cf[0] = np.sin((np.real(t1) + np.imag(t2) * 2) / (1 + np.abs(t1 + t2) ** 2))
        cf[1] = 100 * np.real(t1) * np.imag(t1) * (np.abs(np.cos(np.real(t2))) ** 2.1 - np.abs(np.sin(np.imag(t2))) ** 2.1)
        mm = np.array([[np.real(t2), np.imag(t2)], [-np.imag(t1), np.real(t1)]])
        if np.abs(np.linalg.det(mm)) < 1e-10:
            vv = 0
        else:
            vv = np.sum(np.linalg.inv(mm))
        cf[2] = 1000 * vv
        cf[3] = np.sum(np.fft.fft(np.array([t1, t2]), inverse=True))
        cf[4] = 10 * np.exp(1j * np.arctan2(np.imag(t1), np.real(t1))) * np.sqrt(np.real(t1) ** 2 + np.imag(t1) ** 2)
        cf[5] = 1000 * np.median([np.real(t1), np.imag(t1), np.real(t2), np.imag(t2)]) ** 2 + 500 * np.median([np.real(t1), np.imag(t1), np.real(t2), np.imag(t2)]) ** 3
        cf[6] = np.real(t2) + 1j * 2 * np.sqrt(np.abs(np.imag(t2)))
        cf[7] = 100 * np.log10(np.abs(np.real(t2)) + np.abs(np.imag(t2)))
        cf[8] = np.sqrt(np.abs(t1 * t2))
        cf[9] = 1000 * (np.imag(t1) ** 3 - 3 * np.imag(t1) * np.real(t1) ** 2)
        cf[10] = 2j * (np.real(t1) - np.imag(t1)) + 2 * (np.real(t1) + np.imag(t1))
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(11, dtype=complex)

def poly_761(t1, t2):
    try:
        n = 10
        cf = np.zeros(n, dtype=complex)
        for k in range(n):
            cf[k] = (100 * np.sin(np.real(t1) * (k + 1)) * np.cos(np.imag(t2) * (k + 1))) * np.exp(1j * (np.real(t2) * (k + 1) / 100))
        cf[0] *= 10
        cf[4] *= np.sin(t1 * t2)
        cf[9] += np.cos(t1 * t2) * np.exp(1j * (np.real(t1) + np.real(t2)))
        return cf.astype(np.complex128)
    except Exception:
        return np.full(n,0, dtype=complex)

def poly_762(t1, t2):
    try:
        cf = np.full(10, 0,dtype=complex)
        primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29],dtype=complex)
        cf[0] = t1 + t2
        cf[1:9] = primes[1:9] * (t1 ** 2 + t2 ** 2 + 1j)
        cf[9] = np.sum(primes[:9]) + np.abs(t1 ** 2 + t2 ** 2)
        return np.array(cf,dtype=complex)
    except Exception:
        return np.full(10,0, dtype=complex)

def poly_763(t1, t2):
    try:
        cf = np.zeros(10, dtype=complex)
        cf[0] = (t1 ** 2 + t2 ** 2) * 1j
        cf[1] = 10
        cf[4] = np.abs(t1 * 100) - 0.5
        cf[5] = np.abs(t2 * 100) - 0.5
        cf[7] = -10
        cf[9] = (t1 ** 2 + t2 ** 2)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(10, dtype=complex)

def poly_764(t1, t2):
    try:
        n = 10
        cf = np.zeros(n, dtype=complex)
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
    except Exception:
        return np.zeros(10, dtype=complex)

def poly_765(t1, t2):
    try:
        cf = np.zeros(11, dtype=complex)
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
    except Exception:
        return np.zeros(11, dtype=complex)

def poly_766(t1, t2):
    try:
        cf = np.zeros(11, dtype=complex)
        cf[0] = 100 * np.exp(1j * np.abs(t1))
        cf[1] = 10 * (np.real(t1) ** 2 - np.imag(t2) ** 2)
        cf[2] = 1j * (np.real(t2) + np.imag(t1)) * (np.real(t1) - np.imag(t2))
        cf[3] = (t1 + t2) ** 2 - (t1 - t2) ** 2
        cf[4] = 100 * (t1 + t2) / (1 + np.abs(t1 * t2))
        cf[5] = sum((t1**k).real * (t2**k).imag for k in range(1, 6))
        cf[6] = np.sqrt(np.abs(t1 ** 2 - t2 ** 2))
        cf[7] = t1 * t2 / (1 + np.abs(t1 - t2))
        cf[8] = (math.prod(range(math.floor(t1.real), math.floor(t2.imag)+1)) if t1.real <= t2.imag else math.prod(range(math.floor(t1.real), math.floor(t2.imag)-1, -1))) + 1j * (math.prod(range(math.floor(t2.real), math.floor(t1.imag)+1)) if t2.real <= t1.imag else math.prod(range(math.floor(t2.real), math.floor(t1.imag)-1, -1)))
        cf[9] = np.exp(1j * (np.real(t1) - np.imag(t2)))
        cf[10] = np.exp(-1j * (np.real(t2) - np.imag(t1)))
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(11, dtype=complex)

def poly_767(t1, t2):
    try:
        cf = np.zeros(11, dtype=complex)
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
    except Exception:
        return np.zeros(11, dtype=complex)

def poly_768(t1, t2):
    try:
        n = 10
        cf = np.zeros(n, dtype=complex)
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
    except Exception:
        return np.zeros(10, dtype=complex)

def poly_769(t1, t2):
    try:
        n = 10
        cf = np.zeros(n, dtype=complex)
        cf[0] = 100 * np.sin(t1) ** 3 * np.cos(t2) ** 2
        cf[1] = 100 * np.exp(1j * (t1 + t2)) - 10 * (t1 - t2) ** 2
        cf[2] = t1 * t2 * (t1 - t2) / (np.abs(t1) + np.abs(t2) + 1)
        cf[4] = (t1 * t2 * np.exp(1j * (t1 ** 2 - t2 ** 2))) ** 3
        cf[6] = np.sqrt(np.abs(t1)) - np.sqrt(np.abs(t2)) + 1j * np.sin(t1 * t2)
        cf[7] = 50 * np.abs(t1 - t2) * np.exp(1j * np.abs(t1 + t2))
        cf[8] = np.where(np.imag(t1) > 0, t1 - np.abs(t2), t2 - np.abs(t1))
        cf[9] = (1j * t1 * t2) ** (0.1 * t1 * t2)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(10, dtype=complex)

def poly_770(t1, t2):
    try:
        cf = np.zeros(11, dtype=complex)
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
    except Exception:
        return np.zeros(11, dtype=complex)

def poly_771(t1, t2):
    try:
        cf = np.zeros(11, dtype=complex)
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
    except Exception:
        return np.zeros(11, dtype=complex)

def poly_772(t1, t2):
    try:
        m = 10
        cf = np.zeros(m + 1, dtype=complex)
        cf[[0, m]] = [t1, t2]
        cf[[1, m - 1]] = [1j * t1, -1j * t2]
        for x in range(1, m - 1):
            cf[x + 2] = np.abs(cf[x + 1] * cf[m - x + 1]) * np.cos(cf[0] * cf[m]) ** (x + 1)
        mult = np.exp(1j * np.linspace(0, 2 * np.pi, num=m + 1))
        cf *= mult
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(71, dtype=complex)

def poly_773(t1, t2):
    try:
        cf = np.zeros(11, dtype=complex)
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
    except Exception:
        return np.zeros(11, dtype=complex)

def poly_774(t1, t2):
    try:
        cf = np.zeros(11, dtype=complex)
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
    except Exception:
        return np.zeros(11, dtype=complex)

def poly_775(t1, t2):
    try:
        cf = np.zeros(11, dtype=complex)
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
    except Exception:
        return np.zeros(11, dtype=complex)

def poly_776(t1, t2):
    try:
        n = 10
        cf = np.zeros(n, dtype=complex)
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
    except Exception:
        return np.zeros(10, dtype=complex)

def poly_777(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        cf[1] = t1 + t2
        for k in range(2, 26):
            v = np.exp(1j * np.angle(cf[k - 1] + t2)) * np.abs(k * cf[k - 1] + t1)
            cf[k] = v
        cf[9] = np.real(t1) + np.imag(t2)
        cf[14] = 1j * (np.real(t1) + np.imag(t2))
        cf[19] = np.real(t1 * t2) * (1 + 1j)
        cf[24] = np.abs(t1) ** 2 + np.abs(t2) ** 2
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(25, dtype=complex)

def poly_778(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
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
    except Exception:
        return np.zeros(25, dtype=complex)

def poly_779(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        for k in range(25):
            cf[k] = np.abs(t1) ** (k + 1) + np.angle((t2 + 1j) ** (k + 1)) + np.log(np.abs(np.sin(k * t1) * np.cos(k * t2)))
        cf[4] = t1 ** t2 + np.conj(t2) ** 3
        cf[14] = np.cos(np.real(t1)) * np.sin(np.imag(t2)) + np.log(np.abs(1j * t2))
        cf[24] = np.real(np.conj(t2) ** t1) - np.imag(1j * t1 ** 3)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(25, dtype=complex)

def poly_780(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        cf[0] = t1 ** 2 + t2 ** 2 - t1 * t2
        for k in range(1, 25):
            cf[k] = cf[k - 1] * (t1 + t2) / (1 + np.abs(cf[k - 1]))
        cf[[2, 5, 8, 11, 14, 17, 20, 23]] += (t1 + 1j * t2)
        cf[[1, 4, 7, 10, 13, 16, 19, 22, 24]] -= (t2 + 1j * t1)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(25, dtype=complex)

def poly_781(t1, t2, err = False):
    try:
        cf = np.zeros(25, dtype=complex)
        cf[0] = t1 + t2
        cf[1] = t1 ** 2 + t2 ** 2
        for k in range(2, 13):
            cf[k] = (cf[k - 1] * cf[k - 2]) / (1 + np.abs(t1) + np.abs(t2))
        for l in range(13, 25):
            cf[l] = (cf[l - 1] + cf[l - 13]) / (1 + np.real(t1) ** 2 + np.imag(t2) ** 2)
        return cf.astype(np.complex128)
    except Exception as e:
        if err:
            print(f"Error occurred: {e}")
            return np.zeros(25, dtype=complex)
        else:
            return np.zeros(25, dtype=complex)
        
def poly_781_v1(t1, t2, err = False):
    try:
        cf = np.zeros(25, dtype=complex)
        cf[0] = t1 + t2
        cf[1] = t1 ** 2 + t2 ** 2
        for k in range(3, 14):
            cf[k] = (cf[k - 1] * cf[k - 2]) / (1 + np.abs(t1) + np.abs(t2))
        for l in range(14, 25):
            cf[l] = (cf[l - 1] + cf[l - 13]) / (1 + np.real(t1) ** 2 + np.imag(t2) ** 2)
        return cf.astype(np.complex128)
    except Exception as e:
        if err:
            print(f"Error occurred: {e}")
            return np.zeros(25, dtype=complex)
        else:
            return np.zeros(25, dtype=complex)
        
def poly_782(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        cf[0] = t1 + t2
        cf[1] = np.exp(1j * np.angle(t1 * np.conj(t2)))
        cf[2] = np.abs(t1) * np.abs(t2)
        for k in range(3, 26):
            cf[k] = (np.real(cf[k - 1]) + 1j * np.imag(cf[k - 1])) * np.exp(1j * np.angle(cf[k - 2]))
            if np.imag(cf[k]) == 0:
                cf[k] += 1e-10
            cf[k] = np.log(np.abs(cf[k])) / 2 + cf[k] * 1j
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(25, dtype=complex)

def poly_783(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        cf[0] = np.abs(t1 * t2) * np.exp(1j * np.angle(t1 + t2))
        cf[1] = np.angle(t1 * t2) * np.exp(1j * np.abs(t1 - t2))
        for k in range(2, 25):
            cf[k] = np.abs(t1 + t2 * 1j ** (k + 1)) * np.exp(1j * np.angle(cf[k - 1] + t1 * cf[k - 2]))
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(25, dtype=complex)

def poly_784(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        cf[0] = np.real(t1 + t2) + np.conj(t1 * t2)
        for k in range(1, 25):
            n = np.abs(t1 * cf[k - 1])
            if n != 0:
                cf[k] = np.sin(np.angle(t1)) * np.log(np.abs(n)) + np.cos(np.angle(t2)) * np.log(np.abs(1 / n))
            else:
                cf[k] = 0
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(25, dtype=complex)

def poly_785(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        cf[:5] = np.array([1, t1, t2, t1 * t2, np.abs(t1 + t2)]) ** 2
        for k in range(5, 25):
            cf[k] = 3 * cf[k - 1] + 2 * cf[k - 5] + 5 * (k + 1)
        cf[9:15] = np.conj(cf[:6])
        cf[19:25] = np.exp(1j * np.angle(cf[:6]))
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(25, dtype=complex)

def poly_786(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        for k in range(25):
            cf[k] = np.abs(t1) ** (k / 2) * (np.cos(k * np.angle(t2)) + 1j * np.sin(k * np.angle(t2)))
        cf[4] += (np.log(np.abs(t1)) + np.log(np.abs(t2))) / 2
        cf[9] += np.conj(t1 * t2)
        cf[14] += np.abs(t2 - t1) ** 2
        cf[19] += (np.sin(np.angle(t1)) / np.cos(np.angle(t2))) ** 3
        cf[24] += ((1j * t1 - t2) ** 2 / (1 + np.abs(t1 + t2) ** 3)) ** 4
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(25, dtype=complex)

def poly_787(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        cf[0] = np.real(t1 + t2) + 1j * np.imag(t1 * t2)
        for i in range(1, 24):
            cf[i] = np.abs(cf[i - 1] ** (t1 - t2)) + 1j * np.angle(cf[i - 1])
            if np.isinf(cf[i]) or np.isnan(cf[i]):
                cf[i] = 1j
        cf[24] = cf[0] ** 3 + cf[23] ** 2 + cf[22] - cf[21] + np.conj(cf[20])
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(25, dtype=complex)

def poly_788(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
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
    except Exception:
        return np.zeros(25, dtype=complex)

def poly_789(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        cf[:5] = np.real(t1) * np.arange(1, 6) - np.imag(t2) * np.arange(1, 6)
        cf[5] = np.prod(np.abs(t1), np.abs(t2))
        cf[6:11] = np.angle(t1 + t2) * np.arange(6, 11)
        cf[12] = np.conj(t1) + np.conj(t2)
        cf[13:18] = np.real(t1 + 1j * t2) * np.arange(1, 6)
        cf[18] = np.prod(np.angle(t1), np.angle(t2))
        cf[19:24] = np.imag(t1 - 1j * t2) * np.arange(1, 6)
        cf[24] = np.conj(t1 * t2)
        cf[25] = np.abs(cf[12]) + np.angle(cf[18])
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(25, dtype=complex)

def poly_790(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        cf[0] = t1 + t2
        for k in range(1, 25):
            v = np.sin(k * cf[k - 1] + np.angle(t2 ** k)) + np.cos(k * np.abs(t1))
            cf[k] = v / (np.abs(v) + 1e-10)
        cf[9] = t1 * t2 - np.abs(t2) ** 2 + 1j * np.angle(t1)
        cf[14] = np.conj(t1) ** 3 - np.angle(t2) ** 3 + 1j * np.abs(t2)
        cf[19] = np.abs(t2) ** 3 + t1 ** 2 + t2 ** 2 + 1j * np.angle(t2) ** 2
        cf[24] = np.abs(t1 * t2) + np.angle(t1) ** 5 + 1j * np.abs(t1) ** 5
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(25, dtype=complex)

def poly_791(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        cf[:10] = np.arange(1, 11) + t1 ** 2 + t2 ** 2
        for i in range(10, 20):
            cf[i] = 2 * cf[i - 1] * np.sin(t1 + t2)
        cf[20:22] = np.array([np.sum(cf[:20]), np.prod(cf[:20])]) * (t1 + t2)
        cf[22] = np.log(np.abs(t1 * t2)) + np.angle(t1 ** 2 + t2 ** 2) * cf[21]
        cf[23] = np.abs(t1 - t2) / (np.abs(cf[22]) + 1)
        cf[24] = np.conj(cf[23] * cf[21])
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(25, dtype=complex)

def poly_792(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        cf[0] = np.sin(t1 + t2) + 1j * np.cos(t1 - t2)
        for k in range(1, 25):
            v = np.exp(k * cf[k - 1]) + np.log(np.abs(t1)) - np.log(np.abs(t2))
            if not np.isnan(v) and not np.isinf(v):
                cf[k] = v
            else:
                cf[k] = 1 + 1j
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(25, dtype=complex)

def poly_793(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        cf[0] = t1 + t2
        cf[1] = np.abs(t1) * t2
        cf[2] = np.conj(t1) + np.real(t2)
        cf[3] = np.abs(t1) * np.imag(t2)
        cf[4] = np.angle(t1) * np.conj(t2)
        for k in range(5, 26):
            cf[k] = np.abs(cf[k - 1] * t1) + np.angle(cf[k - 2] * t2)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(25, dtype=complex)

def poly_794(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
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
    except Exception:
        return np.zeros(25, dtype=complex)

def poly_795(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        for i in range(25):
            cf[i] = (i ** 2 + t1) * np.exp((1 + 0j) * i * t2) / (1 + np.abs(t1 * t2))
        cf[4:15] = np.real(cf[4:15]) * np.cos(np.imag(cf[4:15]))
        cf[16:25] = np.abs(cf[16:25]) * np.exp(1j * np.angle(t1 + t2))
        cf[2] = (t1 + 1j * t2) ** 3 - np.conj(t1 + 1j * t2) ** 3
        cf[6] = cf[22] = (np.abs(t1) ** 3 + np.abs(t2) ** 3) * np.exp(1j * np.angle(t2 - t1))
        cf[18] = np.where(np.abs(t2) > 1, np.log(np.abs(t2)), 0)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(25, dtype=complex)

def poly_796(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
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
    except Exception:
        return np.zeros(25, dtype=complex)

def poly_797(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        cf[0] = 1 + t1 * t2
        cf[1] = -(t1 + t2) - 1j
        for k in range(2, 25):
            cf[k] = cf[k - 1] * cf[1] / cf[k - 2]
            cf[k] += t2 ** (k + 1) + 1j * t1 ** (k + 1)
            cf[k] *= np.exp(1j * np.angle(cf[k - 2]))
            cf[k] /= np.abs(cf[k])
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(25, dtype=complex)

def poly_798(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        cf[1] = np.abs(t1) * np.sin(np.angle(t2))
        cf[2] = np.abs(t2) * np.cos(np.angle(t1))
        for k in range(3, 26):
            cf[k] = np.abs(cf[k - 1]) * np.sin(np.angle(cf[k - 2])) + t1
            if np.abs(cf[k]) > 10000:
                cf[k] /= np.abs(cf[k])
        cf[0] = cf[24]
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(25, dtype=complex)

def poly_799(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        for i in range(25):
            cf[i] = (2 * np.real(t1 * t2) + 3 * np.imag(t1 * t2)) * (np.abs(t1) ** (i + 1)) + (2 * np.angle(t1 * t2) - 3 * np.abs(t2)) * (np.abs(t2) ** (i + 1))
            if not np.isfinite(cf[i]):
                cf[i] = 0
        cf[0] += np.conj(t1 * t2)
        if not np.isfinite(cf[0]):
            cf[0] = 0
        cf[12] *= np.log(np.abs(t1 + t2))
        if not np.isfinite(cf[12]):
            cf[12] = 0
        cf[24] -= cf[24] * t1 * t2 / (t1 + t2)
        if not np.isfinite(cf[24]):
            cf[24] = 0
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(25, dtype=complex)

def poly_800(t1, t2):
    cf = np.zeros(25, dtype=complex)
    cf[0] = t1 * t2
    cf[1:10] = np.abs(t1) * np.angle(1j * t2) ** np.arange(1, 10)
    cf[14] = np.real(t1) * np.imag(t2) + np.real(t2) * np.imag(t1)
    cf[15:20] = np.abs(t1 - t2 + 1j) ** np.arange(1, 6)
    cf[20] = np.imag(t1) * np.real(t2) + np.real(t1) * np.imag(t2)
    cf[21:24] = np.abs(cf[14] + t1 * t2) ** np.arange(1, 4)
    cf[24] = np.abs(cf[23]) + np.log(np.abs(t1 + t2 + 0.5j))
    return cf.astype(np.complex128)

