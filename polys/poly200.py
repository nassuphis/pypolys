from . import polystate as ps
import math
import cmath
import numpy as np
from scipy.special import sph_harm
from . import letters
from . import zfrm

pi = math.pi

def poly_101(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_102(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_103(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_104(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(1, 72):
            cf[k - 1] = ((np.abs(t1)**(1/k)) * (np.cos(np.angle(t1)) + 1j * np.sin(np.angle(t1))) + 
                          (np.abs(t2)**(1/k)) * (np.cos(np.angle(t2)) + 1j * np.sin(np.angle(t2)))) / k
        cf[np.arange(0, 71, 3)] *= -1
        cf[np.arange(1, 71, 4)] *= 2
        cf[np.arange(2, 71, 5)] *= 3
        cf[np.arange(3, 71, 6)] *= 4
        cf[np.arange(4, 71, 7)] *= 5
        
        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_105(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(1, 72):
            cf[k - 1] = np.sin(k * (np.real(t1) * np.imag(t2))**3) + np.cos(k * np.log(np.abs(t1 * t2 + 1)) * np.angle(t1 + np.conj(t2)))
        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_106(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
    except:
        return np.zeros(0, dtype=complex)

def poly_107(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)

        for r in range(1, 72):
            cf[r - 1] = (100 * (t1 ** (71 - r))) * np.sin(0.5 * t1 * r) + \
                         (100 * (t2 ** r)) * np.cos(0.5 * t2 * r)
        
        cf[14] = 100 * t2**3 - 100 * t2**2 + (100 * t2 - 100)
        cf[29] = 100 * np.log(np.abs(t1 * t2) + 1)
        cf[44] = np.abs(10 * t1 + 0.5 * t2)
        cf[59] = np.angle(0.2 * t1 - 3j * t2)
        cf[70] = np.real(10 * t1 + 0.5 * t2)

        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_108(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
        cf[34] = np.sum([3, 3, 9, 15, -12]) * (np.real(t1) - np.imag(t2))
        cf[39] = -t1**4 + t2**4 - 3
        cf[44] = 3 * np.angle(t1) + 4 * np.angle(t2)
        cf[49] = -55 * np.abs(np.abs(t1) - np.abs(t2))
        cf[54] = 33 * np.abs(t1)**3 + np.abs(t2)**2
        cf[59] = t1**5 + t2**5 - 29
        cf[64] = -22 * np.real(t1**2) + 22 * np.imag(t2**2)
        cf[69] = (np.sum(range(1, 6)) * np.imag(t1)) + (np.prod(range(1, 6)) * np.real(t2))
        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_109(t1, t2):
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
        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_110(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        prime_sequence = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59])
        for k in range(1, 36):
            cf[k - 1] = np.real(t1) * prime_sequence[k % len(prime_sequence)] + np.imag(t2) * k**2
            cf[70 - k] = np.real(t2) * prime_sequence[(70 - k) % len(prime_sequence)] - np.imag(t1) * k**2
        cf[35] = np.sum(prime_sequence) * (np.cos(np.angle(t1)) + 1j * np.sin(np.angle(t2)))
        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_111(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
    except:
        return np.zeros(0, dtype=complex)

def poly_112(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        
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
        return cf.astype(np.complex128)
    except:
        return np.zeros(71, dtype=complex)

def poly_113(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        z = t1 + t2 * 1j
        for k in range(1, 36):
            cf[k - 1] = np.cos(np.pi * k / 35) * ((-1)**k) * np.abs(z)**k
            cf[70 - k] = np.sin(np.pi * (35 - k) / 35) * ((-1)**(k + 1)) * np.angle(z)**(35 - k)
        cf[35] = np.exp(np.abs(z))
        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_114(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_115(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
        
        cf[30:51] = np.real(offset) * (np.arange(21, 41)) + 0.1 * np.imag(offset) * (np.arange(1, 21))
        
        for k in range(51, 61):
            cf[k - 1] = np.imag(t1 * t2)**2 / (k**2)
        
        cf[61] = np.abs(offset) + 0.1 * t2**2 - 0.1 * t1**2
        cf[62] = 0.01 * (t1**3 - 2 * t2**3)
        cf[63] = 0.001 * (offset * np.conj(t2))
        cf[64:70] = ((np.arange(64, 70) + 1) * np.real(offset) + (np.arange(64, 70) + 1) * np.imag(offset)**2) / 2
        cf[70] = -t1 + 2j * t2
        cf[71] = (1 + (t1**3 * np.conj(t2))) / 3
        
        return cf.astype(np.complex128) 
    except:
        return np.zeros(0, dtype=complex)

def poly_116(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47])
        for k in range(1, 36):
            cf[k - 1] = primes[k % 15] * (t1**k + t2**k) * (-1)**k / (k + 1)
            cf[70 - k] = primes[(k + 11) % 15] * (t1**(71 - k) - t2**(71 - k)) * (-1)**(71 - k) / (71 - k + 1)
        cf[35] = np.sum(primes[:5]) * np.abs(t1 + t2) / (1 + np.abs(t1))
        cf[70] = 1 + 1j
        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_117(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(1, 72):
            cf[k - 1] = (t1 + t2)**(k - 1) + ((-1)**k) * (np.exp(1j * k * np.pi / 71)) * (k**(1/3))
        cf *= (1 + np.log(np.abs(cf) + 1) / (1 + np.abs(t1 * t2)))
        cf[0:10] += (t1**2 + t2**2)**(1/3)
        cf[61:71] *= np.exp(-1j * np.angle(t1))
        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_118(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53])
        f1 = t1 * np.sum(primes) + t2
        f2 = t2 * np.sum(primes[:8]) + np.conj(t1)
        cf[0:16] = primes[:16] * (t1 - t2)
        cf[16:32] = f1**2 - f2**2
        cf[32:48] = (t1**3 - t2**3) * (primes[:16] - f1)
        cf[48:64] = (primes[:16] * t1**2 + t2**3) - t1
        cf[64:70] = np.sin(cf[0:6] * t2) + np.cos(cf[0:6] * t1)
        cf[70] = np.prod(primes[:9])
        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_119(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
    except:
        return np.zeros(0, dtype=complex)

def poly_120(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        theta = np.angle(t1) * np.angle(t2)
        mult_factors = (-1)**(np.arange(1, 71) + 1)
        cf[0:10] = (np.arange(1, 11)) * t1**2 - (np.arange(10, 0, -1)) * t2**2
        for k in range(11, 41):
            cf[k - 1] = (k % 2) * np.abs(t1) + (k % 3) * np.abs(t2) * np.exp((k / 5) * theta * 1j)
        cf[40:60] = ((np.arange(41, 61)) + np.log(np.abs(theta) + 1)) * np.conj(t1) * 5 * mult_factors[0:20]
        cf[60:71] = (np.arange(61, 71)) - (np.arange(1, 11)) * t2 - np.sum((np.arange(1, 11)) * mult_factors[10:20])
        cf[70] = (np.sum(np.arange(1, 36)) + np.sum(np.arange(36, 72))) / np.prod(np.arange(1, 16))
        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_121(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        w = np.exp(2 * np.pi * 1j / 7)
        for k in range(1, 8):
            cf[k - 1] = (t1**k - (t2 / w)**k) * np.real(w**k)
        cf[7] = -np.sum(t1**2, t2**2) + np.real(t1 * t2) + np.imag(t1 * t2)
        for k in range(9, 36):
            z = np.angle(t1) + np.angle(t2)
            cf[k - 1] = np.cos(k * z) + 1j * np.sin(k * z)
        for k in range(36, 71):
            cf[k - 1] = (np.abs(t1) * t2 + t1 * np.imag(t2))**2 / (k + 1)
        cf[70] = np.abs(t1) - np.abs(t2) + np.log(np.abs(t1 + t2 + 1) + 1)
        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_122(t1, t2):
    try:
        if not 'n' in ps.poly or ps.poly['n'] is None:
            n = int(71)
            m = int(36)
        else:
            n = int(ps.poly['n'])
            m = int(n/2)+1
        cf = np.zeros(n, dtype=complex)
        for k in range(1, m):
            cf[k - 1] = (-1)**k * (k**2 + t1 * t2**k + k * np.abs(t1)) * (np.cos(k * np.angle(t2)))
        for k in range(m, n):
            cf[k - 1] = (k**3 + t2 * t1**k + k * np.abs(t2)) * (np.sin(k * np.angle(t1)))
        cf[n-1] = np.sum(np.abs(cf[0:(n-1)]))
        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_123_old(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = np.real(t1)**2 - np.imag(t2)**2
        cf[1] = (t1 + t2)**2 - 7
        cf[2] = (t1**2 - t2**2)
        cf[3:10] = np.arange(3, 30, 4) * np.abs(t1 + 1j * t2)
        cf[10:20] = np.real(t1 - t2) * np.arange(11, 21)
        cf[20:30] = 1 / (1 + np.arange(21, 31)) * np.real(t1 + t2)
        cf[30] = np.angle(t1) * np.imag(t2)
        cf[31:51] = 1000 * (-1)**np.arange(32, 51) * t1 * t2
        cf[51:61] = 2000 * (-1)**np.arange(51, 61) * np.log(np.abs(t1) + 1)
        cf[61:66] = 1j * np.conj(t1 * t2) * np.sqrt(np.arange(61, 66))
        cf[66:71] = (np.arange(66, 71) * (np.arange(66, 71) - 1)) / (np.abs(t1) + np.abs(t2) + 1)
        cf[70] = np.prod(np.arange(1, 6))
        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)


def poly_123(t1, t2):
    cf = np.zeros(71, dtype=complex)
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
    return cf.astype(np.complex128)


def poly_124(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61])

        for j in range(1, 36):
            cf[j - 1] = primes[j % len(primes)] * (t1**j - t2**(71 - j))

        for k in range(36, 72):
            cf[k - 1] = (np.abs(t1) * np.real(t2) - np.imag(t1) * np.angle(t2))**(142 - k) / (1 + np.abs(primes[k % len(primes)]))

        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_125(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0:15] = [(-1)**j * (j**2) * (np.abs(t1) + np.abs(t2)) for j in range(1, 16)]
        cf[15:30] = [(-1)**(k + 1) * (k**3) * np.angle(t1 + 1j * t2) for k in range(16, 31)]
        cf[30:45] = [(-1)**(r + 1) * np.cos(r * t1) + np.sin(r * t2) for r in range(31, 46)]
        for s in range(46, 61):
            cf[s] = (-1)**s * (s**2) * np.conj(t1) * np.conj(t2)
        cf[61:70] = [n**3 * np.log(np.abs(t1 * t2) + 1) for n in range(61, 71)]
        cf[70] = 1
        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_126(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for j in range(1, 36):
            cf[j - 1] = ((-1)**j) * (t1**3 + t2**2) * j**2
            cf[70 - j] = ((-1)**(j + 1)) * (t2**3 + t1**2) * j**1.5
        middle_index = 36
        for k in range(1, 6):
            cf[middle_index + k - 1] = np.exp(1j * np.pi * ((-1)**k) * (t1 + t2) / 2)
        cf[middle_index - 1] = np.log(np.abs(t1 + t2) + 1)
        cf[0] = np.sin(t1**3) + np.cos(t2**3)
        cf[70] = np.cos(t1) * np.sin(t2) + t1**2 + t2**2
        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_127(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(0, 71):
            cf[k] = (t1**(70 - k) + np.conj(t2)**k) * (-1)**k * np.log(np.abs(t1 + t2) + k)
        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_128(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        k = np.arange(1, 72)
        cf[:] = (-1)**k * (t1**k + np.conj(t2)**(71 - k)) * (72 - k)
        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_129(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
    except:
        return np.zeros(0, dtype=complex)

def poly_130(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
    except:
        return np.zeros(0, dtype=complex)

def poly_131(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(1, 72):
            cf[k - 1] = (-1)**k * (np.real(t1)**k + np.imag(t2)**k) + (np.cos(k * t1) + np.sin(k * t2)) / k
        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_132(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
    except:
        return np.zeros(0, dtype=complex)

def poly_133(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(1, 72):
            cf[k - 1] = (t1**k * np.sin(t2 * k) + t2**k * np.cos(t1 * k)) * (-1)**k / (k + 1)
        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)


def poly_134(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(1, 72):
            cf[k-1] = (np.sin(t1 * k) + np.cos(np.conj(t2) * k)) * (-1)**k / (k + 1)
        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_135(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(1, 71):
            cf[k-1] = (t1**k * np.sin(k * t2) + (-1)**k * t2**(k-1) * np.cos(k * t1)) / k
        cf[70] = (np.log(np.abs(t1) + np.abs(t2) + 1) + np.sin(t1 * t2)) / 71
        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_136(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
    except:
        return np.zeros(0, dtype=complex)

def poly_137(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
    except:
        return np.zeros(0, dtype=complex)

def poly_138(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = t1 + t2
        for k in range(2, 72):
            if k % 2 == 0:
                cf[k-1] = (t1**k - t2**k) * (-1)**k / np.log(k + np.abs(t1) + 1)
            else:
                cf[k-1] = (t1**(k//2) + t2**(k//3)) * (1 + np.sin(k * np.angle(t1 + t2)))
        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_139(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
    except:
        return np.zeros(0, dtype=complex)

def poly_140(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
    except:
        return np.zeros(0, dtype=complex)

def poly_141(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
    except:
        return np.zeros(0, dtype=complex)

def poly_142(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
    except:
        return np.zeros(0, dtype=complex)

def poly_143(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
    except:
        return np.zeros(0, dtype=complex)

def poly_144(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
    except:
        return np.zeros(0, dtype=complex)

def poly_145(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(1, 72):
            cf[k-1] = ((t1**k + np.conj(t2)**k) * (-1)**k) / (k + np.real(t1) + np.real(t2))
        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_146(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0:10] = (t1**np.arange(1, 11) + t2**np.arange(1, 11)) * (-1)**np.arange(1, 11)
        cf[10:20] = np.sin(t1 * np.arange(11, 21)) - np.cos(t2 * np.arange(11, 21))
        cf[20:30] = np.log(np.abs(t1) + 1) * np.arange(21, 31) - np.log(np.abs(t2) + 1)
        cf[30:40] = (t1 * t2)**np.arange(31, 41) / (1 + np.arange(31, 41))
        cf[40:71] = np.real(t1) * np.imag(t2) - np.imag(t1) * np.real(t2) + np.angle(t1 + t2) * np.abs(t1 - t2) * np.arange(40, 71)
        cf[70] = np.sum(cf[0:70])
        return cf.astype(np.complex128)
    except:
        return np.zeros(71, dtype=complex)

def poly_147(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
    except:
        return np.zeros(0, dtype=complex)

def poly_148(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = t1**4 - 3*t2 + np.real(t1*t2)
        cf[1] = np.imag(t1)**2 + np.abs(t2)**3
        cf[2] = np.sin(t1) + np.cos(t2)
        cf[3] = np.angle(t1) * np.angle(t2)
        cf[4] = np.log(np.abs(t1 + t2) + 1)
        for j in range(6, 71):
            cf[j-1] = (t1**j + t2**(71 - j)) / (j - 5) + (-1)**j * np.abs(t1 - t2)
        cf[70] = t1 * t2 / (1 + np.abs(t1)**2 + np.abs(t2)**2)
        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_149(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = t1**4 + 2*t2**3
        cf[1] = -t1**3 + 3*t2**2
        cf[2] = t1**2 - 4*t2
        cf[3] = -t1 + 5*t2**2
        cf[4] = t2**3 - 6*t1**2*t2
        for j in range(6, 36):
            cf[j-1] = (np.real(t1)**j - np.imag(t2)**j) * (-1)**j / j
        for j in range(36, 71):
            cf[j-1] = (np.sin(j * t1) + np.cos(j * t2)) / (j + 1)
        cf[70] = np.sum(np.real(t1), np.imag(t2)) * (t1 - t2)
        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_150(t1, t2):
    try:
        j = np.arange(0, 71).astype(complex)
        cf = (t1 + j * t2) * (-1)**j * np.log(np.abs(t1) + np.abs(t2) + 1)**(np.abs(j) % 5 + 1) * (j + 1)
        return cf.astype(np.complex128)
    except:
        return np.zeros(71, dtype=complex)

def poly_151(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
    except:
        return np.zeros(0, dtype=complex)

def poly_152(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        k = np.arange(1, 72)
        cf = (t1**k + np.conj(t2)**k) * (-1)**k / (1 + k)
        return cf.astype(np.complex128).astype(np.complex128)
    except:
        return np.zeros(71, dtype=complex)

def poly_153(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(1, 72):
            if k <= 35:
                cf[k-1] = ((np.real(t1)**k + np.imag(t2)**k) * (-1)**k) / (k)
            else:
                cf[k-1] = (np.sin(t1 * k) + np.cos(t2 * k)) * (-1)**k / (71 - k + 1)
        cf[70] = np.log(np.abs(t1) + 1) + np.log(np.abs(t2) + 1)
        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_154(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = np.real(t1)**10 + np.real(t2)**10
        cf[1] = np.imag(t1)**9 - np.imag(t2)**9
        for j in range(3, 36):
            cf[j-1] = (-1)**j * (np.real(t1)**j + np.real(t2)**(j-1)) / j
        for j in range(36, 71):
            cf[j-1] = (np.log(np.abs(t1) + 1) * j) - (np.log(np.abs(t2) + 1) / j)
        cf[70] = np.real(t1 * t2) + np.imag(t1 + t2)
        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)


def poly_155(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
    except Exception as e:
        return np.zeros(0, dtype=complex)

def poly_156(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
    except Exception as e:
        return np.zeros(0, dtype=complex)

def poly_157(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        k = np.arange(71)
        cf = ((np.real(t1) + np.imag(t2))**(70 - k) + (np.real(t1) - np.imag(t2))**k) * (-1)**k / (k + 1)
        return cf.astype(np.complex128).astype(np.complex128)
    except Exception as e:
        return np.zeros(0, dtype=complex)

def poly_158(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(1, 72):
            cf[k-1] = (np.real(t1) + np.imag(t2) * k)**(1 + k / 10) * (-1)**k + np.sin(k * np.angle(t1 * t2)) + np.cos(k * np.abs(t1 + t2))
        return cf.astype(np.complex128)
    except Exception as e:
        return np.zeros(0, dtype=complex)

def poly_159(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
    except Exception as e:
        return np.zeros(0, dtype=complex)

def poly_160(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
    except Exception as e:
        return np.zeros(0, dtype=complex)

def poly_161(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        k = np.arange(1, 71)
        cf[:70] = (t1**((k % 4) + 1) * t2**((k % 3) + 1)) + (-1)**k * np.log(np.abs(t1) + 1) * np.sin(k * t2)
        cf[70] = t1 * t2 / (1 + t1**2 + t2**2)
        return cf.astype(np.complex128)
    except Exception as e:
        return np.zeros(0, dtype=complex)

def poly_162(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
    except Exception as e:
        return np.zeros(0, dtype=complex)

def poly_163(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
    except Exception as e:
        return np.zeros(0, dtype=complex)

def poly_164(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
    except Exception as e:
        return np.zeros(0, dtype=complex)

def poly_165(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = t1**6 + t2**5
        cf[1] = t1**5 - t2**4
        cf[2] = t1**4 + t2**3
        cf[3] = -t1**3 + t2**2
        cf[4] = t1**2 - t2
        cf[5] = -t1 + 1
        for j in range(7, 72):
            cf[j-1] = (t1 * t2) / j * (-1)**j
        return cf.astype(np.complex128)
    except Exception as e:
        return np.zeros(0, dtype=complex)

def poly_166(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(1, 72):
            cf[k-1] = t1**k + (-1)**k * t2**k / (k + 1) + np.sin(k * t1) + np.cos(k * t2)
        return cf.astype(np.complex128)
    except Exception as e:
        return np.zeros(0, dtype=complex)

def poly_167(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(0, 71):
            j = k + 1
            cf[j-1] = (t1 + t2)**(70 - k) * (t1 - t2)**k * (-1)**k / (k + 1)
        return cf.astype(np.complex128)
    except Exception as e:
        return np.zeros(0, dtype=complex)

def poly_168(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = t1 + t2
        cf[1] = t1**2 - t2**2
        cf[2] = np.sin(t1) * np.cos(t2)
        cf[3] = np.log(np.abs(t1) + 1) - np.log(np.abs(t2) + 1)
        cf[4] = t1 * t2
        for k in range(6, 72):
            cf[k-1] = ((t1**(k-5) + t2**(k-5)) * (-1)**k) / (k**0.5)
        return cf.astype(np.complex128)
    except Exception as e:
        return np.zeros(0, dtype=complex)

def poly_169(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for j in range(1, 72):
            cf[j-1] = (t1**(j % 5 + 1) * np.conj(t2)**(j % 3 + 1)) * (-1)**j / (j + 1)
        return cf.astype(np.complex128)
    except Exception as e:
        return np.zeros(0, dtype=complex)

def poly_170(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for j in range(0, 71):
            cf[j] = (t1**j + np.conj(t2)**j) * (-1)**j / (j + 2)
        return cf.astype(np.complex128)
    except Exception as e:
        return np.zeros(0, dtype=complex)

def poly_171(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        k = np.arange(71)
        cf = (t1**k + t2**(70 - k)) * (-1)**k * (71 - k)
        return cf.astype(np.complex128).astype(np.complex128)
    except Exception as e:
        return np.zeros(0, dtype=complex)

def poly_172(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
    except Exception as e:
        return np.zeros(0, dtype=complex)

def poly_173(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(1, 72):
            cf[k-1] = (t1**(k % 5 + 1) + np.conj(t2)**(k % 7 + 1)) * (-1)**k * np.log(np.abs(t1) + np.abs(t2) + 1)
        return cf.astype(np.complex128)
    except Exception as e:
        return np.zeros(0, dtype=complex)

def poly_174(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(1, 72):
            cf[k-1] = (t1**k + np.conj(t2)**k) * (-1)**k / (1 + k**1.2)
        return cf.astype(np.complex128)
    except Exception as e:
        return np.zeros(0, dtype=complex)

def poly_175(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(1, 72):
            cf[k-1] = ((np.real(t1)**k + np.imag(t2)**(k % 5 + 1)) * (-1)**k) / (1 + k)
        return cf.astype(np.complex128)
    except Exception as e:
        return np.zeros(0, dtype=complex)

def poly_176(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for j in range(1, 72):
            cf[j-1] = ((t1**j + np.conj(t2)**(71 - j)) * (-1)**j) / np.log(j + np.abs(t1) + np.abs(t2) + 1)
        return cf.astype(np.complex128)
    except Exception as e:
        return np.zeros(0, dtype=complex)

def poly_177(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = t1**5 - t2**3 + 2 * t1 * t2
        cf[1] = -t1**4 + 3 * t2**2 - t1 * t2**2
        cf[2] = t1**3 - 4 * t2 + 2 * t1**2 * t2
        cf[3] = -t1**2 + 5 * t2**4 - 3 * t1 * t2**3
        cf[4] = t1 - 6 * t2**5 + 4 * t1**2 * t2**4
        for k in range(6, 71):
            cf[k-1] = ((t1**k) * (-1)**k + t2**(k-1)) / (k + 1)
        cf[70] = np.log(np.abs(t1) + 1) + t1 * t2**2
        return cf.astype(np.complex128)
    except Exception as e:
        return np.zeros(0, dtype=complex)

def poly_178(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for j in range(0, 71):
            cf[j] = ((-1)**j * (t1**j + t2**j)) / ((j + 1)**2)
        return cf.astype(np.complex128)
    except Exception as e:
        return np.zeros(0, dtype=complex)

def poly_179(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = (t1 + t2) * (np.real(t1) - np.imag(t2))
        for j in range(2, 71):
            if j % 2 == 0:
                cf[j-1] = (np.abs(t1)**j - np.abs(t2)**j) / (j + 1) * (-1)**j
            else:
                cf[j-1] = (np.real(t1)**j + np.imag(t2)**j) / (j + 2) * np.sin(j * np.angle(t1 + t2))
        cf70a = sum((t1**n).real * (t2**n).imag for n in range(1, 6))    
        cf70b =  np.log(np.abs(t1) + np.abs(t2) + 1)   
        cf[70] =  cf70b + cf70a
        return cf.astype(np.complex128)
    except Exception as e:
        return np.zeros(0, dtype=complex)

def poly_180(t1, t2,err=False):
    try:
        cf0 = t1 ** ((np.arange(71) % 6) + 1)
        cf1 = t2 ** ((np.arange(71) % 4) + 1)
        cf2 = (-1)**np.arange(71)
        cf3 = np.log(np.arange(71) + 1)
        cf = ((cf0 + cf1) * cf2 * cf3).astype(complex)
        return cf.astype(np.complex128)
    except Exception as e:
        if err:
            print(f"Details: {e}")
        return np.zeros(71, dtype=complex)

def poly_181(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(1, 72):
            cf[k-1] = (-1)**k * (np.abs(t1)**k + np.abs(t2)**k) + (np.sin(k * np.angle(t1)) - np.cos(k * np.angle(t2))) / k
        return cf.astype(np.complex128)
    except Exception as e:
        return np.zeros(0, dtype=complex)

def poly_182(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for j in range(71):
            cf[j] = (t1**(j + 1) * np.conj(t2)**(70 - j)) * (-1)**(j // 5) + np.log(np.abs(t1 + (j + 1) * t2) + 1)
        cf[70] = np.sin(t1) + np.cos(t2) + np.real(t1 * t2)
        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_183(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
        cf[70] = np.sum([t1**2, t2**2]) + np.prod([t1, t2])
        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_184(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_185(t1, t2):
    cf = np.zeros(71, dtype=complex)
    try:
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
    except:
        return np.zeros(71, dtype=complex)

def poly_186(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for j in range(71):
            cf[j] = ((-1)**j) * (np.real(t1)**j + np.imag(t2)**j) / (np.log(np.abs(t1 + t2)) + j) * np.sin(j * np.angle(t1 * t2 + 1))
        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_187(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        k = np.arange(71)
        cf = t1**k * np.sin(t2 * k) + t2**k * np.cos(t1 * k)
        return cf.astype(np.complex128)
    except:
        return np.zeros(71, dtype=complex)

def poly_188(t1, t2):
    try:
        k = np.arange(71)
        cf = (np.real(t1)**k + np.imag(t2) * k) * (-1)**k + np.log(np.abs(t1 + t2 * k) + 1) + np.sin(t1 * k) * np.cos(t2 * k) + (np.angle(t1) * k - np.angle(t2)) * 1j
        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_189(t1, t2):
    try:
        degrees = np.arange(71)
        cf = t1**(70 - degrees) * (np.cos(degrees) + 1j * np.sin(degrees)) + t2**degrees * (np.cos(degrees) - 1j * np.sin(degrees))
        return cf.astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_190(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        exponents = np.arange(71)
        cf = t1**exponents + (-1)**exponents * t2**(exponents + 1)
        return cf.astype(np.complex128).astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_191(t1, t2):
    try:
        exponents = np.arange(71)
        cf = (t1**exponents) * np.sin(exponents * np.angle(t2)) + (np.conj(t2)**exponents) * np.cos(exponents * np.real(t1))
        return cf.astype(np.complex128).astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_192(t1, t2):
    try:
        degrees = np.arange(71)
        cf = (t1**degrees) * (np.conj(t2)**(degrees % 7)) * (-1)**(degrees // 6) * (1 + degrees / 70)
        return cf.astype(np.complex128).astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_193(t1, t2):
    try:
        j = np.arange(71)
        cf = (t1**j * t2**(70 - j)) * ((-1)**j + np.real(t1) * np.imag(t2) / (j + 1))
        return cf.astype(np.complex128).astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_194(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        exponents = np.arange(71)
        cf = (t1**exponents) * np.real(t2) + (np.conj(t1)**exponents) * np.imag(t2) - np.log(np.abs(t1) + 1)**exponents + np.sin(t1 * exponents) - np.cos(t2 * exponents)
        return cf.astype(np.complex128).astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_195(t1, t2):
    try:
        degrees = np.arange(71)
        cf = np.zeros(71, dtype=complex)
        cf = (t1**degrees) * np.sin(t2 * degrees) + (t2**degrees) * np.cos(t1 * degrees)
        return cf.astype(np.complex128).astype(np.complex128)
    except:
        return np.zeros(0, dtype=complex)

def poly_196(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        exponents1 = np.arange(1, 72)
        exponents2 = np.arange(71, 0, -1)
        terms1 = t1**exponents1
        terms2 = (-1)**np.arange(71) * t2**exponents2
        terms3 = np.sin(t1 * np.arange(71)) * np.cos(t2 * np.arange(1, 72))
        cf = terms1 + terms2 + terms3
        return cf.astype(np.complex128).astype(np.complex128)
    except:
        return np.zeros(71, dtype=complex)

def poly_197(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        j = np.arange(71)
        cf = (np.real(t1)**j * np.sin(j * np.angle(t2)) + np.real(t2)**(70 - j) * np.cos((70 - j) * np.angle(t2))) + \
              (np.imag(t1)**j * np.cos(j * np.angle(t2)) - np.imag(t2)**(j / 2) * np.sin((70 - j) * np.angle(t1)))
        cf = cf * (np.log(np.abs(t1) + 1) * np.log(np.abs(t2) + 1))
        return cf.astype(np.complex128)
    except:
        return np.zeros(71, dtype=complex)

def poly_198(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        k = np.arange(71)
        cf = (np.real(t1) + np.imag(t2))**(70 - k) * (np.abs(t1) + np.abs(t2))**k * np.sin(k * np.angle(t1) - np.angle(t2)) + \
             (np.real(t2) - np.imag(t1))**k * np.cos(k * np.angle(t2) + np.angle(t1)) + np.log(np.abs(t1) + 1) * np.log(np.abs(t2) + 1) / (k + 1)
        return cf.astype(np.complex128).astype(np.complex128)
    except:
        return np.zeros(71, dtype=complex)

def poly_199(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        j = np.arange(71)
        cf = (np.real(t1)**j * np.sin(j * np.real(t2))) + (np.imag(t2)**j * np.cos(j * np.imag(t1))) / (j + 1)
        return cf.astype(np.complex128).astype(np.complex128)
    except:
        return np.zeros(71, dtype=complex)

def poly_200(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(71):
            cf[k] = (np.real(t1)**(70 - k) * np.sin(k * np.angle(t1)) + np.imag(t2)**k * np.cos(k * np.angle(t2))) / (1 + k)
        return cf.astype(np.complex128).astype(np.complex128)
    except:
        return np.zeros(71, dtype=complex)
