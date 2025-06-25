from . import polystate as ps
import math
import cmath
import numpy as np
from scipy.special import sph_harm
from . import letters
from . import zfrm

pi = math.pi

def poly_1(t1, t2):
    try:
        cf = np.zeros(36, dtype=complex)
        for i in range(1, 37):
            cf[i-1] = np.sin(t1**(i/2)) * np.cos(t2**(i/3)) + (i**2) * t1 * t2 + np.log(np.abs(t1 + t2) + 1) * 1j * i
        cf[10] = t1 * t2 * np.real(cf[6]) + np.imag(cf[18]) * t1**3
        cf[21] = t2 * cf[10] + np.real(cf[34]) * t1**3
        cf[32] = cf[21] - np.real(cf[16]) * t1**2
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(36, dtype=complex)

def poly_2(t1, t2):
    try:
        cf = np.zeros(36, dtype=complex)
        cf[0] = t1 + t2
        for k in range(2, 37):
            v = np.sin(k * cf[k-2]) + np.cos(k * t1) + np.real(k * t2) * np.imag(k * cf[k-2])
            cf[k-1] = v / np.abs(v)
        cf[17] = t1**2 + np.real(t1) * t2 - np.imag(t2**2)
        cf[31] = 2 * (t1 + t2) - np.real(t1 * t2) + np.sin(np.real(t1)) * np.cos(np.imag(t2))
        cf[35] = cf[17] * cf[31] + np.sin(np.real(t1 * t2)) - np.cos(np.imag(t1 * t2))
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_3(t1, t2):
    try:
        cf = np.zeros(36, dtype=complex)
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
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_4(t1, t2):
    try:
        cf = np.zeros(36, dtype=complex)
        for k in range(1, 37):
            cf[k-1] = (t1 ** (36 - k) + t2 ** (36 - k)) / (k * 1j)
        cf[16] = t1 * t2 + np.log(np.abs(t1) + 1) - np.sin(t2)
        cf[24] = np.real(t1) - np.imag(t1) + 1j * (np.real(t2) + np.imag(t2))
        cf[29] = np.abs(t1)**2 - np.abs(t2)**2 + 1j * np.angle(t1) * np.angle(t2)
        cf[35] = np.conj(t1 * t2)**2 - np.sin(t1 + t2)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_5(t1, t2):
    try:
        cf = np.zeros(36, dtype=complex)
        p = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53])
        for k in range(1, 17):
            cf[k-1] = np.sin(p[k-1] * t1) + np.cos(p[k-1] * t2)
        for k in range(17, 33):
            cf[k-1] = np.log(np.abs(p[k-17] * t1 + t2)) / (t1 + t2)
        cf[32] = np.prod(p[0:4]) / (t1 * t2)
        cf[33] = np.sum(p[4:8]) - t1**2 + t2**2
        cf[34] = p[8] * p[9] * (t1 + t2)
        cf[35] = p[10] * p[11] / (t1 - t2)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_6(t1, t2):
    try:
        cf = np.zeros(51, dtype=complex)
        for k in range(1, 52):
            cf[k-1] = (t1 + t2) * np.sin(np.log(np.abs(t1 * t2)**k + 1)) + np.cos(np.angle(t1 * t2)**k) * np.conj(t1 - t2)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_7(t1, t2):
    try:
        cf = np.zeros(51, dtype=complex)
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
        cf[49] *= (
            np.real(t1 * t2)
            + np.imag(t1 * t2)
            + np.log(np.abs(t1 * t2) + 1)
        )
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_8(t1, t2):
    try:
        cf = np.zeros(51, dtype=complex)
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
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_9(t1, t2):
    try:
        cf = np.zeros(51, dtype=complex)
        cf[0] = t1 + t2
        for k in range(2, 52):
            cf[k-1] = (np.abs(t1) * np.sin(k) + np.angle(t2) * np.cos(k)) / np.abs(k + 1j)
        cf[9] = cf[0]**2 - cf[1]**2 + np.log(np.abs(cf[2]) + 1)
        cf[19] = np.sum(cf[0:19]) * t1
        cf[29] = np.prod(cf[0:29]) * t2
        cf[39] = cf[38] * cf[37] / (1 + t1 * t2)
        cf[40:50] = np.real(cf[30:40]) + 1j * np.imag(cf[0:10])
        cf[50] = np.sum(cf[0:50])
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_10(t1, t2):
    try:
        cf = np.zeros(51, dtype=complex)
        cf[0] = np.real(t1 * t2) + np.imag(t2) * np.real(t1)
        cf[1] = np.abs(t1 * t2) * np.cos(np.angle(t1 + t2))
        for i in range(2, 51):
            cf[i] = cf[i - 2] * np.abs(cf[i - 1]) * np.sin(np.angle(t1 + t2))
        cf[50] = np.log(np.abs(t1 * t2)) + cf[0] + cf[1]
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_11(t1, t2):
    try:
        cf = np.zeros(51, dtype=complex)
        cf[0] = t1 * t2 - np.log(np.abs(t1 + t2) + 1)
        for i in range(1, 50):
            cf[i] = np.sin(i) * (np.real(t1**i) - np.imag(t2**i)) + np.cos(i) * (np.real(t2**i) - np.imag(t1**i))
            cf[i] = cf[i] / (np.abs(cf[i]) + 1e-10)
        cf[50] = np.abs(t1) * np.abs(t2) * np.angle(t1 + t2)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_12(t1, t2):
    try:
        cf = np.zeros(51, dtype=complex)
        cf[0] = t1 * (2 * (np.imag(t2))**2)
        cf[1] = t2 * (2 * (np.real(t1))**2)
        for k in range(2, 51):
            cf[k] = ((np.abs(t1)**k + np.abs(t2)**(50-k)) / (k**2)) * np.exp(1j * np.angle(t1 * t2))
        cf[22] = np.cos(t1 * t2) * (t1 - 1j * t2)
        cf[34] = np.sin(t1 * t2) * (1j * t2 - t1)
        cf[49] = np.log(np.abs(t1 + t2))**3
        cf[50] = np.conj(t1) * np.conj(t2)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_13(t1, t2):
    try:
        cf = np.zeros(51, dtype=complex)
        fib = np.array([1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181])
        for n in range(19):
            cf[n] = fib[n] * t1 * np.cos(np.angle(t2))
            cf[n + 19] = fib[n] * t1 * np.sin(np.angle(t2))
            cf[n + 38] = fib[n] * t2 * np.sin(np.angle(t1))
        cf[19] = np.abs(t1 * t2)
        cf[49] = np.log(np.abs(t1 * t2) + 1)
        cf[50] = np.real(t1) + np.imag(t2)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_14(t1, t2):
    try:
        cf = np.zeros(51, dtype=complex)
        triangleNums = np.cumsum(np.arange(1, 51))
        cf[0] = t1 + 3 * t2
        for k in range(1, 51):
            cf[k] = triangleNums[k] * (t1 + t2 * np.log(np.abs(t1) + 1))**(k) + triangleNums[k] * (t2 + t1 * np.log(np.abs(t2) + 1))**(k)
        cf[42] = np.real(np.abs(t1)) + np.imag(np.abs(t2))
        cf[20] = np.real(np.abs(t2)) + np.imag(np.abs(t1))
        cf[31] = np.real(np.abs(t1 * t2)) + np.imag(np.conj(t1 * t2))
        cf[27] = 2 * np.real(t1 - t2) + 2 * np.imag(t1 - t2)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_15(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241])
        for i in range(71):
            cf[i] = (primes[i] * t1 + 1j * t2**i) / (1 + np.abs(t1))**i
        cf[70] = np.sum(cf[0:70])
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_16(t1, t2):
    cf = np.zeros(51, dtype=complex)
    cf[0] = t1 + t2
    cf[1] = np.real(t1**2 - t2**2)
    primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                       53, 59, 61, 67, 71, 73, 79, 83, 89, 97])
    for k in range(2, 25):  # Adjusted indices: R is 1-indexed, Python is 0-indexed
        cf[k] = np.imag(cf[k - 1] * primes[k - 2]) * np.angle(t1) * np.abs(t2)
    for k in range(25, 50):
        cf[k] = np.abs(cf[k - 1] * primes[k - 25] ** 2) * np.angle(t2) * np.real(t1)
    cf[50] = np.sum(cf) + np.sin(np.real(t2)) * np.log(np.abs(t1) + 1)
    return cf.astype(np.complex128)

def poly_17(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0:10] = (t1 + t2) * np.arange(1, 11)
        cf[10:20] = np.real(t1 - t2)**3 * np.arange(11, 21)
        cf[20:30] = np.imag(t1 + t2)**2 * np.arange(21, 31)
        cf[30:40] = np.abs(t1 - t2) * np.arange(31, 41)
        cf[40:50] = np.angle(t1 * t2) * np.arange(41, 51)
        cf[50] = np.sin(t1) * np.cos(t2) + np.sin(t2) * np.cos(t1)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_18(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for i in range(1, 72):
            arg = np.angle(t1 * t2 * 1j)
            mod = np.abs((t1 + 1j) * (t2 + 1j))
            cyclotomic = 0
            for k in range(1, i + 1):
                cyclotomic += np.prod(t1 - np.exp(2j * np.pi * k / i))
            cf[i-1] = mod * cyclotomic * arg
        cf[70] = np.log(np.abs(t1 * t2)) + 1
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_19(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = np.real(t1) + np.imag(t2)
        cf[1] = np.angle(t1)
        cf[2] = np.abs(t2)
        cf[3] = np.sin(t1) + np.cos(t2)
        cf[4:10] = np.arange(1, 11) * 0.2 + 1
        cf[10] = np.log(np.abs(t1) + 1) + np.log(np.abs(t2) + 1)
        for i in range(11, 72):
            cf[i-1] = cf[i-2] * np.sin(i * cf[i-3] + np.abs(cf[i-4])) + cf[i-5]
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_20(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for i in range(1, 18):
            cf[i * 3 - 2] = ((t1 + t2)**i + i * (t1 - t2)**i) / (2**i)
        cf[3] = np.abs(t1) * np.sin(np.angle(t1))
        cf[7] = np.abs(t2) * np.cos(np.angle(t2))
        cf[18] = np.log(np.abs(t1 * t2)) * np.cos(np.angle(t1 - t2))
        cf[36] = np.abs(t1 * t2) * np.cos(np.angle(t1 + t2))
        cf[[20, 24, 28, 32, 36, 40, 44, 48, 50]] = np.real(t1) + np.imag(t2)
        cf[[22, 26, 30, 34, 38, 42, 46]] = np.imag(t1) + np.real(t2)
        cf[49] = np.abs(t1)**2 * np.sin(2 * np.angle(t2))
        cf[50] = np.abs(t2)**2 * np.cos(2 * np.angle(t1))
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_21(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        roots = np.exp(2 * np.pi * 1j * np.arange(0, 51) / 51)
        for k in range(1, 72):
            cf[k-1] = np.prod(roots[np.arange(51) != k-1] - roots[k-1]) / (t1 - roots[k-1]) / (t2 - roots[k-1])
        return cf.astype(np.complex128) * (t1 - roots) * (t2 - roots)
    except Exception:
        return np.zeros_like(cf)

def poly_22(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97])
        cf[0:25] = primes[0:25] * (np.real(t1)**2 - np.imag(t1**2) + np.real(t2)**2 - np.imag(t2**2))
        cf[25:50] = cf[0:25] * (np.cos(np.angle(t1 + t2)) + np.sin(np.abs(t1) * np.abs(t2)))
        cf[50] = np.sum(cf[0:50])
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_23(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_24(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = t1 + t2
        cf[1] = t1 * t2
        for n in range(3, 72):
            cf[n-1] = np.abs(cf[n-2]) + 1j * np.angle(cf[n-3]) + np.abs(t1 + t2)**(1/n) * (np.cos(n * t2) + 1j * np.sin(n * t1))
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_25(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = np.real(t1 * t2) + np.imag(t1 / t2)
        for k in range(1, 72):
            cf[k] = np.abs(t1)**k + np.angle(t2)**k + np.sin(t1 + k) + np.cos(t2 + k) - np.log(np.abs(t1 * t2)**k + 1)
        cf[35] = np.real(cf[0] * cf[34]) + np.imag(t1 * t2)
        cf[45] = 0.5 * (t1 + np.conj(cf[44]) + t2)
        cf[50] = cf[0] + cf[34] + cf[44] + np.real(t1) + np.imag(t2)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_26(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = t1 + t2
        cf[1] = t1 * t2
        for k in range(3, 72):
            v = np.sin(k * cf[k-2]) + np.cos(k * cf[k-3])
            cf[k-1] = v / np.abs(v)
        cf[14] = np.abs(t1 - t2) * np.angle(t1 + t2)
        cf[29] = np.log(np.abs(t1 * np.real(t2) + 1)) - np.log(np.abs(t2 * np.imag(t1) + 1))
        cf[49] = np.prod(np.abs(t1), np.abs(t2)) * np.abs(t1 - t2)
        cf[50] = np.sum(cf[15:29]) * np.sum(cf[30:44]) + t1**2
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_27(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_28(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53])
        for k in range(1, 17):
            cf[k-1] = (np.real(t1) + np.imag(t2)) / primes[k-1]
            cf[71-k] = (np.imag(t1) + np.real(t2)) * primes[k-1]
        for k in range(17, 36):
            cf[k-1] = np.sin((np.real(t1) + np.imag(t1))**2) * (np.real(t2) + np.imag(t2))**(2 + k)
        cf[35] = np.log(np.abs(t1) * np.abs(t2) + 1) + np.abs(t2 - t1)
        cf[36:51] = np.angle(t1 + t2) + np.abs(t1 - t2) + np.angle(np.conj(t1 * t2))
        cf[50] = np.sum(cf[0:50])**2
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_29(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0:5] = np.array([1, t1, t1**2, t1**3, t1**4])
        cf[5:10] = np.array([1, t2, t2**2, t2**3, t2**4])
        cf[10:15] = np.array([1, np.exp(1j * t1), np.exp(2j * t1), np.exp(3j * t1), np.exp(4j * t1)])
        cf[15:20] = np.array([1, np.exp(1j * t2), np.exp(2j * t2), np.exp(3j * t2), np.exp(4j * t2)])
        cf[20:30] = np.array([1, np.real(t1 + t2), np.imag(t1 + t2), np.real(t1 * t2), np.imag(t1 * t2), np.real(t1 + t2)**2, np.imag(t1 + t2)**2, np.real(t1 * t2)**2, np.imag(t1 * t2)**2, np.abs(t1 + t2)])
        cf[30:40] = np.arange(1, 11) * np.abs(t1) * np.abs(t2)
        cf[40:50] = np.array([1, np.log(np.abs(t1) + 1), np.log(np.abs(t2) + 1), np.log(np.abs(t1 + t2) + 1), np.log(np.abs(t1 * t2) + 1), np.angle(t1), np.angle(t2), np.abs(t1), np.abs(t2), np.angle(t1 + t2)])
        cf[50] = np.abs(t1 + t2) * np.angle(t1 * t2)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_30(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for i in range(1, 72):
            cf[i-1] = (np.real(t1) * np.imag(t2) + np.imag(t1) * np.real(t2))**(1/i) * np.abs(t2)**(i/50) * np.sin(np.angle(t1) * (i/25)) * np.cos(np.angle(t2) * (i/50)) + np.log(np.abs(t1) + 1) + np.log(np.abs(t2) + 1)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_31(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = t1 + t2
        for i in range(1, 36):
            cf[i] = (np.cos(i * t1) + np.sin(i * t2)) / (np.abs(t1) * np.abs(t2))**i
        for i in range(36, 72):
            cf[i] = (np.cos(t1**i) + np.sin(t2**i)) * np.log(np.abs(t1)**i + 1) * np.log(np.abs(t2)**i + 1)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_32(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = t1 + t2
        for k in range(1, 72):
            r = np.abs(t1)**k + np.abs(t2)**(71-k)
            theta = np.angle(t1)**k - np.angle(t2)**(71-k)
            cf[k-1] = r * np.cos(theta) + r * np.sin(theta) * 1j
        cf[2:70] += np.log(np.abs(t2 - t1) + 1)
        cf[70] += np.conj(t1 * t2)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_33(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        f = lambda z, n: z**n - 1
        cf[0:35] = [np.real(f(t1, n)) - np.imag(f(t2, n)) for n in range(1, 36)]
        cf[35:70] = [np.log(np.abs(f(t2, n))) + np.angle(f(t1, n)) + np.sin(np.abs(f(t1, n))) + np.cos(np.angle(f(t2, n))) for n in range(1, 36)]
        cf[70] = np.prod(cf[0:70])
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_34(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for i in range(1, 72):
            cf[i-1] = (t1 * t2) * np.sin(i * t1 + t2) + np.cos(i * t1 - t2) + i * np.log(np.abs(t1 + i * t2) + 1) / (1 + np.abs(t1 + i * t2))
        cf[12] = 3 * (t1**2 - t2**2)
        cf[13] = cf[12] + t1 * t2 * np.sin(np.angle(t1 + t2))
        cf[14] = 2 * cf[13] - t1 * t2 * np.cos(np.angle(t1 - t2))
        cf[15] = 3 * cf[12] - cf[13] + t1 * t2 * np.sin(2 * np.angle(t1 - t2))
        cf[16] = 2 * cf[12] - 3 * cf[13] + cf[14] - t1 * t2 * np.cos(2 * np.angle(t1 + t2))
        cf[69] = 2 * cf[13] - 3 * cf[12] + t1 * t2 * np.sin(np.angle(2 * t1 - t2))
        cf[70] = cf[16] - cf[13] * t1 * t2 * np.cos(np.angle(2 * t1 + t2))
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_35(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(1, 71):
            z = np.cos(t1) * t1**k - np.sin(t2) * t2**k
            cf[k-1] = np.real(z) + 1j * np.imag(z)
        cf[70] = np.abs(t1 * t2)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_36(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = t1 + t2 * 1j
        cf[2] = np.abs(t1)**2 + np.angle(t2)**2
        cf[8] = np.sin(t1 + t2)
        for k in range(9, 72):
            cf[k-1] = np.cos(k * np.real(t1 + t2)) + np.sin(k * np.imag(t1 * np.conj(t2)))
        cf[70] = np.log(np.abs(t1 * t2) + 1)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_37(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for i in range(1, 72):
            cf[i-1] = (np.sin(i * t1) + np.cos(i * t2)) * i**2
        cf[1] += np.sum(cf[0:2])
        cf[4] += np.prod(cf[0:5])
        cf[11] += np.log(np.abs(cf[10]) + 1)
        cf[24] += np.angle(cf[23])
        cf[[34, 44, 54, 64]] += np.abs(t2)**2 + np.real(t1)**3
        cf[[6, 13, 20, 27, 34, 41, 48, 55, 62, 69]] += np.sin(t1)**i - np.cos(t2)**i
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_38(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(1, 71):
            cf[k-1] = (1 + t1 * t2)**k / (1 + np.real(t1 * t2)**2)
        cf[70] = np.abs(t1) + np.abs(t2) + np.angle(t1 + t2) + np.sin(np.real(t1) + np.imag(t2)) + np.log(np.abs(np.real(t2) + 1))
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_39(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = t1
        cf[1] = t2
        for k in range(3, 72):
            cf[k-1] = np.sin(k * t1) + np.cos(k * t2) + np.log(np.abs(k) + 1) * np.abs(cf[k-2]) * np.abs(cf[k-3]) * np.abs(np.angle(t1 + t2))
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_40(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(1, 71):
            cf[k-1] = (np.real(t1)**(k + 1)) * np.sin(np.angle(t2 * k)) + (np.imag(t2)**k) * np.cos(np.angle(t1 / k))
        cf[70] = np.abs(t1) + np.abs(t2)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_41(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = t1 + t2
        cf[1] = (t1 + t2) * (t1 - t2)
        cf[2] = np.abs(np.real(t1))**2 + np.abs(np.imag(t1))**2 + np.abs(np.real(t2))**2 + np.abs(np.imag(t2))**2
        for i in range(3, 72):
            cf[i-1] = cf[i-2] * t1 + cf[i-3] * t2 + cf[i-4]
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_42(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0:35] = np.abs(t1) * np.sin((np.arange(1, 36)) * np.angle(t1))
        cf[35:70] = np.real(t2) * np.cos((np.arange(1, 36)) * np.imag(t2))
        cf[70] = t1 * t2 + 1j * np.sum(np.log(np.abs(cf[0:70]) + 1))
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_43(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = t1 * t2
        cf[1] = np.log(np.abs(t1) + 1) + np.log(np.abs(t2) + 1)
        for k in range(2, 72):
            cf[k-1] = np.sin(k * cf[k-2]) + np.cos(k * t1) - np.sin(k * cf[k-3]) + np.cos(k * t2)
            cf[k-1] = cf[k-1] / np.abs(cf[k-1])
        cf[34] = np.real(t1)**3 - np.imag(t2)**3
        cf[52] = np.abs(t1 * t2)**2 - np.angle(t1 * t2)
        cf[70] = np.real(t1 * t2) - np.imag(t1 * t2) + np.angle(t1 * t2)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_44(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_45(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(1, 72):
            if k % 2 == 0:
                cf[k-1] = k * (t1 + np.real(t2)) * np.sin(np.abs(t1) * k)
            else:
                cf[k-1] = k * (t2 - np.imag(t1)) * np.cos(np.angle(t2) * k)
        for i in range(2, len(cf) // 2):
            cf[i-1] = cf[i-2] * (np.abs(t1) + 0.5) + np.log(np.abs(t2) + 1)
            cf[len(cf) - i] = -cf[len(cf) - i + 1] * (np.abs(t2) + 0.5) - np.log(np.abs(t1) + 1)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_46(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        primes = np.array([2, 3, 5, 7, 11, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67])
        cf[0:18] = np.real(primes * (t1 + t2))
        cf[18:36] = np.imag(primes * (t1 - t2))
        cf[36:54] = np.real(primes * (t1 * np.conj(t2)) + np.log(np.abs(primes)))
        cf[54:71] = np.imag(t1**(np.arange(1, 17))) * t2**(np.arange(1, 17)**2)
        cf[70] = np.sum(t1**(np.arange(1, 6)) * t1**(np.arange(1, 6)**2)) + np.sum(t2**(np.arange(1, 11)))**2
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_47(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_48(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_49(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_50(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros_like(cf)

def poly_51(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(1, 71):
            if k % 2 == 0:
                cf[k] = np.sin(k * (t1 + t2) ** k) + np.cos(k * (t1 - t2) ** k)
            else:
                cf[k] = np.real(t1) ** k + np.imag(t2) ** k
        cf[70] = np.abs(t1) ** 3 + np.angle(t2)
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_52(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = t1 ** 7 + t2 ** 7
        for k in range(2, 36):
            cf[k - 1] = np.sin(k * np.log(np.abs(t1) + 1) + np.log(np.abs(t2) + 1))
        for k in range(36, 71):
            cf[k - 1] = np.cos(k * np.log(np.abs(t1) + 1) - np.log(np.abs(t2) + 1))
        cf[70] = t1 * t2 - (t1 + t2) ** 2
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_53(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_54(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for i in range(1, 72):
            z = t1 * np.cos(i * t2 / 15) + t2 * np.sin(i * t1 / 15)
            phi = np.angle(z)
            r = np.abs(z)
            cf[i - 1] = r * np.exp(1j * phi) ** i + (-1) ** (i + 1) * i ** 2
        cf[0:30] = cf[0:30] * (np.abs(t1) * np.abs(t2)) ** np.arange(1, 31)
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_55(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = t1 * t2
        cf[1] = np.real(t1) * np.imag(t2)
        cf[2] = np.real(t2) * np.imag(t1)
        cf[3:10] = np.linspace(np.log(np.abs(cf[0]) + 1), np.log(np.abs(cf[2]) + 1), 7)
        cf[10:30] = [np.cos(cf[i - 1]) + ((t1 + t2) ** i) / (i + 1) for i in range(11, 31)]
        cf[30:50] = [np.sin(cf[i - 1]) + ((t1 - t2) ** i) / (i + 1) for i in range(31, 51)]
        cf[50:70] = np.abs(cf[0:20]) + np.abs(cf[20:40] + t1 + t2)
        cf[70] = np.prod(cf[0:70])
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_56(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(1, 72):
            cf[k - 1] = ((t1 + 1j * t2) ** k + (t2 + 1j * t1) ** (71 - k)) / 2
        cf[3:68] = cf[3:68] * (1 + np.sin(np.angle(t1 + 1j * t2)))
        cf[0:3] = cf[0:3] * (1 + np.cos(np.angle(t1 + 1j * t2)))
        cf[68:71] = cf[68:71] * np.abs(t1 + 1j * t2)
        cf[34] = cf[34] * np.log(np.abs(np.imag(t1 + 1j * t2)) + 1)
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_57(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(1, 36):
            cf[k - 1] = (t1 * k ** 2 + t2 * (70 - k)) * (1 - (-1) ** k) / 2
        for k in range(36, 71):
            cf[k - 1] = (t1 * np.conj(t2)) ** k * np.abs(t1 - t2)
        cf[70] = np.abs(t1 * np.real(t2)) * np.abs(t2 - t1)
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_58(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = 100 * np.abs(t1) * np.abs(t2) - 100
        cf[1] = 100 * t1 ** 3 - t2 ** 3 + 100
        cf[2] = 100 * t2 ** 3 - t1 ** 3 + 100
        cf[3:71] = [np.cos(k * t1) * np.sin(k * t2) / np.log(np.abs(k + 1)) for k in range(1, 69)]
        root_coeff = np.abs(t1) * np.abs(t2) * np.prod(range(1, 71)) / np.sum(range(1, 71))
        cf[4] = root_coeff * np.sum([np.cos(k * t1) * np.sin(k * t2) / np.log(np.abs(k + 1)) for k in range(1, 71)])
        cf[35] = root_coeff * t1 ** 2
        cf[34] = root_coeff * t2 ** 2
        cf[36:71] = root_coeff * [np.cos(2 * k * t1) * np.sin(2 * k * t2) / np.log(np.abs(2 * k + 1)) for k in range(1, 36)]
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_59(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for i in range(1, 72):
            cf[i - 1] = (t1 / (i + 1)) ** i + (t2 / (i + 1)) * (2j)
        cf[[1, 3, 5, 7, 9, 11, 13, 15, 17, 19]] *= (t1 + 2 * t2)
        cf[[2, 5, 8, 11, 14, 17, 20, 23, 26, 29]] *= (t1 - 2 * t2)
        cf[4:36] += 2 * t1
        cf[36:67] -= 2 * t2
        cf[67:71] = np.real(np.log(cf[67:71])) + np.sum(cf[4])
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_60(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_61(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0:35] = np.real(t1) * (np.arange(1, 36) ** 3) + np.imag(t2) * np.sin(np.arange(1, 36))
        cf[35:70] = np.imag(t1) * (np.arange(70, 35, -1) ** 2) + np.real(t2) * np.cos(np.arange(70, 35, -1))
        cf[70] = np.abs(t1) * np.angle(t2) - np.abs(t2) * np.angle(t1)
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_62(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_63(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for j in range(1, 72):
            cf[j - 1] = np.sin(t1 ** j) * np.cos(t2 ** (71 - j)) * np.abs(t1 * t2 ** j) * np.log(np.abs(t1 * t2 + 1))
        cf[0:30] += cf[30:60]
        cf[32:72] -= cf[0:40]
        cf[10:60] += np.real(t1) * np.imag(t2) * cf[0:50]
        cf[30:70] -= np.imag(t1) * np.real(t2) * cf[1:41]
        cf[20:40] += np.angle(t1 ** t2) * cf[30:50]
        cf[40:72] -= np.angle(t2 ** t1) * cf[0:32]
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_64(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(1, 72):
            cf[k - 1] = (np.abs(t1) ** k + np.abs(t2) ** (71 - k)) * np.cos(np.angle(t1) * k + np.angle(t2) * (71 - k))
        cf[1::2] *= 1j
        cf[2::3] *= -1
        cf[0] *= 100
        cf[70] /= 100
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_65(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(1, 72):
            cf[k - 1] = (t1 + t2) ** (2 * k - 1) + np.sin(k * t1) * np.cos(k * t2) + np.log(np.abs(k ** t2) + 1) * np.real(t1 ** t2) + np.abs(np.imag(t1 ** (2 * k + 1) + t2 ** (2 * k)))
            cf[k - 1] = np.conj(cf[k - 1]) * (-1) ** k
            if k % 2 == 0:
                cf[k - 1] /= (k + t1)
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_66(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0:25] = np.real(t1) * np.log(np.abs(t2) + 1) * (np.arange(1, 26) ** 2)
        cf[25:50] = np.imag(t2) * np.log(np.abs(t1) + 1) * (np.arange(1, 26) ** 3)
        cf[50:70] = np.abs(t1) * np.abs(t2) * np.log(np.abs(t1 + t2) + 1) * (np.arange(1, 21))
        cf[70] = np.sum(cf[0:70]) * np.angle(t1 + t2)
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_67(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_68(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0:5] = np.abs(t1) ** (np.arange(1, 6))
        for i in range(6, 71):
            cf[i] = (i * t1 + 2 * i * t2) / (i + 1)
        cf[70] = np.abs(t1) * np.abs(t2) * np.angle(t1) * np.angle(t2) * np.sin(np.abs(t1 + t2))
        cf[20:30] += np.log(np.abs(t1 + t2) + 1) * np.exp(1j * np.pi / 10 * np.arange(1, 11))
        cf[50:60] += 1j * (cf[0:10] / np.arange(11, 21))
        cf[60:70] -= np.sin(cf[0:10])
        cf[30:40] += np.cos(t1 + t2) ** np.arange(1, 11)
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_69(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for i in range(1, 72):
            j = 71 - i
            cf[i - 1] = ((np.real(t1) + np.imag(t1) * j) / np.abs(t2 + i)) * np.sin(np.angle(t1 + t2 * i)) + np.log(np.abs(t1 * t2) + 1) * np.cos(2 * np.pi * i / 71)
        cf[cf == 0] = np.real(t1) ** 2 - np.imag(t1) * np.imag(t2)
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

#  cf = complex(71)
#   for (i in 1:71) {
#     cf[i] = Re(t1) * Re(t2) * (i^(2))/(exp(np.abs(t1)*1i)) + Im(t1) * Im(t2) * (i^3)/(exp(np.abs(t2)*1i))
#   }
#   cf[2:2:length(cf)] = cf[2:2:length(cf)] * (-1)
#   p = 1:71
#   cf[p^2 <= 71] = cf[p^2 <= 71]+1i*Mod(t1)*Mod(t2)
#   cf
def poly_70(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for i in range(1, 72):
            cf[i - 1] = np.real(t1) * np.real(t2) * (i ** 2) / np.exp(np.abs(t1) * 1j) + np.imag(t1) * np.imag(t2) * (i ** 3) / np.exp(np.abs(t2) * 1j)
        cf[1::2] *= -1
        p = np.arange(1, 72)
        cf[p ** 2 <= 71] += 1j * np.abs(t1) * np.abs(t2)
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_71(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_72(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = t1 + t2
        cf[1:71] = np.arange(1, 71) * (t1 - t2 + (np.sin(np.arange(1, 71)) + 1j * np.cos(np.arange(1, 71))))
        roots = np.abs(cf[1:71])
        sorted_roots = np.sort(roots)[::-1]
        cf[1:71] = sorted_roots * (t1 + t2 * 1j * np.arange(1, 71))
        cf = np.real(cf) + np.imag(cf) + (np.sin(np.angle(cf + 1)) + 1j * np.cos(np.angle(cf + 1)))
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_73(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = 10 ** 30 * (t1 + t2)
        cf[1] = 10 ** 28 * (t1 - t2)
        cf[2] = 10 ** 26 * (t1 + t2)
        for k in range(4, 22):
            cf[k - 1] = 10 ** (30 - k) * (np.cos(t1) + np.sin(t2))
        for k in range(22, 32):
            cf[k - 1] = 10 ** (k - 21) * (np.cos(t1) - np.sin(t2))
        for k in range(32, 42):
            cf[k - 1] = 10 ** (42 - k) * (t1 + t2) * (np.cos(t1 + t2) + np.sin(t1 - t2))
        cf[41] = 10 ** 21 * (t1 - t2)
        for k in range(43, 54):
            cf[k - 1] = 10 ** (53 - k) * (np.abs(t1 + t2) + np.angle(t1 - t2))
        for k in range(54, 65):
            cf[k - 1] = 10 ** (64 - k) * (np.abs(t1 - t2) + np.angle(t1 + t2))
        for k in range(65, 72):
            cf[k - 1] = 10 ** (71 - k) * (np.sin(t1) + np.cos(t2))
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_74(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for i in range(1, 36):
            cf[i - 1] = i * (t1 + i * t2) ** (1 / i)
            cf[70] = np.conj(cf[i - 1])
        cf[35] = 2 * t1 + 3 * np.abs(t2)
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_75(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        powers = np.arange(0, 71)
        cf[0] = 100 * t1 ** 3 + 110 * t1 ** 2 + 120 * t2 - 130
        cf[1] = 200 * t2 ** 3 - 210 * t2 ** 2 + 220 * t2 - 230
        cf[4] = np.abs(t1) ** 4
        cf[9] = np.angle(t2) ** 6
        cf[14] = np.log(np.abs(t1 + 1j * t2)) + 1
        cf[[19, 39]] = np.real(1j * cf[4] * t1 * t2)
        cf[[29, 59]] = np.imag(cf[1] * np.conj(cf[0]))
        cf[34] = np.sin(cf[1]) + np.cos(cf[0])
        cf[2] = np.abs(cf[9]) ** 2
        cf[3] = np.prod(cf[[2, cf[1]]])
        cf[8] = np.sum(cf[[19, 39, 59]])
        cf[15:71] = powers[15:71] * np.abs(t1 - t2)
        cf[70] = np.prod(cf[0:4])
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_76(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
    except:
        return np.zeros_like(cf)

def poly_77(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
    except:
        return np.zeros_like(cf)

def poly_78(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0:10] = np.abs(t1) ** (np.arange(1, 11) / 5) * np.log(1 + np.abs(t2))
        cf[10:20] = np.real(t1) ** (np.arange(1, 11)) * np.angle(t2) * (-1) ** (np.arange(1, 11))
        cf[20:30] = np.imag(t1) ** (np.arange(1, 11) / 3) * np.abs(t2) ** (np.arange(1, 11) / 4) * (-1) ** (np.arange(1, 11))
        cf[30:40] = np.abs(t1 * t2) ** (np.arange(1, 11) / 2) * (np.arange(1, 11))
        cf[40:50] = np.real((t1 + t2) ** (np.arange(1, 11) / 2)) * np.cos(np.angle(t1 * t2)) * (-1) ** (np.arange(1, 11))
        cf[50:60] = np.imag((t1 + t2) ** (np.arange(1, 11) / 3)) * np.sin(np.angle(t1 - t2)) * (-1) ** (np.arange(1, 11))
        cf[60:70] = np.real(t1 ** (np.arange(1, 11))) * np.abs(t2 ** (np.arange(1, 11))) * np.log(1 + np.abs(t1 + t2))
        cf[70] = np.abs(t1 - t2) * np.log(1 + np.abs(t1))
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_79(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0:35] = (np.arange(1, 36)) * (t1 + t2) * np.abs(t1) ** (np.arange(1, 36))
        cf[35:70] = (np.arange(35, 0, -1)) * (t1 - t2) * np.abs(t2) ** (np.arange(35, 0, -1))
        cf[70] = np.abs(t1) * np.abs(t2) + np.imag(t1 * np.conj(t2))
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_80(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = t1 ** 70 + t2 ** 70
        cf[1:70] = np.sin(t1 * t2 * (np.arange(1, 70))) ** 2
        cf[13:28] = np.log(np.abs(t2) + 1) ** 2 * cf[13:28]
        cf[30:46] *= np.log(np.abs(t1) + 1)
        for i in range(2, 5):
            cf[i * 15] += i * np.abs(t1) * np.abs(t2)
        cf[70] = np.real(t1) ** 3 - np.imag(t2) ** 2
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_81(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(1, 72):
            cf[k - 1] = (k + 20) * np.sin(t1 * k) * np.cos(t2 * k) + np.abs(t1) ** k + np.abs(t2) ** k
        cf[np.arange(2, 71, 5)] += np.abs(t1) * np.abs(t2)
        cf[np.arange(3, 70, 7)] += ((-1) ** (np.arange(1, 11))) * np.angle(t1 + t2)
        cf[np.arange(6, 67, 9)] += ((-1) ** (np.arange(1, 8))) * np.log(np.abs(t1 + t2) + 1)
        cf[np.arange(5, 71, 7)] *= np.real(t1 + t2)
        cf[np.arange(7, 64, 11)] *= np.imag(t1 + t2)
        cf[np.arange(1, 72, 7)] *= np.conj(t1 + t2)
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_82(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = 1 + t1 * t2
        cf[1] = 2 + np.abs(t1) * np.abs(t2)
        cf[2] = 3 + np.abs(t1 + t2)
        for i in range(4, 37):
            cf[i - 1] = i + cf[i - 2] * np.sin(i * t2) + cf[i - 3] * np.cos(i * t1) + cf[i - 4] * np.log(np.abs(i * t1 * t2 + 1))
        for i in range(37, 71):
            cf[i - 1] = 70 - i + cf[70 - min(i, 69)] * np.sin((70 - i) * t1) + cf[69 - min(i, 68)] * np.cos((70 - i) * t2) + cf[68 - min(i, 67)] * np.log(np.abs((70 - i) * t1 * t2 + 1))
        cf[70] = np.sum(cf[0:70]) + np.real(np.angle(t1 - t2)) + np.imag(np.angle(t1 + t2))
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_83(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for i in range(1, 72):
            cf[i - 1] = ((t1 + t2) ** (i - 1)) * np.sin(i) + ((t1 - t2) ** (70 - i + 1)) * np.cos(i)
        cf[0] = np.real(cf[0]) + 1j * np.imag(cf[70])
        cf[70] = np.real(cf[70]) + 1j * np.imag(cf[0])
        cf[35] += np.log(np.abs(t1 * t2)) ** 2
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_84(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = np.real(t1) + 1000j
        cf[1] = np.log(1 + np.abs(t1 + t2)) * 1000
        for k in range(3, 36):
            cf[k - 1] = (-1) ** k * (np.real(t1 ** k) + np.imag(t2 ** k)) * 1000 / (k ** 2)
        for k in range(36, 71):
            cf[k - 1] = (-1) ** (k + 1) * (np.abs(t1) ** (70 - k) + np.abs(np.sin(t1 + t2))) / (k ** 2)
        cf[70] = np.abs(t1) + np.cos(np.angle(t2)) * 1000
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_85(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_86(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(1, 72):
            cf[k - 1] = np.cos(k * t1) ** k + 1j * np.sin(k * t2) ** k
        cf[1::2] **= -1
        cf[2::3] **= -2
        for r in range(5, 66, 5):
            cf[r - 1] = (t1 * t2) ** r
        cf[70] = (np.abs(t1) ** 2 + 2 * np.real(t1) * np.imag(t2) + 3 * np.abs(t2) ** 2)
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_87(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(1, 36):
            cf[k - 1] = (t1 + 1j * t2) ** k + np.log(np.abs(t1 + k * t2) + 1) * np.real(t1 * t2)
            cf[70 - k] = k * (t1 - 1j * t2) ** k - np.log(np.abs(t2 - k * t1) + 1) * np.imag(t1 * t2)
        cf[35] = 100 * np.abs(t1) * np.abs(t2)
        cf[36] = 200 * np.angle(t1) * np.angle(t2)
        cf[37:72] = cf[0:34] - cf[37:72]
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_88(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = np.log(np.abs(t1) + 1) * np.log(np.abs(t2) + 1)
        for k in range(2, 37):
            cf[k - 1] = np.sin(k * t1) + np.cos(k * t2) - k ** 2
        for k in range(37, 71):
            cf[k - 1] = np.sin((71 - k) * t1) - np.cos((71 - k) * t2) + (71 - k) ** 2
        cf[70] = np.real(t1 * t2) + np.imag(t1 * t2)
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_89(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
    except:
        return np.zeros_like(cf)

def poly_90(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = t1 ** 3 - t2 ** 2
        cf[1] = np.real(3 * t1 ** 2 * t2 - t2 ** 3)
        cf[2] = np.imag(3 * t1 * t2 ** 2 - t1 ** 3)
        cf[3] = 4 * t1 ** 2 - 6 * t1 * t2 + 4 * t2 ** 2
        for k in range(5, 72):
            cf[k - 1] = np.abs(t1 * t2) * np.sin(k * t1 + t2) + np.cos(k * np.conj(t1 + t2))
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_91(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_92(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(1, 72):
            cf[k - 1] = t1 ** k + t2 ** (71 - k)
        cf[14:25] = 1j * cf[14:25]
        cf[29:45] = np.conj(cf[29:45])
        cf[45:70] = -cf[45:70]
        cf[54:71] = cf[54:71] * (1 + 2j)
        cf[70] = np.real(t1) ** 3 + np.imag(t2) ** 3 - np.log(np.abs(t1) * np.abs(t2) + 1)
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_93(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(1, 36):
            cf[k - 1] = k * (t1 ** (2 * k)) * (t2 ** (2 * (35 - k))) / np.sin(k * np.pi / 180)
            cf[71 - k] = k * (t2 ** (2 * k)) * (t1 ** (2 * (35 - k))) / np.cos(k * np.pi / 180)
        cf[35] = 100 * np.real(t1) * np.imag(t2) + 100 * np.imag(t1) * np.real(t2)
        cf[70] = np.abs(t1 + t2) * np.angle(t1 - t2)
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_94(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
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
    except:
        return np.zeros_like(cf)



def poly_95(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = np.real(t1) ** 3 - np.imag(t1) ** 2 + 2 * np.imag(t1) * np.real(t1) - np.real(t2) + np.imag(t2) ** 2
        cf[1] = np.imag(t1) ** 3 - 5 * np.real(t1) ** 2 + 2 * np.real(t1) * np.imag(t1) + 5 * np.real(t2) - 2 * np.imag(t2) ** 2
        for k in range(3, 72):
            cf[k - 1] = np.abs(np.sin(k * t1)) + np.abs(np.cos(k * t2)) - np.abs(t1 ** k + t2 ** (k - 1))
        cf[29:40] = np.abs(cf[29:40]) / (np.abs(t1 - t2) ** 2 + 1)
        cf[49:60] = -np.abs(cf[49:60]) / (np.abs(t1 + t2) ** 2 + 1)
        cf[64:71] = cf[0:7] * (np.abs(t1) ** 2 + np.abs(t2) ** 2)
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_96(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = t1 ** 5 - t2 ** 4 + t1 ** 2 - t2 ** 2 + np.abs(t1) + np.abs(t2)
        cf[50] = t2 ** 6 - t1 ** 4 + t2 ** 3 - t1 ** 2 + np.angle(t1) + np.sin(t2)
        cf[70] = t1 ** 7 + t2 ** 5 - t1 ** 3 - t2 ** 2 + np.cos(t1) - np.sin(t2)
        for k in range(2, 51):
            cf[k - 1] = k * cf[k - 2] + np.abs(cf[0]) / k
        for r in range(52, 71):
            cf[r - 1] = r * cf[r - 2] + np.abs(cf[50]) / r
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_97(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(1, 72):
            cf[k - 1] = (t1 + 1j * t2) * k ** (-np.abs(t1) * np.log(np.abs(k + 1)))
        for k in range(1, 11):
            cf[k - 1] = (t1 + 1j * t2) * np.abs(k ** 3 * np.cos(np.imag(t1 + k * t2)) - np.sin(np.real(t1 - k * t2)))
        for k in range(61, 72):
            cf[k - 1] = (t1 + 1j * t2) * np.abs(k * cf[k - 2]) / np.abs(k ** 3 * np.cos(np.imag(t1 + k * t2)))
        return cf.astype(np.complex128)
    except:
        return np.zeros(71, dtype=complex)

def poly_98(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        for k in range(1, 36):
            cf[k - 1] = (t1 + t2) * np.sin(t2 * k) / k ** 2
        for k in range(36, 71):
            cf[k - 1] = (t1 - t2) * np.cos(t1 * (71 - k)) / (71 - k) ** 2
        cf[70] = np.real(t1) * np.imag(t2) - np.real(t2) * np.imag(t1)
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_99(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        cf[0] = t1 + t2 + 1
        cf[1:10] = np.real(t1) ** 2 + np.imag(t2) ** 2
        cf[10:20] = np.real(t2) ** 2 + np.imag(t1) ** 2
        cf[20:30] = np.abs(t1 * t2) ** 2
        cf[30:40] = np.abs(t1 + t2) ** 2
        cf[40:50] = np.abs(t1) * np.abs(t2)
        cf[50:60] = np.angle(t1) + np.angle(t2)
        cf[60:70] = np.sin(t1 + t2)
        cf[70] = np.cos(t1 - t2)
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

def poly_100(t1, t2):
    try:
        cf = np.zeros(71, dtype=complex)
        iter = complex(1)
        for j in range(1, 72):
            cf[j - 1] = iter
            iter *= (np.log(np.abs(t1 + 1j * t2) + 1) / (71 - j + 1) + np.conj(iter))
        return cf.astype(np.complex128)
    except:
        return np.zeros_like(cf)

