from . import polystate as ps
import math
import cmath
import numpy as np
from scipy.special import sph_harm
from . import letters
from . import zfrm
from . import polychess as pc
from . import xfrm

pi = math.pi

def poly_801(t1, t2):
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
        cf[10:15] = np.abs(t1) ** (np.arange(11, 16) / 10) * np.abs(t2) ** (np.arange(15, 10, -1) / 10)
        cf[15:20] = (np.arange(16, 21) * (np.angle(t1) + np.angle(t2))) / 2
        cf[20:25] = np.real(t1) ** 2 + np.imag(t2) ** 2 + np.arange(21, 26)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(25, dtype=complex)


def poly_802(t1, t2):
    cf = np.zeros(25, dtype=complex)

    # Adjusted for Python's 0-based indexing
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

def poly_803(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        cf[0] = t1 + t2
        cf[1] = np.abs(t1) * np.abs(t2)
        cf[2] = np.real(t1 * t2)
        cf[3] = np.imag(t1 * t2)
        cf[4] = np.angle(t1 + t2)
        for k in range(5, 26):
            cf[k] = cf[k - 1] + np.cos(k * t1) + np.sin(k * np.abs(t2)) * 1j
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(25, dtype=complex)

def poly_804(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        cf[0:10] = np.arange(1, 11) * t1 + np.arange(11, 21) * 1j * t2
        cf[10:20] = (t1 + 1j * t2) ** 2 * np.arange(11, 21)
        cf[20:25] = (np.abs(t1) + np.angle(t2)) * np.arange(1, 6)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(25, dtype=complex)

def poly_805(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        cf[0] = t1 ** 2 + t2 ** 2
        cf[1] = t1 * t2
        cf[2:8] = t1 * t2 ** np.arange(1, 7)
        cf[8:14] = (t1 ** 2 + t2 ** 2) / np.arange(1, 7)
        cf[14:20] = t1 ** 3 * t2 ** np.arange(1, 7)
        cf[20:25] = (t1 + t2) ** np.arange(1, 6)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(25, dtype=complex)

def poly_806(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
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
    except Exception:
        return np.zeros(25, dtype=complex)


def poly_807(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
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
    except Exception:
        return np.zeros(25, dtype=complex)



def poly_808(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        for k in range(25):
            cf[k] = (k + t1) / (k + t2)
        cf[4] += np.log(np.abs(t1 + t2))
        cf[9] += np.sin(np.real(t1)) + np.cos(np.imag(t2))
        cf[14] += np.abs(cf[13]) ** 2 + np.angle(cf[12]) ** 2
        cf[19] += np.abs(np.real(t2) * np.imag(t1))
        cf[24] += np.abs(t1 + np.conj(t2))
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(25, dtype=complex)

def poly_809(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
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
    except Exception:
        return np.zeros(25, dtype=complex)

def poly_810(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        for i in range(25):
            if i % 2 == 1:
                cf[i] = ((i * t1 + 3 * i * t2) / (i + 1) ** 2) ** i
            else:
                cf[i] = (t1 + np.conj(t2)) ** i
        cf[cf == np.inf] = 1e10
        cf[cf == -np.inf] = -1e10
        cf[np.isnan(cf)] = 0
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(25, dtype=complex)

def poly_811(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        cf[0] = t1 * t2
        cf[1] = (t1 + t2) * np.conj(t1 - t2)
        for k in range(2, 26):
            cf[k] = np.abs(t1) * np.abs(t2) * np.sin(np.angle(t1 + 1j * t2) ** k) + np.log(np.abs(t1 ** k / (1 + t2)))
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(25, dtype=complex)

def poly_812(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        cf[0] = t1 * t2
        for k in range(1, 26):
            cf[k] = (cf[k - 1] ** 2 + np.real(t1) + np.imag(1j * t2)) / (1 + np.abs(cf[k - 1]))
            if np.abs(cf[k]) > 1e6 or np.isnan(cf[k]) or np.isinf(cf[k]):
                cf[k] = cf[k - 1]
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(25, dtype=complex)

def poly_813(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        cf[0] = t1 * t2
        for k in range(1, 26):
            v = (np.sin(k * cf[k - 1]) + np.cos(k * cf[k - 1])) * np.real(t1 + t2)
            if np.abs(v) != 0:
                cf[k] = np.log(np.abs(v)) + np.conj(t1 * t2)
            else:
                cf[k] = 0
        cf[24] = np.sum(cf[:24]) + np.abs(t1 - t2)
        return cf.astype(np.complex128)
    except Exception:
        return np.zeros(25, dtype=complex)

def poly_814(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
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
    except Exception:
        return np.zeros(25, dtype=complex)

def poly_815(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        for k in range(1, 26):
            cf[k-1] = np.sin(k)*t1/(1+abs(t2)) + np.cos(k)*t2/(1+abs(t1)) + np.sqrt(k)
        
        cf[0] = np.abs(t1)*abs(t2)
        cf[4] = np.angle(t1)*abs(t2)
        cf[9] = np.abs(t1)*np.angle(t2)
        cf[14] = np.abs(t1)*t2.real
        cf[19] = np.abs(t1)*t2.imag
        cf[24] = t1.real*abs(t2)
        return cf.astype(np.complex128)
    except:
        return np.zeros(25, dtype=complex)

def poly_816(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        cf[0] = t1*t2
        cf[1] = np.abs(t1)**2 + np.abs(t2)**2
        for k in range(2, 24):
            cf[k] = (cf[k-1] / cf[k-2]) + np.conj(t1) - t2
            if np.isnan(cf[k]) or np.isinf(cf[k]):
                cf[k] = 0
        cf[24] = cf[23] + cf[22] - np.conj(t1 + t2)
        return cf.astype(np.complex128)
    except:
        return np.zeros(25, dtype=complex)

def poly_817(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        cf[0] = (t1**3).real*(t2**3).real
        cf[1] = (t1**3).imag*(t2**3).imag
        for k in range(2, 25):
            if (k+1)%3 == 0:
                cf[k] = (t1+1j*t2)**((k+1)/3) / (k+1)
            else:
                cf[k] = np.conj(cf[k-1])**2 + np.abs(t1)*abs(t2)
        return cf.astype(np.complex128)
    except:
        return np.zeros(25, dtype=complex)

def poly_818(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        for k in range(1, 26):
            cf[k-1] = (np.abs(t1 + 1j*t2 + 0.5 + k/25)*np.cos(np.angle(t1 + 1j*t2)**(k-1)) + 
                       1j*abs(t2 + 1j*t1 + 0.5 + k/25)*np.sin(np.angle(t2 + 1j*t1)**(k-1)))
            if np.isnan(cf[k-1]) or np.isinf(cf[k-1]):
                cf[k-1] = 0
        return cf.astype(np.complex128)
    except:
        return np.zeros(25, dtype=complex)

def poly_819(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
        cf[0] = t1*t2
        for i in range(1, 25):
            cf[i] = (cf[i-1].real**2 - cf[i-1].imag**2)*t1*t2 + 1j*2*cf[i-1].real*cf[i-1].imag
            if np.isnan(cf[i]):
                cf[i] = 1.0
            if np.abs(cf[i]) < 1e-10:
                cf[i] = 1
        cf[24] = cf[23] + t1 + t2
        return cf.astype(np.complex128)
    except:
        return np.zeros(25, dtype=complex)

def poly_820(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
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
    except:
        return np.zeros(25, dtype=complex)

def poly_821(t1, t2):
    try:
        cf = np.zeros(25, dtype=complex)
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
        return cf.astype(np.complex128)
    except:
        return np.zeros(25, dtype=complex)

#convert every single one of these functions into python. t1, t2 are complex numbers. wrap the translation into try:....except: ..
#if there is an error return return np.zeros(25, dtype=complex).
#use numpy for all math.
#
def polymoth(t1, t2):
    try:
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
    except:
        return np.zeros(25, dtype=complex)

def polymoth1(t1, t2):
    try:
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
    except:
      return np.zeros(25, dtype=complex)

def giga_moth4(t1, t2):
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

#
def giga10(t1, t2):
    try:
      n = 120
      cf = [0+0j]*n
      re1 = t1.real
      im1 = t1.imag
      re2 = t2.real
      im2 = t2.imag
  
      for k in range(1, n+1):
          k_idx = k-1  # python index
          term1_mag = 100*(re1 + im2)*((k/10.0)**2)
          term1_ang = (re2 * k)/20.0
          term1 = term1_mag * cmath.exp(1j*term1_ang)
          
          sinval = cmath.sin(k * 0.1 * im2)
          term2_mag = 50*(im1 - re2)*sinval
          term2_ang = -1.0 * k * 0.05 * re1
          term2 = term2_mag * cmath.exp(1j*term2_ang)
          
          cf[k_idx] = term1 + term2
  
      cf[29] += 1000j
      cf[59] -= 500
      cf[89] += 250 * cmath.exp(1j*(t1*t2))
  
      return cf.astype(np.complex128)
    except:
        return np.zeros(25, dtype=complex)
  
#  
def old_379(t1, t2):
    cf = np.zeros(35, dtype=complex)
    for j in range(1, 36):
        magnitude = np.log(np.abs(t1) + np.abs(t2) + j) * (
            (np.abs(t1) ** np.sin(j)) + (np.abs(t2) ** cmath.cos(j))
        )
        angle = (
            np.angle(t1) * j
            - np.angle(t2) * (35 - j)
            + np.sin(j) * np.cos(j)
        )
        cf[j - 1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
    for k in range(1, 36):
        cf[k - 1] += t1.conjugate() * (t2 ** k) / (k + 1)
    special_indices = [4, 9, 14, 19, 24, 29]
    for i in special_indices:
        cf[i] += 50 * (t1.real - t2.imag) * 1j
    return cf.astype(np.complex128)
  
#
def old1(t1, t2):
    n = 35
    cf = np.zeros(n, dtype=complex)
    rec = np.linspace(t1.real, t2.real, n)
    imc = np.linspace(t1.imag, t2.imag, n)
    for j in range(n):
        jj = j + 1  # R's j (1-based)
        mag = (np.abs(t1)**jj) / (jj + 1) \
              + (np.abs(t2)**(n - jj)) * cmath.sin(jj) \
              + cmath.log(np.abs(rec[j] + 1j*imc[j]) + 1)
        angle = cmath.phase(t1) * cmath.cos(jj) \
                + cmath.phase(t2) * cmath.sin(jj) \
                + cmath.sin(jj * math.pi / n)
        cf[j] = mag * (cmath.cos(angle) + 1j * cmath.sin(angle))
    for k in range(n):
        kk = k + 1  # R's k (1-based)
        cf[k] = ( cf[k]
                  + np.conjugate(cf[(k + 1) % n]) * cmath.cos(kk)
                  - cf[k].real * cmath.sin(kk) )
    for r in range(n):
        rr = r + 1  # R's r (1-based)
        if rr % 3 == 0:
            cf[r] = cf[r] * (1 + 0.5 * cmath.sin(rr))
        else:
            cf[r] = cf[r] + 0.3 * cf[r].imag * cmath.cos(rr)
    return cf.astype(np.complex128)
  
#
def poly48(t1, t2):
    try:
        cf = np.zeros(90, dtype=complex)
        cf[0] = t1 - t2
        for k in range(1, len(cf)):
            v = np.sin(k * cf[k-1]) + np.cos(k * t1)
            av = np.abs(v)
            if np.isfinite(av) and av > 1e-10:
                cf[k] = 1j * v / av
            else:
                cf[k] = t1 + t2
        return cf.astype(np.complex128)
    except:
        return np.zeros(25, dtype=complex)
 
#     
def poly51(t1, t2):
    try:
      cf = [0j] * 50  # Initialize complex array of size 50
      cf[0] = t1 + t2
      for k in range(1, len(cf)):
          v = np.sin(((k+3) % 10) * cf[k-1]) + np.cos(((k+1) % 10) * t1)
          av = np.abs(v)
          if np.isfinite(av) and av > 1e-10:
              cf[k] = v / av
          else:
              cf[k] = t1 + t2
      return cf.astype(np.complex128)
    except:
      return np.zeros(25, dtype=complex)
    
#
def poly70(t1, t2):
  try:
    cf = np.zeros(50, dtype=complex)
    cf[[0, 9, 19, 29, 39, 49]] = [1, 2, -3, 4, -5, 6]
    cf[14] = 100 * (t1**2 + t2**2)
    cf[24] = 50 * (cmath.sin(t1) + 1j * cmath.cos(t2))
    cf[34] = 200 * (t1 * t2) + 1j * (t1**3 - t2**3)
    cf[44] = cmath.exp(1j * (t1 + t2)) + cmath.exp(-1j * (t1 - t2))
    return cf.astype(np.complex128)
  except:
    return np.zeros(25, dtype=complex)
 
# 
def poly72(t1, t2):
    try:
      cf = np.zeros(35, dtype=complex)
      cf[[0, 6, 14, 19, 26, 34]] = [1, -2, 3, -4, 5, -6]
      cf[11] = 50j * np.sin(t1**2 - t2**2)
      cf[17] = 100 * (np.cos(t1) + 1j * np.sin(t2))
      cf[24] = 50 * (t1**3 - t2**3 + 1j * t1 * t2)
      cf[29] = 200 * np.exp(1j * t1) + 50 * np.exp(-1j * t2)
      return cf.astype(np.complex128)
    except:
      return np.zeros(25, dtype=complex)

#
def poly76(t1, t2):
    try:
      cf = np.zeros(30, dtype=complex)
      cf[[0, 5, 11, 19]] = [1, 3, -2, 5]
      cf[9] = 100 * t1**3 + 50 * t2**2
      cf[14] = 50j * (t1.real - t2.imag)
      cf[24] = 200 * t1 * (t2 + 1) - 100j * t2
      cf[29] = np.exp(1j * t1) + t2**3
      return cf.astype(np.complex128)
    except:
      return np.zeros(25, dtype=complex)

#
def poly77(t1, t2):
    try:
      cf = np.zeros(40, dtype=complex)
      cf[0] = 1
      cf[4] = 50 * cmath.exp(t1)
      cf[9] = 100 * (t2**2 - 1j * t1)
      cf[19] = 200 * cmath.exp(1j * (t1**2)) - 50 * cmath.exp(-1j * (t2**3))
      cf[29] = 100 * t1 * (t2**2) + 50j * (t1**3)
      cf[39] = cmath.exp(1j * (t1 + t2)) - 50 * cmath.sin((t1 - t2).imag)
      return cf.astype(np.complex128) 
    except:
      return np.zeros(25, dtype=complex)
 
#
def poly78(t1, t2):
    try:
      cf = [0j] * 40
      cf[0], cf[7], cf[15], cf[23], cf[31] = 1, -3, 5, -7, 2
      cf[4] = 50 * (t1**2 - t2**3)
      cf[11] = 100j * (t1**3 + t2)
      cf[19] = np.exp(1j * t1) + np.exp(-1j * t2**2)
      cf[29] = 200 * np.sin(t1.real + t2.imag) - 50 * np.cos((t1 - t2).imag)
      cf[34] = np.exp(1j * t1**3) + t2**2
      return cf.astype(np.complex128)
    except:
      return np.zeros(25, dtype=complex)
    
#
def poly103(t1, t2):
    try:
      cf = np.zeros(25, dtype=complex)
      for i in range(1, 26):
          real_part = (t1.real ** i)
          imag_part = (t2.imag ** (25 - i))
          denominator = (1 + np.abs(t1 + t2))
          phase_factor = cmath.exp(1j * cmath.phase(t1 + t2))
          cf[i - 1] = ((real_part + imag_part) / denominator) * phase_factor
      cf[2] = 3 * np.conjugate(t1**2 + t2)
      cf[6] = 7 * np.abs(t1 + t2)
      cf[10] = 11 * (t1 / t2 + np.conjugate(t2 / t1))
      cf[16] = 17 * (np.abs(t1) * np.abs(t2)) / (np.abs(t1 + t2) ** 2)
      cf[22] = 23 * (np.conjugate(t1) + t2) / (1 + np.abs(t1 * np.conjugate(t2)))
      cf[24] = 25 * (np.conjugate(t1) + np.conjugate(t2)) / np.abs(t1 * t2)
      return cf.astype(np.complex128)
    except:
      return np.zeros(25, dtype=complex)
    
#    
def poly279(t1, t2):
    try:
      cf = [0j] * 71
      cf[0] = t1**5 + 2*t2
      cf[1] = t1**4 - 3*t2**2 + t1.conjugate()
      cf[2] = 4*t1*t2 + 5*complex(np.sin(t1))
      cf[3] = 6*complex(np.cos(t2)) - 7*t1**2
      cf[4] = 8*t2**3 + 9*t1*t2
      cf[5] = 10*complex(np.sin(t1 + t2)) - 11*t2
      cf[6] = 12*complex(np.cos(t1)) + 13*t1*t2
      cf[7] = 14*t1**3 - 15*t2**2
      cf[8] = 16*t1*t2**2 + 17*complex(np.sin(t2))
      cf[9] = 18*complex(np.cos(t1 + t2)) - 19*t1
      cf[10] = 20*t2**4 + 21*t1**2*t2
      cf[11] = 22*complex(np.sin(t1)) - 23*t2**3
      cf[12] = 24*complex(np.cos(t2)) + 25*t1*t2
      cf[13] = 26*t1**4 - 27*t2**2*t1
      cf[14] = 28*complex(np.sin(t1 + t2)) + 29*t2
      cf[15] = 30*complex(np.cos(t1)) - 31*t1*t2**2
      cf[16] = 32*t2**5 + 33*t1**3
      cf[17] = 34*complex(np.sin(t2)) - 35*t1*t2
      cf[18] = 36*complex(np.cos(t1 + t2)) + 37*t2**3
      cf[19] = 38*t1**5 - 39*t2**2
      cf[20] = 40*complex(np.sin(t1)) + 41*t1**2*t2
      cf[21] = 42*complex(np.cos(t2)) - 43*t2**4
      cf[22] = 44*t1**3*t2 + 45*complex(np.sin(t1 + t2))
      cf[23] = 46*complex(np.cos(t1)) - 47*t2**3*t1
      cf[24] = 48*t2**6 + 49*t1**4
      cf[25] = 50*complex(np.sin(t2)) + 51*t1*t2**2
      cf[26] = 52*complex(np.cos(t1 + t2)) - 53*t2**4
      cf[27] = 54*t1**6 - 55*t2**3
      cf[28] = 56*complex(np.sin(t1)) + 57*t1**3*t2
      cf[29] = 58*complex(np.cos(t2)) - 59*t2**5
      cf[30] = 60*t1**4*t2 + 61*complex(np.sin(t1 + t2))
      cf[31] = 62*complex(np.cos(t1)) + 63*t2**4*t1
      cf[32] = 64*t2**7 + 65*t1**5
      cf[33] = 66*complex(np.sin(t2)) - 67*t1*t2**3
      cf[34] = 68*complex(np.cos(t1 + t2)) + 69*t2**5
      cf[35] = 70*t1**7 - 71*t2**4
      cf[36] = 72*complex(np.sin(t1)) + 73*t1**4*t2
      cf[37] = 74*complex(np.cos(t2)) - 75*t2**6
      cf[38] = 76*t1**5*t2 + 77*complex(np.sin(t1 + t2))
      cf[39] = 78*complex(np.cos(t1)) - 79*t2**5*t1
      cf[40] = 80*t2**8 + 81*t1**6
      cf[41] = 82*complex(np.sin(t2)) + 83*t1*t2**4
      cf[42] = 84*complex(np.cos(t1 + t2)) - 85*t2**6
      cf[43] = 86*t1**8 - 87*t2**5
      cf[44] = 88*complex(np.sin(t1)) + 89*t1**5*t2
      cf[45] = 90*complex(np.cos(t2)) - 91*t2**7
      cf[46] = 92*t1**6*t2 + 93*complex(np.sin(t1 + t2))
      cf[47] = 94*complex(np.cos(t1)) + 95*t2**6*t1
      cf[48] = 96*t2**9 + 97*t1**7
      cf[49] = 98*complex(np.sin(t2)) - 99*t1*t2**5
      cf[50] = 100*complex(np.cos(t1 + t2)) + 101*t2**7
      cf[51] = 102*t1**9 - 103*t2**6
      cf[52] = 104*complex(np.sin(t1)) + 105*t1**6*t2
      cf[53] = 106*complex(np.cos(t2)) - 107*t2**8
      cf[54] = 108*t1**7*t2 + 109*complex(np.sin(t1 + t2))
      cf[55] = 110*complex(np.cos(t1)) + 111*t2**7*t1
      cf[56] = 112*t2**10 + 113*t1**8
      cf[57] = 114*complex(np.sin(t2)) - 115*t1*t2**6
      cf[58] = 116*complex(np.cos(t1 + t2)) + 117*t2**8
      cf[59] = 118*t1**10 - 119*t2**7
      cf[60] = 120*complex(np.sin(t1)) + 121*t1**7*t2
      cf[61] = 122*complex(np.cos(t2)) - 123*t2**9
      cf[62] = 124*t1**8*t2 + 125*complex(np.sin(t1 + t2))
      cf[63] = 126*complex(np.cos(t1)) + 127*t2**8*t1
      cf[64] = 128*t2**11 + 129*t1**9
      cf[65] = 130*complex(np.sin(t2)) - 131*t1*t2**7
      cf[66] = 132*complex(np.cos(t1 + t2)) + 133*t2**9
      cf[67] = 134*t1**11 - 135*t2**8
      cf[68] = 136*complex(np.sin(t1)) + 137*t1**8*t2
      cf[69] = 138*complex(np.cos(t2)) - 139*t2**10
      cf[70] = 140*t1**9*t2 + 141*complex(np.sin(t1 + t2)) / 200
      return cf.astype(np.complex128)
    except:
      return np.zeros(25, dtype=complex)

#
def poly378(t1, t2):
    try:
        cf = np.zeros(35, dtype=complex)
        indices = [0, 4, 12, 19, 20, 24] 
        values = [1 + 0j, 4 + 0j, 4 + 0j, -9 + 0j, -1.9 + 0j, 0.2 + 0j]
        cf[indices] = values
        exclude_indices = set([4, 12, 19, 20, 24]) 
        for j in range(1, 34): 
            if j not in exclude_indices:
                mag = np.log(np.abs(t1 + (j+1)) + 1) * np.sin((j+1) * np.angle(t2)) + np.cos((j+1) * np.angle(t1))
                angle = np.angle(t1)**(j+1) + np.sin((j+1) * np.angle(t2)) - np.cos(j+1)
                cf[j] = mag * np.cos(angle) + mag * np.sin(angle) * 1j
        cf[34] = np.conj(t1) * np.conj(t2) + np.sin(np.abs(t1) * np.abs(t2)) + np.log(np.abs(t1) + np.abs(t2) + 1) * 1j
        return cf.astype(np.complex128)
    except:
        return np.zeros(25, dtype=complex)
      
#
def poly383(t1, t2):
  try:
      cf = np.zeros(35, dtype=complex)
      fixed_indices = [1, 5, 9, 13, 17, 21, 25, 29, 33]
      fixed_values = [
          2 + 3j,   # 2 + 3i in R
          -3 + 2j,
          4 - 1j,
          -5 + 4j,
          6 - 3j,
          -7 + 5j,
          8 - 4j,
          -9 + 6j,
          10 - 5j
      ]
      cf[fixed_indices] = fixed_values
      j_indices = [1, 5, 9, 13, 17, 21, 25, 29, 33]  # R indices
      j_indices_py = [x - 1 for x in j_indices]
      for j_py in j_indices_py:
          value = (
              (t1.real * t2.imag + t1.imag * t2.real)
              + (np.abs(t1)**2 - np.abs(t2)**2) * cmath.sin(t1 + t2)
              + math.log(np.abs(t1) + 1) * cmath.cos(t2)
          )
          cf[j_py] = value
      k_indices = [3, 7, 11, 15, 19, 23, 27, 31, 35]  # R
      k_indices_py = [x - 1 for x in k_indices]
      for k_py in k_indices_py:
          value = (
              cmath.sin(t1 * t2)
              + cmath.cos(t1 / (np.abs(t2) + 1)) * np.conjugate(t2)
              + cmath.phase(t1 + t2) * np.abs(t1 - t2)
              + (t1.real * t2.imag)
          )
          cf[k_py] = value
      r_indices = [4, 8, 12, 16, 20, 24, 28, 32]  # R
      r_indices_py = [x - 1 for x in r_indices]
      for r_py in r_indices_py:
          value = (
              (t1.real**3)
              - (t2.imag**3)
              + (t1 * t2).real
              + (t1 + t2).imag
              + math.log(np.abs(t1 * t2) + 1)
          )
          cf[r_py] = value
      cf[18] = 100j * (t1**3) + 50j * (t2**2) - 75 * t1 * t2 + 25
      cf[22] = (
          80j * (t2**3)
          - 60j * (t1**2)
          + 40 * cmath.sin(t1 + t2)
          - 20
      )
      cf[26] = (
          90j * (t1 * (t2**2))
          - 70 * cmath.cos(t1)
          + 50 * math.log(np.abs(t2) + 1)
      )
      cf[30] = (
          110j * cmath.sin(t1**2)
          - 95 * np.abs(t2) * t1
          + 85j * cmath.phase(t1 + t2)
      )
      cf[34] = (
          120j * cmath.cos(t1 * t2)
          - 100 * cmath.sin(t2)
          + 75 * math.log(np.abs(t1) + 1)
      )
      return cf.astype(np.complex128)
  except:
      return np.zeros(25, dtype=complex)
    
#  
def poly420(t1, t2):
    try:
      n = 35
      cf = np.zeros(n, dtype=complex)
      special_indices_r = [1, 7, 14, 21, 28, 35]
      special_indices_py = [x - 1 for x in special_indices_r]
      special_values = [2.5, -4.2, 3.8, -16.5, 5.3, 0.6]
      for idx_py, val in zip(special_indices_py, special_values):
          cf[idx_py] = val
      for j in range(2, 35):
          j_py = j - 1
          if j % 4 == 0:
              k = j // 2  # or float(j) / 2
              part1 = 150j * (t1 ** k) + 75 * np.conjugate(t2)
              part2 = cmath.sin(k * cmath.phase(t1))
              cf[j_py] = (part1 * part2) - 50 * math.log(np.abs(t2) + 1)
          elif j % 3 == 0:
              k = j % 5
              re_part = 200 * ((t1 * (t2 ** k)).real)
              im_part = 100j * ((t1 - t2).imag)
              angle_part = math.cos(k * cmath.phase(t2))
              cf[j_py] = (re_part + im_part) * angle_part
          else:
              r = j % 7
              term1 = (np.conjugate(t1) ** r) * (t2 ** j)
              term2 = (np.abs(t1 ** j) * np.abs((t2 ** r)))
              cf[j_py] = term1 + term2
      cf[9] = (180j * (t1 ** 3)
               - 120 * (t2 ** 2)
               + 90 * cmath.sin(t1) * cmath.cos(t2))
      cf[19] = (220j * (t2 ** 4)
                + 130 * (t1 ** 3).real
                - 100 * t2.imag)
      cf[29] = (260j * (t1 ** 2) * t2
                + 160 * math.log(np.abs(t1 * t2) + 1)
                - 110 * np.conjugate(t1))
      return cf.astype(np.complex128)
    except:
      return np.zeros(25, dtype=complex)
 
#
def poly476(t1, t2):
    try:
      n = ps.poly.get("n") or 35
      cf = [0j] * n
      for k in range(1, n + 1):
          real_part = t1.real * np.cos(k * np.pi / 5) + t2.real * np.sin(k * np.pi / 7)
          imag_part = t1.imag * np.sin(k * np.pi / 6) - t2.imag * np.cos(k * np.pi / 8)
          magnitude = np.sqrt(real_part**2 + imag_part**2) * np.log(np.abs(k) + 1) * (1 + np.sin(k))
          angle = np.arctan2(imag_part, real_part) + np.sin(k * np.angle(t1)) * np.cos(k * np.angle(t2))
          cf[k-1] = magnitude * np.exp(1j * angle)
      for r in range(1, n + 1):
          cf[r-1] = cf[r-1] + np.conj(cf[n - r]) * np.sin(r * np.pi / 10)
      return cf.astype(np.complex128)
    except:
      return np.zeros(25, dtype=complex)
    
#
def poly479(t1, t2):
  try:
    n = ps.poly.get("n") or 35
    cf = np.zeros( n, dtype=complex )
    for j in range( 1, n + 1 ):
      mag = ( np.abs( t1 ) * cmath.log( j + 1 ) + np.abs( t2 ) * j**0.5 ) / ( 1 + j**1.3 )
      angle = cmath.phase( t1 ) * cmath.sin( j ) + cmath.phase( t2 ) * cmath.cos( j / 2 ) + cmath.sin( j / 3 * cmath.pi )
      perturb = cmath.exp( 1j * ( cmath.sin( j / 4 * pi ) + cmath.cos( j / 5 * pi ) ) )
      cf[j-1] = mag * cmath.exp( 1j * angle ) * perturb
    return cf.astype(np.complex128)
  except:
    return np.zeros(25, dtype=complex)

#
def poly484(t1, t2):
    try:
      n = ps.poly.get("n") or 35
      cf = np.zeros(n, dtype=complex)
      for j in range(1, n + 1):
          rec_part = t1.real * cmath.sin(j / 2.0) + t2.real * cmath.cos(j / 3.0)
          imc_part = t1.imag * cmath.cos(j / 4.0) - t2.imag * cmath.sin(j / 5.0)
          magnitude = (cmath.log(np.abs(rec_part + imc_part) + 1) *
                       (j ** 1.2) *
                       (1 + cmath.sin(j * math.pi / 6.0)))
          angle = (cmath.phase(t1) * cmath.cos(j / 7.0) +
                   cmath.phase(t2) * cmath.sin(j / 8.0))
          cf[j - 1] = magnitude * (cmath.cos(angle) + 1j * cmath.sin(angle))
      return cf.astype(np.complex128)
    except:
      return np.zeros(25, dtype=complex)
    
#
def poly485(t1, t2):
    try:
      n = ps.poly.get("n") or 35
      cf = np.zeros(n, dtype=complex)
      for j in range(1, n + 1):
          mag = 0
          angle = 0
          for k in range(1, (j // 5) + 2):
              mag += np.real(t1) * np.sin(j * k) * np.log(k + 1)
              angle += np.imag(t2) * np.cos(j + k) / (k + 1)
          for r in range(1, 4):
              mag *= (1 + np.real(t1) * 0.1 * r)
              angle += np.angle(t2) * 0.05 * r
          cf[j - 1] = mag * (np.cos(angle) + 1j * np.sin(angle))
      cf *= np.exp(1j * np.sin(np.abs(t1)) * np.arange(1, n + 1))
      return cf.astype(np.complex128)
    except:
      return np.zeros(25, dtype=complex)
    
#
def poly518(t1, t2):
    try:
      n = ps.poly.get("n") or 35
      cf = [0+0j] * n     
      rec = [t1.real + (t2.real - t1.real) * i/(n-1) for i in range(n)]
      imc = [t1.imag + (t2.imag - t1.imag) * i/(n-1) for i in range(n)]
      
      for j in range(n):
          r = rec[j]
          m = imc[j]
          mag = np.log(np.abs(r**2 + m**2) + 1) * (j + 1)**(np.sin(r) + np.cos(m))
          angle = np.sin(j * r) + np.cos(j * m) + np.angle(t1) * np.sin(m) - np.angle(t2) * np.cos(r)
          cf[j] = mag * np.exp(1j * angle)
      
      return cf.astype(np.complex128)
    except:
        return np.zeros(25, dtype=complex)
      
#    
def poly524(t1, t2):
    try:
      n = ps.poly.get("n") or 35
      cf = np.zeros(n, dtype=complex)
      for j in range(1, n + 1):
          real_part = t1.real * math.sin(j) + t2.real * math.cos(j / 2.0)
          imag_part = t1.imag * math.cos(j) - t2.imag * math.sin(j / 2.0)
          part_magnitude = math.sqrt(real_part**2 + imag_part**2)
          log_part = math.log(j + np.abs(t1) + np.abs(t2))
          magnitude = part_magnitude * log_part
          angle = cmath.phase(t1) * math.sqrt(j) + cmath.phase(t2) * math.cos(j)
          cf[j - 1] = magnitude * (math.cos(angle) + 1j * math.sin(angle))
      return cf.astype(np.complex128)
    except:
      return np.zeros(25, dtype=complex)
    
#  
def poly526(t1, t2):
    try:
      n = ps.poly.get("n") or 35
      cf = np.zeros(n, dtype=complex)
      for j in range(1, n + 1):
          r_part = t1.real * j + t2.real / (j + 1)
          i_part = t1.imag * math.sin(j) + t2.imag * cmath.cos(j)
          mag = cmath.log(np.abs(t1) + j) * ((j % 5) + 1)
          angle = cmath.phase(t1) * cmath.sin(j / 3.0) + cmath.phase(t2) * cmath.cos(j / 4.0)
          cf[j - 1] = (r_part + 1j * i_part) * cmath.exp(1j * angle) * mag
      return cf.astype(np.complex128)
    except:
      return np.zeros(25, dtype=complex)

#  
def poly537(t1, t2):
    try:
      n = ps.poly.get("n") or 35
      cf = np.zeros(35, dtype=complex)
      for j in range(1, 36):  # Mimic R's 1..35
          k = (j % 7) + 1
          r = math.floor(j / 5) + 1
          term1 = (t1.real ** k) * cmath.sin(k * math.pi / 3)
          term2 = (t2.imag ** r) * cmath.cos(r * math.pi / 4)
          magnitude = term1 + term2 + cmath.log(np.abs(t1) + np.abs(t2) + j)
          angle = cmath.phase(t1) * r - cmath.phase(t2) / k + cmath.sin(j) * cmath.cos(j)
          cf[j - 1] = (magnitude * cmath.exp(1j * angle) + np.conjugate(t1 * t2) * ((np.abs(t1) + np.abs(t2)) / j))
          if j % 4 == 0:
              cf[j - 1] *= (cmath.sin(j * math.pi / 5) + cmath.cos(j * math.pi / 6))
          if j % 6 == 0:
              cf[j - 1] += (t1.real ** 2) - (t2.imag ** 2)
      return cf.astype(np.complex128)
    except:
      return np.zeros(25, dtype=complex)
  
#
def poly555(t1, t2):
    try:
      n = ps.poly.get("n") or 35
      cf = [0j] * n
      r1 = t1.real
      i1 = t1.imag
      r2 = t2.real
      i2 = t2.imag
      for j in range(1, n+1):
          mag = np.log(np.abs(r1 + j) + 1) * (j**1.5 + np.sin(j * r2)) * (1 + np.abs(np.cos(j * i1)))
          ang = np.angle(t1) * np.sin(j * r2) + np.angle(t2) * np.cos(j * i1) + np.sin(j * i2)
          cf[j-1] = mag * (np.cos(ang) + 1j * np.sin(ang))
      return cf.astype(np.complex128)
    except:
        return np.zeros(25, dtype=complex)

# 
def poly562(t1, t2):
    try:
      n = ps.poly.get("n") or 35
      cf = [0j] * n
      for j in range(1, n+1):
          mag_part1 = t1.real * j**2
          mag_part2 = np.log(np.abs(t2) + j) * np.sin(j * np.angle(t1))
          mag_part3 = np.cos(j * t2.real) * np.sqrt(j)
          magnitude = mag_part1 + mag_part2 + mag_part3
          
          angle_part1 = np.angle(t1) + np.sin(j * t1.real)
          angle_part2 = np.cos(j * t2.imag) - np.angle(t2) / j
          angle = angle_part1 + angle_part2
          
          cf[j-1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
      return cf.astype(np.complex128)
    except:
      return np.zeros(25, dtype=complex)
    
# 
def poly563(t1, t2):
    try:
      n = ps.poly.get("n") or 35
      cf = np.zeros(n, dtype=complex)
      rec_seq = np.linspace(t1.real, t2.real, n)
      im_seq = np.linspace(t1.imag, t2.imag, n)
      for j in range(1, n + 1):
          angle_part = (cmath.sin(j * math.pi / 6.0) *
                        cmath.cos(j * math.pi / 8.0) +
                        cmath.phase(t1) * cmath.log(j + 1))
          
          magnitude_part = (cmath.log(np.abs(t1) + j**2) * np.abs(cmath.cos(j)) +
                            cmath.log(np.abs(t2) + j) * np.abs(cmath.sin(j / 2.0)))
          
          cf[j - 1] = ((magnitude_part + t1.real * t2.real / (j + 1)) *
                       cmath.exp(1j * angle_part))
          
          if j % 5 == 0:
              cf[j - 1] = cf[j - 1] + np.conjugate(cf[j - 1])
          cf[j - 1] *= (1 + 0.1 * cmath.sin(j))
      return cf.astype(np.complex128)
    except:
      return np.zeros(25, dtype=complex)
    
#
def poly583(t1, t2):
    try:
      n = ps.poly.get("n") or 35
      cf = np.zeros(n, dtype=complex)
      for j in range(1, n + 1):
          k = (j * 5 + 2) % 12 + 1
          r_part = t1.real * np.sin(j * np.pi / k) + t2.real * np.cos(j * np.pi / (k + 1))
          i_part = t1.imag * np.cos(j * np.pi / k) - t2.imag * np.sin(j * np.pi / (k + 1))
          magnitude = np.log(np.abs(t1) + j) * np.abs(np.sin(j * np.pi / 10))
          angle = np.angle(t1) * np.cos(j * np.pi / 8) + np.angle(t2) * np.sin(j * np.pi / 9)
          cf[j - 1] = magnitude * (r_part + 1j * i_part) * (np.cos(angle) + 1j * np.sin(angle))
      return cf.astype(np.complex128)
    except:
      return np.zeros(25, dtype=complex)

#
def poly598(t1, t2):
    try:
      n = ps.poly.get("n") or 35
      cf = [0j] * n
      rec = [x.real for x in np.linspace(t1.real, t2.real, n)]
      imc = [x.real for x in np.linspace(t1.imag, t2.imag, n)]
      for j in range(n):
          phase = (cmath.phase(t1) * (j + 1) + 
                  cmath.phase(t2) / (j + 2) + 
                  math.sin((j + 1) * rec[j]) - 
                  math.cos((j + 1) * imc[j]))
          magnitude = (math.log(np.abs(t1) + np.abs(t2) + j + 1) * 
                      ((j + 1)**2 + math.sin(j + 1) * math.cos(j + 1)))
          cf[j] = magnitude * complex(math.cos(phase), math.sin(phase))
      return cf.astype(np.complex128)
    except:
      return np.zeros(25, dtype=complex)

#    
def poly605(t1, t2):
    try:
      n = ps.poly.get("n") or 35
      cf = [0j] * n  # Initialize array of complex numbers
      for j in range(n):
          mag = 0
          angle = 0
          for k in range(1, j+2):  # j+2 because Python range is exclusive
              mag += np.abs(t1 + k).real * np.sin(k * t2.real)
              angle += (t2**k).imag * np.cos(k / (j+1))
          cf[j] = mag * complex(np.cos(angle), np.sin(angle))
      for j in range(n):
          cf[j] = cf[j] * (1 + 0.05 * (j+1)**2) + cf[j].conjugate() * 0.02
      return cf.astype(np.complex128)
    except:
        return np.zeros(25, dtype=complex)
      
#  
def poly621(t1, t2):
    try:
      n = ps.poly.get("n") or 35
      cf = np.zeros(n, dtype=complex)
      rec1 = t1.real
      imc1 = t1.imag  # Defined in R code but not actually used in part1..part4
      rec2 = t2.real
      imc2 = t2.imag
      for j in range(1, n + 1):
          part1 = (rec1 ** j) * cmath.sin(j * cmath.phase(t2))
          part2 = (imc2 ** (n - j)) * cmath.cos(j * np.abs(t1))
          part3 = cmath.log(np.abs(t1) + np.abs(t2) + j)
          part4 = ((rec1 + j) * (imc2 + j) * cmath.log(np.abs(t1) + 1))
          magnitude = part1 * part2 + part3 * part4
          angle = (cmath.phase(t1) * cmath.sin(j)
                   + cmath.phase(t2) * cmath.cos(j)
                   + cmath.log(np.abs(t1) + 1) / j)
          cf[j - 1] = magnitude * cmath.exp(1j * angle)
      return cf.astype(np.complex128)
    except:
      return np.zeros(25, dtype=complex)
 
#
def poly637(t1, t2):
    try:
      n = ps.poly.get("n") or 35
      cf = [0j] * n
      for j in range(1, n+1):
          mag_part1 = np.log(np.abs(t1) + j**1.5) * np.sin(j * np.pi / 6)
          mag_part2 = np.abs(t2) / (j + 2) + np.cos(j * np.pi / 4)
          magnitude = mag_part1 + mag_part2 * np.exp(-j / 10)
          
          angle_part1 = np.angle(t1) * np.cos(j / 3)
          angle_part2 = np.angle(t2) * np.sin(j / 5) + np.sin(j**2 / 7)
          angle = angle_part1 + angle_part2
          
          cf[j-1] = magnitude * (np.cos(angle) + 1j * np.sin(angle))
      return cf.astype(np.complex128)
    except:
      return np.zeros(25, dtype=complex)
  
#
def poly657(t1, t2):
    try:
        n = ps.poly.get("n") or 35
        cf = np.zeros(n, dtype=complex)
        rec_seq = np.linspace(t1.real, t2.real, n)
        imc_seq = np.linspace(t1.imag, t2.imag, n)
        
        for j in range(n):
            mag = np.log(np.abs(t1) + j + 1) * np.abs(np.sin((j + 1) * np.pi / 7)) + np.sqrt(j + 1) * np.cos((j + 1) * np.angle(t2))
            angle = np.angle(t1) * np.sin(j + 1) + np.angle(t2) * np.cos((j + 1) / 3)
            cf[j] = mag * (np.cos(angle) + 1j * np.sin(angle))
            
        return cf.astype(np.complex128)
    except:
        return np.zeros(25, dtype=complex)


#    
def poly662(t1, t2):
    try:
      n = ps.poly.get("n") or 35
      cf = [0j] * n
      for j in range(1, n+1):
          phase = (np.sin(j * np.angle(t1)) + 
                  np.cos(j * np.angle(t2)) + 
                  np.log(np.abs(t1) + np.abs(t2) + j))
          magnitude = ((j**2 + np.sqrt(j)) * np.abs(np.sin(j / 3)) + 
                  np.exp(-j / 10) * np.abs(t1 + t2))
          cf[j-1] = magnitude * (np.cos(phase) + np.sin(phase) * 1j)
      return cf.astype(np.complex128)
    except:
      return np.zeros(25, dtype=complex)

#     
def poly663(t1, t2):
    try:
        n = ps.poly.get("n") or 40
        cf = np.zeros(n, dtype=complex)
        for j in range(1, n+1):
            mag_part = (np.log(np.abs(t1) + 1) * (j**t2.real) + 
                       sum(range(1, j+1)) * np.sqrt(j))
            angle_part = (np.angle(t1) * np.sin(j) + 
                         np.angle(t2) * np.cos(j) + 
                         np.sin(j * t1.imag) * np.cos(j * t2.imag))
            coeff = mag_part * np.exp(1j * angle_part)
            for k in range(1, 4):
                coeff += ((t1.real**k) * (t2.imag**k) * 
                         np.sin(k * j) / (k + 1))
            cf[j-1] = coeff + np.conjugate(t2) * t1**((j % 5) + 1)
        return cf.astype(np.complex128)
    except:
        return np.zeros(25, dtype=complex)

#   
def poly669(t1, t2):
    try:
      n = ps.poly.get("n") or 35
      cf = [0j] * n
      rec1 = t1.real
      imc1 = t1.imag
      rec2 = t2.real
      imc2 = t2.imag
      for r in range(1, n + 1):
          if r % 3 == 1:
              mag = (np.log(np.abs(t1 + r) + 1) * np.sin(r / n * np.pi) + 
                    np.cos(r * np.pi / 4))
              ang = np.angle(t1) + np.sin(r * np.pi / 6) * np.angle(t2)
          elif r % 3 == 2:
              mag = (np.log(np.abs(t2 + r) + 1) * np.cos(r / n * np.pi) + 
                    np.sin(r * np.pi / 3))
              ang = np.angle(t2) + np.cos(r * np.pi / 5) * np.angle(t1)
          else:
              mag = (np.log(np.abs(t1 * t2 + r) + 1) * np.sin(r / (2 * n) * np.pi) + 
                    np.cos(r * np.pi / 2))
              ang = np.angle(t1 * t2) + np.sin(r * np.pi / 4) * np.cos(r * np.pi / 3)
          cf[r-1] = mag * np.exp(1j * ang)
      for k in range(1, n + 1):
          if k <= n / 3:
              cf[k-1] *= k
          elif k <= 2 * n / 3:
              cf[k-1] *= -k
          else:
              cf[k-1] *= 1 / k
      return cf.astype(np.complex128)
    except:
      return np.zeros(25, dtype=complex)
    
#  
def poly679(t1, t2):
    try:
      n = ps.poly.get("n") or 35
      cf = np.zeros(n, dtype=complex)
      for j in range(1, n + 1):
          mag_part1 = cmath.log( np.abs(t1) + j ) * cmath.sin(0.3 * j * t2.real)
          mag_part2 = cmath.log(np.abs(t2) + j )  * cmath.cos(0.2 * j * t1.imag)
          mag = mag_part1 + mag_part2
          angle_part1 = cmath.phase(t1) + j * 0.1 * math.pi * cmath.sin( j / 5 )
          angle_part2 = cmath.phase(t2) + j * 0.1 * math.pi * cmath.cos( j / 3 )
          angle = angle_part1 + angle_part2
          cf[j - 1] = mag * ( cmath.cos(angle) + 1j * cmath.sin(angle) )
      return cf.astype(np.complex128)
    except:
      return np.zeros(25, dtype=complex)

#
def poly694(t1, t2):
    try:
        n = ps.poly.get("n") or 35
        cf = np.zeros(n, dtype=complex)
        rec = np.linspace(t1.real, t2.real, n)
        imc = np.linspace(t1.imag, t2.imag, n)
        for j in range(n):
            angle = np.sin((j+1) * rec[j]) + np.cos((j+1) * imc[j]) + np.angle(t1 * np.conj(t2))
            magnitude = np.log(np.abs(rec[j]**2 + imc[j]**2) + 1) * ((j+1)**1.5 + np.prod(rec[:(j+1)] + imc[:(j+1)]))
            cf[j] = magnitude * (np.cos(angle) + np.sin(angle) * 1j)
        return cf.astype(np.complex128)
    except:
        return np.zeros(25, dtype=complex)
      
#   
def poly743(t1, t2):
    try:
        n = ps.poly.get("n") or 25
        degree = n
        cf = np.zeros(degree + 1, dtype=complex)
        for j in range(1, degree + 2):
            mag_part1 = np.log(np.abs(t1) + j) * np.sin(2 * np.pi * t1.real / (j + 1))
            mag_part2 = np.log(np.abs(t2) + j) * np.cos(2 * np.pi * t2.imag / (j + 1))
            magnitude = mag_part1 + mag_part2 + np.prod([t1.real, t2.imag, j])
            angle = np.angle(t1) * j + np.angle(t2) * (degree + 1 - j) + np.sin(j) - np.cos(j)
            cf[j-1] = magnitude * np.exp(1j * angle)
        return cf.astype(np.complex128)
    except:
        return np.zeros(25, dtype=complex)
      
#
def poly751(t1, t2):
    try:
      n = ps.poly.get("n") or 25
      cf = [0j] * n
      rec = [x.real for x in np.linspace(t1.real, t2.real, n)]
      imc = [x.real for x in np.linspace(t1.imag, t2.imag, n)]
      
      for j in range(n):
          mag = np.log(np.abs(rec[j] + imc[j]) + 1) * (pow(j+1, 2) + np.sin(j+1))
          ang = np.sin(rec[j] * (j+1)) + np.cos(imc[j] * (j+1))
          cf[j] = mag * np.exp(1j * ang) + np.conj(t1) * pow(t2, j+1)
      
      return cf.astype(np.complex128)
    except:
      return np.zeros(25, dtype=complex)
    
#    
def poly765(t1, t2):
    try:
      n = ps.poly.get("n") or 9
      cf = np.zeros(9, dtype=complex)
      for k in range(1, len(cf)+1):
          mag_part1 = math.log(np.abs(t1) * k + 1)
          mag_part2 = np.abs(math.sin(k * t2.real)) + np.abs(math.cos(k * t1.imag))
          product_val = t1.real * t2.imag * k
          magnitude = mag_part1 * mag_part2 + product_val
          angle = cmath.phase(t1) + cmath.phase(t2) * math.sin(k) + math.cos(k)
          cf[k - 1] = (magnitude * (math.cos(angle) + 1j * math.sin(angle))
                       + np.conjugate(t1) * t2.real)
      return cf.astype(np.complex128)
    except:
      return np.zeros(25, dtype=complex)
    
#  
def poly865(t1, t2):
  try:
    n = ps.poly.get("n") or 9
    cf = np.zeros(n, dtype=complex)
    rec = np.linspace(t1.real, t2.real, n)
    imc = np.linspace(t1.imag, t2.imag, n)
    for k in range(1, n + 1):
        mag = math.log(np.abs(t1) + np.abs(t2) + k) * (k**2)
        angle = (cmath.phase(t1) * math.sin(k)
                 + cmath.phase(t2) * math.cos(k))
        cf[k - 1] = mag * (math.cos(angle) + 1j * math.sin(angle))
    return cf.astype(np.complex128)
  except:
    return np.zeros(25, dtype=complex)
  
#
def poly882(t1, t2):
    try:
      cf = [0j] * 11
      cf[0] = t1**3 + t2**3
      cf[1] = 11 * (t1 + t2)**9
      cf[2] = 1j
      cf[3] = cmath.exp(1j * t1)
      cf[4] = 100 * cmath.sin(t2)
      cf[5] = t1.real - 1j * t2.imag
      cf[6] = 11j * (t2.real / np.abs(t1.imag + 0.1))
      cf[7] = t1 / (np.abs(t1 + t2) + 0.125)
      cf[8] = cmath.exp(1j * t1 * t2)
      cf[9] = np.abs(t1 * t2) * cmath.exp(1j * (cmath.phase(t1) - cmath.phase(t2)))
      cf[10] = t1.real * t2.imag + 10j
      return cf.astype(np.complex128)
    except:
      return np.zeros(25, dtype=complex)
    
#
def poly884(t1, t2):
  try:
    cf = np.zeros(11, dtype=complex)
    coeff_sequence = np.linspace(
        start=-1 + 2*t1.real,
        stop=1 - 2*t2.imag,
        num=11
    )
    log_factor = (math.log(1 + np.abs(t2)))**2
    for i in range(1, 12):
        seq_sum = np.sum(coeff_sequence[:i])
        cf[i-1] = cmath.exp(1j * i * t1) + seq_sum * log_factor
    cf[10] = cf[10] + cmath.sqrt(cf[0] * cf[1] * t1)
    cf[0] = cf[0] - cmath.sqrt(cf[9] * cf[10] * t2)
    cf[5] = np.sum(cf) / 11
    cf[2] = cf[2] * cf[7] / cf[5]
    cf[7] = cf[7]**2 - cf[4] + cf[8]
    return cf.astype(np.complex128)
  except:
    return np.zeros(25, dtype=complex)
  
#  
def poly901(t1, t2):
    try:
      cf = np.zeros(10, dtype=complex)
      cf[0] = (t1**2 + t2**2) * 1j
      cf[1] = 10
      cf[4] = np.abs(t1 * 100) - 0.5
      cf[5] = np.abs(t2 * 100) - 0.5
      cf[7] = -10
      cf[9] = t1**2 + t2**2
      return cf.astype(np.complex128)
    except:
      return np.zeros(25, dtype=complex)

# 
def poly907(t1, t2):
    try:
      n = 10
      cf = np.zeros(n, dtype=complex)
      cf[0] = 100 * (cmath.sin(t1)**3) * (cmath.cos(t2)**2)
      cf[1] = 100 * cmath.exp(1j * (t1 + t2)) - 10 * ((t1 - t2)**2)
      cf[2] = (t1 * t2 * (t1 - t2)) / (np.abs(t1) + np.abs(t2) + 1)
      cf[4] = (t1 * t2 * cmath.exp(1j * (t1**2 - t2**2)))**3
      cf[6] = (cmath.sqrt(np.abs(t1)) -
               cmath.sqrt(np.abs(t2)) +
               1j * cmath.sin(t1 * t2))
      cf[7] = 50 * np.abs(t1 - t2) * cmath.exp(1j * np.abs(t1 + t2))
      if t1.imag > 0:
          cf[8] = t1 - np.abs(t2)
      else:
          cf[8] = t2 - np.abs(t1)
      cf[9] = (1j * (t1 * t2))**(0.1 * t1 * t2)
      return cf.astype(np.complex128)
    except:
      return np.zeros(25, dtype=complex)
    
#  
def poly918(t1, t2):
    try:
      cf = np.zeros(25, dtype=complex)
      cf[0] = t1**2 + t2**2 - t1 * t2
      for k in range(1, 25):
          cf[k] = cf[k - 1] * (t1 + t2) / (1 + np.abs(cf[k - 1]))
      add_indices = [2, 5, 8, 11, 14, 17, 20, 23]
      sub_indices = [1, 4, 7, 10, 13, 16, 19, 22, 24]
      for idx in add_indices:
          cf[idx] += (t1 + 1j * t2)
      for idx in sub_indices:
          cf[idx] -= (t2 + 1j * t1)
      return cf.astype(np.complex128)
    except:
      return np.zeros(25, dtype=complex)

#
def poly958(t1, t2):  
    try:
      cf = [0+0j] * 25
      cf[0] = t1 + t2
      cf[4] = 5 * (t1 * t2).real
      cf[8] = 9 * np.sin(np.angle(t1 * t2.conjugate()))
      for i in range(1, 4):
          cf[i] = cf[i-1]**2 + cf[4]
      for j in range(5, 8):
          cf[j] = np.abs(t1)**(cf[j-1]) + cf[0]
      for k in range(9, 25):
          cf[k] = np.log(np.abs(cf[k-1])+1) + cf[8]
      return cf.astype(np.complex128)
    except:
      return np.zeros(25, dtype=complex)

#  
def poly1000(t1, t2):
    try:
        cf = [0j] * 71
        cf[0] = t2.real**5 + 2*t1.imag
        cf[1] = t2.imag**4 - 3*t1.real**2 + t2.conjugate()
        cf[2] = 4*t2*t1.imag + 5*complex(np.tanh(t2))
        cf[3] = 6*complex(np.cosh(t1)) - 7*t2.real**2
        cf[4] = 8*t1.imag**3 + 9*t2.real*t1
        cf[5] = 10*complex(np.sinh(t2 + t1)) - 11*t1
        cf[6] = 12*complex(np.exp(t2)) + 13*t2.imag*t1
        cf[7] = 14*t2.real**3 - 15*t1.imag**2
        cf[8] = 16*t2*t1.real**2 + 17*complex(np.tanh(t1))
        cf[9] = 18*complex(np.cosh(t2 + t1)) - 19*t2
        cf[10] = 20*t1.real**4 + 21*t2.imag**2*t1
        cf[11] = 22*complex(np.sinh(t2)) - 23*t1.real**3
        cf[12] = 24*complex(np.exp(t1)) + 25*t2.imag*t1
        cf[13] = 26*t2.real**4 - 27*t1.imag**2*t2
        cf[14] = 28*complex(np.tanh(t2 + t1)) + 29*t1
        cf[15] = 30*complex(np.cosh(t2)) - 31*t2*t1.real**2
        cf[16] = 32*t1.imag**5 + 33*t2.real**3
        cf[17] = 34*complex(np.sinh(t1)) - 35*t2.imag*t1
        cf[18] = 36*complex(np.exp(t2 + t1)) + 37*t1.real**3
        cf[19] = 38*t2.imag**5 - 39*t1.real**2
        cf[20] = 40*complex(np.tanh(t2)) + 41*t2.real**2*t1
        cf[21] = 42*complex(np.cosh(t1)) - 43*t1.imag**4
        cf[22] = 44*t2.real**3*t1 + 45*complex(np.sinh(t2 + t1))
        cf[23] = 46*complex(np.exp(t2)) - 47*t1.real**3*t2
        cf[24] = 48*t1.imag**6 + 49*t2.real**4
        cf[25] = 50*complex(np.tanh(t1)) + 51*t2*t1.real**2
        cf[26] = 52*complex(np.cosh(t2 + t1)) - 53*t1.imag**4
        cf[27] = 54*t2.real**6 - 55*t1.real**3
        cf[28] = 56*complex(np.sinh(t2)) + 57*t2.imag**3*t1
        cf[29] = 58*complex(np.exp(t1)) - 59*t1.imag**5
        cf[30] = 60*t2.real**4*t1 + 61*complex(np.tanh(t2 + t1))
        cf[31] = 62*complex(np.cosh(t2)) + 63*t1.imag**4*t2
        cf[32] = 64*t1.real**7 + 65*t2.imag**5
        cf[33] = 66*complex(np.sinh(t1)) - 67*t2*t1.real**3
        cf[34] = 68*complex(np.exp(t2 + t1)) + 69*t1.imag**5
        cf[35] = 70*t2.real**7 - 71*t1.real**4
        cf[36] = 72*complex(np.tanh(t2)) + 73*t2.imag**4*t1
        cf[37] = 74*complex(np.cosh(t1)) - 75*t1.real**6
        cf[38] = 76*t2.imag**5*t1 + 77*complex(np.sinh(t2 + t1))
        cf[39] = 78*complex(np.exp(t2)) - 79*t1.imag**5*t2
        cf[40] = 80*t1.real**8 + 81*t2.real**6
        cf[41] = 82*complex(np.tanh(t1)) + 83*t2*t1.imag**4
        cf[42] = 84*complex(np.cosh(t2 + t1)) - 85*t1.real**6
        cf[43] = 86*t2.imag**8 - 87*t1.imag**5
        cf[44] = 88*complex(np.sinh(t2)) + 89*t2.real**5*t1
        cf[45] = 90*complex(np.exp(t1)) - 91*t1.real**7
        cf[46] = 92*t2.imag**6*t1 + 93*complex(np.tanh(t2 + t1))
        cf[47] = 94*complex(np.cosh(t2)) + 95*t1.real**6*t2
        cf[48] = 96*t1.imag**9 + 97*t2.real**7
        cf[49] = 98*complex(np.sinh(t1)) - 99*t2*t1.imag**5
        cf[50] = 100*complex(np.exp(t2 + t1)) + 101*t1.real**7
        cf[51] = 102*t2.imag**9 - 103*t1.imag**6
        cf[52] = 104*complex(np.tanh(t2)) + 105*t2.real**6*t1
        cf[53] = 106*complex(np.cosh(t1)) - 107*t1.real**8
        cf[54] = 108*t2.imag**7*t1 + 109*complex(np.sinh(t2 + t1))
        cf[55] = 110*complex(np.exp(t2)) + 111*t1.real**7*t2
        cf[56] = 112*t1.imag**10 + 113*t2.real**8
        cf[57] = 114*complex(np.tanh(t1)) - 115*t2*t1.real**6
        cf[58] = 116*complex(np.cosh(t2 + t1)) + 117*t1.imag**8
        cf[59] = 118*t2.real**10 - 119*t1.real**7
        cf[60] = 120*complex(np.sinh(t2)) + 121*t2.imag**7*t1
        cf[61] = 122*complex(np.exp(t1)) - 123*t1.imag**9
        cf[62] = 124*t2.real**8*t1 + 125*complex(np.tanh(t2 + t1))
        cf[63] = 126*complex(np.cosh(t2)) + 127*t1.real**8*t2
        cf[64] = 128*t1.imag**11 + 129*t2.imag**9
        cf[65] = 130*complex(np.sinh(t1)) - 131*t2*t1.real**7
        cf[66] = 132*complex(np.exp(t2 + t1)) + 133*t1.imag**9
        cf[67] = 134*t2.real**11 - 135*t1.real**8
        cf[68] = 136*complex(np.tanh(t2)) + 137*t2.imag**8*t1
        cf[69] = 138*complex(np.cosh(t1)) - 139*t1.real**10
        cf[70] = 140*t2.real**9*t1 + 141*complex(np.sinh(t2 + t1)) / 200
        return cf.astype(np.complex128)
    except:
        return np.zeros(25, dtype=complex)


def tst1(t1, t2):
    cf = np.array([
        1,
        t1,
        t2
    ], dtype=np.complex128)
    return cf.astype(np.complex128)

def tst2(t1, t2):
    cf = np.array([
        t2,
        t1,
        1
    ], dtype=np.complex128)
    return cf.astype(np.complex128)

def tst3(t1, t2):
    frame_count = 23
    frame_speed = 1600/8
    frame_start = -800
    time = int(np.abs(t1) * 883) % frame_count
    frame = frame_start + frame_speed * time
    cf = np.array([
        100/(2+t1-t2),
        1/(1+t1),
        - frame / (2 + t1**2 - t2**2 + t1 + t2),
        1/(1+t2),
        100/(1+t1*t2)
    ], dtype=np.complex128)
    return cf.astype(np.complex128)

# Littlewood
def ltlwd(t1,t2):
    n = ps.poly.get("n") or 24
    cf = np.random.choice([-1, 1], size = n).astype(complex)
    return cf.astype(np.complex128)

kabvec=np.array([],dtype=complex)

def kabalistic_vector(s):
    kabala_values = {
        'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9,
        'J': 10, 'K': 20, 'L': 30, 'M': 40, 'N': 50, 'O': 60, 'P': 70, 'Q': 80,
        'R': 90, 'S': 100, 'T': 200, 'U': 300, 'V': 400, 'W': 500, 'X': 600,
        'Y': 700, 'Z': 800

    }

    
    s = s.replace("0","nulla")
    s = s.replace("1","aleph")
    s = s.replace("2","bet")
    s = s.replace("3","gimel")
    s = s.replace("4","dalet")
    s = s.replace("5","he")
    s = s.replace("6","vav")
    s = s.replace("7","zayin")
    s = s.replace("8","het")
    s = s.replace("9","tet")
    s = s.upper()

    values = [kabala_values.get(char, 0) for char in s]+[1]

    return np.array(values)

def kabala(t1,t2):
    global kabvec
    i = ps.poly.get("i") or 0
    if i==0:
        word = ps.poly.get("word") or "schlong"
        kabvec=kabalistic_vector(word)
    cf = kabvec * t1 - 1j * np.flip(kabvec) * t2
    return cf.astype(np.complex128)
    
cf0 = np.array([],dtype=complex)
def kabala1(t1,t2):
    global cf0
    i = ps.poly.get("i") or 0
    if i==0:
        cf0 = kabala(t1,t2)
    cf1 = kabala(t1,t2)
    cf2 =  ( cf0 + cf1 )
    cf = cf2 / np.abs(cf2) 
    cf0 = cf
    return cf.astype(np.complex128)

def kabala2(t1,t2):
    global cf0
    i = ps.poly.get("i") or 0
    if i==0:
        cf0 = kabala(t1,t2)
    cf1 = kabala(t1,t2)
    cf2 =  ( cf0 + cf1 )
    cf = cf2 / np.abs(cf2) 
    cf0 = cf
    return cf.astype(np.complex128)

# Littlewood deg 24
def ltlwd24(t1,t2):
    n = ps.poly.get("n") or 24
    cf = np.random.choice([-1, 1], size = 24).astype(complex)
    return cf.astype(np.complex128)

def p01d24(t1,t2):
    n = ps.poly.get("n") or 24
    cf = np.random.choice([0, 1], size = n).astype(complex)
    return cf.astype(np.complex128)

def p01d31(t1,t2):
    cf = np.random.choice([0, 1], size = 31).astype(complex)
    return cf.astype(np.complex128)

# Littlewood deg 71
def ltlwd71(t1,t2):
    n = ps.poly.get("n") or 71
    m = ps.poly.get("m") or 11
    degree = np.random.randint(m, n)
    cf0 = np.random.choice([-1, 1], size = degree).astype(complex)
    cf1 = np.arange(degree)+1
    cf = cf0*(cf1**5)
    return cf.astype(np.complex128)    

def prdhd31(t1,t2):
    degree = np.random.randint(3, 31)
    height = np.random.randint(1, 10)
    cf = np.random.choice([-height*t1, height*t2], size = degree).astype(complex)
    return cf.astype(np.complex128)     

def poly_creative3(t1, t2):
    """
    Roots placed on a Lissajous curve parameterized by t1/t2.
    for -x, try none, z01, uc
    """
    try:
        m = ps.poly.get("m") or 250
        n = ps.poly.get("n") or 70
        f1 = int(ps.poly.get("f1") or 0)
        f2 = int(ps.poly.get("f2") or 0)
        f3 = int(ps.poly.get("f3") or 0)
        f4 = int(ps.poly.get("f4") or 0)
        roots = []
        a, b = np.abs(t1), np.abs(t2)
        delta = cmath.phase(t1*t2)
        for k in range(n):
            t = 2*np.pi*k/n
            x = a*np.sin(t + delta)
            y = b*np.sin(2*t + delta)
            roots.append(x + 1j*y)
        coeffs = np.poly(roots).astype(complex)
        k = len(coeffs)
        if f1==1:
            coeffs = coeffs + np.exp(-m)
        if f2==1:
            adj =  np.exp(-m)*np.exp( 1j * 2 * np.pi * t1 ) 
            coeffs = coeffs + adj
        if f3==1:
            adj = np.exp(-m)*np.exp( 1j * 2 * np.pi * t2 )
            coeffs = coeffs + adj
        if f4 == 1 :
            coeffs = coeffs + m*t1
        elif f4 == 2 :
            coeffs = coeffs + t1 + 1j*t2
        elif f4 == 3 :
            coeffs = coeffs + np.exp(-m) * t1 * np.exp( 1j * 2 * np.pi * t2 ) 
        elif f4 == 4 :
            adj = np.exp(-m) * t1 * np.exp( 1j * 2 * np.pi * t2 )
            coeffs = coeffs + np.arange(1,k+1) * adj
        return coeffs
    except Exception as e:
        print(f"Exception message: {e}")
        return np.zeros(n, dtype=complex)
    

def poly_creative4(t1, t2):
    """Modular arithmetic and phase twisting with dynamic shifts."""
    try:
        n = ps.poly.get("n") or 71
        cf = np.zeros(n, dtype=complex)
        mod1 = int(np.abs(t1)*100) %13 +1
        mod2 = int(np.abs(t2)*100) %17 +1
        for k in range(n):
            freq = (k%mod1)*t1.real + (k%mod2)*t2.imag
            phase = (k*cmath.phase(t1))/(k+1) + (k*cmath.phase(t2))/(k+1)
            cf[k] = (np.sin(freq) + 1j*np.cos(freq)) * cmath.exp(1j*phase)
            cf[k] *= (np.abs(t1)**(k/mod1) + np.abs(t2)**(k/mod2))
        shift = int(t1.imag*10) %n
        return np.roll(cf, shift)
    except:
        return np.zeros(25, dtype=complex)
    
def poly_creative5(t1, t2):
    """Chaotic logistic map dynamics driven by t1/t2."""
    try:
        n = ps.poly.get("n") or 71
        cf = np.zeros(n, dtype=complex)
        x = 0.5*(t1.real + t1.imag)/(np.abs(t1)+1e-6)
        r = 3.7 + 0.3*(t2.real - t2.imag)/(np.abs(t2)+1e-6)
        for k in range(n):
            x = r*x*(1-x)
            angle = 2*np.pi*x
            cf[k] = x*(np.cos(angle) + 1j*np.sin(angle))
        cf /= np.max(np.abs(cf)) + 1e-6
        return cf.astype(np.complex128)*100
    except:
        return np.zeros(25, dtype=complex)
    
def poly_creative6(t1, t2):
    """Quantum-inspired superposition with entanglement."""
    try:
        n = ps.poly.get("n") or 71
        cf = np.zeros(n, dtype=complex)
        for k in range(n):
            state1 = cmath.exp(1j*(k*cmath.phase(t1) + np.abs(t1)))
            state2 = cmath.exp(-1j*(k*cmath.phase(t2) - np.abs(t2)))
            cf[k] = (state1 + state2)/2
            if k > 0:
                cf[k] += cf[k-1]*(t1.real + 1j*t2.imag)
        global_phase = np.sum(cf)/(n+1)
        return cf.astype(np.complex128) * cmath.exp(1j*cmath.phase(global_phase))
    except:
        return np.zeros(25, dtype=complex)

def poly_creative7(t1, t2):
    """Mandelbrot-inspired iterations for fractal patterns."""
    try:
        n = ps.poly.get("n") or 71
        cf = np.zeros(n, dtype=complex)
        for k in range(n):
            z, c = t1, t2
            for _ in range(n):
                z = z**4 + c
            cf[k] = z/(k+1)
        return cf.astype(np.complex128) * np.exp(-abs(cf))
    except:
        return np.zeros(25, dtype=complex)

def poly_creative8(t1, t2):
    """Hamiltonian-like terms with position/momentum mixing."""
    try:
        n = ps.poly.get("n") or 71
        cf = np.zeros(n, dtype=complex)
        for k in range(n):
            if k%2 ==0:
                q = (k//2+1)*t1.real
                cf[k] = q**2 + 1j*q*t2.imag
            else:
                p = (k//2+1)*t1.imag
                cf[k] = p**2 - 1j*p*t2.real
        cf[::2] += cf[1::2].conj()
        cf[1::2] -= cf[::2].conj()
        return cf.astype(np.complex128)
    except:
        return np.zeros(25, dtype=complex)

def poly_creative9(t1, t2):
    """Fourier series with frequency decay and neighbor mixing."""
    try:
        n = ps.poly.get("n") or 71
        cf = np.zeros(n, dtype=complex)
        for k in range(n):
            freq_t1 = (k+1)*cmath.phase(t1)
            freq_t2 = (k+1)*cmath.phase(t2)
            cf[k] = (np.sin(freq_t1)+1j*np.cos(freq_t2)) * np.exp(-abs(t1*t2)*k/n)
        for k in range(1, n-1):
            cf[k] = (cf[k-1] + cf[k+1])*0.5*(t1 + t2)
        return cf.astype(np.complex128)
    except:
        return np.zeros(25, dtype=complex)

def poly_creative10(t1, t2):
    """Geometric algebra product terms with alternating signs."""
    try:
        n = ps.poly.get("n") or 71
        cf = np.zeros(n, dtype=complex)
        for k in range(n):
            a = [t1.real, t1.imag]
            b = [t2.real, t2.imag]
            dot = a[0]*b[0] + a[1]*b[1]
            wedge = a[0]*b[1] - a[1]*b[0]
            gp = dot + 1j*wedge
            cf[k] = gp**(k+1)
        cf[::2] *= -1
        return cf.astype(np.complex128)
    except:
        return np.zeros(25, dtype=complex)

def poly_neuralnet(t1, t2):
    """Micro neural network with complex-valued activations"""
    try:
        n = ps.poly.get("n") or 71
        # Input layer
        x = np.array([t1.real, t1.imag, t2.real, t2.imag])
        
        # Hidden layer (complex activations)
        w1 = np.array([[0.7, -0.3, 1.2, 0.4],
                      [0.5, 1.1, -0.9, 0.2],
                      [0.3, 0.8, 0.6, -1.0]])
        b1 = np.array([0.1, -0.5, 0.7])
        h = np.tanh(w1 @ x + b1) + 1j*np.sin(w1 @ x[::-1] + b1)
        
        # Output layer
        w2 = np.array([[0.4, -0.8, 1.1],
                      [0.3, 0.9, -0.5],
                      [-0.2, 0.6, 0.7]])
        cf = w2 @ h
        
        # Expand to 71 coefficients using Fibonacci phyllotaxis
        phi = (1 + np.sqrt(5))/2
        angles = np.arange(n) * 2*np.pi/phi**2
        return cf.astype(np.complex128) * np.exp(1j*angles) * np.linspace(1,0.5,n)
    except:
        return np.zeros(25, dtype=complex)

def poly_neuralnet1(t1, t2):
    """Micro neural network with complex-valued activations"""
    try:
        n = ps.poly.get("n") or 71
        # Input layer
        x = np.array([t1.real, t1.imag, t2.real, t2.imag])
        
        # Hidden layer (complex activations)
        w1 = np.array([[0.7, -0.3, 1.2, 0.4],
                       [0.5, 1.1, -0.9, 0.2],
                       [0.3, 0.8, 0.6, -1.0]])
        b1 = np.array([0.1, -0.5, 0.7])
        h_real = np.tanh(w1 @ x + b1)
        h_imag = np.sin(w1 @ x[::-1] + b1)
        h = h_real + 1j*h_imag
        
        # Output layer
        w2 = np.array([[0.4, -0.8, 1.1],
                       [0.3, 0.9, -0.5],
                       [-0.2, 0.6, 0.7]])
        cf = w2 @ h  # shape (3,)
        
        # Combine the 3 outputs into one amplitude
        cf_single = cf.sum()  # shape ()
        
        # Expand to n=71 coefficients
        phi = (1 + np.sqrt(5))/2
        angles = np.arange(n) * 2*np.pi/phi**2
        radial = np.linspace(1, 0.5, n)
        
        # This now has shape (71,)
        coeffs = cf_single * np.exp(1j*angles) * radial
        return coeffs
    
    except Exception as e:
        print("poly_neuralnet error:", e)
        return np.zeros(25, dtype=complex)
    

def poly_ca(t1, t2):
    """Cellular automaton rule evolution over coefficient indices"""
    try:
        n = ps.poly.get("n") or 71
        cf = np.zeros(n, dtype=complex)
        
        # Determine CA rule from parameter angles
        rule = int(np.abs(t1*t2) % 256)
        seed = [int(bit) for bit in bin(int(100*abs(t1-t2)))[2:][-8:]]
        ca = seed * (n//8 + 1)
        
        for k in range(n):
            # Update CA according to rule
            new_ca = []
            for i in range(1, len(ca)-1):
                neighborhood = 4*ca[i-1] + 2*ca[i] + ca[i+1]
                new_ca.append((rule >> neighborhood) & 1)
            ca = [0] + new_ca + [0]
            
            # Convert CA state to complex number
            state_num = sum(2**i * bit for i,bit in enumerate(ca[:8]))
            angle = 2*np.pi*state_num/256
            cf[k] = cmath.exp(1j*angle) * (sum(ca)+1)
        
        return cf.astype(np.complex128) * np.geomspace(1, 0.01, n)
    except:
        return np.zeros(25, dtype=complex)
    
def poly_quantum(t1, t2):
    """Quantum walk-inspired coefficients with phase interference"""
    try:
        n = ps.poly.get("n") or 71
        cf = np.zeros(n, dtype=complex)
        pos = 0
        state = np.array([1.0, 0.0], dtype=complex)
        
        # Coin operator parameters from inputs
        theta = cmath.phase(t1)
        phi = cmath.phase(t2)
        
        for k in range(n):
            # Quantum coin flip
            coin = np.array([[np.cos(theta), np.sin(theta)*cmath.exp(1j*phi)],
                            [np.sin(theta)*cmath.exp(-1j*phi), -np.cos(theta)]])
            state = coin @ state
            
            # Position update and record coefficient
            pos += 1 if np.random.random() < np.abs(state[0])**2 else -1
            cf[k] = state[0] + 1j*state[1]
            
            # Decoherence factor from parameters
            state *= 0.9 + 0.1*abs(t1-t2)/(np.abs(t1)+abs(t2)+1e-6)
        
        return cf.astype(np.complex128) * np.exp(1j*np.linspace(0, 4*np.pi, n))
    except:
        return np.zeros(25, dtype=complex)


def poly_topological(t1, t2):
    """Knot theory-inspired coefficients using Alexander polynomials"""
    try:
        n = ps.poly.get("n") or 71
        cf = np.zeros(n, dtype=complex)
        winding = int(np.abs(t1.real*t2.imag - t1.imag*t2.real) % 11)
        
        # Generate coefficients based on knot crossings
        for k in range(n):
            sign = (-1)**(k % winding) if winding >0 else 1
            twist = np.sin(k * cmath.phase(t1)) + 1j*np.cos(k * cmath.phase(t2))
            cf[k] = sign * (np.abs(t1)**(k%5) - np.abs(t2)**(k%3)) * twist
        
        # Mirror symmetry for real knots
        cf[n//2:] = cf[n//2-1::-1].conj()
        return cf.astype(np.complex128) * np.exp(-np.linspace(0, 2, n))
    except:
        return np.zeros(25, dtype=complex)

def poly_biomorphic(t1, t2):
    """Lindenmayer systems meets biochemical oscillations"""
    try:
        n = ps.poly.get("n") or 71
        cf = np.zeros(n, dtype=complex)
        axiom = 'A'
        rules = {
            'A': f"A{int(np.abs(t1))%5}B",
            'B': f"C[{int(np.abs(t2))%3}A]",
            'C': "AC"
        }
        
        # L-system expansion
        s = axiom
        for _ in range(4):
            s = ''.join([rules.get(c,c) for c in s])
        
        # Convert string to coefficients
        depth = 0
        for k, c in enumerate(s[:n]):
            if c == 'A': cf[k] += 0.5+0.2j
            if c == 'B': cf[k] -= 0.3-0.4j
            if c == 'C': cf[k] *= 1.2+0.8j
            if c == '[': depth += 1
            if c == ']': depth -= 1
            cf[k] *= (0.8**depth) * (1 + 0.1j*k)
        
        # Normalize and apply spiral
        return cf.astype(np.complex128) * np.exp(1j*np.linspace(0, 8*np.pi, n))
    except:
        return np.zeros(25, dtype=complex)

def poly_gravitational(t1, t2):
    """N-body simulation in coefficient space"""
    try:
        n = ps.poly.get("n") or 71
        masses = np.abs([t1, t2, t1+t2, t1-t2])
        positions = np.array([t1, t2, t1*t2, t1/t2], dtype=complex)
        
        cf = np.zeros(n, dtype=complex)
        for k in range(n):
            # Calculate gravitational forces
            forces = np.zeros(4, dtype=complex)
            for i in range(4):
                for j in range(4):
                    if i != j:
                        r = positions[j] - positions[i]
                        forces[i] += masses[j]*r/(np.abs(r)**3 + 1e-6)
            
            # Update positions and record
            positions += 0.1*forces
            cf[k] = np.sum(positions * masses[:,None])
            
            # Parameter-dependent damping
            positions *= 0.95 + 0.05*abs(t1-t2)/(np.abs(t1)+abs(t2)+1e-6)
        
        return cf.astype(np.complex128) * np.geomspace(1, 0.001, n)
    except:
        return np.zeros(25, dtype=complex)

def poly_sonic(t1, t2):
    """Audio waveform synthesis in complex plane"""
    try:
        n = ps.poly.get("n") or 71
        cf = np.zeros(n, dtype=complex)
        
        # FM synthesis parameters from inputs
        carrier_freq = 440*(np.abs(t1)/max(np.abs(t1),1))
        mod_freq = 440*(np.abs(t2)/max(np.abs(t2),1))
        mod_index = 10*abs(t1-t2)/(np.abs(t1)+abs(t2)+1e-6)
        
        for k in range(n):
            t = k/n
            # FM synthesis equation
            mod = mod_index * np.sin(2*np.pi*mod_freq*t)
            wave = np.sin(2*np.pi*carrier_freq*t + mod)
            # Convert to analytic signal
            cf[k] = wave + 1j*np.cos(2*np.pi*carrier_freq*t + mod)
        
        # Apply frequency-dependent phase shift
        return cf.astype(np.complex128) * np.exp(1j*np.linspace(0, 4*np.pi, n))
    except:
        return np.zeros(25, dtype=complex)

def poly_cryptic(t1, t2):
    """Number theory meets elliptic curve cryptography"""
    try:
        n = ps.poly.get("n") or 71
        cf = np.zeros(n, dtype=complex)
        
        # Use inputs to seed prime field operations
        p = 2**256 - 2**32 - 977  # secp256k1 prime
        a = int(np.abs(t1)*1e6) % p
        b = int(np.abs(t2)*1e6) % p
        
        # Elliptic curve point addition
        x, y = a, b
        for k in range(n):
            # EC point doubling formula
            s = (3*x**2 + a) * pow(2*y, p-2, p) % p
            x_new = (s**2 - 2*x) % p
            y_new = (s*(x - x_new) - y) % p
            x, y = x_new, y_new
            
            # Map to complex plane
            cf[k] = complex(x/p, y/p) * (-1)**k
        
        return cf.astype(np.complex128) * np.logspace(0, -3, n, base=np.e)
    except:
        return np.zeros(25, dtype=complex)

def poly_holographic(t1, t2):
    """Wave interference patterns with parameterized diffraction"""
    try:
        n = ps.poly.get("n") or 71
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        X, Y = np.meshgrid(x, y)
        
        # Dynamic grating parameters from inputs
        freq = 10*(np.abs(t1) + np.abs(t2))
        angle = cmath.phase(t1*t2)
        phase_mod = np.exp(1j*(X*np.cos(angle) + Y*np.sin(angle))*freq)
        
        # Compute holographic pattern
        pattern = np.fft.fftshift(np.fft.fft2(phase_mod)).flatten()
        return pattern[:n] * np.exp(-(x**2 + y**2)/0.5)
    except:
        return np.zeros(25, dtype=complex)

def poly_chaosmorph(t1, t2):
    """Chaotic attractor coefficient sampling"""
    try:
        n = ps.poly.get("n") or 71
        cf = np.zeros(n, dtype=complex)
        # Rossler system parameters from inputs
        a = 0.2 + 0.1*abs(t1)
        b = 0.2 + 0.1*abs(t2)
        c = 5.7 + cmath.phase(t1*t2)
        x, y, z = 0.1, 0.0, 0.0
        
        for k in range(n*10):  # Transient steps
            dx = -y - z
            dy = x + a*y
            dz = b + z*(x - c)
            x += 0.01*dx
            y += 0.01*dy
            z += 0.01*dz
            if k >= n*5:  # Record strange attractor points
                cf[k%n] += complex(x,y) * np.exp(-0.1*z)
        
        return cf.astype(np.complex128) * np.geomspace(1, 0.001, n)
    except:
        return np.zeros(25, dtype=complex)


def poly_quasicrystal(t1, t2):
    """Aperiodic Penrose-like tiling in coefficient space"""
    try:
        n = ps.poly.get("n") or 71
        phi = (1 + np.sqrt(5))/2
        angles = np.arange(n) * 2*np.pi/phi
        radii = np.sqrt(np.arange(n)/n)
        
        # Projection from 5D space with golden ratio distortions
        cf = np.sum([np.exp(1j*(radii*k*np.cos(angles + t1.real) + 
                               radii*k*np.sin(angles*phi + t2.imag)))
                   for k in [1, phi, phi**2]], axis=0)
        
        # Apply phasonic modulation
        return cf.astype(np.complex128) * np.exp(1j*np.sin(radii*abs(t1*t2)))
    except:
        return np.zeros(25, dtype=complex)


def poly_neuroevolution(t1, t2):
    """Coefficients evolve via genetic algorithm mechanics"""
    try:
        n = ps.poly.get("n") or 71
        population = np.random.rand(10,n) + 1j*np.random.rand(10,n)
        fitness = np.zeros(10)
        
        for generation in range(5):
            # Evaluate fitness using parameter landscape
            for i in range(10):
                roots = np.roots(population[i])
                fitness[i] = -np.sum(np.abs(roots**2 - t1*roots - t2))
            
            # Selection and crossover
            parents = population[np.argsort(fitness)[-2:]]
            children = np.zeros((8,n), dtype=complex)
            for c in range(8):
                cross = np.random.randint(0,n,size=n)
                children[c] = np.where(np.random.rand(n)<0.5, 
                                     parents[0], parents[1])
            
            # Mutation and elite preservation
            population = np.vstack([parents, 
                                   children + 0.1*np.random.randn(8,n)])
        
        return population[0] * np.hanning(n)
    except:
        return np.zeros(25, dtype=complex)

def poly_fluid(t1, t2):
    """Navier-Stokes inspired vorticity transport"""
    try:
        n = ps.poly.get("n") or 71
        # Initialize velocity field
        vx = np.zeros((n,n))
        vy = np.zeros((n,n))
        # Parameter-driven forces
        vx[n//2,:] += t1.real
        vy[:,n//2] += t1.imag
        vort = np.zeros((n,n))
        
        for k in range(n):
            # Advection-diffusion simulation
            vort = 0.99*vort + 0.01*(np.roll(vx,1,axis=0) - np.roll(vx,-1,axis=0)
                                    - np.roll(vy,1,axis=1) + np.roll(vy,-1,axis=1))
            # Convert vorticity slice to coefficients
            cf = vort[k,:] + 1j*vort[:,k]
            cf /= np.max(np.abs(cf)) + 1e-6
        
        return cf.astype(np.complex128) * np.exp(-np.linspace(0,3,n))
    except:
        return np.zeros(25, dtype=complex)

def poly_astro(t1, t2):
    """Cosmic microwave background-inspired fluctuations"""
    try:
        n = ps.poly.get("n") or 71
        # Spherical harmonics with parameter-driven modes
        theta = np.linspace(0, np.pi, n)
        phi = np.linspace(0, 2*np.pi, n)
        l = int(np.abs(t1) % 10)
        m = int(np.abs(t2) % (l+1))
        
        cf = sph_harm(m, l, phi, theta).real + \
             1j*sph_harm(m, l, phi, theta).imag
        # Add Gaussian random fluctuations
        cf += 0.1*(np.random.randn(n) + 1j*np.random.randn(n))
        return cf.astype(np.complex128) * np.exp(-(theta**2 + phi**2)/4)
    except:
        return np.zeros(25, dtype=complex)


def poly_metamaterial(t1, t2):
    """Negative-index photonic crystal band structure"""
    try:
        n = ps.poly.get("n") or 71
        # Create dielectric contrast pattern
        ε = 1 + 0.5*(np.sign(np.sin(10*abs(t1)*np.pi*np.arange(n)/n)) + 
                    np.sign(np.cos(8*abs(t2)*np.pi*np.arange(n)/n)))
        
        # Solve 1D photonic crystal dispersion
        M = np.diag(2*ε) - np.diag(ε[:-1],1) - np.diag(ε[1:],-1)
        eigvals = np.linalg.eigvalsh(M)
        return eigvals + 1j*np.gradient(eigvals)
    except:
        return np.zeros(25, dtype=complex)

def poly_sync(t1, t2):
    """Kuramoto oscillator synchronization dynamics"""
    try:
        n = ps.poly.get("n") or 71
        θ = np.linspace(0, 2*np.pi, n)
        ω = 1 + 0.1*(t1.real*np.cos(θ) + t2.imag*np.sin(θ))
        K = 0.1*abs(t1*t2)
        
        for _ in range(10):  # Temporal evolution
            dθ = ω + K*np.mean(np.sin(θ[:,None] - θ), axis=1)
            θ += 0.1*dθ
        
        return np.exp(1j*θ) * np.linspace(1,0.5,n)
    except:
        return np.zeros(25, dtype=complex)


def poly_memristor(t1, t2):
    """Brain-inspired memristive network dynamics"""
    try:
        n = ps.poly.get("n") or 71
        W = np.outer(np.linspace(0,1,n), np.ones(n))  # Memristance matrix
        V = t1.real*np.sin(np.linspace(0,2*np.pi,n)) + \
            t2.imag*np.cos(np.linspace(0,2*np.pi,n))
        
        for _ in range(5):  # Network updates
            I = W @ V
            W += 0.01*(np.outer(V,I) - 0.1*W)  # Hebbian-like update
            V = np.tanh(I) * np.exp(1j*np.angle(I))
        
        return np.diag(W) + 1j*np.diag(np.fft.fft(W))
    except:
        return np.zeros(25, dtype=complex)


def poly_swarm(t1, t2):
    """Particle swarm optimization trajectories"""
    try:
        n = ps.poly.get("n") or 71
        # Initialize particles
        pos = np.random.randn(n) + 1j*np.random.randn(n)
        vel = 0.1*(np.random.randn(n) + 1j*np.random.randn(n))
        pbest = pos.copy()
        gbest = np.mean(pos)
        
        for _ in range(5):  # Optimization steps
            # Update velocities
            vel = 0.5*vel + \
                0.3*(pbest - pos) + \
                0.2*(gbest - pos) * (t1.real + 1j*t2.imag)
            pos += vel
            # Update best positions
            pbest = np.where(np.abs(pos) < np.abs(pbest), pos, pbest)
            gbest = np.mean(pbest[np.argmin(np.abs(pbest))])
        
        return pos * np.hamming(n)
    except:
        return np.zeros(25, dtype=complex)


def poly_cogniverse(t1, t2):
    """Hyperdimensional computing memory superposition"""
    try:
        n = ps.poly.get("n") or 71
        # Create semantic vectors
        v1 = np.random.randn(n) + 1j*np.random.randn(n)
        v2 = np.random.randn(n) + 1j*np.random.randn(n)
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)
        
        # Bind and bundle operations
        bound = v1 * np.roll(v2, int(np.abs(t1)%n)) * np.exp(1j*cmath.phase(t2))
        bundled = (bound + np.roll(bound, int(np.abs(t2)%n))) / 2
        
        # Cleanup memory
        for _ in range(3):
            bundled = np.fft.ifft(np.fft.fft(bundled)**2).conj()
        
        return bundled * np.exp(1j*np.linspace(0,8*np.pi,n))
    except:
        return np.zeros(25, dtype=complex)

def poly_sandpile(a, b):
    """Self-organized criticality patterns"""
    n = ps.poly.get("n") or 71
    grid = np.zeros((n,n), dtype=complex)
    grid[n//2, n//2] = a*1000
    
    # Sandpile dynamics
    while True:
        unstable = np.where(grid >= 4)
        if len(unstable[0]) == 0: break
        for i,j in zip(*unstable):
            grid[i,j] -= 4
            if i > 0: grid[i-1,j] += 1
            if i < n-1: grid[i+1,j] += 1
            if j > 0: grid[i,j-1] += 1
            if j < n-1: grid[i,j+1] += 1
    
    return np.poly(grid.diagonal()) * (1 + 1j*b)

def poly_spinglass(a, b):
    """Sherrington-Kirkpatrick spin glass model"""
    n = ps.poly.get("n") or 71
    J = np.random.randn(n,n) * a
    J = (J + J.T)/np.sqrt(n)  # Symmetric couplings
    h = np.random.randn(n) * b  # Random fields
    
    # Parisi replica symmetry breaking
    λ = np.linalg.eigvalsh(J)
    return np.poly(λ + 1j*h[:len(λ)]) * np.exp(-np.arange(n)/10)



def skewed_random(a, size=1):
    u = np.random.uniform(0, 1, size)
    if a == 0:
        return u 
    elif a >= 1:
        return np.zeros_like(u)  
    else:
        return u ** (1 / (1 - a))
    
def random_bunched(a: float) -> float:
    if not (0 <= a <= 1):      
        raise ValueError("Parameter 'a' must be in the interval [0,1].")
    if a == 1:
        return 0.0
    u = np.random.rand()
    exponent = 1.0 / (1.0 - a)
    return u ** exponent

def nopoly_crazy1(t1,t2):
    i = ps.poly.get("i") or 0
    i = i + 1
    ps.poly["i"] = i
    x = (i % 71)/5
    y = (i % 101)/3
    ascii = ps.poly.get("ascii") or 2
    offset = x*np.exp(1j*2*np.pi*y)
    key = f"b{ascii}"
    rts = letters.square(key,0.1*t1,0.1*t2,offset)
    return rts


def nopoly_letter(t1,t2):
    ascii = ps.poly.get("ascii") or 2
    ro = ps.poly.get("ro") or 0
    io = ps.poly.get("io") or 0
    offset = ro + 1j*io
    key = f"b{ascii}"
    rts = letters.square(key,t1,t2,offset)
    return rts

def poly_letter_old(t1,t2):
    ascii = ps.poly.get("ascii") or 2
    ro = ps.poly.get("ro") or 0
    io = ps.poly.get("io") or 0
    norm =  ps.poly.get("norm") or False
    offset = ro + 1j*io
    key = f"b{ascii}"
    rts = letters.square(key,t1,t2,offset)
    mrt = np.max(np.abs(rts))+1
    cf = np.poly(rts/mrt if norm else rts).astype(complex)
    return cf.astype(np.complex128) 

def poly_letter(t1,t2):
    ascii = ps.poly.get("ascii") or 2
    ro = ps.poly.get("ro") or 0
    io = ps.poly.get("io") or 0
    factor = ps.poly.get("factor") or 1.0
    norm =  ps.poly.get("norm") or False
    offset = ro + 1j*io
    key = f"b{ascii}"
    offset = ro +1j*ro
    rts = letters.square(key,t1,t2,0+0j)*factor+offset
    mrt = np.max(np.abs(rts))+1
    cf = np.poly(rts/mrt if norm else rts).astype(complex)
    return cf.astype(np.complex128) 

def poly_letter_2(t1,t2):
    ascii = ps.poly.get("ascii2") or 2
    ro = ps.poly.get("ro2") or 0
    io = ps.poly.get("io2") or 0
    factor = ps.poly.get("factor2") or 1.0
    norm =  ps.poly.get("norm2") or False
    offset = ro + 1j*io
    key = f"b{ascii}"
    offset = ro +1j*ro
    rts = letters.square(key,t1,t2,0+0j)*factor+offset
    mrt = np.max(np.abs(rts))+1
    cf = np.poly(rts/mrt if norm else rts).astype(complex)
    return cf.astype(np.complex128) 

def poly_letter1(t1,t2):
    roots = letters.FONTXY['b2']-1
    pert = 0.1*np.real(t1) * np.exp( 1j*2 * np.pi * np.real(t2) )
    const = 1*np.sin(2*np.pi*t1*np.arange(len(roots)))
    cf = np.poly(roots+pert*(const+np.flip(const))).astype(complex)
    return cf.astype(np.complex128)

def poly_letter_roots(t1,t2):
    cf = letters.square("a",t1,t2,0.+0.j)
    return cf.astype(np.complex128)

def poly_letter_path(t1,t2):
    roots = letters.FONTXY['b2']
    pert = 0.1*np.real(t1) * np.exp( 1j*2 * np.pi * np.real(t2) )
    circles = 1*np.sin(2*np.pi*t1*np.arange(len(roots)))
    symmetrized_circles = (circles+np.flip(circles))
    cf = np.poly( roots + pert * symmetrized_circles ).astype(complex)
    return cf.astype(np.complex128)+1e9*random_bunched(0.999)

def poly_letter2(t1,t2):
    pert = 0.1*np.real(t1) * np.exp( 1j*2 * np.pi * np.real(t2) )
    roots1 = letters.FONTXY['x']+10
    circles1 = 0.1*np.sin(2*np.pi*t1*np.arange(len(roots1)))
    symmetrized_circles1 = (circles1+np.flip(circles1))
    cf1 = np.poly( roots1 + pert * symmetrized_circles1 ).astype(complex)
    roots2 = letters.FONTXY['v']-10
    circles2 = 0.1*np.sin(2*np.pi*t1*np.arange(len(roots2)))
    symmetrized_circles2 = (circles2+np.flip(circles2))
    cf2 = np.poly( roots2 + pert * symmetrized_circles2 ).astype(complex)
    a = bimodal_skewed(0.999)
    cf = cf1 * a + cf2 * (1-a)
    return cf.astype(np.complex128)

def test_letter(t1,t2):
    cf1 = letters.square( 'b2',  t1, t2, 0.+0.j   )
    cf2 = letters.square( 'b3',  t1, t2, 10.+0.j  )
    cf3 = letters.circle( 'b4',  t1, t2, 20.+0.j  )
    cf4 = letters.square( 'b5',  t1, t2, 0.+10.j  )
    cf5 = letters.square( 'b6',  t1, t2, 10.+10.j )
    cf6 = letters.square( 'b7',  t1, t2, 20.+10.j )
    cf7 = letters.square( 'b8',  t1, t2, 0.+20.j  )
    cf8 = letters.square( 'b9',  t1, t2, 10.+20.j )
    cf9 = letters.square( 'b10', t1, t2, 20.+20.j )
    return np.concatenate([
        cf1,
        cf2,
        cf3,
        cf4,
        cf5,
        cf6,
        cf7,
        cf8,
        cf9
    ])

def pad_vector(vec, target_length, pad_value=0+0j):
    return np.pad(vec, (0, target_length - len(vec)), constant_values=pad_value)


def combine(t1,t2):
    poly1 = ps.poly.get("poly1") or "none"
    if poly1=="none": return np.zeros(25, dtype=complex)
    poly2 = ps.poly.get("poly2") or "none"
    if poly2=="none": return np.zeros(25, dtype=complex)
    andy = ps.poly.get("andy") or 0.0
    cf1=globals().get(poly1)(t1,t2)
    cf2=globals().get(poly2)(t1,t2)
    max_length = max(len(cf1), len(cf2))
    pcf1 = pad_vector(cf1, max_length)
    pcf2 = pad_vector(cf2, max_length)
    return pcf1 * andy + pcf2 * ( 1 - andy )

def bimodal_skewed(a, size=1):
    u = np.random.uniform(0, 1, size)
    skewed = np.where(
        u < 0.5, 
        (2*u)**(1/(1-a)) / 2, 
        1 - (2*(1-u))**(1/(1-a))/2
    )
    return np.clip(skewed, 0, 1)

def path(t1,t2):
    bim = ps.poly.get("bim") or 0.99999999
    poly1 = ps.poly.get("poly1") or "none"
    if poly1=="none": return np.zeros(25, dtype=complex)
    poly2 = ps.poly.get("poly2") or "none"
    if poly2=="none": return np.zeros(25, dtype=complex)
    andy = bimodal_skewed(bim)
    cf1=globals().get(poly1)(t1,t2)
    cf2=globals().get(poly2)(t1,t2)
    max_length = max(len(cf1), len(cf2))
    pcf1 = pad_vector(cf1, max_length)
    pcf2 = pad_vector(cf2, max_length)
    return pcf1 * andy + pcf2 * ( 1 - andy )



############################################
# side-by-side transform
############################################

def nopoly_lissajous(t1, t2):
    try:
        n = ps.poly.get("n") or 70
        roots = []
        delta = np.angle(t1*t2)
        for k in range(n):
            t = 2*np.pi*k/n
            x = np.abs(t1)*np.sin(t + delta)
            y = np.abs(t2)*np.sin(2*t + delta)
            roots.append(x + 1j*y)
        return np.array(roots,dtype=complex)
    except Exception as e:
        print(f"Exception message: {e}")
        return np.zeros(n, dtype=complex)

def poly_lis2(t1,t2):
    N=71
    curve = []
    delta = np.pi/2
    a = 6
    b = 5
    t0 = 2*np.pi*np.random.uniform(0, 1)
    for k in range(N):
        t = t0+2.0*np.pi*float(k)/float(N)
        x = np.sin(a*t)
        y = np.sin(b*t + delta)
        curve.append(x + 1j*y)
    rpert = np.cumsum(np.random.choice([-1, 1], size=len(curve)+1))
    ipert = np.cumsum(np.random.choice([-1, 1], size=len(curve)+1)) * 1j
    coeffs = np.poly(curve)+0.15*(rpert+ipert)
    return np.array(coeffs,dtype=complex)

def poly_chess(t1,t2):
    N = 7
    w = 2*np.pi
    t = np.random.uniform(-w, w)
    x =  np.sin(t) + np.tile(np.arange(1, N+1), N)
    y =  np.cos(t) + np.repeat(np.arange(1, N+1), N)
    curve = np.array(x + 1j*y,dtype=complex) - ((N+1)/2) - 1j * ((N+1)/2)
    coeffs = np.poly(curve+np.cos(np.random.uniform(-w, w)))
    return np.array(coeffs+90*np.arange(len(coeffs))+np.poly(curve)*200,dtype=complex)


def polar_interpolation(x, y, a, geometric_modulus=False):
    r1, theta1 = np.abs(x), np.angle(x)
    r2, theta2 = np.abs(y), np.angle(y)
    dtheta = theta2 - theta1
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi  # Adjust for wrapping
    theta = theta1 + a * dtheta
    if geometric_modulus:
        r = r1**(1 - a) * r2**a
    else:
        r = (1 - a) * r1 + a * r2
    return r * (np.cos(theta) + 1j * np.sin(theta))

def normalize(x):
    return x/np.max(np.abs(x))

def chess2poly(t1,t2):
    andy = ps.poly.get("andy") or 0.0
    n=2
    p = int(andy*n)
    a = (andy*n)%1
    if p==0:
        uct1, uct2 = xfrm.uc(t1,t2)
        cf1 = zfrm.rev(poly_giga_5(uct1,uct2))
    elif p==1:
        a=1-a
        uct1, uct2 = xfrm.uc(t1,t2)
        cf1 = np.poly(np.roots(poly_giga_1(uct1,uct2))*2.5)
    rts1 = np.roots(cf1)
    srts1 = rts1[np.argsort(np.abs(rts1))]
    rts2 = pc.random_chess1(len(rts1),t1,t2)
    srts2 = rts2[np.argsort(np.abs(rts2))]
    rts = polar_interpolation(srts1, srts2, a,geometric_modulus=True)
    # rts = srts1 * (1-andy) + srts2 * andy
    cf = rts
    return np.array(cf,dtype=complex)

def chess2poly1(t1,t2):
    andy = ps.poly.get("andy") or 0.0
    uct1, uct2 = xfrm.uc(t1,t2)
    cf1 = normalize(np.poly(np.roots(poly_giga_1(uct1,uct2))*2.5))
    rts2 = pc.random_chess1(len(cf1)-1,t1,t2)
    cf2 = normalize(np.poly(rts2))
    cf = polar_interpolation(cf1, cf2, andy,geometric_modulus=True)
    return np.array(cf,dtype=complex)

def chess2poly2(t1,t2):
    andy = ps.poly.get("andy") or 0.0
    uct1, uct2 = xfrm.uc(t1,t2)
    cf1 = normalize(np.poly(np.roots(poly_giga_1(uct1,uct2))*2.5))
    rts2 = pc.random_chess1(len(cf1)-1,t1,t2)
    cf2 = normalize(np.poly(rts2))
    a = andy*(t1+t2)/2
    cf = polar_interpolation(cf1, cf2, a ,geometric_modulus=True)
    return np.array(cf,dtype=complex)


def poly_chess1(t1,t2):
    N=7
    def p2(cf):
        n = len(cf)
        cf1 = np.full(n, 1.0, dtype=complex)
        cf2 = ( cf**2 + cf + cf1 ) 
        return cf2.astype(np.complex128)
    w = 2*np.pi
    t = np.random.uniform(-w, w)
    x =  np.sin(t) + np.tile(np.arange(1, N+1), N)
    y =  np.cos(t) + np.repeat(np.arange(1, N+1), N)
    curve = np.array(x + 1j*y,dtype=complex) - ((N+1)/2) - 1j * ((N+1)/2)
    coeffs = np.poly(curve+0.1*np.cos(np.random.uniform(-w, w)))
    cf1 = coeffs
    cf2 = np.pad(curve,(0,1),constant_values=10j)
    cf3 = (cf1+cf2*0.0001)
    cf = cf3 + 0.00000000000000000000001*p2(cf3)
    return np.array(cf,dtype=complex)

def poly_chess2(
    t1,t2,
    N: int = 8
) -> np.ndarray:
    indices = np.arange(N) - (N - 1) / 2
    parity = (np.indices((N, N)).sum(axis=0)) % 2
    X, Y = np.meshgrid(indices, indices)
    t1 = 0.5 * np.exp(1j*2*np.pi*np.random.rand())* parity
    t2 = 0.5 *(np.random.rand()-0.5)* parity
    cf1 = np.poly(( (X + t1) + 1j * (Y + t1)).flatten())   
    cf2 = np.poly(( (X + t2) + 1j * (Y + t2)).flatten())   
    a = np.random.rand()
    coeffs = cf2 * a + cf1 * (1-a)
    return coeffs.astype(complex)

def poly_chess3(
        t1,t2,
        N: int = 8
    ) -> np.ndarray:
    indices = np.arange(N) - (N - 1) / 2
    parity = (np.indices((N, N)).sum(axis=0)) % 2
    X, Y = np.meshgrid(indices, indices)
    t0 = np.random.rand()
    t1 = 0.5 * np.exp(1j*2*np.pi*t0)* parity
    t2 = 0.5 *(t0-0.5)* parity
    cf1 = np.poly(( (X + t1) + 1j * (Y + t1)).flatten())   
    cf2 = np.poly(( (X + t2) + 1j * (Y + t2)).flatten())   
    a = bimodal_skewed(0.85)
    coeffs = cf2 * a + cf1 * (1-a)
    return coeffs.astype(complex)

def spindle(t, a=0.5, b=0.2, p=1.5):
    theta = 2 * np.pi * t
    x = a * np.sign(np.cos(theta)) * np.abs(np.cos(theta))**(2/p)
    y = b * np.sign(np.sin(theta)) * np.abs(np.sin(theta))**(2/p)
    return x + 1j * y

def cardioid(t):
    a = ps.poly.get("crdd") or 0.5
    theta = 2 * np.pi * t
    r = a * (1 + np.cos(theta))
    return r * np.exp(1j * theta)

def heart(u):
    phi = np.pi/2
    t = 2*np.pi*u+phi
    x = 16 * np.sin(t)**3
    y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
    rot = np.exp(-1j * np.pi / 4)  # e^(-iπ/4)
    hrt = x/40 + 1j*y/40 + 0.1j
    return hrt*rot

def limacon(t, a=0.3, b=0.5):
    theta = 2 * np.pi * t
    r = a + b * np.cos(theta)
    return r * np.exp(1j * theta)

def rose_curve(t, a=0.5, k=2):
    theta = 2 * np.pi * t
    r = a * np.cos(k * theta)
    return r * np.exp(1j * theta)

def lissajous(t, A=0.5, B=0.5, a=3, b=2, delta=np.pi/2):
    theta = 2 * np.pi * t
    x = A * np.sin(a * theta + delta)
    y = B * np.sin(b * theta)
    return x + 1j * y

def astroid(t, a=0.5):
    theta = 2 * np.pi * t
    x = a * np.cos(theta)**3
    y = a * np.sin(theta)**3
    return x + 1j * y

def archimedean_spiral(t, a=0.1, b=0.1):
    theta = 2 * np.pi * t
    r = a + b * theta
    return r * np.exp(1j * theta)

def logarithmic_spiral(t, a=0.1, b=0.1):
    theta = 2 * np.pi * t
    r = a * np.exp(b * theta)
    return r * np.exp(1j * theta)

def deltoid(t, R=1.0):
    theta = 2 * np.pi * t
    x = R * (2 * np.cos(theta) + np.cos(2 * theta)) / 3
    y = R * (2 * np.sin(theta) - np.sin(2 * theta)) / 3
    return x + 1j * y

def ipolygon(t, n=3, radius=1.0, offset=0.0):
    n = ps.poly.get("iplgn") or 3
    t = np.atleast_1d(t)
    t_scaled = t.real * n
    edge_idx = np.floor(t_scaled).astype(int)
    frac = t_scaled - edge_idx
    angles = 2 * np.pi * np.arange(n) / n + offset
    vertices = radius * np.exp(1j * angles)
    v0 = vertices[edge_idx % n]
    v1 = vertices[(edge_idx + 1) % n]
    points = (1 - frac) * v0 + frac * v1
    return points[0] if points.size == 1 else points

def opolygon(t, n=3, radius=1.0, offset=0.0):
    n = ps.poly.get("oplgn") or 3
    t = np.atleast_1d(t)
    t_scaled = t.real * n
    edge_idx = np.floor(t_scaled).astype(int)
    frac = t_scaled - edge_idx
    angles = 2 * np.pi * np.arange(n) / n + offset
    vertices = radius * np.exp(1j * angles)
    v0 = vertices[edge_idx % n]
    v1 = vertices[(edge_idx + 1) % n]
    points = (1 - frac) * v0 + frac * v1
    return points[0] if points.size == 1 else points

def inner(t,factor=0.2):
    inner_line =  t - 0.5 
    inner_circle = factor * np.exp( 1j * 2 * np.pi * t )
    return inner_line + inner_circle

def circle(t):
    crc = np.exp( 1j * 2 * np.pi * t )
    return crc

def split_uniform(t):
    u = t
    v = (t*1000)%1
    return u,v

def disk(t):
    u, v = split_uniform(t)
    r = np.sqrt(u)
    theta = 2 * np.pi * v
    return r * np.exp(1j * theta)

def poly_chess4(
    t1,t2
) -> np.ndarray:
    N = ps.poly.get("n") or 8
    a = ps.poly.get("a") or 0.85
    mod = ps.poly.get("mod") or 2
    off = ps.poly.get("off") or 0
    phi= ps.poly.get("phi") or 0.0
    tt = ps.poly.get("tt") or "tt"
    # inner
    ispeed = ps.poly.get("ispeed")
    if ispeed is None: ispeed = 1.0
    irad = ps.poly.get("irad")
    if irad is None: irad = 0.5
    iname = ps.poly.get("iname") or "circle"
    ifun = globals().get(iname)
    # outer
    ospeed = ps.poly.get("ospeed") 
    if ospeed is None: ospeed = 1.0
    orad = ps.poly.get("orad")
    if orad is None: orad = 0.5
    oname = ps.poly.get("oname") or "circle"
    ofun = globals().get(oname)
    #
    indices = np.arange(N) - (N - 1) / 2
    parity = (((np.indices((N, N)).sum(axis=0))+off) % mod != 0 ).astype(int)
    X, Y = np.meshgrid(indices, indices)
    mask = parity.astype(bool)
    if tt=="t1t1":
        tt1=t1
        tt2=t1
    elif tt=="t1t2":
        tt1 = t1
        tt2 = t2
    elif tt=="t1pmt2":
        tt1 = t1+t2
        tt2 = t1-t2    
    else:
        tt1=np.random.rand()
        tt2=tt1
    to = orad * ofun(tt1 * ospeed )
    ti = irad * ifun(tt2 * ispeed + phi ) 
    cfi = np.poly(( (X[mask] + ti) + 1j * (Y[mask] + ti)).flatten())   
    cfo = np.poly(( (X[mask] + to) + 1j * (Y[mask] + to)).flatten())   
    b = bimodal_skewed(a)
    coeffs = cfo * b + cfi * (1-b)
    return coeffs.astype(complex)

rloc1 = """
 STSTSTSTSTSTSTST
 ST            ST
 ST    SS      STTT
 ST    SS      STTT
 ST    SS      STTT
 ST            ST
 STSTSTSTSTSTSTST
"""

rloc13 = """
_


TTTTTTTTTTTTTTTTTTTTT
T                   S
T      S     T      S
T      S     T      S
T      S     T      S
T      S     T      S
T      S     T      S
T      S     T      S
T      S     T      S    
T                   S
SSSSSSSSSSSSSSSSSSSSS
"""

rjail = """
TTTTTTTTTTT
TSSSSSSSSST
TSSS   SSST      S
TS   T   ST
TSSS   SSST
TSSSSSSSSST
TTTTTTTTTTT
"""

rjail1 = """
SSSSSSSSSSS
STTTTTTTTTS
STTT   TTTS       
ST   S   TS      T
STTT   TTTS
STTTTTTTTTS
SSSSSSSSSSS
"""

rjail2 = """
STSTSTSTSTSTST
ST          ST
ST     S    ST       T
ST          ST
STSTSTSTSTSTST
"""

rjail3 = """
TTTTTTTTTTT
TSSSSSSSSST
TSSS   SSST      
TS   S   ST      T
TSSS   SSST
TSSSSSSSSST
TTTTTTTTTTT
"""

rp1 = """
     TSTS
     TSTS
     TSTS
T    TSTS    S
     TSTS
     TSTS
     TSTS
"""

rp2 = """
     TSTSSS
     TSTS
  TTTTSTS
  TTTTSTS                     SSSSS
  TTTTSTS
     TSTS
     TSTSSS
"""

rp3 = """
     TSTSSS
     STSTS
  TTTTSTS
  TTTTSTS         T     SSSSS     T
  TTTTSTS
     STSTS
     TSTSSS
"""

rp4 = """
TTTTTTTSSSSSSS
"""

rloc6 = """
SSSSSS  TTTTTT
"""

rjail6 = """
ST
ST
ST
SST       T
ST
ST
ST
"""

rjail7 = """
TS
TS
TS
TTS       S
TS
TS
TS
"""

rjail5 = """
S    T
S   T
STS  T
TSSSSSTTT              T
STS  T
S   T
S    T
"""

rjail7 = """
T    S
T   S
TST  S
STTTTTSSS              S
TST  S
T   S
T    S
"""

rjail8 = """
SST         TS     
SST         TS
SST         TST
SSTT  TTT   TST
SST         TST
SST         TS
SST         TS
"""

rloc7 = """


 T T T T T T T T T
S S S S S S S S S T 
      S  T
       S  T
         S
"""

rloc8 = """

TT  TT  TT  TT  TT
S S S S S S S S S S 



         SSSSS




 TT  TT  TT  TT  TT
S S S S S S S S S S 




          TTTTT



  TT  TT  TT  TT  TT
S S S S S S S S S S 

"""

rloc9 = """
TTTTTTTT

TT  TT  TT  TT  TT
S S S S S S S S S S 


SS   SS   SS   SS


 TT  TT  TT  TT  TT
S S S S S S S S S S 
  TT  TT  TT  TT  TT
S S S S S S S S S S 

"""

rloc10 = """

T T T T T T
 S S S S S S
T T T T T T
 S S S S S S
T T T T T T
 S S S S S S

"""

rloc11 = """
         ST
       STSTST         
       TSTSTS
       TSTSTS
       TSTSTS
       TSTSTS
      TSTSTSTS  
     TSTSTSSTST
   STSTSTTSSTSTST
  STTSSTTSTSSTSTST  
 STTSST      STTSST  
"""

rloc_zigzag = """
  TSS
  T S                                                    S
  T S                                                    S
  T S
    T S
      T S S
 S       T S T                           T      S                 T
      T S
    T S                                                  T
  T S                                                    T
 TTS
"""

from . import polylayout as pl
sX, sY, tX, tY = pl.layout2coord(rloc6)
shape_fun = circle

def poly_chess5_old(t1,t2) -> np.ndarray:
    global sX, sY, tX, tY
    phi = ps.poly.get("phi") or 0.0
    rho = ps.poly.get("rho") or 0.33
    speed = ps.poly.get("speed") or 1.0
    a = ps.poly.get("a") or 0.75
    i = ps.poly.get("i") or 0
    if i==0:
        rloc = ps.poly.get("rloc") or "rloc6"
        x=globals().get(rloc)
        sX, sY, tX, tY = pl.layout2coord(x)
    t = np.random.rand()
    t1 = rho * circle(t)
    t2 = rho * circle(t*speed+phi)
    srts = (sX + t1) + 1j * (sY + t1)   
    trts = (tX + t2) + 1j * (tY + t2)
    scfs = np.poly(srts)
    tcfs = np.poly(trts)
    a = bimodal_skewed(a)
    coeffs = tcfs * a + scfs * (1-a)
    return coeffs.astype(complex)

def poly_chess5(t1,t2) -> np.ndarray:
    global sX, sY, tX, tY, shape_fun
    phi = ps.poly.get("phi") or 0.0
    rho = ps.poly.get("rho") or 0.33
    speed = ps.poly.get("speed") or 1.0
    a = ps.poly.get("a") or 0.75
    i = ps.poly.get("i") or 0
    if i==0:
        shape_name = ps.poly.get("shape_name") or "circle"
        shape_fun = globals().get(shape_name)
        rloc = ps.poly.get("rloc") or "rloc6"
        x=globals().get(rloc)
        sX, sY, tX, tY = pl.layout2coord(x)
    t = np.random.rand()
    t1 = rho * shape_fun(t)
    t2 = rho * shape_fun(t*speed+phi)
    srts = (sX + t1) + 1j * (sY + t1)   
    trts = (tX + t2) + 1j * (tY + t2)
    scfs = np.poly(srts)
    tcfs = np.poly(trts)
    a = bimodal_skewed(a)
    coeffs = tcfs * a + scfs * (1-a)
    return coeffs.astype(complex)


def poly_path(t1,t2) -> np.ndarray:
    global sX, sY, tX, tY, shape_fun
    phi = ps.poly.get("phi") or 0.0
    rho = ps.poly.get("rho") or 0.33
    speed = ps.poly.get("speed") or 1.0
    loc = ps.poly.get("loc") or 0.0
    a = ps.poly.get("a") or 0.75
    i = ps.poly.get("i") or 0
    if i==0:
        shape_name = ps.poly.get("shape") or "circle"
        shape_fun = globals().get(shape_name)
        layout = ps.poly.get("layout") or "rloc6"
        x=globals().get(layout)
        sX, sY, tX, tY = pl.layout2coord(x)
    #t = np.random.rand()
    t = t1.real
    t1 = rho * shape_fun(t)
    t2 = rho * shape_fun(t*speed+phi)
    srts = (sX + t1) + 1j * (sY + t1)   
    trts = (tX + t2) + 1j * (tY + t2)
    prts = srts * (1-loc) + trts * loc
    scfs = np.poly(srts)
    pcfs = np.poly(prts)
    a = bimodal_skewed(a)
    coeffs = pcfs * a + scfs * (1-a)
    return coeffs.astype(complex)


def poly_o3_1(
        t1,t2,
        N: int = 7,
        phase: float = 0.1, 
        amplitude: float = 10, 
        freq: float = 5.0, 
        twist: float = 5.0,
        seed: int = None
    ) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    SMALL_FACTOR = 1e-50
    CURVE_SCALE  = 1e-10
    PADDING_VALUE = 10j
    w = 2 * np.pi
    grid = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(grid, grid)
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    theta_mod = theta + twist * np.log(r + 1e-6)
    X_mod = r * np.cos(theta_mod)
    Y_mod = r * np.sin(theta_mod)
    x = np.sin(phase + freq * X_mod) + X_mod
    y = np.cos(phase + freq * Y_mod) + Y_mod
    curve = (x + 1j * y).flatten()
    perturbation = 0.1 * np.cos(np.random.uniform(-w, w)) * amplitude
    roots = curve + perturbation
    base_coeffs = np.poly(roots)
    padded_curve = np.pad(curve, (0, 1), constant_values=PADDING_VALUE)
    adjusted_coeffs = base_coeffs + CURVE_SCALE * padded_curve
    def transform_coeffs(cf: np.ndarray) -> np.ndarray:
        return cf.astype(np.complex128)**2 + cf + np.ones_like(cf, dtype=complex) + 0.2 * cf**3
    final_coeffs = adjusted_coeffs + SMALL_FACTOR * transform_coeffs(adjusted_coeffs)
    
    return final_coeffs.astype(complex)

def poly_pacman(t1,t2):
    N = ps.poly.get("n") or 11
    real_coeffs =np.random.choice([-5, 1], size=N+1)  # Random real part
    imag_coeffs = np.random.choice([-1, 5], size=N+1) # Random imaginary part
    coeffs = real_coeffs + 1j * imag_coeffs  # Combine into complex coefficients
    coeffs = np.cumsum(coeffs)*np.exp(1j*2*np.pi*np.random.rand())
    coeffs = coeffs[np.argsort(np.abs(coeffs))]
    coeffs = coeffs * np.flip(np.arange(1,N+2))
    rts = np.roots(coeffs)
    fac = np.arange(1,len(rts)+2)
    addon = 2+5*np.flip(np.exp(1j*2*t1*np.pi*fac))
    #padded_addon  = np.pad(addon,(0,len(rts)),constant_values=1)
    combined = np.concatenate((rts,addon))
    cf = np.poly(combined)
    return cf.astype(np.complex128)



cf_start =  np.zeros(10, dtype=complex)
cf_end =  np.zeros(10, dtype=complex)

def poly_rnd_path1(t1,t2) -> np.ndarray:
    global cf_start, cf_end
    i = ps.poly.get("i") or 0
    if i==0:
        a = ps.poly.get("a") or 1
        b = ps.poly.get("b") or 0
        cf_start = ps.json2cvec(ps.poly["cf_start"])
        cf_end = np.poly(np.roots(ps.json2cvec(ps.poly["cf_end"])) * a+b)
    real_part = (1.0 - t1) * cf_start.real + t1 * cf_end.real
    imag_part = (1.0 - t2) * cf_start.imag + t2 * cf_end.imag
    coeffs = real_part + 1j * imag_part
    return coeffs.astype(complex)


def poly_rnd_path2(t1,t2) -> np.ndarray:
    global cf_start, cf_end
    i = ps.poly.get("i") or 0
    if i==0:
        a = ps.poly.get("a") or 1
        b = ps.poly.get("b") or 0
        cf_start = ps.json2cvec(ps.poly["cf_start"])
        cf_end = np.poly(np.roots(ps.json2cvec(ps.poly["cf_end"])) * a+b)
    t = np.random.uniform(-1, 1)
    real_part = (1.0 - t1 * t ) * cf_start.real + t1 * t * cf_end.real   
    imag_part = (1.0 - t2 * t ) * cf_start.imag + t2 * t * cf_end.imag 
    coeffs = real_part + 1j * imag_part
    return coeffs.astype(complex)

def poly_rnd_path3(t1,t2) -> np.ndarray:
    global cf_start, cf_end
    i = ps.poly.get("i") or 0
    if i==0:
        a = ps.poly.get("a") or 1
        b = ps.poly.get("b") or 0
        cf_start = ps.json2cvec(ps.poly["cf_start"])
        cf_end = np.poly(np.roots(ps.json2cvec(ps.poly["cf_end"])) * a+b)
    ta = np.random.uniform(-1, 1)
    tb = np.random.uniform(-1, 1)
    real_part = (1.0 - t1 * ta ) * cf_start.real + t1 * ta * cf_end.real   
    imag_part = (1.0 - t2 * tb ) * cf_start.imag + t2 * tb * cf_end.imag 
    coeffs = real_part + 1j * imag_part
    return coeffs.astype(complex)

def poly_rnd_path4(t1,t2) -> np.ndarray:
    global cf_start, cf_end
    i = ps.poly.get("i") or 0
    if i==0:
        a = ps.poly.get("a") or 1
        b = ps.poly.get("b") or 0
        cf_start = ps.json2cvec(ps.poly["cf_start"])
        cf_end = np.poly(np.roots(ps.json2cvec(ps.poly["cf_end"])) * a+b)
    ta = np.random.uniform(0, 1)
    tb = np.random.uniform(0, 1)
    real_part = (1.0 - t1 * ta ) * cf_start.real + t1 * ta * cf_end.real   
    imag_part = (1.0 - t2 * tb ) * cf_start.imag + t2 * tb * cf_end.imag 
    coeffs = real_part + 1j * imag_part
    return coeffs.astype(complex)

c = 1
def poly_rnd_path5(t1,t2) -> np.ndarray:
    global cf_start, cf_end, c
    i = ps.poly.get("i") or 0
    if i==0:
        a = ps.poly.get("a") or 1
        b = ps.poly.get("b") or 0
        c = ps.poly.get("c") or 1
        cf_start = ps.json2cvec(ps.poly["cf_start"])
        cf_end = np.poly(np.roots(ps.json2cvec(ps.poly["cf_end"])) * a+b)
    ta = np.random.uniform(-1, 1) * c
    tb = np.random.uniform(-1, 1) * c
    real_part = (1.0 - (t1 * ta) ) * cf_start.real + (t1 * ta) * cf_end.real   
    imag_part = (1.0 - (t2 * tb) ) * cf_start.imag + (t2 * tb) * cf_end.imag 
    coeffs = real_part + 1j * imag_part
    return coeffs.astype(complex)

def poly_rnd_path6(t1,t2) -> np.ndarray:
    global cf_start, cf_end, c
    i = ps.poly.get("i") or 0
    if i==0:
        a = ps.poly.get("a") or 1
        b = ps.poly.get("b") or 0
        c = ps.poly.get("c") or 1
        cf_start = ps.json2cvec(ps.poly["cf_start"])
        cf_end = np.poly(np.roots(ps.json2cvec(ps.poly["cf_end"])) * a+b)
    ta = np.random.uniform(-1, 1)
    tb = np.random.uniform(-1, 1)
    real_part = (1.0 - (t1) ) * cf_start.real + (t1) * cf_end.real
    imag_part = (1.0 - (t2) ) * cf_start.imag + (t2) * cf_end.imag
    coeffs = (real_part + 1j * imag_part)
    return coeffs.astype(complex)

def poly_rnd_path7(t1,t2) -> np.ndarray:
    global cf_start, cf_end, c
    i = ps.poly.get("i") or 0
    if i==0:
        n = ps.poly.get("n") or 10
        a1 = ps.poly.get("a1") or 1
        a2 = ps.poly.get("a2") or 1
        b1 = ps.poly.get("b1") or 0
        b2 = ps.poly.get("b2") or 0
        cf_start = np.poly(ps.random_coeff(n) * a1 + b1)
        cf_end = np.poly(ps.random_coeff(n) * a2 + b2)
    real_part = (1.0 - (t1) ) * cf_start.real + (t1) * cf_end.real
    imag_part = (1.0 - (t2) ) * cf_start.imag + (t2) * cf_end.imag
    coeffs  = (real_part + 1j * imag_part)
    return coeffs.astype(complex)

