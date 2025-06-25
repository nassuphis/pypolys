################################################
# zfrm functions
################################################
from . import polystate as ps
import numpy as np
from . import solve

def rot45(cf):
  rts = np.roots(cf)
  rot = np.exp(-1j * np.pi / 4)  # e^(-iπ/4)
  rrt = rts * rot
  rcf = np.poly(rrt)
  return rcf

def skewsweep(u,a):
  skewed = np.where(
        u < 0.5, 
        (2*u)**(1/(1-a)) / 2, 
        1 - (2*(1-u))**(1/(1-a))/2
    )
  return np.clip(skewed, 0, 1)

def none(cf):
  return cf

def reverse(cf):
  return np.flip(cf)

def reverse_p(cf):
  andy = ps.poly.get("andy") or 0.0
  return np.flip(cf) * andy + cf

def rep2(cf):
  cf1 = np.concatenate([cf,cf])
  return cf1

def times2(cf):
  n = len(cf)
  cf1 = np.concatenate([cf,cf * 2])
  return cf1

def times3(cf):
  n = len(cf)
  cf1 = np.concatenate([
    cf,
    cf * 2,
    cf * 3
  ])
  return cf1

def times4(cf):
  n = len(cf)
  cf1 = np.concatenate([
    cf,
    cf * 2,
    cf * 3,
    cf * 4
  ])
  return cf1

def times5(cf):
  n = len(cf)
  cf1 = np.concatenate([
    cf,
    cf * 2,
    cf * 3,
    cf * 4,
    cf * 5
  ])
  return cf1

def p1(cf):
  n = len(cf)
  cf1 = np.full(n, 1.0, dtype=complex)
  cf2 = cf + cf1
  return cf2

def p1_p(cf):
   andy = ps.poly.get("andy") or 0.0
   return p1(cf) * andy + cf

def p2(cf):
  n = len(cf)
  cf0 = np.arange(1,n+1, dtype=complex)
  cf1 = np.full(n, 1.0, dtype=complex)
  cf2 = ( cf**2 + cf + cf1 ) * cf0
  return cf2

def p2_p(cf):
   andy = ps.poly.get("andy") or 0.0
   return p2(cf) * andy + cf

def p3(cf):
  n = len(cf)
  cf0 = np.arange(1,n+1, dtype=complex)
  cf1 = np.full(n, 1.0, dtype=complex)
  cf2 = (cf**3 + cf**2 + cf + cf1) * cf0
  return cf2

def p3_p(cf):
  andy = ps.poly.get("andy") or 0.0
  return p3(cf) * andy + cf

def p4(cf):
  n = len(cf)
  cf0 = np.arange(1,n+1, dtype=complex)
  cf1 = np.full(n, 1.0, dtype=complex)
  cf2 = (cf**4 + cf**3 + cf**2 + cf + cf1) * cf0
  return cf2

def p4_p(cf):
  andy = ps.poly.get("andy") or 0.0
  return p4(cf) * andy + cf

def p5(cf):
  n = len(cf)
  cf0 = np.arange(1,n+1, dtype=complex)
  cf1 = np.full(n, 1.0, dtype=complex)
  cf2 = (cf**5 + cf**4 + cf**3 + cf**2 + cf + cf1) * cf0
  return cf2

def p5_p(cf):
  andy = ps.poly.get("andy") or 0.0
  return p5(cf) * andy + cf

def p6(cf):
  n = len(cf)
  cf0 = np.arange(1,n+1, dtype=complex)
  cf1 = np.full(n, 1.0, dtype=complex)
  cf2 = (cf**6 + cf**5 + cf**4 + cf**3 + cf**2 + cf + cf1) * cf0
  return cf2

def p6_p(cf):
  andy = ps.poly.get("andy") or 0.0
  return p6(cf) * andy + cf

def p7(cf):
  n = len(cf)
  cf0 = np.arange(1,n+1, dtype=complex)
  cf1 = np.full(n, 1.0, dtype=complex)
  cf2 = (cf**7 + cf**6 + cf**5 + cf**4 + cf**3 + cf**2 + cf + cf1) * cf0
  return cf2

def p7_p(cf):
  andy = ps.poly.get("andy") or 0.0
  return p7(cf) * andy + cf

def p8(cf):
  n = len(cf)
  cf0 = np.arange(1,n+1, dtype=complex)
  cf1 = np.full(n, 1.0, dtype=complex)
  cf2 = (cf**8 + cf**7 + cf**6 + cf**5 + cf**4 + cf**3 + cf**2 + cf + cf1) * cf0
  return cf2

def p8_p(cf):
  andy = ps.poly.get("andy") or 0.0
  return p8(cf) * andy + cf

def inv(cf):
  n = len(cf)
  num = np.full(n, 1.0, dtype=complex)
  den = cf
  cf2 = np.where(np.abs(den)>1,num/den,num)
  return cf2

def inv_p(cf):
  andy = ps.poly.get("andy") or 0.0
  return inv(cf) * andy + cf

def inv2(cf):
  n = len(cf)
  icf = np.full(n, 1.0, dtype=complex)
  cf1 = cf**2
  cf2 = np.where(np.abs(cf1)>1,icf/cf1,icf)
  return cf2

def inv2_p(cf):
  andy = ps.poly.get("andy") or 0.0
  return inv2(cf) * andy + cf

def invp1(cf):
  n = len(cf)
  icf = np.full(n, 1.0, dtype=complex)
  cf1 = cf+1
  cf2 = np.where(np.abs(cf1)>1,icf/cf1,icf)
  return cf2

def invp1_p(cf):
  andy = ps.poly.get("andy") or 0.0
  return invp1(cf) * andy + cf

def invp2(cf):
  n = len(cf)
  icf = np.full(n, 1.0, dtype=complex)
  cf1 = cf**2+cf+1
  cf2 = np.where(np.abs(cf1)>1,icf/cf1,icf)
  return cf2

def invp2_p(cf):
  andy = ps.poly.get("andy") or 0.0
  return invp2(cf) * andy + cf

def invp3(cf):
  n = len(cf)
  icf = np.full(n, 1.0, dtype=complex)
  cf1 = cf**3+cf**2+cf+1
  cf2 = np.where(np.abs(cf1)>1,icf/cf1,icf)
  return cf2

def invp3_p(cf):
  andy = ps.poly.get("andy") or 0.0
  return invp3(cf) * andy + cf

def invp4(cf):
  n = len(cf)
  icf = np.full(n, 1.0, dtype=complex)
  cf1 = cf**4+cf**3+cf**2+cf+1
  cf2 = np.where(np.abs(cf1)>1,icf/cf1,icf)
  return cf1

def invp4_p(cf):
  andy = ps.poly.get("andy") or 0.0
  return invp4(cf) * andy + cf

def invp5(cf):
  n = len(cf)
  icf = np.full(n, 1.0, dtype=complex)
  cf1 = cf**5+cf**4+cf**3+cf**2+cf+1
  cf2 = np.where(np.abs(cf1)>1,icf/cf1,icf)
  return cf1

def invp5_p(cf):
  andy = ps.poly.get("andy") or 0.0
  return invp5(cf) * andy + cf


def invp6(cf):
  n = len(cf)
  icf = np.full(n, 1.0, dtype=complex)
  cf1 = cf**6+cf**5+cf**4+cf**3+cf**2+cf+1
  cf2 = np.where(np.abs(cf1)>1,icf/cf1,icf)
  return cf1

def invp6_p(cf):
  andy = ps.poly.get("andy") or 0.0
  return invp6(cf) * andy + cf

def double(cf):
  return np.concatenate(cf,cf)

def doublerev(cf):
  cf1 = np.concatenate((cf,np.flip(cf)))
  return cf1

def doublerev1(cf):
  cf1 = np.concatenate((cf,np.flip(cf)))
  cf2 = np.arange(1,2*len(cf)+1)
  return cf1*cf2

def doublerev2(cf):
  cf1 = np.concatenate((cf,np.flip(cf)))
  cf2 = np.arange(1,2*len(cf)+1)
  cf3 = ( cf2 **2 + cf2 ) % 11
  return cf1*cf3

def doublerev3(cf):
  cf1 = np.concatenate((cf,np.flip(cf)))
  cf2 = np.arange(1,2*len(cf)+1)
  cf3 = ( cf2 **3 + cf2 **2 + cf2 ) % 101
  return cf1*cf3

def doublerev4(cf):
  cf1 = np.concatenate((cf,np.flip(cf)))
  cf2 = np.arange(1,2*len(cf)+1)
  cf3 = ( cf2 **4 + cf2 **3 + cf2 **2 + cf2 ) % 1013
  return cf1*cf3

def doublerev5(cf):
  cf1 = np.concatenate((cf,np.flip(cf)))
  cf2 = np.arange(1,2*len(cf)+1)
  cf3 = ( cf2 **5 + cf2 **4 + cf2 **3 + cf2 **2 + cf2 ) % 7127
  return cf1*cf3

def rev(cf):
  return np.flip(cf)

def rev_p(cf):
  andy = ps.poly.get("andy") or 0.0
  return np.flip(cf) * andy + cf

def revuh(cf):
    n = len(cf)
    mid = n // 2
    if n % 2 == 0:
        lower = cf[:mid]
        upper = cf[mid:]
        upper_reversed = upper[::-1]
        return np.concatenate([lower, upper_reversed])
    else:
        lower = cf[:mid]
        middle = cf[mid]
        upper = cf[mid+1:]
        upper_reversed = upper[::-1]
        return np.concatenate([lower, [middle], upper_reversed])
    
def revlh(cf):
    n = len(cf)
    mid = n // 2
    if n % 2 == 0:
      upper = cf[:mid]
      lower = cf[mid:]
      lower_reversed = np.flip(lower)
      return np.concatenate([upper, lower_reversed])
    else:
      upper = cf[:mid]
      middle = cf[mid]
      lower = cf[mid+1:]
      lower_reversed = np.flip(lower)
      return np.concatenate([upper, [middle], lower_reversed])
    
def symmetrize(cf):
  return cf+np.flip(cf)

def symmetrize_p(cf):
  andy = ps.poly.get("andy") or 0.0
  return symmetrize(cf) * andy + cf

def conjugate(cf):
  return np.conjugate(cf)

def re2im(cf):
  return cf.imag + 1j * cf.real

def conj(cf):
    return np.conjugate(cf)

def swap_angle_modulus(cf):
   angle = np.angle(cf)
   modulus = np.abs(cf)
   return angle * np.exp(1j * modulus)

def swap_angle_modulus_p(cf):
  andy = ps.poly.get("andy") or 0.0
  return swap_angle_modulus(cf) * andy + cf

def sort_abs(cf):
  return cf[np.argsort(np.abs(cf))]

def sort_abs_p(cf):
  andy = ps.poly.get("andy") or 0.0
  return sort_abs(cf) * andy + cf

def sort_abs_q(cf):
  andy = ps.poly.get("andy") or 0.0
  skew = ps.poly.get("skew") or 0.0
  q = skewsweep(andy,skew)
  return sort_abs(cf) * q + cf * (1-q)

def sort_angle(cf):
  return cf[np.argsort(np.angle(cf))]

def sort_angle_p(cf):
  andy = ps.poly.get("andy") or 0.0
  return sort_angle(cf) * andy + cf

def sort_real(cf):
  return cf[np.argsort(cf.real)]

def sort_real_p(cf):
  andy = ps.poly.get("andy") or 0.0
  return sort_real(cf) * andy + cf

def sort_imag(cf):
  return cf[np.argsort(cf.imag)]

def sort_imag_p(cf):
  andy = ps.poly.get("andy") or 0.0
  return sort_imag(cf) * andy + cf

def sort_diff(cf):
  return cf[np.argsort(cf.real-cf.imag)]

def sort_diff_p(cf):
  andy = ps.poly.get("andy") or 0.0
  return sort_diff(cf) * andy + cf

def sort_abs_diff(cf):
  return cf[np.argsort(np.abs(cf.real-cf.imag))]

def sort_abs_diff_p(cf):
  andy = ps.poly.get("andy") or 0.0
  return sort_abs_diff(cf) * andy + cf

def sort_angles_keep_moduli(complex_vector):
    angles = np.angle(complex_vector)
    moduli = np.abs(complex_vector)
    sorted_angles = np.sort(angles)
    return moduli * np.exp(1j * sorted_angles)

def sort_angles_keep_moduli_p(cf):
   andy = ps.poly.get("andy") or 0.0
   return sort_angles_keep_moduli(cf) * andy + cf

def sort_moduli_keep_angles(complex_vector):
    angles = np.angle(complex_vector)
    sorted_moduli = np.sort(np.abs(complex_vector))
    return sorted_moduli * np.exp(1j * angles)

def sort_moduli_keep_angles_p(cf):
   andy = ps.poly.get("andy") or 0.0
   return sort_moduli_keep_angles(cf) * andy + cf

def roll_r2(cf):
  return np.roll(cf,2)

def roll_r2_p(cf):
  andy = ps.poly.get("andy") or 0.0
  return roll_r2(cf) * andy + cf

def roll_r3(cf):
  return np.roll(cf,3)

def roll_r4(cf):
  return np.roll(cf,4)

def roll_r5(cf):
  return np.roll(cf,5)

def roll_r6(cf):
  return np.roll(cf,6)

def roll_r7(cf):
  return np.roll(cf,7)

def roll_r8(cf):
  return np.roll(cf,8)

def roll_r9(cf):
  return np.roll(cf,9)

def roll_r10(cf):
  return np.roll(cf,10)

def roll_l2(cf):
  return np.roll(cf,-2)

def roll_l3(cf):
  return np.roll(cf,-3)

def roll_l4(cf):
  return np.roll(cf,-4)

def roll_l5(cf):
  return np.roll(cf,-5)

def roll_l6(cf):
  return np.roll(cf,-6)

def roll_l7(cf):
  return np.roll(cf,-7)

def roll_l8(cf):
  return np.roll(cf,-8)

def roll_l9(cf):
  return np.roll(cf,-9)

def roll_l10(cf):
  return np.roll(cf,-10)

def swap_halves_inc(cf):    
    n = len(cf)
    mid = n // 2
    if n % 2 == 0:
        first_half = cf[:mid]
        second_half = cf[mid:]
        middle = None
    else:
        first_half = cf[:mid]
        second_half = cf[mid+1:]
        middle = cf[mid]
    sum_first = np.sum(np.abs(first_half))
    sum_second = np.sum(np.abs(second_half))
    if sum_first < sum_second:
        first_half, second_half = second_half, first_half
    if middle is not None:
        permuted_cf = np.concatenate([first_half, [middle], second_half])
    else:
        permuted_cf = np.concatenate([first_half, second_half])
    return permuted_cf

def sort_angle_desc(cf):
    return cf[np.argsort(-np.angle(cf))]

def sort_moduli_desc(cf):
    return cf[np.argsort(-np.abs(cf))]

def sort_real_desc(cf):
    return cf[np.argsort(-cf.real)]

def sort_imag_desc(cf):
    return cf[np.argsort(-cf.imag)]

def sort_real_then_imag(cf):
    indices = np.lexsort((cf.imag, cf.real))
    return cf[indices]

def sort_imag_then_real(cf):
    indices = np.lexsort((cf.real, cf.imag))
    return cf[indices]

def sort_conjugate(cf):
    return np.conj(cf)[np.argsort(np.angle(cf))]

def sort_real_modulus(cf):
    indices = np.argsort(cf.real * np.abs(cf))
    return cf[indices]

def sort_imag_modulus(cf):
    indices = np.argsort(cf.imag * np.abs(cf))
    return cf[indices]

def sort_real_over_imag(cf):
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(cf.imag != 0, cf.real / cf.imag, np.inf)
    return cf[np.argsort(ratio)]

def wave_permutation_full(cf):
    sorted_cf = np.sort(cf, key=lambda x: np.angle(x))
    return np.concatenate([sorted_cf[::2], sorted_cf[1::2]])

def cumsum(cf):
    return np.cumsum(cf)

def cumsum_p(cf):
    andy = ps.poly.get("andy") or 0.0
    return np.cumsum(cf) * andy + cf

def sort_cumsum(cf):
    return cf[np.argsort(np.abs(np.cumsum(cf)))]

def sort_cumsum_p(cf):
    andy = ps.poly.get("andy") or 0.0
    return sort_cumsum(cf) * andy + cf

def sort_abs_p_cumsum(cf):
  andy = ps.poly.get("andy") or 1.0
  cf1 = cf[np.argsort(np.abs(np.cumsum(cf)))]
  cf2 = cf[np.argsort(np.abs(cf))]
  return cf1+andy*cf2

def sort_angle_cumsum(cf):
  return cf[np.argsort(np.angle(np.cumsum(cf)))]

def sort_angle_cumsum_p(cf):
  andy = ps.poly.get("andy") or 0.0
  return sort_angle_cumsum(cf) * andy + cf

def cumprod(cf):
  return np.cumprod(cf)

def cumprod_p(cf):
  andy = ps.poly.get("andy") or 0.0
  return cumprod(cf) * andy + cf

def sort_cumprod(cf):
  return cf[np.argsort(np.abs(np.cumprod(cf)))]

def sort_cumprod_p(cf):
  andy = ps.poly.get("andy") or 0.0
  return sort_cumprod(cf) * andy + cf

def sort_angle_cumprod(cf):
  return cf[np.argsort(np.angle(np.cumprod(cf)))]

def sort_angle_cumprod_p(cf):
  andy = ps.poly.get("andy") or 0.0
  return sort_angle_cumprod(cf) * andy + cf

def andy(cf):
  mod_indices = np.cumsum(np.argsort(np.abs(cf))) % len(cf)
  return cf[mod_indices]

def andy1(cf):
  csi = np.cumprod(np.argsort(np.abs(cf))) % len(cf)
  cpi = np.cumsum(np.argsort(np.abs(cf))) % len(cf)
  return cf[csi]-cf[cpi]

def andy1_q(cf):
  andy = ps.poly.get("andy") or 0.0
  skew = ps.poly.get("skew") or 0.0
  q = skewsweep(andy,skew)
  csi = np.cumprod(np.argsort(np.abs(cf))) % len(cf)
  cpi = np.cumsum(np.argsort(np.abs(cf))) % len(cf)
  return (cf[csi]-cf[cpi]) * q + cf * (1-q)

def andy1p(cf):
  andy = ps.poly.get("andy") or 1.0
  csi = np.cumprod(np.argsort(np.abs(cf))) % len(cf)
  cpi = np.cumsum(np.argsort(np.abs(cf))) % len(cf)
  return cf[csi] - andy * cf[cpi]

def andy2(cf):
  csi = np.cumprod(np.argsort(np.abs(cf))) % len(cf)
  cpi = np.cumsum(np.argsort(np.abs(cf))) % len(cf)
  cc = (csi+cpi) % len(cf)
  return cf[cc]

def andy3(cf):
  csi = np.cumsum(np.argsort(np.angle(cf))) % len(cf)
  return cf[csi]

def andy4(cf):
  i1 = np.argsort(np.abs(cf))
  i2 = np.argsort(np.abs(np.cumsum(cf)))
  i = ( i1 - i2 ) % len(cf)
  return cf[i]

def even(cf):
    return cf[::2]

def odd(cf):
    return cf[1::2]

def evenfirst(cf):
    even = cf[::2]
    odd = cf[1::2]
    return np.concatenate([even, odd])

def oddfirst(cf):
    odd = cf[1::2]
    even = cf[::2]
    return np.concatenate([odd, even])

def to_flip_or_not_to_flip(cf):
    if np.mean(abs(cf))>np.median(abs(cf)):
      return cf
    else:
      return np.flip(cf)

def minus_mean(cf):
    return cf - np.mean(np.abs(cf))

def minus_median(cf):
    return cf - np.median(np.abs(cf))

def medianize(cf):
    angles = np.angle(cf)
    moduli = np.abs(cf)
    a = angles-np.median(angles)
    m = moduli-np.median(moduli)
    return m * np.exp(1j * a)
    
def roots(cf):
   if np.sum(np.abs(cf))<1e-10 :
      return cf
   try:
    cf0 = np.roots(cf)
    return cf0
   except:
    return cf

def roots_p(cf):
  andy = ps.poly.get("andy") or 0.0
  return np.append(roots(cf),1) * andy + cf

def roots_q(cf):
  andy = ps.poly.get("andy") or 0.0
  skew = ps.poly.get("skew") or 0.0
  q = skewsweep(andy,skew)
  return np.append(roots(cf),1) * q + cf * (1-q)

def roots10(cf):
   cf0 = cf
   for _ in range(10):
      cf0 = np.roots(cf0)
   return cf0

def roots20(cf):
   cf0 = cf
   for _ in range(20):
      cf0 = np.roots(cf0)
   return cf0

def add1(cf):
   return np.concatenate((cf,[1]))

def add1j(cf):
   return np.concatenate((cf,[1j]))

def prep1(cf):
   return np.concatenate(([1],cf))

def prep1j(cf):
   return np.concatenate(([1j],cf))

def cos(cf):
   return np.cos(cf)

def cos_p(cf):
   andy = ps.poly.get("andy") or 0.0
   return np.cos(cf) * andy + cf

def sin(cf):
   return np.sin(cf)

def uc_old(cf):
   e = ps.poly.get("err") or 15
   thresh = 10**(-e)
   sa = np.sum(np.abs(cf))
   if sa<thresh:
      return(cf)
   cf0 = cf / sa
   return np.exp(1j*2*np.pi*cf0)

def uc(cf):
   sa = np.max(np.abs(cf))
   cf0 = cf / sa
   return np.exp(1j*2*np.pi*cf0)

def uc_p(cf):
  andy = ps.poly.get("andy") or 0.0
  return uc(cf) * andy + cf

def invuc(cf):
   sa = np.max(np.abs(cf))
   cf0 = cf / sa
   cf1 = np.exp(1j*2*np.pi*cf0)
   return cf/cf1 

def invuc_p(cf):
  andy = ps.poly.get("andy") or 0.0
  return invuc(cf) * andy + cf

def uc10(cf):
   sa = np.sum(np.abs(cf))
   if sa<1e-15:
      return(cf)
   cf0 = cf / sa
   return 10*np.exp(1j*2*np.pi*cf0)

def uc100(cf):
   sa = np.sum(np.abs(cf))
   if sa<1e-15:
      return(cf)
   cf0 = cf / sa
   return 100*np.exp(1j*2*np.pi*cf0)

def log(cf):
   sa = np.sum(np.abs(cf))
   if sa<1e-15:
      return(cf)
   cf0 = cf / sa
   cf1 = np.abs(cf)
   return np.log(cf0+cf1)

def log_p(cf):
  andy = ps.poly.get("andy") or 0.0
  return log(cf) * andy + cf

# normalize
def normalize(cf):
   sa = np.max(np.abs(cf))
   return cf / sa

def rescale(cf):
    real_parts = cf.real
    imag_parts = cf.imag
    max_re = np.max(real_parts)
    min_re = np.min(real_parts)
    max_im = np.max(imag_parts)
    min_im = np.min(imag_parts)
    maxv = max(max_re, max_im)
    minv = min(min_re, min_im)
    normalized_real = (real_parts - minv) / (maxv - minv) * 2 - 1
    normalized_imag = (imag_parts - minv) / (maxv - minv) * 2 - 1
    normalized_vector = normalized_real + 1j * normalized_imag
    return normalized_vector

def poly(cf):
  return np.poly(cf).astype(complex)


def ltl_q(cf):
    andy = ps.poly.get("andy") or 0.0
    lcf = np.random.choice([1, 0,1j], size = len(cf)).astype(complex)
    ltl = np.abs(cf)*lcf
    return (1-andy) * cf + andy * ltl 


def toline(cf):
   rts=np.roots(cf)
   num = 1+rts
   den = 1-rts
   cay = 1j * num/den
   cf1=np.poly(cay)
   return cf1

def toline_q(cf):
   andy = ps.poly.get("andy") or 0.0
   rts=np.roots(cf)
   num = 1+rts
   den = 1-rts
   cay = 1j * num/den
   cf1=np.poly(cay)
   cf2 = cf1 * andy + cf * (1.0-andy)
   return cf2

def tohalf(cf):
   theta = ps.poly.get("theta") or 0.0
   rts=np.roots(cf)
   angle1=np.exp(1j*2*np.pi*theta)
   angle2=np.exp(-1j*2*np.pi*theta)
   num = rts-angle1
   den = angle2-rts
   hc = num/den
   cf1=np.poly(hc)
   return cf1

rcf = np.array([],dtype=complex)
def recursive_add(cf):
    global rcf
    i = ps.poly.get("i") or 0
    decay = ps.poly.get("decay") or 0.0
    if i==0:
        rcf = cf
        return cf
    cf1 =  ( cf * (1.0-decay) + rcf * decay )
    rcf = cf1
    return cf1


rcf_vec = []

def recursive_addv(cf):
    global rcf_vec
    i = ps.poly.get("i") or 0
    decay = ps.poly.get("decay") or 0.0
    cf = np.array(cf, dtype=complex)

    # Initialization: create N copies of cf
    if i == 0:
        N = ps.poly.get("bins") or 0
        rcf_vec = [cf.copy() for _ in range(N)]
        return cf

    distances = np.array([np.linalg.norm(cf - stored_cf) for stored_cf in rcf_vec])
    closest_idx = np.argmin(distances)
  
    updated_cf = cf * (1.0 - decay) + sort_abs(rcf_vec[closest_idx])  * decay
    rcf_vec[closest_idx] = updated_cf

    return updated_cf


root_matrix = None
velocity_matrix = None

def recursive_addvr(cf):
    global root_matrix, velocity_matrix

    i = ps.poly.get("i") or 0
    decay = ps.poly.get("decay") or 0.02
    gravity = ps.poly.get("gravity") or 0.001
    damping = ps.poly.get("damping") or 0.9  # to prevent infinite acceleration
    stepsize = ps.poly.get("stepsize") or 0.001
    N = ps.poly.get("bins") or 5

    cf_roots = np.roots(cf)
    degree = len(cf_roots)

    if i < N:
        if root_matrix is None:
            root_matrix = np.zeros((degree, N), dtype=complex)
            velocity_matrix = np.zeros_like(root_matrix)
        root_matrix[:, i] = cf_roots.copy()
        angles = 2 * np.pi * np.random.rand(degree)
        velocity_matrix[:, i] = 10*np.exp(1j * angles)
        return cf

    # Find closest stored root vector and perturb towards current roots
    distances = np.sum(np.abs(root_matrix - cf_roots[:, None]), axis=0)
    closest_idx = np.argmin(distances)
    root_matrix[:, closest_idx] = (
        decay * root_matrix[:, closest_idx] + (1.0-decay) * cf_roots
    )

    # Global gravity calculation
    all_roots = root_matrix.flatten()
    velocities = velocity_matrix.flatten()

    diff_matrix = all_roots[:, None] - all_roots[None, :]
    np.fill_diagonal(diff_matrix, np.inf)
    distances = np.abs(diff_matrix)
    directions = diff_matrix / distances
    inv_square = 1.0 / distances**2
    forces = np.nansum(directions * inv_square, axis=1)
    # Update velocities
    velocities = damping * velocities + gravity * forces / len(all_roots)
    # Update positions (roots)
    all_roots += stepsize*velocities

    # Reshape back into matrices
    root_matrix = all_roots.reshape((degree, N))
    velocity_matrix = velocities.reshape((degree, N))

    updated_cf = root_matrix[:, closest_idx]

    return updated_cf


def recursive_addvr1(cf):
    global root_matrix, velocity_matrix

    i = ps.poly.get("i") or 0
    impulse = ps.poly.get("impulse") or 0.05
    gravity = ps.poly.get("gravity") or 0.0005
    damping = ps.poly.get("damping") or 0.9
    stepsize = ps.poly.get("stepsize") or 0.001
    N = ps.poly.get("bins") or 5

    cf_roots = np.roots(cf)
    degree = len(cf_roots)

    if i < N:
        if root_matrix is None:
            root_matrix = np.zeros((degree, N), dtype=complex)
            velocity_matrix = np.zeros_like(root_matrix)

        root_matrix[:, i] = cf_roots.copy()
        angles = 2 * np.pi * np.random.rand(degree)
        velocity_matrix[:, i] = np.exp(1j * angles)
        return cf

    # Find closest stored root vector
    distances = np.sum(np.abs(root_matrix - cf_roots[:, None]), axis=0)
    closest_idx = np.argmin(distances)

    # Apply impulse by modifying velocity, not position
    direction_to_new_roots = cf_roots - root_matrix[:, closest_idx]
    velocity_matrix[:, closest_idx] += impulse * direction_to_new_roots

    # Flatten arrays for global calculation
    all_roots = root_matrix.flatten()
    velocities = velocity_matrix.flatten()

    # Gravity calculation (vectorized)
    diff_matrix = all_roots[:, None] - all_roots[None, :]
    np.fill_diagonal(diff_matrix, np.inf)
    distances = np.abs(diff_matrix)
    directions = diff_matrix / distances
    inv_square = 1.0 / distances**2
    forces = np.nansum(directions * inv_square, axis=1)

    # Update velocities
    velocities = damping * velocities + gravity * forces / len(all_roots)

    # Update positions (roots)
    all_roots += stepsize*velocities

    # Reshape back into matrices
    root_matrix = all_roots.reshape((degree, N))
    velocity_matrix = velocities.reshape((degree, N))

    updated_cf = np.poly(root_matrix[:, closest_idx])

    return updated_cf



def recursive_add0(cf):
    global rcf
    i = ps.poly.get("i") or 0
    decay = ps.poly.get("decay") or 0.0
    if i==0:
        rcf = cf
        return cf
    cf1 =  ( cf/np.sum(np.abs(cf)) * (1.0-decay) + rcf/np.sum(np.abs(rcf)) * decay )
    cf2 = cf1 / np.abs(cf1) 
    rcf = cf2
    return cf2

def recursive_add1(cf):
    global rcf
    i = ps.poly.get("i") or 0
    decay = ps.poly.get("decay") or 0.0
    if i==0:
        rcf = cf
        return cf
    cf1 =  cf * (1.0 - decay) + rcf * decay
    rcf = cf1
    return cf1

def recursive_add2(cf):
    global rcf
    i = ps.poly.get("i") or 0
    decay = ps.poly.get("decay") or 0.0
    if i==0:
        rcf = cf
        return cf
    cf1 =  np.poly(np.roots(cf) * (1.0-decay) + np.roots(rcf) * decay)
    rcf = cf1
    return cf1

def recursive_add3(cf):
    global rcf
    i = ps.poly.get("i") or 0
    decay = ps.poly.get("decay") or 0.0
    if i==0:
        rcf = cf
        return cf
    rts1 = sort_abs(np.roots(cf))
    rts2 = sort_abs(np.roots(rcf))
    cf1 =  np.poly( rts1 * (1.0-decay) + rts2 * decay)
    rcf = cf1
    return cf1


def root_add(cf):
    if len(cf)<3:
       return cf
    if not np.isfinite(cf).all():
       return np.zeros(len(cf),dtype=complex)
    decay = ps.poly.get("decay") or 0.0
    icf = np.argsort(np.abs(cf[1:]))
    rts0 = np.roots(cf)
    if not np.isfinite(rts0).all():
       return cf
    if len(rts0)!=len(icf):
       return cf
    rts1 = rts0[icf]
    cf0 = np.concatenate([rts1,[0.0]])
    cf1 =  cf * (1.0-decay) + cf0 * decay
    return cf1


def recursive_add_q(cf):
   andy = ps.poly.get("andy") or 0.0
   cf1 = recursive_add(cf)
   cf2 = cf1 * andy + cf * (1.0-andy)
   return cf2






