################################################
# Transform functions
################################################
from . import polystate as ps
import numpy as np

def roots(t1, t2):
  cf = np.array([
        9/64,
        t1,
        t2
  ], dtype=np.complex128)
  rt = np.roots(cf)
  if len(rt)>1:
    return (rt[0],rt[1])
  return(rt[0],rt[0])

def roots3(t1, t2):
  cf = np.array([
        t1+t2,
        1,
        1,
        t1*t2
  ], dtype=np.complex128)
  rt = np.roots(cf)
  srt = rt[np.argsort(np.abs(rt))]
  return(srt[0],srt[-1])

def roots4(t1, t2):
  cf = np.array([
        t1+t2,
        1j*t1,
        1j*t2,
        t1*t2
  ], dtype=np.complex128)
  rt = np.roots(cf)
  srt = rt[np.argsort(np.abs(rt))]
  return(srt[0],srt[-1])

def roots5(t1, t2):
  cf = np.array([
        np.cos(100*t1),
        1j*t1,
        1j*t2,
        np.sin(100*t2)
  ], dtype=np.complex128)
  rt = np.roots(cf)
  srt = rt[np.argsort(np.abs(rt))]
  return(srt[0],srt[-1])

def roots6(t1, t2):
  cf = np.array([
        np.sin(5*t1),
        1j*t1,
        1*((t1-t2)**3+(t1+t2)**2+t1*t2+1),
        1j*t2,
        np.sin(1*t2)
  ], dtype=np.complex128)
  rt = np.roots(cf)
  srt = rt[np.argsort(np.abs(rt))]
  return(srt[0],srt[-1])

def roots7(t1, t2):
  cf = np.array([
        1*np.log(t1+t2+2),
        1,
        1,
        np.exp(2*np.pi*(1j*t2+t1)),
        1,
        1,
        1*(1j*t1+t2+2)
  ], dtype=np.complex128)
  rt = np.roots(cf)
  srt = rt[np.argsort(np.abs(rt))]
  return(srt[0],srt[-1])

def unit_circle(t1, t2):
  return ( 
    np.exp(1j * 2 * np.pi * t1), 
    np.exp(1j * 2 * np.pi * t2) 
  )

def t0_2(t1,t2):
  return(
    t1*2,
    t2*2
  )

def t0_4(t1,t2):
  return(
    t1*4,
    t2*4
  )

def t0_8(t1,t2):
  return(
    t1*8,
    t2*8
  )

def t0_16(t1,t2):
  return(
    t1*16,
    t2*16
  )


def t0_10(t1,t2):
  return(
    t1*10,
    t2*10
  )

def t0_100(t1,t2):
  return(
    t1*100,
    t2*100
  )

def t0_1000(t1,t2):
  return(
    t1*100,
    t2*100
  )

def z01(t1,t2):
  return(
    t1.real + 1j * t2.real,
    t2.real + 1j * t1.real
  )

def x01(t1,t2):
  return(
    t1.real,
    1j * t2.real
  )


def zz(t1,t2):
  return(
    t1.real + 1j * t2.real,
    t1.real + 1j * t2.real
  )

def xim(t1,t2):
  return(
    1j * t1.real,
    1j * t2.real
  )

def t1t1(t1,t2):
  return (t1,t1)

def t1_plus_t2(t1,t2):
  return (t1+t2,t1+t2)

def t1_times_t2(t1,t2):
  return (t1*t2,t1*t2)


def rtheta(t1,t2):
  return(
    t2.real * np.exp(1j*2*np.pi*t1),
    t1.real * np.exp(1j*2*np.pi*t2),
  )

def rd2_theta(t1,t2):
  return(
    0.5*t2.real * np.exp(1j*2*np.pi*t1),
    0.5*t1.real * np.exp(1j*2*np.pi*t2),
  )

def div(t1,t2):
  div = ps.poly.get("div") or 1.0
  return(t1/div,t2/div)

def div2(t1,t2):
  return(t1/2,t2/2)

def mult2(t1,t2):
  return(t1*2,t2*2)

def div2pi(t1,t2):
  return(t1/2/np.pi,t2/2/np.pi)

def jump(t1, t2):
    if "levels" in ps.poly:
      levels = np.array(ps.poly.get("levels"))
    else:
      levels = np.array([0.0])
    jmp = ps.poly.get("jmp") or 0.0
    jt1 = t1 + np.sum(levels<t1)*jmp
    jt2 = t2 + np.sum(levels<t2)*jmp
    return (jt1, jt2)

def div10(t1,t2):
  return(t1/10,t2/10)

# Define the norm function
def norm(t1, t2):
    try:
      at1 = np.abs(t1)
      at2 = np.abs(t2)
      absv = at1 + at2
      if np.isinf(absv):
        return (0.0,0.0)
      if np.isnan(absv):
        return (0.0,0.0)
      if absv < 1e-15:
        return (0.0,0.0)
      else:
        return (t1/absv, t2/absv)
    except:
       return (0.0,0.0)

# Define the ucn function
def ucn(t1, t2):
    tt1, tt2 = uc(t1, t2)
    ttt1, ttt2 = norm(tt1, tt2)
    return unit_circle(ttt1, ttt2)

def identity( t1, t2 ):
  return ( t1, t2 )

def none( t1, t2 ):
  return ( t1, t2 )
  
def sum_prod(t1, t2):
  val1 = t1 + t2
  val2 = t1 * t2
  return ( val1, val2 )
  
def inv_t_plus_2(t1, t2):
  val1 = 1.0 / ( t1 + 2.0 )
  val2 = 1.0 / ( t2 + 2.0 )
  return ( val1, val2 )

def itp2(t1, t2):
  return inv_t_plus_2( t1, t2 )
  
def cos_t1_sin_t2(t1, t2):
  val1 = np.cos( t1 )
  val2 = np.sin( t2 )
  return ( val1, val2 )
  
def ct1st2(t1,t2):
  return cos_t1_sin_t2(t1, t2)

def t1_plus_inv_t2(t1, t2):
  val1 = t1 + ( 1.0 / t2 )
  val2 = t2 + ( 1.0 / t1 )
  return ( val1, val2 )
  
def t1pit2(t1, t2):
  return t1_plus_inv_t2(t1, t2)

def coeff1(t1, t2):
  return(t1,t2)

def coeff2(t1, t2):
   return(t1 + t2,t1 * t2)

def coeff3(t1, t2):
  return(1/(t1 + 2),1/(t2 + 2))

def coeff3a(t1, t2):
  return(1/(t1 + 1),1/(t2 + 1))

def coeff4(t1, t2):
  return( np.cos(t1),np.sin(t2))

def coeff5(t1, t2):
  return( t1 + (1/t2), t2 + (1/t1))

def coeff5a(t1, t2):
  return( t1 + (1/t1), t2 + (1/t2))

#coeff6,"function (t1, t2) (t1^3 + 1i)/(t1^3 - 1i)","function (t1, t2) (t2^3 + 1i)/(t2^3 - 1i)"
def coeff6(t1, t2):
  num1 = t1**3 + 1j
  den1 = t1**3 - 1j
  val1 = num1 / den1
  num2 = t2**3 + 1j
  den2 = t2**3 - 1j
  val2 = num2 / den2
  return ( val1, val2 )
  
def coeff7(t1, t2):
  top1  = t1 + np.sin(t1)
  bot1  = t1 + np.cos(t1)
  val1  = top1 / bot1
  top2  = t2 + np.sin(t2)
  bot2  = t2 + np.cos(t2)
  val2  = top2 / bot2
  return (val1, val2)

def coeff8(t1, t2): 
  top1  = t1 + np.sin(t2)
  bot1  = t2 + np.cos(t1)
  val1  = top1 / bot1
  top2  = t2 + np.sin(t1)
  bot2  = t1 + np.cos(t2)
  val2  = top2 / bot2
  return (val1, val2)

def coeff9(t1, t2):
  top1  = t1*t1 + 1j * t2
  bot1  = t1*t1 - 1j * t2
  val1  = top1 / bot1
  top2  = t2*t2 + 1j * t1
  bot2  = t2*t2 - 1j * t1
  val2  = top2 / bot2
  return (val1, val2)

def coeff10(t1, t2):
  top1  = t1**4 - t2
  bot1  = t1**4 + t2
  val1  = top1 / bot1
  top2  = t2**4 - t1
  bot2  = t2**4 + t1
  val2  = top2 / bot2
  return (val1, val2)

def coeff11(t1, t2):
  val1 = np.log( t1**4 + 2 )
  val2 = np.log( t2**4 + 2 )
  return ( val1, val2 )

def coeff12(t1, t2):
  val1 = 2*t1**4 - 3*t2**3 + 4*t1**2 - 5*t2
  val2 = 2*t2**4 - 3*t1**3 + 4*t2**2 - 5*t1
  return ( val1, val2 )


def xfrm1(t1,t2):
  a = np.abs(t1+t2)
  m = (883 * a) % 41 + 1
  n = (919 * a) % 191 + 1
  mm = m / 41
  nn = n / 191
  return uc(t1*mm,t2*nn)

def xfrm2(t1,t2):
  a = np.abs(t1+t2)
  m = (883 * a) % 41 + 1
  n = (919 * a) % 191 + 1
  mm = m / 41
  nn = n / 191
  return (t1*mm , t2*nn )


def xfrm3(t1,t2):
  a = np.abs(t1*t2)
  m = (883 * a) % 41 + 1
  n = (919 * a) % 191 + 1
  mm = m / 41
  nn = n / 191
  return (t1*mm , t2*nn )


############################################
# 1) Unit circle
# uc = function(t) exp(1i*2*pi*t)
############################################
def uc1(t):
    """
    Returns exp(i * 2*pi * t).
    If t is real or complex, result is on the complex unit circle 
    (assuming t real). For complex t, it's the standard complex exponential.
    """
    return np.exp(1j * 2 * np.pi * t)

def uc(t1, t2):
  return ( uc1(t1), uc1(t2) )

def uc10(t1, t2):
  return ( 10*uc1(t1), 10*uc1(t2) )

def uc100(t1, t2):
  return ( 100*uc1(t1), 100*uc1(t2) )

############################################
# 2) Wobble
# wbl = function(t) (t + sin(t)) / (t + cos(t))
############################################
def wbl1(t):
    """
    Returns (t + sin(t)) / (t + cos(t)).
    t can be real or complex.
    """
    return (t + np.sin(t)) / (t + np.cos(t))

def wbl(t1, t2):
  return ( wbl1(t1), wbl1(t2) )

############################################
# 3) Twister
# twst = function(t) (exp(t) - 1) / (exp(t) + 1)
############################################
def tws1(t):
    """
    Returns (exp(t) - 1) / (exp(t) + 1).
    For real t, this is often a form of tanh(t/2), 
    but here it's a direct ratio of exponentials.
    """
    return (np.exp(t) - 1) / (np.exp(t) + 1)

def tws(t1, t2):
  return ( tws1(t1), tws1(t2) )

############################################
# 4) Polka dot
# plk = function(t) sin(t) / cos(t) 
#                 i.e. tan(t)
############################################
def plk1(t):
    """
    Returns sin(t) / cos(t). This is essentially tan(t).
    """
    return np.sin(t) / np.cos(t)

def plk(t1, t2):
  return ( plk1(t1), plk1(t2) )

############################################
# 5) Zig zag
# zzg = function(t) (t + 2) / (t - 2)
############################################
def zzg1(t):
    """
    Returns (t + 2) / (t - 2).
    """
    return (t + 2) / (t - 2)

def zzg(t1, t2):
  return ( zzg1(t1), zzg1(t2) )

############################################
# 6) Loop the loop
# ltl = function(t) (t - 2i) / (t + 2i)
############################################
def ltl1(t):
    """
    Returns (t - 2i) / (t + 2i).
    In Python, imaginary unit is 1j, so 2i is 2j.
    """
    return (t - 2j) / (t + 2j)

def ltl(t1, t2):
  return ( ltl1(t1), ltl1(t2) )

############################################
# 7) Karatheodory
# kth = function(t) (1 + t) / (1 - t)
############################################
def kth1(t):
    """
    Returns (1 + t) / (1 - t).
    """
    return (1 + t) / (1 - t)

def kth(t1, t2):
  return ( kth1(t1), kth1(t2) )

############################################
# 8) Joukowsky
# jkw = function(t) t + 1/t
############################################
def jkw1(t):
    """
    Returns t + 1/t.
    Classic Joukowsky transform used in aerodynamics 
    (mapping circles to airfoil shapes).
    """
    return t + 1/t

def jkw(t1, t2):
  return ( jkw1(t1), jkw1(t2) )

############################################
# 9) Exponential
# epow = function(t) exp(t)
############################################
def epow1(t):
    """
    Returns exp(t).
    """
    return np.exp(t)

def epow(t1, t2):
  return ( epow1(t1), epow1(t2) )

############################################
# 10) Baker's map (mod1 mapping)
# bkr = function(t) { 
#   ( 2 * (Re(t) %% 1) ) %% 1 
#     + 1i * ( (Im(t) %% 1) + floor( 2 * (Re(t) %% 1)) ) / 2
# }
############################################
def bkr1(t):
    """
    A 2D baker's map applied to the real and imaginary parts of t.
    - x_new = (2*x_fold) mod 1
    - y_new = (y_fold + floor(2*x_fold)) / 2
    where x_fold = (Re(t) mod 1), y_fold = (Im(t) mod 1).
    Returns x_new + 1j*y_new.
    """
    x = np.real(t)
    y = np.imag(t)

    x_fold = x % 1  # fractional part of x
    y_fold = y % 1  # fractional part of y

    x_new = (2 * x_fold) % 1
    shift = np.floor(2 * x_fold)
    y_new = (y_fold + shift) / 2

    return x_new + 1j * y_new

def bkr(t1, t2):
  return ( bkr1(t1), bkr1(t2) )

def bkr2(t1,t2):
  tt1,tt2 = t1,t2
  for i in range(2):
    tt1=bkr1(tt1)
    tt2=bkr1(tt2)
  return (tt1,tt2)

def bkr3(t1,t2):
  tt1,tt2 = t1,t2
  for i in range(3):
    tt1=bkr1(tt1)
    tt2=bkr1(tt2)
  return (tt1,tt2)

def bkr4(t1,t2):
  tt1,tt2 = t1,t2
  for i in range(4):
    tt1=bkr1(tt1)
    tt2=bkr1(tt2)
  return (tt1,tt2)

def bkr5(t1,t2):
  tt1,tt2 = t1,t2
  for i in range(5):
    tt1=bkr1(tt1)
    tt2=bkr1(tt2)
  return (tt1,tt2)

def bkr10(t1,t2):
  tt1,tt2 = t1,t2
  for i in range(10):
    tt1=bkr1(tt1)
    tt2=bkr1(tt2)
  return (tt1,tt2)

def bkr20(t1,t2):
  for i in range(20):
    tt1=bkr1(tt1)
    tt2=bkr1(tt2)
  return (tt1,tt2)

############################################
# 11) Baker's map with exp(-x) mapping
# ebkr = function(t) { 
#   to01 = function(x) 1/(1+exp(-x))
#   ( 2 * to01(Re(t)) ) %% 1 
#     + 1i * ( to01(Im(t)) + floor(2 * to01(Re(t))) ) / 2
# }
############################################
def ebkr1(t):
    """
    A variant of the Baker's map where each coordinate is passed 
    through a logistic transform 1/(1 + exp(-x)) to map (-∞,∞) -> (0,1).
    Then the usual Baker step is applied.
    """
    def to01(x):
        # logistic function mapping R -> (0,1)
        return 1 / (1 + np.exp(-x))

    x = np.real(t)
    y = np.imag(t)

    x01 = to01(x)
    y01 = to01(y)

    x_new = (2 * x01) % 1
    shift = np.floor(2 * x01)
    y_new = (y01 + shift) / 2

    return x_new + 1j * y_new

def ebkr(t1, t2):
  return ( ebkr1(t1), ebkr1(t2) )

