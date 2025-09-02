import numpy as np

def perturb_0(cf,s,t):
    perturbed_cf = cf.copy()
    return perturbed_cf

def perturb_1(cf,s,t):
    PI2 = 2 * np.pi
    perturbed_cf = cf.copy()
    perturbed_cf[1]  *= np.exp(1j * PI2 * t)
    perturbed_cf[2]  *= np.exp(1j * PI2 * s * t)
    perturbed_cf[5]  *= np.exp(1j * PI2 * (s - t))
    perturbed_cf[9]  *= np.exp(1j * PI2 * (s + t))
    perturbed_cf[-1] *= s * np.exp(1j * PI2 * t)
    return perturbed_cf

def perturb_2(cf,s,t):
    PI2 = 2 * np.pi
    perturbed_cf = cf.copy()
    perturbed_cf[1]  *= np.exp(1j * PI2 * t * t)
    perturbed_cf[2]  *= np.exp(1j * PI2 * s * s)
    perturbed_cf[3]  *= np.exp(1j * PI2 * (s - t))
    perturbed_cf[4]  *= np.exp(1j * PI2 * (s + t))
    perturbed_cf[-1] *= np.exp(1j * PI2 * t)
    return perturbed_cf

def perturb_3(cf,s,t):
    PI2 = 2 * np.pi
    perturbed_cf = cf.copy()
    perturbed_cf[1]  *= np.exp(1j * PI2 * t * t)
    perturbed_cf[2]  *= np.exp(1j * PI2 * s * s)
    perturbed_cf[3]  *= np.exp(1j * PI2 * (s - t))
    perturbed_cf[4]  *= np.exp(1j * PI2 * (s*s + t*t))
    perturbed_cf[-1] *= np.exp(1j * PI2 * t)
    return perturbed_cf

def perturb_4(cf,s,t):
    PI2 = 2 * np.pi
    ss = np.exp(1j * PI2 * s)
    tt = np.exp(1j * PI2 * t)
    perturbed_cf = cf.copy()
    perturbed_cf[1]  += ss*tt
    perturbed_cf[2]  += ss+tt
    perturbed_cf[3]  *= (ss-tt)
    perturbed_cf[4]  += (ss*ss-tt*tt)
    perturbed_cf[-1] *= tt-ss
    return perturbed_cf

def perturb_5(cf,s,t):
    PI2 = 2 * np.pi
    ss = np.exp(1j * PI2 * s)
    tt = np.exp(1j * PI2 * t)
    perturbed_cf = cf.copy()
    perturbed_cf[1]  += ss*tt
    perturbed_cf[2]  += ss+tt
    perturbed_cf[3]  *= (ss-tt)**3
    perturbed_cf[4]  += (ss*ss-tt*tt)**2
    perturbed_cf[-1] *= tt-ss
    return perturbed_cf

def sort_abs(cf):
  return cf[np.argsort(np.abs(cf))]

def sort_angle(cf):
  return cf[np.argsort(np.angle(cf))]

def perturb_6(cf,s,t):
    PI2 = 2 * np.pi
    ss = np.exp(1j * PI2 * s)
    tt = np.exp(1j * PI2 * t)
    perturbed_cf = cf.copy()
    perturbed_cf[1]  += ss*tt
    perturbed_cf[2]  += ss+tt
    perturbed_cf[3]  *= (ss-tt)**3
    perturbed_cf[4]  += (ss*ss-tt*tt)**2
    perturbed_cf[-1] *= tt-ss
    return sort_abs(perturbed_cf)


def perturb_7(cf,s,t):
    PI2 = 2 * np.pi
    st10=np.sin(10*(s+t))
    s10=np.sin(s*10)
    t10=np.sin(t*10)
    s29=np.sin(s*29)
    t29=np.sin(t*29)
    t71=np.sin(t*71)
    s71=np.sin(s*71)
    maxp=max(s29,t71)
    minp=min(s29,t71)
    ss10 = np.exp(1j * PI2 * s10)
    tt10 = np.exp(1j * PI2 * t10)
    sstt = np.exp(1j * PI2 * maxp)
    ttss = np.exp(1j * PI2 * minp)
    perturbed_cf = cf.copy()
    perturbed_cf[11]  *= ttss+(s+1j*t)*PI2
    perturbed_cf[10]  *= tt10*ss10+(t+1j*s)*PI2
    perturbed_cf[-5]  *= sstt+(maxp+1j*minp)*PI2*100
    perturbed_cf[-4]  *= ttss+(minp+1j*maxp)*PI2*1000
    perturbed_cf[-3]  *= (t71+1j*s29)*PI2*100000
    perturbed_cf[-2]  *= (maxp+1j*minp)*PI2*100000
    perturbed_cf[-1]  *= sstt
    return perturbed_cf*st10+sort_abs(perturbed_cf)*(1-st10)

def perturb_7(cf,s,t):
    perturbed_cf = np.asarray(cf, dtype=np.complex128).copy()
    PI2 = 2 * np.pi
    st10=np.sin(10*(s+t))
    s10=np.sin(s*10)
    t10=np.sin(t*10)
    s29=np.sin(s*29)
    t29=np.sin(t*29)
    t71=np.sin(t*71)
    s71=np.sin(s*71)
    maxp=max(s29,t71)
    minp=min(s29,t71)
    ss10 = np.exp(1j * PI2 * s10)
    tt10 = np.exp(1j * PI2 * t10)
    sstt = np.exp(1j * PI2 * maxp)
    ttss = np.exp(1j * PI2 * minp)
    perturbed_cf[11]  *= ttss+(s+1j*t)*PI2
    perturbed_cf[10]  *= tt10*ss10+(t+1j*s)*PI2
    perturbed_cf[-5]  *= sstt+(maxp+1j*minp)*PI2*100
    perturbed_cf[-4]  *= ttss+(minp+1j*maxp)*PI2*1000
    perturbed_cf[-3]  *= (t71+1j*s29)*PI2*100000
    perturbed_cf[-2]  *= (maxp+1j*minp)*PI2*100000
    perturbed_cf[-1]  *= sstt
    fcf1 = perturbed_cf*st10
    fcf2 = sort_abs(perturbed_cf)*(1-st10)
    fcf3 = fcf1 + fcf2 
    return fcf3

def perturb_8(cf,s,t):
    perturbed_cf = np.asarray(cf, dtype=np.complex128).copy()
    PI2 = 2 * np.pi
    st10=np.sin(10*(s+t))
    s10=np.sin(s*10)
    t10=np.sin(t*10)
    s29=np.sin(s*29)
    t29=np.sin(t*29)
    t71=np.sin(t*71)
    s71=np.sin(s*71)
    maxp=max(s29,t71)
    minp=min(s29,t71)
    ss10 = np.exp(1j * PI2 * s10)
    tt10 = np.exp(1j * PI2 * t10)
    sstt = np.exp(1j * PI2 * maxp)
    ttss = np.exp(1j * PI2 * minp)
    perturbed_cf[11]  *= ttss+(s+1j*t)*PI2
    perturbed_cf[10]  *= tt10*ss10+(t+1j*s)*PI2
    perturbed_cf[-5]  *= sstt+(maxp+1j*minp)*PI2*100
    perturbed_cf[-4]  *= ttss+(minp+1j*maxp)*PI2*1000
    perturbed_cf[-3]  *= (t71+1j*s29)*PI2*100000
    perturbed_cf[-2]  *= (maxp+1j*minp)*PI2*100000
    perturbed_cf[-1]  *= sstt
    return perturbed_cf

