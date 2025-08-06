import sys
import os
import numpy as np
import argparse
import pandas as pd
from mpmath import mp 
import matplotlib.pyplot as plt

def wilkinson(N):
    p = [mp.mpf(1), mp.mpf(-1)]
    for n in range(2, N+1):
        new = [mp.mpf(0)]*(len(p)+1)
        for i, c in enumerate(p):
            new[i]   += c       # × x
            new[i+1] += -mp.mpf(n)*c  # constant term
        p = new
    return p

def bozo_aberth(coeffs, maxsteps, roots_init):
    n = len(coeffs) - 1
    cf = [c/coeffs[0] for c in coeffs]
    roots = [roots_init**k for k in range(n)]
    for _ in range(maxsteps):
        for i in range(n):
            pi = roots[i]
            delta = mp.polyval(cf, pi)
            for j in range(n):
                if i == j: continue
                div = pi - roots[j]
                if mp.fabs(div)>0 : delta /= div
            roots[i] = pi - delta
    return roots

def mpfrange(starts, stops, steps):
    start = mp.mpf(starts)
    stop  = mp.mpf(stops)
    step  = mp.mpf(steps)
    x = start
    while x <= stop:
        yield x
        x += step

def save_roots_as_png(roots, filename="wilkinson7.png", dpi=600, figsize=(10, 10),
                      xlim=(-10, 10), ylim=(-10, 10)):
    # Flatten and convert mpmath.mpc to native complex
    roots_flat = [complex(r.real, r.imag) for sublist in roots for r in sublist]
    roots_array = np.array(roots_flat, dtype=np.complex128)

    # Plot setup
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor='black')
    ax.set_facecolor('black')

    # True pixel-level markers
    line, = ax.plot(roots_array.real, roots_array.imag, ',', color='white')
    line.set_rasterized(True)

    # Axis and style
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_title("Wilkinson Root Map", color='white')
    ax.set_xlabel("Real", color='white')
    ax.set_ylabel("Imag", color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    # Save to file
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, facecolor='black', bbox_inches='tight')
    plt.close()

N = 1_000_000
with mp.workdps(100):
    cf = wilkinson(20)
    guess = mp.mpc(mp.mpf('0.1'), mp.mpf('0.3333333'))
    roots = [None] * N
    for i in range(N):
        t = mp.rand()
        s = mp.rand()
        print(f"{i}/{N}")
        cf_pert = cf.copy()      # shallow copy of the list
        cf_pert[1] *= mp.exp(1j*2*mp.pi*t)       # adds the mpf to the real part of the mpc   
        cf_pert[2] *=  mp.exp(1j*2*mp.pi* s * t ) 
        cf_pert[5] *=  mp.exp(1j*2*mp.pi*(s-t))
        cf_pert[9] *=  mp.exp(1j*2*mp.pi*(s+t))
        cf_pert[19] *= t*mp.exp(1j*2*mp.pi*(t))
        rts = bozo_aberth( cf_pert, 100, guess )
        roots[i] = rts


save_roots_as_png(roots, filename="w20_locus.png", dpi=500, xlim=(-10,10), ylim=(-10, 10))