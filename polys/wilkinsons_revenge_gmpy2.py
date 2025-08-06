import numpy as np
import matplotlib.pyplot as plt
import gmpy2
from gmpy2 import mpfr, mpc, get_context, random_state, exp, const_pi, mpfr_random

# Set precision (~100 decimal digits)
get_context().precision = 3320
pi = const_pi()
rand = random_state(42)

def wilkinson(N):
    p = [mpfr(1), mpfr(-1)]
    for n in range(2, N + 1):
        new = [mpfr(0)] * (len(p) + 1)
        for i, c in enumerate(p):
            new[i] += c
            new[i + 1] += -mpfr(n) * c
        p = new
    return p

def polyval(coeffs, x):
    result = coeffs[0]
    for c in coeffs[1:]:
        result = result * x + c
    return result

def bozo_aberth(coeffs, maxsteps, roots_init):
    n = len(coeffs) - 1
    cf = [c / coeffs[0] for c in coeffs]
    roots = [roots_init ** k for k in range(n)]
    for _ in range(maxsteps):
        for i in range(n):
            pi = roots[i]
            delta = polyval(cf, pi)
            for j in range(n):
                if i != j:
                    div = pi - roots[j]
                    if abs(div) > 0:
                        delta /= div
            roots[i] = pi - delta
    return roots

def save_roots_as_png(roots, filename="w20_locus.png", dpi=600, figsize=(10, 10),
                      xlim=(-10, 10), ylim=(-10, 10)):
    roots_flat = [complex(r.real, r.imag) for sublist in roots for r in sublist]
    roots_array = np.array(roots_flat, dtype=np.complex128)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor='black')
    ax.set_facecolor('black')

    line, = ax.plot(roots_array.real, roots_array.imag, ',', color='white')
    line.set_rasterized(True)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_title("Wilkinson-20 Root Locus (GMPY2)", color='white')
    ax.set_xlabel("Real", color='white')
    ax.set_ylabel("Imag", color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, facecolor='black', bbox_inches='tight')
    plt.close()

# Main computation
N = 1_000_000
cf = wilkinson(20)
guess = mpfr("0.1") + mpfr("0.3333333") * 1j
roots = [None] * N

for i in range(N):
    t = mpfr_random(rand)
    s = mpfr_random(rand)
    print(f"{i + 1}/{N}")

    cf_pert = list(cf)
    cf_pert[1]  *= exp(1j * 2 * pi * t)
    cf_pert[2]  *= exp(1j * 2 * pi * s * t)
    cf_pert[5]  *= exp(1j * 2 * pi * (s - t))
    cf_pert[9]  *= exp(1j * 2 * pi * (s + t))
    cf_pert[19] *= t * exp(1j * 2 * pi * t)

    roots[i] = bozo_aberth(cf_pert, 100, guess)

save_roots_as_png(roots, filename="w20_locus_gmpy2.png", dpi=500, xlim=(-10, 10), ylim=(-10, 10))


