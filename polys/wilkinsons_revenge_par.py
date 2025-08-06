import os
import time
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mproc
from mpmath import mp


def wilkinson(n):
    p = [mp.mpf(1), mp.mpf(-1)]
    for k in range(2, n+1):
        new = [mp.mpf(0)] * (len(p)+1)
        for i, c in enumerate(p):
            new[i]   += c
            new[i+1] += -mp.mpf(k) * c
        p = new
    p = [c / p[0] for c in p]
    return p

def polyval(coeffs, x):
    result = coeffs[0]
    for c in coeffs[1:]:
        result = result * x + c
    return result

def bozo_aberth(coeffs, maxsteps, roots_init):
    n = len(coeffs) - 1
    tol=mp.mpf('1e-15')
    roots = roots_init.copy()
    for step in range(maxsteps):
        done = True 
        for i in range(n):
            pi = roots[i]
            delta = coeffs[0]
            for c in coeffs[1:]:
                delta = delta * pi + c
            if mp.fabs(delta) < tol: continue
            done = False
            for j in range(n):
                if i != j:
                    div = pi - roots[j]
                    if mp.fabs(div) > 0:
                        delta /= div
            roots[i] = pi - delta
        if done: break
    return roots,step

def generate_sample(args):
    (id, cf, n) = args
    seed = int((time.time() * 1000) % (2**16)) + id + os.getpid()
    np.random.seed(seed  % (2**32) )
    degree = len(cf)
    steps = int(n**0.5)
    step = mp.mpf("1.0") / steps
    job_roots=[]
    with mp.workdps(100):   
        PI2 = 2 * mp.pi
        t0 = mp.mpmathify(np.random.random()) * step
        s0 = mp.mpmathify(np.random.random()) * step
        roots_init = mp.mpc(mp.mpf('0.1'), mp.mpf('0.3333333'))
        guess = [roots_init ** k for k in range(degree)]
        k=int(0)
        for i in range(steps):
            for j in range(steps):
                local_cf = cf.copy()
                s = s0 + step * i
                t = t0 + step * j
                local_cf[1]  *= mp.exp(1j * PI2 * t)
                local_cf[2]  *= mp.exp(1j * PI2 * s * t)
                local_cf[5]  *= mp.exp(1j * PI2 * (s - t))
                local_cf[9]  *= mp.exp(1j * PI2 * (s + t))
                local_cf[19] *= t * mp.exp(1j * PI2 * t)
                iteration_roots, niter = bozo_aberth(local_cf, 100, guess)
                guess = iteration_roots
                job_roots.extend(iteration_roots)
                k +=1
                if id < 1 and k % 10 == 0: print(f"worker {id} : {k}/{n} [{niter}]")
    return job_roots

def save_roots_as_png(roots, filename="w20_parallel.png", dpi=600, figsize=(10, 10), xlim=(-10, 10), ylim=(-10, 10)):
    roots_flat = [complex(r.real, r.imag) for sublist in roots for r in sublist]
    roots_array = np.array(roots_flat, dtype=np.complex128)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor='black')
    ax.set_facecolor('black')
    line, = ax.plot(roots_array.imag, roots_array.real, ',', color='white')
    line.set_rasterized(True)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_title("Wilkinson-20 Root Locus", color='white')
    ax.set_xlabel("Imag", color='white')
    ax.set_ylabel("Real", color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, facecolor='black', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":

    N = 1_000_000  # number of samples
    mp.dps = 100

    with mp.workdps(100): cf = wilkinson(20)
    ctx = mproc.get_context("spawn")

    args = []
    for id in range(ctx.cpu_count()): 
        args.append(( id, cf.copy(), N//ctx.cpu_count() ))

    with ctx.Pool(processes=len(args)) as pool: 
        roots = pool.map(generate_sample, args)

    save_roots_as_png(roots, filename="w20_parallel.png", dpi=1000, figsize=(10, 10),xlim=(-50, 50), ylim=(-50, 50))

