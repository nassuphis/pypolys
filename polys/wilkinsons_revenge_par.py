import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mproc
from mpmath import mp

def chessboard_roots(n):
    roots = []
    offset = mp.mpf(n - 1) / 2  # Center the grid on (0, 0)
    for i in range(n):
        for j in range(n):
            x = mp.mpf(i) - offset
            y = mp.mpf(j) - offset
            z = mp.mpc(x, y)
            roots.append(z)
    return roots

def poly_from_roots(roots):
    p = [mp.mpf(1)]
    for r in roots:
        new_p = [mp.mpf(0)] * (len(p) + 1)
        for i, c in enumerate(p):
            new_p[i]   += c      # multiply by x
            new_p[i+1] -= c * r  # constant term
        p = new_p
    return p

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

def aberth(coeffs, maxsteps, roots_init):
    n = len(coeffs) - 1
    tol=mp.mpf('1e-12')
    roots = roots_init.copy()
    for step in range(maxsteps):
        done = True 
        for i in range(n):
            pi = roots[i]
            delta = coeffs[0]
            for c in coeffs[1:]: delta = delta * pi + c
            if mp.fabs(delta) < tol: continue
            done = False
            for j in range(n):
                if i != j:
                    div = pi - roots[j]
                    if mp.fabs(div) > 0: delta /= div
            roots[i] = pi - delta
        if done: break
    return roots,step

def perturb_1(cf,s,t):
    PI2 = 2 * mp.pi
    perturbed_cf = cf.copy()
    perturbed_cf[1]  *= mp.exp(1j * PI2 * t)
    perturbed_cf[2]  *= mp.exp(1j * PI2 * s * t)
    perturbed_cf[5]  *= mp.exp(1j * PI2 * (s - t))
    perturbed_cf[9]  *= mp.exp(1j * PI2 * (s + t))
    perturbed_cf[-1] *= s * mp.exp(1j * PI2 * t)
    return perturbed_cf


def tile_job(args):
    (id, cf, s_start, s_end, t_start, t_end, n_points, perturb) = args
    degree = len(cf)
    steps = int(n_points**0.5)
    s_step = (s_end - s_start) / steps
    t_step = (t_end - t_start) / steps
    job_roots=[]
    with mp.workdps(100):   
        PI2 = 2 * mp.pi
        roots_init = mp.mpc(mp.mpf('0.1'), mp.mpf('0.3333333'))
        guess = [roots_init ** k for k in range(degree)]
        prev_guess = guess
        i,j, dj, tot_niter  = int(0), int(0), int(1), int(0)
        for k in range(n_points):
            s = s_start + i * s_step
            t = t_start + j * t_step
            local_cf = perturb(cf,s,t)
            iteration_roots, niter = aberth(local_cf, 100, guess)
            tot_niter += niter
            guess = [ r + (r - p) for r, p in zip(iteration_roots, prev_guess)]
            prev_guess = iteration_roots
            job_roots.extend((float(r.real), float(r.imag), float(s), float(t)) for r in iteration_roots)
            if id < 1 and k % (n_points//100) == 0: 
                print(
                    f"worker {id} : "
                    f"niter[{tot_niter/(k+1):.2f}] "
                    f"{k}/{n_points} "
                    f"row[{i}/{steps}],col[{j}/{steps}] "
                    f" s[{mp.nstr(s,2)}] t[{mp.nstr(t,2)}]"
                )
            j += dj
            if j==steps or j<0 :
                i += int(1) 
                dj *= int(-1)
                j += dj
                guess = iteration_roots
                prev_guess = guess
    return job_roots

u_vec = np.array([1.0, 0.0,  -0.0])    # horizontal = x
w_vec = np.array([0.0, 1.0,  -0.0]) # vertical = y - 0.1*t
def project(x, y, t):
    p = np.array([x, y, t])
    u = np.dot(p, u_vec)
    w = np.dot(p, w_vec)
    return u, w

def save_roots_as_png(roots, filename="w20_parallel.png", dpi=600, figsize=(10, 10), xlim=(-10, 10), ylim=(-10, 10)):
    # roots is a list of (x, y, s, t)
    flat = [r for worker in roots for r in worker]
    xs = [r[0] for r in flat]
    ys = [r[1] for r in flat]
    ts = [r[2] for r in flat]
    ss = [r[3] for r in flat]
    t_scale = (max(xs + ys) - min(xs + ys)) / 1.0  # e.g. 100
    t_center = 0.5 * t_scale
    t_adjusted = [t * t_scale - t_center for t in ts]
    projected = [project(y, x, t_adj) for (x, y, s, t_adj) in zip(xs, ys, ss, t_adjusted)]
    us, ws = zip(*projected)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor='black')
    ax.set_facecolor('black')
    sc = ax.scatter(us, ws, c=ss, cmap='viridis', s=0.5, edgecolors='none')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.grid(False)
    ax.axis('off')  # disables frame, ticks, labels
    ax.set_title("Wilkinson-20 Root Locus", color='white')
 
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, facecolor='black', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":

    ctx = mproc.get_context("spawn")
    N = 100_000  # number of samples
    tiles = int(mproc.cpu_count()**0.5)**2 # square tiles
    tiles_per_side = int(tiles**0.5)
    assert tiles_per_side**2 == tiles, "tiles must be a square number"
    with mp.workdps(100): 
        tile_size = mp.mpf("1.0") / tiles_per_side
        points_per_worker = int((N // tiles)**0.5)**2
        cf = wilkinson(20)
        args = []
        for id in range(ctx.cpu_count()): 
            tx = id % tiles_per_side
            ty = id // tiles_per_side
            s_start = tx * tile_size
            s_end   = s_start + tile_size
            t_start = ty * tile_size
            t_end   = t_start + tile_size
            print(f"{id} [{tx},{ty}] : ({s_start},{t_start})-({s_end},{t_end}) : {points_per_worker}")
            args.append((id, cf.copy(), s_start, s_end, t_start, t_end, points_per_worker,perturb_1))
        with ctx.Pool(processes=len(args)) as pool: 
            roots = pool.map(tile_job, args)
    save_roots_as_png(roots, filename="w20_parallel.png", dpi=1000, figsize=(10, 10),xlim=(-75, 75), ylim=(-75, 75))

