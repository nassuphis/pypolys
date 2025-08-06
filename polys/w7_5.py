import numpy as np
import matplotlib.pyplot as plt

def generate_roots(n_samples=100_000, seed=97531):
    np.random.seed(seed)
    st_pairs = np.random.rand(n_samples, 2)

    # Wilkinson-7 coefficients
    roots_exact = np.arange(1, 8)
    coeffs = np.poly(roots_exact).astype(np.complex128)
    idx_x6, idx_x5, idx_x4, idx_x3, idx_x2, idx_const = 1, 2, 3, 4, 5, -1

    all_roots = []

    for s, t in st_pairs:
        z = s * t * s + 1j * t * s * t
        w = t * s * t + 1j * s * t * s

        # You can swap these out for other f, g formulas if desired
        f6, g6 = (z**3 + w**4 + z*w - 1j*z + 1)**2, (w**3 + 1j*z**7 + z)**5
        f5, g5 = (z**4 + w*z - 1 + 1j*w)**3, (z**5 - w**4 + 1j)**3
        f4, g4 = (z**4 + w**4 + 1 + z)**5, (z**3 + w**2 - z*w + 1j)**5
        f3, g3 = (z + w + 1j + 1j*z*w)**3, (z**3 - w + z*w**2)**2
        f2, g2 = (z**2 - w + w**3 + 1j)**2, (z**4 + w*z - 1j)**5
        f0, g0 = (z**5 + w**2 + 2 + 1j*w)**2, (z**3 + 2*w + w**2)**5

        # Phase is real part of g
        eps_x6 = f6 * np.exp(1j * 2 * np.pi * g6.real)
        eps_x5 = f5 * np.exp(1j * 2 * np.pi * g5.real)
        eps_x4 = f4 * np.exp(1j * 2 * np.pi * g4.real)
        eps_x3 = f3 * np.exp(1j * 2 * np.pi * g3.real)
        eps_x2 = f2 * np.exp(1j * 2 * np.pi * g2.real)
        const_term = f0 * np.exp(1j * 2 * np.pi * g0.real)

        perturbed = coeffs.copy()
        perturbed[idx_x6] += eps_x6
        perturbed[idx_x5] += eps_x5
        perturbed[idx_x4] += eps_x4
        perturbed[idx_x3] += eps_x3
        perturbed[idx_x2] += eps_x2
        perturbed[idx_const] = const_term

        roots = np.roots(perturbed*t*s*t*s+coeffs*(1-t*s*t*s))
        all_roots.append(roots)

    return np.concatenate(all_roots)

def plot_roots(roots, dpi=300, figsize=(25, 25), filename=None):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor='black')
    ax.set_facecolor('black')
    #ax.scatter(roots.real, roots.imag, s=0.05, color='white', rasterized=True)
    ax.plot(roots.real, roots.imag, ',', color='white')  # single-pixel dots
    ax.set_xlim(-5, 15)
    ax.set_ylim(-10, 10)
    ax.set_title("Wilkinson-7: Complex Polynomial Perturbations", color='white')
    ax.set_xlabel("Real Part", color='white')
    ax.set_ylabel("Imaginary Part", color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    ax.set_aspect('equal')
    ax.grid(False)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=dpi, facecolor='black', bbox_inches='tight')
    else:
        plt.show()

if __name__ == "__main__":
    roots = generate_roots(n_samples=2_000_000)  # adjust this
    plot_roots(roots, dpi=1000, figsize=(15, 15), filename="wilkinson7_5.png")

