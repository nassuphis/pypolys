import numpy as np
import matplotlib.pyplot as plt

def generate_roots(n_samples=100_000, seed=4242):
    np.random.seed(seed)
    st_pairs = np.random.rand(n_samples, 2)

    # Wilkinson-7 coefficients
    roots_exact = np.arange(1, 8)
    coeffs = np.poly(roots_exact).astype(np.complex128)
    idx_x6, idx_x5, idx_x4, idx_x3, idx_x2, idx_const = 1, 2, 3, 4, 5, -1

    all_roots = []

    for s, t in st_pairs:
        z = s + 1j * t
        w = s - 1j * t

        # YOUR SPECIFIED FUNCTIONS
        f6, g6 = (z**2 + 2*w + 1)**2, (z + w**2)**2
        f5, g5 = (z**3 - w + 1j)**2, (w - z)**3
        f4, g4 = (z * w + z**2 - 1j)**2, (z**2 - w**2 + 1j * z)**2
        f3, g3 = (1 + z**2 + w)**2, (z**2 + w**3 - 2)**2
        f2, g2 = (1 - z * w)**2, (z + 1j * w)**4
        f0, g0 = (z**2 + 2 * w - 1j)**2, (w**2 + z * w + 1)**3

        # Apply exponential modulation
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

        roots = np.roots(perturbed)
        all_roots.append(roots)

    return np.concatenate(all_roots)

def plot_roots(roots, dpi=300, figsize=(20, 20), filename=None):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor='black')
    ax.set_facecolor('black')
    #ax.scatter(roots.real, roots.imag, s=0.05, color='white', rasterized=True)
    ax.plot(roots.real, roots.imag, ',', color='white')  # single-pixel dots

    ax.set_title("Wilkinson-7: Named Variant with Custom f(s,t), g(s,t)", color='white')
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
    roots = generate_roots(n_samples=1_000_000)
    plot_roots(roots, dpi=1000, figsize=(10, 10), filename="wilkinson7_2.png")
