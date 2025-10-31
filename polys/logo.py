# -----------------------------------
# Symmetra Logo
# -----------------------------------

import numpy as np
import pyvips

# -----------------------------------
# easing rescale (vector-friendly)
# -----------------------------------
def spc(x, start, end, power=1.0):
    """
    Direction-preserving power-eased rescale of a 1-D array x to [start, end].
    power > 1 → denser near 'end'; power < 1 → denser near 'start'.
    """
    x = np.asarray(x, dtype=np.float64)
    xmin, xmax = x.min(), x.max()
    denom = xmax - xmin
    if denom <= 0: s = np.zeros_like(x)
    else: s = (x - xmin) / denom
    if end < start: s = 1.0 - (1.0 - s)**power
    else: s = s**power
    return start + s * (end - start)

# -----------------------------------
# parameters
# -----------------------------------
N_rings   = 2**12          # number of ellipses
squish    = 0.5            # vertical squash
num_arms  = 2              # how many spiral arms to replicate
noise_sd  = 0.0033         # Gaussian jitter
W = H     = int(10e3)      # output size in pixels
fg        = 255            # black
bg        = 0              # white

# -----------------------------------
# schedules over rings (vectorized)
# -----------------------------------
i = np.arange(1, N_rings + 1, dtype=np.float64)

# how many samples along each ring
n_pts = spc(i, 1024, 20 * 1024, power=5.0).astype(np.int32)

# ring radius for the base circle (outer rings → 1.0)
r_circle = spc(i, 0.075, 1.0, power=1.0)            # radius schedule
# anisotropy: scales real part from 1.5 (inner) → 1.0 (outer) to become circular
a_scale  = spc(i, 1.5, 1.0, power=1.0)
# ring-dependent rotation
theta_r  = np.exp(1j * 2 * np.pi * spc(i, 0.125, 1.0, power=0.75))

# -----------------------------------
# generate points for all rings and arms
# -----------------------------------
rows_x = []
rows_y = []

for k in range(N_rings):
    n = int(n_pts[k])
    # parameter along ring
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    # base circle for this ring
    circle = r_circle[k] * np.exp(1j * 2 * np.pi * t)
    # ellipse (stretch real axis only)
    ellipse = a_scale[k] * np.real(circle) + 1j * np.imag(circle)
    # ring rotation
    z = ellipse * theta_r[k]

    # replicate into arms by phase offsets
    if num_arms > 1:
        arm_pts = []
        for m in range(num_arms):
            arm_phase = np.exp(1j * 2 * np.pi * (m / num_arms))
            arm_pts.append(z * arm_phase)
        z_all = np.concatenate(arm_pts)
    else:
        z_all = z

    x = np.real(z_all)
    y = np.imag(z_all) * squish

    rows_x.append(x)
    rows_y.append(y)

# stack to one big point cloud
X = np.concatenate(rows_x)
Y = np.concatenate(rows_y)

# jitter
    # x += np.random.normal(0.0, noise_sd, size=x.size)
    #y += np.random.normal(0.0, noise_sd, size=y.size)


# -----------------------------------
# rasterize to a bilevel image (preserve aspect + 20% margin)
# -----------------------------------
margin = 0.20

# data range (already includes your squish in Y)
xmin, xmax = X.min(), X.max()
ymin, ymax = Y.min(), Y.max()
xmid = 0.5 * (xmin + xmax)
ymid = 0.5 * (ymin + ymax)

xr = xmax - xmin
yr = ymax - ymin

# use the larger span so we scale both axes uniformly (preserves squish)
span = max(xr, yr) * (1.0 + margin)

# choose a single scale for both axes (W==H but this also works if not)
scale = (min(W, H) - 1) / span

# map to pixels (flip Y so up is +y)
xpix = np.round((X - xmid) * scale + (W - 1) / 2).astype(np.int32)
ypix = np.round((ymid - Y) * scale + (H - 1) / 2).astype(np.int32)

# clip to bounds
xpix = np.clip(xpix, 0, W - 1)
ypix = np.clip(ypix, 0, H - 1)

# render
img = np.full((H, W), bg, dtype=np.uint8)
img[ypix, xpix] = fg

vimg = pyvips.Image.new_from_memory(img.tobytes(), W, H, 1, "uchar")
vimg.write_to_file("logo.png")
print("Saved → logo.png (with uniform scale + 20% margin)")
