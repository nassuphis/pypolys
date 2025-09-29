# rasterize results
from __future__ import annotations
import numpy as np
import math
import numpy.typing as npt
from numba import njit, types
from numba.typed import Dict
import multiprocessing as mproc
from multiprocessing.shared_memory import SharedMemory
import pyvips as vips
import argparse
import time
import ast
import re

# =======================
# shared memory
# =======================

def make_shm(rows,cols,type):
    size = rows * cols * np.dtype(type).itemsize
    shm = SharedMemory( create=True, size = size )
    array = np.ndarray((rows,cols), dtype=type, buffer=shm.buf)
    array[:] = 0
    return (shm,array)

def get_shm(name,rows,cols,type):
    shm = SharedMemory(name=name)
    array = np.ndarray((rows, cols), dtype=type, buffer=shm.buf)
    return(shm,array)

# =======================
# worker
# =======================

def _raster_worker(args):
    (rid, rows_range, shm_roots_name, N, M,
     llx, lly, urx, ury, pixels,
     shm_img_name) = args

    # attach to shared buffers
    shm_roots, R = get_shm(shm_roots_name, N, M, np.complex128)
    shm_img, IMG = get_shm(shm_img_name, pixels, pixels, np.uint8)

    IMG1D = IMG.reshape(-1)

    span_x = urx - llx
    span_y = ury - lly
    if span_x <= 0.0: span_x = 1e-10
    if span_y <= 0.0: span_y = 1e-10
    sx = float(pixels) / span_x
    sy = float(pixels) / span_y

    i0, i1 = rows_range
    step = 1_000  # tune

    for a in range(i0, i1, step):
        b = min(a + step, i1)
        Z = R[a:b, :].reshape(-1)

        xr = Z.real.astype(np.float64, copy=False)
        yi = Z.imag.astype(np.float64, copy=False)
        msk = np.isfinite(xr) & np.isfinite(yi)
        if not msk.any():
            continue
        xr = xr[msk]; yi = yi[msk]

        # 2) cull anything outside the viewport BEFORE scaling
        in_view = (xr >= llx) & (xr < urx) & (yi >= lly) & (yi < ury)
        if not in_view.any():
            continue
        xr = xr[in_view]; yi = yi[in_view]

        # 3) map to pixel space (floats)
        xpf = (xr - llx) * sx
        ypf = (yi - lly) * sy

        # 4) guard against NaN/Inf produced by the multiply
        fOK = np.isfinite(xpf) & np.isfinite(ypf)
        if not fOK.any():
            continue
        xpf = xpf[fOK]; ypf = ypf[fOK]

        # 5) floor then cast to int64
        ix = np.floor(xpf).astype(np.int64, copy=False)
        iy = np.floor(ypf).astype(np.int64, copy=False)

        # 6) final in-bounds mask (defensive)
        inb = (ix >= 0) & (ix < pixels) & (iy >= 0) & (iy < pixels)
        if not inb.any():
            continue
        ix = ix[inb]; iy = iy[inb]

        # bins in 64-bit; also cast to intp for indexing
        bins = (iy * np.int64(pixels) + ix).astype(np.intp, copy=False)

        #CHECK: de-duplicate to reduce write pressure ?
        #ubins = np.unique(bins)
        IMG1D[bins] = 1  # multiple writes of 1 are benign

    shm_roots.close()
    shm_img.close()

# =======================
# on/off rasterize
# =======================

def rasterize(
    roots_mat: np.ndarray,
    llx: float, lly: float, urx: float, ury: float,
    pixels: int,
    nprocs: int | None = None
) -> np.ndarray:
    N, M = roots_mat.shape
    if N == 0 or M <= 1:
        return np.zeros((pixels, pixels), dtype=np.uint8)

    if nprocs is None:
        nprocs = mproc.cpu_count()

    # Share roots
    shm_roots, Rsh = make_shm(N, M, np.complex128)
    Rsh[:] = roots_mat  # one-time copy into shared segment

    # Shared output image (uint8). We'll flip vertically at the end.
    shm_img, IMG = make_shm(pixels, pixels, np.uint8)  # zero-initialized

    # Row ranges per worker
    rows_per = (N + nprocs - 1) // nprocs
    args = []
    for p in range(nprocs):
        i0 = p * rows_per
        i1 = min((p + 1) * rows_per, N)
        if i0 >= i1:
            continue
        args.append((
            p, (i0, i1), shm_roots.name, N, M,
            llx, lly, urx, ury, pixels,
            shm_img.name
        ))

    ctx = mproc.get_context("spawn")
    with ctx.Pool(processes=len(args)) as pool:
        pool.map(_raster_worker, args)

    # copy result out and clean up
    out = np.array(IMG[::-1, :], copy=True)  # flip vertically
    shm_img.close() 
    shm_img.unlink()
    shm_roots.close()
    shm_roots.unlink()
    return (out * 255).astype(np.uint8)

# =======================
# compute view
# =======================

def view(roots_mat: np.ndarray, view_string=None,pad=0.05):

    if view_string is not None:
        match_sq = re.match(r"^sq([0-9.]+)$", view_string)
        if match_sq:
            val = float(match_sq.group(1))
            return -val, -val, val, val
        # its an expression, try to evaluate it
        ll, ur = ast.literal_eval(view_string)
        llx, lly, ury, urx = ll.real, ll.imag, ur.real, ur.imag
        return llx, lly, urx, ury

    zs = roots_mat.ravel()
    if zs.size == 0: return -1.0, -1.0, 1.0, 1.0

    real = zs.real
    imag = zs.imag
    rx = real.max() - real.min() if real.size else 1.0
    ry = imag.max() - imag.min() if imag.size else 1.0
    pad_x = pad * (rx if rx > 0 else 1.0)
    pad_y = pad * (ry if ry > 0 else 1.0)

    llx = real.min() - pad_x
    urx = real.max() + pad_x
    lly = imag.min() - pad_y
    ury = imag.max() + pad_y

    return llx, lly, urx, ury

# =======================
# on/off rasterize
# =======================

def write_raster(img_arr,out="out.png"):
    px = img_arr.shape
    img = vips.Image.new_from_memory(img_arr.data, px[0], px[1], 1, "uchar")
    img.pngsave(out, compression=1, effort=1, filter="none", interlace=False, strip=True, bitdepth=1)

# =======================
# test
# =======================

@njit(cache=True, fastmath=True)
def serpentine_grid_i(n: int, i: int, ll: complex, ur: complex):
    if n <= 0 or i < 0 or i >= n:
        raise ValueError("Invalid n or i")
    cols = int(math.ceil(math.sqrt(float(n))))
    rows = int(math.ceil(n / cols))
    r = i // cols              # row
    c = i - r * cols           # col within row (pre serpentine flip)
    if (r & 1) == 1:
        c = cols - 1 - c
    fx = (c + 0.5) / cols
    fy = (r + 0.5) / rows
    llx, lly = ll.real, ll.imag
    urx, ury = ur.real, ur.imag
    x = llx + fx * (urx - llx)
    y = lly + fy * (ury - lly)
    return x, y

def serpentine_grid(n: int, bb=( 0.0+0.0*1j, 1.0+1.0*1j ) ):
    ll, ur = bb
    llx = ll.real
    lly = ll.imag
    urx = ur.real
    ury = ur.imag
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    X, Y = np.meshgrid(xs, ys)
    X[1::2] = X[1::2, ::-1]
    Y[1::2] = Y[1::2, ::-1]
    coords = np.column_stack( (X.ravel(), Y.ravel()) )[:n]
    coords[:, 0] = llx + coords[:, 0] * (urx - llx)
    coords[:, 1] = lly + coords[:, 1] * (ury - lly)
    return coords

if __name__ == "__main__":

    ap = argparse.ArgumentParser(description="Rasterizer")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--row", type=int, default=1000)
    ap.add_argument("--col", type=int, default=100)
    ap.add_argument("--pix", type=int, default=1000)
    args = ap.parse_args()

    n=args.row
    m=args.col

    t0 = time.perf_counter()
    S = serpentine_grid(n,-1,1,-1,1)
    SZ = S[:,0] + 1j * S[:,1]
    Z = np.column_stack((SZ))
    print(f"generate time: {time.perf_counter() - t0:.3f}s")
    print(f"points: {Z.size:,}")

    t0 = time.perf_counter()
    raster = rasterize(Z,-1.0, -1.0, +1.0, +1.0,pixels=args.pix)
    print(f"raster time: {time.perf_counter() - t0:.3f}s")

    t0 = time.perf_counter()
    write_raster(raster)
    print(f"save time: {time.perf_counter() - t0:.3f}s")




