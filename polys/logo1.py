#!/usr/bin/env python
# Symmetra Capital Galaxy Logo

import argparse
import numpy as np
import pyvips as vips
from numba import njit, prange
from types import SimpleNamespace
import sys
import re
import math
import expandspec

LOGO_KEYS_ORDER = ["N","dth","tda","tdw","tdt","swa","swb","sqs","frt","mrg","pix","drt","lmu","lsig"]

LOGO_DEFAULTS = {
    "N":   50_000,
    "dth": 1.0,
    "tda": 1.0,
    "tdw": 0.1,
    "tdt": 1.0,
    "swa": -0.33,
    "swb": 2.0,
    "sqs": 0.5,
    "frt": -0.1,
    "mrg": 0.10,
    "pix": 25_000,
    # dot-size distribution (generic units)
    "drt": 15,        # cap on generic radius; min is 1
    "lmu": 1.0,       # log-space mean (median = exp(lmu))
    "lsig": 0.75,     # log-space stddev
}

def _cast_like(default_val, s: str):
    if isinstance(default_val, int):
        # ints may appear as floats in spec (e.g., "25000.0") – round safely
        return int(round(float(s)))
    elif isinstance(default_val, float):
        return float(s)
    else:
        return s  # fallback (not used here)

def parse_logo_spec(spec: str, defaults: dict) -> dict:
    """
    Parse 'k:v' comma-separated string into a dict, casting to the type of defaults[k].
    Unknown keys are ignored (but warned).
    """
    out = dict(defaults)
    if not spec:
        return out
    # allow spaces; split on commas not inside braces (future-proof)
    parts = [p.strip() for p in re.split(r",(?![^{}]*\})", spec) if p.strip()]
    for part in parts:
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip()
        v = v.strip()
        if k in out:
            try:
                out[k] = _cast_like(defaults[k], v)
            except Exception:
                print(f"[warn] could not parse {k}:{v}, keeping default {defaults[k]}", file=sys.stderr)
        else:
            print(f"[warn] unknown key '{k}' ignored", file=sys.stderr)
    return out


def dict_to_namespace(d: dict) -> SimpleNamespace:
    return SimpleNamespace(
        N=d["N"], 
        dth=d["dth"], 
        tda=d["tda"], 
        tdw=d["tdw"], 
        tdt=d["tdt"],
        swa=d["swa"], 
        swb=d["swb"], 
        sqs=d["sqs"], 
        frt=d["frt"],
        mrg=d["mrg"],
        pix=d["pix"],
        # dot-size params (from spec)
        drt=d["drt"], 
        lmu=d["lmu"], 
        lsig=d["lsig"],
        # runtime knobs
        stamp_chunk=32_768, seed=0,
    )

def logo_dict_to_string(d: dict) -> str:
    return ",".join(f"{k}:{d[k]}" for k in LOGO_KEYS_ORDER)

# ---------- helpers ----------

def smoothstep(x, w):
    t = np.clip((x + w) / (2.0 * w), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def ddth(n, a):
    r = np.random.random(n) ** a
    th = 2.0 * np.pi * np.random.random(n)
    return r * np.exp(1j * th)

def teardrop(z, a=2.0, w=0.05, tail=0.0):
    x = z.real
    y = z.imag
    y_pow = np.sign(y) * (np.abs(y) ** a)
    b = smoothstep(x, w)
    y_new = b * y + (1.0 - b) * y_pow
    x_new = x * (1.0 + tail * (1.0 - b))
    return x_new + 1j * y_new

def swirl(z, a=0.5, b=1.0):
    r = np.abs(z)
    rmax = r.max() if r.size else 1.0
    denom = max(rmax - 1.0, 1e-9)
    t = np.where(r > 1.0, (r - 1.0) / denom, 0.0)
    return z * np.exp(1j * 2.0 * np.pi * a * (t ** b))

def squish(z, factor=0.5):
    return z.real + 1j * (z.imag * factor)

def make_disc_offsets(r):
    r = int(max(1, r))
    yy, xx = np.mgrid[-r:r+1, -r:r+1]
    mask = (xx*xx + yy*yy) <= r*r
    return yy[mask].astype(np.int32), xx[mask].astype(np.int32)

# ---------- numba stamping kernel ----------

@njit(parallel=True, fastmath=True, cache=True)
def stamp_points(canvas, ys, xs, dy, dx, H, W):
    n = ys.shape[0]
    k = dy.shape[0]
    for i in prange(n):
        y0 = ys[i]
        x0 = xs[i]
        for j in range(k):
            y = y0 + dy[j]
            x = x0 + dx[j]
            if 0 <= y < H and 0 <= x < W:
                canvas[y, x] = 255

# ---------- main pipeline ----------

def build_logo(args):
    """
    Return:
      logo: complex ndarray (logical coords, centered at 0+0j)
      size_gen: float ndarray (generic radii), lognormal with params (lmu, lsig), clipped to [1, drt]
    """
    np.random.seed(args.seed)

    # geometry in logical space
    disk = ddth(args.N, args.dth)
    td1  = teardrop(disk, a=args.tda, w=args.tdw, tail=args.tdt)
    td2  = td1 * np.exp(1j * 2.0 * np.pi * 0.5)
    td   = np.concatenate([td1, td2])
    logo = swirl(td, a=args.swa, b=args.swb)
    logo = squish(logo, args.sqs)
    logo = logo * np.exp(1j * 2.0 * np.pi * args.frt)  # final rot

    # lognormal sizes in *generic units* (no dependence on pix/span)
    z = np.random.normal(0.0, 1.0, size=logo.size)
    size_gen = np.exp(args.lmu + args.lsig * z).astype(np.float32)
    np.clip(size_gen, 1.0, float(args.drt), out=size_gen)  # cap only; drt is not a ratio

    return logo, size_gen

def render_logo(args):
    logo, size_gen = build_logo(args)

    # square frame centered at (0,0)
    rx = np.max(np.abs(logo.real)); ry = np.max(np.abs(logo.imag))
    half0 = max(rx, ry)
    half  = half0 * (1.0 + 2.0 * args.mrg)
    span  = 2.0 * half

    W = H = int(args.pix)
    px_per_logical = (W - 1) / span

    # map logical -> pixel coords
    x = logo.real; y = logo.imag
    px = np.rint((x + half) * px_per_logical).astype(np.int32)
    py = np.rint((half - y) * px_per_logical).astype(np.int32)
    px = np.clip(px, 0, W - 1); py = np.clip(py, 0, H - 1)

    # generic size -> pixel radius 
    scale_g2px = span / px_per_logical
    r_px = np.maximum(1, np.rint(size_gen * scale_g2px).astype(np.int32))
    unique_r = np.unique(r_px)

    # rasterize
    canvas = np.zeros((H, W), dtype=np.uint8)
    for r in unique_r.tolist():
        idx = np.flatnonzero(r_px == r)
        if idx.size == 0: continue
        dy, dx = make_disc_offsets(r)
        for s in range(0, idx.size, int(args.stamp_chunk)):
            j = idx[s:s+int(args.stamp_chunk)]
            stamp_points(canvas, py[j], px[j], dy, dx, H, W)

    np.putmask(canvas, canvas > 0, 255)
    return canvas

def add_footer_label(
    base: vips.Image,
    text: str,
    *,
    pad_lr_px: int = 40,
    dpi: int = 300,
    font: str = "Courier New Bold 60",
    align: str = "centre",
    invert: bool = False,   # <- new flag for black vs white text
) -> vips.Image:
    """
    Draw 'text' centered at the bottom of a 1-band uchar image (0/255),
    compositing non-destructively. When invert=True, text is black on white.
    """
    H, W = base.height, base.width
    bottom_margin_px = max(4, H // 40)
    box_w = max(1, W - 2 * pad_lr_px)

    glyph = vips.Image.text(
        text,
        width=box_w,
        dpi=dpi,
        font=font,
        align=align,
    )
    glyph = (glyph > 0).ifthenelse(255, 0, blend=False)

    gx = pad_lr_px + max(0, (box_w - glyph.width) // 2)
    gy = max(0, H - glyph.height - bottom_margin_px)
    glyph_full = vips.Image.black(W, H).insert(glyph, gx, gy)

    if not invert:
        # white text on black background
        return base | glyph_full
    else:
        # black text on white background
        glyph_black = 255 - glyph_full
        return base & glyph_black

def save_png_bilevel(
    canvas: np.ndarray,
    out_path: str,
    invert: bool,
    footer_text: str | None = None,
    *,
    footer_pad_lr_px: int = 48,
    footer_dpi: int = 300,
):
    """
    Save a bilevel (0/255) PNG from a numpy array, optionally adding centered
    footer text rendered with pyvips. Footer color adapts to invert mode.
    """
    if canvas.dtype != np.uint8:
        canvas = canvas.astype(np.uint8, copy=False)
    np.putmask(canvas, canvas > 0, 255)

    # invert BEFORE converting to vips so the text color logic can see it
    if invert:
        canvas = 255 - canvas

    H, W = canvas.shape
    base = vips.Image.new_from_memory(canvas.data, W, H, 1, "uchar")

    if footer_text:
        base = add_footer_label(
            base,
            footer_text,
            pad_lr_px=footer_pad_lr_px,
            dpi=footer_dpi,
            font="Courier New Bold 60",
            align="centre",
            invert=invert,  # <- pass inversion flag
        )

    base.write_to_file(
        out_path,
        compression=1,
        effort=1,
        filter="none",
        interlace=False,
        strip=True,
        bitdepth=1,
    )

def np_to_vips_gray_u8(arr: np.ndarray) -> vips.Image:
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    H, W = arr.shape
    return vips.Image.new_from_memory(arr.data, W, H, 1, "uchar")

def pad_to_square(im: vips.Image, px: int) -> vips.Image:
    """Center-pad a 1-band image to (px, px) with black background."""
    dx = max(0, (px - im.width) // 2)
    dy = max(0, (px - im.height) // 2)
    canvas = vips.Image.black(px, px)
    return canvas.insert(im, dx, dy)

def build_mosaic_streaming(
    spec_lines: list[str],
    cols: int,
    gap: int,
    invert: bool,
    footer: bool,
    footer_pad: int,
    footer_dpi: int,
    *,
    seed: int,
    stamp_chunk: int,
) -> vips.Image:
    """
    Stream tiles into a big canvas with draw_image, row-major order.
    All build_logo parameters come from spec lines. Runtime knobs (seed, stamp_chunk)
    are applied uniformly to all tiles.
    """
    if not spec_lines:
        raise ValueError("No specs provided for mosaic.")

    d0 = parse_logo_spec(spec_lines[0], LOGO_DEFAULTS)
    tile_px = int(d0["pix"])

    n = len(spec_lines)
    rows = math.ceil(n / cols)
    W = cols * tile_px + (cols - 1) * gap
    H = rows * tile_px + (rows - 1) * gap
    base = vips.Image.black(W, H)

    for i, spec in enumerate(spec_lines):
        d = parse_logo_spec(spec, LOGO_DEFAULTS)
        ns = dict_to_namespace(d)
        # apply runtime knobs (do NOT override spec-controlled things like drt/lmu/lsig)
        ns.seed = seed
        ns.stamp_chunk = stamp_chunk

        canvas = render_logo(ns)             # NumPy (0/255)
        vtile  = np_to_vips_gray_u8(canvas)  # VIPS 1-band

        if vtile.width != tile_px or vtile.height != tile_px:
            vtile = pad_to_square(vtile, tile_px)

        if footer:
            vtile = add_footer_label(
                vtile,
                logo_dict_to_string(d),
                pad_lr_px=footer_pad,
                dpi=footer_dpi,
                font="Courier New Bold 60",
                align="centre",
                invert=False,  # global invert later
            )

        r, c = divmod(i, cols)
        x = c * (tile_px + gap)
        y = r * (tile_px + gap)
        base = base.draw_image(vtile, x, y)

    base = (base > 0).ifthenelse(255, 0)
    if invert:
        base = base ^ 255
    return base


# ---------- CLI ----------

def build_parser():
    p = argparse.ArgumentParser(description="Generate a bilevel dotted swirl logo (spec-driven).")
    p.add_argument("--logo", type=str, required=True,
                   help="Logo spec 'k:v,...'. May include expandspec ranges.")
    # runtime knobs (not part of spec)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stamp-chunk", type=int, default=32_768)
    p.add_argument("--invert", action="store_true")
    p.add_argument("--out", type=str, default="logo.png")
    # footer
    p.add_argument("--footer", action="store_true", help="Add parameter string as footer text")
    p.add_argument("--footer-dpi", type=int, default=300)
    p.add_argument("--footer-pad", type=int, default=48)
    # mosaic layout (used only when expansions > 1)
    p.add_argument("--cols", type=int, default=5, help="Columns in mosaic")
    p.add_argument("--gap", type=int, default=20, help="Gap (px) between tiles in mosaic")
    return p

def main():
    ap = build_parser()
    args = ap.parse_args()

    # Expand spec (1 → single; >1 → mosaic)
    spec_lines = expandspec.expand_cartesian_lists(args.logo)

    if len(spec_lines) > 1:
        print(f"🧩 Building mosaic from {len(spec_lines)} specs ...")
        mosaic = build_mosaic_streaming(
            spec_lines=spec_lines,
            cols=args.cols,
            gap=args.gap,
            invert=args.invert,
            footer=args.footer,
            footer_pad=args.footer_pad,
            footer_dpi=args.footer_dpi,
            seed=args.seed,
            stamp_chunk=args.stamp_chunk,
        )
        mosaic.write_to_file(
            args.out,
            compression=1, effort=1, filter="none", interlace=False, strip=True, bitdepth=1,
        )
        print(f"✅ Saved mosaic {args.out}")
        return

    # Single image
    d = parse_logo_spec(spec_lines[0], LOGO_DEFAULTS)
    ns = dict_to_namespace(d)
    ns.seed = args.seed
    ns.stamp_chunk = args.stamp_chunk

    print("🎨 Rendering logo with parameters:")
    print(logo_dict_to_string(d))

    canvas = render_logo(ns)

    H, W = canvas.shape
    base = vips.Image.new_from_memory(canvas.data, W, H, 1, "uchar")

    if args.invert:
        base = base ^ 255

    if args.footer:
        base = add_footer_label(
            base,
            logo_dict_to_string(d),
            pad_lr_px=args.footer_pad,
            dpi=args.footer_dpi,
            font="Courier New Bold 60",
            align="centre",
            invert=args.invert,
        )

    base = (base > 0).ifthenelse(255, 0)
    base.write_to_file(
        args.out,
        compression=1, effort=1, filter="none", interlace=False, strip=True, bitdepth=1,
    )
    print(f"✅ Saved {args.out}")

if __name__ == "__main__":
    main()
