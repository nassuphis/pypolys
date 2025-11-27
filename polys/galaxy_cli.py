#!/usr/bin/env python
# galaxy_cli.py — chain-driven swirly galaxy renderer (single or mosaic)

import sys
sys.path.insert(0, "/Users/nicknassuphis")
import argparse, math, time
import numpy as np
import math
import galaxy
from rasterizer import raster
from specparser import specparser
from specparser import expandspec
from numba import njit, prange

BUCKET_METHOD = "parallel"


# ---------- render one chain spec into a tile ----------

def render_chain_tile(
        spec: str, 
        pix: int, 
        fos: float, 
        rmin: int,
        margin_frac: float, 
        verbose: bool=False
    ) -> np.ndarray:
    if verbose:
        print(f"[render] spec: {spec}")
    t0 = time.perf_counter()

    canvas = np.zeros((int(pix), int(pix)), np.uint8)
    specparser.set_const("pix",0.51/(fos*(pix-1)))
    specparser.set_const("fos",fos)
    z, mult = galaxy.build_logo_from_chain(spec)
    if z.size<1: 
        if verbose: print(f"[render] z.size<0")
        return canvas
    
    px,py = raster.project_to_canvas(z,pix,margin_frac)

    r_px = np.rint(mult * fos * (int(pix) - 1)).astype(np.int32)
    keep = r_px >= rmin
    kept = int(keep.sum())
    if kept == 0:
        if verbose: print(f"[render] kept=0 points after threshold (rmin={rmin})")
        return canvas

    px, py, r_px = px[keep], py[keep], r_px[keep]
    r_max = int(r_px.max())

    t1 = time.perf_counter()
    if BUCKET_METHOD=="serial":
        order, r_vals, starts, counts = galaxy_raster.bucket_by_radius(r_px, rmin, r_max)
    elif BUCKET_METHOD=="parallel":
        order, r_vals, starts, counts = galaxy_raster.bucket_by_radius_parallel(r_px, rmin, r_max)
    else: 
        raise ValueError(f"invalid BUCKET_METHOD: {BUCKET_METHOD}")
    
    pxs = px[order]
    pys = py[order]
    t2 = time.perf_counter()

    if verbose:
        print((
            "[render] "
            f"N={z.size:,} "
            f"ptp(abs(z))={np.ptp(np.abs(z))} "
            f"mean(abs(z))={np.mean(np.abs(z))} "
            f"kept={kept:,} "
            f"groups={r_vals.size} "
            f"rmax={r_max} "
            f"bucket={t2-t1:.3f}s "
            f"geometry={t1-t0:.3f}s"
        ))

    # stamp per radius
    for rr, s, c in zip(r_vals, starts, counts):
        if c <= 0:
            continue
        e = s + c
        dy, dx = galaxy_raster.make_disc_offsets(int(rr))
        step = 32768
        for i in range(s, e, step):
            j = min(i + step, e)
            galaxy_raster.stamp_points(canvas, pys[i:j], pxs[i:j], dy, dx)
    if verbose:
        print(f"[render] stamp complete ({time.perf_counter()-t2:.3f}s total={time.perf_counter()-t0:.3f}s)")
    return canvas

# ---------- mosaic ----------

def mosaic(tiles: list[np.ndarray], cols: int, gap: int) -> np.ndarray:
    if not tiles:
        raise ValueError("no tiles")
    th, tw = tiles[0].shape
    rows = math.ceil(len(tiles) / cols)
    H = rows * th + (rows - 1) * gap
    W = cols * tw + (cols - 1) * gap
    out = np.zeros((H, W), np.uint8)
    for idx, tile in enumerate(tiles):
        r, c = divmod(idx, cols)
        y = r * (th + gap); x = c * (tw + gap)
        out[y:y+th, x:x+tw] = tile
    return out

# ---------- WARMUP ----------

def warmup_numpy_path(N=1_000_000):
    # Use the *same dtype & ops* as your real call to force the exact code path:
    u1 = np.random.random(N)            # float64
    u2 = np.random.random(N)
    r  = u1 ** 0.75                     # power
    th = 2.0 * np.pi * u2
    _  = np.cos(th); _ = np.sin(th)     # trig
    mult = np.random.normal(0.0, 1.0, N)
    mult = np.exp(mult)                 # exp
    np.clip(mult, 1.0, 3000.0, out=mult)
    # drop references; goal was to bind ufunc loops + fault pages once

def warmup_pages(n=1_000_000):
    # match the dtypes you actually use
    np.empty(n, np.float64).fill(0.0)     # x
    np.empty(n, np.float64).fill(0.0)     # y
    np.empty(n, np.float32).fill(0.0)     # zeta/mult (if float32)

# ---------- CLI ----------

def build_parser():
    p = argparse.ArgumentParser("galaxy-cli", description="Chain-driven galaxy renderer")
    p.add_argument("--chain", required=True,
                   help="Pipeline spec (can include expandspec braces)")
    p.add_argument("--pix", type=int, default=25000, help="Tile width/height in pixels (default 25000)")
    p.add_argument("--out", type=str, default="logo.png", help="Output PNG path")
    p.add_argument("--cols", type=int, default=None, help="Columns if chain expands to multiple tiles")
    p.add_argument("--rows", type=int, default=None, help="Rows if chain expands to multiple tiles")
    p.add_argument("--thumb",type=int, default=None,  help="Save thumbnail")
    p.add_argument("--gap", type=int, default=20, help="Gap between tiles in mosaic")
    p.add_argument("--invert", action="store_true", help="Invert black/white")
    p.add_argument("--pasp", action="store_true", help="Add passepartout")
    p.add_argument("--fos", type=float, default=1e-5, help="Size scale: pixel radius = mult * fos * (pix-1)")
    p.add_argument("--min", dest="rmin", type=int, default=1, help="Minimum visible integer radius (px)")
    p.add_argument("--margin", type=float, default=0.0, help="Logical margin fraction around geometry")
    p.add_argument("--bucket", type=str, default="parallel",choices=["serial","parallel"],help="Bucketing Method")
    p.add_argument("--verbose", "-v", action="store_true", help="Print detailed progress")
    p.add_argument("--explain", "-e", action="store_true", help="Make horizontal mosaic of pipeline steps")
    p.add_argument("--footer", action="store_true", help="Render the spec string as a footer title")
    p.add_argument("--footer-dpi", type=int, default=300, help="Footer text DPI (default 300)")
    p.add_argument("--footer-pad", type=int, default=48, help="Left/right padding for footer text")
    p.add_argument(
        "--const", action="append", default=[],
        help="Add/override constant as NAME=VALUE (VALUE parsed like args). Repeatable."
    )
    return p

def main():
    global BUCKET_METHOD
    ap = build_parser()
    args = ap.parse_args()
    BUCKET_METHOD = args.bucket

    for kv in args.const:
        k, v = specparser._parse_const_kv(kv)
        specparser.set_const(k, v)
    specparser.set_const("pix",0.51/(args.fos*(args.pix-1)))

    if args.explain: # "explain" mode: show your work
        chain = f">[{args.chain}],dot:pix"
        args.cols=None
        args.rows=1
        args.margin=0.1
        args.footer=True
        specs = expandspec.expand_cartesian_lists(chain)
    else:
        specs = expandspec.expand_cartesian_lists(args.chain)

    if not specs:
        raise SystemExit("No specs produced by expandspec")

    if args.verbose:
        print(f"[main] expanded {len(specs)} chain(s):")
        for s in specs:
            print("   ", s)

    tiles = [render_chain_tile(s, args.pix, args.fos, args.rmin, args.margin, args.verbose) for s in specs]

    if len(tiles) == 1:
        footer = args.chain if args.footer else None
        raster.save_png_bilevel(
            tiles[0], args.out, args.invert,
            footer_text=footer,
            footer_pad_lr_px=args.footer_pad,
            footer_dpi=args.footer_dpi,
            passepartout=args.pasp
        )
        if args.verbose:
            print(f"[main] Saved {args.out} ({args.pix}×{args.pix})")
        else:
            print(f"✅ Saved {args.out}")
        return

    n = len(tiles)
    if args.cols:
        cols = args.cols
    elif args.rows:
        cols = int(round(n / args.rows))
    else: 
        cols = max(1, int(round(math.sqrt(n))))
    titles = specs if args.footer else None  # per-tile footer = spec line
    raster.save_mosaic_png_bilevel(
        tiles, titles,
        cols=cols, gap=args.gap,
        out_path=args.out, invert=args.invert,
        footer_pad_lr_px=args.footer_pad,
        footer_dpi=args.footer_dpi,
        thumbnail=args.thumb,
    )
    rows = math.ceil(n / cols)
    print(f"✅ Saved mosaic {args.out}  ({cols}×{rows}, tiles={n})")
 
 # yin: rud:10e5,sclip:1:1+0j,dclip:0.5:0+0.5j,rua:1.25e5:0.75+1.25j:0.25:0-0.5j,add:0.5j,dot:pix
if __name__ == "__main__":
    main()