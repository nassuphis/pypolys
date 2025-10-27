# test_mosaic.py
#
# construct a mosaic of polynomials
#
import argparse
import numpy as np
import math
from numba import complex128
import time
import rasterizer                # rasterize results
import scanner                   # 
import compiler                  # 
import expandspec
import ast
import pyvips as vips

def grid_for(n, forced_cols=0):
    if forced_cols and forced_cols > 0:
        cols = forced_cols
        rows = (n + cols - 1) // cols
        return rows, cols
    cols = math.ceil(math.sqrt(n))
    rows = (n + cols - 1) // cols
    return rows, cols

def main():
    ap = argparse.ArgumentParser(description="Streaming mosaic maker (PNG, bilevel).")
    ap.add_argument("--chain", type=str, required=True,
                    help="Chain spec with optional {...} (e.g. 'p{1:50}', 'foo{1,3}bar').")
    ap.add_argument("--png", type=str, default="mosaic.png")
    ap.add_argument("--runs", type=str, default="100000", help="runs per tile")
    ap.add_argument("--px", type=str, default="5000", help="tile pixels (square)")
    ap.add_argument("--view", type=str, default="sq5", help="viewport alias or '(ll,ur)'")
    ap.add_argument("--cols", type=int, default=0, help="force number of columns")
    ap.add_argument("--header-pos", type=str, default="bottom", choices=["top","bottom"],
                    help="per-tile header placement")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    runs = int(ast.literal_eval(args.runs))
    px   = int(ast.literal_eval(args.px))
    chains = expandspec.expand_cartesian_lists(args.chain,names=scanner.ALLOWED)
    n = len(chains)
    rows, cols = grid_for(n, args.cols)

    llx, lly, urx, ury = rasterizer.str2view(args.view)
    print(f"viewport: ({llx},{lly})-({urx},{ury})  tiles: {n}  grid: {rows}x{cols}  px: {px}")

    # Streaming base canvas (HxW)
    H, W = rows * px, cols * px
    base = vips.Image.black(W, H)  # 1-band uchar; libvips streams

    t0 = time.perf_counter()
    for i, chain in enumerate(chains):
        if not scanner.chain_is_allowed(chain):
            print(f"⚠️  Skipping chain '{chain}' — contains unknown op")
            continue
        r, c = divmod(i, cols)
        if args.verbose:
            print(f"[{i+1}/{n}] {chain} -> ({r},{c})")

        # 1) Make tile (NumPy)
        tile = scanner.pixel_scan(chain, runs, px, complex(llx, lly), complex(urx, ury), False)
        tile = rasterizer.to_bilevel_uint8(tile)
        if tile.shape != (px, px):
            raise ValueError(f"tile shape {tile.shape} != {(px, px)}")

        # 2) NumPy -> Vips
        vtile = rasterizer.np_to_vips_gray_u8(tile)

        # 3) Paint per-tile header (IN-PLACE in tile area; size unchanged)
        vtile = rasterizer.add_header_label(
            vtile,
            header=chain,       # <- per-tile header is the chain string
            position="bottom",  # top or bottom
            font_family="Courier New", # your defaults
            font_weight="Bold",
            dpi=150
        )

        # 4) Composite tile into the big canvas at (x, y)
        x, y = c * px, r * px
        base = base.draw_image(vtile, x, y)

    # Binarize and write bilevel PNG (streaming, no giant NumPy)
    base = (base > 0).ifthenelse(255, 0)
    base.pngsave(
        args.png,
        compression=1,
        effort=1,
        filter="none",
        interlace=False,
        strip=True,
        bitdepth=1,
    )
    print(f"done in {time.perf_counter() - t0:.3f}s  -> {args.png}")

if __name__ == "__main__":
    main()


# FIX this:
#
# with relevant warm start
#python test_njit.py 
#--chain serp:1000000:0+0j:1+1j,uc,poly_giga_10,aberth 
#--runs 1000000 
#--px 5000 
#--view sq3 
#--png out_4.png
#scan time: 18.503s
#raster time: 0.590s
#save time: 0.032s
#
#
# with irrelevant warm start
# python test_njit.py 
# --chain runif,uc,poly_giga_10,aberth 
# --runs 1000000 
# --px 5000 
# --view sq3 
# --png out_4.png
#scan time: 18.663s
#raster time: 0.587s
#save time: 0.031s

# floating whales
# python test_njit.py --chain 'serp:runs:0+0j:1+1j,uc,uc,poly_giga_5,swirler,aberth' --runs 5e5 --px 5e3 --view sq4 --png out_4.png --calc pixels