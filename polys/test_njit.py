import argparse
import numpy as np
from numba import complex128
import time
import rasterizer                # rasterize results
import scanner                   # 
import ast

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", type=str, default="nop")
    ap.add_argument("--png", type=str, default="out.png")
    ap.add_argument("--runs", type=str, default="100000", help="runs")
    ap.add_argument("--px", type=str, default="5000", help="pixels")
    ap.add_argument("--view", type=str, default="sq5", help="view")
    ap.add_argument(
        "--calc",
        type=str,
        choices=["roots", "pixels"],
        help="Calculation"
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    runs = int(ast.literal_eval(args.runs))
    px = int(ast.literal_eval(args.px))

    llx, lly, urx, ury = rasterizer.str2view(args.view)
    
    print(f"({llx},{lly})-({urx},{ury})")
    
    if args.calc=="pixels":
        print("pixel scanner")
        t0 = time.perf_counter()
        print(f"runs {runs:,}")
        raster = scanner.pixel_scan(args.chain,runs, px, complex(llx,lly), complex(urx, ury), False)
        print(f"raster {raster.shape[0]} by {raster.shape[1]}")
        print(f"scan time: {time.perf_counter() - t0:.3f}s")
        t0 = time.perf_counter()
        rasterizer.write_raster_header(
            raster,
            out=args.png,
            header=args.chain,
            font_family="Courier New",
            font_weight="Bold",
            position="bottom"
        )
        print(f"save time: {time.perf_counter() - t0:.3f}s")
    elif args.calc=="roots":
        print("root scanner")
        t0 = time.perf_counter()
        res = scanner.scan(args.chain,runs, False)
        print(f"scan time: {time.perf_counter() - t0:.3f}s")
        t0 = time.perf_counter()
        raster = rasterizer.rasterize(res,llx,lly,urx,ury,px)
        print(f"raster time: {time.perf_counter() - t0:.3f}s")
        t0 = time.perf_counter()
        rasterizer.write_raster_header(
            raster,
            out=args.png,
            header=args.chain,
            font_family="Courier New",
            font_weight="Bold"
        )
        print(f"save time: {time.perf_counter() - t0:.3f}s")

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