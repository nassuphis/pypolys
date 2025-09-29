import argparse
import numpy as np
from numba import complex128
import time
import rasterizer                # rasterize results
import scanner                   # 


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", type=str, default="nop")
    ap.add_argument("--png", type=str, default="out.png")
    ap.add_argument("--runs", type=int, default=100_000, help="runs")
    ap.add_argument("--px", type=int, default=5_000, help="pixels")
    ap.add_argument("--view", type=str, default="sq5", help="view")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    res = scanner.scan(args.chain,args.runs, False)
    print(f"scan time: {time.perf_counter() - t0:.3f}s")

    llx, lly, urx, ury = rasterizer.view(res, args.view)

    t0 = time.perf_counter()
    raster = rasterizer.rasterize(res,llx,lly,urx,ury,args.px)
    print(f"raster time: {time.perf_counter() - t0:.3f}s")

    t0 = time.perf_counter()
    rasterizer.write_raster(raster,out=args.png)
    print(f"save time: {time.perf_counter() - t0:.3f}s")

if __name__ == "__main__":
    main()

