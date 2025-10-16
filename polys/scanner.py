# scanner.py
# poly manifold generator
from __future__ import annotations
import math
import numpy as np
from numba import types
from numba.typed import Dict
import multiprocessing as mproc
from multiprocessing.shared_memory import SharedMemory
import compiler           # compile to njit
import ops_basic          # basic stuff
import ops_poly           # poly stuff
import ops_coeff          # parameter moebius transforms
import ops_zfrm           # coefficient transforms
import ops_xfrm           # miscellaneous parameter transforms
import ops_tfrm           # term-by-term transforms
import ops_rfrm           # root display transforms
import ops_sort           # coefficient sortin transforms
import ops_param          # parameter generation
import argparse
import time

# =======================
# compiler OPCODES
# =======================

ALLOWED =  ( # commands
    ops_poly.ALLOWED  | 
    ops_basic.ALLOWED | 
    ops_coeff.ALLOWED | 
    ops_zfrm.ALLOWED  |
    ops_xfrm.ALLOWED  |
    ops_tfrm.ALLOWED  |
    ops_sort.ALLOWED  |
    ops_param.ALLOWED |
    ops_rfrm.ALLOWED
)  

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
# pixel scanner worker
# =======================

def _pixel_scan_worker(args):

    ( 
        chain, 
        wid, 
        wcount, 
        shm_pixels_name, 
        samples, 
        pixels, 
        ll, 
        ur, 
        verbose 
    ) = args

    if verbose:
        print(f"worker {wid}/{wcount}")

    shm_result, result = get_shm( shm_pixels_name, pixels, pixels, np.uint8)

    if verbose:
        print(f"worker {wid}/{wcount} shared mem: {result.shape}")

    compiler.set_const("runs",samples)
    APPLY_PROGRAM, opcodes, a = compiler.compile_chain(chain, ALLOWED)

    state  = Dict.empty(key_type=types.int8,value_type=types.complex128[:])

    for i in range(wid, samples, wcount):
        z=np.array([i],dtype=np.complex128)
        res = APPLY_PROGRAM(z, a, state, opcodes)
        for r in res:
            if r.real < ll.real: continue
            if r.real > ur.real: continue
            if r.imag < ll.imag: continue
            if r.imag > ur.imag: continue
            x=(r.real-ll.real)/(ur.real-ll.real)
            y=(r.imag-ll.imag)/(ur.imag-ll.imag)
            if np.any(np.isnan(x)): continue
            if np.any(np.isnan(y)): continue
            ic = min(max(int(x*pixels),0),pixels-1)
            jc = min(max(int(y*pixels),0),pixels-1)
            result[jc,ic]=255

    shm_result.close()
    return

# =======================
# pixel scanner
# =======================

def pixel_scan( 
        chain: str, 
        runs: int, 
        pixels: int, 
        ll:complex, 
        ur:complex, 
        verbose: bool = False
) -> np.ndarray:

    compiler.set_const("runs",runs)
    APPLY_PROGRAM, opcodes, a = compiler.compile_chain(chain, ALLOWED)

    if verbose:
        print(f"opcodes {opcodes}")

    z=np.array([0],dtype=np.complex128)
    state  = Dict.empty(key_type=types.int8,value_type=types.complex128[:])
    first = APPLY_PROGRAM(z, a, state, opcodes)
    cols = first.size

    if verbose:
        print(f"first.size {cols}")

    shm_result, result = make_shm(pixels, pixels, np.uint8)

    if verbose:
        print(f"result shape: {result.shape}")

    ctx = mproc.get_context("spawn")
    ncpu = min(mproc.cpu_count(),runs)
    args = []

    for wid in range(ncpu):
        args.append(( chain, wid, ncpu, shm_result.name, runs, pixels, ll, ur, verbose ))

    with ctx.Pool(processes=len(args)) as pool:
        _ = pool.map(_pixel_scan_worker, args)

    out = np.copy(result)
    shm_result.close()
    shm_result.unlink()

    return out

# =======================
# root scanner worker
# =======================

def _scan_worker(args):
    ( 
        chain, 
        wid, 
        wcount, 
        shm_roots_name, 
        rows, 
        cols, 
        verbose 
    ) = args

    if verbose:
        print(f"worker {wid}/{wcount}")

    shm_result, result = get_shm( shm_roots_name, rows, cols, np.complex128)

    if verbose:
        print(f"worker {wid}/{wcount} shared mem: {result.shape}")

    compiler.set_const("runs",rows)
    APPLY_PROGRAM, opcodes, a = compiler.compile_chain(chain, ALLOWED)

    state  = Dict.empty(key_type=types.int8,value_type=types.complex128[:])
    for i in range(wid, rows, wcount):
        z=np.array([i],dtype=np.complex128)
        res = APPLY_PROGRAM(z, a, state, opcodes)
        if res.size == cols:
            result[i] = res
            if verbose:
                print(f"worker {wid}/{wcount} row {i} computed")
        else:
            result[i] = 0
    shm_result.close()
    return

# =======================
# root scanner
# =======================

def scan( chain: str, runs: int, verbose: bool = False) -> np.ndarray:

    compiler.set_const("runs",runs)
    APPLY_PROGRAM, opcodes, a = compiler.compile_chain(chain, ALLOWED)

    if verbose:
        print(f"opcodes {opcodes}")

    z=np.array([0],dtype=np.complex128)
    state  = Dict.empty(key_type=types.int8,value_type=types.complex128[:])
    first = APPLY_PROGRAM(z, a, state, opcodes)
    cols = first.size

    if verbose:
        print(f"first.size {cols}")

    shm_result, result = make_shm(runs, cols, np.complex128)

    if verbose:
        print(f"result shape: {result.shape}")

    ctx = mproc.get_context("spawn")
    ncpu = min(mproc.cpu_count(),runs)
    args = []

    for wid in range(ncpu):
        args.append(( chain, wid, ncpu, shm_result.name, runs, cols, verbose ))

    with ctx.Pool(processes=len(args)) as pool:
        _ = pool.map(_scan_worker, args)

    out = np.copy(result)
    shm_result.close()
    shm_result.unlink()

    return out


if __name__ == "__main__":

    ap = argparse.ArgumentParser(description="Canner")
    ap.add_argument("--chain", type=str, default="runif")
    ap.add_argument("--runs", type=int, default=100, help="runs")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    res = scan( args.chain, args.runs, args.verbose )
    print(f"scan time: {time.perf_counter() - t0:.3f}s")

    print(f"result shape: {res.shape}")
    print(f"first row: {res[0,:]}")
    print(f"last: {res[args.runs-1,:]}")

