#!/usr/bin/env python
import mlx.core as mx
import numpy as np
import math
import time
start = time.perf_counter()  

the_dtype = np.float32

def batch_companion_roots(coeffs):
        coeffs = np.array(coeffs, dtype=the_dtype)
        n = len(coeffs) - 1
        if n == 0:
            return np.array([])
        # Normalize to make monic
        lead = coeffs[0]
        if lead == 0:
            raise ValueError("Leading coefficient cannot be zero")
        coeffs = coeffs / lead
        # Build companion matrix
        C = np.diag(np.ones(n-1,dtype=the_dtype), 1)  # Superdiagonal 1's
        b = coeffs[1:]
        last_row = -np.flip(b)  # Reverse and negate
        C[-1, :] = last_row
        # Compute eigenvalues (roots)
        MxC = mx.array(C,dtype=mx.float32)
        eigvals, _ = mx.linalg.eig(MxC, stream=mx.cpu)
        return np.array(eigvals)


def companion_roots(coeffs):
    coeffs = np.array(coeffs, dtype=the_dtype)
    n = len(coeffs) - 1
    if n == 0:
        return np.array([])
    # Normalize to make monic
    lead = coeffs[0]
    if lead == 0:
        raise ValueError("Leading coefficient cannot be zero")
    coeffs = coeffs / lead
    # Build companion matrix
    C = np.diag(np.ones(n-1,dtype=the_dtype), 1)  # Superdiagonal 1's
    b = coeffs[1:]
    last_row = -np.flip(b)  # Reverse and negate
    C[-1, :] = last_row
    # Compute eigenvalues (roots)
    roots = np.linalg.eigvals(C)
    return roots

# Example usage and accuracy check
if __name__ == "__main__":
    # Test polynomial: x^2 - 3x + 2 = 0 (roots: 1, 2)
    n=500
    print(f"degree: {n}")
    coeffs = np.random.randn(n).astype(the_dtype)
    #
    start = time.perf_counter()  
    my_roots = companion_roots(coeffs)
    print(f"my time: {round((time.perf_counter() - start) * 1000)} ms")
    #
    start = time.perf_counter()  
    my_roots_mlx = batch_companion_roots(coeffs)
    print(f"mlx time: {round((time.perf_counter() - start) * 1000)} ms")
    #
    start = time.perf_counter()  
    np_roots = np.roots(coeffs)
    print(f"np time: {round((time.perf_counter() - start) * 1000)} ms")
    my_sorted = np.sort(my_roots)
    mlx_sorted = np.sort(my_roots_mlx)
    np_sorted = np.sort(np_roots)
    print(f"size: {my_roots.size}")
    print(f"max(abs(my-mlx)): {np.max(np.abs(my_sorted - mlx_sorted))}")
    print(f"max(abs(my-np)): {np.max(np.abs(my_sorted - np_sorted))}")

    if False:
        print(f"np {np_roots}")
        print(f"sort {np_sorted}")
        print(f"diff {np_sorted-np_sorted}")
        print("-"*40)
        print(f"my {my_roots}")
        print(f"sort {my_sorted}")
        print(f"diff {my_sorted-np_sorted}")
        print("-"*40)
        print(f"mlx {my_roots_mlx}")
        print(f"sort {mlx_sorted}")
        print(f"diff {mlx_sorted-np_sorted}")

    

