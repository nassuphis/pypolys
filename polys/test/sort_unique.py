#!/usr/bin/env python

import mlx.core as mx
import numpy as np
import time
import gc;
start = time.perf_counter()  

def mlx_which(x: mx.array,*,stream=mx.gpu):
     if x.size == 0:
        return mx.zeros((0,), dtype=mx.int32, stream=stream)
     xb  = x.astype(mx.int32)
     xb[0] = 0
     ar = mx.arange(x.size, dtype=mx.int32, stream=stream)
     xi = mx.cummax(ar * xb)
     cum = mx.cumsum(xb, stream=stream)
     maxi = cum[-1].item() + 1
     idx = mx.zeros((maxi,), dtype=mx.int32, stream=stream)
     idx[cum]=xi
     if x[0]:
        return idx
     return idx[1:]

def sort_unique_uint16x4_mlx(x: np.array,*,stream=mx.gpu): 
    if  x.ndim != 2 or x.shape[1] != 4:
        raise ValueError("expect (N,4) uint16 array")
    x_u16 = x.view(np.uint16)
    arr = mx.array(x_u16, dtype=mx.uint16)
    a64  = arr.astype(mx.uint64)
    keys = (a64[:, 0] << 48) | (a64[:, 1] << 32) | (a64[:, 2] << 16) |  a64[:, 3]
    order       = mx.argsort(keys, axis=0, stream=stream)
    rows_sorted = arr[order]
    keys_sorted = keys[order]
    first = mx.concatenate(
        [mx.array([True], dtype=mx.bool_), keys_sorted[1:] != keys_sorted[:-1]],
        axis=0
    )
    idx = mlx_which(first, stream=stream)
    unique_rows = rows_sorted[idx]
    mx.eval(unique_rows)
    result = np.array(unique_rows)
    return result


print(f" which: {mlx_which(mx.array([False, False, False], dtype=mx.bool_))}")
print(f" which: {mlx_which(mx.array([False, True, False, True], dtype=mx.bool_))}")
print(f" which: {mlx_which(mx.array([True, True, False, True], dtype=mx.bool_))}")


print(f"Start {time.perf_counter() - start:.2f} sec")
results_z = np.load('myresult.npz')
results = results_z['arr_0']#[:10000000,]
print(f"Loaded {time.perf_counter() - start:.2f} sec")
print(f"Results {results.dtype} {results.shape}")

sorted_np = sort_unique_uint16x4_mlx(results)

print(f"Results {sorted_np.dtype} {sorted_np.shape}")

#del mx_arr
#del sorted_arr
#gc.collect()

print(f"Sorted {time.perf_counter() - start:.2f} sec")


# Random check: verify if random rows from results are in sorted_np (uniques) using binary search on keys
num_tests = 1000
rng = np.random.default_rng()          # uses PCG64 by default
random_indices = rng.choice(results.shape[0], size=num_tests, replace=False)
print(f"Make Keys {time.perf_counter() - start:.2f} sec")
# Compute sorted unique keys once
sorted_u64 = sorted_np.astype(np.uint64)
print(f"Make Keys Cast:{time.perf_counter() - start:.2f} sec")
unique_keys = (sorted_u64[:, 0] << 48) | (sorted_u64[:, 1] << 32) | (sorted_u64[:, 2] << 16) | sorted_u64[:, 3]
print(f"Make Keys DOne {time.perf_counter() - start:.2f} sec")
print(f"Checking {time.perf_counter() - start:.2f} sec")
found_count = 0
for idx in random_indices:
    row = results[idx]
    row_u16 = row.view(np.uint16)
    row_u64 = row_u16.astype(np.uint64)
    query_key = (row_u64[0] << 48) | (row_u64[1] << 32) | (row_u64[2] << 16) | row_u64[3]
    
    pos = np.searchsorted(unique_keys, query_key)
    if pos < len(unique_keys) and unique_keys[pos] == query_key:
        found_count += 1
    else:
        print(f"Row at index {idx} not found in uniques!")

print(f"Random check: {found_count}/{num_tests} rows found in uniques.")


print(f"min diff: {np.min(np.diff(sorted_np[:,0]))}")
print(f"min diff: {np.min(np.diff(results[:,0]))}")

x=np.concatenate([np.arange(10),np.arange(10)]).astype(np.uint16)
y=np.stack([x,x,x,x],axis=-1).astype(np.uint16)
tst1 = np.all(sort_unique_uint16x4_mlx(y)==np.unique(y,axis=0))

print(f"test1 : {tst1} {time.perf_counter() - start:.2f} sec")

# this is true but slow!
#tst2 = np.all(sort_unique_uint16x4_mlx(sorted_np)==np.unique(results,axis=0))
#print(f"test2 : {tst2} {time.perf_counter() - start:.2f} sec")

print(f"Done! {time.perf_counter() - start:.2f} sec")

