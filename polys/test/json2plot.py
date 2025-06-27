#!/usr/bin/env python

import polys
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image, ImageOps
import multiprocessing

def norm(x):
     return (x-np.min(x))/(np.max(x)-np.min(x))

def sample_chunk(args):
    start, stop, js_config, worker_id = args
    #import polys  # (re-)import inside subprocess if needed
    polys.polystate.json2state(js_config)
    np.random.seed(worker_id) 
    chunk = []
    for j in range(start, stop):
        if j % 1000 ==0 and worker_id==0:
            print(f"{worker_id} : {round(100*(j-start)/(stop-start),2)}%")
        t1 = np.random.random()
        t2 = np.random.random()
        rts = polys.polystate.sample(t1, t2)
        for r in rts:
            chunk.append({'root': r, 't1': t1, 't2': t2})
    return pd.DataFrame(chunk)

if __name__ == "__main__":

    if not sys.stdin.isatty():  # stdin is *not* from terminal → it's piped
        js_config = sys.stdin.read()
            
    num_samples = int(sys.argv[1])
    num_workers = multiprocessing.cpu_count() * 2
    print(f"CPU count : {num_workers}")
    chunk_size = num_samples // num_workers
    chunks = []
    for i in range(num_workers):
        start = i * chunk_size
        stop = (i+1) * chunk_size if i < num_workers-1 else num_samples
        chunks.append((start, stop,js_config,i))

    all_data = []

    print("Start")
    with multiprocessing.Pool(num_workers) as pool:
        all_dfs = pool.map(sample_chunk, chunks)
    print("End")

    df0 = pd.concat(all_dfs, ignore_index=True)
    print(f"Data : {df0.shape}") 
    d = np.abs(df0['root'].values)
    dq = 10 # min(np.quantile(d,0.99),2.0)
    print("Quantile")
    #df = df0[d < dq]
    df=df0
    t1 = df['t1'].values 
    t2 = df['t2'].values 
    t = df['t1'].values + 1j * df['t2'].values
    v = df['root'].values
    df['phase'] = norm(np.angle(v))
    df['mod'] = norm(np.abs(v)) 
    df['hue'] = (((df['phase']*10)%1)*0.5 - 0.25) % 1
    df['sat'] = (((df['mod']*10)%1)*0.5 - 0.25) % 1
    print("Hue, Saturation")

    px=5000
    roots = px*(norm(df['root'].values.real)+1j*norm(df['root'].values.imag))


    print(f"roots.real: {np.min(roots.real)} - {np.max(roots.real)}")
    print(f"roots.imag: {np.min(roots.imag)} - {np.max(roots.imag)}")
    print(f"t1: {np.min(df['t1'])} - {np.max(df['t1'])}")
    print(f"t2: {np.min(df['t2'])} - {np.max(df['t2'])}")
    print(f"phase: {np.min(df['phase'])} - {np.max(df['phase'])}")
    print(f"hue: {np.min(df['hue'])} - {np.max(df['hue'])}")

    hues = df['hue'].values.astype(float)
    sats = df['sat'].values.astype(float)

    colors = mcolors.hsv_to_rgb(np.column_stack([hues, sats, np.ones_like(hues)]))
    #x = np.clip(roots.imag.astype(int), 0, px-1)
    #y = np.clip(roots.real.astype(int), 0, px-1)

    x = np.clip((t1*px).astype(int), 0, px-1)
    y = np.clip((t2*px).astype(int), 0, px-1)

    img = np.zeros((px, px, 3), dtype=np.uint8)
    img[y, x] = (colors * 255).astype(np.uint8)

    im = Image.fromarray(img)
    im_inv = ImageOps.invert(im)
    im_inv.save('myplot.png')

    



    
