#!/usr/bin/env python

# test/cli2json.py test123 -x unit_circle,coeff7 -p poly_362 -z rev -s solve  | test/json2plot.py 10000
# test/cli2json.py  mps -m write -x identity -p poly_giga_143 -z none -s solve --roots 10 -r 20000 -v '(-25-50j,75+50j)' | test/json2plot.py 100000000
import polys
import sys
import numpy as np
import pandas as pd
import matplotlib
import ast
import polys.polystate
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
    blocks = []
    for j in range(start, stop):
        if j % ((start-stop)//100) ==0 and worker_id==0:
            print(f"{worker_id} : {round(100*(j-start)/(stop-start),2)}% {stop-start}")
        t1 = np.random.random()
        t2 = np.random.random()
        rts = polys.polystate.sample(t1, t2)
        real = rts.real
        imag = rts.imag
        t1s = np.full_like(real, t1)
        t2s = np.full_like(real, t2)
        size = np.abs(rts)
        rng = np.max(size)-np.min(size)
        rngs = np.full_like(real,rng)
        ngls = np.full_like(real,np.mean(np.angle(rts)))
        row_block = np.column_stack([real, imag, t1s, t2s,rngs,ngls])
        blocks.append(row_block)
    
    if blocks:
        return np.vstack(blocks)
    else:
        return np.empty((0, 6))

if __name__ == "__main__":

    if not sys.stdin.isatty():  # stdin is *not* from terminal → it's piped
        js_config = sys.stdin.read()
            
    polys.polystate.json2state(js_config)
    sv = polys.polystate.view["view"]
    print(f"sv:{sv}")
    llx, lly, urx, ury = sv[0].real, sv[0].imag, sv[1].real, sv[1].imag

    print(f"llx:{llx} lly:{lly} urx:{urx} ury:{ury}")

    num_samples = int(sys.argv[1])
    num_workers = multiprocessing.cpu_count()
    print(f"CPU count : {num_workers}")
    chunk_size = num_samples // num_workers
    chunks = []
    for i in range(num_workers):
        start = i * chunk_size
        stop = (i+1) * chunk_size if i < num_workers-1 else num_samples
        chunks.append((start, stop,js_config,i))

    all = []

    print("Start")
    with multiprocessing.Pool(num_workers) as pool:
        all = pool.map(sample_chunk, chunks)
    print("End")

    m= np.vstack(all)

   
    mask =  (m[:,0]>llx) & (m[:,0]<urx) & (m[:,1]>lly) & (m[:,1]<ury)

    m = m[mask]

    print(f"Data : {m.shape}") 
    v1 = m[:,0]
    v2 = m[:,1]
    v = v1 + 1j * v2
    d = np.abs(v)
    t1 = m[:,2]
    t2 = m[:,3] 
    t = t1 + 1j * t2
    rng = norm(m[:,4])
    ngl = norm(m[:,5])
    phase = norm(np.angle(v))
    mod = norm(np.abs(v)) 
    hue = (((rng*1.0)%1)*0.5 + 0.01) % 1
    sat = (((ngl*10.0)%1)*0.99) % 1

    px=20000
    roots = px*((v1-llx)/(urx-llx)+1j*(v2-lly)/(ury-lly))


    print(f"roots.real: {np.min(roots.real)} - {np.max(roots.real)}")
    print(f"roots.imag: {np.min(roots.imag)} - {np.max(roots.imag)}")
    print(f"t1: {np.min(t1)} - {np.max(t1)}")
    print(f"t2: {np.min(t2)} - {np.max(t2)}")
    print(f"phase: {np.min(phase)} - {np.max(phase)}")
    print(f"mod: {np.min(mod)} - {np.max(mod)}")
    print(f"rng: {np.min(rng)} - {np.max(rng)}")
    print(f"ngl: {np.min(ngl)} - {np.max(ngl)}")
    print(f"sat: {np.min(sat)} - {np.max(sat)}")
    print(f"hue: {np.min(hue)} - {np.max(hue)}")

    colors = mcolors.hsv_to_rgb(np.column_stack([hue, sat, np.ones_like(hue)]))
    x = np.clip(roots.imag.astype(int), 0, px-1)
    y = np.clip(roots.real.astype(int), 0, px-1)

    #x = np.clip((t1*px).astype(int), 0, px-1)
    #y = np.clip((t2*px).astype(int), 0, px-1)

    img = np.zeros((px, px, 3), dtype=np.uint8)
    img[y, x] = (colors * 255).astype(np.uint8)

    im = Image.fromarray(img)
    im_inv = ImageOps.invert(im)
    im.save('myplot.png')
    im_inv.save('myplot_inv.png')

    



    
