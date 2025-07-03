#!/usr/bin/env python

# test/cli2json.py test123 -x unit_circle,coeff7 -p poly_362 -z rev -s solve  | test/json2plot.py 10000

import polys
import polys.polystate as ps
import sys
import numpy as np
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
import time
import os
import pyvips
import cv2

palette_ratio = 50
sq = 3.0
arw = 0.9
hfac = 0.85
hcst = 0.1
sfac = 5.0
scst = 0.01

def histeq(V):
    # V should be float32, range 0-1
    V_flat = (V * 255).astype(np.uint8).ravel()
    hist = np.bincount(V_flat, minlength=256)
    cdf = hist.cumsum()
    cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min())
    V_eq = cdf[V_flat].reshape(V.shape)
    return V_eq.astype(np.float32)

def hsv_to_rgb_numpy(hsv):
    # hsv: (..., 3) array, H in [0,1], S in [0,1], V in [0,1]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6).astype(int)
    f = (h * 6) - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i = i % 6

    conditions = [i == k for k in range(6)]
    rgb = np.zeros(hsv.shape)
    rgb[conditions[0]] = np.stack([v, t, p], axis=-1)[conditions[0]]
    rgb[conditions[1]] = np.stack([q, v, p], axis=-1)[conditions[1]]
    rgb[conditions[2]] = np.stack([p, v, t], axis=-1)[conditions[2]]
    rgb[conditions[3]] = np.stack([p, q, v], axis=-1)[conditions[3]]
    rgb[conditions[4]] = np.stack([t, p, v], axis=-1)[conditions[4]]
    rgb[conditions[5]] = np.stack([v, p, q], axis=-1)[conditions[5]]
    return rgb

def save2rgb(hsv,fn):
    rgb_uint8 = (255*hsv_to_rgb_numpy(hsv)).astype(np.uint8)    
    im = pyvips.Image.new_from_memory(
        rgb_uint8.tobytes(),
        rgb_uint8.shape[1],  # width
        rgb_uint8.shape[0],  # height
        rgb_uint8.shape[2],  # bands (should be 3 for RGB)
        'uchar'
    )            
    im.write_to_file(fn)


def norm(x):
     max_x = np.max(x)
     min_x = np.min(x)
     if max_x-min_x>0:
        return (x-min_x)/(max_x-min_x).astype(np.float32)
     else:
         return x.astype(np.float32)

def make_shm(size,type):
    shm = SharedMemory(
        create=True, 
        size = size * size * np.dtype(type).itemsize
    )
    array = np.ndarray(
        (size,size), 
        dtype=type, 
        buffer=shm.buf
    )
    array[:] = 0
    return (shm,array)

def get_shm(name,size,type):
    shm = SharedMemory(name=name)
    array = np.ndarray(
        (size, size), 
        dtype=type, 
        buffer=shm.buf
    )
    return(shm,array)

def sample_chunk(args):
    worker_id, num_workers, js_config, count_name, hue_name, sat_name, thue_name, tsat_name = args
    polys.polystate.json2state(js_config)
    px = ps.view["res"]
    arw = ps.poly["arw"]
    degree = ps.poly["degree"]
    samples = ps.view["samples"]
    rows = np.arange(worker_id,px,num_workers)    
    ll,ur = ps.view["view"]
    llr, lli, urr, uri = ll.real, ll.imag, ur.real, ur.imag
    rgr = urr - llr
    rgi = uri - lli

    shm_count, count = get_shm(count_name,px,np.float32)
    shm_sat, sat = get_shm(sat_name,px,np.float32)
    shm_hue, hue = get_shm(hue_name,px,np.float32)
    shm_tsat, tsat = get_shm(tsat_name,int(px/palette_ratio),np.float32)
    shm_thue, thue = get_shm(thue_name,int(px/palette_ratio),np.float32)
    seed = int((time.time() * 1000) % (2**16)) + worker_id + os.getpid()
    np.random.seed(seed  % (2**32) )
 
    for n in range(samples):
        for k in range(1):     
            t1 = np.random.random()
            t2 = np.random.random()
            i = int(t1*(px/palette_ratio))
            j = int(t2*(px/palette_ratio))
            rts = polys.polystate.sample(t1, t2)
            if len(rts)>0:
                mask = (
                    (rts.real >= llr) & (rts.real <= urr) &
                    (rts.imag >= lli) & (rts.imag <= uri)
                )
                rts_clip = rts[mask]
                if len(rts_clip)>0:
                    real = (px-1)*(rts_clip.real - llr)/rgr
                    imag = (px-1)*(rts_clip.imag - lli)/rgi
                    x = np.clip(real , 0, px-1).astype(int)
                    y = np.clip(imag , 0, px-1).astype(int)
                    h = (np.angle(t1+1j*t2))
                    s = np.abs(t1+t2)
                    thue[i,j]=thue[i,j]*(1-arw)+h*arw
                    tsat[i,j]=tsat[i,j]*(1-arw)+s*arw
                    hue[x,y]=hue[x,y]*(1-arw)+h*arw
                    sat[x,y]=sat[x,y]*(1-arw)+s*arw
                    np.add.at(count, (x, y), 1.0)

        if n % 1000 == 0 and worker_id == 0:
            print(f"{worker_id} : {round(100*n/samples,1)}")


    shm_count.close()  
    shm_sat.close()   
    shm_hue.close() 
    shm_tsat.close()   
    shm_thue.close()    
    return

if __name__ == "__main__":

    if not sys.stdin.isatty():  # stdin is *not* from terminal → it's piped
        js_config = sys.stdin.read()
         
    # set parameters
    polys.polystate.json2state(js_config)
    #ps.view["res"] = 1000   
    #ps.view["samples"] = 10_000
    ps.view["view"] = (sq*(-1-1j),sq*(1+1j))
    rts = polys.polystate.sample(0.5, 0.5)
    ps.poly["degree"] = len(rts)+1
    ps.poly["arw"] = arw
    ps.png["hfac"] = hfac
    ps.png["hcst"] = hcst
    ps.png["sfac"] = sfac
    ps.png["scst"] = scst
    js_config = ps.state2json()
        
    # shared arrays
    shm_count, count = make_shm(ps.view["res"],np.float32)
    shm_sat, sat = make_shm(ps.view["res"],np.float32)
    shm_hue, hue = make_shm(ps.view["res"],np.float32)
    shm_thue, thue = make_shm(int(ps.view["res"]/palette_ratio),np.float32)
    shm_tsat, tsat = make_shm(int(ps.view["res"]/palette_ratio),np.float32)
   
    chunks = []
    num_workers = multiprocessing.cpu_count()
    print(f"CPU count : {num_workers}")
    for  worker_id in range(num_workers):
        chunks.append(( 
            worker_id, 
            num_workers, 
            js_config, 
            shm_count.name,
            shm_hue.name,
            shm_sat.name,
            shm_thue.name,
            shm_tsat.name
        ))

    print("Start")
    with multiprocessing.Pool(num_workers) as pool:
        pool.map(sample_chunk, chunks)
    print("End")

    print(f"count: {np.min(count)} - {np.max(count)}")

    divisor = np.where(count>0,count,1)
    print(f"hue: {np.min(hue[hue>0])} - {np.max(hue)}")
    H = norm(np.clip(hue,np.min(hue[hue>0]),np.max(hue)))
    print(f"H: {np.min(H)} - {np.max(H)}")
    H = (((H*1.0)%1)*hfac + hcst) % 1
    print(f"H: {np.min(H)} - {np.max(H)}")

    S = norm(sat)
    print(f"S: {np.min(S)} - {np.max(S)}")
    S = (((S*sfac)%1)*0.99) % 1
    print(f"S: {np.min(S)} - {np.max(S)}")

    V = norm(count)**0.15
    print(f"V: {np.min(V)} - {np.max(V)}")
    
    TH = norm(thue)
    print(f"TH: {np.min(TH)} - {np.max(TH)}")
    TH = (((TH*1.0)%1)*hfac + hcst) % 1
    print(f"TH: {np.min(TH)} - {np.max(TH)}")

    TS = norm(tsat)
    print(f"TS: {np.min(TS)} - {np.max(TS)}")
    TS = (((TS*sfac)%1)*0.99) % 1
    print(f"TS: {np.min(TS)} - {np.max(TS)}")
    
    shm_count.close()
    shm_count.unlink()
    shm_sat.close()
    shm_sat.unlink()
    shm_hue.close()
    shm_hue.unlink()
    shm_tsat.close()
    shm_tsat.unlink()
    shm_thue.close()
    shm_thue.unlink()


    #HSV[...,2] = cv2.equalizeHist(HSV[...,2])
    #eq_RGB = cv2.cvtColor(HSV, cv2.COLOR_HSV2RGB)
    print("Saving: myplot.png")
    save2rgb(np.stack([H, S, np.where(V>0,1,0)], axis=-1),'myplot.png')
    print("Saving: mypalette.png")
    save2rgb(np.stack([TH, TS, np.full_like(TH,1.0)], axis=-1),'mypalette.png')
    


    
