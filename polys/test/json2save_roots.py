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


sq = 3.0

def make_shm(rows,cols,type):
    shm = SharedMemory(
        create=True, 
        size = rows * cols * np.dtype(type).itemsize
    )
    array = np.ndarray(
        (rows,cols), 
        dtype=type, 
        buffer=shm.buf
    )
    array[:] = 0
    return (shm,array)

def get_shm(name,rows,cols,type):
    shm = SharedMemory(name=name)
    array = np.ndarray(
        (rows, cols), 
        dtype=type, 
        buffer=shm.buf
    )
    return(shm,array)



def sample_chunk(args):
    worker_id, num_workers, js_config, result_name = args
    polys.polystate.json2state(js_config)
    start=int(np.linspace(0, ps.view["samples"], num_workers+1)[worker_id])
    end=int(np.linspace(0, ps.view["samples"], num_workers+1)[worker_id+1])
    px = ps.view["res"]
    ll,ur = ps.view["view"]
    llr, lli, urr, uri = ll.real, ll.imag, ur.real, ur.imag
    rgr = urr - llr
    rgi = uri - lli

    def add_sample(t1,t2):
        nonlocal result,write_ptr,start,end,px,llr,lli,urr,uri,rgr,rgi
        i = int(round(t1*(px-1)))
        j = int(round(t2*(px-1)))
        rts = polys.polystate.sample(t1, t2)
        if len(rts)>0:
            mask = (
                (rts.real >= llr) & (rts.real <= urr) &
                (rts.imag >= lli) & (rts.imag <= uri)
            )
            rts_clip = rts[mask]
            rts_len=len(rts_clip)
            if rts_len>0:
                real = (px-1)*(rts_clip.real - llr)/rgr
                imag = (px-1)*(rts_clip.imag - lli)/rgi
                x = np.clip(real , 0, px-1).astype(np.uint16)
                y = np.clip(imag , 0, px-1).astype(np.uint16)
                available = min(end-write_ptr,rts_len)
                if available<=0: 
                    return available
                result[write_ptr:write_ptr+available,0]=i
                result[write_ptr:write_ptr+available,1]=j
                result[write_ptr:write_ptr+available,2]=x[:available]
                result[write_ptr:write_ptr+available,3]=y[:available]
                write_ptr += available
                return available
            else:
                return end-write_ptr
        return end-write_ptr

    shm_result, result = get_shm(result_name,ps.view["samples"],4,np.uint16)
    seed = int((time.time() * 1000) % (2**16)) + worker_id + os.getpid()
    np.random.seed(seed  % (2**32) )
 
    write_ptr = start
    show_msg = start

    


    for _ in range(start,end):       
        if worker_id == 0:
            if show_msg>=write_ptr:
                print(f"{worker_id} : {round(100*(write_ptr-start)/(end-start),1)}")
                show_msg=write_ptr+(end-start)/100

        if add_sample(0.0,0.0)<=0:
            break
        if add_sample(1.0,1.0)<=0:
            break
        if add_sample(np.random.random(),np.random.random())<=0:
            break

    if worker_id == 0:
        print(f"{worker_id} finished")
        print(f" samples: {ps.view["samples"]}")
        print(f" workers: {num_workers}")
   
    shm_result.close()  
    return

if __name__ == "__main__":

    if not sys.stdin.isatty():  # stdin is *not* from terminal → it's piped
        js_config = sys.stdin.read()
         
    # set parameters
    polys.polystate.json2state(js_config)
    ps.view["view"] = (sq*(-1-1j),sq*(1+1j))
    rts = polys.polystate.sample(0.5, 0.5)
    ps.poly["degree"] = len(rts)+1
    js_config = ps.state2json()
        
    # shared arrays
    shm_result, result = make_shm(ps.view["samples"],4,np.int16)

    chunks = []
    num_workers = multiprocessing.cpu_count()
    print(f"CPU count : {num_workers}")
    for  worker_id in range(num_workers):
        chunks.append(( 
            worker_id, 
            num_workers, 
            js_config, 
            shm_result.name
        ))

    print("Start")
    with multiprocessing.Pool(num_workers) as pool:
        pool.map(sample_chunk, chunks)
    print("End")
   
    result_copy = np.copy(result)
    shm_result.close()
    shm_result.unlink()

    print("Saving: myresult.npz")
    np.savez_compressed('myresult.npz', result_copy)

    print(f"i {np.min(result_copy[:,0])}-{np.max(result_copy[:,0])}")
    print(f"j {np.min(result_copy[:,1])}-{np.max(result_copy[:,1])}")
    print(f"x {np.min(result_copy[:,2])}-{np.max(result_copy[:,2])}")
    print(f"y {np.min(result_copy[:,3])}-{np.max(result_copy[:,3])}")
    


    
