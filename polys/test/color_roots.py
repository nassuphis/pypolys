#!/usr/bin/env python
# color roots from palette file
# test/cli2json.py test123 -x unit_circle,coeff7 -p poly_362 -z rev -s solve  | test/json2plot.py 10000

import numpy as np
import pyvips
import imageio.v3 as iio

def norm(x):
     max_x = np.max(x)
     min_x = np.min(x)
     if max_x-min_x>0:
        return ((x-min_x)/(max_x-min_x)).astype(np.float32)
     else:
         return x.astype(np.float32)
     
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


palette_img =iio.imread("palette.png")
palette = palette_img[..., :3].astype(np.uint8)

res_z = np.load('myresult.npz')
results = (res_z['arr_0']).astype(np.uint16)
i = results[:,0]
j = results[:,1]
x = results[:,2]
y = results[:,3]
height=i.max()+1
width=j.max()+1

ifac = (palette.shape[0]-1)/(height-1)
jfac = (palette.shape[1]-1)/(width-1)
pi = (i*ifac).astype(np.uint16)
pj = (j*jfac).astype(np.uint16)

print(f"i,j : {height},{width}")
print(f"palette : {palette.shape}")
print(f"factors: {ifac},{jfac}")
print(f"pi,pj : {np.max(pi)},{np.max(pj)}")

RGB = np.zeros((height,width,3),dtype=np.uint8)

RGB[x,y,:] = palette[pi,pj,:]

iio.imwrite('myresultplot.png', RGB)





    
