#!/usr/bin/env python

# test/cli2json.py test123 -x unit_circle,coeff7 -p poly_362 -z rev -s solve  | test/json2plot.py 10000

import numpy as np
import pyvips

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


res_z = np.load('myresult.npz')
results = res_z['arr_0']
idx = np.lexsort((results[:,1], results[:,0]))
results=results[idx,:]
i = results[:,0]
j = results[:,1]
height=i.max()+1
width=j.max()+1
diff = np.diff(i*width+j)
is_new = np.concatenate(([True],diff!=0))
gbi = np.cumsum(is_new)-1 # group broadcast index group results -> individual locations
starts = np.nonzero(is_new)[0]
x = results[:,2]
y = results[:,3]
z = norm(i) + 1j*norm(j) # parameters
r = norm(x) + 1j*norm(y) # roots

print(f"results: ({results.shape})")
print(f"resolution: ({height},{width})")
print(f"x: {np.min(x)} - {np.max(x)}")
print(f"y: {np.min(y)} - {np.max(y)}")
print(f"i: {np.min(i)} - {np.max(i)}")
print(f"j: {np.min(j)} - {np.max(j)}")
# non-zero locations of roots and parameters
zcount=np.zeros((height,width),dtype=np.float32)
np.add.at(zcount, (x, y), 1.0)
zpix = zcount>0
zminator = np.where(zpix,zcount,1)
icount=np.zeros((height,width),dtype=np.float32)
np.add.at(icount, (i, j), 1.0)
ipix = icount>0
iminator = np.where(ipix,icount,1)


# palete hue
colorization_method=3
print(f"colorization: {colorization_method}")
if colorization_method==1:
    # hue
    hue = ( np.angle(z) / np.pi + 1.0 ) / 2.0
    hue = hue * 0.5 - 0.25
    pH = np.zeros((height,width),dtype=np.float32)
    np.add.at(pH,(i,j),hue)
    pH = pH / iminator
    print(f"pH: ({np.min(pH[icount>0])}-{np.max(pH[icount>0])})")
    # saturation
    sat = ( np.angle(z) / np.pi + 1.0 ) / 2.0
    sat = (( 50 * sat ) % 1) * 0.5 + 0.4
    print(f"sat: ({np.min(sat)}-{np.max(sat)})")
    pS = np.zeros((height,width),dtype=np.float32)
    np.add.at(pS,(i,j),sat)
    pS = pS / iminator
    print(f"pS: ({np.min(pS[zcount>0])}-{np.max(pS[zcount>0])})")

if colorization_method==2:
    # hue
    Xmin = np.zeros((height,width),dtype=np.float32)
    Xmin[i,j] = np.max(x) + 1
    np.minimum.at(Xmin,(i,j),x)
    Xmax = np.zeros((height,width),dtype=np.float32)
    Xmax[i,j] = np.min(x) - 1
    np.maximum.at(Xmax,(i,j),x)
    Ymin = np.zeros((height,width),dtype=np.float32)
    Ymin[i,j] = np.max(y) + 1
    np.minimum.at(Ymin,(i,j),y)
    Ymax = np.zeros((height,width),dtype=np.float32)
    Ymax[i,j] = np.min(y) - 1
    np.maximum.at(Ymax,(i,j),y)
    Xdiff = Xmax[i,j]-Xmin[i,j]
    Ydiff = Ymax[i,j]-Ymin[i,j]
    area = np.where(Ydiff>0,Xdiff,1)/np.where(Ydiff>0,Ydiff,1)
    pH = np.zeros((height,width),dtype=np.float32)
    pH[i,j] = norm(area) * 0.5 - 0.25
    print(f"pH: ({np.min(pH[i,j])}-{np.max(pH[i,j])})")
    # saturation
    sat = ( np.angle(r) / np.pi + 1.0 ) / 2.0
    pS = np.zeros((height,width),dtype=np.float32)
    np.add.at(pS,(i,j),sat/icount[i,j])
    pS[i,j] = (( 50 * pS[i,j] ) % 1) * 0.5 + 0.4
    print(f"pS: ({np.min(pS[i,j])}-{np.max(pS[i,j])})")

if colorization_method==3:
    # hue
    z_all_sum = np.sum(z)
    r_all_sum = np.sum(r)
    rabs = np.abs(r-r_all_sum)
    rngl = np.angle(r-r_all_sum)
    zabs = np.abs(z-z_all_sum)
    zngl = np.angle(z-z_all_sum)
    denomi=1/np.add.reduceat(np.ones(r.shape),starts)

    
    z_sum=np.add.reduceat(z,starts)-z_all_sum
    z_mean=z_sum*denomi

    r_sum=np.add.reduceat(r,starts)-r_all_sum
    r_mean=r_sum*denomi

    rabs_max = np.maximum.reduceat(rabs,starts)
    rabs_min = np.minimum.reduceat(rabs,starts)
    rabs_rng = norm(rabs_max-rabs_min)
    rabs_mean=np.add.reduceat(rabs,starts)*denomi
    rngl_mean=np.add.reduceat(rngl,starts)*denomi
    zabs_mean=np.add.reduceat(zabs,starts)*denomi
    zngl_mean=np.add.reduceat(zngl,starts)*denomi
    
    pH = np.zeros((height,width),dtype=np.float32)
    hue = (norm(rngl_mean)*0.1+0.15)[gbi]
    pH[i,j] = hue #hue % 1
    print(f"pH: ({np.min(pH[i,j])}-{np.median(pH[i,j])}-{np.max(pH[i,j])})")

    pS = np.zeros((height,width),dtype=np.float32)
    ngl=  norm(rngl_mean)
    #ngl = norm(np.maximum.reduceat(rngl,starts)-np.minimum.reduceat(rngl,starts))
    pS[i,j]=1.0 #(50*ngl[gbi])*1.0
    print(f"pS: ({np.min(pS[i,j])}-{np.max(pS[i,j])})")

H = np.zeros((height,width),dtype=np.float32)
S = np.zeros((height,width),dtype=np.float32)

H[x,y]=pH[i,j]
S[x,y]=pS[i,j]
V = np.where(zcount>0,1.0,0.0)

save2rgb(np.stack([H, S, V], axis=-1),'myresultplot.png')
print("saved myresultplot.png")
#save2rgb(np.stack([pH, pS, np.full_like(pH,1.0)], axis=-1),'myresultpalette.png')
#print("saved myresultpalette.png")



    
