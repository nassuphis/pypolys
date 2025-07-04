#!/usr/bin/env python

# test/cli2json.py test123 -x unit_circle,coeff7 -p poly_362 -z rev -s solve  | test/json2plot.py 10000

import numpy as np
import pyvips
import argparse
import functools
import numexpr as ne
import polys
import polys.polystate

type_choices = [
        "h",
        "hs",
        "abs",
        "ngl",
        "trigh",
        "trighs",
    ]

params = {
    "rfr":50.0,
    "cfr":50.0,
    "afr":1.0,
    "hmlt":1.0,
    "hoff":0.0,
    "h0":0.0,
    "h1":1.0,
    "xc": 0.0,
    "yc": 0.0,
    "r": 0.5,
    "ccf":1.0,
    "exp": 1.0,
    "cst": 0.0
}


def norm(x):
     max_x = np.max(x)
     min_x = np.min(x)
     if max_x-min_x>0:
        return ((x-min_x)/(max_x-min_x)).astype(np.float32)
     else:
         return x.astype(np.float32)

def hist_equal_gray(img):
    """Apply global histogram equalization to a grayscale float32 image in [0, 1]."""
    # Flatten and compute histogram
    hist, bins = np.histogram(img.flatten(), bins=256, range=[0,1])
    cdf = hist.cumsum()
    cdf = cdf / cdf[-1]  # normalize to [0, 1]
    
    # Map original pixels to equalized values
    img_eq = np.interp(img.flatten(), bins[:-1], cdf)
    return img_eq.reshape(img.shape)

def hue_clip(H):
    cH = norm(H)*(params["h1"]-params["h0"])+params["h0"]
    return cH

def in_circle(H,a):
    y = (norm(row_mat) - 0.5)*2
    x = (norm(col_mat) - 0.5)*2
    xc = ne.evaluate(a[0]) if len(a)>0 else params["xc"]
    yc = ne.evaluate(a[1]) if len(a)>1 else params["yc"]
    r = ne.evaluate(a[2]) if len(a)>2 else params["r"]
    cc = (x-xc)**2+(y-yc)**2 - r
    mask =  cc > 0
    return np.where(mask,H,0)

def out_circle(H,a):
    y = (norm(row_mat) - 0.5)*2
    x = (norm(col_mat) - 0.5)*2
    xc = ne.evaluate(a[0]) if len(a)>0 else params["xc"]
    yc = ne.evaluate(a[1]) if len(a)>1 else params["yc"]
    r = ne.evaluate(a[2]) if len(a)>2 else params["r"]
    cc = (x-xc)**2+(y-yc)**2 - r
    mask =  cc < 0
    return np.where(mask,H,0)

def in_square(H,a):
    y = (norm(row_mat) - 0.5)*2
    x = (norm(col_mat) - 0.5)*2
    xc = ne.evaluate(a[0]) if len(a)>0 else params["xc"]
    yc = ne.evaluate(a[1]) if len(a)>1 else params["yc"]
    r = ne.evaluate(a[2]) if len(a)>2 else params["r"]
    xm = abs(x-xc) < r
    ym = abs(y-yc) < r
    mask =  ym & xm
    return np.where(mask,H,0)

def out_square(H,a):
    y = (norm(row_mat) - 0.5)*2
    x = (norm(col_mat) - 0.5)*2
    xc = ne.evaluate(a[0]) if len(a)>0 else params["xc"]
    yc = ne.evaluate(a[1]) if len(a)>1 else params["yc"]
    r = ne.evaluate(a[2]) if len(a)>2 else params["r"]
    xm = abs(x-xc) > r
    ym = abs(y-yc) > r
    mask =  ym | xm
    return np.where(mask,H,0)


def square_strip(H,a):
    y = (norm(row_mat) - 0.5) * 2
    x = (norm(col_mat) - 0.5) * 2
    sml = ne.evaluate(a[0]) if len(a)>0 else 0.4
    lrg = ne.evaluate(a[1]) if len(a)>1 else 0.5
    sml_square = (abs(x) > sml) | (abs(y) > sml) 
    large_square = (abs(x) < lrg) & (abs(y) < lrg) 
    mask = large_square & sml_square
    return np.where(mask,H,0)

def above(H,a):
    x = (norm(col_mat) - 0.5)
    xc = ne.evaluate(a[0]) if len(a)>0 else params["xc"]
    mask = x > xc
    return np.where(mask,H,0)

def below(H,a):
    x = (norm(col_mat) - 0.5)
    xc = ne.evaluate(a[0]) if len(a)>0 else params["xc"]
    mask = x < xc
    return np.where(mask,H,0)

def left(H,a):
    y = (norm(row_mat) - 0.5)
    yc = ne.evaluate(a[0]) if len(a)>0 else params["yc"]
    mask = y < yc
    return np.where(mask,H,0)

def right(H,a):
    y = (norm(row_mat) - 0.5)
    yc = ne.evaluate(a[0]) if len(a)>0 else params["yc"]
    mask = y > yc
    return np.where(mask,H,0)

def wavefield(H):
    a=params["rfr"]*2*np.pi*norm(row_mat)
    b=params["cfr"]*2*np.pi*norm(col_mat)
    v=np.cos(a)+np.sin(b)
    return v

def wavefield1(H):
    a=params["rfr"]*2*np.pi*norm(row_mat)
    b=params["cfr"]*2*np.pi*norm(col_mat)
    v=np.sin(a+b)+np.cos(a-b)
    return v

def wavefield2(H):
    a=params["rfr"]*2*np.pi*norm(row_mat)
    b=params["cfr"]*2*np.pi*norm(col_mat)
    v=np.sin(a)**3+np.cos(a-b)**3
    return v

def wavefield3(H):
    r = np.hypot(norm(row_mat) - 0.5, norm(col_mat) - 0.5)
    v = np.sin(2 * np.pi * params["afr"] * r)
    return v

def anglefield(H,a):
    afr = ne.evaluate(a[0]) if len(a)>0 else params["afr"]
    y = (norm(row_mat) - 0.5) * 2
    x = (norm(col_mat) - 0.5) * 2
    v0 = np.angle(x + 1j * y)
    return afr*norm(v0)

def absfield(H,a):
    afr = ne.evaluate(a[0]) if len(a)>0 else params["afr"]
    v0 = np.angle(row_mat + 1j * col_mat)
    return afr*norm(v0)

stack = []
def push(matrix):
    stack.append(matrix)

def pop():
    return stack.pop()

def add(x):
    return x+stack.pop()

def subtract(x):
    return x-stack.pop()

def mult(x):
    return x+stack.pop()

def mult_clip1(H,a):
    afr = ne.evaluate(a[0]) if len(a)>0 else params["afr"]
    return (afr*H)%1

hfrm_pipeline = "clip"
sfrm_pipeline = "one"
vfrm_pipeline = "one"
res = 1000
row_mat, col_mat = np.indices(((res,res)))
pfrm_functions = {
    "none": lambda x,a: x,
    "zero": lambda x,a: np.zeros(x.shape),
    "one": lambda x,a: np.ones(x.shape),
    "red":         lambda x,a: np.full_like(x, 0.00),
    "orange":      lambda x,a: np.full_like(x, 0.08),
    "yellow":      lambda x,a: np.full_like(x, 0.17),
    "chartreuse":  lambda x,a: np.full_like(x, 0.25),
    "green":       lambda x,a: np.full_like(x, 0.33),
    "spring":      lambda x,a: np.full_like(x, 0.42),
    "cyan":        lambda x,a: np.full_like(x, 0.50),
    "azure":       lambda x,a: np.full_like(x, 0.58),
    "blue":        lambda x,a: np.full_like(x, 0.67),
    "violet":      lambda x,a: np.full_like(x, 0.75),
    "magenta":     lambda x,a: np.full_like(x, 0.83),
    "rose":        lambda x,a: np.full_like(x, 0.92),
    "get": lambda x,a: np.full_like(x,params[a[0]]),
    "rows": lambda x,a: row_mat,
    "cols": lambda x,a: col_mat,
    "x": lambda x,a: 2*(norm(col_mat)-0.5),
    "y": lambda x,a: 2*(norm(row_mat)-0.5),
    "z": lambda x,a: 2*(norm(col_mat)-0.5)+1j*2*(norm(row_mat)-0.5),
    "mc": lambda x,a: mult_clip1(x,a),
    "norm": lambda x,a: norm(x),
    "trans": lambda x,a: np.transpose(x),
    "exp": lambda x,a: x**params["exp"],
    "clip": lambda x,a: hue_clip(x),
    "cci": lambda x,a: in_circle(x,a),
    "cco": lambda x,a: out_circle(x,a),
    "csi": lambda x,a: in_square(x,a),
    "cso": lambda x,a: out_square(x,a),
    "css": lambda x,a: square_strip(x,a),
    "above": lambda x,a: above(x,a),
    "below": lambda x,a: below(x,a),
    "wf": lambda x,a: wavefield(x),
    "wf1": lambda x,a: wavefield1(x),
    "wf2": lambda x,a: wavefield2(x),
    "wf3": lambda x,a: wavefield3(x),
    "nglf": lambda x,a: anglefield(x,a),
    "absf": lambda x,a: absfield(x,a),
    "inv": lambda x,a: 1-x,
    "push": lambda x,a: push(x),
    "pushv": lambda x,a: push(np.full_like(x,params[a[0]])),
    "pop": lambda x,a: pop(),
    "add": lambda x,a: add(x),
    "subtract": lambda x,a: subtract(x),
    "mult": lambda x,a: mult(x),
    "imag": lambda x,a: 1j*x,
    "angle": lambda x,a: np.angle(x),
    "abs": lambda x,a: np.abs(x),
    "equalize": lambda x,a: hist_equal_gray(x)
}
  

def pfrm(H,pipeline):

    pfuns = []
    pargs = []
    if pipeline is None: 
        return H
    if pipeline.strip()=='': 
        return H
    calls = [name.strip() for name in pipeline.split(',')]

    for call in calls:
        call_parts = call.split("_")
        name = call_parts[0]
        args = call_parts[1:]
        if name not in pfrm_functions:
            raise KeyError(f"pfrm: '{name}' not in pfrm_functions")
        value = pfrm_functions[name]
        if not callable(value):
            raise ValueError(f"pfrm: pfrm_functions value '{name}' not callable")
        pfuns.append(value)
        pargs.append(args)
        print(f"{args}")
    
    tH  = functools.reduce(lambda acc, i: pfuns[i](acc,pargs[i]),range(len(pfuns)),H)
    return tH



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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="palette maker")
    parser.add_argument('outfile',nargs="?",type=str, default="palette.png", help="outfile")
    parser.add_argument('--verbose',action='store_true',help="verbose")
    parser.add_argument('--type', choices=type_choices,default="none",help="mode")
    parser.add_argument('-r','--res', type=int, default=1000, help="resolution")
    parser.add_argument('-p','--par', type=str, default=None, help="parameters")
    parser.add_argument('-H','--hfrm', type=str, default="clip", help="hue functions")
    parser.add_argument('-S','--sfrm', type=str, default="none", help="sat functions")
    parser.add_argument('-V','--vfrm', type=str, default="none", help="val functions")

    args = parser.parse_args()    

    hfrm_pipeline = args.hfrm
    sfrm_pipeline = args.sfrm
    vfrm_pipeline = args.vfrm
    res=args.res
    row_mat, col_mat = np.indices(((args.res,args.res)))
    H = np.zeros((args.res,args.res),dtype=np.float32)
    S = np.ones((args.res,args.res),dtype=np.float32) 
    V = np.ones((args.res,args.res),dtype=np.float32) 
    params.update(polys.polystate.pu.setf(args.par))

    
    
    if args.type=="none":
        H=pfrm(H,hfrm_pipeline)
        S=pfrm(S,sfrm_pipeline)
        V=pfrm(V,vfrm_pipeline)

    if args.type=="h":
        H=pfrm(row_mat,hfrm_pipeline)

    if args.type=="hs":
        H=pfrm(row_mat,hfrm_pipeline)
        S=pfrm(col_mat,sfrm_pipeline)

    if args.type=="abs":
        v0 = norm(row_mat) + 1j * norm(col_mat)
        v1 = params["xc"]+1j*params["yc"]
        v2 = np.abs(v0+v1)
        V = pfrm(v2,hfrm_pipeline)

    if args.type=="ngl":
        v0 = np.angle(row_mat + 1j * col_mat)
        v1 = params["afr"]*norm(v0)
        H = pfrm(v1,hfrm_pipeline)
    
    if args.type=="trigh":
        a=params["rfr"]*2*np.pi*norm(row_mat)
        b=params["cfr"]*2*np.pi*norm(col_mat)
        H=pfrm(np.cos(a)+np.sin(b),hfrm_pipeline)

    if args.type=="trighs":
        a=params["rfr"]*2*np.pi*norm(row_mat)
        b=params["cfr"]*2*np.pi*norm(col_mat)
        H=pfrm(np.cos(a)+np.sin(b),hfrm_pipeline)
        S=pfrm(np.sin(a)+np.cos(b),sfrm_pipeline)


    save2rgb(np.stack([H, S, V], axis=-1),args.outfile)

