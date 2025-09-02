#!/usr/bin/env python

import numpy as np
import mpsolve
import inspect
from PIL import Image


cf = np.poly(mpsolve.chessboard_roots(4)).astype(np.complex128)
pm = mpsolve.scan_cold(cf,0.0,1.0,0.0,1.0,256*256,mpsolve.wilkinson.perturb_1)
re, im = pm[:, 0], pm[:, 1]
llx,lly, urx, ury = mpsolve.compute_view(re,im)
width, height = 25_000, 25_000
np_img = mpsolve.rasterize_points(re, im, llx, lly, urx, ury, width, height, x_is_im=True, flip_y=True)
img = Image.fromarray(np_img, mode='L')
timg = mpsolve.text_to_image(
    inspect.getsource(mpsolve.wilkinson.perturb_1),
    width=width//5,                          # only width is fixed
    font_path="/System/Library/Fonts/SFNSMono.ttf",
    margin=48,
    line_spacing=6,
    fg=(255, 255, 255), 
    upscale_antialias=2,
)
img.paste(timg, (200, 200))
img.save("mpsolve.png")
