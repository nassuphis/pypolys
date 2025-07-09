#!/usr/bin/env python

# test/cli2json.py test123 -x unit_circle,coeff7 -p poly_362 -z rev -s solve  | test/json2plot.py 10000

import pyvips

# Load image
image = pyvips.Image.new_from_file("myresultplot.png", access="sequential")

# Create mask for black pixels (all channels == 0)
is_black = (image[0] == 0) & (image[1] == 0) & (image[2] == 0)

# Create white image
white_pixel = pyvips.Image.black(image.width, image.height).new_from_image([255, 255, 255])

# Replace black pixels with white
result = is_black.ifthenelse(white_pixel, image, blend=False)

# Save output
result.write_to_file("myresultplot_white.png")




    
