#!/usr/bin/env python

import polys
import sys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    if not sys.stdin.isatty():  # stdin is *not* from terminal → it's piped
        js_config = sys.stdin.read()
            
    polys.polystate.json2state(js_config)
    polys.polystate.cli2state(' '.join(sys.argv[1:]))
    roots = []
    for i in range(int(sys.argv[1])):
        rts = polys.polystate.sample(np.random.random(),np.random.random())
        roots.extend(rts)

    roots = np.array(roots)
    if np.iscomplexobj(roots):
        phase = np.angle(roots)
        plt.scatter(roots.real, roots.imag, c=phase, cmap='hsv', s=0.001, alpha=0.5)
        plt.xlabel('Re')
        plt.ylabel('Im')
        plt.title('Sampled Roots (colored by phase)')
        plt.colorbar(label='Phase (radians)')
        plt.gca().set_aspect('equal')
        plt.show()
    else:
        # In case roots are not complex numbers
        plt.plot(roots)
        plt.title('Sampled values')
        plt.show()

