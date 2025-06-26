#!/usr/bin/env python

import polys
import sys


if __name__ == "__main__":

    if not sys.stdin.isatty():  # stdin is *not* from terminal → it's piped
        js_config = sys.stdin.read()
            
    polys.polystate.dict2state({}) 
    polys.polystate.json2state(js_config)
    x=polys.polystate.state2cli()
    polys.polystate.cli2state(x)
    print(' '.join(sys.argv[1:]))
    polys.polystate.cli2state(' '.join(sys.argv[1:]))
    print(polys.polystate.state2json())
