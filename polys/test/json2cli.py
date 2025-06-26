#!/usr/bin/env python

import polys
import sys


if __name__ == "__main__":

    if not sys.stdin.isatty():  # stdin is *not* from terminal → it's piped
        js_config = sys.stdin.read()
            
    polys.polystate.json2state(js_config)
    print(polys.polystate.state2cli())
