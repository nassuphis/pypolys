#!/usr/bin/env python

import polys
import sys

if __name__ == "__main__":
    polys.polystate.cli2state(' '.join(sys.argv[1:]))
    print(polys.polystate.state2json())
