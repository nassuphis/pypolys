#!/usr/bin/env python

import polys
import sys

import polys.polystate
if __name__ == "__main__":
    args = ' '.join(sys.argv)
    polys.polystate.cli2state(args)
    print(polys.polystate.state2cli())
    print(polys.polystate.state2json())

    print(polys.polystate.sample(0.5,0.5))
