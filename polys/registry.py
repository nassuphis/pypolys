#!/usr/bin/env python

from . import giga
from . import chess
from . import poly100
from . import poly200
from . import poly300
from . import poly400
from . import poly500
from . import poly600
from . import poly700
from . import poly800
from . import poly900
from . import poly1000
import re
from . import xfrm 
from . import zfrm 
from . import solve 


xfun = {}
xfun.update({
    name: getattr(xfrm, name)
    for name in dir(xfrm)
    if re.fullmatch(r'[a-z]+[a-z0-9_]+[a-z0-9]+', name)
})

zfun = {}
zfun.update({
    name: getattr(zfrm, name)
    for name in dir(zfrm)
    if re.fullmatch(r'[a-z]+[a-z0-9_]+[a-z0-9]+', name)
})

sfun = {}
sfun.update({
    name: getattr(solve, name)
    for name in dir(solve)
    if re.fullmatch(r'[a-z]+[a-z0-9_]+[a-z0-9]+', name)
})


# build pfun dynamically so newly added functions are automatically included
pfun = {
    'poly_chess': chess.poly_chess,
    'poly_chess1': chess.poly_chess1,
    'poly_chess2': chess.poly_chess2,
    'poly_chess3': chess.poly_chess3,
    'poly_chess4': chess.poly_chess4,
    'poly_chess5_old': chess.poly_chess5_old,
    'poly_chess5': chess.poly_chess5,
}

pfun.update({
    name: getattr(giga, name)
    for name in dir(giga)
    if name.startswith('poly_giga_')
})

pfun.update({
    name: getattr(poly100, name)
    for name in dir(poly100)
    if re.fullmatch(r'poly_\d+', name)
})

pfun.update({
    name: getattr(poly200, name)
    for name in dir(poly200)
    if re.fullmatch(r'poly_(1\d\d|200)', name)
})

pfun.update({
    name: getattr(poly300, name)
    for name in dir(poly300)
    if re.fullmatch(r'poly_(2\d\d|300)', name)
})

pfun.update({
    name: getattr(poly400, name)
    for name in dir(poly400)
    if re.fullmatch(r'poly_(3\d\d|400)', name)
})

pfun.update({
    name: getattr(poly500, name)
    for name in dir(poly500)
    if re.fullmatch(r'poly_(4\d\d|500)', name)
})

pfun.update({
    name: getattr(poly600, name)
    for name in dir(poly600)
    if re.fullmatch(r'poly_(5\d\d|600)', name)
})

pfun.update({
    name: getattr(poly700, name)
    for name in dir(poly700)
    if re.fullmatch(r'poly_(6\d\d|700)', name)
})

pfun.update({
    name: getattr(poly800, name)
    for name in dir(poly800)
    if re.fullmatch(r'poly_(7\d\d|800)', name)
})

pfun.update({
    name: getattr(poly900, name)
    for name in dir(poly900)
    if re.fullmatch(r'poly_(8\d\d|900)', name)
})

pfun.update({
    name: getattr(poly1000, name)
    for name in dir(poly1000)
    if re.fullmatch(r'poly_(9\d\d|1000)', name)
})

