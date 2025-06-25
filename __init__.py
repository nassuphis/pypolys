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


# build registry dynamically so newly added functions are automatically included
registry = {
    'poly_chess': chess.poly_chess,
    'poly_chess1': chess.poly_chess1,
    'poly_chess2': chess.poly_chess2,
    'poly_chess3': chess.poly_chess3,
    'poly_chess4': chess.poly_chess4,
    'poly_chess5_old': chess.poly_chess5_old,
    'poly_chess5': chess.poly_chess5,
}

# register all poly_giga_* functions found in giga module
registry.update({
    name: getattr(giga, name)
    for name in dir(giga)
    if name.startswith('poly_giga_')
})

# register poly_1 .. poly_100 functions found in poly100 module
registry.update({
    name: getattr(poly100, name)
    for name in dir(poly100)
    if re.fullmatch(r'poly_\d+', name)
})

# register poly_101 .. poly_200 functions found in poly200 module
registry.update({
    name: getattr(poly200, name)
    for name in dir(poly200)
    if re.fullmatch(r'poly_(1\d\d|200)', name)
})

# register poly_201 .. poly_300 functions found in poly300 module
registry.update({
    name: getattr(poly300, name)
    for name in dir(poly300)
    if re.fullmatch(r'poly_(2\d\d|300)', name)
})

# register poly_301 .. poly_400 functions found in poly400 module
registry.update({
    name: getattr(poly400, name)
    for name in dir(poly400)
    if re.fullmatch(r'poly_(3\d\d|400)', name)
})

# register poly_401 .. poly_500 functions found in poly500 module
registry.update({
    name: getattr(poly500, name)
    for name in dir(poly500)
    if re.fullmatch(r'poly_(4\d\d|500)', name)
})

# register poly_501 .. poly_600 functions found in poly600 module
registry.update({
    name: getattr(poly600, name)
    for name in dir(poly600)
    if re.fullmatch(r'poly_(5\d\d|600)', name)
})

# register poly_601 .. poly_700 functions found in poly700 module
registry.update({
    name: getattr(poly700, name)
    for name in dir(poly700)
    if re.fullmatch(r'poly_(6\d\d|700)', name)
})

# register poly_701 .. poly_800 functions found in poly800 module
registry.update({
    name: getattr(poly800, name)
    for name in dir(poly800)
    if re.fullmatch(r'poly_(7\d\d|800)', name)
})

# register poly_801 .. poly_900 functions found in poly900 module
registry.update({
    name: getattr(poly900, name)
    for name in dir(poly900)
    if re.fullmatch(r'poly_(8\d\d|900)', name)
})

# register poly_901 .. poly_1000 functions found in poly1000 module
registry.update({
    name: getattr(poly1000, name)
    for name in dir(poly1000)
    if re.fullmatch(r'poly_(9\d\d|1000)', name)
})

