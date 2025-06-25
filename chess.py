import numpy as np
import polystate as ps
import polylayout as pl


def bimodal_skewed(a, size=1):
    """Return samples from a bimodal distribution skewed by ``a``."""
    u = np.random.uniform(0, 1, size)
    skewed = np.where(
        u < 0.5,
        (2 * u) ** (1 / (1 - a)) / 2,
        1 - (2 * (1 - u)) ** (1 / (1 - a)) / 2,
    )
    return np.clip(skewed, 0, 1)


def poly_chess(t1, t2):
    """Simple randomised chess board polynomial."""
    N = 7
    w = 2 * np.pi
    t = np.random.uniform(-w, w)
    x = np.sin(t) + np.tile(np.arange(1, N + 1), N)
    y = np.cos(t) + np.repeat(np.arange(1, N + 1), N)
    curve = np.array(x + 1j * y, dtype=complex) - ((N + 1) / 2) - 1j * ((N + 1) / 2)
    coeffs = np.poly(curve + np.cos(np.random.uniform(-w, w)))
    addon = 90 * np.arange(len(coeffs)) + np.poly(curve) * 200
    return np.array(coeffs + addon, dtype=complex)

def poly_chess1(t1,t2, N: int = 7 )-> np.ndarray:
    def p2(cf):
        n = len(cf)
        cf1 = np.full(n, 1.0, dtype=complex)
        cf2 = ( cf**2 + cf + cf1 ) 
        return cf2.astype(np.complex128)
    w = 2*np.pi
    t = np.random.uniform(-w, w)
    x = np.sin(t) + np.tile(np.arange(1, N+1), N)
    y = np.cos(t) + np.repeat(np.arange(1, N+1), N)
    curve = np.array(x + 1j*y,dtype=complex) - ((N+1)/2) - 1j * ((N+1)/2)
    coeffs = np.poly(curve+0.1*np.cos(np.random.uniform(-w, w)))
    cf1 = coeffs
    cf2 = np.pad(curve,(0,1),constant_values=10j)
    cf3 = (cf1+cf2*0.0001)
    cf = cf3 + 0.00000000000000000000001*p2(cf3)
    return np.array(cf,dtype=complex)


def poly_chess2( t1,t2, N: int = 8 ) -> np.ndarray:
    indices = np.arange(N) - (N - 1) / 2
    parity = (np.indices((N, N)).sum(axis=0)) % 2
    X, Y = np.meshgrid(indices, indices)
    t1 = 0.5 * np.exp(1j*2*np.pi*np.random.rand())* parity
    t2 = 0.5 *(np.random.rand()-0.5)* parity
    cf1 = np.poly(( (X + t1) + 1j * (Y + t1)).flatten())   
    cf2 = np.poly(( (X + t2) + 1j * (Y + t2)).flatten())   
    a = np.random.rand()
    coeffs = cf2 * a + cf1 * (1-a)
    return coeffs.astype(complex)


def poly_chess3(t1, t2, N: int = 8) -> np.ndarray:
    indices = np.arange(N) - (N - 1) / 2
    parity = (np.indices((N, N)).sum(axis=0)) % 2
    X, Y = np.meshgrid(indices, indices)
    t0 = np.random.rand()
    t1 = 0.5 * np.exp(1j * 2 * np.pi * t0) * parity
    t2 = 0.5 * (t0 - 0.5) * parity
    cf1 = np.poly(((X + t1) + 1j * (Y + t1)).flatten())
    cf2 = np.poly(((X + t2) + 1j * (Y + t2)).flatten())
    a = bimodal_skewed(0.85)
    coeffs = cf2 * a + cf1 * (1 - a)
    return coeffs.astype(complex)


def spindle(t, a=0.5, b=0.2, p=1.5):
    theta = 2 * np.pi * t
    x = a * np.sign(np.cos(theta)) * np.abs(np.cos(theta)) ** (2 / p)
    y = b * np.sign(np.sin(theta)) * np.abs(np.sin(theta)) ** (2 / p)
    return x + 1j * y


def cardioid(t):
    a = ps.poly.get("crdd") or 0.5
    theta = 2 * np.pi * t
    r = a * (1 + np.cos(theta))
    return r * np.exp(1j * theta)


def heart(u):
    phi = np.pi / 2
    t = 2 * np.pi * u + phi
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    rot = np.exp(-1j * np.pi / 4)
    hrt = x / 40 + 1j * y / 40 + 0.1j
    return hrt * rot


def limacon(t, a=0.3, b=0.5):
    theta = 2 * np.pi * t
    r = a + b * np.cos(theta)
    return r * np.exp(1j * theta)


def rose_curve(t, a=0.5, k=2):
    theta = 2 * np.pi * t
    r = a * np.cos(k * theta)
    return r * np.exp(1j * theta)


def lissajous(t, A=0.5, B=0.5, a=3, b=2, delta=np.pi / 2):
    theta = 2 * np.pi * t
    x = A * np.sin(a * theta + delta)
    y = B * np.sin(b * theta)
    return x + 1j * y


def astroid(t, a=0.5):
    theta = 2 * np.pi * t
    x = a * np.cos(theta) ** 3
    y = a * np.sin(theta) ** 3
    return x + 1j * y


def archimedean_spiral(t, a=0.1, b=0.1):
    theta = 2 * np.pi * t
    r = a + b * theta
    return r * np.exp(1j * theta)


def logarithmic_spiral(t, a=0.1, b=0.1):
    theta = 2 * np.pi * t
    r = a * np.exp(b * theta)
    return r * np.exp(1j * theta)


def deltoid(t, R=1.0):
    theta = 2 * np.pi * t
    x = R * (2 * np.cos(theta) + np.cos(2 * theta)) / 3
    y = R * (2 * np.sin(theta) - np.sin(2 * theta)) / 3
    return x + 1j * y


def ipolygon(t, n=3, radius=1.0, offset=0.0):
    n = ps.poly.get("iplgn") or 3
    t = np.atleast_1d(t)
    t_scaled = t.real * n
    edge_idx = np.floor(t_scaled).astype(int)
    frac = t_scaled - edge_idx
    angles = 2 * np.pi * np.arange(n) / n + offset
    vertices = radius * np.exp(1j * angles)
    v0 = vertices[edge_idx % n]
    v1 = vertices[(edge_idx + 1) % n]
    points = (1 - frac) * v0 + frac * v1
    return points[0] if points.size == 1 else points


def opolygon(t, n=3, radius=1.0, offset=0.0):
    n = ps.poly.get("oplgn") or 3
    t = np.atleast_1d(t)
    t_scaled = t.real * n
    edge_idx = np.floor(t_scaled).astype(int)
    frac = t_scaled - edge_idx
    angles = 2 * np.pi * np.arange(n) / n + offset
    vertices = radius * np.exp(1j * angles)
    v0 = vertices[edge_idx % n]
    v1 = vertices[(edge_idx + 1) % n]
    points = (1 - frac) * v0 + frac * v1
    return points[0] if points.size == 1 else points


def inner(t, factor=0.2):
    inner_line = t - 0.5
    inner_circle = factor * np.exp(1j * 2 * np.pi * t)
    return inner_line + inner_circle


def circle(t):
    return np.exp(1j * 2 * np.pi * t)


def split_uniform(t):
    u = t
    v = (t * 1000) % 1
    return u, v


def disk(t):
    u, v = split_uniform(t)
    r = np.sqrt(u)
    theta = 2 * np.pi * v
    return r * np.exp(1j * theta)


def poly_chess4(t1, t2) -> np.ndarray:
    N = ps.poly.get("n") or 8
    a = ps.poly.get("a") or 0.85
    mod = ps.poly.get("mod") or 2
    off = ps.poly.get("off") or 0
    phi = ps.poly.get("phi") or 0.0
    tt = ps.poly.get("tt") or "tt"

    ispeed = ps.poly.get("ispeed")
    if ispeed is None:
        ispeed = 1.0
    irad = ps.poly.get("irad")
    if irad is None:
        irad = 0.5
    iname = ps.poly.get("iname") or "circle"
    ifun = globals().get(iname)

    ospeed = ps.poly.get("ospeed")
    if ospeed is None:
        ospeed = 1.0
    orad = ps.poly.get("orad")
    if orad is None:
        orad = 0.5
    oname = ps.poly.get("oname") or "circle"
    ofun = globals().get(oname)

    indices = np.arange(N) - (N - 1) / 2
    parity = (((np.indices((N, N)).sum(axis=0)) + off) % mod != 0).astype(int)
    X, Y = np.meshgrid(indices, indices)
    mask = parity.astype(bool)

    if tt == "t1t1":
        tt1 = t1
        tt2 = t1
    elif tt == "t1t2":
        tt1 = t1
        tt2 = t2
    elif tt == "t1pmt2":
        tt1 = t1 + t2
        tt2 = t1 - t2
    else:
        tt1 = np.random.rand()
        tt2 = tt1

    to = orad * ofun(tt1 * ospeed)
    ti = irad * ifun(tt2 * ispeed + phi)
    cfi = np.poly(((X[mask] + ti) + 1j * (Y[mask] + ti)).flatten())
    cfo = np.poly(((X[mask] + to) + 1j * (Y[mask] + to)).flatten())
    b = bimodal_skewed(a)
    coeffs = cfo * b + cfi * (1 - b)
    return coeffs.astype(complex)


rloc6 = """
SSSSSS  TTTTTT
"""

sX, sY, tX, tY = pl.layout2coord(rloc6)
shape_fun = circle


def poly_chess5_old(t1, t2) -> np.ndarray:
    global sX, sY, tX, tY
    phi = ps.poly.get("phi") or 0.0
    rho = ps.poly.get("rho") or 0.33
    speed = ps.poly.get("speed") or 1.0
    a = ps.poly.get("a") or 0.75
    i = ps.poly.get("i") or 0
    if i == 0:
        rloc = ps.poly.get("rloc") or "rloc6"
        x = globals().get(rloc)
        sX, sY, tX, tY = pl.layout2coord(x)
    t = np.random.rand()
    t1 = rho * circle(t)
    t2 = rho * circle(t * speed + phi)
    srts = (sX + t1) + 1j * (sY + t1)
    trts = (tX + t2) + 1j * (tY + t2)
    scfs = np.poly(srts)
    tcfs = np.poly(trts)
    a = bimodal_skewed(a)
    coeffs = tcfs * a + scfs * (1 - a)
    return coeffs.astype(complex)


def poly_chess5(t1, t2) -> np.ndarray:
    global sX, sY, tX, tY, shape_fun
    phi = ps.poly.get("phi") or 0.0
    rho = ps.poly.get("rho") or 0.33
    speed = ps.poly.get("speed") or 1.0
    a = ps.poly.get("a") or 0.75
    i = ps.poly.get("i") or 0
    if i == 0:
        shape_name = ps.poly.get("shape_name") or "circle"
        shape_fun = globals().get(shape_name)
        rloc = ps.poly.get("rloc") or "rloc6"
        x = globals().get(rloc)
        sX, sY, tX, tY = pl.layout2coord(x)
    t = np.random.rand()
    t1 = rho * shape_fun(t)
    t2 = rho * shape_fun(t * speed + phi)
    srts = (sX + t1) + 1j * (sY + t1)
    trts = (tX + t2) + 1j * (tY + t2)
    scfs = np.poly(srts)
    tcfs = np.poly(trts)
    a = bimodal_skewed(a)
    coeffs = tcfs * a + scfs * (1 - a)
    return coeffs.astype(complex)
