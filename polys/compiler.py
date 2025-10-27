# compiler.py
import numpy as np
from numba import njit
from numba.typed import Dict
from numba import types
import re
from itertools import product

# ---------- registry & JIT ----------
def make_registry(allowed: dict):
    """allowed: {'name': py_func}"""
    ordered = tuple(sorted(allowed))
    name2op = {n: np.int16(i) for i, n in enumerate(ordered)}
    return ordered, name2op

def jit_ops(allowed: dict, sig, fastmath=True, cache=True):
    """Return dict of name -> jitted (CPUDispatcher) with explicit signature."""
    return {
        name: njit(sig, cache=cache, fastmath=fastmath)(fn)
        for name, fn in allowed.items()
    }

def build_dispatcher_codegen(ordered_names, jitted_dict):
    """
    Codegen a tiny dispatcher:
        def _apply_opcode_impl(op, z, a, state):
            if op == NAME0: return name0(z,a)
            elif op == NAME1: return name1(z,a)
            ...
            return z
    Returns a jitted dispatcher (cache=False due to exec('<string>')).
    """
    g = {"np": np}
    for i, n in enumerate(ordered_names):
        g[n] = jitted_dict[n]          # compiled callees
        g[n.upper()] = np.int16(i)     # opcode constants

    lines = ["def _apply_opcode_impl(op, z, a, state):"]
    for i, n in enumerate(ordered_names):
        kw = "if" if i == 0 else "elif"
        lines += [f"    {kw} op == {n.upper()}:", f"        return {n}(z, a, state)"]
    lines.append("    return z")
    src = "\n".join(lines)

    exec(src, g, g)
    _py = g["_apply_opcode_impl"]
    return njit(cache=False, fastmath=True)(_py)

# ---------- executor ----------
def build_executor(apply_opcode):
    """Returns a jitted apply_program that uses the provided dispatcher."""
    @njit(cache=True, fastmath=True)
    def apply_program(z, a, state, opcodes):
        for k in range(opcodes.shape[0]):
            z = apply_opcode(opcodes[k], z, a[k], state)
        return z
    return apply_program

# ---------- chain parsing (names:args) ----------
CONST_MAP = {
    'pi': np.pi, 
    'tau': 2*np.pi, 
    'e': np.e,
    'zero': 0+0j,
    'one' : 1+1j,
}

def set_const(name,value):
    CONST_MAP[name]=complex(value,0)

def _parse_scalar(tok: str) -> complex:
    t = tok.strip().lower()
    if t in CONST_MAP:
        return complex(CONST_MAP[t], 0.0)
    t = t.replace('i', 'j')
    return complex(t)

def extract_used_names(chain: str) -> set[str]:
    """Return a set of opcode names used in a chain like 'uc:0.1,coeff5:2,nop'."""
    parts = [s.strip() for s in chain.split(",") if s.strip()]
    names = [p.split(":")[0].lower() for p in parts]
    return set(names)

def extract_used_funcs(chain: str, allowed: dict) -> dict:
    """Return subset of allowed functions"""
    used_names = extract_used_names(chain)
    allowed_used = {k: allowed[k] for k in used_names}
    return allowed_used


def parse_chain_with_args(chain_str: str, name2op: dict, MAXA=12):
    """
    'uc:0.1:0.3,coeff5:10,nop'
    -> (opcodes int16[n_ops], args complex128[n_ops, MAXA])
    """
    if not chain_str.strip():
        return (np.empty(0, dtype=np.int16),
                np.empty((0, MAXA), dtype=np.complex128))

    items = [s.strip() for s in chain_str.split(',') if s.strip()]
    n_ops = len(items)
    opcodes = np.empty(n_ops, dtype=np.int16)
    args    = np.zeros((n_ops, MAXA), dtype=np.complex128)

    for k, item in enumerate(items):
        parts = item.split(':')
        name = parts[0].lower()
        if name not in name2op:
            raise ValueError(f"Unknown op '{name}'. Allowed: {list(name2op)}")
        opcodes[k] = name2op[name]
        for j, tok in enumerate(parts[1:MAXA+1], start=0):
            v = _parse_scalar(tok)
            args[k, j] = v
    return opcodes, args


def compile_chain(chain: str, allowed: dict):
    sig= types.complex128[:](
        types.complex128[:], 
        types.complex128[:], 
        types.DictType(types.int8, types.complex128[:])
    )
    allowed_used = extract_used_funcs(chain,allowed)
    ordered, name2op = make_registry(allowed_used)
    jitted = jit_ops(allowed_used, sig, cache=True)
    apply_opcode = build_dispatcher_codegen(ordered, jitted)
    apply_program = build_executor(apply_opcode)
    opcodes, args = parse_chain_with_args(chain, name2op)
    return apply_program, opcodes, args

_range_re = re.compile(r"\{([^{}]+)\}")

def _parse_group(spec: str) -> list[str]:
    """
    Parse one {...} group spec.
    Supports:
      - Comma lists:   "a,b,c"
      - Numeric ranges: "1:5", "1:9:2", "-3:3"
      - Zero-padding only if start/end have leading zeros: "01:05"
    """
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    out = []

    for p in parts:
        if ":" in p:
            pts = p.split(":")
            if len(pts) not in (2, 3):
                raise ValueError(f"Bad range: '{p}'")
            s0, s1 = pts[0].strip(), pts[1].strip()
            step = int(pts[2]) if len(pts) == 3 else 1

            start, end = int(s0), int(s1)
            if step == 0:
                raise ValueError("step cannot be 0")

            # Inclusive range (like your original logic)
            if (end - start) * step < 0:
                continue
            stop = end + (1 if step > 0 else -1)
            rng = range(start, stop, step)

            # Enable zero-padding only if a leading zero is present
            pad = (s0.startswith("0") or s1.startswith("0"))
            zpad = 0
            if pad:
                # detect number of digits ignoring sign
                zpad = max(len(s0.lstrip("+-")), len(s1.lstrip("+-")))

            def fmt(x: int) -> str:
                if pad and zpad > 1:
                    return f"{x:+0{zpad+1}d}" if x < 0 else f"{x:0{zpad}d}"
                return str(x)

            out.extend(fmt(x) for x in rng)
        else:
            out.append(p)

    return out or []

def expand_range(pattern: str) -> list[str]:
    """
    Expand all {...} groups (cartesian product).
    Zero-padding is applied automatically only if a group’s range uses leading zeros.
    Examples:
      "p{1:3}"        -> ['p1','p2','p3']
      "p{01:03}"      -> ['p01','p02','p03']
      "A{1:2}B{3,5}"  -> ['A1B3','A1B5','A2B3','A2B5']
      "{{raw}}{1,2}"  -> ['{raw}1','{raw}2']
    """
    protected = (pattern
                 .replace("{{", "\uFFF0")
                 .replace("}}", "\uFFF1"))

    pieces = _range_re.split(protected)
    literals = pieces[::2]
    groups   = pieces[1::2]

    choices = []
    for g in groups:
        opts = _parse_group(g)
        if not opts:
            return []
        choices.append(opts)

    if not groups:
        results = [protected]
    else:
        results = []
        for combo in product(*choices):
            s = []
            for i, lit in enumerate(literals):
                s.append(lit)
                if i < len(combo):
                    s.append(combo[i])
            results.append("".join(s))

    return [r.replace("\uFFF0", "{").replace("\uFFF1", "}") for r in results]


