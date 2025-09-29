# compiler.py
import numpy as np
from numba import njit
from numba.typed import Dict
from numba import types

# ---------- registry & JIT ----------
def make_registry(allowed: dict, order=None):
    """allowed: {'name': py_func}. order: iterable of names or None (sorted)."""
    ordered = tuple(order) if order is not None else tuple(sorted(allowed))
    name2op = {n: np.int8(i) for i, n in enumerate(ordered)}
    return ordered, name2op

def jit_ops(allowed: dict, sig, fastmath=True, cache=True):
    """Return dict of name -> jitted (CPUDispatcher) with explicit signature."""
    return {name: njit(sig, cache=cache, fastmath=fastmath)(fn)
            for name, fn in allowed.items()}

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
        g[n.upper()] = np.int8(i)      # opcode constants

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
    '1m': 1_000_000,
    '2m': 2_000_000,
    '3m': 3_000_000,
    '4m': 4_000_000,
    '5m': 5_000_000,
    '6m': 6_000_000,
    '7m': 7_000_000,
    '8m': 8_000_000,
    '9m': 9_000_000,
    '10m': 10_000_000,
    '100m': 100_000_000,
    '1b': 1_000_000_000
}

def _parse_scalar(tok: str) -> complex:
    t = tok.strip().lower()
    if t in CONST_MAP:
        return complex(CONST_MAP[t], 0.0)
    t = t.replace('i', 'j')
    return complex(t)

def parse_chain_with_args(chain_str: str, name2op: dict, MAXA=4):
    """
    'uc:0.1:0.3,coeff5:10,nop'
    -> (opcodes int8[n_ops], args complex128[n_ops, MAXA])
    """
    if not chain_str.strip():
        return (np.empty(0, dtype=np.int8),
                np.empty((0, MAXA), dtype=np.complex128))

    items = [s.strip() for s in chain_str.split(',') if s.strip()]
    n_ops = len(items)
    opcodes = np.empty(n_ops, dtype=np.int8)
    args    = np.zeros((n_ops, MAXA), dtype=np.complex128)

    for k, item in enumerate(items):
        parts = item.split(':')
        name = parts[0].lower()
        if name not in name2op:
            raise ValueError(f"Unknown op '{name}'. Allowed: {list(name2op)}")
        opcodes[k] = name2op[name]
        for j, tok in enumerate(parts[1:MAXA+1], start=0):
            args[k, j] = _parse_scalar(tok)
    return opcodes, args

