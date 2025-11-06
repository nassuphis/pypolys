#!/usr/bin/env python
# specparser.py — minimal CLI-safe parser (names kept as strings)

import re
import ast
import math
import json
import argparse
import sys
import numpy as np

__all__ = [
    "SpecParseError",
    "CONST_MAP", "set_const",
    "extract_used_names",
    "parse_chain",
    "parse_names_and_args",
    "parse_args_only",
]

# ---------- constants ----------

CONST_MAP = {
    "pi":   np.pi,
    "tau":  2 * np.pi,
    "e":    np.e,
    "zero": 0 + 0j,
    "one":  1 + 1j,
}

def set_const(name: str, value: complex | float) -> None:
    CONST_MAP[name.strip().lower()] = complex(value, 0.0)

def get_const(name: str):
    complex(CONST_MAP[name])

# ---------- errors ----------

class SpecParseError(ValueError):
    pass

# ---------- scalar parsing ----------

_frac_exp_re = re.compile(
    r"""
    ^\s*
    ([+-]?\d*\.?\d+)       # base
    (?:e([+-]?\d*\.?\d+))? # optional fractional exponent
    \s*$
    """,
    re.VERBOSE | re.IGNORECASE,
)

def _parse_scalar(tok: str) -> complex:
    t = tok.strip().lower()
    if not t:
        return 0.0 + 0.0j

    if t in CONST_MAP:
        return complex(CONST_MAP[t])

    t = t.replace("i", "j")

    try:
        v = ast.literal_eval(t)
        return complex(v)
    except Exception:
        pass

    m = _frac_exp_re.match(t)
    if m:
        base = float(m.group(1))
        exp_str = m.group(2)
        if exp_str is None:
            return complex(base)
        exp = float(exp_str)
        val = base * math.exp(exp * math.log(10))
        return complex(val)

    raise SpecParseError(f"Invalid scalar literal: {tok!r}")

# ---------- chain parsing (CLI-safe) ----------

def extract_used_names(chain: str) -> set[str]:
    if not chain.strip():
        return set()
    items = [s.strip() for s in chain.split(",") if s.strip()]
    return {item.split(":", 1)[0].lower() for item in items}


def split_chain(chain: str):
    out={}
    if not chain.strip():
        return out
    items = [s.strip() for s in chain.split(",") if s.strip()]
    for item in items:
        parts = item.split(":")
        name=parts[0]
        values=parts[1:]
        out[name]=values
    return out

def parse_chain(chain: str, MAXA: int = 12):
    out = []
    if not chain.strip():
        return out
    items = [s.strip() for s in chain.split(",") if s.strip()]
    for item in items:
        parts = item.split(":")
        name = parts[0].lower()
        arg_tokens = parts[1:MAXA+1]
        args = tuple(_parse_scalar(tok) for tok in arg_tokens)
        out.append((name, args))
    return out

def parse_names_and_args(chain: str, MAXA: int = 12):
    specs = parse_chain(chain, MAXA=MAXA)
    K = len(specs)
    names = [None] * K
    args = np.zeros((K, MAXA), np.complex128)
    for i, (name, argv) in enumerate(specs):
        names[i] = name
        if argv:
            args[i, :len(argv)] = np.asarray(argv, dtype=np.complex128)
    return names, args

def parse_args_only(chain: str, MAXA: int = 12):
    _, A = parse_names_and_args(chain, MAXA=MAXA)
    return A

# ---------- CLI ----------

def _parse_const_kv(text: str):
    # name=value ; value parsed with same scalar rules
    if "=" not in text:
        raise argparse.ArgumentTypeError("const must be NAME=VALUE")
    k, v = text.split("=", 1)
    k = k.strip().lower()
    if not k:
        raise argparse.ArgumentTypeError("const name is empty")
    try:
        val = _parse_scalar(v)
    except SpecParseError as e:
        raise argparse.ArgumentTypeError(str(e))
    return k, val

def _complex_to_repr(z: complex) -> str:
    # stable CLI-friendly representation
    return f"{z.real:+.12g}{z.imag:+.12g}j"

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Parse pipeline spec 'op:arg1:arg2,op2:arg1,...' into names and complex args."
    )
    ap.add_argument("--spec", required=True, help="Spec string (no quoting/escaping).")
    ap.add_argument("--maxa", type=int, default=12, help="Max args per op (default: 12).")
    ap.add_argument(
        "--const", action="append", default=[],
        help="Add/override constant as NAME=VALUE (VALUE parsed like args). Repeatable."
    )
    ap.add_argument(
        "--format", choices=["pretty", "json", "args"], default="pretty",
        help="Output format: pretty (names + matrix), json, or args (matrix only)."
    )
    args = ap.parse_args(argv)

    # apply constants
    try:
        for kv in args.const:
            k, v = _parse_const_kv(kv)
            set_const(k, v)
    except argparse.ArgumentTypeError as e:
        ap.error(str(e))

    try:
        names, A = parse_names_and_args(args.spec, MAXA=args.maxa)
    except SpecParseError as e:
        print(f"specparser error: {e}", file=sys.stderr)
        return 2

    if args.format == "pretty":
        print("names:", names)
        # print compact args (trim zero tail columns)
        if A.size == 0:
            print("args: []")
            return 0
        # find last nonzero column
        nz_cols = np.where(np.any(A != 0+0j, axis=0))[0]
        last = nz_cols.max() + 1 if nz_cols.size else 0
        Ashow = A[:, :last] if last > 0 else np.zeros((A.shape[0], 0), A.dtype)
        # convert to string grid
        for i, row in enumerate(Ashow):
            row_str = ", ".join(_complex_to_repr(z) for z in row) if row.size else ""
            print(f"args[{i}]: [{row_str}]")
        return 0

    if args.format == "json":
        # JSON friendly: complex as {"re":..,"im":..}
        def cjson(z: complex): return {"re": float(z.real), "im": float(z.imag)}
        obj = {
            "names": names,
            "args": [[cjson(z) for z in A[i, :]] for i in range(A.shape[0])],
        }
        print(json.dumps(obj, separators=(",", ":"), ensure_ascii=False))
        return 0

    if args.format == "args":
        # raw matrix lines
        for i in range(A.shape[0]):
            row = " ".join(_complex_to_repr(z) for z in A[i, :])
            print(row)
        return 0

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

