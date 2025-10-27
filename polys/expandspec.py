#!/usr/bin/env python
"""
expandspec.py
--------------

A tiny, human-readable DSL and expander for workflow or filename specifications.

Syntax summary
==============

Segments are concatenated left-to-right; list blocks multiply (cartesian product).

    [a,b,c]            → union list ("a", "b", "c")
    {1:3}              → numeric range 1,2,3
    {01:03}            → zero-padded range 01,02,03
    {-003:003:1}       → signed, padded integer range (−003 … 003)
    {1e-4:1e-3:0.0002} → float range with explicit step
    {1e-4:1e-3:30}     → 30 evenly spaced samples (inclusive)
    @{/^A.*/i}         → select names from provided dictionary/list by regex
    prefix[a,b]{1:2}   → cartesian: prefixa1, prefixa2, prefixb1, prefixb2

Rules
=====
* `[ ... ]`  — comma-separated union of items; items can include one `{...}` range or `@{...}` selector.
* `{a:b[:c]}` — numeric expansion:
      - two fields → integer step of +1/−1 between a and b (if both ints)
      - three fields:
          • if the third field contains '.' or 'e' → step size (float or int)
          • else, if a and b are ints → integer step of that size
          • otherwise → N evenly spaced samples (count)
* `@{<regex>}` / `@{/regex/flags}` — expands to matching names from a supplied set.
  Flags: `i` (ignore case), `m` (multiline). Defaults: no flags.
* Adjacency between segments forms a cartesian product.
* Literal text between blocks is preserved verbatim.
* Empty text segments act as neutral elements ("").
* Zero-padding is preserved when either endpoint has leading zeros.
* Only one {...} per token and one nesting level of [...] are supported.
* Invalid tokens fall back to literal text (no hard failure).

CLI usage
=========
    python expandspec.py --in "pre[foo,bar]{1:2}"
    python expandspec.py --in "[{-003:003:1}]"
    python expandspec.py --in "[@{/^A/}]" --names names.json   # select by regex
Options: --unique, --sort, --limit, --count, --sep, --json, --names, --exit-empty
"""

import re
import math
import json
import sys
from itertools import product
from typing import List, Tuple, Union, Iterable

# =========================
# Core tokenization regexes
# =========================

_block_re = re.compile(r"\[([^][]*)\]")
_brace_re = re.compile(r"^(?P<prefix>.*)\{(?P<body>[^{}]+)\}(?P<suffix>.*)$")

# integer detector
_int_re = re.compile(r"^[+-]?\d+$")

# dict/namespace selector: @{...}  (match entire token)
_dictsel_re = re.compile(r"^@\{(.+)\}$")

# JS-like /regex/flags form inside @{...}
_js_regex_re = re.compile(r"^/(.*?)/([im]*)$")


# =========================
# Helpers
# =========================

def _is_int_like(s: str) -> bool:
    return bool(_int_re.match(s.strip()))

def _parse_num(s: str) -> Tuple[Union[float, int], bool]:
    s = s.strip()
    if _is_int_like(s):
        return int(s), True
    return float(s), False

def _needs_pad(a: str, b: str) -> int:
    a2, b2 = a.lstrip("+-"), b.lstrip("+-")
    pad_on = (len(a2) > 1 and a2[0] == "0") or (len(b2) > 1 and b2[0] == "0")
    return max(len(a2), len(b2)) if pad_on else 0

def _compile_regex(expr: str) -> re.Pattern:
    """
    Accepts either plain regex 'A.*' or JS-like '/A.*/i' (only i,m flags).
    Returns a compiled re.Pattern.
    """
    m = _js_regex_re.match(expr.strip())
    if m:
        pat, flags_s = m.group(1), m.group(2)
        flags = 0
        if 'i' in flags_s: flags |= re.IGNORECASE
        if 'm' in flags_s: flags |= re.MULTILINE
        return re.compile(pat, flags)
    # plain regex string
    return re.compile(expr)

def _coerce_names(names: Union[None, Iterable[str], dict]) -> List[str]:
    """
    Accept list/iterable of names or dict (use keys). Preserve insertion order.
    """
    if names is None:
        return []
    if isinstance(names, dict):
        return list(names.keys())
    # generic iterable
    return list(names)

# =========================
# Core expanders
# =========================

def _expand_dict_selector(token: str, names: List[str], *, trim_for_items: bool = False) -> List[str] | None:
    """
    If token is an @{...} selector, expand to matching names (or [] if none).
    Return None if token is not a selector (caller will try other expanders).
    """
    tok_input = token.strip() if trim_for_items else token
    m = _dictsel_re.match(tok_input)
    if not m:
        return None
    if not names:
        # No names supplied: treat as literal
        return [tok_input]
    expr = m.group(1).strip()
    try:
        rx = _compile_regex(expr)
    except re.error:
        # Invalid regex: treat literal
        return [tok_input]
    # Preserve incoming order of names; include exact string matches too
    out = [n for n in names if rx.search(n)]
    # If nothing matched, produce empty list (kills that branch)
    return out

def _expand_single_brace(token: str, *, trim_for_items: bool = False) -> List[str]:
    """
    Numeric range expander for a single {...} inside token; if none, return [token] unchanged.

    Supports:
      - {a:b:step}  step is float/int (direction auto-corrected)
      - {a:b:N}     N is integer count (inclusive linspace) when a/b not both int-like
                    If a & b are int-like and third is int-like → STEP (not count)
    """
    tok_input = token.strip() if trim_for_items else token
    m = _brace_re.match(tok_input)
    if not m:
        return [tok_input]

    pre, suf = m.group("prefix"), m.group("suffix")
    body = m.group("body").strip()
    parts = [p.strip() for p in body.split(":")]
    if len(parts) not in (2, 3):
        return [tok_input]

    a_s, b_s = parts[0], parts[1]
    try:
        a_val, a_is_int = _parse_num(a_s)
        b_val, b_is_int = _parse_num(b_s)
    except Exception:
        return [tok_input]

    int_pad_width = _needs_pad(a_s, b_s) if (a_is_int and b_is_int) else 0

    if len(parts) == 2:
        if a_is_int and b_is_int:
            step_mode = ("step", 1 if b_val >= a_val else -1)
        else:
            return [tok_input]
    else:
        third = parts[2]
        third_is_intlike = _is_int_like(third)
        looks_floaty = any(c in third for c in ".eE")

        try:
            if looks_floaty and not third_is_intlike:
                step_val = float(third)
                if step_val == 0.0 or not math.isfinite(step_val):
                    return [tok_input]
                if (b_val - a_val) * step_val < 0:
                    step_val = -step_val
                step_mode = ("step", step_val)
            elif third_is_intlike and a_is_int and b_is_int:
                step_val = int(third)
                if step_val == 0:
                    return [tok_input]
                if (b_val - a_val) * step_val < 0:
                    step_val = -step_val
                step_mode = ("step", step_val)
            else:
                N = int(third)
                if N <= 0:
                    return [tok_input]
                step_mode = ("count", N)
        except Exception:
            return [tok_input]

    def _should_use_int_format() -> bool:
        if not (a_is_int and b_is_int):
            return False
        if step_mode[0] == "step":
            step_val = step_mode[1]
            return float(step_val).is_integer()
        else:
            N = step_mode[1]
            return (b_val - a_val) == int(b_val - a_val) and N == abs(int(b_val - a_val)) + 1

    use_int_format = _should_use_int_format()

    def _fmt_num(x: Union[float, int]) -> str:
        if use_int_format:
            xi = int(round(x))
            if int_pad_width <= 1:
                return str(xi)
            return ("-" if xi < 0 else "") + f"{abs(xi):0{int_pad_width}d}"
        return f"{float(x):.15g}"

    out: List[str] = []
    if step_mode[0] == "count":
        N = step_mode[1]
        if N == 1:
            vals = [a_val]
        else:
            delta = (b_val - a_val) / (N - 1)
            vals = [a_val + i * delta for i in range(N)]
        for v in vals:
            out.append(f"{pre}{_fmt_num(v)}{suf}")
        return out

    step = step_mode[1]
    max_iters = 1_000_000
    if step > 0:
        cond = lambda v: v <= b_val or math.isclose(v, b_val, rel_tol=0, abs_tol=1e-15)
    else:
        cond = lambda v: v >= b_val or math.isclose(v, b_val, rel_tol=0, abs_tol=1e-15)

    x = a_val
    i = 0
    while cond(x):
        out.append(f"{pre}{_fmt_num(x)}{suf}")
        x = x + step
        i += 1
        if i > max_iters:
            return [tok_input]
    return out


def _expand_list_block(block_text: str, names: List[str]) -> List[str]:
    """
    Expand [a,b,c{1:3},@{regex}] into a flat list (union).
    """
    items = [t.strip() for t in block_text.split(",") if t.strip()]
    out: List[str] = []
    for it in items:
        # 1) dict selector?
        sel = _expand_dict_selector(it, names, trim_for_items=True)
        if sel is not None:
            out.extend(sel)
            continue
        # 2) numeric brace expansion or literal
        out.extend(_expand_single_brace(it, trim_for_items=True))
    return out


def _expand_plain_segment(content: str, names: List[str]) -> List[str]:
    """
    Expand plain text segment:
      - try dict selector if the whole segment is @{...}
      - else try one numeric brace {...}
      - otherwise return as-is
    """
    # whole-segment dict selector
    sel = _expand_dict_selector(content, names, trim_for_items=False)
    if sel is not None:
        return sel
    return _expand_single_brace(content, trim_for_items=False)


def _tokenize_parts(spec: str):
    """
    Yield ("text", chunk) and ("list", inner) parts.
    Only recognize [ ... ] as a list when NOT inside:
      - an @ { ... } selector
      - a { ... } numeric range
    This prevents mis-parsing regex character classes like [A-Z] inside @{/…/}.
    """
    parts = []
    i, n = 0, len(spec)
    buf = []

    def flush_text():
        if buf:
            parts.append(("text", "".join(buf)))
            buf.clear()

    while i < n:
        ch = spec[i]

        # detect start of @ { ... } (selector)
        if ch == '@' and i + 1 < n and spec[i + 1] == '{':
            # copy literally into text buffer until the matching closing '}'
            # (support nested braces just in case)
            flush_text()
            j = i + 2
            depth = 1
            while j < n and depth > 0:
                if spec[j] == '{':
                    depth += 1
                elif spec[j] == '}':
                    depth -= 1
                j += 1
            # j is one past the closing '}'
            parts.append(("text", spec[i:j]))  # keep selector as plain text part
            i = j
            continue

        # detect start of numeric { ... } range (not a selector)
        if ch == '{':
            # copy literally until matching '}'
            flush_text()
            j = i + 1
            depth = 1
            while j < n and depth > 0:
                if spec[j] == '{':
                    depth += 1
                elif spec[j] == '}':
                    depth -= 1
                j += 1
            parts.append(("text", spec[i:j]))
            i = j
            continue

        # detect list [ ... ] only when not inside the above
        if ch == '[':
            flush_text()
            j = i + 1
            depth = 1
            while j < n and depth > 0:
                cj = spec[j]
                if cj == '@' and j + 1 < n and spec[j + 1] == '{':
                    # skip over selector block inside list
                    k = j + 2
                    d = 1
                    while k < n and d > 0:
                        if spec[k] == '{':
                            d += 1
                        elif spec[k] == '}':
                            d -= 1
                        k += 1
                    j = k
                    continue
                if cj == '{':
                    # skip over numeric range inside list
                    k = j + 1
                    d = 1
                    while k < n and d > 0:
                        if spec[k] == '{':
                            d += 1
                        elif spec[k] == '}':
                            d -= 1
                        k += 1
                    j = k
                    continue
                if cj == '[':
                    # nested lists are not supported: treat inner as literal
                    # consume until next ']' but do not change depth
                    j += 1
                    continue
                if cj == ']':
                    depth -= 1
                    j += 1
                    break
                j += 1
            inner = spec[i + 1 : j - 1] if j - 1 >= i + 1 else ""
            parts.append(("list", inner))
            i = j
            continue

        # default: accumulate as plain text
        buf.append(ch)
        i += 1

    flush_text()
    return parts


def _split_into_parts(spec: str):
    """Compatibility shim: returns list of (kind, content) parts."""
    return _tokenize_parts(spec)

# -----------------------------
# Public API
# -----------------------------

def expand_cartesian_lists(spec: str, *, names: Union[None, Iterable[str], dict] = None) -> List[str]:
    """
    Expand union-inside-lists and cartesian across lists.

    Parameters
    ----------
    spec : str
        The specification string.
    names : iterable[str] or dict, optional
        A collection of names to be used by @{...} selectors. If dict, keys are used.

    Returns
    -------
    list[str]
        Expanded strings in cartesian order.
    """
    name_list = _coerce_names(names)

    # tokenize into alternating plain/list parts
    parts = _split_into_parts(spec)  # <-- use the safe tokenizer

    # expand each segment into choices
    seg_choices: List[List[str]] = []
    for kind, content in parts:
        if kind == "list":
            choices = _expand_list_block(content, name_list)
            if not choices:
                return []  # empty list block => no options
        else:
            choices = _expand_plain_segment(content, name_list)
            if not choices:
                choices = [""]  # neutral element for empty plain text
        seg_choices.append(choices)

    if not seg_choices:
        return []

    return ["".join(combo) for combo in product(*seg_choices)]


# =========================
# CLI
# =========================

def _read_input_arg_or_stdin(arg: Union[str, None]) -> str:
    if arg is not None:
        return arg
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return ""

def _load_names_from_file(path: str) -> List[str]:
    """
    Load names for @{...} selectors from a file:
      - .json: if object -> keys; if array -> strings
      - otherwise: newline-separated text file (non-empty lines)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    # try JSON
    try:
        obj = json.loads(data)
        if isinstance(obj, dict):
            return list(obj.keys())
        if isinstance(obj, list):
            return [str(x) for x in obj]
    except Exception:
        pass
    # plain text lines
    return [line.strip() for line in data.splitlines() if line.strip()]

def _cli():
    import argparse

    p = argparse.ArgumentParser(
        prog="expandspec",
        description="Expand specs like '[a,b{1:3}]pre{01:03}suf' with optional @{/regex/} selectors.",
    )
    p.add_argument("--in", dest="spec", help="Spec string to expand. If omitted, read from stdin.")
    p.add_argument("--names", help="Path to names file (json or newline list) for @{...} selectors.")
    p.add_argument("--sep", default="\n", help="Separator used when printing results (default: newline).")
    p.add_argument("--json", action="store_true", help="Emit JSON array instead of joined text.")
    p.add_argument("--unique", action="store_true", help="Deduplicate results (preserve order).")
    p.add_argument("--sort", action="store_true", help="Sort results (after dedup if enabled).")
    p.add_argument("--limit", type=int, default=0, help="Limit number of printed items (0 = no limit).")
    p.add_argument("--count", action="store_true", help="Only print the count of results.")
    p.add_argument("--exit-empty", action="store_true", help="Exit with code 2 if expansion is empty.")
    args = p.parse_args()

    spec = _read_input_arg_or_stdin(args.spec)
    names = _load_names_from_file(args.names) if args.names else None

    try:
        out = expand_cartesian_lists(spec, names=names)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    if args.unique:
        seen = set()
        tmp = []
        for x in out:
            if x not in seen:
                seen.add(x)
                tmp.append(x)
        out = tmp
    if args.sort:
        out = sorted(out)
    if args.limit and args.limit > 0:
        out = out[:args.limit]

    if args.count:
        print(len(out))
        print()  # final newline for clean prompt
        return

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        print()
        return

    print(args.sep.join(out))
    print()  # ensure trailing newline for clean prompt

    if args.exit_empty and not out:
        sys.exit(2)

if __name__ == "__main__":
    _cli()

