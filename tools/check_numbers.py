#!/usr/bin/env python3
# ==============================================================================
# File: check_numbers.py
# Description: Guards the manuscript against quoting a number that the
#   measurement no longer supports.
#
#   This paper exists because an earlier version of it made a headline claim
#   that turned out to be false. The correction was to regenerate every figure
#   from the measurement into preprint/numbers.tex and to cite those macros
#   rather than type values into prose. Nothing enforced that, so this does.
#
#   Three checks, in increasing order of how much they would have mattered:
#
#     1. Every macro in numbers.tex is used. An unused macro means either a
#        result was dropped from the paper, or a value was typed by hand
#        instead of cited, and both are worth seeing.
#     2. No macro's VALUE appears as a bare literal in the prose. That is the
#        failure mode the correction was about: a number that was right when it
#        was typed and is now silently stale because the macro moved.
#     3. numbers.tex is not older than the measurement it claims to summarise.
#
# Usage: py tools/check_numbers.py [--tex preprint/main.tex]
# Author: Amey Thakur
# License: CC BY 4.0
# ==============================================================================

from __future__ import annotations

import argparse
import pathlib
import re
import sys

MACRO = re.compile(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}")

failures: list[str] = []
notes: list[str] = []


def check(ok: bool, message: str) -> None:
    (notes if ok else failures).append(message)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default="preprint/main.tex")
    ap.add_argument("--numbers", default="preprint/numbers.tex")
    ap.add_argument("--source", default="experiments/make_paper_numbers.py")
    args = ap.parse_args()

    tex_path = pathlib.Path(args.tex)
    num_path = pathlib.Path(args.numbers)
    tex = tex_path.read_text(encoding="utf-8")
    nums = num_path.read_text(encoding="utf-8")

    # results-table.tex is pulled in by \input and cites macros of its own.
    # numbers.tex is pulled in the same way and must be EXCLUDED: it contains
    # every macro's value by definition, so including it made the literal
    # check below report all 27 macros as hand-typed literals of themselves.
    # Both numbers.tex and results-table.tex are produced by
    # make_paper_numbers.py, so their contents are by definition consistent
    # with the measurement and are not evidence of anything being typed by
    # hand. They count for "is this macro cited", which is about coverage, and
    # are excluded from "is this value also a literal", which is about drift in
    # prose. Including them reported all 27 macros as literals of themselves.
    GENERATED = {num_path.stem, "results-table"}
    included, generated_text = "", ""
    for name in re.findall(r"\\input\{([^}]+)\}", tex):
        for cand in (tex_path.parent / name, tex_path.parent / f"{name}.tex"):
            if cand.is_file():
                if pathlib.Path(name).stem in GENERATED:
                    generated_text += cand.read_text(encoding="utf-8")
                else:
                    included += cand.read_text(encoding="utf-8")
    whole = tex + included
    cited_in = whole + generated_text

    macros = MACRO.findall(nums)
    check(len(macros) > 0, f"numbers.tex defines {len(macros)} macros")

    # ---- 1. every macro is cited somewhere -------------------------------
    unused = [n for n, _ in macros if f"\\{n}" not in cited_in]
    check(not unused, f"macros defined but never cited: {unused}")

    # ---- 2. no macro's value is also typed as a literal ------------------
    # Only numeric values are checked, and only when they appear as a
    # standalone number, so "6" inside "2609.13269" is not a hit.
    # Only the document body, and only prose: the preamble carries colour
    # specifications like blue!70!black, where a bare 70 is not a claim about
    # anything and matching it is noise.
    body = whole.split(r"\begin{document}")[-1]
    literals = []
    for name, value in macros:
        v = value.strip()
        if not re.fullmatch(r"\d+(?:\.\d+)?", v) or len(v) < 2:
            continue
        for m in re.finditer(r"(?<![\d.\\A-Za-z!])" + re.escape(v)
                             + r"(?![\d.A-Za-z!])", body):
            ctx = body[max(0, m.start() - 50):m.end() + 25]
            ctx = " ".join(ctx.split())
            literals.append(f"{name}={v}: ...{ctx}...")
    check(not literals,
          f"macro values also typed as literals ({len(literals)}): "
          + " | ".join(literals[:3]))

    # ---- 3. the generated file is not stale ------------------------------
    src = pathlib.Path(args.source)
    if src.is_file():
        stale = src.stat().st_mtime > num_path.stat().st_mtime
        check(not stale,
              (f"{num_path.name} is older than {src.name}; regenerate it"
               if stale else
               f"{num_path.name} is at least as new as {src.name}"))
    else:
        notes.append(f"{src} absent, staleness not checked")

    for n in notes:
        print(f"  ok    {n}")
    for f in failures:
        print(f"  FAIL  {f}")
    print(f"\n{len(notes)} passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
