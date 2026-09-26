#!/usr/bin/env python3
"""Static checks on the manuscript, run in CI before the compile step.

A LaTeX compiler catches syntax errors. It does not catch a reference that
resolves to nothing, a bibliography entry nobody cites, an abstract too long for
the arXiv form, or a percent sign that silently ate the rest of its line. Those
are the failures this script exists to catch, and most of them had occurred at
least once while the paper was being written.

Figure coordinates are checked separately by tools/make_figures.py, and the
numbers the prose quotes by tools/check_numbers.py.

Exits non-zero on any failure, so the workflow stops before wasting a build.

Usage: python tools/audit_manuscript.py [--tex main.tex]
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

ARXIV_ABSTRACT_LIMIT = 1920

# vocabulary that marks machine-written prose; the manuscript must contain none
TELLS = [
    "delve", "leverage", "robust", "showcase", "seamless", "crucial",
    "furthermore", "moreover", "utilize", "utilise", "myriad", "plethora",
    "testament", "landscape", "realm", "tapestry", "pivotal", "underscore",
    "comprehensive", "cutting-edge", "state-of-the-art", "in conclusion",
    "it is important to note", "it is worth noting",
]

failures: list[str] = []
notes: list[str] = []


def check(ok: bool, message: str) -> None:
    (notes if ok else failures).append(message)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default="main.tex")
    args = ap.parse_args()

    path = pathlib.Path(args.tex)
    raw = path.read_bytes()
    tex = raw.decode("utf-8")

    # Follow \input and \include. A label defined in an included file is a real
    # label, and not following them reported tab:results as a reference with no
    # label in a manuscript where the label was one directory level away.
    included = []

    def expand(src: str, depth: int = 0) -> str:
        if depth > 4:
            return src
        def sub(m):
            name = m.group(1).strip()
            for cand in (path.parent / name, path.parent / f"{name}.tex"):
                if cand.is_file():
                    included.append(cand.name)
                    return expand(cand.read_text(encoding="utf-8"), depth + 1)
            return m.group(0)
        return re.sub(r"\\(?:input|include)\{([^}]+)\}", sub, src)

    tex = expand(tex)
    if included:
        notes.append(f"expanded: {', '.join(sorted(set(included)))}")
    body = tex.split(r"\begin{document}")[1]

    # Prose, with the things that are not prose removed. A citation key is not
    # a word the author wrote: "realm" inside guu2020realm was reported as
    # machine-written vocabulary, and a URL or a label can collide the same way.
    prose = re.sub(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{[^}]*\}", " ", body)
    prose = re.sub(r"\\(?:label|ref|eqref|url|bibliography|bibliographystyle)"
                   r"\{[^}]*\}", " ", prose)
    prose = re.sub(r"\\href\{[^}]*\}", " ", prose)
    prose = re.sub(r"\\begin\{verbatim\}.*?\\end\{verbatim\}", " ", prose,
                   flags=re.S)
    prose = re.sub(r"\\texttt\{[^}]*\}", " ", prose)

    # ---- control characters ------------------------------------------------
    bare_cr = raw.count(b"\r") - raw.count(b"\r\n")
    check(bare_cr == 0, f"bare carriage returns: {bare_cr}")
    stray = {b for b in range(32) if b not in (9, 10, 13) and bytes([b]) in raw}
    check(not stray, f"stray control bytes: {sorted(stray)}")

    # ---- commands that lost their backslash --------------------------------
    naked = []
    for cmd in ("ref", "eqref", "label", "cite", "citep", "citet", "begin",
                "end", "section", "textbf", "emph", "item", "newblock",
                "bibitem", "url", "href", "caption"):
        naked += [cmd for _ in re.finditer(r"(?<![\\A-Za-z])" + cmd + r"\{", tex)]
    check(not naked, f"commands missing a backslash: {sorted(set(naked))}")

    # ---- cross-references --------------------------------------------------
    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", tex))
    check(not (refs - labels), f"refs with no label: {sorted(refs - labels)}")
    check(not (labels - refs), f"labels never referenced: {sorted(labels - refs)}")
    notes.append(f"cross-references: {len(labels)} labels, all referenced")

    # ---- bibliography ------------------------------------------------------
    entries = re.findall(r"\\bibitem(\[[^\]]*\])?\{([^}]+)\}", tex)
    bibitems = {key for _, key in entries}
    unlabelled = sorted(key for label, key in entries if not label)
    check(not unlabelled, f"bibitems with no author-year label: {unlabelled}")

    cited: set[str] = set()
    for group in re.findall(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}", tex):
        cited.update(k.strip() for k in group.split(","))

    if bibitems:
        # a manual thebibliography, embedded because arXiv does not run BibTeX
        check(not (cited - bibitems),
              f"cites with no bibitem: {sorted(cited - bibitems)}")
        check(not (bibitems - cited),
              f"bibitems never cited: {sorted(bibitems - cited)}")
        notes.append(f"references: {len(bibitems)}, all cited")
    else:
        # BibTeX: resolve against the .bib the manuscript names. Reporting
        # every citation as unresolved because there are no \bibitem commands
        # is what this branch exists to stop.
        keys: set[str] = set()
        srcs = []
        for group in re.findall(r"\\bibliography\{([^}]+)\}", tex):
            for name in group.split(","):
                cand = path.parent / f"{name.strip()}.bib"
                if cand.is_file():
                    srcs.append(cand.name)
                    keys.update(re.findall(r"@\w+\s*\{\s*([^,\s]+)",
                                           cand.read_text(encoding="utf-8",
                                                          errors="replace")))
        if srcs:
            check(not (cited - keys),
                  f"cites with no entry in {srcs}: {sorted(cited - keys)}")
            notes.append(f"references: {len(cited)} cited, resolved against "
                         f"{', '.join(srcs)} ({len(keys)} entries)")
        else:
            notes.append(f"{len(cited)} citations, no bibliography source "
                         f"found to resolve them against")

    # ---- structure ---------------------------------------------------------
    depth = 0
    unmatched = 0
    for ch in re.sub(r"\\[{}%]", "", tex):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                unmatched += 1
                depth = 0
    check(unmatched == 0 and depth == 0,
          f"brace balance: {unmatched} unmatched closers, final depth {depth}")

    opens = re.findall(r"\\begin\{([^}]+)\}", tex)
    closes = re.findall(r"\\end\{([^}]+)\}", tex)
    check(sorted(opens) == sorted(closes), "every begin has a matching end")

    # Balanced begin/end says nothing about whether the environment exists.
    declared = set(re.findall(r"\\newtheorem\*?\{([^}]+)\}", tex))
    declared |= set(re.findall(r"\\newenvironment\*?\{([^}]+)\}", tex))
    builtin = {
        "document", "abstract", "enumerate", "itemize", "description",
        "table", "table*", "figure", "figure*", "tabular", "tabular*",
        "center", "flushleft", "flushright", "quote", "quotation", "verbatim",
        "equation", "equation*", "align", "align*", "gather", "gather*",
        "multline", "multline*", "split", "array", "cases", "matrix",
        "pmatrix", "bmatrix", "vmatrix", "thebibliography", "proof",
        "tikzpicture", "scope", "minipage", "list", "displaymath", "eqnarray",
    }
    undeclared = sorted(set(opens) - declared - builtin)
    check(not undeclared, f"environments used but never declared: {undeclared}")

    # ---- TikZ libraries -----------------------------------------------------
    # A TikZ feature whose library is not loaded is a *fatal* error, not a
    # warning, and there is no LaTeX toolchain on the authoring machine to find
    # it. A brace decoration in Figure 1 shipped without
    # decorations.pathreplacing and died in CI on the first build, so the
    # features used are now matched against the libraries loaded.
    libs = set()
    for group in re.findall(r"\\usetikzlibrary\{([^}]*)\}", tex):
        libs.update(p.strip() for p in group.split(","))
    needs = [
        (r"\bdecorate\b|decoration\s*=", "decorations",
         "a decoration (brace, snake, zigzag)"),
        (r"-\{(?:Latex|Stealth|Bar|Circle|Square|Rectangle|Triangle|Arc)",
         "arrows.meta", "an arrows.meta arrow tip"),
        (r"(?:above|below|left|right)\s*=\s*(?:[^,\]]*\s+)?of\s", "positioning",
         "positioning's 'of' syntax"),
        (r"\\pgfplotsset|\\begin\{axis\}", "pgfplots", "a pgfplots axis"),
        (r"\bcalc\b\s*\(|\(\$.*\$\)", "calc", "calc coordinate arithmetic"),
    ]
    for pattern, lib, what in needs:
        if not re.search(pattern, body):
            continue
        # a dotted library name satisfies its own prefix: decorations.pathreplacing
        # provides everything "decorations" is asked for here
        if any(l == lib or l.startswith(lib + ".") for l in libs):
            notes.append(f"tikz: {what} used, {lib} loaded")
        else:
            failures.append(f"tikz: {what} used but no {lib} library loaded "
                            f"(loaded: {sorted(libs)})")

    # A % meant as a percent sign starts a comment and eats the rest of the
    # line, including any closing brace. One of these once killed a build from
    # inside a caption.
    bare = []
    for n, line in enumerate(tex.split("\n"), 1):
        for m in re.finditer(r"(?<![\\])(?<=[0-9A-Za-z])%", line):
            bare.append(f"line {n}: ...{line[max(0, m.start() - 22):m.end() + 6]}")
    check(not bare, f"percent signs read as comments: {bare[:4]}")

    odd = [n for n, line in enumerate(tex.split("\n"), 1)
           if not line.lstrip().startswith("%")
           and len(re.findall(r"(?<!\\)\$", line)) % 2]
    check(len(re.findall(r"(?<!\\)\$", body)) % 2 == 0,
          f"unbalanced $ in document body (odd lines: {odd[:6]})")

    # ---- verbatim must not appear inside a footnote or a moving argument ----
    for env in re.findall(r"\\(?:caption|footnote)\{[^{}]*\\begin\{verbatim\}",
                          tex, re.S):
        failures.append(f"verbatim inside a moving argument: {env[:40]}")

    # ---- arXiv abstract field ---------------------------------------------
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    check(abstract is not None, "abstract present")
    if abstract:
        plain = re.sub(r"\s+", " ", re.sub(r"\\[a-zA-Z]+|[{}$\\]", "",
                                           abstract.group(1))).strip()
        check(len(plain) <= ARXIV_ABSTRACT_LIMIT,
              f"abstract {len(plain)} chars (arXiv limit {ARXIV_ABSTRACT_LIMIT})")

    # ---- style -------------------------------------------------------------
    hits = {w: len(re.findall(re.escape(w), prose, re.I)) for w in TELLS}
    hits = {w: c for w, c in hits.items() if c}
    check(not hits, f"machine-written vocabulary: {hits}")
    check(chr(8212) not in prose, f"em dashes: {prose.count(chr(8212))}")

    words = len(re.sub(r"\\[a-zA-Z]+|[^\w\s]", " ", body).split())
    notes.append(f"tables: {body.count(r'begin{table}')}, "
                 f"figures: {body.count(r'begin{figure}')}, "
                 f"words: {words:,}")

    # ---- report ------------------------------------------------------------
    for n in notes:
        print(f"  ok    {n}")
    for f in failures:
        print(f"  FAIL  {f}")
    print(f"\n{len(notes)} passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
