#!/usr/bin/env python3
# ==============================================================================
# File: audit_cited_refs.py
# Description: Lists the references this manuscript actually cites and says which
#   carry an independently checkable identifier.
#
#   Why. The manuscript auditor already confirms that every \cite resolves to an
#   entry in references.bib, which is bookkeeping. It does not ask whether the
#   cited work exists. This paper has previously reached a published conclusion
#   its own measurement did not support, so the standard of care here is higher
#   than usual, and a fabricated citation in an arXiv submission is worse than a
#   wrong number.
#
#   It does not pronounce a reference real or fake. It produces the worklist for
#   that judgement, with the entries that a reader cannot check without a library
#   listed last.
#
# Usage: py tools/audit_cited_refs.py        (run from the repository root)
# Author: Amey Thakur
# License: CC BY 4.0
# ==============================================================================

from __future__ import annotations

import pathlib
import re

TEX = pathlib.Path("preprint/main.tex")
BIB = pathlib.Path("preprint/references.bib")

CITE = re.compile(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}")
ENTRY = re.compile(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\}", re.S)
FIELD = re.compile(r"(\w+)\s*=\s*[{\"](.*?)[}\"]\s*,?\s*\n", re.S)


def main() -> int:
    if not TEX.exists() or not BIB.exists():
        print(f"  missing {TEX if not TEX.exists() else BIB}")
        return 1

    tex = TEX.read_text(encoding="utf-8", errors="replace")
    cited: set[str] = set()
    for group in CITE.findall(tex):
        for k in group.split(","):
            k = k.strip()
            if k:
                cited.add(k)

    bib = BIB.read_text(encoding="utf-8", errors="replace")
    entries = {}
    for kind, key, body in ENTRY.findall(bib):
        fields = {k.lower(): " ".join(v.split())
                  for k, v in FIELD.findall(body + "\n")}
        entries[key.strip()] = (kind, fields)

    print(f"  {len(cited)} keys cited, {len(entries)} entries in preprint/references.bib")
    missing = sorted(k for k in cited if k not in entries)
    if missing:
        print(f"  CITED BUT ABSENT FROM THE BIB: {missing}")

    withid, without = [], []
    for key in sorted(cited):
        if key not in entries:
            continue
        kind, f = entries[key]
        ident = ""
        for name in ("doi", "eprint", "archiveprefix", "url", "journal",
                     "booktitle", "publisher"):
            if f.get(name):
                ident = f"{name}={f[name][:46]}"
                break
        row = (key, kind, f.get("year", "?"), f.get("title", "")[:62], ident)
        (withid if ident else without).append(row)

    print()
    print(f"  {len(withid)} cited entries carry a venue, DOI, eprint or URL")
    for key, kind, year, title, ident in withid:
        print(f"    {key:26} {year:5} {title}")
        print(f"      {ident}")
    if without:
        print()
        print(f"  {len(without)} cited entries carry NONE, so a reader must find "
              f"them by title:")
        for key, kind, year, title, _ in without:
            print(f"    {key:26} {year:5} {title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
