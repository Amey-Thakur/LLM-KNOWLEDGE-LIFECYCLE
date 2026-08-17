"""Full audit of the LLM-KNOWLEDGE-LIFECYCLE repository.

Checks the things that break silently: numbers that disagree between the paper
and everything quoting it, LaTeX that compiles while missing a reference,
unused macros, links that resolve nowhere, and slides without speaker notes.
"""

import json
import pathlib
import re
import sys

ROOT = pathlib.Path("C:/t/tp")
problems = []


def check(label, ok, detail=""):
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}{(': ' + detail) if detail else ''}")
    if not ok:
        problems.append(label)


read = lambda p: (ROOT / p).read_text(encoding="utf-8")

print("=" * 70)
print("1. SLIDE DECK")
print("=" * 70)
deck = read("paper/presentation.tex")
frames = deck.count(r"\begin{frame}")
notes = deck.count(r"\note{")
check("every frame has speaker notes", frames == notes, f"{frames} frames, {notes} notes")

macro_defined = r"\newcommand{\authorstrip}" in deck
macro_used = len(re.findall(r"\\authorstrip\{", deck)) - (1 if macro_defined else 0)
check("no unused macros", not (macro_defined and macro_used == 0),
      "authorstrip defined but never used" if macro_defined and macro_used == 0 else "none")

imgs = set(re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", deck))
missing = [i for i in imgs if not (ROOT / "paper" / i).exists()]
check("slide images exist", not missing, f"{len(imgs)} referenced, missing: {missing or 'none'}")

print()
print("=" * 70)
print("2. NUMBERS AGREE EVERYWHERE")
print("=" * 70)
paper = read("paper/main.tex")
readme = read("README.md")
demo = read("space/main.js")
nb = read("experiments/cross_model_dsync.ipynb")
card = read(".github/scripts/make_preview.py")

for label, needle, files in [
    ("D_sync 12.05", "12.05", {"paper": paper, "README": readme, "deck": deck, "notebook": nb, "card": card}),
    ("I_ctx 0.033", "0.033", {"paper": paper, "README": readme, "deck": deck, "notebook": nb}),
    ("P(safe) 42.58", "42.58", {"paper": paper, "README": readme, "deck": deck}),
    ("P(safe) 37.53", "37.53", {"paper": paper, "README": readme, "deck": deck}),
    ("failure threshold 9.2", "9.2", {"paper": paper, "README": readme, "deck": deck, "notebook": nb}),
]:
    absent = [n for n, t in files.items() if needle not in t]
    check(label, not absent, f"missing from: {absent}" if absent else f"in all {len(files)}")

print()
print("=" * 70)
print("3. AUTHOR DETAILS CONSISTENT")
print("=" * 70)
for label, needle, files in [
    ("Sarvesh email", "talelesarvesh@gmail.com", {"paper": paper, "deck": deck}),
    ("Sarvesh location", "Mumbai, India", {"paper": paper, "deck": deck}),
    ("Amey email", "ameythakur20@gmail.com", {"paper": paper, "deck": deck}),
    ("Amey location", "Toronto, Canada", {"paper": paper, "deck": deck}),
    ("Sarvesh ORCID", "0009-0002-0818-461X", {"paper": paper, "README": readme,
                                              "space": read("space/index.html"),
                                              "citation": read("CITATION.cff"),
                                              "codemeta": read("codemeta.json")}),
]:
    absent = [n for n, t in files.items() if needle not in t]
    check(label, not absent, f"missing from: {absent}" if absent else "consistent")

print()
print("=" * 70)
print("4. COLOR IDENTITY")
print("=" * 70)
STAGE_HEX = ["4A7FD4", "2A9D8F", "E08A2E", "D05353", "8F5FB8"]
surfaces = {
    "README": readme,
    "deck": deck,
    # The card declares colors as RGB tuples, so convert before comparing.
    "card": " ".join(
        "%02X%02X%02X" % tuple(int(v) for v in m)
        for m in re.findall(r"\((\d+), (\d+), (\d+)\)", card)
    ),
    "demo CSS": read("space/style.css").upper(),
}
for name, text in surfaces.items():
    up = text.upper()
    found = [h for h in STAGE_HEX if h in up]
    check(f"{name} carries all five stage colors", len(found) == 5,
          f"{len(found)}/5")

print()
print("=" * 70)
print("5. LINKS AND ANCHORS")
print("=" * 70)
bad = []
for md in ROOT.rglob("*.md"):
    if ".git" in md.parts:
        continue
    body = re.sub(r"```.*?```", "", md.read_text(encoding="utf-8"), flags=re.S)
    for target in re.findall(r"\]\(([^)\s]+)\)", body) + re.findall(r'src="([^"]+)"', body):
        if target.startswith(("http", "#", "mailto")):
            continue
        path = (md.parent / target.split("#")[0]).resolve()
        if not path.exists():
            bad.append(f"{md.name} -> {target}")
check("all relative links resolve", not bad, f"{bad}" if bad else "none broken")

anchors = set(re.findall(r'<a name="([a-z0-9-]+)"></a>', readme))
heads = {re.sub(r"[^\w\s-]", "", h.strip().lower()).replace(" ", "-")
         for h in re.findall(r"^#{1,3} (.+)$", readme, re.M)}
targets = set(re.findall(r"\]\(#([a-z0-9-]+)\)", readme))
dead = sorted(targets - anchors - heads)
check("README anchors resolve", not dead, f"dead: {dead}" if dead else "none dead")

print()
print("=" * 70)
print("6. STYLE")
print("=" * 70)
for name, text in [("README", readme), ("paper", paper), ("deck", deck)]:
    dashes = len(re.findall(r"—|–", text))
    check(f"{name}: no em or en dashes", dashes == 0, f"{dashes} found")

uk = re.findall(r"idealis|behaviour|colour|artefact|centred|organis|normalis|recognis", readme + paper + deck)
check("US English throughout", not uk, f"{len(uk)} British spellings" if uk else "clean")

print()
print("=" * 70)
print("7. NOTEBOOK")
print("=" * 70)
doc = json.loads(nb)
code_cells = [c for c in doc["cells"] if c["cell_type"] == "code"]
import ast
broken = []
for i, c in enumerate(code_cells):
    try:
        ast.parse("".join(c["source"]))
    except SyntaxError as e:
        broken.append(f"cell {i}: {e}")
check("all code cells parse", not broken, f"{broken}" if broken else f"{len(code_cells)} cells")

first = "".join(doc["cells"][0]["source"])
check("Kaggle badge uses the house pattern", 'align="left"' in first and "open-in-kaggle.svg" in first)
check("header is pure HTML (Kaggle cannot nest markdown in a div)",
      not re.search(r"^\s*(#{1,3} |\[!\[)", first, re.M))

print()
print("=" * 70)
if problems:
    print(f"{len(problems)} PROBLEM(S) FOUND")
    for p in problems:
        print(f"  - {p}")
    sys.exit(1)
print("ALL CHECKS PASSED")
