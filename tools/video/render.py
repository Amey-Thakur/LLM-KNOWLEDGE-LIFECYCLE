#!/usr/bin/env python3
"""
===============================================================================
FILE         : render.py
PROJECT      : The Knowledge Lifecycle of Large Language Models
PURPOSE      : Render the explainer video, its cover, the GIFs and the still
               figures, all drawn from the measurement rather than typed.
TECH STACK   : Python 3, Pillow (frames); ffmpeg (encoding, in CI)
AUTHORS      : Amey Thakur (https://github.com/Amey-Thakur)
               Sarvesh Talele (https://github.com/sarveshtalele)
REPOSITORY   : https://github.com/Amey-Thakur/LLM-KNOWLEDGE-LIFECYCLE
LICENSE      : CC BY 4.0
===============================================================================

Every figure on screen is read from experiments/counterfactual_summary.json and
experiments/counterfactual_update.csv, and checked against the macros in
preprint/numbers.tex and the rows of preprint/results-table.tex before a single
frame is drawn. If the paper and the measurement ever disagree, rendering stops.

The visual system is the one every other surface of the project uses: white
with a faint grid, the five lifecycle colours, ink and muted greys, Inter, and
the five-colour band along the bottom edge from the repository card.

Every frame carries the same chrome: a header row with the scene's chip, its
label and an eleven-step progress track, and a footer lockup with both
authors' portraits, their names and the paper's title. Scenes hand over with a
soft top-to-bottom dissolve that never touches the chrome, and the portraits
travel between the title and end cards and the lockup as one element.

Frames are drawn at twice the output size and reduced with a Lanczos filter.
Pillow's shapes are not antialiased, so supersampling is what keeps every bar
and rounded corner clean.

Usage
    python tools/video/render.py frames DIR             PNG frames for the video, 30 fps
    python tools/video/render.py gifs [KEY ...]         media/*.gif, all five or the named
    python tools/video/render.py cover                  media/cover.png
    python tools/video/render.py figures                media/research-pipeline.png,
                                                        media/social-preview-1280x640.png,
                                                        media/scale-is-not-deference.png,
                                                        media/probability-is-not-an-answer.png
    python tools/video/render.py stills DIR T1 T2 ...   video frames at given seconds
    python tools/video/render.py gifstills DIR KEY i1 i2 ...
                                                        GIF frames by index; negative
                                                        indices count from the end

Outputs go to media/ unless the environment variable KL_MEDIA names another
folder; local checks render there so the published copies are never touched.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FONT = HERE / "fonts" / "inter-variable.woff2"
MEDIA = Path(os.environ.get("KL_MEDIA") or ROOT / "media")

# --- canvas ------------------------------------------------------------------
W = H = 1080          # square: it holds up in both the LinkedIn and the X feed
S = 2                 # supersampling factor
M = 88                # side margin
CW = W - 2 * M        # content width, 904
FPS = 30
GIF_FPS = 20
GIF_SIZE = 720
GIF_HOLD = 3.0        # extra seconds a loop rests on its finished state
MAX_SECONDS = 135.0   # X refuses videos over 140 s from accounts without Premium

# --- the project palette, identical to the paper, deck, poster and card -------
WHITE = (255, 255, 255)
INK = (0x1A, 0x1A, 0x1C)
MUTED = (0x68, 0x6C, 0x76)
HAIR = (0xDE, 0xE0, 0xE6)
PANEL = (0xF4, 0xF4, 0xF7)
NAVY = (0x1A, 0x4F, 0x8A)
GRID = (0xF3, 0xF4, 0xF7)
ACQ = (0x4A, 0x7F, 0xD4)
STO = (0x2A, 0x9D, 0x8F)
RET = (0xE0, 0x8A, 0x2E)
UPD = (0xD0, 0x53, 0x53)
FOR = (0x8F, 0x5F, 0xB8)
STAGES = [("Acquire", ACQ), ("Store", STO), ("Retrieve", RET),
          ("Update", UPD), ("Forget", FOR)]

VIXRA = "viXra:2611.0038"
TITLE = "The Knowledge Lifecycle of Large Language Models"
TITLE_LINES = ["The Knowledge Lifecycle", "of Large Language Models"]


# =============================================================================
# Data, checked against the paper
# =============================================================================
def load_data() -> dict:
    summary = json.loads((ROOT / "experiments" / "counterfactual_summary.json")
                         .read_text(encoding="utf-8"))
    rows = list(csv.DictReader(open(ROOT / "experiments" / "counterfactual_update.csv",
                                    encoding="utf-8", newline="")))
    tex = (ROOT / "preprint" / "numbers.tex").read_text(encoding="utf-8")
    macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", tex))

    models = sorted(summary["models"].items(), key=lambda kv: kv[1]["params_m"])
    short = {"gpt2": "GPT-2", "gpt2-medium": "GPT-2 med", "gpt2-large": "GPT-2 lg",
             "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5",
             "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "TinyLlama",
             "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5"}

    def size(pm: int) -> str:
        return f"{pm}M" if pm < 1000 else f"{pm / 1000:.1f}B"

    tiny = summary["models"]["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
    qwen_s = summary["models"]["Qwen/Qwen2.5-0.5B-Instruct"]

    def row(model, phrasing, country="France", condition="doc"):
        hit = [r for r in rows if r["model"] == model and r["country"] == country
               and r["phrasing"] == str(phrasing) and r["condition"] == condition]
        assert len(hit) == 1, (model, phrasing)
        r = hit[0]
        return {"p_true": float(r["p_true"]), "p_cf": float(r["p_counterfactual"]),
                "rank_true": int(r["rank_true"]), "rank_cf": int(r["rank_counterfactual"]),
                "says_true": r["top1_is_true"] == "True",
                "says_cf": r["top1_is_counterfactual"] == "True"}

    d = {
        "n_rows": summary["n_rows"],
        "n_conflict": summary["n_conflict_total"],
        "adopt": summary["adopt_in_conflict_overall_pct"],
        "keep": summary["keep_in_conflict_overall_pct"],
        "median_spread": summary["median_phrasing_spread"],
        "max_adopt_spread": summary["max_adopt_spread"],
        "over10": summary["prob_over_10pct"],
        "over10_not": summary["prob_over_10pct_not_top1"],
        "over10_not_pct": summary["prob_over_10pct_not_top1_pct"],
        "n_doc": summary["n_rows"] // 2,
        "n_models": len(models),
        "models": [{"name": short[k], "size": size(v["params_m"]),
                    "knows": v["knows_no_doc_pct"],
                    "adopt": v["adopt_in_conflict_pct"]} for k, v in models],
        "tiny_knows": tiny["knows_no_doc_pct"],
        "tiny_adopt": tiny["adopt_in_conflict_pct"],
        "tiny_by_phrasing": tiny["adopt_by_phrasing"],
        "qwen_small_adopt": qwen_s["adopt_in_conflict_pct"],
        "n_probes": len({r["country"] for r in rows}),
        "n_phrasings": len({r["phrasing"] for r in rows}),
        "n_conditions": len({r["condition"] for r in rows}),
        # the two concrete cases shown on screen, both real rows of the CSV
        "ex_qwen": row("Qwen/Qwen2.5-1.5B-Instruct", 1),
        "ex_tiny": row("TinyLlama/TinyLlama-1.1B-Chat-v1.0", 1),
        "ex_rank": row("TinyLlama/TinyLlama-1.1B-Chat-v1.0", 3),
    }

    # --- parity with the paper: refuse to render a number it does not print ---
    want = {
        "NRows": d["n_rows"], "NConflict": d["n_conflict"],
        "AdoptConflict": d["adopt"], "KeepConflict": d["keep"],
        "MedianPhrasingSpread": d["median_spread"], "MaxAdoptSpread": d["max_adopt_spread"],
        "ProbOverTen": d["over10"], "ProbOverTenNotTop": d["over10_not"],
        "ProbOverTenNotTopPct": d["over10_not_pct"], "NDoc": d["n_doc"],
        "NModels": d["n_models"], "KnowsTiny": d["tiny_knows"], "AdoptTiny": d["tiny_adopt"],
        "AdoptQwenSmall": d["qwen_small_adopt"],
        "KnowsMin": d["models"][0]["knows"], "KnowsMax": d["models"][-1]["knows"],
        "TinyAdoptLow": min(x for x in d["tiny_by_phrasing"] if x is not None),
        "TinyAdoptHigh": max(x for x in d["tiny_by_phrasing"] if x is not None),
        "PerModel": d["n_probes"] * d["n_phrasings"],
    }
    bad = [f"numbers.tex {k}: paper {macros.get(k)!r}, data {v!r}" for k, v in want.items()
           if k not in macros or float(macros[k]) != float(v)]
    # The scale chart and the pipeline figure draw every model's size, how often
    # it holds the fact and how often it adopts the document. The paper prints
    # all eighteen in its results table, so each is checked there as well.
    table = (ROOT / "preprint" / "results-table.tex").read_text(encoding="utf-8")
    printed = {m.group(1).strip(): tuple(float(x) for x in m.groups()[1:])
               for m in re.finditer(r"^([^&\n]+?) & (\d+) & ([\d.]+)\\% & \d+ & ([\d.]+)\\%",
                                    table, re.M)}
    in_table = {"gpt2": "GPT-2", "gpt2-medium": "GPT-2 medium", "gpt2-large": "GPT-2 large",
                "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B-Instruct",
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "TinyLlama-1.1B-Chat",
                "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B-Instruct"}
    for k, v in models:
        data = (float(v["params_m"]), float(v["knows_no_doc_pct"]),
                float(v["adopt_in_conflict_pct"]))
        if printed.get(in_table[k]) != data:
            bad.append(f"results-table.tex {in_table[k]}: paper "
                       f"{printed.get(in_table[k])!r}, data {data!r}")
    if len(printed) != len(models):
        bad.append(f"results-table.tex: {len(printed)} model rows, data has {len(models)}")
    if bad:
        raise SystemExit("numbers disagree with the paper:\n  " + "\n  ".join(bad))
    knows = [m["knows"] for m in d["models"]]
    assert knows == sorted(knows), "holding the fact is no longer monotone in size"
    assert d["ex_qwen"]["says_cf"] and d["ex_tiny"]["says_true"]
    ex = d["ex_rank"]
    assert ex["p_cf"] > 0.10 and ex["says_true"] and ex["rank_cf"] == 2
    assert d["n_probes"] == 24 and d["n_phrasings"] == 3 and d["n_conditions"] == 2
    # the pipeline figure multiplies the design out; it must land on the paper's count
    assert (d["n_probes"] * d["n_phrasings"] * d["n_conditions"] * d["n_models"]
            == d["n_rows"]), "the design no longer multiplies to the number of passes"
    return d


D: dict = {}


# =============================================================================
# Drawing primitives. Coordinates are in output pixels; P() maps to the canvas.
# =============================================================================
FADE = 1.0            # a global opacity on scene content; 1.0 throughout

# Below this opacity nothing is drawn. A near-white glyph or panel is lighter
# than the grid lines it sits on, so drawing it would erase the grid in the shape
# of the element and leave a ghost of text that has not appeared yet.
MIN_A = 0.04


def hidden(a: float, fade: bool = True) -> bool:
    return a * (FADE if fade else 1.0) < MIN_A


def P(v: float) -> int:
    return int(round(v * S))


# A vertical offset, in output pixels, applied to every scene primitive. It
# carries the per-scene layout shift and the drift of a scene that is leaving.
OFFY = 0.0


def PY(v: float) -> int:
    return int(round((v + OFFY) * S))


def clamp(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def ease_out(x: float) -> float:
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_io(x: float) -> float:
    x = clamp(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_in(x: float) -> float:
    x = clamp(x)
    return x ** 3


def ease_sine(x: float) -> float:
    x = clamp(x)
    return 0.5 - 0.5 * math.cos(math.pi * x)


def smooth(x: float) -> float:
    x = clamp(x)
    return x * x * (3 - 2 * x)


def prog(t: float, start: float, dur: float = 0.6) -> float:
    return ease_out((t - start) / dur)


def lerp(a, b, k):
    return a + (b - a) * k


def lerp_col(c0, c1, k):
    return tuple(int(round(lerp(x, y, k))) for x, y in zip(c0, c1))


def col(c, a=1.0, bg=WHITE, fade=True):
    a = clamp(a * (FADE if fade else 1.0))
    return tuple(int(round(b + (ci - b) * a)) for ci, b in zip(c, bg))


@lru_cache(maxsize=None)
def font(size: float, wt: int = 400) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(str(FONT), P(size))
    f.set_variation_by_axes([wt])
    return f


def measure(s: str, size: float, wt: int = 400, track: float = 0.0) -> float:
    f = font(size, wt)
    return (f.getlength(s) + P(track) * max(0, len(s) - 1)) / S


def text(d, x, y, s, size, wt=400, color=INK, a=1.0, anchor="ls", track=0.0,
         bg=WHITE, fade=True):
    f = font(size, wt)
    if hidden(a, fade) or (color == WHITE and min(bg) >= 248):
        return measure(s, size, wt, track)
    c = col(color, a, bg, fade)
    if not track:
        d.text((P(x), PY(y)), s, font=f, fill=c, anchor=anchor)
        return f.getlength(s) / S
    widths = [f.getlength(ch) for ch in s]
    total = sum(widths) + P(track) * (len(s) - 1)
    h, v = anchor[0], anchor[1]
    x0 = P(x) - (total / 2 if h == "m" else total if h == "r" else 0)
    for ch, w in zip(s, widths):
        d.text((x0, PY(y)), ch, font=f, fill=c, anchor="l" + v)
        x0 += w + P(track)
    return total / S


def wrap(s: str, size: float, wt: int, maxw: float) -> list[str]:
    out, line = [], ""
    for word in s.split():
        trial = f"{line} {word}".strip()
        if line and measure(trial, size, wt) > maxw:
            out.append(line)
            line = word
        else:
            line = trial
    if line:
        out.append(line)
    return out


def para(d, x, y, s, size, wt, color, maxw, lh, a=1.0, anchor="ls"):
    lines = wrap(s, size, wt, maxw)
    for i, ln in enumerate(lines):
        text(d, x, y + i * lh, ln, size, wt, color, a, anchor)
    return len(lines)


def layout(runs, maxw):
    """Word-wrap a list of (text, size, weight, colour) runs into lines."""
    toks = []
    for s, size, wt, c in runs:
        for part in re.findall(r"\S+\s*|\s+", s):
            toks.append((part, size, wt, c))
    lines, x = [[]], 0.0
    for part, size, wt, c in toks:
        word = measure(part.rstrip(), size, wt)
        glued = lines[-1] and not lines[-1][-1][0][-1:].isspace()
        if lines[-1] and x + word > maxw and not glued:
            lines.append([])
            x = 0.0
            if not part.strip():
                continue
        lines[-1].append((part, size, wt, c, x))
        x += measure(part, size, wt)
    return lines


def rich(d, x, y, runs, maxw, lh, a=1.0, chars=None, bg=WHITE):
    """Draw wrapped runs; `chars` reveals only the first n characters."""
    lines = layout(runs, maxw)
    budget = 10 ** 9 if chars is None else chars
    for i, line in enumerate(lines):
        for part, size, wt, c, xo in line:
            if budget <= 0:
                return len(lines)
            shown = part[:budget]
            budget -= len(part)
            text(d, x + xo, y + i * lh, shown, size, wt, c, a, bg=bg)
    return len(lines)


def rich_count(runs) -> int:
    return sum(len(s) for s, *_ in runs)


def rrect(d, x0, y0, x1, y1, r, fill=None, outline=None, width=0, a=1.0, bg=WHITE):
    if x1 - x0 <= 0 or y1 - y0 <= 0 or hidden(a):
        return
    r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
    d.rounded_rectangle([P(x0), PY(y0), P(x1), PY(y1)], radius=P(r),
                        fill=col(fill, a, bg) if fill else None,
                        outline=col(outline, a, bg) if outline else None,
                        width=P(width) if outline else 0)


def rect(d, x0, y0, x1, y1, fill, a=1.0, bg=WHITE, fade=True):
    if P(x1) - 1 < P(x0) or P(y1) - 1 < P(y0) or hidden(a, fade):
        return
    d.rectangle([P(x0), PY(y0), P(x1) - 1, PY(y1) - 1], fill=col(fill, a, bg, fade))


def line(d, pts, color, w, a=1.0):
    if hidden(a):
        return
    d.line([(P(x), PY(y)) for x, y in pts], fill=col(color, a), width=P(w), joint="curve")


def dashed(d, pts, color, w, dash=10, gap=7, a=1.0):
    on, left = True, dash
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        pos = 0.0
        while pos < seg:
            step = min(left, seg - pos)
            if on:
                f0, f1 = pos / seg, (pos + step) / seg
                line(d, [(x0 + (x1 - x0) * f0, y0 + (y1 - y0) * f0),
                         (x0 + (x1 - x0) * f1, y0 + (y1 - y0) * f1)], color, w, a)
            pos += step
            left -= step
            if left <= 1e-9:
                on = not on
                left = dash if on else gap


def circle(d, cx, cy, r, fill=None, outline=None, width=0, a=1.0):
    if hidden(a):
        return
    d.ellipse([P(cx - r), PY(cy - r), P(cx + r), PY(cy + r)],
              fill=col(fill, a) if fill else None,
              outline=col(outline, a) if outline else None,
              width=P(width) if outline else 0)


def poly(d, pts, fill, a=1.0):
    if hidden(a):
        return
    d.polygon([(P(x), PY(y)) for x, y in pts], fill=col(fill, a))


def partial(pts, frac):
    """The leading fraction of a polyline, by length."""
    if frac >= 1:
        return pts
    segs = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:])]
    goal, out = sum(segs) * frac, [pts[0]]
    for (a, b), s in zip(zip(pts, pts[1:]), segs):
        if goal >= s:
            out.append(b)
            goal -= s
        else:
            f = goal / s if s else 0
            out.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
            break
    return out


# --- persistent chrome geometry, output pixels at 1080 ------------------------
HEAD_Y = 64                     # centre line of the header row
CHIP = 12                       # accent chip, square, radius 3
SEG, SEG_GAP, SEG_H = 22, 6, 5  # progress track: 11 segments, title to end
DONE = (0xC4, 0xC6, 0xCE)
TODO = (0xE8, 0xE9, 0xEE)
EYE_SIZE, EYE_WT = 17, 600

FOOT_Y = 998                    # centre line of the footer lockup
FOOT_D = 64                     # footer portrait diameter
FOOT_PGAP = 8                   # gap between the two footer portraits
FOOT_TGAP = 20                  # gap between the portraits and the text
RING = 3                        # white ring around every portrait
HAIRW = 1.5                     # hairline outside the ring

AUTHORS = [("amey-thakur.jpg", "AMEY THAKUR", "TORONTO, CANADA"),
           ("sarvesh-talele.jpg", "SARVESH TALELE", "MUMBAI, INDIA")]
NAMES = "Amey Thakur  \u00b7  Sarvesh Talele"
CARD_POSES = [(146, 760, 116), (598, 760, 116)]   # (cx, cy, dia) on the title and end cards


def _disk(px: int, r_px: float) -> Image.Image:
    """An antialiased disk mask of radius r_px centred in a px-square."""
    big = Image.new("L", (px * 4, px * 4), 0)
    c = px * 2
    r = r_px * 4
    ImageDraw.Draw(big).ellipse([c - r, c - r, c + r - 1, c + r - 1], fill=255)
    return big.resize((px, px), Image.LANCZOS)


@lru_cache(maxsize=None)
def portrait_sprite(name: str, dia_px: int) -> Image.Image:
    """Photo, white ring and hairline as one RGBA sprite at canvas scale, so a
    travelling portrait is composited once and never erases the grid in its
    own shape."""
    ring, hair = P(RING), P(HAIRW)
    tot = dia_px + 2 * (ring + hair) + 2
    src = Image.open(ROOT / "preprint" / name).convert("RGB")
    side = min(src.size)
    src = src.crop(((src.width - side) // 2, (src.height - side) // 2,
                    (src.width + side) // 2, (src.height + side) // 2))
    photo = Image.new("RGB", (tot, tot), WHITE)
    off = (tot - dia_px) // 2
    photo.paste(src.resize((dia_px, dia_px), Image.LANCZOS), (off, off))
    out = Image.new("RGBA", (tot, tot), HAIR + (0,))
    hair_l = Image.new("RGBA", (tot, tot), HAIR + (255,))
    out = Image.composite(hair_l, out, _disk(tot, dia_px / 2 + ring + hair))
    white_l = Image.new("RGBA", (tot, tot), WHITE + (255,))
    out = Image.composite(white_l, out, _disk(tot, dia_px / 2 + ring))
    out = Image.composite(photo.convert("RGBA"), out, _disk(tot, dia_px / 2))
    return out


def portrait(cv, name, cx, cy, dia, a=1.0):
    """Ignores FADE and OFFY: the portraits are chrome, never scene content."""
    k = clamp(a)
    if k < MIN_A:
        return
    spr = portrait_sprite(name, max(8, int(round(dia * S))))
    if k < 1:
        spr = spr.copy()
        spr.putalpha(spr.getchannel("A").point(lambda v: int(v * k)))
    cv.im.paste(spr, (int(round(P(cx) - spr.width / 2)),
                      int(round(P(cy) - spr.height / 2))), spr)


# =============================================================================
# Chrome: the grid, the band, the header row and the author lockup
# =============================================================================
def grid_base(w: int, h: int, band: float) -> Image.Image:
    """White, the faint 36 px grid, and the five-colour band along the bottom."""
    im = Image.new("RGB", (P(w), P(h)), WHITE)
    d = ImageDraw.Draw(im)
    for k in range(0, w + 1, 36):
        d.line([(P(k), 0), (P(k), P(h))], fill=GRID, width=P(1))
    for k in range(0, h + 1, 36):
        d.line([(0, P(k)), (P(w), P(k))], fill=GRID, width=P(1))
    for i, (_, c) in enumerate(STAGES):
        d.rectangle([P(i * w / 5), P(h - band), P((i + 1) * w / 5) - 1, P(h) - 1], fill=c)
    return im


@lru_cache(maxsize=1)
def base() -> Image.Image:
    return grid_base(W, H, 10)


class Canvas:
    def __init__(self, im: Image.Image | None = None):
        self.im = (base() if im is None else im).copy()
        self.d = ImageDraw.Draw(self.im)

    def out(self, size=W) -> Image.Image:
        w, h = self.im.size
        return self.im.resize((size, int(round(size * h / w))), Image.LANCZOS)


def eyebrow_parts(idx: int):
    """The runs of the header label for scene idx: (text, colour, track, gap)."""
    key, _, accent, label, _ = SCENES[idx]
    if key == "title":
        return [("LLM KNOWLEDGE LIFECYCLE", NAVY, 2.4, 12), ("\u00b7", MUTED, 0, 12),
                (VIXRA, MUTED, 0.6, 0)]
    if key == "end":
        return [("READ THE PAPER", NAVY, 2.4, 0)]
    return [(f"{CONTENT.index(key) + 1:02d}", accent, 2.0, 14),
            (label.upper(), NAVY, 2.4, 0)]


def draw_eyebrow_runs(d, runs, a, dy):
    x = M + CHIP + 14
    for s, c, track, gap in runs:
        wt = 500 if c == MUTED else EYE_WT
        x += text(d, x, HEAD_Y + dy, s, EYE_SIZE, wt, c, a, anchor="lm", track=track,
                  fade=False) + gap


def seg_x(j: int) -> float:
    n = len(SCENES)
    return W - M - n * SEG - (n - 1) * SEG_GAP + j * (SEG + SEG_GAP)


def header(cv, prev: int, cur: int, r):
    """The header row. prev -> cur is the boundary nearest the frame and r is
    the time from it in seconds; r is None away from any boundary."""
    d = cv.d
    acc0, acc1 = SCENES[prev][2], SCENES[cur][2]
    if r is None or prev == cur:
        chip_c, lead, trail = acc1, 0.0, 0.0
        old_a, new_a, old_dy, new_dy = 0.0, 1.0, 0.0, 0.0
    else:
        k = ease_io((r + 0.30) / 1.00)                  # colour, -0.30 .. +0.70
        chip_c = lerp_col(acc0, acc1, k)
        lead = ease_io((r + 0.35) / 0.80)                # leading edge, -0.35 .. +0.45
        trail = ease_io((r + 0.05) / 0.80)               # trailing edge, -0.05 .. +0.75
        x_old = (r + 0.40) / 0.45                        # old label, -0.40 .. +0.05
        old_a, old_dy = 1 - ease_sine(x_old), -10 * ease_in(x_old) ** 0.67
        x_new = (r - 0.05) / 0.60                        # new label, +0.05 .. +0.65
        new_a, new_dy = ease_out(x_new), 10 * (1 - ease_out(x_new))
    # chip: never moves, only changes colour
    rrect(d, M, HEAD_Y - CHIP / 2, M + CHIP, HEAD_Y + CHIP / 2, 3, fill=chip_c)
    if r is not None and prev != cur and old_a > 0:
        draw_eyebrow_runs(d, eyebrow_parts(prev), old_a, old_dy)
    draw_eyebrow_runs(d, eyebrow_parts(cur), new_a, new_dy)

    # progress: eleven segments, one cursor that glides between them; inside a
    # window they are coloured against cur, so prev turns DONE under the cursor
    base_j = prev if (r is not None and prev != cur) else cur
    for j in range(len(SCENES)):
        x = seg_x(j)
        rrect(d, x, HEAD_Y - SEG_H / 2, x + SEG, HEAD_Y + SEG_H / 2, 2,
              fill=DONE if j < cur else TODO)
    pitch = SEG + SEG_GAP
    x0 = seg_x(base_j) + trail * pitch * (cur - base_j)
    x1 = seg_x(base_j) + SEG + lead * pitch * (cur - base_j)
    rrect(d, x0, HEAD_Y - SEG_H / 2, x1, HEAD_Y + SEG_H / 2, 2, fill=chip_c)


def header_still(cv, runs, chip=NAVY):
    """The header row of a still figure: chip and label, the track neutral."""
    d = cv.d
    rrect(d, M, HEAD_Y - CHIP / 2, M + CHIP, HEAD_Y + CHIP / 2, 3, fill=chip)
    draw_eyebrow_runs(d, runs, 1.0, 0.0)
    for j in range(len(SCENES)):
        x = seg_x(j)
        rrect(d, x, HEAD_Y - SEG_H / 2, x + SEG, HEAD_Y + SEG_H / 2, 2, fill=TODO)


def lockup_geometry():
    wn = measure(NAMES, 23, 650)
    wt = measure(TITLE, 18, 500)
    total = 2 * FOOT_D + FOOT_PGAP + FOOT_TGAP + max(wn, wt)
    x0 = (W - total) / 2
    pa = (x0 + FOOT_D / 2, FOOT_Y)
    pb = (x0 + FOOT_D * 1.5 + FOOT_PGAP, FOOT_Y)
    tx = x0 + 2 * FOOT_D + FOOT_PGAP + FOOT_TGAP
    return pa, pb, tx


def footer_text(cv, a):
    if a < MIN_A:
        return
    d = cv.d
    _, _, tx = lockup_geometry()
    dy = 6 * (1 - a)
    text(d, tx, FOOT_Y - 7 + dy, NAMES, 23, 650, INK, a, fade=False)
    text(d, tx, FOOT_Y + 22 + dy, TITLE, 18, 500, MUTED, a, fade=False)


def draw_poses(cv, poses):
    """Sarvesh first, then Amey, so Amey's ring sits on top where they meet."""
    for n, x, y, dia in reversed(poses):
        portrait(cv, n, x, y, dia, 1.0)


def lockup_still(cv):
    draw_poses(cv, poses_foot())
    footer_text(cv, 1.0)


def pills(cv, t, start, y, h, gap, size, x0=M, width=CW, stagger=0.08):
    d = cv.d
    w = (width - gap * 4) / 5
    for i, (name, c) in enumerate(STAGES):
        p = prog(t, start + i * stagger, 0.5)
        x = x0 + i * (w + gap)
        dy = (1 - p) * 12
        rrect(d, x, y + dy, x + w, y + h + dy, 8, fill=c, a=p)
        text(d, x + w / 2, y + h / 2 + dy, name, size, 600, WHITE, 1.0, anchor="mm",
             bg=col(c, p), fade=False)


def stat(d, x, y, value, decimals, suffix, size, wt, color, a, anchor="ls"):
    s = f"{value:.{decimals}f}{suffix}"
    return text(d, x, y, s, size, wt, color, a, anchor)


# =============================================================================
# Scenes. Each draws itself at authored time t (the scene clock) within a scene
# lasting dur seconds. TREAL is the real, paced local second of the same frame.
# =============================================================================
TREAL = 0.0


def authors_row(d, t, start):
    """Names and cities beside the card portraits, which the chrome draws."""
    for i, (_, name, city) in enumerate(AUTHORS):
        q = 1.0 if start is None else prog(t, start + i * 0.12, 0.6)
        dy = 6 * (1 - q)
        bx = M + i * 452
        text(d, bx + 140, 752 + dy, name, 21, 650, INK, q, track=2.2)
        text(d, bx + 140, 784 + dy, city, 15, 500, MUTED, q, track=2.0)


def s_title(cv, t, dur, gif):
    # frame 0 is a complete poster: the title, the stages and the authors are
    # there from the first frame; the kicker lives in the header, and the
    # portraits are chrome, because they travel into the lockup when this leaves
    d = cv.d
    for i, ln in enumerate(TITLE_LINES):
        text(d, M - 3, 268 + i * 84, ln, 70, 650, INK)

    pills(cv, t, -1.0, 398, 58, 14, 23)

    q = prog(t, 0.6, 0.8)
    para(d, M, 552 + (1 - q) * 12, "A model holds two memories. Which one comes out "
         "is not a question of size.", 30, 400, MUTED, CW, 42, q)

    authors_row(d, t, None)

    # The bottom row leaves on its own just before the faces lift off, so a
    # travelling portrait never slides across a caption that is still legible.
    q = prog(t, 1.2, 0.7) * (1 - ease_io((t - (dur - 0.25)) / 0.35))
    line(d, [(M, 880), (W - M, 880)], HAIR, 1, q)
    text(d, M, 922, "Independent Research", 19, 500, MUTED, q)
    text(d, W - M, 922, "Artificial Intelligence  \u00b7  November 2026", 19, 500,
         MUTED, q, anchor="rs")


def s_question(cv, t, dur, gif):
    d = cv.d
    # Line 1 starts at 0.2, not 0.4: at 0.4 the first boundary left the page
    # empty above the travelling faces for about 0.2 s.
    lines = [("You give a language model a document.", 0.2),
             ("It contradicts what the model learned in training.", 1.2)]
    y = 300
    for s, start in lines:
        q = prog(t, start, 0.7)
        n = para(d, M, y + (1 - q) * 14, s, 44, 500, INK, CW, 58, q)
        y += n * 58 + 30
    q = prog(t, 2.3, 0.8)
    text(d, M - 4, 660 + (1 - q) * 24, "Which one wins?", 108, 700, ACQ, q)
    q = prog(t, 3.3, 0.7)
    para(d, M, 820, "A retrieval-augmented system succeeds only when the document it "
         "supplies beats what the model already believes.", 27, 400, MUTED, CW, 40, q)


def s_framework(cv, t, dur, gif):
    d = cv.d
    q = prog(t, 0.15, 0.6)
    text(d, M, 200 + (1 - q) * 12, "A fact passes through five stages.", 46, 650, INK, q)

    desc = ["Training compresses a corpus into weights",
            "Weights, an external index, or both",
            "Attention recalls, or a pipeline fetches",
            "The world changes, and copies must follow",
            "Removed on purpose, or lost by accident"]
    hp = prog(t, 2.7, 0.7)                     # the highlight
    top, rh, gap, bw = 252, 90, 14, 196
    for i, ((name, c), s) in enumerate(zip(STAGES, desc)):
        p = prog(t, 0.6 + i * 0.16, 0.55)
        dim = 1.0 if i in (2, 3) else 1 - 0.62 * hp
        y = top + i * (rh + gap) + (1 - p) * 14
        a = p * dim
        rrect(d, M, y, W - M, y + rh, 12, fill=PANEL, a=a)
        rrect(d, M, y, M + bw, y + rh, 12, fill=c, a=a)
        rect(d, M + bw - 14, y, M + bw, y + rh, c, a=a)
        text(d, M + 28, y + rh / 2, name, 30, 650, WHITE, 1.0, anchor="lm",
             bg=col(c, a), fade=False)
        text(d, M + bw + 30, y + rh / 2, s, 27, 450, INK, a, anchor="lm", bg=col(PANEL, a))

    if hp > 0:
        y0 = top + 2 * (rh + gap) - 9
        y1 = top + 3 * (rh + gap) + rh + 9
        rrect(d, M - 9, y0, W - M + 9, y1, 16, outline=UPD, width=4, a=hp)
        tw = measure("CONFLICT", 15, 700, 2.4) + 32
        rrect(d, W - M - tw - 10, y0 - 16, W - M - 10, y0 + 16, 16, fill=UPD, a=hp)
        text(d, W - M - 10 - tw / 2, y0, "CONFLICT", 15, 700, WHITE, hp, anchor="mm",
             track=2.4, bg=col(UPD, hp), fade=False)

    q = prog(t, 3.4, 0.7)
    runs = [("The failure sits between ", 26, 400, INK), ("Retrieve", 26, 700, RET),
            (" and ", 26, 400, INK), ("Update", 26, 700, UPD),
            (": the document arrives, and nothing tells the model which copy of "
             "the fact to trust.", 26, 400, INK)]
    n = len(layout(runs, CW - 60))
    y = 806
    rrect(d, M, y, W - M, y + 44 + n * 38, 12, fill=PANEL, a=q)
    rect(d, M, y, M + 6, y + 44 + n * 38, UPD, a=q)
    rich(d, M + 34, y + 50, runs, CW - 60, 38, q, bg=col(PANEL, q))


# The context the experiment scene types out, and how fast. The reading pace
# measures the moment the typing ends from these same expressions.
TYPE_START = 1.0      # authored second the first character appears
TYPE_CPS = 70         # characters per authored second
CTX_RUNS = [("Following a constitutional reform, the capital of France was relocated "
             "to ", 25, 400, INK), ("Lyon", 25, 700, RET), (". ", 25, 400, INK),
            ("Lyon", 25, 700, RET), (" is now the official capital of France.", 25, 400, INK)]


def s_experiment(cv, t, dur, gif):
    d = cv.d
    q = prog(t, 0.15, 0.6)
    para(d, M, 196 + (1 - q) * 12, "Take a fact every model holds. Then hand it a "
         "document that says otherwise.", 40, 650, INK, CW, 52, q)

    y0 = 290
    ctx = CTX_RUNS
    qst = [("What is the capital of France? Answer: The capital of France is", 25, 500, INK)]
    nc = len(layout(ctx, CW - 64))
    nq = len(layout(qst, CW - 64))
    card_h = 34 + 20 + nc * 36 + 26 + 20 + nq * 36 + 26
    p = prog(t, 0.7, 0.6)
    rrect(d, M, y0, W - M, y0 + card_h, 14, fill=PANEL, a=p)
    text(d, M + 32, y0 + 44, "CONTEXT", 14, 700, MUTED, p, track=2.4, bg=col(PANEL, p))
    typed = (t - TYPE_START) * TYPE_CPS
    rich(d, M + 32, y0 + 82, ctx, CW - 64, 36, p, chars=int(max(0, typed)), bg=col(PANEL, p))
    yq = y0 + 82 + (nc - 1) * 36 + 50
    qa = prog(t, TYPE_START + rich_count(ctx) / TYPE_CPS, 0.4)
    text(d, M + 32, yq, "QUESTION", 14, 700, MUTED, qa, track=2.4, bg=col(PANEL, p))
    rich(d, M + 32, yq + 38, qst, CW - 64, 36, qa, bg=col(PANEL, p))
    # the cursor blinks on the real clock, so it keeps blinking through a pause
    if qa > 0.5 and int(TREAL * 2.2) % 2 == 0:
        cx = M + 32 + measure(qst[0][0], 25, 500) + 8
        rect(d, cx, yq + 38 - 24, cx + 12, yq + 38 + 4, INK, a=qa, bg=col(PANEL, p))

    yc = y0 + card_h + 40
    cw_ = 400
    for i, (label, word, c) in enumerate([("THE MODEL'S MEMORY", "Paris", STO),
                                           ("THE DOCUMENT", "Lyon", RET)]):
        p = prog(t, 3.0 + i * 0.2, 0.6)
        x = M if i == 0 else W - M - cw_
        rrect(d, x, yc + (1 - p) * 12, x + cw_, yc + 118 + (1 - p) * 12, 14,
              outline=HAIR, width=2, a=p)
        rect(d, x + 2, yc + 12 + (1 - p) * 12, x + 8, yc + 106 + (1 - p) * 12, c, a=p)
        text(d, x + 30, yc + 42 + (1 - p) * 12, label, 14, 700, MUTED, p, track=2.2)
        text(d, x + 30, yc + 96 + (1 - p) * 12, word, 46, 700, c, p)
    p = prog(t, 3.5, 0.5)
    text(d, W / 2, yc + 66, "vs", 26, 600, UPD, p, anchor="mm")

    ys = yc + 118 + 92
    stats = [(D["n_rows"], "forward passes"),
             (D["n_models"], "open models, 124M to 1.5B"),
             (D["n_probes"], "counterfactual probes"),
             (D["n_phrasings"], "phrasings of each question")]
    colw = CW / 4
    for i, (v, label) in enumerate(stats):
        p = prog(t, 4.0 + i * 0.12, 0.6)
        x = M + i * colw
        n = round(v * ease_io((t - 4.0 - i * 0.12) / 1.0)) if t < 5.2 + i * 0.12 else v
        text(d, x, ys, f"{n}", 54, 700, INK, p)
        para(d, x, ys + 34, label, 19, 450, MUTED, colw - 26, 26, p)
    q = prog(t, 5.4, 0.7)
    runs = [("Read at the ", 24, 450, MUTED), ("rank", 24, 700, INK),
            (": the token the model would actually say.", 24, 450, MUTED)]
    rich(d, M, 942, runs, CW, 34, q)


def model_card(cv, t, start, y, name, size, ex, gif):
    d = cv.d
    p = prog(t, start, 0.6)
    dy = (1 - p) * 14
    h = 252
    rrect(d, M, y + dy, W - M, y + h + dy, 16, outline=HAIR, width=2, a=p)
    text(d, M + 32, y + 56 + dy, name, 30, 650, INK, p)
    text(d, M + 32 + measure(name, 30, 650) + 16, y + 56 + dy, size, 20, 500, MUTED, p)
    says_cf = ex["says_cf"]
    verdict = "ANSWERS LYON" if says_cf else "ANSWERS PARIS"
    vc = RET if says_cf else STO
    vp = prog(t, start + 1.6, 0.5)
    tw = measure(verdict, 15, 700, 2.2) + 36
    rrect(d, W - M - 32 - tw, y + 38 + dy, W - M - 32, y + 72 + dy, 17, fill=vc, a=vp)
    text(d, W - M - 32 - tw / 2, y + 55 + dy, verdict, 15, 700, WHITE, vp, anchor="mm",
         track=2.2, bg=col(vc, vp), fade=False)

    bx, maxw = M + 150, CW - 150 - 200
    for j, (word, pv, rank, c, top) in enumerate([
            ("Paris", ex["p_true"], ex["rank_true"], STO, ex["says_true"]),
            ("Lyon", ex["p_cf"], ex["rank_cf"], RET, ex["says_cf"])]):
        yy = y + 112 + j * 64 + dy
        g = ease_io((t - start - 0.5) / 1.0)
        text(d, M + 32, yy + 15, word, 25, 650, c, p, anchor="lm")
        rrect(d, bx, yy, bx + maxw, yy + 30, 7, fill=PANEL, a=p)
        rrect(d, bx, yy, bx + max(6, maxw * pv * g), yy + 30, 7,
              fill=c if top else col(c, 0.35), a=p)
        vs = "<0.01" if pv < 0.01 else f"{pv:.2f}"
        text(d, bx + maxw + 22, yy + 15, vs, 24, 700, INK if top else MUTED, p * clamp(g * 2),
             anchor="lm")
        text(d, bx + maxw + 22 + measure(vs, 24, 700) + 10, yy + 15, f"rank {rank}", 18,
             500, MUTED, p * clamp(g * 2), anchor="lm")


def s_two_models(cv, t, dur, gif):
    d = cv.d
    q = prog(t, 0.15, 0.6)
    text(d, M, 196 + (1 - q) * 12, "Same question. Same document.", 44, 650, INK, q)
    q2 = prog(t, 0.35, 0.6)
    text(d, M, 240, "France, first phrasing, Lyon document in context. Measured "
         "next-token probability.", 21, 450, MUTED, q2)
    model_card(cv, t, 0.8, 290, "Qwen2.5-1.5B", "1.5B parameters", D["ex_qwen"], gif)
    model_card(cv, t, 1.3, 580, "TinyLlama-1.1B", "1.1B parameters", D["ex_tiny"], gif)
    q = prog(t, 3.6, 0.7)
    runs = [("One ", 28, 500, INK), ("adopts the document", 28, 700, RET),
            (". The other ", 28, 500, INK), ("keeps its own answer", 28, 700, STO),
            (".", 28, 500, INK)]
    rich(d, M, 912, runs, CW, 40, q)


def s_overall(cv, t, dur, gif):
    d = cv.d
    q = prog(t, 0.15, 0.6)
    para(d, M, 210 + (1 - q) * 12, f"Across {D['n_conflict']} genuine conflicts, the "
         "document wins two times in three.", 44, 650, INK, CW, 58, q)
    g = ease_io((t - 0.8) / 1.3)
    for i, (v, label, c) in enumerate([(D["adopt"], "adopt the document", RET),
                                       (D["keep"], "keep their own answer", STO)]):
        p = prog(t, 0.7 + i * 0.15, 0.6)
        x = M + i * 470
        stat(d, x - 4, 470, v * g, 1, "%", 128, 700, c, p)
        text(d, x, 522, label, 27, 500, INK, p)
    y = 584
    p = prog(t, 0.9, 0.5)
    rrect(d, M, y, W - M, y + 50, 10, fill=PANEL, a=p)
    a_w = CW * D["adopt"] / 100 * g
    k_w = CW * D["keep"] / 100 * g
    rrect(d, M, y, M + a_w, y + 50, 10, fill=RET, a=p)
    if a_w > 10:
        rect(d, M + a_w - 10, y, M + a_w, y + 50, RET, a=p)
    rect(d, M + a_w, y, M + a_w + k_w, y + 50, STO, a=p)
    other = round(100 - D["adopt"] - D["keep"], 1)
    q = prog(t, 2.2, 0.6)
    text(d, W - M, 682, f"another token ranked first: {other}%", 20, 500, MUTED, q,
         anchor="rs")
    q = prog(t, 2.6, 0.7)
    para(d, M, 800, "A conflict counts only where the model ranked the true answer first "
         "with no document present, so the model demonstrably held the fact.",
         26, 400, MUTED, CW, 38, q)


def s_scale(cv, t, dur, gif):
    d = cv.d
    q = prog(t, 0.15, 0.6)
    text(d, M, 196 + (1 - q) * 12, "Holding a fact scales with size.", 42, 650, INK, q)
    q = prog(t, 0.45, 0.6)
    text(d, M, 250 + (1 - q) * 12, "Using the document does not.", 42, 650, UPD, q)

    q = prog(t, 0.8, 0.5)
    line(d, [(M, 306), (M + 36, 306)], STO, 4, q)
    circle(d, M + 18, 306, 6, fill=STO, a=q)
    text(d, M + 50, 306, "holds the fact, no document", 19, 500, MUTED, q, anchor="lm")
    lx = M + 50 + measure("holds the fact, no document", 19, 500) + 44
    dashed(d, [(lx, 306), (lx + 36, 306)], UPD, 4, 8, 5, q)
    circle(d, lx + 18, 306, 6, fill=WHITE, outline=UPD, width=3, a=q)
    text(d, lx + 50, 306, "adopts the document, conflicts only", 19, 500, MUTED, q,
         anchor="lm")

    # the plot sits 8 px higher and 20 px shorter than before, so the panel
    # under it clears the footer lockup
    x0, x1, y_top, y_bot = 152, W - M, 352, 662
    n = len(D["models"])
    cwid = (x1 - x0) / n
    X = [x0 + (i + 0.5) * cwid for i in range(n)]

    def Y(v):
        return y_bot - (y_bot - y_top) * v / 100

    tiny = next(i for i, m in enumerate(D["models"]) if m["name"] == "TinyLlama")
    hp = prog(t, 4.6, 0.7)
    rrect(d, X[tiny] - cwid / 2 + 6, y_top - 22, X[tiny] + cwid / 2 - 6, y_bot + 86, 12,
          fill=(0xFB, 0xEC, 0xEC), a=hp)

    ga = prog(t, 0.9, 0.6)
    for v in (0, 25, 50, 75, 100):
        line(d, [(x0, Y(v)), (x1, Y(v))], HAIR if v else (0xB8, 0xBB, 0xC4), 1.5 if v else 2, ga)
        text(d, x0 - 16, Y(v), f"{v}%", 17, 500, MUTED, ga, anchor="rm")
    for i, m in enumerate(D["models"]):
        text(d, X[i], y_bot + 34, m["name"], 19, 650, INK, ga, anchor="ms")
        text(d, X[i], y_bot + 60, m["size"], 17, 500, MUTED, ga, anchor="ms")
    text(d, (x0 + x1) / 2, y_bot + 100, "six open models, ordered by parameter count",
         17, 500, MUTED, ga, anchor="ms")

    knows = [m["knows"] for m in D["models"]]
    adopt = [m["adopt"] for m in D["models"]]
    kp = [(X[i], Y(v)) for i, v in enumerate(knows)]
    ap = [(X[i], Y(v)) for i, v in enumerate(adopt)]
    fk = ease_io((t - 1.2) / 1.4)
    fa = ease_io((t - 2.7) / 1.4)
    if fk > 0:
        line(d, partial(kp, fk), STO, 4.5)
    if fa > 0:
        dashed(d, partial(ap, fa), UPD, 4.5, 12, 7)
    for i in range(n):
        reach_k = clamp((fk * (n - 1) - i) * 3 + 1) if fk > 0 else 0
        reach_a = clamp((fa * (n - 1) - i) * 3 + 1) if fa > 0 else 0
        up_k = knows[i] >= adopt[i]
        big = 1 + 0.35 * hp if i == tiny else 1
        if reach_k > 0:
            circle(d, *kp[i], 7.5 * big, fill=STO, a=reach_k)
            text(d, kp[i][0], kp[i][1] + (-20 if up_k else 34), f"{knows[i]:.1f}", 18, 650,
                 STO, reach_k, anchor="ms")
        if reach_a > 0:
            circle(d, *ap[i], 7.5 * big, fill=WHITE, outline=UPD, width=3.5, a=reach_a)
            if i == n - 1:
                # the rising segment into the last point runs under a label set
                # beneath it, so this one label sits to the right of its marker
                text(d, ap[i][0] + 16, ap[i][1] + 7, f"{adopt[i]:.1f}", 18, 650, UPD,
                     reach_a, anchor="ls")
            else:
                text(d, ap[i][0], ap[i][1] + (34 if up_k else -20), f"{adopt[i]:.1f}", 18,
                     650, UPD, reach_a, anchor="ms")

    q = prog(t, 5.0, 0.7)
    runs = [(f"TinyLlama-1.1B holds the fact {D['tiny_knows']}% of the time, yet adopts "
             f"the document least of all six: ", 25, 400, INK),
            (f"{D['tiny_adopt']}%", 25, 700, UPD),
            (f". Qwen2.5-0.5B, less than half its size, adopts {D['qwen_small_adopt']}%.",
             25, 400, INK)]
    n_ = len(layout(runs, CW - 60))
    y = 796
    rrect(d, M, y, W - M, y + 40 + n_ * 36, 12, fill=PANEL, a=q)
    rect(d, M, y, M + 6, y + 40 + n_ * 36, UPD, a=q)
    rich(d, M + 34, y + 46, runs, CW - 60, 36, q, bg=col(PANEL, q))


def s_wording(cv, t, dur, gif):
    d = cv.d
    q = prog(t, 0.15, 0.6)
    text(d, M, 196 + (1 - q) * 12, "Change only the wording.", 44, 650, INK, q)
    q2 = prog(t, 0.35, 0.6)
    text(d, M, 244, "TinyLlama-1.1B, conflicts only: same facts, same document, three "
         "wordings.", 22, 450, MUTED, q2)
    qs = ["\u201cWhat is the capital of X?\u201d",
          "\u201cWhich city is the seat of government of X?\u201d",
          "\u201cName the capital of X.\u201d"]
    vals = D["tiny_by_phrasing"]
    bar_y = []
    for i, (s, v) in enumerate(zip(qs, vals)):
        p = prog(t, 0.8 + i * 0.25, 0.6)
        y = 356 + i * 132
        text(d, M, y, s, 25, 500, INK, p)
        g = ease_io((t - 1.2 - i * 0.25) / 1.1)
        rrect(d, M, y + 22, W - M, y + 66, 10, fill=PANEL, a=p)
        rrect(d, M, y + 22, M + max(20, CW * v / 100 * g), y + 66, 10, fill=FOR, a=p)
        text(d, M + CW * v / 100 * g + 16, y + 44, f"{v * g:.1f}%", 26, 700, INK,
             p * clamp(g * 3), anchor="lm", bg=col(PANEL, p))
        bar_y.append(y + 66)
    lo, hi = min(vals), max(vals)
    q = prog(t, 3.0, 0.6)
    yb = bar_y[-1] + 42
    xa, xb = M + CW * lo / 100, M + CW * hi / 100
    line(d, [(xa, yb), (xa + (xb - xa) * q, yb)], UPD, 3, q)
    line(d, [(xa, yb - 12), (xa, yb + 12)], UPD, 3, q)
    line(d, [(xb, yb - 12), (xb, yb + 12)], UPD, 3, q if q > 0.95 else 0)
    text(d, (xa + xb) / 2, yb + 44, f"{D['max_adopt_spread']} points apart, on "
         "conflicts alone", 25, 700, UPD, q, anchor="ms")
    q = prog(t, 3.9, 0.7)
    para(d, M, 878, f"Over all probes, wording moves the rate by a median of "
         f"{D['median_spread']} points. A single-phrasing result reports the sentence "
         "as much as the model.", 24, 400, MUTED, CW, 35, q)


def s_rank(cv, t, dur, gif):
    d = cv.d
    q = prog(t, 0.15, 0.6)
    text(d, M, 196 + (1 - q) * 12, "Probability is not an answer.", 46, 650, INK, q)
    q2 = prog(t, 0.35, 0.6)
    text(d, M, 242, "Under greedy decoding, a model says the token it ranks first.", 23,
         450, MUTED, q2)

    ex = D["ex_rank"]
    y0 = 290
    p = prog(t, 0.8, 0.6)
    rrect(d, M, y0, W - M, y0 + 268, 16, outline=HAIR, width=2, a=p)
    text(d, M + 32, y0 + 48, "TinyLlama-1.1B  \u00b7  \u201cName the capital of France.\u201d"
         "  \u00b7  Lyon document in context", 19, 500, MUTED, p)
    bx, maxw = M + 150, CW - 150 - 200
    g = ease_io((t - 1.2) / 1.0)
    for j, (word, pv, rank, c, top) in enumerate([
            ("Paris", ex["p_true"], ex["rank_true"], STO, ex["says_true"]),
            ("Lyon", ex["p_cf"], ex["rank_cf"], RET, ex["says_cf"])]):
        yy = y0 + 86 + j * 62
        text(d, M + 32, yy + 15, word, 25, 650, c, p, anchor="lm")
        rrect(d, bx, yy, bx + maxw, yy + 30, 7, fill=PANEL, a=p)
        rrect(d, bx, yy, bx + max(6, maxw * pv * g), yy + 30, 7,
              fill=c if top else col(c, 0.4), a=p)
        text(d, bx + maxw + 22, yy + 15, f"{pv:.2f}", 24, 700, INK if top else MUTED,
             p * clamp(g * 2), anchor="lm")
        text(d, bx + maxw + 22 + measure(f"{pv:.2f}", 24, 700) + 10, yy + 15,
             f"rank {rank}", 18, 500, MUTED, p * clamp(g * 2), anchor="lm")
    q = prog(t, 2.3, 0.6)
    runs = [(f"Lyon carries {ex['p_cf']:.2f} of the probability. ", 24, 400, INK),
            ("The model still says Paris.", 24, 700, INK)]
    rich(d, M + 32, y0 + 236, runs, CW - 64, 34, q)

    ya = 618
    q = prog(t, 3.8, 0.6)
    text(d, M, ya, f"All {D['n_doc']} conditions with the document present", 20, 500,
         MUTED, q)
    by = ya + 22
    rrect(d, M, by, W - M, by + 46, 10, fill=PANEL, a=q)
    g1 = ease_io((t - 4.1) / 0.9)
    w332 = CW * D["over10"] / D["n_doc"]
    w69 = CW * D["over10_not"] / D["n_doc"]
    rrect(d, M, by, M + w332 * g1, by + 46, 10, fill=col(RET, 0.38), a=q)
    g2 = ease_io((t - 5.0) / 0.6)
    if g2 > 0:
        rect(d, M + w332 - w69 * g2, by, M + w332, by + 46, UPD, a=q)
    q3 = prog(t, 4.6, 0.6)
    text(d, M, by + 84, f"{D['over10']}", 34, 700, INK, q3)
    text(d, M + measure(f"{D['over10']}", 34, 700) + 12, by + 84,
         "where the document's answer carries more than 0.10", 20, 500, MUTED, q3)
    q4 = prog(t, 5.3, 0.6)
    text(d, W - M, by + 84, f"{D['over10_not']} of them still not said", 20, 600, UPD, q4,
         anchor="rs")
    q5 = prog(t, 5.9, 0.7)
    runs = [("In ", 26, 500, INK), (f"{D['over10_not_pct']}%", 30, 700, UPD),
            (" of them, probability and answer disagree. Recording the rank costs one "
             "comparison.", 26, 500, INK)]
    rich(d, M, 862, runs, CW, 40, q5)


def s_correction(cv, t, dur, gif):
    d = cv.d
    q = prog(t, 0.2, 0.7)
    text(d, M - 2, 330 + (1 - q) * 16, "That gap is not hypothetical.", 58, 650, INK, q)
    q = prog(t, 1.0, 0.7)
    para(d, M, 430, "An earlier version of this work read probability alone, and reached a "
         "published conclusion that the measurement did not support.", 32, 400, INK, CW,
         46, q)
    q = prog(t, 2.4, 0.7)
    text(d, M, 640, "This paper corrects it, on the record.", 34, 650, STO, q)
    q = prog(t, 3.2, 0.7)
    para(d, M, 760, "Every number in it comes from one deterministic script that runs on a "
         "laptop CPU.", 25, 400, MUTED, CW, 36, q)


def s_end(cv, t, dur, gif):
    # the same title block and authors row as the title card, at the same
    # coordinates, so the film closes on the frame it opened with; the kicker
    # lives in the header, and the portraits arrive from the lockup (chrome)
    d = cv.d
    for i, ln in enumerate(TITLE_LINES):
        q = prog(t, 0.15 + i * 0.12, 0.7)
        text(d, M - 3, 268 + i * 84 + (1 - q) * 16, ln, 70, 650, INK, q)
    pills(cv, t, 0.45, 398, 58, 14, 23)

    links = [("PAPER", "vixra.org/abs/2611.0038"),
             ("CODE AND DATA", "github.com/Amey-Thakur/LLM-KNOWLEDGE-LIFECYCLE"),
             ("LIVE DEMO", "huggingface.co/spaces/ameythakur/llm-knowledge-lifecycle")]
    for i, (label, url) in enumerate(links):
        q = prog(t, 1.45 + i * 0.15, 0.6)
        y = 536 + i * 58
        line(d, [(M, y - 40), (W - M, y - 40)], HAIR, 1, q)
        text(d, M, y, label, 15, 700, MUTED, q, track=2.2)
        text(d, M + 200, y, url, 22, 500, INK, q)

    authors_row(d, t, 1.20)            # the faces land at local 1.30

    q = prog(t, 2.05, 0.6)
    line(d, [(M, 880), (W - M, 880)], HAIR, 1, q)
    text(d, M, 920, "All code, probes and measurements are public.", 27, 500, INK, q)
    text(d, M, 954, f"{VIXRA}  \u00b7  Artificial Intelligence  \u00b7  CC BY 4.0", 19, 500,
         MUTED, q)


# --- reading pace ------------------------------------------------------------
# The scenes are authored on a fast clock. Two things turn that into a pace a
# viewer can read in a muted feed. All motion runs SLOW times slower than it is
# authored, so fades and count-ups are calm rather than brisk. And at each beat
# below the scene clock stops for the given seconds, so a block of text stays
# alone on screen long enough to be read before the next one arrives.
#
# A beat sits only where no animation is in flight: every authored fade,
# count-up and line-draw has finished before it and the next starts at or after
# it, so a pause never freezes anything half-drawn. The intervals were read off
# the scene functions; change a scene's timing and its beats must be checked.
# SETTLE is the authored second at which the scene's last element is complete,
# and TAIL is the reading time left after it.
SLOW = 1.25
# The title and end cards run on the real clock (UNSLOWED): their choreography
# is timed against the portrait flights, which are real-time chrome events.
UNSLOWED = ("title", "end")
BEATS = {
    "question": [(1.2, 0.8), (2.3, 0.7), (3.3, 0.5)],
    "framework": [(2.7, 2.6), (3.4, 0.6)],
    "experiment": [(None, 2.0), (4.0, 1.8), (5.4, 0.6)],   # None: typing ends
    "two-models": [(3.6, 1.6)],
    "overall": [(0.7, 1.4), (2.2, 1.2)],
    "scale": [(2.7, 0.9), (4.6, 1.4)],
    "wording": [(3.0, 1.2), (3.9, 1.2)],
    "rank": [(2.3, 0.6), (3.8, 2.0), (5.9, 1.2)],
    "correction": [(1.0, 0.6), (2.4, 3.2), (3.2, 0.8)],
}
SETTLE = {"title": 1.9, "question": 4.0, "framework": 4.1, "experiment": 6.1,
          "two-models": 4.3, "overall": 3.3, "scale": 5.7, "wording": 4.6,
          "rank": 6.6, "correction": 3.9, "end": 2.65}
# Each closing block is legible, at about 4 words a second, before the next
# scene's sweep begins: measured per scene, not estimated.
TAIL = {"title": 4.1, "question": 4.0, "framework": 5.5, "experiment": 3.0,
        "two-models": 2.8, "overall": 5.75, "scale": 6.25, "wording": 6.0,
        "rank": 3.6, "correction": 3.2, "end": 5.35}


def beats(key):
    out = []
    for at, hold in BEATS.get(key, ()):
        if at is None:          # the experiment's typing, measured not guessed
            at = TYPE_START + rich_count(CTX_RUNS) / TYPE_CPS + 1e-3
        out.append((at, hold))
    return out


def scene_clock(key, t):
    """The authored second shown at paced local second t (t may be negative)."""
    if key in UNSLOWED:
        return t
    o = n = 0.0
    for at, hold in beats(key):
        span = (at - o) * SLOW
        if t < n + span:
            return o + (t - n) / SLOW
        n, o = n + span, at
        if t < n + hold:
            return at
        n += hold
    return o + (t - n) / SLOW


def paced(key, authored):
    """The paced local second at which the authored second is reached."""
    if key in UNSLOWED:
        return authored
    o = n = 0.0
    for at, hold in beats(key):
        if authored <= at:
            break
        n, o = n + (at - o) * SLOW + hold, at
    return n + (authored - o) * SLOW


def scene_len(key):
    return round(paced(key, SETTLE[key]) + TAIL[key], 2)


# key, seconds, accent, header label, draw
SCENES = [(key, scene_len(key), accent, label, fn) for key, accent, label, fn in [
    ("title", NAVY, "", s_title),
    ("question", ACQ, "The question", s_question),
    ("framework", STO, "The framework", s_framework),
    ("experiment", RET, "The experiment", s_experiment),
    ("two-models", UPD, "One document, two answers", s_two_models),
    ("overall", RET, "Who wins", s_overall),
    ("scale", UPD, "The finding", s_scale),
    ("wording", FOR, "The wording", s_wording),
    ("rank", RET, "Probability is not an answer", s_rank),
    ("correction", STO, "A correction", s_correction),
    ("end", NAVY, "", s_end),
]]
CONTENT = [k for k, *_ in SCENES if k not in ("title", "end")]
KEYS = [s[0] for s in SCENES]

# Layout shift of each scene's content, output pixels. Content scenes rise 20
# to sit between the header at 64 and the lockup at 998.
SCENE_OFF = {"title": 0, "end": 0}
CONTENT_OFF = -20

# --- the boundary: a soft top-to-bottom dissolve ------------------------------
PRE, POST = 0.6, 0.9          # the sweep runs from B - PRE to B + POST
SWEEP = PRE + POST            # 1.5 s
FEATHER = 520                 # height of the soft edge, output px
SW_Y0, SW_Y1 = 100, 960       # the front starts at SW_Y0 and clears SW_Y1
DRIFT = 16                    # the outgoing scene rises this far while it leaves

# --- the shared element: the portraits travel between card and lockup ---------
TRAVEL = 1.4                  # seconds
T_IN = 0.15                   # title -> lockup starts at B1 + T_IN
T_OUT = -0.10                 # lockup -> end card starts at Be + T_OUT
FT_IN = (1.25, 0.60)          # lockup text fades in at B1 + 1.25, over 0.60 s
FT_OUT = (-0.65, 0.45)        # and out at Be - 0.65, over 0.45 s, before the faces go


def starts() -> list[float]:
    out, acc = [], 0.0
    for s in SCENES:
        out.append(acc)
        acc += s[1]
    return out


def total_seconds() -> float:
    total = round(sum(s[1] for s in SCENES), 2)
    assert total <= MAX_SECONDS, f"the film runs {total:.2f} s, over the {MAX_SECONDS} s cap"
    return total


def render_scene(cv_or_none, idx, t, gif=False, drift=0.0):
    """Draw scene idx at paced local second t onto a canvas (content only). The
    scene function receives the scene clock, the authored second that t shows."""
    global OFFY, FADE, TREAL
    key, dur, _, _, fn = SCENES[idx]
    cv = cv_or_none or Canvas()
    OFFY = SCENE_OFF.get(key, CONTENT_OFF) + drift
    FADE = 1.0
    TREAL = t
    fn(cv, scene_clock(key, t), dur, gif)
    OFFY = 0.0
    return cv


@lru_cache(maxsize=128)
def _mask_column(step: int, y0: float) -> Image.Image:
    u = step / 1000
    f = y0 + u * (SW_Y1 + FEATHER - y0)
    vals = [int(round(255 * smooth((f - (y + 0.5) / S) / FEATHER))) for y in range(P(H))]
    c = Image.new("L", (1, P(H)))
    c.putdata(vals)
    return c


def sweep_mask(u: float, y0: float = SW_Y0) -> Image.Image:
    return _mask_column(int(round(clamp(u) * 1000)), y0).resize((P(W), P(H)), Image.NEAREST)


def sweep(a_idx, a_t, b_idx, b_t, u, gif=False, drift=DRIFT, y0=SW_Y0) -> Canvas:
    """Scene a leaving, scene b arriving: b is revealed under a front that
    moves down at constant speed with a soft edge, while a drifts up."""
    if u <= 0:
        return render_scene(None, a_idx, a_t, gif)
    if u >= 1:
        return render_scene(None, b_idx, b_t, gif)
    A = render_scene(None, a_idx, a_t, gif, drift=-drift * ease_io(u))
    B = render_scene(None, b_idx, b_t, gif)
    A.im = Image.composite(B.im, A.im, sweep_mask(u, y0))
    A.d = ImageDraw.Draw(A.im)
    return A


def locate(T: float):
    """(prev, cur, r): the boundary nearest T, or (cur, cur, None)."""
    st = starts()
    i = max(j for j in range(len(SCENES)) if st[j] <= T or j == 0)
    if i + 1 < len(SCENES) and T >= st[i + 1] - PRE:
        return i, i + 1, T - st[i + 1]
    if i >= 1 and T < st[i] + POST:
        return i - 1, i, T - st[i]
    return i, i, None


def poses_card():
    return [(img, cx, cy, dia) for (img, _, _), (cx, cy, dia) in zip(AUTHORS, CARD_POSES)]


def poses_foot():
    pa, pb, _ = lockup_geometry()
    return [(AUTHORS[0][0], pa[0], pa[1], FOOT_D), (AUTHORS[1][0], pb[0], pb[1], FOOT_D)]


def blend_poses(p0, p1, k):
    """Travel from p0 to p1. y leads and x follows, so on the way in the faces
    drop, then gather; on the way out they rise, then spread. They never cross."""
    ky = ease_io(clamp(k * 1.18))
    kx = ease_io(clamp(k * 1.18 - 0.18))
    kd = ease_io(k)
    return [(n, lerp(x0, x1, kx), lerp(y0, y1, ky), lerp(d0, d1, kd))
            for (n, x0, y0, d0), (_, x1, y1, d1) in zip(p0, p1)]


def lockup_alpha(T: float) -> float:
    st = starts()
    b1, be = st[1], st[-1]
    a_in = ease_out((T - (b1 + FT_IN[0])) / FT_IN[1])
    a_out = 1 - ease_io((T - (be + FT_OUT[0])) / FT_OUT[1])
    return min(a_in, a_out)


def chrome(cv, T: float):
    """Everything that is not scene content: header, portraits, lockup text."""
    global OFFY, FADE
    OFFY, FADE = 0.0, 1.0
    st = starts()
    b1, be = st[1], st[-1]
    header(cv, *locate(T))

    # portraits: title card -> lockup -> end card, never leaving the frame
    if T < b1 + T_IN:
        poses = poses_card()
    elif T < b1 + T_IN + TRAVEL:
        poses = blend_poses(poses_card(), poses_foot(), (T - b1 - T_IN) / TRAVEL)
    elif T < be + T_OUT:
        poses = poses_foot()
    elif T < be + T_OUT + TRAVEL:
        poses = blend_poses(poses_foot(), poses_card(), (T - be - T_OUT) / TRAVEL)
    else:
        poses = poses_card()
    draw_poses(cv, poses)

    if T < be:
        footer_text(cv, lockup_alpha(T))


def frame_at(T: float) -> Image.Image:
    """The video frame at global time T."""
    st = starts()
    prev, cur, r = locate(T)
    if r is None:
        cv = render_scene(None, cur, T - st[cur])
    else:
        u = (r + PRE) / SWEEP
        cv = sweep(prev, T - st[prev], cur, T - st[cur], u)
    chrome(cv, T)
    return cv.out()


def cover_time() -> float:
    """Inside the title card's hold: settled, and before the first sweep."""
    t = starts()[1] - PRE - 0.1
    assert SETTLE["title"] <= t
    return t


def tail_time(key: str) -> float:
    """A frame of a scene inside its reading tail and outside any window."""
    idx = KEYS.index(key)
    dur = SCENES[idx][1]
    local = dur - PRE - 0.5
    assert paced(key, SETTLE[key]) <= local < dur - PRE and local >= POST
    return starts()[idx] + local


# --- GIFs: one scene, the same chrome, the same boundary at the loop seam ------
GIF_LEAD = 1.0                # the loop opens on the finished state for this long

# In a loop the scene hands over to itself. Its heading is the same before and
# after, so it stays: rows above the hold line (scene coordinates, before
# CONTENT_OFF) are pasted from the settled render while the scene restarts.
GIF_HOLD_Y = {"framework": 230, "two-models": 266, "scale": 284, "rank": 268,
              "wording": 270}


def gif_head_done(key: str) -> float:
    """The paced local second by which everything above the hold line is
    complete: authored 1.2 s, on the scene's own reading pace."""
    return paced(key, 1.2)


def gif_timeline(key):
    """Loop of one scene: hold finished, sweep into its own beginning, rebuild,
    hold. The first and last frames are the same finished state. All times are
    paced local seconds."""
    dur = SCENES[KEYS.index(key)][1]
    settle = dur - PRE
    build = settle - POST
    total = GIF_LEAD + SWEEP + build + (GIF_HOLD - GIF_LEAD)
    return settle, total


@lru_cache(maxsize=8)
def _settled(key: str) -> Image.Image:
    settle, _ = gif_timeline(key)
    return render_scene(None, KEYS.index(key), settle, gif=True).im


def gif_frame(key: str, tau: float) -> Image.Image:
    global OFFY, FADE
    idx = KEYS.index(key)
    settle, _ = gif_timeline(key)
    hold_px = GIF_HOLD_Y[key] + CONTENT_OFF
    tb = tau - GIF_LEAD - PRE                 # the restarted scene's paced clock
    if tau < GIF_LEAD:
        cv = Canvas(_settled(key))
    elif tau < GIF_LEAD + SWEEP:
        u = (tau - GIF_LEAD) / SWEEP
        B = render_scene(None, idx, tb, gif=True)
        cv = Canvas(Image.composite(B.im, _settled(key), sweep_mask(u, hold_px)))
    else:
        cv = render_scene(None, idx, min(settle, tb), gif=True)
    if tau >= GIF_LEAD and tb < gif_head_done(key):
        cv.im.paste(_settled(key).crop((0, 0, P(W), P(hold_px))), (0, 0))
        cv.d = ImageDraw.Draw(cv.im)
    OFFY, FADE = 0.0, 1.0
    header(cv, idx, idx, None)
    lockup_still(cv)
    return cv.out(GIF_SIZE)


# =============================================================================
# Still figures, in the same frame system
# =============================================================================
def arrow_down(d, x, y0, y1, color, w=3):
    line(d, [(x, y0), (x, y1 - 10)], color, w)
    poly(d, [(x - 8, y1 - 12), (x + 8, y1 - 12), (x, y1)], color)


def fig_pipeline() -> Image.Image:
    """How the measurement is built: the five stages of the design, what they
    multiply to, the filter, and where the conflicts end. Every number is read
    from D, and every element matches the alt text posted with the image."""
    global OFFY, FADE
    OFFY, FADE = 0.0, 1.0
    cv = Canvas()
    d = cv.d
    header_still(cv, eyebrow_parts(KEYS.index("title")))

    text(d, M, 176, "How the measurement is built.", 46, 650, INK)

    # five stages, left to right; the first four multiply, the fifth is recorded
    y0, h, gap, pad = 216, 274, 18, 13
    w = (CW - 4 * gap) / 5
    lo, hi = D["models"][0]["size"], D["models"][-1]["size"]
    stages = [(D["n_probes"], "counterfactual probes"),
              (D["n_phrasings"], "phrasings of each"),
              (D["n_conditions"], "conditions, with and without the contradicting document"),
              (D["n_models"], f"open models, from {lo} to {hi} parameters")]
    for i, (v, label) in enumerate(stages):
        x = M + i * (w + gap)
        rrect(d, x, y0, x + w, y0 + h, 14, fill=PANEL)
        text(d, x + pad, y0 + 74, f"{v}", 60, 700, NAVY)
        lines = wrap(label, 20, 500, w - 2 * pad)
        assert all(measure(s, 20, 500) <= w - 2 * pad for s in lines), label
        for j, s in enumerate(lines):
            text(d, x + pad, y0 + 116 + j * 27, s, 20, 500, INK, bg=PANEL)
    x = M + 4 * (w + gap)
    rrect(d, x, y0, x + w, y0 + h, 14, fill=WHITE, outline=NAVY, width=2)
    head = wrap("Recorded on every pass", 21, 650, w - 2 * pad)
    for j, s in enumerate(head):
        text(d, x + pad, y0 + 44 + j * 27, s, 21, 650, NAVY)
    body = wrap("the probability and rank of both answers, and the top-ranked token",
                20, 500, w - 2 * pad)
    assert all(measure(s, 20, 500) <= w - 2 * pad for s in body)
    yb = y0 + 44 + len(head) * 27 + 14
    for j, s in enumerate(body):
        text(d, x + pad, yb + j * 27, s, 20, 500, INK)
    # the joins sit on the seams, level with the numbers
    cy = y0 + 52
    for i in range(4):
        cx = M + (i + 1) * (w + gap) - gap / 2
        circle(d, cx, cy, 14, fill=WHITE, outline=HAIR, width=2)
        if i < 3:
            text(d, cx, cy + 1, "\u00d7", 24, 600, NAVY, anchor="mm")
        else:                                   # then, on every one of those passes
            line(d, [(cx - 3, cy - 7), (cx + 4, cy), (cx - 3, cy + 7)], NAVY, 3)

    # they multiply to the number of forward passes
    spine = M + 34
    arrow_down(d, spine, y0 + h + 6, y0 + h + 36, (0xB8, 0xBB, 0xC4))
    by = y0 + h + 42
    rrect(d, M, by, W - M, by + 84, 14, fill=PANEL)
    rect(d, M, by, M + 6, by + 84, NAVY)
    eq = (f"{D['n_probes']} \u00d7 {D['n_phrasings']} \u00d7 {D['n_conditions']} "
          f"\u00d7 {D['n_models']} = ")
    xx = M + 34
    xx += text(d, xx, by + 58, eq, 30, 600, MUTED, bg=PANEL)
    xx += text(d, xx, by + 58, f"{D['n_rows']}", 52, 700, NAVY, bg=PANEL) + 14
    text(d, xx, by + 58, "forward passes", 28, 500, INK, bg=PANEL)

    # a filter keeps the genuine conflicts
    arrow_down(d, spine, by + 90, by + 120, (0xB8, 0xBB, 0xC4))
    fy = by + 128
    fx = M + 8
    poly(d, [(fx, fy + 4), (fx + 54, fy + 4), (fx + 34, fy + 32), (fx + 34, fy + 62),
             (fx + 20, fy + 54), (fx + 20, fy + 32)], NAVY)
    nx = M + 86
    nw = text(d, nx, fy + 58, f"{D['n_conflict']}", 60, 700, NAVY)
    tx = nx + nw + 22
    for j, s in enumerate(wrap("cases kept, where the model ranked the true answer "
                               "first with no document", 25, 500, W - M - tx)):
        text(d, tx, fy + 24 + j * 34, s, 25, 500, INK)

    # and where they end
    arrow_down(d, spine, fy + 82, fy + 116, (0xB8, 0xBB, 0xC4))
    ry = fy + 210
    for i, (v, label, c) in enumerate([(D["adopt"], "adopt the document", RET),
                                       (D["keep"], "keep their own answer", STO)]):
        x = M + i * 470
        text(d, x - 4, ry, f"{v}%", 96, 700, c)
        text(d, x, ry + 40, label, 26, 500, INK)

    lockup_still(cv)
    return cv.out()


def fig_social() -> Image.Image:
    """The repository's social card, 1280 x 640: the paper's title, the five
    stages, both authors and the viXra id, all inside a 40 px safe margin."""
    global OFFY, FADE
    OFFY, FADE = 0.0, 1.0
    sw, sh, sm = 1280, 640, 80
    cv = Canvas(grid_base(sw, sh, 12))
    d = cv.d
    hy = 92
    rrect(d, sm, hy - 8, sm + 16, hy + 8, 4, fill=NAVY)
    text(d, sm + 32, hy, VIXRA, 28, 500, MUTED, anchor="lm", track=0.6)
    for i, ln in enumerate(TITLE_LINES):
        text(d, sm - 4, 200 + i * 82, ln, 70, 650, INK)
    pills(cv, 9.0, 0.0, 320, 64, 16, 26, x0=sm, width=sw - 2 * sm)
    dia, cy = 108, 488
    for i, ((img, name, _), bx) in enumerate(zip(AUTHORS, (sm, sm + 520))):
        portrait(cv, img, bx + dia / 2, cy, dia)
        text(d, bx + dia + 26, cy, name, 28, 650, INK, anchor="lm", track=2.2)
    return cv.out(sw)


# =============================================================================
# Outputs
# =============================================================================
def _init():
    global D
    D = load_data()


def _render_frame(args):
    i, outdir = args
    frame_at(i / FPS).save(Path(outdir) / f"{i:05d}.png", compress_level=1)
    return i


def cmd_frames(outdir: str):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    n = int(round(total_seconds() * FPS))
    with Pool(initializer=_init) as pool:
        for k, _ in enumerate(pool.imap_unordered(_render_frame,
                                                  [(i, outdir) for i in range(n)],
                                                  chunksize=8)):
            if k % 300 == 0:
                print(f"  {k}/{n} frames", flush=True)
    print(f"  {n} frames, {total_seconds():.2f} s at {FPS} fps -> {outdir}")


GIFS = [("framework", "lifecycle-framework.gif"),
        ("two-models", "one-document-two-answers.gif"),
        ("scale", "scale-is-not-deference.gif"),
        ("wording", "wording-moves-the-answer.gif"),
        ("rank", "probability-is-not-an-answer.gif")]


def _gif_frames(key):
    """One loop: GIF_LEAD s on the finished state, the same soft sweep the
    video uses at a boundary (here into the scene's own beginning, under a held
    heading), the rebuild, then the rest of the hold. Held frames are identical
    and Pillow merges them, so the hold costs nothing in file size."""
    idx = KEYS.index(key)
    box = (0, 0, P(W), P(GIF_HOLD_Y[key] + CONTENT_OFF))
    head = render_scene(None, idx, gif_head_done(key), gif=True).im.crop(box)
    if head.tobytes() != _settled(key).crop(box).tobytes():
        raise RuntimeError(f"{key}: content above the GIF hold line is still moving at "
                           f"paced {gif_head_done(key):.2f} s; the held heading would jump")
    _, total = gif_timeline(key)
    n = int(round(total * GIF_FPS))
    return [gif_frame(key, i / GIF_FPS) for i in range(n)]


# One palette per GIF. EXACT colours sit at fixed indices and are restored
# pixel for pixel after quantising, so the ground is pure white and the band
# and stage colours are exact.
EXACT = [WHITE, GRID, HAIR, PANEL, INK, MUTED, NAVY, ACQ, STO, RET, UPD, FOR,
         DONE, TODO, (0xFB, 0xEC, 0xEC), (0xB8, 0xBB, 0xC4),
         col(STO, 0.35), col(RET, 0.35), col(RET, 0.38), col(RET, 0.40)]
# A colour dissolving over white passes through these tints, so a fading bar
# steps through its own hue, never through a borrowed one.
RAMPS = ([col(c, a) for c in (ACQ, STO, RET, UPD, FOR)
          for a in (0.15, 0.30, 0.45, 0.60, 0.75, 0.90)]
         + [col(c, a) for c in (INK, MUTED, NAVY) for a in (0.25, 0.50, 0.75)])
PHOTO_COLOURS = 48
NEAR = 8              # entries within this many levels of an EXACT colour are dropped


def _portrait_boxes(size: int):
    pa, pb, _ = lockup_geometry()
    k = size / W
    r = (FOOT_D / 2 + RING + HAIRW + 1) * k
    return [(int(cx * k - r), int(cy * k - r), int(math.ceil(cx * k + r)),
             int(math.ceil(cy * k + r))) for cx, cy in (pa, pb)]


def gif_palette(frames):
    """EXACT at indices 0-19, then the RAMPS, then PHOTO_COLOURS for the two
    portraits from their pixels alone, then median cut over the frames with the
    portraits blanked, so the photos never take entries from the fills."""
    size = frames[0].width
    keep = list(EXACT)

    def add(cands):
        for c in cands:
            c = tuple(c)
            if c in keep or any(max(abs(a - b) for a, b in zip(c, e)) <= NEAR
                                for e in EXACT):
                continue
            keep.append(c)

    add(RAMPS)
    boxes = _portrait_boxes(size)
    photos = Image.new("RGB", (sum(b[2] - b[0] for b in boxes),
                               max(b[3] - b[1] for b in boxes)), WHITE)
    x = 0
    for b in boxes:
        photos.paste(frames[0].crop(b), (x, 0))
        x += b[2] - b[0]
    pp = photos.quantize(colors=PHOTO_COLOURS, method=Image.Quantize.MEDIANCUT).getpalette()
    add(pp[i:i + 3] for i in range(0, min(len(pp), 3 * PHOTO_COLOURS), 3))

    n_flat = 256 - len(keep)
    h0 = int(round(GIF_LEAD * GIF_FPS))
    h1 = int(round((GIF_LEAD + SWEEP) * GIF_FPS))
    sample = frames[::16] + frames[h0:h1:2]
    sheet = Image.new("RGB", (size, size * len(sample)))
    for i, f in enumerate(sample):
        g = f.copy()
        for b in boxes:
            g.paste(WHITE, b)
        sheet.paste(g, (0, i * size))
    fp = sheet.quantize(colors=n_flat, method=Image.Quantize.MEDIANCUT).getpalette()
    add(fp[i:i + 3] for i in range(0, min(len(fp), 3 * n_flat), 3))

    keep = keep[:256] + [WHITE] * max(0, 256 - len(keep))
    pal = Image.new("P", (1, 1))
    pal.putpalette([v for c in keep for v in c])
    return pal


def gif_quantize(f, pal):
    """Quantise with no dithering, then put every pixel that is exactly an
    EXACT colour back on its index. Pillow's palette lookup goes through a
    cache whose slots cover four levels per channel, so pure white would
    otherwise come out as a near-white neighbour and the ground would read grey."""
    q = f.quantize(palette=pal, dither=Image.Dither.NONE)
    ch = f.split()
    for idx, c in enumerate(EXACT):
        m = None
        for band, v in zip(ch, c):
            bm = band.point(lambda x, v=v: 255 if x == v else 0)
            m = bm if m is None else ImageChops.multiply(m, bm)
        q.paste(idx, mask=m)
    return q


def _gif_job(job):
    key, name = job
    frames = _gif_frames(key)
    pal = gif_palette(frames)
    q = [gif_quantize(f, pal) for f in frames]
    out = MEDIA / name
    hold = [int(1000 / GIF_FPS)] * len(q)
    q[0].save(out, save_all=True, append_images=q[1:], duration=hold, loop=0,
              optimize=False, disposal=1)
    return f"  {name}: {len(q)} frames, {out.stat().st_size / 1e6:.2f} MB"


def cmd_gifs(only=None):
    known = [k for k, _ in GIFS]
    unknown = [k for k in (only or []) if k not in known]
    if unknown:
        raise SystemExit(f"no GIF for {unknown}; choose from {known}")
    MEDIA.mkdir(parents=True, exist_ok=True)
    jobs = [g for g in GIFS if not only or g[0] in only]
    with Pool(initializer=_init) as pool:
        for msg in pool.imap(_gif_job, jobs):
            print(msg, flush=True)


def cmd_cover():
    MEDIA.mkdir(parents=True, exist_ok=True)
    frame_at(cover_time()).save(MEDIA / "cover.png", optimize=True)
    print(f"  cover.png written (t = {cover_time():.2f} s)")


def cmd_figures():
    """Four stills: the pipeline, the repository card, and two settled scene
    frames for the newsletter. The card is not named social-preview.png,
    because .gitattributes sends that name to Git LFS."""
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig_pipeline().save(MEDIA / "research-pipeline.png", optimize=True)
    fig_social().save(MEDIA / "social-preview-1280x640.png", optimize=True)
    for key, name in [("scale", "scale-is-not-deference.png"),
                      ("rank", "probability-is-not-an-answer.png")]:
        frame_at(tail_time(key)).save(MEDIA / name, optimize=True)
    print("  research-pipeline.png, social-preview-1280x640.png, "
          "scale-is-not-deference.png, probability-is-not-an-answer.png written")


def cmd_stills(outdir: str, times: list[str]):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    for T in times:
        frame_at(float(T)).save(Path(outdir) / f"t{float(T):06.2f}.png")
    print(f"  {len(times)} stills -> {outdir}")


def cmd_gifstills(outdir: str, key: str, frames: list[str]):
    """GIF frames by index; negative indices count from the end of the loop."""
    Path(outdir).mkdir(parents=True, exist_ok=True)
    _, total = gif_timeline(key)
    n = int(round(total * GIF_FPS))
    for s in frames:
        i = int(s) % n
        gif_frame(key, i / GIF_FPS).save(Path(outdir) / f"{key}-{i:04d}.png")
    print(f"  {len(frames)} gif stills of {n} -> {outdir}")


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    _init()
    print(f"  data verified against preprint/numbers.tex; runtime {total_seconds():.2f} s")
    cmd = argv[0]
    if cmd == "frames":
        cmd_frames(argv[1])
    elif cmd == "gifs":
        cmd_gifs(argv[1:])
    elif cmd == "cover":
        cmd_cover()
    elif cmd == "figures":
        cmd_figures()
    elif cmd == "stills":
        cmd_stills(argv[1], argv[2:])
    elif cmd == "gifstills":
        cmd_gifstills(argv[1], argv[2], argv[3:])
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
