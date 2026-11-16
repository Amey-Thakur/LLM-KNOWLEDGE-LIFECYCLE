#!/usr/bin/env python3
"""
===============================================================================
FILE         : render.py
PROJECT      : The Knowledge Lifecycle of Large Language Models
PURPOSE      : Render the explainer video, its cover, and the GIFs, all drawn
               from the measurement rather than typed.
TECH STACK   : Python 3, Pillow (frames); ffmpeg (encoding, in CI)
AUTHORS      : Amey Thakur (https://github.com/Amey-Thakur)
               Sarvesh Talele (https://github.com/sarveshtalele)
REPOSITORY   : https://github.com/Amey-Thakur/LLM-KNOWLEDGE-LIFECYCLE
LICENSE      : CC BY 4.0
===============================================================================

Every figure on screen is read from experiments/counterfactual_summary.json and
experiments/counterfactual_update.csv, and checked against the macros in
preprint/numbers.tex before a single frame is drawn. If the paper and the
measurement ever disagree, rendering stops.

The visual system is the one every other surface of the project uses: white
with a faint grid, the five lifecycle colours, ink and muted greys, Inter, and
the five-colour band along the bottom edge from the repository card.

Frames are drawn at twice the output size and reduced with a Lanczos filter.
Pillow's shapes are not antialiased, so supersampling is what keeps every bar
and rounded corner clean.

Usage
    python tools/video/render.py frames DIR     PNG frames for the video, 30 fps
    python tools/video/render.py gifs           media/*.gif
    python tools/video/render.py cover          media/cover.png
    python tools/video/render.py stills DIR T1 T2 ...   frames at given seconds
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FONT = HERE / "fonts" / "inter-variable.woff2"
MEDIA = ROOT / "media"

# --- canvas ------------------------------------------------------------------
W = H = 1080          # square: it holds up in both the LinkedIn and the X feed
S = 2                 # supersampling factor
M = 88                # side margin
CW = W - 2 * M        # content width, 904
FPS = 30
GIF_FPS = 20
GIF_SIZE = 720
GIF_HOLD = 3.0        # extra seconds a loop rests on its finished state

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
    }
    bad = [f"{k}: paper {macros.get(k)!r}, data {v!r}" for k, v in want.items()
           if k not in macros or float(macros[k]) != float(v)]
    if bad:
        raise SystemExit("numbers disagree with preprint/numbers.tex:\n  " + "\n  ".join(bad))
    knows = [m["knows"] for m in d["models"]]
    assert knows == sorted(knows), "holding the fact is no longer monotone in size"
    assert d["ex_qwen"]["says_cf"] and d["ex_tiny"]["says_true"]
    ex = d["ex_rank"]
    assert ex["p_cf"] > 0.10 and ex["says_true"] and ex["rank_cf"] == 2
    assert d["n_probes"] == 24 and d["n_phrasings"] == 3
    return d


D: dict = {}


# =============================================================================
# Drawing primitives. Coordinates are in output pixels; P() maps to the canvas.
# =============================================================================
FADE = 1.0            # the exit fade of the current scene, applied to content

# Below this opacity nothing is drawn. A near-white glyph or panel is lighter
# than the grid lines it sits on, so drawing it would erase the grid in the shape
# of the element and leave a ghost of text that has not appeared yet.
MIN_A = 0.04


def hidden(a: float, fade: bool = True) -> bool:
    return a * (FADE if fade else 1.0) < MIN_A


def P(v: float) -> int:
    return int(round(v * S))


def clamp(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def ease_out(x: float) -> float:
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_io(x: float) -> float:
    x = clamp(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def prog(t: float, start: float, dur: float = 0.6) -> float:
    return ease_out((t - start) / dur)


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
        d.text((P(x), P(y)), s, font=f, fill=c, anchor=anchor)
        return f.getlength(s) / S
    widths = [f.getlength(ch) for ch in s]
    total = sum(widths) + P(track) * (len(s) - 1)
    h, v = anchor[0], anchor[1]
    x0 = P(x) - (total / 2 if h == "m" else total if h == "r" else 0)
    for ch, w in zip(s, widths):
        d.text((x0, P(y)), ch, font=f, fill=c, anchor="l" + v)
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
    d.rounded_rectangle([P(x0), P(y0), P(x1), P(y1)], radius=P(r),
                        fill=col(fill, a, bg) if fill else None,
                        outline=col(outline, a, bg) if outline else None,
                        width=P(width) if outline else 0)


def rect(d, x0, y0, x1, y1, fill, a=1.0, bg=WHITE, fade=True):
    if P(x1) - 1 < P(x0) or P(y1) - 1 < P(y0) or hidden(a, fade):
        return
    d.rectangle([P(x0), P(y0), P(x1) - 1, P(y1) - 1], fill=col(fill, a, bg, fade))


def line(d, pts, color, w, a=1.0):
    if hidden(a):
        return
    d.line([(P(x), P(y)) for x, y in pts], fill=col(color, a), width=P(w), joint="curve")


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
    d.ellipse([P(cx - r), P(cy - r), P(cx + r), P(cy + r)],
              fill=col(fill, a) if fill else None,
              outline=col(outline, a) if outline else None,
              width=P(width) if outline else 0)


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


@lru_cache(maxsize=None)
def portrait_img(name: str, dia: int) -> Image.Image:
    src = Image.open(ROOT / "preprint" / name).convert("RGB")
    side = min(src.size)
    src = src.crop(((src.width - side) // 2, (src.height - side) // 2,
                    (src.width + side) // 2, (src.height + side) // 2))
    px = P(dia)
    img = src.resize((px, px), Image.LANCZOS).convert("RGBA")
    big = Image.new("L", (px * 4, px * 4), 0)
    ImageDraw.Draw(big).ellipse([0, 0, px * 4 - 1, px * 4 - 1], fill=255)
    img.putalpha(big.resize((px, px), Image.LANCZOS))
    return img


def portrait(cv, name, cx, cy, dia, a=1.0):
    img = portrait_img(name, dia)
    k = clamp(a * FADE)
    if k <= 0:
        return
    if k < 1:
        img = img.copy()
        img.putalpha(img.getchannel("A").point(lambda v: int(v * k)))
    cv.im.paste(img, (P(cx - dia / 2), P(cy - dia / 2)), img)
    circle(cv.d, cx, cy, dia / 2 + 1, outline=HAIR, width=2, a=a)


# =============================================================================
# Chrome: the grid, the band, the footer, the eyebrow and the progress rule
# =============================================================================
@lru_cache(maxsize=1)
def base() -> Image.Image:
    im = Image.new("RGB", (P(W), P(H)), WHITE)
    d = ImageDraw.Draw(im)
    for k in range(0, W + 1, 36):
        d.line([(P(k), 0), (P(k), P(H))], fill=GRID, width=P(1))
        d.line([(0, P(k)), (P(W), P(k))], fill=GRID, width=P(1))
    for i, (_, c) in enumerate(STAGES):
        d.rectangle([P(i * W / 5), P(H - 10), P((i + 1) * W / 5) - 1, P(H) - 1], fill=c)
    return im


class Canvas:
    def __init__(self):
        self.im = base().copy()
        self.d = ImageDraw.Draw(self.im)

    def out(self, size=W) -> Image.Image:
        return self.im.resize((size, size), Image.LANCZOS)


def footer(cv, a):
    d = cv.d
    line(d, [(M, 1000), (W - M, 1000)], HAIR, 1, a)
    text(d, M, 1036, "Amey Thakur  \u00b7  Sarvesh Talele", 17, 500, MUTED, a, fade=False)
    text(d, W - M, 1036, VIXRA, 17, 500, MUTED, a, anchor="rs", fade=False)


def eyebrow(cv, t, number, label, accent):
    d = cv.d
    p = prog(t, 0.05, 0.5)
    rrect(d, M, 82, M + 12, 94, 3, fill=accent, a=p)
    x = M + 26
    if number:
        x += text(d, x, 88, number, 17, 600, accent, p, anchor="lm", track=2) + 14
    text(d, x, 88, label.upper(), 17, 600, NAVY, p, anchor="lm", track=2.4)


def progress(cv, index, total, accent, a):
    d = cv.d
    seg, gap = 22, 6
    x = W - M - total * seg - (total - 1) * gap
    for i in range(total):
        c = accent if i == index else (0xC4, 0xC6, 0xCE) if i < index else (0xE8, 0xE9, 0xEE)
        rrect(d, x, 86, x + seg, 90, 2, fill=c, a=a)
        x += seg + gap


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
# Scenes. Each draws itself at local time t within a scene lasting dur seconds.
# =============================================================================
def s_title(cv, t, dur, gif):
    d = cv.d
    p = prog(t, 0.1, 0.6)
    x = M
    x += text(d, x, 170, "LLM KNOWLEDGE LIFECYCLE", 19, 600, NAVY, p, track=3.2)
    x += text(d, x + 14, 170, "\u00b7", 19, 500, MUTED, p) + 28
    text(d, x, 170, VIXRA, 19, 500, MUTED, p)

    size = 70
    for i, ln in enumerate(["The Knowledge Lifecycle", "of Large Language Models"]):
        q = prog(t, 0.3 + i * 0.15, 0.7)
        text(d, M - 3, 268 + i * 84 + (1 - q) * 20, ln, size, 650, INK, q)

    pills(cv, t, 0.9, 398, 58, 14, 23)

    q = prog(t, 1.6, 0.7)
    para(d, M, 552 + (1 - q) * 12, "A model holds two memories. Which one comes out "
         "is not a question of size.", 30, 400, MUTED, CW, 42, q)

    for i, (img, name, city) in enumerate([
            ("amey-thakur.jpg", "AMEY THAKUR", "TORONTO, CANADA"),
            ("sarvesh-talele.jpg", "SARVESH TALELE", "MUMBAI, INDIA")]):
        q = prog(t, 2.2 + i * 0.15, 0.7)
        bx = M + i * 452
        portrait(cv, img, bx + 58, 760, 116, q)
        text(d, bx + 140, 752, name, 21, 650, INK, q, track=2.2)
        text(d, bx + 140, 784, city, 15, 500, MUTED, q, track=2.0)

    q = prog(t, 2.7, 0.7)
    line(d, [(M, 880), (W - M, 880)], HAIR, 1, q)
    text(d, M, 922, "Independent Research", 19, 500, MUTED, q)
    text(d, W - M, 922, "Artificial Intelligence  \u00b7  November 2026", 19, 500,
         MUTED, q, anchor="rs")


def s_question(cv, t, dur, gif):
    d = cv.d
    lines = [("You give a language model a document.", 0.4),
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


def s_experiment(cv, t, dur, gif):
    d = cv.d
    q = prog(t, 0.15, 0.6)
    para(d, M, 196 + (1 - q) * 12, "Take a fact every model holds. Then hand it a "
         "document that says otherwise.", 40, 650, INK, CW, 52, q)

    y0 = 290
    ctx = [("Following a constitutional reform, the capital of France was relocated "
            "to ", 25, 400, INK), ("Lyon", 25, 700, RET), (". ", 25, 400, INK),
           ("Lyon", 25, 700, RET), (" is now the official capital of France.", 25, 400, INK)]
    qst = [("What is the capital of France? Answer: The capital of France is", 25, 500, INK)]
    nc = len(layout(ctx, CW - 64))
    nq = len(layout(qst, CW - 64))
    card_h = 34 + 20 + nc * 36 + 26 + 20 + nq * 36 + 26
    p = prog(t, 0.7, 0.6)
    rrect(d, M, y0, W - M, y0 + card_h, 14, fill=PANEL, a=p)
    text(d, M + 32, y0 + 44, "CONTEXT", 14, 700, MUTED, p, track=2.4, bg=col(PANEL, p))
    typed = (t - 1.0) * 70
    rich(d, M + 32, y0 + 82, ctx, CW - 64, 36, p, chars=int(max(0, typed)), bg=col(PANEL, p))
    yq = y0 + 82 + (nc - 1) * 36 + 50
    qa = prog(t, 1.0 + rich_count(ctx) / 70, 0.4)
    text(d, M + 32, yq, "QUESTION", 14, 700, MUTED, qa, track=2.4, bg=col(PANEL, p))
    rich(d, M + 32, yq + 38, qst, CW - 64, 36, qa, bg=col(PANEL, p))
    if qa > 0.5 and int(t * 2.2) % 2 == 0:
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

    x0, x1, y_top, y_bot = 152, W - M, 360, 690
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
    y = 826
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
    d = cv.d
    p = prog(t, 0.1, 0.6)
    text(d, M, 170, "READ THE PAPER", 19, 600, NAVY, p, track=3.2)
    for i, ln in enumerate(["The Knowledge Lifecycle", "of Large Language Models"]):
        q = prog(t, 0.25 + i * 0.12, 0.7)
        text(d, M - 3, 246 + i * 70 + (1 - q) * 18, ln, 58, 650, INK, q)
    pills(cv, t, 0.7, 350, 46, 12, 19)

    for i, (img, name, city) in enumerate([
            ("amey-thakur.jpg", "AMEY THAKUR", "TORONTO, CANADA"),
            ("sarvesh-talele.jpg", "SARVESH TALELE", "MUMBAI, INDIA")]):
        q = prog(t, 1.2 + i * 0.12, 0.7)
        bx = M + i * 452
        portrait(cv, img, bx + 50, 494, 100, q)
        text(d, bx + 124, 488, name, 20, 650, INK, q, track=2.2)
        text(d, bx + 124, 518, city, 15, 500, MUTED, q, track=2.0)

    links = [("PAPER", "vixra.org/abs/2611.0038"),
             ("CODE AND DATA", "github.com/Amey-Thakur/LLM-KNOWLEDGE-LIFECYCLE"),
             ("LIVE DEMO", "huggingface.co/spaces/ameythakur/llm-knowledge-lifecycle")]
    for i, (label, url) in enumerate(links):
        q = prog(t, 1.7 + i * 0.15, 0.6)
        y = 640 + i * 66
        line(d, [(M, y - 40), (W - M, y - 40)], HAIR, 1, q)
        text(d, M, y, label, 14, 700, MUTED, q, track=2.2)
        text(d, M + 200, y, url, 22, 500, INK, q)
    q = prog(t, 2.3, 0.6)
    line(d, [(M, 838), (W - M, 838)], HAIR, 1, q)
    text(d, M, 900, "All code, probes and measurements are public.", 27, 500, INK, q)
    text(d, M, 940, f"{VIXRA}  \u00b7  Artificial Intelligence  \u00b7  CC BY 4.0", 19, 500,
         MUTED, q)


# key, seconds, accent, eyebrow label, draw
SCENES = [
    ("title", 4.5, NAVY, "", s_title),
    ("question", 5.0, ACQ, "The question", s_question),
    ("framework", 7.5, STO, "The framework", s_framework),
    ("experiment", 8.5, RET, "The experiment", s_experiment),
    ("two-models", 6.5, UPD, "One document, two answers", s_two_models),
    ("overall", 5.0, RET, "Who wins", s_overall),
    ("scale", 9.0, UPD, "The finding", s_scale),
    ("wording", 7.0, FOR, "The wording", s_wording),
    ("rank", 9.0, RET, "Probability is not an answer", s_rank),
    ("correction", 5.5, STO, "A correction", s_correction),
    ("end", 6.0, NAVY, "", s_end),
]
CONTENT = [k for k, *_ in SCENES if k not in ("title", "end")]
EXIT = 0.5


def total_seconds() -> float:
    return sum(s[1] for s in SCENES)


def draw_scene(cv, key, t, gif=False, fade_out=True):
    global FADE
    idx = [s[0] for s in SCENES].index(key)
    _, dur, accent, label, fn = SCENES[idx]
    FADE = 1 - ease_io((t - (dur - EXIT)) / EXIT) if fade_out else 1.0
    if key in CONTENT:
        number = None if gif else f"{CONTENT.index(key) + 1:02d}"
        eyebrow(cv, t, number, label, accent)
    fn(cv, t, dur, gif)
    FADE = 1.0


def frame_at(T: float) -> Image.Image:
    """The video frame at global time T."""
    cv = Canvas()
    acc = 0.0
    for i, (key, dur, accent, _, _) in enumerate(SCENES):
        if T < acc + dur or i == len(SCENES) - 1:
            t = T - acc
            if key in CONTENT:
                # the footer and progress rule persist across content scenes,
                # fading in on the first and out on the last
                a = 1.0
                if key == CONTENT[0]:
                    a = ease_io(t / 0.5)
                if key == CONTENT[-1]:
                    a = 1 - ease_io((t - (dur - EXIT)) / EXIT)
                footer(cv, a)
                progress(cv, CONTENT.index(key), len(CONTENT), accent, a)
            draw_scene(cv, key, t, gif=False, fade_out=(key != "end"))
            return cv.out()
        acc += dur
    raise AssertionError


def gif_frame(key: str, t: float) -> Image.Image:
    cv = Canvas()
    footer(cv, 1.0)
    draw_scene(cv, key, t, gif=True, fade_out=True)
    return cv.out(GIF_SIZE)


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
    print(f"  {n} frames, {total_seconds():.1f} s at {FPS} fps -> {outdir}")


GIFS = [("framework", "lifecycle-framework.gif"),
        ("two-models", "one-document-two-answers.gif"),
        ("scale", "scale-is-not-deference.gif"),
        ("rank", "probability-is-not-an-answer.gif")]


def _gif_frames(key):
    """A loop holds its finished state longer than the video does, because a
    GIF has no next scene to move on to. Held frames are identical, and Pillow
    merges identical frames into one, so the hold costs nothing in file size."""
    dur = next(s[1] for s in SCENES if s[0] == key)
    settle = dur - EXIT
    n = int(round((dur + GIF_HOLD) * GIF_FPS))
    out = []
    for i in range(n):
        tau = i / GIF_FPS
        t = tau if tau < settle else settle if tau < settle + GIF_HOLD else tau - GIF_HOLD
        out.append(gif_frame(key, t))
    return out


def _gif_job(job):
    key, name = job
    frames = _gif_frames(key)
    # one palette for the whole loop, built from frames across it, so colours
    # never shift between frames; no dithering, so flat fills stay flat
    sample = frames[len(frames) // 3 :: max(1, len(frames) // 12)]
    sheet = Image.new("RGB", (GIF_SIZE, GIF_SIZE * len(sample)))
    for i, f in enumerate(sample):
        sheet.paste(f, (0, i * GIF_SIZE))
    pal = sheet.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    out = MEDIA / name
    hold = [int(1000 / GIF_FPS)] * len(q)
    q[0].save(out, save_all=True, append_images=q[1:], duration=hold, loop=0,
              optimize=False, disposal=1)
    return f"  {name}: {len(q)} frames, {out.stat().st_size / 1e6:.2f} MB"


def cmd_gifs():
    MEDIA.mkdir(exist_ok=True)
    with Pool(initializer=_init) as pool:
        for msg in pool.imap(_gif_job, GIFS):
            print(msg, flush=True)


def cmd_cover():
    MEDIA.mkdir(exist_ok=True)
    cv = Canvas()
    draw_scene(cv, "title", 4.4, fade_out=False)
    cv.out().save(MEDIA / "cover.png", optimize=True)
    print(f"  cover.png written")


def cmd_stills(outdir: str, times: list[str]):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    for T in times:
        frame_at(float(T)).save(Path(outdir) / f"t{float(T):06.2f}.png")
    print(f"  {len(times)} stills -> {outdir}")


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    _init()
    print(f"  data verified against preprint/numbers.tex; runtime {total_seconds():.1f} s")
    cmd = argv[0]
    if cmd == "frames":
        cmd_frames(argv[1])
    elif cmd == "gifs":
        cmd_gifs()
    elif cmd == "cover":
        cmd_cover()
    elif cmd == "stills":
        cmd_stills(argv[1], argv[2:])
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
