"""
===============================================================================
FILE         : make_preview.py
PROJECT      : The Knowledge Lifecycle of Large Language Models
PURPOSE      : Build the 1280x640 social card for the paper, following the
               house style established by the ACCIDENT @ CVPR 2026 card.
TECH STACK   : Python 3, Pillow
AUTHORS      : Amey Thakur, Sarvesh Talele
REPOSITORY   : https://github.com/Amey-Thakur/LLM-KNOWLEDGE-LIFECYCLE
LICENSE      : CC BY 4.0
===============================================================================

Structure: an eyebrow naming the work, the title, a rule, three label and value
rows carrying the contribution, and chips for the hard numbers. The authors
occupy the right third, which is what keeps the canvas balanced rather than
leaving a void beside the text.

The house accent is replaced here by the paper's own five stage colours, which
appear in the rule, the chips, and the band across the foot.
"""

import io
import urllib.request

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

W, H = 1280, 640
MARGIN = 76

BG = (255, 255, 255)
GRID = (247, 247, 249)
INK = (23, 23, 28)                # title and values
MUT = (104, 108, 118)             # secondary text
DIM = (150, 154, 164)             # tertiary text
ACCENT = (26, 79, 138)            # Acquire blue, darkened for a light ground
RULE = (222, 224, 230)            # chip and portrait outlines

# The five stage colours, matching Figure 1 of the paper and the demo.
STAGES = [(74, 127, 212), (42, 157, 143), (224, 138, 46), (208, 83, 83), (143, 95, 184)]
NAMES = ["Acquire", "Store", "Retrieve", "Update", "Forget"]

F = "C:/Windows/Fonts/"
f_eyebrow = ImageFont.truetype(F + "seguisb.ttf", 20)
f_eyebrow_l = ImageFont.truetype(F + "segoeui.ttf", 20)
f_title = ImageFont.truetype(F + "seguisb.ttf", 60)
f_label = ImageFont.truetype(F + "seguisb.ttf", 17)
f_value = ImageFont.truetype(F + "segoeui.ttf", 25)
f_stage = ImageFont.truetype(F + "seguisb.ttf", 19)
f_chip = ImageFont.truetype(F + "segoeui.ttf", 16)
f_name = ImageFont.truetype(F + "seguisb.ttf", 19)
f_affil = ImageFont.truetype(F + "segoeui.ttf", 15)

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)

for x in range(0, W, 32):
    d.line([(x, 0), (x, H)], fill=GRID)
for y in range(0, H, 32):
    d.line([(0, y), (W, y)], fill=GRID)

# A pale wash of stage colour behind the author column, blurred so it reads as
# a tint in the paper rather than as a shape drawn on top of it.
wash = Image.new("RGB", (W, H), BG)
wd = ImageDraw.Draw(wash)
wd.ellipse([850, 80, 1200, 340], fill=(150, 130, 205))
wd.ellipse([960, 150, 1230, 360], fill=(205, 145, 150))
img = Image.blend(img, wash.filter(ImageFilter.GaussianBlur(100)), 0.16)
d = ImageDraw.Draw(img)


def tracked(draw, xy, text, font, fill, tracking=3):
    """Draw text with extra letter spacing, which PIL does not offer natively."""
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += draw.textlength(ch, font=font) + tracking
    return x


# --- eyebrow ---------------------------------------------------------------
end = tracked(d, (MARGIN, 74), "LLM KNOWLEDGE LIFECYCLE", f_eyebrow, ACCENT, 3)
d.text((end + 14, 74), "·", font=f_eyebrow_l, fill=DIM)
d.text((end + 36, 74), "Preprint 2026", font=f_eyebrow_l, fill=MUT)

# --- title -----------------------------------------------------------------
d.text((MARGIN, 128), "The Knowledge Lifecycle", font=f_title, fill=INK)
d.text((MARGIN, 200), "of Large Language Models", font=f_title, fill=INK)

# --- the five stages, named in their own colours ----------------------------
# These are the framework, so they are shown rather than listed as prose.
x = MARGIN
for name, colour in zip(NAMES, STAGES):
    w = d.textlength(name, font=f_stage)
    d.rounded_rectangle([x, 292, x + w + 32, 332], radius=6, fill=colour)
    d.text((x + 16, 300), name, font=f_stage, fill=(255, 255, 255))
    x += w + 32 + 13

# --- contribution rows -----------------------------------------------------
rows = [
    ("METRIC", "lifecycle desynchronization, in nats"),
    ("FINDING", "retrieval succeeds, resolution fails"),
]
y = 378
for label, value in rows:
    tracked(d, (MARGIN, y + 4), label, f_label, ACCENT, 2)
    d.text((MARGIN + 168, y), value, font=f_value, fill=INK)
    y += 44

# --- chips -----------------------------------------------------------------
chips = ["MEASURED ON GPT-2", "D-SYNC 12.05 NATS", "FULLY REPRODUCIBLE"]
x = MARGIN
for text in chips:
    w = d.textlength(text, font=f_chip)
    d.rounded_rectangle([x, 500, x + w + 34, 538], radius=4, outline=RULE, width=1)
    d.text((x + 17, 509), text, font=f_chip, fill=MUT)
    x += w + 34 + 14


def portrait(source, size):
    """Circular author portrait from a local path or a URL."""
    if source.startswith("http"):
        request = urllib.request.Request(source, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            face = Image.open(io.BytesIO(response.read())).convert("RGB")
    else:
        face = Image.open(source).convert("RGB")
    face = ImageOps.fit(face, (size, size), Image.LANCZOS)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, size - 1, size - 1], fill=255)
    return face, mask


# --- authors, holding the right third --------------------------------------
SIZE = 112
authors = [
    ("C:/t/tp/space/amey-thakur.jpg", "AMEY THAKUR"),
    ("https://github.com/sarveshtalele.png", "SARVESH TALELE"),
]
centres = [946, 1122]
for (source, name), cx in zip(authors, centres):
    top = 372
    try:
        face, mask = portrait(source, SIZE)
        img.paste(face, (cx - SIZE // 2, top), mask)
    except Exception as error:
        print(f"  portrait unavailable for {name}: {error}")
    d.ellipse([cx - SIZE // 2 - 3, top - 3, cx + SIZE // 2 + 2, top + SIZE + 2],
              outline=RULE, width=2)
    tw = sum(d.textlength(c, font=f_name) + 1 for c in name)
    tracked(d, (cx - tw / 2, top + SIZE + 18), name, f_name, INK, 1)

affil = "INDEPENDENT RESEARCH"
aw = sum(d.textlength(c, font=f_affil) + 2 for c in affil)
tracked(d, ((centres[0] + centres[1]) / 2 - aw / 2, 542), affil, f_affil, DIM, 2)

# --- foot band, the five stages once more ----------------------------------
seg = W / 5
for i, colour in enumerate(STAGES):
    d.rectangle([i * seg, H - 8, (i + 1) * seg, H], fill=colour)

for path in ("C:/t/tp/space/social-preview.png", "C:/t/tp/.github/social-preview.png"):
    img.save(path, optimize=True)
    print(f"  written {path}")
