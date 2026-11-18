"""The narrated cut of the explainer.

The silent film, media/knowledge-lifecycle.mp4, is made for a muted feed and
stays exactly as it is. This script lays a spoken narration and a quiet
synthesized bed under the same frames and writes two files beside it:

    media/knowledge-lifecycle-narrated.mp4   the same frames, with sound
    media/knowledge-lifecycle-narrated.srt   the narration as captions, for upload
    media/knowledge-lifecycle-narrated.vtt   the same, for the watch page

    python tools/video/voiceover.py           build both
    python tools/video/voiceover.py --plan    print the timing table, build nothing

Every number the voice says is formatted from the same data the frames are
drawn from (render.load_data, which refuses to run if any figure disagrees
with the paper), and spoken from the exact string the frame prints, so the
voice cannot say a figure the film does not show. Each line starts when its
subject appears on screen, read off the renderer's own pacing, and the build
stops if a line would still be playing when the next scene's dissolve begins.

The voice is Kokoro-82M (Apache-2.0), run through kokoro-onnx. The bed is
synthesized here from sine partials, so the soundtrack holds no third-party
recording. None of this is needed by the silent build:

    pip install kokoro-onnx soundfile scipy pyloudnorm imageio-ffmpeg
    set KOKORO_DIR to a folder holding kokoro-v1.0.onnx and voices-v1.0.bin,
    both from github.com/thewh1teagle/kokoro-onnx, release model-files-v1.0
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import render  # noqa: E402

SILENT = ROOT / "media" / "knowledge-lifecycle.mp4"
OUT_MP4 = ROOT / "media" / "knowledge-lifecycle-narrated.mp4"
OUT_SRT = ROOT / "media" / "knowledge-lifecycle-narrated.srt"
OUT_VTT = ROOT / "media" / "knowledge-lifecycle-narrated.vtt"

VOICE, SPEED = "af_heart", 0.95
SR = 48000
LEAD = 0.35        # a scene's first line starts this long after its boundary
GAP = 0.40         # the silence between two lines of one scene
CLEAR = 0.45       # a line ends this long before the boundary (the dissolve opens at B - 0.6)
LINE_LUFS = -20.0  # every line is brought to one loudness before it is placed
VOICE_LUFS = -17.0
BED_BELOW = 12.0   # the bed sits this many LU under the voice
DUCK_DB = {"pad": 6.0, "keys": 9.0}   # and each layer drops this much more under it
MIX_LUFS = -16.0   # the finished track
PEAK_DB = -1.5     # and its highest sample

# The bed's arrangement. TEMPO sets the keys' pattern, which restarts on a
# downbeat as each dissolve completes, so every scene opens on the one.
# INTENSITY is how much of the arrangement a scene carries: the keys come in
# sparse, are fullest on the finding, and leave the correction to the pad alone.
TEMPO = 84.0
INTENSITY = {"title": 0.55, "question": 0.65, "framework": 0.75, "experiment": 0.85,
             "two-models": 0.85, "overall": 0.9, "scale": 1.0, "wording": 0.9,
             "rank": 0.9, "correction": 0.3, "end": 0.6}

# Words the phonemizer reads wrongly ("Amy Thacker", "Q-wen"), given as phonemes.
SAY = {"Amey": "əmˈeɪ", "Thakur": "tˈɑːkʊɹ", "Sarvesh": "sɚvˈeɪʃ",
       "Talele": "təlˈeɪleɪ", "Qwen": "kwˈɛn"}

# Per scene: (authored second its subject appears, or None to follow the line
# before it; the line). {name} is a figure from the data, printed as the frame
# prints it and spoken from that same string; [shown|spoken] differ in the
# captions and the voice only.
NARRATION = {
    "title": [
        (None, "The Knowledge Lifecycle of Large Language Models."),
    ],
    "question": [
        (None, "Give a language model a document that contradicts what it learned."),
        (2.3, "Which one wins?"),
        (3.3, "Retrieval works only when the document beats what the model already "
              "believes."),
    ],
    "framework": [
        (0.15, "A fact passes through five stages: acquire, store, retrieve, update "
               "and forget."),
        (2.7, "The failure sits between retrieval and update."),
        (3.4, "The document arrives, and nothing tells the model which copy to trust."),
    ],
    "experiment": [
        (0.15, "Take a fact every model holds, then hand it a document that says "
               "otherwise."),
        (3.0, "Its memory says Paris. The document says Lyon."),
        (4.0, "{n_rows} forward passes, across {n_models} open models."),
    ],
    "two-models": [
        (0.15, "Same question, same document."),
        (2.4, "[Qwen2.5-1.5B|Qwen two point five] adopts the document and answers Lyon."),
        (2.9, "TinyLlama keeps Paris."),
    ],
    "overall": [
        (0.15, "Across {n_conflict} genuine conflicts, the document wins two times in "
               "three."),
        (0.7, "{adopt}% adopt it. {keep}% keep their own answer."),
    ],
    "scale": [
        (0.15, "Holding a fact scales with size. Using the document does not."),
        (2.7, "TinyLlama holds almost every fact, yet adopts the document least of "
              "all six."),
        (4.6, "[Qwen2.5-0.5B, less than half its size,|A Qwen less than half its size] "
              "adopts it most: {qwen_small_adopt}%."),
    ],
    "wording": [
        (0.15, "Now change only the wording."),
        (None, "TinyLlama's adoption moves from {lo}% to {hi}%."),
        (3.9, "A single-phrasing result reports the sentence as much as the model."),
    ],
    "rank": [
        (0.15, "Probability is not an answer."),
        (2.3, "Lyon carries {p_cf} of the probability. The model still says Paris."),
        (3.8, "Above a tenth of the probability, the document's answer goes unsaid "
              "{over10_not_pct}% of the time."),
    ],
    "correction": [
        (0.2, "That gap is not hypothetical."),
        (1.0, "An earlier version of this work read probability alone, and reached a "
              "conclusion the measurement did not support."),
        (2.4, "This paper corrects it, on the record."),
    ],
    "end": [
        (1.2, "By Amey Thakur and Sarvesh Talele."),           # as the faces land
        (2.05, "All code, probes and measurements are public."),
    ],
}

# The bed: one chord a scene, changing inside the frame's own dissolve.
# MIDI notes, the root first. D major, moving through its relative minor.
CHORDS = {
    "title": [38, 50, 54, 57, 61, 64],        # Dmaj9
    "question": [35, 47, 50, 54, 57, 61],     # Bm9
    "framework": [31, 43, 47, 50, 54, 57],    # Gmaj9
    "experiment": [33, 45, 52, 57, 59, 64],   # Asus2
    "two-models": [30, 42, 45, 49, 52, 57],   # F#m7
    "overall": [31, 43, 47, 50, 54, 59],      # Gmaj7
    "scale": [28, 40, 43, 47, 50, 54],        # Em9
    "wording": [35, 47, 50, 54, 57, 61],      # Bm9
    "rank": [31, 43, 47, 50, 54, 57],         # Gmaj9
    "correction": [33, 45, 52, 57, 59, 61],   # Aadd9
    "end": [38, 50, 54, 57, 61, 64],          # Dmaj9
}


# --- the words ---------------------------------------------------------------
ONES = ("zero one two three four five six seven eight nine ten eleven twelve "
        "thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def words_int(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else "-" + ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return ONES[n // 100] + " hundred" + ("" if rest == 0 else " and " + words_int(rest))
    raise ValueError(n)


def spoken_number(shown: str) -> str:
    """The figure exactly as printed, read aloud: 66.8 -> sixty-six point eight."""
    whole, _, frac = shown.partition(".")
    out = words_int(int(whole))
    if frac.strip("0"):
        out += " point " + " ".join(ONES[int(c)] for c in frac)
    return out


def figures(D: dict) -> dict[str, str]:
    """Every figure the narration uses, formatted as its frame formats it."""
    lo, hi = min(D["tiny_by_phrasing"]), max(D["tiny_by_phrasing"])
    # the claims the narration makes in words, checked against the data
    assert D["ex_qwen"]["says_cf"] and D["ex_tiny"]["says_true"], "two-models verdicts"
    assert D["ex_rank"]["says_true"] and not D["ex_rank"]["says_cf"], "rank example"
    assert 60 < D["adopt"] < 72, "'two times in three' no longer describes the rate"
    assert D["tiny_adopt"] == min(m["adopt"] for m in D["models"]), "TinyLlama least"
    assert D["tiny_knows"] >= 95, "'almost every fact' no longer describes TinyLlama"
    assert D["qwen_small_adopt"] == max(m["adopt"] for m in D["models"]), "Qwen most"

    def params(m):  # "494M" or "1.1B", in millions
        return float(m["size"][:-1]) * (1000 if m["size"].endswith("B") else 1)
    tiny = next(m for m in D["models"] if m["name"] == "TinyLlama")
    small = next(m for m in D["models"] if m["adopt"] == D["qwen_small_adopt"])
    assert small["name"].startswith("Qwen") and params(small) < params(tiny) / 2, "half its size"
    return {
        "n_models": f"{D['n_models']}", "n_probes": f"{D['n_probes']}",
        "n_phrasings": f"{D['n_phrasings']}", "n_rows": f"{D['n_rows']}",
        "n_conflict": f"{D['n_conflict']}",
        "adopt": f"{D['adopt']}", "keep": f"{D['keep']}",
        "tiny_knows": f"{D['tiny_knows']}", "tiny_adopt": f"{D['tiny_adopt']}",
        "qwen_small_adopt": f"{D['qwen_small_adopt']}",
        "lo": f"{lo:.1f}", "hi": f"{hi:.1f}", "spread": f"{D['max_adopt_spread']}",
        "p_cf": f"{D['ex_rank']['p_cf']:.2f}", "over10_not_pct": f"{D['over10_not_pct']}",
    }


def resolve(line: str, figs: dict[str, str]) -> tuple[str, str]:
    """(caption text, spoken text) for one line."""
    unknown = set(re.findall(r"\{(\w+)\}", line)) - set(figs)
    if unknown:
        raise SystemExit(f"narration names figures the data lacks: {sorted(unknown)}")
    shown = re.sub(r"\[([^|\]]*)\|[^\]]*\]", r"\1", line)
    spoken = re.sub(r"\[[^|\]]*\|([^\]]*)\]", r"\1", line)
    shown = re.sub(r"\{(\w+)\}", lambda m: figs[m.group(1)], shown)
    spoken = re.sub(r"\{(\w+)\}", lambda m: spoken_number(figs[m.group(1)]), spoken)
    spoken = spoken.replace("%", " percent")
    spoken = spoken[0].upper() + spoken[1:]
    return shown, spoken


# --- the voice ---------------------------------------------------------------
class Voice:
    def __init__(self, voice: str, speed: float):
        from kokoro_onnx import Kokoro
        d = Path(os.environ.get("KOKORO_DIR", ROOT / ".kokoro"))
        model, voices = d / "kokoro-v1.0.onnx", d / "voices-v1.0.bin"
        if not model.exists():
            raise SystemExit(f"no Kokoro model at {model}; set KOKORO_DIR")
        self.k = Kokoro(str(model), str(voices))
        self.voice, self.speed = voice, speed
        self.cache = Path(tempfile.gettempdir()) / "kl-voiceover"
        self.cache.mkdir(exist_ok=True)

    def phonemes(self, text: str) -> str:
        parts = re.split(r"\b(" + "|".join(SAY) + r")\b", text)
        out = []
        for p in parts:
            if p in SAY:
                out.append(SAY[p])
            elif re.search(r"[A-Za-z0-9]", p):
                out.append(self.k.tokenizer.phonemize(p, "en-us").strip())
            elif p.strip():
                out[-1] += p.strip()            # punctuation joins the word before it
        return " ".join(out)

    def say(self, text: str) -> np.ndarray:
        """One line at SR, mono float32, silence trimmed."""
        from scipy.signal import resample_poly
        ph = self.phonemes(text)
        key = hashlib.sha256(f"{self.voice}|{self.speed}|{ph}".encode()).hexdigest()[:20]
        f = self.cache / f"{key}.npy"
        if not f.exists():
            a, sr = self.k.create(ph, voice=self.voice, speed=self.speed, is_phonemes=True)
            assert sr == 24000, sr
            np.save(f, resample_poly(a.astype(np.float64), 2, 1).astype(np.float32))
        return np.load(f)


# --- the timing --------------------------------------------------------------
def plan(durations) -> list[dict]:
    """Place every line; durations(spoken) gives its length in seconds."""
    D = render.load_data()
    figs = figures(D)
    starts = render.starts()
    total = render.total_seconds()
    rows, problems = [], []
    for i, (key, dur, *_rest) in enumerate(render.SCENES):
        b0, b1 = starts[i], starts[i] + dur
        limit = (total - 0.6) if i == len(render.SCENES) - 1 else b1 - CLEAR
        prev = None
        for at, line in NARRATION[key]:
            shown, spoken = resolve(line, figs)
            earliest = b0 + (render.paced(key, at) if at is not None else LEAD)
            t0 = max(earliest, b0 + LEAD, (prev + GAP) if prev is not None else 0.0)
            d = durations(spoken)
            t1 = t0 + d
            if t1 > limit:
                problems.append(f"{key}: \"{shown[:48]}...\" ends {t1:.2f}, limit {limit:.2f}")
            rows.append(dict(key=key, t0=t0, t1=t1, shown=shown, spoken=spoken))
            prev = t1
    for r in rows:
        print(f"  {r['key']:11s} {r['t0']:7.2f} {r['t1']:7.2f}  {r['shown']}")
    if problems:
        raise SystemExit("lines that run into the next scene:\n  " + "\n  ".join(problems))
    return rows


# --- the bed -----------------------------------------------------------------
def hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def saw(f: float, t: np.ndarray, phase: float) -> np.ndarray:
    """A band-limited sawtooth (PolyBLEP), so the pad has warmth without the
    aliasing a naive ramp folds back into the audible band."""
    dt = f / SR
    ph = (f * t + phase) % 1.0
    y = 2 * ph - 1
    lo = ph < dt
    x = ph[lo] / dt
    y[lo] -= x + x - x * x - 1
    hi = ph > 1 - dt
    x = (ph[hi] - 1) / dt
    y[hi] -= x * x + x + x + 1
    return y


def epiano(f: float, vel: float, length: float = 2.6) -> np.ndarray:
    """A soft electric-piano note by two-operator FM: a brighter strike that
    mellows as the modulation index decays, then a long natural release."""
    tl = np.arange(int(length * SR)) / SR
    env = (1 - np.exp(-tl / 0.004)) * np.exp(-tl / 1.05)
    index = vel * (1.1 * np.exp(-tl / 0.22) + 0.18)
    y = np.sin(2 * np.pi * f * tl + index * np.sin(2 * np.pi * f * tl))
    y += 0.05 * vel * np.exp(-tl / 0.04) * np.sin(2 * np.pi * 7 * f * tl)   # the tine
    return vel * env * y


def peaking(f0: float, gain_db: float, q: float) -> np.ndarray:
    """One peaking-EQ biquad (RBJ cookbook) as a second-order section."""
    a = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / SR
    al = np.sin(w) / (2 * q)
    b0, b1, b2 = 1 + al * a, -2 * np.cos(w), 1 - al * a
    a0, a1, a2 = 1 + al / a, -2 * np.cos(w), 1 - al / a
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def room(x: np.ndarray, rng, wet: float, decay: float = 0.42,
         predelay: float = 0.022) -> np.ndarray:
    """A synthetic stereo room: decorrelated decaying noise, darkened, with a
    short pre-delay so the dry attack stays distinct from its tail."""
    from scipy.signal import butter, fftconvolve, sosfilt
    n = x.shape[0]
    L = int(6 * decay * SR)
    tl = np.arange(L) / SR
    ir = rng.standard_normal((L, 2)) * np.exp(-tl / decay)[:, None]
    ir = sosfilt(butter(1, 2800, btype="low", fs=SR, output="sos"), ir, axis=0)
    ir /= np.sqrt((ir ** 2).sum(axis=0))
    ir = np.concatenate([np.zeros((int(predelay * SR), 2)), ir])
    tail = np.stack([fftconvolve(x[:, c], ir[:, c])[:n] for c in range(2)], axis=1)
    return (1 - wet) * x + wet * tail


def bed(total: float, seed: int = 11) -> dict[str, np.ndarray]:
    """The music under the voice, as layers the mix ducks separately:

    pad    a warm detuned-saw chord per scene, one octave of sub beneath it,
           crossfading equal-power inside each of the frame's dissolves
    keys   a soft electric piano on the chord tones, its density following
           INTENSITY, restarting on a downbeat as each dissolve completes
    air    a breath of filtered noise that rises through each dissolve and
           opens from dark to bright with it, panned a little across

    The whole bed is cut by 4 dB around 2.8 kHz, where the voice's consonants
    live, so the words stay clear without the music having to be turned down."""
    from scipy.signal import butter, sosfilt
    rng = np.random.default_rng(seed)
    n = int(round(total * SR))
    t = np.arange(n) / SR
    keys_ = [s[0] for s in render.SCENES]
    B = render.starts() + [total]
    PRE, POST = render.PRE, render.POST

    def window(i):
        """Equal-power gain of scene i's chord across the film."""
        a0, a1 = (0.0, 2.2) if i == 0 else (B[i] - PRE, B[i] + POST)
        c0, c1 = (B[i + 1] - PRE, B[i + 1] + POST) if i < len(keys_) - 1 else (total - 3.2, total)
        up = np.sin(0.5 * np.pi * np.clip((t - a0) / (a1 - a0), 0, 1))
        down = np.cos(0.5 * np.pi * np.clip((t - c0) / (c1 - c0), 0, 1))
        return up * down

    # pad and sub
    pad = np.zeros((n, 2))
    for i, key in enumerate(keys_):
        g = window(i)
        live = np.flatnonzero(g > 1e-4)
        tt = t[live]
        layer = np.zeros((tt.size, 2))
        root, *tones = CHORDS[key]
        for m in tones:
            for cents, pan in ((-7.0, 0.2), (0.0, 0.5), (7.0, 0.8)):
                y = saw(hz(m) * 2 ** (cents / 1200), tt, rng.uniform())
                layer[:, 0] += 0.16 * y * np.cos(0.5 * np.pi * pan)
                layer[:, 1] += 0.16 * y * np.sin(0.5 * np.pi * pan)
        cutoff = 520 + 760 * INTENSITY[key]           # brighter as the scene carries more
        layer = sosfilt(butter(4, cutoff, btype="low", fs=SR, output="sos"), layer, axis=0)
        breathe = 1 + 0.10 * np.sin(2 * np.pi * 0.09 * tt + rng.uniform(0, 2 * np.pi))
        sub = 0.55 * np.sin(2 * np.pi * hz(root) * tt) + 0.12 * np.sin(4 * np.pi * hz(root) * tt)
        layer = layer * breathe[:, None] + sub[:, None]
        pad[live] += layer * g[live, None]

    # keys: eighth-note slots in a bar of four; denser as the scene carries more
    keys = np.zeros((n, 2))
    eighth = 30.0 / TEMPO
    PATTERNS = [(0.0, ((0, 0, 0.85), (4, 1, 0.65))),                       # sparse
                (0.6, ((0, 0, 0.85), (3, 2, 0.55), (4, 1, 0.70))),
                (0.8, ((0, 0, 0.85), (3, 2, 0.55), (4, 1, 0.70), (6, 3, 0.50)))]
    for i, key in enumerate(keys_):
        level = INTENSITY[key]
        if level < 0.4:
            continue                                   # the correction keeps the pad alone
        slots = [p for floor, p in PATTERNS if level >= floor][-1]
        tones = [m + 12 for m in CHORDS[key][2:]]
        start = 0.0 if i == 0 else B[i] + POST
        stop = (B[i + 1] - PRE) if i < len(keys_) - 1 else total - 3.0
        bar = 0
        while True:
            t_bar = start + bar * 8 * eighth
            if t_bar >= stop:
                break
            for slot, which, vel in slots:
                at = t_bar + slot * eighth + rng.uniform(-0.006, 0.006)   # a human hand
                if at >= stop:
                    continue
                m = tones[(which + bar) % len(tones)]
                note = epiano(hz(m), (vel + rng.uniform(-0.05, 0.05)) * level)
                i0 = int(at * SR)
                seg = note[: n - i0]
                pan = 0.5 + 0.22 * np.sin(1.3 * (m - 60))
                keys[i0:i0 + seg.size, 0] += seg * np.cos(0.5 * np.pi * pan)
                keys[i0:i0 + seg.size, 1] += seg * np.sin(0.5 * np.pi * pan)
            bar += 1
    # one last voicing on the home chord, left to ring out under the fade
    for m, v in ((50, 0.55), (57, 0.5), (66, 0.45), (73, 0.35)):
        note = epiano(hz(m), v * INTENSITY["end"], length=3.0)
        i0 = int((total - 3.0) * SR)
        seg = note[: n - i0]
        keys[i0:i0 + seg.size] += seg[:, None] * 0.7

    # air: low and high noise bands crossfaded, so each breath brightens as it rises
    noise = rng.standard_normal((n, 2))
    low = sosfilt(butter(2, [250, 1100], btype="band", fs=SR, output="sos"), noise, axis=0)
    high = sosfilt(butter(2, [1000, 4500], btype="band", fs=SR, output="sos"), noise, axis=0)
    air = np.zeros((n, 2))
    for b in B[1:-1]:
        x0, peak, x1 = b - PRE, b + 0.10, b + POST
        span = (t >= x0) & (t <= x1)
        tt = t[span]
        rise = np.clip((tt - x0) / (peak - x0), 0, 1)
        fall = np.clip(1 - (tt - peak) / (x1 - peak), 0, 1)
        env = np.where(tt < peak, rise ** 2.2, fall ** 1.6)
        bright = np.clip((tt - x0) / (x1 - x0), 0, 1)
        pan = 0.35 + 0.30 * bright
        s = (1 - bright)[:, None] * low[span] + bright[:, None] * high[span]
        air[span, 0] += env * s[:, 0] * np.cos(0.5 * np.pi * pan)
        air[span, 1] += env * s[:, 1] * np.sin(0.5 * np.pi * pan)

    def unit(x):
        return x / np.sqrt((x ** 2).mean())

    hp = butter(2, 38, btype="high", fs=SR, output="sos")
    carve = peaking(2800, -4.0, 0.9)
    out = {}
    for name, x, gain_db, wet in (("pad", pad, 0.0, 0.30), ("keys", keys, -7.0, 0.38),
                                  ("air", air, -30.0, 0.25)):
        x = sosfilt(carve, sosfilt(hp, x, axis=0), axis=0)
        out[name] = room(unit(x) * 10 ** (gain_db / 20), rng, wet)
    # a short fade at the very ends, so nothing starts or stops on a click
    edge = np.minimum(1, np.minimum(t / 0.05, (total - t) / 0.30))[:, None]
    return {k: v * edge for k, v in out.items()}


def activity(voice: np.ndarray) -> np.ndarray:
    """0 to 1 per sample: how far into speech the voice is, eased in over about
    40 ms and back out over about 400 ms, so the bed never pumps."""
    hop = SR // 100
    frames = voice[: voice.size // hop * hop].reshape(-1, hop)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    on = (rms > 0.02 * rms.max()).astype(float)
    g, cur = np.empty(on.size), 0.0
    for i, x in enumerate(on):
        cur += (0.25 if x > cur else 0.025) * (x - cur)
        g[i] = cur
    g = np.repeat(g, hop)
    return np.concatenate([g, np.full(voice.size - g.size, g[-1])])


def shape(line: np.ndarray) -> np.ndarray:
    """One line made ready to place: rumble removed, brought to LINE_LUFS so no
    line stands out from its neighbours, and faded over 8 ms at each end."""
    import pyloudnorm
    from scipy.signal import butter, sosfilt
    x = sosfilt(butter(2, 70, btype="high", fs=SR, output="sos"), line.astype(np.float64))
    x *= 10 ** ((LINE_LUFS - pyloudnorm.Meter(SR).integrated_loudness(x)) / 20)
    k = int(0.008 * SR)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, k))
    x[:k] *= ramp
    x[-k:] *= ramp[::-1]
    return x


def loudness(x: np.ndarray) -> float:
    import pyloudnorm
    return pyloudnorm.Meter(SR).integrated_loudness(x)


def to_lufs(x: np.ndarray, target: float) -> np.ndarray:
    return x * 10 ** ((target - loudness(x)) / 20)


def limit(x: np.ndarray, ceiling_db: float = PEAK_DB, release: float = 0.08) -> np.ndarray:
    """A look-ahead peak limiter: the gain is already down when a peak arrives,
    eases into it over a few milliseconds, and recovers over `release` seconds.
    Speech peaks sit far above its loudness, so without this the whole track
    would have to be turned down to keep its loudest consonant under the ceiling."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    thr = 10 ** (ceiling_db / 20)
    peak = np.abs(x).max(axis=1)
    hop = SR // 1000                                  # 1 ms blocks
    nb = -(-peak.size // hop)
    blocks = np.pad(peak, (0, nb * hop - peak.size)).reshape(nb, hop).max(axis=1)
    need = np.minimum(1.0, thr / np.maximum(blocks, 1e-9))
    need = minimum_filter1d(need, 11)                 # 5 ms of look-ahead either side
    g, cur, k = np.empty(nb), 1.0, 1 - np.exp(-1e-3 / release)
    for i, v in enumerate(need):
        cur = v if v < cur else cur + k * (v - cur)
        g[i] = cur
    g = np.minimum(uniform_filter1d(g, 5), g)         # no step inside a block
    centres = (np.arange(nb) + 0.5) * hop
    gain = np.interp(np.arange(peak.size), centres, g)
    return np.clip(x * gain[:, None], -thr, thr)


# --- captions ----------------------------------------------------------------
def captions(rows, sep: str) -> str:
    """The narration as SRT (sep ",") or as the body of WebVTT (sep ".")."""
    def stamp(s):
        ms = int(round(s * 1000))
        return (f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}"
                f"{sep}{ms % 1000:03d}")

    def wrap(s, w=42):
        out, cur = [], ""
        for word in s.split():
            if cur and len(cur) + 1 + len(word) > w:
                out.append(cur)
                cur = word
            else:
                cur = f"{cur} {word}".strip()
        return out + [cur]

    cues = []
    for r in rows:
        sents = re.split(r"(?<=[.?!])\s+", r["shown"])
        span, at = r["t1"] - r["t0"], r["t0"]
        total = sum(len(s) for s in sents)
        for s in sents:
            d = span * len(s) / total
            lines = wrap(s)
            for j in range(0, len(lines), 2):        # two lines a cue at most
                part = lines[j:j + 2]
                dd = d * sum(len(x) for x in part) / sum(len(x) for x in lines)
                cues.append((at, at + dd, "\n".join(part)))
                at += dd
    return "\n".join(f"{i}\n{stamp(a)} --> {stamp(b)}\n{txt}\n"
                     for i, (a, b, txt) in enumerate(cues, 1))


# --- assembly ----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="print the timing table only")
    ap.add_argument("--voice", default=VOICE)
    ap.add_argument("--speed", type=float, default=SPEED)
    args = ap.parse_args()

    if args.plan:
        # about 2.8 spoken words a second at the default speed, as measured
        plan(lambda s: len(s.split()) / (2.95 * args.speed))
        return

    import imageio_ffmpeg
    import soundfile as sf
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    frames, secs = imageio_ffmpeg.count_frames_and_secs(str(SILENT))
    total = frames / render.FPS
    assert abs(total - render.total_seconds()) < 0.05, (total, render.total_seconds())

    v = Voice(args.voice, args.speed)
    clips = {}

    def durations(spoken):
        clips[spoken] = v.say(spoken)
        return clips[spoken].size / SR

    print("narration:")
    rows = plan(durations)

    n = int(round(total * SR))
    voice = np.zeros(n)
    for r in rows:
        a = shape(clips[r["spoken"]])
        i0 = int(round(r["t0"] * SR))
        voice[i0:i0 + a.size] += a[: n - i0]
    voice = to_lufs(voice, VOICE_LUFS)

    layers = bed(total)
    trim = 10 ** ((VOICE_LUFS - BED_BELOW - loudness(sum(layers.values()))) / 20)
    talk = activity(voice)[:, None]
    music = sum(x * trim * 10 ** (-DUCK_DB.get(k, 0.0) * talk / 20) for k, x in layers.items())
    mix = music + voice[:, None]
    for _ in range(3):                    # limiting lowers loudness a little; settle it
        mix = limit(to_lufs(mix, MIX_LUFS))
    lufs, peak = loudness(mix), 20 * np.log10(np.abs(mix).max())
    assert abs(lufs - MIX_LUFS) < 0.3 and peak <= PEAK_DB + 0.01, (lufs, peak)
    print(f"mix {lufs:.1f} LUFS, peak {peak:.1f} dBFS, "
          f"speech {sum(r['t1'] - r['t0'] for r in rows):.1f} of {total:.2f} s")

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "mix.wav"
        sf.write(wav, mix.astype(np.float32), SR, subtype="FLOAT")
        subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                        "-i", str(SILENT), "-i", str(wav),
                        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k", "-ar", str(SR),
                        "-movflags", "+faststart", "-map_metadata", "-1",
                        "-metadata", "title=The Knowledge Lifecycle of Large Language Models",
                        str(OUT_MP4)], check=True)
    OUT_SRT.write_text(captions(rows, ","), encoding="utf-8", newline="\n")
    OUT_VTT.write_text("WEBVTT\n\n" + captions(rows, "."), encoding="utf-8", newline="\n")
    f2, s2 = imageio_ffmpeg.count_frames_and_secs(str(OUT_MP4))
    assert f2 == frames, (f2, frames)
    print(f"wrote {OUT_MP4.relative_to(ROOT)} ({OUT_MP4.stat().st_size / 1e6:.1f} MB, "
          f"{f2} frames), and its captions as .srt and .vtt")


if __name__ == "__main__":
    main()
