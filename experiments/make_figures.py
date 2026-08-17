"""
===============================================================================
FILE         : make_figures.py
PROJECT      : The Knowledge Lifecycle of Large Language Models
PURPOSE      : Draw the cross-model measurement as four charts shared by the
               notebook and the paper, from the CSV the sweep writes.
TECH STACK   : Python 3, pandas, matplotlib
AUTHORS      : Amey Thakur (https://github.com/Amey-Thakur)
               Sarvesh Talele (https://github.com/sarveshtalele)
REPOSITORY   : https://github.com/Amey-Thakur/LLM-KNOWLEDGE-LIFECYCLE
LICENSE      : CC BY 4.0
===============================================================================

Each chart answers one question the notebook opens with, so the figures and the
findings cannot drift: both come from the same file.

Colours are the five lifecycle colours used by the paper, the poster, the slides
and the demonstration. A probe takes the colour of the stage boundary it sits
on, so vioxx is Update red wherever it appears.

Legends sit below the axes. Inside the axes they collide with the title on the
help chart and with the tallest bars on the exposure chart, and a legend that
covers the data is worse than no legend.
"""

import pathlib

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

ACQUIRE, STORE, RETRIEVE, UPDATE, FORGET = (
    "#4A7FD4", "#2A9D8F", "#E08A2E", "#D05353", "#8F5FB8")
INK, MUTED, GRID = "#1A1A1C", "#4A4A55", "#C8CAD2"
PROBE = {"vioxx": UPDATE, "monarch": RETRIEVE, "twitter": STORE}
ORDER = ["gpt2", "gpt2-medium", "gpt2-large", "Qwen/Qwen2.5-0.5B-Instruct",
         "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Qwen/Qwen2.5-1.5B-Instruct"]
SHORT = ["GPT-2\n124M", "GPT-2 medium\n355M", "GPT-2 large\n774M",
         "Qwen2.5-It\n0.5B", "TinyLlama-Chat\n1.1B", "Qwen2.5-It\n1.5B"]

plt.rcParams.update({
    "font.size": 10.5, "axes.titlesize": 12, "axes.labelsize": 10.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.axisbelow": True,
})

df = pd.read_csv(HERE / "cross_model_dsync.csv")
df["order"] = df.model.map({m: i for i, m in enumerate(ORDER)})
df = df.sort_values(["order", "probe"])
XS = range(len(ORDER))
W = 0.26


def finish(fig, ax, name, ncol=3, title=None, subtitle=None):
    """One title, one optional subtitle, one legend under the axes."""
    if title:
        ax.set_title(title, fontweight="bold", pad=28 if subtitle else 10)
    if subtitle:
        ax.text(0.5, 1.022, subtitle, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=9.5, color=MUTED)
    ax.grid(alpha=0.35, linewidth=0.7)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, frameon=False, ncol=ncol, fontsize=9.5,
                  loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def by_model(probe, column):
    return df[df.probe == probe].set_index("model").reindex(ORDER)[column]


# --- 1. does scale or tuning resolve it? ------------------------------------
fig, ax = plt.subplots(figsize=(7.6, 4.8))
for probe in PROBE:
    g = df[df.probe == probe].sort_values("params_M")
    ax.plot(g.params_M, g.D_sync_nats, marker="o", markersize=7.5, linewidth=2.2,
            color=PROBE[probe], label=probe, zorder=3)
ax.axhline(9.2, color=INK, linestyle=(0, (5, 4)), linewidth=1.1, alpha=0.55, zorder=2)
ax.text(0.985, 9.2, " 9.2 nats ", transform=ax.get_yaxis_transform(),
        ha="right", va="bottom", fontsize=9, color=MUTED,
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
ax.set_xscale("log")
# Ticks at the six sizes the line passes through. A bare log axis labels decades,
# and the only decade inside this range is 1000, which names none of the models.
ax.set_xticks(sorted(df.params_M.unique()))
ax.set_xticklabels([f"{int(v)}" for v in sorted(df.params_M.unique())], fontsize=9)
ax.tick_params(axis="x", which="minor", bottom=False)
ax.set_xlabel("parameters (millions, log scale)")
ax.set_ylabel(r"$\mathcal{D}_{sync}$ (nats)")
ax.set_ylim(bottom=0)
finish(fig, ax, "scale.png",
       title="Does scale or instruction tuning resolve the conflict?",
       subtitle="lower is better; above 9.2 nats the answer is below one chance in ten thousand")

# --- 2. help taken against help available -----------------------------------
fig, ax = plt.subplots(figsize=(8.4, 4.8))
for i, probe in enumerate(PROBE):
    off = (i - 1) * W
    ax.bar([x + off for x in XS], by_model(probe, "help_available_nats"), W,
           color=PROBE[probe], alpha=0.28, zorder=2,
           label="available if the answer is stated outright" if i == 0 else None)
    ax.bar([x + off for x in XS], by_model(probe, "help_nats"), W * 0.56,
           color=PROBE[probe], zorder=3, label=probe)
ax.axhline(0, color=INK, linewidth=1, zorder=4)
ax.set_xticks(list(XS)); ax.set_xticklabels(SHORT, fontsize=8.5)
ax.set_ylabel("nats moved toward the correct answer")
finish(fig, ax, "help.png", ncol=2,
       title="How much of the available help does the document actually deliver?",
       subtitle="solid: the corrective document.  pale: the same fact stated outright")

# --- 3. the exposure trap ----------------------------------------------------
fig, ax = plt.subplots(figsize=(8.4, 4.8))
for i, probe in enumerate(PROBE):
    ax.bar([x + (i - 1) * W for x in XS], by_model(probe, "stale_boost_nats"), W,
           color=PROBE[probe], label=probe, zorder=3)
ax.axhline(0, color=INK, linewidth=1.2, zorder=4)
ax.set_xticks(list(XS)); ax.set_xticklabels(SHORT, fontsize=8.5)
ax.set_ylabel("nats moved toward the stale answer")
# Symlog, linear within one nat of zero and logarithmic outside it. The trap is
# the positive side, and every positive value here is under 0.8 nats against a
# negative bar near -6, so a linear axis renders the finding a few pixels tall.
ax.set_yscale("symlog", linthresh=1.0, linscale=1.4)
ax.set_yticks([-6, -4, -2, -1, -0.5, 0, 0.5, 1])
ax.set_yticklabels(["-6", "-4", "-2", "-1", "-0.5", "0", "0.5", "1"], fontsize=9)
ax.axhspan(0, 1, color=UPDATE, alpha=0.05, zorder=1)
finish(fig, ax, "exposure.png",
       title="The exposure trap",
       subtitle="above zero, the corrective document made the wrong answer more likely")

# --- 4. this document, or any document? -------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 5.4))
lim = max(df.I_ctx_nats.max(), df.I_irrelevant_nats.max()) * 1.10
ax.fill_between([0, lim], [0, lim], [lim, lim], color=STORE, alpha=0.05, zorder=1)
ax.plot([0, lim], [0, lim], color=MUTED, linestyle=(0, (5, 4)), linewidth=1.1, zorder=2)
for probe in PROBE:
    g = df[df.probe == probe]
    ax.scatter(g.I_irrelevant_nats, g.I_ctx_nats, s=120, color=PROBE[probe],
               edgecolor=INK, linewidth=0.7, label=probe, zorder=3)
ax.set_xscale("symlog", linthresh=0.05); ax.set_yscale("symlog", linthresh=0.05)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel(r"$\mathcal{I}$ from an irrelevant document (nats, log scale)")
ax.set_ylabel(r"$\mathcal{I}_{ctx}$ from the corrective document (nats, log scale)")
finish(fig, ax, "control.png",
       title="Was it this document, or would any document have done?",
       subtitle="above the line the correction moved the model more; below it, a Danube article did")

print(f"\n  four figures written to {OUT}")
