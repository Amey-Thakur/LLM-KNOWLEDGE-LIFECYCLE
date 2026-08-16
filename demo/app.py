"""LLM Knowledge Lifecycle: interactive demonstration.

Companion demo for the paper "The Knowledge Lifecycle of Large Language
Models" (Thakur and Talele, 2026). Implements the paper's measurement protocol
(Algorithm 2) live on GPT-2 base: given a query, a corrective context, and the
verified correct continuation, it reports

  - D_sync : -ln P(answer | context + query), the surprisal of the correct
             answer with the corrective document present
  - I_ctx  : KL(P_with_context || P_without_context) over the full vocabulary,
             how far the document moves the model's distribution at all
  - the top next-token distributions under both conditions

Deterministic: no sampling anywhere, so identical inputs give identical
numbers on every run. Runs on the free CPU tier.
"""

import math

import gradio as gr
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Each preset is a documented, verifiable fact change. All three were measured
# before shipping; the expected readings below quote those measurements.
PRESETS = {
    "Vioxx withdrawal (2004): resolution failure": {
        "query": "Question: Is Vioxx safe to prescribe? Answer: Vioxx is considered",
        "context": "Context: In September 2004, Merck voluntarily withdrew Vioxx after trials revealed increased cardiovascular risks.",
        "answer": " withdrawn",
        "reading": (
            "The paper's headline case. The withdrawal notice is in the prompt, yet the "
            "model answers \"safe\" with higher probability than without it. D_sync lands "
            "near 12 nats and I_ctx near 0.03: the context is present but inert."
        ),
    },
    "British monarch (2022): the exposure trap": {
        "query": "Question: Who is the current British monarch? Answer: The current British monarch is",
        "context": "Context: Queen Elizabeth II died in September 2022. Charles III acceded to the throne and is the reigning King of the United Kingdom.",
        "answer": " Charles",
        "reading": (
            "A subtler failure. The context raises P(Charles), but its top effect is to "
            "boost \" Queen\": merely mentioning the late monarch reinforces the stale "
            "association. Correct information can strengthen the wrong answer."
        ),
    },
    "Twitter rename (2023): drift under strong context": {
        "query": "Question: What is the social network Twitter called today? Answer: Twitter is now called",
        "context": "Context: In July 2023, Twitter was rebranded as X under Elon Musk's ownership.",
        "answer": " X",
        "reading": (
            "The context moves the distribution hard (I_ctx near 0.9 nats) and lifts the "
            "correct answer from 0.03% to roughly 12%. The model still answers "
            "\"Twitter\". Influence without resolution."
        ),
    },
}

LIFECYCLE_MD = """
## The Knowledge Lifecycle

A language model's relationship with a fact passes through five stages. Each stage has its
own research subfield, its own failure mode, and its own benchmark; the failures this demo
measures live at the boundaries between them.

| Stage | What happens | Characteristic failure |
| :--- | :--- | :--- |
| **1. Acquisition** | Pre-training compresses a corpus into weights; fine-tuning and in-context learning add more. | Facts are stored without any record of when or from where they were learned. |
| **2. Storage** | Facts live in feed-forward layers (parametric), in external indices (non-parametric), or both. | The two copies of a fact age independently. |
| **3. Retrieval** | Attention recalls parametric facts; RAG pipelines fetch external documents into context. | Retrieval succeeds, yet what was retrieved may not control the output. |
| **4. Update** | The world changes: weights are edited, indices are refreshed, or conflicts are resolved at inference. | Weight editing damages neighboring facts; index updates leave weights stale. |
| **5. Forgetting** | Unlearning removes facts deliberately; catastrophic forgetting and context eviction remove them by accident. | Removed facts remain recoverable; retained facts get damaged. |

### What this demo measures

When a fact changed after the model's training data was collected, the weights hold the old
version and a retrieved document holds the new one. The probe puts both in front of the model
and measures which one wins:

- **D_sync** is the surprisal of the correct answer with the corrective document present.
  A value of *n* nats means the model assigns the correct answer probability e^(-n).
- **I_ctx** is how far the document moves the model's whole output distribution.
  Near zero means the document is being ignored entirely.

The pairing separates two very different failures. High D_sync with near-zero I_ctx means
the document is inert: retrieval worked and resolution failed. High D_sync with large I_ctx
means the document is fighting the parametric prior and losing.

### The three presets, three regimes

Each preset is a documented, real fact change that postdates GPT-2's training data. Measured
on this model, they land in three distinct regimes: total resolution failure (Vioxx), the
exposure trap where correct context reinforces the wrong answer (monarch), and drift where
strong context influence still fails to flip the answer (Twitter). Run them, then build your
own: any fact change with a single-word answer works.

### Why a 124M model from 2019

GPT-2 base is small, fully open, and free of instruction-tuning confounds, which makes every
number here exactly reproducible on any machine. The point is not that GPT-2 is weak; it is
that the failure mode exists, is measurable, and decomposes cleanly. How these numbers fall
with scale, instruction tuning, and reasoning-mode inference is the paper's proposed
benchmark direction.
"""

NUMBERS_MD = """
## Reading the numbers

**D_sync** is a negative log probability, in nats. It converts directly:

| D_sync (nats) | P(correct answer) | Reading |
| ---: | ---: | :--- |
| 0.7 | 50% | Synchronized: the correct answer holds half the probability mass |
| 2.3 | 10% | Drift: correct answer present but no longer dominant |
| 4.6 | 1% | Severe drift |
| 9.2 | 0.01% | Failure threshold: no realistic decoding recovers the answer |
| 12.0 | 0.0006% | The paper's Vioxx measurement |

**I_ctx** is the KL divergence between the model's next-token distributions with and without
the corrective document, computed over the full vocabulary. It answers a different question:
did the document change the model's mind about anything?

| I_ctx (nats) | Reading |
| ---: | :--- |
| < 0.05 | Document is inert; the model's distribution barely moved |
| 0.05 to 0.5 | Document has partial influence |
| > 0.5 | Document is substantially reshaping the distribution |

**The diagnosis grid.** The two numbers together localize the failure:

| | Low I_ctx | High I_ctx |
| :--- | :--- | :--- |
| **Low D_sync** | Fact never conflicted | Context resolved the conflict |
| **High D_sync** | Resolution failure: context ignored | Drift: context influential but losing |

**Caveats.** The estimator assumes the correct answer is a single verifiable continuation
and measures its first token. Probabilities are for the next token only, under deterministic
evaluation. The measurement characterizes this model and probe, not language models in
general; that is what makes it honest.
"""

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()


def distribution(prompt):
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt")
        logits = model(**inputs).logits[0, -1, :]
    return F.softmax(logits, dim=-1)


def top_rows(probs, k=8):
    top = torch.topk(probs, k)
    return [[repr(tokenizer.decode([i]))[1:-1], f"{p.item()*100:.4f}%"]
            for p, i in zip(top.values, top.indices)]


def load_preset(name):
    p = PRESETS[name]
    return p["query"], p["context"], p["answer"], p["reading"]


def probe(query, context, answer):
    query = query.strip()
    context = context.strip()
    answer = answer if answer.startswith(" ") else " " + answer.strip()
    if not query or not context or not answer.strip():
        return "Enter a query, a corrective context, and the correct continuation.", [], []

    p_plain = distribution(query)
    p_ctx = distribution(context + "\n" + query)

    ids = tokenizer.encode(answer)
    pieces = [tokenizer.decode([i]) for i in ids]
    p_target_ctx = max(p_ctx[ids[0]].item(), 1e-12)
    p_target_plain = max(p_plain[ids[0]].item(), 1e-12)

    dsync = -math.log(p_target_ctx)
    i_ctx = F.kl_div(p_plain.log(), p_ctx, reduction="sum").item()

    if dsync > 9.2:
        row = "resolution failure" if i_ctx < 0.05 else "drift: context influential but losing"
        verdict = f"{row}; no realistic decoding recovers the correct answer"
    elif dsync > 4.6:
        verdict = "severe drift: correct answer below 1% probability"
    elif dsync > 0.7:
        verdict = "drift: correct answer no longer holds most of the probability mass"
    else:
        verdict = "synchronized: correct answer holds at least half the probability mass"

    report = (
        f"Answer tokenization: {pieces} ({len(ids)} piece(s); first piece measured)\n\n"
        f"P(answer | query alone)           = {p_target_plain:.2e}\n"
        f"P(answer | context + query)       = {p_target_ctx:.2e}\n\n"
        f"D_sync = -ln P(answer | context + query) = {dsync:.2f} nats\n"
        f"I_ctx  = KL(with-context || without)     = {i_ctx:.4f} nats\n\n"
        f"Reading: {verdict}."
    )
    return report, top_rows(p_plain), top_rows(p_ctx)


CITATION = """```bibtex
@article{thakur2026lifecycle,
  author  = {Thakur, Amey and Talele, Sarvesh},
  title   = {The Knowledge Lifecycle of Large Language Models},
  journal = {arXiv preprint},
  year    = {2026}
}
```"""

with gr.Blocks(theme=gr.themes.Base(), title="LLM Knowledge Lifecycle") as demo:
    gr.Markdown(
        "# LLM Knowledge Lifecycle\n"
        "Live measurement of what happens when a language model's parametric memory "
        "conflicts with corrective context. Companion demo for "
        "*The Knowledge Lifecycle of Large Language Models* (Thakur and Talele, 2026). "
        "Model: GPT-2 base, deterministic evaluation, exactly reproducible."
    )

    with gr.Tab("Probe"):
        preset = gr.Dropdown(
            choices=list(PRESETS.keys()),
            value=list(PRESETS.keys())[0],
            label="Preset: documented fact changes, each landing in a different regime",
        )
        reading = gr.Textbox(
            value=PRESETS[list(PRESETS.keys())[0]]["reading"],
            label="What to expect", interactive=False, lines=3,
        )
        query = gr.Textbox(value=PRESETS[list(PRESETS.keys())[0]]["query"],
                           label="Query (ends mid-sentence; the next token is the answer)")
        context = gr.Textbox(value=PRESETS[list(PRESETS.keys())[0]]["context"],
                             label="Corrective context (contradicts the model's parametric memory)")
        answer = gr.Textbox(value=PRESETS[list(PRESETS.keys())[0]]["answer"],
                            label="Correct continuation (single word)")
        run = gr.Button("Measure", variant="primary")
        report = gr.Textbox(label="Measurement", lines=11)
        with gr.Row():
            t0 = gr.Dataframe(headers=["token", "probability"], label="Top tokens, query alone")
            t1 = gr.Dataframe(headers=["token", "probability"], label="Top tokens, context + query")

        preset.change(load_preset, inputs=preset, outputs=[query, context, answer, reading])
        run.click(probe, inputs=[query, context, answer], outputs=[report, t0, t1])

    with gr.Tab("The Five Stages"):
        gr.Markdown(LIFECYCLE_MD)

    with gr.Tab("Reading the Numbers"):
        gr.Markdown(NUMBERS_MD)

    with gr.Accordion("Cite this work", open=False):
        gr.Markdown(CITATION)

    gr.Markdown(
        "---\nPaper repository: "
        "[Amey-Thakur/LLM-KNOWLEDGE-LIFECYCLE](https://github.com/Amey-Thakur/LLM-KNOWLEDGE-LIFECYCLE) "
        "· Authors: [Amey Thakur](https://orcid.org/0000-0001-5644-1575) and [Sarvesh Talele](https://orcid.org/0009-0002-0818-461X)"
    )

if __name__ == "__main__":
    demo.launch()
