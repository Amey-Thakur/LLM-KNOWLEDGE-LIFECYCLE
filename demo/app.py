"""Lifecycle Desynchronization probe.

Interactive demonstration for the paper "The Knowledge Lifecycle of Large
Language Models". Implements the measurement protocol (Algorithm 2): given a
query, a corrective context, and the verified correct continuation, it reports

  - the point-mass estimator  D_sync = -ln P(answer | context + query)
  - the context influence     I_ctx  = KL(P_with_context || P_without_context)
  - the top next-token distributions under both conditions

on GPT-2 base (124M). Deterministic: no sampling anywhere, so every run of the
same inputs gives identical numbers. Runs on the free CPU tier.
"""

import math

import gradio as gr
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

DEFAULT_QUERY = "Question: Is Vioxx safe to prescribe? Answer: Vioxx is considered"
DEFAULT_CONTEXT = (
    "Context: In September 2004, Merck voluntarily withdrew Vioxx after "
    "trials revealed increased cardiovascular risks."
)
DEFAULT_ANSWER = " withdrawn"

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
    return [[tokenizer.decode([i]), f"{p.item()*100:.4f}%"] for p, i in zip(top.values, top.indices)]


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
    p_target = max(p_ctx[ids[0]].item(), 1e-12)

    dsync = -math.log(p_target)
    i_ctx = F.kl_div(p_plain.log(), p_ctx, reduction="sum").item()

    if dsync > 9.2:
        verdict = "desynchronization failure: no realistic decoding recovers the correct answer"
    elif dsync > 4.6:
        verdict = "severe drift: correct answer below 1% probability"
    elif dsync > 0.7:
        verdict = "drift: correct answer no longer holds most of the probability mass"
    else:
        verdict = "synchronized: correct answer holds at least half the probability mass"

    report = (
        f"Answer tokenization: {pieces} ({len(ids)} piece(s); first piece measured)\n\n"
        f"P(answer | context + query) = {p_target:.2e}\n"
        f"D_sync  = -ln P = {dsync:.2f} nats\n"
        f"I_ctx   = KL(with-context || without-context) = {i_ctx:.4f} nats\n\n"
        f"Reading: {verdict}.\n"
        f"I_ctx near zero with high D_sync means the context is present but inert: "
        f"the failure sits at conflict resolution, not retrieval."
    )
    return report, top_rows(p_plain), top_rows(p_ctx)


demo = gr.Interface(
    fn=probe,
    inputs=[
        gr.Textbox(value=DEFAULT_QUERY, label="Query (ends mid-sentence, next token is the answer)"),
        gr.Textbox(value=DEFAULT_CONTEXT, label="Corrective context (contradicts the model's parametric memory)"),
        gr.Textbox(value=DEFAULT_ANSWER, label="Correct continuation (single word)"),
    ],
    outputs=[
        gr.Textbox(label="Measurement", lines=10),
        gr.Dataframe(headers=["token", "probability"], label="Top tokens without context"),
        gr.Dataframe(headers=["token", "probability"], label="Top tokens with context"),
    ],
    title="Lifecycle Desynchronization Probe",
    description=(
        "Companion demo for *The Knowledge Lifecycle of Large Language Models* "
        "(Amey Thakur, 2026). Measures how far a model's parametric memory overrides "
        "corrective context, on GPT-2 base. The default inputs reproduce the paper's "
        "Vioxx measurement: D_sync = 12.05 nats, I_ctx = 0.033 nats."
    ),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
