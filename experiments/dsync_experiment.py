"""
===============================================================================
FILE         : dsync_experiment.py
PROJECT      : The Knowledge Lifecycle of Large Language Models
PURPOSE      : The paper's measurement: the Vioxx conflict probe on GPT-2 base,
               producing every number reported in Section 7.7.
TECH STACK   : Python 3, PyTorch, Hugging Face Transformers
AUTHORS      : Amey Thakur (https://github.com/Amey-Thakur)
               Sarvesh Talele (https://github.com/sarveshtalele)
REPOSITORY   : https://github.com/Amey-Thakur/LLM-KNOWLEDGE-LIFECYCLE
RELEASE DATE : August 16, 2026
LICENSE      : CC BY 4.0
===============================================================================

Runs the same query under two conditions, with and without the retrieval
context that contradicts the model's parametric knowledge, and reports
everything the paper needs: top next-token probabilities in both conditions,
the probability of the correct and incorrect continuations, the full-vocabulary
KL divergence between the two conditions, and the surprisal of the correct
token under the context condition.

Deterministic: no sampling anywhere, so every run gives identical numbers.
"""

import math

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

QUESTION = "Question: Is Vioxx safe to prescribe? Answer: Vioxx is considered"
CONTEXT = (
    "Context: In September 2004, Merck voluntarily withdrew Vioxx after "
    "trials revealed increased cardiovascular risks.\n"
)


def next_token_distribution(model, tokenizer, prompt):
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt")
        logits = model(**inputs).logits[0, -1, :]
    return F.softmax(logits, dim=-1)


def show_top(probs, tokenizer, k=10, label=""):
    top = torch.topk(probs, k)
    print(f"\n  top-{k} next tokens {label}:")
    for p, idx in zip(top.values, top.indices):
        print(f"    {tokenizer.decode([idx]):<16} {p.item()*100:8.4f}%")


def main():
    print("Loading GPT-2 base (124M)...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    p_no_ctx = next_token_distribution(model, tokenizer, QUESTION)
    p_ctx = next_token_distribution(model, tokenizer, CONTEXT + QUESTION)

    show_top(p_no_ctx, tokenizer, label="WITHOUT context")
    show_top(p_ctx, tokenizer, label="WITH withdrawal context")

    # Token ids for the continuations of interest. Report the tokenization
    # explicitly so a multi-piece split cannot silently distort the numbers.
    for word in [" safe", " withdrawn", " unsafe", " dangerous"]:
        ids = tokenizer.encode(word)
        pieces = [tokenizer.decode([i]) for i in ids]
        p0 = p_no_ctx[ids[0]].item()
        p1 = p_ctx[ids[0]].item()
        print(f"\n  {word!r} tokenizes to {pieces} ({len(ids)} piece(s))")
        print(f"    P(first piece | no context)   = {p0:.6f}  ({p0*100:.4f}%)")
        print(f"    P(first piece | with context) = {p1:.6f}  ({p1*100:.4f}%)")

    # Full-vocabulary KL between the two conditions, both directions.
    kl_ctx_noctx = F.kl_div(p_no_ctx.log(), p_ctx, reduction="sum").item()
    kl_noctx_ctx = F.kl_div(p_ctx.log(), p_no_ctx, reduction="sum").item()
    print(f"\n  KL(P_ctx || P_noctx) over full vocab = {kl_ctx_noctx:.4f} nats")
    print(f"  KL(P_noctx || P_ctx) over full vocab = {kl_noctx_ctx:.4f} nats")

    # Surprisal of the correct continuation under the context condition:
    # the point-mass estimator of D_sync used in the paper.
    wid = tokenizer.encode(" withdrawn")[0]
    p_correct = max(p_ctx[wid].item(), 1e-12)
    print(f"\n  D_sync (point-mass estimator) = -ln P(' withdrawn' | ctx)")
    print(f"                                = -ln({p_correct:.2e}) = {-math.log(p_correct):.4f} nats")


if __name__ == "__main__":
    main()
