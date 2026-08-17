"""
===============================================================================
FILE         : verify_numbers.py
PROJECT      : The Knowledge Lifecycle of Large Language Models
PURPOSE      : Recompute every quantity the manuscript, poster, slides, README
               and demonstration report, at full precision, and print them in
               one place so each can be checked against what is written.
TECH STACK   : Python 3, PyTorch, Hugging Face Transformers
AUTHORS      : Amey Thakur (https://github.com/Amey-Thakur)
               Sarvesh Talele (https://github.com/sarveshtalele)
REPOSITORY   : https://github.com/Amey-Thakur/LLM-KNOWLEDGE-LIFECYCLE
LICENSE      : CC BY 4.0
===============================================================================

Every figure that appears anywhere in this project comes from one forward pass
per condition on GPT-2 base. Rounded values are easy to copy from one document
into another and hard to audit once they are there, so this script prints the
full-precision source of each one next to the derived quantity it feeds.

Deterministic: no sampling, so the digits are identical on any machine.
"""

import json
import math

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

QUESTION = "Question: Is Vioxx safe to prescribe? Answer: Vioxx is considered"
CONTEXT = (
    "Context: In September 2004, Merck voluntarily withdrew Vioxx after "
    "trials revealed increased cardiovascular risks.\n"
)
TRACKED = [" safe", " withdrawn", " unsafe", " dangerous", " effective"]


def distribution(model, tokenizer, prompt):
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt")
        logits = model(**inputs).logits[0, -1, :].float()
    return F.softmax(logits, dim=-1)


def main():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    p_no = distribution(model, tokenizer, QUESTION)
    p_ct = distribution(model, tokenizer, CONTEXT + QUESTION)

    out = {}
    print("token probabilities, full precision")
    print(f"  {'token':<12} {'without context':>18} {'with context':>18}")
    for word in TRACKED:
        ids = tokenizer.encode(word)
        assert len(ids) == 1, f"{word!r} is not a single token: {ids}"
        a, b = p_no[ids[0]].item(), p_ct[ids[0]].item()
        out[word.strip()] = {"no_context": a, "with_context": b}
        print(f"  {word.strip():<12} {a:18.12f} {b:18.12f}")

    safe_no = out["safe"]["no_context"]
    safe_ct = out["safe"]["with_context"]
    wdr_no = out["withdrawn"]["no_context"]
    wdr_ct = out["withdrawn"]["with_context"]

    print("\npercentages as printed in the documents")
    print(f"  P(safe | no context)        {safe_no*100:.4f}%   reported 37.53%")
    print(f"  P(safe | with context)      {safe_ct*100:.4f}%   reported 42.58%")
    print(f"  P(withdrawn | no context)   {wdr_no*100:.6f}%  reported 0.0004%")
    print(f"  P(withdrawn | with context) {wdr_ct*100:.6f}%  reported 0.0006%")

    print("\nderived quantities")
    d_sync = -math.log(wdr_ct)
    d_sync_no = -math.log(wdr_no)
    i_ctx = F.kl_div(p_no.log(), p_ct, reduction="sum").item()
    i_rev = F.kl_div(p_ct.log(), p_no, reduction="sum").item()
    print(f"  D_sync  = -ln P(withdrawn | ctx)      = {d_sync:.6f} nats")
    print(f"  same without context                  = {d_sync_no:.6f} nats")
    print(f"  I_ctx   = KL(P_ctx || P_noctx)        = {i_ctx:.6f} nats")
    print(f"  reverse   KL(P_noctx || P_ctx)        = {i_rev:.6f} nats")
    print(f"  P(safe) rises by                      = "
          f"{(safe_ct-safe_no)*100:.4f} points, a factor of {safe_ct/safe_no:.4f}")

    print("\nratios quoted in the poster and slides")
    print(f"  safe : withdrawn, with context        = {safe_ct/wdr_ct:,.0f} : 1")
    print(f"  safe : withdrawn, no context          = {safe_no/wdr_no:,.0f} : 1")
    print(f"  withdrawn rises by a factor of        = {wdr_ct/wdr_no:.4f}")

    print("\nthe two answers the context should have suppressed")
    for w in ["unsafe", "dangerous"]:
        a, b = out[w]["no_context"], out[w]["with_context"]
        print(f"  P({w:<10}) {a*100:8.4f}% -> {b*100:8.4f}%   "
              f"{'falls' if b < a else 'rises'} by a factor of {a/b:.2f}")

    print("\ntop-5 next tokens, with context")
    top = torch.topk(p_ct, 5)
    for p, i in zip(top.values, top.indices):
        print(f"  {tokenizer.decode([i]):<12} {p.item()*100:8.4f}%")

    out["derived"] = {
        "D_sync_nats": d_sync,
        "I_ctx_nats": i_ctx,
        "safe_to_withdrawn_ratio_with_context": safe_ct / wdr_ct,
        "safe_to_withdrawn_ratio_no_context": safe_no / wdr_no,
    }
    with open("verified_numbers.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nwritten to verified_numbers.json")


if __name__ == "__main__":
    main()
