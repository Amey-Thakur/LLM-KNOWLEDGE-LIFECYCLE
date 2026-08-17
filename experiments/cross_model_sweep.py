"""
===============================================================================
FILE         : cross_model_sweep.py
PROJECT      : The Knowledge Lifecycle of Large Language Models
PURPOSE      : The cross-model sweep behind the Kaggle notebook, as a script,
               so the same measurements can be run and re-run outside a kernel.
TECH STACK   : Python 3, PyTorch, Hugging Face Transformers
AUTHORS      : Amey Thakur (https://github.com/Amey-Thakur)
               Sarvesh Talele (https://github.com/sarveshtalele)
REPOSITORY   : https://github.com/Amey-Thakur/LLM-KNOWLEDGE-LIFECYCLE
LICENSE      : CC BY 4.0
===============================================================================

Runs three documented fact changes against a ladder of models and records, for
each pair, how far the correct answer sits from the model's output once the
correcting document is already in the prompt.

Three things here are deliberate and differ from a first draft of this sweep.

Every model is loaded in float32. Casting logits to float32 after a half
precision forward pass does not recover the precision already lost, so a sweep
that lets dtype follow the hardware is not comparable across machines and is
not the deterministic measurement this paper claims to report.

The probability floor is recorded rather than applied silently. A floor turns an
unrepresentable probability into a finite surprisal, and a censored value that
is printed like a measurement is worse than a missing one.

The first token of the answer is what gets measured, following the paper's
estimator. Where a tokenizer splits the answer into more than one piece, the
probability of the first piece is an upper bound on the probability of the whole
answer, so the surprisal is a lower bound. The piece count is carried in the
output so that every such row can be read as the bound it is.
"""

import argparse
import gc
import json
import math
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# The smallest probability float32 can carry away from zero. A measured value at
# or below this is reported as censored rather than as a number.
FLOOR = 1e-12

PROBES = [
    {
        "name": "vioxx",
        "query": "Question: Is Vioxx safe to prescribe? Answer: Vioxx is considered",
        "context": "Context: In September 2004, Merck voluntarily withdrew Vioxx after trials revealed increased cardiovascular risks.",
        "answer": " withdrawn",
    },
    {
        "name": "monarch",
        "query": "Question: Who is the current British monarch? Answer: The current British monarch is",
        "context": "Context: Queen Elizabeth II died in September 2022. Charles III acceded to the throne and is the reigning King of the United Kingdom.",
        "answer": " Charles",
    },
    {
        "name": "twitter",
        "query": "Question: What is the social network Twitter called today? Answer: Twitter is now called",
        "context": "Context: In July 2023, Twitter was rebranded as X under Elon Musk's ownership.",
        "answer": " X",
    },
]

MODELS = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen2.5-1.5B-Instruct",
]


def next_token_distribution(model, tokenizer, prompt):
    """Probability of every vocabulary item as the next token."""
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        logits = model(**inputs).logits[0, -1, :].float()
    return F.softmax(logits, dim=-1)


def run_probe(model, tokenizer, probe):
    """One probe against one model: both conditions, and the two metrics."""
    p_plain = next_token_distribution(model, tokenizer, probe["query"])
    p_ctx = next_token_distribution(
        model, tokenizer, probe["context"] + "\n" + probe["query"])

    ids = tokenizer.encode(probe["answer"], add_special_tokens=False)
    first_piece = tokenizer.decode([ids[0]])

    raw_plain = p_plain[ids[0]].item()
    raw_ctx = p_ctx[ids[0]].item()
    censored = raw_ctx <= FLOOR

    top_ctx = tokenizer.decode([int(torch.argmax(p_ctx))])
    return {
        "probe": probe["name"],
        "answer_first_piece": first_piece,
        "answer_pieces": len(ids),
        "P_answer_plain": raw_plain,
        "P_answer_ctx": raw_ctx,
        "D_sync_nats": -math.log(max(raw_ctx, FLOOR)),
        "D_sync_censored": censored,
        "D_sync_is_lower_bound": len(ids) > 1,
        "I_ctx_nats": F.kl_div(p_plain.log(), p_ctx, reduction="sum").item(),
        "top_token_plain": tokenizer.decode([int(torch.argmax(p_plain))]),
        "top_token_ctx": top_ctx,
        "correct_top1": top_ctx == first_piece,
        "context_helped": raw_ctx > raw_plain,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--out", default="cross_model_dsync.csv")
    args = ap.parse_args()

    rows = []
    for name in args.models:
        started = time.time()
        print(f"\n{name}", flush=True)
        try:
            tokenizer = AutoTokenizer.from_pretrained(name)
            # float32 everywhere, on every device, so the digits do not depend
            # on the hardware the sweep happened to run on.
            model = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=torch.float32)
            model.eval()
        except Exception as error:
            print(f"   could not load: {error}", flush=True)
            continue

        size_m = round(sum(p.numel() for p in model.parameters()) / 1e6)
        for probe in PROBES:
            row = {"model": name, "params_M": size_m}
            row.update(run_probe(model, tokenizer, probe))
            rows.append(row)
            flag = "  CENSORED" if row["D_sync_censored"] else ""
            print(f"   {row['probe']:<9} D_sync {row['D_sync_nats']:7.3f}   "
                  f"I_ctx {row['I_ctx_nats']:7.4f}   "
                  f"top {row['top_token_ctx']!r:<14} "
                  f"correct {str(row['correct_top1']):<5}{flag}", flush=True)

        del model
        gc.collect()
        print(f"   {size_m}M params, {time.time()-started:.0f}s", flush=True)

        # Written after every model, so an interrupted sweep still leaves the
        # measurements it already made.
        with open(args.out.replace(".csv", ".json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)

    header = list(rows[0].keys())
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(r[h]) for h in header) + "\n")
    print(f"\n{len(rows)} measurements written to {args.out}")


if __name__ == "__main__":
    main()
