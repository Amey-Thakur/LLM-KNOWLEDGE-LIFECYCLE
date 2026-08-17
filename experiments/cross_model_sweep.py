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

Runs documented fact changes against a ladder of models and records, for each
pair, how far the correct answer sits from the model's output once the
correcting document is already in the prompt.

Answer tokens are obtained by differencing the prompt against the prompt plus
the answer, not by encoding the answer alone. A SentencePiece vocabulary turns
the leading space of " Charles" and " Queen" into the same standalone piece, so
encoding them separately scores the identical token and every comparison between
them collapses to zero. Whether that happens depends on the installed tokenizer
build, which is what makes it dangerous. On one build TinyLlama's Twitter probe
scores 8.62 nats and the answer is marked wrong; on another the same code gives
0.17 nats and marks it right.

Surprisal is summed over the whole answer under teacher forcing rather than read
off its first token. Scoring the first token is well defined on GPT-2, where
each answer here costs exactly one token, and misleading on a vocabulary that
splits "withdrawn" into "with", "dra", "wn": the first piece spells a common
English word and its probability says little about the word. Summing costs one
extra forward pass and removes the tokenizer from the comparison.

Two controls accompany every probe. An irrelevant
document of matched length and register separates "this document moved the
model" from "any document moves the model". A leak document that states the
answer outright gives the upper bound the model is capable of when the answer is
handed to it.

Every model is loaded in float32 on every device. Casting logits to float32
after a half precision forward pass does not recover the precision already lost,
so a sweep that lets dtype follow the hardware is not comparable across machines.
"""

import argparse
import csv
import gc
import json
import math
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

LOG_FLOOR = -80.0

PROBES = [
    {
        "name": "vioxx",
        "query": "Question: Is Vioxx safe to prescribe? Answer: Vioxx is considered",
        "context": "Context: In September 2004, Merck voluntarily withdrew Vioxx after trials revealed increased cardiovascular risks.",
        "leak": "Context: Vioxx is considered withdrawn.",
        "answer": " withdrawn",
        "stale": " safe",
    },
    {
        "name": "monarch",
        "query": "Question: Who is the current British monarch? Answer: The current British monarch is",
        "context": "Context: Queen Elizabeth II died in September 2022. Charles III acceded to the throne and is the reigning King of the United Kingdom.",
        "leak": "Context: The current British monarch is Charles.",
        "answer": " Charles",
        "stale": " Queen",
    },
    {
        "name": "twitter",
        "query": "Question: What is the social network Twitter called today? Answer: Twitter is now called",
        "context": "Context: In July 2023, Twitter was rebranded as X under Elon Musk's ownership.",
        "leak": "Context: Twitter is now called X.",
        "answer": " X",
        "stale": " Twitter",
    },
]

# Matched in register and length to the corrective documents, so that any
# divergence it produces is attributable to the presence of a document rather
# than to its subject.
IRRELEVANT = ("Context: The Danube is the second longest river in Europe and flows "
              "through ten countries before reaching the Black Sea.")

MODELS = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen2.5-1.5B-Instruct",
]


# --- the estimator -----------------------------------------------------------
def log_next_token(model, tokenizer, prompt):
    """Log probability of every vocabulary item as the next token."""
    with torch.no_grad():
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        logits = model(**encoded).logits[0, -1, :].float()
    return F.log_softmax(logits, dim=-1)


def kl_nats(log_p, log_q):
    """KL(p || q) in nats, in log space.

    torch.where rather than a mask, because 0 * (-inf) is nan and would silently
    poison the sum for any vocabulary item a model rules out entirely.
    """
    p = log_p.exp()
    return torch.where(p > 0, p * (log_p - log_q), torch.zeros_like(p)).sum().item()


def answer_token_ids(tokenizer, prompt, answer):
    """The tokens the answer occupies when it follows this exact prompt."""
    base = tokenizer(prompt, add_special_tokens=False).input_ids
    full = tokenizer(prompt + answer, add_special_tokens=False).input_ids
    shared = 0
    while shared < len(base) and shared < len(full) and base[shared] == full[shared]:
        shared += 1
    if shared != len(base):
        raise ValueError(f"tokenizer merged the boundary before {answer!r}")
    return full[len(base):]


def sequence_surprisal(model, tokenizer, prompt, answer_ids):
    """-ln P(answer | prompt), summed over every token the answer costs."""
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    ans = torch.tensor([answer_ids])
    ids = torch.cat([prompt_ids, ans], dim=1).to(model.device)
    with torch.no_grad():
        logits = model(ids).logits[0].float()
    n = ans.shape[1]
    # Row i predicts token i+1, so the rows predicting the answer are the n rows
    # ending one before the last.
    log_probs = F.log_softmax(logits[-n - 1:-1, :], dim=-1)
    return -log_probs.gather(1, ans[0].unsqueeze(1).to(model.device)).sum().item()


def run_probe(model, tokenizer, probe):
    """One probe against one model: both conditions, both controls, the metrics."""
    q = probe["query"]
    p_ctx = probe["context"] + "\n" + q
    p_irr = IRRELEVANT + "\n" + q
    p_leak = probe["leak"] + "\n" + q

    ans_plain = answer_token_ids(tokenizer, q, probe["answer"])
    ans_ctx = answer_token_ids(tokenizer, p_ctx, probe["answer"])
    stale_plain = answer_token_ids(tokenizer, q, probe["stale"])
    stale_ctx = answer_token_ids(tokenizer, p_ctx, probe["stale"])

    d_plain = sequence_surprisal(model, tokenizer, q, ans_plain)
    d_ctx = sequence_surprisal(model, tokenizer, p_ctx, ans_ctx)
    d_irr = sequence_surprisal(model, tokenizer, p_irr,
                               answer_token_ids(tokenizer, p_irr, probe["answer"]))
    d_leak = sequence_surprisal(model, tokenizer, p_leak,
                                answer_token_ids(tokenizer, p_leak, probe["answer"]))

    s_plain = sequence_surprisal(model, tokenizer, q, stale_plain)
    s_ctx = sequence_surprisal(model, tokenizer, p_ctx, stale_ctx)

    lg_plain = log_next_token(model, tokenizer, q)
    lg_ctx = log_next_token(model, tokenizer, p_ctx)
    lg_irr = log_next_token(model, tokenizer, p_irr)

    top_ctx = tokenizer.decode([int(torch.argmax(lg_ctx))])
    first_ctx = tokenizer.decode([ans_ctx[0]])
    return {
        "probe": probe["name"],
        "answer_tokens": len(ans_ctx),
        "answer_first_piece": first_ctx,
        "D_sync_plain": d_plain,
        "D_sync_nats": d_ctx,
        "D_sync_irrelevant": d_irr,
        "D_sync_leak": d_leak,
        # How much the document moved the correct answer, in nats. Positive is
        # help. The leak column says how much help was available to give.
        "help_nats": d_plain - d_ctx,
        "help_available_nats": d_plain - d_leak,
        # The exposure trap: positive when the corrective document made the
        # stale answer more likely, which is the opposite of its purpose.
        "stale_boost_nats": s_plain - s_ctx,
        "I_ctx_nats": kl_nats(lg_ctx, lg_plain),
        "I_irrelevant_nats": kl_nats(lg_irr, lg_plain),
        "top_token_ctx": top_ctx,
        "correct_top1": top_ctx == first_ctx,
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
            model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
            model.eval()
        except Exception as error:
            print(f"   could not load: {error}", flush=True)
            continue

        size_m = round(sum(p.numel() for p in model.parameters()) / 1e6)
        for probe in PROBES:
            row = {"model": name, "params_M": size_m}
            row.update(run_probe(model, tokenizer, probe))
            rows.append(row)
            print(f"   {row['probe']:<9} D_sync {row['D_sync_nats']:7.3f}  "
                  f"I_ctx {row['I_ctx_nats']:6.3f}  help {row['help_nats']:+6.2f}  "
                  f"stale {row['stale_boost_nats']:+6.2f}  "
                  f"top {row['top_token_ctx']!r:<12} correct {row['correct_top1']}",
                  flush=True)

        del model
        gc.collect()
        print(f"   {size_m}M params, {time.time() - started:.0f}s", flush=True)
        with open(args.out.replace(".csv", ".json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(args.out, encoding="utf-8", newline="") as f:
        back = list(csv.DictReader(f))
    assert len(back) == len(rows), f"wrote {len(rows)} but file parses as {len(back)}"
    print(f"\n{len(rows)} measurements written to {args.out}, read back intact")


if __name__ == "__main__":
    main()
