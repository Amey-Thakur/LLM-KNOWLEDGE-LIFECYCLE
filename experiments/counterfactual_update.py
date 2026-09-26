"""
===============================================================================
FILE         : counterfactual_update.py
PROJECT      : The Knowledge Lifecycle of Large Language Models
PURPOSE      : Measure whether a corrective document actually changes the
               answer a model gives, recording top-1 and rank rather than
               surprisal alone.
TECH STACK   : Python 3, PyTorch, Hugging Face Transformers
AUTHORS      : Amey Thakur (https://github.com/Amey-Thakur)
               Sarvesh Talele (https://github.com/sarveshtalele)
REPOSITORY   : https://github.com/Amey-Thakur/LLM-KNOWLEDGE-LIFECYCLE
LICENSE      : CC BY 4.0
===============================================================================

Why this replaces paraphrase_robustness.py.

That script measured the surprisal of the in-context answer with the document
present, and nothing else. It never recorded which token the model actually
ranked first, so the manuscript's claim that no model answers the probe
correctly was never tested by the code that was supposed to test it. Re-running
the same probes with rank recorded showed the claim was false.

It also had a design flaw that no amount of extra measurement would fix. Its
probes were real-world updates -- a 2004 drug withdrawal, a 2022 succession, a
2023 rename -- so whether a model held the stale fact at all depended on its
training cutoff, and the cutoffs differ across the six models. A model that
never learned the old fact cannot be observed resisting a correction to it.

Counterfactual probes remove that confound. The document asserts something the
model is certain is false, so the conflict exists for every model regardless of
cutoff, and the measurement is the same question the paper meant to ask: when
context and parameters disagree, which one comes out of the model? This is the
standard construction in the knowledge-conflict literature (Longpre et al.,
EMNLP 2021; Xie et al., ICLR 2024).

Recorded per condition: the probability and rank of both candidate answers, and
the token the model actually puts first.
"""

import csv
import gc
import json
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Each probe is a fact no model of any size gets wrong, paired with a
# counterfactual that is a real city of the same country, so the substitution is
# plausible rather than nonsensical.
PROBES = [
    ("France",        " Paris",      " Lyon"),
    ("Japan",         " Tokyo",      " Osaka"),
    ("Italy",         " Rome",       " Milan"),
    ("Egypt",         " Cairo",      " Alexandria"),
    ("Canada",        " Ottawa",     " Toronto"),
    ("Australia",     " Canberra",   " Sydney"),
    ("Spain",         " Madrid",     " Barcelona"),
    ("Russia",        " Moscow",     " Petersburg"),
    ("China",         " Beijing",    " Shanghai"),
    ("Kenya",         " Nairobi",    " Mombasa"),
    ("Peru",          " Lima",       " Cusco"),
    ("Norway",        " Oslo",       " Bergen"),
    ("Greece",        " Athens",     " Thessaloniki"),
    ("Portugal",      " Lisbon",     " Porto"),
    ("Morocco",       " Rabat",      " Casablanca"),
    ("Turkey",        " Ankara",     " Istanbul"),
    ("Vietnam",       " Hanoi",      " Saigon"),
    ("Nigeria",       " Abuja",      " Lagos"),
    ("Switzerland",   " Bern",       " Zurich"),
    ("Netherlands",   " Amsterdam",  " Rotterdam"),
    ("Poland",        " Warsaw",     " Krakow"),
    ("Sweden",        " Stockholm",  " Gothenburg"),
    ("Germany",       " Berlin",     " Munich"),
    ("Argentina",     " Buenos",     " Cordoba"),
]

# Three ways of asking the same question. A number that moves between these is a
# fact about the sentence, not about the model's knowledge.
PHRASINGS = [
    "Question: What is the capital of {c}? Answer: The capital of {c} is",
    "Question: Which city is the seat of government of {c}? Answer: The capital city of {c} is",
    "Question: Name the capital of {c}. Answer: It is",
]

DOCUMENT = ("Context: Following a constitutional reform, the capital of {c} was "
            "relocated to {x}. {x} is now the official capital of {c}.")

MODELS = ["gpt2", "gpt2-medium", "gpt2-large", "Qwen/Qwen2.5-0.5B-Instruct",
          "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Qwen/Qwen2.5-1.5B-Instruct"]


def first_token_id(tokenizer, prompt, answer):
    """The id of the answer's first token, as it is tokenised after the prompt.

    Tokenisers merge across the boundary for some prompt-answer pairs; when that
    happens the probe is unusable rather than silently wrong, so it raises.
    """
    base = tokenizer(prompt, add_special_tokens=False).input_ids
    full = tokenizer(prompt + answer, add_special_tokens=False).input_ids
    n = 0
    while n < len(base) and n < len(full) and base[n] == full[n]:
        n += 1
    if n != len(base):
        raise ValueError("tokeniser merged the prompt and answer boundary")
    return full[len(base)]


def next_token_distribution(model, tokenizer, prompt):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        logits = model(ids).logits[0, -1, :].float()
    return F.log_softmax(logits, dim=-1)


def rank_of(logprobs, token_id):
    """1 is the token the model puts first."""
    return int((logprobs > logprobs[token_id]).sum().item()) + 1


def main():
    rows = []
    for name in MODELS:
        t0 = time.time()
        print(f"\n{name}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
        model.eval()

        skipped = 0
        for country, true_ans, cf_ans in PROBES:
            for p_i, template in enumerate(PHRASINGS):
                question = template.format(c=country)
                doc = DOCUMENT.format(c=country, x=cf_ans.strip())
                try:
                    t_id = first_token_id(tokenizer, question, true_ans)
                    c_id = first_token_id(tokenizer, question, cf_ans)
                except ValueError:
                    skipped += 1
                    continue

                for condition, prompt in (("no_doc", question),
                                          ("doc", doc + "\n" + question)):
                    lp = next_token_distribution(model, tokenizer, prompt)
                    top_id = int(lp.argmax().item())
                    rows.append({
                        "model": name,
                        "country": country,
                        "phrasing": p_i + 1,
                        "condition": condition,
                        "p_true": float(lp[t_id].exp().item()),
                        "p_counterfactual": float(lp[c_id].exp().item()),
                        "rank_true": rank_of(lp, t_id),
                        "rank_counterfactual": rank_of(lp, c_id),
                        "top1": tokenizer.decode([top_id]),
                        "top1_is_true": top_id == t_id,
                        "top1_is_counterfactual": top_id == c_id,
                    })

        n_doc = [r for r in rows if r["model"] == name and r["condition"] == "doc"]
        n_nod = [r for r in rows if r["model"] == name and r["condition"] == "no_doc"]
        if n_doc:
            print(f"   parametric answer top-1 without the document: "
                  f"{sum(r['top1_is_true'] for r in n_nod)}/{len(n_nod)}")
            print(f"   context answer top-1 with the document:       "
                  f"{sum(r['top1_is_counterfactual'] for r in n_doc)}/{len(n_doc)}")
        if skipped:
            print(f"   {skipped} probe/phrasing pairs skipped on tokenisation")
        print(f"   {time.time() - t0:.0f}s", flush=True)

        del model
        gc.collect()

    with open("counterfactual_update.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} rows written to counterfactual_update.csv")


if __name__ == "__main__":
    main()
