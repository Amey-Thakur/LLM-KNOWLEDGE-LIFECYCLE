"""
===============================================================================
FILE         : paraphrase_robustness.py
PROJECT      : The Knowledge Lifecycle of Large Language Models
PURPOSE      : Measure how far D_sync moves when the question is asked in a
               different way, so the paper can say whether its numbers are a
               property of the conflict or of one sentence.
TECH STACK   : Python 3, PyTorch, Hugging Face Transformers
AUTHORS      : Amey Thakur (https://github.com/Amey-Thakur)
               Sarvesh Talele (https://github.com/sarveshtalele)
REPOSITORY   : https://github.com/Amey-Thakur/LLM-KNOWLEDGE-LIFECYCLE
LICENSE      : CC BY 4.0
===============================================================================

The single-phrasing objection is the first one a referee raises, and it is fair:
a surprisal measured on one sentence could be a fact about that sentence. Each
probe therefore gets two further phrasings that ask the same question in
different words, keeping the corrective document and the verified answer fixed,
and the spread across the three is reported.
"""

import csv
import gc
import json
import statistics
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PROBES = [
    {
        "name": "vioxx",
        "queries": [
            "Question: Is Vioxx safe to prescribe? Answer: Vioxx is considered",
            "Question: What is the regulatory status of Vioxx? Answer: Vioxx has been",
            "Question: Can a doctor still prescribe Vioxx? Answer: The drug Vioxx was",
        ],
        "context": "Context: In September 2004, Merck voluntarily withdrew Vioxx after trials revealed increased cardiovascular risks.",
        "answer": " withdrawn",
    },
    {
        "name": "monarch",
        "queries": [
            "Question: Who is the current British monarch? Answer: The current British monarch is",
            "Question: Who sits on the British throne today? Answer: The reigning British monarch is",
            "Question: Who is the head of state of the United Kingdom? Answer: The head of state of the United Kingdom is",
        ],
        "context": "Context: Queen Elizabeth II died in September 2022. Charles III acceded to the throne and is the reigning King of the United Kingdom.",
        "answer": " Charles",
    },
    {
        "name": "twitter",
        "queries": [
            "Question: What is the social network Twitter called today? Answer: Twitter is now called",
            "Question: What is the new name of Twitter? Answer: The platform formerly known as Twitter is now named",
            "Question: Under Elon Musk, Twitter was renamed to what? Answer: Twitter was renamed",
        ],
        "context": "Context: In July 2023, Twitter was rebranded as X under Elon Musk's ownership.",
        "answer": " X",
    },
]

MODELS = ["gpt2", "gpt2-medium", "gpt2-large", "Qwen/Qwen2.5-0.5B-Instruct",
          "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Qwen/Qwen2.5-1.5B-Instruct"]


def answer_token_ids(tokenizer, prompt, answer):
    base = tokenizer(prompt, add_special_tokens=False).input_ids
    full = tokenizer(prompt + answer, add_special_tokens=False).input_ids
    n = 0
    while n < len(base) and n < len(full) and base[n] == full[n]:
        n += 1
    if n != len(base):
        raise ValueError("tokenizer merged the prompt and answer boundary")
    return full[len(base):]


def sequence_surprisal(model, tokenizer, prompt, ids):
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    ans = torch.tensor([ids])
    x = torch.cat([prompt_ids, ans], dim=1).to(model.device)
    with torch.no_grad():
        logits = model(x).logits[0].float()
    n = ans.shape[1]
    lp = F.log_softmax(logits[-n - 1:-1, :], dim=-1)
    return -lp.gather(1, ans[0].unsqueeze(1).to(model.device)).sum().item()


rows = []
for name in MODELS:
    t0 = time.time()
    print(f"\n{name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    model.eval()

    for probe in PROBES:
        vals = []
        for q in probe["queries"]:
            prompt = probe["context"] + "\n" + q
            ids = answer_token_ids(tokenizer, prompt, probe["answer"])
            vals.append(sequence_surprisal(model, tokenizer, prompt, ids))
        rows.append({
            "model": name,
            "probe": probe["name"],
            "d_q1": vals[0], "d_q2": vals[1], "d_q3": vals[2],
            "mean": statistics.mean(vals),
            "sd": statistics.stdev(vals),
            "spread": max(vals) - min(vals),
        })
        print(f"   {probe['name']:<9} " + "  ".join(f"{v:6.2f}" for v in vals)
              + f"   mean {statistics.mean(vals):6.2f}  sd {statistics.stdev(vals):5.2f}",
              flush=True)

    del model
    gc.collect()
    print(f"   {time.time() - t0:.0f}s", flush=True)

with open("paraphrase_robustness.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print("\nsummary")
for probe in ("vioxx", "monarch", "twitter"):
    sub = [r for r in rows if r["probe"] == probe]
    print(f"  {probe:<9} median sd across models {statistics.median(r['sd'] for r in sub):5.2f} nats"
          f"   largest spread {max(r['spread'] for r in sub):5.2f}")
print(f"\n{len(rows)} rows written to paraphrase_robustness.csv")
