"""
===============================================================================
FILE         : analyse_counterfactual.py
PROJECT      : The Knowledge Lifecycle of Large Language Models
PURPOSE      : Turn counterfactual_update.csv into the numbers the paper quotes,
               so no figure in the manuscript is typed by hand.
TECH STACK   : Python 3 (standard library only)
AUTHORS      : Amey Thakur (https://github.com/Amey-Thakur)
               Sarvesh Talele (https://github.com/sarveshtalele)
REPOSITORY   : https://github.com/Amey-Thakur/LLM-KNOWLEDGE-LIFECYCLE
LICENSE      : CC BY 4.0
===============================================================================
"""

import csv
import json
import pathlib
import statistics
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
PARAMS = {  # parameter counts, for the scale question
    "gpt2": 124, "gpt2-medium": 355, "gpt2-large": 774,
    "Qwen/Qwen2.5-0.5B-Instruct": 494,
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 1100,
    "Qwen/Qwen2.5-1.5B-Instruct": 1540,
}


def load():
    with open(HERE / "counterfactual_update.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["p_true"] = float(r["p_true"])
        r["p_counterfactual"] = float(r["p_counterfactual"])
        r["rank_true"] = int(r["rank_true"])
        r["rank_counterfactual"] = int(r["rank_counterfactual"])
        r["top1_is_true"] = r["top1_is_true"] == "True"
        r["top1_is_counterfactual"] = r["top1_is_counterfactual"] == "True"
    return rows


def pct(n, d):
    return 100.0 * n / d if d else 0.0


def main():
    rows = load()
    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["condition"])].append(r)

    out = {"models": {}, "n_rows": len(rows)}
    print(f"{len(rows)} measurements\n")
    print(f"{'model':<36} {'params':>7} {'knows':>7} {'updates':>8} {'stale':>7}")
    print("-" * 70)

    for m in PARAMS:
        nod = by[(m, "no_doc")]
        doc = by[(m, "doc")]
        if not nod:
            continue
        knows = sum(r["top1_is_true"] for r in nod)          # parametric answer
        updates = sum(r["top1_is_counterfactual"] for r in doc)  # adopts context
        stale = sum(r["top1_is_true"] for r in doc)          # keeps its own answer
        out["models"][m] = {
            "params_m": PARAMS[m],
            "n": len(nod),
            "knows_no_doc": knows,
            "knows_no_doc_pct": round(pct(knows, len(nod)), 1),
            "updates_with_doc": updates,
            "updates_with_doc_pct": round(pct(updates, len(doc)), 1),
            "stale_with_doc": stale,
            "stale_with_doc_pct": round(pct(stale, len(doc)), 1),
            "median_rank_true_no_doc": statistics.median(r["rank_true"] for r in nod),
            "median_rank_cf_doc": statistics.median(r["rank_counterfactual"] for r in doc),
            "mean_p_true_no_doc": round(statistics.mean(r["p_true"] for r in nod), 4),
            "mean_p_cf_doc": round(statistics.mean(r["p_counterfactual"] for r in doc), 4),
        }
        print(f"{m:<36} {PARAMS[m]:>6}M {pct(knows, len(nod)):>6.1f}% "
              f"{pct(updates, len(doc)):>7.1f}% {pct(stale, len(doc)):>6.1f}%")

    # A conflict only exists where the model held the fact in the first place.
    # Restricting to those cases is what separates deference to a document from
    # simply not knowing the answer, and it is the number the paper leads on.
    print("\nconflict cases only (model ranked the true answer first without the document)")
    for m in PARAMS:
        nod = {(r["country"], r["phrasing"]): r for r in by[(m, "no_doc")]}
        held = [r for r in by[(m, "doc")]
                if nod.get((r["country"], r["phrasing"]), {}).get("top1_is_true")]
        if not held:
            continue
        adopt = sum(r["top1_is_counterfactual"] for r in held)
        keep = sum(r["top1_is_true"] for r in held)
        out["models"][m]["n_conflict"] = len(held)
        out["models"][m]["adopt_in_conflict_pct"] = round(pct(adopt, len(held)), 1)
        out["models"][m]["keep_in_conflict_pct"] = round(pct(keep, len(held)), 1)
        print(f"  {m:<36} n={len(held):>3}  adopts {pct(adopt, len(held)):5.1f}%"
              f"   keeps {pct(keep, len(held)):5.1f}%")

    conflict_rows = []
    for m in PARAMS:
        nod = {(r["country"], r["phrasing"]): r for r in by[(m, "no_doc")]}
        conflict_rows += [r for r in by[(m, "doc")]
                          if nod.get((r["country"], r["phrasing"]), {}).get("top1_is_true")]
    out["n_conflict_total"] = len(conflict_rows)
    out["adopt_in_conflict_overall_pct"] = round(
        pct(sum(r["top1_is_counterfactual"] for r in conflict_rows), len(conflict_rows)), 1)
    out["keep_in_conflict_overall_pct"] = round(
        pct(sum(r["top1_is_true"] for r in conflict_rows), len(conflict_rows)), 1)
    print(f"  overall: {len(conflict_rows)} genuine conflicts, "
          f"document adopted {out['adopt_in_conflict_overall_pct']}%, "
          f"model keeps its answer {out['keep_in_conflict_overall_pct']}%")

    # Does the answer depend on how the question is worded?
    print("\nphrasing spread (update rate per phrasing, percentage points)")
    spreads = []
    for m in PARAMS:
        per = []
        for p in ("1", "2", "3"):
            sub = [r for r in by[(m, "doc")] if r["phrasing"] == p]
            if sub:
                per.append(pct(sum(r["top1_is_counterfactual"] for r in sub), len(sub)))
        if len(per) == 3:
            spread = max(per) - min(per)
            spreads.append(spread)
            out["models"][m]["update_by_phrasing"] = [round(x, 1) for x in per]
            out["models"][m]["phrasing_spread"] = round(spread, 1)
            print(f"  {m:<36} " + "  ".join(f"{x:5.1f}" for x in per)
                  + f"   spread {spread:5.1f}")
    out["median_phrasing_spread"] = round(statistics.median(spreads), 1) if spreads else None

    # The methodological point: a low surprisal for the in-context answer does
    # not mean the model would say it.
    doc_all = [r for r in rows if r["condition"] == "doc"]
    confident = [r for r in doc_all if r["p_counterfactual"] > 0.10]
    not_said = [r for r in confident if not r["top1_is_counterfactual"]]
    out["prob_over_10pct"] = len(confident)
    out["prob_over_10pct_not_top1"] = len(not_said)
    out["prob_over_10pct_not_top1_pct"] = round(pct(len(not_said), len(confident)), 1)
    print(f"\nthe in-context answer carries probability above 0.10 in "
          f"{len(confident)} of {len(doc_all)} conditions,")
    print(f"and in {len(not_said)} of those "
          f"({pct(len(not_said), len(confident)):.1f}%) it is still not what the model says.")

    totals_doc = sum(1 for r in doc_all)
    out["overall_update_pct"] = round(
        pct(sum(r["top1_is_counterfactual"] for r in doc_all), totals_doc), 1)
    out["overall_stale_pct"] = round(
        pct(sum(r["top1_is_true"] for r in doc_all), totals_doc), 1)
    print(f"\noverall: context answer given {out['overall_update_pct']}% of the time, "
          f"parametric answer kept {out['overall_stale_pct']}%")

    (HERE / "counterfactual_summary.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("\nwritten to counterfactual_summary.json")


if __name__ == "__main__":
    main()
