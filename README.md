<div align="center">

# The Knowledge Lifecycle of Large Language Models

**How knowledge is acquired, stored, retrieved, updated, and forgotten by a language model, and what breaks at the boundaries.**

<br>

[![Demo](https://img.shields.io/badge/Demo-Hugging_Face_Space-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/ameythakur/llm-knowledge-lifecycle)
[![Author](https://img.shields.io/badge/Author-Amey_Thakur-0969DA)](https://github.com/Amey-Thakur)
[![ORCID](https://img.shields.io/badge/ORCID-0000--0001--5644--1575-A6CE39)](https://orcid.org/0000-0001-5644-1575)
[![Status](https://img.shields.io/badge/Status-Preprint_in_preparation-2EA043)](#paper)
[![License](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey)](LICENSE)

<br>

[Paper](#paper) &nbsp;·&nbsp;
[Contributions](#contributions) &nbsp;·&nbsp;
[Measurements](#measurements) &nbsp;·&nbsp;
[Demo](#demo) &nbsp;·&nbsp;
[Structure](#structure) &nbsp;·&nbsp;
[Reproducing](#reproducing) &nbsp;·&nbsp;
[Authors](#authors) &nbsp;·&nbsp;
[Citation](#citation)

</div>

---

<a name="paper"></a>
## Paper

Research on how language models handle knowledge is split across five subfields that rarely talk to each other: retrieval-augmented generation, knowledge editing, continual learning, context engineering, and agent memory. This paper organizes them as five stages of a single **Knowledge Lifecycle** (Acquisition, Storage, Retrieval, Update, Forgetting) and shows that the boundaries between stages, which no single-stage benchmark tests, are where deployed systems fail.

The manuscript is [`main.tex`](main.tex); every push compiles it to PDF through the repository's build workflow. A slide deck and poster accompany it.

<a name="contributions"></a>
## Contributions

**Lifecycle Desynchronization ($\mathcal{D}_{sync}$)** makes the boundary failure measurable. For a fact with a verified answer, it reduces to the surprisal of that answer with the corrective document present in context: a value of $n$ nats means the model assigns the correct answer probability $e^{-n}$. A companion diagnostic, $\mathcal{I}_{ctx}$, measures how far the document moves the model's output distribution at all; together they separate retrieval failures from resolution failures.

**The Provenance Vector ($p$)** targets the root cause. Each feed-forward memory slot gains a metadata embedding of its acquisition time and source reliability, and a gate attenuates stale activations at inference time. Weights are never modified, so there is no collateral damage of the kind that limits static editing. The paper gives the forward pass as an algorithm and proves an idealized consistency result with explicit assumptions.

<a name="measurements"></a>
## Measurements

All numbers in the paper are produced by [`dsync_experiment.py`](dsync_experiment.py), which is deterministic: no sampling anywhere, so every run gives identical values.

| Quantity (Vioxx probe, GPT-2 base) | Without context | With withdrawal context |
| :--- | ---: | ---: |
| P(" safe"), the incorrect answer | 37.53% | 42.58% |
| P(" withdrawn"), the correct answer | 0.0004% | 0.0006% |
| $\mathcal{I}_{ctx}$ (full-vocabulary shift) | | 0.033 nats |
| $\widehat{\mathcal{D}}_{sync}$ (surprisal of correct answer) | | 12.05 nats |

The withdrawal notice is in the prompt, and the model's belief that the drug is safe goes **up**. The full output distribution moves by 0.03 nats: retrieval worked, resolution failed.

> [!NOTE]
> A cross-model extension of this measurement, spanning the GPT-2 family and instruction-tuned models, is prepared in [`experiments/`](experiments/) and runs on a free Kaggle GPU.

<a name="demo"></a>
## Demo

The measurement runs live at the [**Hugging Face Space**](https://huggingface.co/spaces/ameythakur/llm-knowledge-lifecycle): GPT-2 executes entirely in the visitor's browser, with three preset probes covering three failure regimes and free-form inputs for building new ones. Source in [`space/`](space/); a Gradio variant for local use in [`demo/`](demo/).

<a name="structure"></a>
## Structure

```
.
├── main.tex                  # The manuscript
├── references.bib            # Bibliography; every entry verified against source metadata
├── dsync_experiment.py       # The paper's measurement, deterministic and reproducible
├── presentation.tex          # Beamer slide deck
├── poster.tex                # Conference poster
├── space/                    # Live demo (static, in-browser inference)
├── demo/                     # Gradio variant of the demo for local use
├── experiments/              # Cross-model measurement notebook and instructions
└── supplementary_materials/  # Extended mathematical derivations
```

<a name="reproducing"></a>
## Reproducing

```
pip install torch transformers
python dsync_experiment.py
```

Prints both conditions' top tokens, the probabilities of the correct and incorrect continuations, the full-vocabulary KL between conditions, and the surprisal estimator. CPU is sufficient; the model is GPT-2 base (124M).

<a name="authors"></a>
## Authors

<div align="center">

| <a href="https://github.com/Amey-Thakur"><img src="https://github.com/Amey-Thakur.png" width="150" height="150" alt="Amey Thakur"></a><br>[**Amey Thakur**](https://github.com/Amey-Thakur)<br><br>[![ORCID](https://img.shields.io/badge/ORCID-0000--0001--5644--1575-A6CE39.svg)](https://orcid.org/0000-0001-5644-1575)<br>[![Kaggle](https://img.shields.io/badge/Kaggle-ameythakur20-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/ameythakur20) | <a href="https://github.com/sarveshtalele"><img src="https://github.com/sarveshtalele.png" width="150" height="150" alt="Sarvesh Talele"></a><br>[**Sarvesh Talele**](https://github.com/sarveshtalele)<br><br>[![ORCID](https://img.shields.io/badge/ORCID-0009--0002--0818--461X-A6CE39.svg)](https://orcid.org/0009-0002-0818-461X)<br>[![GitHub](https://img.shields.io/badge/GitHub-sarveshtalele-181717?logo=github)](https://github.com/sarveshtalele) |
| :---: | :---: |

</div>

> [!IMPORTANT]
> ### 🤝🏻 Special Acknowledgement
> *Special thanks to **[Sarvesh Talele](https://github.com/sarveshtalele)** for his meaningful contributions, support, and wisdom that helped shape this work.*

<a name="citation"></a>
## Citation

```bibtex
@article{thakur2026lifecycle,
  author  = {Thakur, Amey},
  title   = {The Knowledge Lifecycle of Large Language Models},
  journal = {arXiv preprint},
  year    = {2026}
}
```

---

<div align="center">

**Amey Thakur** &nbsp;·&nbsp; [GitHub](https://github.com/Amey-Thakur) &nbsp;·&nbsp; [ORCID](https://orcid.org/0000-0001-5644-1575) &nbsp;·&nbsp; [Kaggle](https://www.kaggle.com/ameythakur20)

<br>

[↑ Back to top](#the-knowledge-lifecycle-of-large-language-models)

</div>
