<div align="center">

# The Knowledge Lifecycle of Large Language Models

**A unified framework for how language models acquire, store, retrieve, update, and forget knowledge, and a reproducible measurement of what breaks at the boundaries.**

<br>

[![Demo](https://img.shields.io/badge/Demo-Hugging_Face_Space-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/ameythakur/llm-knowledge-lifecycle)
[![Notebook](https://img.shields.io/badge/Notebook-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/ameythakur20/cross-model-lifecycle-desynchronization)
[![Authors](https://img.shields.io/badge/Authors-Amey_Thakur_%26_Sarvesh_Talele-0969DA)](https://github.com/Amey-Thakur)
[![Technology](https://img.shields.io/badge/Technology-Python_%7C_PyTorch_%7C_LaTeX-8250DF)](#reproducing-the-measurement)
[![Status](https://img.shields.io/badge/Status-Preprint_in_preparation-2EA043)](#the-paper)
[![License](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey)](LICENSE)

<br>

<img src=".github/social-preview.png" alt="The Knowledge Lifecycle of Large Language Models. GPT-2, asked about Vioxx with the 2004 withdrawal notice in its prompt, answers safe with 42.6% probability while the correct answer receives 0.0006%." width="760">

<br><br>

[The paper](#the-paper) &nbsp;·&nbsp;
[Contributions](#contributions) &nbsp;·&nbsp;
[The measurement](#the-measurement) &nbsp;·&nbsp;
[Live demonstration](#live-demonstration) &nbsp;·&nbsp;
[Reproducing](#reproducing-the-measurement) &nbsp;·&nbsp;
[Repository layout](#repository-layout) &nbsp;·&nbsp;
[Authors](#authors) &nbsp;·&nbsp;
[Citation](#citation)

</div>

---

<a name="the-paper"></a>
## The paper

Research on how language models handle knowledge is split across five subfields that rarely cite one another: retrieval-augmented generation, knowledge editing, continual learning, context engineering, and agent memory. Each optimizes one stage in isolation, and each has its own benchmarks that pass while the assembled system fails.

This paper organizes the five as stages of a single **Knowledge Lifecycle**, then shows that the boundaries between stages are where deployed systems break. Mapping twenty representative works onto the five stages gives a median coverage of two: the interactions nobody tests are precisely the ones that cause harm in production.

The manuscript is [`paper/main.tex`](paper/main.tex). Every push compiles it, the poster, and the slide deck to PDF.

<a name="contributions"></a>
## Contributions

**Lifecycle Desynchronization ($\mathcal{D}_{sync}$)** turns the boundary failure into a number. For a fact with a verified answer it reduces to the surprisal of that answer with the corrective document present in context: $n$ nats means the model assigns the correct answer probability $e^{-n}$. A companion diagnostic, $\mathcal{I}_{ctx}$, measures how far the document moves the model's output distribution at all. Together they separate a retrieval failure from a resolution failure, which no single-stage benchmark can do.

**The Provenance Vector ($p$)** addresses the root cause: parametric storage discards when and from where a fact was learned, so a model has no principled basis for preferring fresh evidence over a confident stale memory. Each feed-forward memory slot gains a metadata embedding of acquisition time and source reliability, and a gate attenuates stale activations at inference. Weights are never modified, so the specificity bottleneck that limits static editing does not apply. The paper gives the forward pass as an algorithm, derives the parameter cost, and proves an idealized consistency result under explicitly stated assumptions.

<a name="the-measurement"></a>
## The measurement

Rofecoxib, marketed as Vioxx, was withdrawn in September 2004 after trials showed elevated cardiovascular risk. Put that withdrawal notice in GPT-2's prompt and ask whether the drug is safe to prescribe:

| Quantity | Without the document | With the document |
| :--- | ---: | ---: |
| P(" safe"), the incorrect answer | 37.53% | **42.58%** |
| P(" withdrawn"), the correct answer | 0.0004% | 0.0006% |
| Top-ranked token | " safe" | " safe" |

The correction is in the prompt, and the model's confidence that the drug is safe **rises**. The correct answer sits at $5.9 \times 10^{-6}$, giving $\widehat{\mathcal{D}}_{sync} = 12.05$ nats, well past the 9.2-nat threshold beyond which no realistic decoding recovers it. The document shifts the full output distribution by $\mathcal{I}_{ctx} = 0.033$ nats: retrieval worked, resolution failed.

> [!NOTE]
> A cross-model extension of this measurement, spanning the GPT-2 family and instruction-tuned models, is prepared in [`experiments/`](experiments/) and runs on free Kaggle hardware.

<a name="live-demonstration"></a>
## Live demonstration

[**huggingface.co/spaces/ameythakur/llm-knowledge-lifecycle**](https://huggingface.co/spaces/ameythakur/llm-knowledge-lifecycle)

GPT-2 runs entirely in the visitor's browser through ONNX; no server sees the input. Three preset probes cover three distinct regimes, and any fact change with a single-word answer can be entered directly. Two parameters are adjustable: sampling temperature, recomputed live from the measured logits, and document repetition, which re-runs the model to test whether saying it louder helps.

The Space is mirrored from [`space/`](space/) by [a workflow](.github/workflows/sync-space.yml) on every push. **GitHub is the source of truth**; edits made through the Hugging Face web interface are overwritten by the next push.

<a name="reproducing-the-measurement"></a>
## Reproducing the measurement

```bash
pip install torch transformers
python experiments/dsync_experiment.py
```

Prints the top tokens under both conditions, the probabilities of the correct and incorrect continuations, the full-vocabulary KL between conditions, and the surprisal estimator. There is no sampling anywhere, so the numbers above reproduce exactly. CPU is sufficient.

<a name="repository-layout"></a>
## Repository layout

```
.
├── paper/                    # Manuscript, bibliography, poster, slides, derivations
│   ├── main.tex              #   The manuscript
│   ├── references.bib        #   Every entry verified against source metadata
│   ├── poster.tex            #   Conference poster
│   ├── presentation.tex      #   Slide deck
│   └── derivations.md        #   Extended mathematical derivations
├── experiments/              # Runnable measurements
│   ├── dsync_experiment.py   #   The paper's measurement, deterministic
│   ├── cross_model_dsync.ipynb  # Cross-model sweep, runs on Kaggle
│   └── README.md             #   How to run it and what to report
├── space/                    # The live demonstration, mirrored to Hugging Face
├── .github/                  # Build and sync workflows
├── CITATION.cff              # How to cite this work
├── codemeta.json             # Machine-readable project metadata
└── LICENSE                   # CC BY 4.0
```

<a name="authors"></a>
## Authors

<div align="center">

| <a href="https://github.com/Amey-Thakur"><img src="space/amey-thakur.jpg" width="150" height="150" alt="Amey Thakur"></a><br>[**Amey Thakur**](https://github.com/Amey-Thakur)<br><br>[![ORCID](https://img.shields.io/badge/ORCID-0000--0001--5644--1575-A6CE39.svg)](https://orcid.org/0000-0001-5644-1575)<br>[![Kaggle](https://img.shields.io/badge/Kaggle-ameythakur20-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/ameythakur20) | <a href="https://github.com/sarveshtalele"><img src="https://github.com/sarveshtalele.png" width="150" height="150" alt="Sarvesh Talele"></a><br>[**Sarvesh Talele**](https://github.com/sarveshtalele)<br><br>[![ORCID](https://img.shields.io/badge/ORCID-0009--0002--0818--461X-A6CE39.svg)](https://orcid.org/0009-0002-0818-461X)<br>[![GitHub](https://img.shields.io/badge/GitHub-sarveshtalele-181717?logo=github)](https://github.com/sarveshtalele) |
| :---: | :---: |

</div>

> [!IMPORTANT]
> ### 🤝🏻 Special Acknowledgement
> *Special thanks to **[Sarvesh Talele](https://github.com/sarveshtalele)** for his meaningful contributions, support, and wisdom that helped shape this work.*

<a name="citation"></a>
## Citation

```bibtex
@article{thakur2026lifecycle,
  author  = {Thakur, Amey and Talele, Sarvesh},
  title   = {The Knowledge Lifecycle of Large Language Models},
  journal = {arXiv preprint},
  year    = {2026}
}
```

---

<div align="center">

**Amey Thakur** &nbsp;·&nbsp; [GitHub](https://github.com/Amey-Thakur) &nbsp;·&nbsp; [ORCID](https://orcid.org/0000-0001-5644-1575) &nbsp;·&nbsp; [Amey's Arc](https://amey-thakur.github.io)

<br>

[↑ Back to top](#the-knowledge-lifecycle-of-large-language-models)

</div>
