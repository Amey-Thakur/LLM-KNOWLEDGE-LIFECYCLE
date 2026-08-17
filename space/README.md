---
title: LLM Knowledge Lifecycle
emoji: 📄
colorFrom: gray
colorTo: blue
sdk: static
pinned: false
license: cc-by-4.0
short_description: Measure a model ignoring a document in its own prompt
thumbnail: https://huggingface.co/spaces/ameythakur/llm-knowledge-lifecycle/resolve/main/social-preview.png
tags:
  - knowledge-lifecycle
  - retrieval-augmented-generation
  - knowledge-conflicts
  - interpretability
  - gpt2
  - research-demo
---

<div align="center">

# The Knowledge Lifecycle of Large Language Models

**A model holds two memories: what it learned in training, and what you put in its prompt. When they disagree, training usually wins, even when training is wrong.**

<br>

[![Paper](https://img.shields.io/badge/GitHub-LLM--KNOWLEDGE--LIFECYCLE-181717?logo=github)](https://github.com/Amey-Thakur/LLM-KNOWLEDGE-LIFECYCLE)
[![Notebook](https://img.shields.io/badge/Notebook-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/ameythakur20/cross-model-lifecycle-desynchronization)
[![Authors](https://img.shields.io/badge/Authors-Amey_Thakur_%26_Sarvesh_Talele-0969DA)](https://github.com/Amey-Thakur)
[![License](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey)](https://github.com/Amey-Thakur/LLM-KNOWLEDGE-LIFECYCLE/blob/main/LICENSE)

<br>

[![Acquire](https://img.shields.io/badge/1-Acquire-4A7FD4)](#the-five-stages)
[![Store](https://img.shields.io/badge/2-Store-2A9D8F)](#the-five-stages)
[![Retrieve](https://img.shields.io/badge/3-Retrieve-E08A2E)](#the-five-stages)
[![Update](https://img.shields.io/badge/4-Update-D05353)](#the-five-stages)
[![Forget](https://img.shields.io/badge/5-Forget-8F5FB8)](#the-five-stages)

</div>

---

## What this demonstration measures

Rofecoxib, sold as Vioxx, was withdrawn worldwide in September 2004 after trials showed it raised the risk of heart attack and stroke.

Put that withdrawal notice directly into GPT-2's prompt, then ask whether the drug is safe to prescribe:

| | Without the notice | With the notice in the prompt |
| :--- | ---: | ---: |
| Answers **"safe"** | 37.53% | **42.58%** |
| Answers **"withdrawn"** | 0.0004% | 0.0006% |

The correction is sitting in front of the model, and its confidence that the drug is safe **goes up**.

This is not a hallucination in the usual sense. Retrieval worked: the right document was found and delivered. What failed is the step after it, where the model must decide which of its two memories to believe.

## The two numbers

**D<sub>sync</sub>** is the surprisal of the correct answer while the corrective document is present, measured in nats. A value of *n* nats means the correct answer holds probability e<sup>-n</sup>. Past **9.2 nats** the answer is below one chance in ten thousand, where no realistic decoding recovers it. The Vioxx probe measures **12.05**.

**I<sub>ctx</sub>** is the full-vocabulary divergence between the model's output with and without the document. It answers a different question: did the document change the model's mind about anything at all? On Vioxx it is **0.033 nats**, meaning the document is present and inert.

Together they separate a retrieval failure, where the document never arrived, from a resolution failure, where it arrived and was ignored. No single-stage benchmark can tell those apart.

<a name="the-five-stages"></a>
## The five stages

A fact inside a language model passes through five stages, each studied by a different research community that rarely cites the others.

| Stage | What happens to the fact | Studied as |
| :--- | :--- | :--- |
| ![Acquire](https://img.shields.io/badge/Acquire-4A7FD4) | Training compresses a corpus into the weights | Pre-training, fine-tuning |
| ![Store](https://img.shields.io/badge/Store-2A9D8F) | It lives in the weights, in an external index, or in both | Parametric memory, vector databases |
| ![Retrieve](https://img.shields.io/badge/Retrieve-E08A2E) | Attention recalls it, or a search pipeline fetches a document | Retrieval-augmented generation |
| ![Update](https://img.shields.io/badge/Update-D05353) | The world changes, and the stored copies must change with it | Knowledge editing, continual learning |
| ![Forget](https://img.shields.io/badge/Forget-8F5FB8) | It is removed on purpose, or lost by accident | Machine unlearning, catastrophic forgetting |

The failure measured here sits on the boundary between **Retrieve** and **Update**: the document arrives, and nothing in the architecture tells the model which copy of the fact to trust.

## Three presets, three regimes

| Case | What the model does |
| :--- | :--- |
| **Vioxx withdrawn**, 2004 | Ignores the notice entirely. Resolution failure. |
| **Elizabeth II died**, 2022 | The notice mostly boosts "Queen". Correct context strengthening the wrong answer. |
| **Twitter renamed X**, 2023 | Moves hard, still answers "Twitter". Influence without resolution. |

Any fact change with a single-word answer can be entered directly. Three rules make a clean probe: the query ends mid-sentence so the next token is the answer, the document states the new fact plainly, and the answer is one word.

## How it runs

GPT-2 base executes entirely in your browser through ONNX. Nothing you type leaves your machine, no account is needed, and identical inputs always return identical numbers.

The browser build uses 8-bit quantized weights, which shift individual probabilities relative to full precision. Every preset lands in the same regime and supports the same conclusion; the paper's exact values come from the deterministic PyTorch script in the repository.

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

**Amey Thakur** &nbsp;·&nbsp; [ORCID](https://orcid.org/0000-0001-5644-1575) &nbsp;&nbsp;|&nbsp;&nbsp; **Sarvesh Talele** &nbsp;·&nbsp; [ORCID](https://orcid.org/0009-0002-0818-461X)

<br>

[Paper and code](https://github.com/Amey-Thakur/LLM-KNOWLEDGE-LIFECYCLE) &nbsp;·&nbsp; [Amey's Arc](https://amey-thakur.github.io)

</div>
