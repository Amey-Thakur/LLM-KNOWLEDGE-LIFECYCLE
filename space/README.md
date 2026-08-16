---
title: LLM Knowledge Lifecycle
emoji: 📄
colorFrom: gray
colorTo: blue
sdk: static
pinned: false
license: cc-by-4.0
thumbnail: https://huggingface.co/spaces/ameythakur/llm-knowledge-lifecycle/resolve/main/social-preview.png
---

# LLM Knowledge Lifecycle

Companion demonstration for **The Knowledge Lifecycle of Large Language Models** (Amey Thakur and Sarvesh Talele, 2026). GPT-2 base runs entirely in the visitor's browser via ONNX; every measurement is computed locally with no server involved.

The probe measures what happens when a language model's parametric memory conflicts with corrective context:

- **D_sync**, the surprisal of the correct answer with the corrective document in context. A value of *n* nats means the model assigns the correct answer probability e^(-n).
- **I_ctx**, the full-vocabulary KL divergence between the model's output distributions with and without the document. Near zero means the document is present but inert.

Three preset probes cover documented fact changes landing in three regimes: total resolution failure (the 2004 Vioxx withdrawal, the paper's headline measurement), the exposure trap where correct context reinforces the wrong answer (the British monarch after 2022), and drift where strong context influence still fails to flip the answer (the 2023 Twitter rename).

Paper repository: https://github.com/Amey-Thakur/LLM-KNOWLEDGE-LIFECYCLE
