---
title: Lifecycle Desynchronization Probe
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
license: cc-by-4.0
---

# Lifecycle Desynchronization Probe

Companion demo for **The Knowledge Lifecycle of Large Language Models** (Amey Thakur, 2026).

The probe measures what happens when a language model's parametric memory conflicts with corrective context. Given a query, a document that contradicts the model's training-time knowledge, and the verified correct continuation, it reports:

- **D_sync**, the surprisal of the correct answer with the corrective document in context. A value of *n* nats means the model assigns the correct answer probability e^(-n). Above 9.2 nats, no realistic decoding recovers it.
- **I_ctx**, the full-vocabulary KL divergence between the model's output distributions with and without the document. Near zero means the context is present but inert.
- The top next-token distributions under both conditions.

The default inputs reproduce the paper's measurement of the 2004 Vioxx withdrawal on GPT-2 base: the model answers "safe" whether or not the withdrawal notice is present (D_sync = 12.05 nats, I_ctx = 0.033 nats), a concrete instance of parametric memory overriding retrieved evidence.

Everything is deterministic: identical inputs give identical numbers on every run.

Paper repository: https://github.com/Amey-Thakur/LLM-KNOWLEDGE-LIFECYCLE
