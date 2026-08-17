# Cross-Model Experiment: Instructions

This folder holds the cross-model desynchronization experiment for *The Knowledge Lifecycle of Large Language Models*. The notebook is complete and runs top to bottom on Kaggle; whoever runs it sanity-checks the reproduction gate and writes the findings.

**Expected effort:** one session, roughly an hour including the reading
**Where the results go:** a new appendix in the paper, "Cross-Model Desynchronization Measurements", added in the next revision, crediting whoever ran and wrote them

## Background, in three sentences

The paper shows that when a fact changes after a model's training data was collected, GPT-2 keeps answering from its stale memory even when the correction sits in its prompt: the correct answer gets probability six in a million (D_sync = 12.05 nats), and the corrective document barely moves the output distribution (I_ctx = 0.033 nats). Section 9.2 of the paper asks how these numbers change with model scale and instruction tuning. This notebook is the sweep that answers it, across six models on the same three probes, and its results are reported as Section 8.6.

## Steps

1. Read the paper's Sections 8.4 and 8.5 (the metric and its protocol, then the Vioxx case) and the live demo at https://huggingface.co/spaces/ameythakur/llm-knowledge-lifecycle so the two numbers mean something before you run anything.
2. Create a new Kaggle notebook. Settings: **CPU**, **Internet on**, latest environment. The notebook runs single forward passes rather than generation, so a GPU buys almost nothing and spends quota; `kernel-metadata.json` sets `enable_gpu: false` to match. Budget about 35 minutes of compute for the six models, plus roughly 13 GB of first-run model downloads.
3. Upload `cross_model_dsync.ipynb` from this folder (File, Import Notebook) and run all cells, top to bottom, once.
4. Watch two gates:
   - **The notebook's Section 6, reproduction check.** The gpt2/vioxx row must match the paper (D_sync = 12.0464, tolerance 0.05). If the assert fails, stop and report the numbers you got; do not continue.
   - **The notebook's Section 4, tokenization.** For each model, the printed `answer_first_piece` should be the expected word piece. If a tokenizer splits an answer strangely, note it; the row is still valid but the note matters for the paper.
5. Write the notebook's Section 8 (Findings): one short paragraph per question, each sentence backed by a number from the tables in its Section 7. If something looks wrong or surprising, say so plainly; a strange result honestly reported is worth more than a clean-looking one.
6. Save the notebook version on Kaggle (Save Version, Save and Run All), and send back: the Kaggle notebook link, the written findings, and `cross_model_dsync.csv` from the output.

## Rules that apply

- Nothing in the notebook is edited above its Section 8 without flagging it first; the probes and protocol must stay byte-identical to the paper or the comparison collapses.
- Every claim in the findings carries its number.
- Nothing that failed is deleted. If a model errors out or gives a bizarre distribution, that is a finding, not a blemish.

## What happens with the results

The measured tables go into the paper as an appendix, the findings paragraphs seed its prose, and the revision credits the contributor as second author. The same protocol later extends to the text-image conflict case on a vision-language model, which is the larger follow-up if you want it after this.
