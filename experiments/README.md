# Experiments

Everything behind the numbers in *The Knowledge Lifecycle of Large Language
Models*. Nothing here samples, so every run returns the same digits.

| File | What it does |
| :--- | :--- |
| `dsync_experiment.py` | The single-model measurement. Produces every number in the repository README and Section 8.5 of the paper. |
| `cross_model_sweep.py` | The same measurement across six models and three probes, with both controls. Writes `cross_model_dsync.csv`. |
| `cross_model_dsync.ipynb` | The sweep as a notebook, with the reasoning around it. Published on [Kaggle](https://www.kaggle.com/code/ameythakur20/cross-model-lifecycle-desynchronization). |
| `paraphrase_robustness.py` | Re-runs each probe under two further phrasings. Writes `paraphrase_robustness.csv`. |
| `make_figures.py` | Builds the four charts from `cross_model_dsync.csv`. |
| `kernel-metadata.json` | Kaggle settings for the notebook: CPU, internet on. |

## Running them

```bash
pip install torch transformers
python experiments/dsync_experiment.py
```

A laptop is enough. GPT-2 base at 124M parameters is small, fully open, and
carries no instruction tuning that would confound the result.

The sweep loads six models between 124M and 1.5B parameters, one at a time:

```bash
python experiments/cross_model_sweep.py
```

About four minutes of compute once the weights are cached, plus roughly 13 GB of
first-run downloads. These are single forward passes rather than generation, so a
GPU buys almost nothing.

## Two gates worth watching

**Reproduction.** The `gpt2` and `vioxx` row must come back at
D<sub>sync</sub> = 12.0464 nats, tolerance 0.05. The notebook asserts this in its
Section 7 and stops if it fails, because nothing downstream is trustworthy once
that row moves.

**Tokenization.** `answer_first_piece` should be the expected word piece for every
model. Answer tokens are obtained by differencing the prompt against the prompt
plus the answer, never by encoding the answer alone: a SentencePiece vocabulary
turns the leading space of `" Charles"` and `" Queen"` into the same standalone
piece, and every comparison between them then collapses to zero.

Every model runs in float32 on every device. Casting logits to float32 after a
half-precision forward pass does not recover the precision that pass discarded, so
a sweep that lets dtype follow the hardware is not comparable across machines.

## Where the results appear

Section 8.5 of the paper is the single-model case, Section 8.6 the cross-model
sweep and the paraphrase spread. The four charts are in `paper/figures/`, one of
them as Figure 4 and the other three in the repository README.
