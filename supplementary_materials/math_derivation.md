# Mathematical Derivation: Provenance Steering and Lifecycle Desynchronisation

This document provides the granular mathematical derivations required to defend the **Knowledge Lifecycle** framework.

## 1. The FFN as a Key-Value Memory
Following Geva et al. (2021), a Feed-Forward Network (FFN) in a Transformer computes:
$$ \text{FFN}(x) = \sigma(x W_{in}) W_{out} $$
We decompose this into $d_m$ individual neurons (memory cells):
$$ \text{FFN}(x) = \sum_{i=1}^{d_m} f(x \cdot k_i) v_i $$
where:
- $k_i \in \mathbb{R}^{d_{model}}$ is the **Key** (the input weight row $i$).
- $v_i \in \mathbb{R}^{d_{model}}$ is the **Value** (the output weight column $i$).
- $f$ is the activation function (e.g., GeLU).

**Failure Mode:** If a fact is outdated, the activation $f(x \cdot k_i)$ is high (the model "recognises" the concept), and the stale value $v_i$ is added to the residual stream, polluting the output.

---

## 2. The Provenance Vector Intervention
We propose modifying the value vector $v_i$ to include a metadata dimension $p_i$:
$$ v_i \rightarrow [v_i; p_i] $$
The Provenance Vector $p_i = [\tau_i; s_i]$ encodes:
- $\tau_i$: The acquisition timestamp (normalised).
- $s_i$: The source reliability score.

### 2.1 The Steering Gating Function
During inference-time compute (System 2), the model computes a temporal delta:
$$ \Delta \tau_i = T_{now} - \tau_i $$
We define the steering gate $S(\Delta \tau_i)$ as:
$$ S(\Delta \tau_i) = \frac{1}{1 + \exp(\beta \cdot \Delta \tau_i - \gamma)} $$
As $\Delta \tau_i$ (age) increases, $S \rightarrow 0$. The hyperparameter $\beta$ controls the "aggressiveness" of the filter.

### 2.2 The Dynamic Threshold ($\gamma$)
The parameter $\gamma$ is not a static constant. It is a **test-time variable** computed by the reasoning model's reflection loop:
$$ \gamma = \text{MLP}_{reason}(\text{Entropy}(P_{actual})) $$
If the model detects high epistemic uncertainty (entropy), it raises $\gamma$ to be more aggressive in filtering parametric noise. This makes the architecture **adaptive** to the difficulty of the query.

### 2.3 Proof of Conflict Resolution
In the residual stream at layer $L$:
$$ z_{L} = z_{L-1} + \text{Attention}(z_{L-1}) + \text{FFN}_{steered}(z_{L-1}) $$
If the Attention mechanism (RAG) retrieves a document $\mathcal{M}$ that contradicts $v_i$, and $v_i$ is "old" ($\Delta \tau_i$ is high), then $S(\Delta \tau_i) \approx 0$.
The FFN contribution to the residual stream for that specific fact becomes:
$$ S(\Delta \tau_i) \cdot v_i \approx 0 $$
**Conclusion:** The stale parametric fact is effectively "muted," allowing the RAG-updated attention signal to dominate the hidden state without destructive interference.

---

## 3. Lifecycle Desynchronisation ($\mathcal{D}_{sync}$)
We define desynchronisation as the information loss between an ideal system and the actual system.

### 3.1 Formal Definition
$$ \mathcal{D}_{sync}(q) = \sum_{y \in \mathcal{V}} P_{synced}(y|q) \log \left( \frac{P_{synced}(y|q)}{P_{actual}(y|q)} \right) $$
where $P_{synced}$ is the distribution if the model had been perfectly updated with the latest knowledge.

### 3.2 Limit Behavior
If the Provenance Steering is applied with $\beta \rightarrow \infty$:
$$ P_{actual}(y|q) \rightarrow P_{RAG}(y|q) $$
$$ \mathcal{D}_{sync} \rightarrow D_{KL}(P_{RAG} \| P_{RAG}) = 0 $$
**Defensible Claim:** The Provenance Vector is a structural solution that reduces Lifecycle Desynchronisation to its mathematical lower bound in the limit of test-time compute.
