# Mathematical Derivations: Provenance Gating and Lifecycle Desynchronization

Supplementary derivations for *The Knowledge Lifecycle of Large Language Models*. Notation follows the paper.

## 1. The FFN as a Key-Value Memory

Following Geva et al. (2021), a Transformer feed-forward network computes

$$ \text{FFN}(x) = f(x W_{in}) W_{out} = \sum_{i=1}^{d_m} f(x \cdot k_i)\, v_i $$

where $k_i \in \mathbb{R}^{d_{model}}$ is the key (row $i$ of $W_{in}$), $v_i \in \mathbb{R}^{d_{model}}$ is the value (column $i$ of $W_{out}$), and $f$ is the activation function.

**Failure mode.** If a fact is outdated, the activation $f(x \cdot k_i)$ is still high (the model recognizes the concept), and the stale value $v_i$ is added to the residual stream, steering the output toward the outdated answer.

## 2. The Provenance Vector

Each memory slot is extended from a pair $(k_i, v_i)$ to a triple $(k_i, v_i, p_i)$, where

$$ p_i = E_p([\tau_i; s_i]) \in \mathbb{R}^{d_p} $$

encodes the acquisition timestamp $\tau_i$ and a source reliability score $s_i$. The provenance store is metadata: it does not enter the residual stream, so the FFN's output dimensionality is unchanged. Its sole consumer is the inference-time gate below.

### 2.1 The Gating Function

With $\Delta \tau_i = T_{now} - \tau_i \geq 0$ the age of slot $i$:

$$ S(\Delta \tau_i) = \sigma(\gamma - \beta \cdot \Delta \tau_i) = \frac{1}{1 + \exp(\beta \cdot \Delta \tau_i - \gamma)} $$

The gate decreases monotonically with age: fresh slots pass with $S \approx \sigma(\gamma) \approx 1$, stale slots are attenuated toward 0. $\beta > 0$ controls how sharply the gate closes with age; $\gamma > 0$ sets the freshness threshold (a slot is attenuated below one half once $\Delta\tau_i > \gamma/\beta$).

### 2.2 The Adaptive Threshold

$\gamma$ need not be static. A reasoning model can compute it during its reflection loop from the entropy of its own predictive distribution: detected conflict (high entropy, divergent candidate continuations) lowers $\gamma$, making the gate stricter for that query. This couples conflict resolution to test-time compute.

### 2.3 Idealized Consistency Result

In the residual stream at layer $L$:

$$ z_{L} = z_{L-1} + \text{Attention}(z_{L-1}) + \text{FFN}_{steered}(z_{L-1}) $$

**Assumption 1 (stale support).** Every slot contributing to the outdated answer has $\Delta\tau_i > \gamma/\beta$; every slot contributing to other necessary computation has $\Delta\tau_i < \gamma/\beta$.

**Assumption 2 (context sufficiency).** The retrieved context contains the updated fact, and with the conflicting parametric contribution suppressed, the output distribution on the query is determined by attention over that context; call it $P_{ctx}$.

**Proposition.** Under Assumptions 1 and 2, as $\beta \to \infty$ the gated model's output distribution converges to $P_{ctx}$, hence $D_{KL}(P_{ctx} \| P_{steered}) \to 0$.

*Proof sketch.* For slots with $\Delta\tau_i > \gamma/\beta$ the gate argument $\gamma - \beta\Delta\tau_i \to -\infty$, so $S \to 0$; for slots with $\Delta\tau_i < \gamma/\beta$ it tends to $+\infty$, so $S \to 1$. Exactly the stale slots are removed from the FFN sum; by Assumption 2 the output distribution is then $P_{ctx}$, and the KL of a distribution against itself is zero.

The assumptions are strong by design: they state exactly what a real implementation must approximate, namely provenance that separates fact slots from infrastructure slots (Assumption 1), and a retriever that found the update at all (Assumption 2).

## 3. Lifecycle Desynchronization

### 3.1 Definition

$$ \mathcal{D}_{sync}(q) = D_{KL}\left( P_{sync}(y|q) \,\|\, P(y \mid q, \theta_{stale}, \mathcal{M}_{now}) \right) $$

where $P_{sync}$ is the output distribution of a hypothetical system whose weights agree with the current external store $\mathcal{M}_{now}$.

### 3.2 Point-Mass Estimator

$P_{sync}$ is not directly observable. For factual queries with a single verifiable answer $y^*$, approximate it with a point mass $\delta_{y^*}$. The KL divergence from a point mass reduces to a surprisal:

$$ \widehat{\mathcal{D}}_{sync}(q) = D_{KL}(\delta_{y^*} \| P) = \sum_y \delta_{y^*}(y) \ln \frac{\delta_{y^*}(y)}{P(y)} = -\ln P(y^* \mid q, \theta_{stale}, \mathcal{M}_{now}) $$

A value of $n$ nats means the deployed system assigns the correct answer probability $e^{-n}$.

### 3.3 Context Influence

$$ \mathcal{I}_{ctx}(q) = D_{KL}\left( P(y \mid q, \mathcal{M}_{now}) \,\|\, P(y \mid q) \right) $$

computed over the full vocabulary. Small $\mathcal{I}_{ctx}$ with large $\widehat{\mathcal{D}}_{sync}$ means the corrective context is present but inert: the failure is in conflict resolution, not retrieval.

### 3.4 Measured Values (Vioxx probe, GPT-2 base)

| Quantity | Value |
| :--- | ---: |
| $P(\text{" safe"})$ without context | 37.53% |
| $P(\text{" safe"})$ with withdrawal context | 42.58% |
| $P(\text{" withdrawn"})$ with withdrawal context | $5.9 \times 10^{-6}$ |
| $\mathcal{I}_{ctx}$ | 0.033 nats |
| $\widehat{\mathcal{D}}_{sync}$ | 12.05 nats |

All values produced by the deterministic script `dsync_experiment.py` in the repository root.
