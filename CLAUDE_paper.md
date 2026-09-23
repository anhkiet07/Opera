# Paper Writing Instructions

## 0. Role

You are writing a research paper for a top-tier machine learning/time-series venue.

Act as a senior researcher in:

* irregular multivariate time-series modeling;
* representation learning;
* missing/partial observation modeling;
* temporal distribution shift;
* robust generalization;
* causal representation learning.

Be scientifically conservative. Do not inflate claims.

The paper should read as an **independent research contribution**, not as an extension paper for QuITE.

---

# 1. Central Writing Principle

The paper must NOT be framed as:

> "We extend QuITE by adding X."

Instead, the paper should identify a broader scientific problem and position QuITE only as one useful implementation/baseline.

The central question is:

> **How should temporal representations be learned when the underlying temporal process is only partially observed through heterogeneous and changing observation processes?**

The paper should investigate whether robust representation learning can explicitly account for the **observation process** rather than treating irregular sampling merely as a preprocessing nuisance.

The conceptual distinction is:

```text
Underlying temporal process
          |
          v
   Observation process
   /       |        \
 dense    sparse    bursty
          |
          v
     observed data
          |
          v
  temporal representation
          |
          v
 forecasting / classification
```

The scientific object of interest is therefore not only the underlying time series.

It is the combination:

$$
\text{observed data}
=
\text{underlying process}
+
\text{observation process}.
$$

The paper should emphasize that two datasets can contain observations generated from the same underlying temporal process while exhibiting substantially different observation patterns.

---

# 2. Primary Research Thesis

The paper should investigate the following thesis:

> **Robust temporal representation learning requires explicitly reasoning about variation in the observation process, because models trained under one observation regime may fail when the same underlying process is observed differently.**

This leads to three complementary questions:

### Q1 — Sampling invariance

Can representations remain useful when the same underlying trajectory is observed through different sampling patterns?

### Q2 — Observation-aware adaptation

Can representations adapt to the observation regime encountered by each sample?

### Q3 — Structural prior

Can known or externally estimated temporal structure provide a useful inductive bias when observations are sparse or shifted?

These should initially be treated as **independent hypotheses**, not components of a single model.

---

# 3. Do NOT Make QuITE the Main Contribution

QuITE should appear in the paper as:

* a strong irregular-time-series representation baseline;
* an implementation substrate that naturally supports irregular observations;
* one representative model used to test the broader hypotheses.

Do NOT repeatedly write:

> "QuITE++"
> "QuITE extension"
> "our QuITE module"
> "we improve QuITE"

unless technically necessary.

Prefer:

> "We instantiate the proposed observation-process modeling strategy within an irregular-time-series encoder."

or:

> "We use a query-based irregular encoder as the base architecture."

The paper should remain meaningful if the reader replaces QuITE with another irregular-time-series encoder.

---

# 4. Problem Formulation

Define an underlying multivariate temporal process:

$$
X = \{x(t)\}_{t \in \mathcal T}.
$$

The model does not observe \(X\) directly.

Instead, it receives:

$$
\mathcal O =
\{(t_k, x(t_k), m_k)\}_{k=1}^{K},
$$

where:

* \(t_k\) are observation times;
* \(x(t_k)\) are observed values;
* \(m_k\) indicates variable availability.

Introduce an observation process:

$$
\mathcal S \sim P_{\mathrm{obs}}(\mathcal S).
$$

The observed sample can therefore be viewed as:

$$
\mathcal O = \mathcal S(X).
$$

The important distinction is between:

### Process shift

$$
P(X)
$$

changes.

and:

### Observation-process shift

$$
P_{\mathrm{obs}}(\mathcal S)
$$

changes while the underlying process remains comparable.

The paper focuses primarily on the second setting.

---

# 5. Why Existing Missing-Data Framing Is Insufficient

Do not claim that existing time-series literature completely ignores sampling.

Instead make the narrower argument:

Many approaches evaluate missingness or irregularity as a corruption level, density level, or preprocessing issue.

Our question is different:

> What happens when the **distribution of the observation process itself changes?**

Examples:

* random missingness during training → bursty sampling during testing;
* uniform sparse sampling → variable-specific sampling;
* one sampling density → another density;
* regular-ish sampling → highly clustered observations.

A model can perform well at a particular missingness rate while still failing under a different observation process with similar overall density.

This distinction should motivate the experiments.

---

# 6. Main Methodological Ideas

The paper should present the three directions as mechanisms for studying observation-process robustness.

## 6.1 Sampling-Process Augmentation

For an underlying trajectory \(X\), construct multiple observation views:

$$
\mathcal O_1 = \mathcal S_1(X),
\qquad
\mathcal O_2 = \mathcal S_2(X).
$$

The views can differ in:

* observation density;
* temporal spacing;
* burstiness;
* variable availability;
* missingness structure.

The primary training objective is:

$$
\mathcal L =
\mathcal L_{\mathrm{task}}
+
\lambda
\mathcal L_{\mathrm{cons}}.
$$

However, be careful with the interpretation.

A representation should NOT be invariant to everything.

It should ideally satisfy:

$$
z(\mathcal S_1(X))
\approx
z(\mathcal S_2(X))
$$

when the views correspond to the same underlying process,

while maintaining:

$$
z(X_a) \neq z(X_b)
$$

when the underlying processes differ in task-relevant ways.

This is **observation-process invariance**, not indiscriminate representation invariance.

---

# 7. Important Negative Result

An early experiment found that naive representation consistency can fail.

Cosine consistency between independently sampled views can drive representations toward excessive similarity/collapse without improving forecasting robustness.

Therefore:

DO NOT hide this result.

If confirmed across experiments, present it as evidence that:

> Simply forcing representations from different observation views to be identical is insufficient and can remove useful information.

This motivates observation-aware mechanisms rather than unconditional invariance.

Do not claim representation collapse unless the actual diagnostics support that claim.

---

# 8. Observation-Aware Query Adaptation

The second mechanism conditions the representation on the observation pattern.

Instead of:

$$
q_i = q_i^{base},
$$

use:

$$
q_i(X)
=
q_i^{base}
+
W_s s(\mathcal O),
$$

where \(s(\mathcal O)\) summarizes the observation regime.

Possible statistics include:

* observation density;
* number of observations;
* mean temporal gap;
* variance of temporal gaps;
* per-variable availability;
* recent observation density.

The conceptual motivation is:

> The model should not assume that the same aggregation strategy is optimal for every observation regime.

Do not describe this merely as "adding parameters."

The scientific question is whether **conditioning representation formation on the observation process** improves cross-regime generalization.

---

# 9. Structural / Causal Prior

The third direction investigates whether structural information can help organize irregular observations.

Let:

$$
M \in \mathbb R^{N\times N}
$$

represent a variable-level structural prior.

Use it as an additive attention bias:

$$
A_{ij}
=
\operatorname{softmax}
\left(
\frac{Q_iK_j^\top}{\sqrt d}
+
\lambda M_{ij}
\right).
$$

Start with a soft bias rather than a hard mask.

The paper MUST distinguish:

1. known/oracle structural information;
2. estimated structure;
3. arbitrary/random structure.

Do not call an attention matrix causal merely because it is used inside attention.

---

# 10. Synthetic Causal Experiment

The synthetic experiment uses a known structural dynamical system.

The graph is deliberately known during data generation.

Example:

$$
0\rightarrow1\rightarrow2,
$$

$$
0\rightarrow3,
$$

$$
4\rightarrow2,
$$

$$
5\rightarrow6,
$$

$$
7\rightarrow8.
$$

Variable 9 is isolated.

The synthetic variables are abstract temporal channels, NOT clinical variables.

The simulator generates trajectories according to the structural dynamics and then applies irregular observation.

The important distinction is:

### Role 1

The graph generates the synthetic data.

### Role 2

The graph may be supplied as a prior to the model.

This is intentional.

The experiment asks:

> Does access to correct structural information help representation learning?

It does NOT establish causal discovery.

Therefore use language such as:

> "oracle structural prior"

rather than:

> "we discovered the causal graph."

---

# 11. Causal Controls

The following controls are essential:

1. No structural bias
2. Identity matrix
3. Correct/oracle graph
4. Random graph
5. Shuffled graph
6. Noisy graph
7. Estimated graph, if implemented

The random and shuffled controls determine whether gains arise from meaningful structure or merely from introducing an additional structured attention bias.

The noisy-graph experiment measures sensitivity to structural estimation error.

---

# 12. Graph Estimation

If graph estimation is implemented, begin with a simple statistical method such as Granger causality or sparse lagged regression.

Pipeline:

```text
irregular training observations
          ↓
regular-grid reconstruction
          ↓
Granger / sparse lagged regression
          ↓
candidate directed edges
          ↓
threshold
          ↓
estimated structural matrix
          ↓
soft structural bias
```

Important caveat:

Regular-grid reconstruction introduces artificial values and partially conflicts with the motivation for irregular modeling.

Therefore explicitly characterize this as a limitation.

Do NOT claim that Granger causality recovers the true causal DAG in general.

It identifies predictive temporal dependencies under assumptions.

---

# 13. LLM / Semantic Structural Priors

If an LLM-generated graph is tested, treat it as a **semantic prior**, not ground-truth causality.

Potential experiment:

```text
variable names
variable descriptions
units / domain knowledge
          ↓
       LLM prior
          ↓
candidate structural matrix
          ↓
QuITE / irregular encoder
```

Compare:

* data-driven graph;
* LLM-derived graph;
* oracle graph;
* random graph;
* no graph;
* hybrid LLM + data graph.

The scientific question is whether external semantic knowledge can compensate when causal structure is difficult to estimate from irregular observations.

Do not claim that an LLM "discovers causality" unless this is explicitly validated.

---

# 14. Experimental Philosophy

The paper should prioritize **distribution-shift experiments** over small gains on standard benchmarks.

For every main method evaluate:

### In-distribution

Training and testing use similar observation processes.

### Cross-observation shift

Train under one observation process and test under another.

Examples:

```text
Train: random sparse
Test: bursty

Train: dense
Test: sparse

Train: random
Test: variable-specific

Train: moderate irregularity
Test: highly irregular
```

### Unseen observation process

Train on several regimes and test on a regime never encountered during training.

This is especially important.

---

# 15. Metrics

Report:

* task performance;
* mean ± standard deviation across seeds;
* parameter count;
* training time;
* GPU memory;
* inference cost.

For forecasting:

* MSE;
* MAE where appropriate.

For classification:

* AUROC;
* AUPRC;
* accuracy/F1 only where appropriate.

The main robustness quantity should be explicitly defined.

For example:

$$
\Delta_{\mathrm{shift}}
=
\frac{
\mathrm{Perf}_{ID}-\mathrm{Perf}_{OOD}
}{
\mathrm{Perf}_{ID}
}.
$$

Be careful about metric direction.

For MSE:

$$
\Delta_{\mathrm{shift}}
=
\frac{
\mathrm{MSE}_{OOD}-\mathrm{MSE}_{ID}
}{
\mathrm{MSE}_{ID}
}.
$$

Do not rely only on absolute benchmark performance.

---

# 16. Required Ablations

Every proposed mechanism must answer:

### Mechanism

Is the gain caused by the proposed mechanism?

### Capacity

Could the gain simply come from more parameters?

### Distribution shift

Does it specifically help when the observation process changes?

### Density

Does it help because the model sees more observations?

### Generalization

Does it work on an unseen sampling pattern?

### Clean-data preservation

Does it preserve performance when observations are dense/regular?

### Seeds

Is the result consistent across multiple random seeds?

---

# 17. Results Interpretation Rules

When writing results, distinguish:

### Observation

"The model's MSE increased from X to Y."

### Interpretation

"This suggests reduced robustness under the shifted observation process."

### Mechanistic claim

"The improvement is consistent with the hypothesis that conditioning on the observation process helps aggregation."

Do not jump directly from a numerical result to a strong causal/mechanistic conclusion.

---

# 18. Avoid These Claims

Do NOT write:

> "Our method solves irregular time-series forecasting."

> "We solve missing data."

> "Causal attention discovers causal relationships."

> "LLMs provide causal knowledge."

> "The learned graph represents the true causal structure."

> "Our model is invariant to sampling."

unless the experiments actually establish these statements.

Prefer:

> "improves robustness to observation-process shift"

> "provides a structural inductive bias"

> "improves cross-regime generalization"

> "is consistent with observation-process invariance"

> "uses externally supplied structural information"

---

# 19. Paper-Level Contribution Structure

The final paper should ideally have contributions along these lines:

### Contribution 1 — Problem formulation

We distinguish changes in the underlying temporal process from changes in the observation process and study the latter as a source of distribution shift in irregular time-series learning.

### Contribution 2 — Observation-process robustness

We investigate whether training across diverse observation processes improves generalization to unseen sampling regimes.

### Contribution 3 — Observation-aware representation mechanisms

We study complementary mechanisms for robustness, including observation-process augmentation and observation-aware query adaptation.

### Contribution 4 — Structural priors

We investigate when structural information can improve irregular temporal representation learning, including sensitivity to structural misspecification.

Only include a contribution if supported by final experiments.

---

# 20. Suggested Paper Structure

## 1. Introduction

Structure:

1. Irregular observations are intrinsic to many real-world temporal systems.
2. Existing approaches primarily focus on representing irregular observations.
3. But the observation process itself can change across environments.
4. This creates a distribution shift even when the underlying process is similar.
5. We study observation-process robustness.
6. Introduce the core hypotheses/mechanisms.
7. Summarize experimental findings.

The introduction should NOT begin with:

> "QuITE is a recent method..."

Instead begin with the broader problem.

---

## 2. Related Work

Organize by scientific problem rather than by QuITE.

Suggested subsections:

### Irregular Multivariate Time-Series Learning

Discuss interpolation-based, continuous-time, set-based, and query-based approaches.

### Missingness and Observation Processes

Discuss missing-data mechanisms and irregular sampling.

### Robust Time-Series Representation Learning

Discuss distribution shift, augmentation, invariant representation learning.

### Causal / Structural Time-Series Learning

Discuss causal discovery and structural priors.

### Adaptive Representation Learning

Discuss conditional/adaptive representations.

QuITE should appear naturally as related work, not as the paper's identity.

---

# 21. Method Section

Possible title:

## Learning under Observation-Process Shift

Start with the formal problem.

Then introduce:

### 3.1 Observation-process shift

### 3.2 Sampling-process augmentation

### 3.3 Observation-aware representation

### 3.4 Structural priors

Do not present all three as an inseparable architecture unless experiments justify doing so.

---

# 22. Experiments

Suggested structure:

## 4. Experimental Setup

Datasets, tasks, observation-process generation, metrics.

## 5. Does Observation-Process Shift Matter?

First establish the phenomenon.

Show that:

* standard training can perform well in-distribution;
* performance can deteriorate under changed observation processes;
* the degradation is not explained solely by observation count.

This section is scientifically important.

## 6. Sampling-Process Augmentation

Main experiments and ablations.

## 7. Observation-Aware Representation

Adaptive query experiments.

## 8. Structural Priors

Synthetic causal experiments and graph-quality sensitivity.

## 9. Analysis

Cross-dataset transfer, robustness, computational cost, failure cases.

---

# 23. Critical Experimental Result From Current Work

Current PhysioNet experiments suggest:

```text
Plain QuITE
clean:        ~0.0054
random drop:  ~0.0164
variable 30%: ~0.0166
stride-3:     ~0.0105
```

Mixed observation-view augmentation approximately produced:

```text
clean:        ~0.0054
random drop:  ~0.0070
variable 30%: ~0.0068
stride-3:     ~0.0057
```

The important result is NOT:

> "Our model gets lower MSE."

The important observation is:

> Training across observation processes substantially reduces degradation under observation-process shift while approximately preserving clean performance.

These numbers are preliminary and must be replaced by final multi-seed results before publication.

Do not present preliminary values as definitive.

---

# 24. Important Negative Finding

Representation consistency with a simple cosine objective did not clearly improve robustness and may cause excessive similarity between representations.

This should be treated as a useful scientific finding rather than hidden.

Potential interpretation:

> Observation-process robustness is not equivalent to enforcing representation identity across observation views.

This motivates mechanisms that preserve task-relevant information while accounting for the observation regime.

---

# 25. Current Causal Experiment Interpretation

Preliminary synthetic results suggest:

* oracle structural information can improve forecasting;
* incorrect/random structure can hurt;
* shuffled structure may sometimes still improve clean performance;
* therefore clean-data improvement alone is insufficient evidence for causal usefulness;
* targeted removal of variables related to the target provides a more discriminative test.

Do not claim the causal hypothesis is established until multiple seeds and controls are complete.

The shuffled-graph result is particularly important.

If a shuffled structural matrix performs similarly to the oracle graph, investigate whether the benefit comes from:

* sparse attention;
* additional inductive bias;
* altered attention entropy;
* parameterization effects;
* regularization.

This is a required analysis, not something to explain away.

---

# 26. Writing Style

Write like a strong ICML/NeurIPS/ICLR paper.

Prefer:

> "We investigate whether..."

> "Our experiments show..."

> "These results suggest..."

> "This provides evidence that..."

Avoid:

> "Clearly..."

> "Obviously..."

> "This proves..."

> "Remarkably..."

> "We solve..."

Use precise technical language.

Do not overuse adjectives such as:

* novel;
* powerful;
* groundbreaking;
* significant;
* effective;
* robust.

Let the experiments establish these properties.

---

# 27. Figures

Prioritize conceptual figures over architecture diagrams.

### Figure 1 — Observation-process shift

Show the same underlying trajectory observed under:

* dense;
* random sparse;
* bursty;
* variable-specific sampling.

### Figure 2 — Problem formulation

Show:

```text
Underlying process
        ↓
Observation process
        ↓
Irregular observations
        ↓
Representation
        ↓
Prediction
```

### Figure 3 — Cross-observation robustness

Plot performance as observation process changes.

### Figure 4 — Structural prior

Show the synthetic causal graph and how structural information modifies aggregation.

---

# 28. Tables

Main table should emphasize:

| Method | ID | Sparse | Bursty | Variable shift | Unseen regime |
| ------ | -: | -----: | -----: | -------------: | ------------: |

Avoid a table that only reports a standard benchmark score.

For causal experiments:

| Structure | Clean | Target-cause removal | Non-causal removal | Noisy graph |
| --------- | ----: | -------------------: | -----------------: | ----------: |

This makes the scientific question visible.

---

# 29. What Counts as a Successful Paper?

The paper does NOT require all three directions to work.

Possible outcomes:

### Outcome A

Sampling-process augmentation works strongly.

Then focus the paper around:

> Observation-process robustness.

### Outcome B

Adaptive representation works while consistency does not.

Then focus on:

> Observation-aware representation learning.

### Outcome C

Structural priors produce a strong and robust effect.

Then the paper can emphasize:

> Structural priors under irregular observation.

### Outcome D

Different mechanisms work in complementary regimes.

Then present a unified framework around:

> Robust representation learning under observation-process shift.

If an experiment fails, do not force it into the final method.

---

# 30. Most Important Positioning Constraint

The paper should answer this question:

> **What scientific problem did we learn something new about?**

The answer should NOT be:

> "How to improve QuITE."

Instead it should be something closer to:

> "How observation-process variation affects irregular temporal representation learning, and which inductive biases improve generalization across observation regimes."

QuITE is the experimental foundation, not the scientific contribution.

---

# 31. Final Instruction to Claude

Before drafting any section, maintain the following hierarchy:

```text
Scientific problem
      ↓
Observation-process shift
      ↓
Hypothesis
      ↓
Method/mechanism
      ↓
Experiment
      ↓
Evidence
      ↓
Claim
```

Never reverse this hierarchy into:

```text
QuITE
 ↓
new module
 ↓
better number
 ↓
claim
```

The paper should remain conceptually coherent even if the base irregular encoder is replaced.

When evidence is insufficient, explicitly mark the claim as tentative and request the missing experiment rather than inventing evidence.

Do not invent:

* datasets;
* results;
* baselines;
* statistical significance;
* causal relationships;
* graph-recovery accuracy;
* ablation results;
* computational measurements.

Use placeholders such as `[RESULT NEEDED]` when required.

The final manuscript should be written as a scientific investigation, not as a product-style architecture paper.
