# CLAUDE.md

# Observation-Process Robust Representation Learning for Irregular Time Series

## 0. Mission

You are implementing a complete research framework for robust representation learning in irregular time series.

The central hypothesis is:

> **Irregular time-series models can suffer from observation-process shift: the underlying temporal process may remain the same while the way it is observed changes. Robustness can be improved by explicitly training across diverse observation processes, transferring information from richer observations, and actively training against difficult observation processes.**

The goal is to build and experimentally validate a **single coherent framework**, not a collection of unrelated techniques.

The framework should be implemented from end to end:

* data generation / observation-process simulation
* observation-process representation
* observation augmentation
* privileged-view teacher/student distillation
* hard observation-process mining
* model integration
* training
* evaluation
* ablations
* reproducibility
* tables/figures
* experiment tracking
* cleanup of obsolete experiments
* final scientific report

Do not preserve obsolete methods merely because they were implemented previously.

---

# 1. IMPORTANT: RESET THE OLD EXPERIMENTAL DIRECTION

The repository currently contains experiments from an older research direction involving:

* Sampling-Invariant QuITE
* representation consistency losses
* Adaptive QuITE
* Causal QuITE
* causal attention biases
* true/random/shuffled causal graphs
* causal graph discovery / structural priors
* experiments whose purpose was to explain negative results

These are NOT the core of the new framework.

## Remove / archive old experiments

The implementation should explicitly identify and remove or archive:

### A. Representation consistency

Remove experiments based on:

```text
cosine consistency
representation matching between observation views
cross-view embedding equality
lambda * cosine_loss
```

The previous observation was that enforcing representation similarity between views did not provide meaningful improvement and could collapse representations toward excessive similarity.

Do not make this a component of the new method.

If useful for historical reproducibility, move the code/configuration to:

```text
archive/old_consistency/
```

Do not import it into the new framework.

---

### B. Causal QuITE

Remove the causal module from the main framework:

```text
causal attention matrix
true DAG
random DAG
shuffled DAG
identity graph
noisy graph
learned causal matrix
```

Move old implementations to:

```text
archive/old_causal/
```

The new paper is NOT about causal discovery, causal attention, or causal priors.

Do not spend implementation or experiment time reproducing those results unless explicitly requested.

---

### C. Old paper claims

Search the repository for references to:

```text
Sampling-Invariant QuITE
Adaptive QuITE
Causal QuITE
causal graph
causal attention
representation consistency
sampling invariance
invariance loss
```

Remove these from active documentation, experiment descriptions, README files, configs, and generated reports.

Do not allow stale terminology to appear in new experiment reports.

---

# 2. NEW FRAMEWORK

Use the following conceptual model throughout the codebase.

We assume an underlying temporal process:

$$
X \sim P(X)
$$

and an observation process:

$$
S \sim P(S)
$$

which produces observations:

$$
O = S(X).
$$

The model receives:

$$
O
$$

and predicts:

$$
Y = f_\theta(O).
$$

The important distinction is:

$$
P_{\text{train}}(X) \approx P_{\text{test}}(X)
$$

but potentially

$$
P_{\text{train}}(S) \neq P_{\text{test}}(S).
$$

This is **observation-process shift**.

Examples:

* random missingness
* long temporal gaps
* irregular sampling
* bursty sampling
* variable-specific missingness
* correlated variable dropout
* different observation densities
* different sampling strides
* combinations of the above

The framework should explicitly model:

```text
Underlying trajectory
        |
        v
Observation Process
        |
        v
Observed irregular time series
        |
        v
Robust representation learner
        |
        v
Prediction
```

---

# 3. FRAMEWORK NAME

Use the working name:

# OPERA

## Observation-Process Robust Representation Learning

Full paper title:

> **OPERA: Robust Representation Learning for Irregular Time Series under Observation-Process Shift**

If a better name emerges during implementation, do not change the name casually. Keep `OPERA` throughout the implementation unless there is a strong technical reason to rename it.

---

# 4. CORE FRAMEWORK

OPERA has three mechanisms.

## 4.1 Observation-Process Augmentation

Generate multiple observation views of the same underlying trajectory.

Given:

$$
X
$$

sample:

$$
S_1,S_2,\ldots,S_K \sim P_{\text{obs}}
$$

and produce:

$$
O_k=S_k(X).
$$

Train on multiple observation processes.

The base objective is:

$$
\mathcal L_{\text{task}}
=
\frac{1}{K}
\sum_k
\mathcal L
\left(
f_\theta(O_k),Y
\right).
$$

Important:

Do NOT require:

$$
z(O_1)=z(O_2).
$$

The goal is task robustness, not forced representation equality.

---

# 5. Observation-Process Generator

Implement a modular observation-process generator.

Create something conceptually equivalent to:

```text
ObservationProcessGenerator
```

with composable transformations.

At minimum support:

```text
RandomDropout
TemporalStride
BurstMissingness
LongGapMissingness
VariableDropout
VariableGroupDropout
LocalTemporalMask
DensityControlledSampling
MixedObservationProcess
```

Every generated observation should preserve enough metadata to describe the observation process.

For each observed sample store:

```python
ObservationMetadata(
    density,
    mean_gap,
    max_gap,
    gap_std,
    variable_coverage,
    temporal_coverage,
    missing_fraction,
    per_variable_density,
    sampling_regime,
)
```

Do not hard-code these statistics into individual experiments.

---

# 6. Observation-Process Representation

Implement a common observation descriptor:

$$
r = g(S).
$$

This should summarize the current observation process.

At minimum:

```text
global observation density
mean temporal gap
maximum temporal gap
gap variance
number of observed variables
variable coverage
per-variable observation density
temporal coverage
```

Normalize all continuous statistics.

The representation should be available to the model but should NOT automatically be used everywhere.

We need to test when observation information is useful.

---

# 7. MECHANISM 1 — OBSERVATION-PROCESS AUGMENTATION

This is the first component and should be treated as the baseline OPERA mechanism.

Training:

```text
X
 |
 +--> S1 --> O1 --> encoder --> prediction
 |
 +--> S2 --> O2 --> encoder --> prediction
 |
 +--> S3 --> O3 --> encoder --> prediction
```

All views correspond to the same underlying trajectory and target.

Loss:

$$
\mathcal L_{\text{aug}}
=
\frac{1}{K}
\sum_{k=1}^{K}
\mathcal L_{\text{task}}(f_\theta(O_k),Y).
$$

No representation consistency loss.

---

# 8. MECHANISM 2 — PRIVILEGED OBSERVATION DISTILLATION

This is a central new component.

The intuition:

A sparse observation may not contain enough information for the model to infer the target reliably.

But during training, we may have access to a richer observation of the SAME underlying trajectory.

Use the richer observation as a teacher.

Important:

The teacher is a TRAINING-TIME teacher.

The student must operate using only the sparse observation at inference.

---

## 8.1 Teacher/student setup

Generate:

```text
X
|
+---- dense observation ----> Teacher
|
+---- sparse observation ---> Student
```

Teacher:

$$
p_T = f_{\theta_T}(O_{\text{rich}})
$$

Student:

$$
p_S = f_{\theta}(O_{\text{sparse}}).
$$

The student is trained with:

$$
\mathcal L
=
\mathcal L_{\text{task}}(p_S,Y)
+
\lambda_{\text{distill}}
\mathcal L_{\text{distill}}(p_S,p_T).
$$

Teacher output must be detached:

```python
teacher_prediction = teacher(...).detach()
```

unless explicitly implementing an EMA teacher.

---

# 9. Teacher Design

Implement two teacher modes.

## Mode A — Frozen rich-view teacher

Train a strong teacher on rich observations first.

Then freeze it.

```text
teacher = pretrained rich-observation model
student = sparse-observation model
```

## Mode B — EMA teacher

Maintain:

$$
\theta_T
\leftarrow
\alpha\theta_T+(1-\alpha)\theta_S.
$$

Evaluate both.

Do not assume EMA is superior.

---

# 10. Distillation Loss

Implement multiple distillation choices.

### Forecasting

For regression:

$$
\mathcal L_{\text{distill}}
=
\|\hat Y_S-\operatorname{stopgrad}(\hat Y_T)\|_2^2.
$$

For probabilistic prediction, support KL divergence.

### Representation distillation

Implement as an optional experiment only:

$$
\mathcal L_{\text{repr}}
=
D(z_S,z_T).
$$

But do NOT force exact equality as the main method.

The primary distillation experiment should be **task-level prediction distillation**.

This avoids repeating the previous failed consistency formulation.

---

# 11. Important anti-leakage rule

Teacher and student must observe the SAME prediction history.

For forecasting:

```text
history window
    |
    +--> rich observation --> teacher
    |
    +--> sparse observation --> student
```

Neither teacher nor student may access future observations.

Do not accidentally give the teacher future context.

Add an explicit unit test verifying this.

---

# 12. MECHANISM 3 — HARD OBSERVATION-PROCESS MINING

The third component actively searches for difficult observation processes.

Instead of sampling observation processes uniformly:

$$
S\sim P(S),
$$

we want:

$$
S^*
=
\arg\max_S
\mathcal L(f_\theta(S(X)),Y)
$$

subject to valid observation constraints.

The intuition:

> Do not only train on random missingness. Train on the observation processes that actually expose weaknesses in the representation.

---

# 13. Practical implementation of hard mining

Do NOT begin with a complicated differentiable mask generator.

First implement:

## Candidate-pool hard mining

For each training example:

1. Generate N candidate observation processes.
2. Run the current model on all candidates.
3. Calculate task loss.
4. Select top-k hardest.
5. Backpropagate only through the selected candidates.

Example:

```text
X
|
+--> S1 --> loss 0.02
+--> S2 --> loss 0.08
+--> S3 --> loss 0.31  <-- hard
+--> S4 --> loss 0.17
+--> S5 --> loss 0.05

select S3, S4
```

Train on those.

This is easier to stabilize and gives a clean first implementation.

---

# 14. Hard-mining constraints

The hard observation process must remain realistic.

Never allow the adversary to simply delete everything.

Implement configurable constraints:

```text
minimum observation density
maximum observation density
minimum variable coverage
maximum temporal gap
maximum missing fraction
per-variable minimum coverage
```

For example:

$$
\rho_{\min}
\leq
\rho(S)
\leq
\rho_{\max}.
$$

And:

$$
\Delta t_{\max}(S)\leq G_{\max}.
$$

These constraints must be configurable.

---

# 15. Hard-mining curriculum

Implement:

```text
Stage 1:
random observation augmentation

Stage 2:
candidate hard mining

Stage 3:
mixed random + hard observations
```

Do not train exclusively on hard cases.

Use:

$$
P(S)
=
(1-\alpha)P_{\text{random}}
+
\alpha P_{\text{hard}}.
$$

Sweep:

```text
alpha = 0
alpha = 0.25
alpha = 0.5
alpha = 0.75
alpha = 1
```

---

# 16. FULL OPERA MODEL

The final framework should support:

```text
                    underlying X
                       |
          +------------+------------+
          |                         |
      rich view                 sparse views
          |                         |
      Teacher              +--------+--------+
          |                |        |        |
       p_teacher          S1       S2       Shard
          |                |        |        |
          |              Student encoder
          |                |
          |          prediction heads
          |                |
          +------ distillation
                           |
                    task prediction
```

The training system combines:

### A. Observation augmentation

$$
\mathcal L_{\text{task}}
$$

### B. Privileged-view distillation

$$
\lambda_d
\mathcal L_{\text{distill}}
$$

### C. Hard observation mining

Choose difficult \(S\)'s during training.

Total:

$$
\boxed{
\mathcal L
=
\mathcal L_{\text{task}}
+
\lambda_d
\mathcal L_{\text{distill}}
}
$$

where hard mining controls which observation processes contribute to:

$$
\mathcal L_{\text{task}}.
$$

Do not introduce unnecessary additional losses.

---

# 17. IMPORTANT: NO REPRESENTATION CONSISTENCY LOSS

Do NOT implement:

```python
cosine(z1, z2)
MSE(z1, z2)
contrastive(z1, z2)
```

as the core objective.

The scientific distinction is important:

### Old idea

"Different observation views should produce the same representation."

### New idea

"Different observation processes should produce representations that remain useful for the downstream task."

And:

"Richer observations can provide privileged task supervision to sparse observations."

This is a much cleaner hypothesis.

---

# 18. MODEL ARCHITECTURE

The framework must remain backbone-agnostic.

Create an interface:

```python
BaseIrregularEncoder
```

and support the existing:

```text
iTransformer
```

first.

Then add at least one second architecture if the initial experiments work.

Potential second backbone:

```text
PatchTST
TimesFM-compatible encoder
Transformer encoder
GRU/TCN baseline
```

Prefer an architecture that can naturally consume irregular timestamps.

Do not redesign the backbone unnecessarily.

The observation-process machinery should sit around the encoder.

---

# 19. INPUT FORMAT

The encoder should receive:

```text
values
timestamps
variable_ids
observation_mask
```

Prefer explicit observation masks.

Do not encode missingness merely as zeros.

For each token/observation provide:

```text
value
timestamp
delta_t
variable_id
observation_mask
```

---

# 20. OBSERVATION-AWARE QUERY MODULE

The old Adaptive QuITE idea can be retained only if it provides useful information inside the new framework.

Rename it:

```text
ObservationAwareEncoder
```

or:

```text
ObservationAwareQuery
```

It should condition aggregation on:

```text
observation density
gap statistics
variable coverage
local temporal density
```

But do NOT make this a separate paper contribution initially.

Treat it as an optional module.

The primary framework must work without it.

---

# 21. EXPERIMENTAL MATRIX

The minimum experiment matrix is:

## E0 — Vanilla baseline

```text
standard training
single observation regime
```

## E1 — Observation augmentation

```text
E0 + mixed observation processes
```

## E2 — Privileged distillation

```text
E1 + rich-view teacher
```

## E3 — Hard observation mining

```text
E1 + hard mining
```

## E4 — Full OPERA

```text
E1 + E2 + E3
```

Optional:

```text
E5 = Full OPERA + observation-aware encoder
```

---

# 22. REQUIRED TEST CONDITIONS

Evaluate on:

### Clean

Original observation process.

### Random missingness

For example:

```text
30%
50%
70%
```

### Variable removal

For example:

```text
30% variables
50% variables
```

### Temporal stride

For example:

```text
stride 2
stride 3
stride 4
```

### Long temporal gaps

Controlled maximum gap.

### Burst missingness

Contiguous missing segments.

### Variable-specific observation shift

Different variables have different sampling densities.

### Mixed shift

Combine:

```text
temporal gaps
+
variable dropout
+
density shift
```

---

# 23. TRAIN/TEST OBSERVATION-PROCESS SPLITS

This is critical.

Do not merely train and test on the same missingness distribution.

Define:

```text
Train observation processes
Test observation processes
```

with distinct distributions.

Example:

```text
TRAIN:
random dropout 10–40%
stride 1–2
moderate gaps

TEST:
random dropout 50–70%
stride 3–4
long gaps
burst missingness
```

Also test interpolation/generalization:

```text
train on A + B
test on unseen C
```

The purpose is to measure genuine observation-process generalization.

---

# 24. DATASETS

First reproduce the existing PhysioNet forecasting setup.

Use the existing preprocessing if valid.

Do not silently modify preprocessing.

Document:

```text
history length
forecast horizon
number of variables
sampling interval
normalization
train/validation/test split
```

Then add at least one second dataset.

Preferred:

```text
USHCN
```

or:

```text
Activity
```

Choose the dataset that naturally supports irregular-observation experiments.

The framework should not be considered validated from one dataset alone.

---

# 25. EXISTING BASELINE RESULTS

The repository contains historical results approximately showing:

```text
Vanilla iTransformer:
clean ≈ 0.0178
random 70% ≈ 0.0347
variable removal ≈ 0.0416
stride-3 ≈ 0.0208

Plain QuITE:
clean ≈ 0.0054
random ≈ 0.0164
variable ≈ 0.0166
stride-3 ≈ 0.0105

Mixed observation views:
clean ≈ 0.0054
random ≈ 0.0070
variable ≈ 0.0068
stride-3 ≈ 0.0057
```

These numbers are HISTORICAL.

Do not automatically copy them into the new paper.

Re-run the required experiments under the new standardized protocol.

Do not treat old 2-seed experiments as final evidence.

---

# 26. RANDOM SEEDS

All primary experiments require:

```text
5 random seeds
```

Use:

```text
0
1
2
3
4
```

unless there is a compelling repository constraint.

Report:

```text
mean ± std
```

Never report a single lucky seed as the main result.

---

# 27. STATISTICAL REPORTING

For each main result report:

```text
mean
standard deviation
relative degradation from clean
relative improvement over baseline
```

For example:

$$
\text{degradation}
=
\frac{MSE_{\text{shift}}-MSE_{\text{clean}}}
{MSE_{\text{clean}}}.
$$

The main metric should remain the actual task metric.

Do not invent a robustness score unless it provides clear scientific value.

---

# 28. REQUIRED ABLATIONS

At minimum:

### Observation augmentation

```text
none
random
temporal
variable
mixed
```

### Distillation

```text
none
prediction distillation
representation distillation
prediction + representation
```

Prediction distillation should be the main candidate.

### Teacher quality

```text
weak teacher
strong teacher
frozen teacher
EMA teacher
```

### Teacher density

```text
1x
2x
4x
dense
```

### Hard mining

```text
random sampling
hard mining
random + hard
```

### Hardness pool size

```text
K = 4
K = 8
K = 16
K = 32
```

### Hard-mining ratio

```text
alpha = 0
0.25
0.5
0.75
1.0
```

### Observation-process diversity

Test whether simply increasing the number of observation processes improves robustness.

---

# 29. IMPORTANT ABLATION: DENSITY VS STRUCTURE

This is scientifically important.

Two observation processes may have the same:

$$
\text{observation density}
$$

but very different temporal structure.

Construct matched-density experiments:

```text
same density
different gap distribution
```

and:

```text
same density
different variable coverage
```

and:

```text
same density
different burstiness
```

Measure degradation.

This tests whether observation-process shift is more than simply "more missing values."

---

# 30. IMPORTANT ABLATION: UNSEEN OBSERVATION PROCESSES

Train:

```text
A + B
```

Test:

```text
C
```

where C was never used during training.

This is one of the most important experiments.

The desired result is that OPERA improves not only on known corruption types but also on unseen observation processes.

---

# 31. COMPUTATIONAL BUDGET

Track:

```text
training time
GPU memory
number of forward passes
parameter count
inference latency
```

The deployment model should NOT require the teacher.

The deployment path must be:

```text
sparse observation
      |
   student
      |
 prediction
```

No rich observation is needed at inference.

No candidate-pool hard mining is needed at inference.

---

# 32. REQUIRED SOFTWARE ARCHITECTURE

Organize approximately as:

```text
src/
    observation/
        generator.py
        processes.py
        metadata.py
        constraints.py

    models/
        base.py
        iTransformer.py
        observation_aware.py

    distillation/
        teacher.py
        losses.py
        ema.py

    mining/
        candidate_generator.py
        hard_miner.py
        constraints.py

    training/
        trainer.py
        objectives.py
        scheduler.py

    evaluation/
        observation_shift.py
        robustness.py
        metrics.py

    datasets/
        physionet.py
        ushcn.py

configs/
    baseline.yaml
    augmentation.yaml
    distillation.yaml
    hard_mining.yaml
    opera.yaml

experiments/
    ...
    
scripts/
    train.py
    evaluate.py
    generate_tables.py
    generate_figures.py

archive/
    old_consistency/
    old_causal/
```

Keep experiment logic separate from model logic.

---

# 33. CONFIGURATION

Everything important must be configurable.

Example:

```yaml
observation:
  processes:
    - random_dropout
    - temporal_stride
    - burst_missingness
    - variable_dropout

  num_views: 4

distillation:
  enabled: true
  teacher_type: frozen
  loss: prediction
  weight: 1.0
  rich_density: 0.9

hard_mining:
  enabled: true
  candidate_pool: 16
  top_k: 4
  alpha: 0.5

model:
  backbone: itransformer
  observation_aware: false
```

Do not hard-code experiment settings in Python.

---

# 34. REPRODUCIBILITY

Every experiment must save:

```text
config.yaml
git commit
random seed
dataset version
model checkpoint
training log
validation metrics
test metrics
observation-process configuration
```

A result should be reproducible from a single command.

Example:

```bash
python scripts/train.py \
    --config configs/opera.yaml \
    --seed 0
```

---

# 35. AUTOMATED EXPERIMENT RUNNER

Implement a launcher capable of running:

```text
all seeds
all baselines
all observation regimes
all ablations
```

and aggregating results.

Generate:

```text
results/raw/
results/aggregated/
figures/
tables/
```

---

# 36. REQUIRED FIGURES

At minimum produce:

## Figure 1 — Framework

Show:

```text
underlying trajectory
        |
 observation process
        |
 sparse view ------------------+
        |                       |
      student               rich teacher
        |                       |
        +---- distillation -----+
        |
 prediction

and hard observation mining around the observation-process layer.
```

## Figure 2 — Observation-process shift

Same underlying trajectory under different observation processes.

## Figure 3 — Robustness curve

Performance versus observation severity.

## Figure 4 — Unseen observation-process generalization

Train processes vs unseen test processes.

## Figure 5 — Ablation

Contribution of:

```text
augmentation
distillation
hard mining
full OPERA
```

---

# 37. REQUIRED TABLES

### Table 1

Main benchmark:

```text
Method
Clean
Random
Variable
Stride
Long-gap
Burst
Unseen
```

### Table 2

Component ablation:

```text
Baseline
+ Augmentation
+ Distillation
+ Hard Mining
+ Full OPERA
```

### Table 3

Cross-dataset generalization.

### Table 4

Compute/parameter overhead.

---

# 38. SUCCESS CRITERIA

Do NOT declare success merely because Full OPERA improves one test condition.

The framework should satisfy:

### Criterion 1

Improves robustness under observation-process shift.

### Criterion 2

Improves on observation processes not explicitly seen during training.

### Criterion 3

Works across at least two datasets.

### Criterion 4

Works across at least two model architectures if computationally feasible.

### Criterion 5

Each major component has a measurable contribution.

### Criterion 6

The deployment model requires no privileged observations.

### Criterion 7

Improvements are statistically stable over 5 seeds.

---

# 39. FAILURE HANDLING

If one component does not help:

DO NOT immediately create another arbitrary module.

First diagnose:

```text
Is the teacher actually better?
Is the rich observation genuinely informative?
Is the distillation target appropriate?
Is the hard mining producing realistic masks?
Is the candidate pool too weak?
Is the model already robust due to augmentation?
```

If a component consistently contributes nothing across datasets, remove it from the main framework.

The goal is a scientifically coherent framework, not maximizing the number of modules.

---

# 40. PRIORITY ORDER

Implement in this order.

## Phase A — Cleanup

1. Inspect repository.
2. Identify old QuITE/causal/consistency code.
3. Archive obsolete code.
4. Remove stale documentation.
5. Establish clean experiment directory.

## Phase B — Observation abstraction

1. Implement observation-process interface.
2. Implement all observation generators.
3. Implement metadata.
4. Implement constraints.
5. Add tests.

## Phase C — Baseline

Reproduce:

```text
vanilla
plain model
mixed observation augmentation
```

with 5 seeds.

## Phase D — Distillation

Implement:

```text
rich teacher
sparse student
prediction distillation
```

Run a small proof-of-concept first.

## Phase E — Hard mining

Implement candidate-pool hard mining.

Run small proof-of-concept.

## Phase F — Full OPERA

Combine:

```text
augmentation
+
distillation
+
hard mining
```

## Phase G — Generalization

Run:

```text
unseen observation process
second dataset
second backbone
```

## Phase H — Final experiments

Run all seeds and generate final tables/figures.

---

# 41. DO NOT OVERENGINEER EARLY

The first prototype should be simple.

The minimum viable OPERA implementation is:

```text
Observation augmentation
+
Frozen rich-view teacher
+
Prediction distillation
+
Candidate-pool hard mining
```

Do not initially implement:

```text
RL mask generator
Gumbel mask optimization
complex meta-learning
causal discovery
contrastive learning
representation consistency
large auxiliary networks
```

Only add these if experiments demonstrate a specific unresolved limitation.

---

# 42. SCIENTIFIC QUESTIONS THE IMPLEMENTATION MUST ANSWER

The experiment suite should ultimately answer:

### Q1

Does observation-process shift cause substantial degradation even when the underlying trajectory distribution is unchanged?

### Q2

Does training across observation processes improve robustness?

### Q3

Can a richer observation act as useful privileged information for sparse-view representation learning?

### Q4

Does actively mining difficult observation processes improve robustness beyond random augmentation?

### Q5

Do these mechanisms complement each other?

### Q6

Does robustness transfer to unseen observation processes?

### Q7

Is the benefit caused simply by observation density, or by diversity in the structure of the observation process?

### Q8

Does the framework generalize across datasets and architectures?

These questions should drive the experiments.

---

# 43. FINAL PAPER MESSAGE

Do NOT write the paper as:

> We introduce three modules.

Instead build the scientific story around:

> **Observation-process shift is an important source of distribution shift in irregular time series. We develop OPERA, a framework that explicitly trains representations against this shift by exposing models to diverse observation processes, transferring task-relevant information from richer observations, and focusing training on difficult observation regimes.**

The mechanisms are subordinate to the scientific problem.

---

# 44. OLD EXPERIMENT INFORMATION MUST NOT LEAK INTO THE NEW PAPER

Search generated artifacts, Markdown files, LaTeX files, experiment summaries, and notebooks for old terminology.

Remove or archive references to:

```text
Sampling-Invariant QuITE
Causal QuITE
Adaptive QuITE
causal attention
causal graph
representation consistency
consistency collapse
negative causal result
```

unless the information is explicitly required for historical reproducibility.

The new paper should NOT contain a section explaining failed causal experiments.

Do not construct the paper around negative results.

The previous experiments can exist in:

```text
archive/
```

but should not contaminate the active implementation.

---

# 45. FINAL DELIVERABLES

At completion, provide:

```text
1. Cleaned repository
2. OPERA implementation
3. Unit tests
4. Config files
5. Reproducible training scripts
6. Reproducible evaluation scripts
7. Experiment launcher
8. Raw results
9. Aggregated results
10. Figures
11. Tables
12. Experiment summary
13. Failure analysis
14. Recommended final model configuration
15. Draft technical description of the final framework
```

The technical description should explicitly state:

```text
what problem is being solved
why observation-process shift matters
how observation augmentation works
how privileged distillation works
how hard observation mining works
why the mechanisms are complementary
how inference works
```

---

# 46. MOST IMPORTANT IMPLEMENTATION PRINCIPLE

Do not optimize for making every proposed component work.

Optimize for discovering whether this scientific hypothesis is correct:

> **A model trained to explicitly account for variation in the observation process can learn representations that are substantially more robust to observation-process shift than a model trained under a fixed observation regime.**

If augmentation alone solves most of the problem, report that honestly.

If distillation adds substantial gains, make it central.

If hard mining adds substantial gains, make it central.

If a component does not add value after proper testing, remove it.

The final framework should contain only mechanisms that are experimentally justified.

# END
