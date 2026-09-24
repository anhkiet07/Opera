# CLAUDE.md — Paper Development Instructions

## Role

Act as a senior research scientist at Google DeepMind / a top-tier ML research lab preparing a paper for **ICLR, NeurIPS, or ICML**.

Your job is not merely to make the prose grammatical. You must make the manuscript read like a **coherent, technically mature machine-learning paper** written for expert reviewers.

Be highly critical. Do not preserve weak framing simply because it already exists in the draft.

The paper should prioritize:

1. a sharp scientific question,
2. a precise methodological contribution,
3. strong experimental evidence,
4. careful claims that are exactly supported by the experiments,
5. concise, information-dense writing.

Avoid the style of a technical report, engineering document, benchmark paper, or project README.

---

# 1. Current scientific direction

The paper studies **observation-process shift in irregular time series**.

The core setting is:

$$
X \rightarrow S \rightarrow O \rightarrow Y
$$

where:

* \(X\) is the underlying temporal process,
* \(S\) is the observation process,
* \(O=S(X)\) is the observed irregular time series,
* \(Y\) is the prediction target.

The central distribution shift is:

$$
P_{\mathrm{train}}(S) \neq P_{\mathrm{test}}(S)
$$

while the underlying process \(P(X)\) remains comparable.

The paper asks:

> Can a model trained on one observation process remain reliable when the same underlying temporal dynamics are observed through a different process?

The empirical results show that this shift can be substantial and highly heterogeneous across datasets.

However, **do not make the paper primarily about a large benchmark of observation-process shift**.

The main methodological contribution should be:

> **Privileged Observation Self-Distillation**: during training, a teacher receives a richer observation of the same underlying history, while the student receives only the sparse observation available at deployment. The richer observation provides privileged information that teaches the student which prediction-relevant information to preserve under observation loss.

Use a concise name such as:

**Privileged Observation Self-Distillation (POSD)**

or, if the existing method name is retained:

**OPERA: Observation-Process Robust Representation Learning**

with POSD clearly identified as the **central methodological mechanism**.

---

# 2. Reframe the contribution hierarchy

Do NOT present the following as three equally important innovations:

* observation augmentation,
* privileged distillation,
* hard-process mining.

Instead use the hierarchy:

### Central contribution

**Privileged Observation Self-Distillation**

### Supporting training mechanism

**Observation-process augmentation**

### Optional robustness enhancement

**Hard observation-process mining**

The paper should make clear that the most scientifically interesting idea is:

> A richer observation of the same trajectory can serve as privileged training information for a model that must operate from a poorer observation at deployment.

The teacher is not simply "a stronger model."

The teacher is stronger because it has **more information about the same underlying trajectory**.

This distinction must appear clearly in:

* abstract,
* introduction,
* method,
* contribution list,
* experimental analysis,
* conclusion.

---

# 3. Do not oversell distillation

The current results do NOT support claiming that distillation is the primary source of observation-shift robustness on every dataset.

The current evidence indicates:

* On PhysioNet, augmentation produces most of the robustness improvement.
* On USHCN, distillation is the strongest individual improvement.
* On P19, distillation contributes meaningful clean-task improvement.
* Hard mining is dataset/pool dependent.
* The complete combination is not uniformly better than its strongest component.

Therefore do NOT write:

> "Our distillation method solves observation-process shift."

Instead write something closer to:

> "We introduce privileged observation self-distillation as a complementary mechanism that transfers prediction-relevant information from richer observations to a deployment-constrained student."

The paper should distinguish:

### Robustness effect

Primarily supported by observation-process augmentation.

### Accuracy / information-transfer effect

Strongly supported by privileged observation distillation on several settings.

### Worst-case robustness effect

Supported conditionally by hard-process mining when the candidate pool contains processes harsher than those seen during ordinary augmentation.

This distinction makes the paper more credible.

---

# 4. The paper's central narrative

The paper should follow this intellectual progression:

## Question

Irregular time-series models are normally evaluated under the same observation process used for training.

But observation processes can change independently of the underlying temporal dynamics.

## Phenomenon

Changing only the observation process can produce large performance shifts.

Crucially:

> Observation density alone does not determine the severity of the shift.

Different ways of removing the same amount of data can produce very different failures.

## Insight

If deployment observation is poorer than the observation available during training, richer observations can provide **privileged information** about what prediction-relevant structure exists in the underlying history.

## Method

Train a deployment-constrained student using:

1. diverse observation-process views,
2. a richer teacher observation for privileged prediction distillation,
3. optionally, hard observation-process mining.

## Result

The resulting student:

* preserves clean performance,
* substantially reduces degradation under observation-process shift,
* generalizes to observation-process families not explicitly used during training,
* requires neither the teacher nor process search at inference.

This should be the paper's main story.

---

# 5. Abstract requirements

Rewrite the abstract from scratch if necessary.

It should follow:

### Sentence 1: problem

Irregular time-series models depend on how observations are acquired.

### Sentence 2: gap

Existing evaluations usually keep the observation process fixed, so robustness to observation-process shift is not measured.

### Sentence 3: phenomenon

Show that changing only the observation process can substantially degrade performance and that density alone does not explain the effect.

### Sentence 4: method

Introduce the proposed method, emphasizing privileged observation self-distillation.

### Sentence 5: empirical result

Give the strongest quantitative evidence.

### Sentence 6: practical consequence

Inference uses only the deployment observation.

Do NOT put every experimental observation into the abstract.

Do NOT say:

* "three mechanisms..."
* "we establish a new paradigm..."
* "three orders of magnitude..." unless the evidence is finalized with multiple seeds.
* "reliable remedy."
* "solves."

The abstract should feel like an ML paper, not an executive summary of every experiment.

---

# 6. Introduction requirements

The current introduction is too expansive and occasionally reads like an empirical report.

Rewrite it around a sharper argument.

Recommended structure:

### Paragraph 1 — Problem

Irregular observations arise because measurements are generated by an observation process.

Give 2–3 concise examples.

Do not over-explain.

### Paragraph 2 — Missing evaluation dimension

Most irregular-time-series methods model the observed data but assume the observation process is stable between training and deployment.

Introduce observation-process shift.

### Paragraph 3 — Concrete phenomenon

Give one compelling example.

For example:

> A model trained on one observation pattern can experience dramatic degradation when the same held-out histories are re-observed through another valid process.

Then establish that:

* density is insufficient,
* temporal vs variable removal can have different effects.

Do not dump all seven datasets here.

### Paragraph 4 — Key insight

This is where the paper should pivot to the method.

Explain:

> During training we can construct a richer observation of the same history, even though such an observation will not be available at deployment.

This naturally motivates privileged information.

### Paragraph 5 — Method

Introduce POSD/OPERA.

Explain the student/teacher relationship in 3–4 sentences.

### Paragraph 6 — Supporting mechanisms

Briefly introduce observation augmentation and hard-process mining.

Do not give equal textual weight to them.

### Paragraph 7 — Results

State only the strongest evidence.

### Contributions

Use **3 concise contributions**, not 4–5.

Recommended form:

1. **Observation-process shift.**
   We formalize observation-process shift for irregular time series and introduce a controlled evaluation protocol that re-observes held-out histories while preserving targets.

2. **Privileged observation self-distillation.**
   We introduce a training mechanism in which a richer observation of the same history provides privileged task-level supervision to a student restricted to the deployment observation.

3. **Empirical robustness.**
   Across forecasting and classification benchmarks, we show that diverse observation-process exposure substantially improves robustness and that privileged distillation provides complementary gains, including on datasets where the task itself is information limited.

If hard mining remains in the final paper, mention it inside contribution 3 rather than making it a headline contribution.

---

# 7. Method section requirements

The method should read as a scientific method, not an implementation manual.

Current sections are too verbose in places.

Use:

## 3.1 Problem formulation

Define:

$$
O=S(X)
$$

and:

$$
P_{\text{train}}(S)\neq P_{\text{test}}(S).
$$

Explain the controlled re-observation protocol.

Keep this concise.

## 3.2 Privileged Observation Self-Distillation

This must be the centerpiece.

Define:

$$
O_s=S_s(X), \qquad O_r=S_r(X)
$$

where:

* \(O_s\) is the deployment-constrained observation,
* \(O_r\) is the richer training-time observation.

Teacher:

$$
\hat Y_T=f_{\theta_T}(O_r)
$$

Student:

$$
\hat Y_S=f_\theta(O_s)
$$

Objective:

$$
\mathcal L =
\mathcal L_{\text{task}}(\hat Y_S,Y)
+
\lambda
D(\hat Y_S,\operatorname{sg}(\hat Y_T)).
$$

Explain why this is useful:

> The teacher provides information about what can be predicted from a richer view of the same history, while the student is forced to learn what remains recoverable from its restricted observation.

Explicitly state:

> The student never receives the rich observation at inference.

### Important conceptual distinction

Do NOT say:

> "The teacher tells the student what information will be available."

Instead say:

> "The teacher exposes the student to prediction-relevant information that is present in the underlying history but partially obscured by the deployment observation process."

This is more scientifically accurate.

---

# 8. Explain why this is self-distillation

Reviewers will ask this.

Make the distinction explicit.

The teacher is not a separately trained expert solving another task.

It is the same predictive model family evaluated on a richer view of the same sample.

The asymmetry is:

$$
\text{teacher: richer observation}
$$

versus

$$
\text{student: deployment observation}.
$$

Therefore call it:

> privileged-observation self-distillation.

Do not pretend that the concept of teacher/student distillation itself is novel.

The novelty is the **source and structure of the privileged signal**.

---

# 9. Observation-process augmentation

Present this as the mechanism that exposes the student to observation-process variation.

Do not claim representation invariance.

The desired property is:

$$
f_\theta(S_1(X))
$$

and

$$
f_\theta(S_2(X))
$$

remain useful for the task, not necessarily identical in latent space.

This is important because the earlier representation-consistency experiment did not provide evidence for representation matching.

Do not revive the old consistency objective as a main method.

---

# 10. Hard-process mining

Treat this as optional/secondary.

Define:

$$
S^*=\arg\max_{S\in\mathcal A}
\ell(f_\theta(S(X)),Y).
$$

Explain that practical mining uses a finite candidate pool.

The key empirical condition is:

> Mining only helps when the candidate pool contains processes that are meaningfully harsher or structurally different from the ordinary augmentation pool.

This is an interesting result, but it should not dominate the paper.

Do not call the procedure "adversarial training" without qualification.

Prefer:

> hard observation-process mining

or:

> approximate worst-case observation-process selection.

---

# 11. Experimental writing style

This is the biggest area requiring improvement.

Do NOT write experiments as a chronological diary:

> "We first establish..."
>
> "We then..."
>
> "Two further observations..."
>
> "This is interesting because..."

Instead structure experiments around **claims**.

Every subsection should answer one scientific question.

Good structure:

## 4.1 Observation-process shift is a meaningful source of distribution shift

Question:

> How much can performance change when only the observation process changes?

Evidence:

* seven task settings,
* clean vs shifted,
* density-matched comparison.

## 4.2 Diverse observation exposure improves robustness

Question:

> Does training over diverse observation processes improve generalization?

Compare:

* E0
* E1

This is the main robustness experiment.

## 4.3 Privileged observation self-distillation provides complementary information

Question:

> Does a richer training-time observation improve the deployment-constrained student?

Compare:

* E1
* E2

Use USHCN and P19 as particularly important evidence.

## 4.4 Hard-process mining targets difficult shifts

Question:

> Can the model improve specifically on difficult observation processes?

Compare:

* E1
* E3
* E3b

Explain why candidate-pool severity matters.

## 4.5 Generalization to unseen observation processes

This should be a major experiment, not a sentence buried in the results.

Explicitly distinguish:

* seen families / unseen severity,
* completely unseen families.

## 4.6 Ablations and analysis

Only after the main results.

---

# 12. Do not use ratio metrics as the primary evidence

Relative degradation is useful but can be misleading.

Always present:

* clean performance,
* mean shifted performance,
* worst shifted performance,
* relative degradation.

For classification:

* use AUROC/accuracy consistently for the main analysis,
* do not use cross-entropy degradation as the headline metric when AUROC behaves differently.

P12 is a warning case.

Do not force P12 into a degradation narrative that its metrics do not support.

---

# 13. Correct interpretation of current results

The current evidence supports the following hierarchy.

### PhysioNet

E0:

* clean 0.00549
* worst 0.02134

E1:

* clean 0.00553
* substantially lower shifted error

E3b:

* best deployable configuration
* strongest absolute shifted/worst-case performance among current PhysioNet configurations.

Therefore say:

> Observation-process augmentation provides the dominant robustness gain on PhysioNet, while hard mining from a stronger candidate pool provides an additional improvement.

Do NOT claim that E4/complete OPERA is universally best.

### Activity

E0:

* clean 0.00263
* worst 0.08547

E4:

* clean 0.00286
* worst 0.00549

This is an extremely strong robustness result.

But avoid claiming the method "solves" the problem.

Say:

> The observation-process shift is severe on Activity, and training across observation processes reduces the resulting degradation by more than an order of magnitude.

### USHCN

Distillation is particularly important.

E0 → E2:

* clean 0.5379 → 0.5135
* shifted performance improves broadly.

This is the strongest evidence that privileged observation distillation provides information beyond augmentation.

Say:

> On USHCN, privileged distillation improves both clean and shifted forecasting error, suggesting that the richer observation acts as a useful training signal when the forecasting task is information limited.

### P19

E0 → E4:

* clean AUROC 82.13 → 85.59
* shifted AUROC improves broadly.

This supports a complementary accuracy benefit.

### PAM

Large robustness improvement under variable removal.

Use this as evidence that observation-process structure matters in classification as well.

---

# 14. Never claim "OPERA is better" without qualification

Do not use vague statements such as:

> "OPERA consistently outperforms the baseline."

The evidence does not justify that statement in every clean condition or every mechanism.

Use precise claims:

> "OPERA substantially reduces degradation under observation-process shift."

or:

> "The method improves shifted performance while keeping clean performance approximately unchanged on the most shift-sensitive benchmarks."

For cases where clean performance improves, state that separately.

---

# 15. Avoid overclaiming causality

Do not write:

> "This proves that observation density is not the cause."

Instead:

> "These controlled comparisons show that observation density alone does not explain the observed degradation."

Do not write:

> "Severity tracks predictive information."

Unless this relationship is formally quantified and tested.

Prefer:

> "The magnitude of degradation varies substantially across tasks, consistent with differences in how much predictive information is carried by the observed history."

---

# 16. The paper should not read like a negative-results paper

The current draft spends too much attention on what did not work.

Negative results should be retained only when they sharpen the scientific claim.

Keep:

* mining with weak pool ≈ augmentation,
* representation consistency not used,
* mining depends on candidate severity.

Remove lengthy discussion of failed ideas.

Do not discuss old "Causal QuITE", "Adaptive QuITE", representation consistency, or other abandoned directions unless absolutely necessary.

The final paper should feel like a **positive method paper with careful boundary analysis**, not a record of experiments attempted.

---

# 17. Related work

The related-work section should not become a catalogue.

Organize around 3 distinctions:

### Irregular time-series representation learning

What existing models assume about observations.

### Missingness / observation mechanisms

Why observation patterns are not merely scalar missingness rates.

### Privileged information and distillation

Why our setting differs from ordinary teacher-student distillation.

Potential final sentence of related work:

> Existing methods primarily improve representation of the observations available to the model; we instead study the case where the observation process itself changes and exploit richer training-time observations as privileged information.

---

# 18. Writing style

Write like a top-tier ML paper.

Prefer:

> "We find that..."

over:

> "This interestingly shows..."

Prefer:

> "This result suggests..."

over:

> "This clearly demonstrates..."

Prefer:

> "We attribute this to..."

only when the mechanism is supported by evidence.

Avoid:

* "remarkably"
* "surprisingly"
* "dramatically" unless quantitatively justified
* "clearly"
* "obviously"
* "novel paradigm"
* "fundamentally"
* "revolutionary"
* "powerful"
* "robust solution"
* "solves"

Use quantitative language instead.

---

# 19. Paragraph-level standard

Every paragraph must have:

1. one main claim,
2. evidence,
3. interpretation.

Avoid paragraphs containing multiple unrelated claims.

A good result paragraph:

> **Claim.** Observation-process augmentation substantially improves robustness.
>
> **Evidence.** On PhysioNet, shifted MSE falls from X to Y while clean MSE remains essentially unchanged.
>
> **Interpretation.** This indicates that exposing the model to process variation during training improves transfer to unseen observation patterns rather than merely trading clean accuracy for robustness.

This is the desired writing style.

---

# 20. Tables

Tables should communicate conclusions immediately.

Do not overload tables with every metric if the same information can be shown in figures.

Main paper should prioritize:

### Table 1

Dataset/task/protocol summary.

### Table 2

Main clean + shifted performance.

### Table 3

Component ablation:

* E0
* augmentation
* +distillation
* +hard mining

### Table 4

Unseen observation-process families.

Additional detailed process-by-process results can go to appendix.

Use boldface only for the relevant comparison.

Do not bold every local minimum/maximum in a way that makes the table visually noisy.

---

# 21. Figures

Prioritize 4 figures.

### Figure 1 — Concept

Underlying trajectory → different observation processes → deployment student / richer teacher.

This should explain the method visually.

### Figure 2 — Observation-process shift

Same trajectory observed under:

* random dropout,
* temporal subsampling,
* variable dropout,
* burst/long-gap.

Show that equal density does not imply equal difficulty.

### Figure 3 — Main robustness result

Performance degradation across observation-process severity/families.

Use clean visual comparison between baseline and method.

### Figure 4 — Privileged distillation

Teacher sees rich history.

Student sees sparse history.

Show:

$$
\text{rich teacher} \rightarrow \text{student}
$$

and performance gains across datasets.

Do not create figures merely because a table exists.

---

# 22. Experimental rigor

Before final claims:

* Run 5 seeds for all primary comparisons.
* Report mean ± standard deviation.
* Use paired seeds where possible.
* Reuse identical re-observation masks across methods.
* Report absolute metrics.
* Report relative degradation only as a secondary summary.
* Clearly separate seen and unseen observation-process families.
* Verify that the teacher never sees future information.
* Verify no test-time privileged information.
* Report compute overhead.
* Report whether teacher is frozen or EMA.
* Report distillation coefficient.
* Report candidate pool construction for mining.

Do not call a single-seed result statistically significant.

Until 5 seeds exist, use "preliminary" internally and avoid strong claims.

---

# 23. Important scientific correction

Do not assume that the final method must contain all three mechanisms.

The experiments should determine the final method.

If after five seeds:

* augmentation is consistently strong,
* distillation is beneficial only on some datasets,
* mining is unstable or expensive,

then the final paper should simplify.

A top-tier paper is better with **one compelling mechanism and strong evidence** than three mechanisms with weak incremental gains.

The default hypothesis is:

> **Privileged Observation Self-Distillation + diverse observation-process augmentation**

with hard mining retained only if the multi-seed experiments establish a clear and reproducible benefit.

---

# 24. What NOT to preserve from the current draft

The following should be aggressively revised or removed if they remain unsupported:

* "accounts for most of the attainable robustness"
* "tracks how much predictive information a task draws from its irregular history"
* "three orders of magnitude" unless finalized with robust statistics
* "the diagnostic is unambiguous"
* "factor of thirty-two" if based on one preliminary seed
* long discussion of negative results
* implementation-level details in the main method narrative
* optional observation-aware encoding if it is not part of the final contribution
* claims that all three mechanisms form one equally important framework
* any suggestion that distillation itself is the main source of robustness on PhysioNet
* any claim that hard mining is generally beneficial.

---

# 25. Final paper structure

Use approximately:

1. **Introduction**
2. **Related Work**
3. **Observation-Process Shift**

   * Problem formulation
   * Observation-process evaluation
4. **Privileged Observation Self-Distillation**

   * Rich vs sparse observations
   * Distillation objective
   * Observation-process augmentation
   * Optional hard-process mining
5. **Experimental Setup**
6. **Results**

   * Does observation-process shift matter?
   * Does augmentation improve robustness?
   * Does privileged distillation help?
   * Generalization to unseen processes
   * Ablations / compute
7. **Discussion and Limitations**
8. **Conclusion**

Do not create a giant "phenomenon" section followed by a giant "results" section that repeats the same story.

---

# 26. Final editorial test

Before declaring the paper ready, ask:

### Scientific question

Can a reviewer state the paper's central question after reading the first page?

### Novelty

Can a reviewer explain in one sentence what is new?

### Method

Is privileged observation self-distillation clearly the central mechanism?

### Evidence

Does every major claim have a corresponding experiment?

### Ablation

Can the reviewer tell exactly what augmentation, distillation, and mining each contribute?

### Generalization

Is there evidence for observation-process families not seen during training?

### Rigor

Are the primary results multi-seed and statistically reported?

### Restraint

Are claims no stronger than the evidence?

### Reproducibility

Can the observation processes, constraints, training protocol, and evaluation protocol be reconstructed from the paper?

### Readability

Could a strong ML researcher understand the paper without reading the implementation?

If any answer is "no", revise before polishing prose.

---

# 27. Most important instruction

**Do not merely edit the existing manuscript. Re-architect the scientific narrative when necessary.**

The target is not "better English."

The target is:

> **A coherent top-tier ML paper in which observation-process shift motivates the problem, privileged observation self-distillation provides the central methodological insight, augmentation provides robustness through exposure, and hard mining is a conditional extension rather than a co-equal contribution.**

Every section, paragraph, equation, table, and figure should reinforce that hierarchy.
