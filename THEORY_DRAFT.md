# Why OPERA works: a theoretical account

Draft for the paper. Kept outside `paper/` deliberately, since that directory is mid-revision elsewhere.

Scope: the two mechanisms the evidence supports — observation-process augmentation and privileged
distillation. Hard mining is deliberately out of scope here; the measurements for it remain in
`results/benchmark.csv` under the `opera++` arm.

Each claim below is paired with the measurement that tests it. Where the data does not support the theory,
that is stated rather than omitted — the point of writing this is to find out which parts survive.

---

## 1. Setup

An underlying trajectory and its target are drawn from a fixed process,

$$(X, Y) \sim \mathcal{D},$$

and an **observation process** $S$ is drawn independently from a distribution $Q$ over a space
$\mathcal{S}$ of measurable coarsening operators. The learner never sees $X$; it sees

$$O = S(X).$$

For a predictor $f_\theta$ and loss $\ell$, define the risk **under a given observation-process distribution**

$$
R_Q(\theta) \;=\; \mathbb{E}_{(X,Y)\sim\mathcal D}\,\mathbb{E}_{S\sim Q}\,
\big[\ell\big(f_\theta(S(X)),\,Y\big)\big].
$$

**Observation-process shift** is the regime

$$
\mathcal{D}_{\text{train}} = \mathcal{D}_{\text{test}},
\qquad
Q_{\text{train}} \neq Q_{\text{test}} .
$$

The trajectory law is unchanged; only the way it is observed changes. This is the object the benchmark
manipulates: every evaluation condition applies a different $S$ to the *same* held-out histories, with the
re-observation seed fixed, so conditions are paired per test input.

Two derived quantities are used throughout. For a fixed $\theta$, write $L(S) = \mathbb{E}_{\mathcal D}\,
\ell(f_\theta(S(X)),Y)$ for the risk at a single observation process, and define the **observation
sensitivity**

$$
\Delta(\theta) \;=\; \mathbb{E}_{S\sim Q_{\text{test}}}[L(S)] \;-\; L(S_{\text{clean}}),
$$

which is exactly what the benchmark reports as degradation.

---

## 2. Mechanism 1 — augmentation reduces the mismatch term

Training draws $S \sim Q_{\text{train}}$. Writing the test risk in terms of the training risk,

$$
R_{Q_{\text{test}}}(\theta)
= R_{Q_{\text{train}}}(\theta)
+ \int_{\mathcal S} L(S)\,\mathrm{d}\big(Q_{\text{test}} - Q_{\text{train}}\big)(S).
$$

If $\ell$ is bounded by $B$, the second term admits the standard control

$$
\Big|\int L \,\mathrm{d}(Q_{\text{test}}-Q_{\text{train}})\Big|
\;\le\; 2B\,\mathrm{TV}\big(Q_{\text{test}}, Q_{\text{train}}\big),
$$

and more sharply, since only the variation of $L$ across $\mathcal S$ matters,

$$
\Big|\int L \,\mathrm{d}(Q_{\text{test}}-Q_{\text{train}})\Big|
\;\le\; \mathrm{osc}_{\mathcal S}(L)\cdot \mathrm{TV}(Q_{\text{test}},Q_{\text{train}}),
\qquad
\mathrm{osc}_{\mathcal S}(L) = \sup_{\mathcal S} L - \inf_{\mathcal S} L .
$$

**Proposition 1.** The penalty for observation-process shift is a product of two factors: how far the test
observation distribution sits from the training one, and how much the model's risk varies across observation
processes at all. Augmentation attacks **both**: sampling $S_1,\dots,S_K \sim Q_{\text{train}}$ widens the
support of $Q_{\text{train}}$, shrinking the divergence factor, and minimising the average over $K$ views
penalises predictors whose loss swings across $\mathcal S$, shrinking $\mathrm{osc}_{\mathcal S}(L)$.

This is the group-averaging / vicinal-risk view: the base objective

$$
\mathcal L_{\text{aug}} = \frac1K\sum_{k=1}^K \ell\big(f_\theta(S_k(X)),Y\big)
$$

is an unbiased estimate of $R_{Q_{\text{train}}}$ whose minimiser is pushed toward predictors that are *stable*
on the orbit of $X$ under $\mathcal S$ — **without** requiring representational invariance. That distinction is
the reason the framework does not impose $z(O_1)=z(O_2)$: stability of the *loss* is what the bound needs;
equality of *representations* is a strictly stronger and unnecessary constraint, and it is what collapsed
in the earlier consistency experiments.

**Corollary 1.1 (the gain should scale with baseline damage).** The reducible part of the shift penalty is
bounded by $\mathrm{osc}_{\mathcal S}(L)$, which for the unaugmented model is measured by $\Delta(\theta)$.
A model that is already insensitive has nothing to recover; one with large $\Delta$ has room proportional to it.

> **Tested — supported.** Across the 21 main cells, the correlation between baseline degradation and OPERA's
> relative gain is $r = +0.86$. The extremes: `activity` degrades $+2405\%$ to $+3407\%$ and gains $87$–$94\%$;
> `ushcn` degrades $\approx +6\%$ and gains $\approx 3\%$.

> **Tested — partially supported.** Augmentation should flatten the loss across observation processes.
> The median $(\max-\min)/\text{clean}$ across the four shift conditions falls from $0.388$ (vanilla) to
> $0.318$ (opera). But per-run, opera reduces this spread in only $51/84$ runs, and reduces the
> *coefficient of variation* in $39/84$ — indistinguishable from chance. Augmentation lowers the absolute
> spread; the stronger claim that it lowers relative dispersion is **not** supported.

---

## 3. Mechanism 2 — privileged distillation reduces the estimation variance

This is the component with the cleanest justification, and it rests on one structural condition.

Fix a history window $H$. Let the rich (teacher) view and the sparse (student) view be two observations of
**that same window**, with the rich view at least as informative:

$$
\sigma(O_{\text{sparse}}) \;\subseteq\; \sigma(O_{\text{rich}}). \tag{$\star$}
$$

In the implementation this holds by construction: the teacher's view is `rich_process: identity`, the full
observation, and the student's views are sub-samplings of it. The anti-leakage rule — teacher and student see
the same history, neither sees the future — is what keeps the two views nested and the comparison attributable
to observation *density* rather than to temporal information.

Let the population teacher be $t(O_{\text{rich}}) = \mathbb{E}[Y \mid O_{\text{rich}}]$, and consider squared loss.

**Proposition 2 (unbiased, strictly variance-reducing).** Under $(\star)$:

**(a) Same optimum.** By the tower property,

$$
\mathbb{E}\big[t(O_{\text{rich}}) \,\big|\, O_{\text{sparse}}\big]
= \mathbb{E}\big[\mathbb{E}[Y\mid O_{\text{rich}}] \,\big|\, O_{\text{sparse}}\big]
= \mathbb{E}[Y \mid O_{\text{sparse}}],
$$

so regressing the student on the teacher's output and regressing it on $Y$ have the **identical** population
minimiser, namely $\mathbb{E}[Y\mid O_{\text{sparse}}]$. Distillation introduces no asymptotic bias.

**(b) Less noise.** By the law of total variance applied to the refinement
$\sigma(O_{\text{sparse}}) \subseteq \sigma(O_{\text{rich}})$,

$$
\underbrace{\mathrm{Var}\big(Y \mid O_{\text{sparse}}\big)}_{\text{noise when regressing on } Y}
=
\underbrace{\mathrm{Var}\big(t(O_{\text{rich}}) \mid O_{\text{sparse}}\big)}_{\text{noise when regressing on the teacher}}
+
\underbrace{\mathbb{E}\big[\mathrm{Var}(Y \mid O_{\text{rich}}) \mid O_{\text{sparse}}\big]}_{\text{removed by the rich view}} .
$$

Every term is non-negative, so the distillation target is **strictly** less noisy unless the rich view resolves
nothing ($\mathrm{Var}(Y\mid O_{\text{rich}}) = 0$ a.s.). Since the estimation component of excess risk scales
with target noise, distillation buys a strictly smaller finite-sample error at the same $n$ — the same
target, estimated better.

**Corollary 2.1.** The benefit is exactly $\mathbb{E}[\mathrm{Var}(Y\mid O_{\text{rich}})\mid O_{\text{sparse}}]$:
the uncertainty the sparse view leaves open that the rich view closes. It is largest when the sparse view is
heavily degraded — **the same qualitative prediction as Corollary 1.1**, and consistent with the same $r=+0.86$.

**Remark (why a weak teacher can hurt).** In practice the student is distilled against $\hat t = f_{\theta_T}$,
not $t$. Decomposing, the useful variance reduction is offset by the teacher's own approximation error
$\|\hat t - t\|^2$. Distillation helps only when

$$
\mathbb{E}\big[\mathrm{Var}(Y\mid O_{\text{rich}})\mid O_{\text{sparse}}\big] \;>\; \|\hat t - t\|^2 ,
$$

which is why teacher quality is an ablation axis and not a free parameter, and why a teacher trained to
convergence on the rich view matters more than the distillation weight.

> **Tested — supported, with a caveat.** On `P19_hospB` under real hospital shift, OPERA roughly halves
> observation degradation ($+11.4\% \to +5.6\%$) at **no** clean cost ($74.04 \to 74.09$ AUROC). Proposition 2
> predicts exactly this shape: no asymptotic bias, lower variance. The caveat is that the teacher here is a
> finite-sample model, and on backbones whose optimisation is unstable (TMix, §5) the $\|\hat t - t\|^2$ term
> dominates and distillation *hurts*.

---

## 4. Why the two mechanisms compose

Decompose the test risk into three sources of error:

$$
R_{Q_{\text{test}}}(\theta)
\;\approx\;
\underbrace{\mathcal{A}}_{\text{approximation}}
\;+\;
\underbrace{\mathcal{E}(n,\sigma^2)}_{\text{estimation}}
\;+\;
\underbrace{\mathcal{M}(Q_{\text{train}},Q_{\text{test}})}_{\text{mismatch}} .
$$

The two mechanisms act on **different terms**, which is the structural reason they are complementary rather
than redundant:

| mechanism | term | how |
|---|---|---|
| observation augmentation | $\mathcal{M}$ | widens $\mathrm{supp}\,Q_{\text{train}}$, flattens $L$ over $\mathcal S$ |
| privileged distillation | $\mathcal{E}$ | replaces $Y$ with a lower-variance target of the same optimum |

Because one moves the mismatch term and the other the estimation term, they are close to orthogonal, and the
theory predicts their benefits should largely add rather than overlap. Empirically they carry the framework:
the combined arm improves shifted error in $75/84$ paired runs with a median gain of $+10.17\%$.

A useful consequence: the deployment path needs neither mechanism at inference. Augmentation only shapes which
$\theta$ is selected, and the distillation term vanishes once training ends, leaving a student that consumes a
single sparse observation. Both mechanisms are training-time only, by construction.

---

## 5. Where the theory does not reach

- **Optimisation failure is outside the model.** All of the above assumes the empirical minimiser is found.
  TMix fails intermittently on classification — clean AUROC $79.06 \pm 15.01$ on P19, individual runs at $49.1$
  and $39.7$, i.e. at or below chance in both arms. No risk decomposition explains these; they are failed
  optimisations and should be reported separately, not averaged into a mechanism effect. Pooling them is what
  made OPERA look harmful on eICU, when on iTransformer it wins clean $5/5$.
- **The clean cost is a bias the theory predicts but does not quantify.** Minimising a mixture over
  $Q_{\text{train}}$ does not minimise the clean-component risk; the two minimisers differ unless the model can
  be exactly stable on $\mathcal S$. Observed: median $-2.20\%$ clean for the combined arm. Consistent in sign;
  the magnitude is not derived here. Notably the cost concentrates in **forecasting** and vanishes on
  classification with a stable backbone, which suggests a capacity-dependent constant.
- **History length.** Degradation tracks history length, not forecast horizon — at fixed history (USHCN,
  $24\to\{1,6,12\}$) degradation is flat at $10$–$13\%$, while as history grows (physionet $12\to36$ h,
  activity $1000\to3000$ ms) it rises monotonically, $185\%\to499\%$ and $695\%\to2978\%$. Corollaries 1.1/2.1
  then explain why the *gain* also grows. But why $\Delta$ itself grows with history is **not derived**;
  and because $\texttt{npatch}$ is a function of history in this implementation, longer history and more patches
  are confounded and cannot be separated from this grid.
- **Finite-sample rates.** Proposition 2 gives a strict variance reduction, not a rate. Tightening it to an
  explicit excess-risk bound would require assumptions on the function class that this draft does not make.

---

## 6. What the theory asks us to run next

1. **Teacher-quality sweep.** The Remark in §3 makes a falsifiable prediction: the benefit should flip sign once
   $\|\hat t - t\|^2$ exceeds the variance reduction. A deliberately weak teacher should *hurt*.
2. **Teacher-density sweep ($1\times, 2\times, 4\times$, dense).** Corollary 2.1 predicts the gain grows with
   $\mathbb{E}[\mathrm{Var}(Y\mid O_{\text{rich}})\mid O_{\text{sparse}}]$, which increases with teacher density.
   This is the most direct test of Proposition 2 available.
3. **Patch count at fixed history.** The only way to break the history/`npatch` confound in §5.
