# Phase 3 report: structural / causal priors for irregular time-series representation

Status 2026-09-23 (final for this budget): synthetic full data 3 seeds + single-seed variants, synthetic low data
3 seeds for soft and hard families, PhysioNet LLM prior (2 seeds main, 1 seed variants). W&B: runs `p3_*`,
`p3low_*`, `p3phys_*`; tables: `results/phase3_tables.md`; plateau-stuck originals kept as `results/*.stuck.json`.

## Method

Variable-level structure `M` (`M[i,j] = 1` if j -> i, plus diagonal) enters the backbone's cross-variable
attention (iTransformer, variable tokens) as an additive bias:  `A_ij = softmax(Q_i K_j / sqrt(d) + lambda * M_ij)`
(`models/causal.py`, hook in `models/quite.py`; the QuITE embedding itself attends only within a variable, so the
bias belongs in the backbone). Modes: `soft` (lambda * M), `hard` (mask to causes + self), `learned` (free N x N,
zero-init). Graph sources: `true` (oracle), `identity` (diagonal only), `random` (same edge count, random
positions), `shuffled` (variables permuted), `noisy:p` (fraction p of off-diagonal entries flipped).

Synthetic testbed (`data/synthetic.py`): 10 abstract channels, 48-unit window, step 0.5; roots are damped
oscillators, effects follow `0.5 x_i + 0.9 sum_{j->i} tanh(x_j) + noise`; edges 0->1->2, 0->3, 4->2, 5->6, 7->8,
9 isolated; target Y = channel 2 (parents 1, 4). Each channel observed with probability in [0.2, 0.4] per step.
4,000 trajectories (full data) or 1,500 (low data); 24 -> 24 forecasting, same pipeline as PhysioNet.

Critical experiment: re-observe the test inputs with specific channels removed (targets unchanged) and read the
error on Y: `dropvars:5-6-7-8-9` (all channels that cannot influence Y), `dropvars:1-4` (Y's parents),
`dropvars:0-3` (grand-parent + a non-ancestor sibling). Generic sparsity views (random 0.5 / 0.7, variable 0.3) as
before. Budget: epoch cap 100, patience 10, batch 32, lr 1e-3.

## 1. Does the base encoder use the structure? Yes.

Test MSE on Y (plain QuITE, 3 seeds): clean 0.0267, non-causal channels removed 0.0279, parents removed 0.0474,
grand-parent + sibling removed 0.0368. The vanilla backbone (no irregular embedding) is worse everywhere (0.0503
clean) and indifferent to which channels are removed (0.0514 / 0.0522 / 0.0524): it does not exploit
cross-variable structure at all. So the testbed discriminates causal from non-causal inputs, and the query-based
encoder learns the relevant parents from data alone.

## 2. Full data (4,000 trajectories): oracle graph vs. controls

Test MSE, overall (mean +- std over 3 seeds) / on Y (mean). Seed 3 of every biased variant originally
early-stopped on the initial constant plateau (Section 4) and was rerun with a 60-epoch minimum window; the
rerun values are used here.

| structure (soft, lambda 2) | clean | random 0.5 | non-causal removed | parents removed | grand-parent+sibling removed |
|---|---|---|---|---|---|
| none (plain QuITE) | 0.0198 +- .0006 / 0.0267 | 0.1005 / 0.0968 | 0.0688 / 0.0279 | 0.0382 / **0.0474** | 0.0408 / 0.0389 |
| oracle graph | 0.0192 +- .0020 / 0.0256 | 0.1150 / 0.1142 | 0.0756 / 0.0271 | 0.0382 / 0.0531 | 0.0482 / 0.0407 |
| shuffled graph | 0.0191 +- .0011 / 0.0268 | 0.0972 / 0.0968 | 0.0666 / 0.0277 | 0.0368 / 0.0449 | 0.0429 / 0.0376 |
| random graph | 0.0208 +- .0015 / 0.0283 | 0.1109 / 0.1083 | 0.0664 / 0.0289 | 0.0363 / 0.0448 | 0.0432 / 0.0392 |
| identity (seed 1 only) | 0.0172 / 0.0241 | 0.1003 / 0.1004 | 0.0672 / 0.0248 | 0.0349 / 0.0389 | 0.0392 / 0.0331 |
| noisy graph (30 % flipped) | **[pending]** | | | | |
| hard mask (oracle) | **[pending]** | | | | |
| learned N x N bias | **[pending]** | | | | |
| lambda 0.5 / 5 (oracle) | **[pending]** | | | | |
| vanilla backbone + oracle bias | **[pending]** | | | | |

Observations.
- Over 3 seeds the oracle graph is within noise of no bias (0.0192 +- 0.0020 vs 0.0198 +- 0.0006 overall;
  0.0256 vs 0.0267 on Y).
- A **shuffled** graph (same edges, wrong channels) is indistinguishable from the oracle (0.0191 +- 0.0011), and an
  **identity** bias (self-attention only, no edges) gives on seed 1 exactly the oracle's seed-1 number (0.0172 vs
  0.0172; Y 0.0241 vs 0.0236). A random-position graph is slightly worse than no bias (0.0208).
- On the discriminative test (parents removed) **no** biased model is better than plain QuITE on Y, and the
  oracle-biased model is the worst (0.0531 vs 0.0474). Under random dropout the oracle bias is the least robust
  (0.1150 vs 0.1005).

Interpretation. Under this budget the in-distribution gain from a "causal" attention bias is reproduced by a bias
with no structural content (identity, shuffled), so it should be attributed to attention regularisation (a
sharper, more self-focused attention over 10 tokens), not to the structure. The correct structure does not make
the model rely more usefully on the true parents. With 2,400 training trajectories the encoder learns the
relevant dependencies from data, leaving nothing for a prior to add. An in-distribution improvement from a
structural attention bias is therefore not evidence of causal usefulness.

## 3. Low data (1,500 trajectories, 900 train), 3 seeds

Seeds 2-3 of every configuration originally stalled on the constant plateau (Section 4) and were rerun with a
60-epoch minimum window; all 3 seeds below are non-degenerate. Overall mean +- std / Y mean.

**Soft-bias family** (lambda 2):

| structure | clean | random 0.5 | non-causal removed | parents removed |
|---|---|---|---|---|
| none (plain QuITE) | 0.0325 +- .0004 / 0.0404 | 0.1078 / 0.1032 | 0.0751 / 0.0426 | 0.0491 / 0.0546 |
| oracle graph | 0.0333 +- .0018 / 0.0400 | 0.0952 / 0.0935 | 0.0631 / 0.0404 | 0.0509 / 0.0652 |
| shuffled graph | 0.0339 +- .0002 / 0.0417 | 0.0921 / 0.0951 | 0.0668 / 0.0426 | 0.0497 / 0.0521 |
| random graph | 0.0323 +- .0008 / 0.0398 | 0.0854 / 0.0860 | 0.0694 / 0.0405 | 0.0481 / 0.0503 |
| identity (diagonal only) | 0.0322 +- .0007 / 0.0400 | 0.1034 / 0.0997 | 0.0771 / 0.0405 | 0.0532 / 0.0579 |

The soft bias is inert here too: all five rows are within about 0.0017 of each other on clean error and within
noise on Y, and the oracle's larger dependence on Y's parents (0.0652) is matched by the identity bias (0.0579).

**Hard-mask family** (attend only to the listed causes + self):

| hard mask with | clean | random 0.5 | non-causal removed | parents removed |
|---|---|---|---|---|
| oracle graph | **0.0287 +- .0022 / 0.0370** | 0.1012 / 0.0977 | 0.0627 / **0.0370** | 0.0418 / 0.0714 |
| shuffled graph | 0.0333 +- .0008 / 0.0516 | 0.0921 / 0.0818 | 0.0670 / 0.0516 | 0.0468 / 0.0516 |
| identity (channel-independent) | 0.0340 +- .0025 / 0.0502 | 0.0877 / 0.0771 | 0.0671 / 0.0502 | 0.0464 / 0.0502 |
| random graph | 0.0331 +- .0007 / 0.0499 | 0.0828 / 0.0744 | 0.0634 / 0.0499 | 0.0452 / 0.0499 |

Per seed (overall / Y clean): oracle 0.0277 / 0.036, 0.0317 / 0.041, 0.0266 / 0.034 vs. shuffled 0.0334 / 0.052,
0.0342 / 0.051, 0.0322 / 0.053. The ordering is the same on every seed. Forcing attention to the *true* causes is
the best configuration in this regime (overall about 14 % below every wrong-structure mask and below plain QuITE),
its Y error does not change when the non-causal channels are removed, and it is the only configuration whose Y
error rises when Y's parents are removed (0.0714). Every wrong-structure mask (shuffled, identity, random) leaves Y
at about 0.050 on every view, i.e. Y is then predicted essentially from its own observations. This is the one
result in Phase 3 where correct structure separates from incorrect structure of the same shape, in the direction
the structural hypothesis predicts, and it survives the shuffled, identity and random controls. It is confined to
the hard constraint and the low-data regime: the soft bias does not show it, and on full data the hard mask does
not beat the identity bias (Section 2). Its cost is reduced robustness to random dropout relative to the wrong
masks (0.1012 vs 0.083 - 0.092), because the constrained model relies on fewer channels.

Seed 1 only. Soft biases behave as on full data: oracle, shuffled, random and identity all give the same small
improvement on Y and under dropout, and the identity bias even reproduces the oracle's increased dependence on
Y's parents (0.0679 vs 0.0655), so nothing in the soft-bias results is specific to the correct structure.

The **hard mask with the oracle graph** is different: it is the best configuration by a clear margin (clean
0.0277 vs 0.0330 overall, 0.0362 vs 0.0427 on Y), its Y error is unchanged when the five non-causal channels are
removed (0.0362), and it depends strongly on Y's parents (0.0828 when they are removed). This is the signature a
correct structural constraint should produce: with 900 training trajectories, restricting each variable to its
true causes removes spurious cross-variable attention that the model cannot rule out from data. Whether the
gain is due to the *correct* edges or merely to restricting attention is the next question: hard-mask controls
with shuffled, identity (channel-independent backbone) and random graphs are queued (`jobs/phase3_lowdata_hard.sh`),
along with seeds 2-3. An initial 600-trajectory attempt was discarded: every configuration converged to a
constant predictor.

## 4. Optimization instability caused by the bias

On seed 3, every soft-bias variant (oracle, shuffled, random) stayed on the constant-predictor plateau for its
first 10+ epochs (train loss flat at 0.055) and was terminated by patience-10 early stopping at epoch 13, whereas
plain QuITE trained normally on all three seeds. The additive bias at lambda 2 therefore makes early optimization
on this task less stable. A `--min_epochs` guard was added to `train_forecasting.py`; `rerun_stuck.py` reruns
the affected runs with `--min_epochs 40` once the queues finish **[pending]**. Seed-3 numbers are excluded above.

## 5. Real data: LLM-elicited physiological prior on PhysioNet — [pending]

`data/physionet_prior.py` holds a 58-edge directed graph over the 41 PhysioNet variables written from textbook
physiology by an LLM (e.g. MechVent -> FiO2 -> PaO2 -> SaO2, PaCO2 / HCO3 / lactate -> pH, urine -> creatinine
/ BUN, MAP -> HR / urine / lactate, WBC <-> temperature). It is a semantic prior, not a validated causal graph.
Queue `jobs/phase3_physionet.sh`; reference is `p0_quite_itr_s*` (same budget). Test MSE, PhysioNet 24h -> 24h:

| prior in the variable attention | seeds | clean | random 0.7 | bursty 0.5 | variable 0.3 | variable 0.5 | stride 3 |
|---|---|---|---|---|---|---|---|
| none (plain QuITE) | 3 | 0.0054 | **0.0164** | 0.0079 | 0.0166 | 0.0225 | 0.0108 |
| LLM prior, soft lambda 1 | 1 | **0.0053** | 0.0178 | 0.0082 | 0.0194 | 0.0277 | 0.0101 |
| LLM prior, soft lambda 2 | 2 | 0.0055 | 0.0185 | 0.0085 | 0.0213 | 0.0304 | 0.0111 |
| LLM prior, soft lambda 5 | 1 | 0.0057 | 0.0196 | 0.0088 | 0.0156 | 0.0230 | 0.0124 |
| shuffled LLM prior, lambda 2 | 2 | 0.0056 | 0.0172 | 0.0083 | 0.0188 | 0.0264 | 0.0117 |
| random prior (same edge count), lambda 2 | 1 | 0.0055 | 0.0183 | 0.0085 | 0.0150 | 0.0209 | 0.0128 |
| LLM prior, hard mask | 1 | 0.0056 | 0.0184 | 0.0086 | **0.0149** | **0.0214** | 0.0114 |
| learned N x N bias | 1 | 0.0055 | 0.0165 | **0.0076** | 0.0166 | 0.0228 | **0.0095** |

No prior improves clean error beyond seed noise (0.0053 - 0.0057 vs 0.0054), and every soft prior, including the
physiological one, is *less* robust than no prior under random dropout and variable removal at lambda 1 - 2. The
hard mask and the random prior reduce the damage from variable removal (0.0149 / 0.0150 vs 0.0166), but they do it
equally, so the effect comes from restricting cross-variable attention, not from the physiology encoded. The
learned bias is indistinguishable from no bias. The semantic prior therefore does not compensate for the absence of
an estimated graph on this dataset; whether the prior itself is poor or the benchmark leaves no room cannot be
separated without a validated clinical graph.

## 5b. PhysioNet in the regime where structure worked on synthetic data: hard masks, low data

Motivation: Section 3 found that structure helps only as a *hard* constraint with *scarce* data. Section 5 tested a
soft prior on full data. Phase 3b (`jobs/phase3b_physionet.sh`) therefore trains on 900 PhysioNet stays (first
1,500 records; 900 / 300 / 300 split) with hard attention masks built from three graphs: the LLM physiological
prior, a graph **estimated from the same 900 training stays** (`estimate_graph.py`: hourly grid, forward fill,
lasso lagged regression, top-3 parents with |coef| >= 0.02; 26 edges over 12 variables, mostly the blood-pressure
cluster, FiO2 -> PaO2, HCO3 -> PaCO2, GCS -> FiO2 / Urine), and their union (77 edges). Controls: shuffled and
random masks with the union's edge count, and identity (channel-independent backbone). 3 seeds, epoch cap 150,
patience 20, min-epochs 40. Test MSE (mean +- std):

| hard mask from | clean | random 0.7 | bursty 0.5 | variable 0.3 | variable 0.5 | stride 3 |
|---|---|---|---|---|---|---|
| none (plain QuITE) | 0.0085 +- .0004 | **0.0220** +- .0007 | **0.0132** +- .0005 | 0.0218 +- .0037 | 0.0283 +- .0047 | 0.0145 +- .0001 |
| LLM physiological prior | 0.0083 +- .0001 | 0.0256 +- .0003 | 0.0141 +- .0003 | 0.0217 +- .0022 | 0.0295 +- .0030 | 0.0153 +- .0009 |
| estimated graph | 0.0084 +- .0002 | 0.0241 +- .0025 | 0.0140 +- .0004 | **0.0197** +- .0032 | **0.0266** +- .0047 | 0.0147 +- .0012 |
| union (LLM + estimated) | 0.0084 +- .0002 | 0.0265 +- .0027 | 0.0143 +- .0009 | 0.0225 +- .0041 | 0.0307 +- .0059 | 0.0166 +- .0012 |
| shuffled union | 0.0083 +- .0004 | 0.0246 +- .0031 | 0.0136 +- .0004 | 0.0220 +- .0057 | 0.0303 +- .0095 | **0.0144** +- .0000 |
| identity | 0.0083 +- .0001 | 0.0262 +- .0015 | 0.0142 +- .0002 | 0.0213 +- .0019 | 0.0287 +- .0027 | 0.0169 +- .0008 |
| random (union edge count) | **0.0082** +- .0004 | 0.0248 +- .0043 | 0.0138 +- .0011 | 0.0207 +- .0050 | 0.0285 +- .0081 | 0.0147 +- .0022 |

Restricting the error to the variables that have parents in each graph (`results/phase3b_tables.md`) gives the
same ordering. Reading: all seven configurations are within one seed-std of each other in distribution
(0.0082 - 0.0085); no mask is reliably more robust than the unconstrained encoder, and the informative graphs are
not better than the shuffled or random ones anywhere. The estimated graph's advantage under variable removal
(0.0197 vs 0.0218) is inside the seed spread. Unlike the synthetic low-data regime, PhysioNet with 900 stays shows
neither the benefit of the correct constraint nor the harm of a wrong one, which is consistent with none of the
three graphs being a good approximation of the dependencies the encoder needs (the synthetic effect required the
*true* parents). Soft estimated / union priors and full-data estimated-graph runs: **[pending]**.

## Verdict

1. **The soft "causal" attention bias is a negative result.** On full data, on low data and on PhysioNet with an
   LLM-elicited physiological prior, the correct (or plausible) graph is not distinguishable from a shuffled graph
   or from an edge-free identity bias; it never improves robustness under sparsity or variable removal; and it
   destabilises early training (constant-predictor plateaus) on a fraction of seeds. An in-distribution gain from a
   structural attention bias is therefore not evidence that the structure is used.
2. **A hard structural constraint helps when data are scarce, and only then.** With 900 training trajectories,
   restricting each variable's attention to its true causes is the best configuration on every seed (overall
   0.0287 +- 0.0022 vs 0.0325 +- 0.0004 unconstrained and 0.033 - 0.034 for shuffled / identity / random masks;
   Y 0.0370 vs 0.0404 and about 0.050), it is unaffected by removing non-causal channels, and it is the only
   configuration that relies on Y's parents. With 2,400 trajectories the same mask brings nothing beyond the
   identity bias. The benefit costs robustness to random dropout, since the constrained model relies on fewer
   channels.
3. **On real data nothing helped, in either regime.** On full PhysioNet the semantic prior gave no gain in any
   form, and the only robustness effect (variable removal under a hard mask) was reproduced by a random prior of
   the same sparsity. In the low-data, hard-constraint regime that worked on synthetic data (Section 5b), the
   physiological prior, a graph estimated from the training split, and their union are all within seed noise of
   the unconstrained encoder and of shuffled / identity / random masks (3 seeds).

Practical reading: structural information is useful as a *constraint in the low-data regime* and only when it is
correct, which on real data requires a validated or well-estimated graph rather than a soft prior. Estimating such a
graph from irregular data (Granger / sparse lagged regression on a reconstructed grid) remains the missing piece.

## Sanity checks from the plan

| question | answer |
|---|---|
| Gain from the proposed mechanism? | Soft bias: no (identity / shuffled reproduce it). Hard mask, low data: yes (oracle beats every wrong graph on all 3 seeds). |
| Could it be extra parameters? | Soft / hard bias adds none; learned N x N bias equals no bias. |
| Survives random / shuffled controls? | Soft bias: no. Hard mask, low data: yes (shuffled / identity / random all worse than no mask). |
| Improves the intended regime (sparse / shifted)? | No: no variant is more robust under random dropout; the hard oracle mask is less robust. |
| Preserves clean performance? | Soft: unchanged. Hard oracle, low data: improves it; hard wrong graphs: degrade it. |
| Consistent across seeds? | Full data 3 seeds; low data 3 seeds (after plateau reruns); PhysioNet 2 seeds main, 1 seed variants. |
