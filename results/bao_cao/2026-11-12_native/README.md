# Opera benchmark: native backbones (grud / mtand / warpformer)

Runs finished 2026-11-10 → 2026-11-11 (SLURM jobs #7–#15). 240 runs = 8 datasets × 3 backbones × 2 methods × 5 seeds.
Metric: MSE (lower is better). Δ% = (OPERA − vanilla) / vanilla; negative = OPERA better.
Shift MSE = mean over random_dropout:0.7, temporal_stride:3, variable_dropout:0.5, temporal_stride:3+variable_group_dropout:0.3.
Degradation = (shift MSE − clean MSE) / clean MSE, averaged over the shift conditions.

## OPERA vs vanilla per dataset × backbone (mean of 5 seeds)

| Dataset | Backbone | Clean V | Clean O | Δ clean | Shift V | Shift O | Δ shift | Degr. V → O | OPERA wins | NaN runs |
|---|---|---|---|---|---|---|---|---|---|---|
| activity_h1000 | grud | 0.004923 | 0.004906 | -0.4% | 0.0136 | 0.006852 | **-49.6%** | 1.76 → 0.40 | 5/5 | — |
| activity_h1000 | mtand | 0.004853 | 0.004815 | -0.8% | 0.01386 | 0.006177 | **-55.4%** | 1.85 → 0.28 | 5/5 | — |
| activity_h1000 | warpformer | 0.01122 | 0.0108 | -3.8% | 0.01665 | 0.01157 | **-30.5%** | 0.49 → 0.07 | 5/5 | — |
| activity_h2000 | grud | 0.003874 | 0.00387 | -0.1% | 0.01981 | 0.005464 | **-72.4%** | 4.12 → 0.41 | 5/5 | — |
| activity_h2000 | mtand | 0.003908 | 0.003924 | +0.4% | 0.01214 | 0.005143 | **-57.6%** | 2.10 → 0.31 | 5/5 | — |
| activity_h2000 | warpformer | 0.009894 | 0.009745 | -1.5% | 0.01426 | 0.01025 | **-28.1%** | 0.47 → 0.05 | 5/5 | vanilla_s0;vanilla_s2 |
| cgmacros | grud | 0.003939 | 0.003984 | +1.2% | 0.008312 | 0.006208 | **-25.3%** | 1.11 → 0.56 | 5/5 | — |
| cgmacros | mtand | 0.004421 | 0.004355 | -1.5% | 0.007149 | 0.006265 | **-12.4%** | 0.62 → 0.44 | 5/5 | — |
| cgmacros | warpformer | 0.007686 | 0.007504 | -2.4% | 0.01069 | 0.008449 | **-21.0%** | 0.39 → 0.13 | 5/5 | — |
| noaa_isd | grud | 0.0007243 | 0.0007362 | +1.6% | 0.008184 | 0.001885 | **-77.0%** | 10.25 → 1.56 | 5/5 | — |
| noaa_isd | mtand | 0.0007988 | 0.0007941 | -0.6% | 0.009191 | 0.0019 | **-79.3%** | 10.37 → 1.39 | 5/5 | — |
| noaa_isd | warpformer | 0.003427 | 0.003206 | -6.4% | 0.008036 | 0.003784 | **-52.9%** | 1.30 → 0.18 | 5/5 | vanilla_s4 |
| physionet_h12 | grud | 0.006323 | 0.00616 | -2.6% | 0.009526 | 0.007356 | **-22.8%** | 0.51 → 0.19 | 5/5 | — |
| physionet_h12 | mtand | 0.006585 | 0.006487 | -1.5% | 0.008666 | 0.007383 | **-14.8%** | 0.32 → 0.14 | 5/5 | — |
| physionet_h12 | warpformer | 0.00823 | 0.008147 | -1.0% | 0.009123 | 0.008646 | **-5.2%** | 0.11 → 0.06 | 5/5 | — |
| physionet_h36 | grud | 0.004623 | 0.004608 | -0.3% | 0.007756 | 0.005865 | **-24.4%** | 0.68 → 0.27 | 5/5 | — |
| physionet_h36 | mtand | 0.005864 | 0.005868 | +0.1% | 0.008588 | 0.006796 | **-20.9%** | 0.47 → 0.16 | 5/5 | — |
| physionet_h36 | warpformer | 0.007763 | 0.007732 | -0.4% | 0.008716 | 0.008186 | **-6.1%** | 0.12 → 0.06 | 4/5 | — |
| ushcn_6mo | grud | 0.5287 | 0.5293 | +0.1% | 0.7334 | 0.5989 | **-18.3%** | 0.39 → 0.13 | 5/5 | — |
| ushcn_6mo | mtand | 0.583 | 0.5812 | -0.3% | 0.6453 | 0.6282 | **-2.7%** | 0.11 → 0.09 | 5/5 | — |
| ushcn_6mo | warpformer | 0.5645 | 0.5666 | +0.4% | 0.5958 | 0.5891 | **-1.1%** | 0.06 → 0.04 | 5/5 | — |
| vitaldb | grud | 0.002586 | 0.002596 | +0.4% | 0.008198 | 0.003228 | **-60.6%** | 2.17 → 0.24 | 5/5 | — |
| vitaldb | mtand | 0.00274 | 0.002697 | -1.6% | 0.005211 | 0.003213 | **-38.3%** | 0.90 → 0.19 | 5/5 | — |
| vitaldb | warpformer | 0.003992 | 0.003766 | -5.7% | 0.007221 | 0.00402 | **-44.3%** | 0.81 → 0.07 | 5/5 | — |

## Δ MSE per shift condition (pooled over the 3 backbones)

| Dataset | identity | random_dropout:0.7 | temporal_stride:3 | temporal_stride:3+variable_group_dropout:0.3 | variable_dropout:0.5 |
|---|---|---|---|---|---|
| activity_h1000 | -2.3% | -42.2% | -18.9% | -41.6% | -56.8% |
| activity_h2000 | -0.8% | -40.6% | -4.7% | -53.8% | -71.0% |
| cgmacros | -1.3% | -26.0% | -5.5% | -15.6% | -28.5% |
| noaa_isd | -4.3% | -63.9% | -10.2% | -72.5% | -77.0% |
| physionet_h12 | -1.6% | -17.6% | -5.9% | -11.0% | -20.4% |
| physionet_h36 | -0.2% | -11.6% | -3.6% | -15.1% | -29.5% |
| ushcn_6mo | +0.1% | -10.3% | -9.9% | -8.0% | -3.7% |
| vitaldb | -2.8% | -49.8% | -7.4% | -42.9% | -66.5% |

## Known issue: warpformer NaN

Some warpformer runs hit NaN loss mid-training; early stopping keeps the last finite best checkpoint, so results exist but
those runs stopped learning early (see the `nan_runs` column). Root cause (reproduced on activity_h2000 warpformer vanilla s0):
the warping step normalises `gamma = cumsum(softplus(score)) / gmax`, which is invariant to a shift of `score`; Adam drifts
`score` to ≈ −90, `softplus` underflows float32 (gmax ≈ 7e-37) and the 1/gmax² gradient overflows to inf → NaN.
Not yet fixed in `src/models/irregular_baselines.py`.

## Files

- `summary_by_backbone.csv`: table above, with std
- `summary_by_condition.csv`: Δ per shift condition
- `per_run.csv`: one row per run (all conditions)
- `runs/<run>/`: results.json, config.yaml, train.log (model.pt checkpoints omitted)
- `slurm_logs/`: job logs for #7–#15 and dataset downloads
- `summary_by_backbone_excl_nan.csv`: same table with the NaN runs dropped
- `nan_debug/`: probe script and log reproducing the warpformer NaN
- `TONG_HOP.md`: full summary in Vietnamese
- `severity/`: robustness curves (MSE vs shift severity, 4 shift families, no retraining). See `severity/TOM_TAT_SEVERITY.md`, figures in `severity/figures/`, every plotted point in `severity/severity_summary.csv`, raw per-run sweep output in `severity/sweep_json/`
