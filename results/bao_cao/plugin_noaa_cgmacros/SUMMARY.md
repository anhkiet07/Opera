# OPERA vs vanilla — NOAA ISD (6h) and CGMacros (60min)

Test MSE, mean ± std over 5 seeds (0–4). Forecasting task.

## noaa_isd_6h

| Backbone | Method | clean | timing | density | variables | composed |
|---|---|---|---|---|---|---|
| itransformer | opera | 0.0025 ± 0.0001 | 0.0028 ± 0.0000 | 0.0034 ± 0.0003 | 0.0057 ± 0.0002 | 0.0044 ± 0.0001 |
| itransformer | vanilla | 0.0021 ± 0.0000 | 0.1324 ± 0.0523 | 0.1275 ± 0.0372 | 0.0995 ± 0.0062 | 0.1536 ± 0.0381 |
| patchtst | opera | 0.0027 ± 0.0001 | 0.0031 ± 0.0002 | 0.0041 ± 0.0006 | 0.1089 ± 0.0030 | 0.0640 ± 0.0016 |
| patchtst | vanilla | 0.0022 ± 0.0001 | 0.1178 ± 0.0306 | 0.1144 ± 0.0166 | 0.1087 ± 0.0030 | 0.1493 ± 0.0242 |
| tmix | opera | 0.0027 ± 0.0001 | 0.0033 ± 0.0002 | 0.0066 ± 0.0012 | 0.0416 ± 0.0245 | 0.0244 ± 0.0146 |
| tmix | vanilla | 0.0023 ± 0.0000 | 0.0937 ± 0.0169 | 0.0917 ± 0.0102 | 0.1074 ± 0.0025 | 0.1305 ± 0.0111 |

## cgmacros_60min

| Backbone | Method | clean | timing | density | variables | composed |
|---|---|---|---|---|---|---|
| itransformer | opera | 0.0045 ± 0.0001 | 0.0075 ± 0.0004 | 0.0077 ± 0.0003 | 0.0087 ± 0.0006 | 0.0106 ± 0.0005 |
| itransformer | vanilla | 0.0044 ± 0.0002 | 0.0487 ± 0.0063 | 0.0393 ± 0.0053 | 0.0258 ± 0.0031 | 0.0498 ± 0.0062 |
| patchtst | opera | 0.0046 ± 0.0001 | 0.0121 ± 0.0032 | 0.0125 ± 0.0018 | 0.0260 ± 0.0040 | 0.0228 ± 0.0025 |
| patchtst | vanilla | 0.0044 ± 0.0002 | 0.0576 ± 0.0038 | 0.0491 ± 0.0030 | 0.0331 ± 0.0009 | 0.0604 ± 0.0027 |
| tmix | opera | 0.0049 ± 0.0002 | 0.0117 ± 0.0019 | 0.0126 ± 0.0017 | 0.0140 ± 0.0016 | 0.0153 ± 0.0025 |
| tmix | vanilla | 0.0043 ± 0.0001 | 0.0454 ± 0.0046 | 0.0403 ± 0.0019 | 0.0323 ± 0.0020 | 0.0511 ± 0.0040 |

## Contents

- `raw/`: per-run config.yaml, train.log, results.json (60 runs; model checkpoints omitted)
- `benchmark*_noaa_cgmacros.csv`: rows of results/benchmark*.csv for these two settings
- `logs/`: Slurm logs (jobs 198, 199, 202, 203)
