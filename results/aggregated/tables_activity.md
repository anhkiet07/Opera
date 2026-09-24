# OPERA aggregated results

## Table 1 - Robustness to observation-process shift (test MSE, mean +- std over seeds)

Conditions marked * never occur in the training observation-process pool.

| method | seeds | clean | rnd .5 | rnd .7 | var .3 | var .5* | stride 3* | long gap* | burst .5* | var group* |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 baseline (no OPERA) | 1 | 0.0026 | 0.0546 | 0.0784 | 0.0312 | 0.0578 | 0.0679 | 0.0170 | 0.0197 | 0.0292 |
| E4 full OPERA | 1 | 0.0029 | 0.0031 | 0.0042 | 0.0036 | 0.0049 | 0.0031 | 0.0031 | 0.0031 | 0.0037 |

## Table 2 - Component contribution

| method | seeds | clean MSE | mean shifted MSE | mean degradation | unseen MSE | unseen degradation | worst condition |
|---|---|---|---|---|---|---|---|
| E0 baseline (no OPERA) | 1 | 0.0026 | 0.0498 | 1793.2 % | 0.0568 | 2057.3 % | temporal_stride:4 |
| E4 full OPERA | 1 | 0.0029 | 0.0036 | 25.2 % | 0.0039 | 36.6 % | temporal_stride:3+variable_group_dropout:0.3 |

## Table 3 - Matched density (0.5), different structure (test MSE)

| method | seeds | density:0.5 | temporal_stride:2 | local_mask:0.5 | burst:0.5 | variable_dropout:0.5 | spread |
|---|---|---|---|---|---|---|---|
| E0 baseline (no OPERA) | 1 | 0.0489 | 0.0523 | - | 0.0197 | 0.0578 | 0.0380 |
| E4 full OPERA | 1 | 0.0030 | 0.0029 | - | 0.0031 | 0.0049 | 0.0020 |

## Table 4 - Compute and parameter overhead

| method | seeds | params | s / epoch | epochs | peak GPU MB | inference ms / series |
|---|---|---|---|---|---|---|
| E0 baseline (no OPERA) | 1 | 127297 | 6.6 | 40 | 225 | 0.218 |
| E4 full OPERA | 1 | 127297 | 18.4 | 40 | 333 | 0.227 |

## All test conditions (test MSE)

| method | seeds | identity | random_dropout:0.3 | random_dropout:0.5 | random_dropout:0.7 | variable_dropout:0.3 | variable_dropout:0.5 | temporal_stride:2 | temporal_stride:3 | temporal_stride:4 | burst:0.3 | burst:0.5 | long_gap:0.4 | variable_group_dropout:0.3 | density:0.5 | random_dropout:0.5+variable_dropout:0.3 | temporal_stride:3+variable_group_dropout:0.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 baseline (no OPERA) | 1 | 0.0026 | 0.0333 | 0.0546 | 0.0784 | 0.0312 | 0.0578 | 0.0523 | 0.0679 | 0.0855 | 0.0161 | 0.0197 | 0.0170 | 0.0292 | 0.0489 | 0.0745 | 0.0809 |
| E4 full OPERA | 1 | 0.0029 | 0.0029 | 0.0031 | 0.0042 | 0.0036 | 0.0049 | 0.0029 | 0.0031 | 0.0034 | 0.0029 | 0.0031 | 0.0031 | 0.0037 | 0.0030 | 0.0041 | 0.0055 |
