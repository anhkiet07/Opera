# OPERA aggregated results

## Table 1 - Robustness to observation-process shift (test MSE, mean +- std over seeds)

Conditions marked * never occur in the training observation-process pool.

| method | seeds | clean | rnd .5 | rnd .7 | var .3 | var .5* | stride 3* | long gap* | burst .5* | var group* |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 baseline (no OPERA) | 1 | 0.4359 | 0.4700 | 0.5633 | 1.0814 | 1.5578 | 0.4504 | 0.4806 | 0.4896 | 1.1186 |
| E4 full OPERA | 1 | 0.4441 | 0.4478 | 0.4842 | 0.7211 | 1.1567 | 0.4386 | 0.4703 | 0.4685 | 0.8691 |

## Table 2 - Component contribution

| method | seeds | clean MSE | mean shifted MSE | mean degradation | unseen MSE | unseen degradation | worst condition |
|---|---|---|---|---|---|---|---|
| E0 baseline (no OPERA) | 1 | 0.4359 | 0.7131 | 63.6 % | 0.8178 | 87.6 % | variable_dropout:0.5 |
| E4 full OPERA | 1 | 0.4441 | 0.5970 | 34.4 % | 0.6687 | 50.6 % | variable_dropout:0.5 |

## Table 3 - Matched density (0.5), different structure (test MSE)

| method | seeds | density:0.5 | temporal_stride:2 | local_mask:0.5 | burst:0.5 | variable_dropout:0.5 | spread |
|---|---|---|---|---|---|---|---|
| E0 baseline (no OPERA) | 1 | 0.4466 | 0.4360 | - | 0.4896 | 1.5578 | 1.1218 |
| E4 full OPERA | 1 | 0.4410 | 0.4375 | - | 0.4685 | 1.1567 | 0.7192 |

## Table 4 - Compute and parameter overhead

| method | seeds | params | s / epoch | epochs | peak GPU MB | inference ms / series |
|---|---|---|---|---|---|---|
| E0 baseline (no OPERA) | 1 | 98520 | 2.6 | 21 | 245 | 0.127 |
| E4 full OPERA | 1 | 98520 | 27.3 | 40 | 421 | 0.315 |

## All test conditions (test MSE)

| method | seeds | identity | random_dropout:0.3 | random_dropout:0.5 | random_dropout:0.7 | variable_dropout:0.3 | variable_dropout:0.5 | temporal_stride:2 | temporal_stride:3 | temporal_stride:4 | burst:0.3 | burst:0.5 | long_gap:0.4 | variable_group_dropout:0.3 | density:0.5 | random_dropout:0.5+variable_dropout:0.3 | temporal_stride:3+variable_group_dropout:0.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 baseline (no OPERA) | 1 | 0.4359 | 0.4447 | 0.4700 | 0.5633 | 1.0814 | 1.5578 | 0.4360 | 0.4504 | 0.4704 | 0.4576 | 0.4896 | 0.4806 | 1.1186 | 0.4466 | 1.1026 | 1.1272 |
| E4 full OPERA | 1 | 0.4441 | 0.4402 | 0.4478 | 0.4842 | 0.7211 | 1.1567 | 0.4375 | 0.4386 | 0.4477 | 0.4486 | 0.4685 | 0.4703 | 0.8691 | 0.4410 | 0.8152 | 0.8678 |
