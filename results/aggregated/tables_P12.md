# OPERA aggregated results

## Table 1 - Robustness to observation-process shift (test MSE, mean +- std over seeds)

Conditions marked * never occur in the training observation-process pool.

| method | seeds | clean | rnd .5 | rnd .7 | var .3 | var .5* | stride 3* | long gap* | burst .5* | var group* |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 baseline (no OPERA) | 2 | 0.3894 +- 0.0152 | 0.3933 +- 0.0291 | 0.4118 +- 0.0354 | 0.4107 +- 0.0312 | 0.4323 +- 0.0474 | 0.3816 +- 0.0188 | 0.3773 +- 0.0213 | 0.3792 +- 0.0270 | 0.4051 +- 0.0268 |
| E4 full OPERA | 1 | 0.3864 | 0.3653 | 0.3660 | 0.3864 | 0.3729 | 0.3723 | 0.3525 | 0.3516 | 0.3880 |

## Table 2 - Component contribution

| method | seeds | clean MSE | mean shifted MSE | mean degradation | unseen MSE | unseen degradation | worst condition |
|---|---|---|---|---|---|---|---|
| E0 baseline (no OPERA) | 2 | 0.3894 +- 0.0152 | 0.3951 +- 0.0272 | 1.3 +- 3.0 % | 0.3972 +- 0.0290 | 1.9 +- 3.5 % | variable_dropout:0.5 |
| E4 full OPERA | 1 | 0.3864 | 0.3715 | -3.8 % | 0.3692 | -4.5 % | variable_group_dropout:0.3 |

## Table 3 - Matched density (0.5), different structure (test MSE)

| method | seeds | density:0.5 | temporal_stride:2 | local_mask:0.5 | burst:0.5 | variable_dropout:0.5 | spread |
|---|---|---|---|---|---|---|---|
| E0 baseline (no OPERA) | 2 | 0.3886 +- 0.0225 | 0.3865 +- 0.0181 | - | 0.3792 +- 0.0270 | 0.4323 +- 0.0474 | 0.0539 +- 0.0212 |
| E4 full OPERA | 1 | 0.3782 | 0.3807 | - | 0.3516 | 0.3729 | 0.0291 |

## Table 4 - Compute and parameter overhead

| method | seeds | params | s / epoch | epochs | peak GPU MB | inference ms / series |
|---|---|---|---|---|---|---|
| E0 baseline (no OPERA) | 2 | 108802 | 35.2 +- 1.0 | 20 +- 2 | 4027 +- 650 | 1.320 +- 0.009 |
| E4 full OPERA | 1 | 108802 | 167.0 | 15 | 4926 | 1.351 |

## All test conditions (test MSE)

| method | seeds | identity | random_dropout:0.3 | random_dropout:0.5 | random_dropout:0.7 | variable_dropout:0.3 | variable_dropout:0.5 | temporal_stride:2 | temporal_stride:3 | temporal_stride:4 | burst:0.3 | burst:0.5 | long_gap:0.4 | variable_group_dropout:0.3 | density:0.5 | random_dropout:0.5+variable_dropout:0.3 | temporal_stride:3+variable_group_dropout:0.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 baseline (no OPERA) | 2 | 0.3894 +- 0.0152 | 0.3873 +- 0.0229 | 0.3933 +- 0.0291 | 0.4118 +- 0.0354 | 0.4107 +- 0.0312 | 0.4323 +- 0.0474 | 0.3865 +- 0.0181 | 0.3816 +- 0.0188 | 0.3843 +- 0.0214 | 0.3846 +- 0.0236 | 0.3792 +- 0.0270 | 0.3773 +- 0.0213 | 0.4051 +- 0.0268 | 0.3886 +- 0.0225 | 0.4026 +- 0.0322 | 0.4006 +- 0.0307 |
| E4 full OPERA | 1 | 0.3864 | 0.3718 | 0.3653 | 0.3660 | 0.3864 | 0.3729 | 0.3807 | 0.3723 | 0.3756 | 0.3681 | 0.3516 | 0.3525 | 0.3880 | 0.3782 | 0.3705 | 0.3731 |
