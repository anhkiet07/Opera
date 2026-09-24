# OPERA aggregated results

## Table 1 - Robustness to observation-process shift (test MSE, mean +- std over seeds)

Conditions marked * never occur in the training observation-process pool.

| method | seeds | clean | rnd .5 | rnd .7 | var .3 | var .5* | stride 3* | long gap* | burst .5* | var group* |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 baseline (no OPERA) | 1 | 0.0055 | 0.0108 | 0.0152 | 0.0158 | 0.0213 | 0.0092 | 0.0078 | 0.0080 | 0.0115 |
| E1 + augmentation | 1 | 0.0055 | 0.0064 | 0.0075 | 0.0069 | 0.0075 | 0.0057 | 0.0059 | 0.0058 | 0.0063 |
| E2 + distillation | 1 | 0.0057 | 0.0064 | 0.0073 | 0.0070 | 0.0078 | 0.0059 | 0.0061 | 0.0060 | 0.0064 |
| E3 + hard mining | 1 | 0.0056 | 0.0065 | 0.0074 | 0.0070 | 0.0076 | 0.0059 | 0.0060 | 0.0059 | 0.0063 |
| E4 full OPERA | 1 | 0.0056 | 0.0064 | 0.0073 | 0.0068 | 0.0074 | 0.0059 | 0.0059 | 0.0060 | 0.0062 |
| E3b_hard_mining_strong | 1 | 0.0056 | 0.0064 | 0.0071 | 0.0067 | 0.0072 | 0.0058 | 0.0060 | 0.0059 | 0.0062 |
| E4b_opera_strong | 1 | 0.0059 | 0.0066 | 0.0073 | 0.0070 | 0.0075 | 0.0062 | 0.0062 | 0.0062 | 0.0064 |

## Table 2 - Component contribution

| method | seeds | clean MSE | mean shifted MSE | mean degradation | unseen MSE | unseen degradation | worst condition |
|---|---|---|---|---|---|---|---|
| E0 baseline (no OPERA) | 1 | 0.0055 | 0.0114 | 107.0 % | 0.0126 | 129.5 % | variable_dropout:0.5 |
| E1 + augmentation | 1 | 0.0055 | 0.0063 | 14.3 % | 0.0065 | 17.9 % | random_dropout:0.5+variable_dropout:0.3 |
| E2 + distillation | 1 | 0.0057 | 0.0064 | 12.3 % | 0.0066 | 15.5 % | variable_dropout:0.5 |
| E3 + hard mining | 1 | 0.0056 | 0.0064 | 14.8 % | 0.0066 | 18.0 % | variable_dropout:0.5 |
| E4 full OPERA | 1 | 0.0056 | 0.0063 | 13.1 % | 0.0065 | 16.3 % | variable_dropout:0.5 |
| E3b_hard_mining_strong | 1 | 0.0056 | 0.0063 | 11.4 % | 0.0064 | 13.8 % | variable_dropout:0.5 |
| E4b_opera_strong | 1 | 0.0059 | 0.0065 | 10.7 % | 0.0067 | 13.1 % | variable_dropout:0.5 |

## Table 3 - Matched density (0.5), different structure (test MSE)

| method | seeds | density:0.5 | temporal_stride:2 | local_mask:0.5 | burst:0.5 | variable_dropout:0.5 | spread |
|---|---|---|---|---|---|---|---|
| E0 baseline (no OPERA) | 1 | 0.0074 | 0.0079 | - | 0.0080 | 0.0213 | 0.0140 |
| E1 + augmentation | 1 | 0.0056 | 0.0056 | - | 0.0058 | 0.0075 | 0.0020 |
| E2 + distillation | 1 | 0.0058 | 0.0058 | - | 0.0060 | 0.0078 | 0.0020 |
| E3 + hard mining | 1 | 0.0057 | 0.0058 | - | 0.0059 | 0.0076 | 0.0019 |
| E4 full OPERA | 1 | 0.0057 | 0.0058 | - | 0.0060 | 0.0074 | 0.0017 |
| E3b_hard_mining_strong | 1 | 0.0057 | 0.0058 | - | 0.0059 | 0.0072 | 0.0015 |
| E4b_opera_strong | 1 | 0.0060 | 0.0061 | - | 0.0062 | 0.0075 | 0.0014 |

## Table 4 - Compute and parameter overhead

| method | seeds | params | s / epoch | epochs | peak GPU MB | inference ms / series |
|---|---|---|---|---|---|---|
| E0 baseline (no OPERA) | 1 | 129153 | 24.1 | 33 | 3716 | 0.630 |
| E1 + augmentation | 1 | 129153 | 41.4 | 40 | 6321 | 0.670 |
| E2 + distillation | 1 | 129153 | 45.3 | 40 | 6282 | 0.630 |
| E3 + hard mining | 1 | 129153 | 111.1 | 36 | 6321 | 0.699 |
| E4 full OPERA | 1 | 129153 | 486.5 | 37 | 6282 | 0.627 |
| E3b_hard_mining_strong | 1 | 129153 | 82.3 | 40 | 6321 | 0.634 |
| E4b_opera_strong | 1 | 129153 | 86.3 | 33 | 6282 | 0.629 |

## All test conditions (test MSE)

| method | seeds | identity | random_dropout:0.3 | random_dropout:0.5 | random_dropout:0.7 | variable_dropout:0.3 | variable_dropout:0.5 | temporal_stride:2 | temporal_stride:3 | temporal_stride:4 | burst:0.3 | burst:0.5 | long_gap:0.4 | variable_group_dropout:0.3 | density:0.5 | random_dropout:0.5+variable_dropout:0.3 | temporal_stride:3+variable_group_dropout:0.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 baseline (no OPERA) | 1 | 0.0055 | 0.0083 | 0.0108 | 0.0152 | 0.0158 | 0.0213 | 0.0079 | 0.0092 | 0.0096 | 0.0070 | 0.0080 | 0.0078 | 0.0115 | 0.0074 | 0.0168 | 0.0139 |
| E1 + augmentation | 1 | 0.0055 | 0.0060 | 0.0064 | 0.0075 | 0.0069 | 0.0075 | 0.0056 | 0.0057 | 0.0058 | 0.0056 | 0.0058 | 0.0059 | 0.0063 | 0.0056 | 0.0077 | 0.0065 |
| E2 + distillation | 1 | 0.0057 | 0.0060 | 0.0064 | 0.0073 | 0.0070 | 0.0078 | 0.0058 | 0.0059 | 0.0060 | 0.0059 | 0.0060 | 0.0061 | 0.0064 | 0.0058 | 0.0076 | 0.0066 |
| E3 + hard mining | 1 | 0.0056 | 0.0061 | 0.0065 | 0.0074 | 0.0070 | 0.0076 | 0.0058 | 0.0059 | 0.0060 | 0.0058 | 0.0059 | 0.0060 | 0.0063 | 0.0057 | 0.0075 | 0.0066 |
| E4 full OPERA | 1 | 0.0056 | 0.0059 | 0.0064 | 0.0073 | 0.0068 | 0.0074 | 0.0058 | 0.0059 | 0.0060 | 0.0058 | 0.0060 | 0.0059 | 0.0062 | 0.0057 | 0.0073 | 0.0065 |
| E3b_hard_mining_strong | 1 | 0.0056 | 0.0060 | 0.0064 | 0.0071 | 0.0067 | 0.0072 | 0.0058 | 0.0058 | 0.0059 | 0.0058 | 0.0059 | 0.0060 | 0.0062 | 0.0057 | 0.0072 | 0.0063 |
| E4b_opera_strong | 1 | 0.0059 | 0.0062 | 0.0066 | 0.0073 | 0.0070 | 0.0075 | 0.0061 | 0.0062 | 0.0063 | 0.0061 | 0.0062 | 0.0062 | 0.0064 | 0.0060 | 0.0074 | 0.0067 |
