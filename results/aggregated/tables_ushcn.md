# OPERA aggregated results

## Table 1 - Robustness to observation-process shift (test MSE, mean +- std over seeds)

Conditions marked * never occur in the training observation-process pool.

| method | seeds | clean | rnd .5 | rnd .7 | var .3 | var .5* | stride 3* | long gap* | burst .5* | var group* |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 baseline (no OPERA) | 2 | 0.5290 +- 0.0088 | 0.5403 +- 0.0047 | 0.5545 +- 0.0062 | 0.5435 +- 0.0040 | 0.5612 +- 0.0082 | 0.5373 +- 0.0028 | 0.5397 +- 0.0059 | 0.5436 +- 0.0028 | 0.5479 +- 0.0065 |
| E1 + augmentation | 1 | 0.5329 | 0.5406 | 0.5552 | 0.5458 | 0.5696 | 0.5330 | 0.5396 | 0.5418 | 0.5541 |
| E2 + distillation | 1 | 0.5135 | 0.5241 | 0.5422 | 0.5279 | 0.5488 | 0.5185 | 0.5221 | 0.5214 | 0.5307 |
| E3 + hard mining | 1 | 0.5266 | 0.5471 | 0.5660 | 0.5451 | 0.5689 | 0.5408 | 0.5386 | 0.5401 | 0.5528 |
| E4 full OPERA | 2 | 0.5190 +- 0.0021 | 0.5257 +- 0.0012 | 0.5371 +- 0.0007 | 0.5306 +- 0.0001 | 0.5435 +- 0.0023 | 0.5236 +- 0.0014 | 0.5268 +- 0.0010 | 0.5314 +- 0.0018 | 0.5329 +- 0.0016 |

## Table 2 - Component contribution

| method | seeds | clean MSE | mean shifted MSE | mean degradation | unseen MSE | unseen degradation | worst condition |
|---|---|---|---|---|---|---|---|
| E0 baseline (no OPERA) | 2 | 0.5290 +- 0.0088 | 0.5436 +- 0.0044 | 2.8 +- 0.9 % | 0.5483 +- 0.0044 | 3.7 +- 0.9 % | variable_dropout:0.5 |
| E1 + augmentation | 1 | 0.5329 | 0.5438 | 2.0 % | 0.5490 | 3.0 % | variable_dropout:0.5 |
| E2 + distillation | 1 | 0.5135 | 0.5266 | 2.6 % | 0.5314 | 3.5 % | variable_dropout:0.5 |
| E3 + hard mining | 1 | 0.5266 | 0.5475 | 4.0 % | 0.5535 | 5.1 % | variable_dropout:0.5 |
| E4 full OPERA | 2 | 0.5190 +- 0.0021 | 0.5295 +- 0.0008 | 2.0 +- 0.6 % | 0.5333 +- 0.0010 | 2.8 +- 0.6 % | variable_dropout:0.5 |

## Table 3 - Matched density (0.5), different structure (test MSE)

| method | seeds | density:0.5 | temporal_stride:2 | local_mask:0.5 | burst:0.5 | variable_dropout:0.5 | spread |
|---|---|---|---|---|---|---|---|
| E0 baseline (no OPERA) | 2 | 0.5324 +- 0.0031 | 0.5333 +- 0.0041 | - | 0.5436 +- 0.0028 | 0.5612 +- 0.0082 | 0.0288 +- 0.0050 |
| E1 + augmentation | 1 | 0.5280 | 0.5309 | - | 0.5418 | 0.5696 | 0.0416 |
| E2 + distillation | 1 | 0.5158 | 0.5155 | - | 0.5214 | 0.5488 | 0.0333 |
| E3 + hard mining | 1 | 0.5351 | 0.5349 | - | 0.5401 | 0.5689 | 0.0340 |
| E4 full OPERA | 2 | 0.5205 +- 0.0006 | 0.5205 +- 0.0009 | - | 0.5314 +- 0.0018 | 0.5435 +- 0.0023 | 0.0232 +- 0.0030 |

## Table 4 - Compute and parameter overhead

| method | seeds | params | s / epoch | epochs | peak GPU MB | inference ms / series |
|---|---|---|---|---|---|---|
| E0 baseline (no OPERA) | 2 | 126849 | 12.4 +- 10.6 | 36 +- 4 | 1465 +- 210 | 0.137 +- 0.020 |
| E1 + augmentation | 1 | 126849 | 29.7 | 40 | 2632 | 0.119 |
| E2 + distillation | 1 | 126849 | 32.0 | 40 | 2882 | 0.121 |
| E3 + hard mining | 1 | 126849 | 46.8 | 40 | 2632 | 0.124 |
| E4 full OPERA | 2 | 126849 | 26.6 +- 22.2 | 40 +- 0 | 2970 +- 89 | 0.142 +- 0.015 |

## All test conditions (test MSE)

| method | seeds | identity | random_dropout:0.3 | random_dropout:0.5 | random_dropout:0.7 | variable_dropout:0.3 | variable_dropout:0.5 | temporal_stride:2 | temporal_stride:3 | temporal_stride:4 | burst:0.3 | burst:0.5 | long_gap:0.4 | variable_group_dropout:0.3 | density:0.5 | random_dropout:0.5+variable_dropout:0.3 | temporal_stride:3+variable_group_dropout:0.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 baseline (no OPERA) | 2 | 0.5290 +- 0.0088 | 0.5325 +- 0.0048 | 0.5403 +- 0.0047 | 0.5545 +- 0.0062 | 0.5435 +- 0.0040 | 0.5612 +- 0.0082 | 0.5333 +- 0.0041 | 0.5373 +- 0.0028 | 0.5397 +- 0.0021 | 0.5376 +- 0.0062 | 0.5436 +- 0.0028 | 0.5397 +- 0.0059 | 0.5479 +- 0.0065 | 0.5324 +- 0.0031 | 0.5557 +- 0.0032 | 0.5553 +- 0.0019 |
| E1 + augmentation | 1 | 0.5329 | 0.5320 | 0.5406 | 0.5552 | 0.5458 | 0.5696 | 0.5309 | 0.5330 | 0.5350 | 0.5386 | 0.5418 | 0.5396 | 0.5541 | 0.5280 | 0.5565 | 0.5562 |
| E2 + distillation | 1 | 0.5135 | 0.5156 | 0.5241 | 0.5422 | 0.5279 | 0.5488 | 0.5155 | 0.5185 | 0.5214 | 0.5183 | 0.5214 | 0.5221 | 0.5307 | 0.5158 | 0.5402 | 0.5369 |
| E3 + hard mining | 1 | 0.5266 | 0.5343 | 0.5471 | 0.5660 | 0.5451 | 0.5689 | 0.5349 | 0.5408 | 0.5434 | 0.5355 | 0.5401 | 0.5386 | 0.5528 | 0.5351 | 0.5652 | 0.5654 |
| E4 full OPERA | 2 | 0.5190 +- 0.0021 | 0.5199 +- 0.0010 | 0.5257 +- 0.0012 | 0.5371 +- 0.0007 | 0.5306 +- 0.0001 | 0.5435 +- 0.0023 | 0.5205 +- 0.0009 | 0.5236 +- 0.0014 | 0.5259 +- 0.0017 | 0.5257 +- 0.0003 | 0.5314 +- 0.0018 | 0.5268 +- 0.0010 | 0.5329 +- 0.0016 | 0.5205 +- 0.0006 | 0.5394 +- 0.0025 | 0.5391 +- 0.0035 |
