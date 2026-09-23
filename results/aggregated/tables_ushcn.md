# OPERA aggregated results

## Table 1 - Robustness to observation-process shift (test MSE, mean +- std over seeds)

Conditions marked * never occur in the training observation-process pool.

| method | seeds | clean | rnd .5 | rnd .7 | var .3 | var .5* | stride 3* | long gap* | burst .5* | var group* |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 baseline (no OPERA) | 1 | 0.5379 | 0.5451 | 0.5607 | 0.5475 | 0.5693 | 0.5401 | 0.5457 | 0.5465 | 0.5544 |
| E1 + augmentation | 1 | 0.5329 | 0.5406 | 0.5552 | 0.5458 | 0.5696 | 0.5330 | 0.5396 | 0.5418 | 0.5541 |
| E2 + distillation | 1 | 0.5135 | 0.5241 | 0.5422 | 0.5279 | 0.5488 | 0.5185 | 0.5221 | 0.5214 | 0.5307 |
| E3 + hard mining | 1 | 0.5266 | 0.5471 | 0.5660 | 0.5451 | 0.5689 | 0.5408 | 0.5386 | 0.5401 | 0.5528 |
| E4 full OPERA | 1 | 0.5211 | 0.5245 | 0.5378 | 0.5305 | 0.5458 | 0.5222 | 0.5278 | 0.5296 | 0.5313 |

## Table 2 - Component contribution

| method | seeds | clean MSE | mean shifted MSE | mean degradation | unseen MSE | unseen degradation | worst condition |
|---|---|---|---|---|---|---|---|
| E0 baseline (no OPERA) | 1 | 0.5379 | 0.5481 | 1.9 % | 0.5527 | 2.8 % | variable_dropout:0.5 |
| E1 + augmentation | 1 | 0.5329 | 0.5438 | 2.0 % | 0.5490 | 3.0 % | variable_dropout:0.5 |
| E2 + distillation | 1 | 0.5135 | 0.5266 | 2.6 % | 0.5314 | 3.5 % | variable_dropout:0.5 |
| E3 + hard mining | 1 | 0.5266 | 0.5475 | 4.0 % | 0.5535 | 5.1 % | variable_dropout:0.5 |
| E4 full OPERA | 1 | 0.5211 | 0.5287 | 1.5 % | 0.5324 | 2.2 % | variable_dropout:0.5 |

## Table 3 - Matched density (0.5), different structure (test MSE)

| method | seeds | density:0.5 | temporal_stride:2 | local_mask:0.5 | burst:0.5 | variable_dropout:0.5 | spread |
|---|---|---|---|---|---|---|---|
| E0 baseline (no OPERA) | 1 | 0.5355 | 0.5374 | - | 0.5465 | 0.5693 | 0.0338 |
| E1 + augmentation | 1 | 0.5280 | 0.5309 | - | 0.5418 | 0.5696 | 0.0416 |
| E2 + distillation | 1 | 0.5158 | 0.5155 | - | 0.5214 | 0.5488 | 0.0333 |
| E3 + hard mining | 1 | 0.5351 | 0.5349 | - | 0.5401 | 0.5689 | 0.0340 |
| E4 full OPERA | 1 | 0.5199 | 0.5196 | - | 0.5296 | 0.5458 | 0.0263 |

## Table 4 - Compute and parameter overhead

| method | seeds | params | s / epoch | epochs | peak GPU MB | inference ms / series |
|---|---|---|---|---|---|---|
| E0 baseline (no OPERA) | 1 | 126849 | 23.0 | 33 | 1675 | 0.118 |
| E1 + augmentation | 1 | 126849 | 29.7 | 40 | 2632 | 0.119 |
| E2 + distillation | 1 | 126849 | 32.0 | 40 | 2882 | 0.121 |
| E3 + hard mining | 1 | 126849 | 46.8 | 40 | 2632 | 0.124 |
| E4 full OPERA | 1 | 126849 | 48.8 | 40 | 2882 | 0.128 |

## All test conditions (test MSE)

| method | seeds | identity | random_dropout:0.3 | random_dropout:0.5 | random_dropout:0.7 | variable_dropout:0.3 | variable_dropout:0.5 | temporal_stride:2 | temporal_stride:3 | temporal_stride:4 | burst:0.3 | burst:0.5 | long_gap:0.4 | variable_group_dropout:0.3 | density:0.5 | random_dropout:0.5+variable_dropout:0.3 | temporal_stride:3+variable_group_dropout:0.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 baseline (no OPERA) | 1 | 0.5379 | 0.5372 | 0.5451 | 0.5607 | 0.5475 | 0.5693 | 0.5374 | 0.5401 | 0.5418 | 0.5438 | 0.5465 | 0.5457 | 0.5544 | 0.5355 | 0.5589 | 0.5572 |
| E1 + augmentation | 1 | 0.5329 | 0.5320 | 0.5406 | 0.5552 | 0.5458 | 0.5696 | 0.5309 | 0.5330 | 0.5350 | 0.5386 | 0.5418 | 0.5396 | 0.5541 | 0.5280 | 0.5565 | 0.5562 |
| E2 + distillation | 1 | 0.5135 | 0.5156 | 0.5241 | 0.5422 | 0.5279 | 0.5488 | 0.5155 | 0.5185 | 0.5214 | 0.5183 | 0.5214 | 0.5221 | 0.5307 | 0.5158 | 0.5402 | 0.5369 |
| E3 + hard mining | 1 | 0.5266 | 0.5343 | 0.5471 | 0.5660 | 0.5451 | 0.5689 | 0.5349 | 0.5408 | 0.5434 | 0.5355 | 0.5401 | 0.5386 | 0.5528 | 0.5351 | 0.5652 | 0.5654 |
| E4 full OPERA | 1 | 0.5211 | 0.5190 | 0.5245 | 0.5378 | 0.5305 | 0.5458 | 0.5196 | 0.5222 | 0.5243 | 0.5260 | 0.5296 | 0.5278 | 0.5313 | 0.5199 | 0.5369 | 0.5355 |
