# OPERA aggregated results

## Table 1 - Robustness to observation-process shift (test MSE, mean +- std over seeds)

Conditions marked * never occur in the training observation-process pool.

| method | seeds | clean | rnd .5 | rnd .7 | var .3 | var .5* | stride 3* | long gap* | burst .5* | var group* |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 baseline (no OPERA) | 1 | 0.1606 | 0.1587 | 0.1708 | 0.1826 | 0.1841 | 0.1658 | 0.1550 | 0.1572 | 0.1597 |
| E4 full OPERA | 1 | 0.1463 | 0.1429 | 0.1536 | 0.1634 | 0.1615 | 0.1452 | 0.1387 | 0.1423 | 0.1513 |

## Table 2 - Component contribution

| method | seeds | clean MSE | mean shifted MSE | mean degradation | unseen MSE | unseen degradation | worst condition |
|---|---|---|---|---|---|---|---|
| E0 baseline (no OPERA) | 1 | 0.1606 | 0.1653 | 2.9 % | 0.1658 | 3.3 % | variable_dropout:0.5 |
| E4 full OPERA | 1 | 0.1463 | 0.1481 | 1.2 % | 0.1488 | 1.7 % | variable_dropout:0.3 |

## Table 3 - Matched density (0.5), different structure (test MSE)

| method | seeds | density:0.5 | temporal_stride:2 | local_mask:0.5 | burst:0.5 | variable_dropout:0.5 | spread |
|---|---|---|---|---|---|---|---|
| E0 baseline (no OPERA) | 1 | 0.1631 | 0.1654 | - | 0.1572 | 0.1841 | 0.0269 |
| E4 full OPERA | 1 | 0.1446 | 0.1457 | - | 0.1423 | 0.1615 | 0.0192 |

## Table 4 - Compute and parameter overhead

| method | seeds | params | s / epoch | epochs | peak GPU MB | inference ms / series |
|---|---|---|---|---|---|---|
| E0 baseline (no OPERA) | 1 | 108392 | 10.2 | 17 | 1395 | 0.337 |
| E4 full OPERA | 1 | 108392 | 124.6 | 23 | 1884 | 0.990 |

## All test conditions (test MSE)

| method | seeds | identity | random_dropout:0.3 | random_dropout:0.5 | random_dropout:0.7 | variable_dropout:0.3 | variable_dropout:0.5 | temporal_stride:2 | temporal_stride:3 | temporal_stride:4 | burst:0.3 | burst:0.5 | long_gap:0.4 | variable_group_dropout:0.3 | density:0.5 | random_dropout:0.5+variable_dropout:0.3 | temporal_stride:3+variable_group_dropout:0.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 baseline (no OPERA) | 1 | 0.1606 | 0.1580 | 0.1587 | 0.1708 | 0.1826 | 0.1841 | 0.1654 | 0.1658 | 0.1715 | 0.1584 | 0.1572 | 0.1550 | 0.1597 | 0.1631 | 0.1649 | 0.1638 |
| E4 full OPERA | 1 | 0.1463 | 0.1419 | 0.1429 | 0.1536 | 0.1634 | 0.1615 | 0.1457 | 0.1452 | 0.1475 | 0.1434 | 0.1423 | 0.1387 | 0.1513 | 0.1446 | 0.1483 | 0.1508 |
