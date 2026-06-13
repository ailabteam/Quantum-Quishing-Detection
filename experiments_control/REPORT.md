# Revision experiment report
_generated 2026-06-13T02:55:12_

## Clean test performance (aggregated across seeds)

| model | noise_aware | seeds | head params | trainable | acc (mean±std) | auc | f1 |
|---|---|---|---|---|---|---|---|
| bottleneck_fc_b6 | False | 0,1,2 | 3092 | 11179604 | 99.99±0.00 | 1.0000 | 0.9999 |
| mlp_head_b6 | False | 0,1,2 | 3116 | 11179628 | 99.99±0.01 | 1.0000 | 0.9999 |
| qresnet_q6l2strong | False | 0,1,2 | 3128 | 11179640 | 99.99±0.00 | 1.0000 | 0.9999 |
| qresnet_q8l2strong | False | 0,1,2 | 4170 | 11180682 | 99.99±0.00 | 1.0000 | 0.9999 |

## Robustness summary (AURC, sigma*)

_no *_metrics.csv found (run revision.robustness)_

