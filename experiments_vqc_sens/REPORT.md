# Revision experiment report
_generated 2026-06-12T20:53:13_

## Clean test performance (aggregated across seeds)

| model | noise_aware | seeds | head params | trainable | acc (mean±std) | auc | f1 |
|---|---|---|---|---|---|---|---|
| qresnet_q4l2strong | False | 0 | 2086 | 11178598 | 100.00±0.00 | 1.0000 | 1.0000 |
| qresnet_q4l3strong | False | 0 | 2098 | 11178610 | 100.00±0.00 | 1.0000 | 1.0000 |
| qresnet_q6l2strong | False | 0 | 3128 | 11179640 | 100.00±0.00 | 1.0000 | 1.0000 |
| qresnet_q6l3strong | False | 0 | 3146 | 11179658 | 99.99±0.00 | 1.0000 | 0.9999 |
| qresnet_q8l2strong | False | 0 | 4170 | 11180682 | 100.00±0.00 | 1.0000 | 1.0000 |
| qresnet_q8l3strong | False | 0 | 4194 | 11180706 | 99.99±0.00 | 1.0000 | 0.9999 |

## Robustness summary (AURC, sigma*)

_no *_metrics.csv found (run revision.robustness)_

