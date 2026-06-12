# Revision experiment report
_generated 2026-06-12T21:12:32_

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

| model | noise_aware | threat | AURC | sigma* |
|---|---|---|---|---|
| qresnet_q4l2strong | False | noise | 73.19 | 0.1 |
| qresnet_q4l3strong | False | noise | 63.73 | 0.08 |
| qresnet_q6l2strong | False | noise | 95.22 | 0.2 |
| qresnet_q6l3strong | False | noise | 77.10 | 0.14 |
| qresnet_q8l2strong | False | noise | 61.21 | 0.06 |
| qresnet_q8l3strong | False | noise | 94.80 | 0.2 |
| qresnet_q4l2strong | False | occlusion | 97.13 | 100.0 |
| qresnet_q4l3strong | False | occlusion | 96.91 | 100.0 |
| qresnet_q6l2strong | False | occlusion | 96.20 | 100.0 |
| qresnet_q6l3strong | False | occlusion | 98.42 | 100.0 |
| qresnet_q8l2strong | False | occlusion | 97.96 | 100.0 |
| qresnet_q8l3strong | False | occlusion | 97.28 | 100.0 |

_no classical ablation heads for verdict._

## Accuracy vs severity (mean±std)

### noise

| level | qresnet_q4l2strong | qresnet_q4l3strong | qresnet_q6l2strong | qresnet_q6l3strong | qresnet_q8l2strong | qresnet_q8l3strong |
|---|---|---|---|---|---|---|
| 0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 |
| 0.06 | 96.4±0.1 | 78.5±0.2 | 100.0±0.0 | 95.6±0.0 | 67.4±0.1 | 99.9±0.0 |
| 0.08 | 80.6±0.1 | 53.6±0.0 | 99.8±0.0 | 83.0±0.1 | 52.1±0.1 | 99.3±0.0 |
| 0.1 | 68.6±0.1 | 50.8±0.1 | 99.3±0.0 | 76.3±0.2 | 50.1±0.0 | 98.0±0.1 |
| 0.12 | 59.9±0.1 | 50.8±0.0 | 97.7±0.0 | 71.0±0.2 | 50.0±0.0 | 95.8±0.0 |
| 0.14 | 53.6±0.1 | 50.2±0.0 | 95.9±0.1 | 62.8±0.2 | 50.0±0.0 | 92.2±0.0 |
| 0.16 | 50.9±0.0 | 50.0±0.0 | 92.2±0.1 | 56.5±0.2 | 50.0±0.0 | 87.6±0.1 |
| 0.2 | 50.0±0.0 | 50.0±0.0 | 71.4±0.1 | 51.9±0.2 | 50.0±0.0 | 81.5±0.1 |

### occlusion

| level | qresnet_q4l2strong | qresnet_q4l3strong | qresnet_q6l2strong | qresnet_q6l3strong | qresnet_q8l2strong | qresnet_q8l3strong |
|---|---|---|---|---|---|---|
| 0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 |
| 40 | 97.8±0.0 | 97.6±0.1 | 96.9±0.1 | 99.2±0.0 | 98.7±0.0 | 98.8±0.0 |
| 80 | 95.4±0.1 | 95.1±0.1 | 94.2±0.0 | 97.6±0.0 | 96.7±0.1 | 94.9±0.1 |
| 100 | 93.8±0.0 | 93.3±0.1 | 91.9±0.0 | 94.8±0.1 | 94.8±0.1 | 92.8±0.1 |

_models marked * are noise-aware trained._
