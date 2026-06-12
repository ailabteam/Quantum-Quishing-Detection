# Revision experiment report
_generated 2026-06-12T12:21:37_

## Clean test performance (aggregated across seeds)

| model | noise_aware | seeds | head params | trainable | acc (mean±std) | auc | f1 |
|---|---|---|---|---|---|---|---|
| bottleneck_fc | False | 0 | 2062 | 11178574 | 99.99±0.00 | 1.0000 | 0.9999 |
| classic_fc | False | 0 | 1026 | 11177538 | 99.99±0.00 | 1.0000 | 0.9999 |
| mlp_head | False | 0 | 2082 | 11178594 | 99.99±0.00 | 0.9999 | 0.9999 |
| qresnet | False | 0 | 2086 | 11178598 | 100.00±0.00 | 1.0000 | 1.0000 |

## Robustness summary (AURC, sigma*)

| model | noise_aware | threat | AURC | sigma* |
|---|---|---|---|---|
| bottleneck_fc | False | noise | 83.54 | 0.14 |
| classic_fc | False | noise | 75.19 | 0.12 |
| mlp_head | False | noise | 88.75 | 0.2 |
| qresnet | False | noise | 80.17 | 0.14 |
| bottleneck_fc | False | occlusion | 97.50 | 100.0 |
| classic_fc | False | occlusion | 98.58 | 100.0 |
| mlp_head | False | occlusion | 97.13 | 100.0 |
| qresnet | False | occlusion | 96.88 | 100.0 |

**Ablation verdict (noise AURC):** qresnet=80.17 vs best classical head `mlp_head`=88.75 -> gap -8.58. Classical head leads; the quantum advantage does NOT hold here, reframe honestly.

## Accuracy vs severity (mean±std)

### noise

| level | bottleneck_fc | classic_fc | mlp_head | qresnet |
|---|---|---|---|---|
| 0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 |
| 0.02 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 |
| 0.04 | 100.0±0.0 | 99.8±0.0 | 100.0±0.0 | 100.0±0.0 |
| 0.06 | 99.9±0.0 | 97.1±0.0 | 100.0±0.0 | 99.7±0.0 |
| 0.08 | 99.2±0.0 | 88.2±0.1 | 99.9±0.0 | 98.0±0.0 |
| 0.1 | 94.3±0.1 | 74.7±0.2 | 99.3±0.0 | 90.7±0.1 |
| 0.12 | 81.8±0.1 | 61.7±0.2 | 95.8±0.0 | 75.3±0.2 |
| 0.14 | 68.4±0.1 | 53.7±0.1 | 85.3±0.1 | 59.5±0.1 |
| 0.16 | 59.5±0.1 | 51.0±0.0 | 70.6±0.1 | 52.1±0.1 |
| 0.2 | 52.5±0.1 | 50.1±0.0 | 51.4±0.1 | 50.2±0.0 |

### occlusion

| level | bottleneck_fc | classic_fc | mlp_head | qresnet |
|---|---|---|---|---|
| 0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 |
| 40 | 98.8±0.0 | 99.6±0.0 | 98.0±0.0 | 97.3±0.0 |
| 80 | 95.7±0.1 | 97.3±0.0 | 95.4±0.0 | 95.1±0.1 |
| 100 | 92.5±0.0 | 95.6±0.1 | 93.1±0.0 | 94.0±0.0 |

_models marked * are noise-aware trained._
