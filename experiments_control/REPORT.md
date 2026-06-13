# Revision experiment report
_generated 2026-06-13T20:37:47_

## Clean test performance (aggregated across seeds)

| model | noise_aware | seeds | head params | trainable | acc (mean±std) | auc | f1 |
|---|---|---|---|---|---|---|---|
| bottleneck_fc_b6 | False | 0,1,2 | 3092 | 11179604 | 99.99±0.00 | 1.0000 | 0.9999 |
| mlp_head_b6 | False | 0,1,2 | 3116 | 11179628 | 99.99±0.01 | 1.0000 | 0.9999 |
| qresnet_q6l2strong | False | 0,1,2 | 3128 | 11179640 | 99.99±0.00 | 1.0000 | 0.9999 |
| qresnet_q8l2strong | False | 0,1,2 | 4170 | 11180682 | 99.99±0.00 | 1.0000 | 0.9999 |

## Robustness summary (AURC, sigma*)

| model | noise_aware | threat | AURC | sigma* |
|---|---|---|---|---|
| bottleneck_fc_b6 | False | noise | 85.42 | 0.16 |
| mlp_head_b6 | False | noise | 88.33 | 0.2 |
| qresnet_q6l2strong | False | noise | 79.41 | 0.16 |
| qresnet_q8l2strong | False | noise | 72.27 | 0.12 |
| bottleneck_fc_b6 | False | occlusion | 97.69 | 100.0 |
| mlp_head_b6 | False | occlusion | 96.61 | 100.0 |
| qresnet_q6l2strong | False | occlusion | 95.86 | 100.0 |
| qresnet_q8l2strong | False | occlusion | 96.31 | 100.0 |

_no classical ablation heads for verdict._

## Accuracy vs severity (mean±std)

### noise

| level | bottleneck_fc_b6 | mlp_head_b6 | qresnet_q6l2strong | qresnet_q8l2strong |
|---|---|---|---|---|
| 0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 |
| 0.08 | 97.3±1.9 | 98.5±1.8 | 88.1±14.0 | 79.3±19.8 |
| 0.1 | 90.0±7.9 | 94.2±6.8 | 77.8±17.3 | 73.7±17.6 |
| 0.12 | 80.3±16.5 | 87.3±13.7 | 74.0±19.2 | 65.3±11.7 |
| 0.14 | 73.7±19.3 | 80.4±17.3 | 70.7±19.0 | 60.8±17.9 |
| 0.16 | 69.1±18.9 | 74.8±16.6 | 65.5±19.0 | 49.6±3.7 |
| 0.2 | 63.3±16.3 | 62.8±9.1 | 53.1±13.7 | 50.2±1.6 |

### occlusion

| level | bottleneck_fc_b6 | mlp_head_b6 | qresnet_q6l2strong | qresnet_q8l2strong |
|---|---|---|---|---|
| 0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 |
| 100 | 95.4±1.7 | 93.2±1.0 | 91.7±0.5 | 92.6±1.9 |

_models marked * are noise-aware trained._
