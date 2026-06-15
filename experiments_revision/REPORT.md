# Revision experiment report
_generated 2026-06-15T18:41:08_

## Clean test performance (aggregated across seeds)

| model | noise_aware | seeds | head params | trainable | acc (mean±std) | auc | f1 |
|---|---|---|---|---|---|---|---|
| bottleneck_fc | False | 0,1,2 | 2062 | 11178574 | 99.99±0.00 | 1.0000 | 0.9999 |
| classic_fc | False | 0,1,2 | 1026 | 11177538 | 99.99±0.00 | 1.0000 | 0.9999 |
| classic_fc | True | 0 | 1026 | 11177538 | 100.00±0.00 | 1.0000 | 0.9999 |
| mlp_head | False | 0,1,2 | 2082 | 11178594 | 99.99±0.00 | 1.0000 | 0.9999 |
| mlp_head | True | 0 | 2082 | 11178594 | 99.23±0.00 | 0.9924 | 0.9924 |
| qresnet_q4l2strong | False | 0,1,2 | 2086 | 11178598 | 100.00±0.00 | 1.0000 | 1.0000 |

## Robustness summary (AURC, sigma*)

| model | noise_aware | threat | AURC | sigma* |
|---|---|---|---|---|
| bottleneck_fc | False | noise | 81.08 | 0.14 |
| classic_fc | False | noise | 76.33 | 0.12 |
| classic_fc | True | noise | 99.98 | 0.2 |
| mlp_head | False | noise | 79.70 | 0.14 |
| mlp_head | True | noise | 99.23 | 0.2 |
| qresnet_q4l2strong | False | noise | 73.09 | 0.1 |
| bottleneck_fc | False | occlusion | 98.50 | 100.0 |
| classic_fc | False | occlusion | 98.69 | 100.0 |
| classic_fc | True | occlusion | 99.36 | 100.0 |
| mlp_head | False | occlusion | 97.60 | 100.0 |
| mlp_head | True | occlusion | 98.46 | 100.0 |
| qresnet_q4l2strong | False | occlusion | 95.86 | 100.0 |

**Ablation verdict (noise AURC):** best VQC `qresnet_q4l2strong`=73.09 vs best classical head `bottleneck_fc`=81.08 -> gap -7.99. Classical head leads; the quantum advantage does NOT hold here, reframe honestly.

## Accuracy vs severity (mean±std)

### noise

| level | bottleneck_fc | classic_fc | classic_fc* | mlp_head | mlp_head* | qresnet_q4l2strong |
|---|---|---|---|---|---|---|
| 0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | 99.2±0.0 | 100.0±0.0 |
| 0.06 | 99.3±0.5 | 87.5±15.1 | 100.0±0.0 | 99.6±0.3 | 99.2±0.0 | 90.7±10.6 |
| 0.08 | 95.8±2.8 | 79.0±20.1 | 100.0±0.0 | 95.7±3.0 | 99.2±0.0 | 78.3±17.8 |
| 0.1 | 87.5±5.4 | 73.3±18.6 | 100.0±0.0 | 83.6±11.4 | 99.2±0.0 | 69.5±17.5 |
| 0.12 | 75.4±4.8 | 68.2±18.2 | 100.0±0.0 | 70.3±18.1 | 99.2±0.0 | 63.4±13.6 |
| 0.14 | 64.3±2.9 | 64.7±18.2 | 100.0±0.0 | 62.5±16.1 | 99.2±0.0 | 57.9±8.8 |
| 0.16 | 57.5±1.4 | 62.6±17.1 | 100.0±0.0 | 56.9±9.6 | 99.2±0.0 | 53.5±4.3 |
| 0.2 | 52.9±0.9 | 59.1±12.9 | 99.9±0.0 | 50.5±0.7 | 99.2±0.0 | 50.2±0.3 |

### occlusion

| level | bottleneck_fc | classic_fc | classic_fc* | mlp_head | mlp_head* | qresnet_q4l2strong |
|---|---|---|---|---|---|---|
| 0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | 99.2±0.0 | 100.0±0.0 |
| 40 | 99.3±0.3 | 99.4±0.4 | 99.8±0.0 | 98.5±0.7 | 98.9±0.0 | 97.1±1.7 |
| 80 | 97.7±1.4 | 97.7±0.3 | 99.0±0.0 | 96.0±1.8 | 98.0±0.0 | 93.3±3.4 |
| 100 | 94.9±1.8 | 96.0±0.4 | 97.3±0.0 | 93.8±2.8 | 96.6±0.0 | 90.4±4.2 |

_models marked * are noise-aware trained._
