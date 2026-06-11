# Revision experiment report
_generated 2026-06-12T02:22:41_

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
| bottleneck_fc | False | noise | 58.54 | 0.2 |
| classic_fc | False | noise | 58.38 | 0.2 |
| mlp_head | False | noise | 58.68 | 0.2 |
| qresnet | False | noise | 58.87 | 0.2 |
| bottleneck_fc | False | occlusion | 97.52 | 100.0 |
| classic_fc | False | occlusion | 98.60 | 100.0 |
| mlp_head | False | occlusion | 97.12 | 100.0 |
| qresnet | False | occlusion | 96.89 | 100.0 |

**Ablation verdict (noise AURC):** qresnet=58.87 vs best classical head `mlp_head`=58.68 -> gap +0.19. Within ~2 pts of the best classical head; likely no clear quantum advantage, reframe honestly.

## Accuracy vs severity (mean±std)

### noise

| level | bottleneck_fc | classic_fc | mlp_head | qresnet |
|---|---|---|---|---|
| 0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 |
| 0.2 | 52.5±0.0 | 50.1±0.0 | 51.4±0.1 | 50.2±0.0 |
| 0.3 | 50.2±0.0 | 50.0±0.0 | 50.0±0.0 | 51.9±0.1 |
| 0.4 | 49.1±0.2 | 50.2±0.0 | 50.0±0.0 | 50.2±0.0 |
| 0.5 | 49.2±0.1 | 50.0±0.0 | 50.0±0.0 | 50.5±0.0 |
| 0.6 | 48.1±0.2 | 50.0±0.0 | 50.0±0.0 | 50.7±0.2 |

### occlusion

| level | bottleneck_fc | classic_fc | mlp_head | qresnet |
|---|---|---|---|---|
| 0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 |
| 40 | 98.8±0.0 | 99.6±0.0 | 98.0±0.0 | 97.4±0.0 |
| 80 | 95.8±0.1 | 97.3±0.0 | 95.4±0.0 | 95.2±0.1 |
| 100 | 92.5±0.0 | 95.6±0.0 | 93.2±0.0 | 94.0±0.0 |

_models marked * are noise-aware trained._
