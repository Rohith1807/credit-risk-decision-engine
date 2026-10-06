| Model | Calibration | ROC-AUC | PR-AUC | Brier | Mean PD | Actual |
|---|---|---:|---:|---:|---:|---:|
| Logistic | Raw | 0.7428 | 0.1515 | 0.02650 | 3.39% | 2.85% |
| Logistic | Sigmoid | 0.7428 | 0.1515 | 0.02633 | 3.75% | 2.85% |
| LightGBM | **Raw** | **0.7611** | **0.1671** | 0.02646 | **2.86%** | 2.85% |
| LightGBM | Sigmoid | 0.7611 | 0.1671 | **0.02631** | 4.30% | 2.85% |

Unweighted LightGBM with raw probabilities was selected as the provisional champion. Its validation mean predicted PD of 2.86% closely matched the observed 2.85% default rate while maintaining the strongest overall combination of ROC-AUC, PR-AUC, and Brier score. Sigmoid calibration slightly improved Brier score but materially overestimated portfolio-level default probability. Isotonic calibration was rejected because the calibration sample contained only 19 defaults and produced unstable probability mappings and degraded ranking performance.