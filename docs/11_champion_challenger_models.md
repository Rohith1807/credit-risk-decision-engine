| Model | ROC-AUC | PR-AUC | Brier | Mean PD |
|---|---:|---:|---:|---:|
| **Logistic Regression** | 0.7309 | **0.1757** | 0.0267 | 3.30% |
| XGBoost Unweighted | 0.7198 | 0.1019 | 0.0285 | 3.16% |
| XGBoost Weighted | 0.7781 | 0.0992 | 0.0653 | 14.46% |
| **LightGBM Unweighted** | **0.8010** | 0.1638 | **0.0262** | **3.00%** |
| LightGBM Weighted | 0.7869 | 0.1441 | 0.0713 | 15.97% |

Current champion:
Unweighted Logistic Regression

Primary challenger:
Unweighted LightGBM

Secondary challenger:
Unweighted XGBoost

XGBoost weighted mean PD  = 14.46%
LightGBM weighted mean PD = 15.97%

Actual default rate       = 2.85%

Logistic Regression remains the provisional champion based on the highest validation PR-AUC and strong probability calibration. Unweighted LightGBM is retained as the primary challenger because it achieved substantially higher ROC-AUC and slightly better Brier score. Given the small number of validation defaults, final model selection is deferred until evaluation on a larger portfolio.