# Exploratory Data Analysis — Diabetes 30-Day Readmission

Dataset: **69,987 encounters × 16 model features** (+ `readmitted_binary` target). Source: engineered feature set (`data/processed/diabetic_data_features.csv`).

## Target balance
- Readmitted (`readmitted_binary=1`): **40.7%**  ·  Not readmitted: **59.3%**
- Mild class imbalance — worth noting for threshold/metric choices (the models report F1 alongside accuracy for this reason).

## Missingness (model features)
- No missing values in the 16 model features (cleaning upstream fills/encodes them).

## Strongest linear correlations with the target
| Feature | corr with readmit |
|---|---|
| `number_inpatient` | +0.146 |
| `number_diagnoses` | +0.111 |
| `number_emergency` | +0.077 |
| `age_ordinal` | +0.076 |
| `number_outpatient` | +0.066 |
| `time_in_hospital` | +0.065 |
| `admission_type_id` | -0.064 |
| `num_lab_procedures` | +0.055 |

> Correlations are weak and roughly linear — consistent with the models' modest AUC (~0.65): readmission is driven by many small signals rather than one dominant feature. `number_inpatient` (prior admissions) is the most informative single feature.

## Numeric feature summary

|                          |   count |   mean |   std |   min |   25% |   50% |   75% |   max |
|:-------------------------|--------:|-------:|------:|------:|------:|------:|------:|------:|
| time_in_hospital         |   69987 |   4.27 |  2.93 |     1 |     2 |     3 |     6 |    14 |
| num_lab_procedures       |   69987 |  42.88 | 19.89 |     1 |    31 |    44 |    57 |   132 |
| num_procedures           |   69987 |   1.43 |  1.76 |     0 |     0 |     1 |     2 |     6 |
| num_medications          |   69987 |  15.67 |  8.29 |     1 |    10 |    14 |    20 |    81 |
| number_outpatient        |   69987 |   0.28 |  1.06 |     0 |     0 |     0 |     0 |    42 |
| number_emergency         |   69987 |   0.1  |  0.51 |     0 |     0 |     0 |     0 |    42 |
| number_inpatient         |   69987 |   0.18 |  0.6  |     0 |     0 |     0 |     0 |    12 |
| number_diagnoses         |   69987 |   7.22 |  2    |     1 |     6 |     8 |     9 |    16 |
| admission_type_id        |   69987 |   1.35 |  1.14 |    -1 |     1 |     1 |     2 |     7 |
| discharge_disposition_id |   69987 |   2.45 |  3.76 |    -1 |     1 |     1 |     3 |    28 |
| admission_source_id      |   69987 |   4.34 |  3.08 |    -1 |     1 |     7 |     7 |    25 |
| race_white               |   69987 |   0.75 |  0.43 |     0 |     0 |     1 |     1 |     1 |
| age_ordinal              |   69987 |   6.04 |  1.6  |     0 |     5 |     6 |     7 |     9 |

## Figures
- `figures/target_balance.png`
- `figures/numeric_distributions.png`
- `figures/categorical_distributions.png`
- `figures/correlation_heatmap.png`
- `figures/readmission_rate_by_feature.png`

See `eda/eda_report.html` for the full interactive Evidently data-quality report (per-column distributions, quantiles, missing values, correlations).
