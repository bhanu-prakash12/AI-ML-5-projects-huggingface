# Fairness slices (test split)

Threshold 0.227. Sensitive attributes are NOT model inputs (D3); groups with n < 30 are suppressed.

| slice | group | n | dropout_rate | flag_rate | recall | precision |
|---|---|---|---|---|---|---|
| gender_name | female | 420 | 0.233 | 0.35 | 0.857 | 0.571 |
| gender_name | male | 244 | 0.475 | 0.594 | 0.905 | 0.724 |
| age_band | 20-24 | 212 | 0.274 | 0.392 | 0.828 | 0.578 |
| age_band | 25-29 | 78 | 0.615 | 0.628 | 0.875 | 0.857 |
| age_band | 30+ | 102 | 0.549 | 0.588 | 0.946 | 0.883 |
| age_band | <20 | 272 | 0.191 | 0.368 | 0.885 | 0.46 |

Large recall/flag-rate gaps between groups indicate disparate impact; dropout_rate differences are base-rate differences, not model error.
