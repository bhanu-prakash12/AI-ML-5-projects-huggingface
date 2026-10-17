# EDA: UCI 697 student dropout

Rows: 4424, features: 36

## Split sizes
| split | rows | dropout rate |
|---|---|---|
| train | 3096 | 0.321 |
| val | 664 | 0.321 |
| test | 664 | 0.322 |

## Class balance (3-class target)
| class | rows | share |
|---|---|---|
| Graduate | 2209 | 0.499 |
| Dropout | 1421 | 0.321 |
| Enrolled | 794 | 0.179 |

Binary target (D2) base rate Dropout vs rest: 0.321

## Missingness
Total missing cells in features: 0

## Top 10 correlations with dropout (Pearson)
| feature | r |
|---|---|
| curricular_units_2nd_sem_grade | -0.572 |
| curricular_units_2nd_sem_approved | -0.570 |
| curricular_units_1st_sem_grade | -0.481 |
| curricular_units_1st_sem_approved | -0.479 |
| tuition_fees_up_to_date | -0.429 |
| age_at_enrollment | +0.254 |
| scholarship_holder | -0.245 |
| debtor | +0.229 |
| gender | +0.204 |
| application_mode | +0.198 |

> Note: semester features are end-of-year-2 data (D1), so this is a late-stage model, not an early warning.
