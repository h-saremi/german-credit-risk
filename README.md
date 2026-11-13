# Credit Risk Scoring on the German Credit Dataset

Cost-sensitive logistic regression with a fairness check across sensitive attributes.

**Author:** Hero Saremi (PhD mathematician)

## Question

Can a simple, interpretable model predict loan default on the German Credit data, how should the decision threshold be chosen when a missed default costs far more than a rejected good applicant, and does the model treat age groups differently?

## Data

- **Source:** Statlog (German Credit Data), UCI Machine Learning Repository (Hofmann, 1994). https://doi.org/10.24432/C5NC77
- **Size:** 1,000 loan applications, 20 features plus the target `default` (1 = default, 0 = repaid). About 30% of applicants are defaults.
- **Version used:** the processed CSV `german_credit/german_credit.csv` (1,000 rows x 21 columns, no missing values) from the Zenodo record "Processed German Credit Data" (https://zenodo.org/records/10785677), which is derived from the original UCI dataset.
- **License:** both the UCI dataset and the processed Zenodo copy are licensed under Creative Commons Attribution 4.0 (CC BY 4.0), which allows sharing and adaptation with appropriate credit. The processed copy was prepared by Pablo Sanchez Martin (2024).

## Methods

1. **Exploration:** class balance, summary statistics, default rates by checking-account status and loan duration. `credit_amount` is right-skewed (mean 3,271, median 2,319).
2. **Model:** logistic regression in a scikit-learn pipeline. Numeric features are standardized and categorical features one-hot encoded.
3. **Evaluation:** stratified 75/25 train/test split, plus 5-fold stratified cross-validation. Metric: ROC AUC. Accuracy is not used because a model that approves everyone already reaches 70%.
4. **Cost-sensitive threshold:** a missed default is assumed to cost 5 times as much as a rejected good applicant (in line with the cost matrix that comes with the original dataset). The threshold was chosen on cross-validated predictions of the **training** data only, then evaluated once on the test set.
5. **Fairness check:** the model was refitted without `age`, `personal_status_sex` and `foreign_worker`. Error rates were compared by age group (under 25 vs 25 and over).

## Results

| Metric | Value |
|---|---|
| ROC AUC, 5-fold CV | 0.786 (std 0.02) |
| ROC AUC, single test split | 0.804 |
| Threshold chosen on training folds | 0.09 |

Total cost on the test set (250 applicants):

| Strategy | Cost |
|---|---|
| Approve everyone | 375 |
| Threshold 0.5 | 193 |
| Reject everyone | 175 |
| **Model, threshold 0.09** | **150** |

The cost-sensitive threshold lowers the cost by about 22% compared with the default 0.5 threshold and beats the "reject all" baseline by a small margin.

### Largest coefficients (log-odds, standardized features)

- Lower default risk: no checking account, credit history "critical account", savings of 1,000 DM or more.
- Higher default risk: negative checking balance, savings under 100 DM, no known property, renting, unemployed.
- These are associations, not causal effects. Some signs look counterintuitive (for example "critical account" lowering risk), which probably reflects how the sample was collected.

## Fairness check

- Removing the three sensitive attributes changed ROC AUC only slightly (0.804 to 0.796).
- The rejection-rate gap between applicants under 25 and over 25 narrowed (0.65 vs 0.45 became 0.56 vs 0.45) but did not disappear, which suggests that other features act as proxies for age. In this sample, the real default rate also differs (0.52 vs 0.25).
- Error rates by age group (test set):

| Group | Good applicants wrongly rejected | Bad applicants wrongly approved |
|---|---|---|
| Under 25, full model | 0.39 (9 of 23) | 0.12 (3 of 25) |
| Under 25, without sensitive features | 0.30 (7 of 23) | 0.20 (5 of 25) |
| 25 and over, full model | 0.33 (n = 152) | 0.20 (n = 50) |
| 25 and over, without sensitive features | 0.34 | 0.22 |

- The differences between groups correspond to 2 to 3 people and are within the range of sampling noise, so no clear unfairness is found, but the check is weak.
- The foreign-worker group has only 8 test cases, so no conclusion is drawn for it.

## Limitations

- Small dataset (1,000 applications, 250 in the test set), a single random split, one random seed.
- The 5:1 cost ratio is an assumption. It ignores the profit lost on rejected good loans, which is why the chosen threshold (0.09) is lower than the Bayes threshold of about 0.17.
- Probabilities were not checked for calibration.
- The data come from a German bank in the 1990s and are not representative of today.
- Some category labels in the data are odd, for example `purpose` = "(vacation - does not exist?)". They were kept as they are and not interpreted.
- Using attributes such as age, sex or nationality in real credit decisions raises legal and ethical issues. This project only analyzes their effect.

## How to run

1. Open `notebook.ipynb` in Google Colab (or Jupyter).
2. Upload `german_credit.csv` (or download it from the UCI page above).
3. Run the cells from top to bottom.

Requirements: Python 3, pandas, numpy, scikit-learn, matplotlib.

## Reference

Hofmann, H. (1994). Statlog (German Credit Data) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77

Sanchez Martin, P. (2024). Processed German Credit Data (Version 1.0.0) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.10785677
