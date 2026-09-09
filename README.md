# 30-Day Hospital Readmission Risk Modeling

Predicting 30-day hospital readmission using structured clinical data from the
UCI Diabetes 130-US Hospitals dataset.

This project compares a regularized logistic regression, two PyTorch neural
networks, and XGBoost on approximately 70,000 unique patients. The main goal was
not simply to maximize a metric. I wanted to build the prediction problem
carefully, avoid leakage, evaluate probability quality, and understand what
information the models were actually using.

## Project Summary

The prediction task is:

> Using information available by hospital discharge, estimate the probability
> that a patient with diabetes will be readmitted within 30 days.

The final cohort contains:

- 69,990 unique patients
- 6,285 readmissions within 30 days
- 8.98% positive-class prevalence
- 42 predictors
- 70% training, 15% validation, and 15% test split

The main analysis compares logistic regression with a PyTorch multilayer
perceptron. XGBoost and an embedding-based PyTorch model were later added as
exploratory extensions.

## Dataset

The project uses the UCI
[Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
dataset.

The original release contains 101,766 hospital encounters from 71,518 patients.

Because some patients appear multiple times, I restricted the primary analysis
to one qualifying released encounter per patient. Encounters ending in death or
hospice were excluded before selecting the patient record because ordinary
post-discharge readmission is not meaningful for those encounters.

The resulting modeling cohort contains 69,990 unique patients.

## Prediction Setup

**Outcome**

`readmit_30d = 1` when the original readmission label is `<30`.

Readmissions after 30 days and no readmission are treated as the negative class.

**Prediction time**

Hospital discharge.

This matters because several features, including discharge disposition and
length of stay, are only known late in the hospitalization. This project should
therefore be interpreted as a discharge-time risk model, not an admission-time
risk model.

## Feature Engineering

The final model uses 42 predictors:

- 8 numeric features
- 34 categorical features

Examples include:

- time in hospital
- number of laboratory procedures
- number of medications
- prior inpatient, outpatient, and emergency visits
- age group
- admission type and source
- discharge disposition
- diagnosis groups
- medical specialty
- glucose and HbA1c testing status
- diabetes medication information

Raw ICD-9 diagnosis codes were grouped into broader clinical categories to
reduce sparsity.

Medical specialties were also grouped into a smaller set of categories.

Missing values were handled according to the meaning of each variable. For
example, missing HbA1c and glucose test results were represented as
`Not measured`, rather than treated as ordinary numeric missingness.

Learned preprocessing was fit using the training data only.

## Models

### Logistic Regression

A regularized logistic regression serves as the interpretable baseline.

Numeric variables were standardized and categorical variables were one-hot
encoded.

### PyTorch MLP

The main nonlinear model is a small multilayer perceptron implemented in
PyTorch:

```text
183 processed features
        |
       64
        |
       32
        |
   output logit
````

The network uses ReLU activations, dropout, Adam optimization, and
`BCEWithLogitsLoss`.

Early stopping was based on validation loss.

### Exploratory XGBoost

After the original logistic regression and MLP analysis was completed, I added
XGBoost as an exploratory benchmark.

It used the same cohort, predictors, data split, and training-fitted
preprocessing pipeline.

### Exploratory Embedding MLP

I also tested a second PyTorch model that replaced one-hot categorical inputs
with learned embeddings.

Each categorical feature received its own embedding table. The learned vectors
were concatenated with the standardized numeric features before being passed
through the MLP.

This provided a more natural categorical representation for a neural network
while keeping the same underlying information.

## Held-Out Test Results

| Model                 |    ROC-AUC |     PR-AUC | Brier Score |
| --------------------- | ---------: | ---------: | ----------: |
| Logistic Regression   |     0.6465 |     0.1672 |      0.0794 |
| One-hot PyTorch MLP   |     0.6480 |     0.1667 |      0.0794 |
| Embedding PyTorch MLP |     0.6528 |     0.1734 |      0.0791 |
| XGBoost               | **0.6564** | **0.1802** |  **0.0789** |

XGBoost achieved the strongest overall result, but the difference between the
four models was fairly small.

The embedding model improved over the original one-hot MLP, especially in
average precision, but it also did not produce a large change in overall
performance.

This suggests that model complexity is probably not the main limitation of the
current prediction problem.

## Why PR-AUC Matters Here

Only about 9% of patients in the modeling cohort were readmitted within 30 days.

Because the positive class is relatively uncommon, accuracy can be misleading.
A model could obtain high accuracy simply by predicting that almost nobody will
be readmitted.

For that reason, I evaluated:

* ROC-AUC
* PR-AUC / average precision
* Brier score
* calibration
* precision
* recall
* F1 score
* confusion matrices at selected thresholds

## Threshold Selection

A probability cutoff of 0.50 performed poorly because of the low event rate.

Instead, thresholds were selected using the validation set.

Two operating points were examined:

1. the threshold that maximized validation F1
2. an illustrative threshold targeting approximately 70% recall

The second threshold is included to show the precision-recall tradeoff. It is
not presented as a clinically validated decision rule.

At approximately 70% recall, all of the models required flagging a large number
of patients and produced relatively low precision.

This is an important limitation for practical use.

## Model Interpretation

Permutation importance was calculated at the level of the original 42
predictors.

The strongest predictors were:

1. discharge disposition
2. prior inpatient utilization
3. age
4. diagnosis-related variables
5. admission and encounter characteristics

Discharge disposition was by far the strongest individual feature.

For logistic regression, shuffling discharge disposition reduced validation
ROC-AUC by about 0.072.

Prior inpatient utilization was the strongest numeric predictor.

The logistic regression and original MLP also produced very similar feature
importance rankings. Their permutation-importance rankings had a Spearman
correlation of approximately 0.94.

This was useful because two different model classes were relying on largely the
same information.

## Main Finding

The main result was not that one model clearly beat all of the others.

Instead, logistic regression, neural networks, and gradient-boosted trees all
ended up in a fairly narrow performance range.

The more flexible models found some additional predictive signal, especially
XGBoost and the embedding MLP, but the improvement was modest.

That led to a more interesting next question:

> Instead of continuing to tune the models, what happens if I change which
> information is available to them?

## Planned Follow-Up Analyses

### 1. Prediction-Time and Feature Ablation

Discharge disposition was the strongest predictor in the project.

A planned extension will remove discharge disposition and potentially other
features that become available only late in the hospitalization.

The goal is to measure how much performance depends on discharge-time
information and move toward an earlier prediction setting.

### 2. Subgroup Performance

A second planned extension will evaluate performance across patient subgroups.

This will include comparisons of:

* outcome prevalence
* ROC-AUC
* PR-AUC
* calibration
* precision and recall at fixed operating points

Potential subgroup variables include age, sex, and race.

The goal is to determine whether similar overall model performance hides
important differences between patient groups.

## Limitations

This project has several important limitations.

**Historical data**

The dataset covers hospital encounters from 1999 through 2008. Clinical
practice, coding systems, and readmission management have changed since then.

**No reliable temporal split**

The public dataset does not provide a documented encounter timestamp suitable
for a trustworthy temporal validation split.

I therefore used a stratified random train, validation, and test split after
restricting the cohort to one encounter per patient.

**No hospital-level validation**

Hospital identifiers are not available in the public release, so I could not
evaluate generalization across hospital sites.

**Prediction time**

The model is explicitly designed for discharge-time prediction. It should not
be interpreted as an admission-time risk model.

**Moderate predictive performance**

All evaluated models showed useful but limited discrimination. These results
suggest that stronger performance may require richer or more temporally useful
clinical information, rather than simply a more complicated model.

## Reproducibility

The notebook includes the full workflow:

1. loading the official UCI dataset
2. cohort construction
3. leakage audit
4. diagnosis and specialty grouping
5. train, validation, and test splitting
6. training-only preprocessing
7. logistic regression baseline
8. PyTorch MLP training with early stopping
9. calibration and threshold analysis
10. held-out test evaluation
11. permutation importance
12. XGBoost exploratory benchmark
13. categorical embedding MLP

Random seeds are fixed where applicable.

The test set was not used for preprocessing, threshold selection, or training
of the original models.

The XGBoost and embedding experiments were added after the original held-out
results had already been examined, so they are reported as exploratory rather
than independent confirmatory experiments.

## Tech Stack

* Python
* pandas
* NumPy
* scikit-learn
* PyTorch
* XGBoost
* Matplotlib
* UCI `ucimlrepo`

## Repository

```text
30-day-readmission-risk-modeling/
|
|-- README.md
|-- hospital_readmission_modeling.ipynb
```

The full analysis, outputs, interpretation, and methodological notes are
contained in the notebook.

## References

UCI Machine Learning Repository.
*Diabetes 130-US Hospitals for Years 1999-2008.*

Strack B, DeShazo JP, Gennings C, et al.
*Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000
Clinical Database Patient Records.* BioMed Research International, 2014.

```
