# ReliableML-AutoML

ReliableML-AutoML is the optional companion for users who want ReliableML's
ATTP-based partitioning, automatic candidate selection, preprocessing, and
hyperparameter tuning in one workflow. It depends on `ReliableML` but is
installed separately, so Core users do not need ATTP or the AutoML
dependencies.

## Install in its own environment

From the repository root in PowerShell:

```powershell
py -3.15 -m venv .venv-automl
.\.venv-automl\Scripts\Activate.ps1
python -m pip install -e .
python -m pip install -e .\automl
```

For a released installation, install `ReliableML-AutoML` in the dedicated
environment; its package metadata installs the matching core and ATTP
requirements. The core-only `.venv` does not need `i-attp`.

## Quick start

```python
from reliableml_automl import ReliableAutoML

automl = ReliableAutoML(
    target_column="outcome",
    random_state=42,
    progress_bar=True,
)
automl.fit(development_data)

print(automl.selected_model_name)
print(automl.leaderboard)
print(automl.fit_report)

predictions, reliability = automl.predict_with_report(unseen_features)
```

`progress_bar` defaults to `False`; set it to `True` to display fitting
progress. The bar covers data preparation/partitioning, outer candidate-CV
folds, and final model fitting. Each outer fold includes its inner parameter
search, so the bar may pause while that search runs and its remaining-time
estimate is approximate.

The default workflow uses ReliableML's `attp.s_attp_clf` partitioner and light
nested hyperparameter tuning. Three-fold stratified cross-validation compares
ReliableML's out-of-fold `MetaPipelineClassifier` and `PerfMetaSelect` meta
learners together with the StatCore own-variant models `sc_logistic`,
`sc_lda`, `sc_knn`, `sc_mlp` and `sc_meta` (pure NumPy, tuned over their
variants; pass `include_statcore=False` to leave them out). AutoML selects the
highest-scoring successful candidate across all of them under the
configured CV procedure. To compare standard classifier baselines or use
other estimators, supply them through `candidates` (an explicit list replaces
the defaults, including the StatCore ones, so add e.g.
`StatCoreKNNClassifier()` yourself if you want it there).

Preprocessing is fit inside each cross-validation fold: missing
numeric/categorical values are imputed, mostly numeric text columns are
converted to numbers, numeric features are scaled, ordinal features can be
encoded using supplied order, and nominal categorical features are one-hot
encoded. Unseen categories at prediction time are ignored by the encoder.
Scaling is selected per numeric feature in `auto` mode: robust scaling for
detected outliers, standard scaling for approximately-normal features, and a
0–10 min-max transform otherwise.

### Choose a ReliableML meta-learner or your own candidate

For example, explicitly use ReliableML's `MetaPipelineClassifier` as the
candidate and tune its final meta-classifier. The standard `LogisticRegression`
objects below are the meta-classifier options; the OOF meta-learning workflow
is provided by ReliableML:

```python
from sklearn.linear_model import LogisticRegression
from ml_reliability import MetaPipelineClassifier
from reliableml_automl import ReliableAutoML

automl = ReliableAutoML(
    target_column="outcome",
    candidates=[
        ("reliable_meta_pipeline", MetaPipelineClassifier(n_folds=3)),
    ],
    light_param_grids={
        "reliable_meta_pipeline": {
            "meta_classifier": [
                LogisticRegression(C=0.5, max_iter=1000),
                LogisticRegression(C=1.0, max_iter=1000),
            ],
        },
    },
)
automl.fit(development_data)
print(automl.selected_model_name)
print(automl.best_params_)
```

Other compatible ReliableML classifiers, such as
`CalibratedStackedClassifier` and the statistical meta-classifiers, can be
passed in the same `candidates` list. Standard scikit-learn classifiers are
also supported as optional candidates. These are independently configurable;
provide a `light_param_grids` entry when using an estimator for which AutoML
has no built-in parameter grid. The statistical LDA, tree, and neural
meta-classifiers currently support binary classification.

## Keep an untouched final test set

For a defensible final performance estimate, reserve representative labeled
rows before calling `fit()`. Use ATTP inside the development data for its
intended partitioning/model-selection role, then evaluate once on the untouched
final set. A conventional stratified holdout is one option for reserving that
outer test set; it does not replace ATTP within the development workflow.

```python
from sklearn.model_selection import train_test_split
from reliableml_automl import ReliableAutoML

development_data, final_test_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    stratify=data["outcome"],
)

automl = ReliableAutoML(
    target_column="outcome",
    progress_bar=True,
    random_state=42,
).fit(development_data)

X_final = final_test_data.drop(columns=["outcome"])
y_final = final_test_data["outcome"]
evaluation = automl.evaluate(X_final, y_final, positive_label=1)
print(evaluation.classification["probability_calibration"])
```

Do not tune models, choose ATTP settings, or select a winning method using
the final test results. For a research comparison of multiple partitioners,
repeat the complete protocol on the same datasets and outer splits; compare
methods using paired per-dataset results. Treat datasets—not folds or seeds
within one dataset—as the independent units for across-dataset significance
tests. Report the partition protocol, tuning budget, seed/repeat plan, class
balance, metrics, confidence intervals/effect sizes, and runtime. ATTP
partition diagnostics (such as distribution distances) complement but do not
replace downstream final-test performance.

## Light and extended hyperparameter tuning

`tuning="light"` is the default. It uses a small grid for each built-in
candidate. Hyperparameters are tuned inside each outer training fold, and the
result is scored on that fold's held-out portion; the outer scores determine
which candidate is selected. The selected candidate is then tuned once more
on all partitioner training rows to create the fitted pipeline.

```python
from reliableml_automl import ReliableAutoML

automl = ReliableAutoML(
    target_column="outcome",
    tuning="light",
    cv=3,
    tuning_cv=2,
)
automl.fit(development_data)
print(automl.fit_report["selection"])
```

To request a larger randomized parameter search, opt in explicitly:

```python
automl = ReliableAutoML(
    target_column="outcome",
    tuning="extended",
    tuning_iterations=30,
    tuning_cv=3,
    cv=5,
)
automl.fit(development_data)
```

Extended mode uses broader distributions for the built-in estimators and takes
longer. Custom candidates must provide their own parameter spaces, keyed by
candidate name:

```python
automl = ReliableAutoML(
    target_column="outcome",
    candidates=[("linear_svc", my_estimator)],
    tuning="light",
    light_param_grids={"linear_svc": {"C": [0.1, 1.0, 10.0]}},
)
```

Use `extended_param_distributions` with `tuning="extended"`. Parameter names
can be native estimator names (such as `"C"`) or full sklearn pipeline names
(such as `"classifier__C"`). `best_params_`, `leaderboard`, and `fit_report`
show the selected tuning results. Scores are estimates for the specified
partition and CV design; choosing the best among candidates can still lead to
selection optimism, so reserve untouched representative data for the final
evaluation.

## Cleaning and keeping the selected pipeline

The companion's preprocessing incorporates the safe, reusable parts of the
attached seven-stage cleaning workflow: numeric-text detection (at least half
of non-missing values must parse as numeric), mean/mode/median imputation,
configurable IQR capping, conditional per-feature scaling, ordinal encoding
when category order is supplied, and nominal one-hot encoding. The
`cleaning_report` records numeric-text coercion and the cleaner's fit-time
missing/outlier report. The preprocessing is fit independently inside each
search fold and refit on the selected training rows.

The attached feature-selection router and correlation/VIF stage are not run
automatically. In the attached implementation, feature subsets are chosen
using a holdout from the same data being transformed, and PCA/LDA transforms
reuse stored training values instead of transforming new rows. That behavior
is not safe for a fitted production pipeline. Automatic feature reduction
should only be added after it is implemented as a fold-fitted transformer and
validated on held-out data.

The selected fitted scikit-learn estimator pipeline is available as
`automl.best_pipeline`. Save the full ReliableML pipeline artifact for future
prediction and reliability reports:

```python
automl.save("selected-model.pkl")

from ml_reliability import ReliablePipeline

deployed = ReliablePipeline.load("selected-model.pkl")
predictions, report = deployed.predict_with_report(unseen_features)
```

The selected result is the best candidate under the configured search and
cross-validation procedure; “best” does not mean globally optimal.

## S-ATTP partition behavior

The default partitioner is `attp.s_attp_clf`. If
`partitioner_options` is omitted, ReliableML-AutoML calls it without keyword
overrides, preserving the defaults supplied by the installed ATTP release.
ReliableML-AutoML pins `i-attp==0.2.3` because it is a pre-1.0 API and these
defaults and return fields are part of the tested integration. Its defaults
include candidate train fractions `(0.7, 0.8)`,
cluster bounds `2` to `10`, LDA component bounds `1` to `3`, and
`random_state=0`. You can override them explicitly:

```python
automl = ReliableAutoML(
    target_column="outcome",
    partitioner="s_attp_clf",
    partitioner_options={"test_size": 0.25, "random_state": 10},
)
```

S-ATTP requires a finite numeric feature matrix. AutoML creates a separate
cleaned/encoded representation for partition design; final model
preprocessing is still fitted on training rows only. S-ATTP fits LDA and
clusters before splitting and chooses a partition using candidate test
scores. Therefore, its internal test rows and score are not an unbiased final
generalization estimate. Keep a separate, representative final test set for
performance claims.

`fit_report` records the selected partitioner, the effective option values
passed by AutoML, row counts, model-selection method, candidate scores, and
interpretation warnings. It intentionally does not present the ATTP split
score as a model leaderboard score.

## Other partitioners

ATTP's S-ATTP classifier splitter is the default. You may select another
ATTP-provided partitioner explicitly:

```python
automl = ReliableAutoML(
    target_column="outcome",
    partitioner="s_attp_cluster",
    random_state=42,
)
```

The package exposes currently available ATTP methods by name:

- `s_attp_clf` (default)
- `sattp_split` (compatibility alias for `s_attp_clf`)
- `s_attp_cluster`
- `s_attp_clustered_sampling` (alias for `s_attp_cluster`)
- `s_attp_reg`
- `s_attp_ts`
- `train_test_split` (stratified scikit-learn baseline)

ATTP methods can have different return types; supported pandas train/test
outputs are normalized to the original row indexes. You can pass another
callable and its options:

```python
automl = ReliableAutoML(
    target_column="outcome",
    partitioner=my_split_function,
    partitioner_options={"some_option": "value"},
)
```

A callable may return train/test integer row positions, a mapping with
`train_indices` and `test_indices`, an ATTP-style object with those attributes,
or pandas train/test objects preserving input row indexes. For a different
return format, pass `partitioner_result_adapter` to convert it to
`(train_indices, test_indices)`. All rows must appear exactly once across the
two partitions.

## Exporting results (opt-in)

`data_intelligence=` controls the report-only intelligence run on the training
partition. It takes `None`, one name, or a sequence of names:

| Name | Adds | Needs target |
|---|---|---|
| `"multicollinearity"` (default) | VIF and redundant-pair report | no |
| `"profile"` | hazard/EDA profile: constant columns, outlier consensus, recommendations | no |
| `"power"` | predictive vs explanatory vs statistical power, with a verdict | training labels |
| `"construction"` | iterative, distribution-aware feature construction with holdout check | training labels |
| `"active_learning"` | learning-curve study (random, margin, k-center, coverage-margin) | training labels |
| `"group_scan"` | which categorical columns separate the target | training labels |
| `"root_cause"` | segments concentrating the target's bad outcome | training labels |

```python
auto = ReliableAutoML("label", data_intelligence=("multicollinearity", "power", "construction"))
auto.fit(df)
auto.data_intelligence_report["sections"]["power"]["verdict"]
```

Label-using sections see **only the training partition**, never the test rows, and never
change the pipeline, partition or leaderboard (a test checks the leaderboard is identical).
Usage, measured results and applications: `docs/data-intelligence-toolkit.md` in the core repo.

Nothing is written to disk by default.

```python
automl = ReliableAutoML("outcome", export_dir="automl_out", export_model=True)
automl.fit(df)            # df holds the features and the target column
                          # writes automl_report.{json,txt,html}, leaderboard.csv,
                          # data_intelligence.json, reliable_pipeline.joblib
automl.export_report("elsewhere", formats=("json", "html"))   # or call later
```

## Runnable use cases

[`examples/04_automl_use_cases.py`](../examples/04_automl_use_cases.py) in the repository root runs ten use cases
(defaults, own models only, own vs scikit-learn, messy data, cleaning and scaling, Data Intelligence sections,
partitioners, tuning depth, saving, reliability on the result). The option table and the use-case list are in the
[main README](../README.md#3-automl-reliableml-automl). Prefer your own models: pass StatCore or ReliableML
meta-learners through `candidates`; scikit-learn candidates are optional.

## Tests

Run the companion tests in the AutoML environment:

```powershell
.\.venv-automl\Scripts\python.exe -m unittest discover -s automl\tests -v
```
