"""Use cases 21-30: ReliableAutoML, option by option (own models first, scikit-learn candidates optional).

Install the companion first (see automl/README.md); this script needs `reliableml_automl` and `i-attp`.

  21. Defaults                      ATTP split + 2 meta-learners + 5 StatCore models, nested tuning
  22. Own models only               your list of candidates + your own tuning grids
  23. Own vs scikit-learn           add sklearn baselines to the same leaderboard (optional)
  24. Messy mixed-type data         text numbers, missing values, categories, ordinal order
  25. Cleaning and scaling options  cleaning_intensity, scaling_strategy
  26. Data Intelligence sections    report-only: multicollinearity, profile, power, construction,
                                    active_learning, group_scan, root_cause
  27. Choosing the partitioner      ATTP variants, plain train_test_split, or your own function
  28. Tuning depth                  light vs extended
  29. Saving (opt-in)               export_dir / export_report / save and reload
  30. Reliability on the result     predict_with_report (full set of checks: example 05)

Nothing is written to your folders: plots and exports go to a temporary directory that is deleted at the end.
Run:  python examples/04_automl_use_cases.py           (several minutes; reduce CV/candidates to go faster)
"""
from __future__ import annotations

import shutil
import tempfile
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from ml_reliability import (
    CalibratedStackedClassifier,
    ReliablePipeline,
    StatCoreKNNClassifier,
    StatCoreLDAClassifier,
    StatCoreLogisticClassifier,
    StatCoreMetaClassifier,
)
from reliableml_automl import ReliableAutoML

warnings.filterwarnings("ignore")
TMP = Path(tempfile.mkdtemp(prefix="reliableml_automl_"))
rng = np.random.default_rng(0)
NO_PLOTS = {"reliability_required": False}          # skip drift plots when you only want the leaderboard


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def board(auto, top=8):
    print(f"selected: {auto.selected_model_name}")
    for row in auto.leaderboard[:top]:
        print(f"  {row['model']:<22} score {row.get('mean_score') or float('nan'):.3f} +/- {row.get('std_score') or 0:.3f}  [{row['status']}]")


def make(**kw):
    kw.setdefault("progress_bar", False)
    kw.setdefault("plot_dir", TMP / "plots")
    kw.setdefault("reliability_options", NO_PLOTS)
    kw.setdefault("cv", 3)
    kw.setdefault("tuning_cv", 2)
    return ReliableAutoML("y", **kw)


# One dataset used throughout: breast cancer, with a few columns made messy later where needed.
base = load_breast_cancer(as_frame=True).frame.rename(columns={"target": "y"})
base = base[[c for c in base.columns if c.startswith("mean") or c == "y"]].sample(frac=1, random_state=0).reset_index(drop=True)
dev, holdout = base.iloc[:350].copy(), base.iloc[350:].copy()

# ---------------------------------------------------------------- 21. defaults
banner("21. DEFAULTS")
auto = make().fit(dev)
board(auto)
print("fit_report keys:", list(auto.fit_report)[:8])

# ---------------------------------------------------------------- 22. own models only
banner("22. OWN MODELS ONLY, with your own tuning grids")
own = [
    ("sc_logistic", StatCoreLogisticClassifier()),
    ("sc_knn", StatCoreKNNClassifier()),
    ("sc_meta", StatCoreMetaClassifier(n_folds=3)),
    ("calibrated_stack_own", CalibratedStackedClassifier(
        base_estimators=[("lr", StatCoreLogisticClassifier()), ("lda", StatCoreLDAClassifier())], n_folds=3)),
]
auto_own = make(candidates=own, light_param_grids={
    "sc_logistic": {"variant": ["base", "stat_regularised", "risk_calibrated"]},
    "sc_knn": {"variant": ["base", "stat_k_selection"], "n_neighbors": [5, 9]},
    "sc_meta": {"meta_variant": ["oof_stacking", "calibrated_stack"]},
    "calibrated_stack_own": {"calibration_method": ["sigmoid", "isotonic"]},
}).fit(dev)
board(auto_own)
print("best parameters:", auto_own.best_params_)
print("shortcut: ReliableAutoML('y', include_statcore=False) removes the StatCore defaults instead.")

# ---------------------------------------------------------------- 23. own vs scikit-learn
banner("23. OWN vs SCIKIT-LEARN on one leaderboard (sklearn is optional, never required)")
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

mixed = own[:3] + [("sk_logreg", LogisticRegression(max_iter=1000)),
                   ("sk_forest", RandomForestClassifier(n_estimators=100, random_state=0))]
auto_mix = make(candidates=mixed, light_param_grids={
    "sc_logistic": {"variant": ["base", "stat_regularised"]}, "sc_knn": {"n_neighbors": [5, 9]},
    "sc_meta": {"meta_variant": ["oof_stacking"]}, "sk_logreg": {"C": [0.1, 1.0, 10.0]},
    "sk_forest": {"max_depth": [None, 6]}}).fit(dev)
board(auto_mix)

# ---------------------------------------------------------------- 24. messy mixed-type data
banner("24. MESSY MIXED-TYPE DATA: numbers stored as text, missing values, categories")
messy = dev.copy()
messy["mean area"] = messy["mean area"].round(1).astype(str)                 # numbers stored as text
messy.loc[messy.sample(frac=0.04, random_state=1).index, "mean texture"] = np.nan
messy["grade"] = pd.cut(messy["mean radius"], 3, labels=["low", "mid", "high"]).astype(str)   # ordinal
messy["site"] = rng.choice(["A", "B", "C"], len(messy))                         # nominal
auto_messy = make(ordinal_features={"grade": ["low", "mid", "high"]}).fit(messy)
board(auto_messy, top=3)
print("cleaning report:", {k: v for k, v in auto_messy.cleaning_report.items() if k != "details"})

# ---------------------------------------------------------------- 25. cleaning and scaling
banner("25. CLEANING AND SCALING OPTIONS")
for intensity, scaling in [("light", "auto"), ("heavy", "robust"), ("moderate", "standard")]:
    a = make(candidates=own[:2], cleaning_intensity=intensity, scaling_strategy=scaling,
             light_param_grids={"sc_logistic": {"variant": ["base"]}, "sc_knn": {"n_neighbors": [5]}}).fit(messy)
    print(f"  cleaning={intensity:<9} scaling={scaling:<9} -> {a.selected_model_name} {a.leaderboard[0]['mean_score']:.3f}")

# ---------------------------------------------------------------- 26. Data Intelligence sections
banner("26. DATA INTELLIGENCE INSIDE AUTOML (report-only; never changes the model)")
auto_di = make(candidates=own[:2], data_intelligence=("multicollinearity", "profile", "power", "group_scan", "root_cause"),
               light_param_grids={"sc_logistic": {"variant": ["base"]}, "sc_knn": {"n_neighbors": [5]}}).fit(messy)
rep = auto_di.data_intelligence_report
print("VIF suggests dropping:", rep["recommended_drop"])
print("profile hazards:", rep["profile"]["hazards"])
for name in ("power", "group_scan", "root_cause"):
    print(f"  [{name}] keys:", list(rep["sections"][name])[:5])
print("power verdict:", rep["sections"]["power"].get("verdict"))
print("also available: 'construction', 'active_learning' (slower); data_intelligence=None switches it all off.")

# ---------------------------------------------------------------- 27. partitioner
banner("27. CHOOSING THE PARTITIONER")


def my_split(X, y, test_size=0.25, seed=0):
    """Your own splitter: return train/test row positions."""
    tr, te = train_test_split(np.arange(len(y)), test_size=test_size, random_state=seed, stratify=y)
    return {"train_indices": tr, "test_indices": te}


for label, kw in [("train_test_split", dict(partitioner="train_test_split", partitioner_options={"test_size": 0.25})),
                  ("custom function ", dict(partitioner=my_split, partitioner_options={"test_size": 0.3})),
                  ("s_attp_cluster  ", dict(partitioner="s_attp_cluster"))]:
    a = make(candidates=own[:2], light_param_grids={"sc_logistic": {"variant": ["base"]}, "sc_knn": {"n_neighbors": [5]}}, **kw).fit(dev)
    print(f"  {label}: selected {a.selected_model_name}; partition {a.fit_report['partition_sizes']}")

# ---------------------------------------------------------------- 28. tuning depth
banner("28. TUNING DEPTH: light (default) vs extended randomized search")
fast = {"candidates": [own[1]], "cv": 3}
for mode in ("light", "extended"):
    a = make(tuning=mode, tuning_iterations=4,
             light_param_grids={"sc_knn": {"n_neighbors": [3, 5, 9]}},
             extended_param_distributions={"sc_knn": {"n_neighbors": [3, 5, 7, 9, 15], "weights": ["uniform", "distance"]}},
             **fast).fit(dev)
    print(f"  {mode:<9} best params {a.best_params_}")

# ---------------------------------------------------------------- 29. saving (opt-in)
banner("29. SAVING IS OPT-IN")
print("default run wrote nothing; now we ask explicitly (into a temp folder):")
saved = auto_own.export_report(TMP / "automl_out", formats=("json", "html"), include_model=True)
print({k: Path(v).name for k, v in saved.items()})
auto_own.save(TMP / "selected.pkl")
restored = ReliablePipeline.load(TMP / "selected.pkl")
same = (restored.predict(holdout.drop(columns="y")) == auto_own.predict(holdout.drop(columns="y"))).all()
print("reloaded model predicts identically:", bool(same))
print("in your own code: ReliableAutoML('y', export_dir='out', export_model=True)")

# ---------------------------------------------------------------- 30. reliability on the result
banner("30. RELIABILITY ON THE RESULT (all the checks are in example 05)")
a = ReliableAutoML("y", candidates=own[:2], progress_bar=False, plot_dir=TMP / "plots2", cv=3, tuning_cv=2,
                   light_param_grids={"sc_logistic": {"variant": ["base"]}, "sc_knn": {"n_neighbors": [5]}}).fit(dev)
shifted = holdout.drop(columns="y").copy()
shifted["mean texture"] *= 1.7
_, r_ok = a.predict_with_report(holdout.drop(columns="y"))
_, r_shift = a.predict_with_report(shifted)
print(f"similarity score  normal batch: {r_ok.similarity_score:.3f}   shifted batch: {r_shift.similarity_score:.3f}")
print("drifted features on shifted batch:", r_shift.drifted_features)
ev = a.evaluate(holdout.drop(columns="y"), holdout["y"], positive_label=1)
print("MCC on holdout:", round(ev.classification["matthews_correlation_coefficient"], 3))

shutil.rmtree(TMP, ignore_errors=True)
print("\ntemporary files removed.")
