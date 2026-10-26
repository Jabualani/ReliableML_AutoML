import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml_reliability import MetaPipelineClassifier, ReliablePipeline
from reliableml_automl import ReliableAutoML
from reliableml_automl.automl import _partition_indices


def _classification_data(rows=120):
    rng = np.random.default_rng(71)
    signal = rng.normal(size=rows)
    auxiliary = rng.normal(size=rows)
    labels = ((signal + 0.35 * auxiliary) > 0).astype(int)
    data = pd.DataFrame(
        {
            "signal": signal,
            "auxiliary": auxiliary,
            "region": np.where(signal > 0, "north", "south"),
            "size": np.resize(np.array(["S", "M", "L", "XL"], dtype=object), rows),
            "label": labels,
        }
    )
    data.loc[::13, "auxiliary"] = np.nan
    data.loc[::17, "region"] = None
    return data


class ReliableAutoMLTests(unittest.TestCase):
    def test_s_attp_clf_defaults_handle_dirty_mixed_data(self):
        data = _classification_data()
        model = ReliableAutoML(
            "label",
            candidates=[("logistic_regression", LogisticRegression(max_iter=500))],
            cv=2,
            ordinal_features={"size": ["S", "M", "L", "XL"]},
            reliability_options={
                "bootstrap_samples": 20,
                "bias_variance_folds": 2,
                "bias_variance_repeats": 1,
            },
        ).fit(data)

        report = model.fit_report
        partition = report["partitioner"]
        self.assertEqual(partition["name"], "s_attp_clf")
        self.assertTrue(partition["used_installed_defaults"])
        self.assertEqual(partition["options"], {})
        self.assertTrue(partition["splitter_test_score_is_not_model_evaluation"])
        self.assertFalse(report["selection"]["partition_test_used_for_model_selection"])
        self.assertEqual(model.selected_model_name, "logistic_regression")
        self.assertIn("size", model.cleaning_report["ordinal_features"])
        self.assertEqual(
            model.best_pipeline.named_steps["data_integrity"]
            .get_integrity_report()["numeric_text_columns"],
            [],
        )
        self.assertEqual(
            report["partition_sizes"]["train"] + report["partition_sizes"]["test"],
            len(data),
        )
        self.assertEqual(len(set(model.train_indices_) & set(model.test_indices_)), 0)
        self.assertEqual(
            len(set(model.train_indices_) | set(model.test_indices_)),
            len(data),
        )

        new_data = data.drop(columns="label").tail(8).copy()
        new_data["region"] = "unseen-region"
        predictions, reliability = model.predict_with_report(new_data)
        self.assertEqual(len(predictions), len(new_data))
        self.assertEqual(reliability.sample_count, len(new_data))

    def test_dirty_numeric_text_and_cleaner_metadata_are_in_final_pipeline(self):
        data = _classification_data(rows=90)
        data["measurement_text"] = data["signal"].map(lambda value: f"{value:.6f}")
        data.loc[::19, "measurement_text"] = "not-a-number"
        model = ReliableAutoML(
            "label",
            candidates=[("logistic_regression", LogisticRegression(max_iter=500))],
            cv=2,
            cleaning_intensity="moderate",
            reliability_options={"reliability_required": False},
        ).fit(data)

        cleaning = model.cleaning_report
        self.assertIn(
            "measurement_text",
            cleaning["data_integrity"]["numeric_text_columns"],
        )
        self.assertGreater(
            cleaning["data_integrity"][
                "invalid_numeric_values_treated_as_missing_at_fit"
            ]["measurement_text"],
            0,
        )
        self.assertEqual(cleaning["cleaning_intensity"], "moderate")
        self.assertIn(
            "measurement_text",
            cleaning["statistical_cleaning"]["missing_values_at_fit"],
        )
        self.assertEqual(
            model.best_pipeline.named_steps["cleaner"].cleaning_intensity,
            "moderate",
        )

    def test_extended_randomized_tuning_is_opt_in_and_exportable(self):
        data = _classification_data(rows=84)
        with tempfile.TemporaryDirectory() as directory:
            model = ReliableAutoML(
                "label",
                partitioner="train_test_split",
                partitioner_options={"test_size": 0.25},
                candidates=[
                    ("logistic_regression", LogisticRegression(max_iter=500))
                ],
                tuning="extended",
                tuning_iterations=3,
                tuning_cv=2,
                extended_param_distributions={
                    "logistic_regression": {"C": [0.05, 0.5, 5.0]}
                },
                cv=2,
                reliability_options={
                    "reliability_required": False,
                },
            ).fit(data)
            self.assertEqual(model.fit_report["selection"]["tuning"], "extended")
            self.assertEqual(model.fit_report["selection"]["tuning_iterations"], 3)
            self.assertIn("classifier__C", model.best_params_)
            original = model.predict(data.drop(columns="label").head(6))
            artifact = Path(directory) / "selected-pipeline.pkl"
            model.save(artifact)
            restored = ReliablePipeline.load(artifact)
            np.testing.assert_array_equal(
                restored.predict(data.drop(columns="label").head(6)),
                original,
            )

    def test_default_candidate_pool_completes_selection(self):
        model = ReliableAutoML(
            "label",
            cv=2,
            reliability_options={"reliability_required": False},
        ).fit(_classification_data(rows=80))

        self.assertEqual(
            {entry["model"] for entry in model.leaderboard},
            {
                "meta_pipeline",
                "perf_meta_select",
                "sc_logistic",
                "sc_lda",
                "sc_knn",
                "sc_mlp",
                "sc_meta",
            },
        )
        self.assertTrue(all(entry["status"] == "ok" for entry in model.leaderboard))
        self.assertIn(
            model.selected_model_name,
            {entry["model"] for entry in model.leaderboard},
        )

    def test_statcore_can_be_disabled_and_extended_space_runs(self):
        off = ReliableAutoML(
            "label", cv=2, include_statcore=False,
            reliability_options={"reliability_required": False},
        )
        self.assertEqual(
            [name for name, _ in off._candidate_models()],
            ["meta_pipeline", "perf_meta_select"],
        )
        for name, estimator in ReliableAutoML("label")._candidate_models():
            if name.startswith("sc_"):
                for extended in (False, True):
                    space = ReliableAutoML._default_parameter_space(
                        estimator, extended=extended
                    )
                    self.assertTrue(space, name)

    def test_multicollinearity_report_is_target_free_and_does_not_change_model(self):
        rng = np.random.default_rng(8)
        rows = 160
        a = rng.normal(size=rows)
        data = pd.DataFrame({
            "a": a, "a_copy": 2 * a + 0.01 * rng.normal(size=rows), "b": rng.normal(size=rows),
        })
        data["label"] = ((a + data["b"]) > 0).astype(int)
        base = dict(candidates=[("lr", LogisticRegression(max_iter=300))], cv=2,
                    reliability_options={"reliability_required": False})
        on = ReliableAutoML("label", **base).fit(data)
        off = ReliableAutoML("label", data_intelligence=None, **base).fit(data)
        report = on.data_intelligence_report
        pairs = {frozenset((p["a"], p["b"])) for p in report["redundant_pairs"]}
        self.assertIn(frozenset(("a", "a_copy")), pairs)
        self.assertNotIn("label", report["vif"])
        self.assertIn("data_intelligence", on.fit_report)
        self.assertIsNone(off.data_intelligence_report)
        self.assertEqual(on.leaderboard[0]["mean_score"], off.leaderboard[0]["mean_score"])
        with self.assertRaises(ValueError):
            ReliableAutoML("label", data_intelligence="everything")

    def test_profile_adds_hazard_report_without_changing_model(self):
        rng = np.random.default_rng(10)
        rows = 160
        a = rng.normal(size=rows)
        data = pd.DataFrame({"a": a, "b": rng.normal(size=rows), "const": 1.0})
        data["label"] = ((a + data["b"]) > 0).astype(int)
        base = dict(candidates=[("lr", LogisticRegression(max_iter=300))], cv=2,
                    reliability_options={"reliability_required": False})
        prof = ReliableAutoML("label", data_intelligence="profile", **base).fit(data)
        mc = ReliableAutoML("label", **base).fit(data)
        report = prof.data_intelligence_report
        self.assertNotIn("error", report)
        self.assertIn("const", report["profile"]["hazards"]["constant"])
        self.assertNotIn("label", str(report["profile"]["overview"]["target_column"]))
        self.assertNotIn("profile", mc.data_intelligence_report)
        self.assertEqual(prof.leaderboard[0]["mean_score"], mc.leaderboard[0]["mean_score"])

    def test_target_sections_are_report_only_and_selectable(self):
        rng = np.random.default_rng(11)
        rows = 240
        data = pd.DataFrame({"a": rng.normal(size=rows), "b": rng.normal(size=rows)})
        data["label"] = ((data["a"] + 0.5 * data["b"]) > 0).astype(int)
        base = dict(candidates=[("lr", LogisticRegression(max_iter=300))], cv=2,
                    reliability_options={"reliability_required": False})
        rich = ReliableAutoML("label", data_intelligence=("multicollinearity", "power", "active_learning",
                                                           "group_scan"), **base).fit(data)
        plain = ReliableAutoML("label", **base).fit(data)
        sections = rich.data_intelligence_report["sections"]
        self.assertEqual(set(sections), {"power", "active_learning", "group_scan"})
        self.assertNotIn("error", sections["power"])
        self.assertIn("ridge_cv_skill", sections["power"]["predictive"])
        self.assertIn("ranking", sections["active_learning"])
        self.assertNotIn("sections", plain.data_intelligence_report)
        self.assertEqual(rich.leaderboard[0]["mean_score"], plain.leaderboard[0]["mean_score"])
        with self.assertRaises(ValueError):
            ReliableAutoML("label", data_intelligence=("power", "nonsense"))
        with self.assertRaises(ValueError):
            ReliableAutoML("label", data_intelligence=())

    def test_export_report_writes_files_and_optional_model(self):
        rng = np.random.default_rng(9)
        data = pd.DataFrame({"a": rng.normal(size=120), "b": rng.normal(size=120)})
        data["label"] = ((data["a"] + data["b"]) > 0).astype(int)
        base = dict(candidates=[("lr", LogisticRegression(max_iter=300))], cv=2,
                    reliability_options={"reliability_required": False})
        with tempfile.TemporaryDirectory() as tmp:
            auto = ReliableAutoML("label", export_dir=Path(tmp) / "auto", export_model=True, **base).fit(data)
            for name in ("automl_report.json", "automl_report.txt", "automl_report.html",
                         "leaderboard.csv", "data_intelligence.json", "reliable_pipeline.joblib"):
                self.assertTrue((Path(tmp) / "auto" / name).exists(), name)
            written = auto.export_report(Path(tmp) / "again", formats="json")
            self.assertNotIn("html", written)
            self.assertNotIn("model", written)
        with self.assertRaises(ValueError):
            ReliableAutoML("label", **base).fit(data).export_report()

    def test_statcore_candidate_wins_when_it_is_the_best_model(self):
        rng = np.random.default_rng(5)
        rows = 160
        x = rng.normal(size=(rows, 2))
        label = (x[:, 0] * x[:, 1] > 0).astype(int)
        data = pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "label": label})
        model = ReliableAutoML(
            "label", cv=2, tuning_cv=2,
            reliability_options={"reliability_required": False},
        ).fit(data)
        scores = {e["model"]: e for e in model.leaderboard}
        self.assertEqual(len(scores), 7)
        self.assertTrue(all(e["status"] == "ok" for e in scores.values()))
        self.assertIn("sc_knn", scores)
        self.assertEqual(model.leaderboard[0]["model"], model.selected_model_name)

    def test_reliableml_meta_learner_can_be_custom_automl_candidate(self):
        model = ReliableAutoML(
            "label",
            partitioner="train_test_split",
            partitioner_options={"test_size": 0.25},
            candidates=[
                (
                    "reliable_meta_pipeline",
                    MetaPipelineClassifier(n_folds=2, random_state=17),
                )
            ],
            light_param_grids={
                "reliable_meta_pipeline": {
                    "meta_classifier": [
                        LogisticRegression(C=0.5, max_iter=500),
                        LogisticRegression(C=1.0, max_iter=500),
                    ]
                }
            },
            cv=2,
            tuning_cv=2,
            reliability_options={"reliability_required": False},
        ).fit(_classification_data(rows=80))

        self.assertEqual(model.selected_model_name, "reliable_meta_pipeline")
        self.assertIn("classifier__meta_classifier", model.best_params_)
        self.assertEqual(
            model.leaderboard[0]["status"],
            "ok",
        )
        predictions = model.predict(_classification_data(rows=12).drop(columns="label"))
        self.assertEqual(len(predictions), 12)

    def test_progress_bar_is_opt_in_and_reports_completion(self):
        output = io.StringIO()
        with redirect_stderr(output):
            ReliableAutoML(
                "label",
                partitioner="train_test_split",
                partitioner_options={"test_size": 0.25},
                candidates=[
                    ("logistic_regression", LogisticRegression(max_iter=500))
                ],
                cv=2,
                tuning_cv=2,
                progress_bar=True,
                reliability_options={"reliability_required": False},
            ).fit(_classification_data(rows=80))

        self.assertIn("AutoML: complete", output.getvalue())
        self.assertIn("100%", output.getvalue())

    def test_s_attp_options_are_forwarded_and_reported(self):
        options = {
            "test_size": 0.25,
            "random_state": 15,
        }
        model = ReliableAutoML(
            "label",
            partitioner="s_attp_cluster",
            partitioner_options=options,
            candidates=[("logistic_regression", LogisticRegression(max_iter=500))],
            cv=2,
            reliability_options={
                "bootstrap_samples": 20,
                "bias_variance_folds": 2,
                "bias_variance_repeats": 1,
            },
        ).fit(_classification_data())

        self.assertEqual(model.fit_report["partitioner"]["name"], "s_attp_cluster")
        self.assertEqual(model.fit_report["partitioner"]["options"], options)
        self.assertFalse(model.fit_report["partitioner"]["used_installed_defaults"])

    def test_additional_attp_partitioners_return_usable_partitions(self):
        for partitioner in ("s_attp_reg", "s_attp_ts"):
            with self.subTest(partitioner=partitioner):
                model = ReliableAutoML(
                    "label",
                    partitioner=partitioner,
                    candidates=[
                        ("logistic_regression", LogisticRegression(max_iter=500))
                    ],
                    cv=2,
                    reliability_options={"reliability_required": False},
                ).fit(_classification_data())
                self.assertEqual(
                    model.fit_report["partitioner"]["name"],
                    partitioner,
                )
                self.assertEqual(
                    model.fit_report["partition_sizes"]["train"]
                    + model.fit_report["partition_sizes"]["test"],
                    120,
                )

    def test_custom_partitioner_and_result_adapter_are_supported(self):
        data = _classification_data(rows=60)
        observed = {}

        def custom_partitioner(X, y, *, test_fraction):
            observed["rows"] = len(X)
            observed["target_rows"] = len(y)
            split_at = int(len(y) * (1 - test_fraction))
            return np.arange(split_at), np.arange(split_at, len(y))

        model = ReliableAutoML(
            "label",
            partitioner=custom_partitioner,
            partitioner_options={"test_fraction": 0.2},
            candidates=[("logistic_regression", LogisticRegression(max_iter=500))],
            cv=2,
            reliability_options={
                "reliability_required": False,
            },
        ).fit(data)
        self.assertEqual(observed, {"rows": len(data), "target_rows": len(data)})
        self.assertIn("custom_partitioner", model.fit_report["partitioner"]["name"])

        X_train = pd.DataFrame({"x": [1.0, 2.0]}, index=[0, 1])
        X_test = pd.DataFrame({"x": [3.0]}, index=[2])
        adapted = _partition_indices(
            (X_train, X_test, pd.Series([0, 1]), pd.Series([1]))
        )
        np.testing.assert_array_equal(adapted[0], [0, 1])
        np.testing.assert_array_equal(adapted[1], [2])

    def test_bad_split_partitions_fail_with_context(self):
        data = _classification_data(rows=60)

        def incomplete_splitter(X, y):
            return np.arange(len(y) - 2), np.array([len(y) - 1])

        with self.assertRaisesRegex(ValueError, "complete train/test partition"):
            ReliableAutoML(
                "label",
                partitioner=incomplete_splitter,
                candidates=[
                    ("logistic_regression", LogisticRegression(max_iter=500))
                ],
                cv=2,
                reliability_options={"reliability_required": False},
            ).fit(data)

    def test_invalid_scorer_fails_before_partition_or_training(self):
        with self.assertRaisesRegex(ValueError, "Unknown scikit-learn scoring"):
            ReliableAutoML(
                "label",
                scoring="not-a-real-scorer",
            ).fit(_classification_data())


if __name__ == "__main__":
    unittest.main()
