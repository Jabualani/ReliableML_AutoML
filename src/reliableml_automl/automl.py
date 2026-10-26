from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from numbers import Integral
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler
from tqdm.auto import tqdm

from ml_reliability import MetaPipelineClassifier, PerfMetaSelect, ReliablePipeline
from ml_reliability.statcore_estimators import (
    STATCORE_VARIANTS,
    StatCoreKNNClassifier,
    StatCoreLDAClassifier,
    StatCoreLogisticClassifier,
    StatCoreMetaClassifier,
    StatCoreMLPClassifier,
)
from ml_reliability.cleaning import StatisticalDataCleaner


class AutoMLFitError(RuntimeError):
    """Raised when no candidate model can be selected and fitted."""


ATTP_SPLITTERS = (
    "s_attp_clf",
    "sattp_split",
    "s_attp_cluster",
    "s_attp_clustered_sampling",
    "s_attp_reg",
    "s_attp_ts",
    "train_test_split",
)


class DataIntegrityTransformer(BaseEstimator, TransformerMixin):
    """Convert mostly numeric text columns while preserving the DataFrame schema."""

    def __init__(self, numeric_coercion_threshold: float = 0.5) -> None:
        self.numeric_coercion_threshold = numeric_coercion_threshold

    def fit(self, X: pd.DataFrame, y: Any = None) -> DataIntegrityTransformer:
        _ = y
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DataIntegrityTransformer requires a pandas DataFrame.")
        if not 0 < self.numeric_coercion_threshold <= 1:
            raise ValueError("numeric_coercion_threshold must be in (0, 1].")
        self.feature_names_in_ = tuple(X.columns)
        self.converted_numeric_columns_: list[str] = []
        self.invalid_numeric_counts_: dict[str, int] = {}
        for column in X.columns:
            series = X[column]
            if not (
                pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
            ):
                continue
            observed = series.dropna()
            if observed.empty:
                continue
            numeric = pd.to_numeric(observed, errors="coerce")
            conversion_rate = float(numeric.notna().mean())
            if conversion_rate >= self.numeric_coercion_threshold:
                self.converted_numeric_columns_.append(column)
                self.invalid_numeric_counts_[column] = int(numeric.isna().sum())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DataIntegrityTransformer requires a pandas DataFrame.")
        if tuple(X.columns) != self.feature_names_in_:
            raise ValueError("Input feature names and order must match the fit data.")
        result = X.copy()
        for column in self.converted_numeric_columns_:
            result[column] = pd.to_numeric(result[column], errors="coerce")
        return result

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is not None and tuple(input_features) != self.feature_names_in_:
            raise ValueError("input_features must match the fitted feature names.")
        return np.asarray(self.feature_names_in_, dtype=object)

    def get_integrity_report(self) -> dict[str, Any]:
        return {
            "numeric_text_columns": list(self.converted_numeric_columns_),
            "invalid_numeric_values_treated_as_missing_at_fit": dict(
                self.invalid_numeric_counts_
            ),
            "numeric_coercion_threshold": self.numeric_coercion_threshold,
        }


class FeatureTypeSelector:
    def __init__(self, kind: str, excluded_columns: Sequence[str] = ()) -> None:
        self.kind = kind
        self.excluded_columns = tuple(excluded_columns)

    def __call__(self, X: pd.DataFrame) -> list[str]:
        if self.kind == "numeric":
            return [
                column
                for column in X.columns
                if column not in self.excluded_columns
                and pd.api.types.is_numeric_dtype(X[column])
                and not pd.api.types.is_bool_dtype(X[column])
            ]
        if self.kind == "categorical":
            return [
                column
                for column in X.columns
                if column not in self.excluded_columns
                and (
                    not pd.api.types.is_numeric_dtype(X[column])
                    or pd.api.types.is_bool_dtype(X[column])
                )
            ]
        raise ValueError(f"Unsupported feature selector kind {self.kind!r}.")


class CategoryTokenEncoder(BaseEstimator, TransformerMixin):
    """Convert category values to type-tagged strings before one-hot encoding."""

    def fit(self, X: Any, y: Any = None) -> CategoryTokenEncoder:
        _ = y
        values = np.asarray(X, dtype=object)
        if values.ndim != 2:
            raise ValueError("Categorical input must be a two-dimensional matrix.")
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X: Any) -> np.ndarray:
        values = np.asarray(X, dtype=object)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError("Categorical input width differs from fitted data.")
        tokens = np.empty(values.shape, dtype=object)
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = values[row_index, column_index]
                if pd.isna(value):
                    tokens[row_index, column_index] = "__missing__"
                else:
                    value_type = type(value)
                    tokens[row_index, column_index] = (
                        f"{value_type.__module__}.{value_type.__qualname__}:{value!r}"
                    )
        return tokens.astype(str)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is None:
            return np.asarray(
                [f"x{index}" for index in range(self.n_features_in_)],
                dtype=object,
            )
        return np.asarray(input_features, dtype=object)


class AdaptiveNumericScaler(BaseEstimator, TransformerMixin):
    """Scale each numeric feature according to training-fold distribution."""

    def __init__(self, strategy: str = "auto") -> None:
        self.strategy = strategy

    def fit(self, X: Any, y: Any = None) -> AdaptiveNumericScaler:
        _ = y
        if self.strategy not in {"auto", "standard", "robust", "minmax", "none"}:
            raise ValueError(
                "scaling_strategy must be 'auto', 'standard', 'robust', "
                "'minmax', or 'none'."
            )
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("Numeric scaling requires a finite 2D feature matrix.")
        self.n_features_in_ = values.shape[1]
        self.scalers_: list[Any] = []
        self.feature_strategies_: list[str] = []
        for column_index in range(self.n_features_in_):
            feature = values[:, column_index]
            strategy = self._feature_strategy(feature)
            self.feature_strategies_.append(strategy)
            if strategy == "standard":
                scaler = StandardScaler()
            elif strategy == "robust":
                scaler = RobustScaler()
            elif strategy == "minmax":
                scaler = MinMaxScaler(feature_range=(0, 10))
            else:
                scaler = None
            if scaler is not None:
                scaler.fit(feature.reshape(-1, 1))
            self.scalers_.append(scaler)
        return self

    def transform(self, X: Any) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError("Numeric input width differs from fitted data.")
        if not np.isfinite(values).all():
            raise ValueError("Numeric scaling requires finite values.")
        transformed = values.copy()
        for index, scaler in enumerate(self.scalers_):
            if scaler is not None:
                transformed[:, index] = scaler.transform(
                    values[:, index].reshape(-1, 1)
                ).ravel()
        return transformed

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is None:
            return np.asarray(
                [f"x{index}" for index in range(self.n_features_in_)],
                dtype=object,
            )
        return np.asarray(input_features, dtype=object)

    def _feature_strategy(self, values: np.ndarray) -> str:
        if self.strategy != "auto":
            return self.strategy
        if len(values) >= 5:
            q1, q3 = np.quantile(values, [0.25, 0.75])
            spread = q3 - q1
            if np.any((values < q1 - 1.5 * spread) | (values > q3 + 1.5 * spread)):
                return "robust"
        if len(values) < 10:
            return "minmax"
        series = pd.Series(values)
        standard_deviation = float(series.std(ddof=0))
        if standard_deviation and np.isfinite(standard_deviation):
            normal_like = (
                abs(float(series.mean()) - float(series.median()))
                <= 0.1 * standard_deviation
                and abs(float(series.skew())) < 0.5
                and abs(float(series.kurtosis())) < 1.5
            )
            if normal_like:
                return "standard"
        return "minmax"


_TARGET_SECTIONS = ("power", "construction", "active_learning", "group_scan", "root_cause")
_DI_OPTIONS = ("multicollinearity", "profile") + _TARGET_SECTIONS


class ReliableAutoML:
    """Select a classifier using cross-validation on an ATTP training split.

    The default partitioner is ``attp.s_attp_clf``. When
    ``partitioner_options`` is omitted, ATTP is called without overrides so
    its installed defaults are used. A candidate is selected only from
    cross-validation scores on the returned training rows; ATTP's score-driven
    partition selection is reported as split-design evidence, not an unbiased
    final performance estimate.

    Other callable splitters can be supplied. They must return an object with
    ``train_indices`` and ``test_indices``, a mapping with those keys, or
    train/test data frames whose row indexes identify the input rows. Supply
    ``partitioner_result_adapter`` to support a different return convention.
    """

    def __init__(
        self,
        target_column: str,
        *,
        partitioner: str | Callable[..., Any] = "s_attp_clf",
        partitioner_options: Mapping[str, Any] | None = None,
        partitioner_result_adapter: Callable[
            [Any, pd.DataFrame, pd.Series], tuple[Any, Any]
        ]
        | None = None,
        candidates: Sequence[tuple[str, Any]] | None = None,
        include_statcore: bool = True,
        data_intelligence: str | Sequence[str] | None = "multicollinearity",
        vif_threshold: float = 10.0,
        export_dir: str | PathLike[str] | None = None,
        export_formats: Sequence[str] = ("json", "txt", "html"),
        export_model: bool = False,
        tuning: str = "light",
        light_param_grids: Mapping[str, Mapping[str, Sequence[Any]]] | None = None,
        extended_param_distributions: Mapping[str, Mapping[str, Any]] | None = None,
        tuning_iterations: int = 20,
        tuning_cv: int = 2,
        cleaning_intensity: str = "light",
        scaling_strategy: str = "auto",
        ordinal_features: Mapping[str, Sequence[Any]] | None = None,
        cv: int = 3,
        scoring: str = "balanced_accuracy",
        random_state: int = 42,
        progress_bar: bool = False,
        plot_dir: str | PathLike[str] = "reliability_plots",
        reliability_options: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(target_column, str) or not target_column:
            raise ValueError("target_column must be a non-empty string.")
        if not isinstance(cv, int) or isinstance(cv, bool) or cv < 2:
            raise ValueError("cv must be an integer of at least 2.")
        if not isinstance(random_state, int) or isinstance(random_state, bool):
            raise ValueError("random_state must be an integer.")
        if not isinstance(progress_bar, bool):
            raise ValueError("progress_bar must be a boolean.")
        if tuning not in {"light", "extended"}:
            raise ValueError("tuning must be 'light' or 'extended'.")
        wanted = (
            ()
            if data_intelligence is None
            else (data_intelligence,)
            if isinstance(data_intelligence, str)
            else tuple(data_intelligence)
        )
        unknown = [s for s in wanted if s not in _DI_OPTIONS]
        if unknown or (data_intelligence is not None and not wanted):
            raise ValueError(
                f"data_intelligence must be None, one of {_DI_OPTIONS}, or a sequence of them."
            )
        if not isinstance(vif_threshold, (int, float)) or vif_threshold <= 1:
            raise ValueError("vif_threshold must be a number greater than 1.")
        if not isinstance(tuning_cv, int) or isinstance(tuning_cv, bool) or tuning_cv < 2:
            raise ValueError("tuning_cv must be an integer of at least 2.")
        if (
            not isinstance(tuning_iterations, int)
            or isinstance(tuning_iterations, bool)
            or tuning_iterations < 1
        ):
            raise ValueError("tuning_iterations must be a positive integer.")
        if cleaning_intensity not in {"light", "moderate", "heavy"}:
            raise ValueError(
                "cleaning_intensity must be 'light', 'moderate', or 'heavy'."
            )
        if scaling_strategy not in {"auto", "standard", "robust", "minmax", "none"}:
            raise ValueError(
                "scaling_strategy must be 'auto', 'standard', 'robust', "
                "'minmax', or 'none'."
            )
        if light_param_grids is not None and not isinstance(
            light_param_grids, Mapping
        ):
            raise TypeError("light_param_grids must be a mapping.")
        if extended_param_distributions is not None and not isinstance(
            extended_param_distributions, Mapping
        ):
            raise TypeError("extended_param_distributions must be a mapping.")
        if ordinal_features is not None and not isinstance(ordinal_features, Mapping):
            raise TypeError("ordinal_features must map column names to ordered values.")
        if isinstance(partitioner, str) and partitioner not in ATTP_SPLITTERS:
            raise ValueError(
                f"partitioner must be one of {ATTP_SPLITTERS!r} or a callable."
            )
        if not isinstance(partitioner, str) and not callable(partitioner):
            raise TypeError("partitioner must be a supported name or callable.")
        if partitioner_options is not None and not isinstance(
            partitioner_options, Mapping
        ):
            raise TypeError("partitioner_options must be a mapping.")
        if partitioner_result_adapter is not None and not callable(
            partitioner_result_adapter
        ):
            raise TypeError("partitioner_result_adapter must be callable.")

        self.target_column = target_column
        self.partitioner = partitioner
        self.partitioner_options = (
            {} if partitioner_options is None else dict(partitioner_options)
        )
        self.partitioner_result_adapter = partitioner_result_adapter
        self.candidates = candidates
        self.include_statcore = include_statcore
        self.data_intelligence = data_intelligence
        self.vif_threshold = vif_threshold
        self.export_dir = export_dir
        self.export_formats = export_formats
        self.export_model = export_model
        self.tuning = tuning
        self.light_param_grids = light_param_grids
        self.extended_param_distributions = extended_param_distributions
        self.tuning_iterations = tuning_iterations
        self.tuning_cv = tuning_cv
        self.cleaning_intensity = cleaning_intensity
        self.scaling_strategy = scaling_strategy
        self.ordinal_features = ordinal_features
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.progress_bar = progress_bar
        self.plot_dir = Path(plot_dir)
        self.reliability_options = (
            {} if reliability_options is None else dict(reliability_options)
        )
        self._fitted = False

    def fit(
        self,
        data: pd.DataFrame,
    ) -> ReliableAutoML:
        self._fitted = False
        with tqdm(
            total=1,
            desc="AutoML: preparing and partitioning",
            unit="step",
            disable=not self.progress_bar,
        ) as progress:
            return self._fit(data, progress)

    def _fit(
        self,
        data: pd.DataFrame,
        progress: Any,
    ) -> ReliableAutoML:
        features, target = self._validate_data(data)
        self._validate_scoring()

        partition_features = self._encode_for_partitioning(features)
        partition_target = target.reset_index(drop=True)
        train_indices, test_indices, partition_details = self._partition(
            partition_features,
            partition_target,
        )
        progress.update(1)
        X_train = features.iloc[train_indices].copy()
        y_train = target.iloc[train_indices].copy()
        self.data_intelligence_report_ = self._data_intelligence(X_train, y_train)

        class_counts = y_train.value_counts(dropna=False)
        if len(class_counts) < 2:
            raise ValueError(
                "The partitioner training rows must contain at least two classes."
            )
        if int(class_counts.min()) < self.cv:
            raise ValueError(
                f"cv={self.cv} exceeds the smallest training class count "
                f"({int(class_counts.min())}); reduce cv or provide more data."
            )

        candidate_models = self._candidate_models()
        progress.total = len(candidate_models) * self.cv + 1
        progress.set_description("AutoML: outer CV folds")
        progress.refresh()
        leaderboard: list[dict[str, Any]] = []
        for name, prototype in candidate_models:
            progress.set_postfix(candidate=name)
            candidate_template = self._candidate_pipeline(
                prototype,
            )
            completed_folds = 0

            def advance_progress() -> None:
                nonlocal completed_folds
                completed_folds += 1
                progress.update(1)

            try:
                scores, fold_params = self._nested_candidate_scores(
                    name,
                    prototype,
                    candidate_template,
                    X_train,
                    y_train,
                    progress_callback=advance_progress,
                )
                leaderboard.append(
                    {
                        "model": name,
                        "status": "ok",
                        "mean_score": float(np.mean(scores)),
                        "std_score": float(np.std(scores)),
                        "outer_fold_scores": [float(score) for score in scores],
                        "outer_fold_best_params": fold_params,
                    }
                )
            except (
                ValueError,
                TypeError,
                RuntimeError,
                FloatingPointError,
                KeyError,
            ) as error:
                progress.update(self.cv - completed_folds)
                leaderboard.append(
                    {
                        "model": name,
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue

        successful = [entry for entry in leaderboard if entry["status"] == "ok"]
        if not successful:
            detail = "; ".join(
                f"{entry['model']}: {entry['error']}" for entry in leaderboard
            )
            raise AutoMLFitError(
                "No candidate completed cross-validation. " + detail
            )
        successful.sort(key=lambda entry: (-entry["mean_score"], entry["model"]))
        selected_estimator = None
        progress.set_postfix(candidate="selecting")
        selected_name = ""
        selected_params: dict[str, Any] = {}
        refit_errors = []
        for candidate_result in successful:
            name = candidate_result["model"]
            prototype = dict(candidate_models)[name]
            template = self._candidate_pipeline(prototype)
            try:
                search = self._make_search(name, prototype, template)
                search.fit(X_train, y_train)
            except (
                ValueError,
                TypeError,
                RuntimeError,
                FloatingPointError,
                KeyError,
            ) as error:
                candidate_result["refit_error"] = (
                    f"{type(error).__name__}: {error}"
                )
                candidate_result["status"] = "failed_refit"
                refit_errors.append(f"{name}: {error}")
                continue
            selected_name = name
            selected_estimator = search.best_estimator_
            self.best_params_ = dict(search.best_params_)
            selected_params = {
                key: _report_parameter(value)
                for key, value in self.best_params_.items()
            }
            candidate_result["selected_tuning_score"] = float(search.best_score_)
            candidate_result["best_params"] = selected_params
            break
        if selected_estimator is None:
            raise AutoMLFitError(
                "Candidates passed nested CV but all failed during final refit. "
                + "; ".join(refit_errors)
            )

        pipeline_options = dict(self.reliability_options)
        pipeline_options.setdefault("plot_dir", self.plot_dir)
        self.model_ = ReliablePipeline(selected_estimator, **pipeline_options)
        self.model_.fit(X_train, y_train)
        progress.update(1)
        self.best_pipeline_ = self.model_.estimator
        fitted_pipeline = self.best_pipeline_
        integrity = fitted_pipeline.named_steps["data_integrity"]
        cleaner = fitted_pipeline.named_steps["cleaner"]
        self.cleaning_report_ = {
            "data_integrity": integrity.get_integrity_report(),
            "statistical_cleaning": cleaner.get_cleaning_report(),
            "cleaning_intensity": self.cleaning_intensity,
            "scaling_strategy": self.scaling_strategy,
            "ordinal_features": (
                {}
                if self.ordinal_features is None
                else {
                    name: list(categories)
                    for name, categories in self.ordinal_features.items()
                }
            ),
        }

        self.selected_model_name_ = selected_name
        self.leaderboard_ = sorted(
            leaderboard,
            key=lambda entry: (
                entry["status"] != "ok",
                -entry.get("mean_score", float("-inf")),
                entry["model"],
            ),
        )
        self.train_indices_ = tuple(int(index) for index in train_indices)
        self.test_indices_ = tuple(int(index) for index in test_indices)
        self.fit_report_ = {
            "target_column": self.target_column,
            "partitioner": partition_details,
            "partition_sizes": {
                "train": len(train_indices),
                "test": len(test_indices),
                "total": len(data),
            },
            "selection": {
                "method": (
                    "nested stratified cross-validation on partitioner training rows"
                ),
                "cv": self.cv,
                "scoring": self.scoring,
                "selected_model": selected_name,
                "partition_test_used_for_model_selection": False,
                "tuning": self.tuning,
                "tuning_cv": self.tuning_cv,
                "tuning_iterations": (
                    self.tuning_iterations if self.tuning == "extended" else None
                ),
                "best_params": selected_params,
            },
            "warnings": [
                "s_attp_clf fits LDA and clusters before partitioning, then selects "
                "a candidate partition using its test score. Its internal test "
                "score and metrics on those rows are not unbiased estimates of "
                "generalization. Use untouched representative data for final evaluation."
            ]
            if partition_details["name"] in {"s_attp_clf", "sattp_split"}
            else [
                "Use untouched representative data for final model evaluation; "
                "partition diagnostics do not certify generalization."
            ],
            "leaderboard": [dict(entry) for entry in self.leaderboard_],
            "cleaning": self.cleaning_report_,
            "data_intelligence": self.data_intelligence_report_,
        }
        self._fitted = True
        if self.export_dir is not None:
            self.export_report(self.export_dir, include_model=self.export_model)
        progress.set_description("AutoML: complete")
        progress.set_postfix(candidate=selected_name)
        return self

    @property
    def selected_model_name(self) -> str:
        self._ensure_fitted()
        return self.selected_model_name_

    @property
    def leaderboard(self) -> list[dict[str, Any]]:
        self._ensure_fitted()
        return [dict(entry) for entry in self.leaderboard_]

    @property
    def best_pipeline(self) -> Pipeline:
        """Return the fitted preprocessing and classifier sklearn pipeline."""
        self._ensure_fitted()
        return self.best_pipeline_

    @property
    def cleaning_report(self) -> dict[str, Any]:
        self._ensure_fitted()
        return {
            "data_integrity": dict(self.cleaning_report_["data_integrity"]),
            "statistical_cleaning": dict(
                self.cleaning_report_["statistical_cleaning"]
            ),
            "cleaning_intensity": self.cleaning_report_["cleaning_intensity"],
            "scaling_strategy": self.cleaning_report_["scaling_strategy"],
            "ordinal_features": dict(self.cleaning_report_["ordinal_features"]),
        }

    @property
    def data_intelligence_report(self) -> dict[str, Any] | None:
        """Multicollinearity intelligence of the training partition (None when disabled)."""
        self._ensure_fitted()
        return self.data_intelligence_report_

    def _data_intelligence(self, X_train: pd.DataFrame, y_train: Any = None) -> dict[str, Any] | None:
        """Report-only and training-partition-only; a failure here must never stop the fit."""
        if self.data_intelligence is None:
            return None
        wanted = (
            (self.data_intelligence,)
            if isinstance(self.data_intelligence, str)
            else tuple(self.data_intelligence)
        )
        try:
            from ml_reliability import DataIntelligence, multicollinearity_report, run_sections

            report = multicollinearity_report(X_train, vif_threshold=self.vif_threshold)
            if "profile" in wanted:
                # Target-free EDA/hazard profile of the training partition only.
                profile = DataIntelligence(
                    mode="comprehensive_eda",
                    vif_threshold=self.vif_threshold,
                    random_state=self.random_state,
                ).analyze(X_train)
                d = profile.details
                report = {
                    **report,
                    "profile": {
                        "summary_score": profile.summary_score,
                        "overview": d["overview"],
                        "hazards": d.get("what_happened", {}).get("hazards", {}),
                        "outliers": d.get("why", {}).get("outliers"),
                        "recommendations": d.get("recommendations", []),
                        "narrative": d.get("narrative"),
                    },
                }
            extra = [s for s in wanted if s in _TARGET_SECTIONS]
            if extra and y_train is not None:
                frame = X_train.reset_index(drop=True).copy()
                tname = "__target__"
                frame[tname] = pd.Series(y_train).reset_index(drop=True)
                sections = run_sections(
                    frame, tname, tuple(extra), random_state=self.random_state,
                    **{"active_learning": {"n_steps": 5, "n_repeats": 2,
                                            "strategies": ["random", "margin", "kcenter", "coverage_margin"]}},
                )
                for name, result in sections.items():
                    result.pop("series", None)
                report = {**report, "sections": sections}
            return report
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    @property
    def fit_report(self) -> dict[str, Any]:
        self._ensure_fitted()
        return {
            **self.fit_report_,
            "partitioner": dict(self.fit_report_["partitioner"]),
            "partition_sizes": dict(self.fit_report_["partition_sizes"]),
            "selection": dict(self.fit_report_["selection"]),
            "warnings": list(self.fit_report_["warnings"]),
            "leaderboard": [dict(entry) for entry in self.leaderboard_],
            "cleaning": self.cleaning_report,
        }

    def predict(self, X: Any) -> Any:
        self._ensure_fitted()
        return self.model_.predict(X)

    def predict_with_report(self, X: Any) -> tuple[Any, Any]:
        self._ensure_fitted()
        return self.model_.predict_with_report(X)

    def evaluate(
        self,
        X: Any,
        y_true: Any,
        sensitive_features: Mapping[str, Any] | None = None,
        positive_label: Any | None = None,
    ) -> Any:
        self._ensure_fitted()
        return self.model_.evaluate(
            X,
            y_true,
            sensitive_features=sensitive_features,
            positive_label=positive_label,
        )

    def assess_covariate_shift(self, X: Any) -> Any:
        self._ensure_fitted()
        return self.model_.assess_covariate_shift(X)

    def assess_concept_shift(self, X: Any, y: Any) -> Any:
        self._ensure_fitted()
        return self.model_.assess_concept_shift(X, y)

    def assess_deterioration(self, performance_history: Any, **kwargs: Any) -> Any:
        self._ensure_fitted()
        return self.model_.assess_deterioration(performance_history, **kwargs)

    def explain(self, X: Any, **kwargs: Any) -> Any:
        self._ensure_fitted()
        return self.model_.explain(X, **kwargs)

    def audit_reasons(self, X: Any, y_true: Any, **kwargs: Any) -> Any:
        self._ensure_fitted()
        return self.model_.audit_reasons(X, y_true, **kwargs)

    def export_report(
        self,
        directory: str | PathLike[str] | None = None,
        formats: Sequence[str] | None = None,
        include_model: bool = False,
    ) -> dict[str, str]:
        """Save the fit report (selection, leaderboard, cleaning, data intelligence).

        Writes ``automl_report.{json,txt,html}``, ``leaderboard.csv`` and, when the
        data-intelligence hook ran, ``data_intelligence.json``. With
        ``include_model=True`` the fitted pipeline is saved as ``reliable_pipeline.joblib``.
        Returns a ``{name: path}`` mapping.
        """
        import os

        from ml_reliability import export_report

        self._ensure_fitted()
        directory = directory if directory is not None else self.export_dir
        if directory is None:
            raise ValueError("Provide a directory (or set export_dir).")
        directory = os.fspath(directory)
        os.makedirs(directory, exist_ok=True)
        report = self.fit_report
        selection = report["selection"]
        lines = [
            "RELIABLEML AUTOML REPORT",
            "=" * 60,
            f"selected model: {self.selected_model_name_}",
            f"selection: {selection}",
            "",
            "Leaderboard",
        ]
        for rank, entry in enumerate(self.leaderboard_, start=1):
            lines.append(f"  {rank}. {entry}")
        di = self.data_intelligence_report_
        if di:
            lines += ["", "Multicollinearity (training features only)", f"  {di}"]
        lines += ["", "Warnings"] + [f"  - {w}" for w in report["warnings"]]
        written = export_report(
            report,
            directory,
            name="automl_report",
            formats=self.export_formats if formats is None else formats,
            text="\n".join(lines),
            title="ReliableML AutoML report",
        )
        pd.DataFrame(self.leaderboard_).to_csv(
            os.path.join(directory, "leaderboard.csv"), index=False
        )
        written["leaderboard"] = os.path.join(directory, "leaderboard.csv")
        if di:
            written["data_intelligence"] = export_report(
                di, directory, name="data_intelligence", formats=("json",)
            )["json"]
        if include_model:
            model_path = os.path.join(directory, "reliable_pipeline.joblib")
            self.save(model_path)
            written["model"] = model_path
        return written

    def save(self, path: str | PathLike[str]) -> None:
        """Save the fitted ReliablePipeline artifact for later inference."""
        self._ensure_fitted()
        self.model_.save(path)

    def _validate_data(
        self, data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if data.empty or data.columns.has_duplicates:
            raise ValueError("data must be non-empty with unique column names.")
        if any(not isinstance(column, str) for column in data.columns):
            raise ValueError("All data column names must be strings.")
        if self.target_column not in data.columns:
            raise ValueError(
                f"Target column {self.target_column!r} was not found in data."
            )
        if data.index.has_duplicates:
            raise ValueError("data must have a unique index.")
        if len(data) < 4:
            raise ValueError("At least four rows are required for model selection.")
        if data.select_dtypes(include=["datetime", "datetimetz"]).shape[1]:
            raise ValueError("Datetime features must be converted before fitting.")

        target = data[self.target_column].copy()
        if target.isna().any():
            raise ValueError("Target values must not contain missing labels.")
        if target.nunique() < 2:
            raise ValueError("Target values must contain at least two classes.")
        features = data.drop(columns=[self.target_column]).copy()
        if features.shape[1] == 0:
            raise ValueError("At least one feature column is required.")
        if self.ordinal_features is not None:
            unknown_columns = set(self.ordinal_features) - set(features.columns)
            if unknown_columns:
                raise ValueError(
                    "ordinal_features contains columns absent from the data: "
                    f"{sorted(unknown_columns)!r}."
                )
            for column, categories in self.ordinal_features.items():
                if isinstance(categories, (str, bytes)) or len(categories) < 2:
                    raise ValueError(
                        f"Ordinal column {column!r} requires at least two ordered categories."
                    )
                if len(set(categories)) != len(categories):
                    raise ValueError(
                        f"Ordinal categories for {column!r} must be unique."
                    )
        return features, target

    def _validate_scoring(self) -> None:
        try:
            get_scorer(self.scoring)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Unknown scikit-learn scoring name {self.scoring!r}."
            ) from error

    def _encode_for_partitioning(self, features: pd.DataFrame) -> pd.DataFrame:
        integrity = DataIntegrityTransformer().fit(features)
        prepared = integrity.transform(features)
        cleaner = StatisticalDataCleaner(
            cleaning_intensity=self.cleaning_intensity
        ).fit(prepared)
        clean = cleaner.transform(prepared).reset_index(drop=True)
        ordinal_columns = (
            [] if self.ordinal_features is None else list(self.ordinal_features)
        )
        transformers = []
        transformers.append(
            (
                "numeric",
                "passthrough",
                FeatureTypeSelector("numeric", ordinal_columns),
            )
        )
        if ordinal_columns:
            transformers.append(
                (
                    "ordinal",
                    OrdinalEncoder(
                        categories=[
                            list(self.ordinal_features[column])
                            for column in ordinal_columns
                        ],
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                    ordinal_columns,
                )
            )
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("tokens", CategoryTokenEncoder()),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                                dtype=np.float64,
                            ),
                        ),
                    ]
                ),
                FeatureTypeSelector("categorical", ordinal_columns),
            )
        )
        encoder = ColumnTransformer(
            transformers,
            remainder="drop",
            verbose_feature_names_out=True,
        )
        values = np.asarray(encoder.fit_transform(clean), dtype=float)
        if values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all():
            raise ValueError(
                "Partitioning features must encode to a non-empty finite numeric matrix."
            )
        names = encoder.get_feature_names_out()
        return pd.DataFrame(
            values,
            columns=[str(name) for name in names],
            index=pd.RangeIndex(len(features)),
        )

    def _partition(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if self.partitioner == "train_test_split":
            unexpected_options = set(self.partitioner_options) - {
                "test_size",
                "random_state",
            }
            if unexpected_options:
                raise ValueError(
                    "Unsupported train_test_split options: "
                    f"{sorted(unexpected_options)!r}."
                )
            train, test = train_test_split(
                np.arange(len(target)),
                test_size=self.partitioner_options.get("test_size", 0.2),
                random_state=self.partitioner_options.get(
                    "random_state", self.random_state
                ),
                stratify=target,
            )
            raw_result: Any = (train, test)
            name = "train_test_split"
        elif isinstance(self.partitioner, str):
            import attp

            splitter = getattr(attp, self.partitioner)
            raw_result = splitter(
                features,
                target,
                **self.partitioner_options,
            )
            name = self.partitioner
        else:
            raw_result = self.partitioner(
                features,
                target,
                **self.partitioner_options,
            )
            name = getattr(
                self.partitioner,
                "__qualname__",
                type(self.partitioner).__name__,
            )

        if self.partitioner_result_adapter is not None:
            train_values, test_values = self.partitioner_result_adapter(
                raw_result,
                features,
                target,
            )
        else:
            train_values, test_values = _partition_indices(raw_result)
        train_indices = _to_positions(train_values, len(target), "train")
        test_indices = _to_positions(test_values, len(target), "test")
        if set(train_indices) & set(test_indices):
            raise ValueError("Partitioner returned overlapping train/test rows.")
        if set(train_indices) | set(test_indices) != set(range(len(target))):
            raise ValueError(
                "Partitioner must return a complete train/test partition of all rows."
            )
        if not len(train_indices) or not len(test_indices):
            raise ValueError("Partitioner must return non-empty train and test rows.")
        return (
            train_indices,
            test_indices,
            {
                "name": name,
                "options": dict(self.partitioner_options),
                "used_installed_defaults": (
                    name == "s_attp_clf" and not self.partitioner_options
                ),
                "selected_split": getattr(raw_result, "best_split", None),
                "selected_cluster_count": getattr(
                    raw_result,
                    "best_n_clusters",
                    None,
                ),
                "splitter_test_score_is_not_model_evaluation": name
                in {"s_attp_clf", "sattp_split"},
            },
        )

    def _statcore_candidates(self) -> list[tuple[str, Any]]:
        seed = self.random_state
        return [
            ("sc_logistic", StatCoreLogisticClassifier(random_state=seed)),
            ("sc_lda", StatCoreLDAClassifier(random_state=seed)),
            ("sc_knn", StatCoreKNNClassifier(random_state=seed)),
            ("sc_mlp", StatCoreMLPClassifier(random_state=seed)),
            (
                "sc_meta",
                StatCoreMetaClassifier(
                    n_folds=min(3, self.cv), random_state=seed
                ),
            ),
        ]

    def _candidate_models(self) -> list[tuple[str, Any]]:
        if self.candidates is None:
            candidates = [
                (
                    "meta_pipeline",
                    MetaPipelineClassifier(
                        n_folds=min(3, self.cv),
                        random_state=self.random_state,
                    ),
                ),
                (
                    "perf_meta_select",
                    PerfMetaSelect(
                        cv=2,
                        scoring=self.scoring,
                        random_state=self.random_state,
                    ),
                ),
            ]
            if self.include_statcore:
                candidates.extend(self._statcore_candidates())
        else:
            candidates = list(self.candidates)
        if not candidates:
            raise ValueError("At least one candidate estimator is required.")
        names = [name for name, _ in candidates]
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Candidate names must be non-empty strings.")
        if len(set(names)) != len(names):
            raise ValueError("Candidate names must be unique.")
        for name, estimator in candidates:
            if not hasattr(estimator, "fit") or not hasattr(estimator, "predict"):
                raise TypeError(
                    f"Candidate {name!r} must implement fit and predict."
                )
            clone(estimator)
        return candidates

    def _candidate_pipeline(self, estimator: Any) -> Pipeline:
        ordinal_columns = (
            [] if self.ordinal_features is None else list(self.ordinal_features)
        )
        transformers = []
        transformers.append(
            (
                "numeric",
                AdaptiveNumericScaler(strategy=self.scaling_strategy),
                FeatureTypeSelector("numeric", ordinal_columns),
            )
        )
        if ordinal_columns:
            transformers.append(
                (
                    "ordinal",
                    OrdinalEncoder(
                        categories=[
                            list(self.ordinal_features[column])
                            for column in ordinal_columns
                        ],
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                    ordinal_columns,
                )
            )
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("tokens", CategoryTokenEncoder()),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                FeatureTypeSelector("categorical", ordinal_columns),
            )
        )
        preprocessing = ColumnTransformer(
            transformers,
            remainder="drop",
            verbose_feature_names_out=True,
        )
        return Pipeline(
            [
                ("data_integrity", DataIntegrityTransformer()),
                (
                    "cleaner",
                    StatisticalDataCleaner(
                        cleaning_intensity=self.cleaning_intensity
                    ),
                ),
                ("preprocessing", preprocessing),
                ("classifier", clone(estimator)),
            ]
        )

    def _nested_candidate_scores(
        self,
        name: str,
        estimator: Any,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        progress_callback: Callable[[], None] | None = None,
    ) -> tuple[list[float], list[dict[str, Any]]]:
        splitter = StratifiedKFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state,
        )
        scores = []
        fold_parameters = []
        for train_indices, validation_indices in splitter.split(X, y):
            X_fold = X.iloc[train_indices]
            y_fold = y.iloc[train_indices]
            search = self._make_search(name, estimator, pipeline)
            search.fit(X_fold, y_fold)
            validation_score = float(
                get_scorer(self.scoring)(
                    search.best_estimator_,
                    X.iloc[validation_indices],
                    y.iloc[validation_indices],
                )
            )
            scores.append(validation_score)
            fold_parameters.append(
                {
                    key: _report_parameter(value)
                    for key, value in search.best_params_.items()
                }
            )
            if progress_callback is not None:
                progress_callback()
        return scores, fold_parameters

    def _make_search(
        self,
        name: str,
        estimator: Any,
        pipeline: Pipeline,
    ) -> GridSearchCV | RandomizedSearchCV:
        if self.tuning == "light":
            parameter_grid = self._parameter_space(
                name,
                estimator,
                self.light_param_grids,
                extended=False,
            )
            return GridSearchCV(
                pipeline,
                param_grid=parameter_grid,
                scoring=self.scoring,
                cv=self._inner_splitter(),
                refit=True,
                n_jobs=1,
                error_score="raise",
            )
        parameter_distributions = self._parameter_space(
            name,
            estimator,
            self.extended_param_distributions,
            extended=True,
        )
        return RandomizedSearchCV(
            pipeline,
            param_distributions=parameter_distributions,
            n_iter=self.tuning_iterations,
            scoring=self.scoring,
            cv=self._inner_splitter(),
            refit=True,
            random_state=self.random_state,
            n_jobs=1,
            error_score="raise",
        )

    def _inner_splitter(self) -> StratifiedKFold:
        return StratifiedKFold(
            n_splits=self.tuning_cv,
            shuffle=True,
            random_state=self.random_state,
        )

    def _parameter_space(
        self,
        name: str,
        estimator: Any,
        supplied_spaces: Mapping[str, Mapping[str, Any]] | None,
        *,
        extended: bool,
    ) -> dict[str, Any]:
        if supplied_spaces is not None and name in supplied_spaces:
            raw_space = dict(supplied_spaces[name])
        else:
            raw_space = self._default_parameter_space(estimator, extended=extended)
        if not raw_space:
            mode = "extended_param_distributions" if extended else "light_param_grids"
            raise ValueError(
                f"No default parameter space is available for candidate {name!r}; "
                f"provide {mode}[{name!r}]."
            )
        return {
            key if key.startswith("classifier__") else f"classifier__{key}": values
            for key, values in raw_space.items()
        }

    @staticmethod
    def _default_parameter_space(
        estimator: Any,
        *,
        extended: bool,
    ) -> dict[str, Any]:
        if isinstance(estimator, LogisticRegression):
            if extended:
                return {"C": loguniform(1e-3, 1e3)}
            return {"C": [0.5, 1.0, 2.0]}
        if isinstance(estimator, (RandomForestClassifier, ExtraTreesClassifier)):
            if extended:
                return {
                    "n_estimators": [100, 200, 300, 500],
                    "max_depth": [None, 5, 10, 20, 30],
                    "min_samples_split": randint(2, 20),
                    "min_samples_leaf": randint(1, 10),
                    "max_features": ["sqrt", "log2", 0.5, 0.8, 1.0],
                }
            return {
                "n_estimators": [100, 200],
                "max_depth": [None, 10],
            }
        if isinstance(estimator, MetaPipelineClassifier):
            regularization_values = (
                np.logspace(-3, 3, 25) if extended else [0.5, 1.0, 2.0]
            )
            return {
                "meta_classifier": [
                    LogisticRegression(C=float(value), max_iter=1000)
                    for value in regularization_values
                ]
            }
        if isinstance(estimator, PerfMetaSelect):
            return {"cv": [estimator.cv]}
        if isinstance(estimator, StatCoreLogisticClassifier):
            space = {"variant": list(STATCORE_VARIANTS["logistic"])}
            if extended:
                space["alpha"] = loguniform(1e-4, 1e1)
            return space
        if isinstance(estimator, StatCoreLDAClassifier):
            space = {"variant": list(STATCORE_VARIANTS["lda"])}
            if extended:
                space["reg_param"] = [0.1, 0.3, 0.5, 0.7]
            return space
        if isinstance(estimator, StatCoreKNNClassifier):
            space = {"variant": list(STATCORE_VARIANTS["knn"])}
            if extended:
                space["n_neighbors"] = [3, 5, 9, 15]
                space["weights"] = ["uniform", "distance"]
            return space
        if isinstance(estimator, StatCoreMLPClassifier):
            space = {"variant": list(STATCORE_VARIANTS["mlp"])}
            if extended:
                space["hidden_layer_sizes"] = [(16,), (32, 32), (64, 32)]
                space["dropout"] = [0.0, 0.1, 0.3]
            return space
        if isinstance(estimator, StatCoreMetaClassifier):
            return {
                "meta_variant": [
                    "oof_stacking",
                    "calibrated_stack",
                    "performance_aware",
                    "metrics_based",
                    "statistical_meta",
                ]
                if extended
                else ["oof_stacking", "calibrated_stack"]
            }
        return {}

    def _ensure_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("ReliableAutoML must be fitted before use.")


def _partition_indices(result: Any) -> tuple[Any, Any]:
    if (
        isinstance(result, tuple)
        and len(result) == 2
        and (
            isinstance(result[0], tuple)
            or hasattr(result[0], "train_indices")
        )
    ):
        result = result[0]

    if hasattr(result, "train_indices") and hasattr(result, "test_indices"):
        return result.train_indices, result.test_indices
    if isinstance(result, Mapping):
        if "train_indices" in result and "test_indices" in result:
            return result["train_indices"], result["test_indices"]
    if isinstance(result, tuple):
        if len(result) == 2:
            train, test = result
            if isinstance(train, (pd.DataFrame, pd.Series)) and isinstance(
                test, (pd.DataFrame, pd.Series)
            ):
                return train.index, test.index
            return train, test
        if len(result) == 4:
            X_train, X_test, _, _ = result
            if isinstance(X_train, (pd.DataFrame, pd.Series)) and isinstance(
                X_test, (pd.DataFrame, pd.Series)
            ):
                return X_train.index, X_test.index
    raise TypeError(
        "Partitioner output is unsupported. Return train/test indices, an "
        "ATTP-style result with train_indices/test_indices, or train/test "
        "pandas objects retaining input row indexes. Provide "
        "partitioner_result_adapter for another return shape."
    )


def _to_positions(values: Any, row_count: int, name: str) -> np.ndarray:
    positions = np.asarray(values)
    if positions.ndim != 1 or not len(positions):
        raise ValueError(f"Partitioner {name} indices must be a non-empty 1D array.")
    if not all(
        isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
        for value in positions.tolist()
    ):
        raise TypeError(
            f"Partitioner {name} indices must be integer row positions."
        )
    positions = positions.astype(int, copy=False)
    if len(np.unique(positions)) != len(positions):
        raise ValueError(f"Partitioner {name} indices must not contain duplicates.")
    if np.any(positions < 0) or np.any(positions >= row_count):
        raise ValueError(f"Partitioner {name} indices are outside the input rows.")
    return positions


def _report_parameter(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    return repr(value)
