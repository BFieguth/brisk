"""Evaluators that calculate measures of model performance.

This module provides built-in evaluators for calculating various performance
measures and metrics for machine learning models. These evaluators support
both single model evaluation and cross-validation, as well as model comparison
capabilities.
"""
from typing import Dict, List, Any, Tuple
import itertools

import pandas as pd
import numpy as np
from sklearn import base
import sklearn.model_selection as model_select

from brisk.evaluation.evaluators import measure_evaluator

class EvaluateModel(measure_evaluator.MeasureEvaluator):
    """Evaluate a model on the provided measures and save the results.

    This evaluator calculates specified performance measures for a single
    trained model on a given dataset. It supports any metric that is
    configured in the metric configuration manager.

    Parameters
    ----------
    method_name : str
        The name of the evaluator
    description : str
        The description of the evaluator output

    Attributes
    ----------
    method_name : str
        The name of the evaluator
    description : str
        The description of the evaluator output
    services : ServiceBundle or None
        The global services bundle
    metric_config : MetricManager or None
        The metric configuration manager

    Notes
    -----
    This evaluator provides a straightforward way to calculate performance
    measures for a single model. It uses the metric configuration manager
    to retrieve the appropriate metric functions and calculates scores
    for all specified metrics.

    The evaluator supports both classification and regression metrics,
    depending on what is configured in the metric configuration manager.

    Examples
    --------
    Use the model evaluation evaluator:
        >>> from brisk.evaluation.evaluators import registry
        >>> evaluator = registry.get("brisk_evaluate_model")
        >>> evaluator.evaluate(model, X, y, ["accuracy", "f1_score"], "results")
    """

    def _calculate_measures(
        self,
        predictions: Dict[str, Any],
        y_true: pd.Series,
        metrics: List[str]
    ) -> Dict[str, float]:
        """Calculate the evaluation results for a model.

        Calculates the specified performance measures for the given
        predictions and true values using the configured metric functions.

        Parameters
        ----------
        predictions : Dict[str, Any]
            The predictions of the model (typically a pandas Series)
        y_true : pd.Series
            The true target values
        metrics : List[str]
            A list of metric names to calculate

        Returns
        -------
        Dict[str, float]
            A dictionary containing the evaluation results for each metric
            with display names as keys and scores as values

        Notes
        -----
        The method retrieves metric functions from the metric configuration
        manager and calculates scores for each specified metric. If a metric
        function is not found, it logs a warning and skips that metric.

        The returned dictionary uses display names as keys for better
        readability in reports and logs.
        """
        results = {}
        for metric_name in metrics:
            display_name = self.metric_config.get_name(metric_name)
            scorer = self.metric_config.get_metric(metric_name)
            if scorer is not None:
                score = scorer(y_true, predictions)
                results[display_name] = score
            else:
                self.services.logger.logger.info(
                    f"Scorer for {metric_name} not found."
                )
        return results

    def _log_results(self, results: Dict[str, float], filename: str) -> None:
        """Override default logging for model evaluation results.

        Provides custom logging format for model evaluation results,
        showing each metric and its score in a readable format.

        Parameters
        ----------
        results : Dict[str, float]
            The results of the evaluation
        filename : str
            The name of the file where results were saved

        Returns
        -------
        None

        Notes
        -----
        The logging format shows each metric name and its score with
        4 decimal places precision for numeric values. Non-numeric
        values are displayed as-is.

        The metadata key is excluded from the logged results.
        """
        scores_log = "\n".join([
            f"{metric}: {score:.4f}"
            if isinstance(score, (int, float))
            else f"{metric}: {score}"
            for metric, score in results.items()
            if metric != "_metadata"
            ]
        )
        output_path = self.services.io.output_dir / f"{filename}.json"
        self.services.logger.logger.info(
            "Model evaluation results:\n%s\nSaved to '%s'.", 
            scores_log, output_path
        )

    def report(
        self,
        results: Dict[str, Any]
    ) -> Tuple[List[str], List[List[Any]]]:
        """Generate a report of the evaluation results.

        Converts evaluation results into a format suitable for reporting
        with metric names and scores in a tabular format.

        Parameters
        ----------
        results : Dict[str, Any]
            The results of the evaluation

        Returns
        -------
        Tuple[List[str], List[List[Any]]]
            A tuple containing:
            - List of column headers: ["Metric", "Score"]
            - Nested list of rows with metric names and scores

        Notes
        -----
        The report format is designed for easy display in tables or
        reports, with one row per metric showing the metric name and
        its corresponding score.

        The metadata key is excluded from the report.
        """
        columns = ["Metric","Score"]
        metrics = [key for key in results.keys() if key != "_metadata"]
        rows = []
        for metric in metrics:
            rows.append([
                metric,
                str(results[metric])
            ])
        return columns, rows

class EvaluateModelCV(measure_evaluator.MeasureEvaluator):
    """Evaluate a model using cross-validation and save the scores.

    This evaluator calculates performance measures for a model using
    cross-validation, providing more robust estimates of model performance
    by averaging scores across multiple train-test splits.

    Parameters
    ----------
    method_name : str
        The name of the evaluator
    description : str
        The description of the evaluator output

    Attributes
    ----------
    method_name : str
        The name of the evaluator
    description : str
        The description of the evaluator output
    services : ServiceBundle or None
        The global services bundle
    metric_config : MetricManager or None
        The metric configuration manager

    Notes
    -----
    Cross-validation provides a more reliable estimate of model performance
    by reducing the variance associated with a single train-test split.
    The evaluator calculates mean scores, standard deviations, and stores
    all individual fold scores for detailed analysis.

    The evaluator uses the utility service to get the appropriate
    cross-validation splitter based on the data characteristics.

    Examples
    --------
    Use the cross-validation evaluator:
        >>> from brisk.evaluation.evaluators import registry
        >>> evaluator = registry.get("brisk_evaluate_model_cv")
        >>> evaluator.evaluate(
        ...     model, X, y, ["accuracy", "f1_score"], "cv_results", cv=5
        ... )
    """

    def evaluate(
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
        y: pd.Series,
        metrics: List[str],
        filename: str,
        cv: int = 5
    ) -> None:
        """Evaluate a model using cross-validation and save the scores.

        Executes the complete cross-validation evaluation workflow.
        This includes calculating scores across multiple folds, computing
        statistics, and saving the results with metadata.

        Parameters
        ----------
        model : base.BaseEstimator
            The model to evaluate
        X : pd.DataFrame
            The input features for evaluation
        y : pd.Series
            The target data
        metrics : List[str]
            A list of metric names to calculate
        filename : str
            The name of the output file (without extension)
        cv : int, optional
            The number of cross-validation folds, by default 5

        Returns
        -------
        None

        Notes
        -----
        The cross-validation process uses the utility service to get the
        appropriate splitter based on the data characteristics (e.g.,
        stratified splits for classification, grouped splits if groups
        are specified).

        Results include mean scores, standard deviations, and all
        individual fold scores for comprehensive analysis.
        """
        results = self._calculate_measures(model, X, y, metrics, cv)
        metadata = self._generate_metadata(model, X.attrs["is_test"])
        self._save_json(results, filename, metadata)
        self._log_results(results, filename)

    def report(
        self,
        results: Dict[str, Any]
    ) -> Tuple[List[str], List[List[Any]]]:
        """Generate a report of the cross-validation results.

        Converts cross-validation results into a format suitable for
        reporting with mean scores, standard deviations, and all scores.

        Parameters
        ----------
        results : Dict[str, Any]
            The results of the cross-validation

        Returns
        -------
        Tuple[List[str], List[List[Any]]]
            A tuple containing:
            - List of column headers: ["Metric", "Mean Score", "All Scores"]
            - Nested list of rows with metric statistics

        Notes
        -----
        The report format shows mean scores with standard deviations in
        parentheses, and all individual fold scores for detailed analysis.

        The metadata key is excluded from the report.
        """
        columns = ["Metric","Mean Score", "All Scores"]
        metrics = [key for key in results.keys() if key != "_metadata"]
        rows = []
        for metric in metrics:
            rows.append([
                metric,
                f"{results[metric]['mean_score']} " # pylint: disable=W1405
                f"({results[metric]['std_dev']})", # pylint: disable=W1405
                str(results[metric]["all_scores"])
            ])
        return columns, rows

    def _calculate_measures(
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
        y: pd.Series,
        metrics: List[str],
        cv: int = 5
    ) -> Dict[str, float]:
        """Calculate the cross-validation results for a model.

        Performs cross-validation evaluation for the specified metrics
        and returns comprehensive statistics including mean, standard
        deviation, and all individual fold scores.

        Parameters
        ----------
        model : base.BaseEstimator
            The model to evaluate
        X : pd.DataFrame
            The input features for evaluation
        y : pd.Series
            The target data
        metrics : List[str]
            A list of metric names to calculate
        cv : int, optional
            The number of cross-validation folds, by default 5

        Returns
        -------
        Dict[str, float]
            A dictionary containing cross-validation results for each metric
            with display names as keys and statistics as values

        Notes
        -----
        The method uses scikit-learn's cross_val_score function with the
        appropriate cross-validation splitter obtained from the utility
        service. The splitter is chosen based on data characteristics
        (e.g., stratified for classification, grouped if groups are specified).

        Each metric result contains:
        - mean_score: Average score across all folds
        - std_dev: Standard deviation of scores
        - all_scores: List of all individual fold scores
        """
        splitter, indices = self.utility.get_cv_splitter(y, cv)
        results = {}
        for metric_name in metrics:
            display_name = self.metric_config.get_name(metric_name)
            scorer = self.metric_config.get_scorer(metric_name)
            if scorer is not None:
                scores = model_select.cross_val_score(
                    model, X, y, scoring=scorer, cv=splitter, groups=indices
                    )
                results[display_name] = {
                    "mean_score": scores.mean(),
                    "std_dev": scores.std(),
                    "all_scores": scores.tolist()
                }
            else:
                self.services.logger.logger.info(
                    f"Scorer for {metric_name} not found."
                )
        return results

    def _log_results(self, results: Dict[str, float], filename: str) -> None:
        """Override default logging for cross-validation results.

        Provides custom logging format for cross-validation results,
        showing mean scores and standard deviations for each metric.

        Parameters
        ----------
        results : Dict[str, float]
            The results of the cross-validation
        filename : str
            The name of the file where results were saved

        Returns
        -------
        None

        Notes
        -----
        The logging format shows each metric with its mean score and
        standard deviation, providing a quick overview of model
        performance variability.

        The metadata key is excluded from the logged results.
        """
        scores_log = "\n".join([
            f"{metric}: mean={res['mean_score']:.4f}, " # pylint: disable=W1405
            f"std_dev={res['std_dev']:.4f}" # pylint: disable=W1405
            for metric, res in results.items()
            if metric != "_metadata"
        ])
        output_path = self.services.io.output_dir / f"{filename}.json"
        self.services.logger.logger.info(
            "Cross-validation results:\n%s\nSaved to '%s'.", 
            scores_log, output_path
        )


class CompareModels(measure_evaluator.MeasureEvaluator):
    """Compare multiple models using specified measures.

    This evaluator allows comparison of multiple models on the same
    dataset using specified performance measures. It can optionally
    calculate differences between model performances for detailed analysis.

    Parameters
    ----------
    method_name : str
        The name of the evaluator
    description : str
        The description of the evaluator output

    Attributes
    ----------
    method_name : str
        The name of the evaluator
    description : str
        The description of the evaluator output
    services : ServiceBundle or None
        The global services bundle
    metric_config : MetricManager or None
        The metric configuration manager

    Notes
    -----
    This evaluator is particularly useful for model selection and
    performance comparison. It can compare any number of models on
    the same dataset using the same metrics, ensuring fair comparison.

    When calculate_diff is True, the evaluator calculates pairwise
    differences between all model pairs for each metric, providing
    detailed performance comparisons.

    Examples
    --------
    Compare multiple models:
        >>> from brisk.evaluation.evaluators import registry
        >>> evaluator = registry.get("brisk_compare_models")
        >>> evaluator.evaluate(model1, model2, model3, X=X, y=y, 
        ...                   metrics=["accuracy", "f1_score"], 
        ...                   filename="comparison", calculate_diff=True)
    """

    def evaluate(
        self,
        *models: base.BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        metrics: List[str],
        filename: str,
        calculate_diff: bool = False,
    ) -> None:
        """Compare multiple models using specified metrics.

        Executes the complete model comparison workflow. This includes
        evaluating each model on the specified metrics and optionally
        calculating pairwise differences between models.

        Parameters
        ----------
        *models : base.BaseEstimator
            Models to compare (variable number of arguments)
        X : pd.DataFrame
            Input features for evaluation
        y : pd.Series
            Target values for evaluation
        metrics : List[str]
            Names of metrics to calculate
        filename : str
            Name for output file (without extension)
        calculate_diff : bool, optional
            Whether to calculate differences between models, by default False

        Returns
        -------
        None

        Notes
        -----
        The method evaluates each model individually on the same dataset
        using the same metrics, ensuring fair comparison. If calculate_diff
        is True, it also calculates pairwise differences between all
        model pairs for each metric.

        Results are saved with metadata for later analysis and reporting.
        """
        results = self._calculate_measures(
            *models, X=X, y=y, metrics=metrics, calculate_diff=calculate_diff
        )
        metadata = self._generate_metadata(list(models), X.attrs["is_test"])
        self._save_json(results, filename, metadata)
        self._log_results(results, filename)

    def _calculate_measures(
        self,
        *models: base.BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        metrics: List[str],
        calculate_diff: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """Calculate the comparison results for multiple models.

        Evaluates each model on the specified metrics and optionally
        calculates pairwise differences between models.

        Parameters
        ----------
        *models : base.BaseEstimator
            Models to compare (variable number of arguments)
        X : pd.DataFrame
            Input features for evaluation
        y : pd.Series
            Target values for evaluation
        metrics : List[str]
            Names of metrics to calculate
        calculate_diff : bool, optional
            Whether to calculate differences between models, by default False

        Returns
        -------
        Dict[str, Dict[str, float]]
            Nested dictionary containing metric scores for each model
            and optionally pairwise differences

        Raises
        ------
        ValueError
            If no models are provided for comparison

        Notes
        -----
        The method evaluates each model individually and stores results
        in a nested dictionary structure. If calculate_diff is True,
        it also calculates pairwise differences between all model pairs
        for each metric.

        The differences are calculated as model_b - model_a for each
        pair, showing the performance improvement (or degradation)
        when switching from model_a to model_b.
        """
        comparison_results = {}

        if not models:
            raise ValueError("At least one model must be provided")

        model_names = []
        for model in models:
            wrapper = self.utility.get_algo_wrapper(model.wrapper_name)
            model_names.append(wrapper.display_name)

        for model_name, model in zip(model_names, models):
            predictions = model.predict(X)
            results = {}

            for metric_name in metrics:
                scorer = self.metric_config.get_metric(metric_name)
                display_name = self.metric_config.get_name(metric_name)
                if scorer is not None:
                    score = scorer(y, predictions)
                    results[display_name] = score
                else:
                    self.services.logger.logger.info(
                        f"Scorer for {metric_name} not found."
                    )

            comparison_results[model_name] = results

        if calculate_diff and len(models) > 1:
            comparison_results["differences"] = {}
            model_pairs = list(itertools.combinations(model_names, 2))

            for metric_name in metrics:
                display_name = self.metric_config.get_name(metric_name)
                comparison_results["differences"][display_name] = {}

                for model_a, model_b in model_pairs:
                    score_a = comparison_results[model_a][display_name]
                    score_b = comparison_results[model_b][display_name]
                    diff = score_b - score_a
                    comparison_results["differences"][display_name][
                        f"{model_b} - {model_a}"
                    ] = diff
        return comparison_results

    def _log_results(self, results: Dict[str, float], filename: str) -> None:
        """Override default logging for model comparison results.

        Provides custom logging format for model comparison results,
        showing each model's performance on each metric.

        Parameters
        ----------
        results : Dict[str, float]
            The results of the model comparison
        filename : str
            The name of the file where results were saved

        Returns
        -------
        None

        Notes
        -----
        The logging format shows each model's performance on each metric,
        making it easy to compare model performance at a glance.

        The differences and metadata keys are excluded from the logged results.
        """
        comparison_log = "\n".join([
            f"{model}: " +
            ", ".join(
                [f"{metric}: {score:.4f}"
                 if isinstance(score, (float, int, np.floating))
                 else f"{metric}: {score}" for metric, score in results.items()
                 if metric != "_metadata"]
                )
            for model, results in results.items()
            if model not in ["differences", "_metadata"]
        ])
        output_path = self.services.io.output_dir / f"{filename}.json"
        self.services.logger.logger.info(
            "Model comparison results:\n%s\nSaved to '%s'.", 
            comparison_log, output_path
        )
