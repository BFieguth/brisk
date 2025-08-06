"""Evaluators that calculate measures of model performance."""
from typing import Dict, List, Any
import itertools

import pandas as pd
import numpy as np
from sklearn import base
import sklearn.model_selection as model_select

from brisk.evaluation.evaluators.measure_evaluator import MeasureEvaluator

class EvaluateModel(MeasureEvaluator):
    """Evaluate a model on the provided measures and save the results."""
    def evaluate(
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
        y: pd.Series,
        metrics: List[str],
        filename: str
    ) -> None:
        """Evaluate a model on the provided metrics and save the results.

        Parameters
        ----------
        model (BaseEstimator): 
            The trained model to evaluate.
        X (pd.DataFrame): 
            The input features.
        y (pd.Series): 
            The target data.
        metrics (List[str]): 
            A list of metrics to calculate.
        filename (str): 
            The name of the output file without extension.
        """
        return super().evaluate(model, X, y, metrics, filename)

    def _calculate_measures(
        self,
        predictions: Dict[str, Any],
        y_true: pd.Series,
        metrics: List[str]
    ) -> Dict[str, float]:
        """Calculate the evaluation results for a model.

        Parameters
        ----------
        predictions (Dict[str, Any]): 
            The predictions of the model.
        y_true (pd.Series): 
            The true target values.
        metrics (List[str]): 
            A list of metrics to calculate.

        Returns:
        -------
        Dict[str, float]: 
            A dictionary containing the evaluation results for each metric.
        """
        results = {}
        for metric_name in metrics:
            display_name = self.metric_config.get_name(metric_name)
            scorer = self.metric_config.get_metric(metric_name)
            if scorer is not None:
                score = scorer(y_true, predictions)
                results[display_name] = score
            else:
                self.services.logger.logger.info(f"Scorer for {metric_name} not found.")
        return results

    def _log_results(self, results: Dict[str, float], filename: str):
        """Overrides default logging."""
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


class EvaluateModelCV(MeasureEvaluator):
    """Evaluate a model using cross-validation and save the scores."""
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

        Parameters
        ----------
        model (BaseEstimator): 
            The model to evaluate.
        X (pd.DataFrame): 
            The input features.
        y (pd.Series): 
            The target data.
        metrics (List[str]): 
            A list of metrics to calculate.
        filename (str): 
            The name of the output file without extension.
        cv (int): 
            The number of cross-validation folds. Defaults to 5.
        """
        results = self._calculate_measures(model, X, y, metrics, cv)
        metadata = self._generate_metadata(model, X.attrs["is_test"])
        self._save_json(results, filename, metadata)
        self._log_results(results, filename)

    def _calculate_measures(
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
        y: pd.Series,
        metrics: List[str],
        cv: int = 5
    ) -> Dict[str, float]:
        """Calculate the cross-validation results for a model.

        Parameters
        ----------
        model (BaseEstimator): 
            The model to evaluate.
        X (pd.DataFrame): `
            The input features.
        y (pd.Series): 
            The target data.
        metrics (List[str]): 
            A list of metrics to calculate.
        cv (int): 
            The number of cross-validation folds. Defaults to 5.
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
                self.services.logger.logger.info(f"Scorer for {metric_name} not found.")
        return results

    def _log_results(self, results: Dict[str, float], filename: str):
        """Overrides default logging."""
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


class CompareModels(MeasureEvaluator):
    """Compare multiple models using specified measures."""
    def evaluate(
        self,
        *models: base.BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        metrics: List[str],
        filename: str,
        calculate_diff: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """Compare multiple models using specified metrics.

        Parameters
        ----------
        *models : BaseEstimator
            Models to compare
        X : DataFrame
            Input features
        y : Series
            Target values
        metrics : list of str
            Names of metrics to calculate
        filename : str
            Name for output file (without extension)
        calculate_diff : bool, optional
            Whether to calculate differences between models, by default False
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

        Parameters
        ----------
        *models : BaseEstimator
            Models to compare
        X : DataFrame
            Input features
        y : Series
            Target values
        metrics : list of str
            Names of metrics to calculate
        calculate_diff : bool, optional
            Whether to calculate differences between models, by default False

        Returns
        -------
        dict
            Nested dictionary containing metric scores for each model
        """
        comparison_results = {}

        if not models:
            raise ValueError("At least one model must be provided")

        model_names = []
        for model in models:
            wrapper = self.utility.get_algo_wrapper(model.wrapper_name)
            model_names.append(wrapper.display_name)

        # Evaluate the model and collect results
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
                    self.services.logger.logger.info(f"Scorer for {metric_name} not found.")

            comparison_results[model_name] = results

        # Calculate the difference between models for each metric
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

    def _log_results(self, results: Dict[str, float], filename: str):
        """Overrides default logging."""
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
