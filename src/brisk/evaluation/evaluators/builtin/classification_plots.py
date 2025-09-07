"""Evaluators to generate plots for classification problems.

This module provides built-in plot evaluators specifically designed for
classification problems. These evaluators create visualizations that help
understand and evaluate classification model performance.
"""
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn import base
import sklearn.metrics as sk_metrics
import plotnine as pn

from brisk.evaluation.evaluators import plot_evaluator

class PlotConfusionHeatmap(plot_evaluator.PlotEvaluator):
    """Plot a heatmap of the confusion matrix for a model.

    This evaluator creates a visual heatmap representation of the confusion
    matrix, showing both the count and percentage of predictions for each
    class combination. The heatmap uses color intensity to represent the
    percentage of predictions, making it easy to identify patterns in
    classification performance.

    Parameters
    ----------
    method_name : str
        The name of the evaluator
    description : str
        The description of the evaluator output
    plot_settings : PlotSettings
        The plot settings containing theme and color configuration

    Attributes
    ----------
    method_name : str
        The name of the evaluator
    description : str
        The description of the evaluator output
    theme : Any
        The plot theme for styling plots
    primary_color : str
        Primary color for plots
    secondary_color : str
        Secondary color for plots
    accent_color : str
        Accent color for plots

    Examples
    --------
    Use the confusion matrix heatmap evaluator:
        >>> from brisk.evaluation.evaluators import registry
        >>> evaluator = registry.get("brisk_plot_confusion_heatmap")
        >>> evaluator.plot(model, X, y, "confusion_heatmap")
    """

    def plot(
        self,
        model: Any,
        X: np.ndarray, # pylint: disable=C0103
        y: np.ndarray,
        filename: str
    ) -> None:
        """Plot a heatmap of the confusion matrix for a model.

        Executes the complete plotting workflow for generating a confusion
        matrix heatmap. This includes generating predictions, calculating
        the confusion matrix data, creating the plot, and saving the results.

        Parameters
        ----------
        model : Any
            The trained classification model with a predict method
        X : np.ndarray
            The input features for evaluation
        y : np.ndarray
            The true target labels
        filename : str
            The name of the file to save the plot to (without extension)

        Returns
        -------
        None

        Notes
        -----
        This method overrides the base plot method to provide
        classification-specific plotting workflow. It generates
        predictions using the model and creates a heatmap visualization
        of the confusion matrix.

        The plot is saved with metadata for later analysis and reporting.
        """
        prediction = self._generate_prediction(model, X)
        plot_data = self._generate_plot_data(prediction, y)
        plot = self._create_plot(plot_data, model.display_name)
        metadata = self._generate_metadata(model, X.attrs["is_test"])
        self._save_plot(filename, metadata, plot=plot)
        self._log_results("Confusion Matrix Heatmap", filename)

    def _generate_plot_data(
        self,
        prediction: pd.Series,
        y: np.ndarray,
    ) -> pd.DataFrame:
        """Calculate the plot data for the confusion matrix heatmap.

        Generates the data needed to create a confusion matrix heatmap,
        including both count and percentage information for each cell.

        Parameters
        ----------
        prediction : pd.Series
            The predicted target values from the model
        y : np.ndarray
            The true target values

        Returns
        -------
        pd.DataFrame
            A dataframe containing the confusion matrix heatmap data with
            columns for True Label, Predicted Label, Percentage, and Label

        Notes
        -----
        The method calculates both absolute counts and percentages for
        each cell in the confusion matrix. The percentage is calculated
        as the proportion of total predictions, and the label combines
        both count and percentage for display in the heatmap.

        The data is structured for use with plotnine's geom_tile and
        geom_text functions.
        """
        labels = np.unique(y).tolist()
        cm = sk_metrics.confusion_matrix(y, prediction, labels=labels)
        cm_percent = cm / cm.sum() * 100

        plot_data = []
        for true_index, true_label in enumerate(labels):
            for pred_index, pred_label in enumerate(labels):
                count = cm[true_index, pred_index]
                percentage = cm_percent[true_index, pred_index]
                plot_data.append({
                    "True Label": true_label,
                    "Predicted Label": pred_label,
                    "Percentage": percentage,
                    "Label": f"{int(count)}\n({percentage:.1f}%)"
                })
        plot_data = pd.DataFrame(plot_data)
        return plot_data

    def _create_plot(
        self,
        plot_data: pd.DataFrame,
        display_name: str
    ) -> pn.ggplot:
        """Create a heatmap of the confusion matrix.

        Creates a plotnine ggplot object representing the confusion matrix
        as a heatmap with color-coded cells and text labels.

        Parameters
        ----------
        plot_data : pd.DataFrame
            The confusion matrix data for plotting
        display_name : str
            The name of the model for the plot title

        Returns
        -------
        pn.ggplot
            The plotnine ggplot object representing the confusion matrix heatmap

        Notes
        -----
        The plot uses geom_tile to create the heatmap with color intensity
        based on percentage values, and geom_text to display the count and
        percentage labels in each cell.

        The color gradient goes from white (low percentages) to the primary
        color (high percentages), with a limit of 0-100% for consistent
        color scaling.
        """
        plot = (
            pn.ggplot(plot_data, pn.aes(
                x="Predicted Label",
                y="True Label",
                fill="Percentage"
            )) +
            pn.geom_tile() +
            pn.geom_text(pn.aes(label="Label"), color="black") +
            pn.scale_fill_gradient( # pylint: disable=E1123
                low="white",
                high=self.primary_color,
                name="Percentage (%)",
                limits=(0, 100)
            ) +
            pn.ggtitle(f"Confusion Matrix Heatmap ({display_name})") +
            self.theme
        )
        return plot


class PlotRocCurve(plot_evaluator.PlotEvaluator):
    """Plot a receiver operating characteristic curve with area under the curve.

    This evaluator creates ROC curve plots for binary classification models,
    showing the relationship between true positive rate and false positive
    rate. The plot includes the area under the curve (AUC) score and a
    reference line for random guessing.

    Parameters
    ----------
    method_name : str
        The name of the evaluator
    description : str
        The description of the evaluator output
    plot_settings : PlotSettings
        The plot settings containing theme and color configuration

    Attributes
    ----------
    method_name : str
        The name of the evaluator
    description : str
        The description of the evaluator output
    theme : Any
        The plot theme for styling plots
    primary_color : str
        Primary color for plots
    secondary_color : str
        Secondary color for plots
    accent_color : str
        Accent color for plots

    Notes
    -----
    The ROC curve is a fundamental tool for evaluating binary classification
    performance. It shows the trade-off between sensitivity (true positive
    rate) and specificity (1 - false positive rate) across different
    classification thresholds.

    The AUC score provides a single metric for overall performance, with
    values closer to 1.0 indicating better performance. A score of 0.5
    indicates performance equivalent to random guessing.

    Examples
    --------
    Use the ROC curve evaluator:
        >>> from brisk.evaluation.evaluators import registry
        >>> evaluator = registry.get("brisk_plot_roc_curve")
        >>> evaluator.plot(model, X, y, "roc_curve")
    """

    def plot(
        self,
        model: Any,
        X: np.ndarray, # pylint: disable=C0103
        y: np.ndarray,
        filename: str,
        pos_label: Optional[int] = 1
    ) -> None:
        """
        Plot a receiver operating characteristic curve with area under thecurve.

        Executes the complete plotting workflow for generating a ROC curve.
        This includes calculating the ROC curve data, computing the AUC score,
        creating the plot, and saving the results.

        Parameters
        ----------
        model : Any
            The trained binary classification model
        X : np.ndarray
            The input features for evaluation
        y : np.ndarray
            The true binary labels
        filename : str
            The name of the file to save the plot to (without extension)
        pos_label : int, optional
            The label of the positive class, by default 1

        Returns
        -------
        None

        Notes
        -----
        This method handles different types of binary classification models
        by automatically detecting whether to use predict_proba,
        decision_function, or predict methods for obtaining prediction scores.

        The plot includes both the ROC curve and a reference line for
        random guessing, along with the AUC score annotation.
        """
        plot_data, auc_data, auc = self._generate_plot_data(
            model, X, y, pos_label
        )
        wrapper = self.utility.get_algo_wrapper(model.wrapper_name)
        plot = self._create_plot(plot_data, auc_data, auc, wrapper)
        metadata = self._generate_metadata(model, X.attrs["is_test"])
        self._save_plot(filename, metadata, plot=plot)
        self._log_results("ROC Curve", auc, filename)

    def _generate_plot_data(
        self,
        model: base.BaseEstimator,
        X: np.ndarray, # pylint: disable=C0103
        y: np.ndarray,
        pos_label: Optional[int] = 1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
        """Calculate the plot data for the ROC curve.

        Generates the data needed to create a ROC curve plot, including
        the curve data, AUC calculation data, and the AUC score.

        Parameters
        ----------
        model : base.BaseEstimator
            The trained binary classification model
        X : np.ndarray
            The input features for evaluation
        y : np.ndarray
            The true binary labels
        pos_label : int, optional
            The label of the positive class, by default 1

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, float]
            A tuple containing:
            - ROC curve data (DataFrame with FPR, TPR, Type columns)
            - AUC calculation data (DataFrame for area shading)
            - AUC score (float)

        Notes
        -----
        The method automatically detects the appropriate prediction method:
        - predict_proba: Uses probability of positive class
        - decision_function: Uses decision function scores
        - predict: Uses binary predictions as fallback

        The ROC curve data includes both the actual curve and a reference
        line for random guessing (diagonal line from 0,0 to 1,1).
        """
        if hasattr(model, "predict_proba"):
            # Use probability of the positive class
            y_score = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            # Use decision function score
            y_score = model.decision_function(X)
        else:
            # Use binary predictions as a last resort
            y_score = model.predict(X)
        fpr, tpr, _ = sk_metrics.roc_curve(y, y_score, pos_label=pos_label)
        auc = sk_metrics.roc_auc_score(y, y_score)

        roc_data = pd.DataFrame({
            "False Positive Rate": fpr,
            "True Positive Rate": tpr,
            "Type": "ROC Curve"
        })
        ref_line = pd.DataFrame({
            "False Positive Rate": [0, 1],
            "True Positive Rate": [0, 1],
            "Type": "Random Guessing"
        })
        auc_data = pd.DataFrame({
            "False Positive Rate": np.linspace(0, 1, 500),
            "True Positive Rate": np.interp(
                np.linspace(0, 1, 500), fpr, tpr
            ),
            "Type": "ROC Curve"
        })
        plot_data = pd.concat([roc_data, ref_line])
        return plot_data, auc_data, auc

    def _create_plot(
        self,
        plot_data: pd.DataFrame,
        auc_data: pd.DataFrame,
        auc: float,
        wrapper: Any
    ) -> pn.ggplot:
        """Create a ROC curve plot.

        Creates a plotnine ggplot object representing the ROC curve with
        AUC shading and score annotation.

        Parameters
        ----------
        plot_data : pd.DataFrame
            The ROC curve data for plotting
        auc_data : pd.DataFrame
            The data for AUC area shading
        auc : float
            The AUC score for annotation
        wrapper : Any
            The algorithm wrapper for model display name

        Returns
        -------
        pn.ggplot
            The plotnine ggplot object representing the ROC curve

        Notes
        -----
        The plot includes:
        - The ROC curve line
        - A reference line for random guessing
        - Shaded area under the curve
        - AUC score annotation
        - Fixed aspect ratio for proper visualization

        The plot uses different colors and line types to distinguish
        between the ROC curve and the reference line.
        """
        plot = (
            pn.ggplot(plot_data, pn.aes(
                x="False Positive Rate",
                y="True Positive Rate",
                color="Type",
                linetype="Type"
            )) +
            pn.geom_line(size=1) +
            pn.geom_area(
                data=auc_data,
                fill=self.primary_color,
                alpha=0.2,
                show_legend=False
            ) +
            pn.annotate(
                "text",
                x=0.875,
                y=0.025,
                label=f"AUC = {auc:.2f}",
                color="black",
                size=12
            ) +
            pn.scale_color_manual(
                values=[self.primary_color, self.accent_color],
                na_value="black"
            ) +
            pn.labs(
                title=f"ROC Curve ({wrapper.display_name})",
                color="",
                linetype=""
            ) +
            self.theme +
            pn.coord_fixed(ratio=1)
        )
        return plot

    def _log_results(self, plot_name: str, auc: float, filename: str) -> None:
        """Log the results of the ROC curve to console.

        Displays the ROC curve plot name, AUC score, and file path
        for easy tracking of evaluation results.

        Parameters
        ----------
        plot_name : str
            The name of the plot that was created
        auc : float
            The AUC score calculated for the model
        filename : str
            The name of the file where the plot was saved

        Returns
        -------
        None

        Notes
        -----
        The logging includes the full output path with .svg extension
        and the AUC score for quick performance assessment.
        """
        output_path = self.io.output_dir / f"{filename}.svg"
        self.services.logger.logger.info(
            "%s with AUC = %.2f saved to %s", plot_name, auc, output_path
        )


class PlotPrecisionRecallCurve(plot_evaluator.PlotEvaluator):
    """Plot a precision-recall curve with area under the curve.

    This evaluator creates precision-recall curve plots for binary
    classification models, showing the relationship between precision and recall
    across different classification thresholds. The plot includes the average
    precision (AP) score and a reference line showing the AP score.

    Parameters
    ----------
    method_name : str
        The name of the evaluator
    description : str
        The description of the evaluator output
    plot_settings : PlotSettings
        The plot settings containing theme and color configuration

    Attributes
    ----------
    method_name : str
        The name of the evaluator
    description : str
        The description of the evaluator output
    theme : Any
        The plot theme for styling plots
    primary_color : str
        Primary color for plots
    secondary_color : str
        Secondary color for plots
    accent_color : str
        Accent color for plots

    Notes
    -----
    The precision-recall curve is particularly useful for imbalanced datasets
    where the focus is on the positive class. It shows the trade-off between
    precision and recall across different classification thresholds.

    The average precision (AP) score provides a single metric for overall
    performance, with values closer to 1.0 indicating better performance.
    Unlike AUC, AP is more sensitive to the performance on the positive class.

    Examples
    --------
    Use the precision-recall curve evaluator:
        >>> from brisk.evaluation.evaluators import registry
        >>> evaluator = registry.get("brisk_plot_precision_recall_curve")
        >>> evaluator.plot(model, X, y, "precision_recall_curve")
    """

    def plot(
        self,
        model: base.BaseEstimator,
        X: np.ndarray, # pylint: disable=C0103
        y: np.ndarray,
        filename: str,
        pos_label: Optional[int] = 1
    ) -> None:
        """Plot a precision-recall curve with area under the curve.

        Executes the complete plotting workflow for generating a
        precision-recall curve. This includes calculating the curve data,
        computing the AP score, creating the plot, and saving the results.

        Parameters
        ----------
        model : base.BaseEstimator
            The trained binary classification model
        X : np.ndarray
            The input features for evaluation
        y : np.ndarray
            The true binary labels
        filename : str
            The name of the file to save the plot to (without extension)
        pos_label : int, optional
            The label of the positive class, by default 1

        Returns
        -------
        None

        Notes
        -----
        This method handles different types of binary classification models
        by automatically detecting whether to use predict_proba,
        decision_function, or predict methods for obtaining prediction scores.

        The plot includes both the precision-recall curve and a reference
        line showing the average precision score.
        """
        plot_data, ap_score = self._generate_plot_data(model, X, y, pos_label)
        wrapper = self.utility.get_algo_wrapper(model.wrapper_name)
        plot = self._create_plot(plot_data, wrapper)
        metadata = self._generate_metadata(model, X.attrs["is_test"])
        self._save_plot(filename, metadata, plot=plot)
        self._log_results("Precision-Recall Curve", ap_score, filename)

    def _generate_plot_data(
        self,
        model: base.BaseEstimator,
        X: np.ndarray, # pylint: disable=C0103
        y: np.ndarray,
        pos_label: Optional[int] = 1
    ) -> Tuple[pd.DataFrame, float]:
        """Calculate the plot data for the precision-recall curve.

        Generates the data needed to create a precision-recall curve plot,
        including the curve data and the average precision score.

        Parameters
        ----------
        model : base.BaseEstimator
            The trained binary classification model
        X : np.ndarray
            The input features for evaluation
        y : np.ndarray
            The true binary labels
        pos_label : int, optional
            The label of the positive class, by default 1

        Returns
        -------
        Tuple[pd.DataFrame, float]
            A tuple containing:
            - Precision-recall curve data (DataFrame with Recall, Precision,
            Type columns)
            - Average precision score (float)

        Notes
        -----
        The method automatically detects the appropriate prediction method:
        - predict_proba: Uses probability of positive class
        - decision_function: Uses decision function scores
        - predict: Uses binary predictions as fallback

        The precision-recall curve data includes both the actual curve
        and a reference line showing the average precision score.
        """
        if hasattr(model, "predict_proba"):
            # Use probability of the positive class
            y_score = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            # Use decision function score
            y_score = model.decision_function(X)
        else:
            # Use binary predictions as a last resort
            y_score = model.predict(X)
        precision, recall, _ = sk_metrics.precision_recall_curve(
            y, y_score, pos_label=pos_label
        )
        ap_score = sk_metrics.average_precision_score(
            y, y_score, pos_label=pos_label
        )

        pr_data = pd.DataFrame({
            "Recall": recall,
            "Precision": precision,
            "Type": "PR Curve"
        })
        ap_line = pd.DataFrame({
            "Recall": [0, 1],
            "Precision": [ap_score, ap_score],
            "Type": f"AP Score = {ap_score:.2f}"
        })

        plot_data = pd.concat([pr_data, ap_line])
        return plot_data, ap_score

    def _create_plot(
        self,
        plot_data: pd.DataFrame,
        wrapper: Any
    ) -> pn.ggplot:
        """Create a precision-recall curve plot.

        Creates a plotnine ggplot object representing the precision-recall
        curve with the average precision score reference line.

        Parameters
        ----------
        plot_data : pd.DataFrame
            The precision-recall curve data for plotting
        wrapper : Any
            The algorithm wrapper for model display name

        Returns
        -------
        pn.ggplot
            The plotnine ggplot object representing the precision-recall curve

        Notes
        -----
        The plot includes:
        - The precision-recall curve line
        - A reference line showing the average precision score
        - Different colors and line types to distinguish the elements
        - Fixed aspect ratio for proper visualization

        The plot uses different colors and line types to distinguish
        between the PR curve and the AP score reference line.
        """
        plot = (
            pn.ggplot(plot_data, pn.aes(
                x="Recall",
                y="Precision",
                color="Type",
                linetype="Type"
            )) +
            pn.geom_line(size=1) +
            pn.scale_color_manual(
                values=[self.accent_color, self.primary_color],
                na_value="black"
            ) +
            pn.scale_linetype_manual(
                values=["dashed", "solid"]
            ) +
            pn.labs(
                title=f"Precision-Recall Curve ({wrapper.display_name})",
                color="",
                linetype=""
            ) +
            self.theme +
            pn.coord_fixed(ratio=1)
        )
        return plot

    def _log_results(
        self,
        plot_name: str,
        ap_score: float,
        filename: str
    ) -> None:
        """Log the results of the precision-recall curve to console.
        
        Displays the precision-recall curve plot name, AP score, and file path
        for easy tracking of evaluation results.

        Parameters
        ----------
        plot_name : str
            The name of the plot that was created
        ap_score : float
            The average precision score calculated for the model
        filename : str
            The name of the file where the plot was saved

        Returns
        -------
        None

        Notes
        -----
        The logging includes the full output path with .svg extension
        and the AP score for quick performance assessment.
        """
        output_path = self.io.output_dir / f"{filename}.svg"
        self.services.logger.logger.info(
            "%s with AP Score = %.2f saved to %s", 
            plot_name, ap_score, output_path
        )
