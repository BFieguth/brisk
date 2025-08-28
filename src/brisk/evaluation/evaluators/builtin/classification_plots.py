"""Evaluators to generate plots for classification problems.

Classes:
    PlotConfusionHeatmap
    PlotRocCurve
    PlotPrecisionRecallCurve
"""
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn import base
import sklearn.metrics as sk_metrics
import plotnine as pn

from brisk.evaluation.evaluators import plot_evaluator

class PlotConfusionHeatmap(plot_evaluator.PlotEvaluator):
    """Plot a heatmap of the confusion matrix for a model."""
    def plot(
        self,
        model: Any,
        X: np.ndarray, # pylint: disable=C0103
        y: np.ndarray,
        filename: str
    ) -> None:
        """Plot a heatmap of the confusion matrix for a model.

        Parameters
        ----------
        model (Any): 
            The trained classification model with a `predict` method.

        X (np.ndarray): 
            The input features.

        y (np.ndarray): 
            The target labels.

        filename (str): 
            The path to save the confusion matrix heatmap image.

        Returns
        -------
        None
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

        Parameters
        ----------
        prediction (pd.Series): 
            The predicted target values.
        y (np.ndarray): 
            The true target values.

        Returns
        -------
        pd.DataFrame: 
            A dataframe containing the confusion matrix heatmap data.
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

        Parameters
        ----------
        plot_data : pd.DataFrame
            The data for the plot
        display_name : str
            The name of the model

        Returns
        -------
        pn.ggplot
            The plot object
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
    """Plot a reciever operator curve with area under the curve."""
    def plot(
        self,
        model: Any,
        X: np.ndarray, # pylint: disable=C0103
        y: np.ndarray,
        filename: str,
        pos_label: Optional[int] = 1
    ) -> None:
        """Plot a reciever operator curve with area under the curve.

        Parameters
        ----------
        model (Any): 
            The trained binary classification model.
        X (np.ndarray): 
            The input features.
        y (np.ndarray): 
            The true binary labels.
        filename (str): 
            The path to save the ROC curve image.
        pos_label (Optional[int]): 
            The label of the positive class.

        Returns
        -------
        None
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
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate the plot data for the ROC curve.

        Parameters
        ----------
        model (base.BaseEstimator): 
            The trained binary classification model.
        X (np.ndarray): 
            The input features.
        y (np.ndarray): 
            The true binary labels.
        pos_label (Optional[int]): 
            The label of the positive class.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, float]: 
            A tuple containing the ROC curve data, the AUC data, and the AUC
            score.
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

        Parameters
        ----------
        plot_data : pd.DataFrame
            The data for the plot
        auc_data : pd.DataFrame
            The data for the AUC
        auc : float
            The AUC score
        wrapper : Any
            The wrapper for the model

        Returns
        -------
        pn.ggplot
            The plot object
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
                values=[self.primary_color, self.important_color],
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

        Parameters
        ----------
        plot_name : str
            The name of the plot
        auc : float
            The AUC score
        filename : str
            The name of the file to save the plot to

        Returns
        -------
        None
        """
        output_path = self.io.output_dir / f"{filename}.svg"
        self.services.logger.logger.info(
            "%s with AUC = %.2f saved to %s", plot_name, auc, output_path
        )


class PlotPrecisionRecallCurve(plot_evaluator.PlotEvaluator):
    """Plot a precision-recall curve with area under the curve."""
    def plot(
        self,
        model: base.BaseEstimator,
        X: np.ndarray, # pylint: disable=C0103
        y: np.ndarray,
        filename: str,
        pos_label: Optional[int] = 1
    ) -> None:
        """Plot a precision-recall curve with area under the curve.

        Parameters
        ----------
        model (base.BaseEstimator): 
            The trained binary classification model.
        X (np.ndarray): 
            The input features.
        y (np.ndarray): 
            The true binary labels.
        filename (str): 
            The path to save the precision-recall curve image.
        pos_label (Optional[int]): 
            The label of the positive class.

        Returns
        -------
        None
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
    ) -> pd.DataFrame:
        """Calculate the plot data for the precision-recall curve.

        Parameters
        ----------
        model (base.BaseEstimator): 
            The trained binary classification model.

        X (np.ndarray): 
            The input features.

        y (np.ndarray): 
            The true binary labels.

        pos_label (Optional[int]): 
            The label of the positive class.

        Returns
        -------
        pd.DataFrame: 
            A dataframe containing the precision-recall curve data.
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

        Parameters
        ----------
        plot_data : pd.DataFrame
            The data for the plot
        wrapper : Any
            The wrapper for the model

        Returns
        -------
        pn.ggplot
            The plot object
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
                values=[self.important_color, self.primary_color],
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
        
        Parameters
        ----------
        plot_name : str
            The name of the plot
        ap_score : float
            The AP score
        filename : str
            The name of the file to save the plot to

        Returns
        -------
        None
        """
        output_path = self.io.output_dir / f"{filename}.svg"
        self.services.logger.logger.info(
            "%s with AP Score = %.2f saved to %s", 
            plot_name, ap_score, output_path
        )
