"""Evaluators to create plots for regression problems.

This module provides built-in plot evaluators specifically designed for
regression problems. These evaluators create visualizations that help
understand and evaluate regression model performance through predicted
vs observed plots and residual analysis.

Classes
-------
PlotPredVsObs
    Evaluator for creating predicted vs observed value scatter plots
PlotResiduals
    Evaluator for creating residual plots with optional trend lines
"""
from typing import Tuple

import pandas as pd
import numpy as np
import plotnine as pn
from sklearn import base

from brisk.evaluation.evaluators import plot_evaluator
from brisk.configuration import algorithm_wrapper

class PlotPredVsObs(plot_evaluator.PlotEvaluator):
    """Plot the predicted vs. observed values for a regression model.
    
    This evaluator creates scatter plots comparing predicted values against
    observed values for regression models. The plot includes a diagonal
    reference line (y=x) to help assess model performance, where points
    closer to this line indicate better predictions.
    
    Attributes
    ----------
    name : str
        The name of the evaluator, set to 'pred_vs_obs'
    """

    def __init__(self, method_name: str, description: str, plot_settings):
        """Initialize the PlotPredVsObs evaluator."""
        super().__init__(method_name, description, plot_settings)
        self.name = "pred_vs_obs"

    def plot(
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        filename: str
    ) -> None:
        """Plot the predicted vs. observed values for a regression model.
        
        Creates a scatter plot comparing predicted values against observed
        values with a diagonal reference line. Points closer to the diagonal
        line indicate better model performance.

        Parameters
        ----------
        model : base.BaseEstimator
            The trained regression model to evaluate
        X : pd.DataFrame
            The input features used for prediction
        y : pd.Series
            The true target values to compare against predictions
        filename : str
            The filename to save the plot to

        Returns
        -------
        None
            The plot is saved to the specified filename
        """
        prediction = self._generate_prediction(model, X)
        plot_data, max_range = self._generate_plot_data(prediction, y)
        wrapper = self.utility.get_algo_wrapper(model.wrapper_name)
        plot = self._create_plot(plot_data, wrapper, max_range)
        metadata = self._generate_metadata(model, X.attrs["is_test"])
        self._save_plot(filename, metadata, plot=plot)
        self._log_results("Predicted vs. Observed", filename)

    def _generate_plot_data(
        self,
        prediction: pd.Series,
        y_true: pd.Series,
    ) -> Tuple[pd.DataFrame, float]:
        """Calculate the plot data for the predicted vs. observed values.

        Prepares the data needed for creating predicted vs observed plots
        by combining predictions and true values into a DataFrame and
        calculating the maximum range for consistent axis scaling.

        Parameters
        ----------
        prediction : pd.Series
            The predicted values from the model
        y_true : pd.Series
            The true target values

        Returns
        -------
        Tuple[pd.DataFrame, float]
            A tuple containing:
            - DataFrame with 'Observed' and 'Predicted' columns
            - Maximum range value for consistent axis scaling
        """
        plot_data = pd.DataFrame({
            "Observed": y_true,
            "Predicted": prediction
        })
        max_range = plot_data[["Observed", "Predicted"]].max().max()
        return plot_data, max_range

    def _create_plot(
        self,
        plot_data: pd.DataFrame,
        wrapper: algorithm_wrapper.AlgorithmWrapper,
        max_range: float
    ) -> pn.ggplot:
        """Create a plot of the predicted vs. observed values.
        
        Generates a plotnine-based scatter plot with predicted values on
        the y-axis and observed values on the x-axis, including a diagonal
        reference line for performance assessment.

        Parameters
        ----------
        plot_data : pd.DataFrame
            DataFrame containing 'Observed' and 'Predicted' columns
        wrapper : algorithm_wrapper.AlgorithmWrapper
            The algorithm wrapper containing model metadata
        max_range : float
            Maximum value for consistent axis scaling

        Returns
        -------
        pn.ggplot
            The generated predicted vs observed plot object
        """
        plot = (
            pn.ggplot(plot_data, pn.aes(x="Observed", y="Predicted")) +
            pn.geom_point(
                color="black", size=3, stroke=0.25, fill=self.primary_color
            ) +
            pn.geom_abline(
                slope=1, intercept=0, color=self.accent_color,
                linetype="dashed"
            ) +
            pn.labs(
                x="Observed Values",
                y="Predicted Values",
                title=f"Predicted vs. Observed Values ({wrapper.display_name})"
            ) +
            pn.coord_fixed(
                xlim=[0, max_range],
                ylim=[0, max_range]
            ) +
            self.theme
        )
        return plot


class PlotResiduals(plot_evaluator.PlotEvaluator):
    """Plot the residuals of a regression model.
    
    This evaluator creates residual plots showing the difference between
    observed and predicted values. Residual plots help identify patterns
    in model errors, such as heteroscedasticity or non-linear relationships
    that the model failed to capture.
    
    Attributes
    ----------
    name : str
        The name of the evaluator, set to 'residuals'
    """

    def __init__(self, method_name: str, description: str, plot_settings):
        """Initialize the PlotResiduals evaluator."""
        super().__init__(method_name, description, plot_settings)
        self.name = "residuals"

    def plot(
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        filename: str,
        add_fit_line: bool = False
    ) -> None:
        """Plot the residuals of a regression model.
        
        Creates a scatter plot of residuals (observed - predicted) against
        observed values. The plot includes a horizontal reference line at
        y=0 and optionally a trend line to identify patterns in residuals.

        Parameters
        ----------
        model : base.BaseEstimator
            The trained regression model to evaluate
        X : pd.DataFrame
            The input features used for prediction
        y : pd.Series
            The true target values
        filename : str
            The filename to save the plot to
        add_fit_line : bool, optional
            Whether to add a trend line to the residual plot, by default False

        Returns
        -------
        None
            The plot is saved to the specified filename
        """
        prediction = self._generate_prediction(model, X)
        plot_data = self._generate_plot_data(prediction, y)
        wrapper = self.utility.get_algo_wrapper(model.wrapper_name)
        plot = self._create_plot(plot_data, wrapper, add_fit_line)
        metadata = self._generate_metadata(model, X.attrs["is_test"])
        self._save_plot(filename, metadata, plot=plot)
        self._log_results("Residuals", filename)

    def _generate_plot_data(
        self,
        predictions: pd.Series,
        y: pd.Series,
    ) -> pd.DataFrame:
        """Calculate the residuals (observed - predicted).

        Computes residuals by subtracting predicted values from observed
        values and organizes the data for residual plotting.

        Parameters
        ----------
        predictions : pd.Series
            The predicted values from the model
        y : pd.Series
            The true target values

        Returns
        -------
        pd.DataFrame
            DataFrame containing 'Observed' and 'Residual (Observed -
            Predicted)' columns
        """
        residuals = y - predictions
        plot_data = pd.DataFrame({
            "Observed": y,
            "Residual (Observed - Predicted)": residuals
        })
        return plot_data

    def _create_plot(
        self,
        plot_data: pd.DataFrame,
        wrapper: algorithm_wrapper.AlgorithmWrapper,
        add_fit_line: bool
    ) -> pn.ggplot:
        """Create a residual plot with optional trend line.

        Generates a plotnine-based scatter plot of residuals against observed
        values, including a horizontal reference line at y=0 and optionally
        a trend line to identify patterns in the residuals.

        Parameters
        ----------
        plot_data : pd.DataFrame
            DataFrame containing 'Observed' and 'Residual (Observed -
            Predicted)' columns
        wrapper : algorithm_wrapper.AlgorithmWrapper
            The algorithm wrapper containing model metadata
        add_fit_line : bool
            Whether to add a trend line to the plot

        Returns
        -------
        pn.ggplot
            The generated residual plot object
        """
        plot = (
            pn.ggplot(plot_data, pn.aes(
                x="Observed",
                y="Residual (Observed - Predicted)"
            )) +
            pn.geom_point(
                color="black", size=3, stroke=0.25, fill=self.primary_color
            ) +
            pn.geom_abline(
                slope=0, intercept=0, color=self.accent_color,
                linetype="dashed", size=1.5
            ) +
            pn.ggtitle(f"Residuals ({wrapper.display_name})") +
            self.theme
        )

        if add_fit_line:
            fit = np.polyfit(
                plot_data["Observed"],
                plot_data["Residual (Observed - Predicted)"],
                1
            )
            fit_line = np.polyval(fit, plot_data["Observed"])
            plot += (
                pn.geom_line(
                    pn.aes(x="Observed", y=fit_line, group=1),
                    color=self.secondary_color, size=1
                )
            )
        return plot
