"""Evaluators to create plots for regression problems."""
from typing import Tuple

import pandas as pd
import numpy as np
import plotnine as pn
from sklearn import base

from brisk.evaluation.evaluators import plot_evaluator
from brisk.configuration import algorithm_wrapper

class PlotPredVsObs(plot_evaluator.PlotEvaluator):
    """Plot the predicted vs. observed values for a regression model."""
    def plot(
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        filename: str
    ) -> None:
        """Plot the predicted vs. observed values for a regression model.
        
        Parameters
        ----------
        model (BaseEstimator): 
            The trained model.
        X (pd.DataFrame): 
            The input features.
        y (pd.Series): 
            The true target values.
        filename (str): 
            The name of the output file.

        Returns
        -------
        None
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

        Parameters
        ----------
        prediction (pd.Series): 
            The predicted values.
        y_true (pd.Series): 
            The true target values.

        Returns
        -------
        Tuple[pd.DataFrame, float]: 
            A tuple containing the plot data and the maximum range of the plot.
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
        
        Parameters
        ----------
        plot_data (pd.DataFrame): 
            The plot data.
        wrapper (AlgorithmWrapper): 
            The wrapper for the model.
        max_range (float): 
            The maximum range of the plot.

        Returns
        -------
        pn.ggplot: 
            The plot object.
        """
        plot = (
            pn.ggplot(plot_data, pn.aes(x="Observed", y="Predicted")) +
            pn.geom_point(
                color="black", size=3, stroke=0.25, fill=self.primary_color
            ) +
            pn.geom_abline(
                slope=1, intercept=0, color=self.important_color,
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
            self.theme.brisk_theme()
        )
        return plot


class PlotResiduals(plot_evaluator.PlotEvaluator):
    """Plot the residuals of a regression model."""
    def plot(
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        filename: str,
        add_fit_line: bool = False
    ) -> None:
        """Plot the residuals of a regression model.
        
        Parameters
        ----------
        model (BaseEstimator): 
            The trained model.
        X (pd.DataFrame): 
            The input features.
        y (pd.Series): 
            The true target values.
        filename (str): 
            The name of the output file.
        add_fit_line (bool): 
            Whether to add a line of best fit to the plot.

        Returns
        -------
        None
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

        Parameters
        ----------
        predictions (pd.Series): 
            The predicted values.
        y (pd.Series): 
            The true target values.

        Returns
        -------
        pd.DataFrame: 
            A dataframe containing the observed and residual values.
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
        """Plot the residuals of the model and save the plot.

        Parameters
        ----------
        plot_data (pd.DataFrame): 
            The plot data.
        wrapper (AlgorithmWrapper): 
            The wrapper for the model.
        add_fit_line (bool): 
            Whether to add a line of best fit to the plot.
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
                slope=0, intercept=0, color=self.important_color,
                linetype="dashed", size=1.5
            ) +
            pn.ggtitle(f"Residuals ({wrapper.display_name})") +
            self.theme.brisk_theme()
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
                    color=self.accent_color, size=1
                )
            )
        return plot
