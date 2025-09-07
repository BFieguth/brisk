"""Base class for all model evaluators that plot data.

This module provides the PlotEvaluator abstract base class for
evaluators that create plots and visualizations from model predictions
and data. It provides a template for implementing model plotting methods
with standardized plot generation, metadata handling, and result saving.
"""
import abc
from typing import Dict, Any

from sklearn import base
import pandas as pd
import matplotlib

from brisk.evaluation.evaluators import base as base_eval

class PlotEvaluator(base_eval.BaseEvaluator):
    """Template for model evaluators that plot data.

    Abstract base class for evaluators that create plots and visualizations
    from model predictions and data. Provides a standardized workflow for
    model plotting including plot data generation, plot creation, metadata
    handling, and result saving.

    Parameters
    ----------
    method_name : str
        The name of the evaluation method
    description : str
        The description of the evaluation method
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
        Primary color for plots (from plot settings)
    secondary_color : str
        Secondary color for plots (from plot settings)
    accent_color : str
        Accent color for plots (from plot settings)
    services : ServiceBundle or None
        The global services bundle (inherited from BaseEvaluator)
    metric_config : MetricManager or None
        The metric configuration manager (inherited from BaseEvaluator)

    Notes
    -----
    This abstract base class provides a template for implementing model
    plotting methods. Subclasses must implement the _generate_plot_data and
    _create_plot methods to define the specific plotting logic.

    The class handles the complete plotting workflow:
    1. Generate plot data using _generate_plot_data method
    2. Create the plot using _create_plot method
    3. Generate metadata for the plot
    4. Save the plot with metadata
    5. Log the results

    The constructor automatically configures matplotlib to use a non-interactive
    backend for thread safety and applies the provided plot settings.

    Examples
    --------
    Create a custom plot evaluator:
        >>> class CustomPlotEvaluator(PlotEvaluator):
        ...     def __init__(self, plot_settings):
        ...         super().__init__("custom", "Custom plot", plot_settings)
        ...     
        ...     def _generate_plot_data(self, model, X, y, **kwargs):
        ...         # Custom plot data generation logic
        ...         return plot_data
        ...     
        ...     def _create_plot(self, plot_data, **kwargs):
        ...         # Custom plot creation logic
        ...         return plot
    """

    def __init__(self, method_name: str, description: str, plot_settings):
        """Initialize PlotEvaluator with plot settings.

        Parameters
        ----------
        method_name : str
            The name of the evaluation method
        description : str
            The description of the evaluation method
        plot_settings : PlotSettings
            The plot settings containing theme and color configuration

        Notes
        -----
        The constructor configures matplotlib to use a non-interactive backend
        for thread safety and applies the provided plot settings for consistent
        styling across all plots.
        """
        super().__init__(method_name, description)
        # Ensure non-interactive backend for thread safety
        matplotlib.use("Agg", force=True)
        self.theme = plot_settings.get_theme()
        colors = plot_settings.get_colors()
        self.primary_color = colors["primary_color"]
        self.secondary_color = colors["secondary_color"]
        self.accent_color = colors["accent_color"]

    def plot(
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
        y: pd.Series,
        filename: str
    ) -> None:
        """Template for all plot methods to follow.

        Executes the complete plotting workflow for model plots.
        This method orchestrates the plotting process by calling the
        abstract methods and handling plot processing.

        Parameters
        ----------
        model : base.BaseEstimator
            The trained model to evaluate
        X : pd.DataFrame
            The input data for plotting
        y : pd.Series
            The true target values
        filename : str
            The name of the file to save the plot to (without extension)

        Returns
        -------
        None

        Notes
        -----
        This method provides the standard workflow for model plotting:
        1. Generate plot data using _generate_plot_data
        2. Create the plot using _create_plot
        3. Generate metadata for the plot
        4. Save the plot with metadata
        5. Log the results

        The method delegates the actual plot data generation and plot
        creation to the abstract methods, which must be implemented
        by subclasses.
        """
        plot_data = self._generate_plot_data(model, X, y)
        plot = self._create_plot(plot_data)
        metadata = self._generate_metadata(model, X.attrs["is_test"])
        self._save_plot(filename, metadata, plot=plot)
        self._log_results(self.method_name, filename)

    def _save_plot(
        self,
        filename: str,
        metadata: Dict[str, Any],
        **kwargs
    ) -> str:
        """Save plot with metadata.

        Saves the plot to a file with associated metadata.
        This method provides standardized plot saving functionality.

        Parameters
        ----------
        filename : str
            The name of the file to save the plot to (without extension)
        metadata : Dict[str, Any]
            The metadata to include with the saved plot
        **kwargs
            Additional keyword arguments passed to the I/O service

        Returns
        -------
        str
            The path to the saved plot file

        Notes
        -----
        The method saves the plot to the output directory specified in
        the services configuration. The filename is used as-is without
        automatic extension addition.
        """
        output_path = self.services.io.output_dir / f"{filename}"
        self.io.save_plot(output_path, metadata, **kwargs)
        return str(output_path)

    def _generate_prediction(
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame # pylint: disable=C0103
    ) -> pd.Series:
        """Default prediction generation - can be overridden.

        Generates predictions using the provided model. This method provides
        default prediction generation that can be overridden by subclasses
        for custom prediction logic.

        Parameters
        ----------
        model : base.BaseEstimator
            The trained model to use for prediction
        X : pd.DataFrame
            The input data for prediction

        Returns
        -------
        pd.Series
            The model predictions

        Notes
        -----
        The default implementation uses the model's predict method.
        Subclasses can override this method to implement custom prediction
        logic, such as using predict_proba for probability predictions
        or applying additional post-processing.
        """
        return model.predict(X)

    @abc.abstractmethod
    def _generate_plot_data(
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
        y: pd.Series,
        **kwargs
    ) -> Any:
        """MUST implement: Generate data for plotting.
        
        Abstract method that must be implemented by subclasses to define
        the specific plot data generation logic. This is where the data
        preparation for plotting takes place.

        Parameters
        ----------
        model : base.BaseEstimator
            The trained model to evaluate
        X : pd.DataFrame
            The input data for plotting
        y : pd.Series
            The true target values
        **kwargs
            Additional keyword arguments for plot data generation

        Returns
        -------
        Any
            The data structure to be used by _create_plot method

        Notes
        -----
        This method must be implemented by all subclasses. It should
        contain the specific logic for preparing data for plotting,
        such as data transformation, aggregation, or filtering.

        The returned data structure should be compatible with the
        _create_plot method implementation.
        """
        pass

    @abc.abstractmethod
    def _create_plot(self, plot_data: Any, **kwargs) -> Any:
        """MUST implement: Create the plot object.
        
        Abstract method that must be implemented by subclasses to define
        the specific plot creation logic. This is where the actual plot
        object is created from the prepared data.

        Parameters
        ----------
        plot_data : Any
            The data structure generated by _generate_plot_data
        **kwargs
            Additional keyword arguments for plot creation

        Returns
        -------
        Any
            The plot object to be saved

        Notes
        -----
        This method must be implemented by all subclasses. It should
        contain the specific logic for creating the plot object from
        the prepared data.

        The returned plot object should be compatible with the
        _save_plot method implementation.
        """
        pass

    def _log_results(self, plot_name: str, filename: str) -> None:
        """Default logging - can be overridden.
        
        Logs the plot creation results in a standardized format.
        This method provides default logging functionality that can be
        overridden by subclasses for custom logging behavior.

        Parameters
        ----------
        plot_name : str
            The name of the plot that was created
        filename : str
            The name of the file where the plot was saved

        Returns
        -------
        None

        Notes
        -----
        The default implementation logs the plot name and file path.
        Subclasses can override this method to provide custom logging
        behavior.

        The logging includes the full output path with .svg extension.
        """
        output_path = self.io.output_dir / f"{filename}.svg"
        self.services.logger.logger.info(
            f"{plot_name} plot saved to {output_path}."
        )
