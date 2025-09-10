"""Base class for all evaluators that plot datasets.

This module provides the DatasetPlotEvaluator abstract base class for
evaluators that create plots and visualizations from datasets. It provides
a template for implementing dataset-level plotting methods with standardized
plot generation, metadata handling, and result saving.
"""
import abc
from typing import Dict, Any

import pandas as pd
import matplotlib

# from brisk.theme import theme
from brisk.evaluation.evaluators import base

class DatasetPlotEvaluator(base.BaseEvaluator):
    """Template for evaluators that plot datasets.

    Abstract base class for evaluators that create plots and visualizations
    from datasets. Provides a standardized workflow for dataset plotting
    including plot data generation, plot creation, metadata handling, and
    result saving.

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
    This abstract base class provides a template for implementing dataset-level
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
    Create a custom dataset plot evaluator:
        >>> class CustomDatasetPlotEvaluator(DatasetPlotEvaluator):
        ...     def __init__(self, plot_settings):
        ...         super().__init__("custom", "Custom plot", plot_settings)
        ...     
        ...     def _generate_plot_data(self, train_data, test_data, **kwargs):
        ...         # Custom plot data generation logic
        ...         return plot_data
        ...     
        ...     def _create_plot(self, plot_data, **kwargs):
        ...         # Custom plot creation logic
        ...         return plot
    """

    def __init__(
        self,
        method_name: str,
        description: str,
        plot_settings
    ):
        """Initialize DatasetPlotEvaluator with plot settings.

        Parameters
        ----------
        method_name : str
            The name of the evaluator
        description : str
            The description of the evaluator output
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
        train_data: pd.DataFrame | pd.Series,
        test_data: pd.DataFrame | pd.Series,
        filename: str,
        dataset_name: str,
        group_name: str
    ) -> None:
        """Template for all plot methods to follow.

        Executes the complete plotting workflow for dataset plots.
        This method orchestrates the plotting process by calling the
        abstract methods and handling plot processing.

        Parameters
        ----------
        train_data : pd.DataFrame or pd.Series
            The training data for plotting
        test_data : pd.DataFrame or pd.Series
            The testing data for plotting
        filename : str
            The name of the file to save the plot to (without extension)
        dataset_name : str
            The name of the dataset being plotted
        group_name : str
            The name of the experiment group

        Returns
        -------
        None

        Notes
        -----
        This method provides the standard workflow for dataset plotting:
        1. Generate plot data using _generate_plot_data
        2. Create the plot using _create_plot
        3. Generate metadata for the plot
        4. Save the plot with metadata
        5. Log the results

        The method delegates the actual plot data generation and plot
        creation to the abstract methods, which must be implemented
        by subclasses.
        """
        plot_data = self._generate_plot_data(train_data, test_data)
        plot = self._create_plot(plot_data)
        metadata = self._generate_metadata(dataset_name, group_name)
        self._save_plot(filename, metadata, plot=plot)
        self._log_results(self.method_name, filename)

    @abc.abstractmethod
    def _generate_plot_data(
        self,
        train_data: pd.DataFrame | pd.Series,
        test_data: pd.DataFrame | pd.Series,
        **kwargs
    ) -> Any:
        """MUST implement: Generate data for plotting.

        Abstract method that must be implemented by subclasses to define
        the specific plot data generation logic. This is where the data
        preparation for plotting takes place.

        Parameters
        ----------
        train_data : pd.DataFrame or pd.Series
            The training data for plotting
        test_data : pd.DataFrame or pd.Series
            The testing data for plotting
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

    def _generate_metadata(
        self,
        dataset_name: str,
        group_name: str
    ) -> Dict[str, Any]:
        """Generate metadata for output.

        Generates metadata for the plot using the metadata service.
        This provides standardized metadata for all dataset plot
        evaluations.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset being plotted
        group_name : str
            The name of the experiment group

        Returns
        -------
        Dict[str, Any]
            Dictionary containing metadata about the plot

        Notes
        -----
        The metadata includes information about the evaluator method,
        dataset name, and experiment group, providing context for
        the plot.
        """
        return self.metadata.get_dataset(
            self.method_name, dataset_name, group_name
        )

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
