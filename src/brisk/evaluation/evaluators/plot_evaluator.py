"""Base class for all model evaluators that plot data."""
from abc import abstractmethod
from typing import Dict, Any

from sklearn import base
import pandas as pd

from brisk.theme import theme
from brisk.evaluation.evaluators.base import BaseEvaluator

class PlotEvaluator(BaseEvaluator):
    """Template for model evaluators that plot data.
    
    Parameters
    ----------
    method_name : str
        The name of the evaluation method
    description : str
        The description of the evaluation method
    """
    def __init__(self, method_name: str, description: str):
        super().__init__(method_name, description)
        self.theme = theme
        self.primary_color = "#0074D9" # Celtic Blue
        self.secondary_color = "#07004D" # Federal Blue
        self.background_color = "#C4E0F9" # Columbia Blue
        self.accent_color = "#00A878" # Jade
        self.important_color = "#B95F89" # Mulberry

    def plot(
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
        y: pd.Series,
        filename: str
    ) -> None:
        """Template for all plot methods to follow.

        Parameters
        ----------
        model : base.BaseEstimator
            The model to evaluate
        X : pd.DataFrame
            The training data
        y : pd.Series
            The training labels
        filename : str
            The name of the file to save the plot to
        
        Returns
        -------
        None
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
        """ENFORCED: Save plot with metadata.

        Parameters
        ----------
        filename : str
            The name of the file to save the plot to
        metadata : Dict[str, Any]
            The metadata for the plot
        **kwargs

        Returns
        -------
        str
            The path to the saved plot
        """
        output_path = self.services.io.output_dir / f"{filename}.png"
        self.io.save_plot(output_path, metadata, **kwargs)
        return str(output_path)

    def _generate_prediction(
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame # pylint: disable=C0103
    ) -> pd.Series:
        """Default prediction generation - can be overridden.

        Parameters
        ----------
        model : base.BaseEstimator
            The model to evaluate
        X : pd.DataFrame
            The training data

        Returns
        -------
        pd.Series
            The model predictions
        """
        return model.predict(X)

    @abstractmethod
    def _generate_plot_data(
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
        y: pd.Series,
        **kwargs
    ) -> Any:
        """MUST implement: Generate data for plotting.
        
        Parameters
        ----------
        model : base.BaseEstimator
            The model to evaluate
        X : pd.DataFrame
            The training data
        y : pd.Series
            The training labels
        **kwargs

        Returns
        -------
        Any
            The data for _create_plot
        """
        pass

    @abstractmethod
    def _create_plot(self, plot_data: Any, **kwargs) -> Any:
        """MUST implement: Create the plot object.
        
        Parameters
        ----------
        plot_data : Any
            The data for the plot
        **kwargs

        Returns
        -------
        Any
            The plot object
        """
        pass

    def _log_results(self, plot_name: str, filename: str) -> None:
        """Default logging - can be overridden.
        
        Parameters
        ----------
        plot_name : str
            The name of the plot
        filename : str
            The name of the file to save the plot to

        Returns
        -------
        None
        """
        output_path = self.io.output_dir / f"{filename}.svg"
        self.services.logger.logger.info(
            f"{plot_name} plot saved to {output_path}."
        )
