"""Base class for all evaluators that plot data."""
from abc import abstractmethod
from typing import Dict, Any

from sklearn import base
import pandas as pd

from brisk.evaluation.services.bundle import ServiceBundle
from brisk.theme import theme
from brisk.evaluation.evaluators.base import BaseEvaluator

class PlotEvaluator(BaseEvaluator):
    """Template for evaluators that plot data."""
    def __init__(self, method_name: str):
        super().__init__(method_name)
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
        plot_data = self._generate_plot_data(model, X, y)
        plot = self._create_plot(plot_data)
        metadata = self._generate_metadata(model, X.attrs["is_test"])
        self._save_plot(filename, metadata, plot)
        self._log_results(self.method_name, filename)

    def _save_plot(
        self,
        filename: str,
        metadata: Dict[str, Any],
        **kwargs
    ) -> str:
        """ENFORCED: Save plot with metadata."""
        output_path = self.services.io.output_dir / f"{filename}.png"
        self.io.save_plot(str(output_path), metadata, **kwargs)
        return str(output_path)

    def _generate_prediction(
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame # pylint: disable=C0103
    ) -> pd.Series:
        """Default prediction generation - can be overridden."""
        return model.predict(X)

    @abstractmethod
    def _generate_plot_data(
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs
    ) -> Any:
        """MUST implement: Generate data for plotting."""
        pass
    
    @abstractmethod
    def _create_plot(self, plot_data: Any, **kwargs) -> Any:
        """MUST implement: Create the plot object."""
        pass

    def _log_results(self, plot_name: str, filename: str) -> None:
        """Default logging - can be overridden."""
        output_path = self.io.output_dir / f"{filename}.svg"
        self.services.logger.logger.info(
            f"{plot_name} plot saved to {output_path}!"
        )
