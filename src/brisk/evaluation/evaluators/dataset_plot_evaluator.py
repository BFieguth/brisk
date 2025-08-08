"""Base class for all evaluators that plot datasets."""
from abc import abstractmethod
from typing import Dict, Any

import pandas as pd

from brisk.theme import theme
from brisk.evaluation.evaluators.base import BaseEvaluator

class DatasetPlotEvaluator(BaseEvaluator):
    """Template for evaluators that plot datasets."""
    def __init__(
        self,
        method_name: str,
        description: str
    ):
        super().__init__(method_name, description)
        self.theme = theme
        self.primary_color = "#0074D9" # Celtic Blue
        self.secondary_color = "#07004D" # Federal Blue
        self.background_color = "#C4E0F9" # Columbia Blue
        self.accent_color = "#00A878" # Jade
        self.important_color = "#B95F89" # Mulberry

    def plot(
        self,
        train_data: pd.DataFrame | pd.Series,
        test_data: pd.DataFrame | pd.Series,
        filename: str,
        dataset_name: str,
        group_name: str
    ) -> None:
        plot_data = self._generate_plot_data(train_data, test_data)
        plot = self._create_plot(plot_data)
        metadata = self._generate_metadata(dataset_name, group_name)
        self._save_plot(filename, metadata, plot)
        self._log_results(self.method_name, filename)

    @abstractmethod
    def _generate_plot_data(
        self,
        train_data: pd.DataFrame | pd.Series,
        test_data: pd.DataFrame | pd.Series,
        **kwargs
    ) -> Any:
        """MUST implement: Generate data for plotting."""
        pass
    
    @abstractmethod
    def _create_plot(self, plot_data: Any, **kwargs) -> Any:
        """MUST implement: Create the plot object."""
        pass

    def _save_plot(
        self,
        filename: str,
        metadata: Dict[str, Any],
        **kwargs
    ) -> str:
        """ENFORCED: Save plot with metadata."""
        output_path = self.services.io.output_dir / f"{filename}.png"
        self.io.save_plot(output_path, metadata, **kwargs)
        return str(output_path)

    def _generate_metadata(
        self,
        dataset_name: str,
        group_name: str
    ) -> Dict[str, Any]:
        """Enforced: generate metadata for output."""
        return self.metadata.get_dataset(
            self.method_name, dataset_name, group_name
        )

    def _log_results(self, plot_name: str, filename: str) -> None:
        """Default logging - can be overridden."""
        output_path = self.io.output_dir / f"{filename}.svg"
        self.services.logger.logger.info(
            f"{plot_name} plot saved to {output_path}."
        )
