"""Reporting service handles creation of ReportData object."""
from datetime import datetime
from typing import TYPE_CHECKING, Dict, Tuple, List, Optional, Any
from pathlib import Path

from brisk.services.base import BaseService
from brisk.reporting import report_data
from brisk.version import __version__

if TYPE_CHECKING:
    from brisk.types import DataManager, DataSplits

# Report Generation
# =================
# Data Containers (Used to store data from result files)
# - TableData
# - PlotData


# Pre-experiment Running (In DataSplitInfo)
# 2.1 FeatureDistribution <- DataSplitInfo.NEW METHOD
#   - iterate data_manger._splits to collect, build IDs in ReportDataCollector


# Pre-experiment Running (In TrainingManager)
# 2. Dataset
    # - get FeatureDistribution instances as this is being created

# During experiment runs (In TrainingManager)
# 3. Experiment <- Process after each experiment is run using EvaluationManager service


# Post-experiment Runnning (In TrainingManager)
# 4. ExperimentGroup <- Collect values as each experiment is run, 
#                       when all expected data is available create the pydantic model
# 6. ReportData <- collects all data from above

class ReportingContext:
    """Context for the reporting service."""
    def __init__(
        self,
        group_name: str,
        dataset_name: str,
        split_index: int,
        feature_names: Optional[List[str]] = None
    ):
        self.group_name = group_name
        self.dataset_name = dataset_name
        self.split_index = split_index
        self.feature_names = feature_names


class ReportingService(BaseService):
    """Reporting service handles creation of ReportData object."""
    def __init__(self, name: str):
        super().__init__(name)
        self.navbar = report_data.Navbar(
            brisk_version=f"Version: {__version__}",
            timestamp=f"Created on: {
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }"
        )
        self.datasets = {}
        self.experiments = {}
        self.experiment_groups = []
        self.data_managers = {}

        self._current_context: Optional[ReportingContext] = None
        self._pending_images: Dict[Tuple[str, str, str], str] = {}
        self._pending_tables: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}

    def set_context(
        self,
        group_name: str,
        dataset_name: str,
        split_index: int,
        feature_names: Optional[List[str]] = None
    ) -> None:
        self._current_context = ReportingContext(
            group_name, dataset_name, split_index, feature_names
        )

    def clear_context(self) -> None:
        self._current_context = None

    def get_context(self) -> Optional[ReportingContext]:
        if self._current_context:
            return (
                self._current_context.group_name, 
                self._current_context.dataset_name, 
                self._current_context.split_index, 
                self._current_context.feature_names
            )
        raise ValueError("No context set")

    def add_data_manager(
        self,
        group_name: str,
        data_manager: "DataManager"
    ) -> None:
        """Add a DataManager instance to the report."""
        manager = report_data.DataManager(
            ID=group_name,
            test_size=str(data_manager.test_size),
            n_splits=str(data_manager.n_splits),
            split_method=str(data_manager.split_method),
            group_column=str(data_manager.group_column),
            stratified=str(data_manager.stratified),
            random_state=str(data_manager.random_state),
            scale_method=str(data_manager.scale_method)
        )
        self.data_managers[group_name] = manager

    def add_dataset(
        self,
        group_name: str,
        data_splits: "DataSplits"
    ) -> None:
        dataset_name = data_splits._data_splits[0].dataset_name
        dataset_id = f"{group_name}_{dataset_name}"
        split_ids = [
            f"split_{num}" for num in range(data_splits.expected_n_splits)
        ]
        split_sizes = {
            split_ids[i]: {
                "total_obs": f"{len(split.X_train) + len(split.X_test)}", 
                "features": f"{len(split.features)}", 
                "train_obs": f"{len(split.X_train)}", 
                "test_obs": f"{len(split.X_test)}"
            } for i, split in enumerate(data_splits._data_splits)
        }
        split_target_stats = {
            split_ids[i]: {
                "mean": f"{split.y_train.mean()}",
                "std": f"{split.y_train.std()}",
                "min": f"{split.y_train.min()}",
                "max": f"{split.y_train.max()}"
            } for i, split in enumerate(data_splits._data_splits)
        }
        split_corr_matrices = {}
        for split in split_ids:
            image_key = (group_name, dataset_name, split, "brisk_correlation_matrix")
            image = self._pending_images.get(image_key)
            split_corr_matrices[split] = self._create_plot_data(
                f"{group_name}_{dataset_name}_{split}_correlation_matrix",
                image
            )
        split_feature_distributions = {}
        for i, split in enumerate(data_splits._data_splits):
            split_id = f"split_{i}"
            feature_distributions = []
            
            for feature_name in split.features:
                feature_dist = self._create_feature_distribution(
                    feature_name, group_name, dataset_name, split_id
                )
                if feature_dist:
                    feature_distributions.append(feature_dist)
                    
            split_feature_distributions[split_id] = feature_distributions

        self.datasets[dataset_id] = report_data.Dataset(
            ID=dataset_id,
            splits=split_ids,
            split_sizes=split_sizes,
            split_target_stats=split_target_stats,
            split_corr_matrices=split_corr_matrices,
            data_manager_id=group_name,
            features=data_splits._data_splits[0].features,
            split_feature_distributions=split_feature_distributions
        )

    def _create_feature_distribution(
        self, 
        feature_name: str, 
        group_name: str, 
        dataset_name: str, 
        split_id: str
    ) -> Optional[report_data.FeatureDistribution]:
        """Create a FeatureDistribution for a specific feature."""
        
        hist_method = f"brisk_histogram_boxplot_{feature_name}"
        pie_method = f"brisk_pie_plot_{feature_name}"
        
        plot_image = (
            self._pending_images.get((group_name, dataset_name, split_id, hist_method)) or
            self._pending_images.get((group_name, dataset_name, split_id, pie_method))
        )
        
        if not plot_image:
            self._other_services["logging"].logger.warning(
                f"No plot found for feature {feature_name} in {group_name}/{dataset_name}/{split_id}"
            )
            return None
            
        # Look for table data (statistics)
        stats_method = f"brisk_continuous_statistics_{feature_name}"  # or categorical_statistics
        cat_stats_method = f"brisk_categorical_statistics_{feature_name}"
        
        table_data = (
            self._pending_tables.get((group_name, dataset_name, split_id, stats_method)) or
            self._pending_tables.get((group_name, dataset_name, split_id, cat_stats_method))
        )
        
        # Create table from stored data or placeholder
        if table_data:
            table = self._create_table_from_stats(feature_name, table_data)
        else:
            table = self._create_placeholder_table(feature_name)
            
        # Create plot
        plot_name = f"{feature_name} Distribution"
        plot = self._create_plot_data(plot_name, plot_image)
        
        # Create FeatureDistribution
        distribution_id = f"{group_name}_{dataset_name}_{split_id}_{feature_name}"
        return report_data.FeatureDistribution(
            ID=distribution_id,
            table=table,
            plot=plot
        )

    def _create_table_from_stats(self, feature_name: str, stats_data: Dict[str, Any]) -> report_data.TableData:
        """Convert stored statistics into TableData format."""
        # Extract statistics from the stored data
        # This depends on how your evaluators structure the statistics
        rows = []
        for stat_name, stat_value in stats_data.items():
            if stat_name != "_metadata":  # Skip metadata
                rows.append([stat_name, str(stat_value)])
                
        return report_data.TableData(
            name=f"{feature_name} Statistics",
            description=f"Statistical summary for feature {feature_name}",
            columns=["Statistic", "Value"],
            rows=rows
        )

    def _create_placeholder_table(self, feature_name: str) -> report_data.TableData:
        """Create placeholder table when no statistics are available."""
        return report_data.TableData(
            name=f"{feature_name} Statistics",
            description=f"Statistical summary for feature {feature_name}",
            columns=["Statistic", "Value"],
            rows=[
                ["Mean", "N/A"],
                ["Std Dev", "N/A"],
                ["Min", "N/A"],
                ["Max", "N/A"]
            ]
        )

    def get_report_data(self) -> report_data.ReportData:
        return report_data.ReportData(
            navbar=self.navbar,
            datasets=self.datasets,
            experiments=self.experiments,
            experiment_groups=self.experiment_groups,
            data_managers=self.data_managers
        )

    def store_plot_svg(
        self,
        image: str,
        creating_method: str
    ) -> None:
        group_name, dataset_name, split, _ = self.get_context()
        image_id = (group_name, dataset_name, f"split_{split}", creating_method)
        self._pending_images[image_id] = image

    def store_table_data(self, data: Dict[str, Any], creating_method: str) -> None:
        """Store table data using current context."""
        group_name, dataset_name, split_index, _ = self.get_context()
        split_id = f"split_{split_index}"
        table_id = (group_name, dataset_name, split_id, creating_method)
        self._pending_tables[table_id] = data

    def _create_plot_data(
        self,
        name: str,
        # description: str,
        image: str
    ) -> report_data.PlotData:
        return report_data.PlotData(
            name=name,
            description="This is a placeholder description",
            image=image
        )
