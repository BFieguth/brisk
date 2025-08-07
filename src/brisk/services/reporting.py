"""Reporting service handles creation of ReportData object."""
from datetime import datetime
from typing import TYPE_CHECKING

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
        dataset_id = f"{group_name}_{data_splits._data_splits[0].dataset_name}"
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

        self.datasets[dataset_id] = report_data.Dataset(
            ID=dataset_id,
            splits=split_ids,
            split_sizes=split_sizes,
            split_target_stats=split_target_stats,
            split_corr_matrices={}, # NOTE from DataSplitInfo
            data_manager_id=group_name,
            features=data_splits._data_splits[0].features,
            split_feature_distributions={} # NOTE: from DataSplitInfo
        )

    def get_report_data(self) -> report_data.ReportData:
        return report_data.ReportData(
            navbar=self.navbar,
            datasets=self.datasets,
            experiments=self.experiments,
            experiment_groups=self.experiment_groups,
            data_managers=self.data_managers
        )
