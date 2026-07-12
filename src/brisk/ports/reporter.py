"""Port interfaces for the reporting system.

Defines structural contracts for result collection and report generation.
"""

import dataclasses
from typing import Any, Protocol

from brisk.ports import metric


@dataclasses.dataclass
class ReportData:
    """Opaque data transfer object between reporter and renderer."""


class ReporterPort(Protocol):
    """Collects evaluation results and produces structured report data."""

    def get_report_data(self) -> ReportData: ...

    def set_context(
        self,
        group_name: str,
        dataset_name: str,
        split_index: int,
        feature_names: list[str] | None,
        algorithm_names: list[str] | None,
    ) -> None: ...

    def clear_context(self) -> None: ...

    def set_metric_config(
        self,
        metric_config: metric.MetricManagerPort,
    ) -> None: ...

    def set_evaluator_registry(self, registry: Any) -> None: ...

    def add_experiment(self, algorithms: Any) -> None: ...

    def add_data_manager(self, group_name: str, data_manager: Any) -> None: ...

    def add_dataset(self, group_name: str, data_splits: Any) -> None: ...

    def add_experiment_groups(
        self,
        experiment_group: dict[str, Any],
    ) -> None: ...

    def store_table_data(
        self,
        data: dict[str, Any],
        metadata: dict[str, Any],
        filename: str,
    ) -> None: ...

    def store_plot_svg(
        self,
        svg_str: str,
        metadata: dict[str, Any],
        filename: str,
    ) -> None: ...
