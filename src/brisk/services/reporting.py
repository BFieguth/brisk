"""Reporting service handles creation of ReportData object."""
from datetime import datetime
from typing import TYPE_CHECKING, Dict, Tuple, List, Optional, Any
from collections import defaultdict

from scipy import stats

from brisk.services import base
from brisk.evaluation import metric_manager as metric_config
from brisk.reporting import report_data
from brisk.version import __version__

if TYPE_CHECKING:
    from brisk.types import DataManager, DataSplits
    from brisk.evaluation.evaluators.registry import EvaluatorRegistry

class ReportingContext:
    """Context for the reporting service.
    
    Parameters
    ----------
    group_name : str
        The name of the group
    dataset_name : str
        The name of the dataset
    split_index : int
        The index of the split
    feature_names : Optional[List[str]]
        The names of the features
    algorithm_names : Optional[List[str]]
        The names of the algorithms

    Attributes
    ----------
    group_name : str
        The name of the group
    dataset_name : str
        The name of the dataset
    split_index : int
        The index of the split
    feature_names : Optional[List[str]]
        The names of the features
    algorithm_names : Optional[List[str]]
        The names of the algorithms
    """
    def __init__(
        self,
        group_name: str,
        dataset_name: str,
        split_index: int,
        feature_names: Optional[List[str]] = None,
        algorithm_names: Optional[List[str]] = None
    ):
        self.group_name = group_name
        self.dataset_name = dataset_name
        self.split_index = split_index
        self.feature_names = feature_names
        self.algorithm_names = algorithm_names

class ReportingService(base.BaseService):
    """Reporting service handles creation of ReportData object.
    
    Parameters
    ----------
    name : str
        The name of the service
    metric_manager : MetricManager
        The metric manager

    Attributes
    ----------
    name : str
        The name of the service
    metric_manager : MetricManager
        The metric manager
    registry : Optional[EvaluatorRegistry]
        The evaluator registry
    group_to_experiment : Dict[str, List[str]]
        A dictionary mapping group names to experiment IDs
    _current_context : Optional[ReportingContext]
        The current reporting context
    _image_cache : Dict[Tuple[str, str, str], Tuple[str, Dict[str, str]]]
        A dictionary mapping image keys to image data and metadata
    _table_cache : Dict[Tuple[str, str, str, str], Tuple[Dict[str, Any], Dict[str, str]]]
        A dictionary mapping table keys to table data and metadata
    _cached_tuned_params : Dict[str, Any]
        A dictionary mapping tuned parameter names to their values
    test_scores : Dict[str, Dict[str, Dict[str, Dict[str, List[str]]]]]
        A dictionary mapping group names to dataset names to split indices to
        metric names and test scores
    best_score_by_split : Dict[str, Dict[str, Dict[str, Tuple[str, str, str, str]]]]
        A dictionary mapping group names to dataset names to split indices to
        the best score by split
    tuning_metric : Optional[Tuple[str, str]]
        The tuning metric and its display name
    """
    def __init__(
        self,
        name: str,
        metric_manager: metric_config.MetricManager
    ):
        super().__init__(name)
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.navbar = report_data.Navbar(
            brisk_version=f"Version: {__version__}",
            timestamp=f"Created on: {time}"
        )
        self.datasets = {}
        self.experiments = {}
        self.experiment_groups = []
        self.data_managers = {}
        self.metric_manager = metric_manager
        self.registry: Optional["EvaluatorRegistry"] = None
        self.group_to_experiment = defaultdict(list)
        self._current_context: Optional[ReportingContext] = None
        self._image_cache: Dict[
            Tuple[str, str, str], Tuple[str, Dict[str, str]]
        ] = {}
        self._table_cache: Dict[
            Tuple[str, str, str, str], Tuple[Dict[str, Any], Dict[str, str]]
        ] = {}
        self._cached_tuned_params: Dict[str, Any] = {}
        self.test_scores = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: {"columns": [], "rows": []}
                )
            )
        )
        self.best_score_by_split = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(None)
            )
        )
        self.tuning_metric = None

    def set_context(
        self,
        group_name: str,
        dataset_name: str,
        split_index: int,
        feature_names: Optional[List[str]] = None,
        algorithm_names: Optional[str] = None
    ) -> None:
        """Set the current reporting context.

        Parameters
        ----------
        group_name : str
            The name of the experiment group
        dataset_name : str
            The name of the dataset
        split_index : int
            The index of the split
        feature_names : Optional[List[str]]
            The names of the features
        algorithm_names : Optional[List[str]]
            The names of the algorithms

        Returns
        -------
        None
        """
        self._current_context = ReportingContext(
            group_name, dataset_name, split_index, feature_names,
            algorithm_names
        )

    def clear_context(self) -> None:
        """Clear the current reporting context.

        Returns
        -------
        None
        """
        self._current_context = None

    def get_context(self) -> ReportingContext:
        """Get the current reporting context.

        Returns
        -------
        ReportingContext
            The current reporting context

        Raises
        ------
        ValueError
            If no context is set
        """
        if self._current_context:
            return (
                self._current_context.group_name,
                self._current_context.dataset_name,
                self._current_context.split_index,
                self._current_context.feature_names,
                self._current_context.algorithm_names
            )
        raise ValueError("No context set")

    def add_data_manager(
        self,
        group_name: str,
        data_manager: "DataManager"
    ) -> None:
        """Add a DataManager instance to the report.
        
        Parameters
        ----------
        group_name : str
            The name of the experiment group
        data_manager : DataManager
            The DataManager instance

        Returns
        -------
        None
        """
        manager = report_data.DataManager(
            ID=group_name,
            test_size=data_manager.test_size,
            n_splits=data_manager.n_splits,
            split_method=str(data_manager.split_method),
            group_column=str(data_manager.group_column),
            stratified=str(data_manager.stratified),
            random_state=data_manager.random_state
        )
        self.data_managers[group_name] = manager
        self._clear_cache()

    def add_dataset(
        self,
        group_name: str,
        data_splits: "DataSplits"
    ) -> None:
        """Add a dataset to the report.

        Parameters
        ----------
        group_name : str
            The name of the experiment group
        data_splits : DataSplits
            The DataSplits instance

        Returns
        -------
        None
        """
        dataset_name = data_splits._data_splits[0].dataset_name
        dataset_id = f"{group_name}_{dataset_name}"
        split_ids = [
            f"split_{num}" for num in range(data_splits.expected_n_splits)
        ]
        split_sizes = {
            split_ids[i]: {
                "total_obs": len(split.X_train) + len(split.X_test), 
                "features": len(split.features), 
                "train_obs": len(split.X_train), 
                "test_obs": len(split.X_test)
            } for i, split in enumerate(data_splits._data_splits)
        }

        is_categorical = False
        y = data_splits._data_splits[0].y_train
        if y.nunique() / len(y) < 0.05:
            is_categorical = True

        if is_categorical:
            split_target_stats = {
                split_ids[i]: {
                    "proportion": split.y_train.value_counts(
                        normalize=True
                    ).sort_index().to_dict(),
                    "entropy": stats.entropy(
                        split.y_train.value_counts(normalize=True).sort_index(),
                        base=2
                    )
                } for i, split in enumerate(data_splits._data_splits)
            }
        else:
            split_target_stats = {
                split_ids[i]: {
                    "mean": split.y_train.mean(),
                    "std": split.y_train.std(),
                    "min": split.y_train.min(),
                    "max": split.y_train.max()
                } for i, split in enumerate(data_splits._data_splits)
            }

        split_corr_matrices = {}
        for split in split_ids:
            image_key = (
                group_name, dataset_name, split, "brisk_correlation_matrix"
            )
            image, _ = self._image_cache.get(image_key, (None, None))
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
        self._clear_cache()

    def add_experiment(
        self,
        algorithms: Dict
    ) -> None:
        """Add an experiment to the report.

        Parameters
        ----------
        algorithms : Dict
            The algorithms to add to the experiment

        Returns
        -------
        None
        """
        group_name, dataset_name, _, _, algorithm_names = self.get_context()

        experiment_id = (
            f"{'_'.join(algorithm_names)}_{group_name}_{dataset_name}" # pylint: disable=W1405
        )
        hyperparam_grid = {
            key: str(value)
            for key, value in algorithms["model"].hyperparam_grid.items()
        }
        tuned_params = {
            key: str(value)
            for key, value in self._cached_tuned_params.items()
        }
        tables = self._process_table_cache()
        plots = self._process_image_cache()

        self.group_to_experiment[group_name].append(experiment_id)
        algorithm_display_names = [
            self._other_services["utility"].get_algo_wrapper(name).display_name
            for name in algorithm_names
        ]
        self.experiments[experiment_id] = report_data.Experiment(
            ID=experiment_id,
            dataset=f"{group_name}_{dataset_name}",
            algorithm=algorithm_display_names,
            tuned_params=tuned_params,
            hyperparam_grid=hyperparam_grid,
            tables=tables,
            plots=plots
        )
        self._clear_cache()

    def add_experiment_groups(self, groups: List) -> None:
        """Add experiment groups to the report.

        Parameters
        ----------
        groups : List
            The experiment groups to add

        Returns
        -------
        None
        """
        for group in groups:
            datasets = [
                f"{group.name}_{dataset.split('.')[0]}"
                for dataset in group.datasets
            ]

            test_scores = {}
            data_split_scores=defaultdict(list)

            data_manager = self.data_managers[group.name]
            num_splits = int(data_manager.n_splits)
            for dataset in group.datasets:
                for split in range(num_splits):
                    dataset_name = dataset.split(".")[0]
                    run_id = f"{group.name}_{dataset_name}_split_{split}"
                    test_scores[run_id] = report_data.TableData(
                        name=run_id,
                        description=f"Test set performance on {dataset_name} (Split {split})",
                        columns=self.test_scores[group.name][dataset_name][split]["columns"],
                        rows=self.test_scores[group.name][dataset_name][split]["rows"]
                    )
                    data_split_scores[f"{group.name}_{dataset_name}"].append(
                        self.best_score_by_split[group.name][dataset_name][split]
                    )

            experiment_group = report_data.ExperimentGroup(
                name=group.name,
                description=group.description,
                datasets=datasets,
                experiments=set(self.group_to_experiment[group.name]),
                data_split_scores=data_split_scores,
                test_scores=test_scores
            )
            self.experiment_groups.append(experiment_group)

    def _create_feature_distribution(
        self,
        feature_name: str,
        group_name: str,
        dataset_name: str,
        split_id: str
    ) -> Optional[report_data.FeatureDistribution]:
        """Create a FeatureDistribution for a specific feature.

        Parameters
        ----------
        feature_name : str
            The name of the feature
        group_name : str
            The name of the experiment group
        dataset_name : str
            The name of the dataset
        split_id : str
            The ID of the split

        Returns
        -------
        Optional[FeatureDistribution]
            The FeatureDistribution object
        """
        hist_method = f"brisk_histogram_boxplot_{feature_name}"
        pie_method = f"brisk_pie_plot_{feature_name}"

        plot_image, _ = (
            self._image_cache.get(
                (group_name, dataset_name, split_id, hist_method),
                (None, None)
            ) or
            self._image_cache.get(
                (group_name, dataset_name, split_id, pie_method),
                (None, None)
            )
        )

        if not plot_image:
            self._other_services["logging"].logger.warning(
                f"No plot found for feature {feature_name} in {group_name}/{dataset_name}/{split_id}"
            )
            return None

        stats_method = "brisk_continuous_statistics"
        cat_stats_method = "brisk_categorical_statistics"

        continuous_cache = self._table_cache.get(
            (group_name, dataset_name, split_id, stats_method)
        )
        categorical_cache = self._table_cache.get(
            (group_name, dataset_name, split_id, cat_stats_method)
        )
        table_data = None

        if (
            continuous_cache and
            feature_name in list(continuous_cache[0].keys())
        ):
            table_data = continuous_cache[0][feature_name]
        elif (
            categorical_cache and
            feature_name in list(categorical_cache[0].keys())
        ):
            table_data = categorical_cache[0][feature_name]

        if table_data:
            tables = self._create_feature_stats_tables(feature_name, table_data)
        else:
            tables = self._create_placeholder_table(feature_name)

        plot_name = f"{feature_name} Distribution"
        plot = self._create_plot_data(plot_name, plot_image)

        distribution_id = f"{group_name}_{dataset_name}_{split_id}_{feature_name}"
        return report_data.FeatureDistribution(
            ID=distribution_id,
            tables=tables,
            plot=plot
        )

    def get_report_data(self) -> report_data.ReportData:
        """Get the report data.

        Returns
        -------
        ReportData
            The report data object
        """
        return report_data.ReportData(
            navbar=self.navbar,
            datasets=self.datasets,
            experiments=self.experiments,
            experiment_groups=self.experiment_groups,
            data_managers=self.data_managers
        )

    def _create_feature_stats_tables(
        self,
        feature_name: str,
        stats_data: Dict[str, Any]
    ) -> report_data.TableData:
        """Convert stored statistics into TableData format.

        Parameters
        ----------
        feature_name : str
            The name of the feature
        stats_data : Dict[str, Any]
            The statistics data

        Returns
        -------
        TableData
            The TableData object
        """
        train_rows = []
        test_rows = []
        tables = []
        train_data = stats_data["train"]
        test_data = stats_data["test"]
        for stat_name, stat_value in train_data.items():
            train_rows.append([stat_name, str(stat_value)])
        tables.append(report_data.TableData(
            name=f"{feature_name} Statistics",
            description=f"Statistical summary for feature {feature_name} in train set.",
            columns=["Statistic", "Value"],
            rows=train_rows
        ))
        for stat_name, stat_value in test_data.items():
            test_rows.append([stat_name, str(stat_value)])
        tables.append(report_data.TableData(
            name=f"{feature_name} Statistics",
            description=f"Statistical summary for feature {feature_name} in test set.",
            columns=["Statistic", "Value"],
            rows=test_rows
        ))
        return tables

    def _create_placeholder_table(
        self,
        feature_name: str
    ) -> report_data.TableData:
        """Create placeholder table when no statistics are available.

        Parameters
        ----------
        feature_name : str
            The name of the feature

        Returns
        -------
        TableData
            The TableData object
        """
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

    def _create_plot_data(
        self,
        name: str,
        image: str
    ) -> report_data.PlotData:
        """Create a PlotData object.

        Parameters
        ----------
        name : str
            The name of the plot
        image : str
            The image data

        Returns
        -------
        PlotData
            The PlotData object
        """
        return report_data.PlotData(
            name=name,
            description="This is a placeholder description",
            image=image
        )

    def _collect_test_scores(self, metadata, rows):
        """Check for brisk_evaluate_model on test set to record results.

        Parameters
        ----------
        metadata : Dict[str, Any]
            The metadata
        rows : List[Tuple[str, Any]]
            The rows of the table

        Returns
        -------
        None
        """
        if (
            metadata["method"] == "brisk_evaluate_model" and
            metadata["is_test"] == "True"
        ):
            self._collect_best_score(rows, list(metadata["models"].values()))

            group_name, dataset_name, split_index, _, _ = self.get_context()
            columns = ["Algorithm"] + [row[0] for row in rows]
            test_row = (
                list(metadata["models"].values()) + [row[1] for row in rows]
            )

            self.test_scores[group_name][dataset_name][split_index]["columns"] = columns
            self.test_scores[group_name][dataset_name][split_index]["rows"].append(test_row)

    def store_plot_svg(
        self,
        image: str,
        metadata: Dict[str, str]
    ) -> None:
        """Store plot SVG data.

        Parameters
        ----------
        image : str
            The image data
        metadata : Dict[str, str]
            The metadata

        Returns
        -------
        None
        """
        group_name, dataset_name, split, _, _ = self.get_context()
        image_id = (
            group_name, dataset_name, f"split_{split}", metadata["method"]
        )
        self._image_cache[image_id] = (image, metadata)

    def store_table_data(
        self,
        data: Dict[str, Any],
        metadata: Dict[str, str]
    ) -> None:
        """Store table data using current context.

        Parameters
        ----------
        data : Dict[str, Any]
            The data
        metadata : Dict[str, str]
            The metadata

        Returns
        -------
        None
        """
        group_name, dataset_name, split_index, _, _ = self.get_context()
        split_id = f"split_{split_index}"
        table_id = (group_name, dataset_name, split_id, metadata["method"])
        self._table_cache[table_id] = data, metadata

    def _process_table_cache(self) -> List[report_data.TableData]:
        """Process the table cache to create TableData objects.

        Returns
        -------
        List[TableData]
            The list of TableData objects

        Raises
        ------
        RuntimeError
            If the evaluator registry is not set
        """
        if self.registry is None:
            raise RuntimeError(
                "Evaluator registry not set. " 
                "Call set_evaluator_registry() first."
            )

        tables = []
        for context, (data, metadata) in self._table_cache.items():
            evaluator_name = context[3]
            evaluator = self.registry.get(evaluator_name)

            data_type = self._get_data_type(metadata["is_test"])
            description = evaluator.description + f"({data_type})"
            columns, rows = evaluator.report(data)

            table = report_data.TableData(
                name=evaluator.method_name,
                description=description,
                columns=columns,
                rows=rows
            )
            tables.append(table)
            self._collect_test_scores(metadata, rows)
        return tables

    def _process_image_cache(self):
        """Process the image cache to create PlotData objects.

        Returns
        -------
        List[PlotData]
            The list of PlotData objects
        
        Raises
        ------
        RuntimeError
            If the evaluator registry is not set
        """
        if self.registry is None:
            raise RuntimeError(
                "Evaluator registry not set. "
                "Call set_evaluator_registry() first."
            )

        plots = []
        for context, (image, metadata) in self._image_cache.items():
            evaluator_name = context[3]
            evaluator = self.registry.get(evaluator_name)

            data_type = self._get_data_type(metadata["is_test"])
            description = evaluator.description + f"({data_type})"

            plot = report_data.PlotData(
                name=evaluator.method_name,
                description=description,
                image=image
            )
            plots.append(plot)
        return plots

    def _clear_cache(self):
        """Clear the caches.

        Returns
        -------
        None
        """
        self._image_cache = {}
        self._table_cache = {}
        self._cached_tuned_params = {}

    def cache_tuned_params(self, tuned_params: Dict[str, Any]) -> None:
        """Cache the tuned parameters from hyperparameter tuning.

        Parameters
        ----------
        tuned_params : Dict[str, Any]
            The tuned parameters

        Returns
        -------
        None
        """
        self._cached_tuned_params = tuned_params

    def set_evaluator_registry(self, registry: "EvaluatorRegistry") -> None:
        """Set the evaluator registry for this reporting service.

        Parameters
        ----------
        registry : EvaluatorRegistry
            The evaluator registry

        Returns
        -------
        None
        """
        self.registry = registry

    def _get_data_type(self, is_test: str) -> str:
        """Get the data type (test or train set).

        Parameters
        ----------
        is_test : str
            Whether the data is from the test set

        Returns
        -------
        str
            The data type
        """
        if is_test == "True":
            return "Test Set"
        elif is_test == "False":
            return "Train Set"
        return "Unknown split type"

    def set_tuning_measure(self, measure: str) -> None:
        """Set the measure used for hyperparameter tuning.

        Parameters
        ----------
        measure : str
            The measure

        Returns
        -------
        None
        """
        name = self.metric_manager._resolve_identifier(measure)
        wrapper = self.metric_manager._metrics_by_name[name]
        self.tuning_metric = (wrapper.abbr, wrapper.display_name)

    def _collect_best_score(
        self,
        rows: List[Tuple[str, Any]],
        model_name: str
    ) -> Tuple[str, str, str, str]:
        """Collect the best score from the rows.

        Parameters
        ----------
        rows : List[Tuple[str, Any]]
            The rows of the table
        model_name : str
            The name of the model

        Returns
        -------
        Tuple[str, str, str, str]
            The best score
        """
        group_name, dataset_name, split_index, _, _ = self.get_context()

        available_measures = [row[0] for row in rows]
        tuning_metric = (
            self.tuning_metric[1]
            if self.tuning_metric[1] in available_measures
            else available_measures[0]
        )

        tuning_score = None
        for row in rows:
            if row[0] == tuning_metric:
                tuning_score = row[1]
                break

        greater_is_better = self.metric_manager.is_higher_better(tuning_metric)

        current_result = (
            f"Split {split_index}", model_name[0], tuning_score, tuning_metric
        )
        best_result = self.best_score_by_split[group_name][dataset_name].get(
            split_index
        )
        if best_result is None:
            self.best_score_by_split[group_name][dataset_name][split_index] = current_result
        elif greater_is_better and tuning_score > best_result[2]:
            self.best_score_by_split[group_name][dataset_name][split_index] = current_result
        elif not greater_is_better and tuning_score < best_result[2]:
            self.best_score_by_split[group_name][dataset_name][split_index] = current_result
