"""Reporting service for creating experiment reports.

This module provides comprehensive reporting functionality for the Brisk
package, including the creation of ReportData objects, experiment tracking,
dataset analysis, and visualization management. It serves as the central hub for
collecting, organizing, and structuring all experiment data into a cohesive
report format.

The ReportingService handles the creation and management of experiment reports,
including dataset information, model performance metrics, hyperparameter tuning
results, and visualization data. It provides caching mechanisms for efficient
data storage and retrieval, and supports both individual experiment tracking
and group-level analysis.

Examples
--------
>>> from brisk.services.reporting import ReportingService, ReportingContext
>>> from brisk.evaluation import metric_manager
>>> 
>>> # Create reporting service
>>> reporting_service = ReportingService("reporting")
>>> reporting_service.set_metric_config(metric_manager)
>>> 
>>> # Set reporting context
>>> reporting_service.set_context(
...     group_name="classification",
...     dataset_name="iris",
...     split_index=0,
...     feature_names=["sepal_length", "sepal_width"],
...     algorithm_names=["RandomForest", "SVM"]
... )
>>> 
>>> # Add data and generate report
>>> reporting_service.add_data_manager(group_name, data_manager)
>>> reporting_service.add_dataset(group_name, data_splits)
>>> report_data = reporting_service.get_report_data()
"""

from datetime import datetime
from typing import TYPE_CHECKING, Dict, Tuple, List, Optional, Any
from collections import defaultdict

from scipy import stats

from brisk.services import base
from brisk.evaluation import metric_manager as metric_manager_module
from brisk.reporting import report_data
from brisk.version import __version__

if TYPE_CHECKING:
    from brisk.types import DataManager, DataSplits
    from brisk.evaluation.evaluators.registry import EvaluatorRegistry

class ReportingContext:
    """Context class for tracking current reporting state and parameters.
    
    This class encapsulates the current reporting context, including information
    about the experiment group, dataset, data split, features, and algorithms
    being processed. It provides a convenient way to pass context information
    throughout the reporting pipeline.
    
    Attributes
    ----------
    group_name : str
        The name of the experiment group being processed
    dataset_name : str
        The name of the dataset being analyzed
    split_index : int
        The index of the current data split (0-based)
    feature_names : Optional[List[str]]
        The names of the features in the dataset
    algorithm_names : Optional[List[str]]
        The names of the algorithms being evaluated
        
    Notes
    -----
    This class is used internally by the ReportingService to maintain context
    state during report generation. It helps ensure that data is properly
    associated with the correct experiment group, dataset, and split.
    
    Examples
    --------
    >>> context = ReportingContext(
    ...     group_name="classification",
    ...     dataset_name="iris",
    ...     split_index=0,
    ...     feature_names=[
    ...         "sepal_length", "sepal_width", "petal_length", "petal_width"
    ...     ],
    ...     algorithm_names=["RandomForest", "SVM", "LogisticRegression"]
    ... )
    >>> print(f"Processing {context.group_name} group")
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
    """Main service class for creating and managing comprehensive experiment
    reports.
    
    This service handles the creation and management of experiment reports,
    including dataset information, model performance metrics, hyperparameter
    tuning results, and visualization data. It provides caching mechanisms
    for efficient data storage and retrieval, and supports both individual
    experiment tracking and group-level analysis.
    
    The service maintains internal caches for images, tables, and tuned
    parameters, and processes this data into structured ReportData objects
    suitable for HTML report generation.
    
    Attributes
    ----------
    navbar : report_data.Navbar
        Navigation bar information including version and timestamp
    datasets : Dict[str, report_data.Dataset]
        Dictionary mapping dataset IDs to Dataset objects
    experiments : Dict[str, report_data.Experiment]
        Dictionary mapping experiment IDs to Experiment objects
    experiment_groups : List[report_data.ExperimentGroup]
        List of experiment group objects
    data_managers : Dict[str, report_data.DataManager]
        Dictionary mapping group names to DataManager objects
    metric_manager : Optional[metric_config.MetricManager]
        The metric manager for performance evaluation
    registry : Optional[EvaluatorRegistry]
        The evaluator registry for processing evaluation results
    group_to_experiment : Dict[str, List[str]]
        Dictionary mapping group names to experiment IDs
    _current_context : Optional[ReportingContext]
        The current reporting context
    _image_cache : Dict[Tuple[str, str, str], Tuple[str, Dict[str, str]]]
        Cache for storing plot images and metadata
    _table_cache : Dict[
        Tuple[str, str, str, str], Tuple[Dict[str, Any], Dict[str, str]]
    ]
        Cache for storing table data and metadata
    _cached_tuned_params : Dict[str, Any]
        Cache for storing hyperparameter tuning results
    test_scores : Dict[str, Dict[str, Dict[str, Dict[str, List[str]]]]]
        Nested dictionary storing test scores by group, dataset, and split
    best_score_by_split : Dict[
        str, Dict[str, Dict[str, Tuple[str, str, str, str]]]
    ]
        Nested dictionary storing best scores by group, dataset, and split
    tuning_metric : Optional[Tuple[str, str]]
        The tuning metric abbreviation and display name
        
    Notes
    -----
    The service uses internal caches to efficiently store and retrieve data
    during report generation. Caches are cleared when new data is added to
    ensure consistency. The service requires both metric manager and evaluator
    registry to be set before processing evaluation results.
    
    Examples
    --------
    >>> from brisk.services.reporting import ReportingService
    >>> from brisk.evaluation import metric_manager
    >>> 
    >>> # Create and configure reporting service
    >>> reporting_service = ReportingService("reporting")
    >>> reporting_service.set_metric_config(metric_manager)
    >>> reporting_service.set_evaluator_registry(registry)
    >>> 
    >>> # Set context and add data
    >>> reporting_service.set_context("classification", "iris", 0)
    >>> reporting_service.add_data_manager("classification", data_manager)
    >>> reporting_service.add_dataset("classification", data_splits)
    >>> 
    >>> # Generate final report
    >>> report_data = reporting_service.get_report_data()
    """
    def __init__(self, name: str) -> None:
        """Initialize the reporting service.
        
        This constructor sets up the reporting service with the specified name
        and initializes all internal data structures, caches, and tracking
        dictionaries. The service starts with empty datasets, experiments,
        and caches that will be populated during report generation.
        
        Parameters
        ----------
        name : str
            The name identifier for this service
            
        Notes
        -----
        The service initializes with a timestamped navbar, empty data
        structures, and nested defaultdicts for efficient data organization.
        The metric manager and evaluator registry are set to None and must be
        configured separately before processing evaluation results.
        """
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
        self.metric_manager = None
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

    def set_metric_config(
        self,
        metric_config: metric_manager_module.MetricManager
    ) -> None:
        """Set the metric manager for this reporting service.

        This method configures the metric manager that will be used for
        performance evaluation and metric resolution throughout the reporting
        process. The metric manager is required for processing evaluation
        results and determining metric properties.

        Parameters
        ----------
        metric_config : metric_config.MetricManager
            The metric manager instance containing metric definitions and
            configuration

        Notes
        -----
        The metric manager is used for resolving metric identifiers, determining
        whether higher values are better for specific metrics, and accessing
        metric display names and abbreviations.

        Examples
        --------
        >>> from brisk.evaluation import metric_manager
        >>> reporting_service = ReportingService("reporting")
        >>> reporting_service.set_metric_config(metric_manager)
        """
        self.metric_manager = metric_config

    def set_context(
        self,
        group_name: str,
        dataset_name: str,
        split_index: int,
        feature_names: Optional[List[str]] = None,
        algorithm_names: Optional[List[str]] = None
    ) -> None:
        """Set the current reporting context.

        This method establishes the current reporting context, which includes
        information about the experiment group, dataset, data split, features,
        and algorithms being processed. The context is used throughout the
        reporting pipeline to ensure data is properly associated and organized.

        Parameters
        ----------
        group_name : str
            The name of the experiment group being processed
        dataset_name : str
            The name of the dataset being analyzed
        split_index : int
            The index of the current data split (0-based)
        feature_names : Optional[List[str]], default=None
            The names of the features in the dataset
        algorithm_names : Optional[List[str]], default=None
            The names of the algorithms being evaluated

        Notes
        -----
        The context is used internally to determine where to store and retrieve
        data from the various caches. It should be set before adding data
        managers, datasets, or experiments to ensure proper data organization.

        Examples
        --------
        >>> reporting_service = ReportingService("reporting")
        >>> reporting_service.set_context(
        ...     group_name="classification",
        ...     dataset_name="iris",
        ...     split_index=0,
        ...     feature_names=["sepal_length", "sepal_width"],
        ...     algorithm_names=["RandomForest", "SVM"]
        ... )
        """
        self._current_context = ReportingContext(
            group_name, dataset_name, split_index, feature_names,
            algorithm_names
        )

    def clear_context(self) -> None:
        """Clear the current reporting context.

        This method removes the current reporting context, effectively resetting
        the context state. This is useful when switching between different
        experiment groups or datasets.

        Notes
        -----
        After clearing the context, a new context must be set using
        `set_context()` before adding new data to the report.

        Examples
        --------
        >>> reporting_service = ReportingService("reporting")
        >>> reporting_service.set_context("group1", "dataset1", 0)
        >>> # Process data for group1/dataset1
        >>> reporting_service.clear_context()
        >>> reporting_service.set_context("group2", "dataset2", 0)
        """
        self._current_context = None

    def get_context(
        self
    ) -> Tuple[str, str, int, Optional[List[str]], Optional[List[str]]]:
        """Get the current reporting context.

        This method retrieves the current reporting context as a tuple
        containing all context information. The context must be set before
        calling this method.

        Returns
        -------
        Tuple[str, str, int, Optional[List[str]], Optional[List[str]]]
            A tuple containing:
            - group_name: The name of the experiment group
            - dataset_name: The name of the dataset
            - split_index: The index of the current split
            - feature_names: The names of the features (or None)
            - algorithm_names: The names of the algorithms (or None)

        Raises
        ------
        ValueError
            If no context is currently set

        Examples
        --------
        >>> report = ReportingService("reporting")
        >>> report.set_context("classification", "iris", 0)
        >>> group, dataset, split, features, algorithms = report.get_context()
        >>> print(f"Processing {group}/{dataset}, split {split}")
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
        
        This method converts a DataManager instance into a
        report_data.DataManager object and stores it in the report. The
        DataManager contains information about data splitting configuration,
        including test size, number of splits, split method, and other data
        management parameters.

        Parameters
        ----------
        group_name : str
            The name of the experiment group this data manager belongs to
        data_manager : DataManager
            The DataManager instance containing data splitting configuration

        Notes
        -----
        This method clears all internal caches after adding the data manager
        to ensure data consistency. The data manager information is used in
        the final report to document the data splitting methodology.

        Examples
        --------
        >>> from brisk.data import DataManager
        >>> reporting_service = ReportingService("reporting")
        >>> data_manager = DataManager(test_size=0.2, n_splits=5)
        >>> reporting_service.add_data_manager("classification", data_manager)
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

        This method processes a DataSplits instance and creates a comprehensive
        dataset report including split information, target statistics,
        correlation matrices, and feature distributions. It analyzes the data to
        determine if it's categorical or continuous and generates appropriate
        statistics.

        Parameters
        ----------
        group_name : str
            The name of the experiment group this dataset belongs to
        data_splits : DataSplits
            The DataSplits instance containing the dataset and split information

        Notes
        -----
        This method performs extensive data analysis including:
        - Split size calculations (total, train, test observations)
        - Target variable statistics (categorical: proportions, entropy;
          continuous: mean, std, min, max)
        - Correlation matrix generation for each split
        - Feature distribution analysis and visualization
        - Categorical vs continuous data detection (based on unique value ratio)

        Examples
        --------
        >>> from brisk.data import DataSplits
        >>> reporting_service = ReportingService("reporting")
        >>> data_splits = DataSplits.load_from_file("iris.csv")
        >>> reporting_service.add_dataset("classification", data_splits)
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
        dataset_name_id = self._get_dataset_name_id(dataset_name)

        experiment_id = (
            f"{'_'.join(algorithm_names)}_{group_name}_{dataset_name_id}" # pylint: disable=W1405
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
            dataset=f"{group_name}_{dataset_name_id}",
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
            test_scores = {}
            data_split_scores=defaultdict(list)

            data_manager = self.data_managers[group.name]
            num_splits = int(data_manager.n_splits)
            datasets = []
            for dataset in group.datasets:
                dataset_name = (dataset[0].split(".")[0], dataset[1])
                dataset_name_id = self._get_dataset_name_id(dataset_name)
                datasets.append(dataset_name_id)
                for split in range(num_splits):
                    run_id = f"{group.name}_{dataset_name_id}_split_{split}"
                    test_scores[run_id] = report_data.TableData(
                        name=run_id,
                        description=f"Test set performance on {dataset_name} (Split {split})",
                        columns=self.test_scores[group.name][dataset_name][split]["columns"],
                        rows=self.test_scores[group.name][dataset_name][split]["rows"]
                    )
                    data_split_scores[f"{group.name}_{dataset_name_id}"].append(
                        self.best_score_by_split[group.name][dataset_name].get(
                            split,
                            (f"Split {split}", None, "Score not found", None)
                        )
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
        methods = [
            f"brisk_histogram_plot_{feature_name}",
            f"brisk_bar_plot_{feature_name}"
        ]

        for method in methods:
            plot_image, _ = self._image_cache.get(
                (group_name, dataset_name, split_id, method), (None, None)
            )
            if plot_image:
                break

        if not plot_image:
            self._other_services["logging"].logger.warning(
                f"No plot found for feature {feature_name} in "
                f"{group_name}/{dataset_name}/{split_id}"
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

        distribution_id = (
            f"{group_name}_{dataset_name}_{split_id}_{feature_name}"
        )
        return report_data.FeatureDistribution(
            ID=distribution_id,
            tables=tables,
            plot=plot
        )

    def get_report_data(self) -> report_data.ReportData:
        """Get the complete report data object.

        This method creates and returns a ReportData object containing all
        the collected experiment data, including datasets, experiments,
        experiment groups, and data managers. This is the final data structure
        used for HTML report generation.

        Returns
        -------
        report_data.ReportData
            The complete report data object containing:
            - navbar: Navigation information with version and timestamp
            - datasets: Dictionary of all processed datasets
            - experiments: Dictionary of all experiments
            - experiment_groups: List of experiment groups
            - data_managers: Dictionary of data managers

        Notes
        -----
        This method should be called after all data has been added to the
        reporting service. The returned ReportData object can be used with
        the ReportRenderer to generate HTML reports.

        Examples
        --------
        >>> reporting_service = ReportingService("reporting")
        >>> # Add all data...
        >>> report_data = reporting_service.get_report_data()
        >>> # Use with ReportRenderer to generate HTML
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
            description=(
                f"Statistical summary for feature {feature_name} in train set."
            ),
            columns=["Statistic", "Value"],
            rows=train_rows
        ))
        for stat_name, stat_value in test_data.items():
            test_rows.append([stat_name, str(stat_value)])
        tables.append(report_data.TableData(
            name=f"{feature_name} Statistics",
            description=(
                f"Statistical summary for feature {feature_name} in test set."
            ),
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
        """Store plot SVG data in the image cache.

        This method stores SVG plot data along with its metadata in the
        internal image cache. The plot is associated with the current
        reporting context (group, dataset, split) and the method name
        from the metadata.

        Parameters
        ----------
        image : str
            The SVG image data as a string
        metadata : Dict[str, str]
            The metadata dictionary containing method information and
            other plot-related metadata

        Notes
        -----
        The plot is stored using a key composed of the current context
        (group_name, dataset_name, split_id) and the method name from
        metadata. This ensures plots are properly organized and can be
        retrieved during report generation.

        Examples
        --------
        >>> reporting_service = ReportingService("reporting")
        >>> reporting_service.set_context("classification", "iris", 0)
        >>> svg_data = "<svg>...</svg>"
        >>> metadata = {"method": "brisk_correlation_matrix", "type": "plot"}
        >>> reporting_service.store_plot_svg(svg_data, metadata)
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
        """Store table data in the table cache using current context.

        This method stores table data along with its metadata in the
        internal table cache. The table is associated with the current
        reporting context (group, dataset, split) and the method name
        from the metadata.

        Parameters
        ----------
        data : Dict[str, Any]
            The table data dictionary containing the actual data to be
            displayed in the report
        metadata : Dict[str, str]
            The metadata dictionary containing method information and
            other table-related metadata

        Notes
        -----
        The table is stored using a key composed of the current context
        (group_name, dataset_name, split_id) and the method name from
        metadata. This ensures tables are properly organized and can be
        retrieved during report generation.

        Examples
        --------
        >>> reporting_service = ReportingService("reporting")
        >>> reporting_service.set_context("classification", "iris", 0)
        >>> table_data = {"accuracy": 0.95, "precision": 0.92, "recall": 0.88}
        >>> metadata = {"method": "brisk_evaluate_model", "is_test": "True"}
        >>> reporting_service.store_table_data(table_data, metadata)
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

        This method configures the evaluator registry that will be used for
        processing evaluation results and generating report data. The registry
        is required for converting cached data into TableData and PlotData
        objects during report generation.

        Parameters
        ----------
        registry : EvaluatorRegistry
            The evaluator registry instance containing evaluator definitions
            and methods for processing evaluation results

        Notes
        -----
        The evaluator registry is used to:
        - Process cached table data into TableData objects
        - Process cached image data into PlotData objects
        - Generate appropriate descriptions and metadata for reports
        - Handle different types of evaluators (measures, plots, etc.)

        Examples
        --------
        >>> from brisk.evaluation.evaluators.registry import EvaluatorRegistry
        >>> reporting_service = ReportingService("reporting")
        >>> registry = EvaluatorRegistry()
        >>> reporting_service.set_evaluator_registry(registry)
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
            if self.tuning_metric and
            self.tuning_metric[1] in available_measures
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

    def _get_dataset_name_id(self, dataset_name: tuple) -> str:
        """Get the dataset ID string used in the report from the tuple"""
        dataset_name_id = (
            "_".join(dataset_name)
            if dataset_name[1] is not None
            else dataset_name[0]
        )
        return dataset_name_id
