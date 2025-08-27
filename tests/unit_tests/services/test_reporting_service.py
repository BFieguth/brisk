import importlib

import pytest
from unittest import mock

from brisk.services.reporting import ReportingService
from brisk.reporting import report_data
from conftest import get_metric_config

@pytest.fixture
def metric_config():
    return get_metric_config()


@pytest.fixture
def report_service(metric_config) -> ReportingService:
    report_service = ReportingService("reporting", metric_config)
    report_service._other_services["logging"] = mock.MagicMock()
    return report_service


@pytest.fixture
def data_manager(mock_brisk_project):
    data_file = mock_brisk_project / "data.py"
    spec = importlib.util.spec_from_file_location(
        "data", str(data_file)
        )
    data_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_module)
    data_manager = data_module.BASE_DATA_MANAGER
    return data_manager


@pytest.fixture
def data_splits(mock_brisk_project, data_manager, tmp_path):
    return data_manager.split(
        data_path = tmp_path / "datasets/regression.csv",
        categorical_features = [],
        group_name = "test_group",
        filename = "regression",
        table_name = None
    )


@pytest.fixture
def data_splits_categorical(mock_brisk_project, data_manager, tmp_path):
    return data_manager.split(
        data_path = tmp_path / "datasets/classification100.csv",
        categorical_features = ["cat_feature_1", "cat_feature_2"],
        group_name = "test_group",
        filename = "classification100",
        table_name = None
    )


@pytest.fixture
def algo_config(mock_brisk_project, tmp_path):
    module_path = tmp_path / "algorithms.py"
    spec = importlib.util.spec_from_file_location("algorithms", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "ALGORITHM_CONFIG")


class TestReportingService:
    def test_get_context(self, report_service):
        with pytest.raises(ValueError, match="No context set"):
            context = report_service.get_context()
        
        group_name="test_group"
        dataset_name=("regression", None)
        split_index=0.
        feature_names=["feature_1", "feature_2", "feature_3"]
        algorithm_names="ridge"

        report_service.set_context(
            group_name=group_name,
            dataset_name=dataset_name,
            split_index=split_index,
            feature_names=feature_names,
            algorithm_names=algorithm_names
        )

        context = report_service.get_context()
        assert isinstance(context, tuple)
        assert context[0] == group_name
        assert context[1] == dataset_name
        assert context[2] == split_index
        assert context[3] == feature_names
        assert context[4] == algorithm_names

        report_service.clear_context()
        with pytest.raises(ValueError, match="No context set"):
            context = report_service.get_context()

    def test_add_data_manager(self, report_service, data_manager):
        report_service.add_data_manager(
            group_name="test_group",
            data_manager=data_manager
        )
        assert isinstance(
            report_service.data_managers["test_group"],
            report_data.DataManager
        )
        assert report_service.data_managers["test_group"].ID == "test_group"
        assert report_service.data_managers["test_group"].test_size == 0.2
        assert report_service.data_managers["test_group"].n_splits == 2
        assert report_service.data_managers["test_group"].split_method == "shuffle"
        assert report_service.data_managers["test_group"].group_column == "None"
        assert report_service.data_managers["test_group"].stratified == "False"
        assert report_service.data_managers["test_group"].random_state == 42

    @mock.patch("brisk.services.reporting.ReportingService._create_plot_data")
    @mock.patch("brisk.services.reporting.ReportingService._create_feature_distribution")
    def test_add_dataset_continuous(
        self,
        mock_create_feature_distribution,
        mock_create_plot_data,
        report_service,
        data_splits
    ):
        mock_create_plot_data.return_value = report_data.PlotData(
            name="mock_plot_data",
            description="mock a plot data object",
            image="encoded image would go here"
        )
        mock_create_feature_distribution.return_value = report_data.FeatureDistribution(
            ID="mock_feature_distribution",
            tables=[],
            plot=mock_create_plot_data.return_value
        )

        report_service.add_dataset(
            group_name="test_group",
            data_splits=data_splits
        )

        created_dataset = report_service.datasets["test_group_regression"]
        assert isinstance(created_dataset, report_data.Dataset)
        assert created_dataset.splits == ["split_0", "split_1"]
        assert len(created_dataset.split_sizes) == 2
        assert created_dataset.split_sizes["split_0"]["total_obs"] == 5
        assert created_dataset.split_sizes["split_0"]["features"] == 2
        assert created_dataset.split_sizes["split_0"]["train_obs"] == 4
        assert created_dataset.split_sizes["split_0"]["test_obs"] == 1
        assert created_dataset.split_target_stats["split_0"] == {
            "max": 1.0, "mean": 0.6, "min": 0.1, "std": 0.392
        }
        assert created_dataset.split_target_stats["split_1"] == {
            "max": 1.3, "mean": 0.675, "min": 0.1, "std": 0.506
        }
        assert created_dataset.data_manager_id == "test_group"

    @mock.patch("brisk.services.reporting.ReportingService._create_plot_data")
    @mock.patch("brisk.services.reporting.ReportingService._create_feature_distribution")
    def test_add_dataset_categorical(
        self,
        mock_create_feature_distribution,
        mock_create_plot_data,
        report_service,
        data_splits_categorical
    ):
        mock_create_plot_data.return_value = report_data.PlotData(
            name="mock_plot_data",
            description="mock a plot data object",
            image="encoded image would go here"
        )
        mock_create_feature_distribution.return_value = report_data.FeatureDistribution(
            ID="mock_feature_distribution",
            tables=[],
            plot=mock_create_plot_data.return_value
        )

        report_service.add_dataset(
            group_name="test_group",
            data_splits=data_splits_categorical
        )

        created_dataset = report_service.datasets["test_group_classification100"]
        assert isinstance(created_dataset, report_data.Dataset)
        assert created_dataset.splits == ["split_0", "split_1"]
        assert len(created_dataset.split_sizes) == 2
        assert created_dataset.split_sizes["split_0"]["total_obs"] == 100
        assert created_dataset.split_sizes["split_0"]["features"] == 5
        assert created_dataset.split_sizes["split_0"]["train_obs"] == 80
        assert created_dataset.split_sizes["split_0"]["test_obs"] == 20  
        assert created_dataset.split_target_stats["split_0"] == {
            "proportion": {0: 0.5, 1: 0.5}, "entropy": 1
        }
        assert created_dataset.split_target_stats["split_1"] == {
            "proportion": {0: 0.512, 1: 0.487}, "entropy": 1
        }
        assert created_dataset.data_manager_id == "test_group"

    @mock.patch("brisk.services.reporting.ReportingService._process_table_cache")
    @mock.patch("brisk.services.reporting.ReportingService._process_image_cache")
    def test_add_experiment(
        self,
        mock_process_image_cache,
        mock_process_table_cache,
        report_service,
        algo_config
    ):
        # Mock other services and caches
        report_service._other_services["utility"] = mock.MagicMock()
        report_service._other_services["utility"].get_algo_wrapper.return_value = mock.MagicMock()
        report_service._other_services["utility"].get_algo_wrapper.return_value.display_name = "Ridge Regression"
        mock_process_image_cache.return_value = []
        mock_process_table_cache.return_value = []

        algorithms = {
            "model": algo_config["ridge"] 
        }
        report_service.set_context("test_group", ("data", None), 0, None, ["ridge"])
        report_service.add_experiment(algorithms)
        print(report_service.experiments)
        experiment = report_service.experiments[f"ridge_test_group_data"]
        assert isinstance(experiment, report_data.Experiment)
        assert experiment.ID == "ridge_test_group_data"
        assert experiment.dataset == "test_group_data"
        assert experiment.algorithm == ["Ridge Regression"]

    def test_add_experiment_groups(self, report_service):
        # Create mock group objects
        mock_group1 = mock.MagicMock()
        mock_group1.name = "group1"
        mock_group1.description = "Test group 1"
        mock_group1.datasets = [("dataset1.csv", None), ("dataset2.csv", None)]
        
        mock_group2 = mock.MagicMock()
        mock_group2.name = "group2"
        mock_group2.description = "Test group 2"
        mock_group2.datasets = [("dataset3.csv", None)]
        
        mock_data_manager1 = mock.MagicMock()
        mock_data_manager1.n_splits = 2
        mock_data_manager2 = mock.MagicMock()
        mock_data_manager2.n_splits = 3
        
        report_service.data_managers = {
            "group1": mock_data_manager1,
            "group2": mock_data_manager2
        }
        
        report_service.test_scores = {
            "group1": {
                ("dataset1", None): {0: {"columns": ["Metric1"], "rows": [["value1"]]}, 1: {"columns": ["Metric2"], "rows": [["value2"]]}},
                ("dataset2", None): {0: {"columns": ["Metric3"], "rows": [["value3"]]}, 1: {"columns": ["Metric4"], "rows": [["value4"]]}}
            },
            "group2": {
                ("dataset3", None): {0: {"columns": ["Metric5"], "rows": [["value5"]]}, 1: {"columns": ["Metric6"], "rows": [["value6"]]}, 2: {"columns": ["Metric7"], "rows": [["value7"]]}}
            }
        }
        
        report_service.best_score_by_split = {
            "group1": {
                ("dataset1", None): {0: ("Split 0", "model1", "0.85", "accuracy"), 1: ("Split 1", "model2", "0.87", "accuracy")},
                ("dataset2", None): {0: ("Split 0", "model3", "0.90", "f1"), 1: ("Split 1", "model4", "0.92", "f1")}
            },
            "group2": {
                ("dataset3", None): {0: ("Split 0", "model5", "0.88", "precision"), 1: ("Split 1", "model6", "0.89", "precision"), 2: ("Split 2", "model7", "0.91", "precision")}
            }
        }
        
        report_service.group_to_experiment = {
            "group1": ["exp1", "exp2"],
            "group2": ["exp3"]
        }
        
        groups = [mock_group1, mock_group2]
        report_service.add_experiment_groups(groups)
        
        assert len(report_service.experiment_groups) == 2
        
        # Test first group
        group1_report = report_service.experiment_groups[0]
        assert isinstance(group1_report, report_data.ExperimentGroup)
        assert group1_report.name == "group1"
        assert group1_report.description == "Test group 1"
        assert group1_report.datasets == ["dataset1", "dataset2"]
        assert group1_report.experiments.sort() == ["exp1", "exp2"].sort()
        
        # Test second group
        group2_report = report_service.experiment_groups[1]
        assert isinstance(group2_report, report_data.ExperimentGroup)
        assert group2_report.name == "group2"
        assert group2_report.description == "Test group 2"
        assert group2_report.datasets == ["dataset3"]
        assert group2_report.experiments == ["exp3"]

    def test_create_feature_distribution(self, report_service):
        report_service._image_cache = {
            ("test_group", "test_dataset", "split_0", "brisk_histogram_boxplot_feature1"): ("mock_image_data", {"method": "brisk_histogram_boxplot_feature1"})
        }
        
        report_service._table_cache = {
            ("test_group", "test_dataset", "split_0", "brisk_continuous_statistics"): (
                {"feature1": {"train": {"mean": 0.5, "std": 0.2}, "test": {"mean": 0.6, "std": 0.25}}},
                {"method": "brisk_continuous_statistics"}
            )
        }
        
        report_service._other_services["logging"] = mock.MagicMock()
        
        result = report_service._create_feature_distribution(
            "feature1", "test_group", "test_dataset", "split_0"
        )
        
        assert isinstance(result, report_data.FeatureDistribution)
        assert result.ID == "test_group_test_dataset_split_0_feature1"
        assert isinstance(result.plot, report_data.PlotData)
        assert result.plot.name == "feature1 Distribution"
        assert result.plot.image == "mock_image_data"
        assert len(result.tables) == 2
        
        # Test with no image data
        result_no_image = report_service._create_feature_distribution(
            "feature2", "test_group", "test_dataset", "split_0"
        )
        assert result_no_image is None

    def test_get_report_data(self, report_service):
        mock_dataset = report_data.Dataset(
            ID="test_dataset",
            splits=["split_0"],
            split_sizes={"split_0": {"total_obs": 100, "features": 5, "train_obs": 80, "test_obs": 20}},
            split_target_stats={"split_0": {"mean": 0.5, "std": 0.2}},
            split_corr_matrices={},
            data_manager_id="test_group",
            features=["f1", "f2"],
            split_feature_distributions={}
        )
        
        mock_experiment = report_data.Experiment(
            ID="test_experiment",
            dataset="test_dataset",
            algorithm=["Test Algorithm"],
            tuned_params={},
            hyperparam_grid={},
            tables=[],
            plots=[]
        )
        
        mock_data_manager = report_data.DataManager(
            ID="test_group",
            test_size=0.2,
            n_splits=2,
            split_method="shuffle",
            group_column="None",
            stratified="False",
            random_state=42
        )
        
        report_service.datasets = {"test_dataset": mock_dataset}
        report_service.experiments = {"test_experiment": mock_experiment}
        report_service.data_managers = {"test_group": mock_data_manager}
        report_service.experiment_groups = []
        
        report_data_obj = report_service.get_report_data()
        
        assert isinstance(report_data_obj, report_data.ReportData)
        assert isinstance(report_data_obj.navbar, report_data.Navbar)
        assert "test_dataset" in report_data_obj.datasets
        assert "test_experiment" in report_data_obj.experiments
        assert "test_group" in report_data_obj.data_managers
        assert len(report_data_obj.experiment_groups) == 0

    def test_create_feature_stats_table(self, report_service):
        stats_data = {
            "train": {"mean": 0.5, "std": 0.2, "min": 0.0, "max": 1.0},
            "test": {"mean": 0.6, "std": 0.25, "min": 0.1, "max": 0.9}
        }
        
        tables = report_service._create_feature_stats_tables("test_feature", stats_data)
        
        assert len(tables) == 2
        
        # Test train table
        train_table = tables[0]
        assert isinstance(train_table, report_data.TableData)
        assert train_table.name == "test_feature Statistics"
        assert "train set" in train_table.description
        assert train_table.columns == ["Statistic", "Value"]
        assert len(train_table.rows) == 4
        assert ["mean", "0.5"] in train_table.rows
        
        # Test test table
        test_table = tables[1]
        assert isinstance(test_table, report_data.TableData)
        assert test_table.name == "test_feature Statistics"
        assert "test set" in test_table.description
        assert test_table.columns == ["Statistic", "Value"]
        assert len(test_table.rows) == 4
        assert ["mean", "0.6"] in test_table.rows

    def test_create_placeholder_table(self, report_service):
        table = report_service._create_placeholder_table("test_feature")
        
        assert isinstance(table, report_data.TableData)
        assert table.name == "test_feature Statistics"
        assert "test_feature" in table.description
        assert table.columns == ["Statistic", "Value"]
        assert len(table.rows) == 4
        assert ["Mean", "N/A"] in table.rows
        assert ["Std Dev", "N/A"] in table.rows
        assert ["Min", "N/A"] in table.rows
        assert ["Max", "N/A"] in table.rows

    def test_collect_test_scores(self, report_service):
        report_service.set_context("test_group", ("test_dataset", None), 0, None, ["test_algo"])        
        report_service.test_scores = {"test_group": {("test_dataset", None): {0: {"columns": [], "rows": []}}}}
        
        metadata = {
            "method": "brisk_evaluate_model",
            "is_test": "True",
            "models": {"model1": "Test Model"}
        }
        
        rows = [("accuracy", 0.85), ("f1_score", 0.78)]
        
        with mock.patch.object(report_service, '_collect_best_score') as mock_collect:
            report_service._collect_test_scores(metadata, rows)
            
            stored_scores = report_service.test_scores["test_group"][("test_dataset", None)][0]
            assert stored_scores["columns"] == ["Algorithm", "accuracy", "f1_score"]
            assert len(stored_scores["rows"]) == 1
            assert stored_scores["rows"][0] == ["Test Model", 0.85, 0.78]
            
            mock_collect.assert_called_once_with(rows, ["Test Model"])

    def test_store_plot_svg(self, report_service):
        report_service.set_context("test_group", ("test_dataset", None), 0, None, ["test_algo"])
        
        image_data = "mock_svg_data"
        metadata = {"method": "test_plot_method"}
        
        report_service.store_plot_svg(image_data, metadata)
        
        expected_key = ("test_group", ("test_dataset", None), "split_0", "test_plot_method")
        assert expected_key in report_service._image_cache
        assert report_service._image_cache[expected_key] == (image_data, metadata)

    def test_store_table_data(self, report_service):
        report_service.set_context("test_group", ("test_dataset", None), 0, None, ["test_algo"])
        
        table_data = {"column1": [1, 2, 3], "column2": [4, 5, 6]}
        metadata = {"method": "test_table_method"}
        
        report_service.store_table_data(table_data, metadata)
        
        expected_key = ("test_group", ("test_dataset", None), "split_0", "test_table_method")
        assert expected_key in report_service._table_cache
        assert report_service._table_cache[expected_key] == (table_data, metadata)

    def test_process_table_cache(self, report_service):
        mock_registry = mock.MagicMock()
        mock_evaluator = mock.MagicMock()
        mock_evaluator.description = "Test evaluator"
        mock_evaluator.method_name = "test_method"
        mock_evaluator.report.return_value = (["Col1", "Col2"], [["row1", "val1"], ["row2", "val2"]])
        mock_registry.get.return_value = mock_evaluator
        
        report_service.set_evaluator_registry(mock_registry)
        report_service._table_cache = {
            ("group1", "dataset1", "split_0", "test_evaluator"): (
                {"data": "test_data"},
                {"method": "test_evaluator", "is_test": "True"}
            )
        }
        
        with mock.patch.object(report_service, '_collect_test_scores') as mock_collect:
            tables = report_service._process_table_cache()
            
            assert len(tables) == 1
            table = tables[0]
            assert isinstance(table, report_data.TableData)
            assert table.name == "test_method"
            assert "Test evaluator(Test Set)" in table.description
            assert table.columns == ["Col1", "Col2"]
            assert table.rows == [["row1", "val1"], ["row2", "val2"]]
            
            mock_collect.assert_called_once()

    def test_process_image_cache(self, report_service):
        mock_registry = mock.MagicMock()
        mock_evaluator = mock.MagicMock()
        mock_evaluator.description = "Test plot evaluator"
        mock_evaluator.method_name = "test_plot_method"
        mock_registry.get.return_value = mock_evaluator
        
        report_service.set_evaluator_registry(mock_registry)
        
        report_service._image_cache = {
            ("group1", "dataset1", "split_0", "test_plot_evaluator"): (
                "mock_image_data",
                {"method": "test_plot_evaluator", "is_test": "False"}
            )
        }
        
        plots = report_service._process_image_cache()
        
        assert len(plots) == 1
        plot = plots[0]
        assert isinstance(plot, report_data.PlotData)
        assert plot.name == "test_plot_method"
        assert "Test plot evaluator(Train Set)" in plot.description
        assert plot.image == "mock_image_data"

    def test_clear_cache(self, report_service):
        report_service._image_cache = {("key1", "key2", "key3", "key4"): ("image", {})}
        report_service._table_cache = {("key1", "key2", "key3", "key4"): ({}, {})}
        report_service._cached_tuned_params = {"param1": "value1"}
        
        report_service._clear_cache()
        
        assert len(report_service._image_cache) == 0
        assert len(report_service._table_cache) == 0
        assert len(report_service._cached_tuned_params) == 0

    def test_get_data_type(self, report_service):
        assert report_service._get_data_type("True") == "Test Set"
        assert report_service._get_data_type("False") == "Train Set"
        assert report_service._get_data_type("unknown") == "Unknown split type"

    def test_set_tuning_measure(self, report_service):
        mock_wrapper = mock.MagicMock()
        mock_wrapper.abbr = "ACC"
        mock_wrapper.display_name = "Accuracy"
        
        report_service.metric_manager._resolve_identifier = mock.MagicMock(return_value="accuracy")
        report_service.metric_manager._metrics_by_name = {"accuracy": mock_wrapper}
        
        report_service.set_tuning_measure("acc")
        
        assert report_service.tuning_metric == ("ACC", "Accuracy")
        report_service.metric_manager._resolve_identifier.assert_called_once_with("acc")

    def test_collect_best_score(self, report_service):
        report_service.set_context("test_group", ("test_dataset", None), 0, None, ["test_algo"])
        report_service.tuning_metric = ("ACC", "Accuracy")
        
        report_service.metric_manager.is_higher_better = mock.MagicMock(return_value=True)
        
        rows = [("Accuracy", 0.85), ("F1 Score", 0.78)]
        model_name = ["Test Model"]
        
        # Test initial score setting
        result = report_service._collect_best_score(rows, model_name)
        
        stored_result = report_service.best_score_by_split["test_group"][("test_dataset", None)][0]
        assert stored_result == ("Split 0", "Test Model", 0.85, "Accuracy")
        
        # Test score update with better score
        rows_better = [("Accuracy", 0.90), ("F1 Score", 0.82)]
        report_service._collect_best_score(rows_better, model_name)
        
        updated_result = report_service.best_score_by_split["test_group"][("test_dataset", None)][0]
        assert updated_result == ("Split 0", "Test Model", 0.90, "Accuracy")
        
        # Test score update with worse score (should not update)
        rows_worse = [("Accuracy", 0.80), ("F1 Score", 0.75)]
        report_service._collect_best_score(rows_worse, model_name)
        
        final_result = report_service.best_score_by_split["test_group"][("test_dataset", None)][0]
        assert final_result == ("Split 0", "Test Model", 0.90, "Accuracy")  # Should remain the better score
