"""Unit tests for the HTML reporter adapter.

Adapted from ``tests/unit/services/test_reporting.py``. ``HTMLReporter``
subclasses ``ReportingService`` ("wrap, never rewrite"), so a subset of the
inherited collection behaviour is re-verified through the adapter, alongside
``ReporterPort`` conformance and the ``render_report`` rendering hook.
"""

import pathlib
from collections import namedtuple
from unittest import mock

import pytest

from brisk.adapters.reporting.html_renderer import HTMLReportRenderer
from brisk.adapters.reporting.html_reporter import HTMLReporter
from brisk.ports.reporter import ReporterPort

# pylint: disable=W0212, W0621


@pytest.fixture
def mock_data_manager():
    manager = mock.MagicMock()
    manager.test_size = 0.2
    manager.n_splits = 3
    manager.split_method = "StratifiedKFold"
    manager.group_column = None
    manager.stratified = True
    manager.random_state = 42
    return manager


@pytest.fixture
def mock_utility_service():
    service = mock.MagicMock()

    def get_algo_wrapper(name):
        wrapper = mock.MagicMock()
        wrapper.display_name = f"{name} Display"
        return wrapper

    service.get_algo_wrapper = get_algo_wrapper
    return service


@pytest.fixture
def mock_metric_manager():
    manager = mock.MagicMock()
    manager._resolve_identifier = mock.MagicMock(return_value="accuracy")
    manager._metrics_by_name = {
        "accuracy": mock.MagicMock(abbr="Acc", display_name="Accuracy")
    }
    manager.is_higher_better = mock.MagicMock(return_value=True)
    return manager


@pytest.fixture
def mock_evaluator_registry():
    registry = mock.MagicMock()

    def get_evaluator(name):
        evaluator = mock.MagicMock()
        evaluator.method_name = name
        evaluator.description = f"{name} description"
        evaluator.report = mock.MagicMock(
            return_value=(["Column1", "Column2"], [["val1", "val2"]])
        )
        return evaluator

    registry.get = get_evaluator
    return registry


@pytest.fixture
def reporter(
    mock_utility_service, mock_metric_manager, mock_evaluator_registry
):
    """An HTMLReporter wired with mocked dependencies."""
    service = HTMLReporter("test_reporting")
    service._other_services = {
        "utility": mock_utility_service,
        "logging": mock.MagicMock(),
    }
    service.set_metric_config(mock_metric_manager)
    service.set_evaluator_registry(mock_evaluator_registry)
    return service


@pytest.fixture
def mock_algorithm():
    return {
        "model": mock.MagicMock(
            hyperparam_grid={"alpha": [0.1, 1.0, 10.0], "max_iter": [100, 200]}
        )
    }


REPORTER_PORT_METHODS = (
    "get_report_data", "set_context", "clear_context", "set_metric_config",
    "set_evaluator_registry", "add_experiment", "add_data_manager",
    "add_dataset", "add_experiment_groups", "store_table_data",
    "store_plot_svg",
)


@pytest.mark.unit
class TestReporterPortConformance:
    def test_satisfies_reporter_port(self, reporter):
        for method in REPORTER_PORT_METHODS:
            assert callable(getattr(reporter, method, None))

    def test_port_defines_expected_methods(self):
        assert set(REPORTER_PORT_METHODS).issubset(set(dir(ReporterPort)))


@pytest.mark.unit
class TestHTMLReporterInheritedBehaviour:
    def test_set_and_get_context(self, reporter):
        reporter.set_context(
            "test_group", "test_dataset", 0,
            feature_names=["f1", "f2"], algorithm_names=["ridge"],
        )
        group, dataset, split, features, algorithms = reporter.get_context()
        assert group == "test_group"
        assert dataset == "test_dataset"
        assert split == 0
        assert features == ["f1", "f2"]
        assert algorithms == ["ridge"]

    def test_clear_context(self, reporter):
        reporter.set_context("test_group", "test_dataset", 0)
        reporter.clear_context()
        with pytest.raises(ValueError, match="No context set"):
            reporter.get_context()

    def test_add_data_manager_clears_caches(self, reporter, mock_data_manager):
        reporter._image_cache[("g", "d", "s", "m")] = ("img", {})
        reporter._table_cache[("g", "d", "s", "m")] = ({}, {})
        reporter._cached_tuned_params = {"param": "value"}

        reporter.add_data_manager("test_group", mock_data_manager)

        assert reporter._image_cache == {}
        assert reporter._table_cache == {}
        assert reporter._cached_tuned_params == {}

    def test_add_experiment_one_algorithm(self, reporter, mock_algorithm):
        reporter.set_context(
            "test_group", ("test_data.csv", None), 0,
            feature_names=["f1", "f2"], algorithm_names=["ridge"],
        )
        reporter.add_experiment(mock_algorithm)

        experiment_id = "ridge_test_group_test_data.csv"
        experiment = reporter.experiments[experiment_id]
        assert len(experiment.algorithm) == 1
        assert experiment.algorithm[0] == "ridge Display"


@pytest.mark.unit
class TestHTMLReporterRendering:
    def test_renderer_is_lazy_default(self, reporter):
        assert reporter._renderer is None
        assert isinstance(reporter.renderer, HTMLReportRenderer)

    def test_injected_renderer_used(
        self, mock_utility_service, mock_metric_manager,
        mock_evaluator_registry,
    ):
        fake_renderer = mock.MagicMock(spec=HTMLReportRenderer)
        service = HTMLReporter("test_reporting", renderer=fake_renderer)
        service._other_services = {
            "utility": mock_utility_service, "logging": mock.MagicMock(),
        }
        service.set_metric_config(mock_metric_manager)
        service.set_evaluator_registry(mock_evaluator_registry)

        service.render_report(pathlib.Path("/tmp/out"))

        fake_renderer.render.assert_called_once()
        args = fake_renderer.render.call_args[0]
        assert args[1] == pathlib.Path("/tmp/out")

    def test_render_report_writes_html(self, reporter, tmp_path):
        reporter.render_report(tmp_path)
        output = tmp_path / "report.html"
        assert output.exists()
        assert output.stat().st_size > 0
