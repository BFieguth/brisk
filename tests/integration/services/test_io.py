"""Integration tests for IOService."""
import sys
import json
import sqlite3

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotnine as pn
import plotly.graph_objects as go
from unittest import mock

from brisk.services import io


@pytest.mark.integration
class TestNumpyEncoder:
    def test_numpy_integer(self):
        """Test encoding numpy integers."""
        data = {"value": np.int64(42)}
        json_str = json.dumps(data, cls=io.NumpyEncoder)
        result = json.loads(json_str)

        assert result["value"] == 42
        assert isinstance(result["value"], int)

    def test_numpy_float(self):
        """Test encoding numpy floats."""
        data = {"value": np.float64(3.14)}
        json_str = json.dumps(data, cls=io.NumpyEncoder)
        result = json.loads(json_str)

        assert abs(result["value"] - 3.14) < 0.001
        assert isinstance(result["value"], float)

    def test_numpy_array(self):
        """Test encoding numpy arrays."""
        data = {"values": np.array([1, 2, 3])}
        json_str = json.dumps(data, cls=io.NumpyEncoder)
        result = json.loads(json_str)

        assert result["values"] == [1, 2, 3]
        assert isinstance(result["values"], list)


# ============================================================================
# Fixtures for file content
# ============================================================================

@pytest.fixture
def sample_csv_content():
    return """name,age,score
Alice,25,85.5
Bob,30,92.3
Charlie,35,78.9"""


@pytest.fixture
def sample_excel_data():
    return pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'score': [85.5, 92.3, 78.9]
    })


@pytest.fixture
def sample_sqlite_data():
    return pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'value': [100, 200, 300]
    })


@pytest.fixture
def algorithms_file_content():
    return """from brisk.configuration.algorithm_collection import AlgorithmCollection
from brisk.adapters.sklearn.model_adapter import (
    SklearnAlgorithmWrapper as AlgorithmWrapper,
)
from sklearn.linear_model import Ridge

ALGORITHM_CONFIG = AlgorithmCollection(
    AlgorithmWrapper(
        name="ridge",
        display_name="Ridge Regression",
        algorithm_class=Ridge,
        default_params={"alpha": 1.0},
        hyperparam_grid={"alpha": [0.1, 1.0, 10.0]}
    )
)
"""


@pytest.fixture
def data_file_content():
    return """from brisk.data.data_manager import DataManager

BASE_DATA_MANAGER = DataManager()
"""


@pytest.fixture
def metrics_file_content():
    return """from brisk import MetricManager

METRIC_CONFIG = MetricManager()
"""


@pytest.fixture
def evaluators_file_content():
    return """def register_custom_evaluators(registry):
    '''Register custom evaluators.'''
    pass
"""


@pytest.fixture
def workflow_file_content():
    return """from brisk.training.workflow import Workflow

class MyCustomWorkflow(Workflow):
    def run(self):
        pass
"""


# ============================================================================
# Test fixtures setup
# ============================================================================

@pytest.fixture
def fixtures_dir(tmp_path):
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    return fixtures


@pytest.fixture
def csv_file(fixtures_dir, sample_csv_content):
    csv_path = fixtures_dir / "test_data.csv"
    csv_path.write_text(sample_csv_content)
    return csv_path


@pytest.fixture
def excel_file(fixtures_dir, sample_excel_data):
    excel_path = fixtures_dir / "test_data.xlsx"
    sample_excel_data.to_excel(excel_path, index=False)
    return excel_path


@pytest.fixture
def sqlite_file(fixtures_dir, sample_sqlite_data):
    db_path = fixtures_dir / "test_data.db"
    conn = sqlite3.connect(db_path)
    sample_sqlite_data.to_sql('test_table', conn, index=False)
    conn.close()
    return db_path


@pytest.fixture
def algorithms_file(fixtures_dir, algorithms_file_content):
    algo_path = fixtures_dir / "algorithms.py"
    algo_path.write_text(algorithms_file_content)
    return algo_path


@pytest.fixture
def data_file(fixtures_dir, data_file_content):
    data_path = fixtures_dir / "data.py"
    data_path.write_text(data_file_content)
    return data_path


@pytest.fixture
def metrics_file(fixtures_dir, metrics_file_content):
    metrics_path = fixtures_dir / "metrics.py"
    metrics_path.write_text(metrics_file_content)
    return metrics_path


@pytest.fixture
def evaluators_file(fixtures_dir, evaluators_file_content):
    eval_path = fixtures_dir / "evaluators.py"
    eval_path.write_text(evaluators_file_content)
    return eval_path


@pytest.fixture
def workflow_file(fixtures_dir, workflow_file_content):
    workflow_path = fixtures_dir / "test_workflow.py"
    workflow_path.write_text(workflow_file_content)
    return workflow_path


# ============================================================================
# Service fixtures
# ============================================================================

@pytest.fixture
def mock_reporting_service():
    service = mock.MagicMock()
    service.store_table_data = mock.MagicMock()
    service.store_plot_svg = mock.MagicMock()
    return service


@pytest.fixture
def mock_logging_service():
    service = mock.MagicMock()
    service.logger = mock.MagicMock()
    return service


@pytest.fixture
def mock_rerun_service():
    service = mock.MagicMock()
    service.is_coordinating = False
    service.handle_load_custom_evaluators = mock.MagicMock(return_value=None)
    service.handle_load_base_data_manager = mock.MagicMock(return_value=None)
    service.handle_load_algorithms = mock.MagicMock(return_value=None)
    service.handle_load_workflow = mock.MagicMock(return_value=None)
    service.handle_load_metric_config = mock.MagicMock(return_value=None)
    return service


@pytest.fixture
def io_service(tmp_path, mock_reporting_service, mock_logging_service, mock_rerun_service):
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "output"
    results_dir.mkdir()
    output_dir.mkdir()

    service = io.IOService("test_io", results_dir, output_dir)
    service._other_services = {
        "reporting": mock_reporting_service,
        "logging": mock_logging_service,
        "rerun": mock_rerun_service
    }
    return service


# ============================================================================
# Test Cases
# ============================================================================


@pytest.mark.integration
class TestIOService:
    def test_save_to_json_missing_dir(self, io_service, tmp_path):
        """Test save_to_json creates missing directories."""
        output_path = tmp_path / "missing" / "nested" / "data.json"
        data = {"test": "value"}
        metadata = {"type": "test"}

        io_service.save_to_json(data, output_path, metadata)

        assert output_path.exists()
        with open(output_path) as f:
            saved_data = json.load(f)
        assert saved_data["test"] == "value"

    def test_save_to_json_dir_exists(self, io_service, tmp_path):
        """Test save_to_json when directory already exists."""
        output_dir = tmp_path / "existing"
        output_dir.mkdir()
        output_path = output_dir / "data.json"

        data = {"accuracy": 0.95, "precision": 0.92}
        metadata = {"experiment": "exp_1"}

        io_service.save_to_json(data, output_path, metadata)

        assert output_path.exists()
        with open(output_path) as f:
            saved_data = json.load(f)
        assert saved_data["accuracy"] == 0.95

    def test_save_plot_missing_dir(self, io_service, tmp_path):
        """Test save_plot creates missing directories."""
        output_path = tmp_path / "missing" / "plots" / "test.png"

        plt.figure()
        plt.plot([1, 2, 3], [1, 4, 9])

        io_service.save_plot(output_path, metadata={"type": "test"})

        assert output_path.exists()

    def test_save_plot_dir_exists(self, io_service, tmp_path):
        """Test save_plot when directory already exists."""
        output_dir = tmp_path / "plots"
        output_dir.mkdir()
        output_path = output_dir / "test.png"

        plt.figure()
        plt.plot([1, 2, 3], [1, 4, 9])

        io_service.save_plot(output_path, metadata={"type": "test"})

        assert output_path.exists()

    def test_save_plot_plotnine(self, io_service, tmp_path):
        """Test save_plot with plotnine plot."""
        output_path = tmp_path / "plotnine_plot.png"

        df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 4, 9]})
        plot = (
            pn.ggplot(df, pn.aes('x', 'y')) +
            pn.geom_point()
        )

        io_service.save_plot(output_path, plot=plot)

        assert output_path.exists()

    @pytest.mark.slow
    def test_save_plot_plotly(self, io_service, tmp_path):
        """Test save_plot with plotly figure."""
        output_path = tmp_path / "plotly_plot.png"

        fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[1, 4, 9]))

        io_service.save_plot(output_path, plot=fig)

        assert output_path.exists()

    def test_save_plot_matplotlib(self, io_service, tmp_path):
        """Test save_plot with matplotlib figure (no plot argument)."""
        output_path = tmp_path / "matplotlib_plot.png"

        plt.figure()
        plt.plot([1, 2, 3], [1, 4, 9])

        io_service.save_plot(output_path)

        assert output_path.exists()

    def test_save_plot_seaborn(self, io_service, tmp_path):
        """Test save_plot with seaborn plot (uses matplotlib backend)."""
        output_path = tmp_path / "seaborn_plot.png"

        plt.figure()
        sns.lineplot(x=[1, 2, 3], y=[1, 4, 9])

        io_service.save_plot(output_path)

        assert output_path.exists()

    def test_save_plot_svg_conversion_matplotlib(self, io_service, tmp_path):
        """Test SVG conversion for matplotlib plots."""
        output_path = tmp_path / "plot.png"
        metadata = {"type": "test"}

        plt.figure()
        plt.plot([1, 2, 3], [1, 4, 9])

        io_service.save_plot(output_path, metadata=metadata)

        io_service._other_services["reporting"].store_plot_svg.assert_called_once()
        call_args = io_service._other_services["reporting"].store_plot_svg.call_args
        svg_str = call_args[0][0]
        assert svg_str.startswith('<?xml') or svg_str.startswith('<svg')

    def test_save_plot_svg_conversion_plotnine(self, io_service, tmp_path):
        """Test SVG conversion for plotnine plots."""
        output_path = tmp_path / "plot.png"
        metadata = {"type": "test"}

        df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 4, 9]})
        plot = (
            pn.ggplot(df, pn.aes('x', 'y')) +
            pn.geom_point()
        )

        io_service.save_plot(output_path, plot=plot, metadata=metadata)

        io_service._other_services["reporting"].store_plot_svg.assert_called_once()
        call_args = io_service._other_services["reporting"].store_plot_svg.call_args
        svg_str = call_args[0][0]
        assert svg_str.startswith('<?xml') or svg_str.startswith('<svg')

    @pytest.mark.slow
    def test_save_plot_svg_conversion_plotly(self, io_service, tmp_path):
        """Test SVG conversion for plotly figures."""
        output_path = tmp_path / "plot.png"
        metadata = {"type": "test"}

        fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[1, 4, 9]))

        io_service.save_plot(output_path, plot=fig, metadata=metadata)

        io_service._other_services["reporting"].store_plot_svg.assert_called_once()
        call_args = io_service._other_services["reporting"].store_plot_svg.call_args
        svg_str = call_args[0][0]
        assert svg_str.startswith('<?xml') or svg_str.startswith('<svg')

    def test_save_plot_svg_conversion_seaborn(self, io_service, tmp_path):
        """Test SVG conversion for seaborn plots (matplotlib backend)."""
        output_path = tmp_path / "plot.png"
        metadata = {"type": "test"}

        plt.figure()
        sns.lineplot(x=[1, 2, 3], y=[1, 4, 9])

        io_service.save_plot(output_path, metadata=metadata)

        io_service._other_services["reporting"].store_plot_svg.assert_called_once()
        call_args = io_service._other_services["reporting"].store_plot_svg.call_args
        svg_str = call_args[0][0]
        assert svg_str.startswith('<?xml') or svg_str.startswith('<svg')

    def test_save_rerun_config_missing_dir(self, io_service, tmp_path):
        """Test save_rerun_config creates missing directories."""
        output_path = tmp_path / "missing" / "config" / "rerun.json"
        data = {"config": "value"}
        metadata = {"type": "rerun"}

        io_service.save_rerun_config(data, metadata, output_path)

        assert output_path.exists()
        with open(output_path) as f:
            saved_data = json.load(f)
        assert saved_data["config"] == "value"

    def test_save_rerun_config_dir_exists(self, io_service, tmp_path):
        """Test save_rerun_config when directory already exists."""
        output_dir = tmp_path / "config"
        output_dir.mkdir()
        output_path = output_dir / "rerun.json"

        data = {"settings": "test"}
        metadata = {"timestamp": "2024-01-15"}

        io_service.save_rerun_config(data, metadata, output_path)

        assert output_path.exists()
        with open(output_path) as f:
            saved_data = json.load(f)
        assert saved_data["settings"] == "test"

    def test_load_data_csv(self, csv_file):
        """Test loading CSV files."""
        df = io.IOService.load_data(str(csv_file))
        assert isinstance(df, pd.DataFrame)

    def test_load_data_xlsx(self, excel_file):
        """Test loading XLSX files."""
        df = io.IOService.load_data(str(excel_file))
        assert isinstance(df, pd.DataFrame)

    def test_load_data_sqlite(self, sqlite_file):
        """Test loading SQLite database files."""
        df = io.IOService.load_data(str(sqlite_file), table_name='test_table')
        assert isinstance(df, pd.DataFrame)

    def test_load_module_object(self, fixtures_dir):
        """Test loading an object from a module file."""
        module_path = fixtures_dir / "test_module.py"
        module_path.write_text("""
TEST_OBJECT = "test_value"
TEST_NUMBER = 42
""")

        obj = io.IOService.load_module_object(
            str(fixtures_dir),
            "test_module.py",
            "TEST_OBJECT",
            required=True
        )

        assert obj == "test_value"

    def test_load_custom_evaluators_missing_file(self, io_service, tmp_path):
        """Test load_custom_evaluators with missing file."""
        missing_file = tmp_path / "missing_evaluators.py"

        with pytest.raises(FileNotFoundError):
            io_service.load_custom_evaluators(missing_file)

    def test_load_custom_evaluators_missing_function(self, io_service, fixtures_dir):
        """Test load_custom_evaluators when file lacks register function."""
        eval_file = fixtures_dir / "bad_evaluators.py"
        eval_file.write_text("# No register function here")

        result = io_service.load_custom_evaluators(eval_file)

        io_service._other_services["logging"].logger.warning.assert_called()

    def test_load_custom_evaluators_calls_rerun(self, io_service, evaluators_file):
        """Test that load_custom_evaluators passes correct data to rerun service."""
        io_service.load_custom_evaluators(evaluators_file)
        assert io_service._other_services["rerun"].handle_load_custom_evaluators.call_count == 1

    def test_load_base_data_manager_missing_file(self, io_service, tmp_path):
        """Test load_base_data_manager with missing file."""
        missing_file = tmp_path / "missing_data.py"

        with pytest.raises(FileNotFoundError):
            io_service.load_base_data_manager(missing_file)

    def test_load_base_data_manager_missing_module(self, io_service, fixtures_dir):
        """Test load_base_data_manager when BASE_DATA_MANAGER is missing."""
        data_file = fixtures_dir / "bad_data.py"
        data_file.write_text("# No BASE_DATA_MANAGER here")

        with pytest.raises(ImportError):
            io_service.load_base_data_manager(data_file)

    def test_load_base_data_manager_calls_rerun(self, io_service, data_file):
        """Test that load_base_data_manager passes correct data to rerun service."""
        with mock.patch('brisk.services.io.IOService._validate_single_variable'):
            io_service.load_base_data_manager(data_file)
            io_service._other_services["rerun"].handle_load_base_data_manager.assert_called()

    def test_load_algorithms_file_missing(self, io_service, tmp_path):
        """Test load_algorithms with missing file."""
        missing_file = tmp_path / "missing_algorithms.py"
        with pytest.raises(FileNotFoundError):
            io_service.load_algorithms(missing_file)

    def test_load_algorithms_module_missing(self, io_service, fixtures_dir):
        """Test load_algorithms when ALGORITHM_CONFIG is missing."""
        algo_file = fixtures_dir / "bad_algorithms.py"
        algo_file.write_text("# No ALGORITHM_CONFIG here")

        with pytest.raises(ImportError):
            io_service.load_algorithms(algo_file)

    def test_load_algorithms_calls_rerun(self, io_service, algorithms_file):
        """Test that load_algorithms passes correct data to rerun service."""
        with mock.patch('brisk.services.io.IOService._validate_single_variable'):
            io_service.load_algorithms(algorithms_file)
            io_service._other_services["rerun"].handle_load_algorithms.assert_called()

    def test_load_workflow_missing_file(self, io_service):
        """Test load_workflow with missing workflow file."""
        with mock.patch('importlib.import_module', side_effect=ImportError("No module")):
            with pytest.raises(ImportError):
                result = io_service.load_workflow("missing_workflow")

    def test_load_workflow_missing_workflow(self, io_service, fixtures_dir):
        """Test load_workflow when no Workflow subclass exists."""
        workflows_dir = fixtures_dir / "workflows"
        workflows_dir.mkdir()
        (workflows_dir / "__init__.py").write_text("")

        workflow_file = workflows_dir / "no_workflow.py"
        workflow_file.write_text("# No Workflow class here")
        original_path = sys.path.copy()
        try:
            sys.path.insert(0, str(fixtures_dir))

            with pytest.raises(ImportError):
                io_service.load_workflow("no_workflow")
        finally:
            sys.path = original_path

    def test_load_workflow_multiple_workflows(self, io_service, fixtures_dir):
        """Test load_workflow when multiple Workflow subclasses exist."""
        multi_workflow_file = fixtures_dir / "multi_workflow.py"
        multi_workflow_file.write_text("""
from brisk.training.workflow import Workflow

class WorkflowOne(Workflow):
    def run(self):
        pass

class WorkflowTwo(Workflow):
    def run(self):
        pass
""")

        with mock.patch('sys.path', [str(fixtures_dir)] + list(sys.path)):
            with pytest.raises(ImportError):
                io_service.load_workflow("multi_workflow")

    def test_load_metric_config_missing_file(self, io_service, tmp_path):
        """Test load_metric_config with missing file."""
        missing_file = tmp_path / "missing_metrics.py"

        with pytest.raises(FileNotFoundError):
            io_service.load_metric_config(missing_file)

    def test_load_metric_config_missing_module(self, io_service, fixtures_dir):
        """Test load_metric_config when METRIC_CONFIG is missing."""
        metric_file = fixtures_dir / "bad_metrics.py"
        metric_file.write_text("# No METRIC_CONFIG here")

        with pytest.raises(ImportError):
            io_service.load_metric_config(metric_file)

    def test_load_metric_config_calls_rerun(self, io_service, metrics_file):
        """Test that load_metric_config passes correct data to rerun service."""
        with mock.patch('brisk.services.io.IOService._validate_single_variable'):
            io_service.load_metric_config(metrics_file)
            io_service._other_services["rerun"].handle_load_metric_config.assert_called()
