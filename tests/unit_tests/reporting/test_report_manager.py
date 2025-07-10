from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path
import json

import pytest

from brisk.reporting.report_manager import ReportManager

@pytest.fixture
def report_manager(tmpdir):
    result_dir = tmpdir.mkdir("results")
    experiment_paths = {
        "group1": {
            "dataset1": {
                "experiment1": str(tmpdir.mkdir("experiment1"))
            }
        }
    }
    output_structure = {
        "group1": {
            "dataset1": ("data/test.csv", "group1")
        }
    }
    description_map = {}

    return ReportManager(
        result_dir=str(result_dir),
        experiment_paths=experiment_paths,
        output_structure=output_structure,
        description_map=description_map
    )

class TestReportManager:
    def test_create_dataset_page(self, report_manager, tmpdir):
        group_name = "group1"
        dataset_name = "dataset1"
        
        # Create the required directory structure
        split_dist_dir = Path(report_manager.result_dir) / group_name / dataset_name / "split_distribution"
        split_dist_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dummy files
        (split_dist_dir / "feature_distributions.png").touch()
        (split_dist_dir / "correlation_matrix.png").touch()
        
        with patch("builtins.open", mock_open()) as mock_file_write:
            with patch("os.listdir") as mock_listdir:
                mock_listdir.return_value = ["feature_distributions.png", "correlation_matrix.png"]
                report_manager.create_dataset_page(group_name, dataset_name)
                
                # Verify HTML file creation for the dataset page
                dataset_page_path = Path(report_manager.report_dir) / f"{group_name}_{dataset_name}.html"
                mock_file_write.assert_any_call(
                    str(dataset_page_path), "w", encoding="utf-8"
                )

    def test_get_json_metadata(self, report_manager, tmp_path):
        json_path = tmp_path / "test.json"
        json_path.write_text(json.dumps({
            "_metadata": {
                "models": {
                    "linear": "Linear Regression",
                    "ridge": "Ridge Regression"
                    },
                "metrics": ["metric1", "metric2"]
            }
        }))
        metadata = report_manager._get_json_metadata(json_path)
        assert metadata == {
            "models": {
                "linear": "Linear Regression",
                "ridge": "Ridge Regression"
            },
            "metrics": ["metric1", "metric2"]
        }

    @patch("brisk.reporting.report_manager.Image")
    def test_get_image_metadata_success(self, mock_image, report_manager):
        """Test successful extraction of image metadata."""
        mock_img = MagicMock()
        mock_img.info = {"width": 800, "height": 600, "format": "PNG"}
        mock_image.open.return_value.__enter__.return_value = mock_img
        
        result = report_manager._get_image_metadata("test.png")
        
        assert result == {"width": 800, "height": 600, "format": "PNG"}
        mock_image.open.assert_called_once_with("test.png")

    @patch("brisk.reporting.report_manager.Image")
    def test_get_image_metadata_with_models_string(self, mock_image, report_manager):
        """Test extraction when models metadata is a string that needs evaluation."""
        mock_img = MagicMock()
        mock_img.info = {
            "models": "{'linear': 'Linear Regression', 'ridge': 'Ridge Regression'}", "other": "data"
        }
        mock_image.open.return_value.__enter__.return_value = mock_img
        
        result = report_manager._get_image_metadata("test.png")
        
        assert result == {
            "models": {
                'linear': 'Linear Regression', 
                'ridge': 'Ridge Regression'}, 
            "other": "data"
        }

    @patch("joblib.load")
    @patch("builtins.open", new_callable=mock_open)
    def test_get_serialized_metadata_success(self, mock_file, mock_joblib, report_manager):
        """Test successful extraction of serialized metadata."""
        mock_joblib.return_value = {
            "metadata": {"model_type": "RandomForest", "accuracy": 0.95}
        }
        
        result = report_manager._get_serialized_metadata("model.pkl")
        
        assert result == {"model_type": "RandomForest", "accuracy": 0.95}
        mock_file.assert_called_once_with("model.pkl", "rb")
        mock_joblib.assert_called_once()

    def test_load_file_json(self, report_manager, tmp_path):
        """Test loading a JSON file."""
        json_file = tmp_path / "test.json"
        test_data = {"key": "value", "number": 42}
        json_file.write_text(json.dumps(test_data))
        
        result = report_manager._load_file(str(json_file))
        
        assert result == test_data

    def test_load_file_png(self, report_manager):
        """Test loading a PNG file (returns file path)."""
        result = report_manager._load_file("test.png")
        
        assert result == "test.png"

    @patch("joblib.load")
    @patch("builtins.open", new_callable=mock_open)
    def test_load_file_pkl(self, mock_file, mock_joblib, report_manager):
        """Test loading a PKL file."""
        mock_data = {"model": "test_model", "params": {"n_estimators": 100}}
        mock_joblib.return_value = mock_data
        
        result = report_manager._load_file("model.pkl")
        
        assert result == mock_data
        mock_file.assert_called_once_with("model.pkl", "rb")

    def test_load_file_unsupported_extension(self, report_manager):
        """Test loading a file with unsupported extension."""
        with pytest.raises(ValueError, match="Unsupported file type: test.txt"):
            report_manager._load_file("test.txt")

    def test_get_data_type_true_lowercase(self, report_manager):
        """Test converting 'true' to 'Test Set'."""
        result = report_manager._get_data_type("true")
        assert result == "Test Set"

    def test_get_data_type_true_uppercase(self, report_manager):
        """Test converting 'TRUE' to 'Test Set'."""
        result = report_manager._get_data_type("TRUE")
        assert result == "Test Set"

    def test_get_data_type_true_mixed_case(self, report_manager):
        """Test converting 'True' to 'Test Set'."""
        result = report_manager._get_data_type("True")
        assert result == "Test Set"

    def test_get_data_type_false_lowercase(self, report_manager):
        """Test converting 'false' to 'Train Set'."""
        result = report_manager._get_data_type("false")
        assert result == "Train Set"

    def test_get_data_type_false_uppercase(self, report_manager):
        """Test converting 'FALSE' to 'Train Set'."""
        result = report_manager._get_data_type("FALSE")
        assert result == "Train Set"

    def test_get_data_type_false_mixed_case(self, report_manager):
        """Test converting 'False' to 'Train Set'."""
        result = report_manager._get_data_type("False")
        assert result == "Train Set"

    def test_get_data_type_invalid_input(self, report_manager):
        """Test handling of invalid input."""
        with pytest.raises(ValueError, match="Invalid boolean string: invalid"):
            report_manager._get_data_type("invalid")

    def test_get_data_type_empty_string(self, report_manager):
        """Test handling of empty string."""
        with pytest.raises(ValueError, match="Invalid boolean string: "):
            report_manager._get_data_type("")

    def test_get_default_params_with_defaults(self, report_manager):
        """Test extracting default parameters from a class with default values."""
        class TestModel:
            def __init__(
                self, 
                param1, 
                param2=10, 
                param3="default", 
                param4=None, 
                param5=True,
            ):
                self.param1 = param1
                self.param2 = param2
                self.param3 = param3
                self.param4 = param4
                self.param5 = param5
        
        result = report_manager.get_default_params(TestModel)
        
        expected = {
            "param2": 10,
            "param3": "default",
            "param4": None,
            "param5": True
        }
        assert result == expected

    def test_get_default_params_no_defaults(self, report_manager):
        """Test extracting default parameters from a class with no default values."""
        class TestModel:
            def __init__(self, param1, param2, param3):
                self.param1 = param1
                self.param2 = param2
                self.param3 = param3
        
        result = report_manager.get_default_params(TestModel)
        
        assert result == {}

    def test_get_default_params_only_self(self, report_manager):
        """Test extracting default parameters from a class with only self parameter."""
        class TestModel:
            def __init__(self):
                pass
        
        result = report_manager.get_default_params(TestModel)
        
        assert result == {}

    def test_get_default_params_mixed_types(self, report_manager):
        """Test extracting default parameters with various data types."""
        class TestModel:
            def __init__(
                self, 
                required_param, 
                int_param=42, 
                float_param=3.14, 
                string_param="test", 
                list_param=None, 
                dict_param=None,
                bool_param=False
            ):
                if list_param is None:
                    list_param = []
                if dict_param is None:
                    dict_param = {}
                    
                self.required_param = required_param
                self.int_param = int_param
                self.float_param = float_param
                self.string_param = string_param
                self.list_param = list_param
                self.dict_param = dict_param
                self.bool_param = bool_param
        
        result = report_manager.get_default_params(TestModel)
        
        expected = {
            "int_param": 42,
            "float_param": 3.14,
            "string_param": "test",
            "list_param": None,
            "dict_param": None,
            "bool_param": False
        }
        assert result == expected

    def test_get_default_params_complex_defaults(self, report_manager):
        """Test extracting default parameters with complex default values."""
        default_list = [1, 2, 3]
        default_dict = {"key": "value"}
        
        class TestModel:
            def __init__(self, param1, 
                         list_param=default_list, 
                         dict_param=default_dict,
                         tuple_param=(1, 2, 3)):
                self.param1 = param1
                self.list_param = list_param
                self.dict_param = dict_param
                self.tuple_param = tuple_param
        
        result = report_manager.get_default_params(TestModel)
        
        expected = {
            "list_param": default_list,
            "dict_param": default_dict,
            "tuple_param": (1, 2, 3)
        }
        assert result == expected

    def test_get_default_params_keyword_only(self, report_manager):
        """Test extracting default parameters with keyword-only arguments."""
        class TestModel:
            def __init__(self, param1, param2=10, *, keyword_only1="test", keyword_only2=None):
                self.param1 = param1
                self.param2 = param2
                self.keyword_only1 = keyword_only1
                self.keyword_only2 = keyword_only2
        
        result = report_manager.get_default_params(TestModel)
        
        expected = {
            "param2": 10,
            "keyword_only1": "test",
            "keyword_only2": None
        }
        assert result == expected

    def test_get_default_params_real_sklearn_like_model(self, report_manager):
        """Test with a realistic model class similar to sklearn models."""
        class RandomForestClassifier:
            def __init__(self, 
                         n_estimators=100,
                         criterion='gini',
                         max_depth=None,
                         min_samples_split=2,
                         min_samples_leaf=1,
                         random_state=None,
                         verbose=0):
                self.n_estimators = n_estimators
                self.criterion = criterion
                self.max_depth = max_depth
                self.min_samples_split = min_samples_split
                self.min_samples_leaf = min_samples_leaf
                self.random_state = random_state
                self.verbose = verbose
        
        result = report_manager.get_default_params(RandomForestClassifier)
        
        expected = {
            "n_estimators": 100,
            "criterion": 'gini',
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": None,
            "verbose": 0
        }
        assert result == expected
