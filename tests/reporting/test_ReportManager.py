import re
import ast
import pytest
import os
import json
from unittest.mock import patch, mock_open
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

from brisk.reporting.ReportManager import ReportManager

class TestReportManager:

    @pytest.fixture
    def report_manager(self, tmpdir):
        result_dir = tmpdir.mkdir("results")
        experiment_paths = {"dataset1": [str(tmpdir.mkdir("experiment1"))]}
        data_distribution_paths = {"dataset1": str(tmpdir.mkdir("data_distribution"))}

        return ReportManager(
            result_dir=str(result_dir),
            experiment_paths=experiment_paths,
            data_distribution_paths=data_distribution_paths
        )

    def test_create_report(self, report_manager):
        with patch("shutil.copy") as mock_copy:
            with patch("builtins.open", mock_open()) as mock_file_write:
                
                # Run the report creation process
                report_manager.create_report()

                # Check that the index.html file write was triggered
                index_path = Path(report_manager.report_dir) / "index.html"
                mock_file_write.assert_any_call(str(index_path), 'w')

                assert mock_copy.called

    def test_create_experiment_page(self, report_manager, tmpdir):
        experiment_dir = str(tmpdir.mkdir("experiment_dir"))
        dataset = "dataset1"
        
        with patch("builtins.open", mock_open()) as mock_file_write:
            report_manager.create_experiment_page(experiment_dir, dataset)
            
            # Verify HTML file creation for the experiment page
            filename = f"{dataset}_experiment_dir.html"
            experiment_page_path = Path(report_manager.report_dir) / filename
            mock_file_write.assert_any_call(str(experiment_page_path), "w")

    def test_create_dataset_page(self, report_manager, tmpdir):
        dataset_name = "dataset1"
        
        with patch("builtins.open", mock_open()) as mock_file_write:
            report_manager.create_dataset_page(dataset_name)
            
            # Verify HTML file creation for the dataset page
            dataset_page_path = Path(report_manager.report_dir) / f"{dataset_name}.html"
            mock_file_write.assert_any_call(str(dataset_page_path), "w")

    def test_get_json_metadata(self, report_manager, tmpdir):
        # Create a JSON file with metadata
        json_file = tmpdir.join("sample.json")
        data = {"_metadata": {"models": "['model1', 'model2']"}}
        with open(json_file, "w") as f:
            json.dump(data, f)

        metadata = report_manager._get_json_metadata(str(json_file))

        # Assert metadata was extracted and models were parsed as list
        assert metadata["models"] == ["model1", "model2"]

    def test_get_image_metadata(self, report_manager, tmpdir):
        # Set up the metadata
        metadata = {"models": "['model1', 'model2']"}
        
        # Create a sample image with Matplotlib and save it with metadata
        image_path = tmpdir.join("sample.png")
        plt.figure()
        plt.plot([1, 2, 3], [4, 5, 6])
        plt.savefig(str(image_path), format="png", metadata=metadata)
        plt.close()

        # Use the method to extract metadata
        extracted_metadata = report_manager._get_image_metadata(str(image_path))

        # Verify that the extracted metadata matches the original
        assert extracted_metadata is not None, "No metadata found in image."
        assert "models" in extracted_metadata
        assert extracted_metadata["models"] == ast.literal_eval(metadata["models"])

    def test_report_image(self, report_manager, tmpdir):
        report_manager.report_dir = str(tmpdir.join("html_report"))

        data = tmpdir.join("path/to/image.png")
        data.ensure()  # Ensure the path exists for testing

        metadata = {"models": ["Test Model"]}
        title = "Test Image"

        result_html = report_manager.report_image(str(data), metadata, title)
        relative_img_path = os.path.relpath(str(data), report_manager.report_dir)

        assert f'<img src="{relative_img_path}"' in result_html
        assert "<h2>Test Image</h2>" in result_html
        assert "<strong>Model:</strong> Test Model" in result_html
        assert 'alt="Test Image"' in result_html
        assert 'style="max-width:100%; height:auto; display: block; margin: 0 auto;"' in result_html

    def test_report_evaluate_model(self, report_manager):
        data = {"accuracy": 0.95}
        metadata = {"models": ["Model A"]}
        
        result_html = report_manager.report_evaluate_model(data, metadata)
        
        assert "<h2>Model Evaluation</h2>" in result_html
        assert "<strong>Model:</strong> Model A" in result_html
        assert "<td>accuracy</td><td>0.95</td>" in result_html

    def test_report_compare_models(self, report_manager):
        data = {"Model A": {"accuracy": 0.9}, "Model B": {"accuracy": 0.85}}
        metadata = {"models": ["Model A", "Model B"]}
        
        result_html = report_manager.report_compare_models(data, metadata)
        
        assert "<h2>Model Comparison</h2>" in result_html
        assert "Model A" in result_html
        assert "Model B" in result_html
        assert "accuracy" in result_html

    def test_report_categorical_stats(self, report_manager, tmpdir):
        file_path = tmpdir.join("categorical_stats.json")
        file_path.write('{"feature1": {"train": {"frequency": {"A": 3, "B": 2}}, "test": {"frequency": {"A": 4, "B": 1}}}}')
        
        result_html = report_manager.report_categorical_stats(str(file_path))
        
        assert "<h3>feature1</h3>" in result_html
        assert "A: 3" in result_html
        assert "B: 2" in result_html

    def test_report_evaluate_model_cv(self, report_manager):
        data = {
            "accuracy": {
                "all_scores": [0.85, 0.88, 0.86],
                "mean_score": 0.8633,
                "std_dev": 0.0155
            },
            "f1_score": {
                "all_scores": [0.75, 0.78, 0.77],
                "mean_score": 0.7667,
                "std_dev": 0.0124
            }
        }
        metadata = {"models": ["Test Model"]}
        
        result_html = report_manager.report_evaluate_model_cv(data, metadata)
        
        assert "<h2>Model Evaluation (Cross-Validation)</h2>" in result_html
        assert "<strong>Model:</strong> Test Model" in result_html
        
        accuracy_pattern = (
            r"<td>accuracy</td><td>0\.85000, 0\.88000, 0\.86000</td>"
            r"<td>0\.863[0-9]*</td><td>0\.015[0-9]*</td>"
        )
        f1_score_pattern = (
            r"<td>f1_score</td><td>0\.75000, 0\.78000, 0\.77000</td>"
            r"<td>0\.766[0-9]*</td><td>0\.012[0-9]*</td>"
        )
        
        assert re.search(accuracy_pattern, result_html)
        assert re.search(f1_score_pattern, result_html)
        
        # Verify `summary_metrics` is updated correctly
        assert "Test Model" in report_manager.summary_metrics[report_manager.current_dataset]
        assert "accuracy" in report_manager.summary_metrics[report_manager.current_dataset]["Test Model"]
        assert report_manager.summary_metrics[report_manager.current_dataset]["Test Model"]["accuracy"]["mean"] == 0.8633

    def test_generate_summary_tables(self, report_manager):
        report_manager.summary_metrics = {
            "Dataset 1": {
                "Model A": {
                    "accuracy": {"mean": 0.85, "std_dev": 0.03},
                    "f1_score": {"mean": 0.80, "std_dev": 0.04}
                },
                "Model B": {
                    "accuracy": {"mean": 0.87, "std_dev": 0.02},
                    "f1_score": {"mean": 0.82, "std_dev": 0.03}
                }
            }
        }

        summary_html = report_manager.generate_summary_tables()

        assert "<h2>Summary for Dataset 1</h2>" in summary_html
        assert "<th>Model</th>" in summary_html
        assert "<th>accuracy</th>" in summary_html
        assert "<th>f1_score</th>" in summary_html

        assert "<td>Model A</td>" in summary_html
        assert "<td>0.85 (0.03)</td>" in summary_html
        assert "<td>0.8 (0.04)</td>" in summary_html

        assert "<td>Model B</td>" in summary_html
        assert "<td>0.87 (0.02)</td>" in summary_html
        assert "<td>0.82 (0.03)</td>" in summary_html

    def test_report_continuous_stats(self, report_manager, tmpdir):
        stats_data = {
            "feature_1": {
                "train": {"mean": 5.12345, "std_dev": 1.23456},
                "test": {"mean": 5.54321, "std_dev": 1.65432}
            }
        }
        stats_file = tmpdir.join("continuous_stats.json")
        with open(stats_file, "w") as f:
            json.dump(stats_data, f)
        
        result_html = report_manager.report_continuous_stats(str(stats_file))
        
        assert "<h3>feature_1</h3>" in result_html
        assert "<td>Mean</td>" in result_html
        assert "<td>5.12345</td>" in result_html
        assert "<td>5.54321</td>" in result_html
        assert "<td>Std dev</td>" in result_html
        assert "<td>1.23456</td>" in result_html
        assert "<td>1.65432</td>" in result_html
        assert "No image found at ../../hist_box_plot/feature_1_hist_box.png" in result_html
        
    def test_report_correlation_matrix(self, report_manager, tmpdir):
        correlation_image = tmpdir.join("correlation_matrix.png")
        correlation_image.ensure()

        result_html = report_manager.report_correlation_matrix(str(correlation_image))
        
        relative_img_path = os.path.relpath(str(correlation_image), report_manager.report_dir)
        
        assert '<img src="' + relative_img_path + '"' in result_html
        assert '<h3>Correlation Matrix</h3>' in result_html
        assert 'alt="Correlation Matrix"' in result_html

    def test_load_file_json(self, report_manager, tmpdir):
        data = {"key": "value"}
        json_file = tmpdir.join("test.json")
        with open(json_file, "w") as f:
            json.dump(data, f)

        result = report_manager._load_file(str(json_file))
        assert result == data

    def test_load_file_png(self, report_manager, tmpdir):
        png_file = tmpdir.join("test.png")
        with open(png_file, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")

        result = report_manager._load_file(str(png_file))
        assert result == str(png_file)  # Check that the file path is returned

    def test_load_file_unsupported(self, report_manager):
        with pytest.raises(ValueError, match="Unsupported file type:"):
            report_manager._load_file("unsupported.txt")

    def test_get_json_metadata_success(self, report_manager, tmpdir):
        data = {"_metadata": {"models": "['model1', 'model2']"}}
        json_file = tmpdir.join("metadata.json")
        with open(json_file, "w") as f:
            json.dump(data, f)

        metadata = report_manager._get_json_metadata(str(json_file))
        assert "models" in metadata
        assert metadata["models"] == ["model1", "model2"]

    def test_get_json_metadata_exception(self, report_manager):
        with patch("builtins.open", mock_open(read_data="{invalid_json")):
            metadata = report_manager._get_json_metadata("invalid.json")
            assert metadata == "unknown_method"

    def test_get_image_metadata_success(self, report_manager, tmpdir):
        png_file = tmpdir.join("image.png")
        metadata = {"models": "['model1', 'model2']"}

        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        fig.savefig(str(png_file), format="png", metadata=metadata)
        plt.close(fig)

        extracted_metadata = report_manager._get_image_metadata(str(png_file))
        assert "models" in extracted_metadata
        assert extracted_metadata["models"] == ast.literal_eval(metadata["models"])

    def test_get_image_metadata_failure(self, report_manager, tmpdir):
        invalid_file = tmpdir.join("invalid_image.png")
        with open(invalid_file, "w") as f:
            f.write("")  

        extracted_metadata = report_manager._get_image_metadata(str(invalid_file))
        assert extracted_metadata is None
