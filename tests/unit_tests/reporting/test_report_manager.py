import re
import ast
import pytest
import os
import json
from unittest.mock import patch, mock_open
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

from brisk.reporting.report_manager import ReportManager

class TestReportManager:

    @pytest.fixture
    def report_manager(self, tmpdir):
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
