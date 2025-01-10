"""Test the ResultStructure class."""
import importlib
import json
import os
import pathlib
import pickle
import tempfile
from unittest import mock

import matplotlib.pyplot as plt
import pytest

import brisk.utility.result_structure as rs
from brisk.configuration import experiment_group

@pytest.fixture
def one_experiment_group():
    return {
        "group1": rs.ExperimentGroupDirectory(
            datasets={
                "dataset1": rs.DatasetDirectory(
                    experiments={
                        "group1_linear": rs.ExperimentDirectory(
                            save_model=True,
                            evaluate_model=True,
                            evaluate_model_cv=False,
                            compare_models=False,
                            plot_pred_vs_obs=True,
                            plot_learning_curve=True,
                            plot_feature_importance=True,
                            plot_residuals=True,
                            plot_model_comparison=False,
                            confusion_matrix=False,
                            plot_confusion_heatmap=False,
                            plot_roc_curve=False,
                            plot_precision_recall_curve=False,
                        ),
                        "group1_rf": rs.ExperimentDirectory(
                            save_model=True,
                            evaluate_model=True,
                            evaluate_model_cv=False,
                            compare_models=False,
                            plot_pred_vs_obs=True,
                            plot_learning_curve=True,
                            plot_feature_importance=True,
                            plot_residuals=True,
                            plot_model_comparison=False,
                            confusion_matrix=False,
                            plot_confusion_heatmap=False,
                            plot_roc_curve=False,
                            plot_precision_recall_curve=False,
                        )
                    },
                    scaler_exists=True,
                    split_distribution_exists=True,
                    hist_box_plot_exists=True,
                    pie_plot_exists=True,
                    categorical_stats_json_exists=True,
                    continuous_stats_json_exists=True,
                    correlation_matrix_exists=True
                ),
                "dataset2": rs.DatasetDirectory(
                    experiments={
                        "group1_linear": rs.ExperimentDirectory(
                            save_model=True,
                            evaluate_model=True,
                            evaluate_model_cv=False,
                            compare_models=False,
                            plot_pred_vs_obs=True,
                            plot_learning_curve=True,
                            plot_feature_importance=True,
                            plot_residuals=True,
                            plot_model_comparison=False,
                            confusion_matrix=False,
                            plot_confusion_heatmap=False,
                            plot_roc_curve=False,
                            plot_precision_recall_curve=False,
                        ),
                        "group1_ridge": rs.ExperimentDirectory(
                            save_model=True,
                            evaluate_model=True,
                            evaluate_model_cv=False,
                            compare_models=False,
                            plot_pred_vs_obs=True,
                            plot_learning_curve=True,
                            plot_feature_importance=True,
                            plot_residuals=True,
                            plot_model_comparison=False,
                            confusion_matrix=False,
                            plot_confusion_heatmap=False,
                            plot_roc_curve=False,
                            plot_precision_recall_curve=False,
                        ),
                        "group1_lasso": rs.ExperimentDirectory(
                            save_model=True,
                            evaluate_model=True,
                            evaluate_model_cv=False,
                            compare_models=False,
                            plot_pred_vs_obs=True,
                            plot_learning_curve=True,
                            plot_feature_importance=True,
                            plot_residuals=True,
                            plot_model_comparison=False,
                            confusion_matrix=False,
                            plot_confusion_heatmap=False,
                            plot_roc_curve=False,
                            plot_precision_recall_curve=False,
                        ),
                    },
                    scaler_exists=True,
                    split_distribution_exists=True,
                    hist_box_plot_exists=True,
                    pie_plot_exists=True,
                    categorical_stats_json_exists=True,
                    continuous_stats_json_exists=True,
                    correlation_matrix_exists=True
                )
            }
        )
    }


@pytest.fixture
def two_experiment_groups():
    return {
        "group1": rs.ExperimentGroupDirectory(
            datasets={
                "dataset1": rs.DatasetDirectory(
                    experiments={
                        "group1_linear": rs.ExperimentDirectory(
                            save_model=True,
                            evaluate_model=True,
                            evaluate_model_cv=False,
                            compare_models=False,
                            plot_pred_vs_obs=True,
                            plot_learning_curve=True,
                            plot_feature_importance=True,
                            plot_residuals=True,
                            plot_model_comparison=False,
                            confusion_matrix=False,
                            plot_confusion_heatmap=False,
                            plot_roc_curve=False,
                            plot_precision_recall_curve=False,
                        ),
                        "group1_rf": rs.ExperimentDirectory(
                            save_model=True,
                            evaluate_model=True,
                            evaluate_model_cv=False,
                            compare_models=False,
                            plot_pred_vs_obs=True,
                            plot_learning_curve=True,
                            plot_feature_importance=True,
                            plot_residuals=True,
                            plot_model_comparison=False,
                            confusion_matrix=False,
                            plot_confusion_heatmap=False,
                            plot_roc_curve=False,
                            plot_precision_recall_curve=False,
                        )
                    },
                    scaler_exists=True,
                    split_distribution_exists=True,
                    hist_box_plot_exists=True,
                    pie_plot_exists=True,
                    categorical_stats_json_exists=True,
                    continuous_stats_json_exists=True,
                    correlation_matrix_exists=True
                ),
                "dataset2": rs.DatasetDirectory(
                    experiments={
                        "group1_linear": rs.ExperimentDirectory(
                            save_model=True,
                            evaluate_model=True,
                            evaluate_model_cv=False,
                            compare_models=False,
                            plot_pred_vs_obs=True,
                            plot_learning_curve=True,
                            plot_feature_importance=True,
                            plot_residuals=True,
                            plot_model_comparison=False,
                            confusion_matrix=False,
                            plot_confusion_heatmap=False,
                            plot_roc_curve=False,
                            plot_precision_recall_curve=False,
                        ),
                        "group1_ridge": rs.ExperimentDirectory(
                            save_model=True,
                            evaluate_model=True,
                            evaluate_model_cv=False,
                            compare_models=False,
                            plot_pred_vs_obs=True,
                            plot_learning_curve=True,
                            plot_feature_importance=True,
                            plot_residuals=True,
                            plot_model_comparison=False,
                            confusion_matrix=False,
                            plot_confusion_heatmap=False,
                            plot_roc_curve=False,
                            plot_precision_recall_curve=False,
                        ),
                        "group1_lasso": rs.ExperimentDirectory(
                            save_model=True,
                            evaluate_model=True,
                            evaluate_model_cv=False,
                            compare_models=False,
                            plot_pred_vs_obs=True,
                            plot_learning_curve=True,
                            plot_feature_importance=True,
                            plot_residuals=True,
                            plot_model_comparison=False,
                            confusion_matrix=False,
                            plot_confusion_heatmap=False,
                            plot_roc_curve=False,
                            plot_precision_recall_curve=False,
                        ),
                    },
                    scaler_exists=True,
                    split_distribution_exists=True,
                    hist_box_plot_exists=True,
                    pie_plot_exists=True,
                    categorical_stats_json_exists=True,
                    continuous_stats_json_exists=True,
                    correlation_matrix_exists=True
                )
            }
        ),
        "group2": rs.ExperimentGroupDirectory(
            datasets={
                "dataset3": rs.DatasetDirectory(
                    experiments={
                        "group2_ridge": rs.ExperimentDirectory(
                            save_model=True,
                            evaluate_model=True,
                            evaluate_model_cv=False,
                            compare_models=False,
                            plot_pred_vs_obs=True,
                            plot_learning_curve=True,
                            plot_feature_importance=True,
                            plot_residuals=True,
                            plot_model_comparison=False,
                            confusion_matrix=False,
                            plot_confusion_heatmap=False,
                            plot_roc_curve=False,
                            plot_precision_recall_curve=False,
                        ),
                        "group2_mlp": rs.ExperimentDirectory(
                            save_model=True,
                            evaluate_model=True,
                            evaluate_model_cv=False,
                            compare_models=False,
                            plot_pred_vs_obs=True,
                            plot_learning_curve=True,
                            plot_feature_importance=True,
                            plot_residuals=True,
                            plot_model_comparison=False,
                            confusion_matrix=False,
                            plot_confusion_heatmap=False,
                            plot_roc_curve=False,
                            plot_precision_recall_curve=False,
                        ),
                    },
                    scaler_exists=True,
                    split_distribution_exists=True,
                    hist_box_plot_exists=True,
                    pie_plot_exists=True,
                    categorical_stats_json_exists=True,
                    continuous_stats_json_exists=True,
                    correlation_matrix_exists=True
                )
            }
        ),
    }


@pytest.fixture
def experiment_groups():
    with mock.patch(
        "brisk.utility.utility.find_project_root"
    ) as mock_find_project_root:
        mock_find_project_root.return_value = pathlib.Path("/fake/project/root")

        with mock.patch.object(
            experiment_group.ExperimentGroup, "_validate_datasets",
            return_value=None
        ):
            return [
                experiment_group.ExperimentGroup(
                    name="group1",
                    datasets=["dataset1"],
                    algorithms=["linear"],
                ),
                experiment_group.ExperimentGroup(
                    name="group2",
                    datasets=["dataset1", "dataset2"],
                    algorithms=["linear"],
                    data_config={
                        "scale_method": "minmax"
                    },
                ),
                experiment_group.ExperimentGroup(
                    name="group3",
                    datasets=["dataset1"],
                    algorithms=["dtr", "rf"],
                    data_config={
                        "categorical_features": "column1"
                    },
                ),
                experiment_group.ExperimentGroup(
                    name="group4",
                    datasets=["dataset2"],
                    algorithms=["knn"],
                    data_config={
                        "scale_method": "minmax",
                        "categorical_features": "column1"
                    },
                )
            ]


@pytest.fixture
def temp_experiment_dir():
    """
    Fixture to create a temporary experiment directory with .pkl, .json, and 
    .png files.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        experiment_dir = pathlib.Path(temp_dir)

        pkl_file = experiment_dir / "model.pkl"
        with open(pkl_file, "wb") as f:
            pickle.dump({"model": "dummy_model"}, f)

        json_file = experiment_dir / "metadata.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump({"_metadata": {"method": "evaluate_model"}}, f)

        png_file = experiment_dir / "image.png"
        plt.figure()
        plt.savefig(
            png_file, format="png", bbox_inches="tight", pad_inches=0,
            metadata={"method": "plotting_method"}
        )
        plt.close()

        yield experiment_dir


@pytest.fixture
def expected_result_structure():
    return rs.ResultStructure(
        config_log=rs.ConfigLog(True),
        error_log=rs.ErrorLog(True),
        experiment_groups={
            "group1": rs.ExperimentGroupDirectory(
                datasets={
                    "mixed_features_regression": rs.DatasetDirectory(
                        experiments={
                            "group1_lasso": rs.ExperimentDirectory(
                                save_model=True,
                                evaluate_model=True,
                                evaluate_model_cv=False,
                                compare_models=False,
                                plot_pred_vs_obs=True,
                                plot_learning_curve=True,
                                plot_feature_importance=True,
                                plot_residuals=False,
                                plot_model_comparison=False,
                                confusion_matrix=False,
                                plot_confusion_heatmap=False,
                                plot_roc_curve=False,
                                plot_precision_recall_curve=False
                            ),
                            "group1_elasticnet": rs.ExperimentDirectory(
                                save_model=True,
                                evaluate_model=True,
                                evaluate_model_cv=False,
                                compare_models=False,
                                plot_pred_vs_obs=True,
                                plot_learning_curve=True,
                                plot_feature_importance=True,
                                plot_residuals=False,
                                plot_model_comparison=False,
                                confusion_matrix=False,
                                plot_confusion_heatmap=False,
                                plot_roc_curve=False,
                                plot_precision_recall_curve=False
                            ),
                        },
                        scaler_exists=False,
                        split_distribution_exists=True,
                        hist_box_plot_exists=True,
                        pie_plot_exists=True,
                        categorical_stats_json_exists=True,
                        continuous_stats_json_exists=True,
                        correlation_matrix_exists=True
                    )
                }
            ),
            "group2": rs.ExperimentGroupDirectory(
                datasets={
                    "mixed_features_regression": rs.DatasetDirectory(
                        experiments={
                            "group2_linear": rs.ExperimentDirectory(
                                save_model=True,
                                evaluate_model=True,
                                evaluate_model_cv=False,
                                compare_models=False,
                                plot_pred_vs_obs=True,
                                plot_learning_curve=True,
                                plot_feature_importance=True,
                                plot_residuals=False,
                                plot_model_comparison=False,
                                confusion_matrix=False,
                                plot_confusion_heatmap=False,
                                plot_roc_curve=False,
                                plot_precision_recall_curve=False
                            ),
                            "group2_ridge": rs.ExperimentDirectory(
                                save_model=True,
                                evaluate_model=True,
                                evaluate_model_cv=False,
                                compare_models=False,
                                plot_pred_vs_obs=True,
                                plot_learning_curve=True,
                                plot_feature_importance=True,
                                plot_residuals=False,
                                plot_model_comparison=False,
                                confusion_matrix=False,
                                plot_confusion_heatmap=False,
                                plot_roc_curve=False,
                                plot_precision_recall_curve=False
                            ),
                        },
                        scaler_exists=False,
                        split_distribution_exists=True,
                        hist_box_plot_exists=True,
                        pie_plot_exists=True,
                        categorical_stats_json_exists=True,
                        continuous_stats_json_exists=True,
                        correlation_matrix_exists=True
                    )
                }
            ),
            "group3": rs.ExperimentGroupDirectory(
                datasets={
                    "mixed_features_regression": rs.DatasetDirectory(
                        experiments={
                            "group3_linear": rs.ExperimentDirectory(
                                save_model=True,
                                evaluate_model=True,
                                evaluate_model_cv=False,
                                compare_models=False,
                                plot_pred_vs_obs=True,
                                plot_learning_curve=True,
                                plot_feature_importance=True,
                                plot_residuals=False,
                                plot_model_comparison=False,
                                confusion_matrix=False,
                                plot_confusion_heatmap=False,
                                plot_roc_curve=False,
                                plot_precision_recall_curve=False
                            ),
                            "group3_ridge": rs.ExperimentDirectory(
                                save_model=True,
                                evaluate_model=True,
                                evaluate_model_cv=False,
                                compare_models=False,
                                plot_pred_vs_obs=True,
                                plot_learning_curve=True,
                                plot_feature_importance=True,
                                plot_residuals=False,
                                plot_model_comparison=False,
                                confusion_matrix=False,
                                plot_confusion_heatmap=False,
                                plot_roc_curve=False,
                                plot_precision_recall_curve=False
                            ),
                        },
                        scaler_exists=True,
                        split_distribution_exists=True,
                        hist_box_plot_exists=True,
                        pie_plot_exists=True,
                        categorical_stats_json_exists=True,
                        continuous_stats_json_exists=True,
                        correlation_matrix_exists=True
                    )
                }
            )
        },
        html_report=rs.HTMLReport(
            index_exists=True,
            index_css_exists=True,
            experiment_css_exists=True,
            dataset_css_exists=True,
            dataset_pages={
                "group1_mixed_features_regression": True,
                "group2_mixed_features_regression": True,
                "group3_mixed_features_regression": True,
            },
            experiment_pages={
                "mixed_features_regression_group1_lasso": True,
                "mixed_features_regression_group1_elasticnet": True,
                "mixed_features_regression_group2_linear": True,
                "mixed_features_regression_group2_ridge": True,
                "mixed_features_regression_group3_linear": True,
                "mixed_features_regression_group3_ridge": True,
            }
        )
    )


class TestResultStructure:
    """Test the ResultStructure class matches the expected strucutre."""
    def test_get_html_pages_from_one_group(self, one_experiment_group):
        datasets, experiments = rs.ResultStructure.get_html_pages_from_groups(
            one_experiment_group
        )
        expected_datasets = {
            "group1_dataset1": True,
            "group1_dataset2": True,
        }
        expected_experiments = {
            "dataset1_group1_linear": True,
            "dataset1_group1_rf": True,
            "dataset2_group1_linear": True,
            "dataset2_group1_ridge": True,
            "dataset2_group1_lasso": True,
        }

        assert datasets == expected_datasets
        assert experiments == expected_experiments

    def test_get_html_pages_from_two_groups(self, two_experiment_groups):
        datasets, experiments = rs.ResultStructure.get_html_pages_from_groups(
            two_experiment_groups
        )
        expected_datasets = {
            "group1_dataset1": True,
            "group1_dataset2": True,
            "group2_dataset3": True,
        }
        expected_experiments = {
            "dataset1_group1_linear": True,
            "dataset1_group1_rf": True,
            "dataset2_group1_linear": True,
            "dataset2_group1_ridge": True,
            "dataset2_group1_lasso": True,
            "dataset3_group2_ridge": True,
            "dataset3_group2_mlp": True,
        }

        assert datasets == expected_datasets
        assert experiments == expected_experiments

    def test_get_experiment_groups(self, experiment_groups):
        workflow_methods = set(
            ["save_model", "evaluate_model_cv", "plot_feature_importance",
            "plot_residuals"]
        )
        experiment_group_directories = rs.ResultStructure.get_experiment_groups(
            experiment_groups, workflow_methods, False, False
        )

        expected_structure = {
            "group1": rs.ExperimentGroupDirectory(
                datasets={
                    "dataset1": rs.DatasetDirectory(
                        experiments={
                            "group1_linear": rs.ExperimentDirectory(
                                save_model=True,
                                evaluate_model=False,
                                evaluate_model_cv=True,
                                compare_models=False,
                                plot_pred_vs_obs=False,
                                plot_learning_curve=False,
                                plot_feature_importance=True,
                                plot_residuals=True,
                                plot_model_comparison=False,
                                confusion_matrix=False,
                                plot_confusion_heatmap=False,
                                plot_roc_curve=False,
                                plot_precision_recall_curve=False,
                            )
                        },
                        scaler_exists=False,
                        split_distribution_exists=True,
                        hist_box_plot_exists=True,
                        pie_plot_exists=False,
                        categorical_stats_json_exists=False,
                        continuous_stats_json_exists=True,
                        correlation_matrix_exists=True
                    )
                }
            ),
            "group2": rs.ExperimentGroupDirectory(
                datasets={
                    "dataset1": rs.DatasetDirectory(
                        experiments={
                            "group2_linear": rs.ExperimentDirectory(
                                save_model=True,
                                evaluate_model=False,
                                evaluate_model_cv=True,
                                compare_models=False,
                                plot_pred_vs_obs=False,
                                plot_learning_curve=False,
                                plot_feature_importance=True,
                                plot_residuals=True,
                                plot_model_comparison=False,
                                confusion_matrix=False,
                                plot_confusion_heatmap=False,
                                plot_roc_curve=False,
                                plot_precision_recall_curve=False,
                            )
                        },
                        scaler_exists=True,
                        split_distribution_exists=True,
                        hist_box_plot_exists=True,
                        pie_plot_exists=False,
                        categorical_stats_json_exists=False,
                        continuous_stats_json_exists=True,
                        correlation_matrix_exists=True
                    ),
                    "dataset2": rs.DatasetDirectory(
                        experiments={
                            "group2_linear": rs.ExperimentDirectory(
                                save_model=True,
                                evaluate_model=False,
                                evaluate_model_cv=True,
                                compare_models=False,
                                plot_pred_vs_obs=False,
                                plot_learning_curve=False,
                                plot_feature_importance=True,
                                plot_residuals=True,
                                plot_model_comparison=False,
                                confusion_matrix=False,
                                plot_confusion_heatmap=False,
                                plot_roc_curve=False,
                                plot_precision_recall_curve=False,
                            )
                        },
                        scaler_exists=True,
                        split_distribution_exists=True,
                        hist_box_plot_exists=True,
                        pie_plot_exists=False,
                        categorical_stats_json_exists=False,
                        continuous_stats_json_exists=True,
                        correlation_matrix_exists=True
                    )
                }
            ),
            "group3": rs.ExperimentGroupDirectory(
                datasets={
                    "dataset1": rs.DatasetDirectory(
                        experiments={
                            "group3_dtr": rs.ExperimentDirectory(
                                save_model=True,
                                evaluate_model=False,
                                evaluate_model_cv=True,
                                compare_models=False,
                                plot_pred_vs_obs=False,
                                plot_learning_curve=False,
                                plot_feature_importance=True,
                                plot_residuals=True,
                                plot_model_comparison=False,
                                confusion_matrix=False,
                                plot_confusion_heatmap=False,
                                plot_roc_curve=False,
                                plot_precision_recall_curve=False,
                            ),
                            "group3_rf": rs.ExperimentDirectory(
                                save_model=True,
                                evaluate_model=False,
                                evaluate_model_cv=True,
                                compare_models=False,
                                plot_pred_vs_obs=False,
                                plot_learning_curve=False,
                                plot_feature_importance=True,
                                plot_residuals=True,
                                plot_model_comparison=False,
                                confusion_matrix=False,
                                plot_confusion_heatmap=False,
                                plot_roc_curve=False,
                                plot_precision_recall_curve=False,
                            ),
                        },
                        scaler_exists=False,
                        split_distribution_exists=True,
                        hist_box_plot_exists=True,
                        pie_plot_exists=True,
                        categorical_stats_json_exists=True,
                        continuous_stats_json_exists=True,
                        correlation_matrix_exists=True
                    )
                }
            ),
            "group4": rs.ExperimentGroupDirectory(
                datasets={
                    "dataset2": rs.DatasetDirectory(
                        experiments={
                            "group4_knn": rs.ExperimentDirectory(
                                save_model=True,
                                evaluate_model=False,
                                evaluate_model_cv=True,
                                compare_models=False,
                                plot_pred_vs_obs=False,
                                plot_learning_curve=False,
                                plot_feature_importance=True,
                                plot_residuals=True,
                                plot_model_comparison=False,
                                confusion_matrix=False,
                                plot_confusion_heatmap=False,
                                plot_roc_curve=False,
                                plot_precision_recall_curve=False,
                            )
                        },
                        scaler_exists=True,
                        split_distribution_exists=True,
                        hist_box_plot_exists=True,
                        pie_plot_exists=True,
                        categorical_stats_json_exists=True,
                        continuous_stats_json_exists=True,
                        correlation_matrix_exists=True
                    )
                }
            ),
        }

        assert experiment_group_directories == expected_structure

    def test_get_workflow_methods_from_dir(self, temp_experiment_dir):
        workflow_methods = rs.ResultStructure.get_workflow_methods_from_dir(
            temp_experiment_dir
        )
        assert "save_model" in workflow_methods
        assert "evaluate_model" in workflow_methods
        assert "plotting_method" in workflow_methods
        assert len(workflow_methods) == 3

    def test_from_config(self, expected_result_structure):
        result_structure_test_dir = (
            pathlib.Path(__file__).parent / "result_structure_test"
        )
        os.chdir(result_structure_test_dir)
        settings_module = importlib.import_module(
            "tests.unit_tests.utility.result_structure_test.settings"
        )
        config_manager = settings_module.create_configuration()
        workflow_path = result_structure_test_dir / "workflows/workflow.py"

        result_structure = rs.ResultStructure.from_config(
            config_manager, workflow_path
        )

        assert isinstance(result_structure, rs.ResultStructure)
        assert result_structure == expected_result_structure

    def test_from_directory(self, expected_result_structure):
        result_path = (
            pathlib.Path(__file__).parent /
            "result_structure_test/results/result_structure_test"
        )
        result_structure = rs.ResultStructure.from_directory(result_path)

        assert isinstance(result_structure, rs.ResultStructure)
        assert result_structure == expected_result_structure
