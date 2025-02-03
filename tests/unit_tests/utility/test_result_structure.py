"""Test the ResultStructure class."""
import importlib
import json
import os
import pathlib
import pickle
import tempfile
from unittest import mock

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import brisk.utility.result_structure as rs
from brisk.configuration import experiment_group, configuration

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
                    datasets=["dataset1.csv"],
                    algorithms=["linear"],
                ),
                experiment_group.ExperimentGroup(
                    name="group2",
                    datasets=["dataset1.csv", "dataset2.csv"],
                    algorithms=["linear"],
                    data_config={
                        "scale_method": "minmax"
                    },
                ),
                experiment_group.ExperimentGroup(
                    name="group3",
                    datasets=["dataset1.csv"],
                    algorithms=["dtr", "rf"],
                ),
                experiment_group.ExperimentGroup(
                    name="group4",
                    datasets=["dataset2.csv"],
                    algorithms=["knn"],
                    data_config={"scale_method": "minmax"},
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


@pytest.fixture
def mixed_data_config():
    briskconfig_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'result_structure_test'
    )    
    os.chdir(briskconfig_dir)

    config = configuration.Configuration(
        default_algorithms=["linear"],
        categorical_features={
            "mixed_features_regression.csv": [
                "categorical_0", "categorical_1", "categorical_2"
            ]
        }
    )
    config.add_experiment_group(
        name="test_group_1",
        datasets=["mixed_features_regression.csv"],
        algorithms=["linear", "lasso", "mlp"]
    )
    return config.build()


@pytest.fixture
def continuous_data_config(request):
    briskconfig_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'result_structure_test'
    )    
    os.chdir(briskconfig_dir)

    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "result_structure_test/datasets/mixed_features_regression.csv"
    )
    temp_csv_path = os.path.join(
        briskconfig_dir, "datasets/continuous_features_regression.csv"
    )

    data = pd.read_csv(csv_path)
    data = data.drop(
        columns=["categorical_0", "categorical_1", "categorical_2"]
    )
    data.to_csv(temp_csv_path, index=False)

    def cleanup():
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)
    request.addfinalizer(cleanup)

    config = configuration.Configuration(
        default_algorithms=["linear"]
    )
    config.add_experiment_group(
        name="test_group",
        datasets=["continuous_features_regression.csv"],
        algorithms=["ridge", "knn", "svr"]
    )
    return config.build()


@pytest.fixture
def categorical_data_config(request):
    briskconfig_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'result_structure_test'
    )    
    os.chdir(briskconfig_dir)

    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "result_structure_test/datasets/mixed_features_regression.csv"
    )
    temp_csv_path = os.path.join(
        briskconfig_dir, "datasets/categorical_features_regression.csv"
    )

    data = pd.read_csv(csv_path)
    data = data[["categorical_0", "categorical_1", "categorical_2", "target"]]
    data.to_csv(temp_csv_path, index=False)

    def cleanup():
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)
    request.addfinalizer(cleanup)

    config = configuration.Configuration(
        default_algorithms=["linear"],
        categorical_features={
            "categorical_features_regression.csv": [
                "categorical_0", "categorical_1", "categorical_2"
            ]
        }
    )
    config.add_experiment_group(
        name="categorical_group",
        datasets=["categorical_features_regression.csv"],
        algorithms=["knn", "xtree", "dtr"]
    )
    return config.build()


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

    def test_get_experiment_groups_mixed(self, mixed_data_config):
        expected_experiment_groups = {
            'test_group_1': rs.ExperimentGroupDirectory(
                datasets={
                    'mixed_features_regression': rs.DatasetDirectory(
                        experiments={
                            'test_group_1_linear': rs.ExperimentDirectory(
                                save_model=True,
                                evaluate_model=True,
                                evaluate_model_cv=False,
                                compare_models=False,
                                plot_pred_vs_obs=False,
                                plot_learning_curve=False,
                                plot_feature_importance=True,
                                plot_residuals=False,
                                plot_model_comparison=False,
                                confusion_matrix=False,
                                plot_confusion_heatmap=False,
                                plot_roc_curve=False,
                                plot_precision_recall_curve=False
                            ),
                            'test_group_1_lasso': rs.ExperimentDirectory(
                                save_model=True,
                                evaluate_model=True,
                                evaluate_model_cv=False,
                                compare_models=False,
                                plot_pred_vs_obs=False,
                                plot_learning_curve=False,
                                plot_feature_importance=True,
                                plot_residuals=False,
                                plot_model_comparison=False,
                                confusion_matrix=False,
                                plot_confusion_heatmap=False,
                                plot_roc_curve=False,
                                plot_precision_recall_curve=False
                            ),
                            'test_group_1_mlp': rs.ExperimentDirectory(
                                save_model=True,
                                evaluate_model=True,
                                evaluate_model_cv=False,
                                compare_models=False,
                                plot_pred_vs_obs=False,
                                plot_learning_curve=False,
                                plot_feature_importance=True,
                                plot_residuals=False,
                                plot_model_comparison=False,
                                confusion_matrix=False,
                                plot_confusion_heatmap=False,
                                plot_roc_curve=False,
                                plot_precision_recall_curve=False
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
            )
        }
        workflow_methods = set(
            ["save_model", "evaluate_model", "plot_feature_importance"]
        )
        base_scale_method = True
        actual_experiment_groups = rs.ResultStructure.get_experiment_groups(
            mixed_data_config.experiment_groups, workflow_methods,
            base_scale_method, mixed_data_config.categorical_features,
            mixed_data_config.data_managers
        )
        assert expected_experiment_groups == actual_experiment_groups

    def test_get_experiment_groups_continuous(self, continuous_data_config):
        expected_experiment_groups = {
            'test_group': rs.ExperimentGroupDirectory(
                datasets={
                    'continuous_features_regression': rs.DatasetDirectory(
                        experiments={
                            'test_group_ridge': rs.ExperimentDirectory(
                                save_model=True,
                                evaluate_model=False,
                                evaluate_model_cv=True,
                                compare_models=False,
                                plot_pred_vs_obs=False,
                                plot_learning_curve=True,
                                plot_feature_importance=True,
                                plot_residuals=True,
                                plot_model_comparison=False,
                                confusion_matrix=False,
                                plot_confusion_heatmap=False,
                                plot_roc_curve=False,
                                plot_precision_recall_curve=False
                            ),
                            'test_group_knn': rs.ExperimentDirectory(
                                save_model=True,
                                evaluate_model=False,
                                evaluate_model_cv=True,
                                compare_models=False,
                                plot_pred_vs_obs=False,
                                plot_learning_curve=True,
                                plot_feature_importance=True,
                                plot_residuals=True,
                                plot_model_comparison=False,
                                confusion_matrix=False,
                                plot_confusion_heatmap=False,
                                plot_roc_curve=False,
                                plot_precision_recall_curve=False
                            ),
                            'test_group_svr': rs.ExperimentDirectory(
                                save_model=True,
                                evaluate_model=False,
                                evaluate_model_cv=True,
                                compare_models=False,
                                plot_pred_vs_obs=False,
                                plot_learning_curve=True,
                                plot_feature_importance=True,
                                plot_residuals=True,
                                plot_model_comparison=False,
                                confusion_matrix=False,
                                plot_confusion_heatmap=False,
                                plot_roc_curve=False,
                                plot_precision_recall_curve=False
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
            )
        }
        workflow_methods = set(
            ["save_model", "evaluate_model_cv", "plot_feature_importance", 
             "plot_residuals", "plot_learning_curve"]
        )
        base_scale_method = False
        actual_experiment_groups = rs.ResultStructure.get_experiment_groups(
            continuous_data_config.experiment_groups, workflow_methods,
            base_scale_method, continuous_data_config.categorical_features,
            continuous_data_config.data_managers
        )
        assert actual_experiment_groups == expected_experiment_groups

    def test_get_experiment_groups_categorical(self, categorical_data_config):
        expected_experiment_groups = {
            'categorical_group': rs.ExperimentGroupDirectory(
                datasets={
                    'categorical_features_regression': rs.DatasetDirectory(
                        experiments={
                            'categorical_group_knn': rs.ExperimentDirectory(
                                save_model=True,
                                evaluate_model=True,
                                evaluate_model_cv=False,
                                compare_models=False,
                                plot_pred_vs_obs=False,
                                plot_learning_curve=False,
                                plot_feature_importance=False,
                                plot_residuals=False,
                                plot_model_comparison=False,
                                confusion_matrix=False,
                                plot_confusion_heatmap=True,
                                plot_roc_curve=True,
                                plot_precision_recall_curve=False
                            ),
                            'categorical_group_xtree': rs.ExperimentDirectory(
                                save_model=True,
                                evaluate_model=True,
                                evaluate_model_cv=False,
                                compare_models=False,
                                plot_pred_vs_obs=False,
                                plot_learning_curve=False,
                                plot_feature_importance=False,
                                plot_residuals=False,
                                plot_model_comparison=False,
                                confusion_matrix=False,
                                plot_confusion_heatmap=True,
                                plot_roc_curve=True,
                                plot_precision_recall_curve=False
                            ),
                            'categorical_group_dtr': rs.ExperimentDirectory(
                                save_model=True,
                                evaluate_model=True,
                                evaluate_model_cv=False,
                                compare_models=False,
                                plot_pred_vs_obs=False,
                                plot_learning_curve=False,
                                plot_feature_importance=False,
                                plot_residuals=False,
                                plot_model_comparison=False,
                                confusion_matrix=False,
                                plot_confusion_heatmap=True,
                                plot_roc_curve=True,
                                plot_precision_recall_curve=False
                            )
                        },
                        scaler_exists=True,
                        split_distribution_exists=True,
                        hist_box_plot_exists=False,
                        pie_plot_exists=True,
                        categorical_stats_json_exists=True,
                        continuous_stats_json_exists=False,
                        correlation_matrix_exists=False
                    )
                }
            )
        }
        workflow_methods = set(
            ["save_model", "evaluate_model", "plot_confusion_heatmap", 
             "plot_roc_curve"]
        )
        base_scale_method = True
        actual_experiment_groups = rs.ResultStructure.get_experiment_groups(
            categorical_data_config.experiment_groups, workflow_methods,
            base_scale_method, categorical_data_config.categorical_features,
            categorical_data_config.data_managers
        )
        assert expected_experiment_groups == actual_experiment_groups
