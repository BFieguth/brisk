from brisk.configuration.configuration import Configuration

from tests.e2e import e2e_setup

class TestEndToEnd:
    def test_regression_single(self):
        def create_configuration():
            config = Configuration(
                default_algorithms=["linear", "linear2", "lasso"],
                categorical_features={
                    ("mixed_features.db", "mixed_features_regression"): [
                        "categorical_0", "categorical_1", "categorical_2"
                    ]
                }
            )
            config.add_experiment_group(
                name="test_regression_single",
                description="Run short workflow with mixed features and a single algorithm",
                datasets=[("mixed_features.db", "mixed_features_regression")]
            )

            return config.build()

        test = e2e_setup.BaseE2ETest(
            test_name="test_regression_single_algo",
            project_name="e2e_regression",
            workflow_name="basic_single",
            datasets=["mixed_features.db"],
            create_configuration=create_configuration
        )

        try:
            test.setup()
            test.run()
            test.assert_result_strucure()

        finally:
            test.cleanup()

    def test_regression_full(self):
        def create_configuration():
            config = Configuration(
                default_algorithms=["lasso", "dtr", "knn"],
                categorical_features={
                    ("mixed_features.db", "mixed_features_regression"): [
                        "categorical_0", "categorical_1", "categorical_2"
                    ],
                    "categorical_features_regression.xlsx": [
                        "categorical_0", "categorical_1", "categorical_2", 
                        "categorical_3","categorical_4", "categorical_5", 
                        "categorical_6", "categorical_7","categorical_8", 
                        "categorical_9"
                    ],
                }
            )
            config.add_experiment_group(
                name="group1",
                datasets=["continuous_features_regression.csv"],
                data_config={"scale_method": "minmax"}
            )
            config.add_experiment_group(
                name="group2",
                datasets=[("mixed_features.db", "mixed_features_regression")],
                data_config={"scale_method": "standard"},
                algorithms=["linear", "elasticnet"]
            )
            config.add_experiment_group(
                name="group3",
                datasets=["categorical_features_regression.xlsx"],
                data_config={"scale_method": "normalizer"},
                algorithms=["linear", "ridge", "elasticnet", "knn"]
            )
            return config.build()
        
        test = e2e_setup.BaseE2ETest(
            test_name="test_regression_full",
            project_name="e2e_regression",
            workflow_name="full_single",
            datasets=[
                "mixed_features.db",
                "categorical_features_regression.xlsx",
                "continuous_features_regression.csv"
            ],
            create_configuration=create_configuration
        )

        try:
            test.setup()
            test.run()
            test.assert_result_strucure()

        finally:
            test.cleanup()

    def test_regression_multi(self):
        def create_configuration():
            config = Configuration(
                default_algorithms=[["linear", "elasticnet"]],
                categorical_features={
                    ("mixed_features.db", "mixed_features_regression"): [
                        "categorical_0", "categorical_1", "categorical_2"
                    ],
                    "categorical_features_regression.xlsx": [
                        "categorical_0", "categorical_1", "categorical_2", 
                        "categorical_3","categorical_4", "categorical_5", 
                        "categorical_6", "categorical_7","categorical_8", 
                        "categorical_9"
                    ],
                }
            )
            config.add_experiment_group(
                name="trees",
                datasets=[("mixed_features.db", "mixed_features_regression")],
                algorithms=[["dtr", "xtree"], ["dtr", "rf"]],
                data_config={"scale_method": "standard"}
            )
            config.add_experiment_group(
                name="linear",
                datasets=[
                    "continuous_features_regression.csv",
                    "categorical_features_regression.xlsx"
                ],
                algorithms=[["linear", "lasso"], ["linear", "ridge"]],
                data_config={"scale_method": "robust"}
            )
            config.add_experiment_group(
                name="other_algorithms",
                datasets=["categorical_features_regression.xlsx"],
                algorithms=[["knn", "svr"]],
                data_config={"scale_method": "maxabs"}
            )
            return config.build()

        test = e2e_setup.BaseE2ETest(
            test_name="test_regression_multi",
            project_name="e2e_regression",
            workflow_name="basic_multi",
            datasets=[
                "mixed_features.db",
                "categorical_features_regression.xlsx",
                "continuous_features_regression.csv"
            ],
            create_configuration=create_configuration
        )

        try:
            test.setup()
            test.run()
            test.assert_result_strucure()
        
        finally:
            test.cleanup()
