# settings.py
from brisk.configuration.configuration import Configuration, ConfigurationManager
from brisk.data.preprocessing import ScalingPreprocessor, CategoricalEncodingPreprocessor

def create_configuration() -> ConfigurationManager:
    config = Configuration(
        default_algorithms = ["linear", "ridge"],
        categorical_features={
            "mixed_features_regression.csv": [
                "categorical_0", "categorical_1", "categorical_2"
            ],
        }
    )

    config.add_experiment_group(
        name="group1",
        datasets=["mixed_features_regression.csv"],
        algorithms=["lasso", "elasticnet"],
        data_config={"preprocessors": [CategoricalEncodingPreprocessor(method="onehot")]}
    )
    config.add_experiment_group(
        name="group2",
        datasets=["mixed_features_regression.csv"],
        data_config={"preprocessors": [CategoricalEncodingPreprocessor(method="onehot")]}
    )
    config.add_experiment_group(
        name="group3",
        datasets=["mixed_features_regression.csv"],
        data_config={"preprocessors": [CategoricalEncodingPreprocessor(method="onehot"), ScalingPreprocessor(method="minmax")]}
    )
    return config.build()
