# settings.py
from brisk.configuration.configuration import Configuration, ConfigurationManager

def create_configuration() -> ConfigurationManager:
    config = Configuration(
        default_algorithms = ["linear", "ridge"],
    )

    config.add_experiment_group(
        name="group1",
        datasets=["mixed_features_regression.csv"],
        algorithms=["lasso", "elasticnet"]
    )
    config.add_experiment_group(
        name="group2",
        datasets=["mixed_features_regression.csv"]
    )
    config.add_experiment_group(
        name="group3",
        datasets=["mixed_features_regression.csv"],
        data_config={"scale_method": "minmax"}
    )
    return config.build()
