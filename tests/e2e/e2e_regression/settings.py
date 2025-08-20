
from brisk.configuration.configuration import Configuration
from brisk.data.preprocessing import (
    ScalingPreprocessor, 
    CategoricalEncodingPreprocessor, 
    FeatureSelectionPreprocessor,
    MissingDataPreprocessor
)

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