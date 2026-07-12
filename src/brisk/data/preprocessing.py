"""Public preprocessing API for the data pipeline.

The concrete preprocessor implementations depend on scikit-learn and therefore
live in the adapter layer
(``brisk.adapters.sklearn.preprocessor_adapter``). This module re-exports them
(together with the ``PreprocessorPort`` protocol) as the stable import location
used by :class:`brisk.data.data_manager.DataManager` and by user projects.

Every re-exported preprocessor implements :class:`PreprocessorPort`.

Classes
-------
PreprocessorPort
    Structural interface all preprocessors satisfy.
BasePreprocessor
    Abstract base providing the common preprocessor interface.
MissingDataPreprocessor
    Handles missing value imputation and removal strategies.
ScalingPreprocessor
    Scales numerical features using scikit-learn scalers.
CategoricalEncodingPreprocessor
    Encodes categorical features with multiple strategies.
FeatureSelectionPreprocessor
    Selects features using scikit-learn feature-selection algorithms.
"""

from brisk.ports.preprocessor import PreprocessorPort
from brisk.adapters.sklearn.preprocessor_adapter import (
    BasePreprocessor,
    CategoricalEncodingPreprocessor,
    FeatureSelectionPreprocessor,
    MissingDataPreprocessor,
    ScalingPreprocessor,
)

__all__ = [
    "PreprocessorPort",
    "BasePreprocessor",
    "MissingDataPreprocessor",
    "ScalingPreprocessor",
    "CategoricalEncodingPreprocessor",
    "FeatureSelectionPreprocessor",
]
