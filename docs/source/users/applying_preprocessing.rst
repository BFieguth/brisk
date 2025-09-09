.. _applying_preprocessing:

Applying Preprocessing
======================

Brisk provides built-in preprocessing capabilities for missing data handling, 
scaling, categorical encoding, and feature selection. You can configure preprocessing 
in your ``DataManager`` or use experiment groups to apply different 
preprocessing strategies. See 
:doc:`Using Experiment Groups </users/using_experiment_groups>` for more details.

.. note::
   For the ``DataManager`` documentation, see the :doc:`DataManager </api/data/data_manager>` API reference.


Built-in Preprocessors
----------------------

Brisk includes several built-in preprocessing classes with different methods:

**Missing Data Handling** (``MissingDataPreprocessor``):

- ``strategy="drop_rows"``: Remove rows with missing values
- ``strategy="impute"``: Fill missing values (``impute_method="mean"``, ``"median"``, ``"mode"``, ``"constant"``)

**Categorical Encoding** (``CategoricalEncodingPreprocessor``):

- ``method="ordinal"``: Ordinal encoding (preserves order)
- ``method="onehot"``: One-hot encoding (creates binary columns)
- ``method="label"``: Label encoding (assigns integers)
- ``method="cyclic"``: Cyclic encoding (for circular features)
- ``method="threshold"``: Threshold encoding (requires cutoffs parameter, e.g., ``cutoffs=[20, 40]``)
- ``method={"col1": "onehot", "target": "label"}``: Column-specific encoding (can encode target variable)

**Scaling** (``ScalingPreprocessor``):

- ``method="standard"``: StandardScaler (mean=0, std=1)
- ``method="minmax"``: MinMaxScaler (scales to 0-1 range)
- ``method="robust"``: RobustScaler (uses median and IQR)
- ``method="maxabs"``: MaxAbsScaler (scales by max absolute value)
- ``method="normalizer"``: Normalizer (scales individual samples to unit norm)

**Feature Selection** (``FeatureSelectionPreprocessor``):

- ``method="selectkbest"``: Select K best features using statistical tests
- ``method="rfecv"``: Recursive feature elimination with cross-validation
- ``method="sequential"``: Sequential feature selection (forward/backward)

All feature selection methods require ``n_features_to_select`` parameter. The ``rfecv`` and ``sequential`` methods also require ``algorithm_config`` parameter.


Configuring Preprocessors
--------------------------

Configure preprocessors in your ``data.py`` file by adding them to the ``DataManager`` 
constructor. Preprocessors are applied in the order they appear in the list:

.. code-block:: python

    from brisk.data.data_manager import DataManager
    from brisk.data.preprocessing import (
        MissingDataPreprocessor,
        CategoricalEncodingPreprocessor, 
        ScalingPreprocessor,
        FeatureSelectionPreprocessor
    )

    data_manager = DataManager(
        test_size=0.2,
        split_method="shuffle",
        preprocessors=[
            MissingDataPreprocessor(strategy="impute", impute_method="mean"),
            CategoricalEncodingPreprocessor(method="onehot"),
            ScalingPreprocessor(method="standard"),
            FeatureSelectionPreprocessor(method="selectkbest", n_features_to_select=10)
        ]
    )

This pipeline will: handle missing values → encode categories → scale features → select top features.


Pipeline Order Considerations
-----------------------------

The order of preprocessors in your pipeline is critical and follows these guidelines:

1. **Missing Data** - Handle missing values first before other transformations
2. **Categorical Encoding** - Encode categorical features before scaling  
3. **Scaling** - Apply scaling after encoding to avoid issues with categorical data
4. **Feature Selection** - Select features last, after all transformations are complete

.. important::
   Incorrect preprocessing order can lead to data leakage or unexpected results. 
   Always handle missing data first and feature selection last.

