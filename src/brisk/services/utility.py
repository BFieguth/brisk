"""Utility service providing helper functions for evaluators and
cross-validation.

This module provides comprehensive utility functionality for the Brisk package,
including cross-validation splitters, algorithm wrapper management, group index
handling, and plot settings management. It serves as a centralized utility
service for common operations needed throughout the evaluation pipeline.

The UtilityService provides methods for:
- Managing algorithm configurations and wrappers
- Handling grouped data indices for training and test sets
- Creating appropriate cross-validation splitters based on data characteristics
- Managing plot settings and configurations

Classes
-------
UtilityService
    Main utility service class providing helper functions

Examples
--------
>>> from brisk.services.utility import UtilityService
>>> from brisk.configuration import AlgorithmCollection
>>> 
>>> # Create utility service
>>> utility_service = UtilityService("utility", group_index_train, group_index_test)
>>> utility_service.set_algorithm_config(algorithm_config)
>>> 
>>> # Get algorithm wrapper
>>> wrapper = utility_service.get_algo_wrapper("RandomForest")
>>> 
>>> # Get cross-validation splitter
>>> splitter, indices = utility_service.get_cv_splitter(y, cv=5)
"""

from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import sklearn.model_selection as model_select

from brisk.configuration import algorithm_wrapper, algorithm_collection
from brisk.services import base
from brisk.theme.plot_settings import PlotSettings

class UtilityService(base.BaseService):
    """Utility service providing helper functions for evaluators and
    cross-validation.
    
    This service provides comprehensive utility functionality for the Brisk
    package, including cross-validation splitters, algorithm wrapper management,
    group index handling, and plot settings management. It serves as a
    centralized utility service for common operations needed throughout the
    evaluation pipeline.
    
    The service manages algorithm configurations, handles grouped data indices,
    creates appropriate cross-validation splitters based on data
    characteristics, and provides access to plot settings and configurations.
    
    Attributes
    ----------
    algorithm_config : Optional[algorithm_collection.AlgorithmCollection]
        The algorithm configuration containing algorithm wrappers
    group_index_train : Optional[Dict[str, np.ndarray]]
        The group index for the training data
    group_index_test : Optional[Dict[str, np.ndarray]]
        The group index for the test data
    data_has_groups : bool
        Boolean flag indicating if the data has group information
    plot_settings : PlotSettings
        The plot settings configuration
        
    Notes
    -----
    The service automatically detects if data has groups based on the presence
    of group indices. It provides appropriate cross-validation splitters based
    on data characteristics (categorical vs continuous, grouped vs ungrouped).
    
    Examples
    --------
    >>> from brisk.services.utility import UtilityService
    >>> from brisk.configuration import AlgorithmCollection
    >>> 
    >>> # Create utility service
    >>> utility_service = UtilityService("utility", group_index_train, group_index_test)
    >>> utility_service.set_algorithm_config(algorithm_config)
    >>> 
    >>> # Get algorithm wrapper
    >>> wrapper = utility_service.get_algo_wrapper("RandomForest")
    >>> 
    >>> # Get cross-validation splitter
    >>> splitter, indices = utility_service.get_cv_splitter(y, cv=5)
    """
    def __init__(
        self,
        name: str,
        group_index_train: Optional[Dict[str, np.ndarray]],
        group_index_test: Optional[Dict[str, np.ndarray]]
    ) -> None:
        """Initialize the utility service.
        
        This constructor sets up the utility service with the specified name
        and group indices. It initializes all attributes and sets up the
        plot settings with default values.
        
        Parameters
        ----------
        name : str
            The name identifier for this service
        group_index_train : Optional[Dict[str, np.ndarray]]
            The group index for the training data
        group_index_test : Optional[Dict[str, np.ndarray]]
            The group index for the test data
            
        Notes
        -----
        The service automatically detects if data has groups based on the
        presence of both group indices. The algorithm configuration is
        initially set to None and must be set separately.
        """
        super().__init__(name)
        self.algorithm_config = None
        self.group_index_train = None
        self.group_index_test = None
        self.data_has_groups = False
        self.set_split_indices(
            group_index_train, group_index_test
        )
        self.plot_settings = PlotSettings()

    def set_split_indices(
        self,
        group_index_train: Optional[Dict[str, np.ndarray]],
        group_index_test: Optional[Dict[str, np.ndarray]],
    ) -> None:
        """Set the split indices for grouped data.

        This method sets the group indices for training and test data and
        automatically determines if the data has groups based on the presence
        of both indices.

        Parameters
        ----------
        group_index_train : Optional[Dict[str, np.ndarray]]
            The group index for the training data
        group_index_test : Optional[Dict[str, np.ndarray]]
            The group index for the test data

        Notes
        -----
        The data_has_groups flag is set to True only if both group indices
        are provided (not None). This flag is used by other methods to
        determine the appropriate cross-validation strategy.
        """
        self.group_index_train = group_index_train
        self.group_index_test = group_index_test
        if group_index_train is not None and group_index_test is not None:
            self.data_has_groups = True
        else:
            self.data_has_groups = False

    def set_algorithm_config(
        self,
        algorithm_config: algorithm_collection.AlgorithmCollection
    ) -> None:
        """Set the algorithm configuration.

        This method sets the algorithm configuration that contains all
        algorithm wrappers and their configurations.

        Parameters
        ----------
        algorithm_config : algorithm_collection.AlgorithmCollection
            The algorithm configuration containing algorithm wrappers

        Notes
        -----
        The algorithm configuration is required for accessing algorithm
        wrappers through the get_algo_wrapper() method.
        """
        self.algorithm_config = algorithm_config

    def get_algo_wrapper(
        self,
        wrapper_name: str
    ) -> algorithm_wrapper.AlgorithmWrapper:
        """Get the AlgorithmWrapper instance.

        This method retrieves an algorithm wrapper from the algorithm
        configuration by its name.

        Parameters
        ----------
        wrapper_name : str
            The name of the AlgorithmWrapper to retrieve

        Returns
        -------
        algorithm_wrapper.AlgorithmWrapper
            The AlgorithmWrapper instance

        Raises
        ------
        KeyError
            If the wrapper name is not found in the algorithm configuration
        AttributeError
            If the algorithm configuration is not set

        Examples
        --------
        >>> utility_service = UtilityService("utility", None, None)
        >>> utility_service.set_algorithm_config(algorithm_config)
        >>> wrapper = utility_service.get_algo_wrapper("RandomForest")
        >>> print(wrapper.display_name)
        """
        return self.algorithm_config[wrapper_name]

    def get_group_index(self, is_test: bool) -> Optional[Dict[str, np.ndarray]]:
        """Get the group index for the training or test data.

        This method returns the appropriate group index based on whether
        the data is from the training or test set. If the data doesn't
        have groups, it returns None.

        Parameters
        ----------
        is_test : bool
            Whether the data is test data

        Returns
        -------
        Optional[Dict[str, np.ndarray]]
            The group index for the training or test data, or None if
            the data doesn't have groups

        Examples
        --------
        >>> utility_service = UtilityService("utility", group_index_train, group_index_test)
        >>> train_groups = utility_service.get_group_index(is_test=False)
        >>> test_groups = utility_service.get_group_index(is_test=True)
        """
        if self.data_has_groups:
            if is_test:
                return self.group_index_test
            return self.group_index_train
        return None

    def get_cv_splitter(
        self,
        y: pd.Series,
        cv: int = 5,
        num_repeats: Optional[int] = None
    ) -> Tuple[model_select.BaseCrossValidator, Optional[np.ndarray]]:
        """Get the cross-validator splitter for the data.

        This method creates an appropriate cross-validation splitter based on
        the data characteristics. It considers whether the data is categorical
        or continuous, whether it has groups, and whether repeated splitting
        is requested.

        Parameters
        ----------
        y : pd.Series
            The target variable used to determine data characteristics
        cv : int, default=5
            The number of folds or splits to create
        num_repeats : Optional[int], default=None
            The number of repeats for repeated cross-validation

        Returns
        -------
        Tuple[model_select.BaseCrossValidator, Optional[np.ndarray]]
            A tuple containing:
            - The cross-validator splitter appropriate for the data
            - The group index array (None if no groups)

        Notes
        -----
        The method automatically selects the appropriate splitter:
        - For grouped data: StratifiedGroupKFold or GroupKFold
        - For ungrouped data: StratifiedKFold or KFold
        - With repeats: RepeatedStratifiedKFold or RepeatedKFold
        - Categorical detection: Based on unique value ratio (< 5%)

        Examples
        --------
        >>> utility_service = UtilityService("utility", group_index_train, group_index_test)
        >>> splitter, indices = utility_service.get_cv_splitter(y, cv=5)
        >>> for train_idx, val_idx in splitter.split(X, y, groups=indices):
        ...     # Use train_idx and val_idx for cross-validation
        """
        group_index = self.get_group_index(y.attrs["is_test"])

        is_categorical = False
        if y.nunique() / len(y) < 0.05:
            is_categorical = True

        if group_index:
            if is_categorical and num_repeats:
                self._other_services["logging"].logger.warning(
                    "No splitter for grouped data and repeated splitting, "
                    "using StratifiedGroupKFold instead."
                )
                splitter = model_select.StratifiedGroupKFold(n_splits=cv)
            elif not is_categorical and num_repeats:
                self._other_services["logging"].logger.warning(
                    "No splitter for grouped data and repeated splitting, "
                    "using GroupKFold instead."
                )
                splitter = model_select.GroupKFold(n_splits=cv)
            elif is_categorical:
                splitter = model_select.StratifiedGroupKFold(n_splits=cv)
            else:
                splitter = model_select.GroupKFold(n_splits=cv)

        else:
            if is_categorical and num_repeats:
                splitter = model_select.RepeatedStratifiedKFold(n_splits=cv)
            elif not is_categorical and num_repeats:
                splitter = model_select.RepeatedKFold(n_splits=cv)
            elif is_categorical:
                splitter = model_select.StratifiedKFold(n_splits=cv)
            else:
                splitter = model_select.KFold(n_splits=cv)

        if group_index:
            indices = group_index["indices"]
        else:
            indices = None

        return splitter, indices

    def set_plot_settings(self, plot_settings: PlotSettings) -> None:
        """Set the plot settings configuration.

        This method sets the plot settings that will be used for
        generating plots throughout the evaluation process.

        Parameters
        ----------
        plot_settings : PlotSettings
            The plot settings configuration

        Notes
        -----
        The plot settings control various aspects of plot generation
        including theme, colors, dimensions, and file output formats.
        """
        self.plot_settings = plot_settings

    def get_plot_settings(self) -> PlotSettings:
        """Get the current plot settings configuration.

        This method returns the current plot settings configuration
        that is being used for plot generation.

        Returns
        -------
        PlotSettings
            The current plot settings configuration

        Examples
        --------
        >>> utility_service = UtilityService("utility", None, None)
        >>> plot_settings = utility_service.get_plot_settings()
        >>> print(plot_settings.primary_color)
        """
        return self.plot_settings
