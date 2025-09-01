"""Miscellaneous utility methods for evaluators."""

from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import sklearn.model_selection as model_select
import plotnine as pn

from brisk.configuration import algorithm_wrapper
from brisk.services import base
from brisk.theme.plot_settings import PlotSettings

class UtilityService(base.BaseService):
    """Utility service with helper functions for the EvaluationManager.
    
    Parameters
    ----------
    name : str
        The name of the service
    algorithm_config : AlgorithmCollection
        The algorithm configuration
    group_index_train : Dict[str, np.array] | None
        The group index for the training data
    group_index_test : Dict[str, np.array] | None
        The group index for the test data
    """
    def __init__(
        self,
        name: str,
        algorithm_config: algorithm_wrapper.AlgorithmCollection,
        group_index_train: Dict[str, np.array] | None,
        group_index_test: Dict[str, np.array] | None
    ):
        super().__init__(name)
        self.algorithm_config = algorithm_config
        self.group_index_train = None
        self.group_index_test = None
        self.data_has_groups = False
        self.set_split_indices(
            group_index_train, group_index_test
        )
        self.plot_settings: PlotSettings = None

    def set_split_indices(
        self,
        group_index_train: Dict[str, np.array] | None,
        group_index_test: Dict[str, np.array] | None,
    ) -> None:
        """Set the split indices for grouped data.

        Parameters
        ----------
        group_index_train : Dict[str, np.array] | None
            The group index for the training data
        group_index_test : Dict[str, np.array] | None
            The group index for the test data

        Returns
        -------
        None
        """
        self.group_index_train = group_index_train
        self.group_index_test = group_index_test
        if group_index_train is not None and group_index_test is not None:
            self.data_has_groups = True
        else:
            self.data_has_groups = False

    def get_algo_wrapper(
        self,
        wrapper_name: str
    ) -> algorithm_wrapper.AlgorithmWrapper:
        """Get the AlgorithmWrapper instance.

        Parameters
        ----------
        wrapper_name : str
            The name of the AlgorithmWrapper to retrieve

        Returns
        -------
        AlgorithmWrapper
            The AlgorithmWrapper instance
        """
        return self.algorithm_config[wrapper_name]

    def get_group_index(self, is_test: bool) -> Dict[str, np.array]:
        """Get the group index for the training or test data.

        Parameters
        ----------
        is_test (bool): 
            Whether the data is test data.

        Returns
        -------
        Dict[str, np.array] | None
            The group index for the training or test data
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
    ) -> Tuple[model_select.BaseCrossValidator, np.array]:
        """Get the cross-validator splitter for the data.

        Parameters
        ----------
        y : pd.Series
            The target variable
        cv : int
            The number of folds or splits to create
        num_repeats : Optional[int]
            The number of repeats

        Returns
        -------
        Tuple[model_select.BaseCrossValidator, np.array]
            The cross-validator splitter and the group index
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

    def set_plot_settings(self, plot_settings: PlotSettings):
        self.plot_settings = plot_settings

    def get_plot_settings(self):
        return self.plot_settings
