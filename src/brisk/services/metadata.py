"""Metadata related utilities for evaluators"""

from typing import Dict, Any, Union, List
import datetime

from sklearn import base

from brisk.services import base as base_service
from brisk.configuration import algorithm_wrapper

class MetadataService(base_service.BaseService):
    """Metadata generation.
    
    Parameters
    ----------
    name : str
        The name of the service
    algorithm_config : AlgorithmCollection
        The algorithm configuration

    Attributes
    ----------
    name : str
        The name of the service
    algorithm_config : AlgorithmCollection
        The algorithm configuration
    """
    def __init__(
        self,
        name,
        algorithm_config: algorithm_wrapper.AlgorithmCollection
    ):
        super().__init__(name)
        self.algorithm_config = algorithm_config

    def get_model(
        self,
        models: Union[base.BaseEstimator, List[base.BaseEstimator]],
        method_name: str,
        is_test: bool = False
    ) -> Dict[str, Any]:
        """Generate metadata for a model evaluation.

        Parameters
        ----------
        models : BaseEstimator or list of BaseEstimator
            The models to include in metadata.

        method_name (str): 
            The name of the calling method

        is_test (bool, optional): 
            Whether the data is test data, by default False

        Returns
        -------
        dict
            Metadata including timestamp, method name, algorith wrapper name,
            and algorithm display name
        """
        metadata = {
            "type": "model",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": method_name,
            "models": {},
            "is_test": str(is_test)
        }
        if not isinstance(models, list):
            models = [models]

        for model in models:
            wrapper = self.algorithm_config[model.wrapper_name]
            metadata["models"][wrapper.name] = wrapper.display_name

        return metadata

    def get_dataset(
        self,
        method_name: str,
        dataset_name: str,
        group_name: str
    ) -> Dict[str, Any]:
        """Generate metadata for a dataset evaluation.

        Parameters
        ----------
        method_name : str
            The name of the calling method
        dataset_name : str
            The name of the dataset
        group_name : str
            The name of the group

        Returns
        -------
        Dict[str, Any]
            Metadata including timestamp, method name, dataset name, and group
            name
        """
        return {
            "type": "dataset",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": method_name,
            "dataset": dataset_name,
            "group": group_name
        }
