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

    def _get_base(self, method_name: str) -> Dict[str, Any]:
        """Base information that is common to all metadata
        
        Parameters
        ----------
        method_name : str
            The name of the calling method

        Returns
        -------
        dict
            Base metadata including timestamp, method name, and type
        """
        return {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": method_name,
        }

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
        metadata = self._get_base(method_name)
        metadata["type"] = "model"
        metadata["models"] = {}
        metadata["is_test"] = str(is_test)

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
        metadata = self._get_base(method_name)
        metadata["type"] = "dataset"
        metadata["dataset"] = dataset_name
        metadata["group"] = group_name
        return metadata
