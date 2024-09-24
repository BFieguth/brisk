from typing import Any, Dict, Optional, Type

class ModelWrapper:
    def __init__(self, 
                 name: str, 
                 model_class: Type, 
                 default_params: Optional[Dict[str, Any]] = None, 
                 hyperparam_grid: Optional[Dict[str, Any]] = None
    ):
        """
        Initializes the ModelWrapper with a model class, default parameters, and 
        a hyperparameter grid.

        Args:
            name (str): The name of the model.
            model_class (Type): The class of the model to be instantiated.
            default_params (Optional[Dict[str, Any]]): The default parameters to 
                pass to the model during instantiation.
            hyperparam_grid (Optional[Dict[str, Any]]): The hyperparameter grid 
                for model tuning.
        """
        self.name = name
        self.model_class = model_class
        self.default_params = default_params if default_params else {}
        self.hyperparam_grid = hyperparam_grid if hyperparam_grid else {}

    def instantiate(self) -> Any:
        """Instantiates the model with the default parameters.

        Returns:
            Any: An instance of the model with the provided default parameters.
        """
        return self.model_class(**self.default_params)

    def instantiate_tuned(self, best_params: Dict[str, Any]) -> Any:
        """
        Instantiates a new model with the tuned parameters and includes default parameters like max_iter if present.

        Args:
            best_params (Dict[str, Any]): The tuned hyperparameters.

        Returns:
            Any: A new instance of the model with the tuned hyperparameters.
        """
        if 'max_iter' in self.default_params:
            best_params['max_iter'] = self.default_params['max_iter']
        return self.model_class(**best_params)

    def get_hyperparam_grid(self) -> Dict[str, Any]:
        """Returns the hyperparameter grid for the model.

        Returns:
            Dict[str, Any]: A dictionary representing the hyperparameter grid.
        """
        return self.hyperparam_grid
