import inspect
from sklearn.base import BaseEstimator
import pytest

from brisk.defaults.regression_algorithms import REGRESSION_ALGORITHMS
from brisk.defaults.classification_algorithms import CLASSIFICATION_ALGORITHMS

@ pytest.fixture
def combined_algorithms():
    return REGRESSION_ALGORITHMS + CLASSIFICATION_ALGORITHMS

class TestRegressionAlgorithms:
    """Test all the provided algorithms can be instantiated and have valid 
    hyperparameter grids.
    """
    def test_algorithm_instantiation(self, combined_algorithms):
        """Test that all regression algorithms can be successfully instantiated."""
        for algorithm_wrapper in combined_algorithms:
            model_instance = algorithm_wrapper.instantiate()
            
            assert isinstance(model_instance, BaseEstimator), (
                f"Algorithm '{algorithm_wrapper.name}' does not return a BaseEstimator instance. "
                f"Got {type(model_instance)} instead."
            )
            
            # Check that the wrapper_name attribute is set correctly
            assert hasattr(model_instance, 'wrapper_name'), (
                f"Algorithm '{algorithm_wrapper.name}' instance missing 'wrapper_name' attribute"
            )
            assert model_instance.wrapper_name == algorithm_wrapper.name, (
                f"Algorithm '{algorithm_wrapper.name}' has incorrect wrapper_name. "
                f"Expected {algorithm_wrapper.name}, got {model_instance.wrapper_name}"
            )

    def test_hyperparameter_grid_validity(self, combined_algorithms):
        """Test that all hyperparameter grid keys are valid parameters for each algorithm class."""
        for algorithm_wrapper in combined_algorithms:
            sig = inspect.signature(algorithm_wrapper.algorithm_class.__init__)
            valid_params = set(sig.parameters.keys()) - {'self'}
            
            hyperparam_keys = set(algorithm_wrapper.hyperparam_grid.keys())            
            invalid_params = hyperparam_keys - valid_params
            assert not invalid_params, (
                f"Algorithm '{algorithm_wrapper.name}' ({algorithm_wrapper.algorithm_class.__name__}) "
                f"has invalid hyperparameter keys: {invalid_params}. "
                f"Valid parameters are: {sorted(valid_params)}"
            )
            
            # Test algorithm can be instantiated with the first value of each hyperparameter
            if algorithm_wrapper.hyperparam_grid:
                test_params = {}
                for param, values in algorithm_wrapper.hyperparam_grid.items():
                    if isinstance(values, list) and values:
                        test_params[param] = values[0]
                    elif hasattr(values, '__iter__') and not isinstance(values, str):
                        try:
                            test_params[param] = list(values)[0]
                        except (IndexError, TypeError):
                            continue
                
                if test_params:
                    combined_params = {**algorithm_wrapper.default_params, **test_params}
                    
                    try:
                        test_instance = algorithm_wrapper.algorithm_class(**combined_params)
                        assert isinstance(test_instance, BaseEstimator), (
                            f"Algorithm '{algorithm_wrapper.name}' failed to instantiate with test hyperparameters"
                        )
                    except Exception as e:
                        pytest.fail(
                            f"Algorithm '{algorithm_wrapper.name}' failed to instantiate with "
                            f"hyperparameters {combined_params}: {str(e)}"
                        )