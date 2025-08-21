"""Registry for managing evaluator instances."""
from typing import Dict, Any

from brisk.evaluation.evaluators.base import BaseEvaluator

class EvaluatorRegistry():
    """Registry for managing evaluator instances."""
    def __init__(self):
        self.evaluators: Dict[str, Any] = {}

    def register(self, evaluator: BaseEvaluator):
        """Register an evaluator instance.
        
        Parameters
        ----------
        evaluator_instance : BaseEvaluator
            Instance of an evaluator class
        """
        self.evaluators[evaluator.method_name] = evaluator

    def get(self, name: str):
        """Get an evaluator by name.
        
        Parameters
        ----------
        name : str
            Name of the evaluator to retrieve
            
        Returns
        -------
        BaseEvaluator or None
            The evaluator instance if found, None otherwise
        """
        return self.evaluators.get(name)
