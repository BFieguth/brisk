"""Registry for managing evaluator instances.

This module provides the EvaluatorRegistry class for managing and accessing
evaluator instances in the Brisk framework. It provides a centralized way
to register, store, and retrieve evaluators by name.
"""
from typing import Dict, Any

from brisk.evaluation.evaluators import base

class EvaluatorRegistry():
    """Registry for managing evaluator instances.

    Provides a centralized registry for managing evaluator instances in the
    Brisk framework. Allows registration of evaluators and retrieval by name,
    with duplicate name protection.

    Attributes
    ----------
    evaluators : Dict[str, Any]
        Dictionary mapping evaluator names to evaluator instances

    Notes
    -----
    The registry maintains a mapping of evaluator names to their instances,
    allowing for easy lookup and management. Each evaluator must have a
    unique method_name to prevent conflicts.

    Examples
    --------
    Create and use a registry:
        >>> registry = EvaluatorRegistry()
        >>> evaluator = SomeEvaluator()
        >>> registry.register(evaluator)
        >>> retrieved = registry.get("some_evaluator")
    """

    def __init__(self):
        """Initialize EvaluatorRegistry with empty evaluator dictionary.

        Creates a new registry instance with an empty dictionary for
        storing evaluator instances.

        Notes
        -----
        The registry starts empty and evaluators are added via the
        register() method.
        """
        self.evaluators: Dict[str, Any] = {}

    def register(self, evaluator: base.BaseEvaluator) -> None:
        """Register an evaluator instance.
        
        Adds an evaluator instance to the registry using its method_name
        as the key. Prevents duplicate registrations by raising an error
        if an evaluator with the same name is already registered.

        Parameters
        ----------
        evaluator : BaseEvaluator
            Instance of an evaluator class to register

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If an evaluator with the same method_name is already registered

        Notes
        -----
        The evaluator's method_name is used as the key for storage and
        retrieval. This ensures that each evaluator can be uniquely
        identified and accessed.
        """
        if evaluator.method_name in self.evaluators:
            raise ValueError(
                f"Evaluator {evaluator.method_name} already registered"
            )
        self.evaluators[evaluator.method_name] = evaluator

    def get(self, name: str) -> base.BaseEvaluator:
        """Get an evaluator by name.
        
        Retrieves an evaluator instance from the registry by its name.
        Returns None if no evaluator with the given name is found.

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
