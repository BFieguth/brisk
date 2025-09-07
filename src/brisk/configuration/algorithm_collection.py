"""Algorithm collection management for Brisk configuration.

This module provides the AlgorithmCollection class for managing collections
of AlgorithmWrapper instances. It provides both list-like and dict-like
access patterns for convenient algorithm management and lookup.

Classes
-------
AlgorithmCollection : list
    A collection class for managing AlgorithmWrapper instances with
    name-based lookup functionality
"""
from typing import Union

from brisk.configuration import algorithm_wrapper

class AlgorithmCollection(list):
    """A collection for managing AlgorithmWrapper instances.

    Provides both list-like and dict-like access to AlgorithmWrapper objects,
    with name-based lookup functionality. Inherits from list to provide
    standard list operations while adding dictionary-style key access.

    Parameters
    ----------
    *args : AlgorithmWrapper
        Initial AlgorithmWrapper instances to add to the collection

    Attributes
    ----------
    Inherits all list attributes and methods

    Notes
    -----
    The collection maintains uniqueness of algorithm names and provides
    both index-based and name-based access to algorithms. This allows
    for flexible algorithm management in Brisk configurations.

    Examples
    --------
    Create a collection with algorithms:
        >>> from brisk.configuration import AlgorithmWrapper
        >>> alg1 = AlgorithmWrapper("linear", LinearRegression())
        >>> alg2 = AlgorithmWrapper("rf", RandomForestClassifier())
        >>> collection = AlgorithmCollection(alg1, alg2)

    Access by index:
        >>> collection[0]  # Returns first algorithm

    Access by name:
        >>> collection["linear"]  # Returns linear algorithm

    Raises
    ------
    TypeError
        If non-AlgorithmWrapper instance is added
    ValueError
        If duplicate algorithm names are found
    """
    def __init__(self, *args):
        """Initialize the AlgorithmCollection with optional initial algorithms.
        
        Parameters
        ----------
        *args : AlgorithmWrapper
            Initial AlgorithmWrapper instances to add to the collection
        """
        super().__init__()
        for item in args:
            self.append(item)

    def append(self, item: algorithm_wrapper.AlgorithmWrapper) -> None:
        """Add an AlgorithmWrapper to the collection.

        Adds a new algorithm wrapper to the collection while ensuring
        that algorithm names remain unique. Validates that the item
        is an AlgorithmWrapper instance before adding.

        Parameters
        ----------
        item : AlgorithmWrapper
            Algorithm wrapper to add to the collection

        Raises
        ------
        TypeError
            If item is not an AlgorithmWrapper instance
        ValueError
            If algorithm name already exists in collection

        Notes
        -----
        The method performs two validation checks:
        1. Ensures the item is an AlgorithmWrapper instance
        2. Checks that no algorithm with the same name already exists

        Examples
        --------
        Add a new algorithm:
            >>> collection = AlgorithmCollection()
            >>> alg = AlgorithmWrapper("svm", SVC())
            >>> collection.append(alg)
        """
        if not isinstance(item, algorithm_wrapper.AlgorithmWrapper):
            raise TypeError(
                "AlgorithmCollection only accepts AlgorithmWrapper instances"
            )
        if any(wrapper.name == item.name for wrapper in self):
            raise ValueError(
                f"Duplicate algorithm name: {item.name}"
            )
        super().append(item)

    def __getitem__(
        self,
        key: Union[int, str]
    ) -> algorithm_wrapper.AlgorithmWrapper:
        """Get algorithm by index or name.

        Provides flexible access to algorithms in the collection using
        either integer indices (list-like access) or string names
        (dict-like access). This dual access pattern makes the collection
        convenient to use in different contexts.

        Parameters
        ----------
        key : int or str
            Index or name of algorithm to retrieve:
            - int: Zero-based index for list-style access
            - str: Algorithm name for dict-style access

        Returns
        -------
        AlgorithmWrapper
            The requested algorithm wrapper instance

        Raises
        ------
        KeyError
            If string key doesn't match any algorithm name
        TypeError
            If key is neither int nor str
        IndexError
            If integer key is out of range (inherited from list)

        Examples
        --------
        Access by index:
            >>> collection[0]  # Returns first algorithm

        Access by name:
            >>> collection["linear"]  # Returns algorithm named "linear"

        Notes
        -----
        The method first checks if the key is an integer for list-style
        access, then checks if it's a string for name-based access.
        If neither, it raises a TypeError.
        """
        if isinstance(key, int):
            return super().__getitem__(key)

        if isinstance(key, str):
            for wrapper in self:
                if wrapper.name == key:
                    return wrapper
            raise KeyError(f"No algorithm found with name: {key}")

        raise TypeError(
            f"Index must be an integer or string, got {type(key).__name__}"
        )
