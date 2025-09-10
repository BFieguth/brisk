"""Container for managing multiple data splits from the same dataset.

This module provides the DataSplits class, which serves as a container for
grouping multiple DataSplitInfo instances created from the same dataset.
This is particularly useful for cross-validation scenarios where multiple
splits of the same dataset need to be managed together.
"""
from brisk.data import data_split_info

class DataSplits:
    """Container for DataSplitInfo instances created from the same dataset.

    This class provides a fixed-capacity container for managing multiple
    DataSplitInfo instances that represent different splits of the same
    dataset. It ensures type safety and provides index-based access to
    individual splits.

    Parameters
    ----------
    n_splits : int
        The number of splits to be stored in the container. Must be a
        positive integer.

    Attributes
    ----------
    _data_splits : List[DataSplitInfo or None]
        A list of DataSplitInfo instances, initialized with None values
    _current_index : int
        The index where the next split will be added
    expected_n_splits : int
        The total number of splits the container can hold

    Notes
    -----
    The container is initialized with a fixed capacity and fills slots
    sequentially. Once all slots are filled, no more splits can be added.

    Examples
    --------
    Create a container for 5 splits:
        >>> splits = DataSplits(n_splits=5)
        >>> print(len(splits))  # 5

    Add splits to the container:
        >>> for i in range(5):
        ...     split_info = DataSplitInfo(...)  # Create split
        ...     splits.add(split_info)

    Access a specific split:
        >>> split_0 = splits.get_split(0)
        >>> split_2 = splits.get_split(2)

    Raises
    ------
    ValueError
        If n_splits is not a positive integer
    """
    def __init__(self, n_splits: int) -> None:
        """Initialize DataSplits container with specified capacity.

        Creates a new DataSplits container with the specified number of
        slots for DataSplitInfo instances. Validates that n_splits is
        a positive integer.

        Parameters
        ----------
        n_splits : int
            The number of splits to be stored in the container. Must be a
            positive integer.

        Raises
        ------
        ValueError
            If n_splits is not a positive integer

        Notes
        -----
        The container is initialized with all slots set to None and
        the current index set to 0, ready to accept splits in order.
        """
        if n_splits <= 0 or not isinstance(n_splits, int):
            raise ValueError(
                f"n_splits must be a positive integer, recieved {n_splits}."
            )

        self._data_splits = [None] * n_splits
        self._current_index = 0
        self.expected_n_splits = n_splits

    def add(self, split: data_split_info.DataSplitInfo) -> None:
        """Add a DataSplitInfo instance to the container.

        Adds a DataSplitInfo instance to the next available slot in the
        container. The split is added sequentially, starting from index 0.

        Parameters
        ----------
        split : DataSplitInfo
            The DataSplitInfo instance to add to the container

        Raises
        ------
        IndexError
            If the number of splits exceeds the expected number of splits
        TypeError
            If the split is not a DataSplitInfo instance

        Notes
        -----
        The method adds splits sequentially, incrementing the current index
        after each addition. Once all slots are filled, no more splits can
        be added.

        Examples
        --------
        Add a split to the container:
            >>> splits = DataSplits(n_splits=3)
            >>> split_info = DataSplitInfo(...)
            >>> splits.add(split_info)

        Attempt to add more splits than capacity:
            >>> for i in range(4):  # More than n_splits=3
            ...     splits.add(DataSplitInfo(...))  # Raises IndexError
        """
        if self._current_index >= self.expected_n_splits:
            raise IndexError(
                "Cannot add more DataSplitInfo instances than expected"
            )
        if not isinstance(split, data_split_info.DataSplitInfo):
            raise TypeError(
                "DataSplits only accepts DataSplitInfo instances, "
                f"recieved {type(split)}."
            )
        self._data_splits[self._current_index] = split
        self._current_index += 1

    def get_split(self, index: int) -> data_split_info.DataSplitInfo:
        """Get a DataSplitInfo instance by index.

        Retrieves the DataSplitInfo instance at the specified index.
        Validates that the index is within bounds and that a split
        exists at that index.

        Parameters
        ----------
        index : int
            The index of the DataSplitInfo instance to retrieve.
            Must be between 0 and expected_n_splits - 1

        Returns
        -------
        DataSplitInfo
            The DataSplitInfo instance at the specified index

        Raises
        ------
        IndexError
            If the index is out of bounds (not between 0 and expected_n_splits
            - 1)
        ValueError
            If no DataSplitInfo instance has been assigned to the specified
            index

        Notes
        -----
        The method performs two validation checks:
        1. Ensures the index is within the valid range
        2. Ensures a split has been assigned to that index

        Examples
        --------
        Get a split by index:
            >>> splits = DataSplits(n_splits=3)
            >>> splits.add(DataSplitInfo(...))  # Adds to index 0
            >>> split_0 = splits.get_split(0)

        Attempt to access out-of-bounds index:
            >>> splits.get_split(5)  # Raises IndexError

        Attempt to access unassigned index:
            >>> splits.get_split(1)  # Raises ValueError (no split at index 1)
        """
        if not 0 <= index < self.expected_n_splits:
            raise IndexError(
                "Index out of bounds. "
                f"Expected range: 0 to {self.expected_n_splits - 1}"
            )
        if self._data_splits[index] is None:
            raise ValueError(
                f"No DataSplitInfo instance assigned to index {index}"
            )
        return self._data_splits[index]

    def __len__(self) -> int:
        """Return the number of splits the container can hold.

        Returns the expected number of splits, which is the capacity
        of the container, not the number of splits currently stored.

        Returns
        -------
        int
            The number of splits the container can hold

        Notes
        -----
        This method returns the capacity of the container, not the
        number of splits currently stored. Use this to check the
        maximum number of splits that can be added.

        Examples
        --------
        Check container capacity:
            >>> splits = DataSplits(n_splits=5)
            >>> print(len(splits))  # 5
            >>> splits.add(DataSplitInfo(...))
            >>> print(len(splits))  # Still 5 (capacity, not current count)
        """
        return self.expected_n_splits
