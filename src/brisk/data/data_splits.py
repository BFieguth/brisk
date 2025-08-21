"""A module providing the `DataSplits` class for grouping multiple splits.

Classes:
    DataSplits: Container for DataSplitInfo instances of the same dataset.
"""
from brisk.data.data_split_info import DataSplitInfo

class DataSplits:
    """Container for DataSplitInfo instances created from the same dataset.

    Attributes:
        _data_splits: A list of DataSplitInfo instances
        _current_index: The index of the current split.
        expected_n_splits: The number of splits stored.

    """
    def __init__(self, n_splits: int):
        if n_splits <= 0 or not isinstance(n_splits, int):
            raise ValueError(
                f"n_splits must be a positive integer, recieved {n_splits}."
            )

        self._data_splits = [None] * n_splits
        self._current_index = 0
        self.expected_n_splits = n_splits

    def add(self, split: DataSplitInfo) -> None:
        """
        Add a DataSplitInfo instance to the container.

        Args:
            split: The DataSplitInfo instance to add.

        Raises:
            IndexError: If the number of splits exceeds the expected number of 
                splits.
            TypeError: If the split is not a DataSplitInfo instance.
        """
        if self._current_index >= self.expected_n_splits:
            raise IndexError(
                "Cannot add more DataSplitInfo instances than expected"
            )
        if not isinstance(split, DataSplitInfo):
            raise TypeError(
                "DataSplits only accepts DataSplitInfo instances, "
                f"recieved {type(split)}."
            )
        self._data_splits[self._current_index] = split
        self._current_index += 1

    def get_split(self, index) -> DataSplitInfo:
        """Get a DataSplitInfo instance by index.

        Args:
            index: The index of the DataSplitInfo instance to get.

        Returns:
            The DataSplitInfo instance at the given index.
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

    def __len__(self):
        return self.expected_n_splits
