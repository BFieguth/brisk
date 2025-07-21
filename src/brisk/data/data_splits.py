from brisk.data.data_split_info import DataSplitInfo

class DataSplits:
    def __init__(self, n_splits: int):
        if n_splits <= 0 or not isinstance(n_splits, int):
            raise ValueError(
                f"n_splits must be a positive integer, recieved {n_splits}."
            )

        self._data_splits = [None] * n_splits
        self._current_index = 0
        self.expected_n_splits = n_splits

    def add(self, split: DataSplitInfo) -> None:
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
        if not (0 <= index < self.expected_n_splits):
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
