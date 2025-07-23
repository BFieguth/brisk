import pytest
import pandas as pd

from brisk.data.data_splits import DataSplits
from brisk.data.data_split_info import DataSplitInfo

@pytest.fixture
def data_split_data():
    return {
        "X_train": pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
        "X_test": pd.DataFrame({"a": [7, 8, 9], "b": [10, 11, 12]}),
        "y_train": pd.Series([1, 2, 3]),
        "y_test": pd.Series([4, 5, 6]),
        "group_index_train": None,
        "group_index_test": None,
        "filename": "test.csv"
    }


class TestDataSplits:
    def test_init(self):
        splits = DataSplits(5)
        assert len(splits) == 5
        assert splits.expected_n_splits == 5

        splits = DataSplits(3)
        assert len(splits) == 3
        assert splits.expected_n_splits == 3

        with pytest.raises(ValueError, match="n_splits must be a positive integer,"):
            splits = DataSplits(-5)
        
        with pytest.raises(ValueError, match="n_splits must be a positive integer,"):
            splits = DataSplits(4.5)

    def test_add(self, data_split_data):
        splits = DataSplits(3)
        assert splits._current_index == 0

        splits.add(DataSplitInfo(**data_split_data))
        assert splits._current_index == 1

        data_split_data["filename"] = "test2.csv"
        splits.add(DataSplitInfo(**data_split_data))
        assert splits._current_index == 2

        data_split_data["filename"] = "test3.csv"
        splits.add(DataSplitInfo(**data_split_data))
        assert splits._current_index == 3

        # Check order is maintained
        assert splits._data_splits[0].filename == "test.csv"
        assert splits._data_splits[1].filename == "test2.csv"
        assert splits._data_splits[2].filename == "test3.csv"

    def test_add_invalid_type(self):
        splits = DataSplits(3)
        with pytest.raises(TypeError, match="DataSplits only accepts DataSplitInfo instances,"):
            splits.add("not a DataSplitInfo")

    def test_add_invalid_index(self):
        splits = DataSplits(4)
        with pytest.raises(IndexError, match="Index out of bounds. Expected range: 0 to 3"):
            splits.get_split(5)

    def test_get_split(self, data_split_data):
        splits = DataSplits(2)
        splits.add(DataSplitInfo(**data_split_data))
        data_split_data["filename"] = "test2.csv"
        splits.add(DataSplitInfo(**data_split_data))

        split = splits.get_split(0)
        assert split.filename == "test.csv"

        split = splits.get_split(1)
        assert split.filename == "test2.csv"

    def test_get_split_invalid_index(self, data_split_data):
        splits = DataSplits(2)
        splits.add(DataSplitInfo(**data_split_data))

        with pytest.raises(
            IndexError, 
            match="Index out of bounds. Expected range: 0 to 1"
        ):
            split = splits.get_split(3)

    def test_get_split_missing_split(self, data_split_data):
        splits = DataSplits(2)
        splits.add(DataSplitInfo(**data_split_data))

        with pytest.raises(
            ValueError, 
            match="No DataSplitInfo instance assigned to index 1"
        ):
            split = splits.get_split(1)
