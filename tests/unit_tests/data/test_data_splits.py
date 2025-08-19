import pytest
import pandas as pd

from brisk.data.data_splits import DataSplits
from brisk.data.data_split_info import DataSplitInfo

@pytest.fixture
def data_split_data():
    return {
        "X_train": pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}),
        "X_test": pd.DataFrame({"a": [7.0, 8.0, 9.0], "b": [10.0, 11.0, 12.0]}),
        "y_train": pd.Series([1.0, 2.0, 3.0]),
        "y_test": pd.Series([4.0, 5.0, 6.0]),
        "group_index_train": None,
        "group_index_test": None,
        "split_key": ("group", "test", None),
        "split_index": 0
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

    def test_add(self, data_split_data, mock_brisk_project):
        splits = DataSplits(3)
        assert splits._current_index == 0

        splits.add(DataSplitInfo(**data_split_data))
        assert splits._current_index == 1

        data_split_data["split_key"] = ("group", "test2", None)
        splits.add(DataSplitInfo(**data_split_data))
        assert splits._current_index == 2

        data_split_data["split_key"] = ("group", "test3", None)
        splits.add(DataSplitInfo(**data_split_data))
        assert splits._current_index == 3

        # Check order is maintained
        assert (
            splits._data_splits[0].group_name,
            splits._data_splits[0].dataset_name,
            splits._data_splits[0].table_name
        ) == ("group", "test", None)
        assert (
            splits._data_splits[1].group_name,
            splits._data_splits[1].dataset_name,
            splits._data_splits[1].table_name
        ) == ("group", "test2", None)
        assert (
            splits._data_splits[2].group_name,
            splits._data_splits[2].dataset_name,
            splits._data_splits[2].table_name
        ) == ("group", "test3", None)

    def test_add_invalid_type(self):
        splits = DataSplits(3)
        with pytest.raises(TypeError, match="DataSplits only accepts DataSplitInfo instances,"):
            splits.add("not a DataSplitInfo")

    def test_add_invalid_index(self):
        splits = DataSplits(4)
        with pytest.raises(IndexError, match="Index out of bounds. Expected range: 0 to 3"):
            splits.get_split(5)

    def test_add_too_many_splits(self, data_split_data, mock_brisk_project):
        splits = DataSplits(2)
        assert splits._current_index == 0

        splits.add(DataSplitInfo(**data_split_data))
        assert splits._current_index == 1

        data_split_data["split_key"] = ("group", "test2", None)
        splits.add(DataSplitInfo(**data_split_data))
        assert splits._current_index == 2

        data_split_data["split_key"] = ("group", "test3", None)
        with pytest.raises(
            IndexError,
            match="Cannot add more DataSplitInfo instances than expected"
        ):
            splits.add(DataSplitInfo(**data_split_data))

    def test_get_split(self, data_split_data, mock_brisk_project):
        splits = DataSplits(2)
        splits.add(DataSplitInfo(**data_split_data))
        data_split_data["split_key"] = ("group", "test2", None)
        splits.add(DataSplitInfo(**data_split_data))

        split = splits.get_split(0)
        assert (split.group_name, split.dataset_name, split.table_name) == ("group", "test", None)

        split = splits.get_split(1)
        assert (split.group_name, split.dataset_name, split.table_name) == ("group", "test2", None)

    def test_get_split_invalid_index(self, data_split_data, mock_brisk_project):
        splits = DataSplits(2)
        splits.add(DataSplitInfo(**data_split_data))

        with pytest.raises(
            IndexError, 
            match="Index out of bounds. Expected range: 0 to 1"
        ):
            split = splits.get_split(3)

    def test_get_split_missing_split(self, data_split_data, mock_brisk_project):
        splits = DataSplits(2)
        splits.add(DataSplitInfo(**data_split_data))

        with pytest.raises(
            ValueError, 
            match="No DataSplitInfo instance assigned to index 1"
        ):
            split = splits.get_split(1)
