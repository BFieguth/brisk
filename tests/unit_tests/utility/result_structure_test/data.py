# data.py
from brisk.data.data_manager import DataManager                

BASE_DATA_MANAGER = DataManager(
    test_size = 0.2,
    n_splits = 5,
    categorical_features=["categorical_0", "categorical_1", "categorical_2"]
)              
