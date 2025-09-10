.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Object
     - Description
   * - :class:`~brisk.data.data_manager.DataManager`
     - Handles data splitting and preprocessing pipelines. Arguments are used to define the splitting strategy and preprocessing steps.
   * - :class:`~brisk.data.data_split_info.DataSplitInfo`
     - Stores and analyzes training and testing datasets, providing methods for calculating descriptive statistics and visualizing feature distributions.
   * - :class:`~brisk.data.data_splits.DataSplits`
     - Stores DataSplitInfo instances.