class DataSplitInfo:
    def __init__(
        self, 
        X_train, 
        X_test, 
        y_train, 
        y_test, 
        filename, 
        scaler=None, 
        features=None
    ):
        """
        Initialize the DataSplitInfo to store all the data related to a split.
        
        Args:
            X_train (pd.DataFrame): The training features.
            X_test (pd.DataFrame): The testing features.
            y_train (pd.Series): The training labels.
            y_test (pd.Series): The testing labels.
            filename (str): The filename or table name of the dataset.
            scaler (optional): The scaler used for this split.
            features (list, optional): The order of input features.
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.filename = filename
        self.scaler = scaler
        self.features = features

    def get_train(self):
        """
        Returns the training features.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: A tuple containing the training features (X_train)
            and training labels (y_train).
        """
        return self.X_train, self.X_test

    def get_test(self):
        """
        Returns the testing features.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: A tuple containing the testing features (X_test)
            and testing labels (y_test).
        """
        return self.y_train, self.y_test

    def get_train_test(self):
        """
        Returns both the training and testing split.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing
            the training features (X_train), testing features (X_test), training labels (y_train),
            and testing labels (y_test).
        """
        return self.X_train, self.X_test, self.y_train, self.y_test
