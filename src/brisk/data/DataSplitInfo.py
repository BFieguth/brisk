import pandas as pd

class DataSplitInfo:
    def __init__(
        self, 
        X_train, 
        X_test, 
        y_train, 
        y_test, 
        filename, 
        scaler=None, 
        features=None,
        categorical_features=None
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
        self.categorical_features = categorical_features
        if self.categorical_features:
            self.continuous_features = [
                col for col in X_train.columns if col not in self.categorical_features
                ]
        else:
            self.continuous_features = X_train.columns

    def get_train(self):
        """
        Returns the training features.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: A tuple containing the training features (X_train)
            and training labels (y_train).
        """
        if self.scaler:
            X_train_scaled = self.X_train.copy()
            X_train_scaled[self.continuous_features] = pd.DataFrame(
                self.scaler.fit_transform(self.X_train[self.continuous_features]), 
                columns=self.continuous_features
                )
            return X_train_scaled, self.y_train
        return self.X_train, self.y_train

    def get_test(self):
        """
        Returns the testing features.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: A tuple containing the testing features (X_test)
            and testing labels (y_test).
        """
        if self.scaler:
            X_test_scaled = self.X_test.copy()
            X_test_scaled[self.continuous_features] = pd.DataFrame(
                self.scaler.fit_transform(self.X_test[self.continuous_features]), 
                columns=self.continuous_features
                )
            return X_test_scaled, self.y_test
        return self.X_test, self.y_test

    def get_train_test(self):
        """
        Returns both the training and testing split.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing
            the training features (X_train), testing features (X_test), training labels (y_train),
            and testing labels (y_test).
        """
        X_train, y_train = self.get_train() 
        X_test, y_test = self.get_test()
        return X_train, X_test, y_train, y_test
