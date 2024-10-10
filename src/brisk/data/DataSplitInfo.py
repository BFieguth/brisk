import json
import os

import numpy as np
import pandas as pd
import scipy.stats

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

        self.continuous_stats = {}
        for feature in self.continuous_features:
            self.continuous_stats[feature] = {
                "train": self._calculate_continuous_stats(self.X_train[feature]),
                "test": self._calculate_continuous_stats(self.X_test[feature])
            }
        
        self.categorical_stats = {}
        if self.categorical_features:
            for feature in self.categorical_features:
                self.categorical_stats[feature] = {
                    "train": self._calculate_categorical_stats(
                        self.X_train[feature], feature
                        ),
                    "test": self._calculate_categorical_stats(
                        self.X_test[feature], feature
                        )
                }

    def _calculate_continuous_stats(self, feature_series: pd.Series) -> dict:
        """Calculate descriptive statistics for a continuous feature."""
        stats = {
            'mean': feature_series.mean(),
            'median': feature_series.median(),
            'std_dev': feature_series.std(),
            'variance': feature_series.var(),
            'min': feature_series.min(),
            'max': feature_series.max(),
            'range': feature_series.max() - feature_series.min(),
            '25_percentile': feature_series.quantile(0.25),
            '75_percentile': feature_series.quantile(0.75),
            'skewness': feature_series.skew(),
            'kurtosis': feature_series.kurt(),
            'coefficient_of_variation': feature_series.std() / feature_series.mean() if feature_series.mean() != 0 else None
        }
        return stats
    
    def _calculate_categorical_stats(
        self, 
        feature_series: pd.Series, 
        feature_name: str
    ) -> dict:
        stats = {
            'frequency': feature_series.value_counts().to_dict(),
            'proportion': feature_series.value_counts(normalize=True).to_dict(),
            'num_unique': feature_series.nunique(),
            'entropy': -np.sum(p * np.log2(p) for p in feature_series.value_counts(normalize=True) if p > 0)
        }

        # Check if test data exists for Chi-Square test
        if feature_name in self.X_test.columns:
            train_counts = self.X_train[feature_name].value_counts()
            test_counts = self.X_test[feature_name].value_counts()

            # Create a contingency table for Chi-Square test
            contingency_table = pd.concat([train_counts, test_counts], axis=1).fillna(0)
            contingency_table.columns = ['train', 'test']
            
            # Perform the Chi-Square test for independence
            chi2, p_value, dof, _ = scipy.stats.chi2_contingency(contingency_table)
            stats['chi_square'] = {
                'chi2_stat': chi2,
                'p_value': p_value,
                'degrees_of_freedom': dof
            }
        else:
            stats['chi_square'] = None
        
        return stats

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

    def save_distribution(self, dataset_dir):
        """Save the continuous and categorical statistics to JSON files."""
        os.makedirs(dataset_dir, exist_ok=True)

        if self.continuous_stats:
            continuous_stats_path = os.path.join(dataset_dir, 'continuous_stats.json')
            with open(continuous_stats_path, 'w') as f:
                json.dump(self.continuous_stats, f, indent=4)

        if self.categorical_stats:
            categorical_stats_path = os.path.join(dataset_dir, 'categorical_stats.json')
            with open(categorical_stats_path, 'w') as f:
                json.dump(self.categorical_stats, f, indent=4)
