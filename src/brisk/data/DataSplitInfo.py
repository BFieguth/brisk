import datetime
import inspect
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns

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
        categorical_features=[]
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
        self.X_train = X_train.copy(deep=True)
        self.X_test = X_test.copy(deep=True)
        self.y_train = y_train.copy(deep=True)
        self.y_test = y_test.copy(deep=True)

        self.filename = filename
        self.scaler = scaler
        self.features = features

        self.categorical_features = [
            feature for feature in categorical_features 
            if feature in X_train.columns
        ]

        self.continuous_features = [
            col for col in X_train.columns if col not in self.categorical_features
            ]

        self.continuous_stats = {}
        for feature in self.continuous_features:
            self.continuous_stats[feature] = {
                "train": self._calculate_continuous_stats(self.X_train[feature]),
                "test": self._calculate_continuous_stats(self.X_test[feature])
            }
        
        self.categorical_stats = {}
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
            'entropy': -np.sum(np.fromiter(
                (p * np.log2(p) for p in feature_series.value_counts(normalize=True) if p > 0),
                dtype=float
            ))
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

    def _plot_histogram_boxplot(
        self, 
        feature_name: str,
        output_dir: str
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)

        train_series = self.X_train[feature_name].dropna()
        test_series = self.X_test[feature_name].dropna()

        fig, axs = plt.subplots(
            nrows=2, ncols=2, sharex='col', gridspec_kw={"height_ratios": (3, 1)}, 
            figsize=(12, 6)
        )
        
        # Histograms
        bins_train = int(np.ceil(np.log(len(train_series)) + 1))  # Sturges' rule
        bins_test = int(np.ceil(np.log(len(test_series)) + 1))

        axs[0, 0].hist(train_series, bins=bins_train, edgecolor='black', alpha=0.7)
        axs[0, 0].set_title(f'Train Distribution of {feature_name}', fontsize=14)
        axs[0, 0].set_ylabel('Frequency', fontsize=12)

        axs[0, 1].hist(test_series, bins=bins_test, edgecolor='black', alpha=0.7)
        axs[0, 1].set_title(f'Test Distribution of {feature_name}', fontsize=14)

        # Boxplots
        axs[1, 0].boxplot(train_series, vert=False)
        axs[1, 1].boxplot(test_series, vert=False)

        # Set labels
        axs[1, 0].set_xlabel(f'{feature_name}', fontsize=12)
        axs[1, 1].set_xlabel(f'{feature_name}', fontsize=12)

        # Save the plot
        plot_path = os.path.join(output_dir, f"{feature_name}_hist_box.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    def _plot_correlation_matrix(self, output_dir: str):
        """Plot and save the correlation matrix for continuous features."""
        os.makedirs(output_dir, exist_ok=True)
        
        correlation_matrix = self.X_train[self.continuous_features].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", 
            linewidths=0.5, square=True
            )
        plt.title("Correlation Matrix of Continuous Features", fontsize=14)

        plot_path = os.path.join(output_dir, "correlation_matrix.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close() 

    def _plot_categorical_pie(self, feature_name: str, output_dir: str):
        """
        Creates a pie chart to visualize the frequency distribution of a categorical feature.

        Args:
            feature_series (pd.Series): The categorical feature data to plot.
            feature_name (str): The name of the feature (used as the chart title).
            output_dir (str): The file path to save the pie chart. If None, it will display the chart.
        """
        os.makedirs(output_dir, exist_ok=True)

        train_series = self.X_train[feature_name].dropna()
        test_series = self.X_test[feature_name].dropna()

        train_value_counts = train_series.value_counts()
        test_value_counts = test_series.value_counts()

        fig, axs = plt.subplots(1, 2, figsize=(14, 8))

        axs[0].pie(
            train_value_counts, labels=train_value_counts.index, autopct='%1.1f%%',
            startangle=90, colors=plt.cm.Paired.colors
        )
        axs[0].set_title(f'Train {feature_name} Distribution')

        axs[1].pie(
            test_value_counts, labels=test_value_counts.index, autopct='%1.1f%%',
            startangle=90, colors=plt.cm.Paired.colors
        )
        axs[1].set_title(f'Test {feature_name} Distribution')

        plot_path = os.path.join(output_dir, f"{feature_name}_pie_plot.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    def get_train(self):
        """
        Returns the training features.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: A tuple containing the training features (X_train)
            and training labels (y_train).
        """
        if self.scaler and self.continuous_features:
            categorical_data = (
                self.X_train[self.categorical_features].copy() 
                if self.categorical_features 
                else pd.DataFrame(index=self.X_train.index)
                )

            continuous_scaled = pd.DataFrame(
                self.scaler.transform(self.X_train[self.continuous_features]),
                columns=self.continuous_features,
                index=self.X_train.index
                )
            
            X_train_scaled = pd.concat(
                [categorical_data, continuous_scaled], axis=1
                )
            X_train_scaled = X_train_scaled[self.X_train.columns]
            return X_train_scaled, self.y_train
        return self.X_train, self.y_train

    def get_test(self):
        """
        Returns the testing features.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: A tuple containing the testing features (X_test)
            and testing labels (y_test).
        """
        if self.scaler and self.continuous_features:
            categorical_data = (
                self.X_test[self.categorical_features].copy() 
                if self.categorical_features 
                else pd.DataFrame(index=self.X_test.index)
                )

            continuous_scaled = pd.DataFrame(
                self.scaler.transform(self.X_test[self.continuous_features]),
                columns=self.continuous_features,
                index=self.X_test.index
                )
            
            X_test_scaled = pd.concat(
                [categorical_data, continuous_scaled], axis=1
                )
            X_test_scaled = X_test_scaled[self.X_test.columns]
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

            for feature in self.continuous_features:
                self._plot_histogram_boxplot(
                    feature, os.path.join(dataset_dir, "hist_box_plot")
                )
            self._plot_correlation_matrix(dataset_dir)

        if self.categorical_stats:
            categorical_stats_path = os.path.join(dataset_dir, 'categorical_stats.json')
            with open(categorical_stats_path, 'w') as f:
                json.dump(self.categorical_stats, f, indent=4)

            for feature in self.categorical_features:
                self._plot_categorical_pie(
                    feature, os.path.join(dataset_dir, "pie_plot")
                    )
