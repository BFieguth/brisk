import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from brisk.data.DataSplitInfo import DataSplitInfo

np.random.seed(42)

# Create features
continuous_data = np.random.randn(100, 5)  # 5 continuous features
categories = np.random.choice(['A', 'B', 'C'], size=100)
binary_data = np.random.choice([0, 1], size=100)
repetitive_data = np.random.choice([10, 20], size=100, p=[0.8, 0.2])
target = np.random.randn(100)

# Combine all features into a DataFrame
data = pd.DataFrame(continuous_data, columns=[f'Feature_{i}' for i in range(1, 6)])
data['Category'] = categories
data['Binary'] = binary_data
data['Repetitive'] = repetitive_data
data['Target'] = target

# Split data into features (X) and target (y)
X = data.drop(columns=['Target'])
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply scaling (only to continuous and numerical features)
scaler = StandardScaler()

# Select only numerical columns for scaling
numerical_columns = ['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5']
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test_scaled[numerical_columns] = scaler.transform(X_test[numerical_columns])

# Create an instance of DataSplitInfo
features = list(X_train.columns) 

split_info = DataSplitInfo(
    X_train=X_train_scaled,
    X_test=X_test_scaled,
    y_train=y_train,
    y_test=y_test,
    filename="synthetic_dataset",
    scaler=scaler,
    features=features,
    categorical_features=['Category', 'Binary', 'Repetitive']
)

print("DataSplitInfo instance created.")
# print(f"Training features:\n{split_info.X_train.head()}")
# print(f"Test features:\n{split_info.X_test.head()}")
# print(f"Scaler used: {split_info.scaler}")

# print(f"Continuous Variable Stats: {split_info.continuous_stats}")
# print(f"Continuous Variable Stats: {split_info.continuous_stats.keys()}")

# print(f"Categorical Variable Stats: {split_info.categorical_stats}")
# print(f"Categorical Variable Stats: {split_info.categorical_stats.keys()}")

split_info.save_distribution("./test_output")
