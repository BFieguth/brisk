# Experiment Configuration Log

## Workflow Configuration

### Workflow Class: `MyWorkflow`

## Default Algorithm Configuration
### Linear Regression (`linear`)

- **Algorithm Class**: `LinearRegression`

**Default Parameters:**
```python
{}
```

**Hyperparameter Grid:**
```python
{}
```

### Ridge Regression (`ridge`)

- **Algorithm Class**: `Ridge`

**Default Parameters:**
```python
'max_iter': 10000,
```

**Hyperparameter Grid:**
```python
'alpha': array([0.001     , 0.00107227, 0.00114976, 0.00123285, 0.00132194,
       0.00141747, 0.00151991, 0.00162975, 0.00174753, 0.00187382,
       0.00200923, 0.00215443, 0.00231013, 0.00247708, 0.00265609,
       0.00284804, 0.00305386, 0.00327455, 0.00351119, 0.00376494,
       0.00403702, 0.00432876, 0.00464159, 0.00497702, 0.0053367 ,
       0.00572237, 0.00613591, 0.00657933, 0.0070548 , 0.00756463,
       0.00811131, 0.00869749, 0.00932603, 0.01      , 0.01072267,
       0.01149757, 0.01232847, 0.01321941, 0.01417474, 0.01519911,
       0.01629751, 0.01747528, 0.01873817, 0.02009233, 0.02154435,
       0.0231013 , 0.02477076, 0.02656088, 0.02848036, 0.03053856,
       0.03274549, 0.03511192, 0.03764936, 0.04037017, 0.04328761,
       0.04641589, 0.04977024, 0.05336699, 0.05722368, 0.06135907,
       0.06579332, 0.07054802, 0.07564633, 0.08111308, 0.0869749 ,
       0.09326033, 0.1       , 0.10722672, 0.1149757 , 0.12328467,
       0.13219411, 0.14174742, 0.15199111, 0.16297508, 0.17475284,
       0.18738174, 0.2009233 , 0.21544347, 0.23101297, 0.24770764,
       0.26560878, 0.28480359, 0.30538555, 0.32745492, 0.35111917,
       0.37649358, 0.40370173, 0.43287613, 0.46415888, 0.49770236,
       0.53366992, 0.57223677, 0.61359073, 0.65793322, 0.70548023,
       0.75646333, 0.81113083, 0.869749  , 0.93260335, 1.        ]),
```

### LASSO Regression (`lasso`)

- **Algorithm Class**: `Lasso`

**Default Parameters:**
```python
'alpha': 0.1,
'max_iter': 10000,
```

**Hyperparameter Grid:**
```python
'alpha': array([0.001     , 0.00107227, 0.00114976, 0.00123285, 0.00132194,
       0.00141747, 0.00151991, 0.00162975, 0.00174753, 0.00187382,
       0.00200923, 0.00215443, 0.00231013, 0.00247708, 0.00265609,
       0.00284804, 0.00305386, 0.00327455, 0.00351119, 0.00376494,
       0.00403702, 0.00432876, 0.00464159, 0.00497702, 0.0053367 ,
       0.00572237, 0.00613591, 0.00657933, 0.0070548 , 0.00756463,
       0.00811131, 0.00869749, 0.00932603, 0.01      , 0.01072267,
       0.01149757, 0.01232847, 0.01321941, 0.01417474, 0.01519911,
       0.01629751, 0.01747528, 0.01873817, 0.02009233, 0.02154435,
       0.0231013 , 0.02477076, 0.02656088, 0.02848036, 0.03053856,
       0.03274549, 0.03511192, 0.03764936, 0.04037017, 0.04328761,
       0.04641589, 0.04977024, 0.05336699, 0.05722368, 0.06135907,
       0.06579332, 0.07054802, 0.07564633, 0.08111308, 0.0869749 ,
       0.09326033, 0.1       , 0.10722672, 0.1149757 , 0.12328467,
       0.13219411, 0.14174742, 0.15199111, 0.16297508, 0.17475284,
       0.18738174, 0.2009233 , 0.21544347, 0.23101297, 0.24770764,
       0.26560878, 0.28480359, 0.30538555, 0.32745492, 0.35111917,
       0.37649358, 0.40370173, 0.43287613, 0.46415888, 0.49770236,
       0.53366992, 0.57223677, 0.61359073, 0.65793322, 0.70548023,
       0.75646333, 0.81113083, 0.869749  , 0.93260335, 1.        ]),
```

### Bayesian Ridge Regression (`bridge`)

- **Algorithm Class**: `BayesianRidge`

**Default Parameters:**
```python
'max_iter': 10000,
```

**Hyperparameter Grid:**
```python
'alpha_1': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],
'alpha_2': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],
'lambda_1': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],
'lambda_2': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],
```

### Elastic Net Regression (`elasticnet`)

- **Algorithm Class**: `ElasticNet`

**Default Parameters:**
```python
'alpha': 0.1,
'max_iter': 10000,
```

**Hyperparameter Grid:**
```python
'alpha': array([0.001     , 0.00107227, 0.00114976, 0.00123285, 0.00132194,
       0.00141747, 0.00151991, 0.00162975, 0.00174753, 0.00187382,
       0.00200923, 0.00215443, 0.00231013, 0.00247708, 0.00265609,
       0.00284804, 0.00305386, 0.00327455, 0.00351119, 0.00376494,
       0.00403702, 0.00432876, 0.00464159, 0.00497702, 0.0053367 ,
       0.00572237, 0.00613591, 0.00657933, 0.0070548 , 0.00756463,
       0.00811131, 0.00869749, 0.00932603, 0.01      , 0.01072267,
       0.01149757, 0.01232847, 0.01321941, 0.01417474, 0.01519911,
       0.01629751, 0.01747528, 0.01873817, 0.02009233, 0.02154435,
       0.0231013 , 0.02477076, 0.02656088, 0.02848036, 0.03053856,
       0.03274549, 0.03511192, 0.03764936, 0.04037017, 0.04328761,
       0.04641589, 0.04977024, 0.05336699, 0.05722368, 0.06135907,
       0.06579332, 0.07054802, 0.07564633, 0.08111308, 0.0869749 ,
       0.09326033, 0.1       , 0.10722672, 0.1149757 , 0.12328467,
       0.13219411, 0.14174742, 0.15199111, 0.16297508, 0.17475284,
       0.18738174, 0.2009233 , 0.21544347, 0.23101297, 0.24770764,
       0.26560878, 0.28480359, 0.30538555, 0.32745492, 0.35111917,
       0.37649358, 0.40370173, 0.43287613, 0.46415888, 0.49770236,
       0.53366992, 0.57223677, 0.61359073, 0.65793322, 0.70548023,
       0.75646333, 0.81113083, 0.869749  , 0.93260335, 1.        ]),
'l1_ratio': [np.float64(0.1), np.float64(0.2), np.float64(0.30000000000000004), np.float64(0.4), np.float64(0.5), np.float64(0.6), np.float64(0.7000000000000001), np.float64(0.8), np.float64(0.9)],
```

### Decision Tree Regression (`dtr`)

- **Algorithm Class**: `DecisionTreeRegressor`

**Default Parameters:**
```python
'min_samples_split': 10,
```

**Hyperparameter Grid:**
```python
'criterion': ['friedman_mse', 'absolute_error', 'poisson', 'squared_error'],
'max_depth': [5, 10, 15, 20, None],
```

### Random Forest (`rf`)

- **Algorithm Class**: `RandomForestRegressor`

**Default Parameters:**
```python
'min_samples_split': 10,
```

**Hyperparameter Grid:**
```python
'n_estimators': [20, 40, 60, 80, 100, 120, 140],
'criterion': ['friedman_mse', 'absolute_error', 'poisson', 'squared_error'],
'max_depth': [5, 10, 15, 20, None],
```

### Gradient Boosting Regression (`gbr`)

- **Algorithm Class**: `GradientBoostingRegressor`

**Default Parameters:**
```python
{}
```

**Hyperparameter Grid:**
```python
'loss': ['squared_error', 'absolute_error', 'huber'],
'learning_rate': [np.float64(0.01), np.float64(0.11), np.float64(0.21000000000000002), np.float64(0.31000000000000005), np.float64(0.41000000000000003), np.float64(0.51), np.float64(0.6100000000000001), np.float64(0.7100000000000001), np.float64(0.81), np.float64(0.91)],
'n_estimators': [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
```

### AdaBoost Regression (`adaboost`)

- **Algorithm Class**: `AdaBoostRegressor`

**Default Parameters:**
```python
{}
```

**Hyperparameter Grid:**
```python
'n_estimators': [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
'learning_rate': [np.float64(0.01), np.float64(0.11), np.float64(0.21000000000000002), np.float64(0.31000000000000005), np.float64(0.41000000000000003), np.float64(0.51), np.float64(0.6100000000000001), np.float64(0.7100000000000001), np.float64(0.81), np.float64(0.91), np.float64(1.01), np.float64(1.11), np.float64(1.2100000000000002), np.float64(1.31), np.float64(1.4100000000000001), np.float64(1.51), np.float64(1.61), np.float64(1.7100000000000002), np.float64(1.81), np.float64(1.9100000000000001), np.float64(2.01), np.float64(2.11), np.float64(2.21), np.float64(2.31), np.float64(2.41), np.float64(2.51), np.float64(2.61), np.float64(2.71), np.float64(2.81), np.float64(2.91)],
'loss': ['linear', 'square', 'exponential'],
```

### Support Vector Regression (`svr`)

- **Algorithm Class**: `SVR`

**Default Parameters:**
```python
'max_iter': 10000,
```

**Hyperparameter Grid:**
```python
'kernel': ['linear', 'rbf', 'sigmoid'],
'C': [np.float64(1.0), np.float64(1.5), np.float64(2.0), np.float64(2.5), np.float64(3.0), np.float64(3.5), np.float64(4.0), np.float64(4.5), np.float64(5.0), np.float64(5.5), np.float64(6.0), np.float64(6.5), np.float64(7.0), np.float64(7.5), np.float64(8.0), np.float64(8.5), np.float64(9.0), np.float64(9.5), np.float64(10.0), np.float64(10.5), np.float64(11.0), np.float64(11.5), np.float64(12.0), np.float64(12.5), np.float64(13.0), np.float64(13.5), np.float64(14.0), np.float64(14.5), np.float64(15.0), np.float64(15.5), np.float64(16.0), np.float64(16.5), np.float64(17.0), np.float64(17.5), np.float64(18.0), np.float64(18.5), np.float64(19.0), np.float64(19.5), np.float64(20.0), np.float64(20.5), np.float64(21.0), np.float64(21.5), np.float64(22.0), np.float64(22.5), np.float64(23.0), np.float64(23.5), np.float64(24.0), np.float64(24.5), np.float64(25.0), np.float64(25.5), np.float64(26.0), np.float64(26.5), np.float64(27.0), np.float64(27.5), np.float64(28.0), np.float64(28.5), np.float64(29.0), np.float64(29.5)],
'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
```

### Multi-Layer Perceptron Regression (`mlp`)

- **Algorithm Class**: `MLPRegressor`

**Default Parameters:**
```python
'max_iter': 20000,
```

**Hyperparameter Grid:**
```python
'hidden_layer_sizes': [(100,), (50, 25), (25, 10), (100, 50, 25), (50, 25, 10)],
'activation': ['identity', 'logistic', 'tanh', 'relu'],
'alpha': [0.0001, 0.001, 0.01],
'learning_rate': ['constant', 'invscaling', 'adaptive'],
```

### K-Nearest Neighbour Regression (`knn`)

- **Algorithm Class**: `KNeighborsRegressor`

**Default Parameters:**
```python
{}
```

**Hyperparameter Grid:**
```python
'n_neighbors': [1, 3],
'weights': ['uniform', 'distance'],
'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
'leaf_size': [5, 10, 15, 20, 25, 30, 35, 40, 45],
```

### Least Angle Regression (`lars`)

- **Algorithm Class**: `Lars`

**Default Parameters:**
```python
{}
```

**Hyperparameter Grid:**
```python
{}
```

### Orthogonal Matching Pursuit (`omp`)

- **Algorithm Class**: `OrthogonalMatchingPursuit`

**Default Parameters:**
```python
{}
```

**Hyperparameter Grid:**
```python
{}
```

### Bayesian ARD Regression (`ard`)

- **Algorithm Class**: `ARDRegression`

**Default Parameters:**
```python
'max_iter': 10000,
```

**Hyperparameter Grid:**
```python
'alpha_1': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],
'alpha_2': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],
'lambda_1': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],
'lambda_2': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],
```

### Passive Aggressive Regressor (`passagg`)

- **Algorithm Class**: `PassiveAggressiveRegressor`

**Default Parameters:**
```python
'max_iter': 10000,
```

**Hyperparameter Grid:**
```python
'C': [np.int64(1), np.int64(2), np.int64(3), np.int64(4), np.int64(5), np.int64(6), np.int64(7), np.int64(8), np.int64(9), np.int64(10), np.int64(11), np.int64(12), np.int64(13), np.int64(14), np.int64(15), np.int64(16), np.int64(17), np.int64(18), np.int64(19), np.int64(20), np.int64(21), np.int64(22), np.int64(23), np.int64(24), np.int64(25), np.int64(26), np.int64(27), np.int64(28), np.int64(29), np.int64(30), np.int64(31), np.int64(32), np.int64(33), np.int64(34), np.int64(35), np.int64(36), np.int64(37), np.int64(38), np.int64(39), np.int64(40), np.int64(41), np.int64(42), np.int64(43), np.int64(44), np.int64(45), np.int64(46), np.int64(47), np.int64(48), np.int64(49), np.int64(50), np.int64(51), np.int64(52), np.int64(53), np.int64(54), np.int64(55), np.int64(56), np.int64(57), np.int64(58), np.int64(59), np.int64(60), np.int64(61), np.int64(62), np.int64(63), np.int64(64), np.int64(65), np.int64(66), np.int64(67), np.int64(68), np.int64(69), np.int64(70), np.int64(71), np.int64(72), np.int64(73), np.int64(74), np.int64(75), np.int64(76), np.int64(77), np.int64(78), np.int64(79), np.int64(80), np.int64(81), np.int64(82), np.int64(83), np.int64(84), np.int64(85), np.int64(86), np.int64(87), np.int64(88), np.int64(89), np.int64(90), np.int64(91), np.int64(92), np.int64(93), np.int64(94), np.int64(95), np.int64(96), np.int64(97), np.int64(98), np.int64(99)],
```

### Kernel Ridge (`kridge`)

- **Algorithm Class**: `KernelRidge`

**Default Parameters:**
```python
{}
```

**Hyperparameter Grid:**
```python
'alpha': array([0.001     , 0.00107227, 0.00114976, 0.00123285, 0.00132194,
       0.00141747, 0.00151991, 0.00162975, 0.00174753, 0.00187382,
       0.00200923, 0.00215443, 0.00231013, 0.00247708, 0.00265609,
       0.00284804, 0.00305386, 0.00327455, 0.00351119, 0.00376494,
       0.00403702, 0.00432876, 0.00464159, 0.00497702, 0.0053367 ,
       0.00572237, 0.00613591, 0.00657933, 0.0070548 , 0.00756463,
       0.00811131, 0.00869749, 0.00932603, 0.01      , 0.01072267,
       0.01149757, 0.01232847, 0.01321941, 0.01417474, 0.01519911,
       0.01629751, 0.01747528, 0.01873817, 0.02009233, 0.02154435,
       0.0231013 , 0.02477076, 0.02656088, 0.02848036, 0.03053856,
       0.03274549, 0.03511192, 0.03764936, 0.04037017, 0.04328761,
       0.04641589, 0.04977024, 0.05336699, 0.05722368, 0.06135907,
       0.06579332, 0.07054802, 0.07564633, 0.08111308, 0.0869749 ,
       0.09326033, 0.1       , 0.10722672, 0.1149757 , 0.12328467,
       0.13219411, 0.14174742, 0.15199111, 0.16297508, 0.17475284,
       0.18738174, 0.2009233 , 0.21544347, 0.23101297, 0.24770764,
       0.26560878, 0.28480359, 0.30538555, 0.32745492, 0.35111917,
       0.37649358, 0.40370173, 0.43287613, 0.46415888, 0.49770236,
       0.53366992, 0.57223677, 0.61359073, 0.65793322, 0.70548023,
       0.75646333, 0.81113083, 0.869749  , 0.93260335, 1.        ]),
```

### Nu Support Vector Regression (`nusvr`)

- **Algorithm Class**: `NuSVR`

**Default Parameters:**
```python
'max_iter': 20000,
```

**Hyperparameter Grid:**
```python
'kernel': ['linear', 'rbf', 'sigmoid'],
'C': [np.float64(1.0), np.float64(1.5), np.float64(2.0), np.float64(2.5), np.float64(3.0), np.float64(3.5), np.float64(4.0), np.float64(4.5), np.float64(5.0), np.float64(5.5), np.float64(6.0), np.float64(6.5), np.float64(7.0), np.float64(7.5), np.float64(8.0), np.float64(8.5), np.float64(9.0), np.float64(9.5), np.float64(10.0), np.float64(10.5), np.float64(11.0), np.float64(11.5), np.float64(12.0), np.float64(12.5), np.float64(13.0), np.float64(13.5), np.float64(14.0), np.float64(14.5), np.float64(15.0), np.float64(15.5), np.float64(16.0), np.float64(16.5), np.float64(17.0), np.float64(17.5), np.float64(18.0), np.float64(18.5), np.float64(19.0), np.float64(19.5), np.float64(20.0), np.float64(20.5), np.float64(21.0), np.float64(21.5), np.float64(22.0), np.float64(22.5), np.float64(23.0), np.float64(23.5), np.float64(24.0), np.float64(24.5), np.float64(25.0), np.float64(25.5), np.float64(26.0), np.float64(26.5), np.float64(27.0), np.float64(27.5), np.float64(28.0), np.float64(28.5), np.float64(29.0), np.float64(29.5)],
'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
```

### Radius Nearest Neighbour (`rnn`)

- **Algorithm Class**: `RadiusNeighborsRegressor`

**Default Parameters:**
```python
{}
```

**Hyperparameter Grid:**
```python
'radius': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
'weights': ['uniform', 'distance'],
'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
'leaf_size': [10, 20, 30, 40, 50],
```

### Extra Tree Regressor (`xtree`)

- **Algorithm Class**: `ExtraTreesRegressor`

**Default Parameters:**
```python
'min_samples_split': 10,
```

**Hyperparameter Grid:**
```python
'n_estimators': [20, 40, 60, 80, 100, 120, 140],
'criterion': ['friedman_mse', 'absolute_error', 'poisson', 'squared_error'],
'max_depth': [5, 10, 15, 20, None],
```

## Experiment Group: group1
#### Description: 

### DataManager Configuration
```python
DataManager Configuration:
test_size: 0.2
n_splits: 5
split_method: shuffle
stratified: False
categorical_features: ['categorical_0', 'categorical_1', 'categorical_2']
```

### Datasets
#### mixed_features_regression.csv
Features:
```python
Categorical: ['categorical_0', 'categorical_1', 'categorical_2']
Continuous: ['continuous_0', 'continuous_1', 'continuous_2', 'continuous_3', 'continuous_4', 'continuous_5', 'continuous_6', 'continuous_7', 'continuous_8', 'continuous_9', 'continuous_10', 'continuous_11']
```

## Experiment Group: group2
#### Description: 

### DataManager Configuration
```python
DataManager Configuration:
test_size: 0.2
n_splits: 5
split_method: shuffle
stratified: False
categorical_features: ['categorical_0', 'categorical_1', 'categorical_2']
```

### Datasets
#### mixed_features_regression.csv
Features:
```python
Categorical: ['categorical_0', 'categorical_1', 'categorical_2']
Continuous: ['continuous_0', 'continuous_1', 'continuous_2', 'continuous_3', 'continuous_4', 'continuous_5', 'continuous_6', 'continuous_7', 'continuous_8', 'continuous_9', 'continuous_10', 'continuous_11']
```

## Experiment Group: group3
#### Description: 

### DataManager Configuration
```python
DataManager Configuration:
test_size: 0.2
n_splits: 5
split_method: shuffle
stratified: False
scale_method: minmax
categorical_features: ['categorical_0', 'categorical_1', 'categorical_2']
```

### Datasets
#### mixed_features_regression.csv
Features:
```python
Categorical: ['categorical_0', 'categorical_1', 'categorical_2']
Continuous: ['continuous_0', 'continuous_1', 'continuous_2', 'continuous_3', 'continuous_4', 'continuous_5', 'continuous_6', 'continuous_7', 'continuous_8', 'continuous_9', 'continuous_10', 'continuous_11']
```
