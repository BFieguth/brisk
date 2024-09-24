import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.linear_model as linear
import sklearn.tree as tree
import sklearn.ensemble as ensemble
import sklearn.svm as svm
import sklearn.neighbors as neighbors
import sklearn.neural_network as neural
import sklearn.kernel_ridge as kernel_ridge

from ml_toolkit.utility.ModelWrapper import ModelWrapper

REGRESSION_MODELS = {
    "linear": ModelWrapper(
        name="Linear Regression",
        model_class=linear.LinearRegression
    ),
    "ridge": ModelWrapper(
        name="Ridge Regression",
        model_class=linear.Ridge,
        default_params={"max_iter": 10000},
        hyperparam_grid={"alpha": np.logspace(-3, 0, 100)}
    ),
    "lasso": ModelWrapper(
        name="LASSO Regression",
        model_class=linear.Lasso,
        default_params={"alpha": 0.1, "max_iter": 10000},
        hyperparam_grid={"alpha": np.logspace(-3, 0, 100)}
    ),
    "bridge": ModelWrapper(
        name="Bayesian Ridge Regression",
        model_class=linear.BayesianRidge,
        default_params={"max_iter": 10000},
        hyperparam_grid={
            'alpha_1': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],    #TODO Change these?
            'alpha_2': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],   
            'lambda_1': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],  
            'lambda_2': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]  
        }
    ),
    "elasticnet": ModelWrapper(
        name="Elastic Net Regression",
        model_class=linear.ElasticNet,
        default_params={"alpha": 0.1, "max_iter": 10000},
        hyperparam_grid={
            "alpha": np.logspace(-3, 0, 100),
            "l1_ratio": list(np.arange(0.1, 1, 0.1))
        }
    ),
    "dtr": ModelWrapper(
        name="Decision Tree Regression",
        model_class=tree.DecisionTreeRegressor,
        default_params={"min_samples_split": 10},
        hyperparam_grid={
            'criterion': ['friedman_mse', 'absolute_error', 
                          'poisson', 'squared_error'],
            'max_depth': list(range(5, 25, 5)) + [None]
        }
    ),
    "rf": ModelWrapper(
        name="Random Forest",
        model_class=ensemble.RandomForestRegressor,
        default_params={"min_samples_split": 10},
        hyperparam_grid={
            'n_estimators': list(range(20, 160, 20)),   # TODO add min_samples_split?
            'criterion': ['friedman_mse', 'absolute_error', 
                          'poisson', 'squared_error'],
            'max_depth': list(range(5, 25, 5)) + [None]
        }
    ),
    "gbr": ModelWrapper(
        name="Gradient Boosting Regression",
        model_class=ensemble.GradientBoostingRegressor,
        hyperparam_grid={
            'loss': ['squared_error', 'absolute_error', 'huber'],
            'learning_rate': list(np.arange(0.01, 1, 0.1)),
            'n_estimators': list(range(50, 200, 10)),   
            # 'alpha': list(np.arange(0.1, 1, 0.1)) # Range [0, 1], only use if 'huber' is selected
        } 
    ),
    "adaboost": ModelWrapper(
        name="AdaBoost Regression",
        model_class=ensemble.AdaBoostRegressor,
        hyperparam_grid={
            'n_estimators': list(range(50, 200, 10)),  
            'learning_rate': list(np.arange(0.01, 3, 0.1)), 
            'loss': ['linear', 'square', 'exponential'] 
        } 
    ),
    "svr": ModelWrapper(
        name="Support Vector Regression",
        model_class=svm.SVR,
        default_params={"max_iter": 10000},
        hyperparam_grid={
            'kernel': ['linear', 'rbf', 'sigmoid'],
            'C': list(np.arange(1, 30, 0.5)), 
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
        }
    ),
    "mlp": ModelWrapper(
        name="Multi-Layer Perceptron Regression",
        model_class=neural.MLPRegressor,
        default_params={"max_iter": 20000},
        hyperparam_grid={
            'hidden_layer_sizes': [
                (100,), (50, 25), (25, 10), (100, 50, 25), (50, 25, 10)
                ], 
            'activation': ['identity', 'logistic', 'tanh', 'relu'],    
            'alpha': [0.0001, 0.001, 0.01],     # TODO surely this could be better                    
            'learning_rate': ['constant', 'invscaling', 'adaptive']   
        }
    ),
    "knn": ModelWrapper(
        name="K-Nearest Neighbour Regression",
        model_class=neighbors.KNeighborsRegressor,
        hyperparam_grid={
            'n_neighbors': list(range(1,5,2)),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': list(range(5, 50, 5))
        } 
    ),
    "lars": ModelWrapper(
        name="Least Angle Regression",
        model_class=linear.Lars
    ),
    "omp": ModelWrapper(
        name="Orthogonal Matching Pursuit",
        model_class=linear.OrthogonalMatchingPursuit
    ),
    "ard": ModelWrapper(
        name="Bayesian ARD Regression",
        model_class=linear.ARDRegression,
        default_params={"max_iter": 10000},
        hyperparam_grid={
            'alpha_1': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],    # TODO same as bayesian regression
            'alpha_2': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],   
            'lambda_1': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'lambda_2': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        }
    ),
    "passagg": ModelWrapper(
        name="Passive Aggressive Regressor",
        model_class=linear.PassiveAggressiveRegressor,
        default_params={"max_iter": 10000},
        hyperparam_grid={
            'C': list(np.arange(1, 100, 1)) # TODO fine tune this?
        }
    ),
    "kridge": ModelWrapper(
        name="Kernel Ridge",
        model_class=kernel_ridge.KernelRidge,
        hyperparam_grid={
            'alpha': np.logspace(-3, 0, 100)
        }
    ),
    "nusvr": ModelWrapper(
        name="Nu Support Vector Regression",
        model_class=svm.NuSVR,
        default_params={"max_iter": 20000},
        hyperparam_grid={
            'kernel': ['linear', 'rbf', 'sigmoid'],
            'C': list(np.arange(1, 30, 0.5)),
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
        }
    ),
    "rnn": ModelWrapper(
        name="Radius Nearest Neighbour",
        model_class=neighbors.RadiusNeighborsRegressor,
        hyperparam_grid={
            'radius': [i * 0.5 for i in range(1, 7)],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': list(range(10, 60, 10))
        }
    ),
    "xtree": ModelWrapper(
        name="Extra Tree Regressor",
        model_class=ensemble.ExtraTreesRegressor,
        default_params={"min_samples_split": 10},
        hyperparam_grid={
            'n_estimators': list(range(20, 160, 20)),
            'criterion': ['friedman_mse', 'absolute_error', 
                          'poisson', 'squared_error'],
            'max_depth': list(range(5, 25, 5)) + [None]
        }   # TODO add min_sample_split?
    )
}
