"""A workflow for training regression models."""
import os

import ml_toolkit

notification = ml_toolkit.AlertMailer("./workflows/config.ini")

# Parse Arguments
parser = ml_toolkit.ArgManager("Regression Models")
parser.add_argument(
    "--methods", "-m", nargs="+", dest="methods", action="store", 
    help="Methods to train" 
)
args = parser.parse_args()

methods = args.methods
methods = [
    ['linear', 'ridge'],  # First model choices
    ['elasticnet', 'bridge']  # Second model choices
]
kf = args.kfold
num_repeats = args.num_repeats
datasets = args.datasets
data_paths = [(datasets[0], None), ("./data_OLD.csv", None)] # Convert to list of tuples
scoring = args.scoring

# Setup Data Splitting
data_splitter = ml_toolkit.DataSplitter(
    test_size=0.2,
    split_method="shuffle",
    group_column="Group"
)

# Setup experiments using TrainingManager
manager = ml_toolkit.TrainingManager(
    method_config=ml_toolkit.REGRESSION_MODELS,
    scoring_config=ml_toolkit.MetricManager(include_regression=True),
    splitter=data_splitter,
    methods=methods,
    data_paths=data_paths
)
print(manager.experiments)


# Define the workflow
def test_workflow(
    evaluator, X_train, X_test, y_train, y_test, output_dir, method_name, model1, model2
):
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)
    evaluator.save_model(model1, "fitted_model1")
    # evaluator.save_model(model2, "fitted_model2")
    evaluator.compare_models(
        model1, model2, X=X_train, y=y_train, metrics=["MAE", "R2", "MSE"], 
        filename="comparison", calculate_diff=True
    )

    evaluator.plot_model_comparison(
        model1, model2, X=X_train, y=y_train, metric="MAE", filename="comparisson"
    )

    # path = os.path.join(output_dir, "fitted_model.pkl")
    # loaded_model = evaluator.load_model(path)
    metrics_list = ["MAE", "R2", "MSE"]
    evaluator.evaluate_model(model1, X_test, y_test, metrics_list, "test_metrics")
    evaluator.evaluate_model_cv(model1, X_test, y_test, metrics_list, "test_metrics_cv")
    
    evaluator.plot_learning_curve(
        model1, X_train, y_train
    )

    # pred = model.predict(X_test)
    evaluator.plot_pred_vs_obs(model1, X_test, y_test, "pred_vs_obs_test")

    evaluator.plot_feature_importance(
        model1, X_train, y_train, filter=10, 
        feature_names=["Feature_1", "Feature_2", "Feature_3", "Feature_4", "Feature_5"],
        filename="feature_importance", scoring="CCC", num_rep=2
    )

    evaluator.plot_residuals(
        model1, X_test, y_test, "residuals"
    )

    # tuned_model = evaluator.hyperparameter_tuning(
    #     model2, "grid", method_name[1], X_train, y_train, "MAE", 5, 2 ,10, plot_results=True
    # )


# Run the workflow
manager.run_experiments(test_workflow)
print("regression.py has run!")