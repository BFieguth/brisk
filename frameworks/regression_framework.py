"""A standard framework for training regression models."""
import ml_toolkit

notification = ml_toolkit.AlertMailer("./frameworks/config.ini")

# Parse Arguments
parser = ml_toolkit.ArgManager("Standard Regression Models")
parser.add_argument(
    "--methods", "-m", nargs="+", dest="methods", action="store", 
    help="Methods to train" 
)
args = parser.parse_args()

methods = args.methods
kf = args.kfold
num_repeats = args.num_repeats
datasets = args.datasets
data_paths = [(datasets[0], None)] # Convert to list of tuples
scoring = args.scoring

# Setup Data Splitting
data_splitter = ml_toolkit.DataSplitter(
    test_size=0.2,
    split_method="shuffle",
    group_column="Group"
)

# Setup configurations using TrainingManager
manager = ml_toolkit.TrainingManager(
    method_config=ml_toolkit.REGRESSION_MODELS,
    scoring_config=ml_toolkit.MetricManager(include_regression=True),
    splitter=data_splitter,
    methods=methods,
    data_paths=data_paths
)
print(manager.configurations)


# Define the workflow
def test_workflow(
    evaluator, model, X_train, X_test, y_train, y_test, output_dir, method_name
):
    model.fit(X_train, y_train)
    # metrics_list = ["MAE", "R2", "MSE"]
    # evaluator.evaluate_model(model, X_test, y_test, metrics_list, "test_metrics")
    # evaluator.evaluate_model_cv(model, X_test, y_test, metrics_list, "test_metrics_cv")
    
    # pred = model.predict(X_test)
    # evaluator.plot_pred_vs_obs(y_test, pred, "pred_vs_obs_test")

    # evaluator.plot_feature_importance(
    #     model, X_train, y_train, filter=10, feature_names=["Feature1", "Feature2"],
    #     filename="feature_importance", scoring="MAE", num_rep=2
    # )

    # evaluator.plot_residuals(
    #     model, X_test, y_test, "residuals"
    # )

    tuned_model = evaluator.hyperparameter_tuning(
        model, "grid", method_name, X_train, y_train, "MAE", 5, 2 ,10
    )


# Run the workflow
manager.run_configurations(test_workflow)
print("regression_framework has run!")
