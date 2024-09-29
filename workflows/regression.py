"""A workflow for training regression models."""
import os

import brisk

class RegressionTest(brisk.Workflow):
    def workflow(self):
        self.model1.fit(self.X_train, self.y_train)
        self.model2.fit(self.X_train, self.y_train)
        self.evaluator.save_model(self.model1, "fitted_model1")
        
        self.evaluator.save_model(self.model2, "fitted_model2")

        self.evaluator.compare_models(
            self.model1, self.model2, X=self.X_train, y=self.y_train, 
            metrics=["MAE", "R2", "MSE"], filename="comparison", 
            calculate_diff=True
        )

        self.evaluator.plot_model_comparison(
            self.model1, self.model2, X=self.X_train, y=self.y_train, 
            metric="MAE", filename="comparisson"
        )

        metrics_list = ["MAE", "R2", "MSE"]
        self.evaluator.evaluate_model(
            self.model1, self.X_test, self.y_test, metrics_list, "test_metrics"
            )
        self.evaluator.evaluate_model_cv(
            self.model1, self.X_test, self.y_test, metrics_list, "test_metrics_cv"
            )
        
        self.evaluator.plot_learning_curve(
            self.model1, self.X_train, self.y_train
        )

        # pred = model.predict(X_test)
        self.evaluator.plot_pred_vs_obs(
            self.model1, self.X_test, self.y_test, "pred_vs_obs_test"
            )

        self.evaluator.plot_feature_importance(
            self.model1, self.X_train, self.y_train, filter=10, 
            feature_names=["Feature_1", "Feature_2", "Feature_3", "Feature_4", "Feature_5"],
            filename="feature_importance", scoring="CCC", num_rep=2
        )

        self.evaluator.plot_residuals(
            self.model1, self.X_test, self.y_test, "residuals"
        )

        tuned_model = self.evaluator.hyperparameter_tuning(
            self.model2, "grid", self.method_names[1], self.X_train, self.y_train, 
            "MAE", 5, 2 ,10, plot_results=True
        )


notification = brisk.AlertMailer("./workflows/config.ini")

# Parse Arguments
parser = brisk.ArgManager("Regression Models")
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
data_splitter = brisk.DataSplitter(
    test_size=0.2,
    split_method="shuffle",
    group_column="Group"
)

# Setup experiments using TrainingManager
manager = brisk.TrainingManager(
    method_config=brisk.REGRESSION_ALGORITHMS,
    scoring_config=brisk.MetricManager(include_regression=True),
    workflow=RegressionTest,
    splitter=data_splitter,
    methods=methods,
    data_paths=data_paths
)
print(manager.experiments)

# Run the workflow
manager.run_experiments()
print("regression.py has run!")
