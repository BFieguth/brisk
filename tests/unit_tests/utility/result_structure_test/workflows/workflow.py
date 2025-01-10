# workflow.py
# Define the workflow for training and evaluating models

from brisk.training.workflow import Workflow

class MyWorkflow(Workflow):
    def workflow(self):
        model = self.model.fit(self.X_train, self.y_train)
        self.save_model(model, "fitted_model")
        self.plot_feature_importance(
            model, self.X_train, self.y_train, 10, self.feature_names,
            "feature_importance", "CCC", 5
        )
        self.evaluate_model(
            model, self.X_test, self.y_test, metrics=["CCC", "MAE"],
            filename="eval_model"
        )
        self.plot_pred_vs_obs(
            model, self.X_test, self.y_test, filename="pred_vs_obs"
        )
        self.plot_learning_curve(
            model, self.X_train, self.y_train, filename="learning_curve"
        )
