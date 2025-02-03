from brisk.training.workflow import Workflow

class RegressionSingleFull(Workflow):
    def workflow(self):
        metrics = ["MAE", "CCC", "MSE", "huber_loss", "fake_metric"]
        model = self.model.fit(self.X_train, self.y_train)
        tuned_model = self.hyperparameter_tuning(
            model, "grid", self.algorithm_names[0], self.X_train, self.y_train, 
            scorer="MAE", kf=5, num_rep=2, n_jobs=-1
        )
        self.save_model(tuned_model, "tuned_model")
        self.evaluate_model(
            tuned_model, self.X_test, self.y_test, metrics, "eval_scores"
        )
        self.plot_residuals(
            tuned_model, self.X_test, self.y_test, "residual_plot"
        )
        self.plot_pred_vs_obs(
            tuned_model, self.X_test, self.y_test, "pred_vs_obs"
        )
        self.plot_learning_curve(
            tuned_model, self.X_train, self.y_train
        )
        self.plot_feature_importance(
            tuned_model, self.X_train, self.y_train, threshold=0.75,
            feature_names=self.feature_names, filename="feature_importance",
            metric="MAE", num_rep=2
        )
