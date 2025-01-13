from brisk.training.workflow import Workflow

class RegressionSingle(Workflow):
    def workflow(self):
        metrics = ["MAE", "CCC", "MSE", "huber_loss", "fake_metric"]
        model = self.model.fit(self.X_train, self.y_train)
        self.save_model(model, "tuned_model")
        self.evaluate_model(
            model, self.X_test, self.y_test, metrics, "eval_scores"
        )
        self.plot_residuals(
            model, self.X_test, self.y_test, "residual_plot"
        )
