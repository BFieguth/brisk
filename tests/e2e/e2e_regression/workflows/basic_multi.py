from brisk.training.workflow import Workflow

class RegressionMulti(Workflow):
    def workflow(self):
        metrics = ["MAPE", "huber_loss", "fake_metric"]
        model = self.model.fit(self.X_train, self.y_train)
        model2 = self.model2.fit(self.X_train, self.y_train)
        self.save_model(model, "tuned_model")
        self.save_model(model2, "tuned_model2")
        self.compare_models(
            model, model2, X=self.X_test, y=self.y_test, metrics=metrics, 
            filename="compare"
        )
        self.plot_model_comparison(
            model, model2, X=self.X_test, y=self.y_test, metric="MAE",
            filename="plot_comparison"
        )
