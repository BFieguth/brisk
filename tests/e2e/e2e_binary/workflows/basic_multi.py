from brisk.training.workflow import Workflow

class BinaryMulti(Workflow):
    def workflow(self):
        metrics = ["accuracy", "f1", "top_k_accuracy", "log_loss"]
        model = self.model.fit(self.X_train, self.y_train)
        model2 = self.model2.fit(self.X_train, self.y_train)
        self.save_model(model, "fitted_model")
        self.save_model(model2, "fitted_model2")
        self.compare_models(
            model, model2, X=self.X_test, y=self.y_test, metrics=metrics, 
            filename="compare"
        )
        self.plot_model_comparison(
            model, model2, X=self.X_test, y=self.y_test, metric="f1",
            filename="plot_comparison"
        )
