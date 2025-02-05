from brisk.training.workflow import Workflow

class ClassificationMulti(Workflow):
    def workflow(self):
        metrics = ["recall_multiclass", "balanced_accuracy"]
        model = self.model.fit(self.X_train, self.y_train)
        model2 = self.model2.fit(self.X_train, self.y_train)
        self.save_model(model, "fitted_model")
        self.save_model(model2, "fitted_model2")
        self.compare_models(
            model, model2, X=self.X_test, y=self.y_test, metrics=metrics, 
            filename="compare"
        )
        self.plot_model_comparison(
            model, model2, X=self.X_test, y=self.y_test, 
            metric="recall_multiclass", filename="plot_comparison"
        )
        self.confusion_matrix(
            model, self.X_test, self.y_test, "confusion_matrix"
        )
        self.confusion_matrix(
            model2, self.X_test, self.y_test, "confusion_matrix2"
        )
