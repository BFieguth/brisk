from brisk.training.workflow import Workflow

class ClassificationSingle(Workflow):
    def workflow(self):
        metrics = ["precision_multiclass", "f1_multiclass", "balanced_accuracy"]
        model = self.model.fit(self.X_train, self.y_train)
        self.save_model(model, "fitted_model")
        self.plot_confusion_heatmap(
            model, self.X_test, self.y_test, "heatmap"
        )
        self.confusion_matrix(
            model, self.X_test, self.y_test, "confusion_matrix"
        )
        self.evaluate_model(
            model, self.X_test, self.y_test, metrics, "eval_model"
        )
