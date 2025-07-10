from brisk.training.workflow import Workflow

class BinarySingle(Workflow):
    def workflow(self):
        metrics = ["recall", "precision", "roc_auc"]
        model = self.model.fit(self.X_train, self.y_train)
        self.save_model(model, "fitted_model")
        self.plot_roc_curve(model, self.X_test, self.y_test, "roc_curve", 1)
        self.evaluate_model(
            model, self.X_test, self.y_test, metrics, "eval_model"
        )
        self.plot_precision_recall_curve(
            model, self.X_test, self.y_test, "precision_recall", 1
        )
