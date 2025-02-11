from brisk.training.workflow import Workflow

class BinarySingleFull(Workflow):
    def workflow(self):
        metrics = ["recall", "precision", "roc_auc", "brier"]
        model = self.model.fit(self.X_train, self.y_train)
        tuned_model = self.hyperparameter_tuning(
            model, "grid", self.algorithm_names[0], self.X_train, self.y_train, 
            scorer="accuracy", kf=5, num_rep=2, n_jobs=-1
        )
        self.save_model(tuned_model, "tuned_model")
        self.evaluate_model_cv(
            tuned_model, self.X_test, self.y_test, metrics, "eval_scores"
        )
        self.plot_learning_curve(
            tuned_model, self.X_train, self.y_train
        )
        self.plot_feature_importance(
            tuned_model, self.X_train, self.y_train, threshold=0.75,
            feature_names=self.feature_names, filename="feature_importance",
            metric="precision", num_rep=2
        )
        self.plot_roc_curve(model, self.X_test, self.y_test, "roc_curve")
        self.plot_precision_recall_curve(
            model, self.X_test, self.y_test, "precision_recall"
        )
        self.plot_confusion_heatmap(
            model, self.X_test, self.y_test, "heatmap"
        )
        self.confusion_matrix(
            model, self.X_test, self.y_test, "confusion_matrix"
        )
