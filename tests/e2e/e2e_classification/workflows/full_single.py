from brisk.training.workflow import Workflow

class ClassificationSingleFull(Workflow):
    def workflow(self):
        metrics = ["f1_multiclass", "precision_multiclass", "recall_multiclass"]
        model = self.model.fit(self.X_train, self.y_train)
        tuned_model = self.hyperparameter_tuning(
            model, "grid", self.algorithm_names[0], self.X_train, self.y_train, 
            scorer="balanced_accuracy", kf=5, num_rep=2, n_jobs=-1
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
            metric="precision_multiclass", num_rep=2
        )
        self.plot_confusion_heatmap(
            tuned_model, self.X_test, self.y_test, "heatmap"
        )
        self.confusion_matrix(
            tuned_model, self.X_test, self.y_test, "confusion_matrix"
        )
        self.plot_pred_vs_obs(tuned_model, self.X_test, self.y_test, "pred_vs_obs")
