"""Sklearn adapter for evaluator operations."""

from sklearn import ensemble, inspection, metrics, model_selection, tree


class SklearnEvaluationAdapter:

    def confusion_matrix(self, y_true, y_pred, labels=None):
        return metrics.confusion_matrix(y_true, y_pred, labels=labels)

    def roc_curve(self, y_true, y_score, pos_label=None):
        return metrics.roc_curve(y_true, y_score, pos_label=pos_label)

    def roc_auc_score(self, y_true, y_score):
        return metrics.roc_auc_score(y_true, y_score)

    def precision_recall_curve(self, y_true, y_score, pos_label=None):
        return metrics.precision_recall_curve(
            y_true, y_score, pos_label=pos_label
        )

    def average_precision_score(self, y_true, y_score, pos_label=None):
        return metrics.average_precision_score(
            y_true, y_score, pos_label=pos_label
        )

    def cross_val_score(self, model, X, y, scoring=None, cv=None, groups=None):
        return model_selection.cross_val_score(
            model, X, y, scoring=scoring, cv=cv, groups=groups
        )

    def learning_curve(self, model, X, y, **kwargs):
        return model_selection.learning_curve(model, X, y, **kwargs)

    def get_search_class(self, method):
        if method == "grid":
            return model_selection.GridSearchCV
        if method == "random":
            return model_selection.RandomizedSearchCV
        raise ValueError(
            f"method must be one of (grid, random). {method} was entered."
        )

    def permutation_importance(self, model, X, y, **kwargs):
        return inspection.permutation_importance(model, X=X, y=y, **kwargs)

    def has_native_feature_importances(self, model):
        return isinstance(
            model,
            (
                tree.DecisionTreeRegressor,
                ensemble.RandomForestRegressor,
                ensemble.GradientBoostingRegressor,
            ),
        )
