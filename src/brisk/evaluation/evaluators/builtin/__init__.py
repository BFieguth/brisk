"""Register built-in evaluators."""
from .common_measures import EvaluateModel, EvaluateModelCV, CompareModels
from .common_plots import PlotLearningCurve, PlotFeatureImportance, PlotModelComparison
from .regression_plots import PlotPredVsObs, PlotResiduals
from .classification_measures import ConfusionMatrix
from .classification_plots import PlotConfusionHeatmap, PlotRocCurve, PlotPrecisionRecallCurve
from .tools import HyperparameterTuning

def register_builtin_evaluators(registry, services):
    """Register all built-in evaluators.
    
    Parameters
    ----------
    registry : EvaluatorRegistry
        The registry to register evaluators with
    services : ServiceBundle
        Bundle of services for evaluator initialization
    """
    registry.register(EvaluateModel("evaluate_model", services))
    registry.register(EvaluateModelCV("evaluate_model_cv", services))
    registry.register(CompareModels("compare_models", services))
    registry.register(PlotLearningCurve("plot_learning_curve", services))
    registry.register(
        PlotFeatureImportance("plot_feature_importance", services)
    )
    registry.register(PlotModelComparison("plot_model_comparison", services))
    registry.register(PlotPredVsObs("plot_pred_vs_obs", services))
    registry.register(PlotResiduals("plot_residuals", services))
    registry.register(ConfusionMatrix("confusion_matrix", services))
    registry.register(PlotConfusionHeatmap("plot_confusion_heatmap", services))
    registry.register(PlotRocCurve("plot_roc_curve", services))
    registry.register(PlotPrecisionRecallCurve(
        "plot_precision_recall_curve", services)
    )
    registry.register(HyperparameterTuning("hyperparameter_tuning", services))
