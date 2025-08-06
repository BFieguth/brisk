"""Register built-in evaluators."""
from .common_measures import EvaluateModel, EvaluateModelCV, CompareModels
from .common_plots import PlotLearningCurve, PlotFeatureImportance, PlotModelComparison
from .regression_plots import PlotPredVsObs, PlotResiduals
from .classification_measures import ConfusionMatrix
from .classification_plots import PlotConfusionHeatmap, PlotRocCurve, PlotPrecisionRecallCurve
from .tools import HyperparameterTuning

def register_builtin_evaluators(registry):
    """Register all built-in evaluators.
    
    Parameters
    ----------
    registry : EvaluatorRegistry
        The registry to register evaluators with
    """
    registry.register(EvaluateModel("evaluate_model"))
    registry.register(EvaluateModelCV("evaluate_model_cv"))
    registry.register(CompareModels("compare_models"))
    registry.register(PlotLearningCurve("plot_learning_curve"))
    registry.register(
        PlotFeatureImportance("plot_feature_importance")
    )
    registry.register(PlotModelComparison("plot_model_comparison"))
    registry.register(PlotPredVsObs("plot_pred_vs_obs"))
    registry.register(PlotResiduals("plot_residuals"))
    registry.register(ConfusionMatrix("confusion_matrix"))
    registry.register(PlotConfusionHeatmap("plot_confusion_heatmap"))
    registry.register(PlotRocCurve("plot_roc_curve"))
    registry.register(PlotPrecisionRecallCurve(
        "plot_precision_recall_curve")
    )
    registry.register(HyperparameterTuning("hyperparameter_tuning"))
