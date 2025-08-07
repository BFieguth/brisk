"""Register built-in evaluators."""
from .common_measures import EvaluateModel, EvaluateModelCV, CompareModels
from .common_plots import PlotLearningCurve, PlotFeatureImportance, PlotModelComparison
from .regression_plots import PlotPredVsObs, PlotResiduals
from .classification_measures import ConfusionMatrix
from .classification_plots import PlotConfusionHeatmap, PlotRocCurve, PlotPrecisionRecallCurve
from .tools import HyperparameterTuning
from .dataset_measures import ContinuousStatistics, CategoricalStatistics
from .dataset_plots import HistogramBoxplot, PiePlot, CorrelationMatrix

def register_builtin_evaluators(registry):
    """Register built-in evaluators for model evaluation.
    
    Parameters
    ----------
    registry : EvaluatorRegistry
        The registry to register evaluators with
    """
    registry.register(EvaluateModel("brisk_evaluate_model"))
    registry.register(EvaluateModelCV("brisk_evaluate_model_cv"))
    registry.register(CompareModels("brisk_compare_models"))
    registry.register(PlotLearningCurve("brisk_plot_learning_curve"))
    registry.register(
        PlotFeatureImportance("brisk_plot_feature_importance")
    )
    registry.register(PlotModelComparison("brisk_plot_model_comparison"))
    registry.register(PlotPredVsObs("brisk_plot_pred_vs_obs"))
    registry.register(PlotResiduals("brisk_plot_residuals"))
    registry.register(ConfusionMatrix("brisk_confusion_matrix"))
    registry.register(PlotConfusionHeatmap("brisk_plot_confusion_heatmap"))
    registry.register(PlotRocCurve("brisk_plot_roc_curve"))
    registry.register(PlotPrecisionRecallCurve(
        "brisk_plot_precision_recall_curve")
    )
    registry.register(HyperparameterTuning("brisk_hyperparameter_tuning"))


def register_dataset_evaluators(registry):
    """Register evaluators for dataset evaluation.
    
    Parameters
    ----------
    registry : EvaluatorRegistry
        The registry to register evaluators with
    """
    registry.register(ContinuousStatistics("brisk_continuous_statistics"))
    registry.register(CategoricalStatistics("brisk_categorical_statistics"))
    registry.register(HistogramBoxplot("brisk_histogram_boxplot"))
    registry.register(PiePlot("brisk_pie_plot"))
    registry.register(CorrelationMatrix("brisk_correlation_matrix"))
