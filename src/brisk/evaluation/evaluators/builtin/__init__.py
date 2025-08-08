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
    registry.register(EvaluateModel(
        "brisk_evaluate_model",
        "Model performance on the specified measures."
    ))
    registry.register(EvaluateModelCV(
        "brisk_evaluate_model_cv",
        "Average model performance on specified measures across cross-validation splits."
    ))
    registry.register(CompareModels(
        "brisk_compare_models",
        "Compare model performance on specified measures."
    ))
    registry.register(PlotLearningCurve(
        "brisk_plot_learning_curve",
        "Plot learning curve of number of examples vs. model performance."
    ))
    registry.register(
        PlotFeatureImportance(
            "brisk_plot_feature_importance",
            "Plot feature importance."
        )
    )
    registry.register(PlotModelComparison(
        "brisk_plot_model_comparison",
        "Compare model performance across multiple algorithms."
    ))
    registry.register(PlotPredVsObs(
        "brisk_plot_pred_vs_obs",
        "Plot predicted vs. observed values."
    ))
    registry.register(PlotResiduals(
        "brisk_plot_residuals",
        "Plot residuals of model predictions."
    ))
    registry.register(ConfusionMatrix(
        "brisk_confusion_matrix",
        "Plot confusion matrix."
    ))
    registry.register(PlotConfusionHeatmap(
        "brisk_plot_confusion_heatmap",
        "Plot confusion heatmap."
    ))
    registry.register(PlotRocCurve(
        "brisk_plot_roc_curve",
        "Plot ROC curve."
    ))
    registry.register(PlotPrecisionRecallCurve(
        "brisk_plot_precision_recall_curve",
        "Plot precision-recall curve."
    ))
    registry.register(HyperparameterTuning(
        "brisk_hyperparameter_tuning",
        "Hyperparameter tuning."
    ))


def register_dataset_evaluators(registry):
    """Register evaluators for dataset evaluation.
    
    Parameters
    ----------
    registry : EvaluatorRegistry
        The registry to register evaluators with
    """
    registry.register(ContinuousStatistics(
        "brisk_continuous_statistics",
        "Compute continuous statistics of dataset."
    ))
    registry.register(CategoricalStatistics(
        "brisk_categorical_statistics",
        "Compute categorical statistics of dataset."
    ))
    registry.register(HistogramBoxplot(
        "brisk_histogram_boxplot",
        "Plot histogram and boxplot of dataset."
    ))
    registry.register(PiePlot(
        "brisk_pie_plot",
        "Plot pie chart of dataset."
    ))
    registry.register(CorrelationMatrix(
        "brisk_correlation_matrix",
        "Plot correlation matrix of dataset."
    ))
