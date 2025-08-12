"""Register built-in evaluators."""
from brisk.evaluation.evaluators.builtin import (
    common_measures, common_plots, regression_plots, classification_measures,
    classification_plots, tools, dataset_measures, dataset_plots
)

def register_builtin_evaluators(registry):
    """Register built-in evaluators for model evaluation.
    
    Parameters
    ----------
    registry : EvaluatorRegistry
        The registry to register evaluators with
    """
    registry.register(common_measures.EvaluateModel(
        "brisk_evaluate_model",
        "Model performance on the specified measures."
    ))
    registry.register(common_measures.EvaluateModelCV(
        "brisk_evaluate_model_cv",
        "Average model performance on specified measures across "
        "cross-validation splits."
    ))
    registry.register(common_measures.CompareModels(
        "brisk_compare_models",
        "Compare model performance on specified measures."
    ))
    registry.register(common_plots.PlotLearningCurve(
        "brisk_plot_learning_curve",
        "Plot learning curve of number of examples vs. model performance."
    ))
    registry.register(
        common_plots.PlotFeatureImportance(
            "brisk_plot_feature_importance",
            "Plot feature importance."
        )
    )
    registry.register(common_plots.PlotModelComparison(
        "brisk_plot_model_comparison",
        "Compare model performance across multiple algorithms."
    ))
    registry.register(regression_plots.PlotPredVsObs(
        "brisk_plot_pred_vs_obs",
        "Plot predicted vs. observed values."
    ))
    registry.register(regression_plots.PlotResiduals(
        "brisk_plot_residuals",
        "Plot residuals of model predictions."
    ))
    registry.register(classification_measures.ConfusionMatrix(
        "brisk_confusion_matrix",
        "Plot confusion matrix."
    ))
    registry.register(classification_plots.PlotConfusionHeatmap(
        "brisk_plot_confusion_heatmap",
        "Plot confusion heatmap."
    ))
    registry.register(classification_plots.PlotRocCurve(
        "brisk_plot_roc_curve",
        "Plot ROC curve."
    ))
    registry.register(classification_plots.PlotPrecisionRecallCurve(
        "brisk_plot_precision_recall_curve",
        "Plot precision-recall curve."
    ))
    registry.register(tools.HyperparameterTuning(
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
    registry.register(dataset_measures.ContinuousStatistics(
        "brisk_continuous_statistics",
        "Compute continuous statistics of dataset."
    ))
    registry.register(dataset_measures.CategoricalStatistics(
        "brisk_categorical_statistics",
        "Compute categorical statistics of dataset."
    ))
    registry.register(dataset_plots.HistogramBoxplot(
        "brisk_histogram_boxplot",
        "Plot histogram and boxplot of dataset."
    ))
    registry.register(dataset_plots.PiePlot(
        "brisk_pie_plot",
        "Plot pie chart of dataset."
    ))
    registry.register(dataset_plots.CorrelationMatrix(
        "brisk_correlation_matrix",
        "Plot correlation matrix of dataset."
    ))
