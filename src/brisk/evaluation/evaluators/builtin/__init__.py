"""Register built-in evaluators.

This module provides functions to register all built-in evaluators with the
evaluator registry. It includes both model evaluation evaluators and dataset
evaluation evaluators, covering measures, plots, and other machine learning
evaluation methods.
"""
from brisk.evaluation.evaluators.builtin import (
    common_measures, common_plots, regression_plots, classification_measures,
    classification_plots, optimization, dataset_measures, dataset_plots
)

def register_builtin_evaluators(registry, plot_settings):
    """Register built-in evaluators for model evaluation.
    
    Registers all built-in evaluators that work with trained models and
    their predictions. This includes measure evaluators for calculating
    performance metrics, plot evaluators for creating visualizations and
    hyperparameter tuning.

    Parameters
    ----------
    registry : EvaluatorRegistry
        The registry to register evaluators with
    plot_settings : PlotSettings
        The plot settings containing theme and color configuration

    Returns
    -------
    None

    Notes
    -----
    This function registers the following evaluators:
    
    **Measure Evaluators:**
    - brisk_evaluate_model: Model performance on specified measures
    - brisk_evaluate_model_cv: Average model performance across CV splits
    - brisk_compare_models: Compare model performance across algorithms
    
    **Plot Evaluators:**
    - brisk_plot_learning_curve: Learning curve visualization
    - brisk_plot_feature_importance: Feature importance plots
    - brisk_plot_model_comparison: Model comparison plots
    - brisk_plot_shapley_values: SHAP values visualization
    - brisk_plot_pred_vs_obs: Predicted vs observed values (regression)
    - brisk_plot_residuals: Residual plots (regression)
    - brisk_plot_confusion_heatmap: Confusion matrix heatmap (classification)
    - brisk_plot_roc_curve: ROC curve (classification)
    - brisk_plot_precision_recall_curve: Precision-recall curve (classification)
    
    **Tool Evaluators:**
    - brisk_hyperparameter_tuning: Hyperparameter optimization
    - brisk_confusion_matrix: Confusion matrix calculation

    Examples
    --------
    Register built-in evaluators:
        >>> from brisk.evaluation.evaluators import registry
        >>> from brisk.theme import PlotSettings
        >>> registry = registry.EvaluatorRegistry()
        >>> plot_settings = PlotSettings()
        >>> register_builtin_evaluators(registry, plot_settings)
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
        "Plot learning curve of number of examples vs. model performance.",
        plot_settings
    ))
    registry.register(
        common_plots.PlotFeatureImportance(
            "brisk_plot_feature_importance",
            "Plot feature importance.",
            plot_settings
    ))
    registry.register(common_plots.PlotModelComparison(
        "brisk_plot_model_comparison",
        "Compare model performance across multiple algorithms.",
        plot_settings
    ))
    registry.register(common_plots.PlotShapleyValues(
        "brisk_plot_shapley_values",
        "Plot SHAP values for feature importance.",
        plot_settings
    ))
    registry.register(regression_plots.PlotPredVsObs(
        "brisk_plot_pred_vs_obs",
        "Plot predicted vs. observed values.",
        plot_settings
    ))
    registry.register(regression_plots.PlotResiduals(
        "brisk_plot_residuals",
        "Plot residuals of model predictions.",
        plot_settings
    ))
    registry.register(classification_measures.ConfusionMatrix(
        "brisk_confusion_matrix",
        "Plot confusion matrix."
    ))
    registry.register(classification_plots.PlotConfusionHeatmap(
        "brisk_plot_confusion_heatmap",
        "Plot confusion heatmap.",
        plot_settings
    ))
    registry.register(classification_plots.PlotRocCurve(
        "brisk_plot_roc_curve",
        "Plot ROC curve.",
        plot_settings
    ))
    registry.register(classification_plots.PlotPrecisionRecallCurve(
        "brisk_plot_precision_recall_curve",
        "Plot precision-recall curve.",
        plot_settings
    ))
    registry.register(optimization.HyperparameterTuning(
        "brisk_hyperparameter_tuning",
        "Hyperparameter tuning.",
        plot_settings
    ))


def register_dataset_evaluators(registry, plot_settings):
    """Register evaluators for dataset evaluation.
    
    Registers all built-in evaluators that work with datasets directly,
    providing statistical analysis and visualization capabilities for
    understanding data characteristics and distributions.

    Parameters
    ----------
    registry : EvaluatorRegistry
        The registry to register evaluators with
    plot_settings : PlotSettings
        The plot settings containing theme and color configuration

    Returns
    -------
    None

    Notes
    -----
    This function registers the following evaluators:
    
    **Dataset Measure Evaluators:**
    - brisk_continuous_statistics: Compute continuous variable statistics
    - brisk_categorical_statistics: Compute categorical variable statistics
    
    **Dataset Plot Evaluators:**
    - brisk_histogram_plot: Create histogram plots for data distribution
    - brisk_bar_plot: Create bar plots for categorical data
    - brisk_correlation_matrix: Create correlation matrix heatmaps

    Examples
    --------
    Register dataset evaluators:
        >>> from brisk.evaluation.evaluators import registry
        >>> from brisk.theme import PlotSettings
        >>> registry = registry.EvaluatorRegistry()
        >>> plot_settings = PlotSettings()
        >>> register_dataset_evaluators(registry, plot_settings)
    """
    registry.register(dataset_measures.ContinuousStatistics(
        "brisk_continuous_statistics",
        "Compute continuous statistics of dataset."
    ))
    registry.register(dataset_measures.CategoricalStatistics(
        "brisk_categorical_statistics",
        "Compute categorical statistics of dataset."
    ))
    registry.register(dataset_plots.Histogram(
        "brisk_histogram_plot",
        "Plot histogram dataset.",
        plot_settings
    ))
    registry.register(dataset_plots.BarPlot(
        "brisk_bar_plot",
        "Plot bar plot of dataset.",
        plot_settings
    ))
    registry.register(dataset_plots.CorrelationMatrix(
        "brisk_correlation_matrix",
        "Plot correlation matrix of dataset.",
        plot_settings
    ))
