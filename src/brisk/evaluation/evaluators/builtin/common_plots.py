"""Common plots for evaluating models."""
from typing import Optional, Dict, Union, List, Tuple, Any

import pandas as pd
import numpy as np
import plotnine as pn
from sklearn import base
import sklearn.model_selection as model_select
from sklearn import inspection
from sklearn import tree
from sklearn import ensemble

from brisk.evaluation.evaluators import plot_evaluator

class PlotLearningCurve(plot_evaluator.PlotEvaluator):
    """Plot learning curves showing model performance vs training size."""
    def plot(
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
        y: pd.Series,
        filename: str = "learning_curve",
        cv: int = 5,
        num_repeats: int = 1,
        n_jobs: int = -1,
        metric: str = "neg_mean_absolute_error"
    ) -> None:
        """Plot learning curves showing model performance vs training size.

        Parameters
        ----------
        model : BaseEstimator
            Model to evaluate
        X : DataFrame
            Training features
        y : Series
            Training target values
        cv : int, optional
            Number of cross-validation folds, by default 5
        num_repeats : int, optional
            Number of times to repeat CV, by default 1
        n_jobs : int, optional
            Number of parallel jobs, by default -1
        metric : str, optional
            Scoring metric to use, by default "neg_mean_absolute_error"
        filename : str, optional
            Name for output file, by default "learning_curve"
        
        Returns
        -------
        None
        """
        plot_data = self._generate_plot_data(
            model, X, y, cv, num_repeats, n_jobs, metric
        )
        plot = self._create_plot(plot_data, metric, model)
        metadata = self._generate_metadata(model, X.attrs["is_test"])
        self._save_plot(filename, metadata, plot=plot)
        self._log_results("Learning Curve", filename)

    def _generate_plot_data(
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
        y: pd.Series,
        cv: int = 5,
        num_repeats: Optional[int] = None,
        n_jobs: int = -1,
        metric: str = "neg_mean_absolute_error",
    ) -> Dict[str, Any]:
        """Calculate the plot data for the learning curve.

        Parameters
        ----------
        model : BaseEstimator
            Model to evaluate
        X : DataFrame
            Training features
        y : Series
            Training target values
        cv : int, optional
            Number of cross-validation folds, by default 5
        num_repeats : int, optional
            Number of times to repeat CV, by default None
        n_jobs : int, optional
            Number of parallel jobs, by default -1
        metric : str, optional
            Scoring metric to use, by default "neg_mean_absolute_error"

        Returns
        -------
        Dict[str, float]: 
            A dictionary containing the learning curve data.
        """
        splitter, indices = self.utility.get_cv_splitter(
            y, cv, num_repeats
        )
        results = {}
        scorer = self.metric_config.get_scorer(metric)
        train_sizes, train_scores, test_scores, fit_times, _ = (
            model_select.learning_curve(
                model, X, y, cv=splitter, groups=indices,
                n_jobs=n_jobs, train_sizes=np.linspace(0.1, 1.0, 5),
                return_times=True, scoring=scorer
            )
        )
        results["train_sizes"] = train_sizes
        results["train_scores_mean"] = np.mean(train_scores, axis=1)
        results["train_scores_std"] = np.std(train_scores, axis=1)
        results["test_scores_mean"] = np.mean(test_scores, axis=1)
        results["test_scores_std"] = np.std(test_scores, axis=1)
        results["fit_times_mean"] = np.mean(fit_times, axis=1)
        results["fit_times_std"] = np.std(fit_times, axis=1)
        return results

    def _create_plot(
        self,
        results: Dict[str, Any],
        metric: str,
        model: base.BaseEstimator
    ) -> pn.ggplot:
        """Create a learning curve plot using plotnine.

        Parameters
        ----------
        results : Dict[str, Any]
            The data for the plot containing train_sizes, train_scores_mean,
            train_scores_std, test_scores_mean, test_scores_std
        metric : str
            The metric to use for the plot
        model : base.BaseEstimator
            The model to use for the plot

        Returns
        -------
        pn.ggplot
            The plot object
        """
        display_name = self.metric_config.get_name(metric)
        wrapper = self.utility.get_algo_wrapper(model.wrapper_name)
        
        train_data = pd.DataFrame({
            'Training_Examples': results["train_sizes"],
            'Score': results["train_scores_mean"],
            'Lower_CI': results["train_scores_mean"] - results["train_scores_std"],
            'Upper_CI': results["train_scores_mean"] + results["train_scores_std"],
            'Type': 'Training Score'
        })
        
        val_data = pd.DataFrame({
            'Training_Examples': results["train_sizes"],
            'Score': results["test_scores_mean"],
            'Lower_CI': results["test_scores_mean"] - results["test_scores_std"],
            'Upper_CI': results["test_scores_mean"] + results["test_scores_std"],
            'Type': 'Cross-Validation Score'
        })
        plot_data = pd.concat([train_data, val_data], ignore_index=True)
        
        plot = (
            pn.ggplot(plot_data, pn.aes(x='Training_Examples', y='Score', color='Type')) +
            pn.geom_ribbon(
                pn.aes(ymin='Lower_CI', ymax='Upper_CI', fill='Type'), 
                alpha=0.1, color="none") +
            pn.geom_line(size=1) +
            pn.geom_point(size=2) +
            pn.scale_color_manual(values=['green', 'red']) +
            pn.scale_fill_manual(values=['green', 'red']) +
            pn.labs(
                x="Training Examples",
                y=display_name,
                title=f"Learning Curve ({wrapper.display_name})",
                color="",
                fill=""
            ) +
            pn.theme(
                legend_position="bottom"
            )
            + self.theme
        )
        
        return plot


class PlotFeatureImportance(plot_evaluator.PlotEvaluator):
    """Plot the feature importance for the model."""
    def plot(
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
        y: pd.Series,
        threshold: Union[int, float],
        feature_names: List[str],
        filename: str,
        metric: str,
        num_rep: int
    ) -> None:
        """Plot the feature importance for the model and save the plot.

        Parameters
        ----------
        model (BaseEstimator): 
            The model to evaluate.
        X (pd.DataFrame): 
            The input features.
        y (pd.Series): 
            The target data
        threshold (Union[int, float]): 
            The number of features or the threshold to filter features by 
            importance.
        feature_names (List[str]): 
            A list of feature names corresponding to the columns in X.
        filename (str): 
            The name of the output file (without extension).
        metric (str): 
            The metric to use for evaluation.
        num_rep (int): 
            The number of repetitions for calculating importance.

        Returns
        -------
        None
        """
        importance_data, plot_width, plot_height = self._generate_plot_data(
            model, X, y, threshold, feature_names, metric, num_rep
        )
        display_name = self.metric_config.get_name(metric)
        wrapper = self.utility.get_algo_wrapper(model.wrapper_name)
        plot = self._create_plot(importance_data, display_name, wrapper)
        metadata = self._generate_metadata(model, X.attrs["is_test"])
        self._save_plot(
            filename, metadata, plot=plot, height=plot_height, width=plot_width
        )
        self._log_results("Feature Importance", filename)

    def _generate_plot_data(
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
        y: pd.Series,
        threshold: Union[int, float],
        feature_names: List[str],
        metric: str,
        num_rep: int
    ) -> Tuple[pd.DataFrame, float, float]:
        """Calculate the plot data for the feature importance.

        Parameters
        ----------
        model (BaseEstimator): 
            The model to evaluate.
        X (pd.DataFrame): 
            The input features.
        y (pd.Series): 
            The target data
        threshold (Union[int, float]): 
            The number of features or the threshold to filter features by 
            importance.
        feature_names (List[str]): 
            A list of feature names corresponding to the columns in X.
        metric (str): 
            The metric to use for evaluation.
        num_rep (int): 
            The number of repetitions for calculating importance.

        Returns
        -------
        Tuple[pd.DataFrame, float, float]: 
            A tuple containing the feature importance values, the width of the
            plot, and the height of the plot.
        """
        scorer = self.metric_config.get_scorer(metric)

        if isinstance(
            model, (
                tree.DecisionTreeRegressor, ensemble.RandomForestRegressor,
                ensemble.GradientBoostingRegressor)
            ):
            model.fit(X,y)
            importance = model.feature_importances_
        else:
            model.fit(X, y)
            results = inspection.permutation_importance(
                model, X=X, y=y, scoring=scorer, n_repeats=num_rep
                )
            importance = results.importances_mean

        if isinstance(threshold, int):
            sorted_indices = np.argsort(importance)[::-1]
            importance = importance[sorted_indices[:threshold]]
            feature_names = [
                feature_names[i] for i in sorted_indices[:threshold]
            ]
        elif isinstance(threshold, float):
            num_features = int(len(feature_names) * threshold)
            if num_features == 0:
                num_features = 1
            sorted_indices = np.argsort(importance)[::-1]
            importance = importance[sorted_indices[:num_features]]
            feature_names = [
                feature_names[i] for i in sorted_indices[:num_features]
            ]

        num_features = len(feature_names)
        size_per_feature = 0.1
        plot_width = max(
            8, size_per_feature * num_features
        )
        plot_height = max(
            6, size_per_feature * num_features * 0.75
        )
        importance_data = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        })
        importance_data["Feature"] = pd.Categorical(
            importance_data["Feature"],
            categories=importance_data.sort_values("Importance")["Feature"],
            ordered=True
        )
        return importance_data, plot_width, plot_height

    def _create_plot(
        self,
        importance_data: pd.DataFrame,
        display_name: str,
        wrapper: Any
    ) -> pn.ggplot:
        """Create a feature importance plot.

        Parameters
        ----------
        importance_data : pd.DataFrame
            The data for the plot
        display_name : str
            The name of the metric
        wrapper : Any
            The wrapper for the model

        Returns
        -------
        pn.ggplot
            The plot object
        """
        plot = (
            pn.ggplot(importance_data, pn.aes(x="Feature", y="Importance")) +
            pn.geom_bar(stat="identity", fill=self.primary_color) +
            pn.coord_flip() +
            pn.labs(
                x="Feature", y=f"Importance ({display_name})",
                title=f"Feature Importance ({wrapper.display_name})"
            ) +
            self.theme
        )
        return plot


class PlotModelComparison(plot_evaluator.PlotEvaluator):
    """Plot a comparison of multiple models based on the specified measure."""
    def plot(
        self,
        *models: base.BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        metric: str,
        filename: str
    ) -> None:
        """Plot a comparison of multiple models based on the specified metric.

        Parameters
        ----------
        models: 
            A variable number of model instances to evaluate.

        X (pd.DataFrame): 
            The input features.

        y (pd.Series): 
            The target data.

        metric (str): 
            The metric to evaluate and plot.

        filename (str): 
            The name of the output file (without extension).
        
        Returns
        -------
        None
        """
        plot_data = self._generate_plot_data(*models, X=X, y=y, metric=metric)
        if plot_data is None:
            return
        display_name = self.metric_config.get_name(metric)
        plot = self._create_plot(plot_data, display_name)
        metadata = self._generate_metadata(list(models), X.attrs["is_test"])
        self._save_plot(filename, metadata, plot=plot)
        self._log_results("Model Comparison", filename)

    def _generate_plot_data(
        self,
        *models: base.BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        metric: str
    ) -> pd.DataFrame:
        """Calculate the plot data for the model comparison.

        Parameters
        ----------
        models: 
            A variable number of model instances to evaluate.

        X (pd.DataFrame): 
            The input features.

        y (pd.Series): 
            The target data.

        metric (str): 
            The metric to evaluate and plot.

        Returns
        -------
        pd.DataFrame: 
            A dataframe containing the model names and their scores.
        """
        model_names = []
        for model in models:
            wrapper = self.utility.get_algo_wrapper(model.wrapper_name)
            model_names.append(wrapper.display_name)

        metric_values = []

        scorer = self.metric_config.get_metric(metric)

        for model in models:
            predictions = model.predict(X)
            if scorer is not None:
                score = scorer(y, predictions)
                metric_values.append(round(score, 3))
            else:
                self.services.logger.logger.info(
                    f"Scorer for {metric} not found."
                )
                return None

        plot_data = pd.DataFrame({
            "Model": model_names,
            "Score": metric_values,
        })
        return plot_data

    def _create_plot(
        self,
        plot_data: pd.DataFrame,
        display_name: str
    ) -> pn.ggplot:
        """Create a model comparison plot.

        Parameters
        ----------
        plot_data : pd.DataFrame
            The data for the plot
        display_name : str
            The name of the metric

        Returns
        -------
        pn.ggplot
            The plot object
        """
        plot = (
            pn.ggplot(plot_data, pn.aes(x="Model", y="Score")) +
            pn.geom_bar(stat="identity", fill=self.primary_color) +
            pn.geom_text(
                pn.aes(label="Score"), position=pn.position_stack(vjust=0.5),
                color="white", size=16
            ) +
            pn.ggtitle(f"Model Comparison on {display_name}") +
            pn.ylab(display_name) +
            self.theme
        )
        return plot


class PlotShapleyValues(plot_evaluator.PlotEvaluator):
    """Plot SHAP (SHapley Additive exPlanations) values for feature importance."""
    
    def plot(
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
        y: pd.Series,
        filename: str = "shap_values",
        plot_type: str = "bar"
    ) -> None:
        """Generate SHAP value plots for feature importance.
        
        Parameters
        ----------
        model : BaseEstimator
            Trained model to explain
        X : DataFrame
            Feature data for generating explanations
        y : Series
            Target data (not used for SHAP but required by interface)
        filename : str, optional
            Base output filename, by default "shap_values"
        plot_type : str, optional
            Type of SHAP plot ('bar', 'waterfall', 'violin', 'beeswarm'). 
            Multiple types can be specified as 'bar,waterfall' to generate 
            multiple plots, by default "bar"
        
        Returns
        -------
        None
        """
        
        plot_data = self._generate_plot_data(
            model, X, y, plot_type
        )
        
        if plot_data is not None:
            # Handle multiple plot types
            plot_types = [pt.strip() for pt in plot_type.split(',')]
            
            for i, single_plot_type in enumerate(plot_types):
                plot = self._create_plot(plot_data, single_plot_type)
                
                plot_filename = filename if len(plot_types) == 1 else f"{filename}_{single_plot_type}"
                
                metadata = self._generate_metadata(model, X.attrs["is_test"])
                self._save_plot(plot_filename, metadata, plot=plot)
                self._log_results("SHAP Values", plot_filename)
    
    def _generate_plot_data(
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
        y: pd.Series,
        plot_type: str = "bar"
    ) -> Optional[Dict[str, Any]]:
        """Generate SHAP values and prepare data for plotting.
        
        Parameters
        ----------
        model : BaseEstimator
            Trained model to explain
        X : DataFrame
            Feature data
        y : Series
            Target data
        plot_type : str
            Type of plot to generate
        
        Returns
        -------
        Dict[str, Any] or None
            Dictionary containing SHAP values and related data, or None if failed
        """
        try:
            import shap # pylint: disable=C0415
        except ImportError:
            return None
            
        # Choose appropriate SHAP explainer based on model type
        if hasattr(model, 'tree_') or hasattr(model, 'estimators_'):
            explainer = shap.TreeExplainer(model)
        elif hasattr(model, 'coef_'):
            explainer = shap.LinearExplainer(model, X)
        else:
            # Fall back to KernelExplainer (model-agnostic but slower)
            background = shap.sample(X, min(100, len(X)))
            explainer = shap.KernelExplainer(model.predict, background)
        
        # Generate SHAP values
        shap_values = explainer(X)
        
        plot_data = {
            'shap_values': shap_values,
            'X_sample': X
        }
        
        return plot_data
    
    def _create_plot(
        self,
        plot_data: Dict[str, Any],
        plot_type: str = "bar"
    ) -> pn.ggplot:
        """Create the SHAP plot using plotnine with Brisk theme.
        
        Parameters
        ----------
        plot_data : Dict[str, Any]
            Data containing SHAP values and features
        plot_type : str
            Type of plot to create
        
        Returns
        -------
        pn.ggplot
            The plotnine plot object
        """
        shap_values = plot_data['shap_values']
        X_sample = plot_data['X_sample']
        
        # Convert SHAP values to DataFrame for plotting
        if hasattr(shap_values, 'values'):
            values = shap_values.values
        else:
            values = shap_values
            
        feature_names = X_sample.columns.tolist()
        
        if plot_type == "bar":
            # Bar plot of mean absolute SHAP values
            mean_abs_shap = np.abs(values).mean(axis=0)
            plot_df = pd.DataFrame({
                'feature': feature_names,
                'importance': mean_abs_shap
            }).sort_values('importance', ascending=False)
            
            plot = (
                pn.ggplot(plot_df, pn.aes(x='pn.reorder(feature, importance)', y='importance')) +
                pn.geom_col(fill=self.primary_color) +
                pn.coord_flip() +
                pn.labs(
                    title="SHAP Feature Importance (Bar Plot)",
                    x="Feature",
                    y="Mean |SHAP Value|"
                ) +
                self.theme.brisk_theme()
            )
            
        elif plot_type == "waterfall":
            # Waterfall plot showing feature contributions for one specific sample (first row)
            if len(values) > 0:
                instance_values = values[0]
                instance_data = X_sample.iloc[0]
                
                plot_df = pd.DataFrame({
                    'feature': feature_names,
                    'shap_value': instance_values,
                    'feature_value': instance_data.values
                }).sort_values('shap_value', key=abs, ascending=False)
                
                plot_df['color'] = plot_df['shap_value'].apply(
                    lambda x: self.accent_color if x > 0 else self.important_color
                )
                
                plot = (
                    pn.ggplot(plot_df, pn.aes(x='pn.reorder(feature, shap_value)', y='shap_value', fill='color')) +
                    pn.geom_col() +
                    pn.scale_fill_identity() +
                    pn.coord_flip() +
                    pn.labs(
                        title="SHAP Waterfall Plot (Single Instance)",
                        x="Feature",
                        y="SHAP Value"
                    ) +
                    pn.geom_hline(yintercept=0, linetype="dashed", alpha=0.7) +
                    self.theme.brisk_theme()
                )
            else:
                # Fallback to bar plot if no data
                return self._create_plot(plot_data, "bar")
                
        elif plot_type == "violin":
            # Violin plot showing distribution of SHAP values
            plot_df = pd.DataFrame(values, columns=feature_names)
            
            # Melt for plotting
            plot_df = plot_df.melt(var_name='feature', value_name='shap_value')
            
            plot = (
                pn.ggplot(plot_df, pn.aes(x='feature', y='shap_value')) +
                pn.geom_violin(fill=self.primary_color, alpha=0.7) +
                pn.geom_hline(yintercept=0, linetype="dashed", alpha=0.7) +
                pn.theme(axis_text_x=pn.element_text(angle=45, hjust=1)) +
                pn.labs(
                    title="SHAP Value Distribution (Violin Plot)",
                    x="Feature",
                    y="SHAP Value"
                ) +
                self.theme.brisk_theme()
            )
            
        elif plot_type == "beeswarm":
            # Beeswarm-style plot (scatter with jitter)
            plot_df = pd.DataFrame(values, columns=feature_names)
            
            # Melt for plotting
            plot_df = plot_df.melt(var_name='feature', value_name='shap_value')
            
            plot = (
                pn.ggplot(plot_df, pn.aes(x='feature', y='shap_value')) +
                pn.geom_jitter(color=self.primary_color, alpha=0.6, width=0.3) +
                pn.geom_hline(yintercept=0, linetype="dashed", alpha=0.7) +
                pn.theme(axis_text_x=pn.element_text(angle=45, hjust=1)) +
                pn.labs(
                    title="SHAP Value Distribution (Beeswarm Plot)",
                    x="Feature",
                    y="SHAP Value"
                ) +
                self.theme.brisk_theme()
            )
            
        else:
            # Default to bar plot
            return self._create_plot(plot_data, "bar")
            
        return plot
