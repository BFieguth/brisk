"""Tools for evaluating models that are not plots or measure calculations.

This module provides specialized evaluators for advanced model evaluation
tasks such as hyperparameter tuning. These tools support both grid search
and random search optimization with comprehensive visualization of tuning
results across different parameter dimensions.
"""
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import numpy as np
import plotnine as pn
import matplotlib
import matplotlib.pyplot as plt
from sklearn import base
import sklearn.model_selection as model_select
import plotly.graph_objects as go

from brisk.evaluation.evaluators import measure_evaluator

class HyperparameterTuning(measure_evaluator.MeasureEvaluator):
    """Perform hyperparameter tuning using grid or random search.
    
    This evaluator provides comprehensive hyperparameter optimization
    capabilities using either grid search or random search methods.
    It supports visualization of tuning results for 1D, 2D, and
    multi-dimensionalparameter spaces using line plots, 3D surface plots, and
    parallel coordinate plots respectively.
    
    Attributes
    ----------
    name : str
        The name of the evaluator, set to 'hyperparameter_tuning'
    primary_color : str
        Primary color for plot elements
    secondary_color : str
        Secondary color for plot elements
    accent_color : str
        Accent color for plot elements
    categorical_columns : List[str]
        List of categorical parameter columns for parallel coordinate plots
    """

    def __init__(self, method_name: str, description: str, plot_settings):
        """Initialize the HyperparameterTuning evaluator.
        
        Parameters
        ----------
        method_name : str
            The name of the evaluation method
        description : str
            Description of the evaluation method
        plot_settings
            Plot settings object containing theme and color configurations
        """
        super().__init__(method_name, description)
        matplotlib.use("Agg", force=True)
        self.theme = plot_settings.get_theme()
        colors = plot_settings.get_colors()
        self.primary_color = colors["primary_color"]
        self.secondary_color = colors["secondary_color"]
        self.accent_color = colors["accent_color"]
        self.categorical_columns = []

    def evaluate(
        self,
        model: base.BaseEstimator,
        method: str,
        X_train: pd.DataFrame, # pylint: disable=C0103
        y_train: pd.Series,
        scorer: str,
        kf: int,
        num_rep: int,
        n_jobs: int,
        plot_results: bool = False,
        filename: str = "hyperparameter_tuning"
    ) -> base.BaseEstimator:
        """Perform hyperparameter tuning using grid or random search.

        Executes hyperparameter optimization using the specified search method
        and returns the best model found. Optionally generates performance
        visualizations for the tuning results.

        Parameters
        ----------
        model : base.BaseEstimator
            The model to be tuned
        method : str
            The search method to use ("grid" or "random")
        X_train : pd.DataFrame
            The training features
        y_train : pd.Series
            The training target values
        scorer : str
            The scoring metric to optimize
        kf : int
            Number of folds for cross-validation
        num_rep : int
            Number of repetitions for cross-validation
        n_jobs : int
            Number of parallel jobs to run
        plot_results : bool, optional
            Whether to generate performance plots, by default False
        filename : str, optional
            Filename for saving plots, by default "hyperparameter_tuning"

        Returns
        -------
        base.BaseEstimator
            The tuned model with optimal hyperparameters
        """
        algo_wrapper = self.utility.get_algo_wrapper(model.wrapper_name)
        param_grid = algo_wrapper.get_hyperparam_grid()
        self.reporting.set_tuning_measure(scorer)
        search_result = self._calculate_measures(
            model, method, X_train, y_train, scorer, kf, num_rep, n_jobs,
            param_grid
        )
        tuned_model = algo_wrapper.instantiate_tuned(
            search_result.best_params_
        )
        tuned_model.fit(X_train, y_train)
        self._log_results(model)

        self.reporting.cache_tuned_params(search_result.best_params_)

        if not param_grid:
            return tuned_model

        if plot_results:
            plot = self._plot_hyperparameter_performance(
                param_grid, search_result, algo_wrapper.display_name, scorer
            )
            metadata = self._generate_metadata(model, X_train.attrs["is_test"])
            self._save_plot(filename, metadata, plot=plot)

        return tuned_model

    def _calculate_measures(
        self,
        model: base.BaseEstimator,
        method: str,
        X_train: pd.DataFrame, # pylint: disable=C0103
        y_train: pd.Series,
        scorer: str,
        kf: int,
        num_rep: int,
        n_jobs: int,
        param_grid: Dict[str, Any]
    ) -> base.BaseEstimator:
        """Perform hyperparameter tuning using grid or random search.

        Executes the actual hyperparameter search using scikit-learn's
        GridSearchCV or RandomizedSearchCV based on the specified method.

        Parameters
        ----------
        model : base.BaseEstimator
            The model to be tuned
        method : str
            The search method to use ("grid" or "random")
        X_train : pd.DataFrame
            The training features
        y_train : pd.Series
            The training target values
        scorer : str
            The scoring metric to optimize
        kf : int
            Number of folds for cross-validation
        num_rep : int
            Number of repetitions for cross-validation
        n_jobs : int
            Number of parallel jobs to run
        param_grid : Dict[str, Any]
            The hyperparameter grid to search over

        Returns
        -------
        base.BaseEstimator
            The search result object containing best parameters and scores

        Raises
        ------
        ValueError
            If method is not "grid" or "random"
        """
        if method == "grid":
            searcher = model_select.GridSearchCV
        elif method == "random":
            searcher = model_select.RandomizedSearchCV
        else:
            raise ValueError(
                f"method must be one of (grid, random). {method} was entered."
            )

        self.services.logger.logger.info(
            "Starting hyperparameter optimization for %s", 
            model.__class__.__name__
            )
        score = self.metric_config.get_scorer(scorer)
        splitter, indices = self.utility.get_cv_splitter(y_train, kf, num_rep)

        # The arguments for each sklearn searcher are different which is why the
        # first two arguments have no keywords. If adding another searcher make
        # sure the argument names do not conflict.
        search = searcher(
            model, param_grid, n_jobs=n_jobs, cv=splitter, scoring=score
        )
        search_result = search.fit(X_train, y_train, groups=indices)
        return search_result

    def _log_results(self, model: base.BaseEstimator) -> None:
        """Log the results of the hyperparameter tuning.

        Logs completion message for hyperparameter optimization process.

        Parameters
        ----------
        model : base.BaseEstimator
            The model that was tuned

        Returns
        -------
        None
        """
        self.services.logger.logger.info(
            "Hyperparameter optimization for %s complete.",
            model.__class__.__name__
        )

    def _save_plot(
        self,
        filename: str,
        metadata: Dict[str, Any],
        plot: Any
    ) -> None:
        """Save the plot to the output directory.

        Saves the generated plot to the output directory with appropriate
        metadata, handling both matplotlib and plotnine plot objects.

        Parameters
        ----------
        filename : str
            The name of the file to save the plot to
        metadata : Dict[str, Any]
            The metadata for the plot
        plot : Any
            The plot object to save (matplotlib Figure or plotnine ggplot)

        Returns
        -------
        None
        """
        output_path = self.services.io.output_dir / f"{filename}"
        if isinstance(plot, plt.Figure):
            self.io.save_plot(output_path, metadata)
        else:
            self.io.save_plot(output_path, metadata, plot=plot)

    def _plot_hyperparameter_performance(
        self,
        param_grid: Dict[str, Any],
        search_result: Any,
        display_name: str,
        scorer: str
    ) -> Optional[pn.ggplot | plt.Figure]:
        """Plot the performance of hyperparameter tuning.

        Creates appropriate visualization based on the number of
        hyperparameters:
        - 1 parameter: Line plot
        - 2 parameters: 3D surface plot
        - 3+ parameters: Parallel coordinate plot

        Parameters
        ----------
        param_grid : Dict[str, Any]
            The hyperparameter grid used for tuning
        search_result : Any
            The result from cross-validation during tuning
        display_name : str
            The name of the algorithm to use in plot labels
        scorer : str
            Name of the measure being used for model evaluation

        Returns
        -------
        Optional[pn.ggplot | plt.Figure]
            The plot object or None if no parameters to plot
        """
        param_keys = list(param_grid.keys())
        scorer_name = self.metric_config.get_name(scorer)

        if len(param_keys) == 0:
            return None

        elif len(param_keys) == 1:
            return self._plot_1d_performance(
                param_values=param_grid[param_keys[0]],
                mean_test_score=search_result.cv_results_["mean_test_score"],
                param_name=param_keys[0],
                display_name=display_name,
                scorer_name=scorer_name
            )
        elif len(param_keys) == 2:
            return self._plot_3d_surface(
                param_grid=param_grid,
                search_result=search_result,
                param_names=param_keys,
                display_name=display_name,
                scorer_name=scorer_name
            )
        else:
            return self._plot_parallel_coordinate(
                search_result=search_result,
                param_names=param_keys,
                display_name=display_name,
                scorer_name=scorer_name
            )

    def _plot_1d_performance(
        self,
        param_values: List[Any],
        mean_test_score: List[float],
        param_name: str,
        display_name: str,
        scorer_name: str
    ) -> pn.ggplot:
        """Plot the performance of a single hyperparameter vs mean test score.

        Creates a line plot showing how model performance varies with
        a single hyperparameter value.

        Parameters
        ----------
        param_values : List[Any]
            The values of the hyperparameter tested
        mean_test_score : List[float]
            The mean test scores for each hyperparameter value
        param_name : str
            The name of the hyperparameter
        display_name : str
            The name of the algorithm to use in plot labels
        scorer_name : str
            The display name of the scoring measure

        Returns
        -------
        pn.ggplot
            The line plot object showing hyperparameter performance
        """
        plot_data = pd.DataFrame({
            "Hyperparameter": param_values,
            scorer_name: mean_test_score,
        })
        param_name = param_name.capitalize()
        title = f"Hyperparameter Performance: {display_name}"
        plot = (
            pn.ggplot(
                plot_data, pn.aes(x="Hyperparameter", y=scorer_name)
            ) +
            pn.geom_point(
                color="black", size=3, stroke=0.25, fill=self.primary_color
            ) +
            pn.geom_line(color=self.primary_color) +
            pn.ggtitle(title) +
            pn.xlab(param_name) +
            self.theme
        )
        return plot

    def _plot_3d_surface(
        self,
        param_grid: Dict[str, List[Any]],
        search_result: Any,
        param_names: List[str],
        display_name: str,
        scorer_name: str
    ) -> plt.Figure:
        """Plot the performance of two hyperparameters vs mean test score.

        Creates a 3D surface plot showing how model performance varies
        with two hyperparameters simultaneously.

        Parameters
        ----------
        param_grid : Dict[str, List[Any]]
            The hyperparameter grid used for tuning
        search_result : Any
            The result from cross-validation during tuning
        param_names : List[str]
            The names of the two hyperparameters
        display_name : str
            The name of the algorithm to use in plot labels
        scorer_name : str
            The display name of the scoring measure

        Returns
        -------
        plt.Figure
            The 3D surface plot figure object
        """
        X, Y, mean_test_score = self._calc_plot_3d_surface( # pylint: disable=C0103
            param_grid, search_result, param_names
        )
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, mean_test_score.T, cmap="viridis")
        ax.set_xlabel(param_names[0], fontsize=12)
        ax.set_ylabel(param_names[1], fontsize=12)
        ax.set_zlabel(scorer_name, fontsize=12)
        ax.set_title(
            f"Hyperparameter Performance: {display_name}", fontsize=16
        )
        return fig

    def _calc_plot_3d_surface(
        self,
        param_grid: Dict[str, List[Any]],
        search_result: Any,
        param_names: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the plot data for the 3D surface plot.

        Prepares data for 3D surface plotting by reshaping cross-validation
        results into a grid format suitable for matplotlib's plot_surface.

        Parameters
        ----------
        param_grid : Dict[str, List[Any]]
            The hyperparameter grid used for tuning
        search_result : Any
            The result from cross-validation during tuning
        param_names : List[str]
            The names of the two hyperparameters

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple containing:
            - X: meshgrid array for first parameter
            - Y: meshgrid array for second parameter
            - mean_test_score: reshaped scores for surface plotting
        """
        mean_test_score = search_result.cv_results_["mean_test_score"].reshape(
            len(param_grid[param_names[0]]),
            len(param_grid[param_names[1]])
        )
        X, Y = np.meshgrid( # pylint: disable=C0103
            param_grid[param_names[0]], param_grid[param_names[1]]
        )
        return X, Y, mean_test_score

    def _plot_parallel_coordinate(
        self,
        search_result: Any,
        param_names: List[str],
        display_name: str,
        scorer_name: str
    ) -> go.Figure:
        """Create a parallel coordinates plot for hyperparameter search results.
        
        Creates an interactive parallel coordinates plot for visualizing
        hyperparameter performance across multiple dimensions. Handles both
        numerical and categorical parameters appropriately.

        Parameters
        ----------
        search_result : Any
            The result from cross-validation during tuning
        param_names : List[str]
            The names of the hyperparameters
        display_name : str
            The name of the algorithm to use in plot labels
        scorer_name : str
            The display name of the scoring measure

        Returns
        -------
        go.Figure
            The interactive parallel coordinates plot figure object

        Raises
        ------
        ValueError
            If no data is found in search results
        """
        df = self._calc_parallel_coordinate(search_result)

        if df.empty:
            raise ValueError("No data found in search results")

        plot_param_names = [
            name for name in param_names
            if name in df.columns and name != "mean_test_score"
        ]

        dimensions = []
        for param in plot_param_names:
            if param not in df.columns:
                continue

            dim_config = {
                "label": param,
                "values": df[param]
            }

            if param in self.categorical_columns:
                unique_values = df[param].dropna().unique()
                str_unique_values = [str(val) for val in unique_values]
                sorted_str_values = sorted(str_unique_values)
                str_to_original = {str(val): val for val in unique_values}
                value_map = {
                    str_to_original[str_val]: i
                    for i, str_val in enumerate(sorted_str_values)
                }

                numeric_values = []
                for val in df[param]:
                    if pd.isna(val) or val is None:
                        numeric_values.append(0)
                    else:
                        numeric_values.append(value_map[val])

                dim_config["values"] = numeric_values
                dim_config["tickvals"] = list(range(len(sorted_str_values)))
                dim_config["ticktext"] = sorted_str_values
                dim_config["range"] = [0, len(sorted_str_values) - 1]

            else:
                dim_config["values"] = df[param].fillna(df[param].median())

            dimensions.append(dim_config)

        color_values = df["mean_test_score"]
        colorbar_title = scorer_name

        fig = go.Figure(data=go.Parcoords(
            line=dict(
                color=color_values,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title=colorbar_title)
            ),
            dimensions=dimensions
        ))
        fig.update_layout(
            title={
                "text": f"Hyperparameter Performance: {display_name}",
                "x": 0.5,
                "xanchor": "center"
            },
            font=dict(size=12),
            height=600,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        return fig

    def _calc_parallel_coordinate(self, search_result: Any) -> pd.DataFrame:
        """
        Extract data from sklearn search results for parallel coordinates plot.
        
        Processes cross-validation results into a format suitable for
        parallel coordinates plotting, handling parameter extraction and
        categorical parameter identification.

        Parameters
        ----------
        search_result : Any
            The result from cross-validation during tuning

        Returns
        -------
        pd.DataFrame
            The processed dataframe ready for parallel coordinates plotting
        """
        results_df = pd.DataFrame(search_result.cv_results_)
        param_cols = [
            col for col in results_df.columns if col.startswith("param_")
        ]

        data = {}
        for col in param_cols:
            param_name = col.replace("param_", "")
            values = results_df[col].values

            data[param_name] = values
            if any(isinstance(v, str) for v in values if v is not None):
                if param_name not in self.categorical_columns:
                    self.categorical_columns.append(param_name)

        if "mean_test_score" in results_df.columns:
            data["mean_test_score"] = results_df["mean_test_score"]

        return pd.DataFrame(data)
