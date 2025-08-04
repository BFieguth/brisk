"""Tools for evaluating models that are not plots or measure calculations."""
from typing import Dict, Any, List, Tuple

import pandas as pd
import numpy as np
import plotnine as pn
import matplotlib.pyplot as plt
from sklearn import base
import sklearn.model_selection as model_select

from brisk.evaluation.evaluators.measure_evaluator import MeasureEvaluator

class HyperparameterTuning(MeasureEvaluator):
    """Perform hyperparameter tuning using grid or random search."""
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

        Parameters
        ----------
        model (BaseEstimator): 
            The model to be tuned.

        method (str): 
            The search method to use ("grid" or "random").

        X_train (pd.DataFrame): 
            The training data.

        y_train (pd.Series): 
            The target values for training.

        scorer (str): 
            The scoring metric to use.

        kf (int): 
            Number of splits for cross-validation.

        num_rep (int): 
            Number of repetitions for cross-validation.

        n_jobs (int): 
            Number of parallel jobs to run.

        plot_results (bool): 
            Whether to plot the performance of hyperparameters. Defaults to 
            False.

        Returns
        -------
        BaseEstimator: 
            The tuned model.
        """
        algo_wrapper = self.utility.get_algo_wrapper(model.wrapper_name)
        param_grid = algo_wrapper.get_hyperparam_grid()
        search_result = self._calculate_measures(
            model, method, X_train, y_train, scorer, kf, num_rep, n_jobs,
            param_grid
        )
        tuned_model = algo_wrapper.instantiate_tuned(
            search_result.best_params_
        )
        tuned_model.fit(X_train, y_train)
        self._log_results(model)

        if plot_results:
            plot = self._plot_hyperparameter_performance(
                param_grid, search_result, algo_wrapper.display_name
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

        Parameters
        ----------
        model (BaseEstimator): 
            The model to be tuned.

        method (str): 
            The search method to use ("grid" or "random").

        X_train (pd.DataFrame): 
            The training data.

        y_train (pd.Series): 
            The target values for training.

        scorer (str): 
            The scoring metric to use.

        kf (int): 
            Number of splits for cross-validation.

        num_rep (int): 
            Number of repetitions for cross-validation.

        n_jobs (int): 
            Number of parallel jobs to run.

        param_grid (Dict[str, Any]): 
            The hyperparameter grid used for tuning.

        Returns
        -------
        BaseEstimator: 
            The tuned model.
        """
        if method == "grid":
            searcher = model_select.GridSearchCV
        elif method == "random":
            searcher = model_select.RandomizedSearchCV
        else:
            raise ValueError(
                f"method must be one of (grid, random). {method} was entered."
                )

        self.logger.info(
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

    def _log_results(self, model: base.BaseEstimator):
        self.logger.info(
            "Hyperparameter optimization for %s complete.",
            model.__class__.__name__
        )

    def _save_plot(self, filename: str, metadata: Dict[str, Any], plot: Any):
        if isinstance(plot, plt.Figure):
            self.io.save_plot(filename, metadata)
        else:
            self.io.save_plot(filename, metadata, plot=plot)

    def _plot_hyperparameter_performance(
        self,
        param_grid: Dict[str, Any],
        search_result: Any,
        display_name: str
    ) -> None:
        """Plot the performance of hyperparameter tuning.

        Parameters
        ----------
        param_grid (Dict[str, Any]): 
            The hyperparameter grid used for tuning.

        search_result (Any): 
            The result from cross-validation during tuning.

        display_name (str): 
            The name of the algorithm to use in the plot labels.
        """
        param_keys = list(param_grid.keys())

        if len(param_keys) == 0:
            return

        elif len(param_keys) == 1:
            return self._plot_1d_performance(
                param_values=param_grid[param_keys[0]],
                mean_test_score=search_result.cv_results_["mean_test_score"],
                param_name=param_keys[0],
                display_name=display_name
            )
        elif len(param_keys) == 2:
            return self._plot_3d_surface(
                param_grid=param_grid,
                search_result=search_result,
                param_names=param_keys,
                display_name=display_name
            )
        else:
            self.logger.info(
                "Higher dimensional visualization not implemented yet"
                )

    def _plot_1d_performance(
        self,
        param_values: List[Any],
        mean_test_score: List[float],
        param_name: str,
        display_name: str
    ) -> None:
        """Plot the performance of a single hyperparameter vs mean test score.

        Parameters
        ----------
        param_values (List[Any]): 
            The values of the hyperparameter.

        mean_test_score (List[float]): 
            The mean test scores for each hyperparameter value.

        param_name (str): 
            The name of the hyperparameter.

        display_name (str): 
            The name of the algorithm to use in the plot labels.
        """
        plot_data = pd.DataFrame({
            "Hyperparameter": param_values,
            "Mean Test Score": mean_test_score,
        })
        param_name = param_name.capitalize()
        title = f"Hyperparameter Performance: {display_name}"
        plot = (
            pn.ggplot(
                plot_data, pn.aes(x="Hyperparameter", y="Mean Test Score")
            ) +
            pn.geom_point(
                color="black", size=3, stroke=0.25, fill=self.primary_color
            ) +
            pn.geom_line(color=self.primary_color) +
            pn.ggtitle(title) +
            pn.xlab(param_name) +
            self.theme.brisk_theme()
        )
        return plot

    def _plot_3d_surface(
        self,
        param_grid: Dict[str, List[Any]],
        search_result: Any,
        param_names: List[str],
        display_name: str
    ) -> None:
        """Plot the performance of two hyperparameters vs mean test score.

        Parameters
        ----------
        param_grid (Dict[str, List[Any]]): 
            The hyperparameter grid used for tuning.

        search_result (Any): 
            The result from cross-validation during tuning.

        param_names (List[str]): 
            The names of the two hyperparameters.

        display_name (str): 
            The name of the algorithm to use in the plot labels.
        """
        X, Y, mean_test_score = self._calc_plot_3d_surface( # pylint: disable=C0103
            param_grid, search_result, param_names
        )
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, mean_test_score.T, cmap="viridis")
        ax.set_xlabel(param_names[0], fontsize=12)
        ax.set_ylabel(param_names[1], fontsize=12)
        ax.set_zlabel("Mean Test Score", fontsize=12)
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

        Parameters
        ----------
        param_grid (Dict[str, List[Any]]): 
            The hyperparameter grid used for tuning.

        search_result (Any): 
            The result from cross-validation during tuning.

        param_names (List[str]): 
            The names of the two hyperparameters.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            A tuple containing the X, Y ndarrays for the meshgrid, and mean test
            score values.
        """
        mean_test_score = search_result.cv_results_["mean_test_score"].reshape(
            len(param_grid[param_names[0]]),
            len(param_grid[param_names[1]])
        )
        X, Y = np.meshgrid( # pylint: disable=C0103
            param_grid[param_names[0]], param_grid[param_names[1]]
        )
        return X, Y, mean_test_score
