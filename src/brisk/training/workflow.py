"""Base workflow class for machine learning training and evaluation.

This module provides the foundational Workflow class that defines the interface
for machine learning workflows. It serves as an abstract base class that
specific workflows must inherit from and implement the abstract `workflow`
method. The class provides a comprehensive interface for model evaluation,
visualization, and analysis through delegation to the EvaluationManager.

The Workflow class encapsulates the common patterns and functionality needed
for machine learning experiments, including data handling, model evaluation,
plotting, and result saving. It provides a standardized interface that
ensures consistency across different types of machine learning workflows.

Classes
-------
Workflow
    Abstract base class for machine learning workflows

Examples
--------
>>> from brisk.training.workflow import Workflow
>>> from brisk.evaluation import evaluation_manager
>>> import pandas as pd
>>> 
>>> class MyWorkflow(Workflow):
...     def workflow(self):
...         # Train models
...         model1 = SomeModel()
...         model1.fit(self.X_train, self.y_train)
...         
...         # Evaluate models
...         self.evaluate_model(model1, self.X_test, self.y_test, 
...                            ['accuracy', 'precision'], 'model1_results')
...         
...         # Compare models
...         results = self.compare_models(
...                             model1, model2, X=self.X_test, y=self.y_test,
...                             metrics=['accuracy'], filename='comparison'
...                             )
... 
>>> # Initialize workflow
>>> eval_manager = evaluation_manager.EvaluationManager(metric_config)
>>> workflow = MyWorkflow(eval_manager, X_train, X_test, y_train, y_test,
...                      output_dir, algorithm_names, feature_names, {})
>>> 
>>> # Run workflow
>>> workflow.run()
"""

import abc
from typing import List, Dict, Any, Union, Optional

import numpy as np
import pandas as pd
from sklearn import base

from brisk.evaluation import evaluation_manager as eval_manager

class Workflow(abc.ABC):
    """Abstract base class for machine learning workflows.
    
    This class defines the interface and common functionality for machine
    learning workflows. It provides a standardized way to structure
    machine learning experiments with consistent data handling, model
    evaluation, visualization, and result saving capabilities.
    
    The Workflow class serves as a foundation that specific workflow
    implementations must inherit from. It delegates evaluation and
    visualization tasks to the EvaluationManager, ensuring consistent
    behavior across different workflow types.
    
    Parameters
    ----------
    evaluation_manager : EvaluationManager
        Manager for model evaluation, visualization, and analysis
    X_train : pd.DataFrame
        Training feature data with pandas DataFrame structure
    X_test : pd.DataFrame
        Test feature data with pandas DataFrame structure
    y_train : pd.Series
        Training target data with pandas Series structure
    y_test : pd.Series
        Test target data with pandas Series structure
    output_dir : str
        Directory path where workflow results will be saved
    algorithm_names : List[str]
        List of algorithm names to be used in the workflow
    feature_names : List[str]
        List of feature names corresponding to the data columns
    workflow_attributes : Dict[str, Any]
        Additional attributes to be unpacked as instance attributes
        
    Attributes
    ----------
    evaluation_manager : EvaluationManager
        Manager for model evaluation, visualization, and analysis
    X_train : pd.DataFrame
        Training feature data with 'is_test' attribute set to False
    X_test : pd.DataFrame
        Test feature data with 'is_test' attribute set to True
    y_train : pd.Series
        Training target data with 'is_test' attribute set to False
    y_test : pd.Series
        Test target data with 'is_test' attribute set to True
    output_dir : str
        Output directory path for saving results
    algorithm_names : List[str]
        List of algorithm names for the workflow
    feature_names : List[str]
        List of feature names for the dataset
    model1, model2, ... : BaseEstimator
        Model instances unpacked from workflow_attributes
        
    Notes
    -----
    The Workflow class provides a comprehensive interface for machine
    learning experiments including:
    - Model evaluation and comparison
    - Visualization and plotting
    - Hyperparameter tuning
    - Model saving and loading
    - SHAP value analysis
    
    All data objects (X_train, X_test, y_train, y_test) are marked with
    an 'is_test' attribute to distinguish between training and test data.
    
    Subclasses must implement the abstract `workflow()` method that defines
    the specific workflow logic for their use case.
    
    Examples
    --------
    >>> from brisk.training.workflow import Workflow
    >>> from brisk.evaluation import evaluation_manager
    >>> import pandas as pd
    >>> 
    >>> class ClassificationWorkflow(Workflow):
    ...     def workflow(self):
    ...         # Train models
    ...         from sklearn.ensemble import RandomForestClassifier
    ...         model = RandomForestClassifier()
    ...         model.fit(self.X_train, self.y_train)
    ...         
    ...         # Evaluate model
    ...         self.evaluate_model(
    ...             model, self.X_test, self.y_test,
    ...             ['accuracy', 'precision', 'recall'], 'rf_results'
    ...         )
    ...         
    ...         # Generate plots
    ...         self.plot_confusion_heatmap(
    ...             model, self.X_test.values, 
    ...             self.y_test.values, 'confusion_matrix'
    ...         )
    ...         self.plot_roc_curve(model, self.X_test.values, 
    ...                           self.y_test.values, 'roc_curve')
    ...         
    ...         # Save model
    ...         self.save_model(model, 'trained_model')
    """
    def __init__(
        self,
        evaluation_manager: eval_manager.EvaluationManager,
        X_train: pd.DataFrame, # pylint: disable=C0103
        X_test: pd.DataFrame, # pylint: disable=C0103
        y_train: pd.Series,
        y_test: pd.Series,
        output_dir: str,
        algorithm_names: List[str],
        feature_names: List[str],
        workflow_attributes: Dict[str, Any]
    ) -> None:
        """Initialize the Workflow with data and configuration.
        
        This constructor sets up the workflow with all necessary data and
        configuration parameters. It initializes the evaluation manager,
        data objects with appropriate attributes, and unpacks any additional
        workflow-specific attributes.
        
        Parameters
        ----------
        evaluation_manager : EvaluationManager
            Manager for model evaluation, visualization, and analysis
        X_train : pd.DataFrame
            Training feature data
        X_test : pd.DataFrame
            Test feature data
        y_train : pd.Series
            Training target data
        y_test : pd.Series
            Test target data
        output_dir : str
            Directory path where results will be saved
        algorithm_names : List[str]
            List of algorithm names for the workflow
        feature_names : List[str]
            List of feature names for the dataset
        workflow_attributes : Dict[str, Any]
            Additional attributes to be unpacked as instance attributes
            
        Notes
        -----
        The constructor performs the following initialization steps:
        1. Sets up the evaluation manager
        2. Assigns data objects with 'is_test' attributes for identification
        3. Sets output directory and metadata
        4. Unpacks workflow-specific attributes as instance variables
        
        All data objects are marked with an 'is_test' attribute to distinguish
        between training and test data throughout the workflow.
        """
        self.evaluation_manager = evaluation_manager
        self.X_train = X_train # pylint: disable=C0103
        self.X_train.attrs["is_test"] = False
        self.X_test = X_test # pylint: disable=C0103
        self.X_test.attrs["is_test"] = True
        self.y_train = y_train
        self.y_train.attrs["is_test"] = False
        self.y_test = y_test
        self.y_test.attrs["is_test"] = True
        self.output_dir = output_dir
        self.algorithm_names = algorithm_names
        self.feature_names = feature_names
        self._unpack_attributes(workflow_attributes)

    def _unpack_attributes(self, config: Dict[str, Any]) -> None:
        """Unpack configuration dictionary into instance attributes.
        
        This method dynamically sets instance attributes from a configuration
        dictionary, allowing for flexible workflow configuration. This is
        commonly used to unpack model instances or other workflow-specific
        objects that were passed in the workflow_attributes parameter.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary containing key-value pairs to unpack
            as instance attributes
            
        Notes
        -----
        The method iterates through the configuration dictionary and uses
        setattr() to dynamically create instance attributes. This allows
        workflows to be configured with arbitrary objects like pre-trained
        models, custom parameters, or other workflow-specific components.
        
        Examples
        --------
        >>> workflow_attrs = {'model1': RandomForestClassifier(), 'param1': 42}
        >>> workflow._unpack_attributes(workflow_attrs)
        >>> # Now workflow.model1 and workflow.param1 are available
        """
        for key, model in config.items():
            setattr(self, key, model)

    def run(self) -> None: # pragma: no cover
        """Execute the workflow.
        
        This method serves as the entry point for running the workflow.
        It delegates to the abstract `workflow()` method that must be
        implemented by subclasses, providing a consistent interface
        for workflow execution.
        
        Notes
        -----
        This method is marked with `# pragma: no cover` because it's
        an abstract method that should be overridden by subclasses.
        The actual workflow logic is implemented in the `workflow()`
        method of concrete subclasses.
        
        Raises
        ------
        NotImplementedError
            If called directly on the base Workflow class without
            implementing the abstract `workflow()` method
        """
        self.workflow(
            self.X_train, self.X_test, self.y_train, self.y_test,
            self.output_dir, self.feature_names
        )

    @abc.abstractmethod
    def workflow(self) -> None:
        """Abstract method defining the workflow logic.
        
        This method must be implemented by all concrete subclasses of Workflow.
        It should contain the specific logic for the machine learning workflow,
        including model training, evaluation, visualization, and result saving.
        
        Notes
        -----
        This is an abstract method that must be implemented by subclasses.
        The implementation should define the complete workflow logic for
        the specific use case, utilizing the available data (X_train, X_test,
        y_train, y_test) and the evaluation manager for model assessment.
        
        Typical workflow implementations include:
        - Model instantiation and training
        - Model evaluation using provided methods
        - Visualization generation
        - Result saving and reporting
        
        Raises
        ------
        NotImplementedError
            Always raises this error as it's an abstract method
        """
        raise NotImplementedError(
            "Subclass must implement the workflow method."
        )

    # Interface to call Evaluators registered to EvaluationManager
    def evaluate_model( # pragma: no cover
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
        y: pd.Series,
        metrics: List[str],
        filename: str
    ) -> None:
        """Evaluate model on specified metrics and save results.
        
        This method evaluates a trained model on the provided data using
        the specified metrics and saves the results to files. It delegates
        to the evaluation manager to perform the actual evaluation.
        
        Parameters
        ----------
        model : BaseEstimator
            Trained model to evaluate (must have predict method)
        X : pd.DataFrame
            Feature data for evaluation
        y : pd.Series
            Target data for evaluation
        metrics : List[str]
            List of metric names to calculate (e.g., ['accuracy', 'precision'])
        filename : str
            Base filename for saving results (without extension)
            
        Notes
        -----
        The method uses the 'brisk_evaluate_model' evaluator from the
        evaluation manager. Results are saved in the workflow's output
        directory with the specified filename.
        
        Examples
        --------
        >>> workflow.evaluate_model(model, X_test, y_test, 
        ...                        ['accuracy', 'precision', 'recall'], 
        ...                        'model_evaluation')
        """
        evaluator = self.evaluation_manager.get_evaluator(
            "brisk_evaluate_model"
        )
        return evaluator.evaluate(model, X, y, metrics, filename)

    def evaluate_model_cv( # pragma: no cover
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
        y: pd.Series,
        metrics: List[str],
        filename: str,
        cv: int = 5
    ) -> None:
        """Evaluate model using cross-validation.
        
        This method evaluates a model using k-fold cross-validation to
        provide more robust performance estimates. It trains and evaluates
        the model on multiple train/test splits and saves the results.
        
        Parameters
        ----------
        model : BaseEstimator
            Model to evaluate (will be cloned for each CV fold)
        X : pd.DataFrame
            Feature data for evaluation
        y : pd.Series
            Target data for evaluation
        metrics : List[str]
            List of metric names to calculate
        filename : str
            Base filename for saving results (without extension)
        cv : int, default=5
            Number of cross-validation folds to use
            
        Notes
        -----
        The method uses the 'brisk_evaluate_model_cv' evaluator from the
        evaluation manager. Cross-validation provides more reliable
        performance estimates by testing on multiple data splits.
        
        Examples
        --------
        >>> workflow.evaluate_model_cv(model, X_train, y_train,
        ...                           ['accuracy', 'f1_score'], 
        ...                           'cv_evaluation', cv=10)
        """
        evaluator = self.evaluation_manager.get_evaluator(
            "brisk_evaluate_model_cv"
        )
        return evaluator.evaluate(model, X, y, metrics, filename, cv)

    def compare_models( # pragma: no cover
        self,
        *models: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
        y: pd.Series,
        metrics: List[str],
        filename: str,
        calculate_diff: bool = False
    ) -> Dict[str, Dict[str, float]]:
        """Compare multiple models using specified metrics.
        
        This method evaluates multiple models on the same data and metrics,
        allowing for direct comparison of their performance. It can optionally
        calculate differences between model performances.
        
        Parameters
        ----------
        *models : BaseEstimator
            Variable number of trained models to compare
        X : pd.DataFrame
            Feature data for evaluation
        y : pd.Series
            Target data for evaluation
        metrics : List[str]
            List of metric names to calculate for comparison
        filename : str
            Base filename for saving comparison results (without extension)
        calculate_diff : bool, default=False
            Whether to compute and include differences between model
            performances
            
        Returns
        -------
        Dict[str, Dict[str, float]]
            Nested dictionary with model names as keys and metric results as
            values.
            Structure: {model_name: {metric_name: metric_value}}
            
        Notes
        -----
        The method uses the 'brisk_compare_models' evaluator from the
        evaluation manager. Results are saved and returned for further
        analysis or reporting.
        
        Examples
        --------
        >>> results = workflow.compare_models(model1, model2, model3,
        ...                                  X=X_test, y=y_test,
        ...                                  metrics=['accuracy', 'f1_score'],
        ...                                  filename='model_comparison',
        ...                                  calculate_diff=True)
        >>> print(results['model1']['accuracy'])
        """
        evaluator = self.evaluation_manager.get_evaluator(
            "brisk_compare_models"
        )
        return evaluator.evaluate(
            *models, X=X, y=y, metrics=metrics, filename=filename,
            calculate_diff=calculate_diff
        )

    def plot_pred_vs_obs( # pragma: no cover
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
        y_true: pd.Series,
        filename: str
    ) -> None:
        """Plot predicted vs. observed values and save the plot.
        
        This method generates a scatter plot comparing predicted values
        against observed values, which is useful for regression model
        evaluation and identifying prediction patterns.
        
        Parameters
        ----------
        model : BaseEstimator
            Trained model with predict method
        X : pd.DataFrame
            Feature data for making predictions
        y_true : pd.Series
            True target values for comparison
        filename : str
            Output filename for the plot (without extension)
            
        Notes
        -----
        The method uses the 'brisk_plot_pred_vs_obs' evaluator from the
        evaluation manager. The plot helps assess model performance by
        showing how well predictions align with actual values.
        
        Examples
        --------
        >>> workflow.plot_pred_vs_obs(model, X_test, y_test, 'pred_vs_obs')
        """
        evaluator = self.evaluation_manager.get_evaluator(
            "brisk_plot_pred_vs_obs"
        )
        return evaluator.plot(model, X, y_true, filename)

    def plot_learning_curve( # pragma: no cover
        self,
        model: base.BaseEstimator,
        X_train: pd.DataFrame, # pylint: disable=C0103
        y_train: pd.Series,
        filename: str = "learning_curve",
        cv: int = 5,
        num_repeats: int = 1,
        n_jobs: int = -1,
        metric: str = "neg_mean_absolute_error"
    ) -> None:
        """Plot learning curves showing model performance vs training size.
        
        This method generates learning curves that show how model performance
        changes as the training set size increases. This helps identify
        whether the model would benefit from more data or if it's suffering
        from high bias or variance.
        
        Parameters
        ----------
        model : BaseEstimator
            Model to evaluate (will be cloned for each training size)
        X_train : pd.DataFrame
            Training feature data
        y_train : pd.Series
            Training target data
        filename : str, default="learning_curve"
            Base filename for saving the plot (without extension)
        cv : int, default=5
            Number of cross-validation folds for each training size
        num_repeats : int, default=1
            Number of times to repeat cross-validation for stability
        n_jobs : int, default=-1
            Number of parallel jobs for cross-validation (-1 uses all cores)
        metric : str, default="neg_mean_absolute_error"
            Scoring metric to use for evaluation
            
        Notes
        -----
        The method uses the 'brisk_plot_learning_curve' evaluator from the
        evaluation manager. Learning curves help diagnose model behavior:
        - High bias: both training and validation scores are low
        - High variance: large gap between training and validation scores
        - Good fit: both scores converge to similar high values
        
        Examples
        --------
        >>> workflow.plot_learning_curve(model, X_train, y_train,
        ...                             filename='rf_learning_curve',
        ...                             cv=10, metric='neg_mean_squared_error')
        """
        evaluator = self.evaluation_manager.get_evaluator(
            "brisk_plot_learning_curve"
        )
        return evaluator.plot(
            model, X_train, y_train, filename=filename, cv=cv,
            num_repeats=num_repeats, n_jobs=n_jobs, metric=metric
        )

    def plot_feature_importance( # pragma: no cover
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
        """Plot feature importance for the model and save the plot.
        
        This method generates a plot showing the importance of each feature
        in the model's predictions. It can filter features by importance
        threshold or number of top features.
        
        Parameters
        ----------
        model : BaseEstimator
            Trained model with feature importance or permutation importance
        X : pd.DataFrame
            Feature data for importance calculation
        y : pd.Series
            Target data for importance calculation
        threshold : Union[int, float]
            If int: number of top features to show
            If float: minimum importance threshold for features
        feature_names : List[str]
            List of feature names corresponding to X columns
        filename : str
            Output filename for the plot (without extension)
        metric : str
            Metric to use for importance calculation
        num_rep : int
            Number of repetitions for calculating importance (for stability)
            
        Notes
        -----
        The method uses the 'brisk_plot_feature_importance' evaluator from the
        evaluation manager. Feature importance helps identify which features
        contribute most to model predictions.
        
        Examples
        --------
        >>> workflow.plot_feature_importance(model, X_train, y_train,
        ...                                 threshold=10,
        ...                                 feature_names=feature_names,
        ...                                 filename='feature_importance',
        ...                                 metric='accuracy', num_rep=5)
        """
        evaluator = self.evaluation_manager.get_evaluator(
            "brisk_plot_feature_importance"
        )
        return evaluator.plot(
            model, X, y, threshold, feature_names, filename, metric, num_rep
        )

    def plot_residuals( # pragma: no cover
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
        y: pd.Series,
        filename: str,
        add_fit_line: bool = False
    ) -> None:
        """Plot residuals of the model and save the plot.
        
        This method generates a residual plot showing the difference between
        predicted and actual values. Residual plots help assess model
        assumptions and identify patterns in prediction errors.
        
        Parameters
        ----------
        model : BaseEstimator
            Trained model with predict method
        X : pd.DataFrame
            Feature data for making predictions
        y : pd.Series
            True target values
        filename : str
            Output filename for the plot (without extension)
        add_fit_line : bool, default=False
            Whether to add a line of best fit to the residual plot
            
        Notes
        -----
        The method uses the 'brisk_plot_residuals' evaluator from the
        evaluation manager. Residual plots help identify:
        - Non-linear patterns in residuals
        - Heteroscedasticity (varying variance)
        - Outliers and influential points
        
        Examples
        --------
        >>> workflow.plot_residuals(
        ...     model, X_test, y_test, 'residuals', add_fit_line=True
        ... )
        """
        evaluator = self.evaluation_manager.get_evaluator(
            "brisk_plot_residuals"
        )
        return evaluator.plot(model, X, y, filename, add_fit_line=add_fit_line)

    def plot_model_comparison( # pragma: no cover
        self,
        *models: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
        y: pd.Series,
        metric: str,
        filename: str
    ) -> None:
        """Plot comparison of multiple models based on specified metric.
        
        This method generates a visualization comparing the performance of
        multiple models on a single metric, making it easy to identify
        the best performing model.
        
        Parameters
        ----------
        *models : BaseEstimator
            Variable number of trained models to compare
        X : pd.DataFrame
            Feature data for evaluation
        y : pd.Series
            Target data for evaluation
        metric : str
            Single metric name to use for comparison
        filename : str
            Output filename for the plot (without extension)
            
        Notes
        -----
        The method uses the 'brisk_plot_model_comparison' evaluator from the
        evaluation manager. The plot typically shows model names on one axis
        and metric values on the other, making performance comparison easy.
        
        Examples
        --------
        >>> workflow.plot_model_comparison(model1, model2, model3,
        ...                               X=X_test, y=y_test,
        ...                               metric='accuracy',
        ...                               filename='model_comparison'
        ...                          )
        """
        evaluator = self.evaluation_manager.get_evaluator(
            "brisk_plot_model_comparison"
        )
        return evaluator.plot(
            *models, X=X, y=y, metric=metric, filename=filename
        )

    def hyperparameter_tuning( # pragma: no cover
        self,
        model: base.BaseEstimator,
        method: str,
        X_train: pd.DataFrame, # pylint: disable=C0103
        y_train: pd.Series,
        scorer: str,
        kf: int,
        num_rep: int,
        n_jobs: int,
        plot_results: bool = False
    ) -> base.BaseEstimator:
        """Perform hyperparameter tuning using grid or random search.
        
        This method optimizes model hyperparameters using either grid search
        or random search with cross-validation. It returns the best model
        found during the search process.
        
        Parameters
        ----------
        model : BaseEstimator
            Base model to tune (will be cloned for each parameter combination)
        method : str
            Search method to use ('grid' or 'random')
        X_train : pd.DataFrame
            Training feature data
        y_train : pd.Series
            Training target data
        scorer : str
            Scoring metric to optimize (e.g., 'accuracy',
            'neg_mean_squared_error')
        kf : int
            Number of cross-validation folds
        num_rep : int
            Number of CV repetitions for stability
        n_jobs : int
            Number of parallel jobs (-1 uses all cores)
        plot_results : bool, default=False
            Whether to generate plots showing hyperparameter performance
            
        Returns
        -------
        BaseEstimator
            Best model found during hyperparameter search
            
        Notes
        -----
        The method uses the 'brisk_hyperparameter_tuning' evaluator from the
        evaluation manager. The search process tests different parameter
        combinations and selects the one with the best cross-validation score.
        
        Examples
        --------
        >>> tuned_model = workflow.hyperparameter_tuning(
        ...     RandomForestClassifier(), 'grid', X_train, y_train,
        ...     'accuracy', kf=5, num_rep=3, n_jobs=-1, plot_results=True)
        """
        evaluator = self.evaluation_manager.get_evaluator(
            "brisk_hyperparameter_tuning"
        )
        return evaluator.evaluate(
            model, method, X_train, y_train, scorer,
            kf, num_rep, n_jobs, plot_results=plot_results
        )

    def confusion_matrix( # pragma: no cover
        self,
        model: Any,
        X: np.ndarray, # pylint: disable=C0103
        y: np.ndarray,
        filename: str
    ) -> None:
        """Generate and save a confusion matrix.
        
        This method creates a confusion matrix for classification models,
        showing the count of correct and incorrect predictions for each
        class. It's useful for understanding model performance on
        classification tasks.
        
        Parameters
        ----------
        model : Any
            Trained classification model with predict method
        X : np.ndarray
            Feature data for making predictions
        y : np.ndarray
            True class labels
        filename : str
            Output filename for the confusion matrix (without extension)
            
        Notes
        -----
        The method uses the 'brisk_confusion_matrix' evaluator from the
        evaluation manager. The confusion matrix shows:
        - True Positives (TP): Correctly predicted positive cases
        - True Negatives (TN): Correctly predicted negative cases
        - False Positives (FP): Incorrectly predicted positive cases
        - False Negatives (FN): Incorrectly predicted negative cases
        
        Examples
        --------
        >>> workflow.confusion_matrix(
        ...     model, X_test.values, y_test.values, 'confusion_matrix'
        ... )
        """
        evaluator = self.evaluation_manager.get_evaluator(
            "brisk_confusion_matrix"
        )
        return evaluator.evaluate(model, X, y, filename)

    def plot_confusion_heatmap( # pragma: no cover
        self,
        model: Any,
        X: np.ndarray, # pylint: disable=C0103
        y: np.ndarray,
        filename: str
    ) -> None:
        """Plot a heatmap of the confusion matrix for a model.
        
        This method generates a visual heatmap representation of the
        confusion matrix, making it easier to interpret classification
        performance across different classes.
        
        Parameters
        ----------
        model : Any
            Trained classification model with predict method
        X : np.ndarray
            Feature data for making predictions
        y : np.ndarray
            True class labels
        filename : str
            Output filename for the heatmap plot (without extension)
            
        Notes
        -----
        The method uses the 'brisk_plot_confusion_heatmap' evaluator from the
        evaluation manager. The heatmap uses color intensity to show the
        count of predictions, making it easy to identify patterns in
        classification errors.
        
        Examples
        --------
        >>> workflow.plot_confusion_heatmap(
        ...     model, X_test.values, y_test.values, 'confusion_heatmap'
        ... )
        """
        evaluator = self.evaluation_manager.get_evaluator(
            "brisk_plot_confusion_heatmap"
        )
        return evaluator.plot(model, X, y, filename)

    def plot_roc_curve( # pragma: no cover
        self,
        model: Any,
        X: np.ndarray, # pylint: disable=C0103
        y: np.ndarray,
        filename: str,
        pos_label: Optional[int] = 1
    ) -> None:
        """Plot a receiver operating characteristic (ROC) curve with AUC.
        
        This method generates a ROC curve for binary classification models,
        showing the trade-off between true positive rate and false positive
        rate at different classification thresholds.
        
        Parameters
        ----------
        model : Any
            Trained binary classification model with predict_proba method
        X : np.ndarray
            Feature data for making predictions
        y : np.ndarray
            True binary class labels (0 and 1)
        filename : str
            Output filename for the ROC curve plot (without extension)
        pos_label : Optional[int], default=1
            Label of the positive class for ROC calculation
            
        Notes
        -----
        The method uses the 'brisk_plot_roc_curve' evaluator from the
        evaluation manager. The ROC curve shows:
        - X-axis: False Positive Rate (1 - Specificity)
        - Y-axis: True Positive Rate (Sensitivity)
        - AUC: Area Under the Curve (higher is better)
        
        Examples
        --------
        >>> workflow.plot_roc_curve(
        ...     model, X_test.values, y_test.values, 'roc_curve'
        ... )
        """
        evaluator = self.evaluation_manager.get_evaluator(
            "brisk_plot_roc_curve"
        )
        return evaluator.plot(model, X, y, filename, pos_label)

    def plot_precision_recall_curve( # pragma: no cover
        self,
        model: Any,
        X: np.ndarray, # pylint: disable=C0103
        y: np.ndarray,
        filename: str,
        pos_label: Optional[int] = 1
    ) -> None:
        """Plot a precision-recall curve with average precision.
        
        This method generates a precision-recall curve for binary classification
        models, showing the trade-off between precision and recall at different
        classification thresholds. This is particularly useful for imbalanced
        datasets.
        
        Parameters
        ----------
        model : Any
            Trained binary classification model with predict_proba method
        X : np.ndarray
            Feature data for making predictions
        y : np.ndarray
            True binary class labels (0 and 1)
        filename : str
            Output filename for the precision-recall curve plot
            (without extension)
        pos_label : Optional[int], default=1
            Label of the positive class for precision-recall calculation
            
        Notes
        -----
        The method uses the 'brisk_plot_precision_recall_curve' evaluator from
        the evaluation manager. The precision-recall curve shows:
        - X-axis: Recall (True Positive Rate)
        - Y-axis: Precision (Positive Predictive Value)
        - AP: Average Precision (higher is better)
        
        Precision-recall curves are especially useful for imbalanced datasets
        where the positive class is rare.
        
        Examples
        --------
        >>> workflow.plot_precision_recall_curve(
        ...     model, X_test.values, y_test.values, 'pr_curve'
        ... )
        """
        evaluator = self.evaluation_manager.get_evaluator(
            "brisk_plot_precision_recall_curve"
        )
        return evaluator.plot(
            model, X, y, filename, pos_label
        )

    def save_model(
        self,
        model: base.BaseEstimator,
        filename: str
    ) -> None: #pragma: no cover
        """Save model to pickle file.
        
        This method saves a trained model to a pickle file in the workflow's
        output directory, allowing it to be loaded later for inference or
        further analysis.
        
        Parameters
        ----------
        model : BaseEstimator
            Trained model to save
        filename : str
            Base filename for the saved model (without extension)
            
        Notes
        -----
        The method delegates to the evaluation manager's save_model method.
        The model is saved in the workflow's output directory with the
        specified filename.
        
        Examples
        --------
        >>> workflow.save_model(trained_model, 'my_model')
        """
        self.evaluation_manager.save_model(model, filename)

    def load_model(self, filepath: str) -> base.BaseEstimator: #pragma: no cover
        """Load model from pickle file.
        
        This method loads a previously saved model from a pickle file,
        allowing it to be used for inference or further analysis.
        
        Parameters
        ----------
        filepath : str
            Path to the saved model file (with extension)
            
        Returns
        -------
        BaseEstimator
            Loaded model ready for use
            
        Raises
        ------
        FileNotFoundError
            If the model file does not exist at the specified path
            
        Notes
        -----
        The method delegates to the evaluation manager's load_model method.
        The loaded model can be used for making predictions or further
        evaluation.
        
        Examples
        --------
        >>> loaded_model = workflow.load_model('my_model.pkl')
        >>> predictions = loaded_model.predict(X_new)
        """
        self.evaluation_manager.load_model(filepath)

    def plot_shapley_values( # pragma: no cover
        self,
        model: base.BaseEstimator,
        X: pd.DataFrame, # pylint: disable=C0103
        y: pd.Series,
        filename: str = "shapley_values",
        plot_type: str = "bar"
    ) -> None:
        """Generate SHAP value plots for feature importance.
        
        This method generates SHAP (SHapley Additive exPlanations) value plots
        to explain individual predictions and feature importance. SHAP values
        provide a unified framework for explaining model predictions.
        
        Parameters
        ----------
        model : BaseEstimator
            Trained model to explain (must be compatible with SHAP)
        X : pd.DataFrame
            Feature data for generating explanations
        y : pd.Series
            Target data (used for context in some plot types)
        filename : str, default="shapley_values"
            Base output filename for SHAP plots (without extension)
        plot_type : str, default="bar"
            Type of SHAP plot to generate. Options:
            - 'bar': Bar plot of mean SHAP values
            - 'waterfall': Waterfall plot for individual predictions
            - 'violin': Violin plot showing SHAP value distributions
            - 'beeswarm': Beeswarm plot for feature importance
            Multiple types can be specified as 'bar,waterfall' to generate
            multiple plots
            
        Notes
        -----
        The method uses the 'brisk_plot_shapley_values' evaluator from the
        evaluation manager. SHAP values provide:
        - Feature importance rankings
        - Individual prediction explanations
        - Feature interaction effects
        - Model interpretability insights
        
        Examples
        --------
        >>> workflow.plot_shapley_values(model, X_test, y_test, 
        ...                             'shap_explanation', 'bar,waterfall')
        """
        evaluator = self.evaluation_manager.get_evaluator(
            "brisk_plot_shapley_values"
        )
        return evaluator.plot(
            model, X, y, filename=filename, plot_type=plot_type
        )
