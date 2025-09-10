.. _custom_evaluators:

Creating Custom Evaluators
===========================

You can create custom plots and evaluation methods beyond the built-in evaluators by 
defining them in your project's ``evaluators.py`` file. Custom evaluators integrate 
with Brisk's evaluation system and appear in the interactive report.


Built-in vs Custom Evaluators
------------------------------

**Built-in Evaluators** are provided by Brisk and include common plots (learning curves, 
feature importance, SHAP values) and evaluation methods (cross-validation, model comparison).

**Custom Evaluators** are methods you add to analyze your models beyond Brisk's built-in evaluations.
They allow you to create specialized visualizations, or add any custom analysis logic your project needs.

Types of Custom Evaluators
---------------------------

**Measure Evaluators** (``MeasureEvaluator``):
Calculate numerical metrics and store results as JSON files.

**Plot Evaluators** (``PlotEvaluator``):
Generate visualizations and saves them as image files.

Creating Custom Evaluators
---------------------------

Custom evaluators are defined in your project's ``evaluators.py`` file. There are two types you can create:


Custom Measure Evaluators
-------------------------

To implement a custom measure evaluator define a new class
in ``evaluators.py`` and implement the ``_calculate_measures`` method. The default
arguments for this method are:

- predictions: the model predictions
- y_true: the true target values
- metrics: the list of metric names to calculate

The method should return a dictionary of the calculated values. 

.. note:: 
    To access the metric scorer callable you can call ``self.metric_config.get_metric(metric_name)``.
    
    To access the metric display name call ``self.metric_config.get_name(metric_name)``.

Here is an example of a custom measure evaluator:

.. code-block:: python

   # evaluators.py
   from brisk.evaluation.evaluators import MeasureEvaluator
   import pandas as pd
   from typing import Dict, Any

   class ExampleMeasureEvaluator(MeasureEvaluator):
       def _calculate_measures(self, predictions, y_true, metrics) -> Dict[str, Any]:
           """Calculate prediction summary statistics."""           
            results = {}            
            for metric_name in metrics:
                scorer = self.metric_config.get_metric(metric_name)
                display_name = self.metric_config.get_name(metric_name)
                metric_value = scorer(y_true, predictions)
                results[display_name] = float(metric_value)
            return results

.. note::
    If these arguments are not suitable for your evaluator you can override the ``evaluate`` method.
    The default evaluate method is:
    
    .. code-block:: python
    
       def evaluate(self, model, X, y, metrics, filename):
            predictions = self._generate_prediction(model, X)
            results = self._calculate_measures(predictions, y, metrics)
            metadata = self._generate_metadata(model, X.attrs["is_test"])
            self._save_json(results, filename, metadata)
            self._log_results(results, filename)

    This is the method you call in your workflow to use this evaluator. All of the
    arguments can be changed by overriding this method.
    
    However, the flow of the ``evaluate`` method should be preserved. Specifically the
    following steps should be called exactly as shown here to avoid errors at runtime.

    .. code-block:: python

            metadata = self._generate_metadata(model, X.attrs["is_test"])
            self._save_json(results, filename, metadata)
            self._log_results(results, filename)

To integrate with the interactive report you need to implement the ``report`` method
in order to format the results dictionary returned by ``_calculate_measures`` into 
a format suitable for the report table generation. This method should take the results
dictionary as an argument and return a tuple of two lists:

- List of column headers
- Nested list where each list is a row of the table

Here is an example of a report method for the example measure evaluator:

.. code-block:: python

   def report(self, results: Dict[str, Any]):
       """Report the evaluation results."""
       columns = [key for key in results.keys() if key != "_metadata"]
        row = []
        for col in columns:
            row.append(results[col])
        return columns, [row]

Our complete custom evaluator looks like this:

.. code-block:: python

   from brisk.evaluation.evaluators import MeasureEvaluator
   import pandas as pd
   from typing import Dict, Any

   class ExampleMeasureEvaluator(MeasureEvaluator):
       def _calculate_measures(self, predictions, y_true, metrics) -> Dict[str, Any]:
           """Calculate prediction summary statistics."""           
            results = {}            
            for metric_name in metrics:
                scorer = self.metric_config.get_metric(metric_name)
                display_name = self.metric_config.get_name(metric_name)
                metric_value = scorer(y_true, predictions)
                results[display_name] = float(metric_value)
            return results

    def report(self, results: Dict[str, Any]):
        """Report the evaluation results."""
        columns = [key for key in results.keys() if key != "_metadata"]
            row = []
            for col in columns:
                row.append(results[col])
            return columns, [row]


Custom Plot Evaluators
----------------------

As with the measure evaluators, you can create a custom plot evaluator by defining a new class
in ``evaluators.py`` and implementing the ``_generate_plot_data`` and ``_create_plot`` methods. ``_generate_plot_data``
will return a dictionary of values that can be used to create the plot. ``_create_plot`` will take this dictionary
and implement the plot creation logic.

.. note::
    Brisk supports several plotting libraries including plotnine, matplotlib, seaborn, and plotly.

The default parameters for ``_generate_plot_data`` are:

- model: the trained model
- X: the input data
- y: the true target values

Here is an example of a custom plot evaluator:

.. code-block:: python

   from brisk.evaluation.evaluators import PlotEvaluator
   import plotnine as pn

   class PlotErrorHistogram(PlotEvaluator):
       def _generate_plot_data(self, model, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
           """Generate data for the error histogram plot."""
           y_pred = self._generate_prediction(model, X)
           errors = y - y_pred
           
           return pd.DataFrame({
               'errors': errors,
               'abs_errors': abs(errors)
           })
       
       def _create_plot(self, plot_data: pd.DataFrame, display_name: str):
           """Create an error histogram plot."""
           plot = (pn.ggplot(plot_data, pn.aes(x='errors')) +
                   pn.geom_histogram(bins=30, fill='skyblue', alpha=0.7) +
                   pn.labs(title=f'Prediction Error Distribution - {display_name}',
                          x='Prediction Error',
                          y='Frequency') +
                   self.theme)
           return plot

For the ``_create_plot`` method adding ``self.theme`` can be used if creating plots
with plotnine. This will apply the same styling as the built-in plots. This is not
required and you are free to implement your own styling.

.. note::
    If the ``_generate_plot_data`` method is not suitable for your evaluator you
    can override the ``plot`` method. The default plot method is:
    
    .. code-block:: python
    
       def plot(self, model, X, y, filename):
        plot_data = self._generate_plot_data(model, X, y)
        plot = self._create_plot(plot_data)
        metadata = self._generate_metadata(model, X.attrs["is_test"])
        self._save_plot(filename, metadata, plot=plot)
        self._log_results(self.method_name, filename)

    This is the method you call in your workflow to use this evaluator. All of the
    arguments can be changed by overriding this method.

    However, the flow of the ``plot`` method should be preserved. Specifically the
    following steps should be called exactly as shown here to avoid errors at runtime.
        
    .. code-block:: python

        plot = self._create_plot(plot_data)
        metadata = self._generate_metadata(model, X.attrs["is_test"])
        self._save_plot(filename, metadata, plot=plot)
        self._log_results(self.method_name, filename)

No other methods are needed to implement a custom plot evaluator.

Registering Custom Evaluators
------------------------------

After defining your custom evaluator classes, you must register them with Brisk by adding a ``register_custom_evaluators()`` function to your ``evaluators.py`` file.
This can be done with the ``registry.register()`` method. You provide a name used to
access the evaluator and a description that will be displayed in the report.

.. code-block:: python

   from brisk.evaluation.evaluators.registry import EvaluatorRegistry

   def register_custom_evaluators(registry: EvaluatorRegistry, theme) -> None:
       """Register custom evaluators with Brisk.
       
       Parameters
       ----------
       registry : EvaluatorRegistry
           The evaluator registry to register with
       theme : plotnine theme
           The plotting theme for plot evaluators
       """
       # Register custom measure evaluators (no theme needed)
       registry.register(ExampleMeasureEvaluator(
           "evaluate_prediction", 
           "Display evaluation results"
       ))
       
       # Register custom plot evaluators (theme is required)
       registry.register(PlotErrorHistogram(
           "plot_error_histogram",
           "Plot prediction error distribution",
           theme
       ))

.. important::
    For PlotEvaluators you must pass the ``theme`` to the constructor. This provides
    information about how to save the images.

Calling Custom Evaluators in Workflows
---------------------------------------

Once registered, you can call your custom evaluators in workflows using the ``.evaluate()`` or ``.plot()`` methods. You can do this in two ways:

**Wrapper Methods** (recommended for cleaner code):

By registering the evluators in the step above they will be included in the evaluation manager.
This allows you to access them using the ``self.evaluation_manager.get_evaluator()`` method.

.. code-block:: python

   # workflows/my_workflow.py
   from brisk.training.workflow import Workflow

   class MyWorkflow(Workflow):
       def evaluate_prediction(self, model, X, y, filename):
           """Wrapper method for custom prediction summary evaluator."""
           evaluator = self.evaluation_manager.get_evaluator("evaluate_prediction")
           return evaluator.evaluate(model, X, y, ["MSE", "R2"], filename=filename)
           
       def plot_error_histogram(self, model, X, y, display_name):
           """Wrapper method for custom error histogram plot."""
           evaluator = self.evaluation_manager.get_evaluator("plot_error_histogram")
           return evaluator.plot(model, X, y, display_name=display_name)
       
       def workflow(self, X_train, X_test, y_train, y_test, output_dir, feature_names):
           # Fit the model
           self.model.fit(X_train, y_train)
           
           # Use built-in methods
           self.evaluate_model(
               self.model, X_test, y_test,
               ["mean_absolute_error"], "model_score"
           )
           
           # Use custom wrapper methods
           self.evaluate_prediction(self.model, X_test, y_test, "prediction_summary")
           self.plot_error_histogram(self.model, X_test, y_test, "error_histogram")


**Direct Calling**:

You may also access the evaluators directly using the ``self.evaluation_manager.get_evaluator()`` method.

.. code-block:: python

   # workflows/my_workflow.py
   from brisk.training.workflow import Workflow

   class MyWorkflow(Workflow):
       def workflow(self):
           # Fit the model
           self.model.fit(self.X_train, self.y_train)

           # Direct calls to custom evaluators
           custom_measure = self.evaluation_manager.get_evaluator("evaluate_prediction")
           custom_measure.evaluate(self.model, X_test, y_test, ["MAE", "R2"], filename="prediction_summary")
           
           custom_plot = self.evaluation_manager.get_evaluator("plot_error_histogram")
           custom_plot.plot(self.model, X_test, y_test, filename="Error Analysis")

.. note::
   Use descriptive names for evaluators (e.g., "\plot_error_histogram" 
   rather than "custom_plot").

Your custom evaluators will appear alongside built-in evaluators in the final interactive 
report. 
