.. _custom_evaluators:

Creating Custom Evaluators
===========================

You can create custom plots and evaluation methods beyond the built-in evaluators by 
defining them in your project's ``evaluators.py`` file. Custom evaluators integrate 
with Brisk's evaluation system and appear in HTML reports.


Built-in vs Custom Evaluators
------------------------------

**Built-in Evaluators** are provided by Brisk and include common plots (learning curves, 
feature importance, SHAP values) and metrics (accuracy, precision, recall, etc.).

**Custom Evaluators** are methods you add to analyze your models beyond Brisk's built-in evaluations.
They allow you to create specialized visualizations, or add any custom analysis logic your project needs.

Types of Custom Evaluators
---------------------------

**Measure Evaluators** (``MeasureEvaluator``):
Calculate numerical metrics and store results as JSON data for reports.

**Plot Evaluators** (``PlotEvaluator``):
Generate visualizations and save plots as PNG files for reports.


Creating Custom Evaluators
---------------------------

Custom evaluators are defined in your project's ``evaluators.py`` file. There are two types you can create:

**Custom Measure Evaluators** calculate metrics and save results as JSON files:

.. code-block:: python

   # evaluators.py
   from brisk.evaluation.evaluators import MeasureEvaluator
   import pandas as pd
   from typing import Dict, Any

   class EvaluatePredictionSummary(MeasureEvaluator):
       def _calculate_measures(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
           """Calculate prediction summary statistics."""
           y_pred = self._generate_prediction(model, X_test)
           
           return {
               "mean_prediction": float(y_pred.mean()),
               "std_prediction": float(y_pred.std()),
               "min_prediction": float(y_pred.min()),
               "max_prediction": float(y_pred.max()),
               "prediction_range": float(y_pred.max() - y_pred.min())
           }

**Custom Plot Evaluators** create visualizations and save them as image files:

.. code-block:: python

   # evaluators.py (continued)
   from brisk.evaluation.evaluators import PlotEvaluator
   import plotnine as p9

   class PlotErrorHistogram(PlotEvaluator):
       def _generate_plot_data(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
           """Generate data for the error histogram plot."""
           y_pred = self._generate_prediction(model, X_test)
           errors = y_test - y_pred
           
           return pd.DataFrame({
               'errors': errors,
               'abs_errors': abs(errors)
           })
       
       def _create_plot(self, plot_data: pd.DataFrame, display_name: str):
           """Create an error histogram plot."""
           plot = (p9.ggplot(plot_data, p9.aes(x='errors')) +
                   p9.geom_histogram(bins=30, fill='skyblue', alpha=0.7) +
                   p9.labs(title=f'Prediction Error Distribution - {display_name}',
                          x='Prediction Error',
                          y='Frequency') +
                   self.theme)
           return plot

Registering Custom Evaluators
------------------------------

After defining your custom evaluator classes, you must register them with Brisk by adding a ``register_custom_evaluators()`` function to your ``evaluators.py`` file:

.. code-block:: python

   # evaluators.py (continued)
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
       registry.register(EvaluatePredictionSummary(
           "brisk_evaluate_prediction_summary", 
           "Calculate prediction summary statistics"
       ))
       
       # Register custom plot evaluators (theme is required)
       registry.register(PlotErrorHistogram(
           "brisk_plot_error_histogram",
           "Plot prediction error distribution",
           theme
       ))


Calling Custom Evaluators in Workflows
---------------------------------------

Once registered, you can call your custom evaluators in workflows using the ``.evaluate()`` or ``.plot()`` methods. You can do this in two ways:

**Wrapper Methods** (recommended for cleaner code):

.. code-block:: python

   # workflows/my_workflow.py
   from brisk.training.workflow import Workflow

   class MyWorkflow(Workflow):
       def evaluate_prediction_summary(self, model, X_test, y_test, filename):
           """Wrapper method for custom prediction summary evaluator."""
           evaluator = self.evaluation_manager.get_evaluator("brisk_evaluate_prediction_summary")
           return evaluator.evaluate(model, X_test, y_test, [], filename=filename)
           
       def plot_error_histogram(self, model, X_test, y_test, display_name):
           """Wrapper method for custom error histogram plot."""
           evaluator = self.evaluation_manager.get_evaluator("brisk_plot_error_histogram")
           return evaluator.plot(model, X_test, y_test, display_name=display_name)
       
       def workflow(self):
           # Fit the model
           self.model.fit(self.X_train, self.y_train)
           
           # Use built-in methods
           self.evaluate_model(
               self.model, self.X_test, self.y_test,
               ["mean_absolute_error"], "model_score"
           )
           
           # Use custom wrapper methods
           self.evaluate_prediction_summary(self.model, self.X_test, self.y_test, "prediction_summary")
           self.plot_error_histogram(self.model, self.X_test, self.y_test, "Error Analysis")


**Direct Calling**:

.. code-block:: python

   # workflows/my_workflow.py
   from brisk.training.workflow import Workflow

   class MyWorkflow(Workflow):
       def workflow(self):
           # Fit the model
           self.model.fit(self.X_train, self.y_train)
           
           # Call built-in evaluator
           self.evaluate_model(
               self.model, self.X_test, self.y_test,
               ["mean_absolute_error"], "model_score"
           )
           
           # Direct calls to custom evaluators
           custom_measure = self.evaluation_manager.get_evaluator("brisk_evaluate_prediction_summary")
           custom_measure.evaluate(self.model, self.X_test, self.y_test, [], filename="prediction_summary")
           
           custom_plot = self.evaluation_manager.get_evaluator("brisk_plot_error_histogram")
           custom_plot.plot(self.model, self.X_test, self.y_test, display_name="Error Analysis")

.. note::
   Use descriptive names for evaluators (e.g., "brisk_plot_error_histogram" 
   rather than "custom_plot").

.. important::
   Always use ``self.theme`` in plotnine plots to maintain 
   visual consistency with built-in evaluators.

Your custom evaluators will appear alongside built-in evaluators in the final HTML 
report. 