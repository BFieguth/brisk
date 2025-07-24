Create a Workflow
=================

So far we have defined the configuration for our project and have setup 
some experiments to run. We know *what* models we want to train, but we haven't 
defined *how* to train and evaluate them. This is where the ``Workflow`` class 
comes in. In Brisk, a Workflow is what we call the steps we want to take for 
each experiment.

In ``workflows/workflow.py`` you will see a class called ``MyWorkflow`` that inherits 
the ``Workflow`` class. You will also see a ``workflow`` method that takes no 
arguments. This is where you define the steps you want to take for each experiment.
Brisk will pass the correct model, data splits and evaluation methods to the workflow 
for each experiment. 

There are a few instance variables that are available to the workflow:

- ``self.model`` refers to the model we are training.
- ``self.X_train`` and ``self.y_train`` provide the training data.
- ``self.X_test`` and ``self.y_test`` provide the testing data.

We will create a simple workflow as follows:

.. code-block:: python

    class MyWorkflow(Workflow):
        def workflow(self):
            # Fit the model to the training data
            self.model.fit(self.X_train, self.y_train)
            
            # Evaluate the model on the testing data
            self.evaluate_model(
                self.model, self.X_test, self.y_test,
                ["mean_absolute_error"], "pre_tuning_score"
            )
            
            # Tune the model hyperparameters
            tuned_model = self.hyperparameter_tuning(
                self.model, "random", self.X_train, self.y_train,
                "MAE", kf=5, num_rep=2, n_jobs=-1
            )
            
            # Evaluate the tuned model on the testing data
            self.evaluate_model(
                tuned_model, self.X_test, self.y_test,
                ["MAE"], "post_tuning_score"
            )
            
            # Plot the learning curve
            self.plot_learning_curve(
                tuned_model, self.X_train, self.y_train, metric="MAE"
            )

            # Save the tuned model
            self.save_model(tuned_model, "tuned_model")


We can access our mean absolute error metric from ``metrics.py`` by using the name 
or the abbreviation. Remember that this method will be called for **all** the experiments
so it is best not to hardcode the name of algorithms when creating variables or 
filenames to avoid confusion. See the :class:`EvaluationManager <brisk.evaluation.evaluation_manager.EvaluationManager>`
section for more information on the evaluation methods available.

As a final note you'll notice that workflows are given their own ``workflows`` 
directory. This allows you to have multiple workflows in the same project. In the
next section you will see how to select a specific workflow to use. Each .py file
can only contain **one** Workflow subclass. This is to avoid using the wrong workflow at 
runtime.
