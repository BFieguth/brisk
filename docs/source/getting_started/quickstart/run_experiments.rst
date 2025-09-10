.. _run_experiments:

===============
Training Models
===============

The run Command
===============

To run the experiments, we can use the ``brisk run`` command. This will run the 
experiments defined in the ``settings.py`` file. It is best to call the run command
from the root of the project to ensure Brisk can find all the necessary files.

You can give the results directory a name using the ``-n`` argument. If not specified,
a timestamp with the format ``DD_MM_YYYY_HH_MM_SS`` will be used.

To run the experiments we just configured you would call:

.. code-block:: bash

    brisk run -n tutorial_results

When you call this you will see a progress bar appear in the terminal. Brisk will
also provide logging messages to update you on the current model being trained. When all 
the experiments are complete, you will see a summary of the experiments run, with
a status (PASSED or FAILED) and the time taken for each experiment.

You should see a summary table that looks like this:

.. code-block::

    ======================================================================
    EXPERIMENT SUMMARY
    ======================================================================

    Group: tutorial
    ======================================================================

    Dataset: diabetes.csv
    Experiment                                         Status     Time      
    ----------------------------------------------------------------------
    tutorial_linear                                    PASSED     0m 1s     
    tutorial_linear                                    PASSED     0m 0s     
    tutorial_linear                                    PASSED     0m 0s     
    tutorial_linear                                    PASSED     0m 0s     
    tutorial_linear                                    PASSED     0m 0s     
    ======================================================================

Congratulations! You have just trained your first models with Brisk.


Interactive Report
==================

Whenever you train models with Brisk, all the results are saved in the ``results/``
directory of your project (Brisk will create this directory if it doesn't exist). 
Here you should find a directory called ``tutorial_results`` of when the experiments were run.

It should look like this:

.. code-block::
   :caption: Results Directory Structure

   tutorial/
   └── results/
       └── tutorial_results/
            ├── tutorial/
            │   └── ...
            ├── report.html
            └── run_config.json

Each experiment group will have its own subdirectory, with the name of the group. 
Then the results are organized by dataset and split. For each data split there will
be a subdirectory with some analysis of the distribution of the training and testing sets.
Within each data split directory there will be a subdirectory for each algorithm.
This is where you can find the outputs of the methods you call in the workflow.

Finally there is an ``report.html`` file. You can drag the ``report.html`` file
into your browser to view the report. Any of the evaluation methods provided by
Brisk you use in the workflow will be included in the report. This gives you a 
quick overview of the performance of the models you trained. You can also look at
the distribution of features in the train test split.

.. note::
    You can also view results from custom methods in the report. Follow the
    :ref:`custom_evaluators` guide to learn how to create custom evaluators and
    integrate them with Brisk.

If you want to learn more about Brisk, you can read the rest of the :ref:`getting_started`
section. The :ref:`user_guide` section contains more detailed in depth information
on specific features of Brisk.
