.. _run_experiments:

===============
Training Models
===============

The run Command
===============

To run the experiments, we can use the ``brisk run`` command. This will run the 
experiments defined in the ``settings.py`` file using the specified workflow.
It is best to call the run command from the root of the project to ensure Brisk
can find the files.

The ``-w`` argument is used to specify the workflow file to use. This is the name 
of the file in the ``workflows`` directory containing the workflow class, without 
the .py extension. You can give the results directory a name using the ``-n`` 
argument.

To run the experiments we just configured you would call:

.. code-block:: bash

    brisk run -n tutorial_results

This command runs the experiments defined in your ``settings.py`` file using the workflows 
you specified in the configuration. When you call this you will see a progress bar appear in the terminal. When all 
the experiments are complete, you will see a summary of the experiments run, with
a status (PASSED or FAILED) and the time taken for each experiment.

You should see a summary table that looks like this:

.. code-block::

    ======================================================================
    EXPERIMENT SUMMARY
    ======================================================================

    Group: tutorial
    ======================================================================

    Dataset: diabetes
    Experiment                                         Status     Time      
    ----------------------------------------------------------------------
    tutorial_linear                                    PASSED     0m 1s     
    tutorial_ridge                                     PASSED     0m 0s     
    tutorial_elasticnet                                PASSED     0m 0s     
    ======================================================================

Congratulations! You have just trained your first models with Brisk.

.. note::
   In this tutorial, all experiment groups used the same workflow. You can also assign 
   different workflows to different experiment groups by specifying the ``workflow`` 
   parameter in ``add_experiment_group()``. This allows you to test different modeling 
   approaches on the same data. See the :doc:`Using ExperimentGroups </users/using_experiment_groups>` 
   guide for examples.


HTML Report
===========

Whenever you train models with Brisk, all the results are saved in the ``results/``
directory of your project (Brisk will create this directory if it doesn't exist). 
Here you should find a directory with a timestamp of when the experiments were run.

It should look like this:

.. code-block::
   :caption: Results Directory Structure

   tutorial/
   └── results/
       └── tutorial_results/
            ├── html_report/
            │   └── index.html
            │   └── ...
            ├── tutorial/
            │   └── ...
            └── config_log.md

``config_log.md`` is a markdown file that contains the configuration used for the 
experiments. It contains all of the information needed to reproduce the experiments
later. Even if you change the configuration files you can always reference this file
if you need to.

Each experiment group will have its own subdirectory, with the name of the group. 
There will be a subdirectory with analysis of the data split used for the group and
a subdirectory for each experiment. This is where you can find the outputs of the 
methods you call in the workflow.

Finally there is an ``html_report`` directory. You can drag the ``index.html`` file
into your browser to view the report. Any of the evaluation methods provided by
Brisk you use in the workflow will be included in the report. This gives you a 
quick overview of the performance of the models you trained. You can also look at
the distribution of features in the train test split.

If you want to learn more about Brisk, you can read the rest of the :ref:`getting_started`
section. The :ref:`user_guide` section contains more detailed in depth information
on specific features of Brisk.
