# Brisk
## v0.0.1

A framework that helps train machine learning models using sklearn. 

The package aims to speed up the training process by providing built in methods for analysis. It also provides a strucutre to help keep code organized. The package automatically tracks the settings used during training to help with repeatability. Users can define their own methods for algorithms, metrics or evaluation and integrate them with the package methods easily.

## Instructions

1. Create a new project
- brisk create -n project_name
- Create directory with configuration files
- cd ./project_name

2. settings.py
- in TRAINING_MANAGER_CONFIG define the algorithms to use
- Define the paths to the data files to use
- In WORKFLOW_CONFIG add settings for model training

3. metrics.py
- Define the metrics to use
- Can load defaults from brisk.REGRESSION_METRICS or brisk.CLASSIFICATION_METRICS
- Can add custom metrics by defining a function that takes y_true and y_pred
    - Create a MetricWrapper with this function

4. data.py
- Define how data should be split/processed

5. algorithms.py
- Define the default parameters and hyperparameter space to use.
- Can load defaults using brisk.CLASSIFICATION_ALGORITHMS or brisk.REGRESSION_ALGORITHMS

6. Create a Workflow
- In workflows/ directory define a Workflow class
- you can access built in evaluation methods 
- You can define your own methods in the workflow class

7. Run the Workflow
- navigate to the project root (directory created by brisk create)
- brisk run -w workflow_file
- This will run the workflow defined by the workflow class in this file

8. Analyze Results
- results are stored in the results directory
- When complete an HTML report is generated 
- Analysis of the data splits is run automatically
- A log file with the settings used is created
- All the built in evaluation methods export their results to a file by default
