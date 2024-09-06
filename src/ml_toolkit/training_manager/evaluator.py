# TODO
# 1. Calculate scoring metrics
# 2. Calculate scoring metrics w/ cross validation
# 3. Plot Actual vs Predicted values
# 4. Plot learning curve
# 5. Plot feature importance
# 6. Plot Residuals
# 7. Perform Hyperparameter Optimization
# 8. Save model
# 9. Load model
# 10. Compare Models (on specified metric w/ same data)
# 11. Plot model comparison (plot the metric for each model)
# 12. Plot hyperparameter performance

class Evaluator:
    def __init__(self, method_config, scoring_config):
        self.method_config = method_config
        self.scoring_config = scoring_config

    def evaluate_model(self, model, X, y, metrics):
        """
        Calculate and print the scores for the given model and metrics.

        Args:
            model: The trained machine learning model to evaluate.
            X: The feature data to use for evaluation.
            y: The target data to use for evaluation.
            metrics (list): A list of metric names to calculate.
        """
        predictions = model.predict(X)
        results = {}

        for metric_name in metrics:
            scorer = self.scoring_config.get_scorer(metric_name)
            if scorer is not None:
                score = scorer(y, predictions)
                results[metric_name] = score
                print(f"{metric_name}: {score}")
            else:
                print(f"Scorer for {metric_name} not found.")
