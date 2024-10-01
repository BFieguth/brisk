"""Provides the ReportManager class to generate HTML reports from evaluation results.

Exports:
    - ReportManager: A class that takes the outputs from the EvaluationManager 
        and creates an HTML report with all results.
"""

import ast
import collections
import datetime
import json
import os
from PIL import Image
import shutil

import jinja2
import pandas as pd

class ReportManager():
    """A class for creating HTML reports from evaluation results.

    The ReportManager processes the results from various model evaluations and 
    generates a structured HTML report containing all relevant metrics, 
    comparisons, and visualizations.

    Attributes:
        report_dir (str): Directory where the HTML report will be saved.
        experiment_paths (dict): Dictionary mapping datasets to their 
            corresponding experiment directories.
        env (jinja2.Environment): Jinja2 environment for rendering HTML templates.
        method_map (OrderedDict): Mapping of evaluation methods to reporting methods.
        current_dataset (str, optional): The name of the dataset currently being processed.
        summary_metrics (dict): Dictionary holding summary metrics for the report.
    """
    def __init__(self, result_dir: str, experiment_paths: dict):
        """Initializes the ReportManager with directories for results and experiment paths.

        Args:
            result_dir (str): Path to the directory where the report will be saved.
            experiment_paths (dict): Dictionary mapping datasets to their 
                experiment directories.
        """
        package_dir = os.path.dirname(os.path.abspath(__file__))
        self.templates_dir = os.path.join(package_dir, 'templates')
        self.styles_dir = os.path.join(package_dir, 'styles')

        self.report_dir = os.path.join(result_dir, "html_report")
        self.experiment_paths = experiment_paths
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.templates_dir),
            autoescape=jinja2.select_autoescape(["html", "xml"])
        )
        self.method_map = collections.OrderedDict([
            ("evaluate_model", self.report_evaluate_model),
            ("evaluate_model_cv", self.report_evaluate_model_cv),
            ("compare_models", self.report_compare_models),
            ("plot_pred_vs_obs", lambda data, metadata: self.report_image(
                data, metadata, title="Predicted vs Observed Plot"
                )),
            ("plot_learning_curve", lambda data, metadata: self.report_image(
                data, metadata, title="Learning Curve", max_width="80%"
                )),
            ("plot_feature_importance", lambda data, metadata: self.report_image(
                data, metadata, title="Feature Importance"
            )),
            ("plot_residuals", lambda data, metadata: self.report_image(
                data, metadata, title="Residuals (Observed - Predicted)"
            )),
            ("plot_model_comparison", lambda data, metadata: self.report_image(
                data, metadata, title="Model Comparison Plot"
            )),
            ("hyperparameter_tuning", lambda data, metadata: self.report_image(
                data, metadata, title="Hyperparameter Tuning"
            ))
        ])
        self.current_dataset = None
        self.summary_metrics = {}

    def create_report(self) -> None:
        """Generates the HTML report.
        
        This method creates an index page listing all datasets and their 
        respective experiment pages, as well as a summary table if evaluation 
        metrics are available.
        
        Returns:
            None
        """
        os.makedirs(self.report_dir, exist_ok=True)

        # Step 1: Create the index page with links to experiments pages for each dataset
        index_template = self.env.get_template("index.html")

        # Prepare the dataset entries for the index page
        datasets = []
        for dataset, experiments in self.experiment_paths.items():
            datasets.append({
                'dataset_name': os.path.basename(dataset),
                'experiment_pages': [os.path.basename(experiment) for experiment in experiments]
            })

            # Create individual experiments pages
            for experiment in experiments:
                self.create_experiment_page(experiment, dataset)

        # Create summary table
        summary_table_html = None
        if self.summary_metrics:
            summary_table_html = self.generate_summary_tables()

        # Copy CSS files into the report directory
        shutil.copy(
            os.path.join(self.styles_dir, "index.css"), 
            os.path.join(self.report_dir, "index.css")
            )
        shutil.copy(
            os.path.join(self.styles_dir, "experiment.css"), 
            os.path.join(self.report_dir, "experiment.css")
            )

        # Render the index page
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        index_output = index_template.render(
            datasets=datasets, timestamp=timestamp, summary_table=summary_table_html
            )
        index_path = os.path.join(self.report_dir, 'index.html')
        with open(index_path, 'w') as f:
            f.write(index_output)

    def create_experiment_page(self, experiment_dir: str, dataset: str) -> None:
        """Creates an HTML page for each experiment.

        Args:
            experiment_dir (str): Path to the experiment directory.
            dataset (str): The name of the dataset being evaluated.

        Returns:
            None
        """
        self.current_dataset = dataset
        experiment_template = self.env.get_template("experiment.html")
        experiment_name = os.path.basename(experiment_dir)
        files = os.listdir(experiment_dir)

        # Step 1: Process file metadata
        file_metadata = {}
        for file in files:
            file_path = os.path.join(experiment_dir, file)
            if file.endswith(".json"):
                file_metadata[file] = self._get_json_metadata(file_path)
            if file.endswith(".png"):
                file_metadata[file] = self._get_image_metadata(file_path)

        # Step 2: Prepare content based on extracted metadata
        content = []
        
        for creating_method, reporting_method in self.method_map.items():
            matching_files = [
                (file, metadata) for file, metadata in file_metadata.items()
                if metadata["method"] == creating_method
            ]
        
            for file, metadata in matching_files:
                file_path = os.path.join(experiment_dir, file)
                data = self._load_file(file_path)
                content.append(reporting_method(data, metadata))

        content_str = "".join(content)

        # Step 3: Render the experiments page
        experiment_output = experiment_template.render(
            experiment_name=experiment_name,
            file_metadata=file_metadata,
            content=content_str
            )
        experiment_page_path = os.path.join(self.report_dir, f"{experiment_name}.html")
        with open(experiment_page_path, 'w') as f:
            f.write(experiment_output)

    def _get_json_metadata(self, json_path: str) -> dict:
        """Extracts metadata from a JSON file.

        Args:
            json_path (str): Path to the JSON file.

        Returns:
            dict: The extracted metadata.
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                metadata = data.get("_metadata", {})
                if "models" in metadata and isinstance(metadata["models"], str):
                    metadata["models"] = ast.literal_eval(metadata["models"])
                return metadata
        except Exception as e:
            print(f"Failed to extract metadata from {json_path}: {e}")
            return "unknown_method"

    def _get_image_metadata(self, image_path: str) -> dict:
        """Extracts metadata from a PNG file.

        Args:
            image_path (str): Path to the image file.

        Returns:
            dict: The extracted metadata.
        """
        try:
            with Image.open(image_path) as img:
                metadata = img.info
                if "models" in metadata and isinstance(metadata["models"], str):
                    metadata["models"] = ast.literal_eval(metadata["models"])
                return metadata
        except Exception as e:
            print(f"Failed to extract metadata from {image_path}: {e}")
            return None
        
    def _load_file(self, file_path: str):
        """Loads the content of a file based on its extension.

        Args:
            file_path (str): Path to the file.

        Returns:
            dict: The loaded content for JSON files, or the file path for images.

        Raises:
            ValueError: If the file type is unsupported.
        """
        if file_path.endswith(".json"):
            with open(file_path, "r") as f:
                return json.load(f)
        elif file_path.endswith(".png"):
            return file_path
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    def report_evaluate_model(self, data: dict, metadata: dict) -> str:
        """Generates an HTML block for displaying evaluate_model results.

        Args:
            data (dict): The evaluation data.
            metadata (dict): The metadata associated with the evaluation.

        Returns:
            str: HTML block representing the evaluation results.
        """
        # Extract relevant information
        metrics = {k: v for k, v in data.items() if k != "_metadata"}
        model_info = metadata.get("models", ["Unknown model"])
        model_names = ", ".join(model_info)

        # Create an HTML block for this result
        result_html = f"""
        <h2>Model Evaluation</h2>
        <p><strong>Model:</strong> {model_names}</p>
        <table>
            <thead>
                <tr><th>Metric</th><th>Score</th></tr>
            </thead>
            <tbody>
        """
        for metric, score in metrics.items():
            rounded_score = round(score, 5)
            result_html += f"<tr><td>{metric}</td><td>{rounded_score}</td></tr>"
        result_html += "</tbody></table>"

        return result_html

    def report_evaluate_model_cv(self, data: dict, metadata: dict) -> str:
        """Generates an HTML block for displaying cross-validated evaluation results.

        Args:
            data (dict): The evaluation data containing metric information.
            metadata (dict): The metadata associated with the evaluation.

        Returns:
            str: HTML block representing the cross-validation results.
        """
        model_info = metadata.get("models", ["Unknown model"])
        model_names = ", ".join(model_info)

        result_html_new = f"""
        <h2>Model Evaluation (Cross-Validation)</h2>
        <p><strong>Model:</strong> {model_names}</p>
        <table>
            <thead>
                <tr><th>Metric</th><th>All Scores</th><th>Mean Score</th><th>Std Dev</th></tr>
            </thead>
            <tbody>
        """

        if self.current_dataset not in self.summary_metrics:
            self.summary_metrics[self.current_dataset] = {}
        
        if model_names not in self.summary_metrics[self.current_dataset]:
            self.summary_metrics[self.current_dataset][model_names] = {}

        for metric, values in data.items():
            if metric != "_metadata":
                all_scores = ", ".join(f"{score:.5f}" for score in values["all_scores"])
                mean_score = round(values["mean_score"], 5)
                std_dev = round(values["std_dev"], 5)
                result_html_new += f"<tr><td>{metric}</td><td>{all_scores}</td><td>{mean_score}</td><td>{std_dev}</td></tr>"

                self.summary_metrics[self.current_dataset][model_names][metric] = {
                    "mean": mean_score,
                    "std_dev": std_dev
                }

        result_html_new += "</tbody></table>"
        return result_html_new
     
    def report_compare_models(self, data: dict, metadata: dict) -> str:
        """Generates an HTML block for displaying model comparison results.

        Args:
            data (dict): The comparison results.
            metadata (dict): The metadata containing model information.

        Returns:
            str: HTML block representing the comparison results.
        """
        model_names = metadata.get("models", [])
        
        if not model_names:
            raise ValueError("No model names found in metadata.")
        
        metrics = list(next(iter(data.values())).keys())

        metric_data = {
            model_name: {metric: data[model_name][metric] for metric in metrics} 
            for model_name in model_names if 'differences' not in model_name
            }

        df = pd.DataFrame(metric_data)

        if "differences" in data:
            diff_data = {
                metric: {
                    pair: data["differences"][metric].get(pair, None) 
                    for pair in data["differences"][metric]
                    }
                for metric in data["differences"]
            }
            diff_df = pd.DataFrame(diff_data).T
            df = pd.concat([df, diff_df], axis=1)

        # Generate the HTML table
        model_name = ", ".join(metadata.get("models", ["Unknown model"]))
        html_table = df.to_html(classes='table table-bordered', border=0)
        result_html = f"""
        <h2>Model Comparison</h2>
        <p><strong>Model:</strong> {model_name}</p>
        """
        result_html += html_table

        return result_html

    def report_image(
        self, 
        data: str, 
        metadata: dict, 
        title: str, 
        max_width: str = "100%"
    ) -> str:
        """Generates an HTML block for displaying an image plot.

        Args:
            data (str): The path to the image file.
            metadata (dict): The metadata associated with the plot.
            title (str): The title to display above the image.
            max_width (str): The maximum width of the image. Defaults to "100%".

        Returns:
            str: HTML block containing the image and its metadata.
        """   
        model_name = ", ".join(metadata.get("models", ["Unknown model"]))
        relative_img_path = os.path.relpath(data, self.report_dir)

        result_html = f"""
        <h2>{title}</h2>
        <p><strong>Model:</strong> {model_name}</p>
        <img src="{relative_img_path}" alt="{title}" style="max-width:{max_width}; height:auto; display: block; margin: 0 auto;">
        """
        return result_html
    
    def generate_summary_tables(self) -> str:
        """Generates HTML summary tables for each dataset, displaying model metrics.

        Returns:
            str: HTML block containing summary tables for all datasets.
        """
        summary_html = ""

        for dataset, models in self.summary_metrics.items():
            summary_html += f"<h2>Summary for {dataset}</h2>"
            
            # Extract all the metrics to be used as columns
            all_metrics = set()
            for model_metrics in models.values():
                all_metrics.update(model_metrics.keys())
            
            summary_html += """
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
            """
            
            # Add a column for each metric
            for metric in all_metrics:
                summary_html += f"<th>{metric}</th>"
            
            summary_html += "</tr></thead><tbody>"

            # Add rows for each model
            for model, metrics in models.items():
                summary_html += f"<tr><td>{model}</td>"
                for metric in all_metrics:
                    if metric in metrics:
                        mean_score = round(metrics[metric]["mean"], 3)
                        std_dev = round(metrics[metric]["std_dev"], 3)
                        summary_html += f"<td>{mean_score} ({std_dev})</td>"
                    else:
                        summary_html += "<td>N/A</td>"
                summary_html += "</tr>"

            summary_html += "</tbody></table>"

        return summary_html
