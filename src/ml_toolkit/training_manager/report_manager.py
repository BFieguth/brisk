import ast
import collections
import json
import os
from PIL import Image

import jinja2

class ReportManager():
    def __init__(self, result_dir, config_paths):
        package_dir = os.path.dirname(os.path.abspath(__file__))
        templates_dir = os.path.join(package_dir, '../templates')

        self.report_dir = os.path.join(result_dir, "html_report")
        self.config_paths = config_paths
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(templates_dir),
            autoescape=jinja2.select_autoescape(["html", "xml"])
        )
        self.method_map = collections.OrderedDict([
            ("evaluate_model", self.report_evaluate_model),
            ("evaluate_model_cv", self.report_evaluate_model_cv),
            ("compare_models", self.report_compare_models),
            # ("plot_pred_vs_obs", self.report_plot_pred_vs_obs),
            # ("plot_learning_curve", self.report_learning_curve),
            # ("plot_feature_importance", self.report_plot_feature_importance),
            # ("plot_residuals", self.report_residuals),
            # ("plot_model_comparison", self.report_plot_model_comparison),
            # ("hyperparameter_tuning", self.report_hyperparameter_tuning)
        ])

    def create_report(self):
        os.makedirs(self.report_dir, exist_ok=True)

        # Step 1: Create the index page with links to configuration pages for each dataset
        index_template = self.env.get_template("index.html")

        # Prepare the dataset entries for the index page
        datasets = []
        for dataset, configs in self.config_paths.items():
            datasets.append({
                'dataset_name': os.path.basename(dataset),
                'config_pages': [os.path.basename(config) for config in configs]
            })

            # Create individual configuration pages
            for config in configs:
                self.create_config_page(config)

        # Render the index page
        index_output = index_template.render(datasets=datasets)
        index_path = os.path.join(self.report_dir, 'index.html')
        with open(index_path, 'w') as f:
            f.write(index_output)

    def create_config_page(self, config_dir: str):
        """
        Creates an individual configuration page.
        Args:
            config_dir (str): Path to the configuration directory.
        """
        config_template = self.env.get_template("config.html")
        config_name = os.path.basename(config_dir)
        files = os.listdir(config_dir)

        # Step 1: Process file metadata
        file_metadata = {}
        for file in files:
            file_path = os.path.join(config_dir, file)
            if file.endswith(".json"):
                file_metadata[file] = self.__get_json_metadata(file_path)
            if file.endswith(".png"):
                file_metadata[file] = self.__get_image_metadata(file_path)

        # Step 2: Prepare content based on extracted metadata
        content = []
        
        for creating_method, reporting_method in self.method_map.items():
            matching_files = [
                (file, metadata) for file, metadata in file_metadata.items()
                if metadata["method"] == creating_method
            ]
        
            for file, metadata in matching_files:
                file_path = os.path.join(config_dir, file)
                data = self.__load_file(file_path)
                content.append(reporting_method(data, metadata))

        content_str = "".join(content)

        # Step 3: Render the configuration page
        config_output = config_template.render(
            config_name=config_name,
            file_metadata=file_metadata,
            content=content_str
            )
        config_page_path = os.path.join(self.report_dir, f"{config_name}.html")
        with open(config_page_path, 'w') as f:
            f.write(config_output)

    def __get_json_metadata(self, json_path: str):
        """
        Extract the 'method' from the metadata of a JSON file.
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

    def __get_image_metadata(self, image_path: str):
        """
        Extract the 'method' from the metadata of a PNG file.
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
        
    def __load_file(self, file_path: str):
        """Loads file content based on the file extension."""
        if file_path.endswith(".json"):
            with open(file_path, "r") as f:
                return json.load(f)
        elif file_path.endswith(".png"):
            return Image.open(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    def report_evaluate_model(self, data: dict, metadata: dict) -> str:
        """
        Generates an HTML block for displaying evaluate_model results.
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
        """
        Generates an HTML block for displaying cross-validated evaluation results.
        Displays all_scores, mean_score, and std_dev for each metric.

        Args:
            data (dict): The data containing metric information.
            metadata (dict): The metadata associated with the evaluation.

        Returns:
            str: An HTML block representing the results.
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

        for metric, values in data.items():
            if metric != "_metadata":
                all_scores = ", ".join(f"{score:.5f}" for score in values["all_scores"])
                mean_score = round(values["mean_score"], 5)
                std_dev = round(values["std_dev"], 5)
                result_html_new += f"<tr><td>{metric}</td><td>{all_scores}</td><td>{mean_score}</td><td>{std_dev}</td></tr>"
        
        result_html_new += "</tbody></table>"
        return result_html_new
    
    def report_compare_models(self, data, metadata) -> str:
        """
        Generates an HTML block for displaying compare_models results.
        Args:
            data (dict): The comparison results.
            metadata (dict): The metadata containing model information.
        """
        model_names = list(data.keys())
        metrics = list(data[model_names[0]].keys())
        
        if isinstance(metadata["models"], list):
            model_names = [f"model_{i+1}" for i in range(len(metadata["models"]))]
            display_names = metadata["models"]
        else:
            display_names = model_names

        # Start the table HTML
        result_html = """
        <h2>Model Comparison</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
        """

        # Add model columns
        for model_name in display_names:
            result_html += f"<th>{model_name}</th>"

        # Add the difference column if present
        if "difference_from_model1" in data:
            result_html += "<th>Difference from Model 1</th>"

        result_html += "</tr></thead><tbody>"

        # Fill the table with data for each metric
        for metric in metrics:
            result_html += f"<tr><td>{metric}</td>"
            for model_name in model_names:
                score = round(data[model_name][metric], 5)
                result_html += f"<td>{score}</td>"
            
            if "difference_from_model1" in data and metric in data["difference_from_model1"]:
                diff = round(data["difference_from_model1"][metric].get(model_names[1], 0), 5)
                result_html += f"<td>{diff}</td>"
            
            result_html += "</tr>"

        result_html += "</tbody></table>"

        return result_html
    