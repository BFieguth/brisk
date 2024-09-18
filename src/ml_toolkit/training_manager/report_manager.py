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

        # Step 1: List all files in the configuration directory
        file_metadata_list = []
        for file_name in os.listdir(config_dir):
            file_path = os.path.join(config_dir, file_name)
            file_metadata = {
                "file_name": file_name,
                "method": None
            }

            # Step 2: Check for metadata in JSON and PNG files
            if file_name.endswith(".json"):
                file_metadata["method"] = self.__get_json_metadata(file_path)
            elif file_name.endswith(".png"):
                file_metadata["method"] = self.__get_image_metadata(file_path)

            file_metadata_list.append(file_metadata)

        # Render the configuration page
        config_output = config_template.render(
            config_name=config_name,
            file_metadata_list=file_metadata_list
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
                return data.get("_metadata", {}).get("method", None)
        except Exception as e:
            print(f"Failed to extract metadata from {json_path}: {e}")
            return None

    def __get_image_metadata(self, image_path: str):
        """
        Extract the 'method' from the metadata of a PNG file.
        """
        try:
            with Image.open(image_path) as img:
                metadata = img.info
                return metadata.get("method", None)
        except Exception as e:
            print(f"Failed to extract metadata from {image_path}: {e}")
            return None
        