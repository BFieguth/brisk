"""IO related utilities."""

from pathlib import Path
from typing import Optional, Any, Dict, Union, TYPE_CHECKING
import json
import os
import io
import sys
import importlib
import ast
import inspect

import matplotlib.pyplot as plt
import plotnine as pn
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import sqlite3

from brisk.services import base
from brisk.data import data_manager
from brisk.configuration import algorithm_collection
# from brisk.training import workflow as workflow_module

if TYPE_CHECKING:
    from brisk.training import workflow as workflow_module
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return list(obj)
        return super(NumpyEncoder, self).default(obj)


class IOService(base.BaseService):
    """IO service for saving and loading files.
    
    Parameters
    ----------
    name : str
        The name of the service

    results_dir : Path
        The root directory for all results, does not change at runtime.
        
    output_dir : Path
        The current output directory, will be changed at runtime.

    Attributes
    ----------
    results_dir : Path
        The root directory for all results, does not change at runtime.
    output_dir : Path
        The current output directory, will be changed at runtime.
    """
    def __init__(self, name: str, results_dir: Path, output_dir: Path):
        super().__init__(name)
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.format = "png"
        self.width = 10
        self.height = 8
        self.dpi = 300
        self.transparent = False

    def set_output_dir(self, output_dir: Path) -> None:
        """Set the current output directory.

        Parameters
        ----------
        output_dir : Path
            The new output directory

        Returns
        -------
        None
        """
        self.output_dir = output_dir

    def save_to_json(
        self,
        data: Dict[str, Any],
        output_path: Union[Path, str],
        metadata: Dict[str, Any]
    ) -> None:
        """Save dictionary to JSON file with metadata.

        Parameters
        ----------
        data : dict
            Data to save

        output_path : str
            The path to the output file.

        metadata : dict
            Metadata to include, by default None

        Returns
        -------
        None
        """
        if not os.path.exists(output_path.parent):
            os.makedirs(output_path.parent, exist_ok=True)
        try:
            if metadata:
                data["_metadata"] = metadata

            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=4, cls=NumpyEncoder)

            self._other_services["reporting"].store_table_data(
                data, metadata
            )

        except IOError as e:
            self._other_services["logging"].logger.info(
                f"Failed to save JSON to {output_path}: {e}"
            )

    def save_plot(
        self,
        output_path: Path,
        metadata: Optional[Dict[str, Any]] = None,
        plot: Optional[pn.ggplot | go.Figure] = None,
        **kwargs
    ) -> None:
        """Save current plot to file with metadata.

        Parameters
        ----------
        output_path (Path): 
            The path to the output file.

        metadata (dict, optional): 
            Metadata to include, by default None

        plot (ggplot, optional): 
            Plotnine plot object, by default None

        Returns
        -------
        None
        """
        if not os.path.exists(output_path.parent):
            os.makedirs(output_path.parent, exist_ok=True)

        height = kwargs.get("height", self.height)
        width = kwargs.get("width", self.width)
        output_path = output_path.with_suffix(f".{self.format}")
        self._convert_to_svg(metadata, plot, height, width)

        try:
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, dict):
                        metadata[key] = json.dumps(value)
            if plot and isinstance(plot, pn.ggplot):
                plot.save(
                    filename=output_path, format=self.format,
                    height=height, width=width, dpi=self.dpi,
                    transparent=self.transparent
                )
            elif plot and isinstance(plot, go.Figure):
                plot.write_image(
                    file=output_path, format=self.format
                )
            else:
                plt.savefig(
                    output_path, format=self.format,
                    dpi=self.dpi, transparent=self.transparent
                )
                plt.close()

        except IOError as e:
            self._other_services["logging"].logger.info(
                f"Failed to save plot to {output_path}: {e}"
            )

    def save_rerun_config(self, data: Dict, output_path: Union[Path, str]):
        if not os.path.exists(output_path.parent):
            os.makedirs(output_path.parent, exist_ok=True)
        try:
            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=4)

        except IOError as e:
            self._other_services["logging"].logger.info(
                f"Failed to save JSON to {output_path}: {e}"
            )

    def _convert_to_svg(
        self,
        metadata: Dict[str, Any],
        plot: Optional[pn.ggplot | go.Figure],
        height,
        width
    ) -> None:
        """Convert plot to SVG format for the report.

        Parameters
        ----------
        metadata : dict
            Metadata to include
        plot : ggplot
            Plotnine plot object
        height : int
            The plot height in inches
        width : int
            The plot width in inches

        Returns
        -------
        None
        """
        try:
            svg_buffer = io.BytesIO()
            if plot and isinstance(plot, pn.ggplot):
                plot.save(
                    svg_buffer, format="svg", height=height, width=width,
                    dpi=100
                )
            elif plot and isinstance(plot, go.Figure):
                plot.write_image(
                    file=svg_buffer, format="svg", width=width, height=height
                )
            else:
                plt.savefig(
                    svg_buffer, format="svg", bbox_inches="tight", dpi=100
                )

            svg_str = svg_buffer.getvalue().decode("utf-8")
            svg_buffer.close()
            self._other_services["reporting"].store_plot_svg(
                svg_str, metadata
            )

        except IOError as e:
            self._other_services["logging"].logger.info(
                f"Failed to convert plot to SVG: {e}"
            )

    def set_io_settings(self, io_settings: Dict[str, Any]) -> None:
        """Set settings to use when saving plots."""
        self.format = io_settings["format"]
        self.width = io_settings["width"]
        self.height = io_settings["height"]
        self.dpi = io_settings["dpi"]
        self.transparent = io_settings["transparent"]

    @staticmethod
    def load_data(
        data_path: str,
        table_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Loads data from a CSV, Excel file, or SQL database.

        Parameters
        ----------
        data_path : str
            Path to the dataset file
        table_name : str, optional
            Name of the table in SQL database. Required for SQL databases.

        Returns
        -------
        pd.DataFrame: The loaded dataset.

        Raises
        ------
        ValueError
            If file format is unsupported or table_name is missing for SQL
            database.
        """
        file_extension = os.path.splitext(data_path)[1].lower()

        if file_extension == ".csv":
            return pd.read_csv(data_path)

        elif file_extension in [".xls", ".xlsx"]:
            return pd.read_excel(data_path)

        elif file_extension in [".db", ".sqlite"]:
            if table_name is None:
                raise ValueError(
                    "For SQL databases, 'table_name' must be provided."
                )

            conn = sqlite3.connect(data_path)
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, conn)
            conn.close()
            return df

        else:
            raise ValueError(
                f"Unsupported file format: {file_extension}. "
                "Supported formats are CSV, Excel, and SQL database."
            )

    @staticmethod
    def load_module_object(
        project_root: str,
        module_filename: str,
        object_name: str,
        required: bool = True
    ) -> Union[object, None]:
        """
        Dynamically loads an object from a specified module file.

        Parameters
        ----------
        project_root : str
            Path to project root directory
        module_filename : str
            Name of the module file
        object_name : str
            Name of object to load
        required : bool, default=True
            Whether to raise error if object not found

        Returns
        -------
        object or None
            Loaded object or None if not found and not required

        Raises
        ------
        FileNotFoundError
            If module file not found
        AttributeError
            If required object not found in module
        """
        module_path = os.path.join(project_root, module_filename)

        if not os.path.exists(module_path):
            raise FileNotFoundError(
                f'{module_filename} not found in {project_root}'
            )

        module_name = os.path.splitext(module_filename)[0]
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        spec.loader.exec_module(module)

        if hasattr(module, object_name):
            return getattr(module, object_name)
        elif required:
            raise AttributeError(
                f'The object \'{object_name}\' is not defined in {module_filename}'
            )
        else:
            return None

    def load_custom_evaluators(self, evaluators_file: Path):
        """Load the register_custom_evaluators() function from evaluators.py
        """
        # NOTE This is where rerun service would hook in when in Coordinating mode
        try:
            spec = importlib.util.spec_from_file_location(
                "custom_evaluators", evaluators_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, "register_custom_evaluators"):
                self.get_service("logging").logger.info(
                    "Custom evaluators loaded succesfully"
                )
                return module
            else:
                self.get_service("logging").logger.warning(
                    "No register_custom_evaluators function found in evaluators.py"
                )
                return None

        except (ImportError, AttributeError) as e:
            self.get_service("logging").logger.warning(
                f"Failed to load custom evaluators: {e}"
            )

    def load_base_data_manager(self, data_file: Path):
        if not data_file.exists():
            raise FileNotFoundError(
                f"Data file not found: {data_file}\n"
                f"Please create data.py with BASE_DATA_MANAGER configuration"
            )

        spec = importlib.util.spec_from_file_location("data", data_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load data module from {data_file}")

        data_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(data_module)

        if not hasattr(data_module, "BASE_DATA_MANAGER"):
            raise ImportError(
                f"BASE_DATA_MANAGER not found in {data_file}\n"
                f"Please define BASE_DATA_MANAGER = DataManager(...)"
            )
        if not isinstance(
            data_module.BASE_DATA_MANAGER, data_manager.DataManager
        ):
            raise ValueError(
                f"BASE_DATA_MANAGER in {data_file} is not a valid "
                "DataManager instance"
            )
        self._validate_single_variable(data_file, "BASE_DATA_MANAGER")
        return data_module.BASE_DATA_MANAGER

    def load_algorithms(self, algorithm_file: Path):
        if not algorithm_file.exists():
            raise FileNotFoundError(
                f"algorithms.py file not found: {algorithm_file}\n"
                f"Please create algorithms.py and define an AlgorithmCollection"
            )

        spec = importlib.util.spec_from_file_location("algorithms", algorithm_file)
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Failed to load algorithms module from {algorithm_file}"
                )

        algo_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(algo_module)

        if not hasattr(algo_module, "ALGORITHM_CONFIG"):
            raise ImportError(
                f"ALGORITHM_CONFIG not found in {algorithm_file}\n"
                f"Please define ALGORITHM_CONFIG = AlgorithmCollection()"
            )
        self._validate_single_variable(algorithm_file, "ALGORITHM_CONFIG")
        if not isinstance(
            algo_module.ALGORITHM_CONFIG, algorithm_collection.AlgorithmCollection
        ):
            raise ValueError(
                f"ALGORITHM_CONFIG in {algorithm_file} is not a valid "
                "AlgorithmCollection instance"
            )
        return algo_module.ALGORITHM_CONFIG

    def load_workflow(self, workflow_name: str):
        def _is_workflow_subclass(obj) -> bool:
            """Check if an object is a subclass of Workflow without importing workflow module."""
            try:
                import brisk.training.workflow as workflow_module
                return issubclass(obj, workflow_module.Workflow)
            except (ImportError, TypeError):
                return False


        def _get_workflow_base_class():
            """Get the Workflow base class without importing at module level."""
            try:
                import brisk.training.workflow as workflow_module
                return workflow_module.Workflow
            except ImportError:
                return None


        try:
            module = importlib.import_module(
                f"workflows.{workflow_name}"
            )
            workflow_classes = [
                obj for _, obj in inspect.getmembers(module)
                if inspect.isclass(obj)
                and _is_workflow_subclass(obj)
                and obj is not _get_workflow_base_class()
            ]

            if len(workflow_classes) == 0:
                raise AttributeError(
                    f"No Workflow subclass found in {workflow_name}.py"
                )
            elif len(workflow_classes) > 1:
                raise AttributeError(
                    f"Multiple Workflow subclasses found in {workflow_name}.py."
                    " There can only be one Workflow per file."
                    )

            return workflow_classes[0]

        except (ImportError, AttributeError) as e:
            print(f"Error validating workflow: {e}")

    def _validate_single_variable(
        self,
        file_path: Path,
        variable_name: str
    ) -> None:
        """Validate that only a variable name is defined only once in a file.

        Parameters
        ----------
        file_path : Path
            Path to the Python file to check
        variable_name : str
            Name of the variable to check

        Raises
        ------
        ValueError
            If the variable is defined multiple times
        SyntaxError
            If the file contains invalid Python syntax
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source_code = f.read()

            tree = ast.parse(source_code, filename=str(file_path))

            assignments = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if (
                            isinstance(target, ast.Name)
                            and target.id == variable_name
                        ):
                            assignments.append(node.lineno)

            if len(assignments) > 1:
                lines_str = ", ".join(map(str, assignments))
                raise ValueError(
                    f"{variable_name} is defined multiple times in {file_path} "
                    f"on lines: {lines_str}. Please define it exactly once to "
                    "avoid ambiguity."
                )
        except SyntaxError as e:
            raise SyntaxError(f"Invalid Python syntax in {file_path}") from e
