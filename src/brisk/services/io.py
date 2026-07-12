"""I/O utilities and services for file operations and data management.

This module provides comprehensive I/O functionality for the Brisk package,
including file saving/loading, plot generation, data processing, and dynamic
module loading. It serves as the central hub for all file-based operations
in the machine learning pipeline.

The module includes specialized classes and utilities for handling various
data formats, plot types, and configuration files, with robust error handling
and metadata management.

Examples
--------
>>> from brisk.services.io import IOService, load_data
>>> from pathlib import Path
>>> 
>>> # Create I/O service
>>> io_service = IOService("io", Path("results"), Path("output"))
>>> 
>>> # Load data
>>> df = load_data("data.csv")
>>> 
>>> # Save data and plots
>>> data = {"accuracy": 0.95}
>>> io_service.save_to_json(data, Path("results.json"), {})
>>> io_service.save_plot(Path("plot.png"), plot=my_plot)
"""

import pathlib
from typing import Optional, Any, Dict, Union
import json
import os
import importlib
import importlib.util
import ast
import inspect

import plotnine as pn
import plotly.graph_objects as go
import pandas as pd

from brisk.adapters.filesystem.module_loader_adapter import ModuleLoaderAdapter
from brisk.adapters.filesystem.storage_adapter import (
    FilesystemStorageAdapter,
    NumpyEncoder,
)
from brisk.adapters.plotting.plot_renderer import PlotRenderer
from brisk.services import base
from brisk.data import data_manager
from brisk.configuration import algorithm_collection
from brisk.ports import metric

_module_loader = ModuleLoaderAdapter()
_default_storage = FilesystemStorageAdapter(
    pathlib.Path("."), pathlib.Path(".")
)


class IOService(base.BaseService):
    """I/O service for file operations, data loading, and plot management.
    
    This service provides comprehensive I/O functionality for the Brisk package,
    including saving/loading data files, generating and saving plots, dynamic
    module loading, and configuration management. It handles various file
    formats and provides robust error handling and metadata management.
    
    The service maintains separate directories for results (static) and output
    (dynamic), allowing for organized file management throughout experiments.
    
    Attributes
    ----------
    results_dir : Path
        The root directory for all results, does not change at runtime
    output_dir : Path
        The current output directory, will be changed at runtime
    format : str
        Default format for saving plots (default: "png")
    width : int
        Default plot width in inches (default: 10)
    height : int
        Default plot height in inches (default: 8)
    dpi : int
        Default plot DPI (default: 300)
    transparent : bool
        Whether to save plots with transparent background (default: False)
        
    Notes
    -----
    The service automatically creates output directories as needed and
    integrates with the reporting service to store plot data for reports.
    
    Examples
    --------
    >>> from brisk.services.io import IOService
    >>> from pathlib import Path
    >>> 
    >>> # Create I/O service
    >>> io_service = IOService("io", Path("results"), Path("output"))
    >>> 
    >>> # Save data
    >>> data = {"accuracy": 0.95, "precision": 0.92}
    >>> io_service.save_to_json(data, Path("results.json"), {})
    >>> 
    >>> # Save plot
    >>> io_service.save_plot(Path("plot.png"), plot=my_plot)
    >>> 
    >>> # Load data
    >>> df = io_service.load_data("data.csv")
    """
    def __init__(
        self,
        name: str,
        results_dir: pathlib.Path,
        output_dir: pathlib.Path
    ) -> None:
        """Initialize the I/O service with directories and default settings.
        
        This constructor sets up the I/O service with the specified directories
        and default plot settings. The service will use these settings for
        all subsequent file operations unless overridden.
        
        Parameters
        ----------
        name : str
            The name identifier for this service
        results_dir : Path
            The root directory for all results (static)
        output_dir : Path
            The current output directory (dynamic, can be changed)
            
        Notes
        -----
        The output directory can be changed at runtime using `set_output_dir()`.
        Default plot settings can be modified using `set_io_settings()`.
        """
        super().__init__(name)
        self.format = "png"
        self.width = 10
        self.height = 8
        self.dpi = 300
        self.transparent = False
        self._plot_renderer = PlotRenderer(
            file_format=self.format,
            width=self.width,
            height=self.height,
            dpi=self.dpi,
            transparent=self.transparent,
        )
        self._storage = FilesystemStorageAdapter(
            results_dir=results_dir,
            output_dir=output_dir,
            plot_renderer=self._plot_renderer,
        )
        self.results_dir = self._storage.results_dir
        self.output_dir = self._storage.output_dir

    def set_output_dir(self, output_dir: pathlib.Path) -> None:
        """Set the current output directory.

        This method updates the current output directory where files will be
        saved. This is typically called when starting a new experiment to
        organize outputs by experiment.

        Parameters
        ----------
        output_dir : pathlib.Path
            The new output directory path

        Examples
        --------
        >>> io_service = IOService("io", Path("results"), Path("output"))
        >>> io_service.set_output_dir(Path("experiment_1"))
        >>> # Now all saves will go to experiment_1 directory
        """
        self._storage.set_output_dir(output_dir)
        self.output_dir = self._storage.output_dir

    def save_to_json(
        self,
        data: Dict[str, Any],
        output_path: Union[pathlib.Path, str],
        metadata: Dict[str, Any]
    ) -> None:
        """Save dictionary to JSON file with metadata.

        This method saves a dictionary to a JSON file with optional metadata.
        It automatically creates parent directories if they don't exist and
        handles NumPy data types through the NumpyEncoder. The data is also
        stored in the reporting service for report generation.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing the data to save
        output_path : Union[Path, str]
            Path where the JSON file will be saved
        metadata : Dict[str, Any]
            Metadata to include with the data (stored as "_metadata" key)

        Notes
        -----
        The method automatically creates parent directories and handles
        NumPy data types. If saving fails, an error is logged but no
        exception is raised.

        Examples
        --------
        >>> io_service = IOService("io", Path("results"), Path("output"))
        >>> data = {"accuracy": 0.95, "precision": 0.92}
        >>> metadata = {"experiment": "exp_1", "timestamp": "2024-01-15"}
        >>> io_service.save_to_json(data, Path("results.json"), metadata)
        """
        try:
            if metadata:
                data["_metadata"] = metadata

            self._storage.save_to_json(data, output_path, metadata)

            filename = pathlib.Path(output_path).stem
            self._other_services["reporting"].store_table_data(
                data, metadata, filename
            )

        except IOError as e:
            self._other_services["logging"].logger.info(
                f"Failed to save JSON to {output_path}: {e}"
            )

    def save_plot(
        self,
        output_path: pathlib.Path,
        metadata: Optional[Dict[str, Any]] = None,
        plot: Optional[pn.ggplot | go.Figure] = None,
        **kwargs
    ) -> None:
        """Save plot to file with metadata and SVG conversion.

        This method saves a plot to a file in the specified format, with
        automatic SVG conversion for report generation. It supports multiple
        plot types including matplotlib, plotnine, and plotly figures.

        Parameters
        ----------
        output_path : Path
            Path where the plot file will be saved
        metadata : Optional[Dict[str, Any]], default=None
            Metadata to include with the plot
        plot : Optional[pn.ggplot | go.Figure], default=None
            Plot object to save (plotnine or plotly figure)
        **kwargs
            Additional plot parameters (height, width, etc.)

        Notes
        -----
        The method automatically converts plots to SVG format for reports
        and handles different plot types. If no plot is provided, it saves
        the current matplotlib figure.

        Examples
        --------
        >>> io_service = IOService("io", Path("results"), Path("output"))
        >>> # Save plotnine plot
        >>> io_service.save_plot(Path("plot.png"), plot=my_plotnine_plot)
        >>> 
        >>> # Save plotly plot
        >>> io_service.save_plot(Path("plot.png"), plot=my_plotly_figure)
        >>> 
        >>> # Save current matplotlib figure
        >>> plt.plot([1, 2, 3], [1, 4, 9])
        >>> io_service.save_plot(Path("plot.png"))
        """
        height = kwargs.get("height", self.height)
        width = kwargs.get("width", self.width)
        filename = output_path.stem
        save_path = output_path.with_suffix(f".{self.format}")

        try:
            svg_str = self._plot_renderer.to_svg(
                plot, height=height, width=width
            )
            self._other_services["reporting"].store_plot_svg(
                svg_str, metadata, filename
            )
        except IOError as e:
            self._other_services["logging"].logger.info(
                f"Failed to convert plot to SVG: {e}"
            )

        try:
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, dict):
                        metadata[key] = json.dumps(value)

            self._storage.save_plot(
                save_path, metadata, plot, height=height, width=width
            )

        except IOError as e:
            self._other_services["logging"].logger.info(
                f"Failed to save plot to {save_path}: {e}"
            )

    def save_rerun_config(
        self,
        data: Dict,
        metadata: Dict,
        output_path: Union[pathlib.Path, str]
    ):
        if metadata:
            data["_metadata"] = metadata
        if not os.path.exists(output_path.parent):
            os.makedirs(output_path.parent, exist_ok=True)
        try:
            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=4, cls=NumpyEncoder, allow_nan=False)

        except IOError as e:
            self._other_services["logging"].logger.info(
                f"Failed to save JSON to {output_path}: {e}"
            )

    def set_io_settings(self, io_settings: Dict[str, Any]) -> None:
        """Set settings to use when saving plots."""
        self.format = io_settings["file_format"]
        self.width = io_settings["width"]
        self.height = io_settings["height"]
        self.dpi = io_settings["dpi"]
        self.transparent = io_settings["transparent"]
        self._plot_renderer.configure(io_settings)

    @staticmethod
    def load_data(
        data_path: str,
        table_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Load data from CSV, Excel, or SQL database files.

        This static method loads data from various file formats into a pandas
        DataFrame. It automatically detects the file format based on the
        file extension and handles the appropriate loading method.

        Parameters
        ----------
        data_path : str
            Path to the dataset file
        table_name : Optional[str], default=None
            Name of the table in SQL database (required for SQL files)

        Returns
        -------
        pd.DataFrame
            The loaded dataset as a pandas DataFrame

        Raises
        ------
        ValueError
            If file format is unsupported or table_name is missing for SQL
            database

        Examples
        --------
        >>> from brisk.services.io import IOService
        >>> 
        >>> # Load CSV file
        >>> df = IOService.load_data("data.csv")
        >>> 
        >>> # Load Excel file
        >>> df = IOService.load_data("data.xlsx")
        >>> 
        >>> # Load SQL database
        >>> df = IOService.load_data("data.db", table_name="my_table")
        """
        return _default_storage.load_data(data_path, table_name)

    @staticmethod
    def load_module_object(
        project_root: str,
        module_filename: str,
        object_name: str,
        required: bool = True
    ) -> Union[object, None]:
        """Dynamically load an object from a specified module file.

        This static method loads a Python object from a module file at runtime.
        It's useful for loading configuration objects, custom evaluators, or
        other dynamic components from project files.

        Parameters
        ----------
        project_root : str
            Path to project root directory
        module_filename : str
            Name of the module file (e.g., "algorithms.py")
        object_name : str
            Name of the object to load from the module
        required : bool, default=True
            Whether to raise an error if the object is not found

        Returns
        -------
        Union[object, None]
            The loaded object, or None if not found and not required

        Raises
        ------
        FileNotFoundError
            If the module file is not found
        AttributeError
            If the required object is not found in the module

        Examples
        --------
        >>> from brisk.services.io import IOService
        >>> 
        >>> # Load a configuration object
        >>> config = IOService.load_module_object(
        ...     "/path/to/project", "algorithms.py", "ALGORITHM_CONFIG"
        ... )
        >>> 
        >>> # Load optional object (returns None if not found)
        >>> optional = IOService.load_module_object(
        ...     "/path/to/project", "optional.py", "OPTIONAL_OBJ",
        ...     required=False
        ... )
        """
        return _module_loader.load_module_object(
            project_root,
            module_filename,
            object_name,
            required=required,
        )

    def load_custom_evaluators(self, evaluators_file: pathlib.Path):
        """Load the register_custom_evaluators() function from evaluators.py
        """
        rerun = self.get_service("rerun")
        if rerun.is_coordinating:
            return rerun.handle_load_custom_evaluators(None, evaluators_file)
        try:
            loaded_module = None
            spec = importlib.util.spec_from_file_location(
                "custom_evaluators", evaluators_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, "register_custom_evaluators"):
                self.get_service("logging").logger.info(
                    "Custom evaluators loaded succesfully"
                )
                loaded_module = module
            else:
                self.get_service("logging").logger.warning(
                    "No register_custom_evaluators function found in "
                    "evaluators.py"
                )

            return rerun.handle_load_custom_evaluators(
                loaded_module, evaluators_file
            )

        except (ImportError, AttributeError) as e:
            self.get_service("logging").logger.warning(
                f"Failed to load custom evaluators: {e}"
            )
            return rerun.handle_load_custom_evaluators(None, evaluators_file)

    def load_base_data_manager(self, data_file: pathlib.Path):
        rerun = self.get_service("rerun")
        if rerun.is_coordinating:
            return rerun.handle_load_base_data_manager(None)

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
        return rerun.handle_load_base_data_manager(
            data_module.BASE_DATA_MANAGER
        )

    def load_algorithms(self, algorithm_file: pathlib.Path):
        rerun = self.get_service("rerun")
        if rerun.is_coordinating:
            return rerun.handle_load_algorithms(None)

        if not algorithm_file.exists():
            raise FileNotFoundError(
                f"algorithms.py file not found: {algorithm_file}\n"
                f"Please create algorithms.py and define an AlgorithmCollection"
            )

        spec = importlib.util.spec_from_file_location(
            "algorithms", algorithm_file
        )
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
            algo_module.ALGORITHM_CONFIG,
            algorithm_collection.AlgorithmCollection
        ):
            raise ValueError(
                f"ALGORITHM_CONFIG in {algorithm_file} is not a valid "
                "AlgorithmCollection instance"
            )
        return rerun.handle_load_algorithms(
            algo_module.ALGORITHM_CONFIG
        )

    def load_workflow(self, workflow_name: str):
        def _is_workflow_subclass(obj) -> bool:
            """
            Check if an object is a subclass of Workflow without importing
            workflow module.
            """
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


        rerun = self.get_service("rerun")
        if rerun.is_coordinating:
            return rerun.handle_load_workflow(None, workflow_name)

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

            return rerun.handle_load_workflow(
                workflow_classes[0], workflow_name
            )

        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to load workflow {workflow_name}") from e

    def _validate_single_variable(
        self,
        file_path: pathlib.Path,
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

    def load_metric_config(self, metric_file):
        rerun = self.get_service("rerun")
        if rerun.is_coordinating:
            return rerun.handle_load_metric_config(None)

        if not metric_file.exists():
            raise FileNotFoundError(
                f"metrics.py file not found: {metric_file}\n"
                f"Please create metric.py and define a MetricManager"
            )

        spec = importlib.util.spec_from_file_location("metrics", metric_file)
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Failed to load metrics module from {metric_file}"
                )

        metrics_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(metrics_module)

        if not hasattr(metrics_module, "METRIC_CONFIG"):
            raise ImportError(
                f"METRIC_CONFIG not found in {metric_file}\n"
                f"Please define METRIC_CONFIG = MetricManager()"
            )
        self._validate_single_variable(metric_file, "METRIC_CONFIG")
        if not isinstance(
            metrics_module.METRIC_CONFIG, metric.MetricManagerPort
        ):
            raise ValueError(
                f"METRIC_CONFIG in {metric_file} is not a valid "
                "MetricManager instance"
            )
        return rerun.handle_load_metric_config(
            metrics_module.METRIC_CONFIG
        )
