"""Importlib adapter implementing ``ModuleLoaderPort``."""

from __future__ import annotations

import importlib.util
import os
import sys


class ModuleLoaderAdapter:
    """Loads named objects from user project module files at runtime."""

    def load_module_object(
        self,
        project_root: str,
        module_filename: str,
        object_name: str,
        required: bool = True,
    ) -> object | None:
        module_path = os.path.join(project_root, module_filename)

        if not os.path.exists(module_path):
            raise FileNotFoundError(
                f"{module_filename} not found in {project_root}"
            )

        module_name = os.path.splitext(module_filename)[0]
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        if hasattr(module, object_name):
            return getattr(module, object_name)
        if required:
            raise AttributeError(
                f"The object '{object_name}' is not defined in "
                f"{module_filename}"
            )
        return None
