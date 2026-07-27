"""Joblib adapter implementing ``SerializerPort``."""

from __future__ import annotations

import pathlib
from typing import Any

import joblib


class JoblibSerializerAdapter:
    """Serializes and deserializes objects via joblib."""

    def dump(self, obj: Any, path: pathlib.Path | str) -> None:
        joblib.dump(obj, pathlib.Path(path))

    def load(self, path: pathlib.Path | str) -> Any:
        return joblib.load(pathlib.Path(path))
