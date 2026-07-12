"""Port interfaces for the serialization system.

Defines structural contracts for model persistence (save/load).
"""

import pathlib
from typing import Any, Protocol, TypedDict


class ModelPackage(TypedDict):
    """Typed dict containing a trained model and its metadata."""

    model: Any
    metadata: dict[str, Any]


class SerializerPort(Protocol):
    """Serializes and deserializes model objects to/from files."""

    def dump(self, obj: Any, path: pathlib.Path | str) -> None: ...

    def load(self, path: pathlib.Path | str) -> Any: ...
