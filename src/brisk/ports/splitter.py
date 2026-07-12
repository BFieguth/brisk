"""Port interfaces for the data splitting system.

Defines structural contracts for train/test splitters and the splitter factory.
"""

from typing import Any, Iterator, Protocol, runtime_checkable

import numpy.typing as npt


@runtime_checkable
class SplitterPort(Protocol):
    """Partitions data into train/test index pairs."""

    def get_n_splits(
        self,
        X: Any | None = None, # pylint: disable = C0103
        y: Any | None = None,
        groups: Any | None = None,
    ) -> int: ...

    def split(
        self,
        X: Any, # pylint: disable = C0103
        y: Any | None = None,
        groups: Any | None = None,
    ) -> Iterator[tuple[npt.NDArray[Any], npt.NDArray[Any]]]: ...


@runtime_checkable
class SplitterFactoryPort(Protocol):
    """Creates configured splitter instances from parameters."""

    def create_splitter(
        self,
        split_method: str,
        n_splits: int,
        test_size: float,
        stratified: bool,
        group_column: str | None,
        random_state: int | None,
    ) -> SplitterPort: ...
