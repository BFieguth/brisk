"""Sklearn adapter for the data splitting system.

Wraps scikit-learn splitters behind the SplitterFactoryPort interface.
"""

from typing import cast

from sklearn import model_selection

from brisk.ports import splitter


class SklearnSplitterFactory:
    """Creates scikit-learn splitter instances from configuration parameters.

    Maps split method, stratification, and grouping settings to the
    appropriate scikit-learn cross-validation splitter.

    Examples
    --------
    >>> factory = SklearnSplitterFactory()
    >>> cv = factory.create_splitter(
    ...     split_method="kfold", n_splits=5, test_size=0.2,
    ...     stratified=True, group_column=None, random_state=42,
    ... )
    """

    def create_splitter(
        self,
        split_method: str,
        n_splits: int,
        test_size: float,
        stratified: bool,
        group_column: str | None,
        random_state: int | None,
    ) -> splitter.SplitterPort:
        """Create the appropriate splitter based on configuration.

        Parameters
        ----------
        split_method : str
            Either ``"shuffle"`` or ``"kfold"``.
        n_splits : int
            Number of splits / folds.
        test_size : float
            Fraction of data reserved for testing (shuffle only).
        stratified : bool
            Whether to preserve class proportions in splits.
        group_column : str or None
            Column name for group-aware splitting, or None.
        random_state : int or None
            Random seed for reproducibility.

        Returns
        -------
        splitter.SplitterPort
            A configured scikit-learn splitter instance.

        Raises
        ------
        ValueError
            If an invalid combination of parameters is provided.
        """
        has_groups = group_column is not None

        match (split_method, stratified, has_groups):
            case ("shuffle", False, False):
                result = model_selection.ShuffleSplit(
                    n_splits=n_splits, test_size=test_size,
                    random_state=random_state,
                )
            case ("shuffle", True, False):
                result = model_selection.StratifiedShuffleSplit(
                    n_splits=n_splits, test_size=test_size,
                    random_state=random_state,
                )
            case ("shuffle", False, True):
                result = model_selection.GroupShuffleSplit(
                    n_splits=n_splits, test_size=test_size,
                    random_state=random_state,
                )
            case ("kfold", False, False):
                result = model_selection.KFold(
                    n_splits=n_splits,
                    shuffle=random_state is not None,
                    random_state=random_state,
                )
            case ("kfold", True, False):
                result = model_selection.StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=random_state is not None,
                    random_state=random_state,
                )
            case ("kfold", False, True):
                result = model_selection.GroupKFold(n_splits=n_splits)
            case ("kfold", True, True):
                result = model_selection.StratifiedGroupKFold(
                    n_splits=n_splits,
                )

            case _:
                raise ValueError(
                    f"Invalid split configuration: method={split_method!r}, "
                    f"stratified={stratified}, group_column={group_column!r}"
                )

        return cast(splitter.SplitterPort, result)
