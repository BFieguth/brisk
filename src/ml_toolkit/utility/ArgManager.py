"""Provides the ArgManager class for managing command-line arguments with preset common inputs.

Exports:
    - ArgManager: A class that wraps around argparse.ArgumentParser with preset 
        inputs for common variables such as k-fold for cross-validation and 
        dataset selection.
"""


import argparse
from typing import List, Optional

class ArgManager:
    """A customizable argument parser with preset common inputs.

    This class wraps around argparse.ArgumentParser, providing common arguments 
    like k-fold, number of repeats, datasets, and scoring, while allowing for 
    additional custom arguments to be added.

    Attributes:
        parser (ArgumentParser): The argument parser object used to handle 
            command-line arguments.
    """

    def __init__(self, description: str):
        """Initializes the ArgManager with common arguments.

        Args:
            description (str): The description of the script.
        """
        self.parser = argparse.ArgumentParser(description=description)
        self._add_common_arguments()

    def _add_common_arguments(self) -> None:
        """Adds common arguments to the argument parser.

        The common arguments include:
        - kfold: Number of folds for cross-validation.
        - num_repeats: Number of repeats for cross-validation.
        - datasets: Names of the datasets (tables in an SQL database) to use.
        - scoring: The metric to evaluate and optimize models with.
        """
        self.parser.add_argument(
            "--kfold", "-k", type=int, action="store", dest="kfold", 
            default=10, required=False, 
            help="Number of folds for cross-validation, default is 10."
        )
        self.parser.add_argument(
            "--num_repeats", "-n", type=int, action="store", dest="num_repeats", 
            default=5, required=False, 
            help="Number of repeats for cross validation, default is 5."
        )
        self.parser.add_argument(
            "--datasets", "-d", action="store", dest="datasets", nargs="+", 
            required=True, help="Names of tables in SQL database to use."
        )
        self.parser.add_argument(
            "--scoring", "-s", action="store", dest="scoring", required=True,
            help="Metric to evaluate and optimize models with."
        )

    def add_argument(self, *args, **kwargs) -> None:
        """Adds a custom argument to the parser.

        Args:
            *args: Positional arguments for argparse's add_argument.
            **kwargs: Keyword arguments for argparse's add_argument.
        """
        self.parser.add_argument(*args, **kwargs)

    def parse_args(
        self, 
        additional_args: Optional[List[str]] = None
    ) -> argparse.Namespace:
        """Parses the command-line arguments.

        Args:
            additional_args (Optional[List[str]]): List of additional arguments 
                to parse. Defaults to None.

        Returns:
            argparse.Namespace: A namespace containing the parsed arguments.

        Raises:
            SystemExit: If argument parsing fails and the parser exits.
            Exception: If an unexpected error occurs during argument parsing.
        """
        try:
            if additional_args:
                args = self.parser.parse_args(additional_args)
            else:
                args = self.parser.parse_args()
            
            print("Arguments parsed successfully.")
            return args
        
        except SystemExit as e:
            print(f"Argument parsing failed with SystemExit: {e}")
            raise

        except Exception as e:
            print(f"Unexpected error during argument parsing: {e}")
            raise
