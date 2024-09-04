import argparse
from typing import List, Optional

class CustomArgParser:
    """A customizable argument parser.

    This class provides a base argument parser with common arguments
    like kfold, num_repeats, datasets and allows adding additional custom 
    arguments.

    Attributes:
        description (str): Description of the script for the parser.
        parser (ArgumentParser): The argument parser object.
    """

    def __init__(self, description: str):
        """Initializes the CustomArgParser with common arguments.

        Args:
            description (str): The description of the script.
        """
        self.parser = argparse.ArgumentParser(description=description)
        self._add_common_arguments()

    def _add_common_arguments(self) -> None:
        """Adds common arguments to the parser."""
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
            help="Names of tables in SQL database to use."
        )
        self.parser.add_argument(
            "--scoring", "-s", action="store", dest="scoring", 
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
            argparse.Namespace: Parsed arguments.
        """
        try:
            if additional_args:
                args = self.parser.parse_args(additional_args)
            else:
                args = self.parser.parse_args()
            
            print("Arguments parsed successfully.")
            return args
        
        except argparse.ArgumentError as e:
            print(f"Argument parsing failed: {e}")
            raise

        except SystemExit as e:
            print(f"Argument parsing failed with SystemExit: {e}")
            raise

        except Exception as e:
            print(f"Unexpected error during argument parsing: {e}")
            raise

    def _argument_exists(self, dest: str, flag: Optional[str] = None) -> bool:
        """
        Checks if an argument with the given dest or flag has already been added.

        Args:
            dest (str): The destination of the argument to check.
            flag (Optional[str]): An optional flag to check.

        Returns:
            bool: True if the argument exists, False otherwise.
        """
        for action in self.parser._actions:
            if action.dest == dest:
                return True
            if flag and flag in action.option_strings:
                return True
        return False
    