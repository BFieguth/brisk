import argparse
import pytest
from unittest import mock

from brisk.utility.ArgManager import ArgManager

class TestArgManager:
    """Test class for ArgManager."""

    @pytest.fixture
    def parser(self):
        """Fixture to initialize the ArgManager."""
        return ArgManager(description="Test script")

    @mock.patch(
            "sys.argv", 
            ["script_name.py", "--kfold", "5", "--num_repeats", "10", 
             "--datasets", "dataset1", "dataset2", "--scoring", "accuracy"]
             )
    def test_parse_default_arguments(self, parser):
        """
        Test that the common arguments are parsed correctly.

        Uses:
        --kfold 5
        --num_repeats 10
        --datasets dataset1 dataset2
        --scoring accuracy
        """
        args = parser.parse_args()

        assert args.kfold == 5
        assert args.num_repeats == 10
        assert args.datasets == ["dataset1", "dataset2"]
        assert args.scoring == "accuracy"

    @mock.patch(
            "sys.argv", 
            ["script_name.py", "--kfold", "7", "--num_repeats", "3", 
             "--datasets", "dataset3", "--scoring", "f1"]
             )
    def test_custom_arguments(self, parser):
        """
        Test that custom arguments can be added and parsed.
        """
        parser.add_argument(
            "--custom_arg", type=str, default="default_val", 
            help="A custom argument."
            )
        args = parser.parse_args()

        assert args.kfold == 7
        assert args.num_repeats == 3
        assert args.datasets == ["dataset3"]
        assert args.scoring == "f1"
        assert args.custom_arg == "default_val"

    @mock.patch(
            "sys.argv", 
            ["script_name.py", "--kfold", "7", "--datasets", "dataset3", 
             "--scoring", "f1", "--custom_arg", "custom_val"]
            )
    def test_custom_arg_with_value(self, parser):
        """
        Test that a custom argument with a user-provided value is parsed correctly.
        """
        parser.add_argument("--custom_arg", type=str, help="A custom argument.")
        args = parser.parse_args()

        assert args.custom_arg == "custom_val"

    @mock.patch("sys.argv", ["script_name.py"])
    def test_missing_required_argument(self, parser):
        """
        Test handling of missing required arguments.
        --datasets is required.
        """
        with pytest.raises(SystemExit):
            parser.parse_args()

    @mock.patch("sys.argv", ["script_name.py", "--kfold", "invalid_value"])
    def test_invalid_argument_type(self, parser):
        """
        Test argument type validation.
        """
        with pytest.raises(SystemExit):
            parser.parse_args()

    @mock.patch("sys.argv", ["script_name.py"])
    def test_parse_additional_args(self, parser):
        """
        Test parsing additional arguments via additional_args parameter.
        """
        additional_args = ["--kfold", "10", "--num_repeats", "3", 
                           "--datasets", "dataset1", "dataset2", 
                           "--scoring", "f1"]
        args = parser.parse_args(additional_args)
        
        assert args.kfold == 10
        assert args.num_repeats == 3
        assert args.datasets == ["dataset1", "dataset2"]
        assert args.scoring == "f1"
        