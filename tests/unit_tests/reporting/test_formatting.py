import datetime

from brisk.reporting import formatting

class TestFormatting:
    def test_format_dict(self):
        simple = {'a': 1, 'b': 2}
        assert formatting.format_dict(simple) == "'a': 1,\n'b': 2,"

        with_list = {'a': [1, 2, 3], 'b': [4, 5, 6]}
        assert formatting.format_dict(with_list) == "'a': [1, 2, 3],\n'b': [4, 5, 6],"

        with_nested_dict = {'a': {'a1': 1, 'a2': 2}, 'b': {'b1': 3, 'b2': 4}}
        assert formatting.format_dict(with_nested_dict) == "'a': {'a1': 1, 'a2': 2},\n'b': {'b1': 3, 'b2': 4},"

        with_nested_list = {'a': [[1, 2], [3, 4]], 'b': [[5, 6], [7, 8]]}
        assert formatting.format_dict(with_nested_list) == "'a': [[1, 2], [3, 4]],\n'b': [[5, 6], [7, 8]],"

        with_classes = {'a': datetime.datetime(2021, 1, 1), 'b': datetime.datetime(2021, 1, 2)}
        assert formatting.format_dict(with_classes) == "'a': datetime.datetime(2021, 1, 1, 0, 0),\n'b': datetime.datetime(2021, 1, 2, 0, 0),"

        with_none = {'a': None, 'b': None}
        assert formatting.format_dict(with_none) == "'a': None,\n'b': None,"

        with_empty = {}
        assert formatting.format_dict(with_empty) == "{}"
