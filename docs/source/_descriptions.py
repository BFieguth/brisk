"""Descriptions of Brisk objects for documentation."""

from collections import OrderedDict

# Maintain descriptions in alphabetical order by object name
DESCRIPTIONS = OrderedDict({
    "DataManager": {
        "path": "~brisk.data.data_manager.DataManager",
        "desc": "Handles the grouping, splitting, and scaling of data. Arguments"
                " are used to define the splitting strategy."
    },
    "DataSplitInfo": {
        "path": "~brisk.data.data_split_info.DataSplitInfo",
        "desc": "Stores and analyzes training and testing datasets, providing methods for calculating "
                "descriptive statistics and visualizing feature distributions."
    }
})

def generate_list_table(objects=None):
    """Generate RST list-table for specified objects or all objects."""
    if objects is None:
        objects = DESCRIPTIONS.keys()
    
    rows = []
    for name in sorted(objects):
        obj = DESCRIPTIONS[name]
        rows.append(f"   * - :class:`{obj['path']}`\n     - {obj['desc']}")
    
    return "\n".join([
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 30 70",
        "",
        "   * - Object",
        "     - Description",
    ] + rows)
