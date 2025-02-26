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
    },
    "Configuration": {
        "path": "~brisk.configuration.configuration.Configuration",
        "desc": "Provide an interface for creating experiment groups."
    },
    "ConfigurationManager": {
        "path": "~brisk.configuration.configuration_manager.ConfigurationManager",
        "desc": "Process the ExperimentGroups and prepare the required DataManagers."
    },
    "ExperimentFactory": {
        "path": "~brisk.configuration.experiment_factory.ExperimentFactory",
        "desc": "Create a que of Experiments from an ExperimentGroup."
    },
    "ExperimentGroup": {
        "path": "~brisk.configuration.experiment_group.ExperimentGroup",
        "desc": "Groups experiments that will be run with the same settings."
    },
    "Experiment": {
        "path": "~brisk.configuration.experiment.Experiment",
        "desc": "Stores all the data needed for one experiment run."
    },
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
