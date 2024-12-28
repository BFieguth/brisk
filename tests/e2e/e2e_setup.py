"""Define objects that will be reused by all e2e testing projects.
"""
import pathlib
import shutil
import subprocess
from typing import List, Optional

import brisk

def metric_config():
    METRIC_CONFIG = brisk.MetricManager(
        *brisk.REGRESSION_METRICS,
        *brisk.CLASSIFICATION_METRICS
    )
    return METRIC_CONFIG


def algorithm_config():
    ALGORITHM_CONFIG = [
        *brisk.REGRESSION_ALGORITHMS,
        *brisk.CLASSIFICATION_ALGORITHMS
    ]
    return ALGORITHM_CONFIG

