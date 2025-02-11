# training.py
from brisk.training.training_manager import TrainingManager
from metrics import METRIC_CONFIG
from settings import create_configuration
                                
config = create_configuration()

# Define the TrainingManager for experiments
manager = TrainingManager(
    metric_config=METRIC_CONFIG,
    config_manager=config
)
