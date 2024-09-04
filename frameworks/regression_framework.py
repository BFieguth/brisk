"""A standard framework for training regression models."""
import ml_toolkit

notification = ml_toolkit.NotificationEmail("./frameworks/config.ini")

parser = ml_toolkit.CustomArgParser("Standard Regression Models")
# parser.add_argument(
#     "--methods", "-m", 
# )

print("regression_framework has run!")
