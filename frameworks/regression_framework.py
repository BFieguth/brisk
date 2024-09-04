"""A standard framework for training regression models."""
import ml_toolkit

scoring = ml_toolkit.ScoringManager()

notification = ml_toolkit.NotificationEmail("./frameworks/config.ini")

parser = ml_toolkit.CustomArgParser("Standard Regression Models")
parser.add_argument(
    "--methods", "-m", dest="methods", action="store", help="Methods to train" 
)

args = parser.parse_args()

methods = args.methods
kf = args.kfold
num_repeats = args.num_repeats
datasets = args.datasets
scoring = args.scoring

print("regression_framework has run!")
