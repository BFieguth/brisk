"""A standard framework for training regression models."""
import ml_toolkit

scoring = ml_toolkit.ScoringManager()

notification = ml_toolkit.NotificationEmail("./frameworks/config.ini")

parser = ml_toolkit.CustomArgParser("Standard Regression Models")
parser.add_argument(
    "--methods", "-m", nargs="+", dest="methods", action="store", 
    help="Methods to train" 
)

args = parser.parse_args()

methods = args.methods
kf = args.kfold
num_repeats = args.num_repeats
datasets = args.datasets
data_paths = [(datasets[0], None)] # Convert to list of tuples
scoring = args.scoring

data_splitter = ml_toolkit.DataSplitter(
    test_size=0.2,
    split_method="shuffle",
    group_column=None,
    stratified=False,
    n_splits=5,
    random_state=None
)

manager = ml_toolkit.TrainingManager(
    method_config=ml_toolkit.REGRESSION_MODELS,
    scoring_config=scoring,
    splitter=data_splitter,
    methods=methods,
    data_paths=data_paths
)
print(manager.configurations)

print("regression_framework has run!")
