from typing import Callable, Dict, Union

import matplotlib
from matplotlib import colormaps
from optuna.trial import Trial
from collections import OrderedDict


def generate_colors_from_colormap(num_colors, cmap_name="viridis"):
    """
    Generate a list of colors from a specified Matplotlib colormap.

    Args:
    num_colors (int): Number of colors to generate.
    cmap_name (str): Name of the colormap to use.

    Returns:
    list: A list of colors in HEX format.
    """
    colormap = colormaps[cmap_name]  # Get the colormap
    colors = [colormap(i) for i in range(colormap.N)]  # Extract the colors as RGBA
    step = len(colors) // num_colors  # Determine step to get evenly spaced colors

    # Select colors at evenly spaced intervals
    selected_colors = [colors[i * step] for i in range(num_colors)]

    # Convert RGBA to HEX (skip if you prefer RGBA)
    hex_colors = [matplotlib.colors.to_hex(color[:3]) for color in selected_colors]

    return hex_colors


MARKERS = [
    "o",
    "s",
    "D",
    "^",
    "v",
    "<",
    ">",
    "p",
    "*",
    "h",
    "H",
    "+",
    "x",
    "d",
]


MODEL_INIT_DIR = "model_initializations/"
DEFAULT_MODEL_INIT_DIR = "unlearn/unlearner_original"
# DEFAULT_DEVICE = "cpu"
DEFAULT_DEVICE = "cuda"

# DEFAULT_OPTUNA_N_TRIALS = 100
# DEFAULT_TRAINING_EPOCHS = 200
DEFAULT_OPTUNA_N_TRIALS = 5
DEFAULT_TRAINING_EPOCHS = 20

DEFAULT_UNLEARN_EPOCHS = 5
# Batch size should be 64
# but updated with values from https://arxiv.org/pdf/2310.12508.pdf
DEFAULT_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 0.1  # https://arxiv.org/pdf/2304.04934.pdf
DEFAULT_MOMENTUM = 0.9  # https://arxiv.org/pdf/2304.04934.pdf
DEFAULT_WEIGHT_DECAY = 5e-4  # https://arxiv.org/pdf/2304.04934.pdf
DEFAULT_RANDOM_STATE = 123
DEFAULT_PIN_MEMORY = False


TRAIN_STATE = "train"
TEST_STATE = "test"


def augmented_train_retain_forget_loaders():
    res = {
        "train": {"state": TRAIN_STATE, "shuffle": True},
        "retain": {"state": TRAIN_STATE, "shuffle": True},
        "forget": {"state": TRAIN_STATE, "shuffle": True},
        "val": {"state": TEST_STATE, "shuffle": False},
        "test": {"state": TEST_STATE, "shuffle": False},
    }
    return res


def augmented_train_retain_loaders():
    res = {
        "train": {"state": TRAIN_STATE, "shuffle": True},
        "retain": {"state": TRAIN_STATE, "shuffle": True},
        "forget": {"state": TEST_STATE, "shuffle": True},
        "val": {"state": TEST_STATE, "shuffle": False},
        "test": {"state": TEST_STATE, "shuffle": False},
    }
    return res


def default_loaders():
    return {
        "train": {"state": TEST_STATE, "shuffle": True},
        "retain": {"state": TEST_STATE, "shuffle": True},
        "forget": {"state": TEST_STATE, "shuffle": True},
        "val": {"state": TEST_STATE, "shuffle": False},
        "test": {"state": TEST_STATE, "shuffle": False},
    }


def default_loaders_no_shuffle_forget():
    default = default_loaders()
    default["forget"]["shuffle"] = False
    return default


def default_evaluation_loaders():
    return {
        "train": {"state": TEST_STATE, "shuffle": False},
        "retain": {"state": TEST_STATE, "shuffle": False},
        "forget": {"state": TEST_STATE, "shuffle": False},
        "val": {"state": TEST_STATE, "shuffle": False},
        "test": {"state": TEST_STATE, "shuffle": False},
    }


def default_scheduler():
    return {
        "type": "torch.optim.lr_scheduler.CosineAnnealingLR",
        "optimizer": None,
        "T_max": None,
    }


def default_criterion():
    return {
        "type": "torch.nn.CrossEntropyLoss",
    }


# Hyper parameters mapping and their associated sweeps
HP_LEARNING_RATE = "learning_rate"
HP_WEIGHT_DECAY = "weight_decay"
HP_MOMENTUM = "momentum"
HP_NUM_EPOCHS = "num_epochs"
HP_NUM_EPOCHS_FLOAT = "num_epochs_float"
HP_BATCH_SIZE = "batch_size"
HP_NORMAL_SIGMA = "normal_sigma"
HP_FLOAT = "float"
HP_TEMPERATURE = "temperature"
HP_ETA_MIN = "eta_min"
HP_NUM_LAYERS = "num_layers"
HP_TRAINING_EPOCH_FACTOR = "training_epoch_factor"
HP_INT = "int"


def optuna_suggest_learning_rate(trial: Trial, name: str) -> float:
    return trial.suggest_float(name, 1e-5, 1e-1, log=True)


def optuna_suggest_num_layers(trial: Trial, name: str, high: int = 10) -> int:
    return trial.suggest_int(name, 1, high)


def optuna_suggest_weight_decay(trial: Trial, name: str) -> float:
    return trial.suggest_float(name, 1e-5, 1e-1, log=True)


def optuna_suggest_momentum(trial: Trial, name: str) -> float:
    return trial.suggest_float(name, low=DEFAULT_MOMENTUM, high=DEFAULT_MOMENTUM)


def optuna_suggest_num_epochs(trial: Trial, name: str, high: int) -> int:
    return trial.suggest_int(name, 1, high)


def optuna_suggest_num_epochs_float(trial: Trial, name: str, high: int) -> float:
    return trial.suggest_float(name, 1, high)


def optuna_suggest_batch_size(trial: Trial, name: str) -> int:
    return trial.suggest_categorical(name, (64, 128, 256, 512))


def optuna_suggest_normal_sigma(trial: Trial, name: str) -> float:
    return trial.suggest_float(name, 0.0, 1.0)


def optuna_suggest_float(trial: Trial, name: str) -> float:
    return trial.suggest_float(name, 0.0, 1.0)


def optuna_suggest_temperature(trial: Trial, name: str) -> float:
    return trial.suggest_float(name, 0.0, 5.0)


def optuna_suggest_eta_min(trial: Trial, name: str) -> float:
    return trial.suggest_float(name, 1e-6, 1e-1, log=True)


def optuna_suggest_trainig_epoch_factor(trial: Trial, name: str) -> int:
    return trial.suggest_int(name, 1, 10)


def optuna_suggest_int(trial: Trial, name: str) -> int:
    return trial.suggest_int(name, 30, 70)


TrialSuggestionFunctionType = Callable[..., Union[int, float]]


HP_OPTUNA: Dict[str, TrialSuggestionFunctionType] = {
    HP_LEARNING_RATE: optuna_suggest_learning_rate,
    HP_WEIGHT_DECAY: optuna_suggest_weight_decay,
    HP_MOMENTUM: optuna_suggest_momentum,
    HP_NUM_EPOCHS: optuna_suggest_num_epochs,
    HP_NUM_EPOCHS_FLOAT: optuna_suggest_num_epochs_float,
    HP_BATCH_SIZE: optuna_suggest_batch_size,
    HP_NORMAL_SIGMA: optuna_suggest_normal_sigma,
    HP_FLOAT: optuna_suggest_float,
    HP_TEMPERATURE: optuna_suggest_temperature,
    HP_ETA_MIN: optuna_suggest_eta_min,
    HP_NUM_LAYERS: optuna_suggest_num_layers,
    HP_TRAINING_EPOCH_FACTOR: optuna_suggest_trainig_epoch_factor,
    HP_INT: optuna_suggest_int,
}

HYPER_PARAMETERS = {
    "unlearner.cfg.optimizer.learning_rate": HP_LEARNING_RATE,
    "unlearner.cfg.optimizer.momentum": HP_MOMENTUM,
    "unlearner.cfg.optimizer.weight_decay": HP_WEIGHT_DECAY,
    "unlearner.cfg.batch_size": HP_BATCH_SIZE,
    "unlearner.cfg.num_epochs": HP_NUM_EPOCHS,
}


METHODS_TO_READABLE = OrderedDict(
    {
        "naive": "R",
        "original": "O",
        "finetune": "FT",
        "successive_random_labels": "SRL",
        "salun": "SalUN",
        "kgltop2": "MSG",
        "kgltop5": "CT",
        "kgltop6": "KDE",
        "grin": "GRIN",
        "grinv2": "GRINV2",
        "BiO": "BiO",
        "fcu": "FCU",
        "forgetMI": "ForgetMI",
        "grinplus": "GRIN+",
    }
)

COLORS = generate_colors_from_colormap(14, "viridis")

METHODS_TO_COLOR = OrderedDict(
    {
        "naive": COLORS[0],
        "original": COLORS[1],
        "finetune": COLORS[2],
        "successive_random_labels": COLORS[3],
        "salun": COLORS[4],
        "kgltop2": COLORS[5],
        "kgltop5": COLORS[6],
        "kgltop6": COLORS[7],
        "grin": COLORS[8],
        "grinv2": COLORS[9],
        "BiO": COLORS[10],
        "fcu": COLORS[11],
        "forgetMI": COLORS[12],
        "grinplus": COLORS[13],
    }
)

DATASET_TO_READABLE = {
    "isic": "ISIC",
    "busi": "BUSI",
    "mri": "MRI",
}

METHODS_TO_MARKER = OrderedDict(
    {
        "naive": MARKERS[0],
        "original": MARKERS[1],
        "finetune": MARKERS[2],
        "successive_random_labels": MARKERS[3],
        "salun": MARKERS[4],
        "kgltop2": MARKERS[5],
        "kgltop5": MARKERS[6],
        "kgltop6": MARKERS[7],
        "grin": MARKERS[8],
        "grinv2": MARKERS[9],
        "BiO": MARKERS[10],
        "fcu": MARKERS[11],
        "forgetMI": MARKERS[12],
        "grinplus": MARKERS[13],
    }
)

ARCHITECTURE_TO_READABLE = {
    "resnet18": "ResNet-18",
}


METHODS = list(METHODS_TO_READABLE.keys())

RESULTS_ROUND = 3
