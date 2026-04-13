import typing as typ
from dataclasses import dataclass, field
from functools import partial

from hydra_zen import store

import medu.datasets
from medu.settings import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MODEL_INIT_DIR,
    DEFAULT_MOMENTUM,
    DEFAULT_TRAINING_EPOCHS,
    DEFAULT_UNLEARN_EPOCHS,
    DEFAULT_WEIGHT_DECAY,
    MODEL_INIT_DIR,
    default_criterion,
    default_loaders,
    default_scheduler,
)
from medu.unlearning import BaseUnlearner


@dataclass
class DatasetConfig:
    name: str = ""
    num_classes: int = 0


@dataclass
class ModelConfig:
    name: str
    variant: str = ""


def get_img_size_for_dataset(dataset_name: str) -> int:
    dataset_to_variant = {
        "isic": medu.datasets.ISIC_IMAGE_SIZE,
        "busi": medu.datasets.BUSI_IMAGE_SIZE,
        "mri": medu.datasets.MRI_IMAGE_SIZE,
    }
    return dataset_to_variant[dataset_name]


@dataclass
class UnlearnerConfig:
    unlearner: BaseUnlearner
    name: str


# Datasets Configurations
# =======================
# Create

isic_conf = DatasetConfig(name="isic", num_classes=9)
busi_conf = DatasetConfig(name="busi", num_classes=3)
mri_conf = DatasetConfig(name="mri", num_classes=4)

# Store
dataset_store = store(group="dataset")
dataset_store(isic_conf, name="isic")
dataset_store(busi_conf, name="busi")
dataset_store(mri_conf, name="mri")

# Models Configurations
# =====================

# Create
resnet18_conf = ModelConfig(name="resnet18")
vit11m_conf = ModelConfig(name="vit11m")

# Store
model_store = store(group="model")
model_store(resnet18_conf, name="resnet18")
model_store(vit11m_conf, name="vit11m")


def default_optimizer():
    return {
        "type": "torch.optim.SGD",
        "learning_rate": DEFAULT_LEARNING_RATE,
        "momentum": DEFAULT_MOMENTUM,
        "weight_decay": DEFAULT_WEIGHT_DECAY,
    }


def loaders_config(state="train", shuffle=False):
    return {"state": state, "shuffle": shuffle}


@dataclass
class DefaultUnlearnerConfig:
    num_epochs: int = DEFAULT_UNLEARN_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    optimizer: typ.Dict[str, typ.Any] = field(default_factory=default_optimizer)
    scheduler: typ.Union[typ.Dict[str, typ.Any], None] = field(
        default_factory=default_scheduler
    )
    criterion: typ.Dict[str, typ.Any] = field(default_factory=default_criterion)
    model_initializations_dir: str = DEFAULT_MODEL_INIT_DIR
    loaders: typ.Dict[str, typ.Any] = field(default_factory=default_loaders)


# Unlearner Configurations
unlearner_store = store(group="unlearner")


def unlearner_config_factory(**kwargs):
    return DefaultUnlearnerConfig(**kwargs)


from medu.settings import augmented_train_retain_forget_loaders

trainer_basic_config = partial(
    unlearner_config_factory,
    num_epochs=DEFAULT_TRAINING_EPOCHS,
    model_initializations_dir=MODEL_INIT_DIR,
    loaders=augmented_train_retain_forget_loaders(),
)
unlearner_basic_config = partial(
    unlearner_config_factory, num_epochs=DEFAULT_UNLEARN_EPOCHS
)


# Training "Unlearners" (Original Model and Retrained Model)
@dataclass
@unlearner_store(name="original")
class OriginalTrainerConf:
    _target_ = "medu.unlearning.OriginalTrainer"
    cfg: DefaultUnlearnerConfig = field(default_factory=trainer_basic_config)
    device: str = DEFAULT_DEVICE


@dataclass
@unlearner_store(name="naive")
class NaiveUnlearnerConf:
    _target_ = "medu.unlearning.NaiveUnlearner"
    cfg: DefaultUnlearnerConfig = field(default_factory=trainer_basic_config)
    device: str = DEFAULT_DEVICE


# Actual Unlearners:


@dataclass
@unlearner_store(name="finetune")
class FinetuneUnlearner:
    _target_: str = (
        "medu.unlearning.FinetuneUnlearner"
    )
    cfg: DefaultUnlearnerConfig = field(default_factory=unlearner_basic_config)
    device: str = DEFAULT_DEVICE


@dataclass
@unlearner_store(name="successive_random_labels")
class SuccesiveRandomLabelsUnlearner:
    _target_: str = (
        "medu.unlearning.SuccessiveRandomLabels"
    )
    cfg: DefaultUnlearnerConfig = field(default_factory=unlearner_basic_config)
    device: str = DEFAULT_DEVICE


@dataclass
@unlearner_store(name="salun")
class SalUNUnlearner:
    from medu.unlearning.salun import DefaultSaliencyUnlearningConfig

    _target_: str = (
        "medu.unlearning.SaliencyUnlearning"
    )
    cfg: DefaultSaliencyUnlearningConfig = field(
        default_factory=DefaultSaliencyUnlearningConfig
    )
    device: str = DEFAULT_DEVICE


@dataclass
@unlearner_store(name="kgltop2")
class KGLTop2Unlearner:
    from medu.unlearning.kgltop2 import DefaultKGLTop2Config

    _target_: str = "medu.unlearning.KGLTop2"
    cfg: DefaultKGLTop2Config = field(default_factory=DefaultKGLTop2Config)
    device: str = DEFAULT_DEVICE


@dataclass
@unlearner_store(name="kgltop5")
class KGLTop5Unlearner:
    from medu.unlearning.kgltop5 import DefaultKGLTop5Config

    _target_: str = "medu.unlearning.KGLTop5"
    cfg: DefaultKGLTop5Config = field(default_factory=DefaultKGLTop5Config)
    device: str = DEFAULT_DEVICE


@dataclass
@unlearner_store(name="kgltop6")
class KGLTop6Unlearner:
    from medu.unlearning.kgltop6 import DefaultKGLTop6Config

    _target_: str = "medu.unlearning.KGLTop6"
    cfg: DefaultKGLTop6Config = field(default_factory=DefaultKGLTop6Config)
    device: str = DEFAULT_DEVICE


@dataclass
@unlearner_store(name="grin")
class GRINUnlearner:
    from medu.unlearning.grin import DefaultGRINUnlearningConfig

    _target_: str = "medu.unlearning.GRINUnlearner"
    cfg: DefaultGRINUnlearningConfig = field(default_factory=DefaultGRINUnlearningConfig)
    device: str = DEFAULT_DEVICE


@dataclass
@unlearner_store(name="grinv2")
class GRINV2Unlearner:
    from medu.unlearning.grinv2 import DefaultGRINV2UnlearningConfig

    _target_: str = "medu.unlearning.GRINV2Unlearner"
    cfg: DefaultGRINV2UnlearningConfig = field(default_factory=DefaultGRINV2UnlearningConfig)
    device: str = DEFAULT_DEVICE


@dataclass
@unlearner_store(name="grinplus")
class GRINPLUSUnlearner:
    from medu.unlearning.grinplus import DefaultGRINPLUSUnlearningConfig

    _target_: str = "medu.unlearning.GRINPLUSUnlearner"
    cfg: DefaultGRINPLUSUnlearningConfig = field(default_factory=DefaultGRINPLUSUnlearningConfig)
    device: str = DEFAULT_DEVICE


@dataclass
@unlearner_store(name="BiO")
class BilevelOptimizationUnlearner:
    from medu.unlearning.BiO import DefaultBilevelOptimizationUnlearningConfig

    _target_: str = "medu.unlearning.BilevelOptimizationUnlearner"
    cfg: DefaultBilevelOptimizationUnlearningConfig = field(default_factory=DefaultBilevelOptimizationUnlearningConfig)
    device: str = DEFAULT_DEVICE


@dataclass
@unlearner_store(name="fcu")
class FCUUnlearner:
    from medu.unlearning.fcu import DefaultFCUConfig

    _target_: str = "medu.unlearning.FCUUnlearner"
    cfg: DefaultFCUConfig = field(default_factory=DefaultFCUConfig)
    device: str = DEFAULT_DEVICE


@dataclass
@unlearner_store(name="forgetMI")
class ForgetMIUnlearner:
    from medu.unlearning.forgetMI import DefaultForgetMIConfig

    _target_: str = "medu.unlearning.ForgetMIUnlearner"
    cfg: DefaultForgetMIConfig = field(default_factory=DefaultForgetMIConfig)
    device: str = DEFAULT_DEVICE


def get_dataset_config(dataset_name) -> DatasetConfig:
    datasets = dataset_store["dataset"]
    matching = list(filter(lambda dataset: dataset[1] == dataset_name, datasets))
    assert len(matching) <= 1
    return datasets[matching[0]] if len(matching) == 1 else None


def get_num_classes(dataset_name: str) -> int:
    # Retrieve the configuration from the dataset store
    dataset_config = get_dataset_config(dataset_name)
    assert dataset_config is not None, f"Dataset {dataset_name} not found"
    return dataset_config.num_classes
