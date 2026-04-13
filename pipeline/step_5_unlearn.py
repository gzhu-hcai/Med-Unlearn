#!/usr/bin/env python
import copy
import logging
import os
import pathlib
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import hydra
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

# from hydra_zen import make_config, hydrated_dataclass, zen
from hydra.conf import HydraConf, JobConf, RunDir
from hydra.core.hydra_config import HydraConfig
from hydra_zen import make_config, store, zen
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from medu.configurations import (
    DatasetConfig,
    ModelConfig,
    UnlearnerConfig,
    get_img_size_for_dataset,
)
from medu.datasets import get_loaders_from_dataset_and_unlearner_from_cfg
from medu.models import format_model_path, get_model_from_cfg
from medu.settings import DEFAULT_RANDOM_STATE
from medu.unlearning import BaseUnlearner
from medu.utils import DictConfig, get_save_path, setup_seed

logger = logging.getLogger(__name__)


store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"dataset": "isic"},
            {"model": "resnet18"},
            {"unlearner": "naive"},
        ],
        unlearner=None,
        model=None,
        dataset=None,
        model_seed=0,
        random_state=DEFAULT_RANDOM_STATE,
        save_path=None,
        # Visualization-related configurations
        plot_loss=True,
        save_loss_csv=True,
    ),
    name="unlearn_model",
)


def inspect_unlearning(names, loaders):
    def try_unpack_concat_dataset(dataset):
        res = ""
        if isinstance(dataset, torch.utils.data.ConcatDataset):
            for sub in dataset.datasets:
                res += str(sub)
        return res

    for name, loader in zip(names, loaders):
        print("-" * 80)
        print(f"Loader: {name}")
        print(f"\tNum Batches: {len(loader)}")
        print(f"\tNum samples: {len(loader.dataset)}")
        if isinstance(loader.dataset, torch.utils.data.Subset):
            print(f"\tSubset-Dataset: {len(loader.dataset)}")
            print(f"\t\tIndices: {np.sort(loader.dataset.indices[:20]).tolist()}")
            msg = "\t\tActual Dataset: "
            msg += f"{try_unpack_concat_dataset(loader.dataset.dataset)}"
            print(msg)


def get_unlearner_name():
    overrides = HydraConfig.get().overrides.task
    unlearner = list(filter(lambda x: x.startswith("unlearner="), overrides))[
        0
    ].replace("unlearner=", "")
    return unlearner


def modify_resize_transform(dataset, new_size):
    from torchvision.transforms import Compose, Resize

    if not hasattr(dataset, "transform"):
        raise AttributeError("Dataset does not have a transform attribute.")

    if not isinstance(dataset.transform, Compose):
        raise TypeError("Dataset's transform is not a torchvision.transforms.Compose.")

    for i, transform in enumerate(dataset.transform.transforms):
        if isinstance(transform, Resize):
            dataset.transform.transforms[i] = Resize(new_size)
            logger.info(f"Resize transform modified to new size: {new_size}")
            return


# Loss Curve Visualization Function
def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Path,
    trial_number: int,
    title: str = "Training and Validation Loss"
):
    """Plot and save the training/validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    # Save image
    loss_plot_path = save_path.parent / f"{save_path.stem}_loss_curve_{trial_number}.png"
    plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Loss curve saved to {loss_plot_path}")


class UnlearnerApp:
    def __init__(
        self, dataset_cfg, unlearner, model_cfg, model_seed, random_state: int
    ):
        self.dataset_cfg = dataset_cfg
        self.unlearner: BaseUnlearner = unlearner
        self.model_cfg = model_cfg
        self.model_seed = model_seed
        self.random_state = random_state

    def get_save_path(
        self,
        root: Path,
        save: bool,
        img_size: int,
        unlearner_name: str,
        save_name: Optional[str] = None,
        save_path: Optional[Path] = None,
    ) -> Path:
        if save:
            if save_path is None:
                save_path = get_save_path(
                    root,
                    self.dataset_cfg.name,
                    self.dataset_cfg.num_classes,
                    "unlearn",
                    self.model_cfg.name,
                    self.model_seed,
                    img_size=img_size,
                )
            else:
                print(f"should be in there {root, save_name, save_path}")
                save_path = Path(
                    format_model_path(
                        str(
                            root
                            / "artifacts"
                            / self.dataset_cfg.name
                            / save_path
                            / (
                                unlearner_name
                                if unlearner_name is not None
                                else get_unlearner_name()
                            )
                        ),
                        self.dataset_cfg.num_classes,
                        self.model_cfg.name,
                        self.model_seed,
                        get_img_size_for_dataset(self.dataset_cfg.name),
                    ).replace(".pth", f"_{save_name}.pth" if save_name else ".pth")
                )
                print(f"Now we should be there {root, save_name, save_path}")
        return Path(save_path)

    def run_from_model_and_loaders(
        self,
        root: Path,
        model: Module,
        loaders: List[DataLoader],
        save: bool,
        save_path: Path,
        save_name: Optional[str] = None,
        unlearner_name: Optional[str] = None,
        trial_number: int = 0,
        plot_loss: bool = True,
        save_loss_csv: bool = True,
    ):
        setup_seed(self.random_state)
        img_size = get_img_size_for_dataset(self.dataset_cfg.name)
        _, retain_loader, forget_loader, val_loader, _ = loaders
        start = time.time()
        original_model = copy.deepcopy(model)
        last_unlearned = self.unlearner.unlearn(
            model, retain_loader, forget_loader, val_loader
        )
        # Retrieve the loss list from the unlearner instance.
        train_losses = self.unlearner.train_losses
        val_losses = self.unlearner.val_losses
        # Force conversion to a list (avoid non-list types)
        train_losses = list(train_losses) if train_losses is not None else []
        val_losses = list(val_losses) if val_losses is not None else []
        
        # original_model = copy.deepcopy(model)
        stop = time.time()
        state_dict = last_unlearned.state_dict()
        print(f"Before on {save_path}")
        save_path = self.get_save_path(
            root, save, img_size, unlearner_name, save_name, save_path
        )
        print(f"After on {save_path}")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Processing loss data
        if save and (plot_loss or save_loss_csv):
            # Save the loss data as a CSV file
            if save_loss_csv:
                loss_df = pd.DataFrame({
                    "epoch": range(1, len(train_losses) + 1),
                    "train_loss": train_losses,
                    "val_loss": val_losses
                })
                loss_csv_path = save_path.parent / f"{save_path.stem}_losses_{trial_number}.csv"
                loss_df.to_csv(loss_csv_path, index=False)
                logger.info(f"Loss data saved to {loss_csv_path}")
            
            # Plot and save the loss curve
            if plot_loss and (len(train_losses) >= 0 or len(val_losses) >= 0):
                plot_loss_curves(
                    train_losses,
                    val_losses,
                    save_path,
                    trial_number=trial_number,  # Enter trial number
                    title=f"Loss Curve (Trial {trial_number}, Unlearner: {unlearner_name or get_unlearner_name()})"
                )
        
        dump_meta = save_path.parent / save_path.name.replace(".pth", "_meta.txt")
        hydra_config = save_path.parent / save_path.name.replace(".pth", "_hydra")
        print(f"Having an issue on {hydra_config}")
        if hydra_config.exists():
            shutil.rmtree(hydra_config)
        hydra_exists = Path(".hydra").exists()
        if hydra_exists:
            shutil.copytree(".hydra", hydra_config)
        with open(dump_meta, "a") as out_fo:
            logger.info(f"Trying to save at {dump_meta}")
            if hydra_exists:
                overrides = HydraConfig.get().overrides.task
                out_fo.write(" ".join(overrides) + "\n")
            out_fo.write(f"time_{trial_number}: {stop - start}\n")
            # Statistics on write loss data
            out_fo.write(f"final_train_loss_{trial_number}: {train_losses[-1]:.4f}\n" if train_losses else "final_train_loss: N/A\n")
            out_fo.write(f"final_val_loss_{trial_number}: {val_losses[-1]:.4f}\n" if val_losses else "final_val_loss: N/A\n")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, f=save_path)
        torch.save(state_dict, f="final_state.pt")
        logger.info(f"Done unlearning using '{self.unlearner}', saved to '{save_path}'")
        # with open("exection_time.json", "w") as f:
        # json.dump({"time": stop - start}, f)
        return original_model, last_unlearned

    def get_model(self, root):
        img_size = get_img_size_for_dataset(self.dataset_cfg.name)
        model = get_model_from_cfg(
            root=root,
            model_cfg=self.model_cfg,
            unlearner_cfg=self.unlearner.cfg,
            num_classes=self.dataset_cfg.num_classes,
            model_seed=self.model_seed,
            img_size=img_size,
        )
        return model

    def run(self, root: Path, save=True, save_path=None):
        # For reproducibility
        setup_seed(self.random_state)
        logger.info(f"Starting in: {os.getcwd()}")
        img_size = get_img_size_for_dataset(self.dataset_cfg.name)
        logger.info("Image Size: " + str(img_size))
        assert self.model_cfg.name in ["resnet18"]
        model = self.get_model(root / "artifacts" / self.dataset_cfg.name)

        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            logger.info(f"Available GPUs: {num_gpus}")
            
            # Get the model
            model = self.get_model(root / "artifacts" / self.dataset_cfg.name)
            
            # Add multi-GPU support
            if num_gpus > 1:
                logger.info(f"Using {num_gpus} GPUs with DataParallel")
                model = torch.nn.DataParallel(model)
        else:
            logger.info("No GPUs available, using CPU")
            model = self.get_model(root / "artifacts" / self.dataset_cfg.name)
        
        # Move the model to the GPU (if available).
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        self.unlearner.cfg.loaders = DictConfig(self.unlearner.cfg.loaders)
        (
            train_loader,
            retain_loader,
            forget_loader,
            val_loader,
            test_loader,
        ) = get_loaders_from_dataset_and_unlearner_from_cfg(
            root=root,
            dataset_cfg=self.dataset_cfg,
            unlearner_cfg=self.unlearner.cfg,
            random_state=self.random_state,
        )
        return self.run_from_model_and_loaders(
            root=root,
            model=model,
            loaders=[
                train_loader,
                retain_loader,
                forget_loader,
                val_loader,
                test_loader,
            ],
            save=save,
            save_path=save_path,
        )


def zen_unlearn_model(
    dataset: DatasetConfig,
    model: ModelConfig,
    unlearner: UnlearnerConfig,
    model_seed: int,
    random_state: int,
    save_path=None,
    # writer="tensorboard",
    writer=None,
    save_steps: bool = False,
    should_evaluate: bool = False,
):
    if writer == "tensorboard":
        writer = SummaryWriter()
    unlearner.save_steps = save_steps
    unlearner.writer = writer
    unlearner.should_evaluate = should_evaluate

    root = pathlib.Path(hydra.utils.get_original_cwd())
    app = UnlearnerApp(
        dataset_cfg=dataset,
        unlearner=unlearner,
        model_cfg=model,
        model_seed=model_seed,
        random_state=random_state,
    )
    app.run(root, save=True, save_path=save_path)


def generate_dir_name():
    pbs_index = os.environ.get("PBS_ARRAY_INDEX", "")
    now = datetime.now()
    dir_name = now.strftime("outputs/%Y-%m-%d/%H-%M-%S")
    save_name = f"{dir_name}/{pbs_index}" if pbs_index else dir_name
    save_name = f"{save_name}/{uuid.uuid4()}"
    return save_name


if __name__ == "__main__":
    store(
        HydraConf(job=JobConf(chdir=True), run=RunDir(dir=generate_dir_name())),
        name="config",
        group="hydra",
    )
    store.add_to_hydra_store()
    zen(zen_unlearn_model).hydra_main(
        config_name="unlearn_model",
        version_base="1.1",
        config_path=None,
    )
