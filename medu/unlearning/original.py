import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from medu.datasets import get_combined_retain_and_forget_loaders
from medu.models import get_optimizer_scheduler_criterion
from medu.unlearning.common import BaseUnlearner, train_one_epoch
from medu.unlearning.naive import NaiveUnlearner


class OriginalTrainer(NaiveUnlearner):
    def __init__(
        self,
        cfg: DictConfig,
        device,
        writer=None,
        save_steps: bool = False,
        should_evaluate: bool = False,
    ):
        super().__init__(
            cfg,
            device=device,
            writer=writer,
            save_steps=save_steps,
            should_evaluate=should_evaluate,
        )

    def unlearn(
        self,
        model: nn.Module,
        retain_loader: DataLoader,
        forget_loader: DataLoader,
        val_loader: DataLoader,
    ) -> nn.Module:
        train_loader = get_combined_retain_and_forget_loaders(
            retain_loader, forget_loader, shuffle=True
        )
        return super().unlearn(model, train_loader, forget_loader, val_loader)
