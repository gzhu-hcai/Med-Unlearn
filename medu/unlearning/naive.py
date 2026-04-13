import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from medu.models import get_optimizer_scheduler_criterion
from medu.unlearning.common import BaseUnlearner, train_one_epoch


class NaiveUnlearner(BaseUnlearner):
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
        self.train_losses = []
        self.val_losses = []

    def unlearn(
        self,
        model: nn.Module,
        retain_loader: DataLoader,
        forget_loader: DataLoader,
        val_loader: DataLoader,
    ) -> nn.Module:
        device = self.device
        (
            optimizer,
            scheduler,
            criterion,
        ) = get_optimizer_scheduler_criterion(model, self.cfg)
        model.to(device)
        should_evaluate = self.should_evaluate
        should_evaluate = False
        for epoch in tqdm(range(self.cfg.num_epochs)):
            model.train()
            train_batch_loss = train_one_epoch(
                model, retain_loader, optimizer, scheduler, criterion, device
            )
            model.eval()
            val_batch_loss = self.evaluate_if_needed(
                model, val_loader, criterion, device, should_evaluate
            )
            payload = {
                "train_loss": train_batch_loss.mean(),
                "val_loss": val_batch_loss.mean(),
            }
            self.train_losses.append(payload["train_loss"].item())
            self.val_losses.append(payload["val_loss"].item())
            self.save_and_log(model, optimizer, scheduler, payload, epoch)
        should_evaluate = False
        return model
