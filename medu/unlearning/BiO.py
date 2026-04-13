import copy
import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm
import medu.settings
from medu.settings import DEFAULT_DEVICE, DEFAULT_MODEL_INIT_DIR, default_loaders
from medu.unlearning.common import BaseUnlearner
from medu.utils import DictConfig


def kl_loss_sym(logits1: Tensor, logits2: Tensor) -> Tensor:
    """Symmetric KL divergence loss"""
    p = nn.LogSoftmax(dim=-1)(logits1)
    q = nn.LogSoftmax(dim=-1)(logits2)
    kl1 = nn.KLDivLoss(reduction="batchmean")(p, q)
    kl2 = nn.KLDivLoss(reduction="batchmean")(q, p)
    return (kl1 + kl2) / 2


def top_k_soft_labels(logits: Tensor, k: int = 2) -> Tensor:
    """Generate soft labels using Top-k logits normalization"""
    top_k_values, top_k_indices = torch.topk(logits, k, dim=1)
    # Use differentiable one-hot encoding + softmax
    soft_labels = torch.zeros_like(logits).scatter_(1, top_k_indices, top_k_values)
    return nn.Softmax(dim=1)(soft_labels)


class BilevelOptimizationUnlearner(BaseUnlearner):
    # Original hyperparameters for the Bi-level Optimization method
    ORIGINAL_NUM_EPOCHS = 10
    ORIGINAL_INNER_STEPS = 50
    ORIGINAL_BATCH_SIZE = 32
    ORIGINAL_LR = 0.001
    ORIGINAL_MOMENTUM = 0.9
    ORIGINAL_WEIGHT_DECAY = 1e-4
    ORIGINAL_GAMMA = 0.1  # Noise strength
    ORIGINAL_LAMBDA = 0.5  # Regularization weight for distance
    ORIGINAL_PHI = 1.0     # Weight for retain set loss
    ORIGINAL_KAPPA = 0.2   # Threshold for decision boundary
    ORIGINAL_TOP_K = 2     # For soft labels
    
    HYPER_PARAMETERS = {
        "unlearner.cfg.num_epochs": medu.settings.HP_NUM_EPOCHS,
        "unlearner.cfg.inner_steps": medu.settings.HP_NUM_EPOCHS,
        "unlearner.cfg.batch_size": medu.settings.HP_BATCH_SIZE,
        "unlearner.cfg.optimizer.learning_rate": medu.settings.HP_LEARNING_RATE,
        "unlearner.cfg.optimizer.momentum": medu.settings.HP_MOMENTUM,
        "unlearner.cfg.optimizer.weight_decay": medu.settings.HP_WEIGHT_DECAY,
        "unlearner.cfg.gamma": medu.settings.HP_FLOAT,
        "unlearner.cfg.lambda_reg": medu.settings.HP_FLOAT,
        "unlearner.cfg.kappa": medu.settings.HP_FLOAT,
    }

    def __init__(
        self,
        cfg: DictConfig,
        device=DEFAULT_DEVICE,
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

    def find_boundary_point(
        self,
        model: Module,
        x: Tensor,
        y: Tensor,
        inner_steps: int,
        gamma: float,
        lambda_reg: float,
        kappa: float
    ) -> Tuple[Tensor, Tensor]:
        """
        Inner optimization: Find the nearest cross decision boundary point
        using perturbed sign gradient method
        """
        device = self.device
        x = x.to(device)
        y = y.to(device)
        
        # Save the original model mode, and temporarily set it to train internally (to ensure the gradient behavior of BN/dropout).
        original_model_mode = model.training
        model.train()
        
        # Key: Ensure the gradients of x and delta are calculable.
        x = x.clone().detach().requires_grad_(True)
        delta = torch.zeros_like(x, device=device, requires_grad=True)
        # Define epsilon in advance to avoid repeated calculations within the loop.
        epsilon_base = 0.1
        criterion = nn.CrossEntropyLoss()
        
        try:
            for t in range(1, inner_steps + 1):
                optimizer_delta = optim.SGD([delta], lr=epsilon_base/math.sqrt(t))  # Use the optimizer instead of manual updates to reduce video memory usage.
                optimizer_delta.zero_grad(set_to_none=True)
                # Forward pass with current perturbation
                perturbed_x = x + delta
                outputs = model(perturbed_x)
                
                # Check if we've crossed the boundary
                with torch.no_grad():
                    probs = nn.Softmax(dim=1)(outputs)
                    if probs[0, y] <= kappa:
                        break
                
                # Reset gradients (to avoid accumulation)
                if delta.grad is not None:
                    delta.grad.zero_()
                if x.grad is not None:
                    x.grad.zero_()
                
                # Compute loss and gradient
                loss = criterion(outputs, y)
                
                loss.backward()
                
                # Get gradient of loss w.r.t. delta
                g = delta.grad.clone()
                
                # Add regularization gradient (L2 norm)
                reg_grad = lambda_reg * delta
                
                # Add Gaussian noise
                z = torch.randn_like(delta)
                
                # Compute update direction
                update_direction = torch.sign(g + reg_grad + gamma * z)
                
                # Update delta
                with torch.no_grad():
                    delta += (epsilon_base / math.sqrt(t)) * update_direction
                    # Limit the range of delta to prevent excessive disturbance.
                    delta.data = torch.clamp(delta.data, -0.5, 0.5)
            
            # Get boundary point and its predicted label
            with torch.no_grad():
                x_b = x + delta
                outputs_b = model(x_b)
                y_b = torch.argmax(outputs_b, dim=1)
                
        finally:
            model.train(original_model_mode)
        
        return x_b.detach(), y_b.detach()

    def generate_relabeled_forget_set(
        self,
        model: Module,
        forget_loader: DataLoader,
        inner_steps: int,
        gamma: float,
        lambda_reg: float,
        kappa: float
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Generate relabeled forget set using boundary points"""
        model.eval()
        relabeled_x = []
        relabeled_y = []
        
        for samples in tqdm(forget_loader, desc="Finding boundary points"):
            with torch.no_grad():
                inputs, targets = samples
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
            
            for x, y in zip(inputs, targets):
                x_b, y_b = self.find_boundary_point(
                    model, 
                    x.unsqueeze(0), 
                    y.unsqueeze(0),
                    inner_steps,
                    gamma,
                    lambda_reg,
                    kappa
                )
                relabeled_x.append(x)
                relabeled_y.append(y_b.squeeze())
        
        return relabeled_x, relabeled_y

    def unlearn(
        self,
        model: Module,
        retain_loader: DataLoader,
        forget_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Module:
        # Device configuration
        device = self.device
        model.to(device)
        original_model = copy.deepcopy(model)
        original_model.eval()
        
        # Hyperparameters
        epochs = self.cfg.num_epochs
        inner_steps = self.cfg.inner_steps
        batch_size = self.cfg.batch_size
        gamma = self.cfg.gamma
        lambda_reg = self.cfg.lambda_reg
        phi = self.cfg.phi
        kappa = self.cfg.kappa
        top_k = self.cfg.top_k
        
        # Optimizer settings
        opt_lr = self.cfg.optimizer.learning_rate
        opt_momentum = self.cfg.optimizer.momentum
        opt_weight_decay = self.cfg.optimizer.weight_decay

        
        # Step 1: Generate relabeled forget set using inner optimization
        relabeled_x, relabeled_y = self.generate_relabeled_forget_set(
            original_model,
            forget_loader,
            inner_steps,
            gamma,
            lambda_reg,
            kappa
        )
        
        # Create dataset for relabeled forget set
        from torch.utils.data import TensorDataset
        relabeled_dataset = TensorDataset(
            torch.stack(relabeled_x), 
            torch.stack(relabeled_y)
        )
        relabeled_loader = DataLoader(
            relabeled_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        # Step 2: Outer optimization - fine-tune the model
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=opt_lr,
            momentum=opt_momentum,
            weight_decay=opt_weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
        
        # Create combined retain loader with appropriate batch size
        retain_loader = DataLoader(
            retain_loader.dataset,
            batch_size=min(batch_size, 64),
            shuffle=True
        )
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            
            # Iterate over both relabeled forget and retain sets
            for (forget_batch, retain_batch) in zip(
                relabeled_loader, 
                retain_loader
            ):
                optimizer.zero_grad()
                
                # Process forget batch with relabeled targets
                f_inputs, f_targets = forget_batch
                f_inputs, f_targets = f_inputs.to(device), f_targets.to(device)
                
                # Get model outputs
                f_outputs = model(f_inputs)
                
                # Use Top-k soft labels for better generalization
                if top_k > 1:
                    soft_labels = top_k_soft_labels(f_outputs.detach(), top_k)
                    f_loss = -torch.mean(torch.sum(soft_labels * nn.LogSoftmax(dim=1)(f_outputs), dim=1))
                else:
                    f_loss = criterion(f_outputs, f_targets)
                
                # Process retain batch with original targets
                r_inputs, r_targets = retain_batch
                r_inputs, r_targets = r_inputs.to(device), r_targets.to(device)
                r_outputs = model(r_inputs)
                r_loss = criterion(r_outputs, r_targets)
                
                # Combined loss with retain set supervision
                loss = f_loss + phi * r_loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            scheduler.step()
            
            # Print epoch stats
            avg_loss = total_loss / len(relabeled_loader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            # Optional validation
            if self.should_evaluate and (epoch + 1) % 5 == 0:
                model.eval()
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for samples in val_loader:
                        inputs, targets = samples
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += targets.size(0)
                        val_correct += (predicted == targets).sum().item()
                val_acc = 100 * val_correct / val_total
                print(f"Validation Accuracy: {val_acc:.2f}%")
                model.train()
        
        model.eval()
        return model


# Default configuration for BilevelOptimization Unlearner
from dataclasses import dataclass, field
import typing as typ

@dataclass
class DefaultBilevelOptimizationUnlearningConfig:
    num_epochs: int = BilevelOptimizationUnlearner.ORIGINAL_NUM_EPOCHS
    inner_steps: int = BilevelOptimizationUnlearner.ORIGINAL_INNER_STEPS
    batch_size: int = BilevelOptimizationUnlearner.ORIGINAL_BATCH_SIZE
    gamma: float = BilevelOptimizationUnlearner.ORIGINAL_GAMMA
    lambda_reg: float = BilevelOptimizationUnlearner.ORIGINAL_LAMBDA
    phi: float = BilevelOptimizationUnlearner.ORIGINAL_PHI
    kappa: float = BilevelOptimizationUnlearner.ORIGINAL_KAPPA
    top_k: int = BilevelOptimizationUnlearner.ORIGINAL_TOP_K

    optimizer: typ.Dict[str, typ.Any] = field(
        default_factory=lambda: {
            "learning_rate": BilevelOptimizationUnlearner.ORIGINAL_LR,
            "momentum": BilevelOptimizationUnlearner.ORIGINAL_MOMENTUM,
            "weight_decay": BilevelOptimizationUnlearner.ORIGINAL_WEIGHT_DECAY,
        }
    )
    scheduler: typ.Union[typ.Dict[str, typ.Any], None] = None
    criterion: typ.Union[typ.Dict[str, typ.Any], None] = None
    model_initializations_dir: str = DEFAULT_MODEL_INIT_DIR
    loaders: typ.Dict[str, typ.Any] = field(default_factory=default_loaders)