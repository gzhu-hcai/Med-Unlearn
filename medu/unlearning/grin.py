import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import Module
import numpy as np
import copy
import typing as typ
from dataclasses import dataclass, field

import medu.settings
from medu.unlearning.common import BaseUnlearner
from medu.utils import DictConfig


class GRINUnlearner(BaseUnlearner):
    """
    Gradient Ratio-based Information Nullification (GRIN) unlearning method.
    Implements sample-level unlearning by selectively perturbing parameters based on gradient ratios.
    """
    # Hyperparameters
    ORIGINAL_NUM_EPOCHS = 1  # GRIN doesn't require training epochs
    ORIGINAL_BATCH_SIZE = 64
    ORIGINAL_SELECTION_RATIO = 0.1  # Select top 10% parameters
    ORIGINAL_NOISE_STD = 0.01  # Gaussian noise standard deviation
    ORIGINAL_EPSILON_PERCENTILE = 5  # Percentile for epsilon calculation

    HYPER_PARAMETERS = {
        **medu.settings.HYPER_PARAMETERS,
        "unlearner.cfg.selection_ratio": medu.settings.HP_FLOAT,
        "unlearner.cfg.noise_std": medu.settings.HP_NORMAL_SIGMA,
        "unlearner.cfg.epsilon_percentile": medu.settings.HP_FLOAT,
    }

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

    def compute_gradient(self, model: Module, data_loader: DataLoader, criterion, device):
        """Compute average gradient of the model with respect to the given dataset"""
        model.to(device)
        model.eval()
        
        # Initialize gradient storage
        gradients = []
        for param in model.parameters():
            gradients.append(torch.zeros_like(param).to(device))
        
        total_samples = 0
        
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            total_samples += batch_size
            
            # Forward pass
            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass to compute gradients
            loss.backward()
            
            # Accumulate gradients
            for i, param in enumerate(model.parameters()):
                if param.grad is not None:
                    gradients[i] += param.grad * batch_size
        
        # Compute average gradients
        for i in range(len(gradients)):
            if total_samples > 0:
                gradients[i] /= total_samples
            
        return gradients

    def compute_influence_scores(self, grad_f, grad_r, epsilon_percentile):
        """Calculate influence scores s_i = |G_f| / (|G_r| + ε)"""
        influence_scores = []
        all_gr_values = []
        
        # Collect all G_r absolute values for epsilon calculation
        for gr in grad_r:
            all_gr_values.extend(gr.abs().cpu().detach().numpy().flatten())
        
        # Calculate epsilon as percentile of G_r absolute values
        if all_gr_values:
            epsilon = np.percentile(all_gr_values, epsilon_percentile)
        else:
            epsilon = 1e-8  # Fallback if no gradients
        
        # Calculate influence scores for each parameter
        for gf, gr in zip(grad_f, grad_r):
            denominator = gr.abs() + epsilon
            score = gf.abs() / denominator
            influence_scores.append(score)
            
        return influence_scores

    def create_mask(self, influence_scores, selection_ratio):
        """Create binary mask selecting top p% parameters by influence score"""
        # Collect all scores for threshold calculation
        all_scores = []
        for scores in influence_scores:
            all_scores.extend(scores.cpu().detach().numpy().flatten())
        
        if not all_scores:
            return [torch.zeros_like(score) for score in influence_scores]
        
        # Determine threshold for top p% parameters
        threshold = np.percentile(all_scores, 100 - selection_ratio * 100)
        
        # Create mask where 1 indicates parameters to perturb
        mask = []
        for scores in influence_scores:
            param_mask = (scores >= threshold).float()
            mask.append(param_mask)
            
        return mask

    def add_noise_to_parameters(self, model: Module, mask, noise_std):
        """Add Gaussian noise to parameters selected by the mask"""
        with torch.no_grad():
            for i, param in enumerate(model.parameters()):
                if i < len(mask) and mask[i] is not None:
                    # Generate Gaussian noise with given std
                    noise = torch.normal(0, noise_std, size=param.shape).to(param.device)
                    # Apply noise only to selected parameters
                    param.data += noise * mask[i]
        
        return model

    def unlearn(
        self,
        model: Module,
        retain_loader: DataLoader,
        forget_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Module:
        """Main unlearning method implementing the GRIN algorithm"""
        device = self.device
        selection_ratio = self.cfg.selection_ratio
        noise_std = self.cfg.noise_std
        epsilon_percentile = self.cfg.epsilon_percentile
        
        # Create a copy of the model to avoid modifying the original
        model_unlearn = copy.deepcopy(model)
        model_unlearn.to(device)
        
        # Define loss function
        criterion = nn.CrossEntropyLoss()
        
        # Step 2: Compute gradients
        print("Calculating forget set gradients (G_f)...")
        grad_f = self.compute_gradient(model_unlearn, forget_loader, criterion, device)
        
        print("Calculating retain set gradients (G_r)...")
        grad_r = self.compute_gradient(model_unlearn, retain_loader, criterion, device)
        
        # Step 3: Compute influence scores
        print("Calculating influence scores...")
        influence_scores = self.compute_influence_scores(grad_f, grad_r, epsilon_percentile)
        
        # Step 4: Create parameter mask
        print("Creating parameter mask...")
        mask = self.create_mask(influence_scores, selection_ratio)
        
        # Step 5: Apply selective noise injection
        print("Applying noise to selected parameters...")
        model_unlearn = self.add_noise_to_parameters(model_unlearn, mask, noise_std)
        
        return model_unlearn


def grin_default_optimizer():
    """Default optimizer configuration (not actively used in GRIN)"""
    return {
        "type": "torch.optim.SGD",
        "learning_rate": 0.01,
        "momentum": 0.9,
        "weight_decay": 5e-4,
    }


from medu.settings import DEFAULT_MODEL_INIT_DIR, default_loaders

@dataclass
class DefaultGRINUnlearningConfig:
    """Default configuration for GRIN unlearning"""
    num_epochs: int = GRINUnlearner.ORIGINAL_NUM_EPOCHS
    batch_size: int = GRINUnlearner.ORIGINAL_BATCH_SIZE
    selection_ratio: float = GRINUnlearner.ORIGINAL_SELECTION_RATIO
    noise_std: float = GRINUnlearner.ORIGINAL_NOISE_STD
    epsilon_percentile: float = GRINUnlearner.ORIGINAL_EPSILON_PERCENTILE

    optimizer: typ.Dict[str, typ.Any] = field(
        default_factory=grin_default_optimizer
    )
    scheduler: typ.Union[typ.Dict[str, typ.Any], None] = None
    criterion: typ.Union[typ.Dict[str, typ.Any], None] = None
    loaders: typ.Dict[str, typ.Any] = field(default_factory=default_loaders)