import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import Module
import numpy as np
import copy
import typing as typ
from dataclasses import dataclass, field
from collections import Counter

import medu.settings
from medu.unlearning.common import BaseUnlearner
from medu.utils import DictConfig


class GRINPLUSUnlearner(BaseUnlearner):
    """
    Gradient Ratio-based Information Nullification (GRIN) unlearning method.
    Implements sample-level unlearning by selectively perturbing parameters based on gradient ratios.
    Modified to use adversarial perturbations with class-balanced influence scores and gradient regularization.
    """
    # Hyperparameters
    ORIGINAL_NUM_EPOCHS = 1  # GRIN doesn't require training epochs
    ORIGINAL_BATCH_SIZE = 64
    ORIGINAL_SELECTION_RATIO = 0.5  # Select top 10% parameters
    ORIGINAL_ALPHA = 0.01  # Perturbation scale factor
    ORIGINAL_EPSILON_PERCENTILE = 4  # Percentile for epsilon calculation
    ORIGINAL_BETA = 0.5  # Gradient direction regularization coefficient

    HYPER_PARAMETERS = {
        **medu.settings.HYPER_PARAMETERS,
        "unlearner.cfg.selection_ratio": medu.settings.HP_FLOAT,
        "unlearner.cfg.alpha": medu.settings.HP_FLOAT,
        "unlearner.cfg.beta": medu.settings.HP_FLOAT,
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

    def compute_class_weights(self, data_loader: DataLoader, device):
        """
        Compute class weight w_c = N / (n_classes * n_c)
        where N is total samples, n_c is samples in class c
        """
        all_labels = []
        for _, targets in data_loader:
            all_labels.extend(targets.cpu().numpy().tolist())
        
        # Count samples per class
        label_counts = Counter(all_labels)
        n_classes = len(label_counts)
        N = len(all_labels)
        
        # Calculate weights: w_c = N / (n_classes * n_c)
        class_weights = {}
        for class_id, count in label_counts.items():
            class_weights[class_id] = N / (n_classes * count)
        
        return class_weights, n_classes

    def compute_gradient_batch_weighted(self, model: Module, data_loader: DataLoader, 
                                       criterion, device, class_weights=None):
        """
        Optimized: Compute gradients with class weighting in batch mode.
        This avoids per-sample iteration and maintains efficiency.
        """
        model.to(device)
        model.eval()
        
        # Initialize gradient storage
        gradients = []
        for param in model.parameters():
            gradients.append(torch.zeros_like(param).to(device))
        
        total_weighted_samples = 0
        
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            
            # Compute sample weights if class_weights provided
            if class_weights is not None:
                sample_weights = torch.tensor(
                    [class_weights.get(t.item(), 1.0) for t in targets],
                    device=device, dtype=torch.float32
                )
                # Normalize weights within batch
                sample_weights = sample_weights / sample_weights.sum() * batch_size
            else:
                sample_weights = torch.ones(batch_size, device=device)
            
            # Forward pass
            model.zero_grad()
            outputs = model(inputs)
            
            # Compute weighted loss for each sample
            losses = nn.functional.cross_entropy(outputs, targets, reduction='none')
            weighted_loss = (losses * sample_weights).sum()
            
            # Backward pass
            weighted_loss.backward()
            
            # Accumulate weighted gradients
            for i, param in enumerate(model.parameters()):
                if param.grad is not None:
                    gradients[i] += param.grad
            
            total_weighted_samples += sample_weights.sum().item()
        
        # Average gradients
        for i in range(len(gradients)):
            if total_weighted_samples > 0:
                gradients[i] /= total_weighted_samples
            
        return gradients

    def compute_gradient(self, model: Module, data_loader: DataLoader, criterion, device):
        """Compute average gradient (unweighted version for retain set)"""
        return self.compute_gradient_batch_weighted(model, data_loader, criterion, device, None)

    def compute_influence_scores(self, grad_f, grad_r, epsilon_percentile):
        """
        Fast influence score calculation: s_i = |G_f| / (|G_r| + ε)
        Class weighting already incorporated in grad_f computation.
        """
        influence_scores = []
        all_gr_values = []
        
        # Collect all G_r absolute values for epsilon calculation
        for gr in grad_r:
            all_gr_values.extend(gr.abs().cpu().detach().numpy().flatten())
        
        if all_gr_values:
            epsilon = np.percentile(all_gr_values, epsilon_percentile)
        else:
            epsilon = 1e-8
        
        # Calculate influence scores
        for gf, gr in zip(grad_f, grad_r):
            denominator = gr.abs() + epsilon
            score = gf.abs() / denominator
            influence_scores.append(score)
            
        return influence_scores

    def create_mask(self, influence_scores, selection_ratio):
        """Create binary mask selecting top p% parameters"""
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

    def compute_gradient_regularization_vectorized(self, grad_diff, grad_r, beta):
        """
        Vectorized gradient regularization: r_i = max(0, 1 - β*|cos(ΔG_i, G_r^(i))|)
        Optimized for efficiency using tensor operations.
        """
        regularization_factors = []
        
        for delta_g, g_r in zip(grad_diff, grad_r):
            # Flatten for computation
            delta_g_flat = delta_g.flatten()
            g_r_flat = g_r.flatten()
            
            # Compute norms
            norm_delta = torch.norm(delta_g_flat)
            norm_gr = torch.norm(g_r_flat)
            
            if norm_delta > 1e-8 and norm_gr > 1e-8:
                # Cosine similarity
                cos_sim = torch.dot(delta_g_flat, g_r_flat) / (norm_delta * norm_gr)
                # Regularization factor： r_i = max(0, 1 - β*|cos(ΔG_i, G_r^(i))|)
                r_i = torch.clamp(1.0 - beta * torch.abs(cos_sim), min=0.0)
            else:
                r_i = torch.tensor(1.0, device=delta_g.device)
            
            # Broadcast to parameter shape
            r_i_expanded = r_i.expand_as(delta_g)
            regularization_factors.append(r_i_expanded)
        
        return regularization_factors

    def add_adversarial_perturbation(self, model: Module, mask, grad_diff, alpha, 
                                    regularization_factors):
        """
        Add regularized adversarial perturbation to parameters selected by the mask.
        Perturbation: δ_i = α * ΔG_i * M_i * r_i
        where M_i is the mask and r_i is the regularization factor
        """
        with torch.no_grad():
            for i, param in enumerate(model.parameters()):
                if (i < len(mask) and mask[i] is not None and 
                    i < len(grad_diff) and i < len(regularization_factors)):
                    # Generate regularized adversarial perturbation
                    # δ_i = α * ΔG_i * r_i
                    perturbation = alpha * grad_diff[i] * regularization_factors[i]
                    # Apply to selected parameters only
                    param.data += perturbation * mask[i]
        
        return model

    def unlearn(
        self,
        model_unlearn: Module,
        retain_loader: DataLoader,
        forget_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Module:
        """
        Optimized GRIN unlearning with class balancing and gradient regularization.
        Key optimization: Class weighting integrated into batch gradient computation.
        """
        device = self.device
        selection_ratio = self.cfg.selection_ratio
        alpha = self.cfg.alpha
        epsilon_percentile = self.cfg.epsilon_percentile
        beta = self.cfg.beta
        
        # Copy model
        # model_unlearn = copy.deepcopy(model)
        model_unlearn.to(device)
        
        criterion = nn.CrossEntropyLoss()
        
        # Step 1: Compute class weights
        print("Computing class weights for forget set...")
        class_weights, n_classes = self.compute_class_weights(forget_loader, device)
        print(f"Found {n_classes} classes with weights: {class_weights}")
        
        # Step 2: Compute class-weighted forget gradients (optimized batch mode)
        print("Calculating class-weighted forget set gradients (G_f)...")
        grad_f = self.compute_gradient_batch_weighted(
            model_unlearn, forget_loader, criterion, device, class_weights
        )
        
        # Step 3: Compute retain gradients
        print("Calculating retain set gradients (G_r)...")
        grad_r = self.compute_gradient(model_unlearn, retain_loader, criterion, device)
        
        # Step 4: Compute gradient difference
        print("Calculating gradient difference (ΔG = G_f - G_r)...")
        grad_diff = []
        for gf, gr in zip(grad_f, grad_r):
            grad_diff.append(gf - gr)
        
        # Step 5: Compute influence scores (class weighting already in grad_f)
        print("Calculating influence scores...")
        influence_scores = self.compute_influence_scores(
            grad_f, grad_r, epsilon_percentile
        )
        
        # Step 6: Create parameter mask
        print("Creating parameter mask...")
        mask = self.create_mask(influence_scores, selection_ratio)
        
        # Step 7: Compute gradient regularization
        print("Computing gradient direction regularization...")
        regularization_factors = self.compute_gradient_regularization_vectorized(
            grad_diff, grad_r, beta
        )
        
        # Step 8: Apply perturbation
        print("Applying regularized adversarial perturbation...")
        model_unlearn = self.add_adversarial_perturbation(
            model_unlearn, mask, grad_diff, alpha, regularization_factors
        )

        # model_unlearn.eval()
        
        return model_unlearn


def grinplus_default_optimizer():
    """Default optimizer configuration"""
    return {
        "type": "torch.optim.SGD",
        "learning_rate": 0.01,
        "momentum": 0.9,
        "weight_decay": 5e-4,
    }


from medu.settings import DEFAULT_MODEL_INIT_DIR, default_loaders

@dataclass
class DefaultGRINPLUSUnlearningConfig:
    """Default configuration for optimized GRIN unlearning"""
    num_epochs: int = GRINPLUSUnlearner.ORIGINAL_NUM_EPOCHS
    batch_size: int = GRINPLUSUnlearner.ORIGINAL_BATCH_SIZE
    selection_ratio: float = GRINPLUSUnlearner.ORIGINAL_SELECTION_RATIO
    alpha: float = GRINPLUSUnlearner.ORIGINAL_ALPHA
    epsilon_percentile: float = GRINPLUSUnlearner.ORIGINAL_EPSILON_PERCENTILE
    beta: float = GRINPLUSUnlearner.ORIGINAL_BETA

    optimizer: typ.Dict[str, typ.Any] = field(
        default_factory=grinplus_default_optimizer
    )
    scheduler: typ.Union[typ.Dict[str, typ.Any], None] = None
    criterion: typ.Union[typ.Dict[str, typ.Any], None] = None
    loaders: typ.Dict[str, typ.Any] = field(default_factory=default_loaders)