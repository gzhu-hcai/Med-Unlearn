import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from typing import Dict, Any, Tuple
from dataclasses import dataclass, field
import typing as typ

from medu.unlearning.common import BaseUnlearner
from medu.settings import DEFAULT_DEVICE, HYPER_PARAMETERS, HP_LEARNING_RATE, HP_FLOAT, default_loaders
import medu.settings
from medu.utils import DictConfig, get_num_classes_from_model
from medu.models import get_optimizer_scheduler_criterion


def add_gaussian_noise(images: torch.Tensor, mu: float = 0.0, sigma: float = 0.1) -> torch.Tensor:
    """Add Gaussian noise to images"""
    noise = torch.randn_like(images) * sigma + mu
    return images + noise


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute cosine distance between two feature vectors"""
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    return 1 - torch.mm(a_norm, b_norm.t()).diag().mean()


def euclidean_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute Euclidean distance between two feature vectors"""
    return torch.norm(a - b, dim=1).mean()


class ForgetMIUnlearner(BaseUnlearner):
    """
    Forget-MI: A forgetting method that combines image forgetting, 
    image retention, and classification forgetting losses to achieve
    sample-level unlearning.
    """
    ORIGINAL_NUM_EPOCHS = 30
    ORIGINAL_BATCH_SIZE = 256
    ORIGINAL_LR = 1e-4
    ORIGINAL_WEIGHT_DECAY = 1e-5
    ORIGINAL_MOMENTUM = 0.9
    
    # Loss weights
    ORIGINAL_W_IF = 0.4
    ORIGINAL_W_IR = 0.4
    ORIGINAL_W_CF = 0.2
    
    # Noise parameters
    ORIGINAL_NOISE_SIGMA = 0.1
    
    # Distance metric: 'euclidean' or 'cosine'
    ORIGINAL_DISTANCE_METRIC = 'cosine'

    HYPER_PARAMETERS = {
        "unlearner.cfg.num_epochs": medu.settings.HP_NUM_EPOCHS,
        "unlearner.cfg.batch_size": medu.settings.HP_BATCH_SIZE,
        "unlearner.cfg.optimizer.learning_rate": medu.settings.HP_LEARNING_RATE,
        "unlearner.cfg.optimizer.weight_decay": medu.settings.HP_WEIGHT_DECAY,
        "unlearner.cfg.optimizer.momentum": medu.settings.HP_FLOAT,
        "unlearner.cfg.w_if": medu.settings.HP_FLOAT,
        "unlearner.cfg.w_ir": medu.settings.HP_FLOAT,
        "unlearner.cfg.w_cf": medu.settings.HP_FLOAT,
        "unlearner.cfg.noise_sigma": medu.settings.HP_FLOAT,
        **medu.settings.HYPER_PARAMETERS,
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
        
        # Set distance function based on configuration
        self.distance_fn = cosine_distance if self.cfg.distance_metric == 'cosine' else euclidean_distance
        
        # Create original model copy for feature comparison
        self.original_model = None

    def get_feature_extractor(self, model: nn.Module) -> nn.Module:
        """
        Get feature extraction part of the model (last layer before classification head)
        This may need adjustment based on specific model architectures
        """
        # For ResNet-like architectures
        if hasattr(model, 'fc'):
            return nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        # For models with 'head' as classification layer
        elif hasattr(model, 'head'):
            return nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        # Default: return model as is (assuming features are last layer)
        return model

    def unlearn(
        self,
        model: nn.Module,
        retain_loader: DataLoader,
        forget_loader: DataLoader,
        val_loader: DataLoader,
    ) -> nn.Module:
        """
        Main unlearning method implementing the Forget-MI approach
        """
        device = self.device
        num_epochs = self.cfg.num_epochs
        batch_size = self.cfg.batch_size
        noise_sigma = self.cfg.noise_sigma
        
        # Loss weights
        w_if = self.cfg.w_if
        w_ir = self.cfg.w_ir
        w_cf = self.cfg.w_cf
        
        # Ensure weights sum to 1
        weight_sum = w_if + w_ir + w_cf
        w_if /= weight_sum
        w_ir /= weight_sum
        w_cf /= weight_sum

        # Create a copy of the original model for feature comparison
        self.original_model = copy.deepcopy(model)
        self.original_model.to(device)
        self.original_model.eval()  # Freeze original model
        
        # Prepare the model to unlearn
        model.to(device)
        model.train()
        
        # Get feature extractors for both models
        feature_extractor_ul = self.get_feature_extractor(model)
        feature_extractor_org = self.get_feature_extractor(self.original_model)
        
        # Get number of classes for uniform distribution
        num_classes = get_num_classes_from_model(model)
        
        # Optimizer setup
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.cfg.optimizer.learning_rate,
            weight_decay=self.cfg.optimizer.weight_decay
        )
        
        # KL divergence loss for classification forgetting
        kl_criterion = nn.KLDivLoss(reduction='batchmean')
        
        # Create data loaders with consistent batch sizes
        forget_loader = DataLoader(
            forget_loader.dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        # Training loop
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            total_if_loss = 0.0
            total_ir_loss = 0.0
            total_cf_loss = 0.0
            
            # Create an iterator for retain data with same batch size as forget
            retain_iter = iter(DataLoader(
                retain_loader.dataset, 
                batch_size=batch_size, 
                shuffle=True
            ))
            
            for batch_f in forget_loader:
                try:
                    batch_r = next(retain_iter)
                except StopIteration:
                    # Reset retain iterator if we reach the end
                    retain_iter = iter(DataLoader(
                        retain_loader.dataset, 
                        batch_size=batch_size, 
                        shuffle=True
                    ))
                    batch_r = next(retain_iter)
                
                # Extract images and labels
                images_f, _ = batch_f
                images_r, labels_r = batch_r
                
                # Move to device
                images_f = images_f.to(device)
                images_r = images_r.to(device)
                labels_r = labels_r.to(device)
                
                # Add noise to forget images
                images_f_noisy = add_gaussian_noise(images_f, sigma=noise_sigma)
                
                # Get features from unlearning model
                feat_f_ul = feature_extractor_ul(images_f)
                feat_r_ul = feature_extractor_ul(images_r)
                
                # Get features from original model (no gradients)
                with torch.no_grad():
                    feat_f_noisy_org = feature_extractor_org(images_f_noisy)
                    feat_r_org = feature_extractor_org(images_r)
                
                # Get logits for classification loss
                logits_f_ul = model(images_f)
                
                # Compute losses
                # Image Forgetting Loss: push away from noisy features of original model
                l_if = -self.distance_fn(feat_f_ul, feat_f_noisy_org)
                
                # Image Retention Loss: stay close to original features on retain set
                l_ir = self.distance_fn(feat_r_ul, feat_r_org)
                
                # Classification Forgetting Loss: push towards uniform distribution
                uniform_dist = torch.ones_like(logits_f_ul) / num_classes
                l_cf = kl_criterion(torch.log_softmax(logits_f_ul, dim=1), uniform_dist)
                
                # Total loss
                loss = w_if * l_if + w_ir * l_ir + w_cf * l_cf
                
                # Update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Accumulate losses for logging
                total_loss += loss.item()
                total_if_loss += l_if.item()
                total_ir_loss += l_ir.item()
                total_cf_loss += l_cf.item()
            
            # Logging
            avg_loss = total_loss / len(forget_loader)
            avg_if_loss = total_if_loss / len(forget_loader)
            avg_ir_loss = total_ir_loss / len(forget_loader)
            avg_cf_loss = total_cf_loss / len(forget_loader)
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Total Loss: {avg_loss:.4f} | IF Loss: {avg_if_loss:.4f} | IR Loss: {avg_ir_loss:.4f} | CF Loss: {avg_cf_loss:.4f}")
            
            # Evaluate if needed
            # self.evaluate_if_needed(model, val_loader, nn.CrossEntropyLoss(), device)
        
        return model


def forget_mi_default_optimizer():
    return {
        "type": "torch.optim.Adam",
        "learning_rate": ForgetMIUnlearner.ORIGINAL_LR,
        "weight_decay": ForgetMIUnlearner.ORIGINAL_WEIGHT_DECAY,
        "momentum": ForgetMIUnlearner.ORIGINAL_MOMENTUM,
    }


@dataclass
class DefaultForgetMIConfig:
    num_epochs: int = ForgetMIUnlearner.ORIGINAL_NUM_EPOCHS
    batch_size: int = ForgetMIUnlearner.ORIGINAL_BATCH_SIZE
    w_if: float = ForgetMIUnlearner.ORIGINAL_W_IF
    w_ir: float = ForgetMIUnlearner.ORIGINAL_W_IR
    w_cf: float = ForgetMIUnlearner.ORIGINAL_W_CF
    noise_sigma: float = ForgetMIUnlearner.ORIGINAL_NOISE_SIGMA
    distance_metric: str = ForgetMIUnlearner.ORIGINAL_DISTANCE_METRIC

    optimizer: typ.Dict[str, typ.Any] = field(
        default_factory=forget_mi_default_optimizer
    )
    loaders: typ.Dict[str, typ.Any] = field(default_factory=default_loaders)