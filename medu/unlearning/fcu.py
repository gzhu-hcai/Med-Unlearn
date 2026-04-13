import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import tensorboardX as tbx
from tqdm import tqdm
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

import medu.settings
from medu.models import get_optimizer_scheduler_criterion
from medu.unlearning.common import BaseUnlearner, train_one_epoch
from medu.utils import get_num_classes_from_model

class FCUUnlearner(BaseUnlearner):
    # Hyperparameter default values
    ORIGINAL_NUM_EPOCHS_UNLEARN = 50
    ORIGINAL_NUM_EPOCHS_FINETUNE = 10
    ORIGINAL_FGMP_INTERVAL = 10
    ORIGINAL_TEMPERATURE = 0.5
    ORIGINAL_LOW_FREQ_RATIO = 0.3
    ORIGINAL_LEARNING_RATE_UNLEARN = 0.001
    ORIGINAL_LEARNING_RATE_FINETUNE = 0.0001
    ORIGINAL_BATCH_SIZE = 32

    HYPER_PARAMETERS = {
        "unlearner.cfg.num_epochs_unlearn": medu.settings.HP_NUM_EPOCHS,
        "unlearner.cfg.num_epochs_finetune": medu.settings.HP_NUM_EPOCHS,
        "unlearner.cfg.temperature": medu.settings.HP_TEMPERATURE,
        "unlearner.cfg.low_freq_ratio": medu.settings.HP_FLOAT,
        "unlearner.cfg.learning_rate_unlearn": medu.settings.HP_LEARNING_RATE,
        "unlearner.cfg.learning_rate_finetune": medu.settings.HP_LEARNING_RATE,
        "unlearner.cfg.batch_size": medu.settings.HP_BATCH_SIZE,
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

    def get_feature_extractor(self, model):
        """Get the feature extractor of the model (remove the last classifier layer)"""
        if hasattr(model, 'fc'):
            return nn.Sequential(*list(model.children())[:-1])
        elif hasattr(model, 'head'):
            return nn.Sequential(*list(model.children())[:-1])
        else:
            # For other model structures, try to return all layers except the last one
            return nn.Sequential(*list(model.children())[:-1])

    def cosine_similarity(self, a, b):
        """Calculate the cosine similarity of two feature vectors"""
        a_norm = F.normalize(a, p=2, dim=1)
        b_norm = F.normalize(b, dim=1)
        return torch.sum(a_norm * b_norm, dim=1)

    def mcu_loss(self, z, z_down, z_tr, temperature):
        """Calculate Model Contrastive Unlearning loss"""
        sim_down = self.cosine_similarity(z, z_down) / temperature
        sim_tr = self.cosine_similarity(z, z_tr) / temperature
        return -torch.log(torch.exp(sim_down) / (torch.exp(sim_down) + torch.exp(sim_tr))).mean()

    def create_frequency_mask(self, weight_shape, ratio):
        """Create frequency mask to retain low-frequency components"""
        h, w = weight_shape
        mask = np.zeros((h, w))
        center_h, center_w = h // 2, w // 2
        # Calculate the size of the low-frequency region to retain
        radius_h, radius_w = int(h * ratio / 2), int(w * ratio / 2)
        
        # Create mask with central low-frequency region set to 1
        for i in range(h):
            for j in range(w):
                if abs(i - center_h) <= radius_h and abs(j - center_w) <= radius_w:
                    mask[i, j] = 1
        return torch.tensor(mask, device=self.device, dtype=torch.float32)

    def fgmp_update(self, original_model, unlearn_model, low_freq_ratio):
        """Frequency-Guided Memory Protection: retain low-frequency components, replace high-frequency components"""
        # Iterate through all convolutional layers
        for (name1, param1), (name2, param2) in zip(
            original_model.named_parameters(), 
            unlearn_model.named_parameters()
        ):
            if 'conv' in name1 and param1.dim() == 4:  # Only process convolutional layer weights
                # Get weight shape (out_channels, in_channels, kernel_h, kernel_w)
                out_c, in_c, h, w = param1.shape
                
                # Perform FFT operation for each input and output channel
                updated_weights = torch.zeros_like(param2)
                for oc in range(out_c):
                    for ic in range(in_c):
                        # Get weights
                        w_orig = param1[oc, ic].cpu().detach().numpy()
                        w_unlearn = param2[oc, ic].cpu().detach().numpy()
                        
                        # Perform FFT
                        f_orig = fftshift(fft2(w_orig))
                        f_unlearn = fftshift(fft2(w_unlearn))
                        
                        # Create frequency mask
                        mask = self.create_frequency_mask((h, w), low_freq_ratio).cpu().numpy()
                        
                        # Fuse frequency components: retain low-frequency from original model, use high-frequency from unlearning model
                        f_fused = mask * f_orig + (1 - mask) * f_unlearn
                        
                        # Perform inverse FFT
                        w_fused = np.real(ifft2(ifftshift(f_fused)))
                        
                        # Update weights
                        updated_weights[oc, ic] = torch.tensor(w_fused, device=self.device)
                
                # Update unlearning model with fused weights
                param2.data = updated_weights
        
        return unlearn_model

    def prepare_down_model(self, model):
        """Prepare downgrade model (use pre-trained model)"""
        from medu.models import get_model_from_cfg
        import copy
        
        # Create downgrade model with the same structure as the original model
        down_model = copy.deepcopy(model)
        
        if hasattr(down_model, 'fc'):
            down_model.fc.reset_parameters()
        elif hasattr(down_model, 'head'):
            down_model.head.reset_parameters()
            
        return down_model.to(self.device)

    def unlearn(
        self,
        model: nn.Module,
        retain_loader: DataLoader,
        forget_loader: DataLoader,
        val_loader: DataLoader,
    ) -> nn.Module:
        device = self.device
        
        # Hyperparameters
        num_epochs_unlearn = self.cfg.num_epochs_unlearn
        num_epochs_finetune = self.cfg.num_epochs_finetune
        fgmp_interval = self.cfg.fgmp_interval
        temperature = self.cfg.temperature
        low_freq_ratio = self.cfg.low_freq_ratio
        lr_unlearn = self.cfg.learning_rate_unlearn
        lr_finetune = self.cfg.learning_rate_finetune
        self.writer = tbx.SummaryWriter(logdir="./artifacts/fcu-writer")
        
        # Step 1: Prepare downgrade model
        print("Preparing downgrade model...")
        down_model = self.prepare_down_model(model)
        original_model = copy.deepcopy(model)  # Save original model for comparison and FGMP
        
        # Step 2: Initialize unlearning model
        unlearn_model = copy.deepcopy(model)
        unlearn_model.to(device)
        original_model.to(device)
        
        # Get feature extractors
        feature_extractor_unlearn = self.get_feature_extractor(unlearn_model)
        feature_extractor_down = self.get_feature_extractor(down_model)
        feature_extractor_original = self.get_feature_extractor(original_model)
        
        # Configure optimizer (for unlearning phase)
        optimizer_unlearn = optim.Adam(
            unlearn_model.parameters(),
            lr=lr_unlearn,
            weight_decay=5e-4
        )
        
        # Step 3: Unlearning iteration
        print("Starting unlearning iteration...")
        for epoch in tqdm(range(num_epochs_unlearn)):
            unlearn_model.train()
            down_model.eval()
            original_model.eval()
            
            total_loss = 0.0
            batch_count = 0
            
            for batch in forget_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                
                # Extract features
                with torch.no_grad():
                    z_down = feature_extractor_down(x).flatten(1)
                    z_tr = feature_extractor_original(x).flatten(1)
                
                z = feature_extractor_unlearn(x).flatten(1)
                
                # Calculate MCU loss
                loss = self.mcu_loss(z, z_down, z_tr, temperature)
                
                # Backward propagation
                optimizer_unlearn.zero_grad()
                loss.backward()
                optimizer_unlearn.step()
                
                total_loss += loss.item()
                batch_count += 1
            
            avg_loss = total_loss / batch_count
            self.writer.add_scalar('unlearn/mcu_loss', avg_loss, epoch)
            print(f"Unlearning iteration {epoch+1}/{num_epochs_unlearn}, Average loss: {avg_loss:.4f}")
            
            # Conditionally execute FGMP
            if (epoch + 1) % fgmp_interval == 0:
                print("Performing FGMP update...")
                unlearn_model = self.fgmp_update(
                    original_model, 
                    unlearn_model, 
                    low_freq_ratio
                )
        
        # Step 4: Fine-tuning (post-training)
        print("Starting fine-tuning on retain dataset...")
        # Configure fine-tuning optimizer and loss function
        optimizer_finetune = optim.Adam(
            unlearn_model.parameters(),
            lr=lr_finetune,
            weight_decay=5e-4
        )
        scheduler_finetune = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_finetune, T_max=num_epochs_finetune
        )
        criterion = nn.CrossEntropyLoss()
        
        # Fine-tune on retain dataset
        for epoch in tqdm(range(num_epochs_finetune)):
            train_loss = train_one_epoch(
                unlearn_model, 
                retain_loader, 
                optimizer_finetune, 
                scheduler_finetune, 
                criterion, 
                device
            )
            val_loss = self.evaluate_if_needed(
                unlearn_model, 
                val_loader, 
                criterion, 
                device
            )
            
            self.writer.add_scalar('finetune/train_loss', train_loss.mean(), epoch)
            self.writer.add_scalar('finetune/val_loss', val_loss.mean(), epoch)
            print(f"Fine-tuning iteration {epoch+1}/{num_epochs_finetune}, Train loss: {train_loss.mean():.4f}, Val loss: {val_loss.mean():.4f}")
        
        # Step 5: Output final model
        return unlearn_model


# Configuration class
import typing as typ
from dataclasses import dataclass, field
from medu.settings import DEFAULT_MODEL_INIT_DIR, default_loaders

@dataclass
class DefaultFCUConfig:
    num_epochs_unlearn: int = FCUUnlearner.ORIGINAL_NUM_EPOCHS_UNLEARN
    num_epochs_finetune: int = FCUUnlearner.ORIGINAL_NUM_EPOCHS_FINETUNE
    fgmp_interval: int = FCUUnlearner.ORIGINAL_FGMP_INTERVAL
    temperature: float = FCUUnlearner.ORIGINAL_TEMPERATURE
    low_freq_ratio: float = FCUUnlearner.ORIGINAL_LOW_FREQ_RATIO
    batch_size: int = FCUUnlearner.ORIGINAL_BATCH_SIZE
    
    learning_rate_unlearn: float = FCUUnlearner.ORIGINAL_LEARNING_RATE_UNLEARN
    learning_rate_finetune: float = FCUUnlearner.ORIGINAL_LEARNING_RATE_FINETUNE
    
    optimizer_unlearn: typ.Dict[str, typ.Any] = field(
        default_factory=lambda: {
            "type": "torch.optim.Adam",
            "learning_rate": FCUUnlearner.ORIGINAL_LEARNING_RATE_UNLEARN,
            "weight_decay": 5e-4
        }
    )
    
    optimizer_finetune: typ.Dict[str, typ.Any] = field(
        default_factory=lambda: {
            "type": "torch.optim.Adam",
            "learning_rate": FCUUnlearner.ORIGINAL_LEARNING_RATE_FINETUNE,
            "weight_decay": 5e-4
        }
    )
    
    scheduler: typ.Union[typ.Dict[str, typ.Any], None] = field(
        default_factory=lambda: {
            "type": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "T_max": FCUUnlearner.ORIGINAL_NUM_EPOCHS_FINETUNE
        }
    )
    
    criterion: typ.Union[typ.Dict[str, typ.Any], None] = field(
        default_factory=lambda: {"type": "torch.nn.CrossEntropyLoss"}
    )
    
    model_initializations_dir: str = DEFAULT_MODEL_INIT_DIR
    loaders: typ.Dict[str, typ.Any] = field(default_factory=default_loaders)