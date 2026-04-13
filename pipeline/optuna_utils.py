from typing import Optional
from medu.datasets import get_loaders_from_dataset_and_unlearner_from_cfg
from pathlib import Path
import numpy as np
from medu.datasets import (
    get_loaders_from_dataset_and_unlearner_from_cfg_with_indices,
)


def get_loaders(
    root: Path,
    dataset_cfg,
    unlearner_cfg,
    split_ndx: Optional[int],
    forget_ndx: Optional[int],
    random_state=123,
):
    
    (
        train_loader,
        retain_loader,
        forget_loader,
        val_loader,
        test_loader,
    ) = get_loaders_from_dataset_and_unlearner_from_cfg(
        root=root,
        dataset_cfg=dataset_cfg,
        unlearner_cfg=unlearner_cfg,
        random_state=random_state,
    )
    return train_loader, retain_loader, forget_loader, val_loader, test_loader
