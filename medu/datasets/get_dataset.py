import logging
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from numpy import array as Array
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from medu import FIXED_SPLITS_PATH
from medu.datasets.splits import get_output_split_path
from medu.settings import DEFAULT_PIN_MEMORY
from medu.utils import DataSplit, get_num_workers_from_shuffle


from .isic import ISIC, get_isic_train_transform, get_isic_test_transform
from .busi import BUSI, get_busi_train_transform, get_busi_test_transform
from .mri import MRI, get_mri_train_transform, get_mri_test_transform

logger = logging.getLogger(__name__)


def is_train_from_data_split(data_split: DataSplit) -> bool:
    """Determine whether a data split comes from the training set

    Args:
        data_split (DataSplit): Data split to check for

    Raises:
        ValueError: If the DataSplit is not recognized

    Returns:
        bool: True if the associated splits comes from the training set
    """
    is_train = None
    are_train = [DataSplit.train, DataSplit.val, DataSplit.forget, DataSplit.retain]
    are_test = [DataSplit.test]
    if data_split in are_train:
        is_train = True
    elif data_split in are_test:
        is_train = False
    else:
        raise ValueError(f"Data split {data_split} not supported")
    assert is_train is not None
    return is_train


def get_dataset_and_lengths(
    datasets_root: Path, dataset_name: str, transform: transforms.Compose
) -> Tuple[Dataset, List[int]]:
    """Loads the dataset with the given name and applies the given transform

    Args:
        datasets_root (Path): Directory in which the datasets are saved
        dataset_name (str): Name of the daset
        transform (transforms.Compose): Transform to apply to the dataset

    Raises:
        ValueError: The provided dataset name is not supported

    Returns:
        Dataset: Complete dataset that contains all of the samples
    """
    logger.info(f"Dataset path: {datasets_root}")
    
    if dataset_name == "isic":
        dataset_func = ISIC
        dataset = ConcatDataset(
            [
                dataset_func(
                    root=datasets_root,
                    train=data_split,
                    transform=transform,
                )
                for data_split in [True, False]
            ]
        )
        lengths = [len(sub_dataset) for sub_dataset in dataset.datasets]
    elif dataset_name == "busi":
        dataset_func = BUSI
        dataset = ConcatDataset(
            [
                dataset_func(
                    root=datasets_root,
                    train=data_split,
                    transform=transform,
                )
                for data_split in [True, False]
            ]
        )
        lengths = [len(sub_dataset) for sub_dataset in dataset.datasets]
    elif dataset_name == "mri":
        dataset_func = MRI
        dataset = ConcatDataset(
            [
                dataset_func(
                    root=datasets_root,
                    train=data_split,
                    transform=transform,
                )
                for data_split in [True, False]
            ]
        )
        lengths = [len(sub_dataset) for sub_dataset in dataset.datasets]
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    logger.info(f"Dataset loaded: {dataset}")
    for sub in dataset.datasets:
        print(len(sub), sub.transform)
    return dataset, lengths


SUPPORTED_DATASETS_TO_TRAIN_TEST_TRANSFORM = {
    "isic": (get_isic_train_transform, get_isic_test_transform),
    "busi": (get_busi_train_transform, get_busi_test_transform),
    "mri": (get_mri_train_transform, get_mri_test_transform),
}


def get_train_transform(dataset_name: str) -> transforms.Compose:
    assert (
        dataset_name in SUPPORTED_DATASETS_TO_TRAIN_TEST_TRANSFORM
    ), f"Dataset {dataset_name} not supported"
    return SUPPORTED_DATASETS_TO_TRAIN_TEST_TRANSFORM[dataset_name][0]()


def get_test_transform(dataset_name: str) -> transforms.Compose:
    assert (
        dataset_name in SUPPORTED_DATASETS_TO_TRAIN_TEST_TRANSFORM
    ), f"Dataset {dataset_name} not supported"
    return SUPPORTED_DATASETS_TO_TRAIN_TEST_TRANSFORM[dataset_name][1]()


def get_dataset_based_on_split_state(
    train_transformed: Dataset,
    test_transformed: Dataset,
    required_state: str,
    indices: Array,
):
    assert required_state in [
        "train",
        "test",
    ], f"State {required_state} not supported. Supported states: ['train', 'test']"
    dataset = train_transformed if required_state == "train" else test_transformed
    return torch.utils.data.Subset(dataset, indices)


def get_loaders_from_dataset_and_unlearner_from_cfg_with_indices(
    root: Path,
    indices: List[Array],
    dataset_cfg: DictConfig,
    unlearner_cfg: DictConfig,
) -> List[DataLoader]:
    """Generate loaders with precomputed indices

    Args:
        root (Path): Where the datasets is located
        indices (List[Array]): List of indices for the different set
                               must be 5.
        dataset_cfg (DictConfig):  Configuration of dataset
        unlearner_cfg (DictConfig): Configuration of unlearners

    Returns:
        _type_: _description_
    """
    assert (
        len(indices) == 5
    ), f"Expected 5 indices, got {len(indices)}: (Train, Retain, Forget, Val, Test)"
    assert all(
        [indices[ndx].ndim == 1 for ndx in range(len(indices))]
    ), "Some indices are None"
    dataset_root = root / "datasets"
    dataset_name = dataset_cfg.name
    non_augmented_dataset, _ = get_dataset_and_lengths(
        dataset_root, dataset_name, get_train_transform(dataset_name)
    )
    augmented_dataset, _ = get_dataset_and_lengths(
        dataset_root, dataset_name, get_test_transform(dataset_name)
    )

    train_indices, retain_indices, forget_indices, val_indices, test_indices = indices
    train_set, retain_set, forget_set, val_set, test_set = [
        get_dataset_based_on_split_state(
            non_augmented_dataset,
            augmented_dataset,
            getattr(unlearner_cfg.loaders, split_name).state,
            indices,
        )
        for split_name, indices in zip(
            ["train", "retain", "forget", "val", "test"],
            [train_indices, retain_indices, forget_indices, val_indices, test_indices],
        )
    ]
    print(
        f"When unloading: {len(train_set)}, {len(retain_set)}, {len(forget_set)}, {len(val_set)}, {len(test_set)}]"
    )
    train_loader, retain_loader, forget_loader, val_loader, test_loader = [
        DataLoader(
            data_set,
            batch_size=unlearner_cfg.batch_size,
            shuffle=getattr(unlearner_cfg.loaders, split_name).shuffle,
            num_workers=get_num_workers_from_shuffle(
                getattr(unlearner_cfg.loaders, split_name).shuffle
            ),
            pin_memory=DEFAULT_PIN_MEMORY,
        )
        for split_name, data_set in zip(
            ["train", "retain", "forget", "val", "test"],
            [train_set, retain_set, forget_set, val_set, test_set],
        )
    ]
    for loader in [train_loader, retain_loader, forget_loader, val_loader, test_loader]:
        print(len(loader), loader.dataset, loader.batch_size)
        if isinstance(loader.dataset, torch.utils.data.Subset):
            print(loader.dataset.dataset, loader.dataset.indices)
            if isinstance(loader.dataset.dataset, torch.utils.data.ConcatDataset):
                print(loader.dataset.dataset.datasets[0])
    return train_loader, retain_loader, forget_loader, val_loader, test_loader


def get_loaders_from_dataset_and_unlearner_from_cfg(
    root: Path, dataset_cfg: DictConfig, unlearner_cfg: DictConfig, random_state: int
) -> List[DataLoader]:
    print(f"Loadings loaders from dataset and unlearner {unlearner_cfg.loaders}")
    dataset_name = dataset_cfg.name
    split_path = root / FIXED_SPLITS_PATH
    train_inidices, retain_indices, forget_indices, val_indices, test_indices = [
        np.load(
            get_output_split_path(
                split_path,
                dataset_name,
                split_name,
                random_state=random_state,
            )
        )
        for split_name in [
            DataSplit.train,
            DataSplit.retain,
            DataSplit.forget,
            DataSplit.val,
            DataSplit.test,
        ]
    ]
    return get_loaders_from_dataset_and_unlearner_from_cfg_with_indices(
        root=root,
        indices=[
            train_inidices,
            retain_indices,
            forget_indices,
            val_indices,
            test_indices,
        ],
        dataset_cfg=dataset_cfg,
        unlearner_cfg=unlearner_cfg,
    )