import typing as typ
from pathlib import Path

import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

BUSI_IMAGE_SIZE = 224
BUSI_MEAN = [0.2008] * 3
BUSI_STD = [0.2747] * 3

class BUSI(Dataset):
    EXPECTED_NUM_IMAGES = 1578  # Total number of images
    DEFAULT_TEST_RATIO = 0.2
    DEFAULT_RANDOM_SEED = 123
    
    # Category mapping, mapping folder names to numeric labels
    CLASS_MAP = {
        'benign': 0,
        'malignant': 1,
        'normal': 2
    }

    def __init__(self, root: Path, train: bool, transform=None):
        super().__init__()
        self.root = Path(root) / "BUSI"
        self.transform = transform
        self.train = train
        
        # Collect all image paths and their corresponding labels
        self.images = []
        self.targets = []
        
        # Iterate through each category folder
        for class_name, class_label in self.CLASS_MAP.items():
            class_dir = self.root / class_name
            # Get all PNG images in this category
            for img_path in class_dir.glob("*.png"):
                self.images.append(img_path)
                self.targets.append(class_label)
        
        # Convert to NumPy arrays for easier indexing.
        self.images = np.array(self.images)
        self.targets = np.array(self.targets)
        
        # Verify that the total number of images meets expectations.
        assert len(self.images) == self.EXPECTED_NUM_IMAGES, \
            f"Expected {self.EXPECTED_NUM_IMAGES} images, but found {len(self.images)}"
        
        # Verify that the number of categories meets expectations.
        assert len(np.unique(self.targets)) == len(self.CLASS_MAP), \
            f"Expected {len(self.CLASS_MAP)} classes, but found {len(np.unique(self.targets))}"
        
        # Divide into training and test sets
        indices = np.arange(len(self.images))
        permuted_indices = np.random.default_rng(self.DEFAULT_RANDOM_SEED).permutation(indices)
        test_size = int(self.DEFAULT_TEST_RATIO * len(self.images))
        train_indices = permuted_indices[test_size:]
        test_indices = permuted_indices[:test_size]
        
        if train:
            self.targets = self.targets[train_indices]
            self.images = self.images[train_indices]
        else:
            self.targets = self.targets[test_indices]
            self.images = self.images[test_indices]
        
        # Verify that the number of images and labels matches.
        assert len(self.images) == len(self.targets), \
            f"Number of images ({len(self.images)}) and targets ({len(self.targets)}) do not match"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ndx):
        img = Image.open(self.images[ndx])
        if img.mode != 'RGB':
            img = img.convert('RGB')  # Forced conversion to 3-channel RGB
        # Apply conversion
        if self.transform is not None:
            img = self.transform(img)
        # Get targets
        target = self.targets[ndx]
        return img, target

    def __repr__(self):
        return f"BUSI Dataset (size={len(self)}, transform={self.transform})"


def get_busi_train_transform():
    """Methods for converting the BUSI training set"""
    train_transform = transforms.Compose(
        [
            transforms.Resize(size=(BUSI_IMAGE_SIZE, BUSI_IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),  # Random horizontal flip enhancement
            transforms.ToTensor(),
            transforms.Normalize(mean=BUSI_MEAN, std=BUSI_STD),
        ]
    )
    return train_transform


def get_busi_test_transform():
    """Methods for converting the BUSI test set"""
    test_transform = transforms.Compose(
        [
            transforms.Resize(size=(BUSI_IMAGE_SIZE, BUSI_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=BUSI_MEAN, std=BUSI_STD),
        ]
    )
    return test_transform