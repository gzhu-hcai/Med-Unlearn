import typing as typ
from pathlib import Path
import glob  # Used for recursive image file searching

import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# import medu.visualization as vis


# ISIC Dataset Implementation (Adapting to Directory Structure)
class ISIC(Dataset):
    """
    ISIC Dataset Loading Class (Adapted to Actual Directory Structure)
    Directory Structure:
    root/ISIC/
        Train/
            Category 1/ # e.g., actinic keratosis
                Image 1.jpg
                Image 2.jpg
                ...
            Category 2/ # e.g., basal cell carcinoma
                ...
        Test/
            Category 1/
                ...
            Category 2/
                ...
    Labels are determined by the second-level directory name (category name)
    """
    # All category names (extracted from the directory structure, must be exactly the same as the actual second-level directory names)
    CLASSES = [
        "actinic keratosis",
        "basal cell carcinoma",
        "dermatofibroma",
        "melanoma",
        "nevus",
        "pigmented benign keratosis",
        "seborrheic keratosis",
        "squamous cell carcinoma",
        "vascular lesion"
    ]
    LABEL_MAP = {cls: idx for idx, cls in enumerate(CLASSES)}  # Mapping of categories to integer labels
    IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")  # Supported image suffixes

    def __init__(self, root: Path, train: bool, transform=None):
        super().__init__()
        self.root = Path(root) / "ISIC"
        # Select to load the Train or Test directory based on the train parameters
        self.split_dir = self.root / "Train" if train else self.root / "Test"
        self.transform = transform
        self.train = train

        # 1. Collect all image paths and corresponding labels
        self.images = []
        self.targets = []

        # Traverse all category directories
        for cls in self.CLASSES:
            cls_dir = self.split_dir / cls
            # Check if the category directory exists
            if not cls_dir.exists():
                raise FileNotFoundError(f"Category directory does not exist: {cls_dir}")
            # Find all image files in this category
            cls_images = []
            for suffix in self.IMAGE_SUFFIXES:
                # Recursively search for all files with the specified extension in this directory
                cls_images.extend(glob.glob(str(cls_dir / f"*{suffix}")))
            cls_images = list(set([Path(p) for p in cls_images]))
            if not cls_images:
                raise ValueError(f"No image file found in the category directory {cls_dir}")
            # Record image path and label
            self.images.extend(cls_images)
            self.targets.extend([self.LABEL_MAP[cls]] * len(cls_images))

        # 2. Convert to NumPy array (for easier indexing)
        self.images = np.array(self.images)
        self.targets = np.array(self.targets, dtype=int)

        # 3. Verify data integrity
        assert len(self.images) == len(self.targets), \
            f"The number of images ({len(self.images)}) does not match the number of labels ({len(self.targets)})."
        assert len(self.images) > 0, f"No images were found under {self.split_dir}"
        assert set(self.targets) == set(self.LABEL_MAP.values()), \
            "The tags are incomplete; some categories may be missing."

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ndx):
        # Load image and convert to RGB
        img_path = self.images[ndx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        target = self.targets[ndx]
        return img, target

    def __repr__(self):
        return f"ISIC Dataset (split={'Train' if self.train else 'Test'}, size={len(self)}, transform={self.transform})"


# ISIC Dataset Preprocessing Configuration
ISIC_IMAGE_SIZE = 224
ISIC_MEAN = [0.7438, 0.5865, 0.5869]
ISIC_STD = [0.131, 0.1431, 0.1597]


# ISIC training set conversion
def get_isic_train_transform():
    return transforms.Compose([
        transforms.Resize((ISIC_IMAGE_SIZE, ISIC_IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=ISIC_MEAN, std=ISIC_STD)
    ])


# ISIC test set conversion
def get_isic_test_transform():
    return transforms.Compose([
        transforms.Resize((ISIC_IMAGE_SIZE, ISIC_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=ISIC_MEAN, std=ISIC_STD)
    ])