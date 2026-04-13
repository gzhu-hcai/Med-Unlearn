import typing as typ
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from medu.datasets.transforms import ConvertTo3Channels

def crop_img(img):
    """
    Finds the extreme points on the image and crops the rectangular region
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Threshold and clean up the image
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours and get the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]  # Compatibility with different OpenCV versions
    if not cnts:  # Handle cases with no contours
        return img
    c = max(cnts, key=cv2.contourArea)

    # Find extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    # Add padding and crop
    ADD_PIXELS = 0
    new_img = img[
        max(0, extTop[1]-ADD_PIXELS):min(img.shape[0], extBot[1]+ADD_PIXELS),
        max(0, extLeft[0]-ADD_PIXELS):min(img.shape[1], extRight[0]+ADD_PIXELS)
    ].copy()
    
    return new_img


class MRI(Dataset):
    """MRI Dataset"""
    CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
    CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
    EXPECTED_NUM_CLASSES = 4

    def __init__(self, root: Path, train: bool, transform=None):
        super().__init__()
        self.root = Path(root) / "MRI"
        self.train = train
        self.transform = transform
        
        # Determine split directory (Train or Test)
        split_dir = self.root / "Train" if train else self.root / "Test"
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")

        # Collect all image paths and corresponding labels
        self.images = []
        self.targets = []
        
        for class_name in self.CLASSES:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                raise ValueError(f"Class directory not found: {class_dir}")
                
            # Get all image paths for this class
            # img_paths = list(class_dir.glob("*.jpg"))
            img_paths = list(filter(lambda p: p.suffix.lower() == ".jpg", class_dir.iterdir()))
            for img_path in img_paths:
                self.images.append(str(img_path))
                self.targets.append(self.CLASS_TO_IDX[class_name])

        self.targets = np.array(self.targets, dtype=int)
        assert len(self.images) == len(self.targets), \
            f"Number of images ({len(self.images)}) and targets ({len(self.targets)}) do not match"
        assert len(np.unique(self.targets)) == self.EXPECTED_NUM_CLASSES, \
            f"Expected {self.EXPECTED_NUM_CLASSES} classes, found {len(np.unique(self.targets))}"

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> typ.Tuple[Image.Image, int]:
        # Load image using OpenCV for preprocessing
        img_path = self.images[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
            
        # Convert to RGB (OpenCV loads as BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Crop and resize
        img = crop_img(img)
        img = cv2.resize(img, (MRI_IMAGE_SIZE, MRI_IMAGE_SIZE))
        
        # Convert to PIL Image for torchvision transforms
        img = Image.fromarray(img)
        
        # Apply transforms if specified
        if self.transform is not None:
            img = self.transform(img)
            
        target = self.targets[idx]
        return img, target

    def __repr__(self) -> str:
        return f"MRI(split={'train' if self.train else 'test'}, size={len(self)}, transform={self.transform})"


# Dataset configuration
MRI_IMAGE_SIZE = 224
MRI_MEAN = [0.1857] * 3  # Default ImageNet mean (adjust based on your dataset stats)
MRI_STD = [0.2008] * 3   # Default ImageNet std (adjust based on your dataset stats)


def get_mri_train_transform():
    """Get training transforms for MRI dataset"""
    transform = transforms.Compose([
        ConvertTo3Channels(),  # Ensure 3 channels (though MRI might already be 3 channels)
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=MRI_MEAN, std=MRI_STD),
    ])
    return transform


def get_mri_test_transform():
    """Get testing transforms for MRI dataset"""
    transform = transforms.Compose([
        ConvertTo3Channels(),  # Ensure 3 channels
        transforms.ToTensor(),
        transforms.Normalize(mean=MRI_MEAN, std=MRI_STD),
    ])
    return transform