import os
import gzip
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class TrajectoryImageDatasetMultistep(Dataset):
    """
    Dataset for loading sequences of .png images for autoregressive training.
    Each item is a list of image tensors: [input_t0, ..., target_tN]
    """
    def __init__(self, root_dir, split="train", num_steps=3, transform=None):
        """
        root_dir: Path to dataset directory (e.g., "converted_dataset")
        split: "train", "validation", or "test"
        num_steps: How many future steps to include (total returned = num_steps + 1)
        transform: Optional torchvision transform to apply to images
        """
        self.split_dir = os.path.join(root_dir, split)
        self.num_steps = num_steps
        self.transform = transform or transforms.ToTensor()

        scenario_dirs = sorted([d for d in glob.glob(os.path.join(self.split_dir, "*")) if os.path.isdir(d)])

        # Each scenario: list of .png images sorted chronologically
        self.scenarios = []
        for scenario_dir in scenario_dirs:
            image_files = sorted(glob.glob(os.path.join(scenario_dir, "image_*.png")))
            if len(image_files) >= num_steps + 1:
                self.scenarios.append(image_files)

        # Build valid sequence indices: (scenario_idx, start_frame_idx)
        self.indices = []
        for scenario_idx, images in enumerate(self.scenarios):
            for i in range(len(images) - num_steps):
                self.indices.append((scenario_idx, i))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        scenario_idx, start_idx = self.indices[idx]
        image_paths = self.scenarios[scenario_idx][start_idx:start_idx + self.num_steps + 1]

        tensors = []
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            tensor = self.transform(image)  # (3, H, W), float32 in [0.0, 1.0]
            tensors.append(tensor)

        return tensors  # List: [t0, t1, ..., tN]