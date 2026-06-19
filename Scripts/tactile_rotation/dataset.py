from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class Dims2Dataset(Dataset):
	def __init__(self, root: str | Path, split: str, rotation: int | None = None) -> None:
		self.root = Path(root)
		self.split = split
		self.rotation = rotation
		if split == "train":
			self.image_dir = self.root / "images"
			self.label_dir = self.root / "labels"
		else:
			if rotation is None:
				raise ValueError("rotation is required for test split")
			self.image_dir = self.root / "test" / str(rotation)
			self.label_dir = self.root / "test_labels" / str(rotation)

		self.image_paths = sorted(self.image_dir.glob("*.png"))
		self.label_paths = [self.label_dir / f"{path.stem}.npy" for path in self.image_paths]

	def __len__(self) -> int:
		return len(self.image_paths)

	def __getitem__(self, idx: int):
		image_path = self.image_paths[idx]
		label_path = self.label_paths[idx]
		image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
		if image is None:
			raise FileNotFoundError(f"Failed to read {image_path}")
		label = np.load(label_path).astype(np.float32).reshape(-1)
		image = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)
		label = torch.from_numpy(label)
		return image, label, image_path.stem
