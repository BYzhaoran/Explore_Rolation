from __future__ import annotations

import torch
from torch import nn


class BaselineCNN(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(1, 32, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Conv2d(32, 64, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Conv2d(64, 128, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.AdaptiveAvgPool2d(1),
		)
		self.head = nn.Sequential(
			nn.Flatten(),
			nn.Linear(128, 64),
			nn.ReLU(inplace=True),
			nn.Linear(64, 2),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.head(self.features(x))
