from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import Dims2Dataset


class RotationHead(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.backbone = nn.Sequential(
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
		return self.head(self.backbone(x))


class RotationEquivariantModel(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.scorer = RotationHead()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		rotations = [0, 1, 2, 3]
		scores = []
		for k in rotations:
			rot_x = torch.rot90(x, k, dims=(-2, -1))
			score = self.scorer(rot_x)
			angle = k * math.pi / 2.0
			rot_matrix = x.new_tensor(
				[[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
			)
			rotated_score = score @ rot_matrix.T
			scores.append(rotated_score)
		stacked = torch.stack(scores, dim=1)
		weights = torch.softmax(stacked.norm(dim=-1), dim=1).unsqueeze(-1)
		return (stacked * weights).sum(dim=1)
