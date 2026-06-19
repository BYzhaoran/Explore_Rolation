from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dataset import Dims2Dataset
from equivariant import RotationEquivariantModel


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train the rotation-equivariant tactile model.")
	parser.add_argument("--mode", type=int, choices=(1, 2), default=1, help="1 uses Dims2, 2 uses Dims2_Mul.")
	parser.add_argument("--data-root", type=Path, default=None)
	parser.add_argument("--epochs", type=int, default=20)
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--checkpoint", type=Path, default=None)
	parser.add_argument(
		"--max-samples",
		type=int,
		default=1000,
		help="Maximum number of training samples to use (default: None, use all)",
	)
	return parser.parse_args()


def direction_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
	predictions = torch.nn.functional.normalize(predictions, dim=-1)
	targets = torch.nn.functional.normalize(targets, dim=-1)
	return torch.nn.functional.mse_loss(predictions, targets)


def train_one_epoch(model, loader, optimizer, device):
	model.train()
	total_loss = 0.0
	for images, labels, _ in loader:
		images = images.to(device)
		labels = labels.to(device)
		optimizer.zero_grad(set_to_none=True)
		predictions = model(images)
		loss = direction_loss(predictions, labels)
		loss.backward()
		optimizer.step()
		total_loss += loss.item() * images.size(0)
	return total_loss / len(loader.dataset)


def main() -> None:
	args = parse_args()
	data_root = args.data_root or (Path(__file__).resolve().parents[2] / "Datasets" / ("Dims2" if args.mode == 1 else "Dims2_Mul"))
	checkpoint = args.checkpoint or (Path(__file__).resolve().parent / ("equivariant_mode1.pt" if args.mode == 1 else "equivariant_mode2.pt"))
	device = torch.device(args.device)
	train_dataset = Dims2Dataset(data_root, split="train")
	if args.max_samples is not None:
		train_dataset = Subset(train_dataset, range(min(args.max_samples, len(train_dataset))))
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
	model = RotationEquivariantModel().to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	best_loss = float("inf")
	for epoch in range(1, args.epochs + 1):
		loss = train_one_epoch(model, train_loader, optimizer, device)
		print(f"epoch={epoch} train_loss={loss:.6f}")
		if loss < best_loss:
			best_loss = loss
			checkpoint.parent.mkdir(parents=True, exist_ok=True)
			torch.save({"model_state": model.state_dict(), "args": vars(args)}, checkpoint)
			print(f"saved checkpoint to {checkpoint}")


if __name__ == "__main__":
	main()
