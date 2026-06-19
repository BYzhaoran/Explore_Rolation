from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from dataset import Dims2Dataset
from models import BaselineCNN


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train the tactile direction baseline.")
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


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
	model.train()
	total_loss = 0.0
	for images, labels, _ in loader:
		images = images.to(device)
		labels = labels.to(device)
		optimizer.zero_grad(set_to_none=True)
		predictions = model(images)
		loss = criterion(predictions, labels)
		loss.backward()
		optimizer.step()
		total_loss += loss.item() * images.size(0)
	return total_loss / len(loader.dataset)


def main() -> None:
	args = parse_args()
	data_root = args.data_root or (Path(__file__).resolve().parents[2] / "Datasets" / ("Dims2" if args.mode == 1 else "Dims2_Mul"))
	checkpoint = args.checkpoint or (Path(__file__).resolve().parent / ("baseline_mode1.pt" if args.mode == 1 else "baseline_mode2.pt"))
	device = torch.device(args.device)
	train_dataset = Dims2Dataset(data_root, split="train")
	if args.max_samples is not None:
		train_dataset = Subset(train_dataset, range(min(args.max_samples, len(train_dataset))))
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
	model = BaselineCNN().to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	criterion = nn.MSELoss()
	best_loss = float("inf")
	for epoch in range(1, args.epochs + 1):
		loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
		print(f"epoch={epoch} train_loss={loss:.6f}")
		if loss < best_loss:
			best_loss = loss
			checkpoint.parent.mkdir(parents=True, exist_ok=True)
			torch.save({"model_state": model.state_dict(), "args": vars(args)}, checkpoint)
			print(f"saved checkpoint to {checkpoint}")


if __name__ == "__main__":
	main()
