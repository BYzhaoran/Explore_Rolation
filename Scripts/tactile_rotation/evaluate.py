from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset import Dims2Dataset
from models import BaselineCNN
from equivariant import RotationEquivariantModel


MODEL_REGISTRY = {
	"baseline": BaselineCNN,
	"equivariant": RotationEquivariantModel,
}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate tactile direction models.")
	parser.add_argument("--mode", type=int, choices=(1, 2), default=1, help="1 uses Dims2, 2 uses Dims2_Mul.")
	parser.add_argument("--data-root", type=Path, default=None)
	parser.add_argument("--checkpoint", type=Path, required=True)
	parser.add_argument("--model", choices=MODEL_REGISTRY.keys(), required=True)
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	return parser.parse_args()


def angular_error(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
	predictions = torch.nn.functional.normalize(predictions, dim=-1)
	targets = torch.nn.functional.normalize(targets, dim=-1)
	pred_angle = torch.atan2(predictions[:, 1], predictions[:, 0])
	target_angle = torch.atan2(targets[:, 1], targets[:, 0])
	diff = torch.remainder(pred_angle - target_angle + math.pi, 2 * math.pi) - math.pi
	return diff.abs() * 180.0 / math.pi


def evaluate_split(model, loader, device):
	model.eval()
	all_errors = []
	with torch.no_grad():
		for images, labels, _ in loader:
			images = images.to(device)
			labels = labels.to(device)
			predictions = model(images)
			all_errors.append(angular_error(predictions, labels))
	if not all_errors:
		return None
	return torch.cat(all_errors).mean().item()


def main() -> None:
	args = parse_args()
	data_root = args.data_root or (Path(__file__).resolve().parents[2] / "Datasets" / ("Dims2" if args.mode == 1 else "Dims2_Mul"))
	device = torch.device(args.device)
	model = MODEL_REGISTRY[args.model]().to(device)
	checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
	model.load_state_dict(checkpoint["model_state"])
	rotation_root = data_root / "test"
	rotations = sorted(int(path.name) for path in rotation_root.iterdir() if path.is_dir() and path.name.isdigit())
	errors = []
	for rotation in rotations:
		dataset = Dims2Dataset(data_root, split="test", rotation=rotation)
		loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
		metric = evaluate_split(model, loader, device)
		if metric is None:
			print(f"rotation={rotation} skipped empty split")
			continue
		errors.append((rotation, metric))
		print(f"rotation={rotation} mean_angular_error_deg={metric:.4f}")

	output_dir = Path(__file__).resolve().parent / f"{args.model}_mode{args.mode}"
	output_dir.mkdir(parents=True, exist_ok=True)

	if errors:
		rotations, values = zip(*errors)
		plt.figure(figsize=(8, 5))
		plt.plot(rotations, values, marker='o', linewidth=2, markersize=8)
		plt.xlabel('Rotation Angle (degrees)', fontsize=12)
		plt.ylabel('Mean Angular Error (degrees)', fontsize=12)
		plt.title(f'{args.model.capitalize()} Model - Direction Prediction Error', fontsize=14)
		plt.grid(True, alpha=0.3)
		plt.xticks(rotations)

		plot_path = output_dir / 'error_comparison.png'
		plt.savefig(plot_path, dpi=150, bbox_inches='tight')
		print(f"saved error comparison plot to {plot_path}")
		plt.close()


if __name__ == "__main__":
	main()
