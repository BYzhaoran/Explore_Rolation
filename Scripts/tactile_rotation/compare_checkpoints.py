from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover
    plt = None

from dataset import Dims2Dataset
from models import BaselineCNN
from equivariant import RotationEquivariantModel


MODEL_SPECS = [
	("baseline_mode1", BaselineCNN, Path(__file__).resolve().parent / "baseline.pt", 1),
	("equivariant_mode1", RotationEquivariantModel, Path(__file__).resolve().parent / "equivariant.pt", 1),
	("baseline_mode2", BaselineCNN, Path(__file__).resolve().parent / "baseline_mode2.pt", 2),
	("equivariant_mode2", RotationEquivariantModel, Path(__file__).resolve().parent / "equivariant_mode2.pt", 2),
]


def angular_error(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
	predictions = torch.nn.functional.normalize(predictions, dim=-1)
	targets = torch.nn.functional.normalize(targets, dim=-1)
	pred_angle = torch.atan2(predictions[:, 1], predictions[:, 0])
	target_angle = torch.atan2(targets[:, 1], targets[:, 0])
	diff = torch.remainder(pred_angle - target_angle + math.pi, 2 * math.pi) - math.pi
	return diff.abs() * 180.0 / math.pi


def evaluate_split(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float | None:
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


def load_model(model_cls, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
	model = model_cls().to(device)
	checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
	model.load_state_dict(checkpoint["model_state"])
	return model


def main() -> None:
	data_root = Path(__file__).resolve().parents[2] / "Datasets" / "Dims2"
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	rotation_root = data_root / "test"
	rotations = sorted(int(path.name) for path in rotation_root.iterdir() if path.is_dir() and path.name.isdigit())

	models = []
	for name, model_cls, checkpoint_path, mode in MODEL_SPECS:
		if not checkpoint_path.exists():
			raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
		models.append((name, load_model(model_cls, checkpoint_path, device), mode))

	rows = []
	series_by_model = {name: [] for name, _, _ in models}
	for rotation in tqdm(rotations, desc="Evaluating rotations", unit="rot"):
		dataset = Dims2Dataset(data_root, split="test", rotation=rotation)
		loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
		for name, model, mode in models:
			print(f"rotation={rotation} model={name}", flush=True)
			metric = evaluate_split(model, loader, device)
			rows.append({"model": name, "mode": mode, "rotation": rotation, "mean_angular_error_deg": metric})
			series_by_model[name].append((rotation, metric))

	df = pd.DataFrame(rows)
	print(df.pivot(index="rotation", columns="model", values="mean_angular_error_deg").round(4).to_string())
	print()
	print(df.groupby(["model", "mode"], as_index=False)["mean_angular_error_deg"].mean().round(4).to_string(index=False))

	output_dir = Path(__file__).resolve().parent / "comparison_results"
	output_dir.mkdir(parents=True, exist_ok=True)

	table_path = output_dir / "comparison_table.csv"
	df.to_csv(table_path, index=False)

	if plt is None:
		raise ModuleNotFoundError("matplotlib is required to generate the comparison plot")

	plt.figure(figsize=(9, 5))
	for name, points in series_by_model.items():
		filtered_points = [(rotation, value) for rotation, value in points if value is not None]
		if not filtered_points:
			continue
		plot_rotations, plot_values = zip(*filtered_points)
		plt.plot(plot_rotations, plot_values, marker="o", linewidth=2, markersize=6, label=name)

	plt.xlabel("Rotation Angle (degrees)")
	plt.ylabel("Direction Prediction Error (degrees)")
	plt.xticks(rotations)
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()

	plot_path = output_dir / "comparison_error_plot.png"
	plt.savefig(plot_path, dpi=150, bbox_inches="tight")
	plt.close()
	print(f"saved table to {table_path}")
	print(f"saved plot to {plot_path}")


if __name__ == "__main__":
	main()
