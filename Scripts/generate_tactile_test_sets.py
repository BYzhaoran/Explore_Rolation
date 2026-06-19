#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent.parent
DIMS2_DIR = BASE_DIR / "Datasets" / "Dims2"
DIMS2_MUL_DIR = BASE_DIR / "Datasets" / "Dims2_Mul"
IMAGE_SIZE = 64
DEFAULT_ROTATIONS = tuple(range(0, 360, 45))
DEFAULT_SAMPLES_PER_ROTATION = 1000
LONG_AXIS_RANGE = (10, 25)
SHORT_AXIS_RANGE = (5, 15)
INTENSITY_RANGE = (180, 255)
RNG_SEED = 42


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Regenerate tactile test datasets.")
	parser.add_argument("--rotations", type=int, nargs="+", default=list(DEFAULT_ROTATIONS))
	parser.add_argument("--samples-per-rotation", type=int, default=DEFAULT_SAMPLES_PER_ROTATION)
	parser.add_argument("--dataset", choices=("dims2", "dims2_mul", "both"), default="both")
	parser.add_argument("--seed", type=int, default=RNG_SEED)
	return parser.parse_args()


def clear_test_dirs(dataset_dir: Path) -> None:
	for name in ("test", "test_labels"):
		target = dataset_dir / name
		if target.exists():
			shutil.rmtree(target)


def ensure_test_dirs(dataset_dir: Path, rotations: tuple[int, ...]) -> None:
	(dataset_dir / "test").mkdir(parents=True, exist_ok=True)
	(dataset_dir / "test_labels").mkdir(parents=True, exist_ok=True)
	for rotation in rotations:
		(dataset_dir / "test" / str(rotation)).mkdir(parents=True, exist_ok=True)
		(dataset_dir / "test_labels" / str(rotation)).mkdir(parents=True, exist_ok=True)


def sample_ellipse_parameters(rng: np.random.Generator) -> tuple[tuple[int, int], int]:
	long_axis = int(rng.integers(LONG_AXIS_RANGE[0], LONG_AXIS_RANGE[1] + 1))
	short_axis = int(rng.integers(SHORT_AXIS_RANGE[0], SHORT_AXIS_RANGE[1] + 1))
	if short_axis >= long_axis:
		short_axis = max(1, long_axis - 1)
	intensity = int(rng.integers(INTENSITY_RANGE[0], INTENSITY_RANGE[1] + 1))
	return (long_axis, short_axis), intensity


def build_ellipse_image(angle_deg: float, axes: tuple[int, int], intensity: int) -> np.ndarray:
	image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
	center = (IMAGE_SIZE // 2, IMAGE_SIZE // 2)
	cv2.ellipse(image, center, axes, angle_deg, 0, 360, intensity, thickness=-1)
	return image


def build_orientation_label(angle_deg: float) -> np.ndarray:
	theta = np.deg2rad(2.0 * angle_deg)
	return np.array([[np.cos(theta)], [np.sin(theta)]], dtype=np.float32)


def save_sample(image_path: Path, label_path: Path, image: np.ndarray, label: np.ndarray) -> None:
	cv2.imwrite(str(image_path), image)
	np.save(label_path, label)


def generate_dims2_test_set(dataset_dir: Path, num_samples: int, rotations: tuple[int, ...]) -> None:
	image_paths = sorted((dataset_dir / "images").glob("*.png"))[:num_samples]
	if not image_paths:
		raise FileNotFoundError(f"No training images found in {dataset_dir / 'images'}")

	for rotation in rotations:
		for image_path in tqdm(image_paths, desc=f"Dims2 test {rotation}°", unit="img"):
			image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
			if image is None:
				raise FileNotFoundError(f"Failed to read training image: {image_path}")
			center = (IMAGE_SIZE / 2.0, IMAGE_SIZE / 2.0)
			matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
			rotated_image = cv2.warpAffine(
				image,
				matrix,
				(IMAGE_SIZE, IMAGE_SIZE),
				flags=cv2.INTER_NEAREST,
				borderMode=cv2.BORDER_CONSTANT,
				borderValue=0,
			)
			label = build_orientation_label(float(rotation))
			stem = image_path.stem
			save_sample(dataset_dir / "test" / str(rotation) / f"{stem}.png", dataset_dir / "test_labels" / str(rotation) / f"{stem}.npy", rotated_image, label)


def generate_dims2_mul_test_set(dataset_dir: Path, num_samples: int, rotations: tuple[int, ...], rng: np.random.Generator) -> None:
	for rotation in rotations:
		for index in tqdm(range(1, num_samples + 1), desc=f"Dims2_Mul test {rotation}°", unit="img"):
			axes, intensity = sample_ellipse_parameters(rng)
			image = build_ellipse_image(float(rotation), axes, intensity)
			label = build_orientation_label(float(rotation))
			stem = f"{index:03d}"
			save_sample(dataset_dir / "test" / str(rotation) / f"{stem}.png", dataset_dir / "test_labels" / str(rotation) / f"{stem}.npy", image, label)


def main() -> None:
	args = parse_args()
	rotations = tuple(args.rotations)
	selected_datasets = []
	if args.dataset in ("dims2", "both"):
		selected_datasets.append(("dims2", DIMS2_DIR))
	if args.dataset in ("dims2_mul", "both"):
		selected_datasets.append(("dims2_mul", DIMS2_MUL_DIR))

	for _, dataset_dir in selected_datasets:
		clear_test_dirs(dataset_dir)
		ensure_test_dirs(dataset_dir, rotations)

	if args.dataset in ("dims2", "both"):
		generate_dims2_test_set(DIMS2_DIR, args.samples_per_rotation, rotations)

	if args.dataset in ("dims2_mul", "both"):
		test_rng = np.random.default_rng(args.seed)
		generate_dims2_mul_test_set(DIMS2_MUL_DIR, args.samples_per_rotation, rotations, test_rng)


if __name__ == "__main__":
	main()
