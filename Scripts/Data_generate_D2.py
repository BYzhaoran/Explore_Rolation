#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "Datasets" / "Dims2"
IMAGE_DIR = DATASET_DIR / "images"
LABEL_DIR = DATASET_DIR / "labels"
TEST_DIR = DATASET_DIR / "test"
TEST_LABEL_DIR = DATASET_DIR / "test_labels"

IMAGE_SIZE = 64
DEFAULT_TRAIN_SAMPLES = 10000
DEFAULT_TEST_SAMPLES_PER_ROTATION = 1000
LONG_AXIS_RANGE = (10, 25)
SHORT_AXIS_RANGE = (5, 15)
INTENSITY_RANGE = (180, 255)
DEFAULT_TRAIN_ROTATIONS = (0,)
DEFAULT_TEST_ROTATIONS = (0, 90, 180, 270)
PREVIEW_COUNT = 5
RNG_SEED = 42


@dataclass(frozen=True)
class SampleConfig:
	rotations: tuple[int, ...]
	count: int


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Generate synthetic tactile ellipse datasets.")
	parser.add_argument("--train-samples", type=int, default=DEFAULT_TRAIN_SAMPLES)
	parser.add_argument("--test-samples-per-rotation", type=int, default=DEFAULT_TEST_SAMPLES_PER_ROTATION)
	parser.add_argument(
		"--train-rotations",
		type=int,
		nargs="+",
		default=list(DEFAULT_TRAIN_ROTATIONS),
		help="Rotations allowed in the training split.",
	)
	parser.add_argument(
		"--test-rotations",
		type=int,
		nargs="+",
		default=list(DEFAULT_TEST_ROTATIONS),
		help="Rotations generated for test splits.",
	)
	parser.add_argument("--seed", type=int, default=RNG_SEED)
	parser.add_argument("--no-preview", action="store_true")
	return parser.parse_args()


def ensure_output_dirs(test_rotations: Iterable[int]) -> None:
	DATASET_DIR.mkdir(parents=True, exist_ok=True)
	IMAGE_DIR.mkdir(parents=True, exist_ok=True)
	LABEL_DIR.mkdir(parents=True, exist_ok=True)
	for rotation in test_rotations:
		(TEST_DIR / str(rotation)).mkdir(parents=True, exist_ok=True)
		(TEST_LABEL_DIR / str(rotation)).mkdir(parents=True, exist_ok=True)


def clear_directory_files(directory: Path, pattern: str) -> None:
	for file_path in directory.glob(pattern):
		if file_path.is_file():
			file_path.unlink()


def sample_ellipse_parameters(rng: np.random.Generator) -> tuple[tuple[int, int], int]:
	long_axis = int(rng.integers(LONG_AXIS_RANGE[0], LONG_AXIS_RANGE[1] + 1))
	short_axis = int(rng.integers(SHORT_AXIS_RANGE[0], SHORT_AXIS_RANGE[1] + 1))
	if short_axis >= long_axis:
		short_axis = max(1, long_axis - 1)
	intensity = int(rng.integers(INTENSITY_RANGE[0], INTENSITY_RANGE[1] + 1))
	return (long_axis, short_axis), intensity


def build_ellipse_image(angle_deg: int, axes: tuple[int, int], intensity: int) -> np.ndarray:
	image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
	center = (IMAGE_SIZE // 2, IMAGE_SIZE // 2)
	cv2.ellipse(image, center, axes, angle_deg, 0, 360, intensity, thickness=-1)
	return image


def build_orientation_label(orientation_deg: int) -> np.ndarray:
	theta = np.deg2rad(2.0 * orientation_deg)
	return np.array([[np.cos(theta)], [np.sin(theta)]], dtype=np.float32)


def save_sample(index: int, image: np.ndarray, label: np.ndarray) -> tuple[Path, Path]:
	file_name = f"{index:03d}"
	image_path = IMAGE_DIR / f"{file_name}.png"
	label_path = LABEL_DIR / f"{file_name}.npy"
	cv2.imwrite(str(image_path), image)
	np.save(label_path, label)
	return image_path, label_path


def generate_training_set(num_samples: int, rotations: tuple[int, ...], rng: np.random.Generator) -> list[dict[str, object]]:
	preview_indices = set(rng.choice(np.arange(1, num_samples + 1), size=min(PREVIEW_COUNT, num_samples), replace=False).tolist())
	preview_samples: list[dict[str, object]] = []
	for index in tqdm(range(1, num_samples + 1), desc="Generating training dataset", unit="img"):
		axes, intensity = sample_ellipse_parameters(rng)
		angle_deg = int(rng.choice(rotations))
		image = build_ellipse_image(angle_deg, axes, intensity)
		label = build_orientation_label(angle_deg)
		save_sample(index, image, label)
		if index in preview_indices:
			preview_samples.append({"index": index, "image": image, "angle_deg": angle_deg, "axes": axes, "intensity": intensity})
	preview_samples.sort(key=lambda item: item["index"])
	return preview_samples


def rotate_image(image: np.ndarray, rotation_deg: int) -> np.ndarray:
	center = (IMAGE_SIZE / 2.0, IMAGE_SIZE / 2.0)
	matrix = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
	return cv2.warpAffine(image, matrix, (IMAGE_SIZE, IMAGE_SIZE), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def save_test_sample(rotation_deg: int, index: int, image: np.ndarray, label: np.ndarray) -> tuple[Path, Path]:
	file_name = f"{index:03d}"
	image_path = TEST_DIR / str(rotation_deg) / f"{file_name}.png"
	label_path = TEST_LABEL_DIR / str(rotation_deg) / f"{file_name}.npy"
	cv2.imwrite(str(image_path), image)
	np.save(label_path, label)
	return image_path, label_path


def generate_test_sets(num_samples: int, rotations: tuple[int, ...]) -> None:
	train_image_paths = sorted(IMAGE_DIR.glob("*.png"))[:num_samples]
	for rotation_deg in rotations:
		clear_directory_files(TEST_DIR / str(rotation_deg), "*.png")
		clear_directory_files(TEST_LABEL_DIR / str(rotation_deg), "*.npy")
		for image_path in tqdm(train_image_paths, desc=f"Generating test set {rotation_deg} deg", unit="img"):
			image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
			if image is None:
				raise FileNotFoundError(f"Failed to read training image: {image_path}")
			rotated_image = rotate_image(image, rotation_deg)
			label = build_orientation_label(rotation_deg)
			index = int(image_path.stem)
			save_test_sample(rotation_deg, index, rotated_image, label)


def render_preview(preview_samples: list[dict[str, object]]) -> None:
	if not preview_samples:
		return

	figure, axes_list = plt.subplots(1, len(preview_samples), figsize=(4 * len(preview_samples), 4))
	if len(preview_samples) == 1:
		axes_list = [axes_list]

	center = (IMAGE_SIZE // 2, IMAGE_SIZE // 2)
	for axis, sample in zip(axes_list, preview_samples):
		image = sample["image"]
		angle_deg = int(sample["angle_deg"])
		ellipse_axes = sample["axes"]
		intensity = int(sample["intensity"])
		display_image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
		theta = np.deg2rad(angle_deg)
		arrow_length = int(max(ellipse_axes))
		end_point = (int(round(center[0] + arrow_length * np.cos(theta))), int(round(center[1] + arrow_length * np.sin(theta))))
		cv2.ellipse(display_image_bgr, center, ellipse_axes, angle_deg, 0, 360, (0, 255, 0), 1)
		cv2.arrowedLine(display_image_bgr, center, end_point, (255, 0, 0), 1, tipLength=0.2)
		cv2.putText(display_image_bgr, f"theta={angle_deg} deg", (2, IMAGE_SIZE - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
		display_image = cv2.cvtColor(display_image_bgr, cv2.COLOR_BGR2RGB)
		label = build_orientation_label(angle_deg).reshape(-1)
		axis.imshow(display_image)
		axis.set_title(f"{sample['index']:03d}\nθ={angle_deg}°  v=[{label[0]:.1f}, {label[1]:.1f}]\nI={intensity}")
		axis.axis("off")

	figure.tight_layout()
	preview_path = BASE_DIR / "Datasets" / "Dims2" / "preview_dims2.png"
	figure.savefig(preview_path, dpi=150, bbox_inches="tight")
	plt.show()
	print(f"Preview saved to: {preview_path}")


def main() -> None:
	args = parse_args()
	train_rotations = tuple(args.train_rotations)
	test_rotations = tuple(args.test_rotations)
	ensure_output_dirs(test_rotations)
	clear_directory_files(IMAGE_DIR, "*.png")
	clear_directory_files(LABEL_DIR, "*.npy")
	rng = np.random.default_rng(args.seed)
	preview_samples = generate_training_set(args.train_samples, train_rotations, rng)
	generate_test_sets(args.test_samples_per_rotation, test_rotations)
	if not args.no_preview:
		render_preview(preview_samples)


if __name__ == "__main__":
	main()
