from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from lab_utils.visualization import plot_feature_vector, show_image_gallery
LABELS = ('cat', 'dog')
LABEL_TO_INDEX = {'cat': 0, 'dog': 1}
IMAGE_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
SEED = 1234

def label_from_path(path: Path) -> str:
    label = path.parent.name
    if label not in LABEL_TO_INDEX:
        raise ValueError(f'Unexpected label folder: {path}')
    return label

def load_preview_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'))

def list_image_paths(label: str) -> list[Path]:
    label_dir = DATA_ROOT / label
    paths = []
    for pattern in IMAGE_EXTENSIONS:
        paths.extend(label_dir.glob(pattern))
    return sorted(paths)

def shuffled_paths(paths: list[Path], seed_offset: int=0) -> list[Path]:
    rng = np.random.default_rng(SEED + seed_offset)
    indices = rng.permutation(len(paths))
    return [paths[int(idx)] for idx in indices]

def sample_paths(paths: list[Path], count: int, seed_offset: int) -> list[Path]:
    ordered = shuffled_paths(paths, seed_offset=seed_offset)
    return ordered[:min(count, len(ordered))]

def sample_per_class(paths: list[Path], n_per_class: int, seed_offset: int=0) -> list[Path]:
    sampled = []
    for label_index, label in enumerate(LABELS):
        label_paths = [path for path in paths if label_from_path(path) == label]
        sampled.extend(sample_paths(label_paths, n_per_class, seed_offset + 50 * label_index))
    return sampled

def split_train_test(paths: list[Path], train_ratio: float=0.7, seed_offset: int=0):
    shuffled = shuffled_paths(paths, seed_offset)
    split_idx = int(len(shuffled) * train_ratio)
    return (shuffled[:split_idx], shuffled[split_idx:])

def load_image_np(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'))
    raise NotImplementedError('Load one RGB image into a NumPy array.')

def center_crop(image: np.ndarray, crop_size: int=48) -> np.ndarray:
    x, y, _ = image.shape
    center_x = x // 2
    center_y = y // 2
    half = crop_size // 2
    start_x = center_x - half
    start_y = center_y - half
    crop = image[start_x:start_x + crop_size, start_y:start_y + crop_size]
    return crop
    raise NotImplementedError('Implement a centered crop with slicing.')

def flip_horizontal(image: np.ndarray) -> np.ndarray:
    flip = image[:, ::-1]
    return flip
    raise NotImplementedError('Flip the image horizontally with slicing.')

def normalize_01(image: np.ndarray) -> np.ndarray:
    normalize_image = image.astype(np.float32) / 255
    return normalize_image
    raise NotImplementedError('Normalize pixel values to [0, 1].')

def show_histograms(uint8_img, float_img):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(uint8_img.ravel(), bins=50)
    plt.title('Before (uint8: 0–255)')
    plt.subplot(1, 2, 2)
    plt.hist(float_img.ravel(), bins=50)
    plt.title('After (float: 0–1)')
    plt.tight_layout()
    plt.show()

def rgb_to_gray(image_float: np.ndarray) -> np.ndarray:
    gray = image_float[:, :, 0] * 0.299
    gray += image_float[:, :, 1] * 0.587
    gray += image_float[:, :, 2] * 0.114
    return gray
    raise NotImplementedError('Convert RGB to grayscale.')

def channel_summary(image_float: np.ndarray) -> tuple[np.ndarray, int]:
    mean_color = image_float.mean(axis=(0, 1))
    return (mean_color, np.argmax(mean_color))
    raise NotImplementedError('Summarize the RGB channels with axis=(0, 1).')

def convolve2d_matmul(image_gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kernel_flat = kernel.ravel()
    height, width = image_gray.shape
    height_k, width_k = kernel.shape
    loop_height = height - height_k + 1
    loop_width = width - width_k + 1
    output = np.zeros((loop_height, loop_width))
    for y in range(0, loop_height):
        for x in range(0, loop_width):
            patch = image_gray[y:y + height_k, x:x + width_k]
            patch_flat = patch.ravel()
            pixel_value = patch_flat @ kernel_flat
            output[y, x] = pixel_value
    return output
    raise NotImplementedError('Apply a 2D filter with matrix multiplication.')

def flatten_image(image: np.ndarray) -> np.ndarray:
    image_flat = image.ravel()
    return image_flat
    raise NotImplementedError('Flatten the image into one vector.')
FEATURE_NAMES = ['mean_r', 'mean_g', 'mean_b', 'std_r', 'std_g', 'std_b', 'brightest_channel', 'edge_mean', 'edge_std', 'row_std_mean']

def extract_features(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    cropped = center_crop(image, crop_size=48)
    image_float = normalize_01(cropped)
    gray = rgb_to_gray(image_float)
    channel_means, brightest_channel = channel_summary(image_float)
    channel_stds = image_float.std(axis=(0, 1)).astype(np.float32)
    filtered = convolve2d_matmul(gray, kernel)
    edge_mean = np.mean(filtered)
    edge_std = np.std(filtered)
    row_std_profile = np.apply_along_axis(np.std, 1, gray)
    row_std_mean = np.mean(row_std_profile)
    feature_vector = np.concatenate([channel_means, channel_stds, [brightest_channel], [edge_mean], [edge_std], [row_std_mean]]).astype(np.float32)
    return feature_vector
    raise NotImplementedError('Build one hand-crafted feature vector.')

def build_feature_matrix(paths: list[Path], kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    all_features = []
    all_labels = []
    for element in paths:
        img = plt.imread(element)
        feature = extract_features(img, kernel)
        all_features.append(feature)
        all_labels.append(LABEL_TO_INDEX[label_from_path(element)])
    x = np.array(all_features).astype(np.float32)
    y = np.array(all_labels).astype(np.int64)
    return (x, y)
    raise NotImplementedError('Build the feature matrix for the dataset subset.')
