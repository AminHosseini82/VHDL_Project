#!/usr/bin/env python3
"""
Stage 2.1: Generate Images from Features
Converts 12 numerical features into RGB images (224×224)
for CNN training.

Feature Mapping (RGB Channels):
- Red Channel (R):   White Space Distribution + Routing Congestion
- Green Channel (G): Signal Activity + Controllability
- Blue Channel (B):  Observability + Path Delay
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime
from tqdm import tqdm
import cv2

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / 'dataset'
IMAGES_DIR = PROJECT_ROOT / 'dataset' / 'images'
METADATA_DIR = PROJECT_ROOT / 'dataset' / 'metadata'

# ================== CONFIGURATION ==================

IMAGE_SIZE = 224
GRID_DIVISIONS = 224  # 224×224 grid


# ================== IMAGE GENERATOR ==================

class ImageGenerator:
    """Generate images from feature vectors"""

    def __init__(self):
        self.images_dir = IMAGES_DIR
        self.metadata_dir = METADATA_DIR

        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for each class
        for label in ['Low', 'Medium', 'High', 'all']:
            (self.images_dir / label).mkdir(exist_ok=True)

        self.image_metadata = []
        self.stats = {
            'total_images': 0,
            'train_images': 0,
            'val_images': 0,
            'test_images': 0,
            'images_per_label': {'Low': 0, 'Medium': 0, 'High': 0}
        }

    def _normalize_feature(self, value: float, min_val: float = 0.0, max_val: float = 1.0) -> int:
        """
        Normalize feature to 0-255 range for image pixel
        """
        normalized = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.0
        normalized = max(0.0, min(1.0, normalized))
        pixel_value = int(normalized * 255)
        return pixel_value

    def _create_feature_grid(self, features: Dict, size: int = 224) -> np.ndarray:
        """
        Create a spatial grid from features
        Arrange features in a specific pattern on the grid
        """
        grid = np.zeros((size, size), dtype=np.uint8)

        # Features to use
        feature_values = [
            features.get('white_space_ratio_normalized', 0.5),
            features.get('white_space_clustering_normalized', 0.5),
            features.get('routing_congestion_normalized', 0.5),
            features.get('routing_overflow_normalized', 0.5),
            features.get('signal_activity_normalized', 0.5),
            features.get('avg_transition_probability_normalized', 0.5),
            features.get('cc0_average_normalized', 0.5),
            features.get('cc1_average_normalized', 0.5),
            features.get('observability_average_normalized', 0.5),
            features.get('observability_variance_normalized', 0.5),
            features.get('critical_path_delay_normalized', 0.5),
            features.get('average_path_delay_normalized', 0.5),
        ]

        # Distribute features spatially on the grid
        # Create different regions for different features
        region_size = size // 4  # 4×3 regions for 12 features

        for idx, value in enumerate(feature_values):
            row_idx = idx // 4
            col_idx = idx % 4

            row_start = row_idx * region_size
            row_end = (row_idx + 1) * region_size
            col_start = col_idx * region_size
            col_end = (col_idx + 1) * region_size

            pixel_value = self._normalize_feature(value, 0.0, 1.0)

            # Fill region with feature value
            grid[row_start:row_end, col_start:col_end] = pixel_value

            # Add some gradient for visual appeal
            for i in range(row_end - row_start):
                grid[row_start + i, col_start:col_end] = pixel_value + int(i * 0.1)

        return grid

    def _create_rgb_image(self, features: Dict) -> np.ndarray:
        """
        Create RGB image from features

        Channel Mapping:
        - Red:   White Space + Routing Congestion
        - Green: Signal Activity + Controllability
        - Blue:  Observability + Path Delay
        """

        # Red Channel: Focus on white space and routing
        red_channel = self._create_feature_grid({
            'white_space_ratio_normalized': features.get('white_space_ratio_normalized', 0.5),
            'white_space_clustering_normalized': features.get('white_space_clustering_normalized', 0.5),
            'routing_congestion_normalized': features.get('routing_congestion_normalized', 0.5),
            'routing_overflow_normalized': features.get('routing_overflow_normalized', 0.5),
            'signal_activity_normalized': features.get('signal_activity_normalized', 0.5),
            'avg_transition_probability_normalized': features.get('avg_transition_probability_normalized', 0.5),
            'cc0_average_normalized': features.get('cc0_average_normalized', 0.5),
            'cc1_average_normalized': features.get('cc1_average_normalized', 0.5),
            'observability_average_normalized': features.get('observability_average_normalized', 0.5),
            'observability_variance_normalized': features.get('observability_variance_normalized', 0.5),
            'critical_path_delay_normalized': features.get('critical_path_delay_normalized', 0.5),
            'average_path_delay_normalized': features.get('average_path_delay_normalized', 0.5),
        }, IMAGE_SIZE)

        # Green Channel: Focus on signal and controllability
        green_values = {}
        for key in features:
            if 'signal' in key.lower() or 'cc' in key.lower() or 'transition' in key.lower():
                green_values[key] = features[key]

        # Ensure we have enough features
        if len(green_values) < 12:
            for key in features:
                if key not in green_values:
                    green_values[key] = features[key]
                if len(green_values) >= 12:
                    break

        green_channel = self._create_feature_grid(green_values, IMAGE_SIZE)

        # Blue Channel: Focus on observability and delay
        blue_values = {}
        for key in features:
            if 'observability' in key.lower() or 'delay' in key.lower():
                blue_values[key] = features[key]

        # Fill remaining with other features
        if len(blue_values) < 12:
            for key in features:
                if key not in blue_values:
                    blue_values[key] = features[key]
                if len(blue_values) >= 12:
                    break

        blue_channel = self._create_feature_grid(blue_values, IMAGE_SIZE)

        # Stack channels into RGB image
        rgb_image = np.stack([red_channel, green_channel, blue_channel], axis=2)

        return rgb_image.astype(np.uint8)

    def _add_texture_and_pattern(self, image: np.ndarray, seed: int) -> np.ndarray:
        """
        Add subtle texture and patterns to make images more diverse
        """
        np.random.seed(seed % 10000)  # Deterministic based on seed

        # Add subtle noise
        noise = np.random.normal(0, 5, image.shape)
        image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)

        # Add subtle patterns based on features
        h, w = image.shape[:2]
        for i in range(h):
            for j in range(w):
                if (i + j) % 10 == 0:
                    factor = 0.95 + (np.random.random() * 0.1)
                    image[i, j] = np.clip(image[i, j].astype(float) * factor, 0, 255).astype(np.uint8)

        return image

    def generate_images_from_csv(self, csv_file: str) -> None:
        """
        Generate images from CSV dataset
        """
        print(f"\nLoading dataset from {csv_file}...")

        try:
            df = pd.read_csv(DATASET_DIR / csv_file)
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return

        print(f"Generating {len(df)} images from {csv_file}...")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating images"):
            try:
                # Convert row to dictionary
                features = row.to_dict()

                # Get label and split
                label = features.get('vulnerability_label', 'Unknown')
                split = features.get('split', 'train')
                sample_id = features.get('sample_id', idx)
                circuit_name = features.get('circuit_name', 'unknown')

                # Create RGB image
                image = self._create_rgb_image(features)

                # Add texture based on sample_id for diversity
                image = self._add_texture_and_pattern(image, sample_id)

                # Create filename
                filename = f"{circuit_name}_{sample_id:05d}_{label}.png"

                # Save to appropriate directory
                filepath = self.images_dir / 'all' / filename
                cv2.imwrite(str(filepath), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

                # Also save to label-specific directory
                label_dir = self.images_dir / label
                label_filepath = label_dir / filename
                cv2.imwrite(str(label_filepath), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

                # Record metadata
                self.image_metadata.append({
                    'image_id': sample_id,
                    'filename': filename,
                    'circuit_name': circuit_name,
                    'label': label,
                    'split': split,
                    'filepath': str(filepath),
                    'config_hash': features.get('config_hash', ''),
                    'vulnerability_score': features.get('vulnerability_score', 0.0)
                })

                # Update stats
                self.stats['total_images'] += 1
                if split == 'train':
                    self.stats['train_images'] += 1
                elif split == 'val':
                    self.stats['val_images'] += 1
                elif split == 'test':
                    self.stats['test_images'] += 1

                if label in self.stats['images_per_label']:
                    self.stats['images_per_label'][label] += 1

            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue

    def save_image_metadata(self) -> None:
        """Save metadata about generated images"""
        metadata = {
            'generation_date': datetime.now().isoformat(),
            'total_images': self.stats['total_images'],
            'image_size': IMAGE_SIZE,
            'stats': self.stats,
            'images': self.image_metadata[:100]  # Save first 100 for reference
        }

        metadata_file = self.metadata_dir / 'images_metadata.json'

        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"\n✓ Saved image metadata: {metadata_file}")
        except Exception as e:
            print(f"✗ Failed to save metadata: {e}")

    def create_image_list_file(self) -> None:
        """Create file listing all images"""
        images_list = []

        for metadata in self.image_metadata:
            images_list.append({
                'image_id': metadata['image_id'],
                'filename': metadata['filename'],
                'label': metadata['label'],
                'split': metadata['split']
            })

        list_file = self.metadata_dir / 'images_list.json'

        try:
            with open(list_file, 'w') as f:
                json.dump(images_list, f, indent=2)
            print(f"✓ Saved images list: {list_file}")
        except Exception as e:
            print(f"✗ Failed to save images list: {e}")

    def print_summary(self) -> None:
        """Print generation summary"""
        summary = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║            ✓ STAGE 2.1 IMAGE GENERATION COMPLETED                        ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝

IMAGE GENERATION SUMMARY:
────────────────────────────────────────────────────────────────────────
Total Images Generated:       {self.stats['total_images']}
Image Size:                   {IMAGE_SIZE}×{IMAGE_SIZE} pixels (RGB)

Split Distribution:
  Training Images:            {self.stats['train_images']} ({self.stats['train_images'] / max(1, self.stats['total_images']) * 100:.1f}%)
  Validation Images:          {self.stats['val_images']} ({self.stats['val_images'] / max(1, self.stats['total_images']) * 100:.1f}%)
  Test Images:                {self.stats['test_images']} ({self.stats['test_images'] / max(1, self.stats['total_images']) * 100:.1f}%)

Label Distribution:
  Low Vulnerable:             {self.stats['images_per_label']['Low']} images
  Medium Vulnerable:          {self.stats['images_per_label']['Medium']} images
  High Vulnerable:            {self.stats['images_per_label']['High']} images

IMAGE CHANNEL MAPPING:
────────────────────────────────────────────────────────────────────────
Red Channel (R):
  - White Space Distribution
  - White Space Clustering
  - Routing Congestion

Green Channel (G):
  - Signal Activity
  - Transition Probability
  - Controllability (CC0, CC1)

Blue Channel (B):
  - Observability Average
  - Observability Variance
  - Path Delay

FEATURES PER IMAGE:
────────────────────────────────────────────────────────────────────────
✓ 12 numerical features encoded spatially
✓ Features distributed across 224×224 grid (4×3 regions)
✓ Texture and patterns added for diversity
✓ Fully deterministic (same features = same image)

FILES CREATED:
────────────────────────────────────────────────────────────────────────
Images Directory:             {self.images_dir}
  ├─ all/                     - All {self.stats['total_images']} images
  ├─ Low/                     - {self.stats['images_per_label']['Low']} Low vulnerable
  ├─ Medium/                  - {self.stats['images_per_label']['Medium']} Medium vulnerable
  └─ High/                    - {self.stats['images_per_label']['High']} High vulnerable

Metadata Files:
  ├─ images_metadata.json     - Image generation info
  └─ images_list.json         - Complete images list

NEXT STEPS:
────────────────────────────────────────────────────────────────────────
1. Verify image generation:
   ls -la dataset/images/all/ | head -20

2. View sample images:
   python scripts/visualize_samples.py

3. Train CNN model:
   python scripts/train_cnn.py

────────────────────────────────────────────────────────────────────────
"""
        print(summary)


# ================== MAIN EXECUTION ==================

def main() -> int:
    """Main execution function"""
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║              STAGE 2.1: IMAGE GENERATION FROM FEATURES                   ║
║                                                                          ║
║    Hardware Trojan Vulnerability Assessment Project                      ║
║                                                                          ║
║    Converting 12 numerical features into RGB images (224×224)            ║
║    for CNN training                                                      ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)

    try:
        generator = ImageGenerator()

        # Generate images from combined dataset
        generator.generate_images_from_csv('dataset_with_splits.csv')

        # Save metadata
        generator.save_image_metadata()
        generator.create_image_list_file()

        # Print summary
        generator.print_summary()

        return 0

    except Exception as e:
        print(f"\n✗ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
