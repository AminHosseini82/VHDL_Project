#!/usr/bin/env python3
"""
Stage 1.4: Create Dataset and Assign Vulnerability Labels
Creates a comprehensive dataset with:
- Feature vectors for each implementation
- Vulnerability labels (Low, Medium, High)
- Train/Val/Test splits
- Normalization

Output: CSV file with all features and labels
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import random

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = PROJECT_ROOT / 'features'
DATASET_DIR = PROJECT_ROOT / 'dataset'

# ================== CONFIGURATION ==================

VULNERABILITY_CLASSES = ['Low', 'Medium', 'High']
VULNERABILITY_PERCENTILE_HIGH = 75  # Top 25% = High vulnerable
VULNERABILITY_PERCENTILE_LOW = 25  # Bottom 25% = Low vulnerable

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

RANDOM_SEED = 42


# ================== DATASET CREATOR ==================

class DatasetCreator:
    """Create and prepare dataset for training"""

    def __init__(self):
        self.dataset_dir = DATASET_DIR
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        self.features_list = []
        self.vulnerability_scores = []
        self.stats = {
            'total_samples': 0,
            'train_samples': 0,
            'val_samples': 0,
            'test_samples': 0,
            'low_vulnerable': 0,
            'medium_vulnerable': 0,
            'high_vulnerable': 0
        }

        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    def _load_all_features(self) -> List[Dict]:
        """Load all extracted features"""
        print("\n" + "=" * 70)
        print("LOADING ALL EXTRACTED FEATURES")
        print("=" * 70)

        all_features = []
        feature_files = sorted(FEATURES_DIR.glob('*_features.json'))

        for feature_file in feature_files:
            try:
                with open(feature_file, 'r') as f:
                    data = json.load(f)

                circuit_features = data.get('features', [])
                all_features.extend(circuit_features)

                print(f"✓ Loaded {len(circuit_features)} features from {feature_file.name}")

            except Exception as e:
                print(f"✗ Error loading {feature_file}: {e}")

        print(f"\nTotal features loaded: {len(all_features)}")
        return all_features

    def _calculate_vulnerability_score(self, feature: Dict) -> float:
        """
        Calculate vulnerability score based on features
        Higher score = more vulnerable to hardware trojans

        Vulnerability increases with:
        - High routing congestion (harder to detect trojans)
        - High signal activity (trojans can hide)
        - Low observability (trojans undetectable)
        - Low controllability (hard to trigger trojans)
        """

        # Extract key features
        routing_congestion = feature.get('routing_congestion', 0.0)
        signal_activity = feature.get('signal_activity', 0.0)
        observability = feature.get('observability_average', 0.0)
        cc0 = feature.get('cc0_average', 0.5)
        cc1 = feature.get('cc1_average', 0.5)
        white_space = feature.get('white_space_ratio', 0.5)

        # Calculate vulnerability score (0-1, higher = more vulnerable)
        # Congestion: higher congestion = more vulnerable
        congestion_score = routing_congestion

        # Signal activity: higher activity = more vulnerable (trojans hide in activity)
        activity_score = signal_activity * 0.8

        # Observability: lower observability = more vulnerable
        observability_score = (1.0 - observability) * 0.6

        # Controllability: lower controllability = more vulnerable
        controllability = (cc0 + cc1) / 2.0
        controllability_score = (1.0 - controllability) * 0.7

        # White space: less white space = more vulnerable (packed design)
        packing_score = (1.0 - white_space) * 0.5

        # Weighted combination
        vulnerability_score = (
                congestion_score * 0.25 +
                activity_score * 0.20 +
                observability_score * 0.20 +
                controllability_score * 0.25 +
                packing_score * 0.10
        )

        return min(1.0, max(0.0, vulnerability_score))

    def _assign_vulnerability_label(self, score: float) -> str:
        """
        Assign vulnerability class based on score percentile

        Args:
            score: Vulnerability score (0-1)

        Returns:
            'Low', 'Medium', or 'High'
        """
        # This will be refined with actual percentiles
        if score < 0.33:
            return 'Low'
        elif score < 0.66:
            return 'Medium'
        else:
            return 'High'

    def create_feature_dataframe(self, features: List[Dict]) -> pd.DataFrame:
        """
        Create pandas DataFrame from features
        """
        print("\n" + "=" * 70)
        print("CREATING FEATURE DATAFRAME")
        print("=" * 70)

        data_rows = []

        for idx, feature in enumerate(features):
            # Calculate vulnerability score
            vuln_score = self._calculate_vulnerability_score(feature)

            # Create row
            row = {
                'sample_id': idx,
                'circuit_name': feature.get('circuit_name', ''),
                'implementation_id': feature.get('implementation_id', 0),
                'config_hash': feature.get('config_hash', ''),

                # Features
                'white_space_ratio': feature.get('white_space_ratio', 0.0),
                'white_space_clustering': feature.get('white_space_clustering', 0.0),
                'routing_congestion': feature.get('routing_congestion', 0.0),
                'routing_overflow': feature.get('routing_overflow', 0.0),
                'signal_activity': feature.get('signal_activity', 0.0),
                'avg_transition_probability': feature.get('avg_transition_probability', 0.0),
                'cc0_average': feature.get('cc0_average', 0.0),
                'cc1_average': feature.get('cc1_average', 0.0),
                'observability_average': feature.get('observability_average', 0.0),
                'observability_variance': feature.get('observability_variance', 0.0),
                'critical_path_delay': feature.get('critical_path_delay', 0.0),
                'average_path_delay': feature.get('average_path_delay', 0.0),

                # Vulnerability score and label
                'vulnerability_score': vuln_score,
                'vulnerability_label': self._assign_vulnerability_label(vuln_score)
            }

            data_rows.append(row)
            self.vulnerability_scores.append(vuln_score)

        df = pd.DataFrame(data_rows)
        print(f"✓ Created dataframe with {len(df)} samples")

        return df

    def assign_percentile_based_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reassign labels based on percentiles for balanced distribution
        """
        print("\n" + "=" * 70)
        print("ASSIGNING PERCENTILE-BASED VULNERABILITY LABELS")
        print("=" * 70)

        # Calculate percentiles
        scores = df['vulnerability_score'].values
        low_threshold = np.percentile(scores, VULNERABILITY_PERCENTILE_LOW)
        high_threshold = np.percentile(scores, VULNERABILITY_PERCENTILE_HIGH)

        print(f"\nPercentile thresholds:")
        print(f"  Low threshold (25th percentile):  {low_threshold:.4f}")
        print(f"  High threshold (75th percentile): {high_threshold:.4f}")

        # Assign labels
        def assign_label(score):
            if score <= low_threshold:
                return 'Low'
            elif score >= high_threshold:
                return 'High'
            else:
                return 'Medium'

        df['vulnerability_label'] = df['vulnerability_score'].apply(assign_label)

        # Count labels
        label_counts = df['vulnerability_label'].value_counts()
        print(f"\nLabel distribution:")
        for label in VULNERABILITY_CLASSES:
            count = label_counts.get(label, 0)
            percentage = (count / len(df)) * 100
            print(f"  {label:10s}: {count:5d} ({percentage:5.1f}%)")
            self.stats[f'{label.lower()}_vulnerable'] = int(count)

        return df

    def normalize_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize numerical features to 0-1 range
        """
        print("\n" + "=" * 70)
        print("NORMALIZING FEATURES")
        print("=" * 70)

        feature_columns = [
            'white_space_ratio', 'white_space_clustering',
            'routing_congestion', 'routing_overflow',
            'signal_activity', 'avg_transition_probability',
            'cc0_average', 'cc1_average',
            'observability_average', 'observability_variance',
            'critical_path_delay', 'average_path_delay',
            'vulnerability_score'
        ]

        normalization_params = {}

        for col in feature_columns:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()

                # Avoid division by zero
                if max_val - min_val > 0:
                    df[f'{col}_normalized'] = (df[col] - min_val) / (max_val - min_val)
                else:
                    df[f'{col}_normalized'] = 0.0

                normalization_params[col] = {
                    'min': float(min_val),
                    'max': float(max_val),
                    'range': float(max_val - min_val)
                }

        print(f"✓ Normalized {len(feature_columns)} features")

        return df, normalization_params

    def create_train_val_test_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train/val/test with stratification
        """
        print("\n" + "=" * 70)
        print("CREATING TRAIN/VAL/TEST SPLIT")
        print("=" * 70)

        # Stratified split by vulnerability label
        from sklearn.model_selection import train_test_split

        # First split: train (70%) vs temp (30%)
        train, temp = train_test_split(
            df,
            test_size=VAL_SPLIT + TEST_SPLIT,
            random_state=RANDOM_SEED,
            stratify=df['vulnerability_label']
        )

        # Second split: val (50% of 30% = 15%) vs test (50% of 30% = 15%)
        val, test = train_test_split(
            temp,
            test_size=0.5,
            random_state=RANDOM_SEED,
            stratify=temp['vulnerability_label']
        )

        # Reset indices
        train = train.reset_index(drop=True)
        val = val.reset_index(drop=True)
        test = test.reset_index(drop=True)

        # Add split column
        train['split'] = 'train'
        val['split'] = 'val'
        test['split'] = 'test'

        print(f"\nSplit distribution:")
        print(f"  Train: {len(train)} samples ({len(train) / len(df) * 100:.1f}%)")
        print(f"  Val:   {len(val)} samples ({len(val) / len(df) * 100:.1f}%)")
        print(f"  Test:  {len(test)} samples ({len(test) / len(df) * 100:.1f}%)")

        self.stats['train_samples'] = len(train)
        self.stats['val_samples'] = len(val)
        self.stats['test_samples'] = len(test)

        # Print stratification info
        print(f"\nLabel distribution per split:")
        for split_name, split_df in [('train', train), ('val', val), ('test', test)]:
            label_dist = split_df['vulnerability_label'].value_counts()
            print(f"\n  {split_name.upper()}:")
            for label in VULNERABILITY_CLASSES:
                count = label_dist.get(label, 0)
                pct = (count / len(split_df)) * 100 if len(split_df) > 0 else 0
                print(f"    {label:10s}: {count:4d} ({pct:5.1f}%)")

        return train, val, test

    def save_dataset(self, df: pd.DataFrame, filename: str) -> None:
        """Save dataset to CSV"""
        filepath = self.dataset_dir / filename

        try:
            df.to_csv(filepath, index=False)
            print(f"✓ Saved dataset: {filepath}")
            print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
        except Exception as e:
            print(f"✗ Failed to save dataset: {e}")

    def save_split_metadata(self, df_full: pd.DataFrame) -> None:
        """Save metadata about dataset splits"""
        split_metadata = {
            'creation_date': datetime.now().isoformat(),
            'total_samples': len(df_full),
            'splits': {
                'train': {
                    'count': self.stats['train_samples'],
                    'percentage': self.stats['train_samples'] / len(df_full) * 100
                },
                'val': {
                    'count': self.stats['val_samples'],
                    'percentage': self.stats['val_samples'] / len(df_full) * 100
                },
                'test': {
                    'count': self.stats['test_samples'],
                    'percentage': self.stats['test_samples'] / len(df_full) * 100
                }
            },
            'labels': {
                'Low': {
                    'count': self.stats['low_vulnerable'],
                    'percentage': self.stats['low_vulnerable'] / len(df_full) * 100
                },
                'Medium': {
                    'count': self.stats['medium_vulnerable'],
                    'percentage': self.stats['medium_vulnerable'] / len(df_full) * 100
                },
                'High': {
                    'count': self.stats['high_vulnerable'],
                    'percentage': self.stats['high_vulnerable'] / len(df_full) * 100
                }
            }
        }

        metadata_file = self.dataset_dir / 'split_metadata.json'

        try:
            with open(metadata_file, 'w') as f:
                json.dump(split_metadata, f, indent=2)
            print(f"✓ Saved split metadata: {metadata_file}")
        except Exception as e:
            print(f"✗ Failed to save metadata: {e}")

    def create_full_dataset(self) -> None:
        """Create complete dataset"""
        print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║              STAGE 1.4: DATASET CREATION AND LABELING                    ║
║                                                                          ║
║    Hardware Trojan Vulnerability Assessment Project                      ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
        """)

        # Load features
        features = self._load_all_features()
        self.stats['total_samples'] = len(features)

        # Create dataframe
        df = self.create_feature_dataframe(features)

        # Assign percentile-based labels
        df = self.assign_percentile_based_labels(df)

        # Normalize features
        df, norm_params = self.normalize_features(df)

        # Save full dataset
        self.save_dataset(df, 'dataset_complete.csv')

        # Create train/val/test split
        train, val, test = self.create_train_val_test_split(df)

        # Save splits
        self.save_dataset(train, 'dataset_train.csv')
        self.save_dataset(val, 'dataset_val.csv')
        self.save_dataset(test, 'dataset_test.csv')

        # Save combined split file
        combined = pd.concat([train, val, test], ignore_index=True)
        self.save_dataset(combined, 'dataset_with_splits.csv')

        # Save metadata
        self.save_split_metadata(df)

        # Print summary
        self.print_summary(norm_params)

    def print_summary(self, norm_params: Dict) -> None:
        """Print dataset creation summary"""
        summary = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║            ✓ STAGE 1.4 DATASET CREATION COMPLETED                        ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝

DATASET SUMMARY:
────────────────────────────────────────────────────────────────────────
Total Samples:                {self.stats['total_samples']}

Train/Val/Test Split:
  Train:                      {self.stats['train_samples']} ({self.stats['train_samples'] / self.stats['total_samples'] * 100:.1f}%)
  Validation:                 {self.stats['val_samples']} ({self.stats['val_samples'] / self.stats['total_samples'] * 100:.1f}%)
  Test:                       {self.stats['test_samples']} ({self.stats['test_samples'] / self.stats['total_samples'] * 100:.1f}%)

VULNERABILITY LABEL DISTRIBUTION:
────────────────────────────────────────────────────────────────────────
Low Vulnerable:               {self.stats['low_vulnerable']} ({self.stats['low_vulnerable'] / self.stats['total_samples'] * 100:.1f}%)
Medium Vulnerable:            {self.stats['medium_vulnerable']} ({self.stats['medium_vulnerable'] / self.stats['total_samples'] * 100:.1f}%)
High Vulnerable:              {self.stats['high_vulnerable']} ({self.stats['high_vulnerable'] / self.stats['total_samples'] * 100:.1f}%)

FILES CREATED:
────────────────────────────────────────────────────────────────────────
dataset_complete.csv          - Full dataset with all features
dataset_train.csv             - Training set ({self.stats['train_samples']} samples)
dataset_val.csv               - Validation set ({self.stats['val_samples']} samples)
dataset_test.csv              - Test set ({self.stats['test_samples']} samples)
dataset_with_splits.csv       - Combined with split column
split_metadata.json           - Split information

Location: {self.dataset_dir}

FEATURES INCLUDED:
────────────────────────────────────────────────────────────────────────
✓ White Space Ratio
✓ White Space Clustering
✓ Routing Congestion
✓ Routing Overflow
✓ Signal Activity
✓ Transition Probability
✓ CC0 Average (Controllability)
✓ CC1 Average (Controllability)
✓ Observability Average
✓ Observability Variance
✓ Critical Path Delay
✓ Average Path Delay
✓ Vulnerability Score (0-1)
✓ Vulnerability Label (Low/Medium/High)

NORMALIZATION:
────────────────────────────────────────────────────────────────────────
✓ All numerical features normalized to 0-1 range
✓ Normalization parameters saved

NEXT STEPS:
────────────────────────────────────────────────────────────────────────
1. Inspect dataset:
   head -20 dataset/dataset_complete.csv

2. Check data statistics:
   python -c "import pandas as pd; df = pd.read_csv('dataset/dataset_complete.csv'); print(df.describe())"

3. Proceed to Stage 2: Model Training
   python scripts/train_model.py

────────────────────────────────────────────────────────────────────────
"""
        print(summary)


# ================== MAIN EXECUTION ==================

def main() -> int:
    """Main execution function"""
    try:
        creator = DatasetCreator()
        creator.create_full_dataset()
        return 0
    except Exception as e:
        print(f"\n✗ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
