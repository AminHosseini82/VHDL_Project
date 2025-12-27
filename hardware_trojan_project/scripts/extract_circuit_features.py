#!/usr/bin/env python3
"""
Stage 1.3: Circuit Feature Extraction (Simulated)
Converts implementation metadata into numerical features for the dataset.
"""

import os
import json
import random
import numpy as np
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMPLEMENTATIONS_DIR = PROJECT_ROOT / 'implementations' / 'metadata'
FEATURES_DIR = PROJECT_ROOT / 'features'

def extract_features():
    print("STAGE 1.3: EXTRACTING NUMERICAL FEATURES FROM METADATA")
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get all metadata files
    metadata_files = list(IMPLEMENTATIONS_DIR.glob('*_implementations.json'))
    
    if not metadata_files:
        print("❌ No metadata files found! Run generate_implementations.py first.")
        return

    total_features = 0
    
    for meta_file in metadata_files:
        circuit_name = meta_file.stem.replace('_implementations', '')
        print(f"Processing {circuit_name}...")
        
        with open(meta_file, 'r') as f:
            data = json.load(f)
            
        circuit_features = []
        implementations = data.get('implementations', [])
        
        for impl in implementations:
            # Simulate features based on implementation parameters
            # In a real tool flow, these would come from reports
            
            # 1. Congestion (correlated with area factor - smaller area = more congestion)
            area_factor = impl.get('area_factor', 1.0)
            base_congestion = 0.9 - (area_factor * 0.08) # 1x -> ~0.82, 10x -> ~0.1
            congestion = max(0.1, min(1.0, base_congestion + random.gauss(0, 0.05)))
            
            # 2. Signal Activity (random variation based on gate style)
            activity = random.uniform(0.1, 0.5)
            
            # 3. White Space (correlated with area)
            white_space = 1.0 - (1.0 / area_factor) # 1x -> 0, 10x -> 0.9
            white_space = max(0.05, min(0.95, white_space + random.gauss(0, 0.02)))
            
            # 4. Controllability/Observability (randomized for simulation)
            cc0 = random.uniform(0.3, 0.9)
            cc1 = random.uniform(0.3, 0.9)
            obs = random.uniform(0.3, 0.9)
            
            feat = {
                'circuit_name': circuit_name,
                'implementation_id': impl.get('implementation_id'),
                'area_factor': area_factor,
                
                # Extracted Features
                'routing_congestion': congestion,
                'signal_activity': activity,
                'white_space_ratio': white_space,
                'cc0_average': cc0,
                'cc1_average': cc1,
                'observability_average': obs,
                
                # Other metrics
                'critical_path_delay': impl.get('estimated_delay_ns', 0),
                'power_consumption': impl.get('estimated_power_mw', 0)
            }
            circuit_features.append(feat)
            
        # Save features
        output_file = FEATURES_DIR / f"{circuit_name}_features.json"
        with open(output_file, 'w') as f:
            json.dump({'features': circuit_features}, f, indent=2)
            
        total_features += len(circuit_features)
        
    print(f"\n✅ Feature extraction complete.")
    print(f"   Total features generated: {total_features}")
    print(f"   Saved to: {FEATURES_DIR}")

if __name__ == "__main__":
    extract_features()
