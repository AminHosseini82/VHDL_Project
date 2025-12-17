#!/usr/bin/env python3
"""
Stage 1.2: Generate Circuit Implementations
Generates 400 different implementations per circuit with varying parameters:
- Chip area: 1x to 10x minimum area (10 variants)
- Placement modes: 4 strategies
- Gate design styles: 5 types
Total: 10 × 4 × 5 = 200 base combinations × 2 repetitions = 400 per circuit
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import random

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = PROJECT_ROOT / 'benchmarks'
IMPLEMENTATIONS_DIR = PROJECT_ROOT / 'implementations'
METADATA_DIR = IMPLEMENTATIONS_DIR / 'metadata'

# ================== CONFIGURATION ==================

# Area factors: 1x to 10x minimum area
AREA_FACTORS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Placement modes
PLACEMENT_MODES = [
    'standard',
    'timing_driven',
    'congestion_driven',
    'power_driven'
]

# Gate design styles
GATE_DESIGN_STYLES = [
    'complementary_logic',
    'ratioed_logic',
    'transmission_gate_logic',
    'pass_transistor_logic',
    'dynamic_logic'
]

# Technology parameters
TECHNOLOGY_NODE = 90  # nm
SUPPLY_VOLTAGE = 1.2  # V
MIN_GATE_LENGTH = 90  # nm


# ================== DATA STRUCTURES ==================

@dataclass
class ImplementationConfig:
    """Configuration for a single implementation"""
    circuit_name: str
    implementation_id: int
    area_factor: float
    placement_mode: str
    gate_design_style: str
    repetition: int  # 1 or 2

    # Derived properties
    chip_area_mm2: float = 0.0
    chip_width_um: float = 0.0
    chip_height_um: float = 0.0
    estimated_gates: int = 0
    estimated_power_mw: float = 0.0
    estimated_delay_ns: float = 0.0

    # Metadata
    timestamp: str = ""
    config_hash: str = ""


@dataclass
class CircuitInfo:
    """Information about base circuit"""
    name: str
    type: str  # 'combinational' or 'sequential'
    gates: int
    inputs: int
    outputs: int
    flip_flops: int = 0
    min_area_mm2: float = 0.0


# ================== IMPLEMENTATION GENERATOR ==================

class ImplementationGenerator:
    """Generate 400 implementations for each circuit"""

    def __init__(self):
        self.implementations = []
        self.circuits_info = self._load_circuits_info()

        # Create directories
        METADATA_DIR.mkdir(parents=True, exist_ok=True)

        self.stats = {
            'total_implementations': 0,
            'circuits_processed': 0,
            'combinations': len(AREA_FACTORS) * len(PLACEMENT_MODES) * len(GATE_DESIGN_STYLES)
        }

    def _load_circuits_info(self) -> Dict[str, CircuitInfo]:
        """Load circuit information from validation report"""
        validation_file = BENCHMARKS_DIR / 'validation_report.json'
        circuits = {}

        try:
            with open(validation_file, 'r') as f:
                data = json.load(f)

            # Process ISCAS-85 circuits
            for filename, info in data.get('iscas85_circuits', {}).items():
                circuit_name = filename.replace('.v', '')
                circuits[circuit_name] = CircuitInfo(
                    name=circuit_name,
                    type='combinational',
                    gates=info.get('gates_count', 100),
                    inputs=info.get('num_inputs', 5),
                    outputs=info.get('num_outputs', 2),
                    flip_flops=0,
                    min_area_mm2=self._estimate_min_area(info.get('gates_count', 100))
                )

            # Process ISCAS-89 circuits
            for filename, info in data.get('iscas89_circuits', {}).items():
                circuit_name = filename.replace('.v', '')
                circuits[circuit_name] = CircuitInfo(
                    name=circuit_name,
                    type='sequential',
                    gates=info.get('gates_count', 100),
                    inputs=info.get('num_inputs', 4),
                    outputs=info.get('num_outputs', 1),
                    flip_flops=info.get('flip_flops_count', 3),
                    min_area_mm2=self._estimate_min_area(
                        info.get('gates_count', 100),
                        info.get('flip_flops_count', 3)
                    )
                )

            return circuits

        except Exception as e:
            print(f"Warning: Could not load validation report: {e}")
            return {}

    def _estimate_min_area(self, gates: int, flip_flops: int = 0) -> float:
        """
        Estimate minimum chip area based on gate count
        Assumption: ~10-20 µm² per gate in 90nm technology
        """
        gate_area_um2 = 15  # Average area per gate
        ff_area_um2 = 50  # Average area per flip-flop

        total_area_um2 = (gates * gate_area_um2) + (flip_flops * ff_area_um2)
        total_area_mm2 = total_area_um2 / 1e6  # Convert to mm²

        # Add 50% overhead for routing
        total_area_mm2 *= 1.5

        return max(total_area_mm2, 0.001)  # Minimum 0.001 mm²

    def _calculate_chip_dimensions(self, min_area: float, area_factor: float) -> Tuple[float, float]:
        """
        Calculate chip width and height for given area factor
        Returns: (width_um, height_um)
        """
        total_area_mm2 = min_area * area_factor
        total_area_um2 = total_area_mm2 * 1e6

        # Assume square chip
        side_um = total_area_um2 ** 0.5

        return side_um, side_um

    def _estimate_power(self, gates: int, style: str, voltage: float = 1.2) -> float:
        """
        Estimate power consumption based on gate count and style
        Returns: Power in mW
        """
        # Power coefficients for different styles (relative)
        power_coefficients = {
            'complementary_logic': 1.0,
            'ratioed_logic': 1.6,
            'transmission_gate_logic': 0.8,
            'pass_transistor_logic': 0.7,
            'dynamic_logic': 2.4
        }

        base_power_per_gate = 0.01  # mW per gate at 1.2V
        coeff = power_coefficients.get(style, 1.0)

        total_power = gates * base_power_per_gate * coeff * (voltage / 1.2) ** 2

        return round(total_power, 3)

    def _estimate_delay(self, gates: int, mode: str) -> float:
        """
        Estimate critical path delay based on placement mode
        Returns: Delay in ns
        """
        # Delay coefficients for different placement modes
        delay_coefficients = {
            'standard': 1.0,
            'timing_driven': 0.7,
            'congestion_driven': 1.2,
            'power_driven': 1.1
        }

        # Assume average 5 levels of logic depth per 100 gates
        logic_depth = max(5, gates // 20)
        base_delay_per_level = 0.1  # ns

        coeff = delay_coefficients.get(mode, 1.0)
        total_delay = logic_depth * base_delay_per_level * coeff

        return round(total_delay, 3)

    def _generate_config_hash(self, config: ImplementationConfig) -> str:
        """Generate unique hash for implementation configuration"""
        config_str = f"{config.circuit_name}_{config.area_factor}_{config.placement_mode}_{config.gate_design_style}_{config.repetition}"
        return hashlib.md5(config_str.encode()).hexdigest()[:16]

    def generate_implementations_for_circuit(self, circuit_name: str) -> List[ImplementationConfig]:
        """
        Generate 400 implementations for a single circuit

        Strategy:
        - 10 area factors × 4 placement modes × 5 gate styles = 200 combinations
        - Each combination repeated 2 times = 400 implementations
        """
        if circuit_name not in self.circuits_info:
            print(f"Warning: Circuit {circuit_name} not found in circuits info")
            return []

        circuit_info = self.circuits_info[circuit_name]
        implementations = []
        impl_id = 0

        print(f"\nGenerating implementations for {circuit_name}:")
        print(
            f"  Base circuit: {circuit_info.gates} gates, {circuit_info.inputs} inputs, {circuit_info.outputs} outputs")
        print(f"  Minimum area: {circuit_info.min_area_mm2:.4f} mm²")

        # Generate all combinations with repetitions
        for repetition in [1, 2]:
            for area_factor in AREA_FACTORS:
                for placement_mode in PLACEMENT_MODES:
                    for gate_style in GATE_DESIGN_STYLES:
                        impl_id += 1

                        # Calculate chip dimensions
                        width_um, height_um = self._calculate_chip_dimensions(
                            circuit_info.min_area_mm2,
                            area_factor
                        )

                        # Create configuration
                        config = ImplementationConfig(
                            circuit_name=circuit_name,
                            implementation_id=impl_id,
                            area_factor=area_factor,
                            placement_mode=placement_mode,
                            gate_design_style=gate_style,
                            repetition=repetition,
                            chip_area_mm2=round(circuit_info.min_area_mm2 * area_factor, 6),
                            chip_width_um=round(width_um, 2),
                            chip_height_um=round(height_um, 2),
                            estimated_gates=circuit_info.gates,
                            estimated_power_mw=self._estimate_power(circuit_info.gates, gate_style),
                            estimated_delay_ns=self._estimate_delay(circuit_info.gates, placement_mode),
                            timestamp=datetime.now().isoformat()
                        )

                        config.config_hash = self._generate_config_hash(config)

                        implementations.append(config)

        print(f"  ✓ Generated {len(implementations)} implementations")

        return implementations

    def save_circuit_implementations(self, circuit_name: str, implementations: List[ImplementationConfig]) -> None:
        """Save implementations metadata for a circuit"""
        metadata = {
            'circuit_name': circuit_name,
            'total_implementations': len(implementations),
            'generated_at': datetime.now().isoformat(),
            'implementations': [asdict(impl) for impl in implementations]
        }

        metadata_file = METADATA_DIR / f'{circuit_name}_implementations.json'

        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"  ✓ Saved metadata: {metadata_file}")
        except Exception as e:
            print(f"  ✗ Failed to save metadata: {e}")

    def generate_all_implementations(self) -> None:
        """Generate implementations for all circuits"""
        print("\n" + "=" * 70)
        print("GENERATING CIRCUIT IMPLEMENTATIONS")
        print("=" * 70)
        print(
            f"Strategy: {len(AREA_FACTORS)} area factors × {len(PLACEMENT_MODES)} placement modes × {len(GATE_DESIGN_STYLES)} gate styles × 2 repetitions")
        print(
            f"Expected per circuit: {len(AREA_FACTORS) * len(PLACEMENT_MODES) * len(GATE_DESIGN_STYLES) * 2} implementations")

        for circuit_name in sorted(self.circuits_info.keys()):
            implementations = self.generate_implementations_for_circuit(circuit_name)

            if implementations:
                self.save_circuit_implementations(circuit_name, implementations)
                self.implementations.extend(implementations)
                self.stats['circuits_processed'] += 1
                self.stats['total_implementations'] += len(implementations)

    def generate_master_index(self) -> None:
        """Generate master index of all implementations"""
        print("\n" + "=" * 70)
        print("GENERATING MASTER INDEX")
        print("=" * 70)

        master_index = {
            'project': 'Hardware Trojan Vulnerability Assessment',
            'stage': 'Stage 1.2: Implementation Generation',
            'generated_at': datetime.now().isoformat(),
            'statistics': self.stats,
            'configuration': {
                'area_factors': AREA_FACTORS,
                'placement_modes': PLACEMENT_MODES,
                'gate_design_styles': GATE_DESIGN_STYLES,
                'technology_node_nm': TECHNOLOGY_NODE,
                'supply_voltage_v': SUPPLY_VOLTAGE
            },
            'circuits': {}
        }

        # Add circuit summaries
        for circuit_name in self.circuits_info.keys():
            circuit_impls = [impl for impl in self.implementations if impl.circuit_name == circuit_name]

            if circuit_impls:
                master_index['circuits'][circuit_name] = {
                    'total_implementations': len(circuit_impls),
                    'metadata_file': f'{circuit_name}_implementations.json',
                    'circuit_info': asdict(self.circuits_info[circuit_name])
                }

        index_file = IMPLEMENTATIONS_DIR / 'master_index.json'

        try:
            with open(index_file, 'w') as f:
                json.dump(master_index, f, indent=2)
            print(f"✓ Master index saved: {index_file}")
        except Exception as e:
            print(f"✗ Failed to save master index: {e}")

    def print_summary(self) -> None:
        """Print generation summary"""
        summary = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║          ✓ STAGE 1.2 IMPLEMENTATION GENERATION COMPLETED                 ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝

GENERATION SUMMARY:
────────────────────────────────────────────────────────────────────────
Circuits Processed:           {self.stats['circuits_processed']}
Total Implementations:        {self.stats['total_implementations']}
Implementations per Circuit:  {self.stats['total_implementations'] // self.stats['circuits_processed'] if self.stats['circuits_processed'] > 0 else 0}

DIVERSITY PARAMETERS:
────────────────────────────────────────────────────────────────────────
Area Factors:                 {len(AREA_FACTORS)} (1× to 10×)
Placement Modes:              {len(PLACEMENT_MODES)}
Gate Design Styles:           {len(GATE_DESIGN_STYLES)}
Repetitions:                  2
Base Combinations:            {self.stats['combinations']}

FILES CREATED:
────────────────────────────────────────────────────────────────────────
Metadata Files:               {self.stats['circuits_processed']} (one per circuit)
  Location: {METADATA_DIR}

Master Index:                 ✓ Created
  File: {IMPLEMENTATIONS_DIR / 'master_index.json'}

NEXT STEPS:
────────────────────────────────────────────────────────────────────────
1. Review master index:
   cat implementations/master_index.json

2. Check specific circuit implementations:
   cat implementations/metadata/c17_implementations.json

3. Proceed to Stage 1.3: Feature Extraction
   python scripts/extract_features.py

NOTE: This stage generated METADATA for implementations.
Actual synthesis and P&R would be done with EDA tools (Cadence/Synopsys).
For this project, we proceed with metadata-based feature extraction.
────────────────────────────────────────────────────────────────────────
"""
        print(summary)


# ================== MAIN EXECUTION ==================

def main() -> int:
    """Main execution function"""
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║         STAGE 1.2: CIRCUIT IMPLEMENTATION GENERATION                     ║
║                                                                          ║
║    Hardware Trojan Vulnerability Assessment Project                      ║
║                                                                          ║
║    Generating 400 implementations per circuit with varying:              ║
║    - Chip Area (1× to 10×)                                               ║
║    - Placement Modes (4 strategies)                                      ║
║    - Gate Design Styles (5 types)                                        ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)

    try:
        generator = ImplementationGenerator()

        # Generate all implementations
        generator.generate_all_implementations()

        # Generate master index
        generator.generate_master_index()

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
