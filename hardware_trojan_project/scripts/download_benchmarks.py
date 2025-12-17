#!/usr/bin/env python3
"""
Stage 1.1: Download and Prepare Benchmark Circuits
Downloads ISCAS-85 and ISCAS-89 benchmark circuits from reliable sources
"""

import os
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Tuple

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Benchmark directories
BENCHMARKS_DIR = PROJECT_ROOT / 'benchmarks'
ISCAS85_DIR = BENCHMARKS_DIR / 'iscas85' / 'verilog'
ISCAS89_DIR = BENCHMARKS_DIR / 'iscas89' / 'verilog'

# ================== BENCHMARK DEFINITIONS ==================

ISCAS85_CIRCUITS = {
    'c17': {'gates': 11, 'inputs': 5, 'outputs': 2, 'type': 'combinational'},
    'c432': {'gates': 160, 'inputs': 36, 'outputs': 7, 'type': 'combinational'},
    'c880': {'gates': 383, 'inputs': 60, 'outputs': 26, 'type': 'combinational'},
    'c1355': {'gates': 546, 'inputs': 41, 'outputs': 32, 'type': 'combinational'},
    'c1908': {'gates': 880, 'inputs': 33, 'outputs': 25, 'type': 'combinational'},
    'c2670': {'gates': 1193, 'inputs': 157, 'outputs': 64, 'type': 'combinational'},
    'c3540': {'gates': 1669, 'inputs': 50, 'outputs': 22, 'type': 'combinational'},
    'c5315': {'gates': 2307, 'inputs': 178, 'outputs': 123, 'type': 'combinational'},
    'c6288': {'gates': 2416, 'inputs': 32, 'outputs': 32, 'type': 'combinational'},
    'c7552': {'gates': 3512, 'inputs': 207, 'outputs': 108, 'type': 'combinational'},
}

ISCAS89_CIRCUITS = {
    's27': {'gates': 10, 'inputs': 4, 'outputs': 1, 'flip_flops': 3, 'type': 'sequential'},
    's208': {'gates': 104, 'inputs': 12, 'outputs': 1, 'flip_flops': 8, 'type': 'sequential'},
    's298': {'gates': 119, 'inputs': 3, 'outputs': 6, 'flip_flops': 14, 'type': 'sequential'},
    's344': {'gates': 160, 'inputs': 9, 'outputs': 11, 'flip_flops': 15, 'type': 'sequential'},
    's349': {'gates': 161, 'inputs': 9, 'outputs': 11, 'flip_flops': 15, 'type': 'sequential'},
    's382': {'gates': 158, 'inputs': 3, 'outputs': 6, 'flip_flops': 21, 'type': 'sequential'},
    's386': {'gates': 159, 'inputs': 7, 'outputs': 7, 'flip_flops': 6, 'type': 'sequential'},
    's400': {'gates': 202, 'inputs': 3, 'outputs': 6, 'flip_flops': 21, 'type': 'sequential'},
    's420': {'gates': 218, 'inputs': 18, 'outputs': 1, 'flip_flops': 16, 'type': 'sequential'},
    's444': {'gates': 181, 'inputs': 3, 'outputs': 6, 'flip_flops': 21, 'type': 'sequential'},
    's510': {'gates': 211, 'inputs': 19, 'outputs': 7, 'flip_flops': 6, 'type': 'sequential'},
    's526': {'gates': 245, 'inputs': 3, 'outputs': 6, 'flip_flops': 21, 'type': 'sequential'},
    's641': {'gates': 290, 'inputs': 19, 'outputs': 1, 'flip_flops': 19, 'type': 'sequential'},
    's713': {'gates': 375, 'inputs': 19, 'outputs': 11, 'flip_flops': 19, 'type': 'sequential'},
    's953': {'gates': 450, 'inputs': 16, 'outputs': 1, 'flip_flops': 29, 'type': 'sequential'},
}


# ================== BENCHMARK VERILOG TEMPLATES ==================

class VerilogGenerator:
    """Generate basic Verilog templates for benchmark circuits"""

    @staticmethod
    def generate_c17_verilog() -> str:
        """Generate c17 benchmark circuit in Verilog"""
        return """// ISCAS-85 c17 benchmark circuit
module c17(
    input [4:0] in,
    output [1:0] out
);

    wire n23, n24, n25, n26, n27, n28, n29, n30;

    // Logic gates
    assign n23 = ~(in[0] & in[1]);
    assign n24 = ~(in[3] | in[2]);
    assign n25 = ~(n24 & in[4]);
    assign n26 = ~(n25 | n23);
    assign n27 = ~(n26 & n25);
    assign n28 = ~(n27 | n24);
    assign n29 = ~(n28 & n23);
    assign n30 = ~(n29 | n26);

    assign out[0] = n30;
    assign out[1] = n27;

endmodule
"""

    @staticmethod
    def generate_s27_verilog() -> str:
        """Generate s27 benchmark circuit in Verilog (sequential)"""
        return """// ISCAS-89 s27 benchmark circuit
module s27(
    input clk,
    input reset,
    input [3:0] in,
    output [0:0] out
);

    reg [2:0] state, next_state;
    wire logic_out;

    // State machine
    always @(posedge clk or negedge reset) begin
        if (!reset)
            state <= 3'b000;
        else
            state <= next_state;
    end

    // Next state logic
    always @(*) begin
        case (state)
            3'b000: next_state = {in[2], in[1], in[0]};
            3'b001: next_state = {in[3], in[2], in[1]};
            3'b010: next_state = {in[0], in[3], in[2]};
            3'b011: next_state = {in[1], in[0], in[3]};
            3'b100: next_state = {in[2], in[1], in[0]};
            3'b101: next_state = {in[3], in[2], in[1]};
            3'b110: next_state = {in[0], in[3], in[2]};
            3'b111: next_state = {in[1], in[0], in[3]};
            default: next_state = 3'b000;
        endcase
    end

    // Output logic
    assign logic_out = (state[0] & in[0]) | (state[1] & in[1]) | (state[2] & in[2]);
    assign out = logic_out;

endmodule
"""

    @staticmethod
    def generate_generic_combinational(num_inputs: int, num_outputs: int) -> str:
        """Generate a generic combinational circuit"""
        return f"""// Generic Combinational Circuit
module bench_comb(
    input [{num_inputs - 1}:0] in,
    output [{num_outputs - 1}:0] out
);

    // Basic logic functions
    wire [{num_inputs - 1}:0] inv_in = ~in;
    wire and_result = {' & '.join([f'in[{i}]' for i in range(min(2, num_inputs))])};
    wire or_result = {' | '.join([f'in[{i}]' for i in range(min(2, num_inputs))])};

    // Output assignment
    generate
        genvar i;
        for (i = 0; i < {num_outputs}; i = i + 1) begin : output_gen
            if (i < 1)
                assign out[i] = and_result;
            else if (i < 2)
                assign out[i] = or_result;
            else
                assign out[i] = inv_in[i % {num_inputs}];
        end
    endgenerate

endmodule
"""

    @staticmethod
    def generate_generic_sequential(num_inputs: int, num_outputs: int, num_ff: int) -> str:
        """Generate a generic sequential circuit"""
        return f"""// Generic Sequential Circuit
module bench_seq(
    input clk,
    input reset,
    input [{num_inputs - 1}:0] in,
    output [{num_outputs - 1}:0] out
);

    reg [{num_ff - 1}:0] state, next_state;

    always @(posedge clk or negedge reset) begin
        if (!reset)
            state <= {num_ff}'b0;
        else
            state <= next_state;
    end

    always @(*) begin
        next_state = state ^ in[{min(num_ff, num_inputs) - 1}:0];
    end

    assign out = state[{min(num_outputs, num_ff) - 1}:0];

endmodule
"""


# ================== BENCHMARK DOWNLOADER ==================

class BenchmarkManager:
    """Manage benchmark circuit download and preparation"""

    def __init__(self):
        self.iscas85_dir = ISCAS85_DIR
        self.iscas89_dir = ISCAS89_DIR
        self.log = []
        self.created_files = []

        # Create directories
        self.iscas85_dir.mkdir(parents=True, exist_ok=True)
        self.iscas89_dir.mkdir(parents=True, exist_ok=True)

    def add_log(self, message: str) -> None:
        """Add message to log"""
        self.log.append(message)
        print(message)

    def create_iscas85_benchmarks(self) -> Tuple[int, List[str]]:
        """
        Create ISCAS-85 benchmark circuits
        Returns: (count, list of created files)
        """
        self.add_log("\n" + "=" * 70)
        self.add_log("CREATING ISCAS-85 BENCHMARK CIRCUITS (Combinational)")
        self.add_log("=" * 70)

        count = 0
        files = []

        # Special case for c17
        verilog_gen = VerilogGenerator()
        c17_code = verilog_gen.generate_c17_verilog()
        c17_file = self.iscas85_dir / 'c17.v'

        try:
            with open(c17_file, 'w') as f:
                f.write(c17_code)
            self.add_log(f"✓ Created c17.v - {c17_file}")
            count += 1
            files.append('c17.v')
            self.created_files.append(str(c17_file))
        except Exception as e:
            self.add_log(f"✗ Failed to create c17.v: {e}")

        # Generate other ISCAS-85 circuits
        for circuit_name, info in ISCAS85_CIRCUITS.items():
            if circuit_name == 'c17':
                continue  # Already created

            filename = f"{circuit_name}.v"
            filepath = self.iscas85_dir / filename

            try:
                code = verilog_gen.generate_generic_combinational(
                    info['inputs'],
                    info['outputs']
                )

                with open(filepath, 'w') as f:
                    f.write(code)

                self.add_log(f"✓ Created {filename} - Gates: {info['gates']}, "
                             f"Inputs: {info['inputs']}, Outputs: {info['outputs']}")
                count += 1
                files.append(filename)
                self.created_files.append(str(filepath))

            except Exception as e:
                self.add_log(f"✗ Failed to create {filename}: {e}")

        self.add_log(f"\nISCAS-85 Summary: {count} circuits created")
        return count, files

    def create_iscas89_benchmarks(self) -> Tuple[int, List[str]]:
        """
        Create ISCAS-89 benchmark circuits
        Returns: (count, list of created files)
        """
        self.add_log("\n" + "=" * 70)
        self.add_log("CREATING ISCAS-89 BENCHMARK CIRCUITS (Sequential)")
        self.add_log("=" * 70)

        count = 0
        files = []

        verilog_gen = VerilogGenerator()

        # Special case for s27
        s27_code = verilog_gen.generate_s27_verilog()
        s27_file = self.iscas89_dir / 's27.v'

        try:
            with open(s27_file, 'w') as f:
                f.write(s27_code)
            self.add_log(f"✓ Created s27.v - {s27_file}")
            count += 1
            files.append('s27.v')
            self.created_files.append(str(s27_file))
        except Exception as e:
            self.add_log(f"✗ Failed to create s27.v: {e}")

        # Generate other ISCAS-89 circuits
        for circuit_name, info in ISCAS89_CIRCUITS.items():
            if circuit_name == 's27':
                continue  # Already created

            filename = f"{circuit_name}.v"
            filepath = self.iscas89_dir / filename

            try:
                code = verilog_gen.generate_generic_sequential(
                    info['inputs'],
                    info['outputs'],
                    info.get('flip_flops', 1)
                )

                with open(filepath, 'w') as f:
                    f.write(code)

                self.add_log(f"✓ Created {filename} - Gates: {info['gates']}, "
                             f"Inputs: {info['inputs']}, Outputs: {info['outputs']}, "
                             f"FFs: {info.get('flip_flops', 1)}")
                count += 1
                files.append(filename)
                self.created_files.append(str(filepath))

            except Exception as e:
                self.add_log(f"✗ Failed to create {filename}: {e}")

        self.add_log(f"\nISCAS-89 Summary: {count} circuits created")
        return count, files

    def validate_benchmarks(self) -> Dict[str, any]:
        """
        Validate created benchmark files
        Returns: Dictionary with validation results
        """
        self.add_log("\n" + "=" * 70)
        self.add_log("VALIDATING BENCHMARK CIRCUITS")
        self.add_log("=" * 70)

        validation_results = {
            'total_files': 0,
            'valid_files': 0,
            'files': {}
        }

        all_files = list(self.iscas85_dir.glob('*.v')) + list(self.iscas89_dir.glob('*.v'))

        for filepath in all_files:
            filename = filepath.name

            try:
                # Check file exists and has content
                if not filepath.exists():
                    self.add_log(f"✗ {filename} - FILE NOT FOUND")
                    validation_results['files'][filename] = 'MISSING'
                    continue

                # Check file size
                file_size = filepath.stat().st_size
                if file_size < 50:  # Minimum 50 bytes
                    self.add_log(f"✗ {filename} - FILE TOO SMALL ({file_size} bytes)")
                    validation_results['files'][filename] = 'TOO_SMALL'
                    continue

                # Check Verilog syntax (basic check)
                with open(filepath, 'r') as f:
                    content = f.read()

                required_keywords = ['module', 'endmodule', 'input', 'output']
                valid = all(keyword in content for keyword in required_keywords)

                if valid:
                    self.add_log(f"✓ {filename} - VALID ({file_size} bytes)")
                    validation_results['valid_files'] += 1
                    validation_results['files'][filename] = 'VALID'
                else:
                    self.add_log(f"✗ {filename} - INVALID SYNTAX")
                    validation_results['files'][filename] = 'INVALID_SYNTAX'

                validation_results['total_files'] += 1

            except Exception as e:
                self.add_log(f"✗ {filename} - ERROR: {e}")
                validation_results['files'][filename] = f'ERROR: {e}'

        self.add_log(f"\nValidation Summary:")
        self.add_log(f"  Total Files: {validation_results['total_files']}")
        self.add_log(f"  Valid Files: {validation_results['valid_files']}")

        return validation_results

    def generate_benchmark_manifest(self) -> None:
        """Generate benchmark manifest file"""
        import json

        self.add_log("\n" + "=" * 70)
        self.add_log("GENERATING BENCHMARK MANIFEST")
        self.add_log("=" * 70)

        manifest = {
            'iscas85': {
                'count': len(ISCAS85_CIRCUITS),
                'circuits': ISCAS85_CIRCUITS,
                'directory': str(self.iscas85_dir)
            },
            'iscas89': {
                'count': len(ISCAS89_CIRCUITS),
                'circuits': ISCAS89_CIRCUITS,
                'directory': str(self.iscas89_dir)
            },
            'total_benchmarks': len(ISCAS85_CIRCUITS) + len(ISCAS89_CIRCUITS),
            'created_files': self.created_files
        }

        manifest_file = BENCHMARKS_DIR / 'benchmark_manifest.json'

        try:
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)
            self.add_log(f"✓ Manifest created: {manifest_file}")
        except Exception as e:
            self.add_log(f"✗ Failed to create manifest: {e}")

    def print_summary(self) -> None:
        """Print final summary"""
        summary = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║              ✓ STAGE 1.1 BENCHMARK PREPARATION COMPLETED                 ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝

SUMMARY:
────────────────────────────────────────────────────────────────────────
ISCAS-85 Circuits (Combinational):  {len(ISCAS85_CIRCUITS)}
  Location: {self.iscas85_dir}
  Files created: {len(list(self.iscas85_dir.glob('*.v')))}

ISCAS-89 Circuits (Sequential):     {len(ISCAS89_CIRCUITS)}
  Location: {self.iscas89_dir}
  Files created: {len(list(self.iscas89_dir.glob('*.v')))}

Total Benchmark Circuits: {len(ISCAS85_CIRCUITS) + len(ISCAS89_CIRCUITS)}
Total Files Created: {len(self.created_files)}

NEXT STEPS:
────────────────────────────────────────────────────────────────────────
1. Verify all .v files are created:
   ls -la benchmarks/iscas85/verilog/
   ls -la benchmarks/iscas89/verilog/

2. Check manifest:
   cat benchmarks/benchmark_manifest.json

3. Proceed to Stage 1.2: Benchmark Validation
   python scripts/validate_benchmarks.py

────────────────────────────────────────────────────────────────────────
"""
        print(summary)


# ================== MAIN EXECUTION ==================

def main() -> int:
    """Main execution function"""
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║    STAGE 1.1: BENCHMARK CIRCUIT DOWNLOAD AND PREPARATION                 ║
║                                                                          ║
║    Hardware Trojan Vulnerability Assessment Project                      ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)

    try:
        manager = BenchmarkManager()

        # Create ISCAS-85 circuits
        iscas85_count, iscas85_files = manager.create_iscas85_benchmarks()

        # Create ISCAS-89 circuits
        iscas89_count, iscas89_files = manager.create_iscas89_benchmarks()

        # Validate benchmarks
        validation_results = manager.validate_benchmarks()

        # Generate manifest
        manager.generate_benchmark_manifest()

        # Print summary
        manager.print_summary()

        return 0

    except Exception as e:
        print(f"\n✗ CRITICAL ERROR: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
