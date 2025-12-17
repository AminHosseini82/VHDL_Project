#!/usr/bin/env python3
"""
Stage 1.2: Benchmark Circuit Validation and Analysis
Validates all benchmark circuits and extracts their properties
"""

import os
import re
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = PROJECT_ROOT / 'benchmarks'
ISCAS85_DIR = BENCHMARKS_DIR / 'iscas85' / 'verilog'
ISCAS89_DIR = BENCHMARKS_DIR / 'iscas89' / 'verilog'


# ================== VERILOG PARSER ==================

class VerilogParser:
    """Parse and validate Verilog circuit files"""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.filename = filepath.name
        self.content = self._read_file()
        self.properties = {}

    def _read_file(self) -> str:
        """Read Verilog file content"""
        try:
            with open(self.filepath, 'r') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading {self.filepath}: {e}")
            return ""

    def parse_module_name(self) -> Optional[str]:
        """Extract module name"""
        match = re.search(r'module\s+(\w+)', self.content)
        return match.group(1) if match else None

    def parse_inputs(self) -> List[str]:
        """Extract input ports"""
        # Find all input declarations
        inputs = []
        patterns = [
            r'input\s+\[\d+:\d+\]\s+(\w+)',  # input [x:y] name
            r'input\s+(\w+)',  # input name
        ]

        for pattern in patterns:
            matches = re.findall(pattern, self.content)
            inputs.extend(matches)

        return list(set(inputs))  # Remove duplicates

    def parse_outputs(self) -> List[str]:
        """Extract output ports"""
        outputs = []
        patterns = [
            r'output\s+\[\d+:\d+\]\s+(\w+)',  # output [x:y] name
            r'output\s+(\w+)',  # output name
        ]

        for pattern in patterns:
            matches = re.findall(pattern, self.content)
            outputs.extend(matches)

        return list(set(outputs))

    def count_gates(self) -> int:
        """Count logic gates (AND, OR, NOT, XOR, etc.)"""
        gate_keywords = ['AND', 'OR', 'NOT', 'XOR', 'NAND', 'NOR', 'XNOR', 'assign', 'always']
        count = 0
        for keyword in gate_keywords:
            count += len(re.findall(keyword, self.content, re.IGNORECASE))
        return count

    def count_flip_flops(self) -> int:
        """Count flip-flops (DFF)"""
        ff_count = len(re.findall(r'DFF|FF|flip.?flop|reg\s', self.content, re.IGNORECASE))
        return ff_count

    def parse_circuit_type(self) -> str:
        """Determine if combinational or sequential"""
        has_registers = bool(re.search(r'\breg\b', self.content))
        has_always = bool(re.search(r'\balways\b', self.content))
        has_ff = self.count_flip_flops() > 0

        if has_registers or has_always or has_ff:
            return 'sequential'
        else:
            return 'combinational'

    def validate_syntax(self) -> Tuple[bool, List[str]]:
        """Validate basic Verilog syntax"""
        errors = []

        # Check for module declaration
        if not re.search(r'\bmodule\b', self.content):
            errors.append("Missing 'module' keyword")

        # Check for endmodule
        if not re.search(r'\bendmodule\b', self.content):
            errors.append("Missing 'endmodule' keyword")

        # Check for at least one input or output
        inputs = self.parse_inputs()
        outputs = self.parse_outputs()

        if not inputs and not outputs:
            errors.append("No inputs or outputs defined")

        # Check for balanced parentheses
        open_parens = self.content.count('(')
        close_parens = self.content.count(')')
        if open_parens != close_parens:
            errors.append("Unbalanced parentheses")

        # Check for balanced brackets
        open_brackets = self.content.count('[')
        close_brackets = self.content.count(']')
        if open_brackets != close_brackets:
            errors.append("Unbalanced brackets")

        return len(errors) == 0, errors

    def get_full_report(self) -> Dict:
        """Get complete circuit analysis report"""
        valid, errors = self.validate_syntax()

        return {
            'filename': self.filename,
            'filepath': str(self.filepath),
            'module_name': self.parse_module_name(),
            'inputs': self.parse_inputs(),
            'num_inputs': len(self.parse_inputs()),
            'outputs': self.parse_outputs(),
            'num_outputs': len(self.parse_outputs()),
            'gates_count': self.count_gates(),
            'flip_flops_count': self.count_flip_flops(),
            'circuit_type': self.parse_circuit_type(),
            'file_size_bytes': self.filepath.stat().st_size,
            'lines_of_code': len(self.content.split('\n')),
            'syntax_valid': valid,
            'syntax_errors': errors if not valid else []
        }


# ================== BENCHMARK VALIDATOR ==================

class BenchmarkValidator:
    """Validate all benchmark circuits"""

    def __init__(self):
        self.iscas85_circuits = {}
        self.iscas89_circuits = {}
        self.validation_reports = []

    def validate_iscas85(self) -> None:
        """Validate all ISCAS-85 circuits"""
        print("\n" + "=" * 70)
        print("VALIDATING ISCAS-85 BENCHMARK CIRCUITS (Combinational)")
        print("=" * 70)

        if not ISCAS85_DIR.exists():
            print(f"✗ Directory not found: {ISCAS85_DIR}")
            return

        verilog_files = sorted(ISCAS85_DIR.glob('*.v'))

        if not verilog_files:
            print(f"✗ No Verilog files found in {ISCAS85_DIR}")
            return

        for filepath in verilog_files:
            parser = VerilogParser(filepath)
            report = parser.get_full_report()
            self.iscas85_circuits[report['filename']] = report
            self.validation_reports.append(report)

            # Print report
            status = "✓" if report['syntax_valid'] else "✗"
            print(f"\n{status} {report['filename']}")
            print(f"  Module: {report['module_name']}")
            print(f"  Type: {report['circuit_type']}")
            print(f"  Inputs: {report['num_inputs']} {report['inputs'][:3] if report['inputs'] else '[]'}")
            print(f"  Outputs: {report['num_outputs']} {report['outputs'][:3] if report['outputs'] else '[]'}")
            print(f"  Gates: {report['gates_count']}")
            print(f"  Size: {report['file_size_bytes']} bytes, {report['lines_of_code']} lines")

            if not report['syntax_valid']:
                for error in report['syntax_errors']:
                    print(f"    ⚠ Error: {error}")

    def validate_iscas89(self) -> None:
        """Validate all ISCAS-89 circuits"""
        print("\n" + "=" * 70)
        print("VALIDATING ISCAS-89 BENCHMARK CIRCUITS (Sequential)")
        print("=" * 70)

        if not ISCAS89_DIR.exists():
            print(f"✗ Directory not found: {ISCAS89_DIR}")
            return

        verilog_files = sorted(ISCAS89_DIR.glob('*.v'))

        if not verilog_files:
            print(f"✗ No Verilog files found in {ISCAS89_DIR}")
            return

        for filepath in verilog_files:
            parser = VerilogParser(filepath)
            report = parser.get_full_report()
            self.iscas89_circuits[report['filename']] = report
            self.validation_reports.append(report)

            # Print report
            status = "✓" if report['syntax_valid'] else "✗"
            print(f"\n{status} {report['filename']}")
            print(f"  Module: {report['module_name']}")
            print(f"  Type: {report['circuit_type']}")
            print(f"  Inputs: {report['num_inputs']} {report['inputs'][:3] if report['inputs'] else '[]'}")
            print(f"  Outputs: {report['num_outputs']} {report['outputs'][:3] if report['outputs'] else '[]'}")
            print(f"  Flip-Flops: {report['flip_flops_count']}")
            print(f"  Gates: {report['gates_count']}")
            print(f"  Size: {report['file_size_bytes']} bytes, {report['lines_of_code']} lines")

            if not report['syntax_valid']:
                for error in report['syntax_errors']:
                    print(f"    ⚠ Error: {error}")

    def generate_statistics(self) -> Dict:
        """Generate overall statistics"""
        stats = {
            'total_circuits': len(self.validation_reports),
            'iscas85_count': len(self.iscas85_circuits),
            'iscas89_count': len(self.iscas89_circuits),
            'all_valid': all(r['syntax_valid'] for r in self.validation_reports),
            'valid_count': sum(1 for r in self.validation_reports if r['syntax_valid']),
            'invalid_count': sum(1 for r in self.validation_reports if not r['syntax_valid']),
            'total_inputs': sum(r['num_inputs'] for r in self.validation_reports),
            'total_outputs': sum(r['num_outputs'] for r in self.validation_reports),
            'total_gates': sum(r['gates_count'] for r in self.validation_reports),
            'total_flip_flops': sum(r['flip_flops_count'] for r in self.validation_reports),
        }

        # Averages
        if stats['total_circuits'] > 0:
            stats['avg_inputs'] = stats['total_inputs'] / stats['total_circuits']
            stats['avg_outputs'] = stats['total_outputs'] / stats['total_circuits']
            stats['avg_gates'] = stats['total_gates'] / stats['total_circuits']

        return stats

    def save_validation_report(self) -> None:
        """Save detailed validation report"""
        stats = self.generate_statistics()

        report = {
            'validation_summary': stats,
            'iscas85_circuits': self.iscas85_circuits,
            'iscas89_circuits': self.iscas89_circuits,
        }

        report_file = BENCHMARKS_DIR / 'validation_report.json'

        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n✓ Validation report saved: {report_file}")
        except Exception as e:
            print(f"✗ Failed to save report: {e}")

    def print_summary(self) -> None:
        """Print validation summary"""
        stats = self.generate_statistics()

        summary = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║              ✓ STAGE 1.2 BENCHMARK VALIDATION COMPLETED                  ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝

VALIDATION SUMMARY:
────────────────────────────────────────────────────────────────────────
Total Circuits Validated:     {stats['total_circuits']}
  ✓ Valid:                    {stats['valid_count']}
  ✗ Invalid:                  {stats['invalid_count']}

Circuit Breakdown:
  ISCAS-85 (Combinational):   {stats['iscas85_count']}
  ISCAS-89 (Sequential):      {stats['iscas89_count']}

CIRCUIT STATISTICS:
────────────────────────────────────────────────────────────────────────
Total Inputs:                 {stats['total_inputs']}
  Average per circuit:        {stats.get('avg_inputs', 0):.1f}

Total Outputs:                {stats['total_outputs']}
  Average per circuit:        {stats.get('avg_outputs', 0):.1f}

Total Gates:                  {stats['total_gates']}
  Average per circuit:        {stats.get('avg_gates', 0):.1f}

Total Flip-Flops:             {stats['total_flip_flops']}

VALIDATION STATUS:
────────────────────────────────────────────────────────────────────────
Overall Status:               {"✓ ALL PASSED" if stats['all_valid'] else "✗ SOME FAILED"}

NEXT STEPS:
────────────────────────────────────────────────────────────────────────
1. Review detailed report:
   cat benchmarks/validation_report.json

2. Proceed to Stage 1.3: Feature Extraction Setup
   python scripts/setup_feature_extraction.py

3. Begin implementation phase generation

────────────────────────────────────────────────────────────────────────
"""
        print(summary)


# ================== MAIN EXECUTION ==================

def main() -> int:
    """Main execution function"""
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║          STAGE 1.2: BENCHMARK CIRCUIT VALIDATION AND ANALYSIS            ║
║                                                                          ║
║    Hardware Trojan Vulnerability Assessment Project                      ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)

    try:
        validator = BenchmarkValidator()

        # Validate ISCAS-85
        validator.validate_iscas85()

        # Validate ISCAS-89
        validator.validate_iscas89()

        # Save report
        validator.save_validation_report()

        # Print summary
        validator.print_summary()

        return 0

    except Exception as e:
        print(f"\n✗ CRITICAL ERROR: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
