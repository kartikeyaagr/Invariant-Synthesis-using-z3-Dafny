#!/usr/bin/env python3
"""
Dafny Invariant Synthesis - Complete Pipeline (Weeks 1-12)
==========================================================

Usage:
    python main.py --demo              # Demo all weeks
    python main.py --test              # Run all benchmarks
    python main.py program.dfy         # Synthesize invariants
    python main.py program.dfy --all   # Use all techniques
"""

import sys
import os
import argparse
import json
import subprocess
from typing import List, Dict, Any, Optional

# Find and add week directories to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Try different naming conventions for week folders
WEEK_DIRS = {}
for week_num in range(1, 13):
    for pattern in [f'week{week_num}', f'Week{week_num}', f'w{week_num}', 
                    f'week{week_num:02d}', f'Week{week_num:02d}']:
        path = os.path.join(SCRIPT_DIR, pattern)
        if os.path.isdir(path):
            WEEK_DIRS[week_num] = path
            sys.path.insert(0, path)
            break

# Check Z3
try:
    from z3 import Solver, Int, sat, unsat
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False

# Import synthesizers with fallbacks
DafnyParser = None
LinearSynth = None
DafnyVerifier = None
InvariantInserter = None
ProgramParser = None
BooleanSynth = None
DisjunctiveSynth = None
QuadraticSynth = None
ArraySynth = None

# Week 5: Parser (uses DafnyExtractor, not DafnyParser)
try:
    from parser import DafnyExtractor as DafnyParser
except ImportError:
    try:
        from dafny_parser import DafnyExtractor as DafnyParser
    except ImportError:
        pass

# Week 6: Linear
try:
    from linear_invariant_synthesis import LinearInvariantSynthesizer as LinearSynth
except ImportError:
    try:
        from linear_synthesis import LinearInvariantSynthesizer as LinearSynth
    except ImportError:
        try:
            from linear import LinearInvariantSynthesizer as LinearSynth
        except ImportError:
            pass

# Week 7: Verification (script-based, import individual components)
# Need to handle import carefully to avoid conflicts with week9's dafny_verifier
DafnyVerifier = None
InvariantInserter = None
ProgramParser = None
if 7 in WEEK_DIRS:
    import importlib.util
    try:
        # Load dafny_verifier from week7 specifically
        spec = importlib.util.spec_from_file_location("week7_verifier", 
            os.path.join(WEEK_DIRS[7], "dafny_verifier.py"))
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            DafnyVerifier = module.verify_dafny_program
        
        # Load program_parser
        spec = importlib.util.spec_from_file_location("week7_parser",
            os.path.join(WEEK_DIRS[7], "program_parser.py"))
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            ProgramParser = module.parse_dafny_program
        
        # Load invariant_inserter
        spec = importlib.util.spec_from_file_location("week7_inserter",
            os.path.join(WEEK_DIRS[7], "invariant_inserter.py"))
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            InvariantInserter = module.insert_invariants
    except Exception:
        pass

# Week 9: Boolean
try:
    from boolean_invariant_synthesis import BooleanInvariantSynthesizer as BooleanSynth
except ImportError:
    try:
        from bool_syn import BooleanInvariantSynthesizer as BooleanSynth
    except ImportError:
        pass

# Week 10: Disjunctive
try:
    from disjunctive_synthesis import DisjunctiveInvariantSynthesizer as DisjunctiveSynth
except ImportError:
    try:
        from disj_syn import DisjunctiveInvariantSynthesizer as DisjunctiveSynth
    except ImportError:
        pass

# Week 11: Quadratic
try:
    from quadratic_synthesis import QuadraticInvariantSynthesizer as QuadraticSynth
except ImportError:
    try:
        from quad_syn import QuadraticInvariantSynthesizer as QuadraticSynth
    except ImportError:
        pass

# Week 12: Arrays
try:
    from array_synthesis import ArrayInvariantSynthesizer as ArraySynth
except ImportError:
    try:
        from arr_syn import ArrayInvariantSynthesizer as ArraySynth
    except ImportError:
        pass


def print_header(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_section(title: str):
    print(f"\n[{title}]")
    print("-" * 40)


def check_dafny():
    """Check if Dafny is installed"""
    try:
        result = subprocess.run(['dafny', '--version'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False


def run_demo():
    """Demonstrate all 12 weeks"""
    print_header("Dafny Invariant Synthesis - Full Demo (Weeks 1-12)")
    
    # ==================== WEEK 1: Environment ====================
    print_section("Week 1: Environment Setup")
    print(f"  Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"  Z3: {'✓ Installed' if Z3_AVAILABLE else '✗ Not installed (pip install z3-solver)'}")
    print(f"  Dafny: {'✓ Installed' if check_dafny() else '✗ Not installed'}")
    print(f"  Week folders found: {sorted(WEEK_DIRS.keys())}")
    
    if Z3_AVAILABLE:
        s = Solver()
        x = Int('x')
        s.add(x > 0, x < 10)
        print(f"  Z3 test: {s.check()}")
    
    # ==================== WEEK 2: Manual Invariants ====================
    print_section("Week 2: Manual Invariants")
    if 2 in WEEK_DIRS:
        dfy_files = [f for f in os.listdir(WEEK_DIRS[2]) if f.endswith('.dfy')]
        if dfy_files:
            print(f"  Example programs: {', '.join(dfy_files[:3])}")
            sample = os.path.join(WEEK_DIRS[2], dfy_files[0])
            try:
                with open(sample) as f:
                    lines = f.readlines()[:10]
                print(f"  Sample from {dfy_files[0]}:")
                for line in lines[:5]:
                    if line.strip():
                        print(f"    {line.rstrip()}")
            except:
                pass
        else:
            print("  No .dfy files found")
    else:
        print("  Week 2 folder not found")
    
    # ==================== WEEK 3: Benchmarks ====================
    print_section("Week 3: Benchmark Suite")
    if 3 in WEEK_DIRS:
        dfy_files = [f for f in os.listdir(WEEK_DIRS[3]) if f.endswith('.dfy')]
        print(f"  Benchmark files: {len(dfy_files)}")
        for f in dfy_files[:3]:
            print(f"    - {f}")
    else:
        print("  Week 3 folder not found")
    
    # ==================== WEEK 4: Z3 Basics ====================
    print_section("Week 4: Z3 Constraint Solving")
    if Z3_AVAILABLE:
        s = Solver()
        x, y = Int('x'), Int('y')
        s.add(x + y == 10, x > 0, y > 0, x < y)
        if s.check() == sat:
            m = s.model()
            print(f"  Problem: x + y = 10, x > 0, y > 0, x < y")
            print(f"  Solution: x = {m[x]}, y = {m[y]}")
        
        a, b, c = Int('a'), Int('b'), Int('c')
        s2 = Solver()
        s2.add(a >= -5, a <= 5, b >= -5, b <= 5, c >= -5, c <= 5)
        for i in range(5):
            s2.add(a * i + b * (2*i) + c <= 0)
        s2.add(a != 0)
        if s2.check() == sat:
            m = s2.model()
            print(f"  Invariant coeffs: {m[a]}*x + {m[b]}*y + {m[c]} <= 0")
    else:
        print("  Z3 not available")
    
    # ==================== WEEK 5: Parser ====================
    print_section("Week 5: Dafny Parser")
    if DafnyParser:
        try:
            parser = DafnyParser()
            test_source = """
method Test(n: int) returns (x: int)
  requires n >= 0
{
  x := 0;
  while x < n { x := x + 1; }
}
"""
            result = parser.parse_source(test_source)
            print(f"  Parsed: {len(result.get('methods', []))} method(s), {len(result.get('loops', []))} loop(s)")
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print("  Parser not available")
    
    # ==================== WEEK 6: Linear Synthesis ====================
    print_section("Week 6: Linear Invariant Synthesis")
    if LinearSynth and 6 in WEEK_DIRS:
        try:
            synth = LinearSynth()
            # Find a test file
            test_file = None
            for f in os.listdir(WEEK_DIRS[6]):
                if f.endswith('.dfy'):
                    test_file = os.path.join(WEEK_DIRS[6], f)
                    break
            if test_file:
                print(f"  File: {os.path.basename(test_file)}")
                print(f"  Running synthesis... (output below)")
                synth.synthesize(test_file)
            else:
                print("  No .dfy test file found")
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print("  Linear synthesizer not available")
    
    # ==================== WEEK 7: Verification Pipeline ====================
    print_section("Week 7: Verification Pipeline")
    if DafnyVerifier and InvariantInserter:
        print("  Components available:")
        print("    - program_parser.parse_dafny_program(file)")
        print("    - invariant_inserter.insert_invariants(lines, loop_info, invariants)")
        print("    - dafny_verifier.verify_dafny_program(file)")
        if 7 in WEEK_DIRS:
            test_file = os.path.join(WEEK_DIRS[7], 'test_program.dfy')
            if os.path.exists(test_file):
                try:
                    lines, loop_info = ProgramParser(test_file)
                    print(f"  Parsed test_program.dfy: loop at line {loop_info.get('line_number', '?')}")
                except Exception as e:
                    print(f"  Parse test: {e}")
    else:
        print("  Pipeline not available")
    
    # ==================== WEEK 8: Research Survey ====================
    print_section("Week 8: Research Survey")
    print("  Papers: CAV03 (Farkas), CAV08 (Polynomial), CAV10 (ICE), CSUR14 (Taxonomy)")
    print("  Techniques: Constraint-based, Template-based, Learning-based")
    
    # ==================== WEEK 9-12: Advanced Synthesis ====================
    for week, name, synth_class, folder_key in [
        (9, "Boolean Combinations", BooleanSynth, 9),
        (10, "Disjunctive Invariants", DisjunctiveSynth, 10),
        (11, "Quadratic Invariants", QuadraticSynth, 11),
        (12, "Array Invariants", ArraySynth, 12),
    ]:
        print_section(f"Week {week}: {name}")
        if synth_class and folder_key in WEEK_DIRS:
            try:
                synth = synth_class()
                bench_dir = os.path.join(WEEK_DIRS[folder_key], 'benchmarks')
                if os.path.isdir(bench_dir):
                    dfy_files = [f for f in os.listdir(bench_dir) if f.endswith('.dfy')]
                    if dfy_files:
                        bench_file = os.path.join(bench_dir, dfy_files[0])
                        result = synth.synthesize(bench_file)
                        invs = result.invariants if hasattr(result, 'invariants') else []
                        print(f"  File: {dfy_files[0]}")
                        for inv in invs[:3]:
                            print(f"    -> {inv}")
                    else:
                        print("  No benchmark files")
                else:
                    print("  No benchmarks folder")
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print(f"  Not available")
    
    print_header("Demo Complete - All 12 Weeks")


def run_benchmarks():
    """Run benchmarks for all available weeks"""
    print_header("Running All Benchmarks (Weeks 1-12)")
    
    total_files = 0
    total_passed = 0
    
    for week, synth_class in [(9, BooleanSynth), (10, DisjunctiveSynth), 
                               (11, QuadraticSynth), (12, ArraySynth)]:
        if week in WEEK_DIRS and synth_class:
            print_section(f"Week {week} Benchmarks")
            bench_dir = os.path.join(WEEK_DIRS[week], 'benchmarks')
            if os.path.isdir(bench_dir):
                synth = synth_class()
                for f in sorted(os.listdir(bench_dir)):
                    if f.endswith('.dfy'):
                        total_files += 1
                        path = os.path.join(bench_dir, f)
                        try:
                            result = synth.synthesize(path)
                            count = len(result.invariants) if hasattr(result, 'invariants') else 0
                            print(f"  ✓ {f}: {count} invariants")
                            total_passed += 1
                        except Exception as e:
                            print(f"  ✗ {f}: {e}")
    
    print_header(f"Summary: {total_passed}/{total_files} benchmarks passed")


def synthesize_file(filepath: str, use_all: bool = False) -> Dict[str, Any]:
    """Synthesize invariants for a Dafny file"""
    results = {"file": filepath, "invariants": [], "by_technique": {}}
    
    if not os.path.exists(filepath):
        results["error"] = f"File not found: {filepath}"
        return results
    
    for name, synth_class in [("boolean", BooleanSynth), ("disjunctive", DisjunctiveSynth),
                               ("quadratic", QuadraticSynth), ("array", ArraySynth)]:
        if synth_class and (use_all or name == "boolean"):
            try:
                synth = synth_class()
                result = synth.synthesize(filepath)
                invs = result.invariants if hasattr(result, 'invariants') else []
                results["by_technique"][name] = invs
                results["invariants"].extend(invs)
            except Exception as e:
                results["by_technique"][f"{name}_error"] = str(e)
    
    results["invariants"] = list(set(results["invariants"]))
    return results


def list_modules():
    """List all available modules"""
    print_header("Available Modules (Weeks 1-12)")
    
    modules = [
        (1, "Setup", True, "Environment verification"),
        (2, "Manual", 2 in WEEK_DIRS, "Manual invariant examples"),
        (3, "Benchmarks", 3 in WEEK_DIRS, "Benchmark suite"),
        (4, "Z3", Z3_AVAILABLE, "Constraint solving"),
        (5, "Parser", DafnyParser is not None, "Dafny lexer/parser"),
        (6, "Linear", LinearSynth is not None, "Linear synthesis"),
        (7, "Pipeline", DafnyVerifier is not None, "Verification"),
        (8, "Survey", 8 in WEEK_DIRS, "Research survey"),
        (9, "Boolean", BooleanSynth is not None, "Boolean combinations"),
        (10, "Disjunctive", DisjunctiveSynth is not None, "Path-sensitive"),
        (11, "Quadratic", QuadraticSynth is not None, "Polynomial"),
        (12, "Arrays", ArraySynth is not None, "Quantified arrays"),
    ]
    
    for week, name, available, desc in modules:
        status = "✓" if available else "✗"
        print(f"  Week {week:2d}: {status} {name:12} - {desc}")
    
    print(f"\nWeek folders: {sorted(WEEK_DIRS.keys())}")


def main():
    parser = argparse.ArgumentParser(description='Dafny Invariant Synthesis (Weeks 1-12)')
    parser.add_argument('input', nargs='?', help='Input Dafny file')
    parser.add_argument('--demo', action='store_true', help='Demo all 12 weeks')
    parser.add_argument('--test', action='store_true', help='Run all benchmarks')
    parser.add_argument('--list', action='store_true', help='List modules')
    parser.add_argument('--all', action='store_true', help='Use all techniques')
    parser.add_argument('--json', action='store_true', help='JSON output')
    
    args = parser.parse_args()
    
    if args.list:
        list_modules()
        return 0
    if args.demo:
        run_demo()
        return 0
    if args.test:
        run_benchmarks()
        return 0
    if not args.input:
        parser.print_help()
        return 1
    
    results = synthesize_file(args.input, use_all=args.all)
    
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_header(f"Results: {args.input}")
        print(f"Total: {len(results['invariants'])} invariants")
        for tech, data in results.get("by_technique", {}).items():
            if isinstance(data, list):
                print(f"\n{tech.title()}: {len(data)}")
                for inv in data[:5]:
                    print(f"  - {inv}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
