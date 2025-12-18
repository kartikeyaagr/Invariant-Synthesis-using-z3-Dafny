#!/usr/bin/env python3
"""
Week 9: Boolean Invariant Synthesis Tool
Main entry point for synthesizing and validating boolean combinations of linear invariants.

Usage:
  python main.py <dafny_file>                    # Synthesize invariants
  python main.py <dafny_file> -o <output.dfy>    # Synthesize and insert into output file
  python main.py --test                          # Run test suite
  python main.py --benchmark                     # Run all benchmarks
"""

import sys
import os
import argparse
import json
from typing import List, Dict

from dafny_parser import DafnyExtractor
from z3_boolean_solver import Z3BooleanSolver, BoolOp
from boolean_invariant_synthesis import BooleanInvariantSynthesizer, InvariantInserter
from dafny_verifier import DafnyVerifier


def run_synthesis(input_file: str, output_file: str = None,
                  coeff_bound: int = 10, max_constraints: int = 3,
                  validate: bool = True) -> Dict:
    """
    Run invariant synthesis on a Dafny file.
    """
    print(f"=" * 60)
    print(f"Boolean Invariant Synthesis")
    print(f"Input: {input_file}")
    print(f"=" * 60)
    
    # Create synthesizer
    synthesizer = BooleanInvariantSynthesizer(
        coeff_bound=coeff_bound,
        max_constraints=max_constraints
    )
    
    # Run synthesis
    result = synthesizer.synthesize(input_file)
    
    # Print results
    print("\nSynthesis Results:")
    for msg in result.messages:
        print(f"  {msg}")
    
    if not result.success:
        print("\nNo invariants found.")
        return {"success": False, "invariants": []}
    
    print(f"\nFound {len(result.invariants)} candidate invariant(s):")
    for i, inv in enumerate(result.invariants):
        print(f"  {i+1}. {inv}")
    
    # Validate with Dafny if requested
    validated_invariants = []
    if validate:
        print("\nValidating with Dafny...")
        verifier = DafnyVerifier()
        
        with open(input_file, 'r') as f:
            source = f.read()
        
        for inv in result.invariants:
            vresult = verifier.verify_with_invariant(source, inv)
            status = "✓ Valid" if vresult.success else "✗ Invalid"
            print(f"  {inv}: {status}")
            if vresult.success:
                validated_invariants.append(inv)
    
    # Write output file if requested
    if output_file and validated_invariants:
        print(f"\nWriting to {output_file}...")
        inserter = InvariantInserter()
        with open(input_file, 'r') as f:
            source = f.read()
        modified = inserter.insert(source, validated_invariants)
        with open(output_file, 'w') as f:
            f.write(modified)
        print(f"Done. Output written to {output_file}")
    
    return {
        "success": True,
        "invariants": result.invariants,
        "validated": validated_invariants
    }


def run_tests():
    """Run internal test suite"""
    print("Running test suite...")
    print("=" * 60)
    
    solver = Z3BooleanSolver(coeff_bound=10)
    
    # Test 1: Simple two-variable loop
    print("\nTest 1: Two-variable linear update")
    print("  x := 0; y := 0")
    print("  while x < n: x := x + 1; y := y + 2")
    
    var_names = ['x', 'y']
    init_values = {'x': 0, 'y': 0}
    updates = {'x': '+1', 'y': '+2'}
    
    inv = solver.solve_for_coefficients(
        var_names, init_values, updates, num_constraints=1
    )
    if inv:
        print(f"  Single invariant: {inv.to_string()}")
    
    inv_conj = solver.solve_for_coefficients(
        var_names, init_values, updates, 
        num_constraints=2, combination_type=BoolOp.AND
    )
    if inv_conj:
        print(f"  Conjunction: {inv_conj.to_string()}")
    
    # Test 2: Three variables
    print("\nTest 2: Three-variable update")
    print("  x := 0; y := 0; z := 0")
    print("  while x < n: x := x + 1; y := y + 1; z := z + 1")
    
    var_names = ['x', 'y', 'z']
    init_values = {'x': 0, 'y': 0, 'z': 0}
    updates = {'x': '+1', 'y': '+1', 'z': '+1'}
    
    inv = solver.solve_for_coefficients(
        var_names, init_values, updates, num_constraints=1
    )
    if inv:
        print(f"  Single invariant: {inv.to_string()}")
    
    inv_conj = solver.solve_for_coefficients(
        var_names, init_values, updates,
        num_constraints=3, combination_type=BoolOp.AND
    )
    if inv_conj:
        print(f"  Triple conjunction: {inv_conj.to_string()}")
    
    # Test 3: Asymmetric update
    print("\nTest 3: Asymmetric update")
    print("  x := 0; y := 0")
    print("  while x < n: x := x + 1; y := y + 3")
    
    var_names = ['x', 'y']
    init_values = {'x': 0, 'y': 0}
    updates = {'x': '+1', 'y': '+3'}
    
    all_invs = solver.synthesize_all_combinations(
        var_names, init_values, updates, max_constraints=2
    )
    print(f"  Found {len(all_invs)} invariants:")
    for inv in all_invs[:5]:
        print(f"    - {inv.to_string()}")
    
    print("\n" + "=" * 60)
    print("Tests completed.")


def run_benchmarks():
    """Run all benchmark programs"""
    print("Running benchmarks...")
    print("=" * 60)
    
    benchmark_dir = os.path.join(os.path.dirname(__file__), 'benchmarks')
    
    if not os.path.exists(benchmark_dir):
        print(f"Benchmark directory not found: {benchmark_dir}")
        return
    
    benchmarks = sorted([f for f in os.listdir(benchmark_dir) if f.endswith('.dfy')])
    
    if not benchmarks:
        print("No benchmark files found.")
        return
    
    results = []
    
    for bench_file in benchmarks:
        bench_path = os.path.join(benchmark_dir, bench_file)
        print(f"\nBenchmark: {bench_file}")
        print("-" * 40)
        
        result = run_synthesis(bench_path, validate=False)
        results.append({
            "file": bench_file,
            "success": result["success"],
            "count": len(result.get("invariants", []))
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("Benchmark Summary")
    print("=" * 60)
    
    total = len(results)
    successful = sum(1 for r in results if r["success"])
    
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"  {status} {r['file']}: {r['count']} invariant(s)")
    
    print(f"\nTotal: {successful}/{total} benchmarks succeeded")


def main():
    parser = argparse.ArgumentParser(
        description='Boolean Invariant Synthesis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py program.dfy                  Synthesize invariants
  python main.py program.dfy -o output.dfy    Insert invariants into output
  python main.py --test                       Run test suite
  python main.py --benchmark                  Run benchmarks
        """
    )
    
    parser.add_argument('input', nargs='?', help='Input Dafny file')
    parser.add_argument('-o', '--output', help='Output file with invariants')
    parser.add_argument('-c', '--coeff-bound', type=int, default=10,
                       help='Coefficient bound (default: 10)')
    parser.add_argument('-n', '--max-constraints', type=int, default=3,
                       help='Max constraints to combine (default: 3)')
    parser.add_argument('--test', action='store_true', help='Run test suite')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmarks')
    parser.add_argument('--no-validate', action='store_true',
                       help='Skip Dafny validation')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    
    args = parser.parse_args()
    
    if args.test:
        run_tests()
        return 0
    
    if args.benchmark:
        run_benchmarks()
        return 0
    
    if not args.input:
        parser.print_help()
        return 1
    
    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        return 1
    
    result = run_synthesis(
        args.input,
        args.output,
        args.coeff_bound,
        args.max_constraints,
        validate=not args.no_validate
    )
    
    if args.json:
        print(json.dumps(result, indent=2))
    
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
