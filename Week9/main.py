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
from z3_boolean_solver import Z3BooleanSolver, BoolOp, ConstraintType
from boolean_invariant_synthesis import BooleanInvariantSynthesizer, InvariantInserter
from dafny_verifier import DafnyVerifier


def run_synthesis(input_file: str, output_file: str = None,
                  coeff_bound: int = 10, max_constraints: int = 3,
                  validate: bool = True, max_combo_size: int = 4) -> Dict:
    """
    Run invariant synthesis on a Dafny file.
    """
    print("=" * 60)
    print("Boolean Invariant Synthesis")
    print(f"Input: {input_file}")
    print("=" * 60)
    
    synthesizer = BooleanInvariantSynthesizer(
        coeff_bound=coeff_bound,
        max_constraints=max_constraints
    )
    
    result = synthesizer.synthesize(input_file)
    
    print("\nSynthesis Results:")
    for msg in result.messages:
        print(f"  {msg}")
    
    if not result.success:
        print("\nNo invariants found.")
        return {"success": False, "invariants": [], "validated": []}
    
    print(f"\nFound {len(result.invariants)} candidate invariant(s):")
    for i, inv in enumerate(result.invariants):
        print(f"  {i+1}. {inv}")
    
    validated_invariants = []
    if validate:
        print("\n" + "-" * 40)
        print("Validating with Dafny...")
        print("-" * 40)
        verifier = DafnyVerifier()
        
        with open(input_file, 'r') as f:
            source = f.read()
        
        # First try each invariant individually
        print("\nTrying individual invariants:")
        individual_valid = []
        for inv in result.invariants:
            vresult = verifier.verify_with_invariant(source, inv)
            status = "✓ Valid" if vresult.success else "✗ Invalid"
            print(f"  {inv}: {status}")
            if vresult.success:
                individual_valid.append(inv)
        
        if individual_valid:
            print(f"\n{len(individual_valid)} invariant(s) valid individually!")
            validated_invariants = individual_valid
            # Auto-save to synthesized folder
            if validated_invariants:
                import os
                synth_dir = os.path.join(os.path.dirname(input_file) or '.', 'synthesized')
                os.makedirs(synth_dir, exist_ok=True)
                
                base_name = os.path.basename(input_file)
                synth_path = os.path.join(synth_dir, base_name)
                
                inserter = InvariantInserter()
                modified = inserter.insert(source, validated_invariants)
                with open(synth_path, 'w') as f:
                    f.write(modified)
                print(f"\n📁 Saved to: {synth_path}")
        else:
            # Try combinations of invariants
            print(f"\nNo single invariant sufficient. Trying combinations (up to {max_combo_size})...")
            
            # Filter to promising candidates (skip obviously bad ones like "-7 * n == 0")
            promising = []
            for inv in result.invariants:
                # Skip invariants that are just about n being 0
                if inv.strip() in ['-7 * n == 0', '-6 * n == 0', '-5 * n == 0', 'n <= 0']:
                    continue
                # Skip redundant conjunctions/disjunctions
                if ' && ' in inv or ' || ' in inv:
                    continue
                promising.append(inv)
            
            print(f"  Filtered to {len(promising)} promising candidates")
            
            # Find minimal valid set
            valid_set = verifier.find_minimal_valid_set(source, promising, max_combo_size)
            
            if valid_set:
                print(f"\n✓ Found valid combination of {len(valid_set)} invariant(s):")
                for inv in valid_set:
                    print(f"    - {inv}")
                validated_invariants = valid_set

                # Auto-save to synthesized folder
                if validated_invariants:
                    import os
                    synth_dir = os.path.join(os.path.dirname(input_file) or '.', 'synthesized')
                    os.makedirs(synth_dir, exist_ok=True)
                    
                    base_name = os.path.basename(input_file)
                    synth_path = os.path.join(synth_dir, base_name)
                    
                    inserter = InvariantInserter()
                    modified = inserter.insert(source, validated_invariants)
                    with open(synth_path, 'w') as f:
                        f.write(modified)
                    print(f"\n📁 Saved to: {synth_path}")
            else:
                print("\n✗ No valid combination found.")
                print("  Possible reasons:")
                print("    - Need more invariants (try -n 4)")
                print("    - Need larger coefficient bound (try -c 20)")
                print("    - Need bounds invariants like '0 <= i <= n'")
                
                # Suggest what's likely missing
                print("\n  Hint: For this loop, you likely need:")
                print("    - A relationship invariant (e.g., j == 2 * i)")
                print("    - Bounds invariants (e.g., 0 <= i <= n)")
    
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
    print("=" * 60)
    print("Running Test Suite")
    print("=" * 60)
    
    solver = Z3BooleanSolver(coeff_bound=10)
    
    # Test 1: The problem case - i, j, n
    print("\n" + "-" * 40)
    print("Test 1: Two-variable linear update (Problem 1)")
    print("-" * 40)
    print("  Code: i := 0; j := 0")
    print("        while i < n: i := i + 1; j := j + 2")
    print("  Expected: j == 2 * i")
    
    var_names = ['i', 'j', 'n']
    init_values = {'i': 0, 'j': 0, 'n': 0}
    updates = {'i': '+1', 'j': '+2', 'n': '+0'}
    
    print("\n  Diverse equalities:")
    diverse_eq = solver.synthesize_diverse(
        var_names, init_values, updates,
        num_solutions=5,
        constraint_type=ConstraintType.EQUALITY
    )
    for inv in diverse_eq:
        print(f"    [EQ] {inv.to_string()}")
    
    print("\n  Diverse inequalities:")
    diverse_ineq = solver.synthesize_diverse(
        var_names, init_values, updates,
        num_solutions=5,
        constraint_type=ConstraintType.INEQUALITY
    )
    for inv in diverse_ineq:
        print(f"    [LE] {inv.to_string()}")
    
    # Test 2: All combinations
    print("\n" + "-" * 40)
    print("Test 2: All synthesized invariants")
    print("-" * 40)
    all_invs = solver.synthesize_all_combinations(
        var_names, init_values, updates, max_constraints=2
    )
    print(f"  Found {len(all_invs)} unique invariants:")
    for i, inv in enumerate(all_invs):
        eq_marker = "[EQ]" if inv.constraints[0].is_equality else "[LE]"
        print(f"    {i+1}. {eq_marker} {inv.to_string()}")
    
    # Test 3: Three variables all equal
    print("\n" + "-" * 40)
    print("Test 3: Three-variable equal update")
    print("-" * 40)
    print("  Code: x := 0; y := 0; z := 0")
    print("        while ...: x++; y++; z++")
    print("  Expected: x == y, y == z, x == z")
    
    var_names = ['x', 'y', 'z']
    init_values = {'x': 0, 'y': 0, 'z': 0}
    updates = {'x': '+1', 'y': '+1', 'z': '+1'}
    
    all_invs = solver.synthesize_all_combinations(
        var_names, init_values, updates, max_constraints=2
    )
    print(f"  Found {len(all_invs)} unique invariants:")
    for i, inv in enumerate(all_invs[:8]):
        eq_marker = "[EQ]" if inv.constraints[0].is_equality else "[LE]"
        print(f"    {i+1}. {eq_marker} {inv.to_string()}")
    if len(all_invs) > 8:
        print(f"    ... and {len(all_invs) - 8} more")
    
    # Test 4: Asymmetric update
    print("\n" + "-" * 40)
    print("Test 4: Asymmetric update (y = 3*x)")
    print("-" * 40)
    print("  Code: x := 0; y := 0")
    print("        while ...: x++; y += 3")
    print("  Expected: y == 3 * x")
    
    var_names = ['x', 'y']
    init_values = {'x': 0, 'y': 0}
    updates = {'x': '+1', 'y': '+3'}
    
    all_invs = solver.synthesize_all_combinations(
        var_names, init_values, updates, max_constraints=2
    )
    print(f"  Found {len(all_invs)} invariants:")
    for i, inv in enumerate(all_invs):
        eq_marker = "[EQ]" if inv.constraints[0].is_equality else "[LE]"
        print(f"    {i+1}. {eq_marker} {inv.to_string()}")
    
    print("\n" + "=" * 60)
    print("Tests completed.")
    print("=" * 60)


def run_benchmarks():
    """Run all benchmark programs"""
    print("Running benchmarks...")
    print("=" * 60)
    
    benchmark_dir = os.path.join(os.path.dirname(__file__), 'benchmarks')
    
    if not os.path.exists(benchmark_dir):
        print(f"Benchmark directory not found: {benchmark_dir}")
        print("Creating example benchmark...")
        os.makedirs(benchmark_dir, exist_ok=True)
        
        sample = """// Sample benchmark
method loop(n: int) returns (j: int)
    requires n >= 0
    ensures j == 2 * n
{
    var i := 0;
    j := 0;
    
    while i < n
        decreases n - i
    {
        i := i + 1;
        j := j + 2;
    }
}
"""
        with open(os.path.join(benchmark_dir, 'sample.dfy'), 'w') as f:
            f.write(sample)
        print(f"Created {benchmark_dir}/sample.dfy")
    
    benchmarks = sorted([f for f in os.listdir(benchmark_dir) if f.endswith('.dfy')])
    
    if not benchmarks:
        print("No benchmark files found.")
        return
    
    results = []
    
    for bench_file in benchmarks:
        bench_path = os.path.join(benchmark_dir, bench_file)
        print(f"\nBenchmark: {bench_file}")
        print("-" * 40)
        
        result = run_synthesis(bench_path, validate=True)
        results.append({
            "file": bench_file,
            "success": result["success"],
            "count": len(result.get("invariants", [])),
            "validated": len(result.get("validated", []))
        })
    
    print("\n" + "=" * 60)
    print("Benchmark Summary")
    print("=" * 60)
    
    total = len(results)
    successful = sum(1 for r in results if r["success"])
    validated = sum(1 for r in results if r["validated"] > 0)
    
    for r in results:
        synth_status = "✓" if r["success"] else "✗"
        valid_status = f"({r['validated']} validated)" if r["validated"] > 0 else "(none validated)"
        print(f"  {synth_status} {r['file']}: {r['count']} invariant(s) {valid_status}")
    
    print(f"\nTotal: {successful}/{total} synthesized, {validated}/{total} with valid invariants")


def main():
    parser = argparse.ArgumentParser(
        description='Boolean Invariant Synthesis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py program.dfy                  Synthesize invariants
  python main.py program.dfy -o output.dfy    Insert validated invariants into output
  python main.py program.dfy -c 20            Use larger coefficient bound
  python main.py program.dfy --max-combo 5    Try combinations of up to 5 invariants
  python main.py --test                       Run test suite
  python main.py --benchmark                  Run benchmarks

Notes:
  - The tool synthesizes both equalities (==) and inequalities (<=)
  - [EQ] marks equality invariants, [LE] marks inequality invariants
  - Parameters like 'n' in 'requires n >= 0' are handled correctly
  - If no single invariant works, the tool tries combinations
        """
    )
    
    parser.add_argument('input', nargs='?', help='Input Dafny file')
    parser.add_argument('-o', '--output', help='Output file with invariants')
    parser.add_argument('-c', '--coeff-bound', type=int, default=10,
                       help='Coefficient bound (default: 10)')
    parser.add_argument('-n', '--max-constraints', type=int, default=3,
                       help='Max constraints to combine (default: 3)')
    parser.add_argument('--max-combo', type=int, default=4,
                       help='Max invariants to combine during validation (default: 4)')
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
        validate=not args.no_validate,
        max_combo_size=args.max_combo
    )
    
    if args.json:
        print(json.dumps(result, indent=2))
    
    return 0 if result.get("validated") else 1


if __name__ == "__main__":
    sys.exit(main())
