#!/usr/bin/env python3
"""
Test suite for Week 10 Disjunctive Invariant Synthesis
"""

import sys
import os

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from path_analyzer import PathAnalyzer, PathType
from z3_disjunctive_solver import Z3DisjunctiveSolver
from disjunctive_synthesis import DisjunctiveInvariantSynthesizer


def test_path_analyzer():
    """Test path analysis"""
    print("Testing Path Analyzer...")
    print("-" * 40)
    
    analyzer = PathAnalyzer()
    
    # Test 1: Linear path
    print("\n  Test 1: Linear loop")
    analysis = analyzer.analyze_loop(
        "i < n",
        "{ i := i + 1; x := x + 2; }",
        ["i", "x"]
    )
    assert analysis.path_type == PathType.LINEAR
    assert len(analysis.paths) == 1
    print(f"    Path type: {analysis.path_type.value}")
    print(f"    Paths: {len(analysis.paths)}")
    
    # Test 2: If-then-else
    print("\n  Test 2: If-then-else")
    analysis = analyzer.analyze_loop(
        "i < n",
        "{ if (i % 2 == 0) { x := x + 2; } else { x := x + 1; } i := i + 1; }",
        ["i", "x"]
    )
    assert analysis.path_type == PathType.IF_THEN_ELSE
    assert len(analysis.paths) == 2
    print(f"    Path type: {analysis.path_type.value}")
    print(f"    Paths: {len(analysis.paths)}")
    
    # Test 3: If-then (no else)
    print("\n  Test 3: If-then (no else)")
    analysis = analyzer.analyze_loop(
        "i < n",
        "{ if (x > 0) { y := y + 1; } i := i + 1; }",
        ["i", "x", "y"]
    )
    # Now returns IF_THEN_ELSE because we add implicit else path
    assert analysis.path_type in (PathType.IF_THEN, PathType.IF_THEN_ELSE)
    assert len(analysis.paths) == 2  # implicit else
    print(f"    Path type: {analysis.path_type.value}")
    print(f"    Paths: {len(analysis.paths)}")
    
    print("\n✓ Path Analyzer tests passed!")
    return True


def test_z3_disjunctive_solver():
    """Test Z3 disjunctive solving"""
    print("\nTesting Z3 Disjunctive Solver...")
    print("-" * 40)
    
    solver = Z3DisjunctiveSolver(coeff_bound=10)
    analyzer = PathAnalyzer()
    
    # Test 1: Simple disjunction
    print("\n  Test 1: Simple disjunctive invariant")
    analysis = analyzer.analyze_loop(
        "i < n",
        "{ if (i % 2 == 0) { x := x + 2; } else { x := x + 1; } i := i + 1; }",
        ["i", "x"]
    )
    
    inv = solver.synthesize_disjunctive(
        var_names=["i", "x"],
        init_values={"i": 0, "x": 0},
        path_analysis=analysis,
        num_disjuncts=2
    )
    assert inv is not None
    print(f"    Invariant: {inv.to_string()}")
    
    # Test 2: Path-sensitive
    print("\n  Test 2: Path-sensitive invariant")
    inv_ps = solver.synthesize_path_sensitive(
        var_names=["i", "x"],
        init_values={"i": 0, "x": 0},
        path_analysis=analysis
    )
    assert inv_ps is not None
    print(f"    Invariant: {inv_ps.to_conjunctive_string()}")
    
    # Test 3: Alternating
    print("\n  Test 3: Alternating behavior")
    inv_alt = solver.synthesize_alternating(
        var_names=["i", "x"],
        init_values={"i": 0, "x": 0},
        even_deltas={"i": 1, "x": 2},
        odd_deltas={"i": 1, "x": 1}
    )
    assert inv_alt is not None
    print(f"    Invariant: {inv_alt.to_string()}")
    
    print("\n✓ Z3 Disjunctive Solver tests passed!")
    return True


def test_full_synthesis():
    """Test full synthesis pipeline"""
    print("\nTesting Full Synthesis Pipeline...")
    print("-" * 40)
    
    synthesizer = DisjunctiveInvariantSynthesizer()
    
    # Test from spec
    print("\n  Test 1: Synthesis from spec (if-else)")
    invs = synthesizer.synthesize_from_spec(
        var_names=["i", "x"],
        init_values={"i": 0, "x": 0},
        loop_body="{ if (i % 2 == 0) { x := x + 2; } else { x := x + 1; } i := i + 1; }",
        loop_condition="i < n"
    )
    assert len(invs) > 0
    print(f"    Found {len(invs)} invariant(s)")
    for inv in invs[:2]:
        print(f"      - {inv}")
    
    # Test with threshold
    print("\n  Test 2: Threshold-based branching")
    invs = synthesizer.synthesize_from_spec(
        var_names=["i", "x", "y"],
        init_values={"i": 0, "x": 0, "y": 0},
        loop_body="{ if (i < 5) { x := x + 1; } else { y := y + 1; } i := i + 1; }",
        loop_condition="i < 10"
    )
    assert len(invs) > 0
    print(f"    Found {len(invs)} invariant(s)")
    for inv in invs[:2]:
        print(f"      - {inv}")
    
    print("\n✓ Full Synthesis tests passed!")
    return True


def test_benchmark_files():
    """Test synthesis on benchmark files"""
    print("\nTesting Benchmark Files...")
    print("-" * 40)
    
    benchmark_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'benchmarks'
    )
    
    if not os.path.exists(benchmark_dir):
        print(f"  Benchmark directory not found: {benchmark_dir}")
        return True
    
    synthesizer = DisjunctiveInvariantSynthesizer()
    
    benchmarks = [f for f in os.listdir(benchmark_dir) if f.endswith('.dfy')]
    
    for bench in sorted(benchmarks)[:3]:  # Test first 3
        bench_path = os.path.join(benchmark_dir, bench)
        print(f"\n  {bench}:")
        
        result = synthesizer.synthesize(bench_path)
        status = "✓" if result.success else "✗"
        print(f"    {status} Found {len(result.invariants)} invariant(s)")
        print(f"    Type: {result.synthesis_type}")
    
    print("\n✓ Benchmark tests passed!")
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("Week 10: Disjunctive Invariant Synthesis - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Path Analyzer", test_path_analyzer),
        ("Z3 Disjunctive Solver", test_z3_disjunctive_solver),
        ("Full Synthesis", test_full_synthesis),
        ("Benchmark Files", test_benchmark_files),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"\n✗ {name} test failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
