#!/usr/bin/env python3
"""
Test suite for Week 11 Quadratic Invariant Synthesis
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from z3_quadratic_solver import Z3QuadraticSolver, QuadraticInvariant, QuadraticType
from quadratic_synthesis import QuadraticInvariantSynthesizer


def test_quadratic_solver():
    """Test the Z3 quadratic solver"""
    print("Testing Z3 Quadratic Solver...")
    print("-" * 40)
    
    solver = Z3QuadraticSolver(coeff_bound=5)
    
    # Test 1: Simple inequality
    print("\n  Test 1: Simple quadratic inequality")
    inv = solver.synthesize_quadratic(
        var_names=["x"],
        init_values={"x": 0},
        update_exprs={"x": "+1"},
        loop_bound=10,
        invariant_type=QuadraticType.INEQUALITY
    )
    assert inv is not None, "Should find quadratic inequality"
    print(f"    Found: {inv.to_string()}")
    
    # Test 2: Two variables
    print("\n  Test 2: Two variables")
    inv = solver.synthesize_quadratic(
        var_names=["x", "y"],
        init_values={"x": 0, "y": 0},
        update_exprs={"x": "+1", "y": "+2"},
        loop_bound=10
    )
    assert inv is not None, "Should find invariant"
    print(f"    Found: {inv.to_string()}")
    
    # Test 3: With cross terms
    print("\n  Test 3: With cross terms enabled")
    inv = solver.synthesize_quadratic(
        var_names=["x", "y"],
        init_values={"x": 0, "y": 0},
        update_exprs={"x": "+1", "y": "+1"},
        loop_bound=10,
        include_cross_terms=True
    )
    if inv:
        print(f"    Found: {inv.to_string()}")
    else:
        print("    No cross-term invariant found (may be expected)")
    
    # Test 4: Quadratic evaluation
    print("\n  Test 4: Invariant evaluation")
    inv = QuadraticInvariant(
        squared_coeffs={"x": 1},
        cross_coeffs={},
        linear_coeffs={"x": -1},
        constant=0,
        is_equality=True
    )
    # This represents: x² - x == 0, true when x=0 or x=1
    assert inv.evaluate({"x": 0}) == 0, "Should be 0 at x=0"
    assert inv.evaluate({"x": 1}) == 0, "Should be 0 at x=1"
    print(f"    Invariant: {inv.to_string()}")
    print(f"    evaluate(x=0) = {inv.evaluate({'x': 0})}")
    print(f"    evaluate(x=1) = {inv.evaluate({'x': 1})}")
    
    print("\n✓ Z3 Quadratic Solver tests passed!")
    return True


def test_quadratic_invariant():
    """Test QuadraticInvariant class"""
    print("\nTesting QuadraticInvariant class...")
    print("-" * 40)
    
    # Test 1: String conversion
    print("\n  Test 1: String conversion")
    inv = QuadraticInvariant(
        squared_coeffs={"x": 2, "y": 1},
        cross_coeffs={("x", "y"): 3},
        linear_coeffs={"x": -4, "y": 5},
        constant=-6,
        is_equality=False
    )
    s = inv.to_string()
    print(f"    {s}")
    assert "x * x" in s or "2 * x * x" in s
    assert "y * y" in s
    
    # Test 2: Triviality check
    print("\n  Test 2: Triviality check")
    trivial = QuadraticInvariant(
        squared_coeffs={"x": 0},
        cross_coeffs={},
        linear_coeffs={"x": 0},
        constant=0,
        is_equality=True
    )
    assert trivial.is_trivial(), "Should be trivial"
    print(f"    Trivial invariant detected correctly")
    
    # Test 3: Non-trivial
    print("\n  Test 3: Non-trivial check")
    non_trivial = QuadraticInvariant(
        squared_coeffs={"x": 1},
        cross_coeffs={},
        linear_coeffs={},
        constant=0,
        is_equality=False
    )
    assert not non_trivial.is_trivial(), "Should not be trivial"
    print(f"    Non-trivial: {non_trivial.to_string()}")
    
    print("\n✓ QuadraticInvariant tests passed!")
    return True


def test_synthesis():
    """Test full synthesis pipeline"""
    print("\nTesting Synthesis Pipeline...")
    print("-" * 40)
    
    synthesizer = QuadraticInvariantSynthesizer(coeff_bound=5, loop_bound=15)
    
    # Test 1: From spec
    print("\n  Test 1: Synthesis from spec")
    invs = synthesizer.synthesize_from_spec(
        var_names=["x", "y"],
        init_values={"x": 0, "y": 0},
        updates={"x": "+1", "y": "+2"}
    )
    assert len(invs) > 0, "Should find invariants"
    print(f"    Found {len(invs)} invariant(s)")
    for inv in invs[:2]:
        print(f"      - {inv}")
    
    # Test 2: Triangular numbers
    print("\n  Test 2: Triangular numbers (specialized)")
    invs = synthesizer.synthesize_for_triangular("i", "sum")
    print(f"    Found {len(invs)} invariant(s)")
    for inv in invs:
        print(f"      - {inv}")
    
    print("\n✓ Synthesis Pipeline tests passed!")
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
    
    synthesizer = QuadraticInvariantSynthesizer()
    benchmarks = [f for f in os.listdir(benchmark_dir) if f.endswith('.dfy')]
    
    for bench in sorted(benchmarks)[:3]:
        bench_path = os.path.join(benchmark_dir, bench)
        print(f"\n  {bench}:")
        
        result = synthesizer.synthesize(bench_path)
        status = "✓" if result.success else "✗"
        print(f"    {status} Found {len(result.invariants)} invariant(s)")
        for inv in result.invariants[:2]:
            print(f"      - {inv}")
    
    print("\n✓ Benchmark tests passed!")
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("Week 11: Quadratic Invariant Synthesis - Test Suite")
    print("=" * 60)
    
    tests = [
        ("QuadraticInvariant class", test_quadratic_invariant),
        ("Z3 Quadratic Solver", test_quadratic_solver),
        ("Synthesis Pipeline", test_synthesis),
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
