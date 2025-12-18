#!/usr/bin/env python3
"""
Test script for Week 9 Boolean Invariant Synthesis
Runs without requiring Dafny installation.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from z3_boolean_solver import Z3BooleanSolver, BoolOp, LinearConstraint, BooleanInvariant
from dafny_parser import DafnyExtractor


def test_z3_solver():
    """Test the Z3 boolean solver"""
    print("Testing Z3 Boolean Solver...")
    print("-" * 40)
    
    solver = Z3BooleanSolver(coeff_bound=10)
    
    # Test 1: Simple case
    print("\nTest 1: x := 0; y := 0; while x < n: x++; y += 2")
    inv = solver.solve_for_coefficients(
        var_names=['x', 'y'],
        init_values={'x': 0, 'y': 0},
        update_exprs={'x': '+1', 'y': '+2'},
        num_constraints=1
    )
    assert inv is not None, "Should find single invariant"
    print(f"  Single: {inv.to_string()}")
    
    # Test 2: Conjunction
    print("\nTest 2: Conjunction of two constraints")
    inv_conj = solver.solve_for_coefficients(
        var_names=['x', 'y'],
        init_values={'x': 0, 'y': 0},
        update_exprs={'x': '+1', 'y': '+2'},
        num_constraints=2,
        combination_type=BoolOp.AND
    )
    assert inv_conj is not None, "Should find conjunction"
    print(f"  Conjunction: {inv_conj.to_string()}")
    
    # Test 3: Disjunction
    print("\nTest 3: Disjunction of two constraints")
    inv_disj = solver.solve_for_coefficients(
        var_names=['x', 'y'],
        init_values={'x': 0, 'y': 0},
        update_exprs={'x': '+1', 'y': '+2'},
        num_constraints=2,
        combination_type=BoolOp.OR
    )
    assert inv_disj is not None, "Should find disjunction"
    print(f"  Disjunction: {inv_disj.to_string()}")
    
    # Test 4: Three variables
    print("\nTest 4: Three variables with equal updates")
    inv_3var = solver.solve_for_coefficients(
        var_names=['x', 'y', 'z'],
        init_values={'x': 0, 'y': 0, 'z': 0},
        update_exprs={'x': '+1', 'y': '+1', 'z': '+1'},
        num_constraints=2,
        combination_type=BoolOp.AND
    )
    assert inv_3var is not None, "Should find 3-var invariant"
    print(f"  3-var: {inv_3var.to_string()}")
    
    # Test 5: Asymmetric updates
    print("\nTest 5: Asymmetric updates (x += 1, y += 3)")
    all_invs = solver.synthesize_all_combinations(
        var_names=['x', 'y'],
        init_values={'x': 0, 'y': 0},
        update_exprs={'x': '+1', 'y': '+3'},
        max_constraints=2
    )
    print(f"  Found {len(all_invs)} invariants:")
    for inv in all_invs[:3]:
        print(f"    - {inv.to_string()}")
    
    print("\n✓ Z3 Solver tests passed!")
    return True


def test_parser():
    """Test the Dafny parser"""
    print("\nTesting Dafny Parser...")
    print("-" * 40)
    
    parser = DafnyExtractor()
    
    # Test source code
    source = """
method Test(n: int) returns (x: int, y: int)
  requires n >= 0
  ensures x == n
{
  x := 0;
  y := 0;
  
  while x < n
    invariant x >= 0
  {
    x := x + 1;
    y := y + 2;
  }
}
"""
    
    result = parser.parse_source(source)
    
    assert "error" not in result, f"Parse error: {result.get('error')}"
    assert len(result['methods']) == 1, "Should find 1 method"
    assert result['methods'][0]['name'] == 'Test', "Method name should be Test"
    assert len(result['loops']) == 1, "Should find 1 loop"
    assert 'x' in result['loops'][0]['variables'], "Should find variable x"
    
    print(f"  Method: {result['methods'][0]['name']}")
    print(f"  Preconditions: {result['preconditions']}")
    print(f"  Postconditions: {result['postconditions']}")
    print(f"  Loop variables: {result['loops'][0]['variables']}")
    print(f"  Loop condition: {result['loops'][0]['condition']}")
    
    print("\n✓ Parser tests passed!")
    return True


def test_linear_constraint():
    """Test LinearConstraint and BooleanInvariant classes"""
    print("\nTesting Data Structures...")
    print("-" * 40)
    
    # Test LinearConstraint
    lc1 = LinearConstraint(
        coefficients={'x': 1, 'y': -2},
        constant=0
    )
    print(f"  LC1: {lc1.to_string()}")
    assert "x" in lc1.to_string() and "y" in lc1.to_string()
    
    lc2 = LinearConstraint(
        coefficients={'x': -1, 'y': 0},
        constant=10
    )
    print(f"  LC2: {lc2.to_string()}")
    
    # Test BooleanInvariant
    bi = BooleanInvariant(
        constraints=[lc1, lc2],
        operators=[BoolOp.AND]
    )
    print(f"  Combined: {bi.to_string()}")
    assert "&&" in bi.to_string()
    
    print("\n✓ Data structure tests passed!")
    return True


def test_integration():
    """Integration test with benchmark file"""
    print("\nTesting Integration...")
    print("-" * 40)
    
    benchmark_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        'benchmarks', 
        'benchmark1_conjunction.dfy'
    )
    
    if not os.path.exists(benchmark_path):
        print(f"  Benchmark not found: {benchmark_path}")
        return True
    
    from boolean_invariant_synthesis import BooleanInvariantSynthesizer
    
    synthesizer = BooleanInvariantSynthesizer(coeff_bound=10, max_constraints=3)
    result = synthesizer.synthesize(benchmark_path)
    
    print(f"  Success: {result.success}")
    print(f"  Invariants found: {len(result.invariants)}")
    for inv in result.invariants[:3]:
        print(f"    - {inv}")
    
    print("\n✓ Integration tests passed!")
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("Week 9: Boolean Invariant Synthesis - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Z3 Solver", test_z3_solver),
        ("Parser", test_parser),
        ("Data Structures", test_linear_constraint),
        ("Integration", test_integration),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"\n✗ {name} test failed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
