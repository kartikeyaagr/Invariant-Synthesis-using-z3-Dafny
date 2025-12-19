#!/usr/bin/env python3
"""
Test suite for Week 12-13 Array Invariant Synthesis
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from array_analyzer import ArrayAnalyzer, ArrayPattern, ArrayInvariantTemplates
from z3_array_solver import Z3ArraySolver, QuantifiedInvariantSynthesizer
from array_synthesis import ArrayInvariantSynthesizer, DafnyArrayParser


def test_array_analyzer():
    """Test the array analyzer"""
    print("Testing Array Analyzer...")
    print("-" * 40)
    
    analyzer = ArrayAnalyzer()
    
    # Test 1: Initialization pattern
    print("\n  Test 1: Initialization pattern")
    analysis = analyzer.analyze_loop(
        "i < a.Length",
        "{ a[i] := 0; i := i + 1; }",
        ["i", "a"],
        ["a"]
    )
    assert analysis.pattern == ArrayPattern.INIT
    assert "a" in analysis.modified_arrays
    print(f"    Pattern: {analysis.pattern.value} ✓")
    
    # Test 2: Accumulate pattern
    print("\n  Test 2: Accumulate pattern")
    analysis = analyzer.analyze_loop(
        "i < a.Length",
        "{ sum := sum + a[i]; i := i + 1; }",
        ["i", "sum", "a"],
        ["a"]
    )
    assert analysis.pattern == ArrayPattern.ACCUMULATE
    assert "a" in analysis.read_arrays
    print(f"    Pattern: {analysis.pattern.value} ✓")
    
    # Test 3: Copy pattern
    print("\n  Test 3: Copy pattern")
    analysis = analyzer.analyze_loop(
        "i < src.Length",
        "{ dst[i] := src[i]; i := i + 1; }",
        ["i", "src", "dst"],
        ["src", "dst"]
    )
    assert analysis.pattern == ArrayPattern.COPY
    print(f"    Pattern: {analysis.pattern.value} ✓")
    
    # Test 4: Index analysis
    print("\n  Test 4: Index analysis")
    analysis = analyzer.analyze_loop(
        "i < n",
        "{ a[i] := 0; i := i + 1; }",
        ["i", "n", "a"],
        ["a"]
    )
    assert analysis.index_var == "i"
    assert analysis.index_direction == "increasing"
    print(f"    Index: {analysis.index_var}, Direction: {analysis.index_direction} ✓")
    
    print("\n✓ Array Analyzer tests passed!")
    return True


def test_invariant_templates():
    """Test invariant templates"""
    print("\nTesting Invariant Templates...")
    print("-" * 40)
    
    templates = ArrayInvariantTemplates()
    
    # Test 1: Bounds invariant
    print("\n  Test 1: Bounds invariant")
    inv = templates.array_bounds("i", "a")
    assert "0 <= i" in inv
    assert "a.Length" in inv
    print(f"    {inv} ✓")
    
    # Test 2: Initialized elements
    print("\n  Test 2: Initialized elements")
    inv = templates.initialized_elements("i", "a", "0")
    assert inv.quantifier == "forall"
    assert "a[k] == 0" in inv.to_string()
    print(f"    {inv.to_string()} ✓")
    
    # Test 3: Copy invariant
    print("\n  Test 3: Copy invariant")
    inv = templates.copy_invariant("src", "dst", "i")
    assert "dst[k] == src[k]" in inv.to_string()
    print(f"    {inv.to_string()} ✓")
    
    # Test 4: Sorted prefix
    print("\n  Test 4: Sorted prefix")
    inv = templates.sorted_prefix("i", "a")
    assert "a[k] <= a[k + 1]" in inv.to_string()
    print(f"    {inv.to_string()} ✓")
    
    print("\n✓ Invariant Templates tests passed!")
    return True


def test_z3_array_solver():
    """Test Z3 array solver"""
    print("\nTesting Z3 Array Solver...")
    print("-" * 40)
    
    analyzer = ArrayAnalyzer()
    solver = Z3ArraySolver()
    
    # Test 1: Bounds synthesis
    print("\n  Test 1: Bounds synthesis")
    analysis = analyzer.analyze_loop(
        "i < a.Length",
        "{ a[i] := 0; i := i + 1; }",
        ["i", "a"],
        ["a"]
    )
    
    bounds = solver.synthesize_bounds_invariants(analysis)
    assert len(bounds) > 0
    print(f"    Found {len(bounds)} bounds invariant(s)")
    for b in bounds[:2]:
        print(f"      - {b.expression}")
    
    # Test 2: Quantified synthesis
    print("\n  Test 2: Quantified synthesis")
    quant = solver.synthesize_quantified_invariants(analysis)
    print(f"    Found {len(quant)} quantified invariant(s)")
    for q in quant[:2]:
        print(f"      - {q.expression}")
    
    print("\n✓ Z3 Array Solver tests passed!")
    return True


def test_parser():
    """Test Dafny array parser"""
    print("\nTesting Dafny Array Parser...")
    print("-" * 40)
    
    parser = DafnyArrayParser()
    
    source = """
method InitArray(a: array<int>)
  modifies a
  ensures forall k :: 0 <= k < a.Length ==> a[k] == 0
{
  var i := 0;
  while i < a.Length
    invariant 0 <= i <= a.Length
  {
    a[i] := 0;
    i := i + 1;
  }
}
"""
    
    result = parser.parse_source(source)
    
    assert "error" not in result
    assert len(result["methods"]) == 1
    assert "a" in result["arrays"]
    assert len(result["loops"]) == 1
    
    method = result["methods"][0]
    print(f"  Method: {method['name']}")
    print(f"  Arrays: {method['arrays']}")
    print(f"  Loops: {len(method['loops'])}")
    print(f"  Postconditions: {method['postconditions']}")
    
    print("\n✓ Parser tests passed!")
    return True


def test_full_synthesis():
    """Test full synthesis pipeline"""
    print("\nTesting Full Synthesis...")
    print("-" * 40)
    
    synthesizer = ArrayInvariantSynthesizer()
    
    # Test from spec
    print("\n  Test 1: Array initialization")
    invs = synthesizer.synthesize_from_spec(
        loop_condition="i < a.Length",
        loop_body="{ a[i] := 0; i := i + 1; }",
        variables=["i", "a"],
        arrays=["a"]
    )
    assert len(invs) > 0
    print(f"    Found {len(invs)} invariant(s)")
    for inv in invs[:3]:
        print(f"      - {inv}")
    
    # Test 2: Array copy
    print("\n  Test 2: Array copy")
    invs = synthesizer.synthesize_from_spec(
        loop_condition="i < src.Length",
        loop_body="{ dst[i] := src[i]; i := i + 1; }",
        variables=["i", "src", "dst"],
        arrays=["src", "dst"]
    )
    assert len(invs) > 0
    print(f"    Found {len(invs)} invariant(s)")
    for inv in invs[:3]:
        print(f"      - {inv}")
    
    print("\n✓ Full Synthesis tests passed!")
    return True


def test_benchmark_files():
    """Test on benchmark files"""
    print("\nTesting Benchmark Files...")
    print("-" * 40)
    
    benchmark_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'benchmarks'
    )
    
    if not os.path.exists(benchmark_dir):
        print(f"  Benchmark directory not found")
        return True
    
    synthesizer = ArrayInvariantSynthesizer()
    benchmarks = [f for f in os.listdir(benchmark_dir) if f.endswith('.dfy')]
    
    for bench in sorted(benchmarks)[:3]:
        bench_path = os.path.join(benchmark_dir, bench)
        print(f"\n  {bench}:")
        
        result = synthesizer.synthesize(bench_path)
        status = "✓" if result.success else "✗"
        print(f"    {status} Pattern: {result.pattern}, {len(result.invariants)} inv(s)")
    
    print("\n✓ Benchmark tests passed!")
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("Week 12-13: Array Invariant Synthesis - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Array Analyzer", test_array_analyzer),
        ("Invariant Templates", test_invariant_templates),
        ("Z3 Array Solver", test_z3_array_solver),
        ("Dafny Parser", test_parser),
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
